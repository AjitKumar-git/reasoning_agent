"""
Deterministic, resumable corpus iteration for full-corpus training.

The loaders in `src/data_loaders.py` build correctly shaped batches for short runs, but
they re-shuffle with module-level `random` and re-tokenise per record, so a run cannot say
where it is in the corpus or continue from there. This module answers both questions:
every source is a pure function of (seed, position in the stream), and a caller can skip
exactly N already-consumed records on resume.

Holdout is by parquet row group. Row groups are how parquet is physically stored, so
excluding them costs nothing, and a held-out slice drawn this way cannot appear in
training no matter how the within-group order is shuffled.

Every iterator yields raw `(prompt_text, answer_text)` pairs. Tokenisation is separate
because training slices a window out of the answer and the eval path does not.
"""

import hashlib
import json
import os
import random
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import pyarrow.parquet as pq

Pair = Tuple[str, str]

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")

SOURCES = ("openhermes", "lmsys", "knowledge", "cognitive", "gsm8k", "cosmopedia")


# --------------------------------------------------------------------------------------------
# row selection
# --------------------------------------------------------------------------------------------

def group_plan(num_groups: int, seed: int, holdout_every: int, want_heldout: bool) -> List[int]:
    """Ordered row-group indices, split deterministically into train and held-out halves.

    `holdout_every=50` reserves groups 0, 50, 100, ... for evaluation. Passing 0 disables
    the split and every group is a train group.
    """
    order = list(range(num_groups))
    random.Random(seed).shuffle(order)
    if holdout_every <= 0:
        return [] if want_heldout else order
    if want_heldout:
        return [g for g in order if g % holdout_every == 0]
    return [g for g in order if g % holdout_every != 0]


def row_plan(num_rows: int, seed: int, holdout_every: int, want_heldout: bool) -> List[int]:
    """Same split for single-group files, keyed on row index instead of row group."""
    order = list(range(num_rows))
    random.Random(seed).shuffle(order)
    if holdout_every <= 0:
        return [] if want_heldout else order
    if want_heldout:
        return [r for r in order if r % holdout_every == 0]
    return [r for r in order if r % holdout_every != 0]


def _shuffled(rows: List[Pair], seed: int, salt: int) -> List[Pair]:
    order = list(range(len(rows)))
    random.Random(seed * 1000003 + salt).shuffle(order)
    return [rows[i] for i in order]


# --------------------------------------------------------------------------------------------
# per-source readers
# --------------------------------------------------------------------------------------------

def _conversation_pairs(conv: Sequence) -> List[Pair]:
    """Extracts every user->assistant turn from a chat-like conversation list."""
    if not conv or not isinstance(conv, list):
        return []
    out = []
    for i in range(len(conv) - 1):
        u, a = conv[i], conv[i + 1]
        if not isinstance(u, dict) or not isinstance(a, dict):
            continue
        u_role = u.get("from") or u.get("role")
        a_role = a.get("from") or a.get("role")
        if u_role in ("human", "user") and a_role in ("gpt", "assistant", "model"):
            u_text = (u.get("value") or u.get("content") or "").strip()
            a_text = (a.get("value") or a.get("content") or "").strip()
            if u_text and a_text:
                out.append((u_text, a_text))
    return out


def openhermes_pairs(path: str, seed: int = 0, holdout_every: int = 0,
                     want_heldout: bool = False, english_only: bool = True) -> Iterator[Pair]:
    parquet_file = pq.ParquetFile(path)
    for rg_idx in group_plan(parquet_file.num_row_groups, seed, holdout_every, want_heldout):
        table = parquet_file.read_row_group(rg_idx)
        columns = table.column_names
        rows: List[Pair] = []
        if "language" in columns and english_only:
            langs = table["language"].to_pylist()
            convs = table["conversations"].to_pylist()
            for lang, conv in zip(langs, convs):
                if lang and lang.lower() not in ("english", "en"):
                    continue
                rows += _conversation_pairs(conv)
        elif "conversations" in columns:
            for conv in table["conversations"].to_pylist():
                rows += _conversation_pairs(conv)
        elif "instruction" in columns and "output" in columns:
            out_col = table["output"].to_pylist()
            in_col = table["input"].to_pylist() if "input" in columns else [""] * len(out_col)
            for inst, inp, out in zip(table["instruction"].to_pylist(), in_col, out_col):
                prompt = f"{inst.strip()} {inp.strip()}".strip() if inp else inst.strip()
                if prompt and out and out.strip():
                    rows.append((prompt, out.strip()))
        yield from _shuffled(rows, seed, rg_idx)


def lmsys_pairs(data_dir: str, seed: int = 0, holdout_every: int = 0,
                want_heldout: bool = False) -> Iterator[Pair]:
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".parquet"))
    order = list(range(len(files)))
    random.Random(seed).shuffle(order)
    for file_idx in order:
        path = os.path.join(data_dir, files[file_idx])
        parquet_file = pq.ParquetFile(path)
        for rg_idx in group_plan(parquet_file.num_row_groups, seed + file_idx, holdout_every, want_heldout):
            table = parquet_file.read_row_group(rg_idx)
            langs = table["language"].to_pylist()
            convs = table["conversation"].to_pylist()
            rows: List[Pair] = []
            for lang, conv in zip(langs, convs):
                if lang != "English":
                    continue
                rows += _conversation_pairs(conv)
            yield from _shuffled(rows, seed, 7919 * (file_idx + 1) + rg_idx)


def small_parquet_pairs(path: str, seed: int = 0, holdout_every: int = 0,
                        want_heldout: bool = False) -> Iterator[Pair]:
    """Reader for curated files (knowledge, gsm8k, etc.), split by row index."""
    table = pq.read_table(path)
    cols = table.column_names
    if "stimulus" in cols and "articulation" in cols:
        stimuli = table["stimulus"].to_pylist()
        answers = table["articulation"].to_pylist()
    elif "question" in cols and "answer" in cols:
        stimuli = table["question"].to_pylist()
        answers = table["answer"].to_pylist()
    elif "prompt" in cols and "response" in cols:
        stimuli = table["prompt"].to_pylist()
        answers = table["response"].to_pylist()
    else:
        raise ValueError(f"Unrecognized columns in {path}: {cols}")

    for idx in row_plan(len(stimuli), seed, holdout_every, want_heldout):
        q, a = stimuli[idx], answers[idx]
        if q and a and str(q).strip() and str(a).strip():
            yield (str(q).strip(), str(a).strip())


def cosmopedia_pairs(path: str, seed: int = 0, holdout_every: int = 0,
                     want_heldout: bool = False, max_records: int = 50000) -> Iterator[Pair]:
    """Capped Cosmopedia-v2 reader (bounded subset of HuggingFaceTB/smollm-corpus).
    Enforces a strict max_records cap to prevent unbounded disk/memory growth.
    """
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            from datasets import load_dataset
            print(f"[corpus] Downloading bounded Cosmopedia-v2 subset (max_records={max_records})...")
            ds = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
            records = []
            for i, item in enumerate(ds):
                if i >= max_records:
                    break
                text = item.get("text") or item.get("content") or ""
                prompt = item.get("prompt") or ""
                if not prompt and "\n\n" in text:
                    parts = text.split("\n\n", 1)
                    prompt, text = parts[0], parts[1]
                elif not prompt:
                    prompt = "Explain the following topic in depth:"
                if prompt.strip() and text.strip():
                    records.append({"prompt": prompt.strip(), "response": text.strip()})
            import pyarrow as pa
            table = pa.Table.from_pylist(records)
            pq.write_table(table, path)
            print(f"[corpus] Saved bounded Cosmopedia-v2 ({len(records)} records) to {path}")
        except Exception as e:
            print(f"[corpus] Warning: unable to auto-download Cosmopedia subset: {e}")
            return

    table = pq.read_table(path)
    cols = table.column_names
    if "prompt" in cols and "response" in cols:
        prompts = table["prompt"].to_pylist()
        responses = table["response"].to_pylist()
    elif "text" in cols:
        texts = table["text"].to_pylist()
        prompts, responses = [], []
        for t in texts:
            if "\n\n" in str(t):
                p, r = str(t).split("\n\n", 1)
            else:
                p, r = "Explain the following concept:", str(t)
            prompts.append(p)
            responses.append(r)
    else:
        prompts = table[cols[0]].to_pylist()
        responses = table[cols[1]].to_pylist() if len(cols) > 1 else prompts

    n = min(len(prompts), max_records)
    for idx in row_plan(n, seed, holdout_every, want_heldout):
        q, a = prompts[idx], responses[idx]
        if q and a and str(q).strip() and str(a).strip():
            yield (str(q).strip(), str(a).strip())


# --------------------------------------------------------------------------------------------
# mixture
# --------------------------------------------------------------------------------------------

def default_paths() -> Dict[str, str]:
    return {
        "openhermes": os.path.join(DATA_DIR, "openhermes", "openhermes_2_5.parquet"),
        "lmsys": os.path.join(DATA_DIR, "lmsys_chat_1m", "data"),
        "knowledge": os.path.join(DATA_DIR, "corpus", "world_knowledge_canonical_train.parquet"),
        "cognitive": os.path.join(DATA_DIR, "cerebro_cognitive", "cerebro_cognitive_train.parquet"),
        "gsm8k": os.path.join(DATA_DIR, "gsm8k", "gsm8k_train.parquet"),
        "cosmopedia": os.path.join(DATA_DIR, "cosmopedia", "cosmopedia_v2_train.parquet"),
    }


def parse_mix(spec: str) -> List[Tuple[str, float]]:
    """Parses 'openhermes=0.45,lmsys=0.35,knowledge=0.15,cognitive=0.05'."""
    parts = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"bad --mix entry {chunk!r}; expected name=weight")
        name, weight = chunk.split("=", 1)
        name = name.strip()
        if name not in SOURCES:
            raise ValueError(f"unknown source {name!r}; known: {', '.join(SOURCES)}")
        parts.append((name, float(weight)))
    total = sum(w for _, w in parts)
    if total <= 0:
        raise ValueError("--mix weights must sum to something positive")
    return [(n, w / total) for n, w in parts]


class Mixture:
    """Weighted interleave of sources with exact resume by per-source record count.

    `next()` draws a source by weight, then one record from it. Resume passes the counts
    already consumed, each source skips that many records, so the sequence continues where
    the paused run stopped rather than restarting the shuffle.
    """

    def __init__(self, specs: List[Tuple[str, float]], paths: Dict[str, str], seed: int = 0,
                 holdout_every: int = 0, want_heldout: bool = False,
                 consumed: Dict[str, int] = None,
                 holdout_sources: Iterable[str] = ("openhermes", "lmsys", "gsm8k")):
        self.seed = seed
        self.names = [n for n, _ in specs]
        self.weights = [w for _, w in specs]
        # `consumed` counts records already yielded, including the ones skipped to reach the
        # resume point, so reporting it back to the state file never loses ground.
        self.consumed = {n: int((consumed or {}).get(n, 0)) for n in self.names}
        self._iterators: Dict[str, Iterator[Pair]] = {}
        self._factories = {}

        for name in self.names:
            path = paths[name]
            if not os.path.exists(path):
                # Cosmopedia can be auto-downloaded on demand if missing
                if name == "cosmopedia":
                    cosmopedia_pairs(path)
                else:
                    raise FileNotFoundError(f"source {name!r} not found at {path}")
            # Only the sources that carry an eval slice are split.
            every = holdout_every if name in holdout_sources else 0
            if name == "openhermes":
                factory = lambda p=path, e=every: openhermes_pairs(p, seed, e, want_heldout)
            elif name == "lmsys":
                factory = lambda p=path, e=every: lmsys_pairs(p, seed, e, want_heldout)
            elif name == "cosmopedia":
                factory = lambda p=path, e=every: cosmopedia_pairs(p, seed, e, want_heldout)
            else:
                factory = lambda p=path, e=every: small_parquet_pairs(p, seed, e, want_heldout)
            self._factories[name] = factory
            self._iterators[name] = factory()

        for name in self.names:
            count = self.consumed[name]
            for _ in range(count):
                try:
                    next(self._iterators[name])
                except StopIteration:
                    # When a finite source finishes an epoch, loop it exactly as next() does
                    self._iterators[name] = self._factories[name]()
                    try:
                        next(self._iterators[name])
                    except StopIteration:
                        raise RuntimeError(f"source {name!r} is completely empty")
        # Replaying the recorded number of draws reproduces the exact source sequence, so a
        # resumed run sees the same interleaving an uninterrupted run would have seen.
        self._rng = random.Random(seed * 31 + 17)
        for _ in range(sum(self.consumed.values())):
            self._rng.choices(self.names, weights=self.weights, k=1)

    def next(self) -> Tuple[str, str, str]:
        name = self._rng.choices(self.names, weights=self.weights, k=1)[0]
        try:
            prompt, answer = next(self._iterators[name])
        except StopIteration:
            # A finished source restarts from its top so the mixture never dies mid-run.
            self._iterators[name] = self._factories[name]()
            prompt, answer = next(self._iterators[name])
        self.consumed[name] += 1
        return name, prompt, answer

    def state(self) -> Dict[str, int]:
        return dict(self.consumed)


def total_weights(specs: List[Tuple[str, float]]) -> Dict[str, float]:
    return {n: w for n, w in specs}


# --------------------------------------------------------------------------------------------
# tokenisation & provenance & leakage checks
# --------------------------------------------------------------------------------------------

def tokenize_pair(tokenizer, prompt: str, answer: str, max_prompt_len: int,
                  max_answer_tokens: int) -> Tuple[List[int], List[int]]:
    """Tokenises one record, appending EOS only when the answer is seen to its end."""
    prompt_ids = tokenizer.encode(prompt, max_length=max_prompt_len, truncation=True)
    answer_text = f" {answer.strip()}"
    answer_ids = tokenizer.encode(answer_text, max_length=max_answer_tokens, truncation=True)
    if len(answer_ids) < max_answer_tokens:
        answer_ids = answer_ids + [tokenizer.eos_token_id]
    return prompt_ids, answer_ids


def get_file_hash(path: str) -> str:
    """Computes deterministic short md5 hash of file or directory contents."""
    if not os.path.exists(path):
        return "missing"
    if os.path.isdir(path):
        files = sorted(os.listdir(path))
        return hashlib.md5("".join(files).encode()).hexdigest()[:12]
    h = hashlib.md5()
    with open(path, "rb") as f:
        chunk = f.read(65536)
        while chunk:
            h.update(chunk)
            chunk = f.read(65536)
    return h.hexdigest()[:12]


def get_dataset_provenance(paths: Dict[str, str], mix_spec: str) -> Dict[str, Any]:
    """Records dataset file hashes and the mix string for auditability in the state sidecar."""
    prov = {"mix": mix_spec, "hashes": {}}
    for name, p in paths.items():
        prov["hashes"][name] = get_file_hash(p)
    return prov


def assert_no_data_leakage(
    train_prompts: Sequence[str],
    val_prompts: Sequence[str],
    withheld_entities: Sequence[str] = (),
) -> Tuple[int, int]:
    """
    Acceptance criterion 3:
    Asserts 0 duplicated questions across train and val, and 0 withheld entities in train.
    """
    train_set = set(p.strip().lower() for p in train_prompts)
    val_set = set(p.strip().lower() for p in val_prompts)
    duplicates = train_set.intersection(val_set)
    assert len(duplicates) == 0, f"Leakage detected: {len(duplicates)} duplicated prompts across train and val!"

    entity_leaks = 0
    if withheld_entities:
        for p in train_prompts:
            p_lower = p.lower()
            for entity in withheld_entities:
                if entity.lower() in p_lower:
                    entity_leaks += 1
                    break
        assert entity_leaks == 0, f"Leakage detected: {entity_leaks} train prompts contain withheld entities!"

    return len(duplicates), entity_leaks

