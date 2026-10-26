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

import os
import random
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import pyarrow.parquet as pq

Pair = Tuple[str, str]

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")

SOURCES = ("openhermes", "lmsys", "knowledge", "cognitive")


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
    """Reader for the small curated files, split by row index."""
    table = pq.read_table(path)
    stimuli = table["stimulus"].to_pylist()
    answers = table["articulation"].to_pylist()
    for idx in row_plan(len(stimuli), seed, holdout_every, want_heldout):
        q, a = stimuli[idx], answers[idx]
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
                 holdout_sources: Iterable[str] = ("openhermes", "lmsys")):
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
                raise FileNotFoundError(f"source {name!r} not found at {path}")
            # Only the sources that carry an eval slice are split. The knowledge file is
            # already entity-disjoint from its val file, so holding out more rows there
            # would shrink the smallest source for no measurement it does not already have.
            every = holdout_every if name in holdout_sources else 0
            if name == "openhermes":
                factory = lambda p=path, e=every: openhermes_pairs(p, seed, e, want_heldout)
            elif name == "lmsys":
                factory = lambda p=path, e=every: lmsys_pairs(p, seed, e, want_heldout)
            else:
                factory = lambda p=path, e=every: small_parquet_pairs(p, seed, e, want_heldout)
            self._factories[name] = factory
            self._iterators[name] = factory()

        for name, iterator in self._iterators.items():
            for _ in range(self.consumed[name]):
                try:
                    next(iterator)
                except StopIteration:
                    raise RuntimeError(
                        f"state file says {self.consumed[name]:,} records were consumed from "
                        f"{name!r}, but the source holds fewer today. The corpus changed since "
                        f"that state was written; start a fresh run."
                    )
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
# tokenisation
# --------------------------------------------------------------------------------------------

def tokenize_pair(tokenizer, prompt: str, answer: str, max_prompt_len: int,
                  max_answer_tokens: int) -> Tuple[List[int], List[int]]:
    """Tokenises one record, appending EOS only when the answer is seen to its end.

    The previous loaders appended EOS to every answer, including answers cut off at the
    cap. That teaches the model to end a sentence at an arbitrary token, which shows up as
    the short template-shaped replies in the chat logs. A truncated answer simply has no
    EOS; the model only learns to stop where the corpus stops.
    """
    prompt_ids = tokenizer.encode(prompt, max_length=max_prompt_len, truncation=True)
    answer_text = f" {answer.strip()}"
    answer_ids = tokenizer.encode(answer_text, max_length=max_answer_tokens, truncation=True)
    if len(answer_ids) < max_answer_tokens:
        answer_ids = answer_ids + [tokenizer.eos_token_id]
    return prompt_ids, answer_ids
