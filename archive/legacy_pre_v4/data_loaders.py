"""
Unified Data Loaders for CerebroSpheroid Living Organism
Sensory Stimulus (Input) -> Cognitive Articulation (Output)
Zero chatbot/assistant synthetic prompt formatting tags.
"""

import os
os.environ.setdefault("HF_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".hf_cache")))

import re
import random
from typing import Optional
import pyarrow.parquet as pq
import torch
from datasets import load_dataset


def clean_organism_response(text: str) -> str:
    """Removes canned chatbot/assistant boilerplate so the living model thinks as an autonomous cognitive entity."""
    # Strip opening assistant sycophancy and robotic disclaimers
    text = re.sub(
        r"^(?:As an AI language model|As an AI|As an artificial intelligence|I am an AI assistant|I am an AI|I'm an AI|As a helpful assistant)[,\s.:;-]+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^(?:Sure(?: thing)?[!,.]|Certainly[!,.]|Of course[!,.]|I would be happy to help[!,.]|Hello! How can I help you today\?)[ \t]*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


def get_lmsys_stream(data_dir, tokenizer, batch_size=2, max_prompt_len=1024, max_resp_len=128):
    """
    Streams natural environmental language stimuli from LMSYS-Chat-1M without chatbot tags.
    Direct sensory mapping:
      Stimulus      -> {user_query}
      Articulation  -> {direct_knowledge_response}
    """
    parquet_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")])
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    while True:
        random.shuffle(parquet_files)
        for pfile in parquet_files:
            table = pq.read_table(pfile, columns=["language", "conversation"])
            num_rows = table.num_rows
            row_indices = list(range(num_rows))
            random.shuffle(row_indices)

            batch_p, batch_t = [], []
            for idx in row_indices:
                row = table.slice(idx, 1).to_pydict()
                if row["language"][0] != "English":
                    continue

                conv = row["conversation"][0]
                if len(conv) < 2:
                    continue

                for i in range(len(conv) - 1):
                    msg_u = conv[i]
                    msg_a = conv[i + 1]
                    if msg_u.get("role") == "user" and msg_a.get("role") == "assistant":
                        u_text = msg_u.get("content", "").strip()
                        a_text = msg_a.get("content", "").strip()
                        if not u_text or not a_text:
                            continue

                        clean_a = clean_organism_response(a_text)
                        if not clean_a:
                            continue

                        prompt_text = u_text
                        resp_text = f" {clean_a}"

                        p_toks = tokenizer.encode(prompt_text, max_length=max_prompt_len, truncation=True)
                        t_toks = tokenizer.encode(resp_text, max_length=max_resp_len - 1, truncation=True) + [tokenizer.eos_token_id]

                        if len(p_toks) >= 4 and len(t_toks) >= 4:
                            batch_p.append(p_toks)
                            batch_t.append(t_toks)

                            if len(batch_p) == batch_size:
                                max_p = max(len(p) for p in batch_p)
                                max_t = max(len(t) for t in batch_t)
                                p_pad = [p + [tokenizer.eos_token_id] * (max_p - len(p)) for p in batch_p]
                                t_pad = [t + [-100] * (max_t - len(t)) for t in batch_t]
                                yield (
                                    torch.tensor(p_pad, dtype=torch.long),
                                    torch.tensor(t_pad, dtype=torch.long)
                                )
                                batch_p, batch_t = [], []


def clean_gsm8k_solution(raw_solution: str) -> str:
    """Cleans GSM8K chain-of-thought calculation tags and formats clear deductive steps."""
    cleaned = re.sub(r"<<.*?>>", "", raw_solution)
    cleaned = re.sub(r"####\s*(\S+)", r"The answer is \1.", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def get_gsm8k_stream(tokenizer, batch_size=2, max_prompt_len=1024, max_resp_len=128):
    """
    Streams multi-step math reasoning problems from Hugging Face GSM8K.
    Natural stimulus mapping:
      Stimulus      -> {question}
      Articulation  -> {step_by_step_solution}
    """
    while True:
        ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
        batch_p, batch_t = [], []
        for item in ds:
            q = item.get("question", "").strip()
            a = clean_gsm8k_solution(item.get("answer", ""))
            if not q or not a:
                continue

            prompt_text = q
            resp_text = f" {a}"

            p_toks = tokenizer.encode(prompt_text, max_length=max_prompt_len, truncation=True)
            t_toks = tokenizer.encode(resp_text, max_length=max_resp_len - 1, truncation=True) + [tokenizer.eos_token_id]

            if len(p_toks) >= 4 and len(t_toks) >= 4:
                batch_p.append(p_toks)
                batch_t.append(t_toks)

                if len(batch_p) == batch_size:
                    max_p = max(len(p) for p in batch_p)
                    max_t = max(len(t) for t in batch_t)
                    p_pad = [p + [tokenizer.eos_token_id] * (max_p - len(p)) for p in batch_p]
                    t_pad = [t + [-100] * (max_t - len(t)) for t in batch_t]
                    yield (
                        torch.tensor(p_pad, dtype=torch.long),
                        torch.tensor(t_pad, dtype=torch.long)
                    )
                    batch_p, batch_t = [], []


def get_svamp_stream(tokenizer, batch_size=2, max_prompt_len=1024, max_resp_len=128):
    """
    Streams adversarial arithmetic problems from SVAMP with diverse templates.
    Natural stimulus mapping:
      Stimulus      -> {question}
      Articulation  -> {thought_solution}
    """
    ds = load_dataset("ChilleD/SVAMP", split="train")
    records = list(ds)
    templates = [
        "The result of this calculation is {ans}.",
        "After calculating the steps, the answer is {ans}.",
        "Solving the problem yields {ans}.",
        "The final total is {ans}.",
        "The answer is {ans}.",
    ]
    while True:
        random.shuffle(records)
        batch_p, batch_t = [], []
        for r in records:
            q = r["question_concat"].strip()
            ans = str(r["Answer"]).strip()
            prompt_text = q
            resp_text = f" {random.choice(templates).format(ans=ans)}"

            p_toks = tokenizer.encode(prompt_text, max_length=max_prompt_len, truncation=True)
            t_toks = tokenizer.encode(resp_text, max_length=max_resp_len - 1, truncation=True) + [tokenizer.eos_token_id]

            if len(p_toks) >= 4 and len(t_toks) >= 3:
                batch_p.append(p_toks)
                batch_t.append(t_toks)

                if len(batch_p) == batch_size:
                    max_p = max(len(p) for p in batch_p)
                    max_t = max(len(t) for t in batch_t)
                    p_pad = [p + [tokenizer.eos_token_id] * (max_p - len(p)) for p in batch_p]
                    t_pad = [t + [-100] * (max_t - len(t)) for t in batch_t]
                    yield (
                        torch.tensor(p_pad, dtype=torch.long),
                        torch.tensor(t_pad, dtype=torch.long)
                    )
                    batch_p, batch_t = [], []


def get_hybrid_reasoning_stream(
    lmsys_dir: str,
    tokenizer,
    batch_size: int = 2,
    reasoning_ratio: float = 0.70,
    max_prompt_len: int = 1024,
    max_resp_len: int = 128,
):
    """
    Interleaved Dual-Curriculum Stream:
    - 70% Rigorous Step-by-Step Mathematical & Deductive Logic (85% GSM8K + 15% SVAMP)
    All strictly mapped to direct natural stimulus -> articulation pairs without chatbot role tags.
    """
    gsm8k_stream = get_gsm8k_stream(tokenizer, batch_size=batch_size, max_prompt_len=max_prompt_len, max_resp_len=max_resp_len)
    svamp_stream = get_svamp_stream(tokenizer, batch_size=batch_size, max_prompt_len=max_prompt_len, max_resp_len=max_resp_len)
    lmsys_stream = get_lmsys_stream(lmsys_dir, tokenizer, batch_size=batch_size, max_prompt_len=max_prompt_len, max_resp_len=max_resp_len)

    while True:
        if random.random() < reasoning_ratio:
            if random.random() < 0.85:
                yield next(gsm8k_stream)
            else:
                yield next(svamp_stream)
        else:
            yield next(lmsys_stream)


def get_specialized_cognitive_stream(
    tokenizer,
    batch_size: int = 2,
    parquet_path: str = None,
    max_prompt_len: int = 1024,
    max_resp_len: int = 128,
    shuffle_seed: Optional[int] = None,
    allow_synthesis_fallback: bool = True,
):
    """
    Streams from the specialized continuous cognitive reasoning dataset.
    Features:
    1. Multi-step quantitative deduction, rate/distance, finance, combinatorics.
    2. Causal physics, thermodynamics, optics, and biochemistry mechanisms.
    3. Formal transitive logic and algorithmic sequence tracking.
    4. Infinite on-the-fly synthesis fallback so training never terminates prematurely.

    `shuffle_seed` draws epoch orderings from a private `random.Random` instead of the global
    one, so constructing the stream neither depends on nor disturbs process RNG state, and the
    first batches it yields are reproducible from the seed alone. Leave it None for training
    (fresh ordering every run) and set it for anything whose number is going to be compared
    across evaluations -- an unseeded eval stream measures a different set each time.

    `allow_synthesis_fallback=False` makes a missing file an error rather than an infinite
    stream of freshly invented samples. A validation set that silently synthesises is not
    validating the file it claims to.
    """
    from src.generate_specialized_dataset import generate_single_cognitive_sample

    if parquet_path is None:
        parquet_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "cerebro_cognitive", "cerebro_cognitive_train.parquet")
        )

    records = []
    if os.path.exists(parquet_path):
        table = pq.read_table(parquet_path)
        stimuli = table["stimulus"].to_pylist()
        articulations = table["articulation"].to_pylist()
        records = list(zip(stimuli, articulations))
    elif not allow_synthesis_fallback:
        raise FileNotFoundError(
            f"no curriculum file at {parquet_path}; synthesis fallback is disabled because "
            f"this stream is meant to be a fixed evaluation set"
        )

    shuffle_rng = random.Random(shuffle_seed) if shuffle_seed is not None else random

    batch_p, batch_t = [], []

    while True:
        if records:
            shuffle_rng.shuffle(records)
            for q, a in records:
                prompt_text = q
                resp_text = f" {a}"

                p_toks = tokenizer.encode(prompt_text, max_length=max_prompt_len, truncation=True)
                t_toks = tokenizer.encode(resp_text, max_length=max_resp_len - 1, truncation=True) + [tokenizer.eos_token_id]

                if len(p_toks) >= 4 and len(t_toks) >= 4:
                    batch_p.append(p_toks)
                    batch_t.append(t_toks)

                    if len(batch_p) == batch_size:
                        max_p = max(len(p) for p in batch_p)
                        max_t = max(len(t) for t in batch_t)
                        p_pad = [p + [tokenizer.eos_token_id] * (max_p - len(p)) for p in batch_p]
                        t_pad = [t + [-100] * (max_t - len(t)) for t in batch_t]
                        yield (
                            torch.tensor(p_pad, dtype=torch.long),
                            torch.tensor(t_pad, dtype=torch.long)
                        )
                        batch_p, batch_t = [], []
        else:
            # Infinite on-the-fly procedural synthesis
            q, a = generate_single_cognitive_sample()
            prompt_text = q
            resp_text = f" {a}"

            p_toks = tokenizer.encode(prompt_text, max_length=max_prompt_len, truncation=True)
            t_toks = tokenizer.encode(resp_text, max_length=max_resp_len - 1, truncation=True) + [tokenizer.eos_token_id]

            if len(p_toks) >= 4 and len(t_toks) >= 4:
                batch_p.append(p_toks)
                batch_t.append(t_toks)

                if len(batch_p) == batch_size:
                    max_p = max(len(p) for p in batch_p)
                    max_t = max(len(t) for t in batch_t)
                    p_pad = [p + [tokenizer.eos_token_id] * (max_p - len(p)) for p in batch_p]
                    t_pad = [t + [-100] * (max_t - len(t)) for t in batch_t]
                    yield (
                        torch.tensor(p_pad, dtype=torch.long),
                        torch.tensor(t_pad, dtype=torch.long)
                    )
                    batch_p, batch_t = [], []


get_cerebro_cognitive_stream = get_specialized_cognitive_stream


def get_cognitive_and_language_stream(
    tokenizer,
    lmsys_dir: str = None,
    batch_size: int = 2,
    cognitive_ratio: float = 0.50,
    max_prompt_len: int = 512,
    max_resp_len: int = 128,
):
    """
    Interleaved Cognitive Reasoning + Real-World Natural Language Stream.
    1. 50% Specialized Synthetic Cognitive Reasoning (Math, Rates, Geometry, Physics, Logic)
    2. 50% Clean LMSYS-Chat-1M Environmental Discourse (Natural Grammar, General Knowledge, Science)
    Guarantees both deep algorithmic reasoning and human-grade grammatical syntax.
    """
    if lmsys_dir is None:
        lmsys_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "lmsys_chat_1m", "data")
        )

    cognitive_stream = get_specialized_cognitive_stream(
        tokenizer, batch_size=batch_size, max_prompt_len=max_prompt_len, max_resp_len=max_resp_len
    )
    lmsys_stream = get_lmsys_stream(
        lmsys_dir, tokenizer, batch_size=batch_size, max_prompt_len=max_prompt_len, max_resp_len=max_resp_len
    )

    while True:
        if random.random() < cognitive_ratio:
            yield next(cognitive_stream)
        else:
            yield next(lmsys_stream)


def get_world_knowledge_stream(
    tokenizer,
    batch_size: int = 2,
    parquet_path: Optional[str] = None,
    max_prompt_len: int = 1024,
    max_resp_len: int = 256,
    shuffle_seed: Optional[int] = None,
):
    """
    Streams high-density world knowledge, factual QA, geography, capitals, and science facts.
    """
    if parquet_path is None:
        parquet_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "general_knowledge", "world_knowledge_train.parquet")
        )

    if not os.path.exists(parquet_path):
        from src.generate_world_knowledge import main as gen_main
        gen_main()

    table = pq.read_table(parquet_path)
    stimuli = table["stimulus"].to_pylist()
    articulations = table["articulation"].to_pylist()
    records = list(zip(stimuli, articulations))

    shuffle_rng = random.Random(shuffle_seed) if shuffle_seed is not None else random
    batch_p, batch_t = [], []

    while True:
        shuffle_rng.shuffle(records)
        for q, a in records:
            p_toks = tokenizer.encode(q.strip(), max_length=max_prompt_len, truncation=True)
            resp_text = f" {a.strip()}"
            t_toks = tokenizer.encode(resp_text, max_length=max_resp_len - 1, truncation=True) + [tokenizer.eos_token_id]

            if len(p_toks) >= 2 and len(t_toks) >= 2:
                batch_p.append(p_toks)
                batch_t.append(t_toks)

                if len(batch_p) == batch_size:
                    max_p = max(len(p) for p in batch_p)
                    max_t = max(len(t) for t in batch_t)
                    p_pad = [p + [tokenizer.eos_token_id] * (max_p - len(p)) for p in batch_p]
                    t_pad = [t + [-100] * (max_t - len(t)) for t in batch_t]
                    yield (
                        torch.tensor(p_pad, dtype=torch.long),
                        torch.tensor(t_pad, dtype=torch.long),
                    )
                    batch_p, batch_t = [], []


def get_openhermes_stream(
    tokenizer,
    parquet_path: Optional[str] = None,
    batch_size: int = 1,
    max_prompt_len: int = 512,
    max_resp_len: int = 192,
    shuffle_seed: Optional[int] = None,
):
    """
    Data-Level Distillation Stream from Teknium OpenHermes-2.5 (1M high-quality distilled traces).
    Extracts direct reasoning traces:
      Stimulus      -> {human instruction}
      Articulation  -> {distilled frontier model response}
    Zero chatbot boilerplate, pure cognitive distillation.
    """
    if parquet_path is None:
        parquet_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "openhermes", "openhermes_2_5.parquet")
        )

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"OpenHermes parquet not found at: {parquet_path}")

    parquet_file = pq.ParquetFile(parquet_path)
    num_row_groups = parquet_file.num_row_groups
    shuffle_rng = random.Random(shuffle_seed)

    while True:
        group_indices = list(range(num_row_groups))
        shuffle_rng.shuffle(group_indices)

        for rg_idx in group_indices:
            rg = parquet_file.read_row_group(rg_idx)
            cols = rg.column_names

            # Extract conversations / instructions
            rows = []
            if "conversations" in cols:
                convs_col = rg["conversations"].to_pylist()
                for conv in convs_col:
                    if not conv or not isinstance(conv, list):
                        continue
                    # Find human -> gpt turn
                    for i in range(len(conv) - 1):
                        msg_u = conv[i]
                        msg_a = conv[i + 1]
                        u_role = msg_u.get("from") or msg_u.get("role")
                        a_role = msg_a.get("from") or msg_a.get("role")
                        if u_role in ("human", "user") and a_role in ("gpt", "assistant", "model"):
                            u_text = msg_u.get("value") or msg_u.get("content") or ""
                            a_text = msg_a.get("value") or msg_a.get("content") or ""
                            if u_text.strip() and a_text.strip():
                                rows.append((u_text.strip(), a_text.strip()))
            elif "instruction" in cols and "output" in cols:
                inst_col = rg["instruction"].to_pylist()
                out_col = rg["output"].to_pylist()
                in_col = rg["input"].to_pylist() if "input" in cols else [""] * len(inst_col)
                for inst, inp, out in zip(inst_col, in_col, out_col):
                    p = f"{inst.strip()} {inp.strip()}".strip() if inp else inst.strip()
                    if p and out and out.strip():
                        rows.append((p, out.strip()))

            shuffle_rng.shuffle(rows)
            batch_p, batch_t = [], []

            for q, a in rows:
                clean_a = clean_organism_response(a)
                if not clean_a:
                    continue

                p_toks = tokenizer.encode(q, max_length=max_prompt_len, truncation=True)
                resp_text = f" {clean_a}"
                t_toks = tokenizer.encode(resp_text, max_length=max_resp_len - 1, truncation=True) + [tokenizer.eos_token_id]

                if len(p_toks) >= 4 and len(t_toks) >= 4:
                    batch_p.append(p_toks)
                    batch_t.append(t_toks)

                    if len(batch_p) == batch_size:
                        max_p = max(len(p) for p in batch_p)
                        max_t = max(len(t) for t in batch_t)
                        p_pad = [p + [tokenizer.eos_token_id] * (max_p - len(p)) for p in batch_p]
                        t_pad = [t + [-100] * (max_t - len(t)) for t in batch_t]
                        yield (
                            torch.tensor(p_pad, dtype=torch.long),
                            torch.tensor(t_pad, dtype=torch.long),
                        )
                        batch_p, batch_t = [], []


def get_general_llm_stream(
    tokenizer,
    batch_size: int = 2,
    max_prompt_len: int = 1024,
    max_resp_len: int = 256,
    lmsys_dir: Optional[str] = None,
    openhermes_path: Optional[str] = None,
    lmsys_ratio: float = 0.45,
    knowledge_ratio: float = 0.35,
    shuffle_seed: Optional[int] = None,
):
    """
    Unified General LLM + World Knowledge Curriculum Stream:
    1. 45% LMSYS-Chat-1M / OpenHermes real human-LLM conversational discourse
    2. 35% World Knowledge & Factual QA (geography, capitals, landmarks, science, nature)
    3. 20% Specialized Cognitive Reasoning (cerebro_cognitive + GSM8K)
    Guarantees both encyclopedic factual knowledge (e.g. Paris as capital of France) and deep logic.
    """
    if openhermes_path is None:
        openhermes_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "openhermes", "openhermes_2_5.parquet")
        )

    # Use OpenHermes if downloaded, else fallback to LMSYS
    if os.path.exists(openhermes_path):
        llm_stream = get_openhermes_stream(
            tokenizer,
            parquet_path=openhermes_path,
            batch_size=batch_size,
            max_prompt_len=max_prompt_len,
            max_resp_len=max_resp_len,
            shuffle_seed=shuffle_seed,
        )
    else:
        if lmsys_dir is None:
            lmsys_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "data", "lmsys_chat_1m", "data")
            )
        llm_stream = get_lmsys_stream(
            lmsys_dir, tokenizer, batch_size=batch_size, max_prompt_len=max_prompt_len, max_resp_len=max_resp_len
        )

    knowledge_stream = get_world_knowledge_stream(
        tokenizer, batch_size=batch_size, max_prompt_len=max_prompt_len, max_resp_len=max_resp_len, shuffle_seed=shuffle_seed
    )
    cognitive_stream = get_specialized_cognitive_stream(
        tokenizer, batch_size=batch_size, max_prompt_len=max_prompt_len, max_resp_len=max_resp_len
    )

    while True:
        r = random.random()
        if r < lmsys_ratio:
            yield next(llm_stream)
        elif r < (lmsys_ratio + knowledge_ratio):
            yield next(knowledge_stream)
        else:
            yield next(cognitive_stream)


# Re-export frontier reasoning loaders
try:
    from src.frontier_reasoning_loaders import (
        get_numinamath_stream,
        get_proofwriter_stream,
        get_bbh_stream,
        get_frontier_reasoning_stream,
    )
except ImportError:
    pass

