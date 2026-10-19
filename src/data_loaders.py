"""
Unified Data Loaders for CerebroSpheroid Living Organism
Sensory Stimulus (Input) -> Cognitive Articulation (Output)
Zero chatbot/assistant synthetic prompt formatting tags.
"""

import os
os.environ.setdefault("HF_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".hf_cache")))

import re
import random
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
):
    """
    Streams from the specialized continuous cognitive reasoning dataset.
    Features:
    1. Multi-step quantitative deduction, rate/distance, finance, combinatorics.
    2. Causal physics, thermodynamics, optics, and biochemistry mechanisms.
    3. Formal transitive logic and algorithmic sequence tracking.
    4. Infinite on-the-fly synthesis fallback so training never terminates prematurely.
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

    batch_p, batch_t = [], []

    while True:
        if records:
            random.shuffle(records)
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
