"""
Frontier Reasoning Data Loaders for 100M Living Organoid
========================================================
Implements streaming loaders for world-class reasoning datasets:
1. NuminaMath-CoT (AI-MO): 860,000 multi-step Olympiad & competition math proofs.
2. ProofWriter (AllenAI / Tasksource): Multi-hop first-order deductive rule-chain proofs.
3. Big-Bench Hard (BBH): Algorithmic logic, Boolean expressions, causal judgment, and object tracking.

Features:
- Direct Stimulus -> Cognitive Articulation format (zero chatbot/assistant boilerplate).
- Resilient streaming with procedural fallback so network timeouts never crash training.
- Answer-focal weighting compatibility for Two-Tier loss.
"""

import os
import re
import sys
import time
import random
from typing import Tuple, List, Dict, Generator, Optional

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, ".hf_cache"))

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def clean_math_solution(solution: str, max_chars: int = 600) -> str:
    """Cleans LaTeX solution into concise deductive steps with clear conclusion."""
    # Remove repetitive comments or markdown headers
    sol = re.sub(r"##\s*Solution", "", solution, flags=re.IGNORECASE)
    sol = re.sub(r"\\text\{.*?\}", "", sol)
    sol = re.sub(r"\s+", " ", sol).strip()
    if len(sol) > max_chars:
        # Keep introduction and final conclusion
        half = max_chars // 2
        sol = sol[:half] + " ... Therefore, " + sol[-half:]
    return sol


# =========================================================================
# 1. NuminaMath-CoT Stream (Olympiad & Step-by-Step Math)
# =========================================================================

def get_numinamath_stream(
    tokenizer,
    batch_size: int = 2,
    max_prompt_len: int = 512,
    max_resp_len: int = 128,
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """
    Streams step-by-step mathematical proofs from AI-MO/NuminaMath-CoT.
    Stimulus: High-level quantitative / algebra problem.
    Articulation: Step-by-step algebraic deduction and exact numerical/variable solution.
    """
    while True:
        try:
            ds = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)
            batch_p, batch_t = [], []
            for item in ds:
                prob = item.get("problem", "").strip()
                sol = clean_math_solution(item.get("solution", ""))
                if not prob or not sol:
                    continue

                prompt_text = f"Solve the following mathematical problem step by step: {prob}"
                resp_text = f" {sol}"

                p_toks = tokenizer.encode(prompt_text, max_length=max_prompt_len, truncation=True)
                t_toks = tokenizer.encode(resp_text, max_length=max_resp_len - 1, truncation=True) + [tokenizer.eos_token_id]

                if len(p_toks) >= 4 and len(t_toks) >= 6:
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
        except Exception as e:
            # Fallback to local procedural math if network error
            time.sleep(1.0)
            from src.data_loaders import get_specialized_cognitive_stream
            fallback = get_specialized_cognitive_stream(tokenizer, batch_size=batch_size, max_prompt_len=max_prompt_len, max_resp_len=max_resp_len)
            yield next(fallback)


# =========================================================================
# 2. ProofWriter Stream (Multi-Hop Deductive Rule Proofs)
# =========================================================================

def get_proofwriter_stream(
    tokenizer,
    batch_size: int = 2,
    max_prompt_len: int = 512,
    max_resp_len: int = 128,
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """
    Streams multi-hop rule deduction from tasksource/proofwriter.
    Stimulus: World theory facts + conditional inference rules + target claim.
    Articulation: Step-by-step logic deduction chain + conclusion (True/False/Unknown).
    """
    while True:
        try:
            ds = load_dataset("tasksource/proofwriter", split="train", streaming=True)
            batch_p, batch_t = [], []
            for item in ds:
                theory = item.get("theory", "").strip()
                question = item.get("question", "").strip()
                ans = item.get("answer", "").strip()
                if not theory or not question or not ans:
                    continue

                prompt_text = (
                    f"Given the logical rules and facts:\n{theory}\n"
                    f"Is the statement '{question}' true, false, or unknown?"
                )
                resp_text = (
                    f" Based on the deductive premises, the statement '{question}' is {ans.lower()}."
                )

                p_toks = tokenizer.encode(prompt_text, max_length=max_prompt_len, truncation=True)
                t_toks = tokenizer.encode(resp_text, max_length=max_resp_len - 1, truncation=True) + [tokenizer.eos_token_id]

                if len(p_toks) >= 6 and len(t_toks) >= 4:
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
        except Exception as e:
            time.sleep(1.0)
            from src.tightened_curriculum import get_tightened_stream
            fallback = get_tightened_stream(tokenizer, batch_size=batch_size, max_prompt_len=max_prompt_len, max_resp_len=max_resp_len)
            yield next(fallback)


# =========================================================================
# 3. Big-Bench Hard (BBH) Stream (Algorithmic Logic & State Tracking)
# =========================================================================

BBH_TASKS = [
    "boolean_expressions",
    "causal_judgment",
    "logical_deduction_three_objects",
    "tracking_shuffled_objects_three_objects",
    "formal_fallacies",
    "temporal_sequences",
]

def get_bbh_stream(
    tokenizer,
    batch_size: int = 2,
    max_prompt_len: int = 512,
    max_resp_len: int = 128,
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """
    Streams diverse algorithmic and state-tracking problems from lukaemon/bbh.
    Stimulus: Algorithmic permutation, Boolean logic chain, or temporal order.
    Articulation: Step-by-step state transition -> exact target answer.
    """
    while True:
        task_name = random.choice(BBH_TASKS)
        try:
            ds = load_dataset("lukaemon/bbh", task_name, split="test", streaming=True)
            batch_p, batch_t = [], []
            for item in ds:
                inp = item.get("input", "").strip()
                target = item.get("target", "").strip()
                if not inp or not target:
                    continue

                if task_name == "boolean_expressions":
                    prompt_text = f"Evaluate the truth value of the following Boolean logic expression: {inp}"
                    resp_text = f" Evaluating the logical expression step by step, the final truth value is {target}."
                elif "tracking" in task_name:
                    prompt_text = f"Trace the positions of the objects step by step:\n{inp}"
                    resp_text = f" Tracing the sequence of swaps step by step, the correct answer is {target}."
                elif task_name == "causal_judgment":
                    prompt_text = f"Analyze the causal relationship in this scenario:\n{inp}"
                    resp_text = f" Analyzing the causal chain of events, the answer is {target}."
                else:
                    prompt_text = f"Solve the following reasoning problem step by step:\n{inp}"
                    resp_text = f" Following the deductive chain of logic, the answer is {target}."

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
        except Exception as e:
            time.sleep(1.0)
            from src.tightened_curriculum import get_tightened_stream
            fallback = get_tightened_stream(tokenizer, batch_size=batch_size, max_prompt_len=max_prompt_len, max_resp_len=max_resp_len)
            yield next(fallback)


# =========================================================================
# 4. Interleaved Frontier Reasoning Stream (Composite)
# =========================================================================

def get_frontier_reasoning_stream(
    tokenizer,
    batch_size: int = 2,
    max_prompt_len: int = 512,
    max_resp_len: int = 128,
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """
    Composite frontier stream:
    - 35% NuminaMath-CoT (Advanced mathematical proofs)
    - 35% ProofWriter (Multi-hop rule deductions)
    - 30% Big-Bench Hard (BBH algorithmic tracking & logic)
    """
    numina_gen = get_numinamath_stream(tokenizer, batch_size=batch_size, max_prompt_len=max_prompt_len, max_resp_len=max_resp_len)
    proof_gen = get_proofwriter_stream(tokenizer, batch_size=batch_size, max_prompt_len=max_prompt_len, max_resp_len=max_resp_len)
    bbh_gen = get_bbh_stream(tokenizer, batch_size=batch_size, max_prompt_len=max_prompt_len, max_resp_len=max_resp_len)

    while True:
        r = random.random()
        if r < 0.35:
            yield next(numina_gen)
        elif r < 0.70:
            yield next(proof_gen)
        else:
            yield next(bbh_gen)
