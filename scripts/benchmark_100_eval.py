#!/usr/bin/env python3
"""
Comprehensive 100-Question Benchmark Evaluator for CerebroSpheroid V4.
Evaluates model across 7 cognitive categories:
  1. World Facts & Geography (15)
  2. Science & Nature (15)
  3. Basic Arithmetic & Math (15)
  4. Logical & Common-Sense Reasoning (15)
  5. Conversational & Assistant Tasks (15)
  6. In-Context Grounding & Extraction (15)
  7. Coding & Algorithmic Logic (10)
"""

import os
import sys
import json
import time
import argparse

os.environ["HF_HOME"] = os.path.expanduser("~/.cache/hf_home")
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/hf_home")

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.corpus import build_tokenizer
from src.spheroid_v4 import build_spheroid_v4


def load_benchmark_questions(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Benchmark file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def run_benchmark(
    checkpoint_path: str,
    questions_file: str,
    category_filter: str = None,
    limit: int = None,
    output_report: str = None,
    device: str = None,
    thinking_cycles: int = 3,
    max_new_tokens: int = 48,
    temperature: float = 0.6,
    repetition_penalty: float = 1.35,
):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    print("=" * 80)
    print(f"🧠 CEREBRO-SPHEROID V4: 100-QUESTION BENCHMARK SUITE")
    print(f"  • Checkpoint: {checkpoint_path}")
    print(f"  • Device    : {device}")
    print(f"  • Questions : {questions_file}")
    if category_filter:
        print(f"  • Filter    : Category == '{category_filter}'")
    print("=" * 80)

    # 1. Load Tokenizer & Model
    tok = build_tokenizer("llama")
    model = build_spheroid_v4("v4_small", vocab_size=len(tok))
    model.load_compatible_checkpoint(checkpoint_path, device=device)
    model.to(device=device, dtype=torch.float32)
    model.eval()

    # 2. Load Questions
    all_questions = load_benchmark_questions(questions_file)
    if category_filter:
        all_questions = [q for q in all_questions if q.get("category", "").lower() == category_filter.lower()]
    if limit and limit > 0:
        all_questions = all_questions[:limit]

    print(f"\nLoaded {len(all_questions)} questions to evaluate.\n")

    results = []
    category_stats = {}

    start_total_time = time.time()

    for idx, item in enumerate(all_questions, 1):
        qid = item["id"]
        cat = item["category"]
        q_text = item["question"]
        ref = item.get("reference", "")

        if cat not in category_stats:
            category_stats[cat] = {"count": 0, "total_tokens": 0, "total_time": 0.0}

        t0 = time.time()
        with torch.no_grad():
            ans = model.generate_text(
                q_text,
                tokenizer=tok,
                thinking_cycles=thinking_cycles,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                device=device,
                strip_bio_prefix=True,
            )
        elapsed = time.time() - t0
        clean_ans = ans.strip()

        gen_tokens = len(tok.encode(clean_ans, add_special_tokens=False))
        category_stats[cat]["count"] += 1
        category_stats[cat]["total_tokens"] += gen_tokens
        category_stats[cat]["total_time"] += elapsed

        result_entry = {
            "id": qid,
            "category": cat,
            "question": q_text,
            "reference": ref,
            "response": clean_ans,
            "tokens": gen_tokens,
            "time_sec": round(elapsed, 3),
        }
        results.append(result_entry)

        # Print live progress
        print(f"[{idx:03d}/{len(all_questions):03d}] [{cat}]")
        print(f"  Q: {q_text}")
        print(f"  A: {clean_ans}")
        if ref:
            print(f"  Ref: {ref}")
        print(f"  ⏱ {elapsed:.2f}s ({gen_tokens} tokens)")
        print("-" * 80)

    total_time = time.time() - start_total_time
    total_tokens_all = sum(r["tokens"] for r in results)

    print("\n" + "=" * 80)
    print("📊 BENCHMARK SUMMARY REPORT")
    print("=" * 80)
    print(f"Total Questions Evaluated : {len(results)}")
    print(f"Total Time Taken          : {total_time:.1f}s")
    print(f"Total Tokens Generated    : {total_tokens_all}")
    if total_time > 0:
        print(f"Average Generation Speed  : {total_tokens_all / total_time:.1f} tokens/s")
    print("\nBreakdown by Category:")
    print(f"  {'Category':<35} | {'Count':<6} | {'Avg Tokens':<10} | {'Avg Time (s)':<12}")
    print("  " + "-" * 70)
    for cat, stats in category_stats.items():
        avg_tok = stats["total_tokens"] / max(stats["count"], 1)
        avg_t = stats["total_time"] / max(stats["count"], 1)
        print(f"  {cat:<35} | {stats['count']:<6} | {avg_tok:<10.1f} | {avg_t:<12.2f}")
    print("=" * 80)

    # Save to report if requested
    if output_report:
        os.makedirs(os.path.dirname(os.path.abspath(output_report)), exist_ok=True)
        with open(output_report, "w", encoding="utf-8") as f:
            json.dump({"summary": category_stats, "results": results}, f, indent=2)
        print(f"💾 Full evaluation results saved to: {output_report}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CerebroSpheroid on 100 benchmark questions.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cerebro_spheroid_v4_latest.pt")
    parser.add_argument("--questions", type=str, default="eval/benchmark_100_questions.json")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g. cuda:0 or cpu)")
    parser.add_argument("--category", type=str, default=None, help="Filter by category")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--output", type=str, default="data/benchmark_100_results.json")
    parser.add_argument("--cycles", type=int, default=3, help="Internal thinking cycles")
    parser.add_argument("--max_tokens", type=int, default=48, help="Max new tokens")
    parser.add_argument("--temp", type=float, default=0.6, help="Temperature")
    args = parser.parse_args()

    run_benchmark(
        checkpoint_path=args.checkpoint,
        questions_file=args.questions,
        category_filter=args.category,
        limit=args.limit,
        output_report=args.output,
        device=args.device,
        thinking_cycles=args.cycles,
        max_new_tokens=args.max_tokens,
        temperature=args.temp,
    )
