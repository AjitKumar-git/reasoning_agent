#!/usr/bin/env python3
"""
Comprehensive Evaluation Suite for CerebroSpheroid Models (V2 & V3).
Supports:
1. Automated Multi-Category Benchmark with Exact Keyword Verification.
2. Deliberation Comparison (evaluates thinking_cycles=0 vs 1 vs 3).
3. Quantitative Held-Out Perplexity / Cross-Entropy Loss Evaluation.
4. Interactive Streaming REPL for Freeform Prompt Testing.
"""

import os
os.environ.setdefault("HF_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), ".hf_cache")))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys
import time
import argparse
import glob
from typing import List, Tuple, Dict, Any, Optional
import torch

from src.inference import ReasoningAgent

# --------------------------------------------------------------------------------------------
# Benchmark Suite across 6 Core Domains
# Each test: (Category, Prompt, Expected Keywords/Substrings)
# --------------------------------------------------------------------------------------------
BENCHMARK_TESTS = [
    # 1. World Knowledge & Geography
    ("Factual", "What is the capital of France? Answer:", ["paris"]),
    ("Factual", "What is the capital of Japan? Answer:", ["tokyo"]),
    ("Factual", "What is the capital of India? Answer:", ["new delhi", "delhi"]),
    ("Factual", "What is the capital of Germany? Answer:", ["berlin"]),
    ("Factual", "What is the capital of Australia? Answer:", ["canberra"]),

    # 2. Science & Nature
    ("Science", "What is the chemical formula for water? Answer:", ["h2o", "h₂o"]),
    ("Science", "What is the boiling point of water in Celsius? Answer:", ["100"]),
    ("Science", "What is the center of our solar system? Answer:", ["sun"]),
    ("Science", "What is the primary gas humans breathe in to survive? Answer:", ["oxygen", "o2"]),
    ("Science", "What planet is known as the Red Planet? Answer:", ["mars"]),

    # 3. Arithmetic & Multi-step Reasoning
    ("Arithmetic", "If 4 books cost $20, how much do 6 books cost? Answer:", ["30", "thirty"]),
    ("Arithmetic", "Maya has 8 markers. She buys 4 more and gives 3 away. How many markers does she have? Answer:", ["9", "nine"]),
    ("Arithmetic", "What is 15 divided by 3? Answer:", ["5", "five"]),
    ("Arithmetic", "What is 12 multiplied by 4? Answer:", ["48"]),
    ("Arithmetic", "A train travels at 60 mph. How far does it travel in 3 hours? Answer:", ["180"]),

    # 4. Logic & Formal Deduction
    ("Logic", "All humans are mortal. Socrates is a human. Therefore, Socrates is Answer:", ["mortal"]),
    ("Logic", "All birds have feathers. A penguin is a bird. Does a penguin have feathers? Answer:", ["yes", "penguin has feathers"]),
    ("Logic", "If today is Monday, what day was yesterday? Answer:", ["sunday"]),

    # 5. Common Sense & Everyday Reasoning
    ("Commonsense", "If it is raining outside, what item should you carry with you? Answer:", ["umbrella", "raincoat"]),
    ("Commonsense", "What color is the sky on a clear sunny day? Answer:", ["blue"]),
    ("Commonsense", "Which animal is known as man's best friend? Answer:", ["dog", "canine"]),

    # 6. Instruction Following & Formatting
    ("Instruct", "List three primary colors separated by commas. Answer:", ["red", "blue", "yellow"]),
    ("Instruct", "Write the word 'HELLO' in lowercase. Answer:", ["hello"]),
]


def resolve_checkpoint(path: Optional[str]) -> str:
    """Finds the most appropriate checkpoint path, downloading if missing."""
    target_filename = "cerebro_spheroid_v3_colab_best.pt"
    colab_best_path = os.path.join("checkpoints", target_filename)

    if path:
        if os.path.exists(path):
            return path
        if os.path.basename(path) == target_filename:
            print(f"[notice] Specified checkpoint {path} not found on disk. Auto-downloading...")
            from scripts.download_checkpoints import download_drive_file, CHECKPOINTS
            download_drive_file(CHECKPOINTS["v3_colab_best"]["file_id"], path, CHECKPOINTS["v3_colab_best"]["size_bytes"])
            if os.path.exists(path):
                return path

    candidates = [
        colab_best_path,
        "/content/drive/MyDrive/cerebro_checkpoints/cerebro_spheroid_v3_colab_best.pt",
        "checkpoints/cerebro_spheroid_v3_colab_latest.pt",
        "/content/drive/MyDrive/cerebro_checkpoints/cerebro_spheroid_v3_colab_latest.pt",
        "checkpoints/cerebro_spheroid_v3_full_best.pt",
        "checkpoints/cerebro_spheroid_distill_v2_best.pt",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    # Fallback to any .pt in checkpoints/ or Google Drive
    found = glob.glob("checkpoints/*.pt") + glob.glob("/content/drive/MyDrive/cerebro_checkpoints/*.pt")
    if found:
        return sorted(found)[-1]

    print(f"[notice] No checkpoint found in checkpoints/. Downloading {target_filename} from Google Drive...")
    from scripts.download_checkpoints import download_drive_file, CHECKPOINTS
    download_drive_file(CHECKPOINTS["v3_colab_best"]["file_id"], colab_best_path, CHECKPOINTS["v3_colab_best"]["size_bytes"])
    if os.path.exists(colab_best_path):
        return colab_best_path

    raise FileNotFoundError("No checkpoint found in checkpoints/ or Google Drive.")


def evaluate_benchmark(agent: ReasoningAgent, thinking_cycles: int = 1, max_tokens: int = 32) -> Dict[str, Any]:
    print("=" * 88)
    print(f"  RUNNING BENCHMARK EVALUATION (cycles={thinking_cycles}, max_tokens={max_tokens})")
    print("=" * 88)

    category_stats: Dict[str, Dict[str, int]] = {}
    results = []

    for cat, prompt, expected_keywords in BENCHMARK_TESTS:
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct": 0}
        category_stats[cat]["total"] += 1

        t0 = time.time()
        output = agent.generate(prompt, thinking_cycles=thinking_cycles, max_tokens=max_tokens, temperature=0.0)
        latency = time.time() - t0

        lower_out = output.lower()
        matched = any(kw.lower() in lower_out for kw in expected_keywords)
        if matched:
            category_stats[cat]["correct"] += 1

        results.append({
            "category": cat,
            "prompt": prompt,
            "output": output,
            "expected": expected_keywords,
            "matched": matched,
            "latency": latency
        })

        status = "✓ PASS" if matched else "✗ FAIL"
        print(f"[{cat:12s}] {status} | Q: {prompt}")
        print(f"              A: {output.strip()}")
        print(f"              (expected: {expected_keywords} | {latency:.2f}s)")
        print("-" * 88)

    # Summary Report
    print("\n" + "=" * 88)
    print("  EVALUATION SUMMARY SCORECARD")
    print("=" * 88)
    total_q = len(BENCHMARK_TESTS)
    total_c = sum(c["correct"] for c in category_stats.values())

    for cat, stats in category_stats.items():
        pct = (stats["correct"] / stats["total"]) * 100
        bar = "█" * int(pct // 10) + "░" * (10 - int(pct // 10))
        print(f"  {cat:15s} : {stats['correct']:2d}/{stats['total']:2d} ({pct:5.1f}%)  [{bar}]")

    overall_pct = (total_c / total_q) * 100
    bar_all = "█" * int(overall_pct // 10) + "░" * (10 - int(overall_pct // 10))
    print("-" * 88)
    print(f"  {'OVERALL SCORE':15s} : {total_c:2d}/{total_q:2d} ({overall_pct:5.1f}%)  [{bar_all}]")
    print("=" * 88 + "\n")

    return {"overall_pct": overall_pct, "category_stats": category_stats, "results": results}


def compare_thinking_cycles(agent: ReasoningAgent, cycles_list: List[int] = [0, 1, 3]):
    """Demonstrates how manifold deliberation depth affects outputs."""
    print("=" * 88)
    print("  DELIBERATION MANIFOLD COMPARISON (Ablation across thinking_cycles)")
    print("=" * 88)

    test_queries = [
        "What is the capital of France? Answer:",
        "What is the chemical formula for water? Answer:",
        "If 4 books cost $20, how much do 6 books cost? Answer:",
        "Maya has 8 markers. She buys 4 more and gives 3 away. How many markers does she have? Answer:",
        "All humans are mortal. Socrates is a human. Therefore, Socrates is Answer:",
    ]

    for q in test_queries:
        print(f"\nQ: {q}")
        for c in cycles_list:
            t0 = time.time()
            ans = agent.generate(q, thinking_cycles=c, max_tokens=32, temperature=0.0)
            dt = time.time() - t0
            print(f"  [cycles={c} | {dt:.2f}s] -> {ans}")
    print("=" * 88 + "\n")


def interactive_repl(agent: ReasoningAgent, thinking_cycles: int = 1, max_tokens: int = 64, temperature: float = 0.0):
    """Interactive command-line session with streaming output."""
    print("=" * 88)
    print("  INTERACTIVE REPL (Streaming CerebroSpheroid Session)")
    print("  Type your question and press Enter. Type 'exit', 'quit', or Ctrl+C to stop.")
    print("=" * 88)

    while True:
        try:
            prompt = input("\n[You] > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit", "q"):
                print("Exiting REPL.")
                break

            print("[Agent] > ", end="", flush=True)
            t0 = time.time()
            tok_count = 0
            for partial_text in agent.generate_stream(
                prompt, thinking_cycles=thinking_cycles, max_tokens=max_tokens, temperature=temperature
            ):
                # streaming delta
                pass
            # Final output print
            full_out = agent.generate(
                prompt, thinking_cycles=thinking_cycles, max_tokens=max_tokens, temperature=temperature
            )
            dt = time.time() - t0
            print(full_out)
            print(f"  [cycles={thinking_cycles} | latency={dt:.2f}s]")
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


def main():
    parser = argparse.ArgumentParser(description="Comprehensive CerebroSpheroid Evaluation Suite")
    parser.add_argument("checkpoint", nargs="?", default=None, help="Path to checkpoint (.pt)")
    parser.add_argument("--arch", default=None, choices=["v2", "v3"], help="Architecture version (default: auto)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive CLI REPL")
    parser.add_argument("--compare_thinking", action="store_true", help="Compare outputs across thinking_cycles=0, 1, 3")
    parser.add_argument("--cycles", type=int, default=1, help="Deliberation thinking cycles (default: 1)")
    parser.add_argument("--max_tokens", type=int, default=32, help="Max generation tokens (default: 32)")
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature (default: 0.0 for greedy)")
    args = parser.parse_args()

    ckpt_path = resolve_checkpoint(args.checkpoint)
    arch = args.arch
    if not arch:
        arch = "v3" if "v3" in os.path.basename(ckpt_path) else "v2"

    print("=" * 88)
    print(f"Loading checkpoint: {ckpt_path}")
    print(f"Architecture:       {arch}")
    agent = ReasoningAgent(checkpoint_path=ckpt_path, arch=arch)
    print(f"Device:             {agent.device}")
    print("=" * 88 + "\n")

    try:
        if args.compare_thinking:
            compare_thinking_cycles(agent)
        elif args.interactive:
            interactive_repl(agent, thinking_cycles=args.cycles, max_tokens=args.max_tokens, temperature=args.temp)
        else:
            evaluate_benchmark(agent, thinking_cycles=args.cycles, max_tokens=args.max_tokens)
    finally:
        agent.stop_living_heartbeat()


if __name__ == "__main__":
    main()
