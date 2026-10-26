#!/usr/bin/env python3
"""
Comprehensive evaluation probe for CerebroSpheroidV3 after 4,000 steps of full-corpus training.
Tests factual recall, commonsense, arithmetic reasoning, and order sensitivity.
"""

import os
os.environ.setdefault("HF_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), ".hf_cache")))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys
import time
import torch
from src.inference import ReasoningAgent

TEST_QUERIES = [
    # 1. World Knowledge / Factual QA
    ("Factual", "What is the capital of France? Answer:"),
    ("Factual", "What is the capital of Japan? Answer:"),
    ("Factual", "What is the capital of India? Answer:"),
    ("Factual", "What is the chemical formula for water? Answer:"),
    # 2. Science & Biology
    ("Science", "What is the primary function of the heart in the human body? Answer:"),
    ("Science", "What is the boiling point of water in Celsius? Answer:"),
    # 3. Arithmetic & Multi-step Reasoning
    ("Arithmetic", "If 4 books cost $20, how much do 6 books cost? Answer:"),
    ("Arithmetic", "Maya has 8 markers. She buys 4 more and gives 3 away. How many markers does she have? Answer:"),
    ("Arithmetic", "What is 15 divided by 3? Answer:"),
    # 4. Commonsense & Logic
    ("Logic", "All humans are mortal. Socrates is a human. Therefore, Socrates is Answer:"),
]

def run_probe(ckpt_path: str, label: str):
    print("=" * 84)
    print(f"PROBING: {label} [{os.path.basename(ckpt_path)}]")
    print("=" * 84)

    agent = ReasoningAgent(checkpoint_path=ckpt_path, arch="v3")
    print(f"Model architecture: {agent.arch} ({type(agent.model).__name__}) on {agent.device}")
    print("-" * 84)

    for category, prompt in TEST_QUERIES:
        t0 = time.time()
        # Generate with temperature=0.0 (greedy) and 1 thinking cycle
        ans = agent.generate(prompt, thinking_cycles=1, max_tokens=32, temperature=0.0)
        dt = time.time() - t0
        print(f"[{category}]")
        print(f"  Q: {prompt}")
        print(f"  A: {ans}")
        print(f"  (latency: {dt:.2f}s)\n")

    agent.stop_living_heartbeat()

if __name__ == "__main__":
    import glob
    custom_target = sys.argv[1] if len(sys.argv) > 1 else None
    if custom_target and os.path.exists(custom_target):
        run_probe(custom_target, f"Target Checkpoint [{os.path.basename(custom_target)}]")
    else:
        ckpts = [
            ("checkpoints/cerebro_spheroid_v3_colab_best.pt", "V3 Colab (Best Held-out Checkpoint)"),
            ("/content/drive/MyDrive/cerebro_checkpoints/cerebro_spheroid_v3_colab_best.pt", "V3 Colab Google Drive (Best)"),
            ("checkpoints/cerebro_spheroid_v3_full_best.pt", "V3 Full Corpus (Best Held-out Checkpoint)"),
            ("checkpoints/cerebro_spheroid_v3_full_step4000.pt", "V3 Full Corpus (Step 4,000 Checkpoint)"),
        ]
        found = False
        for path, label in ckpts:
            if os.path.exists(path):
                found = True
                run_probe(path, label)
        if not found:
            # Fallback search for any .pt
            any_pt = glob.glob("checkpoints/*.pt") + glob.glob("/content/drive/MyDrive/cerebro_checkpoints/*.pt")
            if any_pt:
                run_probe(any_pt[0], f"Auto-detected Checkpoint [{os.path.basename(any_pt[0])}]")
            else:
                print("No checkpoints found in checkpoints/ or Google Drive.")
