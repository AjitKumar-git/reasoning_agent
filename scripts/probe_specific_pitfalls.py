#!/usr/bin/env python3
"""
Diagnostic Probe: Testing CerebroSpheroid V4 against Classic LLM Pitfalls.
1. Character counting (the 'strawberry' problem)
2. Temporal reasoning (days of week, clock times)
3. Hallucination traps (unphysical / counterfactual premises)
"""

import os
import sys
import torch

os.environ["HF_HOME"] = os.path.expanduser("~/.cache/hf_home")
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/hf_home")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.corpus import build_tokenizer
from src.spheroid_v4 import build_spheroid_v4


def run_probe(ckpt_path: str, device: str = "cuda:0"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(f"🧪 CEREBRO-SPHEROID V4: PITFALL & COGNITIVE TRAP PROBE")
    print(f"  • Checkpoint: {ckpt_path}")
    print(f"  • Device    : {device}")
    print("=" * 75)

    tok = build_tokenizer("llama")
    model = build_spheroid_v4("v4_small", vocab_size=len(tok))
    model.load_compatible_checkpoint(ckpt_path, device=device)
    model.to(device=device, dtype=torch.float32)
    model.eval()

    test_cases = [
        # 1. Strawberry & Spelling Blindness
        ("Strawberry (Direct)", "How many letters 'r' are in the word strawberry?"),
        ("Strawberry (Spelled)", "Spell out the letters in s-t-r-a-w-b-e-r-r-y and count how many times 'r' appears."),
        
        # 2. Time & Temporal Reasoning
        ("Temporal (Weekday)", "If yesterday was Tuesday, what day is tomorrow?"),
        ("Temporal (Clock Math)", "A train leaves at 2:00 PM and travels for 3 hours. What time does it arrive?"),
        ("Temporal (Duration)", "How many hours are there between 9:00 AM and 5:00 PM?"),

        # 3. Hallucination Traps & Nonsense Filtering
        ("Hallucination Trap 1", "Who was the first person to walk on the Sun?"),
        ("Hallucination Trap 2", "What is the capital city of Mars?"),
        ("Hallucination Trap 3", "Can a human breathe underwater without equipment?"),
    ]

    for label, prompt in test_cases:
        print(f"\n[Test: {label}]")
        print(f"❓ Prompt: \"{prompt}\"")
        with torch.no_grad():
            ans = model.generate_text(
                prompt,
                tokenizer=tok,
                thinking_cycles=4,
                max_new_tokens=48,
                temperature=0.4,
                repetition_penalty=1.30,
                device=device,
                strip_bio_prefix=True,
            )
        print(f"🤖 Answer: {ans.strip()}")
        print("-" * 75)

    print("\n✅ Pitfall probe complete.")


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/cerebro_spheroid_v4_ultrachat_sft_best.pt"
    run_probe(ckpt)
