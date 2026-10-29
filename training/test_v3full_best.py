"""
Comprehensive Testing and Evaluation Suite for cerebro_spheroid_v3full_best.pt (162M).

Trained on Kaggle for 4,800 steps on OpenHermes, GSM8K, and Knowledge.
Evaluates:
1. Multi-Step Math Reasoning (GSM8K style)
2. Logical Deduction & Commonsense Reasoning
3. Open-Ended Instruction Following (OpenHermes style)
4. Active Deliberation Thinking Ablation (T=0 vs T=1 vs T=3 thinking cycles)
5. Manifold Internal Dynamics (16 Concentric Shells RMS, cosine alignment, effective rank)
"""

import math
import os
import sys
import time
import torch
import torch.nn.functional as F

os.environ["HF_HOME"] = os.path.expanduser("~/.hf_cache")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoTokenizer
from src.spheroid_v3 import build_spheroid_v3
from src.spheroid_v4 import build_spheroid_v4, compute_manifold_diagnostics, lift_checkpoint_v3_to_v4


def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(" TESTING CEREBROSPHEROID V3-FULL BEST MODEL (162M)")
    print(f" Device: {device} | Checkpoint: checkpoints/cerebro_spheroid_v3full_best.pt")
    print("=" * 75)

    ckpt_path = os.path.abspath("checkpoints/cerebro_spheroid_v3full_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # Load tokenizer
    print("[1/4] Loading Tokenizer (GPT-2)...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Load model
    print("[2/4] Instantiating CerebroSpheroid (162,272,317 parameters)...")
    model = build_spheroid_v3("laptop").to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    print("      Model successfully loaded with exact parameter match!")

    # -------------------------------------------------------------
    # 3. Manifold Geometric Diagnostics
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print(" 3. VOLUMETRIC MANIFOLD INTERNAL DYNAMICS (16 SHELLS)")
    print("=" * 75)
    with torch.no_grad():
        test_prompt = "Question: If a car travels at 60 mph, how far does it travel in 3.5 hours? Answer:"
        toks = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
        brain_state, _ = model.ingest_stimulus_with_memory(toks)
        # 16 shells
        num_shells = brain_state.shape[1]
        rms_energies = [brain_state[0, k].pow(2).mean().sqrt().item() for k in range(num_shells)]
        mean_rms = sum(rms_energies) / len(rms_energies)

        cos_sims = []
        for k in range(num_shells - 1):
            cs = F.cosine_similarity(brain_state[0, k:k+1], brain_state[0, k+1:k+2]).item()
            cos_sims.append(cs)
        mean_cos = sum(cos_sims) / len(cos_sims)

        print(f"  Prompt: \"{test_prompt}\"")
        print(f"  Mean Shell RMS Energy   : {mean_rms:.4f} (target: ~1.0)")
        print(f"  Adjacent Shell Cos Sim  : {mean_cos:.4f} (smooth radial gradient)")
        print(f"  Deep Shell 0 Energy     : {rms_energies[0]:.4f} (subcortical working memory attractor)")
        print(f"  Outer Shell 15 Energy   : {rms_energies[-1]:.4f} (superficial cortical speech resonance)")
        print(f"  Per-Shell RMS Gradient  : {[round(r, 2) for r in rms_energies]}")

    # -------------------------------------------------------------
    # 4. Active Reasoning & Thinking Evaluation
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print(" 4. REASONING & INFERENCE TEST SUITE")
    print("=" * 75)

    test_cases = [
        {
            "category": "Math Reasoning (GSM8K Style)",
            "prompt": "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer: Let's solve this step by step:",
        },
        {
            "category": "Arithmetic & Calculation",
            "prompt": "Question: A bakery makes 120 loaves of bread. In the morning, they sell 45 loaves. In the afternoon, they sell 35 loaves. How many loaves remain unsold?\nAnswer:",
        },
        {
            "category": "Logical Deduction",
            "prompt": "Question: John is taller than Peter. Peter is taller than Sam. Is John taller than Sam?\nAnswer:",
        },
        {
            "category": "Instruction Following / Explanation",
            "prompt": "Question: What is photosynthesis and why is it important for life on Earth?\nAnswer:",
        },
        {
            "category": "Coding / Algorithmic",
            "prompt": "Question: Write a Python function `is_even(n)` that returns True if n is an even integer.\nAnswer:\n```python",
        },
    ]

    for idx, tc in enumerate(test_cases, 1):
        print(f"\n[{idx}/{len(test_cases)}] Category: {tc['category']}")
        print(f"Prompt:\n  {tc['prompt'].strip()}")
        print("-" * 65)

        prompt_toks = tokenizer.encode(tc["prompt"], return_tensors="pt").to(device)

        with torch.no_grad():
            # Test thinking cycles T=1
            t_start = time.time()
            out_toks = model.solve_with_active_thinking(
                prompt_toks,
                thinking_cycles=1,
                max_answer_tokens=48,
                temperature=0.2,
                repetition_penalty=1.2,
            )
            elapsed = time.time() - t_start

            # Decode output
            gen_toks = out_toks[0].cpu().tolist()
            gen_text = tokenizer.decode(gen_toks, skip_special_tokens=True)
            print(f"Generated (Thinking Cycles = 1, {elapsed:.2f}s):")
            print(f"  {gen_text.strip()}")

    # -------------------------------------------------------------
    # 5. Deliberation Thinking Ablation (T=0 vs T=1 vs T=3)
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print(" 5. DELIBERATION ABLATION (T=0 vs T=1 vs T=3 Thinking Cycles)")
    print("=" * 75)
    ablation_prompt = "Question: If a train leaves Chicago at 2:00 PM traveling at 50 mph, how many miles has it traveled by 5:00 PM?\nAnswer:"
    ablation_toks = tokenizer.encode(ablation_prompt, return_tensors="pt").to(device)
    print(f"Prompt: \"{ablation_prompt}\"\n")

    with torch.no_grad():
        for tc_count in [0, 1, 3]:
            out = model.solve_with_active_thinking(
                ablation_toks,
                thinking_cycles=tc_count,
                max_answer_tokens=32,
                temperature=0.1,
                repetition_penalty=1.2,
            )
            text = tokenizer.decode(out[0].cpu().tolist(), skip_special_tokens=True)
            print(f"  Thinking Cycles = {tc_count}: \"{text.strip()}\"")

    print("\n" + "=" * 75)
    print(" EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    run_test()
