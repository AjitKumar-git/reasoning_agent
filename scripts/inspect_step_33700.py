#!/usr/bin/env python3
"""
Inspect and evaluate CerebroSpheroid V4 Step 33,700 Checkpoint.
"""
import os
import sys
import torch

# Ensure repo root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.spheroid_v4 import (
    CerebroSpheroidV4,
    compute_manifold_diagnostics,
)
from src.corpus import build_tokenizer

def main():
    ckpt_path = "checkpoints/cerebro_spheroid_v4_step_33700.pt"
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        return

    print("=" * 75)
    print(" CEREBRO-SPHEROID V4 — STEP 33,700 CHECKPOINT INSPECTION & EVALUATION")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load checkpoint
    print(f"Loading checkpoint: {ckpt_path} ({os.path.getsize(ckpt_path):,} bytes)...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    print(f"• Total checkpoint keys: {len(ckpt)}")
    for k in ["step", "train_loss", "loss", "eval_loss", "metrics", "curriculum"]:
        if k in ckpt:
            print(f"• {k}: {ckpt[k]}")

    # 2. Instantiate Model & Load State Dict
    model = CerebroSpheroidV4().to(device)
    model_keys = set(model.state_dict().keys())

    cleaned_sd = {}
    for k, v in ckpt.items():
        clean_k = k
        if clean_k.startswith("_orig_mod."):
            clean_k = clean_k[10:]
        elif clean_k.startswith("module."):
            clean_k = clean_k[7:]
        if clean_k in model_keys:
            cleaned_sd[clean_k] = v

    load_result = model.load_state_dict(cleaned_sd, strict=True)
    print(f"• Model State: Successfully loaded {len(cleaned_sd)}/{len(model_keys)} parameters (strict=True)!")
    print(f"• Total Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    model.eval()

    # 3. Manifold Diagnostics
    print("\n--- Manifold & Biological Geometry Diagnostics ---")
    state = model.init_brain_state(1, device=device)
    diag = compute_manifold_diagnostics(model, state)
    for k, v in diag.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:8.4f}")
        else:
            print(f"  {k:25s}: {v}")

    # 4. Tokenizer
    print("\nLoading Tokenizer (Llama 32k)...")
    tokenizer = build_tokenizer("llama")

    # 5. Inference Test Suite
    test_prompts = [
        "Hi",
        "Hello! Who are you?",
        "What is 2 + 3?",
        "If you have 3 apples and eat 1, how many are left?",
        "What is the capital of France?",
        "Why is the sky blue?",
        "What is water made of?",
        "What color is a red ball?",
        "Write a python function to add two numbers.",
    ]

    print("\n" + "=" * 75)
    print(" INFERENCE EVALUATION (Enclosed Bio Delimiters: '<stim> ... </stim>' / '<act> ... </act>')")
    print("=" * 75)

    for i, user_q in enumerate(test_prompts, 1):
        print(f"\n[{i}/{len(test_prompts)}] Prompt: {user_q!r}")

        # Greedy generation (T=0.1 to avoid argmax strict flatting if identical logits)
        out_greedy = model.generate_text(
            prompt=user_q,
            tokenizer=tokenizer,
            thinking_cycles=3,
            max_new_tokens=48,
            temperature=0.1,
            repetition_penalty=1.2,
            device=device,
        )
        print(f"  ▶ Greedy / Low-T: {out_greedy!r}")

        # Sampled generation (T=0.7, repetition_penalty=1.25)
        out_sampled = model.generate_text(
            prompt=user_q,
            tokenizer=tokenizer,
            thinking_cycles=3,
            max_new_tokens=48,
            temperature=0.7,
            repetition_penalty=1.25,
            device=device,
        )
        print(f"  ▶ Sampled (T=0.7): {out_sampled!r}")

    print("\n" + "=" * 75)
    print(" EVALUATION COMPLETE")
    print("=" * 75)

if __name__ == "__main__":
    main()
