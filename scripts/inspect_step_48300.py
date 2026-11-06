#!/usr/bin/env python3
"""
Comprehensive evaluation and manifold diagnostic script for Step 48,300 checkpoint.
Evaluates weight integrity, spectral radius, and multi-domain reasoning capabilities.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F

from src.corpus import build_tokenizer, clean_motor_response, format_stimulus_prompt
from src.spheroid_v4 import CerebroSpheroidV4, compute_manifold_diagnostics


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Executing evaluation on: {device}")

    ckpt_path = "checkpoints/cerebro_spheroid_v4_step_48300.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/cerebro_spheroid_v4_latest.pt"
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint file not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path} ({os.path.getsize(ckpt_path):,} bytes)...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    print(f"• Total checkpoint keys: {len(ckpt)}")
    for k in ["step", "train_loss", "loss", "eval_loss", "curriculum"]:
        if k in ckpt:
            print(f"• {k}: {ckpt[k]}")

    # Build model & load state dict
    model = CerebroSpheroidV4().to(device=device)
    model_keys = set(model.state_dict().keys())
    cleaned_sd = {k: v for k, v in ckpt.items() if k in model_keys}
    model.load_state_dict(cleaned_sd, strict=True)
    model.eval()
    print(f"• Model State: Successfully loaded {len(cleaned_sd)}/{len(model_keys)} parameters (strict=True)!")
    print(f"• Total Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 1. Weight Integrity Check
    print("\n" + "=" * 75)
    print(" 1. WEIGHT INTEGRITY & HYPER-LOGIT ATTRACTOR AUDIT")
    print("=" * 75)
    has_nan = False
    has_inf = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"  ❌ NaN detected in {name}")
            has_nan = True
        if torch.isinf(param).any():
            print(f"  ❌ Inf detected in {name}")
            has_inf = True
    if not has_nan and not has_inf:
        print("  ✅ Zero NaNs and zero Infs across all 43 model tensors.")

    head_w = model.motor_dnn.head.weight.data
    head_norms = torch.norm(head_w, dim=1)
    max_norm, max_idx = torch.max(head_norms), torch.argmax(head_norms).item()
    mean_norm = torch.mean(head_norms).item()
    print(f"  • Motor Head Norm: Mean={mean_norm:.4f}, Max={max_norm.item():.4f} (at token {max_idx})")

    # 2. Manifold & Core Dynamics Diagnostics
    print("\n" + "=" * 75)
    print(" 2. MANIFOLD & SPECTRAL RADIUS DIAGNOSTICS")
    print("=" * 75)
    with torch.no_grad():
        state = model.init_brain_state(1, device=device)
        diag = compute_manifold_diagnostics(model, state)
        for k, v in diag.items():
            if isinstance(v, float):
                print(f"  • {k:<32}: {v:.6f}")
            else:
                print(f"  • {k:<32}: {v}")

    # 3. Model Inference with Enclosed Biological Delimiters (<stim> ... </stim> / <act> ... </act>)
    print("\n" + "=" * 75)
    print(" 3. MULTI-DOMAIN REASONING EVALUATION (Step 48,300)")
    print("=" * 75)
    tok = build_tokenizer("llama")

    test_prompts = [
        # Conversational
        ("Greeting", "Hi"),
        ("Identity", "Hello! Who are you?"),
        # Arithmetic / Reasoning
        ("Basic Math", "What is 2 + 3?"),
        ("Word Problem", "If you have 15 books and buy 4 more, how many do you have in total?"),
        # Science / Knowledge
        ("Science", "Why is the sky blue?"),
        ("Physics", "Explain Newton's third law of motion in simple terms."),
        # Code Generation
        ("Coding", "Write a Python function to check if a number is prime."),
    ]

    for category, prompt in test_prompts:
        print(f"\n▶ [{category}] User: {prompt!r}")

        # Greedy / Low-Temperature
        ans_greedy = model.generate_text(
            prompt,
            tokenizer=tok,
            thinking_cycles=3,
            max_new_tokens=64,
            temperature=0.1,
            repetition_penalty=1.25,
            device=device,
            strip_bio_prefix=True,
        )
        print(f"  • Greedy (T=0.1): {ans_greedy!r}")

        # Sampled
        ans_sampled = model.generate_text(
            prompt,
            tokenizer=tok,
            thinking_cycles=3,
            max_new_tokens=64,
            temperature=0.7,
            repetition_penalty=1.25,
            device=device,
            strip_bio_prefix=True,
        )
        print(f"  • Sampled (T=0.7): {ans_sampled!r}")
        print("-" * 70)


if __name__ == "__main__":
    main()
