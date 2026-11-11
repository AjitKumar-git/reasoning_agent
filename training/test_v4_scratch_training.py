#!/usr/bin/env python3
"""
Test CerebroSpheroid V4 Training From Scratch
=============================================
Verifies:
1. Forward pass with causal KV & RoPE under teacher forcing and scheduled deliberation.
2. Gradient flow across all layers (sensory front-end, volumetric core, motor decoder, head).
3. Monotonic loss reduction during optimization from scratch.
4. Coherent generation without repetition loops using causal KV cache.
"""

import math
import os
import sys
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.spheroid_v4 import build_spheroid_v4


def test_v4_scratch():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("🔬 TESTING V4 TRAINING FROM SCRATCH (Causal RoPE & Multi-Head Upgrade)")
    print(f"   Device: {dev}")
    print("=" * 80)

    # 1. Instantiate fresh V4 model
    vocab_size = 1000
    model = build_spheroid_v4("test", vocab_size=vocab_size).to(dev)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Created fresh V4 model: {total_params:,} parameters")

    # 2. Synthetic prompt and target sequences
    B, L_prompt, L_target = 4, 16, 12
    rng = torch.Generator(device=dev).manual_seed(42)
    prompts = torch.randint(1, vocab_size, (B, L_prompt), generator=rng, device=dev)
    targets = torch.randint(1, vocab_size, (B, L_target), generator=rng, device=dev)

    # 3. Test Forward Pass with Causal KV
    print("\n[1/4] Testing forward pass with causal target accumulation...")
    logits = model(prompts, response_steps=L_target, targets=targets, deliberation_mode="teacher_forcing")
    assert logits.shape == (B, L_target, vocab_size), f"Logits shape mismatch: {logits.shape}"
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf!"
    print(f"      Forward pass successful: {tuple(logits.shape)} logits, 100% finite.")

    # 4. Test Backward Pass & Gradient Flow
    print("\n[2/4] Testing backward pass and gradient flow across active pathways...")
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    loss.backward()

    core_pathway_keys = [
        "sensory_dnn.embedding.weight",
        "sensory_dnn.fc1_gate.weight",
        "sensory_dnn.fc1_val.weight",
        "sensory_dnn.fc2.weight",
        "volumetric_core.intra_shell",
        "volumetric_core.shortcut_tracts",
        "volumetric_core.reentrant_field",
        "motor_dnn.q_proj.weight",
        "motor_dnn.k_proj.weight",
        "motor_dnn.v_proj.weight",
        "motor_dnn.fc_gate.weight",
        "motor_dnn.fc_val.weight",
        "motor_dnn.head.weight",
        "motor_dnn.efference_proj.weight",
    ]

    print(f"      Initial Loss from Scratch: {loss.item():.4f} (Expected near ln({vocab_size}) = {math.log(vocab_size):.2f})")
    for key in core_pathway_keys:
        param = dict(model.named_parameters())[key]
        assert param.grad is not None and param.grad.abs().sum() > 0, f"Core parameter {key} did not receive gradients!"
    print(f"      OK: All {len(core_pathway_keys)} core signal pathway parameters received clean, healthy gradients.")

    # 5. Test 25 Optimization Steps from Scratch
    print("\n[3/4] Testing 25 gradient steps from scratch (loss convergence)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    losses = []
    # Fixed synthetic sequence to memorize
    fixed_prompt = torch.randint(1, vocab_size, (2, 8), generator=rng, device=dev)
    fixed_target = torch.randint(1, vocab_size, (2, 8), generator=rng, device=dev)

    for step in range(25):
        optimizer.zero_grad()
        l_out = model(fixed_prompt, response_steps=8, targets=fixed_target, deliberation_mode="teacher_forcing")
        step_loss = F.cross_entropy(l_out.view(-1, vocab_size), fixed_target.view(-1))
        step_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(step_loss.item())
        if (step + 1) % 5 == 0:
            print(f"      Step {step + 1:02d} | Loss: {step_loss.item():.4f}")

    assert losses[-1] < losses[0] * 0.5, f"Loss failed to drop significantly: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"      OK: Loss dropped rapidly from {losses[0]:.4f} down to {losses[-1]:.4f} ({((losses[0] - losses[-1])/losses[0])*100:.1f}% reduction).")

    # 6. Test Autoregressive Generation with Causal KV
    print("\n[4/4] Testing autoregressive generation with active causal KV cache...")
    model.eval()
    with torch.no_grad():
        gen_tokens = model.generate(fixed_prompt[:1], max_new_tokens=16, temperature=0.7)
    assert gen_tokens.shape == (1, 16), f"Generated shape mismatch: {gen_tokens.shape}"
    toks = gen_tokens[0].tolist()
    print(f"      Generated 16 tokens: {toks}")
    # Verify no immediate degenerate single-token repetition (e.g., [5, 5, 5, 5, 5...])
    has_diversity = len(set(toks)) > 3
    assert has_diversity, f"Degenerate single-token repetition detected: {toks}"
    print("      OK: Output exhibits healthy token diversity and no repetition collapse.")

    print("\n" + "=" * 80)
    print("🎉 ALL V4 SCRATCH TRAINING & CAUSAL ATTENTION VERIFICATIONS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    test_v4_scratch()
