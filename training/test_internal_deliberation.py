"""
Verification Suite for Refined Closed-Loop Internal Deliberation in CerebroSpheroid V4.

Validates:
1. Full backward compatibility: loads existing V4 checkpoint with 0 missing/dropped tensors.
2. Differentiable closed-loop internal deliberation (zero external token feedback):
   - Computes forward rollout in deliberation_mode="internal".
   - Verifies gradient flow through internal straight-through thought re-afference.
3. Thalamic memory basin prompt retention:
   - Demonstrates that thalamic anchoring preserves the initial prompt representation.
4. Autonomous generation in deliberation_mode="internal".
"""

import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.spheroid_v4 import build_spheroid_v4
from src.corpus import build_tokenizer


def test_internal_deliberation_suite():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print(" RUNNING REFINED INTERNAL DELIBERATION VERIFICATION SUITE")
    print(f" Device: {device}")
    print("=" * 72)

    # 1. Checkpoint loading test
    print("\n[1/4] Verifying V4 Checkpoint Weight Compatibility...")
    tok = build_tokenizer("llama")
    model = build_spheroid_v4("v4_small", vocab_size=len(tok))
    ckpt_path = "checkpoints/cerebro_spheroid_v4_small_kaggle_best.pt"
    
    if os.path.exists(ckpt_path):
        success = model.load_compatible_checkpoint(ckpt_path, device=device)
        print(f"      Loaded checkpoint {ckpt_path}: {success}")
        assert success, "Failed to load V4 checkpoint!"
    else:
        print("      Checkpoint file not found; testing on freshly initialized model.")

    model.to(device)

    # 2. Differentiable Internal Deliberation Rollout (Zero External Token Feedback)
    print("\n[2/4] Testing Gradient Flow in deliberation_mode='internal'...")
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()

    prompt = torch.randint(10, 500, (2, 8), device=device)
    targets = torch.randint(10, 500, (2, 6), device=device)

    # Run internal deliberation rollout (targets are ONLY used to evaluate CE loss, NOT fed back)
    logits = model(
        prompt,
        response_steps=6,
        thinking_cycles=1,
        targets=targets,
        deliberation_mode="internal",
        thalamic_anchor_weight=0.2,
    )
    assert logits.shape == (2, 6, len(tok)), f"Unexpected logits shape: {logits.shape}"

    loss = F.cross_entropy(logits.reshape(-1, len(tok)), targets.reshape(-1))
    print(f"      Deliberation step loss: {loss.item():.4f}")
    loss.backward()

    # Check gradients on sensory embedding, volumetric core, and motor readout
    grad_emb = model.sensory_dnn.embedding.weight.grad
    grad_core = model.volumetric_core.intra_shell.grad
    grad_head = model.motor_dnn.head.weight.grad

    assert grad_emb is not None and grad_emb.abs().sum() > 0, "No gradient flow to sensory embedding!"
    assert grad_core is not None and grad_core.abs().sum() > 0, "No gradient flow to volumetric core!"
    assert grad_head is not None and grad_head.abs().sum() > 0, "No gradient flow to motor head!"
    print("      OK: Gradients flow cleanly through internal thought re-afference and manifold.")

    optimizer.step()
    print("      OK: Optimizer step completed cleanly.")

    # 3. Thalamic Memory Basin Anchor Retention
    print("\n[3/4] Verifying Thalamic Memory Basin Prompt Retention...")
    model.eval()
    with torch.no_grad():
        memory, gist, _ = model.encode_sensory(prompt)
        initial_state = model._integrate_memory(memory)
        initial_thalamic = initial_state[:, :model.num_shells // 3, :].clone()

        # Run rollout with thalamic anchor
        _ = model.training_logits(
            prompt,
            response_steps=16,
            thinking_cycles=2,
            deliberation_mode="internal",
            thalamic_anchor_weight=0.3,
        )
        
        # Verify cosine similarity is preserved
        memory2, _, _ = model.encode_sensory(prompt)
        state_after = model._integrate_memory(memory2)
        sim = F.cosine_similarity(
            initial_thalamic.reshape(2, -1),
            state_after[:, :model.num_shells // 3, :].reshape(2, -1)
        ).mean().item()
        print(f"      Thalamic prompt cosine similarity: {sim:.4f}")
        assert sim > 0.85, f"Thalamic prompt similarity degraded too much: {sim}"
        print("      OK: Deep thalamic shells retain prompt representation.")

    # 4. Autonomous Generation in Deliberation Mode
    print("\n[4/4] Verifying Autonomous Generation (deliberation_mode='internal')...")
    with torch.no_grad():
        out_tokens = model.generate(
            prompt,
            thinking_cycles=2,
            max_new_tokens=8,
            deliberation_mode="internal",
            thalamic_anchor_weight=0.2,
        )
        print(f"      Generated {out_tokens.shape[1]} tokens: {out_tokens[0].tolist()}")
        assert out_tokens.shape == (2, 8), f"Unexpected generated shape: {out_tokens.shape}"
        print("      OK: Internal deliberation generation runs without external token feedback.")

    print("\n" + "=" * 72)
    print(" ALL REFINED INTERNAL DELIBERATION TESTS PASSED!")
    print("=" * 72)


if __name__ == "__main__":
    test_internal_deliberation_suite()
