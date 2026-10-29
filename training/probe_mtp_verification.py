"""
Empirical Verification Probe for DeepSeek Multi-Token Prediction (MTP)
adapted for the CerebroSpheroid Premotor-Motor Decoder.

Tests:
1. Joint Forward/Loss Computation: L_total = L_main + lambda * L_MTP.
2. Gradient Flow & Backward Pass Credit Assignment from future token t+2 into the core.
3. Overfit Convergence with Multi-Token Supervision.
4. Speculative Decoding Simulation: Measures token acceptance rate (>80% on learned sequence).
"""

import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.deepseek_layers import MultiTokenPredictionHead


def run_mtp_probe():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(f" PROBE 2: DEEPSEEK MULTI-TOKEN PREDICTION (MTP) ON {device}")
    print("=" * 75)

    vocab_size = 32000
    surface_dim = 640
    hidden_dim = 1280
    batch_size = 4
    seq_len = 16

    embedding = nn.Embedding(vocab_size, surface_dim).to(device)
    main_head = nn.Linear(hidden_dim, vocab_size, bias=False).to(device)
    mtp_head = MultiTokenPredictionHead(
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        surface_dim=surface_dim,
    ).to(device)

    # -------------------------------------------------------------
    # 1. Forward Pass & Dual-Loss Computation
    # -------------------------------------------------------------
    print("\n[1/4] Testing MTP Forward Pass & Dual-Loss (L_main + lambda * L_mtp)...")
    # Synthetic target tokens: [B, L]
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    h_states = torch.randn(batch_size, seq_len - 2, hidden_dim, device=device, requires_grad=True)

    # Step t predicts token t+1 (main) and token t+2 (mtp)
    # Main targets: targets[:, 1 : -1]  (token t+1)
    # MTP targets : targets[:, 2 : ]   (token t+2)
    y_t1 = targets[:, 1 : -1]
    y_t2 = targets[:, 2 : ]
    emb_t1 = embedding(y_t1)

    logits_main = main_head(h_states)  # [B, L-2, Vocab]
    logits_mtp = mtp_head(h_states.reshape(-1, hidden_dim), emb_t1.reshape(-1, surface_dim)).view(batch_size, seq_len - 2, vocab_size)

    loss_main = F.cross_entropy(logits_main.reshape(-1, vocab_size), y_t1.reshape(-1))
    loss_mtp = F.cross_entropy(logits_mtp.reshape(-1, vocab_size), y_t2.reshape(-1))
    loss_total = loss_main + 0.3 * loss_mtp

    print(f"  Main Loss (t+1) : {loss_main.item():.4f}")
    print(f"  MTP Loss  (t+2) : {loss_mtp.item():.4f}")
    print(f"  Total Joint Loss: {loss_total.item():.4f}")
    assert math.isfinite(loss_total.item())
    print("  => OK: Joint multi-token loss computes cleanly.")

    # -------------------------------------------------------------
    # 2. Gradient Flow & Credit Assignment
    # -------------------------------------------------------------
    print("\n[2/4] Testing Gradient Flow from Token t+2 into Trunk State...")
    loss_total.backward()

    assert h_states.grad is not None and h_states.grad.norm() > 0.0
    for name, p in mtp_head.named_parameters():
        assert p.grad is not None and p.grad.norm() > 0.0, f"Zero gradient in MTP head {name}"
    print("  => OK: Gradients flow from token t+2 back into trunk representations.")

    # -------------------------------------------------------------
    # 3. Overfit Convergence with Multi-Token Supervision
    # -------------------------------------------------------------
    print("\n[3/4] Testing Overfit Convergence under Multi-Token Supervision...")
    params = list(embedding.parameters()) + list(main_head.parameters()) + list(mtp_head.parameters())
    opt = torch.optim.AdamW(params, lr=0.005)

    fixed_h = torch.randn(batch_size, seq_len - 2, hidden_dim, device=device)

    for step in range(1, 31):
        opt.zero_grad()
        emb = embedding(y_t1)
        l_main = main_head(fixed_h)
        l_mtp = mtp_head(fixed_h.reshape(-1, hidden_dim), emb.reshape(-1, surface_dim)).view(batch_size, seq_len - 2, vocab_size)

        loss1 = F.cross_entropy(l_main.reshape(-1, vocab_size), y_t1.reshape(-1))
        loss2 = F.cross_entropy(l_mtp.reshape(-1, vocab_size), y_t2.reshape(-1))
        loss = loss1 + 0.3 * loss2
        loss.backward()
        opt.step()

        acc_main = (l_main.argmax(dim=-1) == y_t1).float().mean().item() * 100.0
        acc_mtp = (l_mtp.argmax(dim=-1) == y_t2).float().mean().item() * 100.0

        if step % 10 == 0 or step == 1 or step == 30:
            print(f"  Step {step:02d}/30 | Loss: {loss.item():.4f} | Main Acc (t+1): {acc_main:5.1f}% | MTP Acc (t+2): {acc_mtp:5.1f}%")

    assert acc_main > 95.0 and acc_mtp > 95.0, "MTP failed to converge"
    print("  => OK: Both main (t+1) and MTP (t+2) heads achieve >95% accuracy simultaneously.")

    # -------------------------------------------------------------
    # 4. Speculative Decoding Simulation (2x Emission Speed)
    # -------------------------------------------------------------
    print("\n[4/4] Simulating Speculative Decoding (Emitting 2 tokens per step)...")
    with torch.no_grad():
        # Predict candidate token 1
        cand_t1 = main_head(fixed_h).argmax(dim=-1)
        cand_emb_t1 = embedding(cand_t1)
        # Predict candidate token 2 via MTP head
        cand_t2 = mtp_head(fixed_h.reshape(-1, hidden_dim), cand_emb_t1.reshape(-1, surface_dim)).view(batch_size, seq_len - 2, vocab_size).argmax(dim=-1)

        # Verification: how often is candidate t2 equal to ground truth y_t2?
        accepted = (cand_t2 == y_t2).float().mean().item() * 100.0
        print(f"  Speculative Acceptance Rate: {accepted:.1f}%")
        assert accepted > 90.0, "Speculative acceptance rate too low"
        print("  => OK: Speculative decoding successfully delivers dual tokens per step.")

    print("\n" + "=" * 75)
    print(" ALL MTP PROBE VERIFICATION CRITERIA PASSED!")
    print("=" * 75)


if __name__ == "__main__":
    run_mtp_probe()
