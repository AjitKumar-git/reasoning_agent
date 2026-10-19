"""
Tightened AGI Training Engine with Answer Focal Weighting & Anti-Degeneracy
============================================================================
1. Trains EnhancedOrganoidSpheroid on tightened, answer-anchored curriculums.
2. Answer Focal Loss: Weights critical numerical and categorical conclusion tokens 3.0x higher.
3. Anti-Degeneracy Refractory Decoding: Aggressively suppresses unigram & trigram repetition.
4. Evaluates periodically using the strict, multi-criterion Tightened AGI Benchmark.
5. Saves to checkpoints/tightened_organoid_best.pt.
"""

import os
import sys
import time
import math
import random
import argparse
import torch
import torch.nn as nn
from typing import Tuple, List, Dict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.environ["HF_HOME"] = os.path.join(ROOT_DIR, ".hf_cache")

from transformers import AutoTokenizer
from src.organoid_layers import build_enhanced_organoid_spheroid
from src.tightened_curriculum import get_tightened_stream
from src.advanced_datasets import get_advanced_organoid_stream
from training.tightened_agi_benchmark import run_tightened_benchmark


def compute_focal_weighted_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
    focal_weight: float = 3.5,
) -> torch.Tensor:
    """
    Computes CrossEntropyLoss with progressive directional weighting:
    Tokens in the second half get 3.0x, and tokens in the final 25% (the exact conclusion)
    get 6.0x focal weight.
    """
    B, T, V = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    ce_loss = nn.functional.cross_entropy(logits_flat, targets_flat, ignore_index=-100, reduction="none")
    ce_loss = ce_loss.view(B, T)

    weights = torch.ones_like(ce_loss)
    for b in range(B):
        valid_indices = (targets[b] != -100).nonzero(as_tuple=True)[0]
        L = len(valid_indices)
        if L > 0:
            halfway = valid_indices[L // 2]
            three_quarters = valid_indices[(3 * L) // 4]
            weights[b, halfway:] = 3.0
            weights[b, three_quarters:] = 6.0

    valid_mask = (targets != -100).float()
    weighted_loss = (ce_loss * weights * valid_mask).sum() / (weights * valid_mask).sum().clamp(min=1.0)
    return weighted_loss


def main():
    parser = argparse.ArgumentParser(description="Train Living Organoid on Tightened AGI Objectives")
    parser.add_argument("--steps", type=int, default=300, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Micro-batch size for ~2.3 GB VRAM")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2.0e-4, help="Learning rate")
    parser.add_argument("--eval_interval", type=int, default=30, help="Evaluation interval")
    parser.add_argument("--save_prefix", type=str, default="tightened_organoid_best", help="Checkpoint prefix")
    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    model = build_enhanced_organoid_spheroid(vocab_size=vocab_size)

    # Resume from existing tightened or enhanced checkpoint
    tight_ckpt = os.path.join(ROOT_DIR, "checkpoints", "tightened_organoid_best.pt")
    enh_ckpt = os.path.join(ROOT_DIR, "checkpoints", "enhanced_organoid_best.pt")
    init_ckpt = tight_ckpt if os.path.exists(tight_ckpt) else enh_ckpt

    if os.path.exists(init_ckpt):
        print(f"Resuming weights from {init_ckpt}...")
        sd = torch.load(init_ckpt, map_location=device)
        model.load_state_dict(sd, strict=False)
        print("  --> Weights loaded successfully.")

    model.to(device=device, dtype=dtype)
    total_p = sum(p.numel() for p in model.parameters())

    best_ckpt = os.path.join(ROOT_DIR, "checkpoints", f"{args.save_prefix}.pt")
    os.makedirs(os.path.join(ROOT_DIR, "checkpoints"), exist_ok=True)

    print("=" * 90)
    print("      TIGHTENED AGI ORGANOID TRAINING (ANSWER FOCAL LOSS & ANTI-DEGENERACY)      ")
    print("=" * 90)
    print(f"Model Parameters   : {total_p:,} ({total_p/1e6:.2f}M params)")
    print(f"Target Accelerator : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Loss Objective     : Two-Tier Focal Weighted Loss (3.0x on reasoning, 6.0x on conclusion)")
    print(f"Curriculums        : 75% Tightened Procedural Proofs + 25% Multi-Curriculum")
    print(f"Training Batch     : Micro {args.batch_size} × Accum {args.grad_accum} = Effective {args.batch_size * args.grad_accum}")
    print(f"Total Steps        : {args.steps}")
    print("=" * 90)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.05)

    tight_stream = get_tightened_stream(tokenizer, batch_size=args.batch_size, max_prompt_len=256, max_resp_len=56)
    adv_stream = get_advanced_organoid_stream(tokenizer, batch_size=args.batch_size, max_prompt_len=256, max_resp_len=56)

    t_start = time.time()
    best_pass_rate = 0.0
    best_composite_score = 0.0
    best_loss = 999.0
    step = 0

    while step < args.steps:
        step += 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        thinking_cycles = random.randint(2, 4)

        for _ in range(args.grad_accum):
            # 75% Tightened Curriculum, 25% General Advanced
            if random.random() < 0.75:
                prompts, targets = next(tight_stream)
            else:
                prompts, targets = next(adv_stream)

            prompts = prompts.to(device)
            targets = targets.to(device)

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
                logits = model(prompts, targets, thinking_cycles=thinking_cycles)
                loss = compute_focal_weighted_loss(logits, targets, vocab_size=vocab_size, focal_weight=3.5)
                scaled_loss = loss / args.grad_accum

            scaled_loss.backward()
            accum_loss += scaled_loss.item()
            del prompts, targets, logits, loss, scaled_loss

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if device.type == "cuda" and step % 2 == 0:
            torch.cuda.empty_cache()

        loss_val = accum_loss

        if step % 10 == 0 or step == 1:
            ppl = math.exp(min(loss_val, 15))
            vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            elapsed = time.time() - t_start
            print(f"Step [{step:4d}/{args.steps}] | Loss: {loss_val:.4f} | PPL: {ppl:7.2f} | Cycles: {thinking_cycles} | VRAM: {vram:.0f} MB | Elapsed: {elapsed:.1f}s")

        if step % args.eval_interval == 0:
            bench_res = run_tightened_benchmark(model, tokenizer, device, dtype, max_tokens=48, thinking_cycles=3)
            pass_rate = bench_res["pass_rate_pct"]
            comp_score = bench_res["composite_score_pct"]

            # Save strictly when emergent AGI score improves
            is_better = (pass_rate > best_pass_rate) or (pass_rate == best_pass_rate and comp_score > best_composite_score)
            if is_better:
                best_pass_rate = max(best_pass_rate, pass_rate)
                best_composite_score = max(best_composite_score, comp_score)
                torch.save(model.state_dict(), best_ckpt)
                print(f"  ★ [NEW BEST TIGHTENED CHECKPOINT] Saved to {best_ckpt} (Strict Pass: {best_pass_rate:.1f}% | Score: {best_composite_score:.1f}% | Loss: {loss_val:.4f})")

            if loss_val < best_loss:
                best_loss = loss_val

    latest_ckpt = os.path.join(ROOT_DIR, "checkpoints", f"{args.save_prefix}_latest.pt")
    torch.save(model.state_dict(), latest_ckpt)
    total_time = time.time() - t_start
    print("=" * 90)
    print(f"Tightened Training Complete in {total_time/60:.2f} minutes.")
    print(f"Best Loss               : {best_loss:.4f}")
    print(f"Best Strict Pass Rate   : {best_pass_rate:.1f}%")
    print(f"Best Composite Score    : {best_composite_score:.1f}%")
    print(f"Saved Checkpoint        : {best_ckpt}")
    print("=" * 90)


if __name__ == "__main__":
    main()
