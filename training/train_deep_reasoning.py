"""
Deep Reasoning Engine for Living Riemannian Organoid (B^3)
==========================================================
Trains the living organoid manifold on massive multi-source reasoning datasets:
1. 45% Cerebro-Cognitive Parquet (50,000 empirical math, logic, rate, algebra, and geometry samples).
2. 45% Tightened Procedural Proofs (syllogisms, transitivity posets, counterfactual causality, enzyme kinetics, ARC-AGI).
3. 10% Cross-Modal Retinotopic/Tonotopic Sheets (maintains visual and audio grounding).

Objective: Two-Tier Answer-Focal Weighted Cross Entropy (3.0x on deduction, 6.0x on concluding target entity/number).
Zero External Feedback: Strictly maintains the Riemannian core invariant (x_{t+1} != [x, y]).
"""

import os
import sys
import time
import math
import random
import argparse
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
from transformers import AutoTokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, ".hf_cache"))

from src.organoid_layers import build_enhanced_organoid_spheroid
from src.multimodal_layers import build_multimodal_organoid_core
from src.tightened_curriculum import get_tightened_stream
from src.data_loaders import get_cerebro_cognitive_stream
from src.advanced_datasets import get_multimodal_organoid_stream
from training.tightened_agi_benchmark import run_tightened_benchmark


def compute_focal_weighted_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
    step_focal_weight: float = 3.0,
    conclusion_focal_weight: float = 6.0,
) -> torch.Tensor:
    """
    Two-Tier Answer-Focal Loss:
    - First half (prompt & syntactic setup): 1.0x
    - Third quarter (intermediate deduction steps): 3.0x
    - Final quarter (exact target entity / number): 6.0x
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
            weights[b, halfway:] = step_focal_weight
            weights[b, three_quarters:] = conclusion_focal_weight

    valid_mask = (targets != -100).float()
    weighted_loss = (ce_loss * weights * valid_mask).sum() / (weights * valid_mask).sum().clamp(min=1.0)
    return weighted_loss


def main():
    parser = argparse.ArgumentParser(description="Train Living Organoid on Deep Reasoning Datasets")
    parser.add_argument("--steps", type=int, default=360, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=2, help="Micro-batch size for ~1.5 - 2.0 GB VRAM")
    parser.add_argument("--grad_accum", type=int, default=6, help="Gradient accumulation steps (effective batch 12)")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate")
    parser.add_argument("--eval_interval", type=int, default=40, help="Benchmark evaluation interval")
    parser.add_argument("--model_type", type=str, default="multimodal", choices=["multimodal", "enhanced"])
    parser.add_argument("--model_size", type=str, default="100m", choices=["33m", "100m"], help="Model size preset")
    parser.add_argument("--save_prefix", type=str, default="deep_reasoning_organoid_best")
    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    print("=" * 90)
    print("      DEEP REASONING LIVING ORGANOID TRAINING ENGINE (B^3)      ")
    print("=" * 90)
    print(f"Target Accelerator : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Precision          : {dtype}")
    print(f"Architecture Type  : {args.model_type.upper()} [{args.model_size.upper()}]")
    print(f"Datasets           : 45% Cerebro-Cognitive (50k Parquet) + 45% Tightened Proofs + 10% Multimodal")
    print(f"Objective          : Two-Tier Answer-Focal Loss (1.0x / 3.0x / 6.0x)")
    print(f"Effective Batch    : Micro {args.batch_size} × Accum {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"Total Steps        : {args.steps}")
    print("=" * 90)

    # Initialize model architecture
    if args.model_type == "multimodal":
        model = build_multimodal_organoid_core(vocab_size=vocab_size, preset=args.model_size)
    else:
        model = build_enhanced_organoid_spheroid(vocab_size=vocab_size)

    # Resume/Transplant from best existing weights
    mm_100m = os.path.join(ROOT_DIR, "checkpoints", "multimodal_organoid_100m_best.pt")
    mm_best = os.path.join(ROOT_DIR, "checkpoints", "multimodal_organoid_best.pt")
    tight_best = os.path.join(ROOT_DIR, "checkpoints", "tightened_organoid_best.pt")

    if args.model_size == "100m" and os.path.exists(mm_100m):
        init_ckpt = mm_100m
    elif os.path.exists(mm_best) and args.model_type == "multimodal":
        init_ckpt = mm_best
    else:
        init_ckpt = tight_best

    if os.path.exists(init_ckpt):
        print(f"Resuming/transplanting base weights from {init_ckpt}...")
        sd = torch.load(init_ckpt, map_location="cpu")
        model_sd = model.state_dict()
        transferred = 0
        warm_weights = 0
        for k, v in sd.items():
            if k.startswith("sensory_pole.") and args.model_type == "multimodal":
                k = k.replace("sensory_pole.", "lang_pole.")
            if k in model_sd:
                if model_sd[k].shape == v.shape:
                    model_sd[k].copy_(v)
                    transferred += 1
                    warm_weights += v.numel()
                elif model_sd[k].dim() == v.dim():
                    slices = tuple(slice(0, min(s_src, s_tgt)) for s_src, s_tgt in zip(v.shape, model_sd[k].shape))
                    model_sd[k][slices].copy_(v[slices])
                    transferred += 1
                    t_num = 1
                    for sl in slices:
                        t_num *= (sl.stop - sl.start)
                    warm_weights += t_num
        model.load_state_dict(model_sd)
        print(f"  --> Transferred {transferred} weight tensors successfully ({warm_weights:,} warm parameters mapped).")

    model.to(device=device, dtype=dtype)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters   : {total_p:,} ({total_p/1e6:.2f}M)")

    size_suffix = f"_{args.model_size}" if args.model_size == "100m" else ""
    best_ckpt = os.path.join(ROOT_DIR, "checkpoints", f"{args.save_prefix}{size_suffix}.pt")
    canonical_mm = os.path.join(ROOT_DIR, "checkpoints", f"multimodal_organoid{size_suffix}_best.pt")
    os.makedirs(os.path.join(ROOT_DIR, "checkpoints"), exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.05)

    # Initialize data streams
    print("Initializing Multi-Source Reasoning Streams...")
    cog_stream = get_cerebro_cognitive_stream(tokenizer, batch_size=args.batch_size, max_prompt_len=256, max_resp_len=56)
    tight_stream = get_tightened_stream(tokenizer, batch_size=args.batch_size, max_prompt_len=256, max_resp_len=56)
    mm_stream = get_multimodal_organoid_stream(tokenizer, batch_size=args.batch_size, max_prompt_len=256, max_resp_len=56)

    t_start = time.time()
    best_pass_rate = 0.0 if args.model_size == "100m" else (83.3 if args.model_type == "multimodal" else 58.3)
    best_composite_score = 0.0 if args.model_size == "100m" else (66.4 if args.model_type == "multimodal" else 56.4)
    best_loss = 999.0
    step = 0

    print("Beginning Deep Reasoning Manifold Optimization...\n")

    while step < args.steps:
        step += 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        thinking_cycles = random.randint(2, 4)

        for _ in range(args.grad_accum):
            r = random.random()
            images, audio = None, None

            if r < 0.45:
                # 45% Cerebro-Cognitive Parquet (50,000 real problems)
                prompts, targets = next(cog_stream)
            elif r < 0.90:
                # 45% Tightened Procedural Proofs
                prompts, targets = next(tight_stream)
            else:
                # 10% Multimodal Cross-Modal Binding
                if args.model_type == "multimodal":
                    prompts, targets, images, audio = next(mm_stream)
                    images = images.to(device)
                    audio = audio.to(device)
                else:
                    prompts, targets = next(tight_stream)

            prompts = prompts.to(device)
            targets = targets.to(device)

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
                if args.model_type == "multimodal":
                    logits = model(
                        prompt_tokens=prompts,
                        target_tokens=targets,
                        images=images,
                        audio=audio,
                        thinking_cycles=thinking_cycles,
                    )
                else:
                    logits = model(prompts, targets, thinking_cycles=thinking_cycles)

                loss = compute_focal_weighted_loss(logits, targets, vocab_size=vocab_size)
                scaled_loss = loss / args.grad_accum

            scaled_loss.backward()
            accum_loss += scaled_loss.item()
            del prompts, targets, images, audio, logits, loss, scaled_loss

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

            is_better = (pass_rate > best_pass_rate) or (pass_rate == best_pass_rate and comp_score > best_composite_score)
            if is_better:
                best_pass_rate = max(best_pass_rate, pass_rate)
                best_composite_score = max(best_composite_score, comp_score)
                torch.save(model.state_dict(), best_ckpt)
                if args.model_type == "multimodal":
                    torch.save(model.state_dict(), canonical_mm)
                print(f"  ★ [NEW BEST REASONING CHECKPOINT] Saved to {best_ckpt} (Strict Pass: {best_pass_rate:.1f}% | Score: {best_composite_score:.1f}% | Loss: {loss_val:.4f})")

            if loss_val < best_loss:
                best_loss = loss_val

    latest_ckpt = os.path.join(ROOT_DIR, "checkpoints", f"{args.save_prefix}{size_suffix}_latest.pt")
    torch.save(model.state_dict(), latest_ckpt)
    total_time = time.time() - t_start

    print("=" * 90)
    print(f"Deep Reasoning Training Complete in {total_time/60:.2f} minutes.")
    print(f"Best Loss               : {best_loss:.4f}")
    print(f"Best Strict Pass Rate   : {best_pass_rate:.1f}%")
    print(f"Best Composite Score    : {best_composite_score:.1f}%")
    print(f"Saved Checkpoint        : {best_ckpt}")
    print("=" * 90)


if __name__ == "__main__":
    main()
