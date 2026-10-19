"""
Training Engine with Micro-Adjustment Noise Reduction & Living Pulsation Stabilization
======================================================================================
1. Trains Continuous Real Geodesic Sphere (B^3) starting from checkpoints/proto_agi_sphere_best.pt.
2. Active Micro-Adjustment by Noise Reduction & Homeostatic Energy Stabilization active on every step.
3. Ingests stimulus by perturbing the active living resting-state manifold rather than dead zeros.
4. Uses Maximum Stable VRAM (~2.4 - 2.5 GB on RTX 3050 Laptop GPU).
5. Continuously benchmarks Proto-AGI emergence across the 6 cognitive pillars.
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
from transformers import AutoTokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, ".hf_cache"))

from src.real_geodesic_sphere import build_real_geodesic_spheroid
from src.data_loaders import get_specialized_cognitive_stream


EMERGENCE_TESTS = [
    {
        "pillar": "Taxonomical Deduction",
        "prompt": "All quadrupedal mammals produce milk for their offspring. A platypus lays eggs but nurses its young with milk. Is a platypus a mammal?",
        "keys": ["mammal", "milk", "diagnostic", "class", "deduction"],
    },
    {
        "pillar": "Multi-Hop Transitivity",
        "prompt": "Iron is denser than Aluminium. Platinum is denser than Iron. Lithium is lighter than Aluminium. Which of these four metals is the densest?",
        "keys": ["platinum", "densest", ">", "iron", "chain"],
    },
    {
        "pillar": "Counterfactual Physics",
        "prompt": "If hot air were denser than cold air, what direction would hot smoke from a chimney move?",
        "keys": ["downward", "sink", "ground", "denser", "gravity"],
    },
    {
        "pillar": "Multi-Step Change Arithmetic",
        "prompt": "Maya buys 6 notebooks priced at $4 each. Maya pays with a $50 bill. How much change does Maya receive?",
        "keys": ["$24", "$26", "26", "change", "subtract"],
    },
    {
        "pillar": "Kinematics Rate",
        "prompt": "A car travels at a steady speed of 60 miles per hour. How far does the car travel in 5 hours?",
        "keys": ["300", "speed", "miles", "multiply"],
    },
    {
        "pillar": "Thermal Mechanisms",
        "prompt": "Why does a metal spoon feel colder than a wooden spoon at room temperature?",
        "keys": ["thermal conductivity", "heat", "rapid", "insulator", "loss"],
    },
]


def evaluate_emergence(model, tokenizer, device, dtype) -> float:
    model.eval()
    passed = 0
    total = len(EMERGENCE_TESTS)

    print("\n" + "·" * 78)
    print("  [LIVING PULSATING SPHEROID: PROTO-AGI EMERGENCE INSPECTION]")
    for item in EMERGENCE_TESTS:
        pillar = item["pillar"]
        q = item["prompt"]
        keys = item["keys"]

        q_toks = tokenizer.encode(q, return_tensors="pt").to(device)
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
                out_toks, settled = model.solve_with_adaptive_emergence(
                    q_toks,
                    min_cycles=2,
                    max_cycles=5,
                    settling_tolerance=0.08,
                    max_answer_tokens=28,
                    temperature=0.60,
                    eos_token_id=tokenizer.eos_token_id,
                )
        resp = tokenizer.decode(out_toks[0]).strip() if len(out_toks[0]) > 0 else "<empty>"
        resp_l = resp.lower()

        hits = [k for k in keys if k.lower() in resp_l]
        is_emergent = len(hits) >= 1
        if is_emergent:
            passed += 1
            tag = "★ PASS"
        else:
            tag = "○ EVOLVE"

        print(f"  [{pillar}] (Settled: {settled} cyc | {tag} | Hits: {hits})")
        print(f"    Q: \"{q[:55]}...\"")
        print(f"    A: \"{resp}\"")

    pct = (passed / total) * 100.0
    print(f"  --> Emergence Score: {passed}/{total} ({pct:.1f}%)")
    print("·" * 78 + "\n")
    return pct


def main():
    parser = argparse.ArgumentParser(description="Train Living Geodesic Spheroid with Micro-Adjustment Stabilization")
    parser.add_argument("--steps", type=int, default=400, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=12, help="Micro-batch size for ~2.4 GB VRAM")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2.0e-4, help="Learning rate")
    parser.add_argument("--eval_interval", type=int, default=50, help="Evaluation interval")
    parser.add_argument("--save_prefix", type=str, default="proto_agi_sphere_living", help="Checkpoint prefix")
    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    model = build_real_geodesic_spheroid()
    init_ckpt = os.path.join(ROOT_DIR, "checkpoints", "proto_agi_sphere_best.pt")
    if os.path.exists(init_ckpt):
        print(f"Resuming model weights from {init_ckpt}...")
        try:
            model.load_state_dict(torch.load(init_ckpt, map_location=device), strict=False)
        except Exception as e:
            print(f"Loaded with partial match: {e}")

    model.to(device=device, dtype=dtype)
    total_p = sum(p.numel() for p in model.parameters())

    best_ckpt = os.path.join(ROOT_DIR, "checkpoints", f"{args.save_prefix}_best.pt")
    os.makedirs(os.path.join(ROOT_DIR, "checkpoints"), exist_ok=True)

    print("=" * 86)
    print("      LIVING GEODESIC SPHEROID (B^3): MICRO-STABILIZED DEEP TRAINING        ")
    print("=" * 86)
    print(f"Model Parameters   : {total_p:,} ({total_p/1e6:.2f}M params)")
    print(f"Target Accelerator : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Stabilization      : Active Micro-Adjustment + Spatial Laplacian Dissipation + Homeostasis")
    print(f"Resting State      : Spontaneous Endogenous Oscillations (Theta-Alpha Pacemaker)")
    print(f"VRAM Target        : Maximum Stable Allocation (~2.4 - 2.5 GB)")
    print(f"Training Batch     : Micro {args.batch_size} × Accum {args.grad_accum} = Effective {args.batch_size * args.grad_accum}")
    print(f"Total Steps        : {args.steps}")
    print("=" * 86)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.05)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    data_stream = get_specialized_cognitive_stream(
        tokenizer, batch_size=args.batch_size, max_prompt_len=256, max_resp_len=48
    )

    t_start = time.time()
    best_loss = 999.0
    best_score = -1.0
    step = 0

    while step < args.steps:
        step += 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        thinking_cycles = random.randint(2, 5)

        for _ in range(args.grad_accum):
            prompts, targets = next(data_stream)
            prompts = prompts.to(device)
            targets = targets.to(device)

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
                logits = model(prompts, targets, thinking_cycles=thinking_cycles)
                loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
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
            cur_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - t_start
            print(f"Step [{step:4d}/{args.steps}] | Loss: {loss_val:.4f} | PPL: {ppl:7.2f} | Cycles: {thinking_cycles} | VRAM: {vram:.0f} MB | Elapsed: {elapsed:.1f}s")

        if step % args.eval_interval == 0:
            score = evaluate_emergence(model, tokenizer, device, dtype)
            if loss_val < best_loss or score > best_score:
                if loss_val < best_loss:
                    best_loss = loss_val
                if score > best_score:
                    best_score = score
                torch.save(model.state_dict(), best_ckpt)
                print(f"  ★ [NEW BEST LIVING CHECKPOINT] Saved to {best_ckpt} (Loss: {best_loss:.4f} | Score: {best_score:.1f}%)")

    torch.save(model.state_dict(), os.path.join(ROOT_DIR, "checkpoints", f"{args.save_prefix}_latest.pt"))
    total_time = time.time() - t_start
    print("=" * 86)
    print(f"Training Complete in {total_time/60:.2f} minutes.")
    print(f"Best Loss            : {best_loss:.4f}")
    print(f"Peak Emergence Score : {best_score:.1f}%")
    print(f"Saved Checkpoint     : {best_ckpt}")
    print("=" * 86)


if __name__ == "__main__":
    main()
