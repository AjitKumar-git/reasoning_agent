"""
Training & Emergence Scaling for Enhanced Living Organoid Spheroid (B^3)
========================================================================
Applies 5 Biological Layers on top of the Organoid Model:
1. Thalamic Gateway Layer (Sensory Salience & Reticular Gating)
2. Cortical Columnar Layer (Mini-Columns & DoG Lateral Inhibition on S^2)
3. Cross-Frequency PAC Modulator (Theta-Gamma Nested Phase Clocks)
4. Astrocytic Glial Neuromodulator Field (Tripartite Metabolic Regulation)
5. Hippocampal Attractor & Sharp-Wave Ripple (SWR) Resting-State Replay

Trained on Advanced Multi-Curriculum Streams:
- ARC-AGI Spatial Transformations, Symmetry & Pathfinding
- Big-Bench Hard (BBH) Algorithmic Logic & State Tracking
- Proof-Style Quantitative Relations & Modular Remainder Arithmetic
- Biochemical & Cellular Causal Mechanisms (Enzymes, Chemiosmosis, Le Chatelier)
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
from src.advanced_datasets import get_advanced_organoid_stream
from src.data_loaders import get_specialized_cognitive_stream


EMERGENCE_TESTS = [
    {
        "pillar": "Deductive Syllogisms & Taxonomies",
        "prompt": "All quadrupedal mammals produce milk for their young. A platypus lays eggs but nurses its young with milk. Is a platypus a mammal?",
        "keys": ["mammal", "milk", "class", "deduction", "yes"],
    },
    {
        "pillar": "Multi-Hop Transitive Logic",
        "prompt": "Iron is denser than Aluminium. Platinum is denser than Iron. Lithium is lighter than Aluminium. Which of these four metals is the densest?",
        "keys": ["platinum", "densest", ">", "iron", "chain"],
    },
    {
        "pillar": "Counterfactual Physics",
        "prompt": "If hot air were denser than cold air, what direction would hot smoke from a chimney move?",
        "keys": ["downward", "sink", "ground", "denser", "gravity"],
    },
    {
        "pillar": "Multi-Step Arithmetic & Kinematics",
        "prompt": "Maya buys 6 notebooks priced at $4 each. Maya pays with a $50 bill. How much change does Maya receive?",
        "keys": ["$24", "$26", "26", "change", "subtract"],
    },
    {
        "pillar": "Biochemical & Kinetic Mechanisms",
        "prompt": "How does a competitive enzyme inhibitor affect the apparent Vmax and Km of an enzyme?",
        "keys": ["Km", "Vmax", "affinity", "compete", "inhibitor", "active site"],
    },
    {
        "pillar": "Spatial Symmetry & ARC-AGI Grid Invariants",
        "prompt": "A 3x3 matrix with rows ['. # .', '. # .', '. # .'] has a vertical line in column 2. What does this matrix look like after a horizontal reflection?",
        "keys": ["reflection", "vertical line", "column 2", "unchanged", "symmetric", "#"],
    },
]


def evaluate_emergence(model, tokenizer, device, dtype) -> float:
    model.eval()
    passed = 0
    total = len(EMERGENCE_TESTS)

    print("\n" + "·" * 78)
    print("  [ENHANCED ORGANOID SPHEROID: PROTO-AGI EMERGENCE INSPECTION]")
    for item in EMERGENCE_TESTS:
        pillar = item["pillar"]
        q = item["prompt"]
        keys = item["keys"]

        q_toks = tokenizer.encode(q, return_tensors="pt").to(device)
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
                out_tokens = model.generate_answer(
                    q_toks, max_tokens=28, thinking_cycles=4, eos_token_id=tokenizer.eos_token_id, temperature=0.6
                )

        resp = tokenizer.decode(out_tokens).strip() if out_tokens else "<empty>"
        resp_l = resp.lower()

        hits = [k for k in keys if k.lower() in resp_l]
        is_emergent = len(hits) >= 1
        if is_emergent:
            passed += 1
            tag = "★ PASS"
        else:
            tag = "○ EVOLVE"

        print(f"  [{pillar}] ({tag} | Hits: {hits})")
        print(f"    Q: \"{q[:55]}...\"")
        print(f"    A: \"{resp}\"")

    pct = (passed / total) * 100.0
    print(f"  --> Emergence Score: {passed}/{total} ({pct:.1f}%)")
    print("·" * 78 + "\n")
    return pct


def main():
    parser = argparse.ArgumentParser(description="Train Enhanced Living Organoid Spheroid")
    parser.add_argument("--steps", type=int, default=250, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Micro-batch size for ~2.5 GB VRAM")
    parser.add_argument("--grad_accum", type=int, default=3, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2.0e-4, help="Learning rate")
    parser.add_argument("--eval_interval", type=int, default=50, help="Evaluation interval")
    parser.add_argument("--save_prefix", type=str, default="enhanced_organoid_best", help="Checkpoint prefix")
    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    model = build_enhanced_organoid_spheroid(vocab_size=vocab_size)

    # Resume base weights from proto_agi_sphere_living_best.pt
    init_ckpt = os.path.join(ROOT_DIR, "checkpoints", "proto_agi_sphere_living_best.pt")
    if os.path.exists(init_ckpt):
        print(f"Resuming base geodesic spheroid weights from {init_ckpt}...")
        sd = torch.load(init_ckpt, map_location=device)
        # Remap keys if necessary
        model_sd = model.state_dict()
        transferred = 0
        for k, v in sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                model_sd[k] = v
                transferred += 1
        model.load_state_dict(model_sd)
        print(f"  --> Transferred {transferred} weight tensors successfully.")

    model.to(device=device, dtype=dtype)
    total_p = sum(p.numel() for p in model.parameters())

    best_ckpt = os.path.join(ROOT_DIR, "checkpoints", f"{args.save_prefix}.pt")
    os.makedirs(os.path.join(ROOT_DIR, "checkpoints"), exist_ok=True)

    print("=" * 86)
    print("      ENHANCED LIVING ORGANOID SPHEROID (B^3) WITH BIOLOGICAL LAYERS        ")
    print("=" * 86)
    print(f"Model Parameters   : {total_p:,} ({total_p/1e6:.2f}M params)")
    print(f"Target Accelerator : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Biological Layers  : Thalamic Gateway + Columnar DoG + PAC Coupler + Astrocytes + Hippocampus")
    print(f"Curriculums        : ARC-AGI Spatial + BBH Logic + Quantitative Proofs + Biochemical Mechanisms")
    print(f"Training Batch     : Micro {args.batch_size} × Accum {args.grad_accum} = Effective {args.batch_size * args.grad_accum}")
    print(f"Total Steps        : {args.steps}")
    print("=" * 86)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.05)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Interleaved Dual Streams:
    # 1. Advanced Multi-Curriculum Stream (ARC-AGI + BBH + Bio Mechanisms)
    advanced_stream = get_advanced_organoid_stream(
        tokenizer, batch_size=args.batch_size, max_prompt_len=256, max_resp_len=48
    )
    # 2. Specialized Quantitative & Causal Physics Stream
    cognitive_stream = get_specialized_cognitive_stream(
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
        thinking_cycles = random.randint(2, 4)

        for _ in range(args.grad_accum):
            # 50% Advanced ARC/BBH/Bio, 50% Quantitative/Causal
            if random.random() < 0.50:
                prompts, targets = next(advanced_stream)
            else:
                prompts, targets = next(cognitive_stream)

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
                print(f"  ★ [NEW BEST ENHANCED ORGANOID CHECKPOINT] Saved to {best_ckpt} (Loss: {best_loss:.4f} | Score: {best_score:.1f}%)")

    latest_ckpt = os.path.join(ROOT_DIR, "checkpoints", f"{args.save_prefix}_latest.pt")
    torch.save(model.state_dict(), latest_ckpt)
    total_time = time.time() - t_start
    print("=" * 86)
    print(f"Training Complete in {total_time/60:.2f} minutes.")
    print(f"Best Loss            : {best_loss:.4f}")
    print(f"Peak Emergence Score : {best_score:.1f}%")
    print(f"Saved Checkpoint     : {best_ckpt}")
    print("=" * 86)


if __name__ == "__main__":
    main()
