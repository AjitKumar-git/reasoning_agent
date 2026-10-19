"""
Multimodal Training & Emergence Engine for Living Organoid Core (B^3)
=====================================================================
Integrates Visual Retinotopic Sheet + Auditory Tonotopic Sheet into the Core.
- Visual: Ingests 2D image signals [B, 3, 64, 64] onto the West Pole (x < -0.05).
- Auditory: Ingests 1D audio waveforms [B, 1, 16000] onto the South-West Pole (y < -0.05).
- Language: Ingests environmental stimuli onto the South Pole (z < -0.05).
- Cross-Modal Deliberation: Physical 3D wave interference in the unit ball (B^3).
- Motor Readout: Decodes from North Pole (z > 0.05) with internal corollary discharge.
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
from src.multimodal_layers import build_multimodal_organoid_core
from src.tightened_curriculum import get_tightened_stream
from src.advanced_datasets import (
    get_multimodal_organoid_stream,
    get_advanced_organoid_stream,
    generate_synthetic_multimodal_sample,
)
from training.train_tightened_agi import compute_focal_weighted_loss


MULTIMODAL_EVAL_TESTS = [
    {
        "modality": "Cross-Modal Binding",
        "prompt": "Describe the visual geometry and acoustic frequency characteristics present in the sensory field.",
        "keys": ["visual", "sheet", "auditory", "tonotopic", "binds", "geometry"],
    },
    {
        "modality": "Deductive Taxonomies",
        "prompt": "All quadrupedal mammals produce milk for their young. A platypus lays eggs but nurses its young with milk. Is a platypus a mammal?",
        "keys": ["mammal", "milk", "class", "deduction", "yes"],
    },
    {
        "modality": "Multi-Hop Transitive Logic",
        "prompt": "Iron is denser than Aluminium. Platinum is denser than Iron. Lithium is lighter than Aluminium. Which of these four metals is the densest?",
        "keys": ["platinum", "densest", ">", "iron", "chain"],
    },
    {
        "modality": "Multi-Step Arithmetic & Kinematics",
        "prompt": "Maya buys 6 notebooks priced at $4 each. Maya pays with a $50 bill. How much change does Maya receive?",
        "keys": ["$24", "$26", "26", "change", "subtract"],
    },
]


def evaluate_multimodal_emergence(model, tokenizer, device, dtype) -> float:
    model.eval()
    passed = 0
    total = len(MULTIMODAL_EVAL_TESTS)

    print("\n" + "·" * 78)
    print("  [MULTIMODAL ORGANOID CORE: VISION + AUDIO + REASONING INSPECTION]")

    for idx, item in enumerate(MULTIMODAL_EVAL_TESTS):
        modality = item["modality"]
        q = item["prompt"]
        keys = item["keys"]

        q_toks = tokenizer.encode(q, return_tensors="pt").to(device)

        # For cross-modal test, synthesize a paired image and audio
        img_in = None
        aud_in = None
        if idx == 0:
            _, _, sample_img, sample_aud = generate_synthetic_multimodal_sample()
            img_in = sample_img.unsqueeze(0).to(device=device, dtype=torch.float32)
            aud_in = sample_aud.unsqueeze(0).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
                out_toks = model.generate_multimodal(
                    prompt_tokens=q_toks,
                    images=img_in,
                    audio=aud_in,
                    max_tokens=28,
                    thinking_cycles=4,
                    eos_token_id=tokenizer.eos_token_id,
                    temperature=0.6,
                )

        resp = tokenizer.decode(out_toks).strip() if out_toks else "<empty>"
        resp_l = resp.lower()

        hits = [k for k in keys if k.lower() in resp_l]
        is_emergent = len(hits) >= 1
        if is_emergent:
            passed += 1
            tag = "★ PASS"
        else:
            tag = "○ EVOLVE"

        print(f"  [{modality}] ({tag} | Hits: {hits})")
        print(f"    Q: \"{q[:55]}...\"")
        print(f"    A: \"{resp}\"")

    pct = (passed / total) * 100.0
    print(f"  --> Multimodal Emergence Score: {passed}/{total} ({pct:.1f}%)")
    print("·" * 78 + "\n")
    return pct


def main():
    parser = argparse.ArgumentParser(description="Train Multimodal Living Organoid Core")
    parser.add_argument("--steps", type=int, default=120, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=6, help="Micro-batch size for ~2.3 GB VRAM")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2.0e-4, help="Learning rate")
    parser.add_argument("--eval_interval", type=int, default=30, help="Evaluation interval")
    parser.add_argument("--save_prefix", type=str, default="multimodal_organoid_best", help="Checkpoint prefix")
    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    model = build_multimodal_organoid_core(vocab_size=vocab_size)

    # Resume base weights: prefer multimodal_organoid_best.pt, then tightened_organoid_best.pt
    mm_ckpt = os.path.join(ROOT_DIR, "checkpoints", "multimodal_organoid_best.pt")
    tight_ckpt = os.path.join(ROOT_DIR, "checkpoints", "tightened_organoid_best.pt")
    enh_ckpt = os.path.join(ROOT_DIR, "checkpoints", "enhanced_organoid_best.pt")

    if os.path.exists(mm_ckpt):
        init_ckpt = mm_ckpt
    elif os.path.exists(tight_ckpt):
        init_ckpt = tight_ckpt
    else:
        init_ckpt = enh_ckpt

    if os.path.exists(init_ckpt):
        print(f"Resuming base weights from {init_ckpt}...")
        sd = torch.load(init_ckpt, map_location=device)
        model_sd = model.state_dict()
        transferred = 0
        for k, v in sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                model_sd[k] = v
                transferred += 1
            elif k.startswith("sensory_pole."):
                lang_k = k.replace("sensory_pole.", "lang_pole.")
                if lang_k in model_sd and model_sd[lang_k].shape == v.shape:
                    model_sd[lang_k] = v
                    transferred += 1
        model.load_state_dict(model_sd)
        print(f"  --> Transferred {transferred} weight tensors successfully.")

    model.to(device=device, dtype=dtype)
    total_p = sum(p.numel() for p in model.parameters())

    best_ckpt = os.path.join(ROOT_DIR, "checkpoints", f"{args.save_prefix}.pt")
    os.makedirs(os.path.join(ROOT_DIR, "checkpoints"), exist_ok=True)

    print("=" * 86)
    print("      MULTIMODAL LIVING ORGANOID CORE (B^3): VISION + AUDIO + REASONING      ")
    print("=" * 86)
    print(f"Model Parameters   : {total_p:,} ({total_p/1e6:.2f}M params)")
    print(f"Target Accelerator : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Sensory Modalities : Retinotopic Vision (West) + Tonotopic Audio (SW) + Language (South)")
    print(f"Core Dynamics      : 3D Unit Ball Wave Conduction + DoG Lateral Inhibition + Corollary Discharge")
    print(f"Training Batch     : Micro {args.batch_size} × Accum {args.grad_accum} = Effective {args.batch_size * args.grad_accum}")
    print(f"Total Steps        : {args.steps}")
    print("=" * 86)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.05)

    # Streams
    mm_stream = get_multimodal_organoid_stream(tokenizer, batch_size=args.batch_size, max_prompt_len=256, max_resp_len=56)
    tight_stream = get_tightened_stream(tokenizer, batch_size=args.batch_size, max_prompt_len=256, max_resp_len=56)
    cog_stream = get_advanced_organoid_stream(tokenizer, batch_size=args.batch_size, max_prompt_len=256, max_resp_len=56)

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
            r = random.random()
            if r < 0.40:
                prompts, targets, images, audio = next(mm_stream)
                images = images.to(device)
                audio = audio.to(device)
            elif r < 0.80:
                prompts, targets = next(tight_stream)
                images, audio = None, None
            else:
                prompts, targets = next(cog_stream)
                images, audio = None, None

            prompts = prompts.to(device)
            targets = targets.to(device)

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
                logits = model(
                    prompt_tokens=prompts,
                    target_tokens=targets,
                    images=images,
                    audio=audio,
                    thinking_cycles=thinking_cycles,
                )
                loss = compute_focal_weighted_loss(logits, targets, vocab_size=vocab_size, focal_weight=3.5)
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
            score = evaluate_multimodal_emergence(model, tokenizer, device, dtype)
            if loss_val < best_loss or score > best_score:
                if loss_val < best_loss:
                    best_loss = loss_val
                if score > best_score:
                    best_score = score
                torch.save(model.state_dict(), best_ckpt)
                print(f"  ★ [NEW BEST MULTIMODAL CHECKPOINT] Saved to {best_ckpt} (Loss: {best_loss:.4f} | Score: {best_score:.1f}%)")

    latest_ckpt = os.path.join(ROOT_DIR, "checkpoints", f"{args.save_prefix}_latest.pt")
    torch.save(model.state_dict(), latest_ckpt)
    total_time = time.time() - t_start
    print("=" * 86)
    print(f"Multimodal Training Complete in {total_time/60:.2f} minutes.")
    print(f"Best Loss            : {best_loss:.4f}")
    print(f"Peak Emergence Score : {best_score:.1f}%")
    print(f"Saved Checkpoint     : {best_ckpt}")
    print("=" * 86)


if __name__ == "__main__":
    main()
