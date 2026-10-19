"""
Training Script for Minimal Bipolar Hemispheric Wave Spheroid (v2)
- Input on Hemisphere A (Sensory Sheet)
- Physical 3D Spatial Wave Diffusion across Spheroid
- Output on Hemisphere B (Motor Sheet)
- Internal Corollary Discharge into Subcortical Core
"""

import os
import sys
import time
import math
import argparse
import random
import torch
import torch.nn as nn
from transformers import AutoTokenizer

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.hemispheric_spheroid import build_minimal_hemispheric_spheroid
from src.data_loaders import get_specialized_cognitive_stream, get_cognitive_and_language_stream


def main():
    parser = argparse.ArgumentParser(description="Train Minimal Bipolar Hemispheric Spheroid")
    parser.add_argument("--steps", type=int, default=500, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=2, help="Micro-batch size")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (AdamW)")
    parser.add_argument("--eval_interval", type=int, default=50, help="Evaluation and checkpoint interval")
    parser.add_argument("--save_prefix", type=str, default="hemispheric_minimal", help="Checkpoint prefix")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--dataset_mode", type=str, default="cognitive", choices=["hybrid", "cognitive"], help="Dataset stream mode")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    # Instantiate Minimal Bipolar Hemispheric Model (~42M params)
    model = build_minimal_hemispheric_spheroid()
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming model weights from {args.resume}...")
        model.load_state_dict(torch.load(args.resume, map_location="cpu"))
    elif os.path.exists(os.path.join("checkpoints", f"{args.save_prefix}_best.pt")) and not args.resume:
        default_resume = os.path.join("checkpoints", f"{args.save_prefix}_best.pt")
        print(f"Auto-resuming existing best weights from {default_resume}...")
        model.load_state_dict(torch.load(default_resume, map_location="cpu"))

    model.to(device=device, dtype=dtype)

    total_p = sum(p.numel() for p in model.parameters())

    print("=" * 80)
    print("      BIPOLAR HEMISPHERIC WAVE SPHEROID: MINIMAL AI TRAINING             ")
    print("=" * 80)
    print(f"Model Parameters   : {total_p:,} ({total_p/1e6:.2f}M params)")
    print(f"Target Accelerator : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Spatial Topology   : Hemisphere A (z=0) -> 3D Volume (8x8x8) -> Hemisphere B (z=7)")
    print(f"Steps              : {args.steps} | Effective Batch: {args.batch_size * args.grad_accum}")
    print(f"Optimizer          : ADAMW (lr={args.lr:.2e})")
    print("=" * 80)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.05)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    if args.dataset_mode == "hybrid":
        print("Dataset Stream     : 50% Specialized Cognitive Reasoning + 50% Natural Language (LMSYS)")
        data_stream = get_cognitive_and_language_stream(
            tokenizer, batch_size=args.batch_size, cognitive_ratio=0.50, max_prompt_len=256, max_resp_len=64
        )
    else:
        print("Dataset Stream     : 100% Specialized Cognitive Reasoning Stream")
        data_stream = get_specialized_cognitive_stream(
            tokenizer, batch_size=args.batch_size, max_prompt_len=256, max_resp_len=64
        )

    best_ckpt_path = os.path.join("checkpoints", f"{args.save_prefix}_best.pt")
    latest_ckpt_path = os.path.join("checkpoints", f"{args.save_prefix}_latest.pt")
    os.makedirs("checkpoints", exist_ok=True)

    test_queries = [
        "If 5 pencils cost $15, how much do 8 pencils cost?",
        "A cyclist rides at a speed of 12 miles per hour. How far does the cyclist travel in 5 hours?",
        "A rectangular swimming pool is 20 meters long and 10 meters wide. What is its perimeter and area?",
        "Why does hot air rise while cold air sinks?",
        "All birds have feathers. A robin is a bird. Does a robin have feathers?",
        "Why does a metal spoon feel colder than a wooden spoon at room temperature?",
    ]

    t_start = time.time()
    best_loss = 999.0
    step = 0

    while step < args.steps:
        step += 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(args.grad_accum):
            prompts, targets = next(data_stream)
            prompts = prompts.to(device)
            targets = targets.to(device)
            B, ans_len = targets.shape

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
                # 1. Ingest into Hemisphere A (Sensory Surface)
                state_3d, sensory_memory = model.ingest_stimulus(prompts)
                
                # 2. Deliberate across Corpus Callosum towards Hemisphere B
                state_3d = model.deliberate(state_3d, cycles=2)

                refractory_trace = torch.zeros(B, vocab_size, device=device, dtype=dtype)
                logits_list = []

                # 3. Efferent readout on Hemisphere B surface
                for t in range(ans_len):
                    logits_t, state_3d, refractory_trace = model.emit_next_token(
                        state_3d, sensory_memory=sensory_memory, refractory_trace=refractory_trace, step_idx=t
                    )
                    logits_list.append(logits_t)

                predicted_logits = torch.stack(logits_list, dim=1)
                loss = criterion(predicted_logits.view(-1, vocab_size), targets.view(-1))
                scaled_loss = loss / args.grad_accum

            scaled_loss.backward()
            accum_loss += scaled_loss.item()
            del prompts, targets, state_3d, sensory_memory, logits_list, predicted_logits, loss, scaled_loss, refractory_trace

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if device.type == "cuda" and step % 2 == 0:
            torch.cuda.empty_cache()

        loss_val = accum_loss

        # Step Progress Logging
        if step % 10 == 0 or step == 1:
            ppl = math.exp(min(loss_val, 15))
            vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            cur_lr = scheduler.get_last_lr()[0]
            print(f"Step [{step:4,d}/{args.steps}] | Loss: {loss_val:.4f} | PPL: {ppl:8.2f} | LR: {cur_lr:.2e} | VRAM: {vram:.0f} MB")

        # Evaluation & Checkpoint Interval
        if step % args.eval_interval == 0:
            if loss_val < best_loss:
                best_loss = loss_val
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"  ★ [New Best Checkpoint] Saved to {best_ckpt_path} (Loss: {best_loss:.4f})")

            # Cognitive Probe Inspection
            model.eval()
            print("\n" + "·" * 60)
            print(f"  [Cognitive Inspection at Step {step} | Loss: {loss_val:.4f}]")
            for q in test_queries:
                q_toks = tokenizer.encode(q, return_tensors="pt").to(device)
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
                        out_toks = model.solve_with_active_thinking(
                            q_toks, thinking_cycles=2, max_answer_tokens=20, temperature=0.7, eos_token_id=tokenizer.eos_token_id
                        )
                resp = tokenizer.decode(out_toks[0]) if len(out_toks[0]) > 0 else "<empty>"
                print(f"  Q: \"{q[:55]}...\"")
                print(f"  A: \"{resp.strip()}\"")
            print("·" * 60 + "\n")

    torch.save(model.state_dict(), latest_ckpt_path)
    total_time = time.time() - t_start
    print("=" * 80)
    print(f"Training Concluded in {total_time/60:.2f} minutes.")
    print(f"Best Weights  : {best_ckpt_path} (Loss: {best_loss:.4f})")
    print(f"Latest Weights: {latest_ckpt_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
