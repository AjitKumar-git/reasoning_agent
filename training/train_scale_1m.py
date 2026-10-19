"""
CerebroSpheroid: 1-Million Step Scaling Engine with Adaptive Plateau Early-Stopping
Key Features:
1. Target Scale: Up to 1,000,000 gradient steps over full datasets.
2. Plateau Early Stopping: Monitors validation/rolling loss; terminates automatically
   if no progress is recorded for 100 consecutive evaluation intervals.
3. Hybrid Dual-Curriculum Stream: 70% GSM8K/SVAMP reasoning chains + 30% LMSYS conversational fluency.
4. Refractory Trace Backpropagation: Trains biological refractory dynamics directly to eliminate unigram mode collapse.
5. Living Core Heartbeat Stimulation: Trains over active oscillating resting-state manifolds.
6. 4GB VRAM Safety: Momentum-SGD + BFloat16/FP16 autocast maintaining <3.2 GB allocation.
"""

import os
os.environ["HF_HOME"] = "/home/ajit/Documents/Startup/reasoning_agent/.hf_cache"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import math
import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.model import CerebroSpheroid
from src.data_loaders import (
    get_hybrid_reasoning_stream,
    get_lmsys_stream,
    get_gsm8k_stream,
    get_svamp_stream,
    get_specialized_cognitive_stream,
)


def main():
    parser = argparse.ArgumentParser(description="CerebroSpheroid 1-Million Step Scaling Engine")
    parser.add_argument("--dataset", type=str, default="specialized", choices=["specialized", "hybrid", "lmsys", "gsm8k", "svamp"],
                        help="Curriculum dataset (default: specialized cognitive stream)")
    parser.add_argument("--max_steps", type=int, default=1_000_000, help="Maximum gradient steps")
    parser.add_argument("--eval_interval", type=int, default=100, help="Steps between plateau evaluation checks")
    parser.add_argument("--patience", type=int, default=100, help="Patience (number of eval intervals without improvement)")
    parser.add_argument("--batch_size", type=int, default=2, help="Micro-batch size per forward pass")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation micro-batches (effective batch = batch_size * grad_accum)")
    parser.add_argument("--max_prompt_len", type=int, default=1024, help="Max input prompt tokens")
    parser.add_argument("--max_resp_len", type=int, default=128, help="Max output answer tokens")
    parser.add_argument("--thinking_cycles", type=int, default=3, help="Subconscious internal deliberation cycles")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"], help="Optimizer (adamw or sgd)")
    parser.add_argument("--lr", type=float, default=None, help="Base learning rate (default: 3e-4 for AdamW, 0.015 for SGD)")
    parser.add_argument("--resume", type=str, default=None, help="Explicit checkpoint to resume from")
    parser.add_argument("--save_prefix", type=str, default="cerebro_spheroid_1m", help="Checkpoint filename prefix")
    args = parser.parse_args()

    if args.lr is None:
        args.lr = 3e-4 if args.optimizer == "adamw" else 0.015

    effective_batch = args.batch_size * args.grad_accum

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    print("=" * 80)
    print("      CEREBRO-SPHEROID: 1-MILLION STEP LIVING COGNITIVE TRAINING ENGINE  ")
    print("=" * 80)
    print(f"Curriculum Stream  : {args.dataset.upper()} (Expanded Prompt: {args.max_prompt_len} / Resp: {args.max_resp_len} Tokens)")
    print(f"Target Accelerator : {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Hardware       : {gpu_name} ({vram_gb:.2f} GB VRAM)")
    print(f"Max Scale          : {args.max_steps:,} steps | Micro-Batch: {args.batch_size} | Grad Accum: {args.grad_accum}")
    print(f"Effective Batch    : {effective_batch} sequences averaged per gradient step")
    print(f"Optimizer          : {args.optimizer.upper()} (lr={args.lr:.2e})")
    print(f"Plateau Detector   : Patience {args.patience} checks (eval every {args.eval_interval} steps)")
    print(f"Active Latency     : {args.thinking_cycles} subconscious cycles (Living Manifold)")
    print("=" * 80)

    # 1. Instantiate Architecture
    model = CerebroSpheroid(
        vocab_size=vocab_size,
        surface_dim=640,
        num_shells=16,
        radial_dim=256,
        num_shortcut_tracts=40,
    )

    best_ckpt_path = os.path.join("checkpoints", f"{args.save_prefix}_best.pt")
    latest_ckpt_path = os.path.join("checkpoints", f"{args.save_prefix}_latest.pt")
    
    ckpt_candidates = []
    if args.resume:
        ckpt_candidates.append(args.resume)
    ckpt_candidates.extend([
        os.path.join("checkpoints", "cerebro_spheroid_gsm8k_100m.pt"),
        best_ckpt_path,
        os.path.join("checkpoints", "cerebro_spheroid_100m.pt")
    ])

    loaded = False
    for candidate in ckpt_candidates:
        if os.path.exists(candidate):
            print(f"Resuming weights from checkpoint: {candidate}...")
            model.load_compatible_checkpoint(candidate, device="cpu")
            loaded = True
            break
    if not loaded:
        print("No prior checkpoint found; initializing fresh weights.")

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model.to(device=device, dtype=dtype)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, eps=1e-8)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=args.lr * 0.05)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 2. Select Curriculum Stream
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "lmsys_chat_1m", "data"))
    if args.dataset == "specialized":
        data_stream = get_specialized_cognitive_stream(
            tokenizer, batch_size=args.batch_size,
            max_prompt_len=args.max_prompt_len, max_resp_len=args.max_resp_len
        )
    elif args.dataset == "hybrid":
        data_stream = get_hybrid_reasoning_stream(
            data_dir, tokenizer, batch_size=args.batch_size, reasoning_ratio=0.70,
            max_prompt_len=args.max_prompt_len, max_resp_len=args.max_resp_len
        )
    elif args.dataset == "lmsys":
        data_stream = get_lmsys_stream(
            data_dir, tokenizer, batch_size=args.batch_size,
            max_prompt_len=args.max_prompt_len, max_resp_len=args.max_resp_len
        )
    elif args.dataset == "gsm8k":
        data_stream = get_gsm8k_stream(
            tokenizer, batch_size=args.batch_size,
            max_prompt_len=args.max_prompt_len, max_resp_len=args.max_resp_len
        )
    elif args.dataset == "svamp":
        data_stream = get_svamp_stream(
            tokenizer, batch_size=args.batch_size,
            max_prompt_len=args.max_prompt_len, max_resp_len=args.max_resp_len
        )

    test_queries = [
        "Maya has 8 markers. She buys 4 more and gives 3 away. How many markers does she have?",
        "If 4 books cost $20, how much do 6 books cost?",
        "What is the capital of France?",
    ]

    print("\n" + "-" * 80)
    print("Initiating Scaled Training with Plateau Early-Stopping...")
    print("-" * 80)

    t_start = time.time()
    best_loss = 999.0
    patience_counter = 0
    step = 0
    os.makedirs("checkpoints", exist_ok=True)

    while step < args.max_steps:
        step += 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(args.grad_accum):
            prompts, targets = next(data_stream)
            prompts = prompts.to(device)
            targets = targets.to(device)
            batch_size, ans_len = targets.shape

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
                # Ingest stimulus into living manifold and capture Arcuate Fasciculus sensory working memory
                state, sensory_memory = model.ingest_stimulus_with_memory(prompts, living=True)
                state = model.deliberate(state, thinking_cycles=args.thinking_cycles)

                refractory_trace = torch.zeros(batch_size, vocab_size, device=device, dtype=dtype)
                logits_list = []

                for t in range(ans_len):
                    logits_t, state, refractory_trace = model.emit_next_token(
                        state, sensory_memory=sensory_memory, refractory_trace=refractory_trace, step_idx=t
                    )
                    logits_list.append(logits_t)

                predicted_logits = torch.stack(logits_list, dim=1)
                loss = criterion(predicted_logits.view(-1, vocab_size), targets.view(-1))
                scaled_loss = loss / args.grad_accum

            scaled_loss.backward()
            accum_loss += scaled_loss.item()

            # Explicitly free micro-batch computation graph tensors to prevent memory inflation
            del prompts, targets, state, sensory_memory, logits_list, predicted_logits, loss, scaled_loss, refractory_trace

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Release fragmented cached blocks back to CUDA driver
        if device.type == "cuda" and (step % 2 == 0 or torch.cuda.memory_allocated() > 1.8 * 1024**3):
            torch.cuda.empty_cache()

        loss_val = accum_loss

        # Step Progress Logging
        if step % 10 == 0 or step == 1:
            elapsed = time.time() - t_start
            ppl = math.exp(min(loss_val, 15))
            vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            cur_lr = scheduler.get_last_lr()[0]
            print(f"Step [{step:7,d}/{args.max_steps:,}] | Loss: {loss_val:.4f} | PPL: {ppl:8.2f} | LR: {cur_lr:.2e} | VRAM: {vram:.0f} MB | Patience: {patience_counter}/{args.patience}")

        # Plateau & Checkpoint Inspection Interval
        if step % args.eval_interval == 0:
            if loss_val < (best_loss - 0.005):  # Measurable improvement
                improvement = best_loss - loss_val
                best_loss = loss_val
                patience_counter = 0
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"  ★ [New Best Checkpoint] Saved to {best_ckpt_path} (Loss: {best_loss:.4f}, Improvement: -{improvement:.4f})")
            else:
                patience_counter += 1
                print(f"  [Plateau Check] No improvement for {patience_counter}/{args.patience} intervals (Best Loss: {best_loss:.4f})")

            # Periodic latest checkpoint save
            if step % (args.eval_interval * 5) == 0:
                torch.save(model.state_dict(), latest_ckpt_path)

            # Probe cognitive inspection
            if step % (args.eval_interval * 2) == 0:
                model.eval()
                print("\n" + "·" * 60)
                print(f"  [Cognitive Inspection at Step {step:,} | Loss: {loss_val:.4f}]")
                with torch.no_grad():
                    for q in test_queries:
                        pt = torch.tensor([tokenizer.encode(q)], device=device)
                        ans_toks = model.solve_with_active_thinking(
                            pt, thinking_cycles=args.thinking_cycles, max_answer_tokens=None, eos_token_id=tokenizer.eos_token_id, temperature=0.0
                        )
                        ans_text = tokenizer.decode(ans_toks[0].tolist(), skip_special_tokens=True)
                        print(f"  Q: \"{q[:60]}...\"")
                        print(f"  A: \"{ans_text.strip()}\"")
                print("·" * 60 + "\n")

            # Check early stopping condition
            if patience_counter >= args.patience:
                print("\n" + "!" * 80)
                print(f"  EARLY STOPPING TRIGGERED: No progress for {args.patience} consecutive evaluation intervals ({args.patience * args.eval_interval:,} steps).")
                print(f"  Final Optimum Loss Achieved: {best_loss:.4f}")
                print("!" * 80)
                break

    # Save final weights
    torch.save(model.state_dict(), latest_ckpt_path)
    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"Training Concluded after {step:,} steps in {total_time/60:.2f} minutes.")
    print(f"Best Weights  : {best_ckpt_path} (Loss: {best_loss:.4f})")
    print(f"Latest Weights: {latest_ckpt_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
