"""
CerebroSpheroid: Google Colab Cloud Training Engine
Optimized for high-VRAM Cloud Accelerators (T4 16GB, V100 16GB/32GB, A100 40GB/80GB).

Key Advantages in Google Colab:
1. Speedup: 10x - 30x faster than laptop GPU.
2. Large Batch Sizes: batch_size=8 or 16 natively in bfloat16.
3. Persistent Google Drive Sync: Saves best checkpoints directly to Google Drive.
4. Unbounded Generation: Full EOS-boundary learning and autonomous thought completion.
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

# Adjust paths to import local src modules
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.model import CerebroSpheroid
from src.data_loaders import (
    get_specialized_cognitive_stream,
    get_hybrid_reasoning_stream,
    get_gsm8k_stream,
    get_svamp_stream,
    get_lmsys_stream,
)


def main():
    parser = argparse.ArgumentParser(description="CerebroSpheroid Colab Cloud Training Engine")
    parser.add_argument("--dataset", type=str, default="specialized", choices=["specialized", "hybrid", "gsm8k", "svamp", "lmsys"])
    parser.add_argument("--max_steps", type=int, default=100_000, help="Target gradient steps")
    parser.add_argument("--eval_interval", type=int, default=50, help="Steps between evaluations")
    parser.add_argument("--batch_size", type=int, default=8, help="Micro-batch size per GPU forward pass")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--max_prompt_len", type=int, default=1024, help="Max input prompt tokens")
    parser.add_argument("--max_resp_len", type=int, default=128, help="Max output answer tokens")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (AdamW)")
    parser.add_argument("--drive_dir", type=str, default="/content/drive/MyDrive/cerebro_checkpoints",
                        help="Google Drive directory for persistent checkpoint storage")
    parser.add_argument("--resume", type=str, default=None, help="Specific checkpoint path to resume from")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    effective_batch = args.batch_size * args.grad_accum

    print("=" * 80)
    print("      CEREBRO-SPHEROID: GOOGLE COLAB HIGH-THROUGHPUT ENGINE             ")
    print("=" * 80)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Cloud Accelerator  : {gpu_name} ({vram_gb:.2f} GB VRAM)")
    else:
        print("Warning: CUDA is not available. Please select a GPU runtime in Colab!")
    print(f"Curriculum Stream  : {args.dataset.upper()} (Prompt: {args.max_prompt_len} / Resp: {args.max_resp_len} Tokens)")
    print(f"Optimization Plan  : Micro-Batch: {args.batch_size} | Grad Accum: {args.grad_accum} -> Effective Batch: {effective_batch}")
    print(f"Target Steps       : {args.max_steps:,} steps | Optimizer: ADAMW (lr={args.lr:.2e})")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    # 1. Instantiate Model Architecture
    model = CerebroSpheroid(
        vocab_size=vocab_size,
        surface_dim=640,
        num_shells=16,
        radial_dim=256,
        num_shortcut_tracts=40,
    )

    # Checkpoint paths (Local + Google Drive sync)
    os.makedirs("checkpoints", exist_ok=True)
    drive_available = os.path.exists(os.path.dirname(args.drive_dir))
    if drive_available:
        os.makedirs(args.drive_dir, exist_ok=True)
        print(f"✓ Google Drive sync enabled: {args.drive_dir}")

    local_best_ckpt = os.path.join("checkpoints", "cerebro_spheroid_colab_best.pt")
    drive_best_ckpt = os.path.join(args.drive_dir, "cerebro_spheroid_colab_best.pt") if drive_available else None

    # Resume candidates
    ckpt_candidates = []
    if args.resume:
        ckpt_candidates.append(args.resume)
    if drive_best_ckpt:
        ckpt_candidates.append(drive_best_ckpt)
    ckpt_candidates.extend([
        local_best_ckpt,
        os.path.join("checkpoints", "cerebro_spheroid_1m_best.pt"),
        os.path.join("checkpoints", "cerebro_spheroid_gsm8k_100m.pt"),
        os.path.join("checkpoints", "cerebro_spheroid_100m.pt"),
    ])

    loaded = False
    for candidate in ckpt_candidates:
        if candidate and os.path.exists(candidate):
            print(f"Resuming weights from checkpoint: {candidate}...")
            model.load_compatible_checkpoint(candidate, device="cpu")
            loaded = True
            break
    if not loaded:
        print("No prior checkpoint found; starting with fresh living manifold weights.")

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model.to(device=device, dtype=dtype)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=args.lr * 0.05)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 2. Setup Dataset Streams
    lmsys_dir = os.path.join(repo_root, "data", "lmsys_chat_1m", "data")
    if args.dataset == "specialized":
        cog_parquet = os.path.join(repo_root, "data", "cerebro_cognitive", "cerebro_cognitive_train.parquet")
        data_stream = get_specialized_cognitive_stream(
            tokenizer, batch_size=args.batch_size, parquet_path=cog_parquet,
            max_prompt_len=args.max_prompt_len, max_resp_len=args.max_resp_len
        )
    elif args.dataset == "hybrid":
        if os.path.exists(lmsys_dir):
            data_stream = get_hybrid_reasoning_stream(
                lmsys_dir, tokenizer, batch_size=args.batch_size, reasoning_ratio=0.70,
                max_prompt_len=args.max_prompt_len, max_resp_len=args.max_resp_len
            )
        else:
            # Fallback to GSM8K + SVAMP streaming without local LMSYS
            print("Notice: Local LMSYS parquets not found; using pure GSM8K + SVAMP cloud streams.")
            gsm = get_gsm8k_stream(tokenizer, batch_size=args.batch_size, max_prompt_len=args.max_prompt_len, max_resp_len=args.max_resp_len)
            svm = get_svamp_stream(tokenizer, batch_size=args.batch_size, max_prompt_len=args.max_prompt_len, max_resp_len=args.max_resp_len)
            def _pure_reasoning_stream():
                while True:
                    yield next(gsm) if random.random() < 0.85 else next(svm)
            data_stream = _pure_reasoning_stream()
    elif args.dataset == "gsm8k":
        data_stream = get_gsm8k_stream(tokenizer, batch_size=args.batch_size, max_prompt_len=args.max_prompt_len, max_resp_len=args.max_resp_len)
    elif args.dataset == "svamp":
        data_stream = get_svamp_stream(tokenizer, batch_size=args.batch_size, max_prompt_len=args.max_prompt_len, max_resp_len=args.max_resp_len)
    elif args.dataset == "lmsys":
        data_stream = get_lmsys_stream(lmsys_dir, tokenizer, batch_size=args.batch_size, max_prompt_len=args.max_prompt_len, max_resp_len=args.max_resp_len)

    test_queries = [
        "Maya has 8 markers. She buys 4 more and gives 3 away. How many markers does she have?",
        "If 4 books cost $20, how much do 6 books cost?",
        "What is the capital of France?",
    ]

    print("\n" + "-" * 80)
    print("Initiating High-Speed Cloud Training...")
    print("-" * 80)

    t_start = time.time()
    best_loss = 999.0
    step = 0

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
                state, sensory_memory = model.ingest_stimulus_with_memory(prompts, living=True)
                state = model.deliberate(state, thinking_cycles=3)

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
            del prompts, targets, state, sensory_memory, logits_list, predicted_logits, loss, scaled_loss, refractory_trace

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if device.type == "cuda" and step % 5 == 0:
            torch.cuda.empty_cache()

        loss_val = accum_loss

        # Step Progress Logging
        if step % 25 == 0 or step == 1:
            elapsed = time.time() - t_start
            steps_per_sec = step / max(elapsed, 0.01)
            ppl = math.exp(min(loss_val, 15))
            vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            cur_lr = scheduler.get_last_lr()[0]
            print(f"Step [{step:6,d}/{args.max_steps:,}] | Loss: {loss_val:.4f} | PPL: {ppl:7.2f} | Speed: {steps_per_sec:.2f} step/s | VRAM: {vram:.0f} MB | LR: {cur_lr:.2e}")

        # Evaluation & Checkpointing Interval
        if step % args.eval_interval == 0:
            if loss_val < (best_loss - 0.005):
                improvement = best_loss - loss_val
                best_loss = loss_val
                torch.save(model.state_dict(), local_best_ckpt)
                if drive_available:
                    torch.save(model.state_dict(), drive_best_ckpt)
                print(f"  ★ [New Best Checkpoint] Saved to {local_best_ckpt} (Loss: {best_loss:.4f}, Improvement: -{improvement:.4f})")

            # Cognitive Probe Inspection (Unbounded Generation until EOS)
            if step % (args.eval_interval * 2) == 0:
                model.eval()
                print("\n" + "·" * 60)
                print(f"  [Cognitive Inspection at Step {step:,} | Loss: {loss_val:.4f}]")
                with torch.no_grad():
                    for q in test_queries:
                        pt = torch.tensor([tokenizer.encode(q)], device=device)
                        ans_toks = model.solve_with_active_thinking(
                            pt, thinking_cycles=3, max_answer_tokens=None, eos_token_id=tokenizer.eos_token_id, temperature=0.0
                        )
                        ans_text = tokenizer.decode(ans_toks[0].tolist(), skip_special_tokens=True)
                        print(f"  Q: \"{q[:60]}...\"")
                        print(f"  A: \"{ans_text.strip()}\"")
                print("·" * 60 + "\n")

    print("\n" + "=" * 80)
    print(f"Colab Cloud Training Complete! Final Optimum Loss: {best_loss:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
