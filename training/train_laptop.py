"""
CerebroSpheroid: Unified Laptop Training Engine
Supports multi-domain datasets:
  --dataset lmsys  : 1,000,000 real-world human-LLM conversations from local parquets
  --dataset gsm8k  : Multi-step mathematical reasoning from OpenAI GSM8K
  --dataset svamp  : Adversarial arithmetic challenge edge cases from SVAMP
"""

import os
os.environ.setdefault(
    "HF_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".hf_cache"))
)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import math
import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.model import CerebroSpheroid
from src.spheroid_v3 import CerebroSpheroidV3, training_rollout
from src.data_loaders import get_lmsys_stream, get_gsm8k_stream, get_svamp_stream
from src.inference import ReasoningAgent


def main():
    parser = argparse.ArgumentParser(description="Unified CerebroSpheroid Laptop Trainer")
    parser.add_argument("--dataset", type=str, default="lmsys", choices=["lmsys", "gsm8k", "svamp"], help="Dataset")
    parser.add_argument("--steps", type=int, default=300, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Micro-batch size per step")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--thinking_cycles", type=int, default=3, help="Subconscious thinking cycles")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"], help="Optimizer (adamw or sgd)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 3e-4 for AdamW, 0.015 for SGD)")
    parser.add_argument("--resume", type=str, default=None, help="Explicit checkpoint path to resume")
    parser.add_argument("--save_name", type=str, default=None, help="Custom checkpoint name")
    parser.add_argument("--arch", type=str, default="v2", choices=["v2", "v3"], help="Manifold generation to train")
    args = parser.parse_args()

    if args.lr is None:
        args.lr = 3e-4 if args.optimizer == "adamw" else 0.015

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    print("=" * 80)
    print("      CEREBRO-SPHEROID: UNIFIED COGNITIVE TRAINING ENGINE               ")
    print("=" * 80)
    print(f"Dataset Selected   : {args.dataset.upper()}")
    print(f"Target Accelerator : {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Model          : {gpu_name} ({vram_gb:.2f} GB VRAM)")
    print(f"Planned Steps      : {args.steps} steps | Batch Size: {args.batch_size}")
    print(f"Optimizer          : {args.optimizer.upper()} (lr={args.lr:.2e})")
    print(f"Thinking Latency   : {args.thinking_cycles} cycles (Subconscious deliberation)")
    print(f"Manifold           : {args.arch.upper()}")
    print("=" * 80)

    # 1. Instantiate Model Architecture
    model_cls = CerebroSpheroidV3 if args.arch == "v3" else CerebroSpheroid
    model = model_cls(
        vocab_size=vocab_size,
        surface_dim=640,
        num_shells=16,
        radial_dim=256,
        num_shortcut_tracts=40,
    )

    # Load existing checkpoint if available
    save_filename = args.save_name or f"cerebro_spheroid_{args.dataset}_100m.pt"
    ckpt_path = os.path.join("checkpoints", save_filename)

    ckpt_candidates = []
    if args.resume:
        ckpt_candidates.append(args.resume)
    ckpt_candidates.extend([
        ckpt_path,
        os.path.join("checkpoints", "cerebro_spheroid_gsm8k_100m.pt"),
        os.path.join("checkpoints", "cerebro_spheroid_100m.pt"),
    ])

    loaded = False
    for candidate in ckpt_candidates:
        if os.path.exists(candidate):
            print(f"Resuming from checkpoint: {candidate}...")
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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 2. Select Data Stream
    if args.dataset == "lmsys":
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "lmsys_chat_1m", "data"))
        data_stream = get_lmsys_stream(data_dir, tokenizer, batch_size=args.batch_size)
    elif args.dataset == "gsm8k":
        data_stream = get_gsm8k_stream(tokenizer, batch_size=args.batch_size)
    elif args.dataset == "svamp":
        data_stream = get_svamp_stream(tokenizer, batch_size=args.batch_size)

    test_queries = [
        "Maya has 8 markers. She buys 4 more and gives 3 away. How many markers does she have?",
        "What is the capital of France?",
        "How can identity protection services help protect me against identity theft?",
    ]

    print("\n" + "-" * 80)
    print(f"Starting Training on {args.dataset.upper()}...")
    print("-" * 80)

    t_start = time.time()
    best_loss = 999.0

    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(args.grad_accum):
            prompts, targets = next(data_stream)
            prompts = prompts.to(device)
            targets = targets.to(device)
            ans_len = targets.shape[1]

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
                predicted_logits = training_rollout(
                    model, prompts, ans_len, thinking_cycles=args.thinking_cycles
                )
                loss = criterion(predicted_logits.reshape(-1, vocab_size), targets.reshape(-1))
                scaled_loss = loss / args.grad_accum

            scaled_loss.backward()
            accum_loss += scaled_loss.item() * args.grad_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_val = accum_loss
        if loss_val < best_loss:
            best_loss = loss_val
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)

        if step % 200 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)

        if step % 25 == 0 or step == 1 or step == args.steps:
            elapsed = time.time() - t_start
            ppl = math.exp(min(loss_val, 15))
            vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            cur_lr = scheduler.get_last_lr()[0]
            print(f"Step [{step:4d}/{args.steps}] | Loss: {loss_val:.4f} | PPL: {ppl:8.2f} | LR: {cur_lr:.2e} | VRAM: {vram:.0f} MB | Time: {elapsed:.1f}s")

        if step % 100 == 0 or step == args.steps:
            model.eval()
            print("\n" + "·" * 60)
            print(f"  [Cognitive Inspection at Step {step} | Loss: {loss_val:.4f}]")
            with torch.no_grad():
                for q in test_queries:
                    pt = torch.tensor([tokenizer.encode(q)], device=device)
                    ans_toks = model.solve_with_active_thinking(
                        pt, thinking_cycles=args.thinking_cycles, max_answer_tokens=16, temperature=0.0
                    )
                    ans_text = tokenizer.decode(ans_toks[0].tolist(), skip_special_tokens=True)
                    print(f"  Q: \"{q[:60]}...\"")
                    print(f"  A: \"{ans_text.strip()}\"")
            print("·" * 60 + "\n")

    # Final save
    torch.save(model.state_dict(), ckpt_path)
    print("\n" + "=" * 80)
    print(f"Training Complete! Saved checkpoint to: {ckpt_path} (Best Loss: {best_loss:.4f})")
    print("=" * 80)


if __name__ == "__main__":
    main()
