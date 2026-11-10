#!/usr/bin/env python3
"""
CerebroSpheroid V4: Local Supervised Fine-Tuning (SFT) Engine.
Optimized for local NVIDIA RTX 3050 GPU (4 GB VRAM) with mixed precision.
Ingests local OpenHermes & LMSYS conversational datasets to align reasoning & dialogue.
"""

import os
import sys
import time
import math
import random
import argparse
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow.parquet as pq

# Add project root to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.spheroid_v4 import build_spheroid_v4
from src.corpus import build_tokenizer


def load_conversation_pairs(parquet_path: str, max_records: int = 50000):
    """Loads and yields clean (prompt, response) instruction pairs from local parquet (UltraChat, GSM8k, or CerebroCognitive)."""
    print(f"📖 Reading conversational records from {parquet_path}...")
    schema = pq.read_schema(parquet_path)
    pairs = []

    if "question" in schema.names and "answer" in schema.names:
        table = pq.read_table(parquet_path, columns=["question", "answer"])
        q_col = table["question"]
        a_col = table["answer"]
        for i in range(min(len(q_col), max_records)):
            q = q_col[i].as_py()
            a = a_col[i].as_py()
            if q and a:
                pairs.append((q.strip(), a.strip()))
        print(f"✅ Loaded {len(pairs):,} question-answer pairs.")
        return pairs

    col_name = "messages" if "messages" in schema.names else "conversations"
    table = pq.read_table(parquet_path, columns=[col_name])
    conv_col = table[col_name]
    
    for i in range(len(conv_col)):
        if len(pairs) >= max_records:
            break
        raw = conv_col[i].as_py()
        if not raw:
            continue
        try:
            if isinstance(raw, str):
                conv = ast.literal_eval(raw)
            else:
                conv = raw
            
            # Extract all (user, assistant) dialogue turns
            history = ""
            for turn_idx in range(len(conv) - 1):
                m_user = conv[turn_idx]
                m_asst = conv[turn_idx + 1]
                role_u = m_user.get("from") or m_user.get("role") or ""
                role_a = m_asst.get("from") or m_asst.get("role") or ""
                val_u = (m_user.get("value") or m_user.get("content") or "").strip()
                val_a = (m_asst.get("value") or m_asst.get("content") or "").strip()
                
                if role_u in ("user", "human") and role_a in ("assistant", "gpt"):
                    if len(val_u) >= 5 and len(val_a) >= 5:
                        prompt_text = f"{history}User: {val_u}" if history else val_u
                        pairs.append((prompt_text, val_a))
                        history += f"User: {val_u}\nAssistant: {val_a}\n"
                        if len(history) > 600:
                            history = history[-400:]
        except Exception:
            continue
            
    print(f"✅ Loaded {len(pairs):,} conversational dialogue pairs.")
    return pairs


def build_batch(tokenizer, pair_chunk, max_prompt_len=256, max_answer_len=128, device="cuda"):
    """
    Constructs model-ready tensors using 1-token biological delimiters:
    [20436] (sensory stimulus <stim>) + prompt + [1044] (motor action <act>) + response
    """
    STIM_ID = 20436  # <stim>
    ACT_ID = 1044    # <act>
    PAD_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    EOS_ID = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2

    prompts_list = []
    targets_list = []
    feedback_list = []

    for p, a in pair_chunk:
        p_ids = [STIM_ID] + tokenizer.encode(p, add_special_tokens=False)[: max_prompt_len - 1]
        a_ids = [ACT_ID] + tokenizer.encode(a, add_special_tokens=False)[: max_answer_len - 2] + [EOS_ID]

        prompts_list.append(p_ids)
        targets_list.append(a_ids)

    # Pad prompts to max in batch
    p_max = max(len(x) for x in prompts_list)
    t_max = max(len(x) for x in targets_list)

    padded_prompts = []
    padded_targets = []
    padded_feedback = []

    for p_ids, a_ids in zip(prompts_list, targets_list):
        p_pad = [PAD_ID] * (p_max - len(p_ids)) + p_ids
        t_pad = a_ids + [-100] * (t_max - len(a_ids))
        
        # Feedback is shifted target with teacher forcing
        f_pad = [-100] + a_ids[:-1] + [-100] * (t_max - len(a_ids))
        f_pad = f_pad[:t_max]

        padded_prompts.append(p_pad)
        padded_targets.append(t_pad)
        padded_feedback.append(f_pad)

    p_tensor = torch.tensor(padded_prompts, dtype=torch.long, device=device)
    t_tensor = torch.tensor(padded_targets, dtype=torch.long, device=device)
    f_tensor = torch.tensor(padded_feedback, dtype=torch.long, device=device)

    return p_tensor, t_tensor, f_tensor


def run_sft_inference_eval(model, tokenizer, device, thinking_cycles=3):
    """Runs zero-shot qualitative evaluation on diverse benchmark prompts from the 100-question suite."""
    bench_file = os.path.join(ROOT_DIR, "data", "benchmark_100_questions.json")
    if os.path.exists(bench_file):
        import json
        with open(bench_file, "r", encoding="utf-8") as f:
            bench_items = json.load(f)
        # Select representative prompts from diverse categories:
        # 1 conversational, 1 geography, 1 math, 1 science, 1 logic, 1 grounded extraction
        sample_ids = [61, 1, 31, 16, 46, 76]
        id_map = {item["id"]: item for item in bench_items}
        benchmark_prompts = [id_map[i]["question"] for i in sample_ids if i in id_map]
    else:
        benchmark_prompts = [
            "Hello! Who are you?",
            "What is the capital of France?",
            "What is 2 + 3?",
            "Why is the sky blue?",
        ]
    model.eval()
    print("\n" + "=" * 65)
    print("🤖 Live SFT Model Inference Probe (Diverse Benchmark Suite):")
    print("=" * 65)
    with torch.no_grad():
        for q in benchmark_prompts:
            ans = model.generate_text(
                q,
                tokenizer=tokenizer,
                thinking_cycles=thinking_cycles,
                max_new_tokens=48,
                temperature=0.6,
                repetition_penalty=1.30,
                device=device,
                strip_bio_prefix=True,
            )
            print(f"User:     {q}")
            print(f"Spheroid: {ans.strip()}")
            print("-" * 65)
    print()
    model.train()


def main():
    parser = argparse.ArgumentParser(description="CerebroSpheroid V4 Local GPU Fine-Tuner")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/cerebro_spheroid_v4_step142200.pt",
                        help="Path to initial pre-trained checkpoint")
    parser.add_argument("--dataset_path", type=str, default="data/ultrachat/ultrachat_200k_train.parquet",
                        help="Path to local UltraChat-200k conversational parquet dataset")
    parser.add_argument("--save_prefix", type=str, default="cerebro_spheroid_v4_ultrachat_sft",
                        help="Prefix for saved fine-tuned checkpoints")
    parser.add_argument("--batch_size", type=int, default=4, help="Micro-batch size (fit in 4 GB VRAM)")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation micro-steps (effective batch: 32)")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Peak SFT learning rate")
    parser.add_argument("--min_lr", type=float, default=2.0e-5, help="Cosine decayed minimum learning rate")
    parser.add_argument("--steps", type=int, default=300, help="Total SFT steps to run locally")
    parser.add_argument("--eval_every", type=int, default=50, help="Steps between live generation probes")
    parser.add_argument("--save_every", type=int, default=100, help="Steps between checkpoint saves")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 75)
    print("🧠 CEREBRO-SPHEROID V4: LOCAL GPU SUPERVISED FINE-TUNING")
    print(f"Device           : {device} ({torch.cuda.get_device_name(device) if 'cuda' in args.device else 'CPU'})")
    print(f"Initial Weights  : {args.ckpt_path}")
    print(f"Dataset Source   : {args.dataset_path}")
    print(f"Batch Config     : {args.batch_size} micro-batch x {args.grad_accum} accum = {args.batch_size * args.grad_accum} effective")
    print(f"Peak Learning Rate: {args.lr:.2e} -> Min: {args.min_lr:.2e}")
    print(f"Target Horizon   : {args.steps} steps")
    print("=" * 75)

    # 1. Tokenizer
    tokenizer = build_tokenizer("llama")

    # 2. Model Initialization
    model = build_spheroid_v4("v4_small", vocab_size=len(tokenizer))
    if os.path.exists(args.ckpt_path):
        model.load_compatible_checkpoint(args.ckpt_path, device=device)
    else:
        print(f"⚠️ Warning: Checkpoint {args.ckpt_path} not found. Initializing from scratch.")
    model.to(device)

    # 3. Optimizer with Warmup & Cosine Decay
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )

    # 4. Dataset
    pairs = load_conversation_pairs(args.dataset_path, max_records=25000)
    rng = random.Random(1337)
    rng.shuffle(pairs)

    # Initial Pre-SFT Baseline Generation Probe
    print("\n🔍 Running Baseline Generation Probe Before SFT:")
    run_sft_inference_eval(model, tokenizer, device)

    # 5. Training Loop
    print("\n🚀 Commencing Local SFT Training Run...")
    model.train()
    pair_idx = 0
    t_start = time.time()
    total_tokens_trained = 0
    best_loss = float("inf")
    scaler = torch.amp.GradScaler('cuda', enabled=('cuda' in args.device))

    for step in range(1, args.steps + 1):
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        # Learning Rate Schedule (Cosine with 50-step warmup)
        if step <= 50:
            lr = args.min_lr + (args.lr - args.min_lr) * (float(step) / 50.0)
        else:
            prog = float(step - 50) / max(1.0, float(args.steps - 50))
            lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * min(1.0, prog)))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        accum_loss = 0.0
        step_tokens = 0

        for _ in range(args.grad_accum):
            chunk = []
            while len(chunk) < args.batch_size:
                chunk.append(pairs[pair_idx % len(pairs)])
                pair_idx += 1

            p_batch, t_batch, f_batch = build_batch(tokenizer, chunk, max_prompt_len=256, max_answer_len=128, device=device)
            target_mask = (t_batch >= 0)
            step_tokens += int(target_mask.sum().item())

            with torch.amp.autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, enabled=('cuda' in args.device)):
                logits = model(
                    p_batch,
                    response_steps=t_batch.shape[1],
                    targets=t_batch,
                    feedback_tokens=f_batch,
                    deliberation_mode="teacher_forcing",
                    sampling_p=1.0,
                    repetition_penalty=1.0,
                )
                loss = F.cross_entropy(logits.view(-1, len(tokenizer)), t_batch.view(-1), ignore_index=-100)
                loss_scaled = loss / float(args.grad_accum)

            scaler.scale(loss_scaled).backward()
            accum_loss += float(loss.item()) / float(args.grad_accum)

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        dt = time.time() - t0
        total_tokens_trained += step_tokens
        tok_s = step_tokens / max(dt, 1e-4)

        if accum_loss < best_loss:
            best_loss = accum_loss
            # Save best checkpoint
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "params": model.state_dict(),
                "step": step,
                "loss": best_loss,
                "source": "ultrachat_sft",
            }, f"checkpoints/{args.save_prefix}_best.pt")

        if step % 10 == 0 or step == 1:
            ppl = math.exp(min(accum_loss, 15.0))
            vram_mb = torch.cuda.memory_allocated(device) / 1024**2 if "cuda" in args.device else 0.0
            print(f"Step {step:4d}/{args.steps} | Loss: {accum_loss:.4f} | PPL: {ppl:6.2f} | LR: {lr:.2e} | Speed: {tok_s:5.0f} tok/s | VRAM: {vram_mb:.0f} MB")

        # Periodic Generation Evaluation Probe
        if step % args.eval_every == 0:
            run_sft_inference_eval(model, tokenizer, device)

        # Periodic Latest Checkpoint Save
        if step % args.save_every == 0 or step == args.steps:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "params": model.state_dict(),
                "step": step,
                "loss": accum_loss,
                "source": "ultrachat_sft",
            }, f"checkpoints/{args.save_prefix}_latest.pt")
            print(f"💾 Checkpoint saved at Step {step} -> checkpoints/{args.save_prefix}_latest.pt")

    total_time = time.time() - t_start
    print("\n" + "=" * 75)
    print(f"🎉 UltraChat Supervised Fine-Tuning Complete in {total_time:.1f}s!")
    print(f"   • Total Tokens Trained: {total_tokens_trained:,}")
    print(f"   • Final Loss: {accum_loss:.4f} (Best: {best_loss:.4f})")
    print(f"   • Saved Checkpoints: checkpoints/{args.save_prefix}_best.pt")
    print("=" * 75)

    # Final Evaluation Probe
    print("\n🏆 Final Post-SFT Model Evaluation:")
    run_sft_inference_eval(model, tokenizer, device)


if __name__ == "__main__":
    main()
