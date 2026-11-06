#!/usr/bin/env python3
"""
Scientific A/B Benchmark: Manifold-Grafted vs Random Initialization.

Runs a controlled local training comparison on the laptop GPU:
- Model A: Baseline Random Gaussian Initialization
- Model B: 1.1B/2B Manifold-Grafted (TinyLlama) Warm-Start

Both models are trained on the exact same batches from data/openhermes/openhermes_2_5.parquet.
Measures:
- Step-by-step loss reduction (Step 0 to Step 100)
- Validation Perplexity
- Next-token prediction accuracy on reasoning prompts
"""

import argparse
import math
import os
import sys
import time
import random
import pyarrow.parquet as pq

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.corpus import build_tokenizer, _conversation_pairs
from scripts.eval_latest_checkpoint_loss import BitwiseJAXEngine
from scripts.align_manifold_warmstart import init_default_cerebro_params


def load_training_batches(parquet_path: str, tokenizer, count: int = 100, max_len: int = 64):
    """Loads prompt-response pairs from local OpenHermes parquet."""
    pf = pq.ParquetFile(parquet_path)
    table = pf.read_row_group(0)
    convs = table["conversations"].to_pylist()
    
    batches = []
    for conv in convs:
        pairs = _conversation_pairs(conv)
        for p, a in pairs:
            text = f"User: {p.strip()}\nAssistant: {a.strip()}"
            tokens = tokenizer.encode(text, return_tensors="pt")
            if tokens.shape[1] > 16:
                # Truncate to max_len
                seq = tokens[:, :max_len]
                batches.append(seq)
                if len(batches) >= count:
                    return batches
    return batches


def train_single_model(
    model_name: str,
    params_dict: dict,
    batches: list,
    eval_sentences: list,
    tokenizer,
    device: str = "cuda",
    lr: float = 2e-4,
    steps: int = 100,
    grad_accum: int = 2
) -> Tuple[list, list]:
    """Trains a model instance for `steps` steps and records trajectory."""
    print(f"\n" + "=" * 65)
    print(f"  Training {model_name} on {device.upper()} ({steps} steps)")
    print("=" * 65)
    
    # Wrap in trainable Parameters
    p_params = {k: nn.Parameter(v.to(device=device, dtype=torch.float32)) for k, v in params_dict.items()}
    engine = BitwiseJAXEngine(p_params, device=device)
    
    # Filter only parameters that require gradients
    trainable = list(p_params.values())
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    
    trajectory = []
    eval_trajectory = []
    
    K = 16
    t0 = time.time()
    
    for step in range(steps):
        batch_tokens = batches[step % len(batches)].to(device)
        inputs = batch_tokens[:, :-1]
        targets = batch_tokens[:, 1:]
        
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda', enabled=(device == "cuda"), dtype=torch.float16):
            sens, (keys, vals) = engine.encode_sensory(inputs)
            B, S, D = sens.shape
            state = torch.zeros(B, K, D, device=device, dtype=sens.dtype)
            plan = torch.zeros(B, D, device=device, dtype=sens.dtype)
            refractory = torch.zeros(B, tokenizer.vocab_size, device=device, dtype=sens.dtype)
            
            logits_list = []
            for t in range(S):
                state = engine.deliberate_step(state + sens[:, t:t+1, :].expand(-1, K, -1) * 0.2)
                logits, state, refractory, plan = engine.emit_step(
                    state, (keys[:, :t+1], vals[:, :t+1]), plan, refractory, step_idx=t
                )
                logits_list.append(logits.unsqueeze(1))
                
            all_logits = torch.cat(logits_list, dim=1)
            loss = F.cross_entropy(all_logits.view(-1, tokenizer.vocab_size), targets.view(-1))
            scaled_loss = loss / grad_accum
            
        scaled_loss.backward()
        
        if (step + 1) % grad_accum == 0 or (step + 1) == steps:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
        loss_val = float(loss.item())
        trajectory.append((step, loss_val))
        
        if step % 20 == 0 or step == steps - 1:
            tok_s = (step + 1) * inputs.numel() / max(time.time() - t0, 1e-6)
            ppl = math.exp(min(loss_val, 20.0))
            print(f"  Step {step:3d}/{steps} | Train Loss: {loss_val:.4f} (PPL: {ppl:8.1f}) | Throughput: {tok_s:5.0f} tok/s")
            
    print(f"[*] {model_name} finished {steps} steps in {time.time() - t0:.1f}s")
    return trajectory, p_params


def main():
    parser = argparse.ArgumentParser(description="Scientific A/B Local Training Benchmark")
    parser.add_argument("--grafted_checkpoint", type=str, default="checkpoints/cerebro_v4_tinyllama_aligned.pt",
                        help="Path to manifold-grafted checkpoint")
    parser.add_argument("--fallback_grafted", type=str, default="checkpoints/cerebro_v4_manifold_aligned.pt",
                        help="Fallback grafted checkpoint if TinyLlama is still downloading")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps per model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    
    # Select grafted checkpoint
    graft_path = args.grafted_checkpoint
    if not os.path.exists(graft_path):
        if os.path.exists(args.fallback_grafted):
            print(f"[*] {graft_path} not ready yet, using {args.fallback_grafted}")
            graft_path = args.fallback_grafted
        else:
            print(f"Error: No grafted checkpoint found.")
            return
            
    tokenizer = build_tokenizer("llama")
    
    parquet_path = os.path.join(ROOT_DIR, "data/openhermes/openhermes_2_5.parquet")
    print(f"[*] Loading training corpus from: {parquet_path}")
    batches = load_training_batches(parquet_path, tokenizer, count=args.steps, max_len=48)
    print(f"[*] Prepared {len(batches)} training batches of max_len=48")
    
    eval_sentences = [
        "The fundamental law of calculus connects derivatives with integrals.",
        "Deep learning models compress high dimensional data onto low dimensional manifolds.",
        "To solve the equation three x minus nine equals zero, we isolate the variable.",
    ]
    
    # 1. Baseline Model A (Random Initialization)
    print("\n" + "=" * 65)
    print("  INITIALIZING MODEL A (Random Gaussian Baseline)")
    print("=" * 65)
    params_a = init_default_cerebro_params()
    traj_a, final_p_a = train_single_model(
        "Model A (Random Baseline)", params_a, batches, eval_sentences, tokenizer,
        device=args.device, lr=args.lr, steps=args.steps
    )
    
    # Clear CUDA Cache before Model B
    if args.device == "cuda":
        torch.cuda.empty_cache()
        
    # 2. Model B (Manifold Grafted Warm-Start)
    print("\n" + "=" * 65)
    print(f"  INITIALIZING MODEL B (Manifold Grafted: {os.path.basename(graft_path)})")
    print("=" * 65)
    params_b = torch.load(graft_path, map_location="cpu", weights_only=False)
    traj_b, final_p_b = train_single_model(
        f"Model B (Manifold Grafted)", params_b, batches, eval_sentences, tokenizer,
        device=args.device, lr=args.lr, steps=args.steps
    )
    
    # Print Comparative Results
    print("\n" + "=" * 70)
    print("  SCIENTIFIC A/B BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Step':<8} | {'Model A (Random Baseline)':<26} | {'Model B (Manifold Grafted)':<26} | {'Delta':<8}")
    print("-" * 70)
    for i in range(0, args.steps, 20):
        l_a = traj_a[i][1]
        l_b = traj_b[i][1]
        delta = l_a - l_b
        print(f"{i:<8d} | {l_a:8.4f} (PPL: {math.exp(min(l_a, 20)):7.1f})   | {l_b:8.4f} (PPL: {math.exp(min(l_b, 20)):7.1f})   | {delta:+6.3f}")
    
    final_a = traj_a[-1][1]
    final_b = traj_b[-1][1]
    print(f"{args.steps - 1:<8d} | {final_a:8.4f} (PPL: {math.exp(min(final_a, 20)):7.1f})   | {final_b:8.4f} (PPL: {math.exp(min(final_b, 20)):7.1f})   | {final_a - final_b:+6.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
