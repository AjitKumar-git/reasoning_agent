"""
CerebroSpheroid Cognitive Training on Cloud TPU (v3-8 / v4) via JAX.

Why JAX on TPU
--------------
1. Zero recompilation lag: JAX compiles the fused deliberate-then-emit scan loop once,
   avoiding PyTorch/XLA graph explosion.
2. Direct PJRT integration: Native TPU execution with bfloat16 / float32 matrix engines.
3. Clean zero-feedback contract: Supports both V2 and V3 models with exact arithmetic parity.

Usage on Kaggle TPU VM:
----------------------
    # Accelerator: TPU v3-8
    python3 kaggle_tpu_jax.py --scale laptop --max_steps 4000 --batch_size 16 --lr 0.001
"""

import os
import sys
import time
import math
import argparse
from typing import Optional, Dict, Any, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, ".hf_cache"))
sys.path.insert(0, ROOT_DIR)

import numpy as np
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

from src.jax.model import (
    ModelConfig,
    init_params,
    training_rollout,
    compute_loss,
)
from src.jax.convert import save_jax_checkpoint, load_torch_checkpoint_to_jax
from training.train_jax import init_adamw, update_adamw, generate_synthetic_batch, PROMPT_LEN, RESP_LEN


def main():
    parser = argparse.ArgumentParser(description="CerebroSpheroid JAX TPU Training")
    parser.add_argument("--scale", type=str, default="laptop", choices=["laptop", "small", "2b", "260m", "test"])
    parser.add_argument("--max_steps", type=int, default=2000, help="Max gradient steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per step")
    parser.add_argument("--thinking_cycles", type=int, default=3, help="Subconscious deliberation cycles")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--arch", type=str, default="v2", choices=["v2", "v3"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--save_path", type=str, default="checkpoints/cerebro_spheroid_tpu_jax.npz")
    parser.add_argument("--init_from_pt", type=str, default=None)

    args = parser.parse_args()

    print("=" * 80)
    print("      CEREBRO-SPHEROID: CLOUD TPU TRAINING ENGINE (JAX / PJRT)          ")
    print("=" * 80)
    devices = jax.devices()
    print(f"JAX Platform      : {jax.default_backend().upper()}")
    print(f"TPU Devices ({len(devices)}) : {devices}")
    print(f"Architecture      : {args.arch.upper()}")
    print(f"Model Scale       : {args.scale}")
    print(f"Batch Size        : {args.batch_size}")
    print(f"Deliberation      : {args.thinking_cycles} cycles")
    print(f"Learning Rate     : {args.lr}")
    print(f"Max Steps         : {args.max_steps}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    config = ModelConfig.from_scale(scale=args.scale, arch=args.arch)
    config.vocab_size = tokenizer.vocab_size

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)

    if args.init_from_pt and os.path.exists(args.init_from_pt):
        print(f"Loading weights from: {args.init_from_pt}")
        params, config = load_torch_checkpoint_to_jax(args.init_from_pt, arch=args.arch)
    else:
        print("Initializing model parameters...")
        params = init_params(config, key=init_key)

    total_params = sum(p.size for p in params.values() if jnp.issubdtype(p.dtype, jnp.inexact))
    print(f"Total trainable parameters: {total_params:,}")

    opt_state = init_adamw(params)

    @jax.jit
    def train_step(p, opt_s, q_batch, r_batch, current_lr):
        def _loss_wrapper(param_dict):
            loss, metrics = compute_loss(
                param_dict, config, q_batch, r_batch,
                thinking_cycles=args.thinking_cycles, pad_token_id=pad_id
            )
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(_loss_wrapper, has_aux=True, allow_int=True)(p)
        new_p, new_opt_s = update_adamw(p, grads, opt_s, lr=current_lr)
        return new_p, new_opt_s, loss, metrics

    @jax.jit
    def eval_step(p, q_batch, r_batch):
        loss, metrics = compute_loss(
            p, config, q_batch, r_batch,
            thinking_cycles=args.thinking_cycles, pad_token_id=pad_id
        )
        return loss, metrics

    val_q, val_r = generate_synthetic_batch(tokenizer, batch_size=args.batch_size * 2, seed=args.seed + 999)
    val_q_jax = jnp.array(val_q)
    val_r_jax = jnp.array(val_r)

    print("\nCompiling fused XLA graph...")
    t_start = time.time()
    best_eval_loss = float("inf")

    for step in range(1, args.max_steps + 1):
        q_np, r_np = generate_synthetic_batch(tokenizer, batch_size=args.batch_size, seed=args.seed + step)
        q_jax = jnp.array(q_np)
        r_jax = jnp.array(r_np)

        t0 = time.time()
        progress = float(step) / max(1, args.max_steps)
        scheduled_lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

        params, opt_state, loss_val, metrics = train_step(params, opt_state, q_jax, r_jax, scheduled_lr)
        loss_float = float(loss_val)
        dt = time.time() - t0

        if step == 1:
            print(f"✓ Fused XLA graph compiled in {dt:.2f}s!")

        if step % 10 == 0 or step == 1:
            tokens_per_sec = (args.batch_size * (PROMPT_LEN + RESP_LEN)) / max(dt, 1e-5)
            print(f"Step {step:5d}/{args.max_steps} | Loss: {loss_float:.4f} | PPL: {float(metrics['perplexity']):.2f} | {dt*1000:6.1f} ms/step | {tokens_per_sec:7.1f} tok/s")

        if step % args.eval_interval == 0 or step == args.max_steps:
            eval_loss, eval_metrics = eval_step(params, val_q_jax, val_r_jax)
            eval_loss_float = float(eval_loss)
            is_best = eval_loss_float < best_eval_loss
            star = " ★ (New Best)" if is_best else ""
            print(f"\n---> [Eval @ Step {step}] Loss: {eval_loss_float:.4f} | PPL: {float(eval_metrics['perplexity']):.2f}{star}\n")

            if is_best:
                best_eval_loss = eval_loss_float
                save_jax_checkpoint(
                    args.save_path, params, config,
                    extra_metadata={"best_step": step, "best_loss": eval_loss_float}
                )

    total_time = time.time() - t_start
    print("=" * 80)
    print(f"TPU Training completed in {total_time:.1f}s ({args.max_steps / total_time:.2f} steps/s)")
    print(f"Best Evaluation Loss: {best_eval_loss:.4f}")
    print(f"Saved Checkpoint: {args.save_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
