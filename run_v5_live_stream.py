#!/usr/bin/env python3
"""
Real-Time Continuous Streamer for CerebroSpheroid V5.
- Monitors the active training checkpoint for new step updates.
- Loops through continuous reasoning prompts or accepts interactive inputs.
- Emits tokens with a typewriter pacing to watch in real time.
"""

import os
import sys
import time
import argparse
import torch
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
V5_DIR = os.path.abspath(os.path.join(ROOT_DIR, "..", "cerebro_spheroid_v5"))
sys.path.insert(0, V5_DIR)

from src.jax.v5_model import ModelConfigV5, PRESETS_V5, training_rollout, generate_small_world_indices

DEFAULT_PROMPTS = [
    "Question: What is the capital of France?\nAnswer:",
    "The process of photosynthesis occurs in",
    "In deep space, gravity behaves as",
    "The fundamental principle of neural network optimization is",
    "Question: Why is the sky blue?\nAnswer:",
    "A continuous biological brain manifold differs from a transformer because",
]


def load_v5_checkpoint(ckpt_path: str):
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    step = data.get("step", "unknown")
    loss = data.get("loss", "unknown")
    eval_loss = data.get("eval_loss", "unknown")
    jax_params = {k: jnp.array(v.float().numpy()) for k, v in data["model_state"].items()}
    return jax_params, step, loss, eval_loss


def run_single_stream(jax_params, cfg, buffers, tokenizer, prompt, steps=60, rep_penalty=1.2, delay=0.03):
    prompt_ids = jnp.array([tokenizer.encode(prompt)], dtype=jnp.int32)
    
    print("\n" + "=" * 68)
    print(f"PROMPT: {prompt}")
    print("=" * 68)
    print("[Deliberating in 16 Concentric Laminar Shells across 4 Cycles...]")
    sys.stdout.flush()

    logits = training_rollout(
        params=jax_params,
        prompts=prompt_ids,
        target_len=steps,
        buffers=buffers,
        config=cfg,
        feedback_tokens=None,
        sampling_p=1.0,
        repetition_penalty=rep_penalty,
    )

    print("\n--- EMISSION STREAM ---")
    sys.stdout.flush()

    for t in range(logits.shape[1]):
        tok_id = int(jnp.argmax(logits[0, t]))
        piece = tokenizer.decode([tok_id])
        sys.stdout.write(piece)
        sys.stdout.flush()
        if delay > 0:
            time.sleep(delay)

    print("\n--- [STREAM COMPLETED] ---\n")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=os.path.join(V5_DIR, "checkpoints", "v5_small_cloud_latest.pt"))
    parser.add_argument("--steps", type=int, default=65)
    parser.add_argument("--delay", type=float, default=0.04, help="Pacing delay per token (seconds)")
    parser.add_argument("--loop", action="store_true", default=True, help="Loop through prompts continuously")
    args = parser.parse_args()

    cache_dir = os.path.abspath(os.path.join(ROOT_DIR, ".hf_cache", "hub"))
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b", cache_dir=cache_dir)
    cfg = ModelConfigV5(**PRESETS_V5["v5_small"])
    buffers = generate_small_world_indices(cfg.num_shells, cfg.num_shortcut_tracts)

    last_step = None
    jax_params = None

    print(f"[*] Starting CerebroSpheroid V5 Real-Time Output Engine...")
    print(f"[*] Monitoring checkpoint: {args.checkpoint}")
    sys.stdout.flush()

    prompt_idx = 0

    while True:
        # Check if checkpoint changed or needs initial load
        if not os.path.exists(args.checkpoint):
            print(f"[!] Checkpoint not found at {args.checkpoint}. Retrying in 5s...")
            time.sleep(5)
            continue

        try:
            mtime = os.path.getmtime(args.checkpoint)
            jax_params, step, loss, eval_loss = load_v5_checkpoint(args.checkpoint)
            if step != last_step:
                print(f"\n[★ Checkpoint Loaded] Step: {step} | Train Loss: {loss:.4f} | Eval Loss: {eval_loss}")
                last_step = step
        except Exception as e:
            print(f"[!] Error loading checkpoint: {e}")
            time.sleep(3)
            continue

        prompt = DEFAULT_PROMPTS[prompt_idx % len(DEFAULT_PROMPTS)]
        prompt_idx += 1

        run_single_stream(
            jax_params=jax_params,
            cfg=cfg,
            buffers=buffers,
            tokenizer=tokenizer,
            prompt=prompt,
            steps=args.steps,
            delay=args.delay,
        )

        time.sleep(2.0)


if __name__ == "__main__":
    main()
