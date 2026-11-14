#!/usr/bin/env python3
"""
Continuous Output Runner for the V5 model actively training in cerebro_spheroid_v5.
Loads the cloud/TPU JAX-trained checkpoint (step 64,750, loss 3.42) using exact native
JAX training_rollout closed-loop emission.
"""

import os
import sys
import argparse
import torch
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
V5_DIR = os.path.abspath(os.path.join(ROOT_DIR, "..", "cerebro_spheroid_v5"))
sys.path.insert(0, V5_DIR)

from src.jax.v5_model import ModelConfigV5, PRESETS_V5, training_rollout, generate_small_world_indices


def main():
    parser = argparse.ArgumentParser(description="Stream continuous output from V5 actively training model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(V5_DIR, "checkpoints", "v5_small_cloud_latest.pt"),
        help="Path to latest V5 checkpoint",
    )
    parser.add_argument("--prompt", type=str, default="Question: What is photosynthesis?\nAnswer:")
    parser.add_argument("--steps", type=int, default=80, help="Number of continuous tokens to emit")
    parser.add_argument("--preset", type=str, default="v5_small")
    parser.add_argument("--rep_penalty", type=float, default=1.2)
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    step = data.get("step", "unknown")
    loss = data.get("loss", "unknown")
    eval_loss = data.get("eval_loss", "unknown")
    print(f"Checkpoint Info -> Step: {step} | Train Loss: {loss:.4f} | Eval Loss: {eval_loss}")

    # Load Tokenizer
    cache_dir = os.path.abspath(os.path.join(ROOT_DIR, ".hf_cache", "hub"))
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b", cache_dir=cache_dir)

    # Convert checkpoint PyTorch weights to JAX params
    jax_params = {k: jnp.array(v.float().numpy()) for k, v in data["model_state"].items()}
    cfg = ModelConfigV5(**PRESETS_V5[args.preset])
    buffers = generate_small_world_indices(cfg.num_shells, cfg.num_shortcut_tracts)

    prompt_ids = jnp.array([tokenizer.encode(args.prompt)], dtype=jnp.int32)
    print(f"\nPrompt: {args.prompt!r}")
    print(f"Ingesting into 16 Concentric Laminar Shells across 4 Deliberation Cycles...")

    logits = training_rollout(
        params=jax_params,
        prompts=prompt_ids,
        target_len=args.steps,
        buffers=buffers,
        config=cfg,
        feedback_tokens=None,
        sampling_p=1.0,
        repetition_penalty=args.rep_penalty,
    )

    print("\n" + "=" * 65)
    print(" CONTINUOUS EMISSION STREAM (CEREBRO-SPHEROID V5 CLOUD MODEL)")
    print("=" * 65 + "\n")

    emitted_ids = []
    for t in range(logits.shape[1]):
        tok_id = int(jnp.argmax(logits[0, t]))
        emitted_ids.append(tok_id)
        piece = tokenizer.decode([tok_id])
        sys.stdout.write(piece)
        sys.stdout.flush()

    print("\n\n" + "=" * 65)
    print(" END OF CONTINUOUS STREAM")
    print("=" * 65)


if __name__ == "__main__":
    main()
