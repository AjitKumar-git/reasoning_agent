"""
Comprehensive Generation Evaluation Script for CerebroSpheroid Checkpoints.
Evaluates both greedy (temp=0.0) and stochastic (temp=0.7) outputs on standard prompts.
"""

import os
import sys
import jax
import jax.numpy as jnp

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.corpus import build_tokenizer
from src.jax import ModelConfigV4, load_torch_pt_to_jax, generate_v4

TEST_PROMPTS = [
    "What is the capital of France?",
    "If a basket has 15 apples and you take away 6, how many apples remain?",
    "Explain in simple terms why the sky is blue.",
    "The process of photosynthesis converts sunlight into",
    "Hello! How are you doing today?",
]

def evaluate_checkpoint(ckpt_path: str, max_new_tokens: int = 48):
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found: {ckpt_path}")
        return

    print("=" * 80)
    print(f"🔍 Evaluating Checkpoint: {ckpt_path}")
    print("=" * 80)

    config = ModelConfigV4(vocab_size=32000, surface_dim=640, num_shells=16, radial_dim=256, num_shortcut_tracts=40)
    params, buffers = load_torch_pt_to_jax(ckpt_path, config)
    tokenizer = build_tokenizer("llama")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_id = tokenizer.eos_token_id

    for idx, prompt_text in enumerate(TEST_PROMPTS, 1):
        print(f"\n--- [Prompt {idx}]: {prompt_text} ---")
        input_ids = jnp.array([tokenizer.encode(prompt_text)])

        for temp in [0.0, 0.7]:
            temp_desc = "Greedy (temp=0.0)" if temp == 0.0 else f"Sampling (temp={temp})"
            try:
                out_tokens = generate_v4(
                    params,
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    buffers=buffers,
                    config=config,
                    temperature=temp,
                    repetition_penalty=1.25,
                    thalamic_anchor_weight=0.02,
                    pad_id=pad_id,
                    eos_token_id=eos_id,
                )
                generated_text = tokenizer.decode(out_tokens[0])
                print(f"  [{temp_desc}]:")
                print(f"  >>> {generated_text.strip()}")
            except Exception as e:
                print(f"  [{temp_desc}] Error during generation: {e}")

if __name__ == "__main__":
    target_ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/cerebro_spheroid_v4_step_158500_remote_latest.pt"
    evaluate_checkpoint(target_ckpt)
