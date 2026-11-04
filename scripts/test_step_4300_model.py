"""
Comprehensive Evaluation Suite for CerebroSpheroid V4 Checkpoint (Step 4,300).
Evaluates parameter diagnostics, held-out rollout loss, and multi-prompt token generation.
"""

import os
import sys
import math
import time
import torch
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import jax
import jax.numpy as jnp
from src.corpus import build_tokenizer
from src.jax import (
    ModelConfigV4,
    PRESETS_V4,
    load_torch_pt_to_jax,
    training_rollout_v4,
    emit_step,
    deliberate_step,
    encode_sensory,
    generate_v4,
)


PROMPTS = [
    "What is 2 + 2?",
    "If you have 5 apples and you give 2 away, how many are left?",
    "What is the capital of France?",
    "Explain what photosynthesis is in simple terms.",
    "Solve for x: 3x - 9 = 0.",
    "Write a short haiku about the ocean.",
]


def test_model(ckpt_path: str):
    print("=" * 80)
    print(f"🧠 CEREBRO-SPHEROID V4: COMPREHENSIVE OFFLINE EVALUATION")
    print(f"Checkpoint File : {ckpt_path}")
    print(f"File Size       : {os.path.getsize(ckpt_path) / 1e6:.1f} MB")
    print("=" * 80)

    # 1. Parameter Inspection
    print("\n[1/4] Inspecting Parameters and Weight Health...")
    config = ModelConfigV4(vocab_size=32000, **PRESETS_V4["small"])
    params, buffers = load_torch_pt_to_jax(ckpt_path, config)

    total_params = sum(int(np.prod(v.shape)) for v in params.values())
    nan_keys = []
    inf_keys = []
    for k, v in params.items():
        if bool(jnp.isnan(v).any()):
            nan_keys.append(k)
        if bool(jnp.isinf(v).any()):
            inf_keys.append(k)

    print(f"   Total Active Parameters : {total_params:,}")
    print(f"   Total Parameter Arrays  : {len(params)}")
    print(f"   NaN Status              : {'CLEAN (0 NaNs)' if not nan_keys else f'FAIL ({len(nan_keys)} keys)'}")
    print(f"   Inf Status              : {'CLEAN (0 Infs)' if not inf_keys else f'FAIL ({len(inf_keys)} keys)'}")
    assert not nan_keys and not inf_keys, "Checkpoint contains non-finite parameters!"

    # 2. Tokenizer Setup
    print("\n[2/4] Initializing LLaMA Tokenizer...")
    tok = build_tokenizer("llama")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    # 3. Dynamic Forward Rollout and Simulated Loss Test
    print("\n[3/4] Testing Dynamic Rollout & Loss on Sample Sequences...")
    sample_text = "The quick brown fox jumps over the lazy dog."
    sample_target = " A classic pangram containing every letter."
    enc_q = tok.encode(sample_text, return_tensors="np")
    enc_r = tok.encode(sample_target, return_tensors="np")

    q_arr = jnp.array(enc_q)
    r_arr = jnp.array(enc_r)
    target_len = r_arr.shape[1]

    t0 = time.time()
    logits = training_rollout_v4(params, q_arr, target_len=target_len, buffers=buffers, config=config)
    rollout_time = time.time() - t0

    # Cross-entropy loss
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_one_hot = jax.nn.one_hot(r_arr, config.vocab_size)
    sample_loss = float(-jnp.mean(jnp.sum(log_probs * target_one_hot, axis=-1)))
    sample_ppl = math.exp(min(sample_loss, 20.0))

    print(f"   Rollout Latency : {rollout_time * 1000:.1f} ms")
    print(f"   Logits Shape    : {logits.shape}")
    print(f"   Sample Loss     : {sample_loss:.4f} (Perplexity: {sample_ppl:.2f})")
    print(f"   Logits Mean/Std : {float(jnp.mean(logits)):.4f} / {float(jnp.std(logits)):.4f}")

    # 4. Multi-Prompt Deliberative Generation with Plan-First & Two-Tier Refractory
    print("\n[4/4] Generating Responses across Multi-Domain Prompts (Plan-First + Refractory)...")
    print("-" * 80)

    PERSONAS = {
        0: "Reasoner",
        1: "Explorer",
        2: "Concise",
        3: "Verifier",
    }

    for p in PROMPTS:
        formatted_prompt = f"User: {p}\nAssistant:"
        toks = jnp.array(tok.encode(formatted_prompt, return_tensors="np"))
        print(f"PROMPT: {p}")

        # Test canonical generation (Reasoner persona, repetition_penalty=1.3)
        gen_tokens = generate_v4(
            params,
            toks,
            max_new_tokens=24,
            buffers=buffers,
            config=config,
            persona_id=0,
            temperature=0.7,
            repetition_penalty=1.3,
            pad_id=pad_id,
            eos_token_id=tok.eos_token_id,
        )
        decoded = tok.decode(gen_tokens[0]).strip()
        print(f"  [Persona 0: Reasoner] : {decoded}")

        # Test Explorer persona
        gen_exp = generate_v4(
            params,
            toks,
            max_new_tokens=24,
            buffers=buffers,
            config=config,
            persona_id=1,
            temperature=0.7,
            repetition_penalty=1.3,
            pad_id=pad_id,
            eos_token_id=tok.eos_token_id,
        )
        decoded_exp = tok.decode(gen_exp[0]).strip()
        print(f"  [Persona 1: Explorer] : {decoded_exp}")
        print("-" * 80)

    print("\n✅ Comprehensive Evaluation Completed Successfully!")


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/cerebro_spheroid_v4_step_4300.pt"
    test_model(ckpt)
