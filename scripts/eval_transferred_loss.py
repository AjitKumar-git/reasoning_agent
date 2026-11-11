#!/usr/bin/env python3
"""
Evaluate Transferred Weight Centroid Model Loss
===============================================
Measures exact Cross-Entropy loss and perplexity of the transferred model
on held-out OpenHermes-2.5 and Cosmopedia-v2 benchmark splits.
"""

import math
import os
import random
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5
from scripts.eval_latest_checkpoint_loss import load_openhermes_eval_records, load_cosmopedia_eval_records
from training.train_v4_full import build_batch


def evaluate_transferred(
    ckpt_path: str = "checkpoints/centroid_pythia70m_transferred.pt",
    model_id: str = "EleutherAI/pythia-70m",
    eval_count: int = 48,
    batch_size: int = 8,
):
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("🔬 EVALUATING TRANSFERRED WEIGHT CENTROID MODEL")
    print(f"   Checkpoint: {ckpt_path}")
    print(f"   Tokenizer : {model_id}")
    print(f"   Device    : {dev}")
    print("=" * 80)

    # 1. Load Tokenizer
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir="/tmp/hf_cache")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0
    pad_id = tok.pad_token_id

    # 2. Load Checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ModelConfigV5(**ckpt["config"])
    model = CerebroSpheroidV5(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(dev).eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Loaded Centroid Model: {total_params:,} parameters")

    # 3. Load Benchmark Records
    n_per_split = eval_count // 2
    records = load_openhermes_eval_records(count=n_per_split) + load_cosmopedia_eval_records(count=n_per_split)
    print(f"📥 Loaded {len(records)} held-out records (OpenHermes-2.5 + Cosmopedia-v2).")

    # 4. Measure Cross-Entropy Loss
    rng = random.Random(1337)
    tf_losses = []
    auto_losses = []

    for i in range(0, len(records), batch_size):
        chunk = records[i : i + batch_size]
        batch = build_batch(
            tok,
            chunk,
            window=64,
            rng=rng,
            max_prompt_len=128,
            max_answer_tokens=64,
            pad_token_id=pad_id,
        )
        if batch is None:
            continue
        p_t, t_t, _ = batch
        p_t, t_t = p_t.to(dev), t_t.to(dev)

        with torch.no_grad():
            # Teacher-forcing (sampling_p = 1.0)
            loss_tf, _ = model(p_t, targets=t_t, sampling_p=1.0, pad_id=pad_id)
            # Autonomous internal deliberation (sampling_p = 0.0)
            loss_auto, _ = model(p_t, targets=t_t, sampling_p=0.0, pad_id=pad_id)

            tf_losses.append(loss_tf.item())
            auto_losses.append(loss_auto.item())

    mean_tf = sum(tf_losses) / max(1, len(tf_losses))
    mean_auto = sum(auto_losses) / max(1, len(auto_losses))
    ppl_tf = math.exp(min(mean_tf, 20.0))
    ppl_auto = math.exp(min(mean_auto, 20.0))

    print("\n" + "=" * 80)
    print("📈 EVALUATION RESULTS FOR TRANSFERRED WEIGHT MODEL:")
    print(f"   Teacher-Forcing Loss     : {mean_tf:.4f} (Perplexity: {ppl_tf:.2f})")
    print(f"   Autonomous Feedback Loss : {mean_auto:.4f} (Perplexity: {ppl_auto:.2f})")
    print(f"   Autonomous Gap           : {mean_auto - mean_tf:+.4f} loss units")
    print("=" * 80)

    # 5. Test Sample Generation
    test_prompts = [
        "What is the capital of France?",
        "Explain why the sky is blue in simple words.",
        "What is 2 + 3?",
    ]
    print("\n📝 Sample Text Generation:")
    for q in test_prompts:
        prompt_ids = tok.encode(q, return_tensors="pt").to(dev)
        with torch.no_grad():
            gen_out = model.generate(
                prompt_ids,
                max_new_tokens=28,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=pad_id,
            )
        generated_tokens = gen_out[0, prompt_ids.size(1):]
        text = tok.decode(generated_tokens, skip_special_tokens=True).strip()
        print(f"   Prompt : \"{q}\"")
        print(f"   Output : \"{text}\"\n")

    return {
        "tf_loss": mean_tf,
        "auto_loss": mean_auto,
        "tf_ppl": ppl_tf,
        "auto_ppl": ppl_auto,
    }


if __name__ == "__main__":
    evaluate_transferred()
