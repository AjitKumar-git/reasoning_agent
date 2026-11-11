#!/usr/bin/env python3
"""
Benchmark: CerebroSpheroid V4 (155,600 Best) vs. Upgraded Centroid Architecture
================================================================================

Measures the exact structural improvements across:
1. Teacher-Forcing Cross-Entropy Loss
2. Free-Running Autonomous Generation Loss (The Teacher-Forcing vs Autonomous Gap)
3. Token Sequence Retention (No Prompt Mean-Pooling)
4. Qualitative Text Generation on Standard Prompts
"""

import math
import os
import sys
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V5_ROOT = os.path.abspath(os.path.join(ROOT_DIR, "..", "cerebro_spheroid_v5"))
sys.path.insert(0, ROOT_DIR)
if os.path.exists(V5_ROOT):
    sys.path.append(V5_ROOT)

from src.corpus import build_tokenizer
from src.spheroid_v4 import build_spheroid_v4
from scripts.eval_latest_checkpoint_loss import load_openhermes_eval_records, load_cosmopedia_eval_records
from training.train_v4_full import build_batch

try:
    from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5
    HAS_V5 = True
except ImportError:
    HAS_V5 = False


def evaluate_v4(model, eval_records, tokenizer, device, max_batches=8):
    model.eval()
    tf_losses = []
    auto_losses = []
    
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    rng = random = __import__("random").Random(1337)
    
    for i in range(0, min(len(eval_records), max_batches * 8), 8):
        chunk = eval_records[i : i + 8]
        batch = build_batch(tokenizer, chunk, window=64, rng=rng, max_prompt_len=128, max_answer_tokens=64, pad_token_id=pad_id)
        if batch is None:
            continue
        p_t, t_t, f_t = batch
        p_t, t_t, f_t = p_t.to(device), t_t.to(device), f_t.to(device)
        
        with torch.no_grad():
            # 1. Teacher forcing
            tf_logits = model.training_logits(p_t, response_steps=t_t.size(1), thinking_cycles=3, targets=t_t, deliberation_mode="teacher_forcing")
            tf_loss = F.cross_entropy(tf_logits.view(-1, tf_logits.size(-1)), t_t.view(-1), ignore_index=-100)
            tf_losses.append(tf_loss.item())
            
            # 2. Autonomous internal deliberation
            auto_logits = model.training_logits(p_t, response_steps=t_t.size(1), thinking_cycles=3, deliberation_mode="internal")
            auto_loss = F.cross_entropy(auto_logits.view(-1, auto_logits.size(-1)), t_t.view(-1), ignore_index=-100)
            auto_losses.append(auto_loss.item())

    mean_tf = sum(tf_losses) / max(1, len(tf_losses))
    mean_auto = sum(auto_losses) / max(1, len(auto_losses))
    return mean_tf, mean_auto


def main():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok = build_tokenizer("llama")
    print("=" * 80)
    print("🔬 COMPARATIVE BENCHMARK: CEREBRO-SPHEROID V4 VS. UPGRADED CENTROID")
    print(f"   Execution Device: {dev}")
    print("=" * 80)

    # 1. Load V4 Champion Checkpoint (Step 155,600)
    v4_path = "checkpoints/cerebro_spheroid_v4_step_155600_best.pt"
    v4_model = build_spheroid_v4("v4_small", vocab_size=len(tok))
    v4_model.load_compatible_checkpoint(v4_path, device=dev)
    v4_model.to(dev).eval()
    print(f"✅ Loaded V4 Champion Checkpoint: {v4_path} ({sum(p.numel() for p in v4_model.parameters()):,} params)")

    # 2. Load Evaluation Records
    print("\n📥 Loading Held-Out Benchmark Records (OpenHermes-2.5 + Cosmopedia-v2)...")
    records = load_openhermes_eval_records(count=24) + load_cosmopedia_eval_records(count=24)
    print(f"   Loaded {len(records)} held-out evaluation pairs.")

    # 3. Evaluate V4
    print("\n📊 Evaluating CerebroSpheroid V4 (Step 155,600)...")
    v4_tf, v4_auto = evaluate_v4(v4_model, records, tok, dev)
    print(f"   V4 Teacher-Forcing Loss : {v4_tf:.4f} (Perplexity: {math.exp(v4_tf):.2f})")
    print(f"   V4 Autonomous Free Loss : {v4_auto:.4f} (Perplexity: {math.exp(v4_auto):.2f})")
    print(f"   V4 Autonomous Degradation Gap: +{v4_auto - v4_tf:.4f} loss units")

    # 4. Generate Qualitative Outputs from V4
    test_prompts = [
        "What is the capital of France?",
        "Explain why the sky is blue in simple words.",
        "What is 2 + 3?",
    ]
    print("\n📝 V4 Sample Generation Outputs (Step 155,600):")
    for q in test_prompts:
        gen = v4_model.generate_text(q, tokenizer=tok, thinking_cycles=3, max_new_tokens=28, temperature=0.3, repetition_penalty=1.2, device=dev)
        print(f"   Prompt : \"{q}\"")
        print(f"   Output : \"{gen}\"\n")


if __name__ == "__main__":
    main()
