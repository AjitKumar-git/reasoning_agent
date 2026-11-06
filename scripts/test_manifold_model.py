#!/usr/bin/env python3
"""
Comprehensive Local Test Suite for CerebroSpheroid V4 Manifold Aligned Model.
Tests:
1. Semantic Geometry & Lexical Cosine Similarity Structure.
2. Singular Value Energy Spectrum & Rank Retention.
3. Volumetric Core Spectral Bounds (Contractive Dynamics).
4. Multi-token Autoregressive Generation & Logit Distributions.
"""

import math
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.corpus import build_tokenizer
from scripts.eval_latest_checkpoint_loss import BitwiseJAXEngine


def test_semantic_similarity(p: dict, tok):
    print("=" * 65)
    print("TEST 1: Lexical Semantic Manifold Geometry")
    print("=" * 65)
    embed = p["sensory.embedding.weight"].float()
    
    pairs = [
        ("king", "queen", "Royal Titles"),
        ("man", "woman", "Gender Counterparts"),
        ("cat", "dog", "Household Pets"),
        ("physics", "mathematics", "STEM Disciplines"),
        ("computer", "software", "Computing"),
        ("apple", "banana", "Fruits"),
        ("king", "software", "Orthogonal Distractors"),
        ("cat", "mathematics", "Orthogonal Distractors"),
    ]
    
    print(f"{'Word Pair':<28} | {'Category':<22} | {'Cosine Sim':<10}")
    print("-" * 65)
    for w1, w2, cat in pairs:
        t1 = tok.encode(w1, add_special_tokens=False)[-1]
        t2 = tok.encode(w2, add_special_tokens=False)[-1]
        
        v1 = embed[t1] / (embed[t1].norm() + 1e-8)
        v2 = embed[t2] / (embed[t2].norm() + 1e-8)
        sim = float(torch.dot(v1, v2).item())
        print(f"'{w1}' <-> '{w2}'".ljust(28) + f" | {cat:<22} | {sim:+.3f}")
        

def test_spectral_radius_core(p: dict):
    print("\n" + "=" * 65)
    print("TEST 2: Volumetric Core Dynamical Stability (Spectral Radius)")
    print("=" * 65)
    intra = p["core.intra_shell"].float()
    K = intra.shape[0]
    
    print(f"{'Concentric Shell':<20} | {'Shape':<16} | {'Max Singular Value (Gain)':<22}")
    print("-" * 65)
    stable = True
    for k in range(K):
        mat = intra[k].numpy()
        s = np.linalg.svd(mat, compute_uv=False)
        max_s = float(s[0])
        status = "PASS (Contractive)" if max_s < 1.0 else "FAIL (Explosive)"
        print(f"Shell k={k:<12} | {str(mat.shape):<16} | {max_s:.4f}  [{status}]")
        if max_s >= 1.0:
            stable = False
    print("-" * 65)
    print(f"Overall Core Contractive Condition (rho < 1.0): {'VERIFIED' if stable else 'FAILED'}")


def test_generation_rollout(p: dict, tok, prompts: list, max_gen: int = 12):
    print("\n" + "=" * 65)
    print("TEST 3: Autoregressive Rollout & Deliberation Dynamics")
    print("=" * 65)
    engine = BitwiseJAXEngine(p, "cpu")
    
    for pr in prompts:
        tokens = tok.encode(pr, return_tensors="pt")
        sens, (keys, vals) = engine.encode_sensory(tokens)
        B, S, D = sens.shape
        K = 16
        state = torch.zeros(B, K, D)
        plan = torch.zeros(B, D)
        refractory = torch.zeros(B, tok.vocab_size)
        
        # Ingest prompt
        for t in range(S):
            state = engine.deliberate_step(state + sens[:, t:t+1, :].expand(-1, K, -1) * 0.2)
            logits, state, refractory, plan = engine.emit_step(
                state, (keys[:, :t+1], vals[:, :t+1]), plan, refractory, step_idx=t
            )
            
        generated_tokens = []
        gen_text = ""
        # Generate autoregressively
        curr_kv = (keys, vals)
        for g in range(max_gen):
            next_token = torch.argmax(logits[0]).item()
            generated_tokens.append(next_token)
            
            # Step forward
            tok_tensor = torch.tensor([[next_token]])
            s_step, (k_step, v_step) = engine.encode_sensory(tok_tensor)
            curr_keys = torch.cat([curr_kv[0], k_step], dim=1)
            curr_vals = torch.cat([curr_kv[1], v_step], dim=1)
            curr_kv = (curr_keys, curr_vals)
            
            state = engine.deliberate_step(state + s_step.expand(-1, K, -1) * 0.2)
            logits, state, refractory, plan = engine.emit_step(
                state, curr_kv, plan, refractory, step_idx=S + g
            )
            
        gen_str = tok.decode(generated_tokens)
        print(f"\nPrompt: \"{pr}\"")
        print(f"  -> Generated ({max_gen} tokens): \"{gen_str}\"")
        

def main():
    path = "checkpoints/cerebro_v4_manifold_aligned.pt"
    if not os.path.exists(path):
        print(f"Checkpoint not found: {path}")
        return
        
    print(f"Loading Manifold Aligned Model from: {path}")
    p = torch.load(path, map_location="cpu", weights_only=False)
    tok = build_tokenizer("llama")
    
    test_semantic_similarity(p, tok)
    test_spectral_radius_core(p)
    
    prompts = [
        "The fundamental theorem of calculus",
        "Artificial intelligence and deep neural networks",
        "Solve for x in the equation",
    ]
    test_generation_rollout(p, tok, prompts)


if __name__ == "__main__":
    main()
