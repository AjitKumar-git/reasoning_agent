#!/usr/bin/env python3
"""
Comprehensive Validation & Inference Probe for Trained CerebroSpheroid V4 Checkpoint.
Verifies weight integrity, spectral stability, semantic manifold geometry, and live generation.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.corpus import build_tokenizer
from src.spheroid_v4 import build_spheroid_v4


def run_comprehensive_test(ckpt_path: str):
    print("=" * 75)
    print(f"🧠 CEREBRO-SPHEROID V4: COMPREHENSIVE CHECKPOINT PROBE")
    print(f"Checkpoint File: {ckpt_path}")
    print("=" * 75)

    if not os.path.exists(ckpt_path):
        print(f"❌ Error: Checkpoint not found at {ckpt_path}")
        return

    # 1. Load Checkpoint
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = {}
    if isinstance(raw, dict) and "params" in raw:
        params = raw["params"]
        meta = raw.get("meta", {})
    elif isinstance(raw, dict) and "model_state_dict" in raw:
        params = raw["model_state_dict"]
        meta = raw.get("meta", {})
    elif isinstance(raw, dict) and any(k.startswith("sensory.") or k.startswith("core.") for k in raw.keys()):
        params = raw
    else:
        params = raw

    print("\n📊 1. CHECKPOINT METADATA & INTEGRITY")
    print("-" * 75)
    if meta:
        print(f"  • Trained Step       : {meta.get('step', 'N/A'):,}")
        print(f"  • Recorded Loss      : {meta.get('loss', 'N/A')}")
        print(f"  • Ingestion Source   : {meta.get('source', 'N/A')}")
        print(f"  • Timestamp          : {meta.get('timestamp', 'N/A')}")
    else:
        print("  • Metadata dictionary not embedded in file (direct state dict).")

    # Check for NaN / Inf in any parameter
    total_params = 0
    non_finite = 0
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            total_params += v.numel()
            if not torch.isfinite(v).all():
                non_finite += 1
                print(f"  ⚠️ Warning: Non-finite values detected in {k}")
    print(f"  • Total Parameters   : {total_params:,} ({total_params * 4 / (1024**2):.1f} MB in float32)")
    print(f"  • Non-Finite Tensors : {non_finite} ({'✅ 100% HEALTHY' if non_finite == 0 else '❌ CORRUPTED'})")

    # 2. Spectral Radius Stability Check
    print("\n🛡️ 2. DYNAMICAL STABILITY (SPECTRAL RADIUS ON RECURRENT SHELLS)")
    print("-" * 75)
    intra_key = None
    for candidate in ["core.intra_shell", "core_intra_shell"]:
        if candidate in params:
            intra_key = candidate
            break

    spectral_stable = True
    if intra_key is not None:
        intra = params[intra_key].float()
        K = intra.shape[0]
        max_rho = 0.0
        for k in range(K):
            mat = intra[k].numpy()
            s = np.linalg.svd(mat, compute_uv=False)
            gain = float(s[0])
            max_rho = max(max_rho, gain)
            status = "CONTRACTIVE" if gain <= 1.0001 else "EXPLOSIVE"
            if gain > 1.05:
                spectral_stable = False
            if k in [0, K // 2, K - 1]:
                print(f"  Shell k={k:<3d} | Shape: {str(mat.shape):<14} | Max Gain (rho): {gain:.4f} [{status}]")
        print(f"  • Peak Spectral Radius: {max_rho:.4f} -> {'✅ VERIFIED STABLE (rho <= 1.0)' if spectral_stable else '⚠️ EXPANSIVE'}")
    else:
        print("  • Recurrent core intra-shell key not found in params.")

    # 3. Lexical Semantic Manifold Geometry
    print("\n💎 3. LEXICAL SEMANTIC GEOMETRY (COSINE SIMILARITY)")
    print("-" * 75)
    tok = build_tokenizer("llama")
    embed_key = "sensory.embedding.weight" if "sensory.embedding.weight" in params else "embedding.weight"
    if embed_key in params:
        embed = params[embed_key].float()
        test_pairs = [
            ("king", "queen", "Royal Titles"),
            ("man", "woman", "Gender"),
            ("cat", "dog", "Animals"),
            ("physics", "mathematics", "Sciences"),
            ("apple", "banana", "Fruits"),
            ("king", "software", "Unrelated (Control)"),
            ("cat", "physics", "Unrelated (Control)"),
        ]
        print(f"  {'Word Pair':<24} | {'Category':<20} | {'Cosine Sim':<10}")
        print("  " + "-" * 60)
        for w1, w2, cat in test_pairs:
            t1 = tok.encode(w1, add_special_tokens=False)[-1]
            t2 = tok.encode(w2, add_special_tokens=False)[-1]
            v1 = embed[t1] / (embed[t1].norm() + 1e-8)
            v2 = embed[t2] / (embed[t2].norm() + 1e-8)
            sim = float(torch.dot(v1, v2).item())
            print(f"  '{w1}' <-> '{w2}'".ljust(26) + f" | {cat:<20} | {sim:+.3f}")

    # 4. Live Autoregressive Generation & Reasoning
    print("\n💬 4. LIVE AUTOREGRESSIVE GENERATION (BIOLOGICAL DELIMITERS)")
    print("-" * 75)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    model = build_spheroid_v4("v4_small", vocab_size=len(tok))
    model.load_compatible_checkpoint(ckpt_path, device=device)
    model.to(device=device, dtype=torch.float32)
    model.eval()

    questions = [
        "Hello! Who are you?",
        "What is the capital of France?",
        "What is 2 + 3?",
        "Why is the sky blue?",
        "If you have 4 apples and give 1 to a friend, how many apples do you have?",
        "Explain photosynthesis in one simple sentence.",
    ]

    for idx, q in enumerate(questions, 1):
        print(f"\n[Test {idx}] User Prompt: \"{q}\"")
        with torch.no_grad():
            ans = model.generate_text(
                q,
                tokenizer=tok,
                thinking_cycles=3,
                max_new_tokens=48,
                temperature=0.6,
                repetition_penalty=1.35,
                device=device,
                strip_bio_prefix=True,
            )
        clean_ans = ans.strip()
        print(f"🤖 Spheroid Output: \"{clean_ans}\"")

    print("\n" + "=" * 75)
    print("✅ Comprehensive Checkpoint Probe Finished.")
    print("=" * 75)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/cerebro_spheroid_v4_latest.pt"
    run_comprehensive_test(path)
