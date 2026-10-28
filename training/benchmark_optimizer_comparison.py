"""
Head-to-head Empirical Benchmark on CerebroSpheroid V4 (25M Probe):
SpheroidOptimizer (Orthogonal Polar + Spectral Clamp) vs AdamW vs SGD.

Measures:
1. Convergence Rate & Cross-Entropy Loss
2. Manifold Effective Rank Preservation (Spectral Health)
3. Volumetric Core Spectral Norm: rho(W_recurrent)
4. Peak Allocated GPU VRAM
"""

import math
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.spheroid_v4 import build_spheroid_v4, compute_manifold_diagnostics
from src.spheroid_optimizer import build_spheroid_optimizer


def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Benchmark] Running on: {device}")

    # Standard multi-task dataset for comparison
    prompts = torch.tensor([
        [101, 205, 309, 412, 515, 620],
        [701, 802, 903, 1004, 1105, 1206],
        [301, 402, 503, 604, 705, 806],
        [901, 902, 903, 904, 905, 906],
    ], device=device)

    targets = torch.tensor([
        [1501, 1502, 1503, 1504],
        [2501, 2502, 2503, 2504],
        [3501, 3502, 3503, 3504],
        [4501, 4502, 4503, 4504],
    ], device=device)

    def evaluate_optimizer(opt_name: str, steps: int = 30):
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        model = build_spheroid_v4("v4_probe_25m", use_causal_ingestion=True).to(device)
        model.train()

        # Build optimizer
        if opt_name == "spheroid":
            opt = build_spheroid_optimizer(
                model,
                lr=0.005,
                lr_1d=0.001,
                weight_decay=0.01,
                spectral_clamp=1.0,
            )
        elif opt_name == "adamw":
            opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
        elif opt_name == "sgd":
            opt = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=0.01)
        else:
            raise ValueError(f"Unknown optimizer {opt_name}")

        losses = []
        accuracies = []

        for step in range(1, steps + 1):
            opt.zero_grad()
            logits = model.training_logits(
                prompt_tokens=prompts,
                response_steps=targets.shape[1],
                targets=targets,
                thinking_cycles=1,
            )
            loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            preds = logits.argmax(dim=-1)
            acc = (preds == targets).float().mean().item() * 100.0

            losses.append(loss.item())
            accuracies.append(acc)

        # Post-training manifold diagnostics
        model.eval()
        with torch.no_grad():
            state = model.ingest_stimulus(prompts)
            diag = compute_manifold_diagnostics(model, state)
            
            # Measure spectral norm of intra_shell recurrence
            W_intra = model.volumetric_core.intra_shell # [K, D, D]
            sigmas = [torch.linalg.matrix_norm(W_intra[i], ord=2).item() for i in range(W_intra.shape[0])]
            max_sigma = max(sigmas)

        peak_vram_mb = (
            torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        )

        return {
            "name": opt_name,
            "loss_init": losses[0],
            "loss_final": losses[-1],
            "acc_final": accuracies[-1],
            "effective_rank": diag["mean_effective_rank"],
            "shell_rms": diag["mean_shell_rms"],
            "max_spectral_norm": max_sigma,
            "peak_vram_mb": peak_vram_mb,
        }

    print("\n" + "=" * 75)
    print(f" BENCHMARKING OPTIMIZERS ON CEREBROSPHEROID V4 (25M PROBE, 30 STEPS)")
    print("=" * 75)

    results = []
    for name in ["spheroid", "adamw", "sgd"]:
        print(f"--> Running {name.upper()}...")
        res = evaluate_optimizer(name, steps=30)
        results.append(res)
        print(f"    Loss: {res['loss_init']:.4f} -> {res['loss_final']:.4f} | Acc: {res['acc_final']:.1f}%")
        print(f"    Spectral Norm: {res['max_spectral_norm']:.3f} | Rank: {res['effective_rank']:.2f} | Peak VRAM: {res['peak_vram_mb']:.1f} MB")

    print("\n" + "=" * 75)
    print(f"{'Optimizer':<12} | {'Init Loss':<10} | {'Final Loss':<10} | {'Acc':<6} | {'Max Spec':<9} | {'Rank':<6} | {'Peak VRAM'}")
    print("-" * 75)
    for r in results:
        print(
            f"{r['name'].upper():<12} | {r['loss_init']:<10.4f} | {r['loss_final']:<10.4f} | "
            f"{r['acc_final']:<5.1f}% | {r['max_spectral_norm']:<9.3f} | {r['effective_rank']:<6.2f} | {r['peak_vram_mb']:.1f} MB"
        )
    print("=" * 75)


if __name__ == "__main__":
    run_benchmark()
