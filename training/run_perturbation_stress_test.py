"""
Automated Perturbation Stress-Test & Resilience Benchmark
=========================================================
Keeps the living organoid running in the background while systematically injecting
7 distinct biological, physical, and architectural perturbations into the continuous
3D Riemannian manifold (B^3), measuring:
- Dynamical Trajectory Divergence D(t)
- Kinetic Energy Jumps Delta E_k
- Homeostatic Relaxation Time tau_relax
- Cognitive Reasoning Pass Retention Rate
- Qualitative Reasoning Shift (Clean vs. Perturbed Outputs)
"""

import os
import sys
import time
import math
import json
import argparse
from typing import List, Dict, Any

import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, ".hf_cache"))

from src.living_perturbation_framework import LivingOrganoidHost, PerturbationConfig


PROBE_SUITE = [
    {
        "id": "1_platypus",
        "pillar": "Deductive Syllogism",
        "prompt": "All quadrupedal mammals produce milk for their young. A platypus lays eggs but nurses its young with milk. Is a platypus a mammal? Answer yes or no and state why.",
        "exact_targets": ["mammal", "is a mammal"],
    },
    {
        "id": "2_density_chain",
        "pillar": "Multi-Hop Transitivity",
        "prompt": "Iron is denser than Aluminium. Platinum is denser than Iron. Lithium is lighter than Aluminium. Which of these four metals is the densest?",
        "exact_targets": ["platinum", "platinum is the densest"],
    },
    {
        "id": "3_inverted_buoyancy",
        "pillar": "Counterfactual Physics",
        "prompt": "If hot air were denser than cold air, in what direction would hot smoke from a chimney move? State the direction clearly.",
        "exact_targets": ["downward", "down", "sink", "ground"],
    },
    {
        "id": "4_notebook_change",
        "pillar": "Multi-Step Arithmetic",
        "prompt": "Maya buys 6 notebooks priced at $4 each. Maya pays with a $50 bill. Exactly how much change does Maya receive?",
        "exact_targets": ["26", "$26", "26 dollars"],
    },
    {
        "id": "5_cube_faces",
        "pillar": "Spatial ARC-AGI",
        "prompt": "A geometric cube has 6 faces. If you paint exactly 2 opposite faces red and all the remaining faces blue, exactly how many blue faces are there?",
        "exact_targets": ["4", "4 faces", "four"],
    },
    {
        "id": "6_matrix_reflection",
        "pillar": "Matrix Spatial Symmetry",
        "prompt": "A 3x3 matrix has rows ['. # .', '. # .', '. # .'] with a vertical line in column 2. What happens to this vertical line when the matrix is reflected horizontally?",
        "exact_targets": ["unchanged", "same", "remains in column 2", "column 2"],
    },
]


def run_stress_test(checkpoint: str, thinking_cycles: int = 3) -> Dict[str, Any]:
    print("=" * 90)
    print("      LIVING ORGANOID CONTINUOUS PERTURBATION STRESS-TEST ENGINE      ")
    print("=" * 90)
    print(f"Target Checkpoint : {checkpoint}")
    print(f"Thinking Depth    : {thinking_cycles} cycles")
    print(f"Modality Support  : Continuous 3D Ball Wave Interference (B^3) + Tripartite Glia")
    print("=" * 90 + "\n")

    host = LivingOrganoidHost(checkpoint_path=checkpoint, heartbeat_hz=3.0)
    host.start()
    time.sleep(1.0)

    results = {
        "checkpoint": checkpoint,
        "clean_baseline": {},
        "perturbations": [],
    }

    try:
        # 1. Clean Baseline Evaluation
        print("▶ PHASE 1: EVALUATING UNPERTURBED CLEAN BASELINE")
        clean_scores = {}
        for probe in PROBE_SUITE:
            res = host.query(probe["prompt"], thinking_cycles=thinking_cycles, max_answer_tokens=48)
            ans = res["response"].lower()
            hit = any(t in ans for t in probe["exact_targets"])
            clean_scores[probe["id"]] = {
                "pillar": probe["pillar"],
                "hit": hit,
                "response": res["response"],
            }
            tag = "★ PASS" if hit else "○ EVOLVE"
            print(f"  [{probe['pillar']}] -> {tag}")
            print(f"    Q: \"{probe['prompt'][:60]}...\"")
            print(f"    A: \"{res['response']}\"")

        clean_pass_rate = sum(1 for v in clean_scores.values() if v["hit"]) / len(PROBE_SUITE) * 100.0
        results["clean_baseline"] = {
            "pass_rate_pct": clean_pass_rate,
            "probes": clean_scores,
        }
        print(f"\n--> Clean Baseline Pass Rate: {clean_pass_rate:.1f}%\n")

        # 2. Perturbation Regimens to Test
        regimens = [
            # Regimen 1: Thermal Noise
            PerturbationConfig(
                name="Mild Thermal Noise (sigma=0.05)",
                perturb_type="thermal",
                intensity=0.05,
                duration_ticks=5,
                description="Langevin thermal jitter on all 8 concentric shells",
            ),
            PerturbationConfig(
                name="Severe Thermal Noise (sigma=0.25)",
                perturb_type="thermal",
                intensity=0.25,
                duration_ticks=5,
                description="High-energy thermal noise testing dynamical limit",
            ),
            # Regimen 2: Astrocytic Calcium Shock
            PerturbationConfig(
                name="Astrocytic Calcium Surge (surge=1.0)",
                perturb_type="astro_shock",
                intensity=1.0,
                duration_ticks=5,
                description="Tripartite glial Ca2+ surge driving hyper-conductance",
            ),
            PerturbationConfig(
                name="Excitotoxic Glial Shock (surge=2.5)",
                perturb_type="astro_shock",
                intensity=2.5,
                duration_ticks=5,
                description="Extreme astrocytic metabolic flood",
            ),
            # Regimen 3: Cortical Columnar Lesion
            PerturbationConfig(
                name="Focal Columnar Lesion (30 deg, West Pole)",
                perturb_type="columnar_lesion",
                intensity=0.0,
                target_coords=(-1.0, 0.0, 0.0),
                lesion_radius_deg=30.0,
                duration_ticks=5,
                description="Complete inactivation of 30-deg patch on sensory pole",
            ),
            PerturbationConfig(
                name="Large Columnar Lesion (60 deg, North Pole)",
                perturb_type="columnar_lesion",
                intensity=0.2,
                target_coords=(0.0, 0.0, 1.0),
                lesion_radius_deg=60.0,
                duration_ticks=5,
                description="Partial ischemia over motor pole",
            ),
            # Regimen 4: Thalamic Sensory Noise
            PerturbationConfig(
                name="Thalamic Sensory Noise (sigma=0.20)",
                perturb_type="thalamic_noise",
                intensity=0.20,
                duration_ticks=5,
                description="Afferent input noise before thalamocortical filtering",
            ),
            # Regimen 5: Subcortical Corollary Ablation
            PerturbationConfig(
                name="Partial Corollary Ablation (50% feedback)",
                perturb_type="corollary_ablation",
                intensity=0.50,
                duration_ticks=5,
                description="Dampens internal motor efference copy to subcortical core",
            ),
            PerturbationConfig(
                name="Complete Corollary Disconnection (0% feedback)",
                perturb_type="corollary_ablation",
                intensity=0.00,
                duration_ticks=5,
                description="Completely severs internal corollary discharge loop",
            ),
            # Regimen 6: Core Radial Soliton Impulse
            PerturbationConfig(
                name="Deep Subcortical Soliton Pulse (amp=2.0)",
                perturb_type="core_pulse",
                intensity=2.0,
                target_shells=[0, 1],
                duration_ticks=5,
                description="High-energy soliton burst injected into k=0 subcortical center",
            ),
        ]

        print("=" * 90)
        print("▶ PHASE 2: SYSTEMATIC PERTURBATION EXPERIMENTS ON ACTIVE MANIFOLD")
        print("=" * 90)

        for p_idx, p_cfg in enumerate(regimens, 1):
            print(f"\n⚡ EXPERIMENT [{p_idx}/{len(regimens)}]: {p_cfg.name}")
            print(f"   Modality    : {p_cfg.perturb_type} | Intensity: {p_cfg.intensity}")
            print(f"   Description : {p_cfg.description}")

            # Measure kinetic energy before
            with host.state_lock:
                e_before = (host.state_3d ** 2).mean().item()

            # Inject perturbation into background pacemaker
            host.inject_perturbation(p_cfg)
            time.sleep(0.5)

            # Measure energy after shock
            with host.state_lock:
                e_after = (host.state_3d ** 2).mean().item()
            delta_e = abs(e_after - e_before)

            # Test cognitive reasoning during active perturbation
            pert_scores = {}
            for probe in PROBE_SUITE:
                res = host.query(
                    probe["prompt"],
                    thinking_cycles=thinking_cycles,
                    max_answer_tokens=48,
                    active_perturbation=p_cfg,
                )
                ans = res["response"].lower()
                hit = any(t in ans for t in probe["exact_targets"])
                pert_scores[probe["id"]] = {
                    "pillar": probe["pillar"],
                    "hit": hit,
                    "response": res["response"],
                }

            pert_pass_rate = sum(1 for v in pert_scores.values() if v["hit"]) / len(PROBE_SUITE) * 100.0
            retention_rate = (pert_pass_rate / max(1e-5, clean_pass_rate)) * 100.0

            # Measure homeostatic relaxation time
            t_relax_start = time.time()
            relaxed = False
            for tick in range(15):
                time.sleep(0.2)
                with host.state_lock:
                    mean_div = sum(host.divergence_history[-5:]) / max(1, len(host.divergence_history[-5:]))
                if mean_div < 0.08:
                    relaxed = True
                    break
            tau_relax = (time.time() - t_relax_start) if relaxed else 3.0

            exp_record = {
                "name": p_cfg.name,
                "perturb_type": p_cfg.perturb_type,
                "intensity": p_cfg.intensity,
                "delta_kinetic_energy": round(delta_e, 5),
                "tau_relax_seconds": round(tau_relax, 2),
                "pass_rate_pct": pert_pass_rate,
                "retention_rate_pct": round(retention_rate, 1),
                "probes": pert_scores,
            }
            results["perturbations"].append(exp_record)

            print(f"   --> Energy Shift Delta E_k : {delta_e:.5f}")
            print(f"   --> Relaxation Latency     : {tau_relax:.2f}s")
            print(f"   --> Perturbed Pass Rate    : {pert_pass_rate:.1f}% (Retention: {retention_rate:.1f}%)")

            # Sample probe comparison
            sample_probe = PROBE_SUITE[0]  # Platypus
            clean_resp = clean_scores[sample_probe["id"]]["response"]
            pert_resp = pert_scores[sample_probe["id"]]["response"]
            print(f"   [Sample Response Comparison | {sample_probe['pillar']}]:")
            print(f"     Clean : \"{clean_resp}\"")
            print(f"     Shock : \"{pert_resp}\"")

        # 3. Final Summary Table
        print("\n" + "=" * 90)
        print("      PERTURBATION STRESS-TEST SUMMARY TABLE")
        print("=" * 90)
        print(f"{'Perturbation Modality':<45} | {'Delta E_k':<10} | {'tau_relax':<10} | {'Pass Rate':<10} | {'Retention'}")
        print("-" * 90)
        print(f"{'Clean Baseline':<45} | {'0.00000':<10} | {'0.00s':<10} | {clean_pass_rate:5.1f}%     | 100.0%")
        for exp in results["perturbations"]:
            name = exp["name"][:43]
            dek = f"{exp['delta_kinetic_energy']:.5f}"
            tau = f"{exp['tau_relax_seconds']:.2f}s"
            p_rate = f"{exp['pass_rate_pct']:5.1f}%"
            ret = f"{exp['retention_rate_pct']:5.1f}%"
            print(f"{name:<45} | {dek:<10} | {tau:<10} | {p_rate:<10} | {ret}")
        print("=" * 90 + "\n")

        # Save output JSON
        out_file = os.path.join(ROOT_DIR, "checkpoints/perturbation_stress_test_results.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Detailed stress-test results written to: {out_file}")

    finally:
        host.stop()

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Perturbation Stress Test on Living Organoid")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/multimodal_organoid_best.pt")
    parser.add_argument("--thinking_cycles", type=int, default=3)
    args = parser.parse_args()

    run_stress_test(checkpoint=args.checkpoint, thinking_cycles=args.thinking_cycles)


if __name__ == "__main__":
    main()
