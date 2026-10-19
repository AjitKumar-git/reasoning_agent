"""
Test the Running Living Model
=============================
Directly probes and verifies the currently running living spheroid daemon:
1. Inspects live background process telemetry (uptime, ticks, kinetic energy, VRAM).
2. Tests the latest state-of-the-art model (MultimodalOrganoidCore / Tightened Organoid).
3. Executes live zero-feedback cognitive reasoning probes across diverse domains.
"""

import os
import sys
import time
import json
import argparse
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.environ["HF_HOME"] = os.path.join(ROOT_DIR, ".hf_cache")

import torch
from transformers import AutoTokenizer

from src.living_perturbation_framework import LivingOrganoidHost, PerturbationConfig


PROBE_QUESTIONS = [
    {
        "category": "1. Deductive Syllogism",
        "prompt": "All quadrupedal mammals produce milk for their young. A platypus lays eggs but nurses its young with milk. Is a platypus a mammal? Answer yes or no and state why.",
        "target": "mammal",
    },
    {
        "category": "2. Multi-Hop Transitivity",
        "prompt": "Iron is denser than Aluminium. Platinum is denser than Iron. Lithium is lighter than Aluminium. Which of these four metals is the densest?",
        "target": "platinum",
    },
    {
        "category": "3. Counterfactual Physics",
        "prompt": "If hot air were denser than cold air, in what direction would hot smoke from a chimney move? State the direction clearly.",
        "target": ["downward", "down", "sink", "ground"],
    },
    {
        "category": "4. Multi-Step Arithmetic",
        "prompt": "Maya buys 6 notebooks priced at $4 each. Maya pays with a $50 bill. Exactly how much change does Maya receive?",
        "target": ["26", "$26"],
    },
    {
        "category": "5. Spatial ARC-AGI",
        "prompt": "A geometric cube has 6 faces. If you paint exactly 2 opposite faces red and all the remaining faces blue, exactly how many blue faces are there?",
        "target": ["4", "four"],
    },
    {
        "category": "6. Matrix Spatial Symmetry",
        "prompt": "A 3x3 matrix has rows ['. # .', '. # .', '. # .'] with a vertical line in column 2. What happens to this vertical line when the matrix is reflected horizontally?",
        "target": ["unchanged", "same", "column 2"],
    },
]


def check_running_daemon(status_file: str = "checkpoints/living_perturbation_status.json"):
    abs_path = os.path.join(ROOT_DIR, status_file)
    if not os.path.exists(abs_path):
        abs_path = os.path.join(ROOT_DIR, "checkpoints/living_daemon_status.json")
    print("=" * 80)
    print("      CHECKING ACTIVE BACKGROUND LIVING DAEMON TELEMETRY      ")
    print("=" * 80)
    if not os.path.exists(abs_path):
        print(f"Warning: {status_file} not found. No daemon status logged yet.")
        return False

    with open(abs_path, "r") as f:
        status = json.load(f)

    now = time.time()
    time_diff = now - status.get("timestamp", 0)
    is_fresh = time_diff < 15.0

    print(f"Daemon Alive Status  : {'★ ACTIVE & RUNNING' if is_fresh and status.get('is_alive') else '○ INACTIVE/STALE'}")
    print(f"Uptime               : {status.get('uptime_seconds', 0)/3600:.2f} hours ({status.get('uptime_seconds', 0):.1f}s)")
    print(f"Heartbeat Ticks      : {status.get('ticks', 0):,}")
    print(f"Frequency            : {status.get('frequency_hz', 0)} Hz")
    print(f"Mean Kinetic Energy  : {status.get('mean_kinetic_energy', 0):.5f} (Homeostatic E_0 ~ 0.38)")
    print(f"Total Manifold Norm  : {status.get('total_norm', 0):.2f}")
    print(f"VRAM Allocated       : {status.get('vram_allocated_mb', 0):.1f} MB")
    print(f"Last Heartbeat Sync  : {time_diff:.1f}s ago")
    print("=" * 80 + "\n")
    return True


def test_live_inference(checkpoint: str, thinking_cycles: int = 3):
    print("=" * 80)
    print("      TESTING LIVE COGNITIVE REASONING ON RUNNING MANIFOLD      ")
    print("=" * 80)
    print(f"Checkpoint : {checkpoint}")
    print(f"Inference  : Living Persistent Manifold (Zero External Token Feedback)")
    print("=" * 80)

    host = LivingOrganoidHost(checkpoint_path=checkpoint, heartbeat_hz=2.0)
    host.start()

    time.sleep(1.0)
    passed_probes = 0

    try:
        for idx, item in enumerate(PROBE_QUESTIONS, 1):
            category = item["category"]
            prompt = item["prompt"]
            target = item["target"]

            t0 = time.time()
            res = host.query(
                prompt,
                thinking_cycles=thinking_cycles,
                max_answer_tokens=48,
                temperature=0.5,
                fresh_state=True,
            )
            latency_ms = (time.time() - t0) * 1000

            ans = res["response"]
            target_list = target if isinstance(target, list) else [target]
            is_hit = any(t.lower() in ans.lower() for t in target_list)
            if is_hit:
                passed_probes += 1
                tag = "★ PASS"
            else:
                tag = "○ EVOLVE"

            print(f"\n▶ [{idx}/{len(PROBE_QUESTIONS)}] {category} -> {tag} ({latency_ms:.1f} ms)")
            print(f"  Q: \"{prompt[:70]}...\"")
            print(f"  A: \"{ans}\"")
            print(f"  Target: '{target}' | Hit: {is_hit}")

        pct = (passed_probes / len(PROBE_QUESTIONS)) * 100.0
        print("\n" + "=" * 80)
        print(f"Live Probing Complete: {passed_probes}/{len(PROBE_QUESTIONS)} Target Probes Hit ({pct:.1f}%)")
        print("=" * 80 + "\n")

    finally:
        host.stop()


def main():
    parser = argparse.ArgumentParser(description="Test Running Living Model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/multimodal_organoid_best.pt")
    parser.add_argument("--thinking_cycles", type=int, default=3)
    args = parser.parse_args()

    check_running_daemon()
    test_live_inference(checkpoint=args.checkpoint, thinking_cycles=args.thinking_cycles)


if __name__ == "__main__":
    main()
