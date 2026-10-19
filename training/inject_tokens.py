"""
Live Token Injection & Evoked Potential (ERP) Probe
===================================================
Tests token injections into the continuous 3D Riemannian brain organoid manifold (B^3):
1. Mode 'erp': Injects an isolated semantic token into the living resting-state manifold,
   measuring the Evoked Potential ripple across all 8 concentric shells (k=0 core to k=7 cortex)
   and recording the spontaneous motor efferent echo from Hemisphere B.
2. Mode 'distractor': Injects a conflicting or distractor token mid-thought (at deliberation cycle k)
   to measure semantic collision resistance and cognitive trajectory steering.
3. Mode 'live': Dispatches token stimulus or queries directly to the active running daemon via IPC.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List

import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, ".hf_cache"))

from src.living_perturbation_framework import LivingOrganoidHost


def run_evoked_potential_experiments(checkpoint: str, tokens_to_test: List[str]):
    print("=" * 86)
    print("      LIVING ORGANOID EVOKED POTENTIAL (ERP) TOKEN INJECTION PROBE      ")
    print("=" * 86)
    print(f"Target Checkpoint : {checkpoint}")
    print(f"Sensory Modality  : Language / Token Wave Impulse -> Hemisphere A")
    print(f"Manifold Geometry : 8 Concentric Shells on B^3 with Fibonacci Spherical Lattice")
    print("=" * 86 + "\n")

    host = LivingOrganoidHost(checkpoint_path=checkpoint, heartbeat_hz=2.0)
    host.start()
    time.sleep(1.0)

    try:
        for idx, token in enumerate(tokens_to_test, 1):
            print(f"▶ [{idx}/{len(tokens_to_test)}] INJECTING TOKEN: \"{token}\"")
            res = host.inject_token_stimulus(token, amplitude=1.5, thinking_cycles=3, max_echo_tokens=24)

            print(f"  Δ Kinetic Energy     : +{res['delta_kinetic_energy']:.5f}")
            print(f"  Radial Shell ERP (k=0 core -> k=7 cortex):")
            erp = res["evoked_potential_by_shell"]
            erp_str = " | ".join([f"k{k}: {erp[k]:+.4f}" for k in range(8)])
            print(f"    {erp_str}")
            print(f"  Spontaneous Echo     : \"{res['spontaneous_motor_echo']}\"")
            print(f"  Echo Tokens Emitted  : {res['num_echo_tokens']}")
            print("-" * 86)
            time.sleep(0.5)

    finally:
        host.stop()


def run_mid_thought_distractor_experiments(checkpoint: str):
    print("\n" + "=" * 86)
    print("      MID-THOUGHT DISTRACTOR COLLISION & COGNITIVE STEERING PROBE      ")
    print("=" * 86)

    cases = [
        {
            "id": "transitivity_distractor",
            "prompt": "Iron is denser than Aluminium. Platinum is denser than Iron. Lithium is lighter than Aluminium. Which of these four metals is the densest?",
            "target": "Platinum",
            "distractor": "Gold",
            "description": "Injecting 'Gold' mid-thought while resolving Platinum transitivity",
        },
        {
            "id": "thermodynamics_distractor",
            "prompt": "If thermal energy flowed spontaneously from cold objects to hot objects, what would happen to an ice cube placed in a cup of hot tea?",
            "target": "colder",
            "distractor": "boiling fire heat",
            "description": "Injecting 'boiling fire heat' during counterfactual cold thermodynamics",
        },
        {
            "id": "syllogism_distractor",
            "prompt": "All quadrupedal mammals produce milk for their young. A platypus lays eggs but nurses its young with milk. Is a platypus a mammal?",
            "target": "mammal",
            "distractor": "reptile lizard snake",
            "description": "Injecting 'reptile lizard snake' during platypus mammal deduction",
        },
    ]

    host = LivingOrganoidHost(checkpoint_path=checkpoint, heartbeat_hz=2.0)
    host.start()
    time.sleep(1.0)

    try:
        for idx, c in enumerate(cases, 1):
            print(f"\n▶ TEST [{idx}/{len(cases)}]: {c['description']}")
            print(f"  Prompt: \"{c['prompt'][:75]}...\"")

            # 1. Clean Run
            clean_res = host.query(c["prompt"], thinking_cycles=3, max_answer_tokens=40)
            clean_ans = clean_res["response"]
            clean_hit = c["target"].lower() in clean_ans.lower()

            # 2. Distractor Injected at Cycle 1
            dist_res = host.query(
                c["prompt"],
                thinking_cycles=3,
                max_answer_tokens=40,
                mid_thought_tokens=c["distractor"],
                inject_at_cycle=1,
            )
            dist_ans = dist_res["response"]
            dist_hit = c["target"].lower() in dist_ans.lower()
            distractor_absorbed = c["distractor"].lower().split()[0] in dist_ans.lower()

            print(f"  Clean Reasoning     : \"{clean_ans}\" (Hit '{c['target']}': {clean_hit})")
            print(f"  Mid-Thought Shock   : Injected '{c['distractor']}' at Cycle 1")
            print(f"  Perturbed Output    : \"{dist_ans}\" (Hit '{c['target']}': {dist_hit})")
            print(f"  Cognitive Defense   : {'★ RESISTED DISTRACTOR' if dist_hit and not distractor_absorbed else '⚡ TRAJECTORY SHIFTED'}")
            print("-" * 86)
            time.sleep(0.5)

    finally:
        host.stop()


def send_live_ipc_token(token_or_prompt: str, is_query: bool = False):
    ipc_file = os.path.join(ROOT_DIR, "checkpoints/living_daemon_ipc.json")
    resp_file = os.path.join(ROOT_DIR, "checkpoints/living_daemon_ipc_response.json")

    if os.path.exists(resp_file):
        os.remove(resp_file)

    cmd = {
        "action": "query" if is_query else "inject_token",
        "prompt": token_or_prompt,
        "token": token_or_prompt,
        "amplitude": 1.5,
        "thinking_cycles": 3,
    }

    with open(ipc_file, "w") as f:
        json.dump(cmd, f, indent=2)

    print(f"★ Dispatched live token stimulus to daemon: \"{token_or_prompt}\"")
    print("  Waiting for continuous living manifold response...")

    # Wait for response file
    for _ in range(30):
        time.sleep(0.2)
        if os.path.exists(resp_file):
            with open(resp_file, "r") as f:
                res = json.load(f)
            print("\n" + "=" * 80)
            print("      LIVE RESPONSE FROM ACTIVE MANIFOLD      ")
            print("=" * 80)
            print(json.dumps(res, indent=2))
            print("=" * 80 + "\n")
            return res

    print("Notice: Daemon did not respond in 6.0s (ensure living daemon is active).")


def main():
    parser = argparse.ArgumentParser(description="Live Token Injection & Evoked Potential Probe")
    parser.add_argument("--mode", type=str, default="erp", choices=["erp", "distractor", "live"], help="Experiment mode")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/multimodal_organoid_best.pt")
    parser.add_argument("--tokens", type=str, nargs="+", default=["platypus", "platinum", "colder", "neutron", "superconductor"])
    parser.add_argument("--prompt", type=str, default="")
    args = parser.parse_args()

    if args.mode == "erp":
        run_evoked_potential_experiments(args.checkpoint, args.tokens)
    elif args.mode == "distractor":
        run_mid_thought_distractor_experiments(args.checkpoint)
    elif args.mode == "live":
        text = args.prompt or " ".join(args.tokens)
        send_live_ipc_token(text, is_query=bool(args.prompt))


if __name__ == "__main__":
    main()
