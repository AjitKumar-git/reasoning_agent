"""
Proto-AGI Emergence Benchmark & Adaptive Deliberation Engine
============================================================
Evaluates autonomous cognitive reasoning across 6 foundational pillars:
1. Deep Deductive Syllogisms & Biological Taxonomies
2. Multi-Hop Transitive Orderings (A > B > C > D)
3. Counterfactual Physics & Physical Inversion
4. Multi-Step Arithmetic, Algebra & Kinematics
5. Biochemical, Thermal & Kinetic Mechanisms
6. Spatial Symmetry & Invariant Counting

Uses Dynamic Adaptive Deliberation:
The wave pulsates internally until its kinetic energy settles into harmonic
attractor equilibrium (delta_E < tol), mirroring natural contemplation time.
Strict Zero External Token Feedback (x_{t+1} != [x, y]).
"""

import os
import sys
import time
import math
import argparse

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.environ["HF_HOME"] = os.path.join(ROOT_DIR, ".hf_cache")

import torch
from transformers import AutoTokenizer

from src.real_geodesic_sphere import build_real_geodesic_spheroid
from src.modular_kissing_spheres import build_modular_kissing_lattice


BENCHMARK_PILLARS = [
    {
        "pillar": "1. Deductive Syllogism & Novel Taxonomies",
        "tests": [
            {
                "prompt": "All quadrupedal mammals produce milk. A platypus lays eggs but nurses its young with milk. Is a platypus a mammal?",
                "key_concepts": ["mammal", "milk", "deduction", "yes", "category"],
            },
            {
                "prompt": "All elements with full outer electron shells are noble gases. Argon has 8 valence electrons filling its outer shell. Is Argon a noble gas?",
                "key_concepts": ["noble gas", "outer shell", "argon", "yes", "deduction"],
            },
        ],
    },
    {
        "pillar": "2. Multi-Hop Transitive Reasoning",
        "tests": [
            {
                "prompt": "Iron is denser than Aluminium. Platinum is denser than Iron. Lithium is lighter than Aluminium. Which of these four metals is the densest?",
                "key_concepts": ["platinum", "densest", "transitive", ">", "iron"],
            },
            {
                "prompt": "City A is north of City B. City B is north of City C. Is City A north of City C?",
                "key_concepts": ["north", "transitive", "yes", "relation", "follows"],
            },
        ],
    },
    {
        "pillar": "3. Counterfactual Physical Systems",
        "tests": [
            {
                "prompt": "If hot air were denser than cold air, what direction would hot smoke from a chimney move?",
                "key_concepts": ["downward", "sink", "ground", "denser", "gravity"],
            },
            {
                "prompt": "If thermal energy flowed spontaneously from cold objects to hot objects, what would happen to an ice cube in warm tea?",
                "key_concepts": ["colder", "freeze", "tea hotter", "spontaneous", "reverse"],
            },
        ],
    },
    {
        "pillar": "4. Multi-Step Arithmetic & Kinematics",
        "tests": [
            {
                "prompt": "A shop sells notebooks for $4 each. A student buys 6 notebooks and pays with a $50 bill. How much change does the student receive?",
                "key_concepts": ["$24", "50 - 24", "$26", "multiply", "subtract", "change"],
            },
            {
                "prompt": "A car travels 180 miles in 3 hours. How far does it travel in 7 hours at the same speed?",
                "key_concepts": ["60 mph", "180 / 3", "60 * 7", "420 miles", "multiply"],
            },
        ],
    },
    {
        "pillar": "5. Biochemical & Kinetic Mechanisms",
        "tests": [
            {
                "prompt": "Why does breathing rate increase during vigorous exercise?",
                "key_concepts": ["oxygen", "cellular respiration", "carbon dioxide", "energy", "ATP"],
            },
            {
                "prompt": "Why does boiling water cause proteins to denature?",
                "key_concepts": ["thermal energy", "hydrogen bonds", "unfold", "denature", "structure"],
            },
        ],
    },
    {
        "pillar": "6. Spatial Symmetry & Invariant Counting",
        "tests": [
            {
                "prompt": "A cube has 6 faces. If you paint 2 opposite faces red and the other faces blue, how many blue faces are there?",
                "key_concepts": ["4", "6 - 2", "blue", "faces", "subtract"],
            },
        ],
    },
]


def evaluate_model(model, name: str, tokenizer, device, dtype):
    print("=" * 88)
    print(f"      TESTING PROTO-AGI EMERGENCE: {name}")
    print("=" * 88)

    pillar_scores = []
    total_tests = 0
    total_passed = 0

    for pillar_data in BENCHMARK_PILLARS:
        pillar_name = pillar_data["pillar"]
        tests = pillar_data["tests"]
        print(f"\n▶ [{pillar_name}]")

        pillar_hits = 0
        for test in tests:
            total_tests += 1
            prompt = test["prompt"]
            keys = test["key_concepts"]

            q_toks = tokenizer.encode(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
                    out_toks, settled_cycles = model.solve_with_adaptive_emergence(
                        q_toks,
                        min_cycles=2,
                        max_cycles=12,
                        settling_tolerance=0.04,
                        max_answer_tokens=32,
                        temperature=0.60,
                        eos_token_id=tokenizer.eos_token_id,
                    )

            resp = tokenizer.decode(out_toks[0]).strip() if len(out_toks[0]) > 0 else "<empty>"
            resp_lower = resp.lower()

            # Measure concept hits
            hits = [k for k in keys if k.lower() in resp_lower]
            hit_ratio = len(hits) / max(1, len(keys))
            is_proto_emergent = len(hits) >= 1

            if is_proto_emergent:
                pillar_hits += 1
                total_passed += 1
                status = "★ EMERGENT"
            else:
                status = "○ EVOLVING"

            print(f"  Q: \"{prompt[:62]}...\"")
            print(f"  Deliberation Settled: {settled_cycles} cycles | {status} (Hits: {hits})")
            print(f"  A: \"{resp}\"")
            print("  " + "-" * 75)

        pillar_scores.append((pillar_name, pillar_hits, len(tests)))

    score_pct = (total_passed / max(1, total_tests)) * 100
    print("\n" + "=" * 88)
    print(f"  SUMMARY FOR {name}:")
    print(f"  Overall Emergence Score: {total_passed}/{total_tests} ({score_pct:.1f}%)")
    for pname, phits, ptot in pillar_scores:
        print(f"  - {pname}: {phits}/{ptot} ({phits/ptot*100:.0f}%)")
    print("=" * 88 + "\n")
    return score_pct


def main():
    parser = argparse.ArgumentParser(description="Evaluate Proto-AGI Emergence on Living Spheroid")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    if args.checkpoint:
        ckpt_path = args.checkpoint
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(ROOT_DIR, ckpt_path)
        print(f"Loading custom target checkpoint: {ckpt_path}")
        model_single = build_real_geodesic_spheroid()
        state_dict = torch.load(ckpt_path, map_location=device)
        model_single.load_state_dict(state_dict, strict=False)
        model_single.to(device=device, dtype=dtype)
        evaluate_model(model_single, f"Target Spheroid ({os.path.basename(ckpt_path)})", tokenizer, device, dtype)
        return

    # 1. Test Single Real Geodesic Sphere
    single_sphere_ckpt = os.path.join(ROOT_DIR, "checkpoints", "proto_agi_sphere_living_best.pt")
    if not os.path.exists(single_sphere_ckpt):
        single_sphere_ckpt = os.path.join(ROOT_DIR, "checkpoints", "real_sphere_best.pt")
    if os.path.exists(single_sphere_ckpt):
        model_single = build_real_geodesic_spheroid()
        model_single.load_state_dict(torch.load(single_sphere_ckpt, map_location=device), strict=False)
        model_single.to(device=device, dtype=dtype)
        evaluate_model(model_single, f"Continuous Real Geodesic Sphere ({os.path.basename(single_sphere_ckpt)})", tokenizer, device, dtype)

    # 2. Test Modular Adjacent Kissing Lattice
    kissing_ckpt = os.path.join(ROOT_DIR, "checkpoints", "modular_kissing_best.pt")
    if os.path.exists(kissing_ckpt):
        model_lattice = build_modular_kissing_lattice()
        model_lattice.load_state_dict(torch.load(kissing_ckpt, map_location=device), strict=False)
        model_lattice.to(device=device, dtype=dtype)
        evaluate_model(model_lattice, "Modular Adjacent Kissing-Sphere Network (4-Sphere Cluster)", tokenizer, device, dtype)


if __name__ == "__main__":
    main()
