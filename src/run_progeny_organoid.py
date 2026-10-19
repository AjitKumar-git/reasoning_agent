"""
Autonomous Progeny Organoid Runtime Engine
==========================================
Bootstraps an exported biological progeny organoid colony:
1. Loads progeny_manifest.json and colony_genome.pt.
2. Runs active living resting-state pulsation with continuous micro-adjustments.
3. Processes reasoning queries.
4. Triggers secondary biological mitosis when cognitive workload saturates,
   growing the colony into higher-generation organoids in 3D Riemannian space.
"""

import os
import sys
import time
import json
import argparse
import torch
from transformers import AutoTokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, ".hf_cache"))

from src.self_replicating_spheroid import build_self_replicating_colony, SelfReplicatingColony


def run_progeny_colony(manifest_path: str = "checkpoints/progeny/progeny_manifest.json"):
    full_manifest_path = os.path.join(ROOT_DIR, manifest_path)
    if not os.path.exists(full_manifest_path):
        print(f"Error: Manifest {full_manifest_path} not found.")
        return

    with open(full_manifest_path, "r") as f:
        manifest = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    print("=" * 86)
    print("      AUTONOMOUS PROGENY ORGANOID COLONY: RUNTIME ENGINE                    ")
    print("=" * 86)
    print(f"Colony Size        : {manifest['colony_size']} Organoids")
    print(f"Total Replications : {manifest['total_replications']}")
    print(f"Sensory Root       : {manifest['sensory_root']}")
    print(f"Motor Root         : {manifest['motor_root']}")
    print(f"Contact Patches    : {len(manifest['contact_patches'])} welded tangent disks")
    print(f"Device             : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 86)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Reconstruct fully welded multi-sphere colony from manifest
    colony = SelfReplicatingColony.from_manifest(full_manifest_path, device=device, dtype=dtype)
    colony.eval()

    test_queries = [
        "All birds have feathers. A robin is a bird. Does a robin have feathers?",
        "If 5 pencils cost $15, how much do 8 pencils cost?",
        "Iron is denser than Aluminium. Platinum is denser than Iron. Lithium is lighter than Aluminium. Which is densest?",
    ]

    print("\n--- Testing Colony Reasoning Across Replicated Touching Spheres ---\n")
    for q in test_queries:
        print(f"Q: \"{q}\"")
        q_toks = tokenizer.encode(q, return_tensors="pt").to(device)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
                out = colony.solve_with_active_thinking(
                    q_toks, thinking_cycles=3, max_answer_tokens=28, temperature=0.65
                )

        ans = tokenizer.decode(out[0]).strip() if len(out[0]) > 0 else "<empty>"
        print(f"A: \"{ans}\"")
        print("-" * 75)

    # Test autonomous secondary mitosis
    print("\nInitiating Secondary Mitosis on latest generation daughter...")
    latest_id = list(manifest["organoids"].keys())[-1]
    daughter_id, _ = colony.replicate_spheroid(latest_id)
    print(f"New Daughter Organoid {daughter_id} welded to colony! New size: {len(colony.organoids)}")

    updated_manifest = colony.export_autonomous_progeny()
    print(f"Updated Progeny Manifest saved to {updated_manifest}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="checkpoints/progeny/progeny_manifest.json")
    args = parser.parse_args()

    run_progeny_colony(manifest_path=args.manifest)
