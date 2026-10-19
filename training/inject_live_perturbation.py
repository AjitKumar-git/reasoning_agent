"""
Live Perturbation Injector CLI
==============================
Sends real-time perturbation instructions to the active living daemon via IPC:
Supports:
- thermal: Gaussian thermal noise on manifold
- astro_shock: Astrocytic Ca2+ / glutamate surge
- columnar_lesion: Geodesic inactivation patch on S^2
- pac_jitter: Theta-gamma carrier clock desynchronization
- corollary_ablation: Dampen or sever motor-to-core efference copy
- core_pulse: Radial energy pulse into deep subcortical core
"""

import os
import sys
import json
import argparse

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IPC_FILE = os.path.join(ROOT_DIR, "checkpoints/living_daemon_ipc.json")


def main():
    parser = argparse.ArgumentParser(description="Inject Live Perturbation into Living Organoid")
    parser.add_argument("--type", type=str, required=True, choices=["thermal", "astro_shock", "columnar_lesion", "thalamic_noise", "pac_jitter", "corollary_ablation", "core_pulse"], help="Perturbation type")
    parser.add_argument("--intensity", type=float, default=0.10, help="Shock intensity / scale")
    parser.add_argument("--duration", type=int, default=5, help="Duration in ticks")
    parser.add_argument("--shells", type=int, nargs="+", default=None, help="Target shells (e.g. 0 1 for core)")
    parser.add_argument("--radius_deg", type=float, default=30.0, help="Lesion angular radius in degrees")
    parser.add_argument("--name", type=str, default="", help="Custom descriptive name")
    args = parser.parse_args()

    name = args.name or f"Live {args.type.replace('_', ' ').title()} ({args.intensity})"

    cmd = {
        "action": "perturb",
        "name": name,
        "type": args.type,
        "intensity": args.intensity,
        "duration": args.duration,
        "shells": args.shells,
        "radius_deg": args.radius_deg,
    }

    os.makedirs(os.path.dirname(IPC_FILE), exist_ok=True)
    with open(IPC_FILE, "w") as f:
        json.dump(cmd, f, indent=2)

    print(f"★ Perturbation command dispatched to: {IPC_FILE}")
    print(f"  Name      : {name}")
    print(f"  Type      : {args.type}")
    print(f"  Intensity : {args.intensity}")
    print(f"  Duration  : {args.duration} ticks")


if __name__ == "__main__":
    main()
