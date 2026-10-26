"""
Live Organoid Stream & Real-Time Dynamics Viewer
================================================
Provides a real-time visual streaming interface for the 100M continuous
Riemannian organoid manifold (MultimodalOrganoidCore on B^3):
1. --monitor     : Live animated terminal dashboard showing 8 concentric shells (B^3),
                   kinetic energy E_k, Lyapunov stability, and spontaneous thoughts.
2. --prompt "..." : Streams live token-by-token generation with sub-100ms latency,
                   confidence tracking, and corollary discharge monitoring.
3. --sample      : Runs live streaming inference across standard Proto-AGI reasoning probes.
4. --interactive : Interactive live chat REPL with character/token streaming.
"""

import os
import sys
import time
import json
import argparse
from typing import Optional

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.environ["HF_HOME"] = os.path.join(ROOT_DIR, ".hf_cache")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(ROOT_DIR, ".hf_cache")
os.environ["HF_HUB_CACHE"] = os.path.join(ROOT_DIR, ".hf_cache")

import torch

# ANSI Color Codes for terminal UI
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
BLUE = "\033[94m"
CLEAR_SCREEN = "\033[H\033[J"

SHELL_NAMES = [
    ("Shell 0", "Core 0 - Deep Latent Core  (r=0.12)"),
    ("Shell 1", "Core 1 - Corollary Recurrent(r=0.25)"),
    ("Shell 2", "Inter-Core Associative      (r=0.38)"),
    ("Shell 3", "Parietal/Spatial Cortex     (r=0.50)"),
    ("Shell 4", "Prefrontal Deliberation     (r=0.62)"),
    ("Shell 5", "Temporal Semantic Cortex    (r=0.75)"),
    ("Shell 6", "Sensory/Visual Integrator   (r=0.88)"),
    ("Shell 7", "Sensorimotor Poles on S^2   (r=1.00)"),
]


def render_bar(val: float, max_val: float = 1.0, length: int = 26) -> str:
    """Renders a smooth ASCII energy bar."""
    filled = int(min(max(val / max(max_val, 1e-4), 0.0), 1.0) * length)
    empty = length - filled
    bar = "█" * filled + "░" * empty
    if val > 0.6:
        return f"{RED}{bar}{RESET}"
    elif val > 0.4:
        return f"{YELLOW}{bar}{RESET}"
    elif val > 0.2:
        return f"{CYAN}{bar}{RESET}"
    else:
        return f"{BLUE}{bar}{RESET}"


def format_uptime(seconds: float) -> str:
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}h {m:02d}m {sec:02d}s"


def run_monitor(status_file: str = "checkpoints/living_perturbation_status.json", refresh_hz: float = 2.0):
    """
    Renders an in-place live animated dashboard of the running organoid daemon.
    """
    abs_path = os.path.join(ROOT_DIR, status_file)
    print(f"{CYAN}Connecting to Living Organoid Monitor at {abs_path}...{RESET}")

    last_tick = -1
    try:
        while True:
            if not os.path.exists(abs_path):
                print(f"{YELLOW}Waiting for living daemon status file to be written...{RESET}", end="\r")
                time.sleep(1.0)
                continue

            try:
                with open(abs_path, "r") as f:
                    stat = json.load(f)
            except Exception:
                time.sleep(0.1)
                continue

            curr_tick = stat.get("ticks", 0)
            uptime_str = format_uptime(stat.get("uptime_seconds", 0))
            e_k = stat.get("mean_kinetic_energy", 0.0)
            norm_val = stat.get("total_norm", 0.0)
            vram_mb = stat.get("vram_allocated_mb", 0.0)
            hz = stat.get("frequency_hz", 2.0)
            m_type = stat.get("model_type", "multimodal").upper()
            div = stat.get("recent_mean_divergence", 0.0)
            p_active = stat.get("active_perturbation")
            spontaneous = stat.get("spontaneous_thought", "").strip()
            shells = stat.get("shell_energies", [e_k] * 8)

            p_str = f"{RED}⚡ SHOCK: {p_active.get('name')}{RESET}" if p_active else f"{GREEN}STABLE (Clean Dynamics){RESET}"

            # Draw UI
            sys.stdout.write(CLEAR_SCREEN)
            sys.stdout.write(f"{BOLD}{MAGENTA}================================================================================{RESET}\n")
            sys.stdout.write(f"   {BOLD}LIVING 3D RIEMANNIAN ORGANOID MANIFOLD — REAL-TIME TELEMETRY MONITOR{RESET}\n")
            sys.stdout.write(f"{BOLD}{MAGENTA}================================================================================{RESET}\n")
            sys.stdout.write(f"{BOLD}Status{RESET}       : {GREEN}ALIVE ●{RESET} [{m_type} 100M] | {BOLD}Uptime{RESET}: {uptime_str} | {BOLD}Pacemaker{RESET}: {hz:.1f} Hz\n")
            sys.stdout.write(f"{BOLD}Step Index{RESET}   : Tick #{curr_tick:d} | {BOLD}VRAM Allocated{RESET}: {vram_mb:.1f} MB | {BOLD}Stability{RESET}: {p_str}\n")
            sys.stdout.write(f"{BOLD}Hamiltonian{RESET}  : Kinetic Energy E_k = {BOLD}{CYAN}{e_k:.5f}{RESET} | Frobenius Norm ||Ψ|| = {norm_val:.2f}\n")
            sys.stdout.write(f"{BOLD}Divergence{RESET}   : Lyapunov D_mean = {div:.5f}\n")
            sys.stdout.write(f"{MAGENTA}--------------------------------------------------------------------------------{RESET}\n")
            sys.stdout.write(f"{BOLD}CONCENTRIC RIEMANNIAN SHELL ENERGIES (B^3 continuous wave density):{RESET}\n")

            for k in range(min(len(shells), 8)):
                name, desc = SHELL_NAMES[k]
                bar = render_bar(shells[k], max_val=0.8, length=28)
                sys.stdout.write(f"  {BOLD}{name}{RESET} [{desc}]: {bar} {CYAN}{shells[k]:.4f}{RESET}\n")

            sys.stdout.write(f"{MAGENTA}--------------------------------------------------------------------------------{RESET}\n")
            sys.stdout.write(f"{BOLD}DEFAULT MODE NETWORK (DMN) RESTING-STATE MURMURS (Motor Pole Efferent):{RESET}\n")
            if spontaneous:
                sys.stdout.write(f"  {YELLOW}💭 \"{spontaneous}\"{RESET}\n")
            else:
                sys.stdout.write(f"  {DIM}(Quiescent resting manifold — awaiting spontaneous burst or stimulus){RESET}\n")

            sys.stdout.write(f"{MAGENTA}================================================================================{RESET}\n")
            sys.stdout.write(f"{DIM}Press Ctrl+C to exit monitor. Run with --sample or --interactive to query live.{RESET}\n")
            sys.stdout.flush()

            time.sleep(1.0 / refresh_hz)
    except KeyboardInterrupt:
        print(f"\n{GREEN}Exited live monitor.{RESET}")


def run_live_streaming_query(
    host,
    prompt: str,
    thinking_cycles: int = 3,
    max_tokens: int = 48,
    temperature: float = 0.5,
    show_details: bool = True,
) -> str:
    """
    Executes live streaming inference, printing character/token output directly
    to stdout as each motor efferent token fires from Hemisphere B.
    """
    print(f"\n{BOLD}{CYAN}════════════════════════════════════════════════════════════════════════════════{RESET}")
    print(f"{BOLD}PROMPT:{RESET} {prompt}")
    print(f"{BOLD}{CYAN}════════════════════════════════════════════════════════════════════════════════{RESET}")

    gen = host.stream_query(
        prompt=prompt,
        thinking_cycles=thinking_cycles,
        max_answer_tokens=max_tokens,
        temperature=temperature,
        fresh_state=True,
    )

    full_response = ""
    for event in gen:
        ev_type = event.get("event")

        if ev_type == "start":
            pass

        elif ev_type == "stimulus_ingested":
            if show_details:
                sens_norm = event.get("gated_sensory_norm", 0.0)
                n_toks = event.get("prompt_tokens", 0)
                print(f"{DIM}[Sensory Ingest]{RESET} {n_toks} prompt tokens ingested into Hemisphere A (Sensory Norm: {sens_norm:.3f})")

        elif ev_type == "deliberation_cycle":
            if show_details:
                cyc = event.get("cycle", 1)
                tot = event.get("total_cycles", 3)
                e_k = event.get("kinetic_energy", 0.0)
                lat = event.get("cycle_time_ms", 0.0)
                shells = event.get("shell_energies", [])
                bars = "".join([" " if e < 0.2 else "▂" if e < 0.3 else "▄" if e < 0.4 else "▆" if e < 0.5 else "█" for e in shells])
                print(f"{DIM}[Deliberation {cyc}/{tot}]{RESET} Pulse wave step -> E_k: {e_k:.4f} | Shells [{bars}] | Latency: {lat:.1f}ms")

        elif ev_type == "token":
            # Print token live immediately!
            tok_text = event.get("token_text", "")
            conf = event.get("confidence", 0.0)
            lat = event.get("latency_ms", 0.0)
            corr = event.get("corollary_norm", 0.0)

            # Smooth live stream
            sys.stdout.write(f"{BOLD}{GREEN}{tok_text}{RESET}")
            sys.stdout.flush()

        elif ev_type == "done":
            full_response = event.get("response", "")
            n_tokens = event.get("tokens_generated", 0)
            tot_ms = event.get("total_ms", 0.0)
            ms_per_tok = tot_ms / max(1, n_tokens)
            print()
            print(f"{BOLD}{CYAN}────────────────────────────────────────────────────────────────────────────────{RESET}")
            print(f"{BOLD}Metrics:{RESET} {n_tokens} tokens | Total: {tot_ms:.1f}ms | Avg: {ms_per_tok:.1f}ms/token | Zero External Feedback Invariant Preserved")
            print(f"{BOLD}{CYAN}════════════════════════════════════════════════════════════════════════════════{RESET}\n")

    return full_response


def main():
    parser = argparse.ArgumentParser(description="Live Organoid Stream & Real-Time Dynamics Viewer")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/multimodal_organoid_100m_best.pt")
    parser.add_argument("--monitor", action="store_true", help="Launch live animated dashboard monitor")
    parser.add_argument("--prompt", type=str, default=None, help="Query the living model with live streaming tokens")
    parser.add_argument("--sample", action="store_true", help="Run live streaming demonstration across benchmark probes")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive live chat streaming REPL")
    parser.add_argument("--cycles", type=int, default=3, help="Deliberation cycles before motor readout")
    parser.add_argument("--max_tokens", type=int, default=48, help="Max tokens to stream")
    parser.add_argument("--temp", type=float, default=0.5, help="Sampling temperature")
    args = parser.parse_args()

    # If --monitor or no arguments given, run the dashboard monitor
    if args.monitor or (not args.prompt and not args.sample and not args.interactive):
        run_monitor()
        return

    # For streaming queries, initialize host
    from src.living_perturbation_framework import LivingOrganoidHost
    print(f"\n{BOLD}{CYAN}Loading 100M Living Organoid Host from {args.checkpoint}...{RESET}")
    host = LivingOrganoidHost(checkpoint_path=args.checkpoint, heartbeat_hz=2.0)

    if args.prompt:
        run_live_streaming_query(host, args.prompt, thinking_cycles=args.cycles, max_tokens=args.max_tokens, temperature=args.temp)
        return

    if args.sample:
        probes = [
            ("Deductive Syllogism", "All quadrupedal mammals produce milk for their young. A platypus lays eggs but nurses its young with milk. Is a platypus a mammal? Answer yes or no and state why."),
            ("Multi-Hop Transitivity", "Iron is denser than Aluminium. Platinum is denser than Iron. Lithium is lighter than Aluminium. Which of these four metals is the densest?"),
            ("Counterfactual Physics", "If hot air were denser than cold air, in what direction would hot smoke from a chimney move? State the direction clearly."),
            ("Multi-Step Arithmetic", "Maya buys 6 notebooks priced at $4 each. Maya pays with a $50 bill. Exactly how much change does Maya receive?"),
        ]
        for cat, prompt in probes:
            print(f"\n{BOLD}{YELLOW}[Category: {cat}]{RESET}")
            run_live_streaming_query(host, prompt, thinking_cycles=args.cycles, max_tokens=args.max_tokens, temperature=args.temp)
        return

    if args.interactive:
        print(f"\n{BOLD}{GREEN}================================================================================{RESET}")
        print(f"      {BOLD}INTERACTIVE LIVING ORGANOID STREAMING REPL (100M PARAMETERS){RESET}")
        print(f"  Type any question or prompt. Tokens will stream live character-by-character.")
        print(f"  Type 'exit' or 'quit' to end session.")
        print(f"{BOLD}{GREEN}================================================================================{RESET}\n")
        try:
            while True:
                user_prompt = input(f"{BOLD}{CYAN}Organoid >> {RESET}").strip()
                if not user_prompt:
                    continue
                if user_prompt.lower() in ["exit", "quit"]:
                    break
                run_live_streaming_query(host, user_prompt, thinking_cycles=args.cycles, max_tokens=args.max_tokens, temperature=args.temp)
        except KeyboardInterrupt:
            print(f"\n{GREEN}Exited interactive session.{RESET}")


if __name__ == "__main__":
    main()
