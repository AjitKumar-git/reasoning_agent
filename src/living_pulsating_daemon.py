"""
Continuous Living Pulsating Spheroid Daemon
===========================================
Maintains an active, continuous resting-state biological manifold in the background:
1. Endogenous Theta/Alpha Pacemaker Waves continuously circulate in the 3D unit ball B^3.
2. Active Micro-Adjustment by Noise Reduction & Homeostatic Energy Stabilization on every tick:
   - Dissipates localized spatial phase turbulence via S^2 Laplacian micro-damping.
   - Homeostatically clamps mean metabolic energy (E_0 = 0.40).
   - Soft Lyapunov saturation prevents numerical runaway.
3. Zero-Feedback Live Reasoning:
   - Ingests sensory prompts by perturbing the living manifold rather than starting from dead zeros.
   - Decodes output from Hemisphere B while subcortical core continues pulsating.
4. Telemetry: Continuously updates checkpoints/living_daemon_status.json with live metrics.
"""

import os
import sys
import time
import math
import json
import argparse
import threading
import torch
from transformers import AutoTokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, ".hf_cache"))

from src.real_geodesic_sphere import build_real_geodesic_spheroid


class LivingPulsatingDaemon:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        heartbeat_hz: float = 2.0,
        status_file: str = "checkpoints/living_daemon_status.json",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        self.heartbeat_hz = heartbeat_hz
        self.status_file = os.path.join(ROOT_DIR, status_file)
        os.makedirs(os.path.dirname(self.status_file), exist_ok=True)

        print(f"Loading tokenizer & RealGeodesicSpheroid onto {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = build_real_geodesic_spheroid()
        if os.path.exists(checkpoint_path):
            print(f"Loading weights from {checkpoint_path}...")
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        else:
            print(f"Notice: {checkpoint_path} not found, initializing fresh weights.")

        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        # Living State Manifold
        self.state_lock = threading.Lock()
        self.living_state = self.model.init_brain_state(1, self.device, living=True)
        self.ticks = 0
        self.stop_event = threading.Event()
        self.thread = None
        self.start_time = time.time()

    def start(self):
        """Starts the continuous living background pulsation loop."""
        if self.thread is not None and self.thread.is_alive():
            return

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._heartbeat_worker, daemon=True)
        self.thread.start()
        print(f"★ Living Pulsating Daemon started at {self.heartbeat_hz:.1f} Hz on {self.device}!")

    def stop(self):
        """Stops the daemon."""
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
        print("Living Pulsating Daemon halted.")

    def _heartbeat_worker(self):
        interval = 1.0 / max(self.heartbeat_hz, 0.1)

        while not self.stop_event.is_set():
            t0 = time.time()
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda" if self.device.type == "cuda" else "cpu", dtype=self.dtype):
                    with self.state_lock:
                        self.ticks += 1
                        # Micro-adjusted, noise-reduced, homeostatically stabilized pulsation tick
                        self.living_state = self.model.live_heartbeat_tick(self.living_state, t_step=self.ticks)

                        # Telemetry
                        kinetic_energy = (self.living_state ** 2).mean().item()
                        state_norm = torch.norm(self.living_state).item()

            # Record telemetry every 10 ticks
            if self.ticks % 10 == 0:
                vram_mb = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
                uptime = time.time() - self.start_time
                status = {
                    "is_alive": True,
                    "ticks": self.ticks,
                    "uptime_seconds": round(uptime, 1),
                    "mean_kinetic_energy": round(kinetic_energy, 5),
                    "total_norm": round(state_norm, 3),
                    "vram_allocated_mb": round(vram_mb, 1),
                    "frequency_hz": self.heartbeat_hz,
                    "timestamp": time.time(),
                }
                try:
                    with open(self.status_file, "w") as f:
                        json.dump(status, f, indent=2)
                except Exception:
                    pass

            elapsed = time.time() - t0
            sleep_time = max(0.0, interval - elapsed)
            time.sleep(sleep_time)

    def query(self, prompt: str, thinking_cycles: int = 3, max_answer_tokens: int = 32) -> str:
        """
        Ingests a prompt directly into the active living resting-state manifold,
        deliberates, and reads out answer tokens while keeping the manifold living.
        """
        q_toks = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if self.device.type == "cuda" else "cpu", dtype=self.dtype):
                with self.state_lock:
                    # Ingest by perturbing the current living resting-state
                    state_3d, sensory_memory = self.model.ingest_stimulus(
                        q_toks, initial_state=self.living_state, living=True
                    )
                    # Deliberate
                    past_states = []
                    for _ in range(thinking_cycles):
                        state_3d, past_states = self.model.core.pulse_wave_step(state_3d, past_states)

                    # Readout tokens
                    refractory_trace = torch.zeros(1, self.model.vocab_size, device=self.device, dtype=self.dtype)
                    out_tokens = []
                    for t in range(max_answer_tokens):
                        logits, state_3d, refractory_trace, past_states = self.model.emit_next_token(
                            state_3d, sensory_memory, refractory_trace, past_states=past_states, step_idx=t
                        )
                        if t < 3:
                            logits[..., self.tokenizer.eos_token_id] -= 50.0

                        top_idx = torch.argmax(logits, dim=-1)
                        out_tokens.append(top_idx)
                        if (top_idx == self.tokenizer.eos_token_id).all():
                            break

                    # Update living state with residual memory from thought cycle
                    self.living_state = state_3d.detach()

        if len(out_tokens) == 0:
            return "<empty>"
        out_tensor = torch.cat(out_tokens, dim=-1)
        return self.tokenizer.decode(out_tensor[0]).strip()


def run_daemon_service(checkpoint: str, duration_seconds: int = 3600):
    daemon = LivingPulsatingDaemon(checkpoint_path=checkpoint, heartbeat_hz=2.0)
    daemon.start()

    print("\n" + "=" * 80)
    print("      LIVING CONTINUOUS PULSATING SPHEROID DAEMON ACTIVE                  ")
    print("=" * 80)
    print(f"Checkpoint  : {checkpoint}")
    print(f"Frequency   : {daemon.heartbeat_hz} Hz (Continuous Theta-Alpha Pacemaker)")
    print(f"Telemetry   : checkpoints/living_daemon_status.json")
    print(f"Status      : Model stays alive and continuously pulsating on GPU memory")
    print("=" * 80 + "\n")

    t_end = time.time() + duration_seconds
    try:
        while time.time() < t_end:
            time.sleep(5.0)
            with open(os.path.join(ROOT_DIR, "checkpoints/living_daemon_status.json"), "r") as f:
                telemetry = json.load(f)
            print(f"[Heartbeat #{telemetry['ticks']:6d}] | Kinetic Energy: {telemetry['mean_kinetic_energy']:.5f} | Norm: {telemetry['total_norm']:.2f} | VRAM: {telemetry['vram_allocated_mb']} MB")
    except KeyboardInterrupt:
        print("Shutting down daemon...")
    finally:
        daemon.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/proto_agi_sphere_best.pt")
    parser.add_argument("--duration", type=int, default=7200, help="Seconds to keep daemon active")
    args = parser.parse_args()

    run_daemon_service(checkpoint=args.checkpoint, duration_seconds=args.duration)
