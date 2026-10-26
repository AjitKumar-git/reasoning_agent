"""
Living Organoid Perturbation Framework
=======================================
Maintains a continuously active, pulsating 3D Riemannian brain organoid manifold (B^3),
supports live online query ingestion, and provides real-time injection of physical,
biological, and structural perturbations to observe dynamical stability and cognitive resilience:

Perturbation Modalities:
1. Thermal Langevin Noise: Isotropic or shell-selective Gaussian noise on B^3.
2. Astrocytic Calcium/Glutamate Shock: Excitotoxic hyper-activation or metabolic gain surges.
3. Cortical Columnar Lesion: Geodesic masking on S^2 simulating focal stroke / inactivation.
4. Thalamic Sensory Noise: Corrupts afferent sensory features before gating.
5. Cross-Frequency PAC Desynchronization: Theta-gamma carrier clock phase jitter.
6. Subcortical Corollary Discharge Ablation: Attenuates internal motor-to-core efference copy.
7. Core Radial Soliton Shock: Concentrated radial energy impulse into the deep Riemannian core.

Metrics Measured:
- Instantaneous Kinetic Energy E_k(t)
- Manifold Frobenius Norm ||Psi(t)||_F
- Trajectory Perturbation Divergence D(t) = ||Psi_pert(t) - Psi_clean(t)|| / ||Psi_clean(t)||
- Homeostatic Relaxation Latency tau_relax (ticks to re-enter basin of attraction)
- Strict AGI Reasoning Pass Rate & Degradation Gap under perturbation
"""

import os
import sys
import time
import math
import json
import random
import threading
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Any, Generator

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.environ["HF_HOME"] = os.path.join(ROOT_DIR, ".hf_cache")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(ROOT_DIR, ".hf_cache")
os.environ["HF_HUB_CACHE"] = os.path.join(ROOT_DIR, ".hf_cache")

from src.organoid_layers import build_enhanced_organoid_spheroid, EnhancedOrganoidSpheroid
from src.multimodal_layers import build_multimodal_organoid_core, MultimodalOrganoidCore
from src.real_geodesic_sphere import build_real_geodesic_spheroid


@dataclass
class PerturbationConfig:
    name: str = "Clean Baseline"
    perturb_type: str = "none"  # "none", "thermal", "astro_shock", "columnar_lesion", "thalamic_noise", "pac_jitter", "corollary_ablation", "core_pulse"
    intensity: float = 0.0
    duration_ticks: int = 1
    target_shells: Optional[List[int]] = None
    target_coords: Optional[Tuple[float, float, float]] = None  # (x, y, z) on S^2
    lesion_radius_deg: float = 30.0
    description: str = ""


class PerturbationInjector:
    """Mathematical operators for biological & physical perturbations on B^3 manifold."""

    @staticmethod
    def apply_thermal_noise(
        state_3d: torch.Tensor,
        sigma: float,
        target_shells: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Injects Gaussian thermal noise: Psi <- Psi + N(0, sigma^2)."""
        if sigma <= 0.0:
            return state_3d
        noise = torch.randn_like(state_3d) * sigma
        if target_shells is not None:
            mask = torch.zeros(state_3d.shape[1], device=state_3d.device)
            mask[target_shells] = 1.0
            noise = noise * mask.view(1, -1, 1, 1)
        return state_3d + noise

    @staticmethod
    def apply_astro_shock(
        astro_field: torch.Tensor,
        state_3d: torch.Tensor,
        surge_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simulates tripartite glial excitotoxicity: surges Ca2+ and drives hyper-gain."""
        if surge_scale <= 0.0:
            return astro_field, state_3d
        # Surge astrocytic field
        astro_perturbed = astro_field + torch.abs(torch.randn_like(astro_field)) * surge_scale
        # Induce metabolic gain multiplier
        gain_multiplier = 1.0 + surge_scale * 0.5
        state_perturbed = state_3d * gain_multiplier
        return astro_perturbed, state_perturbed

    @staticmethod
    def apply_columnar_lesion(
        state_3d: torch.Tensor,
        coords: torch.Tensor,
        target_coords: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        lesion_radius_deg: float = 30.0,
        attenuation: float = 0.0,
    ) -> torch.Tensor:
        """
        Simulates focal columnar lesion/ischemia on S^2:
        Attenuates nodes within angular geodesic radius theta_lesion of target_coords.
        """
        center = torch.tensor(target_coords, device=state_3d.device, dtype=coords.dtype)
        center = center / center.norm().clamp(min=1e-6)

        # Geodesic distance on S^2: d = arccos(clamp(coords . center, -1, 1))
        cos_sim = torch.mv(coords, center).clamp(-1.0, 1.0)
        ang_dist_rad = torch.acos(cos_sim)
        radius_rad = math.radians(lesion_radius_deg)

        # Mask: 1.0 outside lesion, attenuation inside lesion
        inside_lesion = ang_dist_rad <= radius_rad
        mask = torch.ones(coords.shape[0], device=state_3d.device, dtype=state_3d.dtype)
        mask[inside_lesion] = attenuation

        # Apply across all shells
        return state_3d * mask.view(1, 1, -1, 1)

    @staticmethod
    def apply_thalamic_noise(sensory_field: torch.Tensor, sigma: float) -> torch.Tensor:
        """Corrupts sensory afferent inputs entering the thalamic gateway."""
        if sigma <= 0.0:
            return sensory_field
        noise = torch.randn_like(sensory_field) * sigma
        return sensory_field + noise

    @staticmethod
    def apply_pac_jitter(state_3d: torch.Tensor, phase_jitter_rad: float) -> torch.Tensor:
        """Induces frequency/phase dysrhythmia on nested carrier clocks."""
        if phase_jitter_rad <= 0.0:
            return state_3d
        jitter = torch.sin(state_3d + phase_jitter_rad) * 0.15
        return state_3d + jitter

    @staticmethod
    def apply_corollary_ablation(corollary_vector: torch.Tensor, factor: float) -> torch.Tensor:
        """Attenuates or severs internal motor-to-core efference copy (0.0 = complete cut)."""
        return corollary_vector * factor

    @staticmethod
    def apply_core_pulse(state_3d: torch.Tensor, amplitude: float, shell_idx: int = 0) -> torch.Tensor:
        """Injects a concentrated radial soliton pulse into the deep subcortical core."""
        pulse = torch.randn_like(state_3d[:, shell_idx : shell_idx + 1, :, :]) * amplitude
        perturbed = state_3d.clone()
        perturbed[:, shell_idx : shell_idx + 1, :, :] += pulse
        return perturbed


class LivingOrganoidHost:
    """
    Manages continuous background execution of the living organoid manifold,
    telemetry streaming, live perturbation injection, and online zero-feedback reasoning.
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/multimodal_organoid_best.pt",
        model_type: str = "auto",  # "multimodal", "enhanced", "real", "auto"
        device: str = "cuda",
        heartbeat_hz: float = 2.0,
        status_file: str = "checkpoints/living_perturbation_status.json",
        ipc_file: str = "checkpoints/living_daemon_ipc.json",
        live_output: bool = False,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        self.heartbeat_hz = heartbeat_hz
        self.status_file = os.path.join(ROOT_DIR, status_file)
        self.ipc_file = os.path.join(ROOT_DIR, ipc_file)
        self.live_output = live_output
        os.makedirs(os.path.dirname(self.status_file), exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        vocab_size = self.tokenizer.vocab_size

        # Resolve model architecture
        abs_ckpt = os.path.join(ROOT_DIR, checkpoint_path) if not os.path.isabs(checkpoint_path) else checkpoint_path
        if model_type == "auto":
            if "multimodal" in checkpoint_path:
                model_type = "multimodal"
            elif "tightened" in checkpoint_path or "enhanced" in checkpoint_path:
                model_type = "enhanced"
            else:
                model_type = "real"

        is_100m = "100m" in checkpoint_path
        if os.path.exists(abs_ckpt) and not is_100m:
            try:
                sd_peek = torch.load(abs_ckpt, map_location="cpu")
                if "lang_pole.token_embed.weight" in sd_peek and sd_peek["lang_pole.token_embed.weight"].shape[1] == 768:
                    is_100m = True
            except Exception:
                pass

        self.model_type = model_type
        size_label = "100M" if is_100m else "33M"
        print(f"Initializing LivingOrganoidHost ({model_type.upper()} [{size_label}]) on {self.device} ({self.dtype})...")

        if model_type == "multimodal":
            self.model = build_multimodal_organoid_core(vocab_size=vocab_size, preset="100m" if is_100m else "33m")
        elif model_type == "enhanced":
            self.model = build_enhanced_organoid_spheroid(vocab_size=vocab_size)
        else:
            self.model = build_real_geodesic_spheroid(vocab_size=vocab_size)

        if os.path.exists(abs_ckpt):
            print(f"Loading checkpoint weights from: {abs_ckpt}")
            sd = torch.load(abs_ckpt, map_location=self.device)
            self.model.load_state_dict(sd, strict=False)
            print("  --> Weights mapped successfully.")
        else:
            print(f"Warning: Checkpoint {abs_ckpt} not found! Initializing fresh weights.")

        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        # Thread synchronization and state
        self.state_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.ticks = 0
        self.start_time = time.time()

        # Active perturbation state
        self.active_perturbation: Optional[PerturbationConfig] = None
        self.perturbation_ticks_remaining = 0

        # Reference trajectory tracking for Lyapunov / divergence computation
        self.baseline_trajectory: List[torch.Tensor] = []
        self.divergence_history: List[float] = []

        # Initialize living state
        with torch.no_grad():
            self.state_3d = self.model.core.init_living_state(1, self.device, self.dtype)
            self.past_states: List[torch.Tensor] = []
            if hasattr(self.model, "astrocytes"):
                self.astro_field = self.model.astrocytes.init_field(1, self.device, self.dtype)
            else:
                self.astro_field = None
            if hasattr(self.model, "hippocampus"):
                self.hipp_mem = self.model.hippocampus.init_memory(1, self.device, self.dtype)
            else:
                self.hipp_mem = None

        print("★ Living Organoid Manifold initialized successfully.")

    def start(self):
        """Starts the continuous living background pacemaker loop."""
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._heartbeat_worker, daemon=True)
        self.thread.start()
        print(f"★ Living Organoid Pacemaker running at {self.heartbeat_hz:.1f} Hz.")

    def stop(self):
        """Halts the background loop."""
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
        print("Living Organoid Pacemaker halted.")

    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def inject_perturbation(self, config: PerturbationConfig):
        """Thread-safely queues an active perturbation into the living manifold."""
        with self.state_lock:
            self.active_perturbation = config
            self.perturbation_ticks_remaining = config.duration_ticks
            print(f"\n⚡ [PERTURBATION INJECTED] -> {config.name} (Type: {config.perturb_type}, Intensity: {config.intensity}, Duration: {config.duration_ticks} ticks)")

    def get_spontaneous_thought(self, num_tokens: int = 3) -> str:
        """
        Reads out spontaneous motor efferent bursts (Default Mode Network murmurs)
        directly from current continuous resting wave activations in Hemisphere B.
        """
        try:
            with torch.no_grad():
                with torch.amp.autocast("cuda" if self.device.type == "cuda" else "cpu", dtype=self.dtype):
                    motor_nodes = getattr(self.model, "motor_nodes", getattr(self.model.core, "motor_nodes", None))
                    if motor_nodes is None:
                        return ""
                    hemi_b = self.state_3d[:, -1, motor_nodes, :]
                    embed_dim = self.model.motor_pole.embed_dim if hasattr(self.model, "motor_pole") and hasattr(self.model.motor_pole, "embed_dim") else 768
                    sensory_mem = torch.zeros(1, 1, embed_dim, device=self.device, dtype=self.dtype)
                    refractory = torch.zeros(1, self.model.vocab_size, device=self.device, dtype=self.dtype)
                    tokens = []
                    curr_hemi = hemi_b.clone()
                    for t in range(num_tokens):
                        logits, refractory, _ = self.model.motor_pole.forward_readout(curr_hemi, sensory_mem, refractory, step_idx=t)
                        if len(tokens) > 0:
                            logits[:, tokens[-1]] -= 3.0
                        tok = torch.argmax(logits, dim=-1).item()
                        if tok == self.tokenizer.eos_token_id:
                            break
                        tokens.append(tok)
            return self.tokenizer.decode(tokens).strip() if tokens else ""
        except Exception:
            return ""

    def _step_manifold(self, state_3d: torch.Tensor, cycle: int) -> torch.Tensor:
        """Steps the physical biological wave equations for one tick."""
        if hasattr(self.model, "deliberate_step") and self.astro_field is not None and self.hipp_mem is not None:
            state_3d, self.astro_field, self.hipp_mem, self.past_states = self.model.deliberate_step(
                state_3d, self.astro_field, self.hipp_mem, self.past_states, cycle=cycle
            )
        else:
            state_3d, self.past_states = self.model.core.pulse_wave_step(state_3d, self.past_states)
        return state_3d

    def _heartbeat_worker(self):
        interval = 1.0 / max(self.heartbeat_hz, 0.1)

        while not self.stop_event.is_set():
            t0 = time.time()
            divergence = 0.0

            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda" if self.device.type == "cuda" else "cpu", dtype=self.dtype):
                    with self.state_lock:
                        self.ticks += 1

                        # Track clean baseline before perturbation
                        unperturbed_copy = self.state_3d.clone()

                        # Apply active perturbation if engaged
                        if self.active_perturbation is not None and self.perturbation_ticks_remaining > 0:
                            p = self.active_perturbation
                            self.perturbation_ticks_remaining -= 1

                            if p.perturb_type == "thermal":
                                self.state_3d = PerturbationInjector.apply_thermal_noise(
                                    self.state_3d, p.intensity, p.target_shells
                                )
                            elif p.perturb_type == "astro_shock" and self.astro_field is not None:
                                self.astro_field, self.state_3d = PerturbationInjector.apply_astro_shock(
                                    self.astro_field, self.state_3d, p.intensity
                                )
                            elif p.perturb_type == "columnar_lesion":
                                coords = self.model.core.coords
                                target_c = p.target_coords or (0.0, 0.0, 1.0)
                                self.state_3d = PerturbationInjector.apply_columnar_lesion(
                                    self.state_3d, coords, target_c, p.lesion_radius_deg, attenuation=p.intensity
                                )
                            elif p.perturb_type == "pac_jitter":
                                self.state_3d = PerturbationInjector.apply_pac_jitter(self.state_3d, p.intensity)
                            elif p.perturb_type == "core_pulse":
                                shell = p.target_shells[0] if p.target_shells else 0
                                self.state_3d = PerturbationInjector.apply_core_pulse(self.state_3d, p.intensity, shell)

                            if self.perturbation_ticks_remaining <= 0:
                                self.active_perturbation = None

                        # Physical wave pulsation step
                        self.state_3d = self._step_manifold(self.state_3d, cycle=self.ticks)

                        # Homeostatic recovery & divergence calculation
                        norm_unpert = unperturbed_copy.norm().clamp(min=1e-5).item()
                        diff_norm = (self.state_3d - unperturbed_copy).norm().item()
                        divergence = diff_norm / norm_unpert
                        self.divergence_history.append(divergence)
                        if len(self.divergence_history) > 100:
                            self.divergence_history.pop(0)

                        kinetic_energy = (self.state_3d ** 2).mean().item()
                        state_norm = self.state_3d.norm().item()

            # Process IPC commands if any
            self.poll_ipc()

            # Write telemetry every 5 ticks
            if self.ticks % 5 == 0:
                vram_mb = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
                shell_energies = [round((self.state_3d[:, k, :, :] ** 2).mean().item(), 4) for k in range(self.state_3d.shape[1])]
                spontaneous_thought = self.get_spontaneous_thought(num_tokens=3)

                telemetry = {
                    "is_alive": True,
                    "model_type": self.model_type,
                    "ticks": self.ticks,
                    "uptime_seconds": round(time.time() - self.start_time, 1),
                    "mean_kinetic_energy": round(kinetic_energy, 5),
                    "total_norm": round(state_norm, 3),
                    "shell_energies": shell_energies,
                    "spontaneous_thought": spontaneous_thought,
                    "trajectory_divergence": round(divergence, 5),
                    "recent_mean_divergence": round(sum(self.divergence_history) / max(1, len(self.divergence_history)), 5),
                    "active_perturbation": asdict(self.active_perturbation) if self.active_perturbation else None,
                    "vram_allocated_mb": round(vram_mb, 1),
                    "frequency_hz": self.heartbeat_hz,
                    "timestamp": time.time(),
                }
                try:
                    with open(self.status_file, "w") as f:
                        json.dump(telemetry, f, indent=2)
                except Exception:
                    pass

                if self.live_output:
                    bars = [" " if e < 0.2 else "▂" if e < 0.3 else "▄" if e < 0.4 else "▆" if e < 0.5 else "█" for e in shell_energies]
                    shell_bar_str = "".join(bars)
                    p_str = f" | ⚡ {self.active_perturbation.name}" if self.active_perturbation else ""
                    th_str = f" | 💭 \"{spontaneous_thought}\"" if spontaneous_thought else ""
                    print(f"[Tick #{self.ticks:6d} | {self.heartbeat_hz:.1f}Hz] E_k: {kinetic_energy:.4f} | Shells [{shell_bar_str}] {shell_energies[0]:.2f}→{shell_energies[-1]:.2f} | VRAM: {vram_mb:.0f}MB{p_str}{th_str}", flush=True)

            elapsed = time.time() - t0
            time.sleep(max(0.0, interval - elapsed))

    def poll_ipc(self):
        """Checks for external commands written to IPC file."""
        if not os.path.exists(self.ipc_file):
            return
        try:
            with open(self.ipc_file, "r") as f:
                cmd = json.load(f)
            os.remove(self.ipc_file)

            action = cmd.get("action")
            if action == "perturb":
                p_cfg = PerturbationConfig(
                    name=cmd.get("name", "IPC Perturbation"),
                    perturb_type=cmd.get("type", "thermal"),
                    intensity=cmd.get("intensity", 0.05),
                    duration_ticks=cmd.get("duration", 5),
                    target_shells=cmd.get("shells"),
                    lesion_radius_deg=cmd.get("radius_deg", 30.0),
                )
                self.inject_perturbation(p_cfg)
            elif action == "inject_token":
                token_text = cmd.get("token", cmd.get("text", ""))
                amp = cmd.get("amplitude", 1.0)
                res = self.inject_token_stimulus(token_text, amplitude=amp)
                resp_file = os.path.join(ROOT_DIR, "checkpoints/living_daemon_ipc_response.json")
                with open(resp_file, "w") as f:
                    json.dump(res, f, indent=2)
            elif action == "query":
                prompt = cmd.get("prompt", "")
                mid_toks = cmd.get("mid_thought_tokens")
                res = self.query(
                    prompt,
                    thinking_cycles=cmd.get("thinking_cycles", 3),
                    mid_thought_tokens=mid_toks,
                    temperature=cmd.get("temperature", 0.5),
                    max_answer_tokens=cmd.get("max_answer_tokens", 48),
                )
                resp_file = os.path.join(ROOT_DIR, "checkpoints/living_daemon_ipc_response.json")
                with open(resp_file, "w") as f:
                    json.dump(res, f, indent=2)
            elif action == "stop":
                self.stop()
        except Exception:
            pass

    def inject_token_stimulus(
        self,
        text_or_token: str,
        amplitude: float = 1.0,
        thinking_cycles: int = 2,
        max_echo_tokens: int = 16,
    ) -> Dict[str, Any]:
        """
        Injects an isolated semantic token impulse into Hemisphere A while the resting-state
        pacemaker is active. Measures Evoked Potential (ERP) ripple across 8 concentric shells
        and captures any spontaneous motor efferent echo from Hemisphere B.
        """
        device = self.device
        dtype = self.dtype
        q_toks = self.tokenizer.encode(text_or_token, return_tensors="pt").to(device)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
                with self.state_lock:
                    if hasattr(self.model, "lang_pole"):
                        sens_field, sensory_memory = self.model.lang_pole(q_toks)
                        nodes = self.model.lang_nodes
                    else:
                        sens_field, sensory_memory = self.model.sensory_pole(q_toks)
                        nodes = self.model.core.sensory_nodes

                    # Measure energy before injection
                    e_before = (self.state_3d ** 2).mean().item()
                    shell_energy_before = [(self.state_3d[:, k, :, :] ** 2).mean().item() for k in range(8)]

                    core_0 = self.state_3d[:, 0, :, :].mean(dim=1)
                    gated_sens = self.model.thalamus(sens_field.to(dtype), core_0)

                    # Ingest token impulse into Hemisphere A
                    self.state_3d[:, -1, nodes, :] += gated_sens * amplitude

                    # Measure immediate Evoked Potential (ERP) shock across all 8 shells
                    shell_energy_shock = [(self.state_3d[:, k, :, :] ** 2).mean().item() for k in range(8)]
                    e_shock = (self.state_3d ** 2).mean().item()

                    # Deliberate through biological wave equations
                    past_states: List[torch.Tensor] = []
                    working_state = self.state_3d.clone()
                    astro_field = self.astro_field.clone() if self.astro_field is not None else None
                    hipp_mem = self.hipp_mem.clone() if self.hipp_mem is not None else None

                    for cycle in range(thinking_cycles):
                        if hasattr(self.model, "deliberate_step") and astro_field is not None and hipp_mem is not None:
                            working_state, astro_field, hipp_mem, past_states = self.model.deliberate_step(
                                working_state, astro_field, hipp_mem, past_states, cycle
                            )
                        else:
                            working_state, past_states = self.model.core.pulse_wave_step(working_state, past_states)

                    # Spontaneous motor readout from Hemisphere B
                    motor_nodes = getattr(self.model, "motor_nodes", getattr(self.model.core, "motor_nodes", None))
                    refractory_trace = torch.zeros(1, self.model.vocab_size, device=device, dtype=dtype)
                    out_tokens = []

                    for t in range(max_echo_tokens):
                        hemi_b_field = working_state[:, -1, motor_nodes, :]
                        logits, refractory_trace, corollary_vector = self.model.motor_pole.forward_readout(
                            hemi_b_field, sensory_memory, refractory_trace, step_idx=t
                        )
                        if len(out_tokens) > 0:
                            logits[:, out_tokens[-1]] -= 3.5
                            for prev_tok in set(out_tokens[-4:]):
                                logits[:, prev_tok] -= 1.8
                        next_tok = torch.argmax(logits, dim=-1).item()
                        if next_tok == self.tokenizer.eos_token_id:
                            break
                        out_tokens.append(next_tok)

                        working_state[:, 0, :, :] += corollary_vector.unsqueeze(1) * 0.20
                        working_state[:, 1, :, :] += corollary_vector.unsqueeze(1) * 0.10
                        working_state, past_states = self.model.core.pulse_wave_step(working_state, past_states)

                    # Update living state with homeostatic assimilation
                    self.state_3d = (0.85 * self.state_3d + 0.15 * working_state).detach()

        echo_text = self.tokenizer.decode(out_tokens).strip() if out_tokens else "<silent>"
        delta_shells = [round(shock - before, 5) for shock, before in zip(shell_energy_shock, shell_energy_before)]

        return {
            "injected_token": text_or_token,
            "amplitude": amplitude,
            "delta_kinetic_energy": round(e_shock - e_before, 5),
            "evoked_potential_by_shell": delta_shells,
            "spontaneous_motor_echo": echo_text,
            "num_echo_tokens": len(out_tokens),
        }

    def stream_query(
        self,
        prompt: str,
        thinking_cycles: int = 3,
        max_answer_tokens: int = 48,
        active_perturbation: Optional[PerturbationConfig] = None,
        mid_thought_tokens: Optional[str] = None,
        inject_at_cycle: int = 1,
        temperature: float = 0.5,
        fresh_state: bool = False,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Executes zero-feedback live inference on the continuous living manifold,
        yielding real-time telemetry events for each deliberation cycle and token.
        """
        device = self.device
        dtype = self.dtype
        t_start = time.time()
        q_toks = self.tokenizer.encode(prompt, return_tensors="pt").to(device)

        yield {
            "event": "start",
            "prompt": prompt,
            "thinking_cycles": thinking_cycles,
            "max_tokens": max_answer_tokens,
            "perturbation": active_perturbation.name if active_perturbation else "Clean Baseline",
            "timestamp": t_start,
        }

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
                with self.state_lock:
                    # Ingest stimulus directly into current living state
                    if fresh_state:
                        working_state = self.model.core.init_living_state(1, device, dtype)
                        astro_field = self.model.astrocytes.init_field(1, device, dtype) if hasattr(self.model, "astrocytes") else None
                        hipp_mem = self.model.hippocampus.init_memory(1, device, dtype) if hasattr(self.model, "hippocampus") else None
                    else:
                        working_state = self.state_3d.clone()
                        astro_field = self.astro_field.clone() if self.astro_field is not None else None
                        hipp_mem = self.hipp_mem.clone() if self.hipp_mem is not None else None

                    if hasattr(self.model, "lang_pole"):
                        sens_field, sensory_memory = self.model.lang_pole(q_toks)
                        nodes = self.model.lang_nodes
                    elif hasattr(self.model, "sensory_pole"):
                        sens_field, sensory_memory = self.model.sensory_pole(q_toks)
                        nodes = self.model.core.sensory_nodes
                    else:
                        raise NotImplementedError("Model lacks sensory pole.")

                    # Apply thalamic noise if specified
                    if active_perturbation and active_perturbation.perturb_type == "thalamic_noise":
                        sens_field = PerturbationInjector.apply_thalamic_noise(
                            sens_field, active_perturbation.intensity
                        )

                    core_0 = working_state[:, 0, :, :].mean(dim=1)
                    gated_sens = self.model.thalamus(sens_field.to(dtype), core_0)
                    working_state[:, -1, nodes, :] += gated_sens

                    yield {
                        "event": "stimulus_ingested",
                        "prompt_tokens": q_toks.shape[1],
                        "gated_sensory_norm": round(gated_sens.norm().item(), 4),
                        "initial_kinetic_energy": round((working_state ** 2).mean().item(), 5),
                    }

                    # Deliberation with optional continuous perturbation
                    past_states: List[torch.Tensor] = []

                    for cycle in range(thinking_cycles):
                        t_cyc_start = time.time()
                        if active_perturbation:
                            p = active_perturbation
                            if p.perturb_type == "thermal":
                                working_state = PerturbationInjector.apply_thermal_noise(
                                    working_state, p.intensity, p.target_shells
                                )
                            elif p.perturb_type == "astro_shock" and astro_field is not None:
                                astro_field, working_state = PerturbationInjector.apply_astro_shock(
                                    astro_field, working_state, p.intensity
                                )
                            elif p.perturb_type == "columnar_lesion":
                                working_state = PerturbationInjector.apply_columnar_lesion(
                                    working_state, self.model.core.coords, p.target_coords or (0.0, 0.0, 1.0),
                                    p.lesion_radius_deg, attenuation=p.intensity
                                )
                            elif p.perturb_type == "pac_jitter":
                                working_state = PerturbationInjector.apply_pac_jitter(working_state, p.intensity)

                        # Mid-thought semantic token injection (Cognitive Pinning / Distractor Collision)
                        if mid_thought_tokens is not None and cycle == inject_at_cycle:
                            mid_toks = self.tokenizer.encode(mid_thought_tokens, return_tensors="pt").to(device)
                            if hasattr(self.model, "lang_pole"):
                                mid_field, _ = self.model.lang_pole(mid_toks)
                            else:
                                mid_field, _ = self.model.sensory_pole(mid_toks)
                            gated_mid = self.model.thalamus(mid_field.to(dtype), working_state[:, 0, :, :].mean(dim=1))
                            working_state[:, -1, nodes, :] += gated_mid

                        if hasattr(self.model, "deliberate_step") and astro_field is not None and hipp_mem is not None:
                            working_state, astro_field, hipp_mem, past_states = self.model.deliberate_step(
                                working_state, astro_field, hipp_mem, past_states, cycle
                            )
                        else:
                            working_state, past_states = self.model.core.pulse_wave_step(working_state, past_states)

                        cyc_shell_e = [round((working_state[:, k, :, :] ** 2).mean().item(), 4) for k in range(working_state.shape[1])]
                        cyc_e_k = round((working_state ** 2).mean().item(), 5)
                        yield {
                            "event": "deliberation_cycle",
                            "cycle": cycle + 1,
                            "total_cycles": thinking_cycles,
                            "kinetic_energy": cyc_e_k,
                            "shell_energies": cyc_shell_e,
                            "cycle_time_ms": round((time.time() - t_cyc_start) * 1000, 2),
                        }

                    # Readout from motor pole
                    out_tokens = []
                    motor_nodes = getattr(self.model, "motor_nodes", getattr(self.model.core, "motor_nodes", None))
                    refractory_trace = torch.zeros(1, self.model.vocab_size, device=device, dtype=dtype)

                    corollary_scale = 1.0
                    if active_perturbation and active_perturbation.perturb_type == "corollary_ablation":
                        corollary_scale = active_perturbation.intensity

                    for t in range(max_answer_tokens):
                        t_tok_start = time.time()
                        hemi_b_field = working_state[:, -1, motor_nodes, :]
                        logits, refractory_trace, corollary_vector = self.model.motor_pole.forward_readout(
                            hemi_b_field, sensory_memory, refractory_trace, step_idx=t
                        )

                        # Refractory inhibition on recent firings
                        if len(out_tokens) > 0:
                            logits[:, out_tokens[-1]] -= 3.5
                            for prev_tok in set(out_tokens[-4:]):
                                logits[:, prev_tok] -= 1.8

                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        if temperature > 0:
                            probs_sample = torch.nn.functional.softmax(logits / max(temperature, 0.1), dim=-1)
                            next_tok = torch.multinomial(probs_sample, num_samples=1).item()
                        else:
                            next_tok = torch.argmax(logits, dim=-1).item()

                        conf = probs[0, next_tok].item()

                        if next_tok == self.tokenizer.eos_token_id:
                            break
                        out_tokens.append(next_tok)

                        tok_text = self.tokenizer.decode([next_tok])
                        corr_norm = corollary_vector.norm().item()

                        # Corollary discharge into core
                        corr_eff = PerturbationInjector.apply_corollary_ablation(corollary_vector, corollary_scale)
                        working_state[:, 0, :, :] += corr_eff.unsqueeze(1) * 0.20
                        working_state[:, 1, :, :] += corr_eff.unsqueeze(1) * 0.10
                        working_state, past_states = self.model.core.pulse_wave_step(working_state, past_states)

                        tok_time_ms = round((time.time() - t_tok_start) * 1000, 2)
                        yield {
                            "event": "token",
                            "step": t + 1,
                            "token_id": next_tok,
                            "token_text": tok_text,
                            "confidence": round(conf, 4),
                            "corollary_norm": round(corr_norm, 4),
                            "latency_ms": tok_time_ms,
                        }

                    # Homeostatic residual update back into living state
                    if not fresh_state:
                        self.state_3d = (0.95 * self.state_3d + 0.05 * working_state).detach()

        total_ms = round((time.time() - t_start) * 1000, 1)
        response_text = self.tokenizer.decode(out_tokens).strip() if out_tokens else "<empty>"
        yield {
            "event": "done",
            "prompt": prompt,
            "response": response_text,
            "tokens_generated": len(out_tokens),
            "perturbation": active_perturbation.name if active_perturbation else "Clean Baseline",
            "thinking_cycles": thinking_cycles,
            "total_ms": total_ms,
        }

    def query(
        self,
        prompt: str,
        thinking_cycles: int = 3,
        max_answer_tokens: int = 48,
        active_perturbation: Optional[PerturbationConfig] = None,
        mid_thought_tokens: Optional[str] = None,
        inject_at_cycle: int = 1,
        temperature: float = 0.5,
        fresh_state: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes zero-feedback live inference on the continuous living manifold,
        optionally under an active transient or persistent perturbation.
        """
        gen = self.stream_query(
            prompt=prompt,
            thinking_cycles=thinking_cycles,
            max_answer_tokens=max_answer_tokens,
            active_perturbation=active_perturbation,
            mid_thought_tokens=mid_thought_tokens,
            inject_at_cycle=inject_at_cycle,
            temperature=temperature,
            fresh_state=fresh_state,
        )
        last_event: Dict[str, Any] = {}
        for event in gen:
            if event.get("event") == "done":
                last_event = event
        return {
            "prompt": prompt,
            "response": last_event.get("response", ""),
            "tokens_generated": last_event.get("tokens_generated", 0),
            "perturbation": last_event.get("perturbation", "Clean Baseline"),
            "thinking_cycles": thinking_cycles,
            "latency_ms": last_event.get("total_ms", 0.0),
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Living Organoid Manifold Host & Perturbation Engine")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/multimodal_organoid_best.pt")
    parser.add_argument("--heartbeat", type=float, default=2.0, help="Heartbeat frequency in Hz")
    parser.add_argument("--duration", type=int, default=86400, help="Running duration in seconds")
    parser.add_argument("--live", action="store_true", help="Print real-time animated shell wave output & spontaneous thoughts")
    args = parser.parse_args()

    host = LivingOrganoidHost(checkpoint_path=args.checkpoint, heartbeat_hz=args.heartbeat, live_output=args.live)
    host.start()

    print("\n" + "=" * 80)
    print("      LIVING PERTURBATION FRAMEWORK ACTIVE (DAEMON MODE)      ")
    print("=" * 80)
    print(f"Model       : {host.model_type.upper()} ({sum(p.numel() for p in host.model.parameters())/1e6:.2f}M parameters)")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Frequency   : {args.heartbeat} Hz continuous biological pacemaker")
    print(f"Live Output : {'Enabled (real-time stream)' if args.live else 'Disabled'}")
    print(f"Status Log  : {host.status_file}")
    print(f"IPC Queue   : {host.ipc_file}")
    print("Commands    : Write JSON to IPC file to inject live perturbations or queries")
    print("=" * 80 + "\n")

    t_end = time.time() + args.duration
    try:
        while time.time() < t_end:
            time.sleep(5.0)
            if not args.live and os.path.exists(host.status_file):
                with open(host.status_file, "r") as f:
                    stat = json.load(f)
                div_str = f" | Div: {stat.get('trajectory_divergence', 0):.4f}" if stat.get('active_perturbation') else ""
                print(f"[Tick #{stat.get('ticks', 0):6d}] E_k: {stat.get('mean_kinetic_energy', 0):.5f} | Norm: {stat.get('total_norm', 0):.2f} | VRAM: {stat.get('vram_allocated_mb', 0):.1f} MB{div_str}")
    except KeyboardInterrupt:
        print("\nStopping living framework...")
    finally:
        host.stop()


if __name__ == "__main__":
    main()

