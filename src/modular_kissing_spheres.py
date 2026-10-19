"""
Modular Adjacent Kissing-Sphere Network (B^3 Multimodal Lattice)
================================================================
Continuous Riemannian Spherical Organoids with Tangent Contact Patches:
1. Multiple autonomous unit-ball spheres S_i = {x in R^3 : ||x - C_i|| <= R} placed in 3D space.
2. Tangent Kissing Interfaces: Two spheres S_1, S_2 with ||C_1 - C_2|| = 2R touch at a tangent point.
3. Continuous Boundary Flux Operator:
   - Contact Patch D_12: Geodesic cap on S_1 facing C_2 (x . n_12 >= cos(theta_c)).
   - Contact Patch D_21: Opposing geodesic cap on S_2 facing C_1.
   - Bilateral boundary wave energy flux:
       J_{1->2} = W_flux * Psi_1[D_12]
       Psi_2[D_21] += kappa * J_{1->2}
4. Deep Deliberation Range with Retrograde Wave Interference:
   - Forward Wave: Ingested at Sensory Sphere, diffuses into Cognitive Sphere.
   - Retrograde Wave: Pulsed backwards from Motor Sphere into Cognitive Sphere.
   - Interference Standing Wave: Psi_standing = Psi_fwd + Psi_retro + gamma * (Psi_fwd * Psi_retro).
5. Zero External Token Feedback (x_{t+1} != [x, y]).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict

from src.real_geodesic_sphere import (
    build_fibonacci_geodesic_mesh,
    NoiseReductionAutoregressor,
    RealGeodesicSpheroidCore,
)


# =========================================================================
# 1. Tangent Contact Patch Operator (Kissing Interface)
# =========================================================================

class TangentContactPatch(nn.Module):
    """
    Computes and manages bidirectional wave flux across a circular tangent
    contact patch between two touching spheres S_A and S_B.
    """
    def __init__(
        self,
        name: str,
        sphere_a_idx: int,
        sphere_b_idx: int,
        center_a: torch.Tensor,
        center_b: torch.Tensor,
        coords_s2: torch.Tensor,
        channels: int = 64,
        angular_aperture_deg: float = 32.0,
    ):
        super().__init__()
        self.name = name
        self.sphere_a_idx = sphere_a_idx
        self.sphere_b_idx = sphere_b_idx
        self.channels = channels

        # Tangent contact direction unit vector: from A to B
        displacement = center_b - center_a
        dist = displacement.norm()
        normal_ab = (displacement / (dist + 1e-8)).to(coords_s2.dtype)
        normal_ba = -normal_ab

        cos_threshold = math.cos(math.radians(angular_aperture_deg))

        # Project S^2 mesh coords onto normal
        proj_a = torch.mv(coords_s2, normal_ab)
        patch_a_indices = (proj_a >= cos_threshold).nonzero(as_tuple=True)[0]

        proj_b = torch.mv(coords_s2, normal_ba)
        patch_b_indices = (proj_b >= cos_threshold).nonzero(as_tuple=True)[0]

        # Ensure equal patch sizes by trimming to min
        min_nodes = min(len(patch_a_indices), len(patch_b_indices))
        assert min_nodes > 0, f"Contact aperture too small between sphere {sphere_a_idx} and {sphere_b_idx}"
        patch_a_indices = patch_a_indices[:min_nodes]
        patch_b_indices = patch_b_indices[:min_nodes]

        self.num_contact_nodes = min_nodes
        self.register_buffer("patch_a", patch_a_indices)
        self.register_buffer("patch_b", patch_b_indices)

        # Trainable bidirectional wave flux couplers
        self.flux_a_to_b = nn.Linear(channels, channels, bias=False)
        self.flux_b_to_a = nn.Linear(channels, channels, bias=False)
        self.norm_a = nn.LayerNorm(channels)
        self.norm_b = nn.LayerNorm(channels)
        self.coupling_strength = nn.Parameter(torch.tensor(0.40, dtype=torch.float32))

    def exchange_flux(
        self,
        state_a: torch.Tensor,
        state_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates boundary flux on the outermost cortical shell (k = -1, r = 1.0)
        state_a: [B, K, M, C]
        state_b: [B, K, M, C]
        Returns flux perturbations to be injected into state_a and state_b.
        """
        dtype = state_a.dtype
        # Wave field on contact patches
        field_a = state_a[:, -1, self.patch_a, :]  # [B, N_c, C]
        field_b = state_b[:, -1, self.patch_b, :]  # [B, N_c, C]

        flux_ab = self.norm_b(self.flux_a_to_b(field_a))
        flux_ba = self.norm_a(self.flux_b_to_a(field_b))

        kappa = torch.tanh(self.coupling_strength).to(dtype)

        # Perturbation tensors matching full state shape
        delta_b = torch.zeros_like(state_b)
        delta_a = torch.zeros_like(state_a)

        delta_b[:, -1, self.patch_b, :] = (kappa * flux_ab).to(dtype)
        delta_a[:, -1, self.patch_a, :] = (kappa * flux_ba).to(dtype)

        return delta_a, delta_b


# =========================================================================
# 2. Retrograde Wave Interference Operator
# =========================================================================

class RetrogradeWaveInterference(nn.Module):
    """
    Implements non-linear standing-wave interference between the forward
    deliberation wave (Psi_fwd) and the retrograde feedback counter-wave (Psi_retro).
    Creates sharp interference fringes:
        Psi_standing = Psi_fwd + Psi_retro + gamma * (Psi_fwd * Psi_retro)
    """
    def __init__(self, channels: int = 64):
        super().__init__()
        self.channels = channels
        self.fringe_gate = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.retro_damping = nn.Parameter(torch.tensor(0.85, dtype=torch.float32))
        self.fringe_proj = nn.Linear(channels, channels, bias=False)
        self.norm = nn.LayerNorm(channels)

    def forward(self, psi_fwd: torch.Tensor, psi_retro: torch.Tensor) -> torch.Tensor:
        """
        psi_fwd: [B, K, M, C]
        psi_retro: [B, K, M, C]
        """
        dtype = psi_fwd.dtype
        gamma = torch.tanh(self.fringe_gate).to(dtype)
        decay = torch.sigmoid(self.retro_damping).to(dtype)

        # Non-linear quadratic interference fringe
        fringe = gamma * self.norm(self.fringe_proj(psi_fwd * psi_retro))
        standing_wave = psi_fwd + (decay * psi_retro) + fringe
        return standing_wave


# =========================================================================
# 3. Modular Kissing Spheroid Organoid
# =========================================================================

class KissingSphereOrganoid(nn.Module):
    """
    A single autonomous continuous Riemannian spherical organoid (B^3).
    Has 3D spatial center C_i, radius R=1.0, and K Fibonacci geodesic shells.
    """
    def __init__(
        self,
        name: str,
        center: Tuple[float, float, float],
        num_shells: int = 8,
        nodes_per_shell: int = 128,
        channels: int = 64,
        soliton_lambda: float = 0.08,
    ):
        super().__init__()
        self.name = name
        self.center = torch.tensor(center, dtype=torch.float32)
        self.num_shells = num_shells
        self.nodes_per_shell = nodes_per_shell
        self.channels = channels

        self.core = RealGeodesicSpheroidCore(
            num_shells=num_shells,
            nodes_per_shell=nodes_per_shell,
            channels=channels,
            soliton_lambda=soliton_lambda,
        )

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.core.init_state(batch_size, device, dtype)

    def pulse(
        self,
        state: torch.Tensor,
        past_states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        return self.core.pulse_wave_step(state, past_states)


# =========================================================================
# 4. Modular Kissing-Sphere Lattice (Multimodal Cluster)
# =========================================================================

class ModularKissingLattice(nn.Module):
    """
    Full Modular Spheroid Network configured with kissing contacts:
    - S_sensory at (0, 0, -2): Ingests sensory prompt tokens onto South Pole.
    - S_cognitive at (0, 0,  0): Central reasoning spheroid with deep deliberation.
    - S_motor    at (0, 0, +2): Motor efferent readout on North Pole.
    - Optional lateral S_aux at (+2, 0, 0): Memory / Multimodal buffer.
    
    Connected via tangent kissing contact patches with bilateral wave flux.
    """
    def __init__(
        self,
        vocab_size: int = 50257,
        embed_dim: int = 256,
        channels: int = 64,
        num_shells: int = 8,
        nodes_per_shell: int = 128,
        include_aux_sphere: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.channels = channels
        self.num_shells = num_shells
        self.nodes_per_shell = nodes_per_shell

        # 1. Instantiate Spheres
        self.s_sensory = KissingSphereOrganoid("sensory", (0.0, 0.0, -2.0), num_shells, nodes_per_shell, channels)
        self.s_cog = KissingSphereOrganoid("cognitive", (0.0, 0.0, 0.0), num_shells, nodes_per_shell, channels)
        self.s_motor = KissingSphereOrganoid("motor", (0.0, 0.0, 2.0), num_shells, nodes_per_shell, channels)

        self.include_aux = include_aux_sphere
        if include_aux_sphere:
            self.s_aux = KissingSphereOrganoid("auxiliary", (2.0, 0.0, 0.0), num_shells, nodes_per_shell, channels)

        # 2. Build Tangent Contact Interfaces
        coords = self.s_cog.core.coords

        # Patch 1: Sensory (0, 0, -2) <-> Cognitive (0, 0, 0)
        self.patch_sensory_cog = TangentContactPatch(
            name="sensory_to_cog",
            sphere_a_idx=0,
            sphere_b_idx=1,
            center_a=self.s_sensory.center,
            center_b=self.s_cog.center,
            coords_s2=coords,
            channels=channels,
            angular_aperture_deg=32.0,
        )

        # Patch 2: Cognitive (0, 0, 0) <-> Motor (0, 0, 2)
        self.patch_cog_motor = TangentContactPatch(
            name="cog_to_motor",
            sphere_a_idx=1,
            sphere_b_idx=2,
            center_a=self.s_cog.center,
            center_b=self.s_motor.center,
            coords_s2=coords,
            channels=channels,
            angular_aperture_deg=32.0,
        )

        # Patch 3: Cognitive (0, 0, 0) <-> Auxiliary (2, 0, 0)
        if include_aux_sphere:
            self.patch_cog_aux = TangentContactPatch(
                name="cog_to_aux",
                sphere_a_idx=1,
                sphere_b_idx=3,
                center_a=self.s_cog.center,
                center_b=self.s_aux.center,
                coords_s2=coords,
                channels=channels,
                angular_aperture_deg=32.0,
            )

        # 3. Sensory Afferent & Motor Efferent
        num_sensory = len(self.s_sensory.core.sensory_nodes)
        num_motor = len(self.s_motor.core.motor_nodes)

        from src.real_geodesic_sphere import GeodesicSensoryAfferent, GeodesicMotorEfferent
        self.sensory_retina = GeodesicSensoryAfferent(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            channels=channels,
            num_sensory_nodes=num_sensory,
        )
        self.motor_broca = GeodesicMotorEfferent(
            channels=channels,
            embed_dim=embed_dim,
            vocab_size=vocab_size,
            num_motor_nodes=num_motor,
        )

        # 4. Retrograde Interference Operator
        self.retrograde_interference = RetrogradeWaveInterference(channels=channels)

        # 5. Corollary Discharge Highway into Subcortical Core
        self.corollary_proj = nn.Linear(vocab_size, channels, bias=False)

    def initialize_lattice(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        states = {
            "sensory": self.s_sensory.init_state(batch_size, device, dtype),
            "cognitive": self.s_cog.init_state(batch_size, device, dtype),
            "motor": self.s_motor.init_state(batch_size, device, dtype),
        }
        if self.include_aux:
            states["auxiliary"] = self.s_aux.init_state(batch_size, device, dtype)
        return states

    def pulse_multimodal_lattice(
        self,
        states: Dict[str, torch.Tensor],
        past_states_dict: Optional[Dict[str, List[torch.Tensor]]] = None,
        retrograde_wave: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]]]:
        """
        Executes one synchronized multi-sphere wave pulse:
        1. Compute boundary flux across all tangent contact patches.
        2. Apply boundary flux perturbations to sphere outer shells.
        3. If retrograde wave exists, apply standing-wave interference inside cognitive sphere.
        4. Internal continuous Laplace-Beltrami + radial + soliton wave evolution in each sphere.
        """
        if past_states_dict is None:
            past_states_dict = {k: [] for k in states.keys()}

        # 1. Compute Tangent Contact Fluxes
        d_sensory, d_cog_from_sensory = self.patch_sensory_cog.exchange_flux(states["sensory"], states["cognitive"])
        d_cog_from_motor, d_motor = self.patch_cog_motor.exchange_flux(states["cognitive"], states["motor"])

        # Accumulate boundary flux on outer shells
        states["sensory"] = states["sensory"] + d_sensory
        states["cognitive"] = states["cognitive"] + d_cog_from_sensory + d_cog_from_motor
        states["motor"] = states["motor"] + d_motor

        if self.include_aux:
            d_cog_from_aux, d_aux = self.patch_cog_aux.exchange_flux(states["cognitive"], states["auxiliary"])
            states["cognitive"] = states["cognitive"] + d_cog_from_aux
            states["auxiliary"] = states["auxiliary"] + d_aux

        # 2. Retrograde Standing Wave Interference inside Cognitive Sphere
        if retrograde_wave is not None:
            states["cognitive"] = self.retrograde_interference(states["cognitive"], retrograde_wave)

        # 3. Synchronized Internal Pulse across all Spheres
        new_states = {}
        new_past = {}
        for name, organoid in [
            ("sensory", self.s_sensory),
            ("cognitive", self.s_cog),
            ("motor", self.s_motor),
        ] + ([("auxiliary", self.s_aux)] if self.include_aux else []):
            st, pst = organoid.pulse(states[name], past_states_dict[name])
            new_states[name] = st
            new_past[name] = pst

        return new_states, new_past

    def ingest_prompt(self, prompt_tokens: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Ingests sensory tokens exclusively onto South Pole of Sensory Sphere.
        """
        B = prompt_tokens.shape[0]
        device = prompt_tokens.device
        dtype = self.sensory_retina.norm.weight.dtype

        states = self.initialize_lattice(B, device, dtype)
        sensory_field, sensory_memory = self.sensory_retina(prompt_tokens)

        # Ingest onto sensory sphere outermost shell (k = -1), South Pole
        states["sensory"][:, -1, self.s_sensory.core.sensory_nodes, :] = sensory_field.to(dtype)
        return states, sensory_memory

    def forward(
        self,
        prompt_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        thinking_cycles: int = 4,
        enable_retrograde: bool = True,
    ) -> torch.Tensor:
        """
        Forward Pass through the Kissing-Sphere Network:
        1. Ingest Prompt into Sensory Sphere.
        2. Deliberate across the Kissing Lattice for `thinking_cycles` (e.g. 4, 6, 8).
        3. Readout tokens from North Pole of Motor Sphere.
        4. Internal Corollary Discharge into subcortical core.
        """
        B, ans_len = target_tokens.shape
        device = prompt_tokens.device
        dtype = self.sensory_retina.norm.weight.dtype

        states, sensory_memory = self.ingest_prompt(prompt_tokens)
        past_states_dict = {k: [] for k in states.keys()}

        # 1. Forward Deliberation across Kissing Spheres
        retrograde_wave = None
        for cycle_idx in range(thinking_cycles):
            states, past_states_dict = self.pulse_multimodal_lattice(
                states, past_states_dict, retrograde_wave=retrograde_wave
            )
            # If retrograde enabled, synthesize a counter-wave from motor sphere into cognitive
            if enable_retrograde and cycle_idx >= 1:
                retrograde_wave = states["motor"].clone().detach() * 0.30

        # 2. Sequential Token Readout from Motor Sphere
        refractory_trace = torch.zeros(B, self.vocab_size, device=device, dtype=dtype)
        logits_list = []

        for t in range(ans_len):
            motor_field = states["motor"][:, -1, self.s_motor.core.motor_nodes, :]
            logits_t, refractory_trace, corollary = self.motor_broca.forward_readout(
                motor_field, sensory_memory, refractory_trace, step_idx=t
            )
            logits_list.append(logits_t)

            # Corollary Discharge into Core of Cognitive Sphere (k=0, 1)
            coro_expanded = corollary.unsqueeze(1).to(dtype)
            states["cognitive"][:, 0, :, :] += coro_expanded * 0.25
            states["cognitive"][:, 1, :, :] += coro_expanded * 0.15

            # Pulse the entire lattice for next token step
            states, past_states_dict = self.pulse_multimodal_lattice(
                states, past_states_dict, retrograde_wave=None
            )

        return torch.stack(logits_list, dim=1)

    @torch.no_grad()
    def solve_with_active_thinking(
        self,
        prompt_tokens: torch.Tensor,
        thinking_cycles: int = 4,
        max_answer_tokens: int = 36,
        temperature: float = 0.65,
        eos_token_id: Optional[int] = None,
        enable_retrograde: bool = True,
    ) -> torch.Tensor:
        """
        Autonomous Multi-Sphere Inference with Zero External Feedback.
        """
        self.eval()
        B = prompt_tokens.shape[0]
        device = prompt_tokens.device
        dtype = self.sensory_retina.norm.weight.dtype

        states, sensory_memory = self.ingest_prompt(prompt_tokens)
        past_states_dict = {k: [] for k in states.keys()}

        # Deep Deliberation across Kissing Spheres
        retrograde_wave = None
        for cycle_idx in range(thinking_cycles):
            states, past_states_dict = self.pulse_multimodal_lattice(
                states, past_states_dict, retrograde_wave=retrograde_wave
            )
            if enable_retrograde and cycle_idx >= 1:
                retrograde_wave = states["motor"].clone() * 0.30

        refractory_trace = torch.zeros(B, self.vocab_size, device=device, dtype=dtype)
        generated = []

        for step_idx in range(max_answer_tokens):
            motor_field = states["motor"][:, -1, self.s_motor.core.motor_nodes, :]
            logits, refractory_trace, corollary = self.motor_broca.forward_readout(
                motor_field, sensory_memory, refractory_trace, step_idx=step_idx
            )

            if eos_token_id is not None and step_idx < 3:
                logits[..., eos_token_id] -= 50.0

            logits = logits / max(temperature, 1e-4)
            top_k = 40
            val, idx = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(val, dim=-1)
            sampled_rel = torch.multinomial(probs.squeeze(1) if probs.dim() == 3 else probs, num_samples=1)
            next_tok = torch.gather(idx.squeeze(1) if idx.dim() == 3 else idx, -1, sampled_rel)
            generated.append(next_tok)

            if eos_token_id is not None and (next_tok == eos_token_id).all():
                break

            # Corollary Discharge into Core of Cognitive Sphere
            coro_expanded = corollary.unsqueeze(1).to(dtype)
            states["cognitive"][:, 0, :, :] += coro_expanded * 0.25
            states["cognitive"][:, 1, :, :] += coro_expanded * 0.15

            # Pulse the multi-sphere lattice
            states, past_states_dict = self.pulse_multimodal_lattice(
                states, past_states_dict, retrograde_wave=None
            )

        if len(generated) == 0:
            return torch.empty((B, 0), dtype=torch.long, device=device)
        return torch.cat(generated, dim=-1)

    def deliberate_adaptive(
        self,
        states: Dict[str, torch.Tensor],
        min_cycles: int = 2,
        max_cycles: int = 16,
        settling_tolerance: float = 0.05,
        enable_retrograde: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], int]:
        """
        Dynamically pulses the multi-sphere lattice until internal kinetic energy
        in the central cognitive sphere settles below tolerance threshold.
        """
        past_states_dict = {k: [] for k in states.keys()}
        actual_cycles = 0
        retrograde_wave = None

        for cycle in range(max_cycles):
            prev_cog = states["cognitive"].clone()
            states, past_states_dict = self.pulse_multimodal_lattice(
                states, past_states_dict, retrograde_wave=retrograde_wave
            )
            actual_cycles = cycle + 1

            if enable_retrograde and cycle >= 1:
                retrograde_wave = states["motor"].clone().detach() * 0.30

            if cycle >= min_cycles:
                delta_norm = torch.norm(states["cognitive"] - prev_cog)
                base_norm = torch.norm(states["cognitive"]).clamp(min=1e-6)
                kinetic_delta = (delta_norm / base_norm).item()
                if kinetic_delta < settling_tolerance:
                    break

        return states, past_states_dict, actual_cycles

    @torch.no_grad()
    def solve_with_adaptive_emergence(
        self,
        prompt_tokens: torch.Tensor,
        min_cycles: int = 2,
        max_cycles: int = 16,
        settling_tolerance: float = 0.05,
        max_answer_tokens: int = 40,
        temperature: float = 0.65,
        eos_token_id: Optional[int] = None,
        enable_retrograde: bool = True,
    ) -> Tuple[torch.Tensor, int]:
        """
        Autonomous Multi-Sphere Inference with Dynamic Energy Settling.
        Returns: (generated_tokens, settled_deliberation_cycles)
        """
        self.eval()
        B = prompt_tokens.shape[0]
        device = prompt_tokens.device
        dtype = self.sensory_retina.norm.weight.dtype

        states, sensory_memory = self.ingest_prompt(prompt_tokens)
        states, past_states_dict, settled_cycles = self.deliberate_adaptive(
            states,
            min_cycles=min_cycles,
            max_cycles=max_cycles,
            settling_tolerance=settling_tolerance,
            enable_retrograde=enable_retrograde,
        )

        refractory_trace = torch.zeros(B, self.vocab_size, device=device, dtype=dtype)
        generated = []

        for step_idx in range(max_answer_tokens):
            motor_field = states["motor"][:, -1, self.s_motor.core.motor_nodes, :]
            logits, refractory_trace, corollary = self.motor_broca.forward_readout(
                motor_field, sensory_memory, refractory_trace, step_idx=step_idx
            )

            if eos_token_id is not None and step_idx < 3:
                logits[..., eos_token_id] -= 50.0

            logits = logits / max(temperature, 1e-4)
            top_k = 40
            val, idx = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(val, dim=-1)
            sampled_rel = torch.multinomial(probs.squeeze(1) if probs.dim() == 3 else probs, num_samples=1)
            next_tok = torch.gather(idx.squeeze(1) if idx.dim() == 3 else idx, -1, sampled_rel)
            generated.append(next_tok)

            if eos_token_id is not None and (next_tok == eos_token_id).all():
                break

            coro_expanded = corollary.unsqueeze(1).to(dtype)
            states["cognitive"][:, 0, :, :] += coro_expanded * 0.25
            states["cognitive"][:, 1, :, :] += coro_expanded * 0.15

            states, past_states_dict = self.pulse_multimodal_lattice(
                states, past_states_dict, retrograde_wave=None
            )

        if len(generated) == 0:
            return torch.empty((B, 0), dtype=torch.long, device=device), settled_cycles
        return torch.cat(generated, dim=-1), settled_cycles



def build_modular_kissing_lattice() -> ModularKissingLattice:
    """Builds a 4-sphere modular kissing lattice (Sensory, Cognitive, Motor, Auxiliary)."""
    return ModularKissingLattice(
        vocab_size=50257,
        embed_dim=256,
        channels=64,
        num_shells=8,
        nodes_per_shell=128,
        include_aux_sphere=True,
    )
