"""
Real Continuous Geodesic Spheroid (B^3) Neural Architecture
============================================================
Physical 3D Ball & Spherical Geometry:
1. Concentric Fibonacci Geodesic Shells on S^2 (Equidistant, zero polar singularities).
2. Continuous Laplace-Beltrami Tangential Diffusion + Radial Inter-Shell Wave Conduction.
3. Non-Linear Cubic Soliton Stabilization (-lambda * Psi^3) to prevent semantic dispersion.
4. Active Noise-Reduction Autoregressor (suppresses high-frequency thermal dispersion).
5. True Hemispheric Bipolarity:
   - Hemisphere A (South Pole, z < 0, r = 1.0): Sensory Afferent Ingestion.
   - Equatorial Belt (z ≈ 0): Trans-Callosal White-Matter Highway.
   - Hemisphere B (North Pole, z > 0, r = 1.0): Motor Efferent Readout.
   - Deep Subcortical Core (r < 0.25): Corollary Discharge Sink.
6. Strict Zero External Token Feedback: x_{t+1} != [x, y].
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


# =========================================================================
# 1. Geodesic Mesh Engine on S^2
# =========================================================================

def build_fibonacci_geodesic_mesh(num_nodes: int = 128, k_neighbors: int = 7):
    """
    Constructs an equidistant, singularity-free Fibonacci spherical lattice on S^2.
    Precomputes:
    - 3D Coordinates on the unit sphere S^2.
    - Sparse Geodesic Laplace-Beltrami Diffusion Matrix L_S2.
    - Hemisphere A (South Pole: z < -0.05) & Hemisphere B (North Pole: z > 0.05) index sets.
    - Antipodal / Equatorial Mirror Map for trans-callosal axonal bridges.
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0  # Golden ratio
    indices = np.arange(num_nodes)
    z = 1.0 - (2.0 * indices) / max(1, num_nodes - 1)
    r_xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = 2.0 * math.pi * indices / phi

    x = r_xy * np.cos(theta)
    y = r_xy * np.sin(theta)
    coords = torch.tensor(np.stack([x, y, z], axis=1), dtype=torch.float32)

    # Compute pairwise geodesic angular distance: d_ij = acos(x_i . x_j)
    cos_sim = torch.mm(coords, coords.t()).clamp(-1.0, 1.0)
    geodesic_dist = torch.acos(cos_sim)

    # Top-K nearest geodesic neighbors per node
    topk_dist, topk_idx = torch.topk(geodesic_dist, k=min(k_neighbors, num_nodes), largest=False)

    sigma = 0.35
    W = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    for i in range(num_nodes):
        weights = torch.exp(- (topk_dist[i] ** 2) / (2.0 * sigma * sigma))
        W[i, topk_idx[i]] = weights

    # Normalized Laplace-Beltrami Operator: L = D^-1 * W - I
    deg = W.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    L_S2 = (W / deg) - torch.eye(num_nodes, dtype=torch.float32)

    south_pole_indices = (coords[:, 2] < -0.05).nonzero(as_tuple=True)[0]  # Hemisphere A
    north_pole_indices = (coords[:, 2] > 0.05).nonzero(as_tuple=True)[0]   # Hemisphere B
    equator_indices = (coords[:, 2].abs() <= 0.25).nonzero(as_tuple=True)[0]

    # Symmetric antipodal mapping: (x, y, -z) <-> (x, y, +z)
    mirror_coords = coords.clone()
    mirror_coords[:, 2] = -mirror_coords[:, 2]
    cross_dist = torch.cdist(coords, mirror_coords)
    callosal_map = cross_dist.argmin(dim=-1)

    return coords, L_S2, south_pole_indices, north_pole_indices, equator_indices, callosal_map


# =========================================================================
# 2. Noise-Reduction Autoregressor (Active Denoising Filter)
# =========================================================================

class NoiseReductionAutoregressor(nn.Module):
    """
    Active Denoising Filter operating on the pulsating wave state:
    - Maintains an autoregressive moving average of coherent spatial modes.
    - Isolates chaotic stochastic dispersion from the coherent reasoning wave.
    - Suppresses incoherent high-frequency noise while preserving sharp symbolic edges.
    """
    def __init__(self, channels: int = 64, order: int = 2):
        super().__init__()
        self.channels = channels
        self.order = order
        # Autoregressive temporal filter weights
        self.ar_weights = nn.Parameter(torch.tensor([0.65, 0.25], dtype=torch.float32).view(1, 1, 1, order))
        self.noise_gate = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.SiLU(),
            nn.Linear(channels // 2, channels),
            nn.Sigmoid()
        )

    def filter_step(self, current_state: torch.Tensor, past_states: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        current_state: [B, K, M, C]
        past_states: list of previous [B, K, M, C] states
        Returns filtered_state, updated past_states
        """
        if len(past_states) < self.order:
            past_states = [current_state.clone()] + past_states
            return current_state, past_states

        # Stack past states: [B, K, M, C, order]
        stacked = torch.stack(past_states[:self.order], dim=-1)
        w = F.softmax(self.ar_weights, dim=-1)
        coherent_prediction = (stacked * w.unsqueeze(3)).sum(dim=-1)

        # Residual high-frequency noise
        noise_residual = current_state - coherent_prediction
        damping_factor = self.noise_gate(noise_residual)

        # Non-linear noise cancellation: keep coherent component, damp uncorrelated jitter
        filtered_state = coherent_prediction + (1.0 - damping_factor) * torch.tanh(noise_residual)
        updated_past = [filtered_state.detach()] + past_states[:self.order - 1]
        return filtered_state, updated_past


class PulsationMicroStabilizer(nn.Module):
    """
    Active Micro-Adjustment & Homeostatic Energy Stabilizer:
    1. Spatial Laplacian micro-smoothing on S^2: dissipates high-frequency localized phase turbulence.
    2. Homeostatic Energy Normalization: continuously regulates baseline metabolic energy E_0,
       preventing synaptic quiescence (collapse to 0) and epileptiform runaway excitation.
    3. Soft Lyapunov saturation clamp: guarantees long-term numerical boundedness during infinite pulsation.
    """
    def __init__(self, channels: int = 64, target_energy: float = 0.40):
        super().__init__()
        self.channels = channels
        self.target_energy = target_energy
        self.micro_damping = nn.Parameter(torch.tensor(0.08, dtype=torch.float32))
        self.homeostatic_rate = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def forward(self, state_3d: torch.Tensor, L_S2: torch.Tensor) -> torch.Tensor:
        B, K, M, C = state_3d.shape
        dtype = state_3d.dtype

        # 1. Spatial Laplacian micro-dissipation on S^2 shells
        flat = state_3d.view(B * K, M, C)
        laplacian = torch.matmul(L_S2.to(dtype), flat).view(B, K, M, C)
        eta = torch.sigmoid(self.micro_damping).to(dtype) * 0.15
        state_smooth = state_3d + eta * laplacian

        # 2. Homeostatic energy clamp
        current_energy = (state_smooth ** 2).mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
        target_e = torch.tensor(self.target_energy, device=state_3d.device, dtype=dtype)
        scale = torch.sqrt(target_e / current_energy)
        rate = torch.sigmoid(self.homeostatic_rate).to(dtype)
        bounded_scale = 1.0 + rate * (scale.clamp(0.75, 1.30) - 1.0)
        state_regulated = state_smooth * bounded_scale

        # 3. Soft Lyapunov saturation clamp
        return torch.tanh(state_regulated / 3.0) * 3.0


# =========================================================================
# 3. Real Geodesic Spheroid 3D Brain Core (B^3)
# =========================================================================

class RealGeodesicSpheroidCore(nn.Module):
    """
    Physical 3D Spheroid Ball Manifold:
    State tensor S_t in R^{B x K_shells x M_nodes x C_channels}
    Governed by the Non-Linear Klein-Gordon Soliton Equation on S^2 x R.
    """
    def __init__(
        self,
        num_shells: int = 8,
        nodes_per_shell: int = 128,
        channels: int = 64,
        soliton_lambda: float = 0.08,
    ):
        super().__init__()
        self.K = num_shells
        self.M = nodes_per_shell
        self.C = channels
        self.soliton_lambda = soliton_lambda

        # Precompute Geodesic Sphere Topology
        coords, L_S2, south_idx, north_idx, eq_idx, callosal_map = build_fibonacci_geodesic_mesh(
            num_nodes=nodes_per_shell, k_neighbors=7
        )
        self.register_buffer("coords", coords)
        self.register_buffer("L_S2", L_S2)
        self.register_buffer("sensory_nodes", south_idx)
        self.register_buffer("motor_nodes", north_idx)
        self.register_buffer("equator_nodes", eq_idx)
        self.register_buffer("callosal_map", callosal_map)

        # 1. Tangential Surface Diffusion Transform
        self.tangent_proj = nn.Linear(channels, channels, bias=False)
        self.tangent_norm = nn.LayerNorm(channels)

        # 2. Radial Inter-Shell Wave Conduction (1D finite difference / conv along shells)
        self.radial_wave_conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.radial_norm = nn.LayerNorm(channels)

        # 3. Trans-Callosal Equatorial White-Matter Highway
        self.callosal_bridge = nn.Linear(channels, channels, bias=False)

        # 4. Inertial Damping Gate (Unit-Gain Stability)
        self.inertia_gate = nn.Parameter(torch.zeros(1, num_shells, 1, channels))

        # 5. Active Noise-Reduction Autoregressor
        self.noise_filter = NoiseReductionAutoregressor(channels=channels, order=2)

        # 6. Active Micro-Adjustment & Homeostatic Stabilizer
        self.micro_stabilizer = PulsationMicroStabilizer(channels=channels, target_energy=0.40)

        # 7. Spontaneous Endogenous Oscillators (Living Resting-State Pacemaker)
        self.rhythm_freq = nn.Parameter(torch.tensor([0.15, 0.35], dtype=torch.float32))
        self.spontaneous_amp = nn.Parameter(torch.tensor(0.04, dtype=torch.float32))

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.K, self.M, self.C, device=device, dtype=dtype)

    def init_living_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Initializes an active living resting-state manifold with spontaneous endogenous oscillations.
        Simulates the baseline resting-state network (Default Mode Network) of a living neural system.
        """
        quiescent = self.init_state(batch_size, device, dtype)
        state, _ = self.tick_living_dynamics(quiescent, t_step=0, spontaneous_noise=False)
        return state

    def tick_living_dynamics(
        self, state_3d: torch.Tensor, t_step: int = 0, spontaneous_noise: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Advances the 3D unit ball by one living resting-state clock tick (spontaneous theta/alpha waves).
        Continuously maintains metabolic homeostatic activity even without sensory input.
        """
        B, K, M, C = state_3d.shape
        device = state_3d.device
        dtype = state_3d.dtype

        # Radial depth phase distribution across concentric shells
        shell_depths = torch.linspace(0.0, 1.0, K, device=device, dtype=torch.float32).view(1, K, 1, 1)
        w1, w2 = self.rhythm_freq[0], self.rhythm_freq[1]
        t_val = float(t_step)
        osc = torch.sin(w1 * t_val + shell_depths * math.pi) + 0.5 * torch.cos(w2 * t_val + shell_depths * 2.0 * math.pi)
        amp = self.spontaneous_amp.clamp(min=0.01, max=0.20)
        rhythm_field = osc.expand(B, K, M, C) * amp

        if spontaneous_noise and self.training:
            noise = torch.randn_like(rhythm_field) * 0.005
            rhythm_field = rhythm_field + noise

        perturbed = state_3d + rhythm_field.to(dtype)
        # Pulse through the continuous 3D volume
        return self.pulse_wave_step(perturbed)

    def pulse_wave_step(
        self,
        S_t: torch.Tensor,
        past_states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Executes one continuous pulsation cycle across the 3D unit ball B^3.
        S_t shape: [B, K, M, C]
        """
        B, K, M, C = S_t.shape

        # 1. Surface Tangential Diffusion (Laplace-Beltrami across S^2)
        # S_t reshaped to [B * K, M, C]
        S_flat = S_t.view(B * K, M, C)
        diffused_S2 = torch.matmul(self.L_S2.to(S_t.dtype), S_flat).view(B, K, M, C)
        F_tangent = self.tangent_norm(self.tangent_proj(diffused_S2))

        # 2. Radial Conduction (between shells k=0..K-1)
        # Permute to [B * M, C, K] for 1D radial finite-difference convolution
        S_radial_in = S_t.permute(0, 2, 3, 1).contiguous().view(B * M, C, K)
        F_radial = self.radial_wave_conv(S_radial_in).view(B, M, C, K).permute(0, 3, 1, 2).contiguous()
        F_radial = self.radial_norm(F_radial)

        # 3. Trans-Callosal Highway (Equatorial antipodal resonance)
        S_mirror = S_t[:, :, self.callosal_map, :]
        F_callosal = self.callosal_bridge(S_mirror)

        # 4. Non-Linear Cubic Soliton Self-Focusing (-lambda * S^3)
        # Prevents wave dispersion, locking mathematical and symbolic packets
        soliton_compression = -self.soliton_lambda * (S_t ** 3).clamp(-10.0, 10.0)

        # 5. Total Combined Wave Acceleration Field
        F_total = torch.tanh(F_tangent + F_radial + 0.35 * F_callosal + soliton_compression)

        # 6. Contractive Inertial Damping
        gamma = torch.sigmoid(self.inertia_gate.to(S_t.dtype))
        S_raw_next = gamma * S_t + (1.0 - gamma) * F_total

        # 7. Active Noise-Reduction Autoregressive Filter
        if past_states is None:
            past_states = []
        S_clean_next, updated_past = self.noise_filter.filter_step(S_raw_next, past_states)

        # 8. Active Micro-Adjustment & Homeostatic Energy Stabilization
        S_stabilized = self.micro_stabilizer(S_clean_next, self.L_S2)

        return S_stabilized, updated_past


# =========================================================================
# 4. Sensory Afferent (Hemisphere A) & Motor Efferent (Hemisphere B)
# =========================================================================

class GeodesicSensoryAfferent(nn.Module):
    """
    Ingests 1D prompt tokens and projects them exclusively onto the
    curved surface of Hemisphere A (South Pole: z < -0.05, outermost shell r = 1.0).
    """
    def __init__(self, vocab_size: int, embed_dim: int, channels: int, num_sensory_nodes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(num_sensory_nodes)
        self.norm = nn.LayerNorm(channels)

    def forward(self, prompt_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
        - sensory_surface_field: [B, num_sensory_nodes, channels]
        - sensory_memory: [B, L, embed_dim] for Arcuate Fasciculus cross-resonance
        """
        x_emb = self.embedding(prompt_tokens)  # [B, L, D]
        x_trans = x_emb.transpose(1, 2)         # [B, D, L]
        h = F.gelu(self.conv1(x_trans))
        h = F.gelu(self.conv2(h))
        sensory_nodes_field = self.pool(h).transpose(1, 2)  # [B, num_sensory_nodes, C]
        sensory_nodes_field = self.norm(sensory_nodes_field)
        return sensory_nodes_field, x_emb


class GeodesicMotorEfferent(nn.Module):
    """
    Decodes language exclusively from the curved surface of
    Hemisphere B (North Pole: z > 0.05, outermost shell r = 1.0).
    Equipped with:
    - Arcuate Fasciculus Wernicke-Broca Attention Resonance
    - Corollary Discharge Projection into Deep Subcortical Core (r < 0.25)
    - Biological Refractory Trace (anti-repetition)
    Strict Zero External Token Feedback (x_{t+1} != [x, y]).
    """
    def __init__(self, channels: int, embed_dim: int, vocab_size: int, num_motor_nodes: int):
        super().__init__()
        self.num_motor_nodes = num_motor_nodes
        self.channels = channels
        self.vocab_size = vocab_size

        self.motor_pool = nn.Linear(num_motor_nodes * channels, channels * 2)
        self.q_proj = nn.Linear(channels * 2, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.af_norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(channels * 2 + embed_dim, vocab_size)

        # Corollary Discharge projection back into subcortical core (k=0, 1)
        self.corollary_proj = nn.Linear(channels * 2 + embed_dim, channels)

    def forward_readout(
        self,
        hemisphere_b_field: torch.Tensor,
        sensory_memory: torch.Tensor,
        refractory_trace: torch.Tensor,
        step_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        hemisphere_b_field: [B, num_motor_nodes, C]
        sensory_memory: [B, L, D]
        refractory_trace: [B, vocab_size]
        """
        B = hemisphere_b_field.shape[0]
        motor_flat = hemisphere_b_field.reshape(B, -1)
        motor_feat = F.silu(self.motor_pool(motor_flat))  # [B, C*2]

        # Arcuate Fasciculus Cross-Resonance
        Q = self.q_proj(motor_feat).unsqueeze(1)          # [B, 1, D]
        K = self.k_proj(sensory_memory)                   # [B, L, D]
        V = self.v_proj(sensory_memory)                   # [B, L, D]

        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(Q.shape[-1])
        attn = F.softmax(scores, dim=-1)
        af_resonance = self.af_norm(torch.bmm(attn, V).squeeze(1))  # [B, D]

        # Combined Motor Logits
        joint = torch.cat([motor_feat, af_resonance], dim=-1)
        raw_logits = self.head(joint)

        # Biological Refractory Adaptation (suppresses repeating tokens without autoregression)
        adapted_logits = raw_logits - 1.8 * refractory_trace

        # Update refractory trace with hard action potential spike
        top_tok = torch.argmax(raw_logits.detach(), dim=-1)
        spike = torch.zeros_like(refractory_trace)
        spike.scatter_(1, top_tok.unsqueeze(1), 2.5)
        updated_refractory = 0.70 * refractory_trace + 0.30 * spike

        # Compute corollary discharge vector to inject into core
        corollary_vector = torch.tanh(self.corollary_proj(joint))  # [B, C]

        return adapted_logits, updated_refractory, corollary_vector


# =========================================================================
# 5. Master Real Continuous Geodesic Spheroid Model
# =========================================================================

class RealGeodesicSpheroid(nn.Module):
    """
    True Continuous 3D Geodesic Brain Manifold:
    - B^3 Unit Ball with K concentric Fibonacci S^2 shells.
    - Hemisphere A (Sensory) -> 3D Volume -> Hemisphere B (Motor).
    - Internal Corollary Discharge into Subcortical Core.
    - Active Noise Reduction Autoregressor.
    - Strict Zero Output Feedback: x_{t+1} != [x, y].
    """
    def __init__(
        self,
        vocab_size: int = 50257,
        num_shells: int = 8,
        nodes_per_shell: int = 128,
        channels: int = 64,
        embed_dim: int = 256,
        soliton_lambda: float = 0.08,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_shells = num_shells
        self.nodes_per_shell = nodes_per_shell
        self.channels = channels

        # 3D Geodesic Ball Core
        self.core = RealGeodesicSpheroidCore(
            num_shells=num_shells,
            nodes_per_shell=nodes_per_shell,
            channels=channels,
            soliton_lambda=soliton_lambda,
        )

        num_sensory = len(self.core.sensory_nodes)
        num_motor = len(self.core.motor_nodes)

        # Sensory Afferent Pole (Hemisphere A)
        self.sensory_pole = GeodesicSensoryAfferent(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            channels=channels,
            num_sensory_nodes=num_sensory,
        )

        # Motor Efferent Pole (Hemisphere B)
        self.motor_pole = GeodesicMotorEfferent(
            channels=channels,
            embed_dim=embed_dim,
            vocab_size=vocab_size,
            num_motor_nodes=num_motor,
        )

    def init_brain_state(self, batch_size: int, device: torch.device, living: bool = True) -> torch.Tensor:
        dtype = self.core.inertia_gate.dtype
        if living:
            return self.core.init_living_state(batch_size, device, dtype)
        return self.core.init_state(batch_size, device, dtype)

    def live_heartbeat_tick(self, state_3d: torch.Tensor, t_step: int = 0) -> torch.Tensor:
        """Continuously pulses the 3D sphere at resting state with micro-adjustment & stabilization."""
        next_state, _ = self.core.tick_living_dynamics(state_3d, t_step=t_step, spontaneous_noise=False)
        return next_state

    def ingest_stimulus(
        self,
        prompt_tokens: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        living: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Injects sensory energy exclusively into Hemisphere A surface nodes
        on the outermost shell (k = K - 1), perturbing the active living resting-state.
        """
        B = prompt_tokens.shape[0]
        dtype = self.core.inertia_gate.dtype
        device = prompt_tokens.device

        if initial_state is not None:
            state_3d = initial_state.clone()
        elif living:
            state_3d = self.core.init_living_state(B, device, dtype)
        else:
            state_3d = self.core.init_state(B, device, dtype)

        sensory_field, sensory_memory = self.sensory_pole(prompt_tokens)

        # Inward boundary condition on Hemisphere A (outer shell k = -1)
        state_3d[:, -1, self.core.sensory_nodes, :] += sensory_field.to(dtype)
        return state_3d, sensory_memory

    def deliberate(
        self,
        state_3d: torch.Tensor,
        cycles: int = 3,
    ) -> torch.Tensor:
        """
        Permits internal 3D wave diffusion across the corpus callosum
        before emitting any tokens (Active Deliberation / Zero Token Emission).
        """
        past_states = []
        for _ in range(cycles):
            state_3d, past_states = self.core.pulse_wave_step(state_3d, past_states)
        return state_3d

    def emit_next_token(
        self,
        state_3d: torch.Tensor,
        sensory_memory: torch.Tensor,
        refractory_trace: torch.Tensor,
        past_states: Optional[List[torch.Tensor]] = None,
        step_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Emits token logits from Hemisphere B surface, injects corollary discharge
        into subcortical core, and pulses the 3D volume for the next step.
        """
        # Readout from Hemisphere B on the outermost cortical shell (k = -1)
        hemi_b_field = state_3d[:, -1, self.core.motor_nodes, :]
        logits, updated_refractory, corollary_vector = self.motor_pole.forward_readout(
            hemi_b_field, sensory_memory, refractory_trace, step_idx=step_idx
        )

        # Internal Corollary Discharge: Motor signal projected into deep core (k = 0, 1)
        state_3d = state_3d.clone()
        state_3d[:, 0, :, :] += corollary_vector.unsqueeze(1) * 0.25
        state_3d[:, 1, :, :] += corollary_vector.unsqueeze(1) * 0.15

        # Pulse the continuous 3D volume for the next token step
        state_3d, past_states = self.core.pulse_wave_step(state_3d, past_states)

        return logits, state_3d, updated_refractory, past_states

    def forward(
        self,
        prompt_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        thinking_cycles: int = 2,
    ) -> torch.Tensor:
        """
        Teacher-forced forward training pass with zero external feedback.
        """
        B, ans_len = target_tokens.shape
        device = prompt_tokens.device
        dtype = self.core.inertia_gate.dtype

        # 1. Ingest Stimulus onto Hemisphere A
        state_3d, sensory_memory = self.ingest_stimulus(prompt_tokens)

        # 2. Deliberate across Geodesic Corpus Callosum
        past_states = []
        for _ in range(thinking_cycles):
            state_3d, past_states = self.core.pulse_wave_step(state_3d, past_states)

        refractory_trace = torch.zeros(B, self.vocab_size, device=device, dtype=dtype)
        logits_list = []

        # 3. Readout on Hemisphere B with Internal Corollary Discharge
        for t in range(ans_len):
            logits_t, state_3d, refractory_trace, past_states = self.emit_next_token(
                state_3d, sensory_memory, refractory_trace, past_states=past_states, step_idx=t
            )
            logits_list.append(logits_t)

        return torch.stack(logits_list, dim=1)

    @torch.no_grad()
    def solve_with_active_thinking(
        self,
        prompt_tokens: torch.Tensor,
        thinking_cycles: int = 3,
        max_answer_tokens: int = 40,
        temperature: float = 0.65,
        repetition_penalty: float = 1.3,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autonomous biological inference:
        Zero token feedback into the prompt retina.
        """
        self.eval()
        B = prompt_tokens.shape[0]
        device = prompt_tokens.device
        dtype = self.core.inertia_gate.dtype

        state_3d, sensory_memory = self.ingest_stimulus(prompt_tokens)

        past_states = []
        for _ in range(thinking_cycles):
            state_3d, past_states = self.core.pulse_wave_step(state_3d, past_states)

        refractory_trace = torch.zeros(B, self.vocab_size, device=device, dtype=dtype)
        generated = []

        for step_idx in range(max_answer_tokens):
            logits, state_3d, refractory_trace, past_states = self.emit_next_token(
                state_3d, sensory_memory, refractory_trace, past_states=past_states, step_idx=step_idx
            )

            # Mask EOS for the first 3 tokens
            if eos_token_id is not None and step_idx < 3:
                logits[..., eos_token_id] -= 50.0

            logits = logits / max(temperature, 1e-4)

            # Nucleus top-k sampling
            top_k = 40
            val, idx = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(val, dim=-1)
            sampled_rel = torch.multinomial(probs.squeeze(1) if probs.dim() == 3 else probs, num_samples=1)
            next_tok = torch.gather(idx.squeeze(1) if idx.dim() == 3 else idx, -1, sampled_rel)

            generated.append(next_tok)
            if eos_token_id is not None and (next_tok == eos_token_id).all():
                break

        if len(generated) == 0:
            return torch.empty((B, 0), dtype=torch.long, device=device)
        return torch.cat(generated, dim=-1)

    def deliberate_adaptive(
        self,
        state_3d: torch.Tensor,
        min_cycles: int = 2,
        max_cycles: int = 16,
        settling_tolerance: float = 0.05,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], int]:
        """
        Dynamically pulses the 3D unit ball until the internal kinetic wave energy
        settles into harmonic attractor equilibrium (||Psi_t - Psi_{t-1}|| / ||Psi_t|| < tol).
        """
        past_states = []
        actual_cycles = 0
        for cycle in range(max_cycles):
            prev_state = state_3d.clone()
            state_3d, past_states = self.core.pulse_wave_step(state_3d, past_states)
            actual_cycles = cycle + 1
            if cycle >= min_cycles:
                delta_norm = torch.norm(state_3d - prev_state)
                base_norm = torch.norm(state_3d).clamp(min=1e-6)
                kinetic_delta = (delta_norm / base_norm).item()
                if kinetic_delta < settling_tolerance:
                    break
        return state_3d, past_states, actual_cycles

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
    ) -> Tuple[torch.Tensor, int]:
        """
        Autonomous biological inference with dynamic energy settling:
        Pulsates internally until the physical wave settles into equilibrium before token emission.
        Returns: (generated_tokens, settled_deliberation_cycles)
        """
        self.eval()
        B = prompt_tokens.shape[0]
        device = prompt_tokens.device
        dtype = self.core.inertia_gate.dtype

        state_3d, sensory_memory = self.ingest_stimulus(prompt_tokens)
        state_3d, past_states, settled_cycles = self.deliberate_adaptive(
            state_3d, min_cycles=min_cycles, max_cycles=max_cycles, settling_tolerance=settling_tolerance
        )

        refractory_trace = torch.zeros(B, self.vocab_size, device=device, dtype=dtype)
        generated = []

        for step_idx in range(max_answer_tokens):
            logits, state_3d, refractory_trace, past_states = self.emit_next_token(
                state_3d, sensory_memory, refractory_trace, past_states=past_states, step_idx=step_idx
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

        if len(generated) == 0:
            return torch.empty((B, 0), dtype=torch.long, device=device), settled_cycles
        return torch.cat(generated, dim=-1), settled_cycles


def build_real_geodesic_spheroid() -> RealGeodesicSpheroid:
    """Builds a real continuous geodesic spheroid (B^3) with 1,024 3D spatial nodes."""
    return RealGeodesicSpheroid(
        vocab_size=50257,
        num_shells=8,
        nodes_per_shell=128,
        channels=64,
        embed_dim=256,
        soliton_lambda=0.08,
    )
