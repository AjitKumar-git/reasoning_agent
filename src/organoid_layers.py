"""
Biological & Mathematical Cortical Layers for Living Organoid Spheres (B^3)
=============================================================================
Layers designed to be applied on top of the Real Continuous Geodesic Spheroid:
1. CorticalColumnarLayer:
   - Vertically integrated laminar hyper-columns across concentric radial shells.
   - Geodesic Difference-of-Gaussians (DoG / Mexican Hat) lateral inhibition on S^2.
   - Prevents spatial wave smearing and creates sharp topological concept attractors.

2. ThalamicGatewayLayer:
   - Reticular dynamic arousal & salience gating for afferent sensory waves.
   - Modulated by the deep subcortical core state (S_core).
   - Filters ambient noise and amplifies salient stimulus features before cortical diffusion.

3. PhaseAmplitudeCouplingLayer (PAC):
   - Cross-frequency coupling: slow theta carrier rhythm modulates fast gamma cognitive solitons.
   - Replaces artificial positional tokens with continuous biological phase clocks.
   - Binds multi-step reasoning steps into coherent syntactic structures.

4. AstrocyticNeuromodulatorField:
   - Slow tripartite glial chemical potential field (Ca2+ / gliotransmitters).
   - Regulates localized synaptic gain and homeostatic plasticity.
   - Prevents epileptiform runaway excitation while sensitizing quiescent reasoning paths.

5. HippocampalAttractorLayer:
   - Modern continuous energy attractor memory.
   - Sharp-Wave Ripple (SWR) resting-state replay into subcortical Shell 0.
   - Provides zero-shot working memory persistence across deliberative cycles.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict

from src.real_geodesic_sphere import (
    build_fibonacci_geodesic_mesh,
    RealGeodesicSpheroidCore,
    GeodesicSensoryAfferent,
    GeodesicMotorEfferent,
)


# =========================================================================
# 1. Cortical Columnar & Lateral Inhibition Layer
# =========================================================================

class CorticalColumnarLayer(nn.Module):
    """
    Cortical Mini-Column Projection & Spherical Difference-of-Gaussians (DoG)
    Lateral Inhibition Operator:
    - Biological cortex is organized in radial mini-columns traversing Layers I-VI (Shells 0..K-1).
    - Local recurrent GABAergic interneurons produce a Mexican-Hat lateral inhibition profile:
      short-range excitation surrounded by long-range inhibitory suppression.
    - Mathematical formulation on S^2:
        I_DoG = I - alpha * L_S2 + beta * (L_S2 @ L_S2)
    """
    def __init__(self, num_shells: int = 8, nodes_per_shell: int = 128, channels: int = 64):
        super().__init__()
        self.K = num_shells
        self.M = nodes_per_shell
        self.C = channels

        # Trainable radial laminar integration weights (Layers I through VI)
        self.laminar_weights = nn.Parameter(torch.ones(num_shells, 1, 1) / math.sqrt(num_shells))
        self.column_transform = nn.Linear(channels, channels, bias=False)
        self.norm = nn.LayerNorm(channels)

        # Lateral inhibition parameters (Mexican Hat profile)
        self.alpha_excite = nn.Parameter(torch.tensor(0.18, dtype=torch.float32))
        self.beta_inhibit = nn.Parameter(torch.tensor(0.06, dtype=torch.float32))

    def forward(self, state_3d: torch.Tensor, L_S2: torch.Tensor) -> torch.Tensor:
        """
        state_3d: [B, K, M, C]
        L_S2: [M, M] Laplace-Beltrami operator on S^2
        Returns sharpened state_3d: [B, K, M, C]
        """
        B, K, M, C = state_3d.shape
        dtype = state_3d.dtype

        # 1. Integrate radial mini-columns: [B, M, C]
        normed_laminar = F.softmax(self.laminar_weights, dim=0).to(dtype)
        column_rep = (state_3d * normed_laminar).sum(dim=1)  # [B, M, C]
        column_proj = self.norm(self.column_transform(column_rep))

        # 2. Compute Bi-harmonic Operator on S^2: L^2 = L @ L
        L_f = L_S2.to(dtype)
        L2_f = torch.matmul(L_f, L_f)

        # 3. Apply Difference-of-Gaussians (DoG) filter across surface nodes
        L_act = torch.matmul(L_f, column_proj)   # [B, M, C]
        L2_act = torch.matmul(L2_f, column_proj) # [B, M, C]

        alpha = torch.sigmoid(self.alpha_excite).to(dtype) * 0.35
        beta = torch.sigmoid(self.beta_inhibit).to(dtype) * 0.15
        dog_response = column_proj + alpha * L_act - beta * L2_act

        # 4. Soft non-linear winner-take-all / lateral suppression
        sharp_column = torch.tanh(dog_response)

        # 5. Modulate outer cortical shells (k >= K // 2)
        modulated = state_3d.clone()
        for k in range(K // 2, K):
            factor = float(k - K // 2 + 1) / float(K // 2)
            modulated[:, k, :, :] = modulated[:, k, :, :] + factor * 0.20 * sharp_column

        return modulated


# =========================================================================
# 2. Thalamocortical Dynamic Gating Layer
# =========================================================================

class ThalamicGatewayLayer(nn.Module):
    """
    Thalamocortical Dynamic Arousal & Sensory Salience Gateway:
    - Sensory afferents pass through the thalamus before reaching cortex.
    - Gates incoming sensory stimulus based on subcortical core state.
    - Amplifies salient concepts, suppresses ambient noise.
    """
    def __init__(self, channels: int = 64):
        super().__init__()
        self.channels = channels
        self.sensory_proj = nn.Linear(channels, channels)
        self.core_proj = nn.Linear(channels, channels)
        self.gate_mlp = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.SiLU(),
            nn.Linear(channels // 2, channels),
            nn.Sigmoid()
        )
        self.arousal_bias = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, sensory_field: torch.Tensor, core_state: torch.Tensor) -> torch.Tensor:
        """
        sensory_field: [B, M_sens, C]
        core_state: [B, C]
        Returns gated sensory_field: [B, M_sens, C]
        """
        M_sens = sensory_field.shape[1]
        dtype = sensory_field.dtype
        core_expanded = self.core_proj(core_state).unsqueeze(1).expand(-1, M_sens, -1)
        sens_proj = self.sensory_proj(sensory_field)

        resonance = sens_proj * core_expanded + self.arousal_bias.to(dtype)
        gate = self.gate_mlp(resonance)
        return sensory_field * (0.5 + 0.5 * gate)


# =========================================================================
# 3. Cross-Frequency Phase-Amplitude Coupling (PAC) Layer
# =========================================================================

class PhaseAmplitudeCouplingLayer(nn.Module):
    """
    Theta-Gamma Cross-Frequency Phase-Amplitude Coupler:
    - Slow carrier rhythm (theta 4-8 Hz) modulates high-frequency gamma soliton waves.
    - Binds multi-step reasoning thoughts into coherent syntactic structures.
    """
    def __init__(self, channels: int = 64, num_frequency_bins: int = 4):
        super().__init__()
        self.channels = channels
        self.carrier_freqs = nn.Parameter(torch.tensor([0.08, 0.20, 0.45, 0.90], dtype=torch.float32))
        self.phase_offsets = nn.Parameter(torch.zeros(num_frequency_bins, channels))
        self.pac_coupling_gain = nn.Parameter(torch.tensor(0.30, dtype=torch.float32))
        self.envelope_transform = nn.Linear(channels, channels)

    def forward(self, state_3d: torch.Tensor, deliberation_step: int) -> torch.Tensor:
        B, K, M, C = state_3d.shape
        t = float(deliberation_step)
        device = state_3d.device
        dtype = state_3d.dtype

        phases = self.carrier_freqs.unsqueeze(1) * t + self.phase_offsets
        oscillations = torch.cos(phases).sum(dim=0)
        envelope = 1.0 + torch.tanh(self.pac_coupling_gain) * torch.tanh(self.envelope_transform(oscillations))
        envelope = envelope.view(1, 1, 1, C).to(device=device, dtype=dtype)
        return state_3d * envelope


# =========================================================================
# 4. Astrocytic Glial Neuromodulator Field
# =========================================================================

class AstrocyticNeuromodulatorField(nn.Module):
    """
    Slow Astrocytic Metabolic & Tripartite Neuromodulatory Field:
    - Glial astrocytes diffuse chemical potential (Ca2+) at slow timescales.
    - Prevents runaway excitation and regulates local synaptic gain.
    """
    def __init__(self, num_shells: int = 8, nodes_per_shell: int = 128, field_dim: int = 16, channels: int = 64):
        super().__init__()
        self.K = num_shells
        self.M = nodes_per_shell
        self.field_dim = field_dim
        self.channels = channels

        self.decay_rate = nn.Parameter(torch.tensor(0.85, dtype=torch.float32))
        self.diffusion_rate = nn.Parameter(torch.tensor(0.12, dtype=torch.float32))
        self.activity_to_glial = nn.Linear(channels, field_dim, bias=False)
        self.glial_to_gain = nn.Linear(field_dim, channels)

    def init_field(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.K, self.M, self.field_dim, device=device, dtype=dtype)

    def update_field(
        self,
        current_field: torch.Tensor,
        wave_state: torch.Tensor,
        L_S2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, K, M, C = wave_state.shape
        dtype = wave_state.dtype

        activity = self.activity_to_glial(torch.tanh(wave_state ** 2))
        flat_field = current_field.view(B * K, M, self.field_dim)
        diffused_field = torch.matmul(L_S2.to(dtype), flat_field).view(B, K, M, self.field_dim)

        gamma = torch.sigmoid(self.decay_rate).to(dtype)
        diff_k = torch.sigmoid(self.diffusion_rate).to(dtype) * 0.25

        updated_field = gamma * current_field + (1.0 - gamma) * activity + diff_k * diffused_field
        updated_field = torch.clamp(updated_field, -2.0, 2.0)

        raw_gain = self.glial_to_gain(updated_field)
        synaptic_gain = 0.70 + 0.60 * torch.sigmoid(raw_gain)

        return updated_field, synaptic_gain


# =========================================================================
# 5. Hippocampal Attractor & Sharp-Wave Ripple Replay Layer
# =========================================================================

class HippocampalAttractorLayer(nn.Module):
    """
    Hippocampal CA3 Recurrent Attractor & Sharp-Wave Ripple (SWR) Replay Layer:
    - Stores peak cognitive wave resonances during deliberation.
    - Replays compressed memory traces into subcortical Shell 0.
    """
    def __init__(self, memory_slots: int = 32, channels: int = 64):
        super().__init__()
        self.memory_slots = memory_slots
        self.channels = channels

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

        self.temperature = nn.Parameter(torch.tensor(math.sqrt(channels), dtype=torch.float32))
        self.swr_gain = nn.Parameter(torch.tensor(0.20, dtype=torch.float32))

    def init_memory(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.memory_slots, self.channels, device=device, dtype=dtype)

    def write_attractor(self, memory_bank: torch.Tensor, core_state: torch.Tensor) -> torch.Tensor:
        new_slot = self.norm(core_state).unsqueeze(1)
        return torch.cat([new_slot, memory_bank[:, :-1, :]], dim=1)

    def sharp_wave_replay(self, memory_bank: torch.Tensor, query_state: torch.Tensor) -> torch.Tensor:
        B, slots, C = memory_bank.shape
        q = self.q_proj(query_state).unsqueeze(1)
        k = self.k_proj(memory_bank)
        v = self.v_proj(memory_bank)

        temp = self.temperature.clamp(min=1.0)
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / temp
        attn_weights = F.softmax(attn_scores, dim=-1)

        recalled = torch.bmm(attn_weights, v).squeeze(1)
        gain = torch.tanh(self.swr_gain)
        return gain * self.out_proj(recalled)


# =========================================================================
# 6. Unified Enhanced Living Organoid Spheroid (Integrated Architecture)
# =========================================================================

class EnhancedOrganoidSpheroid(nn.Module):
    """
    Top-Level Living Organoid Architecture integrating 5 biological layers:
    1. Geodesic Sensory Afferent Ingestion (Hemisphere A, South Pole)
    2. Thalamic Gateway Layer (Sensory Filtering & Dynamic Arousal)
    3. Astrocytic Glial Neuromodulation Field (Ca2+ diffusion & gain control)
    4. 3D Geodesic Spheroid Ball Wave Core (Laplace-Beltrami + Radial Conduction + Cubic Soliton)
    5. Cortical Columnar Layer (Mini-Columns + DoG Lateral Inhibition on S^2)
    6. Cross-Frequency PAC Modulator (Theta-Gamma nested clock)
    7. Hippocampal Attractor & SWR Replay into Deep Subcortical Shell 0
    8. Geodesic Motor Efferent Decoder (Hemisphere B, North Pole)
    9. Internal Corollary Discharge (Efference Copy) preserving x_{t+1} != [x, y]
    """
    def __init__(
        self,
        num_shells: int = 8,
        nodes_per_shell: int = 128,
        channels: int = 64,
        embed_dim: int = 256,
        vocab_size: int = 50257,
        memory_slots: int = 32,
        soliton_lambda: float = 0.08,
    ):
        super().__init__()
        self.num_shells = num_shells
        self.nodes_per_shell = nodes_per_shell
        self.channels = channels
        self.vocab_size = vocab_size

        # 3D Geodesic Ball Wave Core
        self.core = RealGeodesicSpheroidCore(
            num_shells=num_shells,
            nodes_per_shell=nodes_per_shell,
            channels=channels,
            soliton_lambda=soliton_lambda,
        )

        num_sensory = len(self.core.sensory_nodes)
        num_motor = len(self.core.motor_nodes)

        # Biological Sensory & Motor Poles
        self.sensory_pole = GeodesicSensoryAfferent(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            channels=channels,
            num_sensory_nodes=num_sensory,
        )
        self.motor_pole = GeodesicMotorEfferent(
            channels=channels,
            embed_dim=embed_dim,
            vocab_size=vocab_size,
            num_motor_nodes=num_motor,
        )

        # Applied Biological Layers
        self.thalamus = ThalamicGatewayLayer(channels=channels)
        self.columnar_layer = CorticalColumnarLayer(
            num_shells=num_shells,
            nodes_per_shell=nodes_per_shell,
            channels=channels,
        )
        self.pac_coupler = PhaseAmplitudeCouplingLayer(channels=channels)
        self.astrocytes = AstrocyticNeuromodulatorField(
            num_shells=num_shells,
            nodes_per_shell=nodes_per_shell,
            field_dim=16,
            channels=channels,
        )
        self.hippocampus = HippocampalAttractorLayer(
            memory_slots=memory_slots,
            channels=channels,
        )

    def deliberate_step(
        self,
        state_3d: torch.Tensor,
        astro_field: torch.Tensor,
        hipp_mem: torch.Tensor,
        past_states: List[torch.Tensor],
        cycle: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Executes a single deliberation step through all biological layers."""
        # A. Astrocytic metabolic gain modulation
        astro_field, synaptic_gain = self.astrocytes.update_field(
            astro_field, state_3d, self.core.L_S2
        )
        state_3d = state_3d * synaptic_gain

        # B. 3D Geodesic Wave Core Step (Laplace-Beltrami + Radial Conduction + Soliton)
        state_3d, past_states = self.core.pulse_wave_step(state_3d, past_states)

        # C. Cortical Columnar Lateral Inhibition (DoG / Mexican Hat on S^2)
        state_3d = self.columnar_layer(state_3d, self.core.L_S2)

        # D. Cross-Frequency PAC Carrier Phase Modulation
        state_3d = self.pac_coupler(state_3d, deliberation_step=cycle)

        # E. Hippocampal Sharp-Wave Ripple Replay into Subcortical Core (Shell 0)
        core_now = state_3d[:, 0, :, :].mean(dim=1)
        swr_replay = self.hippocampus.sharp_wave_replay(hipp_mem, core_now)
        state_3d[:, 0, :, :] = state_3d[:, 0, :, :] + swr_replay.unsqueeze(1) * 0.20
        hipp_mem = self.hippocampus.write_attractor(hipp_mem, core_now)

        return state_3d, astro_field, hipp_mem, past_states

    def forward(
        self,
        prompt_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        thinking_cycles: int = 3,
    ) -> torch.Tensor:
        """Teacher-forced forward training pass with zero external token feedback."""
        B, ans_len = target_tokens.shape
        device = prompt_tokens.device
        dtype = self.core.inertia_gate.dtype

        # 1. Initialize Living Resting State
        state_3d = self.core.init_living_state(B, device, dtype)
        astro_field = self.astrocytes.init_field(B, device, dtype)
        hipp_mem = self.hippocampus.init_memory(B, device, dtype)
        past_states: List[torch.Tensor] = []

        # 2. Ingest Stimulus onto Hemisphere A
        sensory_field, sensory_memory = self.sensory_pole(prompt_tokens)

        # 3. Thalamic Gating: filter afferents based on subcortical core arousal
        core_0 = state_3d[:, 0, :, :].mean(dim=1)
        gated_sensory_field = self.thalamus(sensory_field.to(dtype), core_0)
        state_3d[:, -1, self.core.sensory_nodes, :] += gated_sensory_field

        # 4. Multi-Cycle Deliberation
        for cycle in range(thinking_cycles):
            state_3d, astro_field, hipp_mem, past_states = self.deliberate_step(
                state_3d, astro_field, hipp_mem, past_states, cycle
            )

        # 5. Token Generation from Hemisphere B with Corollary Discharge
        logits_list = []
        refractory_trace = torch.zeros(B, self.vocab_size, device=device, dtype=dtype)

        for t in range(ans_len):
            hemi_b_field = state_3d[:, -1, self.core.motor_nodes, :]
            logits, refractory_trace, corollary_vector = self.motor_pole.forward_readout(
                hemi_b_field, sensory_memory, refractory_trace, step_idx=t
            )
            logits_list.append(logits.unsqueeze(1))

            # Internal Corollary Discharge: Route efference copy into deep subcortical Core
            state_3d[:, 0, :, :] += corollary_vector.unsqueeze(1) * 0.20
            state_3d[:, 1, :, :] += corollary_vector.unsqueeze(1) * 0.10

            # Pulse the living volume for the next step
            state_3d, past_states = self.core.pulse_wave_step(state_3d, past_states)

        return torch.cat(logits_list, dim=1)

    def generate_answer(
        self,
        prompt_tokens: torch.Tensor,
        max_tokens: int = 32,
        thinking_cycles: int = 3,
        eos_token_id: int = 50256,
        temperature: float = 0.6,
    ) -> List[int]:
        """
        Unbounded generation from Hemisphere B with internal corollary discharge.
        Strictly zero token feedback (x_{t+1} != [x, y]).
        """
        B = prompt_tokens.size(0)
        device = prompt_tokens.device
        dtype = self.core.inertia_gate.dtype

        # 1. Living resting state
        state_3d = self.core.init_living_state(B, device, dtype)
        astro_field = self.astrocytes.init_field(B, device, dtype)
        hipp_mem = self.hippocampus.init_memory(B, device, dtype)
        past_states: List[torch.Tensor] = []

        # 2. Ingest stimulus
        sensory_field, sensory_memory = self.sensory_pole(prompt_tokens)
        core_0 = state_3d[:, 0, :, :].mean(dim=1)
        gated_sensory_field = self.thalamus(sensory_field.to(dtype), core_0)
        state_3d[:, -1, self.core.sensory_nodes, :] += gated_sensory_field

        # 3. Deliberation cycles
        for cycle in range(thinking_cycles):
            state_3d, astro_field, hipp_mem, past_states = self.deliberate_step(
                state_3d, astro_field, hipp_mem, past_states, cycle
            )

        # 4. Generate
        out_tokens = []
        refractory_trace = torch.zeros(B, self.vocab_size, device=device, dtype=dtype)

        for t in range(max_tokens):
            hemi_b_field = state_3d[:, -1, self.core.motor_nodes, :]
            logits, refractory_trace, corollary_vector = self.motor_pole.forward_readout(
                hemi_b_field, sensory_memory, refractory_trace, step_idx=t
            )
            # Biological refractory suppression on recently fired motor tokens
            if len(out_tokens) > 0:
                logits[:, out_tokens[-1]] -= 3.5
                for prev_tok in set(out_tokens[-4:]):
                    logits[:, prev_tok] -= 1.8

            if temperature > 0:
                probs = F.softmax(logits / max(temperature, 0.1), dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).item()
            else:
                next_tok = torch.argmax(logits, dim=-1).item()

            if next_tok == eos_token_id:
                break
            out_tokens.append(next_tok)

            # Internal corollary discharge
            state_3d[:, 0, :, :] += corollary_vector.unsqueeze(1) * 0.20
            state_3d[:, 1, :, :] += corollary_vector.unsqueeze(1) * 0.10
            state_3d, past_states = self.core.pulse_wave_step(state_3d, past_states)

        return out_tokens


def build_enhanced_organoid_spheroid(
    num_shells: int = 8,
    nodes_per_shell: int = 128,
    channels: int = 64,
    embed_dim: int = 256,
    vocab_size: int = 50257,
) -> EnhancedOrganoidSpheroid:
    return EnhancedOrganoidSpheroid(
        num_shells=num_shells,
        nodes_per_shell=nodes_per_shell,
        channels=channels,
        embed_dim=embed_dim,
        vocab_size=vocab_size,
    )
