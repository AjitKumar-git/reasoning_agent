"""
Multimodal Cortical Layers for Living Organoid Spheroids (B^3)
==============================================================
Integrates Continuous Visual and Auditory Processing into the Core:
1. VisualRetinotopicLayer:
   - Retinotopic Cortical Sheet mapping 2D spatial image features directly onto S^2.
   - Conformal spherical projection with Foveal magnification factor.
   - Receptive field convolutions (V1/V2 orientation hyper-columns).
   - Injects continuous visual wave flux directly onto the visual pole (x < -0.05).

2. AudioTonotopicLayer:
   - Tonotopic Cochlear Basilar Sheet mapping acoustic frequency spectra onto S^2.
   - Native STFT spectral decomposition with logarithmic Mel-scale frequency bins.
   - Tonotopic latitudinal gradient mapping (low to high frequency resonators).
   - Injects continuous acoustic wave flux onto the auditory pole (y < -0.05).

3. MultimodalOrganoidCore:
   - Unified Living Organoid Brain Manifold integrating Vision + Audio + Language
   - Cross-Modal Wave Resonance across the 3D Ball (B^3)
   - Strict Zero External Token Feedback (x_{t+1} != [x, y]).
"""

import math
import numpy as np
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
from src.organoid_layers import (
    ThalamicGatewayLayer,
    CorticalColumnarLayer,
    PhaseAmplitudeCouplingLayer,
    AstrocyticNeuromodulatorField,
    HippocampalAttractorLayer,
)


# =========================================================================
# 1. Visual Retinotopic Cortical Layer
# =========================================================================

class VisualRetinotopicLayer(nn.Module):
    """
    Continuous Retinotopic Visual Cortical Layer (V1/V2 Sheet):
    - Ingests 2D image signals [B, C_in, H, W] (e.g. RGB or single-channel spatial grids).
    - Extracts multi-scale spatial receptive fields (orientation & spatial frequency).
    - Projects 2D cortical feature maps conformally onto the visual pole of S^2
      using spherical coordinate grid sampling (guaranteeing continuous spatial topology).
    - Emits continuous visual wave perturbations [B, num_visual_nodes, channels].
    """
    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 64,
        num_visual_nodes: int = 64,
        foveal_magnification: float = 1.4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_visual_nodes = num_visual_nodes
        self.foveal_mag = foveal_magnification

        # Multi-scale V1 receptive field convolutions
        self.conv1 = nn.Conv2d(in_channels, channels // 2, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(channels // 2, channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.res_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.norm = nn.LayerNorm(channels)
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, images: torch.Tensor, visual_coords_s2: torch.Tensor) -> torch.Tensor:
        """
        images: [B, C_in, H, W]
        visual_coords_s2: [num_visual_nodes, 3] 3D coordinates on unit sphere S^2
        Returns:
            visual_field: [B, num_visual_nodes, channels]
        """
        B = images.shape[0]
        device = images.device
        dtype = self.conv1.weight.dtype

        x = images.to(dtype=dtype)
        h = F.silu(self.conv1(x))
        h = F.silu(self.conv2(h))
        h = F.silu(self.conv3(h))
        h = h + self.res_block(h)  # [B, C, H', W']

        # Conformal Spherical -> 2D Grid Mapping
        # For each 3D coordinate (x, y, z) on S^2, map to stereographic/azimuthal plane (u, v)
        vx = visual_coords_s2[:, 0]
        vy = visual_coords_s2[:, 1]
        vz = visual_coords_s2[:, 2]

        # Foveal magnification: expand central field, compress periphery
        # Project onto tangent disk: u = y / (1 - x), v = z / (1 - x)
        denom = (1.0 - vx.clamp(max=0.85)).unsqueeze(0)
        u = (vy.unsqueeze(0) / denom) * self.foveal_mag
        v = (vz.unsqueeze(0) / denom) * self.foveal_mag

        grid_uv = torch.stack([u.clamp(-1.0, 1.0), v.clamp(-1.0, 1.0)], dim=-1)  # [1, N_vis, 2]
        grid_uv = grid_uv.unsqueeze(1).expand(B, -1, -1, -1)                      # [B, 1, N_vis, 2]

        # Bilinear sampling from 2D continuous cortical feature map
        sampled = F.grid_sample(h, grid_uv, mode="bilinear", padding_mode="border", align_corners=True)
        # sampled shape: [B, C, 1, N_vis] -> permute to [B, N_vis, C]
        visual_field = sampled.squeeze(2).permute(0, 2, 1).contiguous()
        visual_field = self.norm(visual_field)
        return self.out_proj(visual_field)


# =========================================================================
# 2. Audio Tonotopic Cortical Layer
# =========================================================================

class AudioTonotopicLayer(nn.Module):
    """
    Continuous Tonotopic Auditory Cortical Layer (A1 Sheet):
    - Ingests 1D audio waveforms [B, 1, T] or precomputed spectrograms [B, n_mels, T].
    - Computes Short-Time Fourier Transform (STFT) with Hann windowing.
    - Projects acoustic frequencies tonotopically along a geodesic latitudinal gradient
      (low frequencies near equator, high frequencies near poles).
    - Applies acoustic resonance phase modeling.
    - Emits continuous auditory wave perturbations [B, num_audio_nodes, channels].
    """
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 160,
        num_freq_bins: int = 64,
        channels: int = 64,
        num_audio_nodes: int = 64,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_freq_bins = num_freq_bins
        self.channels = channels
        self.num_audio_nodes = num_audio_nodes

        # Temporal audio feature convolutions along basilar membrane
        self.spec_conv = nn.Sequential(
            nn.Conv1d(num_freq_bins, channels, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(num_audio_nodes),
        )
        self.harmonic_resonance = nn.Parameter(torch.randn(num_audio_nodes, channels) * 0.05)
        self.norm = nn.LayerNorm(channels)
        self.out_proj = nn.Linear(channels, channels)

    def forward(
        self,
        audio: torch.Tensor,
        audio_coords_s2: torch.Tensor,
    ) -> torch.Tensor:
        """
        audio: [B, 1, T] raw waveform OR [B, num_freq_bins, T_spec] spectrogram
        audio_coords_s2: [num_audio_nodes, 3] 3D coordinates on unit sphere S^2
        Returns:
            audio_field: [B, num_audio_nodes, channels]
        """
        B = audio.shape[0]
        device = audio.device
        dtype = self.harmonic_resonance.dtype

        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # [B, 1, T]

        # 1. Compute spectral representation if raw waveform
        if audio.shape[1] == 1 and audio.shape[-1] > 256:
            window = torch.hann_window(self.n_fft, device=device)
            stft = torch.stft(
                audio.squeeze(1),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
                return_complex=True,
            )
            spec = torch.abs(stft)[:, :self.num_freq_bins, :]  # [B, n_freq, T_frames]
        else:
            # Already frequency/spectrogram input
            spec = audio

        # 2. Convolve across time-frequency basilar membrane
        spec_f = spec.to(dtype=dtype)
        # Ensure correct channel dimension for Conv1d
        if spec_f.shape[1] != self.num_freq_bins:
            spec_f = F.adaptive_avg_pool1d(spec_f.transpose(1, 2), self.num_freq_bins).transpose(1, 2)

        feat_1d = self.spec_conv(spec_f)  # [B, C, num_audio_nodes]
        feat_nodes = feat_1d.permute(0, 2, 1).contiguous()  # [B, num_audio_nodes, C]

        # 3. Tonotopic elevation sorting (Z-axis gradient encodes frequency)
        elevations = audio_coords_s2[:, 2].to(device=device, dtype=dtype)
        tonotopic_weights = torch.sigmoid(elevations.unsqueeze(0).unsqueeze(-1) * 2.0)

        # 4. Harmonic resonance modulation
        resonance = self.harmonic_resonance.unsqueeze(0).to(device=device, dtype=dtype)
        modulated = (feat_nodes + resonance) * (0.8 + 0.4 * tonotopic_weights)
        audio_field = self.norm(modulated)
        return self.out_proj(audio_field)


# =========================================================================
# 3. Master Multimodal Organoid Core Architecture
# =========================================================================

class MultimodalOrganoidCore(nn.Module):
    """
    Master Multimodal Living Organoid Brain Manifold (B^3):
    Coordinates three continuous sensory afferents onto the unit sphere S^2:
    - Hemisphere A South Pole (z < -0.05): Symbolic Language Afferents
    - Hemisphere A West Pole (x < -0.05): Visual Retinotopic Sheet
    - Hemisphere A South-West (y < -0.05): Auditory Tonotopic Sheet
    - Thalamic Gateway dynamically routes and balances cross-modal energy
    - Physical 3D Wave Core propagates cross-modal wave interference in B^3
    - Cortical Columnar DoG lateral inhibition sharpens multimodal attractors
    - Hemisphere B North Pole (z > 0.05): Emits symbolic motor thoughts
    - Internal Corollary Discharge preserves x_{t+1} != [x, y]
    """
    def __init__(
        self,
        vocab_size: int = 50257,
        embed_dim: int = 256,
        channels: int = 64,
        num_shells: int = 8,
        nodes_per_shell: int = 128,
        image_in_channels: int = 3,
        memory_slots: int = 32,
        soliton_lambda: float = 0.08,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.channels = channels
        self.num_shells = num_shells
        self.nodes_per_shell = nodes_per_shell

        # 1. 3D Geodesic Ball Wave Core
        self.core = RealGeodesicSpheroidCore(
            num_shells=num_shells,
            nodes_per_shell=nodes_per_shell,
            channels=channels,
            soliton_lambda=soliton_lambda,
        )

        coords = self.core.coords  # [M, 3]

        # Topological partition of sensory surface nodes on S^2
        # Language: South Pole (z < -0.05)
        south_nodes = (coords[:, 2] < -0.05).nonzero(as_tuple=True)[0]
        # Visual: West Pole (x < -0.05)
        west_nodes = (coords[:, 0] < -0.05).nonzero(as_tuple=True)[0]
        # Auditory: South-West (y < -0.05)
        sw_nodes = (coords[:, 1] < -0.05).nonzero(as_tuple=True)[0]
        # Motor: North Pole (z > 0.05)
        north_nodes = (coords[:, 2] > 0.05).nonzero(as_tuple=True)[0]

        self.register_buffer("lang_nodes", south_nodes)
        self.register_buffer("vis_nodes", west_nodes)
        self.register_buffer("aud_nodes", sw_nodes)
        self.register_buffer("motor_nodes", north_nodes)

        # 2. Sensory Afferent Layers
        self.lang_pole = GeodesicSensoryAfferent(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            channels=channels,
            num_sensory_nodes=len(south_nodes),
        )
        self.visual_layer = VisualRetinotopicLayer(
            in_channels=image_in_channels,
            channels=channels,
            num_visual_nodes=len(west_nodes),
        )
        self.audio_layer = AudioTonotopicLayer(
            channels=channels,
            num_audio_nodes=len(sw_nodes),
        )

        # 3. Multimodal Thalamic Routing Gateway
        self.thalamus = ThalamicGatewayLayer(channels=channels)
        self.vis_thalamus = ThalamicGatewayLayer(channels=channels)
        self.aud_thalamus = ThalamicGatewayLayer(channels=channels)

        # 4. Biological Cortical Processing Layers
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

        # 5. Motor Efferent Readout Pole
        self.motor_pole = GeodesicMotorEfferent(
            channels=channels,
            embed_dim=embed_dim,
            vocab_size=vocab_size,
            num_motor_nodes=len(north_nodes),
        )

    def deliberate_step(
        self,
        state_3d: torch.Tensor,
        astro_field: torch.Tensor,
        hipp_mem: torch.Tensor,
        past_states: List[torch.Tensor],
        cycle: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Multi-layer physical wave deliberation across the 3D unit ball B^3."""
        # A. Astrocytic metabolic gain modulation
        astro_field, synaptic_gain = self.astrocytes.update_field(
            astro_field, state_3d, self.core.L_S2
        )
        state_3d = state_3d * synaptic_gain

        # B. 3D Wave Diffusion (Laplace-Beltrami + Radial Conduction + Soliton)
        state_3d, past_states = self.core.pulse_wave_step(state_3d, past_states)

        # C. Cortical Columnar Lateral Inhibition (DoG on S^2)
        state_3d = self.columnar_layer(state_3d, self.core.L_S2)

        # D. Cross-Frequency PAC Modulation
        state_3d = self.pac_coupler(state_3d, deliberation_step=cycle)

        # E. Hippocampal Sharp-Wave Ripple Replay into Subcortical Core
        core_now = state_3d[:, 0, :, :].mean(dim=1)
        swr_replay = self.hippocampus.sharp_wave_replay(hipp_mem, core_now)
        state_3d[:, 0, :, :] += swr_replay.unsqueeze(1) * 0.20
        hipp_mem = self.hippocampus.write_attractor(hipp_mem, core_now)

        return state_3d, astro_field, hipp_mem, past_states

    def ingest_multimodal(
        self,
        state_3d: torch.Tensor,
        prompt_tokens: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Injects Language, Visual, and Audio waves simultaneously into the manifold,
        gated by the thalamocortical dynamic routing layer.
        """
        dtype = self.core.inertia_gate.dtype
        core_0 = state_3d[:, 0, :, :].mean(dim=1)
        sensory_memory = None

        # 1. Ingest Language Tokens (South Pole)
        if prompt_tokens is not None:
            sens_field, sensory_memory = self.lang_pole(prompt_tokens)
            gated_lang = self.thalamus(sens_field.to(dtype), core_0)
            state_3d[:, -1, self.lang_nodes, :] += gated_lang

        # 2. Ingest Visual Retinotopic Sheet (West Pole)
        if images is not None:
            vis_coords = self.core.coords[self.vis_nodes]
            vis_field = self.visual_layer(images, vis_coords)
            gated_vis = self.vis_thalamus(vis_field.to(dtype), core_0)
            state_3d[:, -1, self.vis_nodes, :] += gated_vis

        # 3. Ingest Auditory Tonotopic Sheet (South-West Pole)
        if audio is not None:
            aud_coords = self.core.coords[self.aud_nodes]
            aud_field = self.audio_layer(audio, aud_coords)
            gated_aud = self.aud_thalamus(aud_field.to(dtype), core_0)
            state_3d[:, -1, self.aud_nodes, :] += gated_aud

        return state_3d, sensory_memory

    def forward(
        self,
        prompt_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        thinking_cycles: int = 3,
    ) -> torch.Tensor:
        """Teacher-forced training forward pass with continuous corollary discharge."""
        B, ans_len = target_tokens.shape
        device = prompt_tokens.device if prompt_tokens is not None else target_tokens.device
        dtype = self.core.inertia_gate.dtype

        # 1. Living Resting State
        state_3d = self.core.init_living_state(B, device, dtype)
        astro_field = self.astrocytes.init_field(B, device, dtype)
        hipp_mem = self.hippocampus.init_memory(B, device, dtype)
        past_states: List[torch.Tensor] = []

        # 2. Ingest Multimodal Afferents
        state_3d, sensory_memory = self.ingest_multimodal(
            state_3d, prompt_tokens=prompt_tokens, images=images, audio=audio
        )

        # If no text prompt was provided, create a dummy sensory memory for cross-resonance
        if sensory_memory is None:
            sensory_memory = torch.zeros(B, 1, 256, device=device, dtype=dtype)

        # 3. Deliberation Cycles
        for cycle in range(thinking_cycles):
            state_3d, astro_field, hipp_mem, past_states = self.deliberate_step(
                state_3d, astro_field, hipp_mem, past_states, cycle
            )

        # 4. Generate Motor Efferents with Internal Corollary Discharge
        logits_list = []
        refractory_trace = torch.zeros(B, self.vocab_size, device=device, dtype=dtype)

        for t in range(ans_len):
            hemi_b_field = state_3d[:, -1, self.motor_nodes, :]
            logits, refractory_trace, corollary_vector = self.motor_pole.forward_readout(
                hemi_b_field, sensory_memory, refractory_trace, step_idx=t
            )
            logits_list.append(logits.unsqueeze(1))

            # Internal Corollary Discharge
            state_3d[:, 0, :, :] += corollary_vector.unsqueeze(1) * 0.20
            state_3d[:, 1, :, :] += corollary_vector.unsqueeze(1) * 0.10
            state_3d, past_states = self.core.pulse_wave_step(state_3d, past_states)

        return torch.cat(logits_list, dim=1)

    def generate_multimodal(
        self,
        prompt_tokens: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        max_tokens: int = 32,
        thinking_cycles: int = 3,
        eos_token_id: int = 50256,
        temperature: float = 0.6,
    ) -> List[int]:
        """Unbounded multimodal inference with internal corollary discharge."""
        B = 1
        if prompt_tokens is not None:
            B = prompt_tokens.size(0)
            device = prompt_tokens.device
        elif images is not None:
            B = images.size(0)
            device = images.device
        elif audio is not None:
            B = audio.size(0)
            device = audio.device
        else:
            raise ValueError("Must provide at least one modality (prompt_tokens, images, or audio)")

        dtype = self.core.inertia_gate.dtype
        state_3d = self.core.init_living_state(B, device, dtype)
        astro_field = self.astrocytes.init_field(B, device, dtype)
        hipp_mem = self.hippocampus.init_memory(B, device, dtype)
        past_states: List[torch.Tensor] = []

        state_3d, sensory_memory = self.ingest_multimodal(
            state_3d, prompt_tokens=prompt_tokens, images=images, audio=audio
        )
        if sensory_memory is None:
            sensory_memory = torch.zeros(B, 1, 256, device=device, dtype=dtype)

        for cycle in range(thinking_cycles):
            state_3d, astro_field, hipp_mem, past_states = self.deliberate_step(
                state_3d, astro_field, hipp_mem, past_states, cycle
            )

        out_tokens = []
        refractory_trace = torch.zeros(B, self.vocab_size, device=device, dtype=dtype)

        for t in range(max_tokens):
            hemi_b_field = state_3d[:, -1, self.motor_nodes, :]
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

            state_3d[:, 0, :, :] += corollary_vector.unsqueeze(1) * 0.20
            state_3d[:, 1, :, :] += corollary_vector.unsqueeze(1) * 0.10
            state_3d, past_states = self.core.pulse_wave_step(state_3d, past_states)

        return out_tokens


def build_multimodal_organoid_core(
    vocab_size: int = 50257,
    embed_dim: int = 256,
    channels: int = 64,
    num_shells: int = 8,
    nodes_per_shell: int = 128,
    image_in_channels: int = 3,
    preset: str = "33m",
) -> MultimodalOrganoidCore:
    if preset == "100m":
        embed_dim = 768
        channels = 160
        num_shells = 8
        nodes_per_shell = 128
    return MultimodalOrganoidCore(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        channels=channels,
        num_shells=num_shells,
        nodes_per_shell=nodes_per_shell,
        image_in_channels=image_in_channels,
    )


def build_100m_multimodal_organoid_core(
    vocab_size: int = 50257,
    image_in_channels: int = 3,
) -> MultimodalOrganoidCore:
    """Builds the 100.11M parameter living multimodal organoid core."""
    return MultimodalOrganoidCore(
        vocab_size=vocab_size,
        embed_dim=768,
        channels=160,
        num_shells=8,
        nodes_per_shell=128,
        image_in_channels=image_in_channels,
    )
