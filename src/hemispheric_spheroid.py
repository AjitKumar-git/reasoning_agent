"""
Bipolar Hemispheric Wave Spheroid (v2)
Anatomically Polarized Continuous Brain Manifold:
1. Sensory Hemisphere A (Surface Only): 1D->2D Spatial Afferent CNN
2. 3D Volumetric Waveflow: Trans-Callosal 3D Wave Diffusion across 3D Spheroid Grid
3. Motor Hemisphere B (Surface Only): 2D->1D Efferent Deconvolutional CNN
4. Internal Corollary Discharge into Subcortical Core (Shell 0 / Deep Voxels)
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SensoryAfferentCNN(nn.Module):
    """
    Hemisphere A (Sensory Pole Surface):
    Converts 1D sequential language stimulus into a 2D topographical cortical sheet.
    """
    def __init__(self, vocab_size: int = 50257, embed_dim: int = 256, channels: int = 128, grid_size: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.channels = channels
        self.grid_size = grid_size  # H = W = grid_size
        
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 1024, embed_dim) * 0.02)
        
        # 1D Temporal Convolutions extracting linguistic feature hierarchies
        self.conv1d_1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(embed_dim, channels, kernel_size=3, padding=1)
        self.norm1d = nn.RMSNorm(channels)
        
        # Spatial Projection from Sequence Tokens into 2D Cortical Grid Sheet (H x W)
        self.to_grid = nn.Linear(channels, channels * grid_size * grid_size)
        self.surface_conv2d = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.surface_norm = nn.GroupNorm(8, channels)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Maps [B, L] -> (Sensory Working Memory [B, L, C], 2D Cortical Sheet [B, C, H, W])
        """
        B, L = input_ids.shape
        x = self.token_embed(input_ids) + self.pos_embed[:, :L, :]  # [B, L, embed_dim]
        
        # 1D Temporal Convolutions
        h1d = F.silu(self.conv1d_1(x.transpose(1, 2)))
        h1d = self.conv1d_2(h1d).transpose(1, 2)  # [B, L, channels]
        sensory_memory = self.norm1d(h1d)
        
        # Pool temporal sequence into global topographical cortical excitation map
        pooled = sensory_memory.mean(dim=1)  # [B, channels]
        grid_flat = self.to_grid(pooled)    # [B, channels * H * W]
        grid_2d = grid_flat.view(B, self.channels, self.grid_size, self.grid_size)
        
        # Surface 2D lateral diffusion
        surface_sheet = self.surface_norm(F.silu(self.surface_conv2d(grid_2d)) + grid_2d)
        return sensory_memory, surface_sheet


class VolumetricSpheroidCore3D(nn.Module):
    """
    3D Spheroid Manifold:
    Physical 3D volumetric wave chamber partitioned into:
    - Hemisphere A (Input Pole): Depth slices z in [0, D // 2)
    - Corpus Callosum (Central Bridge): Depth slices around z = D // 2
    - Hemisphere B (Output Pole): Depth slices z in [D // 2, D)
    """
    def __init__(self, channels: int = 128, depth: int = 8, grid_size: int = 8, num_callosal_tracts: int = 24):
        super().__init__()
        self.channels = channels
        self.depth = depth          # Z-axis (polar axis from Left to Right Hemisphere)
        self.grid_size = grid_size  # Y-axis and X-axis
        
        # 3D Physical Spatial Wave Diffusion Convolutions
        self.wave_conv3d = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.cross_conv3d = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.vol_norm = nn.GroupNorm(8, channels)
        
        # Trans-Callosal Axonal Shortcut Tracts (Fast-bridges linking Hemisphere A to Hemisphere B)
        self.callosal_weight = nn.Parameter(torch.randn(num_callosal_tracts, channels, channels) * 0.02)
        # Random paired loci bridging Hemisphere A (z < D/2) and Hemisphere B (z >= D/2)
        src_z = torch.randint(0, depth // 2, (num_callosal_tracts,))
        dst_z = torch.randint(depth // 2, depth, (num_callosal_tracts,))
        src_y = torch.randint(0, grid_size, (num_callosal_tracts,))
        dst_y = torch.randint(0, grid_size, (num_callosal_tracts,))
        src_x = torch.randint(0, grid_size, (num_callosal_tracts,))
        dst_x = torch.randint(0, grid_size, (num_callosal_tracts,))
        
        self.register_buffer("src_z", src_z)
        self.register_buffer("dst_z", dst_z)
        self.register_buffer("src_y", src_y)
        self.register_buffer("dst_y", dst_y)
        self.register_buffer("src_x", src_x)
        self.register_buffer("dst_x", dst_x)
        
        # Inertia gate for contractive unit-gain dynamics
        self.inertia_gate = nn.Parameter(torch.tensor(0.85))

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initializes quiescent 3D volumetric state: [B, C, Depth, H, W]"""
        return torch.zeros(
            batch_size, self.channels, self.depth, self.grid_size, self.grid_size,
            device=device, dtype=self.wave_conv3d.weight.dtype
        )

    def inject_hemisphere_a(self, state: torch.Tensor, surface_sheet: torch.Tensor) -> torch.Tensor:
        """Injects sensory activation exclusively into the outer surface of Hemisphere A (z=0)."""
        new_state = state.clone()
        new_state[:, :, 0, :, :] = new_state[:, :, 0, :, :] + surface_sheet.to(state.dtype)
        return self.vol_norm(new_state)

    def inject_corollary_discharge(self, state: torch.Tensor, efference_copy: torch.Tensor) -> torch.Tensor:
        """
        Injects internal motor corollary discharge into subcortical deep core (central depth, central voxels).
        Preserves zero external token feedback while informing internal cerebellar state of recent articulation.
        """
        new_state = state.clone()
        mid_z = self.depth // 2
        mid_y = self.grid_size // 2
        mid_x = self.grid_size // 2
        # efference_copy is [B, C] -> inject into central subcortical core
        new_state[:, :, mid_z, mid_y, mid_x] = new_state[:, :, mid_z, mid_y, mid_x] + efference_copy.to(state.dtype)
        return self.vol_norm(new_state)

    def wave_deliberation_step(self, state: torch.Tensor) -> torch.Tensor:
        """
        Propagates 3D physical wave across space from Hemisphere A across Corpus Callosum to Hemisphere B:
        S_{t+1} = GroupNorm(gamma * S_t + (1 - gamma) * tanh(Conv3D(S) + CallosalTracts(S)))
        """
        # 1. 3D Spatial Laplacian Wave Diffusion
        diffused = self.wave_conv3d(state) + self.cross_conv3d(state)
        
        # 2. Trans-Callosal White-Matter Axonal Bridges
        B = state.shape[0]
        src_voxels = state[:, :, self.src_z, self.src_y, self.src_x]  # [B, C, N_tracts]
        # Einsum over tracts
        tract_flow = torch.einsum('bct,tco->bot', src_voxels, self.callosal_weight)  # [B, C, N_tracts]
        
        shortcut_field = torch.zeros_like(state)
        for t in range(len(self.src_z)):
            shortcut_field[:, :, self.dst_z[t], self.dst_y[t], self.dst_x[t]] += tract_flow[:, :, t]
        
        total_velocity = diffused + 0.3 * shortcut_field
        gamma = torch.sigmoid(self.inertia_gate)
        next_state = gamma * state + (1.0 - gamma) * torch.tanh(total_velocity)
        return self.vol_norm(next_state)

    def tap_hemisphere_b(self, state: torch.Tensor) -> torch.Tensor:
        """Taps the emergent spatial wave on the surface of Hemisphere B (z = Depth - 1)."""
        return state[:, :, -1, :, :]  # [B, C, H, W]


class MotorEfferentCNN(nn.Module):
    """
    Hemisphere B (Motor Pole Surface):
    2D->1D Deconvolutional CNN reading emergent surface wave topology on Hemisphere B
    and converting it to human-readable language token logits.
    """
    def __init__(self, vocab_size: int = 50257, channels: int = 128, grid_size: int = 8, max_phase_steps: int = 1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.channels = channels
        self.grid_size = grid_size
        self.max_phase_steps = max_phase_steps
        
        # Premotor Cadence Clock
        self.phase_embed = nn.Embedding(max_phase_steps, channels)
        
        # Surface 2D spatial feature extractor
        self.conv2d_1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1)  # [H/2, W/2]
        self.norm2d = nn.GroupNorm(8, channels * 2)
        
        # Arcuate Fasciculus Cross-Resonance with input sensory memory
        self.q_proj = nn.Linear(channels * 2, channels * 2, bias=False)
        self.k_proj = nn.Linear(channels, channels * 2, bias=False)
        self.v_proj = nn.Linear(channels, channels * 2, bias=False)
        self.res_norm = nn.RMSNorm(channels * 2)
        
        # SwiGLU Gated Motor Synthesis
        fused_dim = channels * 2 + channels * 2
        self.fc_gate = nn.Linear(fused_dim, channels * 4, bias=False)
        self.fc_val = nn.Linear(fused_dim, channels * 4, bias=False)
        self.norm_mid = nn.RMSNorm(channels * 4)
        
        # Language Output Head
        self.head = nn.Linear(channels * 4, vocab_size, bias=False)
        
        # Corollary Discharge Projection (Efference Copy to central subcortical core)
        self.efference_proj = nn.Linear(channels * 4, channels, bias=False)
        nn.init.normal_(self.efference_proj.weight, mean=0.0, std=0.01)
        
        # Refractory parameters
        self.refractory_strength = nn.Parameter(torch.tensor(1.2))
        self.refractory_decay = nn.Parameter(torch.tensor(0.75))

    def forward(
        self,
        surface_b: torch.Tensor,
        sensory_memory: Optional[torch.Tensor] = None,
        refractory_trace: Optional[torch.Tensor] = None,
        step_idx: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Maps emergent Hemisphere B surface [B, C, H, W] -> (logits [B, V], next_trace, efference_copy [B, C])
        """
        B = surface_b.shape[0]
        
        # Premotor Phase Clock
        p_idx = min(max(0, step_idx), self.max_phase_steps - 1)
        phase_vec = self.phase_embed(torch.tensor(p_idx, device=surface_b.device)).view(1, self.channels, 1, 1)
        modulated_surface = surface_b + phase_vec
        
        # Spatial 2D Convolutions across Hemisphere B surface
        h2d = F.silu(self.conv2d_1(modulated_surface))
        h2d = self.norm2d(F.silu(self.conv2d_2(h2d)))  # [B, C*2, H/2, W/2]
        motor_spatial_vec = h2d.mean(dim=[2, 3])       # [B, C*2]
        
        # Arcuate Fasciculus Cross-Resonance with input prompt working memory
        if sensory_memory is not None and sensory_memory.ndim == 3:
            q = self.q_proj(motor_spatial_vec).unsqueeze(1)    # [B, 1, C*2]
            k = self.k_proj(sensory_memory)                     # [B, L, C*2]
            v = self.v_proj(sensory_memory)                     # [B, L, C*2]
            scale = 1.0 / math.sqrt(self.channels * 2)
            attn_scores = torch.bmm(q, k.transpose(1, 2)) * scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            resonance = self.res_norm(torch.bmm(attn_weights, v).squeeze(1))  # [B, C*2]
        else:
            resonance = torch.zeros_like(motor_spatial_vec)
        
        # SwiGLU Gated Motor Synthesis
        fused = torch.cat([motor_spatial_vec, resonance], dim=-1)
        h = self.norm_mid(F.silu(self.fc_gate(fused)) * self.fc_val(fused))
        
        raw_logits = self.head(h)
        efference_copy = self.efference_proj(h)  # [B, C]
        
        # Refractory dynamics
        if refractory_trace is not None:
            alpha = torch.clamp(self.refractory_strength, min=0.1, max=5.0)
            adapted_logits = raw_logits - alpha * refractory_trace.to(raw_logits.dtype)
            decay = torch.sigmoid(self.refractory_decay)
            current_prob = F.softmax(raw_logits, dim=-1).detach()
            next_trace = (decay * refractory_trace + (1.0 - decay) * current_prob).detach()
            return adapted_logits, next_trace, efference_copy

        return raw_logits, None, efference_copy


# =========================================================================
# Master Bipolar Hemispheric Spheroid Architecture
# =========================================================================

class BipolarHemisphericSpheroid(nn.Module):
    """
    Bipolar Hemispheric Wave Spheroid:
    1. Input Pole: Hemisphere A (Sensory Afferent 1D->2D CNN)
    2. Transmission: 3D Volumetric Wave Diffusion across Corpus Callosum
    3. Output Pole: Hemisphere B (Motor Efferent 2D->1D CNN)
    4. Internal Corollary Discharge into Subcortical Core
    Zero External Feedback ($x_{t+1} \neq [x, y]$).
    """
    def __init__(
        self,
        vocab_size: int = 50257,
        channels: int = 128,
        depth: int = 8,
        grid_size: int = 8,
        num_callosal_tracts: int = 32,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.channels = channels
        self.depth = depth
        self.grid_size = grid_size
        
        self.sensory_cnn = SensoryAfferentCNN(
            vocab_size=vocab_size, embed_dim=256, channels=channels, grid_size=grid_size
        )
        self.wave_core3d = VolumetricSpheroidCore3D(
            channels=channels, depth=depth, grid_size=grid_size, num_callosal_tracts=num_callosal_tracts
        )
        self.motor_cnn = MotorEfferentCNN(
            vocab_size=vocab_size, channels=channels, grid_size=grid_size
        )

    def ingest_stimulus(
        self, prompt_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ingests environmental sensory stimulus exclusively onto Hemisphere A surface:
        1. Encodes 1D tokens into 2D surface sheet on Hemisphere A.
        2. Initializes 3D volumetric core and injects activation at z=0.
        3. Returns (state_3d, sensory_memory)
        """
        B = prompt_tokens.shape[0]
        sensory_memory, surface_a = self.sensory_cnn(prompt_tokens)
        
        state_3d = self.wave_core3d.init_state(B, prompt_tokens.device)
        state_3d = self.wave_core3d.inject_hemisphere_a(state_3d, surface_a)
        return state_3d, sensory_memory

    def deliberate(self, state_3d: torch.Tensor, cycles: int = 3) -> torch.Tensor:
        """Physical 3D continuous wave diffusion across Corpus Callosum towards Hemisphere B."""
        curr = state_3d
        for _ in range(cycles):
            curr = self.wave_core3d.wave_deliberation_step(curr)
        return curr

    def emit_next_token(
        self,
        state_3d: torch.Tensor,
        sensory_memory: Optional[torch.Tensor] = None,
        refractory_trace: Optional[torch.Tensor] = None,
        step_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Taps Hemisphere B surface, decodes output logits, and loops efference copy into subcortical core.
        """
        # 1. Tap emergent spatial wave on Hemisphere B surface
        surface_b = self.wave_core3d.tap_hemisphere_b(state_3d)
        
        # 2. Decode language logits & extract internal corollary discharge
        logits, next_trace, efference_copy = self.motor_cnn(
            surface_b, sensory_memory=sensory_memory, refractory_trace=refractory_trace, step_idx=step_idx
        )
        
        # 3. Advance internal 3D wave with internal corollary discharge in central subcortical core
        state_with_eff = self.wave_core3d.inject_corollary_discharge(state_3d, efference_copy)
        next_state_3d = self.wave_core3d.wave_deliberation_step(state_with_eff)
        
        return logits, next_state_3d, next_trace

    @torch.no_grad()
    def solve_with_active_thinking(
        self,
        prompt_tokens: torch.Tensor,
        thinking_cycles: int = 3,
        max_answer_tokens: Optional[int] = None,
        temperature: float = 0.6,
        repetition_penalty: float = 1.2,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Bipolar Hemispheric Active Generation:
        Stimulus enters Hemisphere A -> Waves flow through 3D Volume -> Articulates on Hemisphere B.
        Zero external token feedback.
        """
        self.eval()
        state_3d, sensory_memory = self.ingest_stimulus(prompt_tokens)
        state_3d = self.deliberate(state_3d, cycles=thinking_cycles)

        B = prompt_tokens.shape[0]
        refractory_trace = torch.zeros(B, self.vocab_size, device=prompt_tokens.device, dtype=self.motor_cnn.head.weight.dtype)

        ceiling = max_answer_tokens if max_answer_tokens is not None else 512
        generated = []

        for step_idx in range(ceiling):
            logits, state_3d, refractory_trace = self.emit_next_token(
                state_3d, sensory_memory=sensory_memory, refractory_trace=refractory_trace, step_idx=step_idx
            )

            if eos_token_id is not None and step_idx < 3:
                logits[..., eos_token_id] -= 50.0

            if repetition_penalty > 0 and len(generated) > 0:
                for gen_tok in generated:
                    logits.scatter_add_(-1, gen_tok, torch.full_like(gen_tok, -repetition_penalty, dtype=logits.dtype))

            if temperature <= 0.05:
                next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / max(temperature, 1e-4), dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)

            generated.append(next_tok)

            if eos_token_id is not None and (next_tok == eos_token_id).all():
                break

        if len(generated) == 0:
            return torch.empty((B, 0), dtype=torch.long, device=prompt_tokens.device)
        return torch.cat(generated, dim=-1)


def build_minimal_hemispheric_spheroid() -> BipolarHemisphericSpheroid:
    """Builds a minimal, high-throughput Bipolar Hemispheric Spheroid (~20M params)."""
    return BipolarHemisphericSpheroid(
        vocab_size=50257,
        channels=128,
        depth=8,
        grid_size=8,
        num_callosal_tracts=32,
    )
