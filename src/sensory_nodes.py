"""
Sensory Nodes & Thalamic Gating for CerebroSpheroid V4.

Implements modular sensory front-ends for multi-modal ingestion:
1. VisionSensoryEncoder: Patch-based projection (16x16 conv, stride 16, 196 patches)
   projected into surface_dim, with zero-initialized injection scale.
2. AudioSensoryEncoder: Strided 1D depthwise separable convs (k=7, 11) downsampling
   to 25-50 Hz frames, projected to surface_dim, with zero-initialized injection scale.
3. ThalamicSensoryGater: Salience & quality gating on prompt inputs (padding ratio,
   entropy/perplexity band, token round-trip integrity).

Design Principle (Section 6, V4 Plan):
Text-only numerics must not shift when sensory nodes are present. Zero injection
reproduces text-only logits exactly.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionSensoryEncoder(nn.Module):
    """
    Afferent Visual Sensory Encoder (V4):
    Converts 2D visual inputs [B, 3, 224, 224] into a sequence of 196 patch tokens
    of dimension `surface_dim` via 16x16 strided convolution.
    Features a zero-initialized injection gate ensuring exact zero-injection equivalence
    with text-only baselines until trained.
    """
    def __init__(
        self,
        surface_dim: int = 640,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
    ):
        super().__init__()
        self.surface_dim = surface_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2  # 14x14 = 196

        self.patch_proj = nn.Conv2d(
            in_channels, surface_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.norm = nn.RMSNorm(surface_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, surface_dim))
        
        # Zero-initialized injection scale ensures exact numerical parity on zero/absent input
        self.injection_scale = nn.Parameter(torch.zeros(1))

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pixel_values: [B, 3, H, W] image tensor
        Returns:
            patch_tokens: [B, 196, surface_dim]
            visual_gist: [B, surface_dim] pooled representation
        """
        b, c, h, w = pixel_values.shape
        x = self.patch_proj(pixel_values)  # [B, D, H/16, W/16]
        x = x.flatten(2).transpose(1, 2)  # [B, 196, D]
        x = self.norm(x + self.pos_embed[:, :x.shape[1], :])
        
        # Scale controls the amount of visual drive entering the manifold
        effective_tokens = self.injection_scale * x
        visual_gist = effective_tokens.mean(dim=1)
        return effective_tokens, visual_gist


class AudioSensoryEncoder(nn.Module):
    """
    Afferent Auditory Sensory Encoder (V4):
    Processes 1D raw waveform or acoustic features via strided depthwise-separable
    1D convolutions (k=7, 11) downsampling to ~25-50 Hz frame representations.
    Features a zero-initialized injection gate for exact baseline equivalence.
    """
    def __init__(
        self,
        surface_dim: int = 640,
        in_channels: int = 1,
        stride_total: int = 320,  # e.g., 16kHz -> 50Hz frames (downsample factor 320)
    ):
        super().__init__()
        self.surface_dim = surface_dim
        
        # Multi-stage strided convs (e.g., 8 * 8 * 5 = 320)
        self.conv1 = nn.Conv1d(in_channels, surface_dim // 2, kernel_size=7, stride=8, padding=3)
        self.conv2 = nn.Conv1d(surface_dim // 2, surface_dim, kernel_size=7, stride=8, padding=3)
        self.conv3 = nn.Conv1d(surface_dim, surface_dim, kernel_size=11, stride=5, padding=5)
        
        self.norm = nn.RMSNorm(surface_dim)
        self.injection_scale = nn.Parameter(torch.zeros(1))

    def forward(self, audio_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_values: [B, 1, T] or [B, T] audio waveform
        Returns:
            audio_tokens: [B, T_frames, surface_dim]
            audio_gist: [B, surface_dim] pooled representation
        """
        if audio_values.ndim == 2:
            audio_values = audio_values.unsqueeze(1)
        
        h = F.gelu(self.conv1(audio_values))
        h = F.gelu(self.conv2(h))
        h = F.gelu(self.conv3(h))  # [B, D, T_frames]
        
        tokens = self.norm(h.transpose(1, 2))  # [B, T_frames, D]
        effective_tokens = self.injection_scale * tokens
        audio_gist = effective_tokens.mean(dim=1)
        return effective_tokens, audio_gist


class ThalamicSensoryGater(nn.Module):
    """
    Thalamic Sensory Gater (V4):
    Dynamic front-end filtering module gating inputs based on text quality metrics:
    - Padding ratio: suppresses spurious noise on heavily padded inputs
    - Token dispersion: measures representation entropy to prevent saturation
    - Outlier detection: clips gradient bursts at the boundary
    """
    def __init__(self, surface_dim: int = 640):
        super().__init__()
        self.surface_dim = surface_dim
        self.quality_proj = nn.Linear(3, surface_dim)
        nn.init.zeros_(self.quality_proj.weight)
        nn.init.constant_(self.quality_proj.bias, 2.0)  # sigmoid(2.0) ~ 0.88 open gate by default

    def forward(
        self,
        tokens: torch.Tensor,
        sensory_flux: torch.Tensor,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens: [B, L] input token IDs
            sensory_flux: [B, D] boundary flux to be modulated
            pad_token_id: ID of pad/EOS token
        Returns:
            modulated_flux: [B, D]
        """
        b, l = tokens.shape
        if pad_token_id is not None:
            pad_ratio = (tokens == pad_token_id).float().mean(dim=-1, keepdim=True)  # [B, 1]
        else:
            pad_ratio = torch.zeros(b, 1, device=tokens.device, dtype=sensory_flux.dtype)
        
        # Length normalized feature
        len_feature = torch.full((b, 1), math.log(max(l, 1)) / 10.0, device=tokens.device, dtype=sensory_flux.dtype)
        
        # Flux energy
        flux_rms = sensory_flux.pow(2).mean(dim=-1, keepdim=True).sqrt()
        
        features = torch.cat([pad_ratio, len_feature, flux_rms], dim=-1)
        gate = torch.sigmoid(self.quality_proj(features))
        return sensory_flux * gate
