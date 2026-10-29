"""
DeepSeek Advancements for CerebroSpheroid V4:
1. Multi-Head Latent Attention (MLA) for Arcuate Fasciculus KV compression.
2. Multi-Token Prediction (MTP) for speculative decoding and forward planning.

References:
- DeepSeek-V2 / DeepSeek-V3 Technical Reports:
  Multi-Head Latent Attention (MLA) compresses the KV cache by >80% via low-rank projection
  and decoupled rotary position keys.
  Multi-Token Prediction (MTP) trains auxiliary heads to anticipate future tokens in parallel.
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# 1. Multi-Head Latent Attention (MLA)
# =========================================================================

def apply_rotary_pos_emb(x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
    """Vectorized RoPE for decoupled positional keys and queries."""
    B, L, D = x.shape
    d_half = D // 2
    device = x.device
    dtype = x.dtype

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_half, device=device, dtype=torch.float32) / d_half))
    t = torch.arange(L, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [L, d_half]
    sin, cos = freqs.sin().to(dtype), freqs.cos().to(dtype)
    cos_full = torch.cat([cos, cos], dim=-1).unsqueeze(0)  # [1, L, D]
    sin_full = torch.cat([sin, sin], dim=-1).unsqueeze(0)  # [1, L, D]

    x1, x2 = x[..., :d_half], x[..., d_half:d_half * 2]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos_full + rotated * sin_full


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) for the Arcuate Fasciculus cross-resonance bypass.
    
    Instead of caching full [B, L, D] keys and [B, L, D] values per token, MLA projects
    sensory memory into a low-rank latent vector c_kv of dimension d_c (e.g. D // 4),
    along with a small decoupled positional key k_rope of dimension d_r (e.g. 32).
    
    KV Cache Compression:
        Standard MHA KV Cache : 2 * D floats/token
        MLA Cache              : (d_c + d_r) floats/token
        Compression Ratio      : ~75% - 85% memory savings!
    """

    def __init__(
        self,
        surface_dim: int = 640,
        num_heads: int = 8,
        latent_dim: Optional[int] = None,
        rope_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.surface_dim = surface_dim
        self.num_heads = num_heads
        self.head_dim = surface_dim // num_heads
        self.latent_dim = latent_dim or (surface_dim // 4)  # 640 -> 160
        self.rope_dim = rope_dim

        # Latent KV down-projection (applied ONCE to sensory memory)
        self.w_dkv = nn.Linear(surface_dim, self.latent_dim, bias=False)
        self.norm_kv = nn.RMSNorm(self.latent_dim)

        # Decoupled RoPE Key projection
        self.w_kr = nn.Linear(surface_dim, self.rope_dim, bias=False)

        # Latent Key/Value Up-projections (applied during attention)
        self.w_uk = nn.Linear(self.latent_dim, self.surface_dim, bias=False)
        self.w_uv = nn.Linear(self.latent_dim, self.surface_dim, bias=False)

        # Query Projections (from motor speech surface)
        self.w_q = nn.Linear(surface_dim, self.surface_dim, bias=False)
        self.w_qr = nn.Linear(surface_dim, self.rope_dim, bias=False)

        # Output projection and normalization
        self.out_proj = nn.Linear(surface_dim, surface_dim, bias=False)
        self.norm_out = nn.RMSNorm(surface_dim)
        self.dropout = dropout

    def compress_sensory(self, sensory_memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compresses sensory working memory into the compact MLA latent cache.
        Returns:
            c_kv   : [B, L, latent_dim] (compressed content)
            k_rope : [B, L, rope_dim]   (decoupled positional key)
        """
        c_kv = self.norm_kv(self.w_dkv(sensory_memory))
        k_rope = apply_rotary_pos_emb(self.w_kr(sensory_memory))
        return c_kv, k_rope

    def forward(
        self,
        motor_query: torch.Tensor,
        mla_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Performs multi-head latent attention.
        Args:
            motor_query : [B, 1, surface_dim] or [B, Q, surface_dim]
            mla_cache   : Tuple of (c_kv [B, L, latent_dim], k_rope [B, L, rope_dim])
        Returns:
            resonance   : [B, Q, surface_dim]
        """
        c_kv, k_rope = mla_cache
        B, L, _ = c_kv.shape
        Q_len = motor_query.shape[1]

        # 1. Generate Queries
        q_content = self.w_q(motor_query)  # [B, Q, D]
        q_rope = apply_rotary_pos_emb(self.w_qr(motor_query))  # [B, Q, rope_dim]

        # Reshape to multi-head: [B, num_heads, Q, head_dim]
        q_heads = q_content.view(B, Q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Up-project Keys and Values from Latent Cache
        k_content = self.w_uk(c_kv)  # [B, L, D]
        v_content = self.w_uv(c_kv)  # [B, L, D]

        k_heads = k_content.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v_heads = v_content.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Combine content attention score with decoupled RoPE positional score:
        # Score = (Q_content @ K_content^T + Q_rope @ K_rope^T) / sqrt(d_h + d_r)
        # We achieve this via SDPA by appending RoPE keys along the head dimension:
        q_rope_expanded = q_rope.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # [B, H, Q, d_r]
        k_rope_expanded = k_rope.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # [B, H, L, d_r]

        q_full = torch.cat([q_heads, q_rope_expanded], dim=-1)  # [B, H, Q, d_h + d_r]
        k_full = torch.cat([k_heads, k_rope_expanded], dim=-1)  # [B, H, L, d_h + d_r]

        # SDPA accelerated attention computation
        scale = 1.0 / math.sqrt(self.head_dim + self.rope_dim)
        attn_out = F.scaled_dot_product_attention(
            q_full, k_full, v_heads, scale=scale, dropout_p=self.dropout if self.training else 0.0
        )  # [B, H, Q, d_h]

        # Recombine heads and project
        out = attn_out.transpose(1, 2).contiguous().view(B, Q_len, self.surface_dim)
        return self.norm_out(self.out_proj(out))


# =========================================================================
# 2. Multi-Token Prediction (MTP) Auxiliary Head
# =========================================================================

class MultiTokenPredictionHead(nn.Module):
    """
    DeepSeek-V3 Multi-Token Prediction (MTP) Module.
    
    Given the current fused motor representation h_t at step t, and the embedding of
    token y_{t+1}, predicts the subsequent token y_{t+2} in parallel.
    
    Benefits:
    1. Richer Representations: Forces the recurrent core and motor decoder to anticipate
       and plan 2 tokens into the future.
    2. Speculative Decoding: At inference time, generates 2 candidate tokens per step,
       achieving ~1.6x - 2.0x faster emission speed with exact verification.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int = 32000,
        surface_dim: int = 640,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.surface_dim = surface_dim

        # Projection combining step t motor representation and token t+1 embedding
        self.norm_in = nn.RMSNorm(hidden_dim + surface_dim)
        self.fc_gate = nn.Linear(hidden_dim + surface_dim, hidden_dim, bias=False)
        self.fc_val = nn.Linear(hidden_dim + surface_dim, hidden_dim, bias=False)
        self.norm_mid = nn.RMSNorm(hidden_dim)

        # Prediction head for step t+2
        self.head_t2 = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(
        self,
        h_t: torch.Tensor,
        emb_t1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predicts logits for token t+2 given motor state h_t and embedding of token t+1.
        Args:
            h_t    : [B, hidden_dim] (representation at step t)
            emb_t1 : [B, surface_dim] (embedding of token t+1)
        Returns:
            logits_t2 : [B, vocab_size] (logits for token t+2)
        """
        fused = self.norm_in(torch.cat([h_t, emb_t1], dim=-1))
        h = self.norm_mid(F.silu(self.fc_gate(fused)) * self.fc_val(fused))
        return self.head_t2(h)
