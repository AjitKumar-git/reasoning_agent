"""
Associative Fast-Weight Synaptic Memory Module for CerebroSpheroid
Establishes inter-token syntactic and semantic relations (subject-verb agreement,
entity binding, discourse coherence) in O(1) constant memory without storing or
re-feeding past token sequences.

Mathematical Formulation:
  Readout (Step t):
    q_t = W_q(h_t)                              # [B, H, d_k]
    r_t = (M_{t-1} @ q_t) / sqrt(d_k)           # [B, H, d_v]
    assoc_context = W_o(RMSNorm(flatten(r_t)))  # [B, D]

  Synaptic Update:
    k_t = L2_Norm(W_k(h_t))                     # [B, H, d_k]
    v_t = W_v(h_t)                              # [B, H, d_v]
    M_t = sigmoid(decay) * M_{t-1} + sigmoid(eta) * (v_t @ k_t^T)
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AssociativeFastWeightMemory(nn.Module):
    """
    Multi-Head Hebbian Fast-Weight Associative Matrix.
    Replaces autoregressive KV caching with a constant-sized synaptic matrix
    that binds inter-token relations directly in latent space.
    """
    def __init__(
        self,
        dim: int = 640,
        num_heads: int = 4,
        decay_init: float = 0.92,
        write_rate_init: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = dim // num_heads

        # Linear projections for Associative Queries, Keys, Values
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.RMSNorm(dim)

        # Learnable plasticity dynamics: decay factor and write rate
        # Initialized such that sigmoid(decay_param) == decay_init
        decay_logit = math.log(decay_init / (1.0 - decay_init))
        self.decay_param = nn.Parameter(torch.tensor(decay_logit))

        write_logit = math.log(write_rate_init / (1.0 - write_rate_init))
        self.write_param = nn.Parameter(torch.tensor(write_logit))

        # Output modulation gate
        self.output_gate = nn.Parameter(torch.tensor(0.5))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.k_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)

    def init_memory(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Initializes empty synaptic matrix M_0 = 0 [B, H, d_k, d_v]."""
        return torch.zeros(
            batch_size, self.num_heads, self.head_dim, self.head_dim,
            device=device, dtype=dtype
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Step-wise associative readout and synaptic update.
        Args:
          hidden_state: [B, D] active motor/articulatory representation
          memory: [B, H, d_k, d_v] prior associative matrix M_{t-1}
        Returns:
          context: [B, D] associative inter-token context
          next_memory: [B, H, d_k, d_v] updated matrix M_t
        """
        B, D = hidden_state.shape
        if memory is None:
            memory = self.init_memory(B, hidden_state.device, dtype=hidden_state.dtype)

        # Project Q, K, V
        q = self.q_proj(hidden_state).view(B, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_state).view(B, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_state).view(B, self.num_heads, self.head_dim)

        # 1. Readout from M_{t-1}: r_t = (M_{t-1} @ q) / sqrt(d)
        scale = 1.0 / math.sqrt(self.head_dim)
        # memory is [B, H, d_v, d_k], q is [B, H, d_k, 1]
        q_col = q.unsqueeze(-1)
        r = torch.matmul(memory, q_col).squeeze(-1) * scale  # [B, H, d_v]
        r_flat = r.contiguous().view(B, D)
        context = self.out_proj(self.norm(r_flat)) * torch.tanh(self.output_gate)

        # 2. Plastic Synaptic Update: M_t = lambda * M_{t-1} + eta * (v @ k^T)
        k_norm = F.normalize(k, p=2, dim=-1)  # Bounded keys prevent runaway divergence
        vk_outer = torch.matmul(v.unsqueeze(-1), k_norm.unsqueeze(-2))  # [B, H, d_v, d_k]

        decay = torch.sigmoid(self.decay_param)
        write_rate = torch.sigmoid(self.write_param)
        next_memory = decay * memory + write_rate * vk_outer

        return context, next_memory
