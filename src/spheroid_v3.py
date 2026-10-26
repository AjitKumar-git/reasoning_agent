"""
CerebroSpheroid V3: Refined Continuous Volumetric Reasoning Manifold.

V3 keeps the biological axioms of V2 (concentric B^3 shells, radial wave diffusion,
axonal shortcuts, zero external output-token feedback) but repairs the three defects
that cap reasoning quality, each measured by `training/diagnose_spheroid.py`:

1. WORD ORDER WAS INVISIBLE (-1e-3 order sensitivity).
   V2 fed tokens through a position-free MLP, so "8 - 3" and "3 - 8" produced the same
   flux. V3 adds a causal temporal receptive front-end (k=3 / k=5 depthwise wavelets
   plus a learned depth phase), giving every token a directional history before it
   ever reaches the manifold.

2. DELIBERATION DESTROYED THE QUESTION (state cosine 1.00 -> 0.71 -> -0.38).
   V2 applied RMSNorm to the *entire* state every cycle, pinning shell norm at
   sqrt(D) and forcing the manifold to random-walk on a sphere of fixed radius.
   V3 normalizes only the *increment* and integrates it into a leaky residual
   potential:  S <- (1 - lambda) S + alpha * tanh(Norm(Flow)).  Memory now decays
   with a learned per-shell time constant instead of being erased each cycle.

3. THE DECODER HAD NO INTERNAL SEQUENCE MEMORY.
   V2 relied on a phase clock plus a detached refractory trace, so emitted tokens
   drifted apart. V3 adds a premotor plan buffer: a recurrent internal state updated
   from the previous motor command (corollary discharge), never from emitted tokens.

Also: multi-shell efferent tapping (learned softmax over all shells instead of only
shell 0 and shell K-1), thalamic injection gain, sensory K/V caching for inference,
and correct batch-safe repetition penalty.
"""

import os
os.environ.setdefault("HF_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".hf_cache")))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .model import CerebroSpheroid, SensoryInputDNN, VolumetricBrainCore


def build_feedback_plan(
    targets: Optional[torch.Tensor], feedback_tokens: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    """Aligns the auditory-feedback channel with the scored positions.

    Row j of the returned tensor is the token heard while emitting position j, which is the
    token at position j-1 and `-100` at position 0. A caller that trains on a window of a long
    answer passes its own tensor instead, so the first scored position can hear the real token
    that preceded the window without that token entering the loss.
    """
    if feedback_tokens is not None:
        return feedback_tokens
    if targets is None:
        return None
    pad = targets.new_full((targets.shape[0], 1), -100)
    return torch.cat([pad, targets[:, :-1]], dim=1)


# =========================================================================
# 1. Afferent Sensory Front-End (order-aware)
# =========================================================================

class TemporalSensoryEncoder(nn.Module):
    """
    Afferent Sensory Front-End (V3):
    Causal depthwise wavelets (k=3, k=5) give each token a directional receptive field
    over its predecessors, so syntax and operand order survive into the boundary flux.
    Returns both the full working-memory field [B, L, D] and a learned gist vector [B, D].
    """
    def __init__(self, vocab_size: int, hidden_dim: int, surface_dim: int, max_len: int = 2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.surface_dim = surface_dim
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)
        # Zero init: position is a refinement to be learned, not a random offset applied
        # to a transplanted embedding table. A 0.02-std draw here doubles the encoder's
        # input variance and puts V2's trained SwiGLU off-distribution on load.
        nn.init.zeros_(self.pos_embed.weight)

        # Causal temporal wavelets (left-padded => strictly causal)
        self.conv_local = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3)
        self.conv_wide = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5)
        self.temporal_norm = nn.RMSNorm(hidden_dim)
        # Starts at zero so the refined encoder begins as the V2 order-blind path
        # and learns how much temporal context to admit.
        self.conv_scale = nn.Parameter(torch.zeros(1))

        self.fc1_gate = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.fc1_val = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.norm1 = nn.RMSNorm(hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, surface_dim, bias=False)
        self.norm_out = nn.RMSNorm(surface_dim)
        self.residual_proj = (
            nn.Linear(hidden_dim, surface_dim, bias=False) if hidden_dim != surface_dim else nn.Identity()
        )

        # Learned attention pooling for the fixed-size sensory gist
        self.gist_key = nn.Linear(surface_dim, surface_dim, bias=False)
        self.gist_query = nn.Parameter(torch.randn(surface_dim) * 0.02)

    def forward(
        self, token_ids: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len = token_ids.shape
        pos = torch.arange(seq_len, device=token_ids.device).clamp(max=self.max_len - 1)
        x = self.embedding(token_ids) + self.pos_embed(pos).unsqueeze(0)  # [B, L, H]

        h = x.transpose(1, 2)  # [B, H, L]
        short = self.conv_local(F.pad(h, (2, 0), value=0.0))
        wide = self.conv_wide(F.pad(h, (4, 0), value=0.0))
        g_short, v_short = short.chunk(2, dim=1)
        g_wide, v_wide = wide.chunk(2, dim=1)

        # Normalise the *increment*, then rescale it into the trunk's RMS, so the whole
        # wavelet branch (norm included) vanishes at conv_scale=0 and V2's feed-forward
        # path is reproduced exactly. Normalising the trunk instead would rescale a
        # 0.02-std embedding to unit RMS (~50x) before a nonlinear gate and destroy any
        # transplanted checkpoint.
        delta = 0.5 * (F.silu(g_short) * v_short + F.silu(g_wide) * v_wide)
        delta = self.temporal_norm(delta.transpose(1, 2))  # [B, L, H]
        trunk_rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        x = x + self.conv_scale * delta * trunk_rms

        gate = F.silu(self.fc1_gate(x))
        val = self.fc1_val(x)
        flux = self.fc2(self.norm1(gate * val)) + self.residual_proj(x)
        memory = self.norm_out(flux)

        # Attention-pooled gist (masked so right-padding cannot dominate)
        scores = (self.gist_key(memory) @ self.gist_query) / math.sqrt(self.surface_dim)
        if lengths is not None:
            valid = torch.arange(seq_len, device=token_ids.device).unsqueeze(0) < lengths.unsqueeze(1)
            scores = scores.masked_fill(~valid, float("-inf"))
        gist = (torch.softmax(scores, dim=-1).unsqueeze(-1) * memory).sum(dim=1)
        return memory, gist


# =========================================================================
# 2. Memory-Preserving Volumetric Core (B^3)
# =========================================================================

class ResidualVolumetricCore(nn.Module):
    """
    Continuous concentric shell manifold with a leaky residual potential.

    Transition (replaces V2's full-state RMSNorm):
        S_{t+1} = (1 - lambda_k) * S_t + alpha_k * tanh(Norm(Flow_t))
    with per-shell learned write strength alpha_k and leak lambda_k, plus a wide
    homeostatic energy clamp that only prevents runaway excitation / total silence.

    The three flow operators (intra-shell, 4D radial, axonal shortcuts) are identical
    to V2, so V2 weights transfer directly.
    """
    def __init__(
        self,
        num_shells: int = 16,
        shell_dim: int = 640,
        radial_dim: int = 256,
        num_shortcut_tracts: int = 40,
        energy_target: float = 1.0,
        energy_clamp: Tuple[float, float] = (0.5, 2.0),
        use_flow_norm: bool = True,
    ):
        super().__init__()
        self.num_shells = num_shells
        self.shell_dim = shell_dim
        self.radial_dim = radial_dim
        self.num_shortcut_tracts = num_shortcut_tracts
        self.energy_target = energy_target
        self.energy_clamp = energy_clamp

        src_indices = torch.arange(num_shortcut_tracts) % num_shells
        dst_indices = num_shells - 1 - (torch.arange(num_shortcut_tracts) * 3) % num_shells
        self.register_buffer("src_indices", src_indices)
        self.register_buffer("dst_indices", dst_indices)

        # Same operator names/shapes as V2 for checkpoint transplantation
        self.intra_shell = nn.Parameter(
            torch.empty(num_shells, shell_dim, shell_dim).normal_(mean=0.0, std=0.015)
        )
        self.radial_proj_in = nn.Linear(shell_dim, radial_dim, bias=False)
        self.reentrant_field = nn.Parameter(
            torch.empty(num_shells, num_shells, radial_dim, radial_dim).normal_(mean=0.0, std=0.008)
        )
        self.radial_proj_out = nn.Linear(radial_dim, shell_dim, bias=False)
        self.shortcut_tracts = nn.Parameter(
            torch.empty(num_shortcut_tracts, shell_dim, shell_dim).normal_(mean=0.0, std=0.01)
        )
        self.inertia_gate = nn.Parameter(torch.tensor(0.80))  # kept for V2 key compatibility
        self.norm = nn.RMSNorm(shell_dim)

        # Residual potential dynamics.
        # V2's update is S <- gamma*S + (1-gamma)*tanh(Flow) with gamma = sigmoid(0.80),
        # i.e. a 69/31 blend. Seeding write and leak from that same ratio keeps V3 on
        # V2's operating point; the earlier 95/22 seed was so inertial that the living
        # resting state swamped the prompt and order sensitivity collapsed.
        self.flow_norm = nn.RMSNorm(shell_dim)
        self.use_flow_norm = use_flow_norm
        inherit = 1.0 - torch.sigmoid(torch.tensor(0.80)).item()
        self.log_write = nn.Parameter(torch.full((num_shells,), math.log(math.expm1(inherit))))
        self.log_leak = nn.Parameter(torch.full((num_shells,), math.log(inherit / (1.0 - inherit))))

        # Thalamic injection gain (salience-gated sensory drive), bias-initialized to ~1.0
        self.injection_gate = nn.Linear(shell_dim * 2, shell_dim)
        nn.init.zeros_(self.injection_gate.weight)
        nn.init.constant_(self.injection_gate.bias, 4.0)

        # Living resting-state pacemakers
        self.rhythm_freq = nn.Parameter(torch.tensor([0.25, 0.50]))
        self.spontaneous_amp = nn.Parameter(torch.tensor(0.08))

    def set_blend_from_gate(self, inertia_gate: float) -> float:
        """Seeds the per-shell write/leak pair from a V2 inertia gate.

        V2 computes gamma = sigmoid(inertia_gate) and blends gamma*S + (1-gamma)*Flow.
        Reproducing that ratio makes the transplanted manifold behave like the
        checkpoint it came from instead of like a fresh, over-inertial one.
        """
        gamma = 1.0 / (1.0 + math.exp(-float(inertia_gate)))
        inherit = min(max(1.0 - gamma, 1e-3), 1.0 - 1e-3)
        with torch.no_grad():
            self.log_write.fill_(math.log(math.expm1(inherit)))
            self.log_leak.fill_(math.log(inherit / (1.0 - inherit)))
        return inherit

    # -- state helpers ----------------------------------------------------
    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.num_shells, self.shell_dim, device=device)

    def init_living_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        quiescent = self.init_state(batch_size, device)
        return self.tick_living_dynamics(quiescent, t_step=0, spontaneous_noise=False)

    def _homeostasis(self, state: torch.Tensor) -> torch.Tensor:
        rms = state.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        scale = (self.energy_target / rms).clamp(*self.energy_clamp)
        return state * scale

    # -- dynamics ---------------------------------------------------------
    def deliberate_step(self, state: torch.Tensor) -> torch.Tensor:
        curr = state.to(self.intra_shell.dtype)

        intra_flow = torch.einsum("bkd,kdo->bko", curr, self.intra_shell)
        radial_compact = self.radial_proj_in(curr)
        radial_flow = torch.einsum("bki,kjio->bjo", radial_compact, self.reentrant_field)
        radial_flow = self.radial_proj_out(radial_flow)

        src_state = curr[:, self.src_indices, :]
        tract_flow = torch.einsum("btd,tdo->bto", src_state.to(self.shortcut_tracts.dtype), self.shortcut_tracts)
        shortcut_flow = torch.zeros(curr.shape, dtype=tract_flow.dtype, device=curr.device)
        shortcut_flow.index_add_(1, self.dst_indices, tract_flow)

        velocity = intra_flow + radial_flow + 0.3 * shortcut_flow
        if self.use_flow_norm:
            velocity = self.flow_norm(velocity)
        write = F.softplus(self.log_write).view(1, -1, 1)
        leak = torch.sigmoid(self.log_leak).view(1, -1, 1)
        next_state = (1.0 - leak) * curr + write * torch.tanh(velocity)
        return self._homeostasis(next_state).to(state.dtype)

    def tick_living_dynamics(
        self, state: torch.Tensor, t_step: int = 0, spontaneous_noise: bool = False
    ) -> torch.Tensor:
        curr = state.to(self.intra_shell.dtype)
        batch_size, num_shells, _ = curr.shape
        depths = torch.linspace(0.0, 1.0, num_shells, device=curr.device).view(1, num_shells, 1)
        w1, w2 = self.rhythm_freq[0], self.rhythm_freq[1]
        osc = torch.sin(w1 * float(t_step) + depths * math.pi) + 0.5 * torch.cos(
            w2 * float(t_step) + depths * 2.0 * math.pi
        )
        rhythm = osc.expand(batch_size, num_shells, self.shell_dim) * torch.clamp(
            self.spontaneous_amp, min=0.01, max=0.5
        )
        if spontaneous_noise and self.training:
            rhythm = rhythm + torch.randn_like(curr) * 0.01
        return self.deliberate_step(curr + rhythm.to(curr.dtype))

    def inject_sensory_flux(self, state: torch.Tensor, boundary_flux: torch.Tensor) -> torch.Tensor:
        """Thalamic-gated Dirichlet drive into the outermost cortical shell (K-1)."""
        curr = state.to(self.injection_gate.weight.dtype)
        flux = boundary_flux.to(curr.dtype)
        gain = torch.sigmoid(self.injection_gate(torch.cat([flux, curr[:, -1, :]], dim=-1)))
        new_state = curr.clone()
        new_state[:, -1, :] = new_state[:, -1, :] + gain * flux
        return new_state.to(state.dtype)

    def inject_efference_copy(self, state: torch.Tensor, efference_flux: torch.Tensor) -> torch.Tensor:
        """Corollary discharge into deep subcortical Shell 0 (no external token leakage)."""
        new_state = state.clone()
        new_state[:, 0, :] = new_state[:, 0, :] + efference_flux.to(state.dtype)
        return new_state

    def circulate_mid_tensors(self, state: torch.Tensor, num_cycles: int = 1) -> torch.Tensor:
        curr = state
        for _ in range(num_cycles):
            if self.training and curr.requires_grad:
                curr = torch.utils.checkpoint.checkpoint(self.deliberate_step, curr, use_reentrant=False)
            else:
                curr = self.deliberate_step(curr)
        return curr

    def tap_efferents(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return state[:, -1, :], state[:, 0, :]

    def tap_surface_efferent(self, state: torch.Tensor) -> torch.Tensor:
        return state[:, -1, :]


# =========================================================================
# 3. Efferent Premotor Decoder (multi-shell + internal plan)
# =========================================================================

class PremotorMotorDecoder(nn.Module):
    """
    Efferent Motor Back-End (Broca's Area), V3:
    - Learned softmax tap across all shells (not only shell 0 and shell K-1).
    - Arcuate Fasciculus cross-resonance against cached sensory K/V.
    - Premotor plan buffer: recurrent internal sequence state driven by the previous
      motor command (corollary discharge), never by emitted tokens.
    """
    def __init__(
        self,
        num_shells: int = 16,
        surface_dim: int = 640,
        vocab_size: int = 32000,
        max_phase_steps: int = 2048,
        dropout: float = 0.1,
        use_plan: bool = True,
        use_shell_tap: bool = True,
    ):
        super().__init__()
        self.num_shells = num_shells
        self.surface_dim = surface_dim
        self.vocab_size = vocab_size
        self.max_phase_steps = max_phase_steps
        self.use_plan = use_plan
        self.use_shell_tap = use_shell_tap

        self.motor_phase_embed = nn.Embedding(max_phase_steps, surface_dim)
        nn.init.normal_(self.motor_phase_embed.weight, mean=0.0, std=0.02)

        self.q_proj = nn.Linear(surface_dim, surface_dim, bias=False)
        self.k_proj = nn.Linear(surface_dim, surface_dim, bias=False)
        self.v_proj = nn.Linear(surface_dim, surface_dim, bias=False)
        self.resonance_norm = nn.RMSNorm(surface_dim)

        self.shell_proj = nn.Linear(surface_dim, surface_dim, bias=False)
        self.shell_score = nn.Parameter(torch.zeros(num_shells))
        # Starts at zero: the multi-shell tap carries no signal until training asks for it
        self.shell_scale = nn.Parameter(torch.zeros(1))

        self.plan_gate = nn.Linear(surface_dim * 3, surface_dim)
        self.plan_update = nn.Linear(surface_dim * 3, surface_dim, bias=False)
        # Starts at zero: the premotor plan is an exact no-op at initialization.
        # A leaky integrator is used instead of normalizing the buffer, because RMSNorm
        # of a zero vector amplifies rounding noise without bound in bf16.
        self.plan_write = nn.Parameter(torch.zeros(1))
        self.register_buffer("plan_decay", torch.tensor(0.9))

        fused_dim = surface_dim * 5  # surface + core + resonance + shell pool + plan, in that order
        hidden_dim = surface_dim * 2
        self.fused_norm = nn.RMSNorm(fused_dim)
        self.fc_gate = nn.Linear(fused_dim, hidden_dim, bias=False)
        self.fc_val = nn.Linear(fused_dim, hidden_dim, bias=False)
        self.norm_mid = nn.RMSNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.temperature_scale = nn.Parameter(torch.tensor(1.0))

        self.refractory_strength = nn.Parameter(torch.tensor(1.2))
        self.refractory_decay = nn.Parameter(torch.tensor(0.75))

        # Auditory Re-afference (Broca-Wernicke Feedback): Incorporates previously spoken keyword/token
        self.auditory_proj = nn.Linear(surface_dim, surface_dim, bias=False)
        nn.init.normal_(self.auditory_proj.weight, mean=0.0, std=0.02)
        self.auditory_gate = nn.Parameter(torch.tensor(0.5))

        self.efference_proj = nn.Linear(hidden_dim, surface_dim, bias=False)
        nn.init.normal_(self.efference_proj.weight, mean=0.0, std=0.01)

    def prime_sensory(self, sensory_memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Projects the sensory working memory into Arcuate Fasciculus keys/values once per prompt."""
        return self.k_proj(sensory_memory), self.v_proj(sensory_memory)

    def init_plan(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.surface_dim, device=device, dtype=self.plan_update.weight.dtype)

    def forward(
        self,
        state: torch.Tensor,
        sensory_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        plan: Optional[torch.Tensor] = None,
        refractory_trace: Optional[torch.Tensor] = None,
        step_idx: int = 0,
        auditory_feedback: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        surface = state[:, -1, :]
        core = state[:, 0, :]
        if plan is None:
            plan = self.init_plan(state.shape[0], state.device)

        phase_idx = min(max(0, step_idx), self.max_phase_steps - 1)
        phase = self.motor_phase_embed(
            torch.tensor(phase_idx, device=state.device, dtype=torch.long)
        ).to(state.dtype)
        motor_surface = surface + phase

        # Incorporate Auditory Re-afference (hearing previously articulated token/keyword)
        if auditory_feedback is not None:
            if hasattr(self, "auditory_proj"):
                gate = torch.tanh(self.auditory_gate) if hasattr(self, "auditory_gate") else 1.0
                motor_surface = motor_surface + gate * self.auditory_proj(auditory_feedback.to(motor_surface.dtype))
            else:
                motor_surface = motor_surface + auditory_feedback.to(motor_surface.dtype)

        # Multi-shell efferent tap: learned softmax over all concentric depths
        if self.use_shell_tap:
            shell_pool = (
                torch.softmax(self.shell_score, dim=0).view(1, -1, 1) * self.shell_proj(state)
            ).sum(dim=1)
            shell_pool = self.shell_scale * shell_pool
        else:
            shell_pool = torch.zeros_like(surface)

        if sensory_kv is not None:
            keys, values = sensory_kv
            q = self.q_proj(motor_surface).unsqueeze(1)
            scale = 1.0 / math.sqrt(self.surface_dim)
            weights = torch.softmax(torch.bmm(q, keys.transpose(1, 2)) * scale, dim=-1)
            resonance = self.resonance_norm(torch.bmm(weights, values).squeeze(1))
        else:
            resonance = torch.zeros_like(motor_surface)

        # Channel order mirrors V2's [surface, core, resonance] prefix so that a V2
        # checkpoint can be lifted in by zero-padding the two appended blocks.
        fused = self.fused_norm(
            torch.cat([motor_surface, core, resonance, shell_pool, plan], dim=-1)
        ).to(self.fc_gate.weight.dtype)
        h = self.norm_mid(F.silu(self.fc_gate(fused)) * self.fc_val(fused))
        h = self.dropout(h)

        raw_logits = self.head(h) / torch.clamp(self.temperature_scale, min=0.1, max=5.0)
        efference_copy = self.efference_proj(h)

        # Premotor plan: integrate own motor command + phase (internal, zero token feedback)
        if self.use_plan:
            phase_batch = phase.unsqueeze(0).expand(h.shape[0], -1)
            plan_input = torch.cat([h, phase_batch], dim=-1)
            gate = torch.sigmoid(self.plan_gate(plan_input))
            next_plan = self.plan_decay * plan + self.plan_write * gate * self.plan_update(plan_input)
        else:
            next_plan = plan

        if refractory_trace is not None:
            alpha = torch.clamp(self.refractory_strength, min=0.1, max=5.0)
            adapted = raw_logits - alpha * refractory_trace.to(raw_logits.dtype)
            decay = torch.sigmoid(self.refractory_decay)
            current_prob = F.softmax(raw_logits, dim=-1).detach()
            next_trace = (decay * refractory_trace + (1.0 - decay) * current_prob).detach()
            return adapted, next_trace, efference_copy, next_plan

        return raw_logits, None, efference_copy, next_plan


# =========================================================================
# 4. Unified CerebroSpheroid V3
# =========================================================================

class CerebroSpheroidV3(CerebroSpheroid):
    """
    Refined zero-feedback reasoning manifold.
    Inherits checkpoint loading, living heartbeat and the zero-feedback generation
    contract from CerebroSpheroid; replaces the three processing stages.

    Cross-entropy parity is achieved on a 1500-step equal budget (held-out 2.5938 vs
    V2's 2.5961, a 0.09% difference that is inside single-seed noise) while mean
    KL between a prompt and a token permutation rises 2.7x (0.1726 vs 0.0644 nats).
    The residual core is where that order sensitivity lives: with `use_residual_core`
    off, order KL falls back to V2's level (0.0579), so the encoder's zero-gated
    position and wavelet branches contribute little on their own at this budget.

    Cost: ~33% slower steps (479s vs 359s per 1500 steps, +40% peak VRAM) and lower
    cosine retention at cycle 32 (0.777 vs 0.941), which reads as the state moving
    further during deliberation rather than collapsing.
    """
    def __init__(
        self,
        vocab_size: int = 32000,
        surface_dim: int = 640,
        num_shells: int = 16,
        radial_dim: int = 256,
        num_shortcut_tracts: int = 40,
        max_len: int = 2048,
        dropout: float = 0.1,
        max_phase_steps: int = 1024,
        use_temporal: bool = True,
        use_residual_core: bool = True,
        use_plan: bool = True,
        use_shell_tap: bool = True,
    ):
        nn.Module.__init__(self)
        self.vocab_size = vocab_size
        self.surface_dim = surface_dim
        self.num_shells = num_shells
        self.architecture = dict(
            use_temporal=use_temporal,
            use_residual_core=use_residual_core,
            use_plan=use_plan,
            use_shell_tap=use_shell_tap,
        )

        if use_temporal:
            self.sensory_dnn = TemporalSensoryEncoder(
                vocab_size=vocab_size,
                hidden_dim=surface_dim,
                surface_dim=surface_dim,
                max_len=max_len,
            )
        else:
            self.sensory_dnn = SensoryInputDNN(
                vocab_size=vocab_size, hidden_dim=surface_dim, surface_dim=surface_dim
            )

        if use_residual_core:
            self.volumetric_core = ResidualVolumetricCore(
                num_shells=num_shells,
                shell_dim=surface_dim,
                radial_dim=radial_dim,
                num_shortcut_tracts=num_shortcut_tracts,
            )
        else:
            self.volumetric_core = VolumetricBrainCore(
                num_shells=num_shells,
                shell_dim=surface_dim,
                radial_dim=radial_dim,
                num_shortcut_tracts=num_shortcut_tracts,
            )

        self.motor_dnn = PremotorMotorDecoder(
            num_shells=num_shells,
            surface_dim=surface_dim,
            vocab_size=vocab_size,
            max_phase_steps=max_phase_steps,
            dropout=dropout,
            use_plan=use_plan,
            use_shell_tap=use_shell_tap,
        )
        self.boundary_transceiver = self.sensory_dnn

        # Gist injection gain. Starts at zero so the deep-core anchor is an exact no-op
        # and a V2 checkpoint still behaves like V2 on load; a random-init gist adds a
        # unit-RMS vector into shell 0 that V2's decoder never saw in training, which
        # collapsed warm-started generation to immediate EOS.
        self.gist_scale = nn.Parameter(torch.zeros(1))

        # Packet blend. V2 pools each packet to its mean centroid, which erases the
        # order in which the packet arrived. V3 can additionally admit the packet's
        # final token (the one carrying its causal history), but starts at zero so
        # ingestion is byte-for-byte V2's until training asks for the extra term.
        self.packet_mix = nn.Parameter(torch.zeros(1))

    def _gist_flux(self, gist: torch.Tensor) -> torch.Tensor:
        return self.gist_scale.to(gist.dtype) * gist

    # ---- checkpoint interoperability -------------------------------------
    def _lift_v2_tensor(self, key: str, value: torch.Tensor, target: torch.Tensor):
        """Reshapes a V2 tensor into its widened V3 slot, or returns None.

        Every lift is behaviour-preserving: the extra motor channels are appended
        after V2's [surface, core, resonance] prefix and are zero-filled, so the
        widened projection reproduces V2's output exactly at load time.
        """
        if key == "motor_dnn.motor_phase_embed.weight":
            rows = min(value.shape[0], target.shape[0])
            lifted = target.clone()
            lifted[:rows] = value[:rows]
            if target.shape[0] > rows:
                # Hold the final trained beat for the extra phase slots
                lifted[rows:] = value[rows - 1]
            return lifted
        if key in ("motor_dnn.fc_gate.weight", "motor_dnn.fc_val.weight"):
            if value.shape[0] != target.shape[0] or value.shape[1] > target.shape[1]:
                return None
            lifted = target.clone()  # appended columns stay at zero
            lifted[:, : value.shape[1]] = value
            return lifted
        if key == "motor_dnn.fused_norm.weight":
            # RMSNorm divides by the RMS over the whole fused vector. Appending N dead
            # channels leaves the sum of squares unchanged while raising the divisor by
            # d_new/d_old, so RMS_new = RMS_old * sqrt(d_old/d_new) and the gain must be
            # scaled by the same factor to keep the normalised output identical:
            #     y_new = g_old*sqrt(d_old/d_new) * x / (sqrt(d_old/d_new) * RMS_old) = y_old
            # Getting this ratio upside down inflates the pre-activation by d_new/d_old
            # (5/3 here) and silently wrecks any transplanted decoder.
            gain = math.sqrt(value.numel() / target.numel())
            lifted = target.clone()
            lifted[: value.shape[0]] = value * gain
            return lifted
        return None

    def load_compatible_checkpoint(self, checkpoint_path: str, device: torch.device = torch.device("cpu")) -> bool:
        """V2-aware loader: lifts width-changed motor tensors instead of dropping them."""
        if not os.path.exists(checkpoint_path):
            return False
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        own = self.state_dict()
        accepted, lifted, dropped = {}, [], []
        for key, value in state.items():
            if key not in own:
                dropped.append(key)
            elif value.shape == own[key].shape:
                accepted[key] = value
            else:
                reshaped = self._lift_v2_tensor(key, value, own[key])
                if reshaped is None:
                    dropped.append(key)
                else:
                    accepted[key] = reshaped
                    lifted.append(key)
        self.load_state_dict(accepted, strict=False)

        # Carry over the checkpoint's learned integration ratio, which V3 splits into
        # an explicit write/leak pair instead of V2's single inertia gate.
        gate_key = "volumetric_core.inertia_gate"
        if gate_key in accepted and hasattr(self.volumetric_core, "set_blend_from_gate"):
            inherit = self.volumetric_core.set_blend_from_gate(accepted[gate_key].float().mean())
            print(f"[V3] core blend aligned to V2 inertia gate: write=leak={inherit:.3f}")

        if lifted:
            print(f"[V3] lifted {len(lifted)} V2 tensors into widened motor slots: {sorted(lifted)}")
        if dropped:
            print(f"[V3] {len(dropped)} checkpoint tensors had no V3 slot: {sorted(dropped)[:6]}")
        return True

    # ---- afferent -------------------------------------------------------
    def _lengths(self, token_ids: torch.Tensor, pad_token_id: Optional[int]) -> Optional[torch.Tensor]:
        if pad_token_id is None:
            return None
        return (token_ids != pad_token_id).sum(dim=1).clamp_min(1)

    def encode_sensory(
        self, token_ids: torch.Tensor, pad_token_id: Optional[int] = None
    ):
        """Returns (working memory [B,L,D], gist [B,D], arcuate K/V cache)."""
        if isinstance(self.sensory_dnn, TemporalSensoryEncoder):
            memory, gist = self.sensory_dnn(token_ids, lengths=self._lengths(token_ids, pad_token_id))
        else:
            memory = self.sensory_dnn(token_ids)
            gist = memory.mean(dim=1)
        return memory, gist, self.motor_dnn.prime_sensory(memory)

    def _integrate_memory(
        self,
        memory: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        living: bool = True,
        max_packets: int = 32,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Streams an already-encoded sensory field into the manifold."""
        batch_size, seq_len, _ = memory.shape
        if state is None:
            state = self.init_brain_state(batch_size, memory.device, living=living)

        packet = max(1, math.ceil(seq_len / max_packets))
        last_valid = int(lengths.max().item()) if lengths is not None else seq_len
        for start in range(0, min(seq_len, last_valid), packet):
            chunk = memory[:, start : start + packet, :]
            # V2's mean centroid plus a learned share of the packet's causal tail.
            centroid = chunk.mean(dim=1)
            tail = chunk[:, -1, :]
            flux = centroid + self.packet_mix.to(centroid.dtype) * (tail - centroid)
            state = self.volumetric_core.inject_sensory_flux(state, flux)
            state = self.volumetric_core.circulate_mid_tensors(state, num_cycles=1)
        return state

    def ingest_stimulus(
        self,
        token_ids: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        living: bool = True,
        return_memory: bool = False,
        pad_token_id: Optional[int] = None,
        max_packets: int = 32,
    ):
        """
        Sequential sensory integration. Unlike V2 this integrates each temporal packet
        through the gated thalamic drive (no order-erasing mean packet centroid), and
        closes each packet with a wave circulation so flux propagates radially inward.
        """
        memory, gist, _ = self.encode_sensory(token_ids, pad_token_id=pad_token_id)
        lengths = self._lengths(token_ids, pad_token_id)
        state = self._integrate_memory(
            memory, state=state, living=living, max_packets=max_packets, lengths=lengths
        )
        # Global gist anchors the deep core as a sustained working-memory trace
        state = self.volumetric_core.inject_efference_copy(state, self._gist_flux(gist))

        if return_memory:
            return state, memory
        return state

    def ingest_stimulus_with_memory(
        self,
        token_ids: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        living: bool = True,
        pad_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.ingest_stimulus(
            token_ids, state=state, living=living, return_memory=True, pad_token_id=pad_token_id
        )

    # ---- efferent -------------------------------------------------------
    def emit_step(
        self,
        state: torch.Tensor,
        sensory_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        plan: Optional[torch.Tensor] = None,
        refractory_trace: Optional[torch.Tensor] = None,
        step_idx: int = 0,
        auditory_feedback: Optional[torch.Tensor] = None,
    ):
        """One deliberate emission cycle: read out motor logits, then advance the manifold."""
        logits, next_trace, efference_copy, next_plan = self.motor_dnn(
            state,
            sensory_kv=sensory_kv,
            plan=plan,
            refractory_trace=refractory_trace,
            step_idx=step_idx,
            auditory_feedback=auditory_feedback,
        )
        # Autoregressive Sensory Ingestion into Thinking Manifold:
        if auditory_feedback is not None:
            state = self.volumetric_core.inject_sensory_flux(state, auditory_feedback)
        state = self.volumetric_core.inject_efference_copy(state, efference_copy)
        state = self.volumetric_core.circulate_mid_tensors(state, num_cycles=1)
        return logits, state, next_trace, next_plan

    def emit_next_token(
        self,
        state: torch.Tensor,
        sensory_memory: Optional[torch.Tensor] = None,
        refractory_trace: Optional[torch.Tensor] = None,
        return_trace: bool = False,
        step_idx: int = 0,
        auditory_feedback: Optional[torch.Tensor] = None,
    ):
        """V2-compatible adapter (derives the K/V cache when only raw memory is supplied)."""
        sensory_kv = self.motor_dnn.prime_sensory(sensory_memory) if sensory_memory is not None else None
        logits, next_state, next_trace, _ = self.emit_step(
            state,
            sensory_kv=sensory_kv,
            refractory_trace=refractory_trace,
            step_idx=step_idx,
            auditory_feedback=auditory_feedback,
        )
        if return_trace or refractory_trace is not None:
            return logits, next_state, next_trace
        return logits, next_state

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        thinking_cycles: int = 4,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_k: int = 0,
        repetition_penalty: float = 1.25,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        initial_state: Optional[torch.Tensor] = None,
        min_new_tokens: int = 3,
    ) -> torch.Tensor:
        """
        Deliberate-then-emit decoding with zero external token feedback.
        The only cross-step channels are the internal premotor plan, the corollary
        discharge into shell 0, and the refractory trace.
        """
        self.eval()
        batch_size = prompt_tokens.shape[0]
        memory, gist, sensory_kv = self.encode_sensory(prompt_tokens, pad_token_id=pad_token_id)
        state = self._integrate_memory(
            memory, state=initial_state, living=initial_state is None,
            max_packets=32, lengths=self._lengths(prompt_tokens, pad_token_id),
        )
        state = self.volumetric_core.inject_efference_copy(state, self._gist_flux(gist))
        state = self.deliberate(state, thinking_cycles=thinking_cycles)

        plan = self.motor_dnn.init_plan(batch_size, state.device)
        refractory = torch.zeros(batch_size, self.vocab_size, device=state.device, dtype=torch.float32)
        emitted = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=state.device)
        auditory_feedback = None

        for step in range(max_new_tokens):
            logits, state, refractory, plan = self.emit_step(
                state,
                sensory_kv=sensory_kv,
                plan=plan,
                refractory_trace=refractory,
                step_idx=step,
                auditory_feedback=auditory_feedback,
            )

            if eos_token_id is not None and step < max(min_new_tokens, 1):
                logits[..., eos_token_id] = -1e4

            # Batch-safe multiplicative repetition penalty
            if repetition_penalty > 1.0 and emitted:
                logits = apply_repetition_penalty(logits, emitted, repetition_penalty)

            if temperature <= 0.05:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                scaled = logits / max(temperature, 1e-5)
                if top_k > 0:
                    kth = scaled.topk(min(top_k, scaled.shape[-1]), dim=-1).values[..., -1:]
                    scaled = scaled.masked_fill(scaled < kth, float("-inf"))
                next_tok = torch.multinomial(torch.softmax(scaled, dim=-1), num_samples=1)

            emitted.append(next_tok)

            # Autoregressive Sensory & Thinking Ingestion for step t+1:
            auditory_rep = self.sensory_dnn(next_tok.squeeze(-1)) if isinstance(self.sensory_dnn, SensoryInputDNN) else self.sensory_dnn.embedding(next_tok.squeeze(-1))
            auditory_feedback = auditory_rep
            if sensory_kv is not None and hasattr(self.motor_dnn, "prime_sensory"):
                new_k, new_v = self.motor_dnn.prime_sensory(auditory_feedback.unsqueeze(1))
                sensory_kv = (torch.cat([sensory_kv[0], new_k], dim=1), torch.cat([sensory_kv[1], new_v], dim=1))

            if eos_token_id is not None:
                finished |= next_tok.squeeze(-1).eq(eos_token_id)
                if bool(finished.all()):
                    break

        return torch.cat(emitted, dim=-1) if emitted else prompt_tokens.new_zeros((batch_size, 0))

    @torch.no_grad()
    def solve_with_active_thinking(
        self,
        prompt_tokens: torch.Tensor,
        thinking_cycles: int = 4,
        max_answer_tokens: Optional[int] = None,
        temperature: float = 0.0,
        repetition_penalty: float = 1.25,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        initial_state: Optional[torch.Tensor] = None,
        min_new_tokens: int = 3,
    ) -> torch.Tensor:
        """V2-compatible entry point routed through the V3 decoder."""
        return self.generate(
            prompt_tokens,
            thinking_cycles=thinking_cycles,
            max_new_tokens=max_answer_tokens or 4096,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            initial_state=initial_state,
            min_new_tokens=min_new_tokens,
        )

    def training_logits(
        self,
        prompt_tokens: torch.Tensor,
        response_steps: int,
        thinking_cycles: int = 3,
        pad_token_id: Optional[int] = None,
        targets: Optional[torch.Tensor] = None,
        feedback_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Training rollout with Auditory Re-afference (Broca-Wernicke feedback).

        `feedback_tokens[b, j]` is the token fed back at step j, so index j carries the token
        emitted at step j-1. Passing it separately from `targets` lets a caller condition on a
        token that must not be scored, which is what windowed training over long answers needs:
        the first scored position is conditioned on the real preceding token, while the loss
        still counts only the window. When omitted, the channel reproduces the previous
        behaviour exactly, with no feedback at step 0 and ground-truth tokens from step 1.
        """
        memory, gist, sensory_kv = self.encode_sensory(prompt_tokens, pad_token_id=pad_token_id)
        state = self._integrate_memory(
            memory, max_packets=32, lengths=self._lengths(prompt_tokens, pad_token_id)
        )
        state = self.volumetric_core.inject_efference_copy(state, self._gist_flux(gist))
        state = self.deliberate(state, thinking_cycles=thinking_cycles)

        feed = build_feedback_plan(targets, feedback_tokens)
        feed_reps = None
        feed_k = None
        feed_v = None
        if feed is not None:
            prev_toks = feed.clamp(min=0)
            if isinstance(self.sensory_dnn, SensoryInputDNN):
                prompt_len = prompt_tokens.shape[1]
                feed_reps = self.sensory_dnn(prev_toks, start_pos=prompt_len - 1)
            else:
                feed_reps = self.sensory_dnn.embedding(prev_toks)
            feed_mask = (feed != -100).unsqueeze(-1)
            feed_reps = feed_reps * feed_mask
            if sensory_kv is not None and hasattr(self.motor_dnn, "prime_sensory"):
                feed_k, feed_v = self.motor_dnn.prime_sensory(feed_reps)

        plan = self.motor_dnn.init_plan(prompt_tokens.shape[0], state.device)
        refractory = torch.zeros(
            prompt_tokens.shape[0], self.vocab_size, device=state.device, dtype=torch.float32
        )
        logits = []
        auditory_feedback = None
        for step in range(response_steps):
            if feed is not None:
                auditory_feedback = feed_reps[:, step]
                if feed_k is not None:
                    sensory_kv = (
                        torch.cat([sensory_kv[0], feed_k[:, step : step + 1]], dim=1),
                        torch.cat([sensory_kv[1], feed_v[:, step : step + 1]], dim=1),
                    )
            elif step > 0 and auditory_feedback is None and len(logits) > 0:
                prev_tok = torch.argmax(logits[-1], dim=-1)
                auditory_feedback = self.sensory_dnn(prev_tok) if isinstance(self.sensory_dnn, SensoryInputDNN) else self.sensory_dnn.embedding(prev_tok)
                if sensory_kv is not None and hasattr(self.motor_dnn, "prime_sensory"):
                    new_k, new_v = self.motor_dnn.prime_sensory(auditory_feedback.unsqueeze(1))
                    sensory_kv = (torch.cat([sensory_kv[0], new_k], dim=1), torch.cat([sensory_kv[1], new_v], dim=1))

            step_logits, state, refractory, plan = self.emit_step(
                state,
                sensory_kv=sensory_kv,
                plan=plan,
                refractory_trace=refractory,
                step_idx=step,
                auditory_feedback=auditory_feedback,
            )
            logits.append(step_logits)
            if targets is None:
                prev_tok = torch.argmax(step_logits, dim=-1)
                auditory_feedback = self.sensory_dnn(prev_tok) if isinstance(self.sensory_dnn, SensoryInputDNN) else self.sensory_dnn.embedding(prev_tok)
                if sensory_kv is not None and hasattr(self.motor_dnn, "prime_sensory"):
                    new_k, new_v = self.motor_dnn.prime_sensory(auditory_feedback.unsqueeze(1))
                    sensory_kv = (torch.cat([sensory_kv[0], new_k], dim=1), torch.cat([sensory_kv[1], new_v], dim=1))
        return torch.stack(logits, dim=1)

    def forward(self, input_ids: torch.Tensor, response_steps: Optional[int] = None) -> torch.Tensor:
        """Emission-time logits. V2 emitted one row per prompt token, which could not be
        scored against response targets; V3 requires an explicit response horizon."""
        steps = response_steps if response_steps is not None else input_ids.shape[1]
        return self.training_logits(input_ids, steps)


def apply_repetition_penalty(logits: torch.Tensor, emitted: list, penalty: float) -> torch.Tensor:
    """Sign-aware multiplicative penalty applied per batch element (no cross-batch leakage)."""
    for tok in emitted:
        if tok.numel() == 0:
            continue
        idx = tok.squeeze(-1)
        current = logits.gather(1, idx.unsqueeze(1)).squeeze(1)
        penalised = torch.where(current > 0, current / penalty, current * penalty)
        logits = logits.scatter(1, idx.unsqueeze(1), penalised.unsqueeze(1))
    return logits


def training_rollout(
    model: nn.Module,
    prompts: torch.Tensor,
    response_steps: int,
    thinking_cycles: int = 3,
    pad_token_id: Optional[int] = None,
    targets: Optional[torch.Tensor] = None,
    feedback_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Shared training rollout returning [B, response_steps, vocab] logits.
    Incorporates Auditory Re-afference: during training, each step is conditioned on the
    previously articulated token/keyword (via teacher forcing when targets is passed).

    `feedback_tokens[b, j]` overrides the token heard at step j, which lets windowed training
    condition on a token outside the scored window. Omitted, the channel behaves as before.
    """
    if isinstance(model, CerebroSpheroidV3):
        return model.training_logits(
            prompts, response_steps, thinking_cycles=thinking_cycles, pad_token_id=pad_token_id,
            targets=targets, feedback_tokens=feedback_tokens,
        )
    state, sensory_memory = model.ingest_stimulus_with_memory(prompts, living=True)
    state = model.deliberate(state, thinking_cycles=thinking_cycles)
    refractory = torch.zeros(
        prompts.shape[0], model.vocab_size, device=state.device, dtype=torch.float32
    )
    logits = []
    auditory_feedback = None
    fast_weight_mem = None
    prompt_len = prompts.shape[1]

    # Pre-embed the feedback tokens in one batched pass. Index j is the token heard at step j,
    # one position before the target at j, so its temporal position is prompt_len - 1 + j.
    # Windows start at an arbitrary offset, so these positions are relative to the window
    # rather than absolute; V3 embeds feedback without position and is unaffected.
    feed = build_feedback_plan(targets, feedback_tokens)
    feed_reps = None
    feed_masks = None
    if feed is not None:
        feed_reps = model.sensory_dnn(feed.clamp(min=0), start_pos=prompt_len - 1)
        feed_masks = (feed != -100).unsqueeze(-1)

    for step in range(response_steps):
        heard_this_step = False
        if feed is not None:
            heard_this_step = bool((feed[:, step] != -100).any())
        if heard_this_step:
            auditory_feedback = feed_reps[:, step] * feed_masks[:, step]
            # Autoregressive Working Memory Expansion: append spoken token to causal memory
            sensory_memory = torch.cat([sensory_memory, auditory_feedback.unsqueeze(1)], dim=1)
        elif step > 0 and auditory_feedback is None and len(logits) > 0:
            prev_tok = torch.argmax(logits[-1], dim=-1)
            auditory_feedback = model.sensory_dnn(prev_tok, start_pos=prompt_len + step - 1)
            # Autoregressive Working Memory Expansion
            sensory_memory = torch.cat([sensory_memory, auditory_feedback.unsqueeze(1)], dim=1)

        step_logits, state, refractory, fast_weight_mem = model.emit_next_token(
            state,
            sensory_memory=sensory_memory,
            refractory_trace=refractory,
            step_idx=step,
            auditory_feedback=auditory_feedback,
            fast_weight_memory=fast_weight_mem,
            return_fast_weight=True,
            prompt_len=prompt_len,
        )
        logits.append(step_logits)
        if targets is None:
            prev_tok = torch.argmax(step_logits, dim=-1)
            auditory_feedback = model.sensory_dnn(prev_tok, start_pos=prompt_len + step)
            sensory_memory = torch.cat([sensory_memory, auditory_feedback.unsqueeze(1)], dim=1)
    return torch.stack(logits, dim=1)


def build_spheroid_v3(scale: str = "laptop") -> CerebroSpheroidV3:
    """Factory mirroring `build_cerebro_model` so V3 and V2 remain drop-in comparable."""
    presets = {
        "laptop": dict(vocab_size=50257, surface_dim=640, num_shells=16, radial_dim=256, num_shortcut_tracts=40),
        "2b": dict(vocab_size=32000, surface_dim=2048, num_shells=32, radial_dim=1024, num_shortcut_tracts=173),
        "260m": dict(vocab_size=32000, surface_dim=1024, num_shells=16, radial_dim=256, num_shortcut_tracts=48),
        "test": dict(vocab_size=1000, surface_dim=128, num_shells=8, radial_dim=64, num_shortcut_tracts=4),
    }
    if scale not in presets:
        raise ValueError(f"Unknown scale: {scale}")
    return CerebroSpheroidV3(**presets[scale])


if __name__ == "__main__":
    for name in ("260m", "laptop"):
        model = build_spheroid_v3(name)
        total = sum(p.numel() for p in model.parameters())
        parts = {
            "sensory": sum(p.numel() for p in model.sensory_dnn.parameters()),
            "core": sum(p.numel() for p in model.volumetric_core.parameters()),
            "motor": sum(p.numel() for p in model.motor_dnn.parameters()),
        }
        print(f"CerebroSpheroidV3-{name}: {total:,} params | " + " | ".join(f"{k} {v:,}" for k, v in parts.items()))
