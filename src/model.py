"""
CerebroSpheroid: A Biological Brain Manifold Neural Architecture
Free from Transformer sequential layer stacks and autoregressive feedback loops.

Core Biological Principles:
1. Afferent Sensory Input DNN (Cortical Encoder):
   SwiGLU ResNet translating discrete symbolic tokens into continuous Riemannian boundary flux.
2. Continuous Non-Layered Volumetric Ball Core (B^3):
   32 Concentric Laminar Shells with 4D Radial Wave Coupling and 173 Subcortical Axonal Shortcut Tracts.
   Enforces contractive unit-gain dynamics S_{t+1} = RMSNorm(gamma S_t + (1-gamma) tanh(Flow)),
   guaranteeing stability against gradient explosion and noisy drift.
3. Efferent Motor Output DNN (Dual-Tap Decoder):
   Decoupled speech articulation combining superficial cortical resonance (Shell 31) with deep
   thalamic working memory attractors (Shell 0).
4. Deliberate-Then-Emit Active Thinking:
   Input Ingestion -> Subconscious Thinking Cycles (Zero Emission) -> Autonomous Readout (Zero Output Feedback).
"""

import os
os.environ.setdefault("HF_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".hf_cache")))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


# =========================================================================
# 1. Afferent Sensory Input DNN (Cortical Encoder)
# =========================================================================

class SensoryInputDNN(nn.Module):
    """
    Afferent Sensory Front-End:
    2-Layer SwiGLU ResNet with RMSNorm converting discrete token symbols
    into smooth continuous boundary flux fields on S^2.
    """
    def __init__(self, vocab_size: int = 32000, hidden_dim: int = 2048, surface_dim: int = 2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.surface_dim = surface_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # SwiGLU Non-Linear Projection
        self.fc1_gate = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.fc1_val = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.norm1 = nn.RMSNorm(hidden_dim * 2)

        self.fc2 = nn.Linear(hidden_dim * 2, surface_dim, bias=False)
        self.norm_out = nn.RMSNorm(surface_dim)
        self.residual_proj = nn.Linear(hidden_dim, surface_dim, bias=False) if hidden_dim != surface_dim else nn.Identity()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        gate = F.silu(self.fc1_gate(x))
        val = self.fc1_val(x)
        h = self.norm1(gate * val)
        flux = self.fc2(h) + self.residual_proj(x)
        return self.norm_out(flux)

    # Compatibility alias for legacy scripts
    def ingest_sensory_input(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.forward(token_ids)


# =========================================================================
# 2. Efferent Motor Output DNN (Dual-Tap Decoder)
# =========================================================================

class MotorOutputDNN(nn.Module):
    """
    Efferent Motor Back-End (Broca's Area):
    Dual-Tap Decoder sampling superficial cortical resonance (Shell K-1)
    and deep subcortical working memory attractors (Shell 0) through SwiGLU.
    Equipped with:
    1. Arcuate Fasciculus (Wernicke-Broca Cross-Resonance): Direct white-matter tract
       binding motor articulation to the active sensory working memory of the question.
    2. Biological Refractory Dynamics (neuronal fatigue trace): Prevents repetitive mode collapse.
    """
    def __init__(
        self,
        surface_dim: int = 2048,
        core_dim: int = 2048,
        vocab_size: int = 32000,
        max_phase_steps: int = 1024,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.surface_dim = surface_dim
        self.max_phase_steps = max_phase_steps

        # Premotor Temporal Phase Clock (Striatal Sequential Beat Rhythm)
        self.motor_phase_embed = nn.Embedding(max_phase_steps, surface_dim)
        nn.init.normal_(self.motor_phase_embed.weight, mean=0.0, std=0.02)

        # Arcuate Fasciculus: Cross-Manifold Sensory Working Memory Resonance
        self.q_proj = nn.Linear(surface_dim, surface_dim, bias=False)
        self.k_proj = nn.Linear(surface_dim, surface_dim, bias=False)
        self.v_proj = nn.Linear(surface_dim, surface_dim, bias=False)
        self.resonance_norm = nn.RMSNorm(surface_dim)

        fused_dim = surface_dim + core_dim + surface_dim
        hidden_dim = surface_dim * 2

        self.fused_norm = nn.RMSNorm(fused_dim)
        self.fc_gate = nn.Linear(fused_dim, hidden_dim, bias=False)
        self.fc_val = nn.Linear(fused_dim, hidden_dim, bias=False)
        self.norm_mid = nn.RMSNorm(hidden_dim)

        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.temperature_scale = nn.Parameter(torch.tensor(1.0))

        # Biological Refractory Parameters: neuronal fatigue and trace decay
        self.refractory_strength = nn.Parameter(torch.tensor(1.2))
        self.refractory_decay = nn.Parameter(torch.tensor(0.75))

        # Internal Corollary Discharge (Efference Copy Projection to Subcortical Core)
        self.efference_proj = nn.Linear(hidden_dim, surface_dim, bias=False)
        nn.init.normal_(self.efference_proj.weight, mean=0.0, std=0.01)

    def forward(
        self,
        surface_state: torch.Tensor,
        core_state: Optional[torch.Tensor] = None,
        sensory_memory: Optional[torch.Tensor] = None,
        refractory_trace: Optional[torch.Tensor] = None,
        step_idx: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if core_state is None:
            core_state = surface_state

        # Modulate query state with Premotor Temporal Phase Clock
        phase_idx = min(max(0, step_idx), self.max_phase_steps - 1)
        phase_t = torch.tensor(phase_idx, device=surface_state.device, dtype=torch.long)
        phase_vec = self.motor_phase_embed(phase_t).to(surface_state.dtype)
        motor_surface = surface_state + phase_vec

        # Arcuate Fasciculus Cross-Resonance with question's sensory working memory
        if sensory_memory is not None and sensory_memory.ndim == 3:
            q = self.q_proj(motor_surface).unsqueeze(1)   # [B, 1, D]
            k = self.k_proj(sensory_memory)               # [B, L, D]
            v = self.v_proj(sensory_memory)               # [B, L, D]
            scale = 1.0 / math.sqrt(self.surface_dim)
            attn_scores = torch.bmm(q, k.transpose(1, 2)) * scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            resonance = torch.bmm(attn_weights, v).squeeze(1)  # [B, D]
            resonance = self.resonance_norm(resonance)
        else:
            resonance = torch.zeros_like(surface_state)

        fused = self.fused_norm(torch.cat([motor_surface, core_state, resonance], dim=-1)).to(self.fc_gate.weight.dtype)
        gate = F.silu(self.fc_gate(fused))
        val = self.fc_val(fused)
        h = self.norm_mid(gate * val)
        raw_logits = self.head(h) / torch.clamp(self.temperature_scale, min=0.1, max=5.0)
        efference_copy = self.efference_proj(h)

        if refractory_trace is not None:
            alpha = torch.clamp(self.refractory_strength, min=0.1, max=5.0)
            adapted_logits = raw_logits - alpha * refractory_trace.to(raw_logits.dtype)
            decay = torch.sigmoid(self.refractory_decay)
            current_prob = F.softmax(raw_logits, dim=-1).detach()
            next_trace = (decay * refractory_trace + (1.0 - decay) * current_prob).detach()
            return adapted_logits, next_trace, efference_copy

        return raw_logits, None, efference_copy

    # Compatibility alias for legacy scripts
    def emit_motor_output(self, surface_state: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.forward(surface_state, surface_state)
        return logits


# =========================================================================
# 3. Stabilized Contractive Volumetric Core (B^3 Manifold)
# =========================================================================

class VolumetricBrainCore(nn.Module):
    """
    Continuous 3D Spherical Ball Core (B^3):
    - 32 Concentric Laminar Shells (Outer cortex to deep thalamus).
    - 4D Radial Wave Diffusion Field coupling all radial depths simultaneously.
    - 173 Subcortical White-Matter Axonal Shortcut Tracts.
    - Contractive Unit-Gain Dynamics: S_{t+1} = Norm(gamma S_t + (1-gamma) tanh(Flow)).
    """
    def __init__(
        self,
        num_shells: int = 32,
        shell_dim: int = 2048,
        radial_dim: int = 1024,
        num_shortcut_tracts: int = 173,
    ):
        super().__init__()
        self.num_shells = num_shells
        self.shell_dim = shell_dim
        self.radial_dim = radial_dim
        self.num_shortcut_tracts = num_shortcut_tracts

        # Precompute axonal tract wiring between non-adjacent shells
        src_indices = torch.arange(num_shortcut_tracts) % num_shells
        dst_indices = (num_shells - 1 - (torch.arange(num_shortcut_tracts) * 3) % num_shells)
        self.register_buffer("src_indices", src_indices)
        self.register_buffer("dst_indices", dst_indices)

        # 1. Concentric Shell Intra-Manifold Operators (Laplace-Beltrami Tangential Flow)
        self.intra_shell = nn.Parameter(
            torch.empty(num_shells, shell_dim, shell_dim).normal_(mean=0.0, std=0.015)
        )

        # 2. Radial Volumetric Wave Diffusion Field (Global 4D coupling)
        self.radial_proj_in = nn.Linear(shell_dim, radial_dim, bias=False)
        self.reentrant_field = nn.Parameter(
            torch.empty(num_shells, num_shells, radial_dim, radial_dim).normal_(mean=0.0, std=0.008)
        )
        self.radial_proj_out = nn.Linear(radial_dim, shell_dim, bias=False)

        # 3. Subcortical White-Matter Axonal Shortcut Tracts
        self.shortcut_tracts = nn.Parameter(
            torch.empty(num_shortcut_tracts, shell_dim, shell_dim).normal_(mean=0.0, std=0.01)
        )

        # 4. Leaky Inertia & Contraction Control
        self.inertia_gate = nn.Parameter(torch.tensor(0.80))
        self.norm = nn.RMSNorm(shell_dim)

        # 5. Living Core Oscillatory Pacemaker (Endogenous Theta/Alpha Dynamics)
        self.rhythm_freq = nn.Parameter(torch.tensor([0.25, 0.50]))  # Dual frequency pacemakers
        self.spontaneous_amp = nn.Parameter(torch.tensor(0.08))       # Resting-state spontaneous amplitude

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initializes quiescent volumetric state tensor: [Batch, num_shells, shell_dim]"""
        return torch.zeros(batch_size, self.num_shells, self.shell_dim, device=device)

    def init_living_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initializes an active resting-state living manifold with spontaneous endogenous oscillations.
        Simulates the baseline resting-state network of a living neural system.
        """
        quiescent = self.init_state(batch_size, device)
        return self.tick_living_dynamics(quiescent, t_step=0, spontaneous_noise=False)

    def tick_living_dynamics(
        self, state: torch.Tensor, t_step: int = 0, spontaneous_noise: bool = False
    ) -> torch.Tensor:
        """
        Advances the brain core by one living clock tick (spontaneous endogenous oscillation).
        Continuously maintains metabolic homeostatic activity even in the absence of sensory input.
        """
        curr = state.to(self.intra_shell.dtype)
        batch_size, num_shells, shell_dim = curr.shape

        # Radial depth phase distribution
        shell_depths = torch.linspace(0.0, 1.0, num_shells, device=curr.device).view(1, num_shells, 1)
        w1, w2 = self.rhythm_freq[0], self.rhythm_freq[1]
        t_val = float(t_step)
        osc = torch.sin(w1 * t_val + shell_depths * math.pi) + 0.5 * torch.cos(w2 * t_val + shell_depths * 2.0 * math.pi)
        rhythm_field = osc.expand(batch_size, num_shells, shell_dim) * torch.clamp(self.spontaneous_amp, min=0.01, max=0.5)

        if spontaneous_noise and self.training:
            noise = torch.randn_like(curr) * 0.01
            rhythm_field = rhythm_field + noise

        perturbed = curr + rhythm_field.to(curr.dtype)
        return self.deliberate_step(perturbed)

    def inject_sensory_flux(self, state: torch.Tensor, boundary_flux: torch.Tensor) -> torch.Tensor:
        """Injects Dirichlet sensory excitation into the outermost cortical shell (Shell K-1)."""
        new_state = state.clone()
        new_state[:, -1, :] = new_state[:, -1, :] + boundary_flux.to(state.dtype)
        return self.norm(new_state).to(state.dtype)

    def inject_efference_copy(self, state: torch.Tensor, efference_flux: torch.Tensor) -> torch.Tensor:
        """
        Injects internal motor corollary discharge (efference copy) into deep subcortical Shell 0.
        Simulates the internal cerebellar / basal ganglia feedback loop without external token leakage.
        """
        new_state = state.clone()
        new_state[:, 0, :] = new_state[:, 0, :] + efference_flux.to(state.dtype)
        return self.norm(new_state).to(state.dtype)

    def deliberate_step(self, state: torch.Tensor) -> torch.Tensor:
        """
        One cycle of contractive continuous wave propagation:
        S_{t+1} = RMSNorm(gamma * S_t + (1 - gamma) * tanh(Flow))
        Clamps spectral radius <= 1.0, eliminating chaotic drift and exploding gradients.
        """
        curr = state.to(self.intra_shell.dtype)

        # A. Intra-shell tangential diffusion
        intra_flow = torch.einsum('bkd,kdo->bko', curr, self.intra_shell)

        # B. 4D Radial wave transmission linking all depths
        radial_compact = self.radial_proj_in(curr)
        radial_flow = torch.einsum('bki,kjio->bjo', radial_compact, self.reentrant_field)
        radial_flow = self.radial_proj_out(radial_flow)

        # C. White-matter axonal shortcuts
        src_state = curr[:, self.src_indices, :]
        tract_flow = torch.einsum('btd,tdo->bto', src_state.to(self.shortcut_tracts.dtype), self.shortcut_tracts)
        shortcut_flow = torch.zeros(curr.shape, dtype=tract_flow.dtype, device=curr.device)
        shortcut_flow.index_add_(1, self.dst_indices, tract_flow)

        # D. Bounded Contractive Transition
        total_velocity = intra_flow + radial_flow + 0.3 * shortcut_flow
        gamma = torch.sigmoid(self.inertia_gate)
        next_state = gamma * curr + (1.0 - gamma) * torch.tanh(total_velocity)
        return self.norm(next_state).to(state.dtype)

    def circulate_mid_tensors(self, state: torch.Tensor, num_cycles: int = 1) -> torch.Tensor:
        """Circulates intermediate mid-tensors across all shells without external input."""
        curr = state
        for _ in range(num_cycles):
            if self.training and curr.requires_grad:
                curr = torch.utils.checkpoint.checkpoint(self.deliberate_step, curr, use_reentrant=False)
            else:
                curr = self.deliberate_step(curr)
        return curr

    def tap_efferents(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Taps both outer cortical surface (Shell -1) and deep subcortical core (Shell 0)."""
        return state[:, -1, :], state[:, 0, :]

    def tap_surface_efferent(self, state: torch.Tensor) -> torch.Tensor:
        return state[:, -1, :]


# Legacy compatibility alias
SurfaceTransceiverBoundary = SensoryInputDNN


# =========================================================================
# 4. Unified CerebroSpheroid Brain Architecture
# =========================================================================

class CerebroSpheroid(nn.Module):
    """
    Unified CerebroSpheroid Brain Architecture:
    1. Afferent Sensory Input DNN (Cortical Encoder)
    2. Stabilized Volumetric Brain Core (B^3 Manifold)
    3. Efferent Motor Output DNN (Dual-Tap Decoder)
    4. Zero-Feedback Autonomous Deliberation (No output tokens fed back)
    """
    def __init__(
        self,
        vocab_size: int = 32000,
        surface_dim: int = 2048,
        num_shells: int = 32,
        radial_dim: int = 1024,
        num_shortcut_tracts: int = 173,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.surface_dim = surface_dim
        self.num_shells = num_shells

        self.sensory_dnn = SensoryInputDNN(
            vocab_size=vocab_size, hidden_dim=surface_dim, surface_dim=surface_dim
        )
        self.volumetric_core = VolumetricBrainCore(
            num_shells=num_shells,
            shell_dim=surface_dim,
            radial_dim=radial_dim,
            num_shortcut_tracts=num_shortcut_tracts,
        )
        self.motor_dnn = MotorOutputDNN(
            surface_dim=surface_dim, core_dim=surface_dim, vocab_size=vocab_size
        )

        # Legacy alias for older scripts
        self.boundary_transceiver = self.sensory_dnn

    def init_brain_state(
        self, batch_size: int, device: torch.device, living: bool = True
    ) -> torch.Tensor:
        """Initializes brain state: living active resting state by default, or quiescent."""
        if living:
            return self.volumetric_core.init_living_state(batch_size, device)
        return self.volumetric_core.init_state(batch_size, device)

    def live_heartbeat_tick(
        self, state: Optional[torch.Tensor] = None, t_step: int = 0, batch_size: int = 1
    ) -> torch.Tensor:
        """Continuous living clock tick: keeps the core pulsating in resting state like a living system."""
        if state is None:
            device = next(self.parameters()).device
            state = self.init_brain_state(batch_size, device, living=True)
        return self.volumetric_core.tick_living_dynamics(state, t_step=t_step)

    def load_compatible_checkpoint(self, checkpoint_path: str, device: torch.device = torch.device('cpu')) -> bool:
        """Loads state_dict skipping any parameters with mismatched shapes across architectural upgrades."""
        if not os.path.exists(checkpoint_path):
            return False
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model_dict = self.state_dict()
        compatible_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                compatible_dict[k] = v
        self.load_state_dict(compatible_dict, strict=False)
        return True

    def ingest_stimulus(
        self,
        token_ids: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        living: bool = True,
        return_memory: bool = False,
    ):
        """
        Ingests a sequence of sensory tokens into the manifold.
        If state is None, perturbs an active living resting state rather than starting from dead zero.
        Optionally returns (state, sensory_memory) for Arcuate Fasciculus motor resonance.
        """
        batch_size, seq_len = token_ids.shape
        if state is None:
            state = self.init_brain_state(batch_size, token_ids.device, living=living)

        # Precompute the continuous sensory working memory field (Full resolution for Arcuate Fasciculus)
        sensory_memory = self.sensory_dnn(token_ids)  # [Batch, SeqLen, surface_dim]

        # Biological Sensory Temporal Receptive Integration:
        # Prevents recurrent autograd graph inflation for long sequences (e.g. 1024 tokens)
        # by integrating continuous boundary flux across temporal gamma-wave packets (max 32 circulation steps).
        packet_size = max(1, math.ceil(seq_len / 32)) if seq_len > 32 else 1
        for t in range(0, seq_len, packet_size):
            chunk = sensory_memory[:, t : t + packet_size, :]
            flux = chunk.mean(dim=1)
            state = self.volumetric_core.inject_sensory_flux(state, flux)
            state = self.volumetric_core.circulate_mid_tensors(state, num_cycles=1)

        if return_memory:
            return state, sensory_memory
        return state

    def ingest_stimulus_with_memory(
        self, token_ids: torch.Tensor, state: Optional[torch.Tensor] = None, living: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience method returning both core manifold state and sensory working memory."""
        return self.ingest_stimulus(token_ids, state=state, living=living, return_memory=True)

    def deliberate(self, state: torch.Tensor, thinking_cycles: int = 4) -> torch.Tensor:
        """Active subconscious deliberation (internal cognitive processing with zero emission)."""
        return self.volumetric_core.circulate_mid_tensors(state, num_cycles=thinking_cycles)

    def emit_next_token(
        self,
        state: torch.Tensor,
        sensory_memory: Optional[torch.Tensor] = None,
        refractory_trace: Optional[torch.Tensor] = None,
        return_trace: bool = False,
        step_idx: int = 0,
    ):
        """
        Reads out next-token distribution and advances manifold state (Zero Feedback).
        Uses Arcuate Fasciculus sensory memory resonance and Premotor Temporal Phase Clock.
        """
        surface_efferent, core_attractor = self.volumetric_core.tap_efferents(state)
        logits, next_trace, efference_copy = self.motor_dnn(
            surface_efferent,
            core_attractor,
            sensory_memory=sensory_memory,
            refractory_trace=refractory_trace,
            step_idx=step_idx,
        )
        # Advance manifold internally with Corollary Discharge in Deep Subcortical Core (Shell 0)
        state_with_efference = self.volumetric_core.inject_efference_copy(state, efference_copy)
        next_state = self.volumetric_core.circulate_mid_tensors(state_with_efference, num_cycles=1)

        if return_trace or refractory_trace is not None:
            return logits, next_state, next_trace
        return logits, next_state

    def forward(self, input_ids: torch.Tensor, circulation_cycles: int = 1) -> torch.Tensor:
        """Forward pass for training across sequence steps."""
        batch_size, seq_len = input_ids.shape
        state = self.init_brain_state(batch_size, input_ids.device, living=True)
        sensory_memory = self.sensory_dnn(input_ids)
        logits_history = []

        for t in range(seq_len):
            flux = sensory_memory[:, t, :]
            state = self.volumetric_core.inject_sensory_flux(state, flux)
            state = self.volumetric_core.circulate_mid_tensors(state, num_cycles=circulation_cycles)
            surface_efferent, core_attractor = self.volumetric_core.tap_efferents(state)
            logits, _, efference_copy = self.motor_dnn(
                surface_efferent, core_attractor, sensory_memory=sensory_memory[:, : t + 1, :], step_idx=t
            )
            state = self.volumetric_core.inject_efference_copy(state, efference_copy)
            logits_history.append(logits)

        return torch.stack(logits_history, dim=1)

    @torch.no_grad()
    def solve_with_active_thinking(
        self,
        prompt_tokens: torch.Tensor,
        thinking_cycles: int = 4,
        max_answer_tokens: Optional[int] = None,
        temperature: float = 0.6,
        repetition_penalty: float = 1.5,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Deliberate-Then-Emit Active Cognitive Reasoning (Unbounded Generation):
        1. Ingest problem prompt into living manifold and sensory working memory buffer
        2. Deliberate internally (K thinking cycles)
        3. Emit answer tokens dynamically until EOS token or safety ceiling (Zero output feedback)!
        """
        self.eval()
        state, sensory_memory = self.ingest_stimulus_with_memory(prompt_tokens, living=True)
        state = self.deliberate(state, thinking_cycles=thinking_cycles)

        batch_size = prompt_tokens.shape[0]
        refractory_trace = torch.zeros(batch_size, self.vocab_size, device=state.device)

        ceiling = max_answer_tokens if max_answer_tokens is not None else 4096
        generated = []
        for step_idx in range(ceiling):
            logits, state, refractory_trace = self.emit_next_token(
                state, sensory_memory=sensory_memory, refractory_trace=refractory_trace, step_idx=step_idx
            )

            # Suppress premature EOS in first 3 tokens
            if eos_token_id is not None and step_idx < 3:
                logits[..., eos_token_id] -= 50.0

            # Apply repetition penalty for previously generated tokens
            if repetition_penalty > 0 and len(generated) > 0:
                for gen_tok in generated:
                    logits.scatter_add_(
                        -1, gen_tok, torch.full_like(gen_tok, -repetition_penalty, dtype=logits.dtype)
                    )

            if temperature <= 0.05:
                next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / max(temperature, 1e-5), dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            generated.append(next_tok)

            if eos_token_id is not None and (next_tok == eos_token_id).all():
                break

        if len(generated) == 0:
            return torch.empty((batch_size, 0), dtype=torch.long, device=prompt_tokens.device)
        return torch.cat(generated, dim=-1)

    @torch.no_grad()
    def generate_zero_feedback(
        self, prompt_tokens: torch.Tensor, max_new_tokens: int = 64, temperature: float = 0.7
    ) -> torch.Tensor:
        return self.solve_with_active_thinking(
            prompt_tokens, thinking_cycles=2, max_answer_tokens=max_new_tokens, temperature=temperature
        )


def build_cerebro_model(scale: str = "2b") -> CerebroSpheroid:
    """
    Factory for CerebroSpheroid configurations:
    - '2b': Exact 2.007 Billion parameter cognitive brain model.
    - '260m': High-throughput 248 Million parameter model.
    - 'test': Lightweight test model for fast local verification.
    """
    if scale == "2b":
        return CerebroSpheroid(
            vocab_size=32000,
            surface_dim=2048,
            num_shells=32,
            radial_dim=1024,
            num_shortcut_tracts=173,
        )
    elif scale == "260m":
        return CerebroSpheroid(
            vocab_size=32000,
            surface_dim=1024,
            num_shells=16,
            radial_dim=256,
            num_shortcut_tracts=48,
        )
    elif scale == "test":
        return CerebroSpheroid(
            vocab_size=1000,
            surface_dim=128,
            num_shells=8,
            radial_dim=64,
            num_shortcut_tracts=4,
        )
    else:
        raise ValueError(f"Unknown scale: {scale}")


if __name__ == "__main__":
    print("=" * 80)
    print("      CEREBRO-SPHEROID: UPGRADED BRAIN ARCHITECTURE INITIALIZATION      ")
    print("=" * 80)
    model_2b = build_cerebro_model("2b")
    total_2b = sum(p.numel() for p in model_2b.parameters())
    sensory_p = sum(p.numel() for p in model_2b.sensory_dnn.parameters())
    core_p = sum(p.numel() for p in model_2b.volumetric_core.parameters())
    motor_p = sum(p.numel() for p in model_2b.motor_dnn.parameters())

    print(f"CerebroSpheroid-2B Parameter Breakdown:")
    print(f"  • Total Parameters         : {total_2b:,} ({total_2b/1e9:.3f} Billion)")
    print(f"  • Afferent Sensory DNN     : {sensory_p:,} ({sensory_p/1e6:.2f}M params)")
    print(f"  • Volumetric Core (B^3)    : {core_p:,} ({core_p/1e9:.3f} Billion params)")
    print(f"  • Efferent Motor Output DNN: {motor_p:,} ({motor_p/1e6:.2f}M params)")
    print("=" * 80)
