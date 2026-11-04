"""
CerebroSpheroid V4: Scaled, Measured Reasoning Manifold.

100% Parameter and Computational Parity with Pure JAX v4_model.py:
1. Afferent Sensory Front-End:
   - 2-Layer SwiGLU projection without positional contamination (SensoryInputDNNV4).
   - Pre-computed prompt and feedback sensory projections (Arcuate Fasciculus K/V).
2. Volumetric Recurrent Brain Core (B^3):
   - Small-world geodesic shortcut tracts with precomputed axonal index routing.
   - Radial re-entrant wave coupling between concentric spherical shells.
   - RMSNorm and softplus leaky integration (ResidualVolumetricCoreV4).
3. Efferent Motor Output Decoder (Broca's Area):
   - SDPA accelerated cross-resonance attention with sensory prompt.
   - Multi-shell softmax pooling and efference copy feedback.
   - Premotor recurrent plan buffer with contractive decay.
   - Built-in Agentic Persona steering basis (0=Reasoner, 1=Explorer, 2=Concise, 3=Verifier).
   - Two-tier refractory adaptation & repetition suppression (PremotorMotorDecoderV4).
4. Full Checkpoint Interoperability:
   - Dual-key compatibility: seamless loading/saving between TPU JAX and GPU PyTorch.
"""

import math
import os
import random
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_module

from .sensory_nodes import AudioSensoryEncoder, ThalamicSensoryGater, VisionSensoryEncoder


def get_shell_zones(num_shells: int) -> Dict[str, range]:
    """Derives zone bounds dynamically from num_shells.

    Thalamic (deep): Shells 0 to K//3
    Associative (intermediate): Shells K//3 to K - max(1, K//4)
    Superficial (cortical): Shells K - max(1, K//4) to K
    """
    thalamic_end = num_shells // 3
    associative_end = num_shells - max(1, num_shells // 4)
    return {
        "thalamic": range(0, thalamic_end),
        "associative": range(thalamic_end, associative_end),
        "superficial": range(associative_end, num_shells),
    }


class SensoryInputDNNV4(nn.Module):
    """
    Afferent Sensory Front-End:
    2-Layer SwiGLU ResNet with RMSNorm converting discrete token symbols
    into smooth continuous boundary flux fields on S^2.
    Bitwise matches pure JAX v4_model.py encode_sensory.
    """
    def __init__(self, vocab_size: int = 32000, surface_dim: int = 640):
        super().__init__()
        self.vocab_size = vocab_size
        self.surface_dim = surface_dim
        self.embedding = nn.Embedding(vocab_size, surface_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        self.fc1_gate = nn.Linear(surface_dim, surface_dim * 2, bias=False)
        self.fc1_val = nn.Linear(surface_dim, surface_dim * 2, bias=False)
        self.norm1 = nn.RMSNorm(surface_dim * 2)

        self.fc2 = nn.Linear(surface_dim * 2, surface_dim, bias=False)
        self.norm_out = nn.RMSNorm(surface_dim)

    def forward(self, token_ids_or_flux: torch.Tensor) -> torch.Tensor:
        if token_ids_or_flux.dtype in (torch.int32, torch.int64, torch.long):
            x = self.embedding(token_ids_or_flux)
        else:
            x = token_ids_or_flux

        gate = F.silu(self.fc1_gate(x))
        val = self.fc1_val(x)
        h = self.norm1(gate * val)
        flux = self.fc2(h) + x
        return self.norm_out(flux)


class ResidualVolumetricCoreV4(nn.Module):
    """
    Systolic Volumetric Core V4 matching pure JAX v4_model.py bitwise.
    Contains exactly 9 core parameters and small-world geodesic shortcut buffers.
    """
    def __init__(
        self,
        num_shells: int = 16,
        shell_dim: int = 640,
        radial_dim: int = 256,
        num_shortcut_tracts: int = 40,
    ):
        super().__init__()
        self.num_shells = num_shells
        self.shell_dim = shell_dim
        self.radial_dim = radial_dim
        self.num_shortcut_tracts = num_shortcut_tracts

        src_indices = torch.arange(num_shortcut_tracts) % num_shells
        dst_indices = num_shells - 1 - (torch.arange(num_shortcut_tracts) * 3) % num_shells
        self.register_buffer("src_indices", src_indices)
        self.register_buffer("dst_indices", dst_indices)

        self.intra_shell = nn.Parameter(torch.empty(num_shells, shell_dim, shell_dim).normal_(mean=0.0, std=0.015))
        self.reentrant_field = nn.Parameter(torch.empty(num_shells, num_shells, radial_dim, radial_dim).normal_(mean=0.0, std=0.008))
        self.shortcut_tracts = nn.Parameter(torch.empty(num_shortcut_tracts, shell_dim, shell_dim).normal_(mean=0.0, std=0.01))

        self.radial_proj_in = nn.Linear(shell_dim, radial_dim, bias=False)
        self.radial_proj_out = nn.Linear(radial_dim, shell_dim, bias=False)

        self.flow_norm = nn.RMSNorm(shell_dim)
        self.norm = nn.RMSNorm(shell_dim)

        self.log_write = nn.Parameter(torch.full((num_shells,), math.log(math.expm1(0.35)), dtype=torch.float32))
        self.log_leak = nn.Parameter(torch.full((num_shells,), math.log(0.35 / 0.65), dtype=torch.float32))

    def deliberate_step(self, state: torch.Tensor) -> torch.Tensor:
        curr = state.to(self.intra_shell.dtype)
        # 1. Intra-shell recurrent propagation
        intra = torch.einsum("bkd,kdo->bko", curr, self.intra_shell)

        # 2. Axonal shortcut tracts
        src_state = curr[:, self.src_indices, :]
        tracts = torch.einsum("bti,tij->btj", src_state, self.shortcut_tracts)
        tract_acc = torch.zeros_like(curr)
        tract_acc.index_add_(1, self.dst_indices, tracts)

        # 3. Radial re-entrant wave coupling
        rad_in = self.radial_proj_in(curr)
        reentrant = torch.einsum("bki,kjio->bjo", rad_in, self.reentrant_field)
        reentrant = self.radial_proj_out(F.silu(reentrant))

        # Flow norm and leaky integration
        flow = self.flow_norm(intra + tract_acc + reentrant)
        leak = torch.sigmoid(self.log_leak).view(1, -1, 1)
        write = F.softplus(self.log_write).view(1, -1, 1)

        raw = (1.0 - leak) * curr + write * torch.tanh(flow)
        return self.norm(raw).to(state.dtype)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.num_shells, self.shell_dim, device=device)

    def inject_efference_copy(self, state: torch.Tensor, efference_flux: torch.Tensor) -> torch.Tensor:
        new_state = state.clone()
        new_state[:, 0, :] = new_state[:, 0, :] + efference_flux.to(state.dtype)
        return new_state

    def circulate_mid_tensors(self, state: torch.Tensor, num_cycles: int = 1) -> torch.Tensor:
        curr = state
        for _ in range(num_cycles):
            curr = self.deliberate_step(curr)
        return curr


class PremotorMotorDecoderV4(nn.Module):
    """
    V4 Premotor Motor Decoder:
    - Retains internal plan buffer, corollary discharge, and multi-shell softmax tap.
    - Upgrades Arcuate Fasciculus cross-resonance to F.scaled_dot_product_attention (SDPA).
    - Preserves exact 1:1 parameter naming and shapes with pure JAX v4_model.py.
    """
    def __init__(
        self,
        num_shells: int = 16,
        surface_dim: int = 640,
        vocab_size: int = 32000,
        max_phase_steps: int = 1024,
        dropout: float = 0.0,
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

        self.phase_embed = nn.Embedding(max_phase_steps, surface_dim)
        nn.init.normal_(self.phase_embed.weight, mean=0.0, std=0.02)

        self.q_proj = nn.Linear(surface_dim, surface_dim, bias=False)
        self.k_proj = nn.Linear(surface_dim, surface_dim, bias=False)
        self.v_proj = nn.Linear(surface_dim, surface_dim, bias=False)
        self.resonance_norm = nn.RMSNorm(surface_dim)

        self.shell_proj = nn.Linear(surface_dim, surface_dim, bias=False)
        self.shell_score = nn.Parameter(torch.zeros(num_shells))
        self.shell_scale = nn.Parameter(torch.zeros(1))

        self.plan_gate = nn.Linear(surface_dim * 3, surface_dim, bias=True)
        self.plan_update = nn.Linear(surface_dim * 3, surface_dim, bias=False)
        self.plan_write = nn.Parameter(torch.zeros(1))
        self.plan_decay = nn.Parameter(torch.tensor(0.90))

        fused_dim = surface_dim * 5
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

        self.efference_proj = nn.Linear(hidden_dim, surface_dim, bias=False)
        nn.init.normal_(self.efference_proj.weight, mean=0.0, std=0.01)

        self.persona_embeddings = nn.Parameter(torch.empty(4, surface_dim).normal_(mean=0.0, std=0.05))
        self.persona_scale = nn.Parameter(torch.tensor(0.5))
        self.plan_init = nn.Linear(surface_dim * 2, surface_dim, bias=True)

    @property
    def motor_phase_embed(self):
        return self.phase_embed

    def init_plan(
        self,
        delib_summary: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        persona_id: Optional[int] = None,
    ) -> torch.Tensor:
        if delib_summary is not None:
            raw = self.plan_init(delib_summary)
            plan = torch.tanh(raw)
            if persona_id is not None:
                plan = plan + self.persona_scale * self.persona_embeddings[persona_id]
            return plan
        return torch.zeros(batch_size, self.surface_dim, device=device)

    def prime_sensory(self, sensory_memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k_proj(sensory_memory), self.v_proj(sensory_memory)

    def forward(
        self,
        state: torch.Tensor,
        sensory_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        plan: Optional[torch.Tensor] = None,
        refractory_trace: Optional[torch.Tensor] = None,
        step_idx: int = 0,
        auditory_feedback: Optional[torch.Tensor] = None,
        repetition_penalty: float = 1.25,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        B, K, D = state.shape
        surface = state[:, -1, :]
        core = state[:, 0, :]
        if plan is None:
            plan = self.init_plan(batch_size=B, device=state.device)

        max_phase = self.phase_embed.weight.shape[0]
        phase_idx = min(max(0, int(step_idx)), max_phase - 1)
        phase = self.phase_embed.weight[phase_idx].to(state.dtype)
        motor_surface = surface + phase

        if auditory_feedback is not None:
            motor_surface = motor_surface + auditory_feedback.to(motor_surface.dtype)

        # Arcuate Fasciculus cross-resonance via Scaled Dot-Product Attention (SDPA)
        if sensory_kv is not None:
            keys, values = sensory_kv
            q = self.q_proj(motor_surface).unsqueeze(1)
            res = F.scaled_dot_product_attention(q.to(keys.dtype), keys, values)
            resonance = self.resonance_norm(res.squeeze(1).to(motor_surface.dtype))
        else:
            resonance = torch.zeros_like(motor_surface)

        # Multi-shell softmax tap
        if self.use_shell_tap:
            shell_weights = F.softmax(self.shell_score, dim=0).view(1, -1, 1)
            shell_proj = self.shell_proj(state)
            shell_pool = torch.sum(shell_weights * shell_proj, dim=1) * self.shell_scale
        else:
            shell_pool = torch.zeros_like(motor_surface)

        fused = self.fused_norm(
            torch.cat([motor_surface, core, resonance, shell_pool, plan], dim=-1)
        )
        h = self.norm_mid(F.silu(self.fc_gate(fused)) * self.fc_val(fused))
        h = self.dropout(h)

        # Premotor recurrent plan update
        if self.use_plan:
            phase_batch = phase.unsqueeze(0).expand(B, -1)
            plan_input = torch.cat([h, phase_batch], dim=-1)
            p_gate = torch.sigmoid(self.plan_gate(plan_input))
            p_decay = torch.clamp(self.plan_decay, 0.0, 0.92)
            p_update = self.plan_update(plan_input)
            next_plan = torch.clamp(p_decay * plan + self.plan_write * p_gate * p_update, -50.0, 50.0)
        else:
            next_plan = plan

        temp = torch.clamp(self.temperature_scale, min=0.1, max=5.0)
        raw_logits = self.head(h) / temp
        efference_copy = self.efference_proj(h)

        # Two-tier refractory adaptation & repetition suppression
        if refractory_trace is not None:
            alpha = torch.clamp(self.refractory_strength, 0.1, 5.0)
            tier1_penalty = 3.0 * alpha * refractory_trace
            active_mask = (refractory_trace > 0.02).to(raw_logits.dtype)
            rep_scale = torch.where(raw_logits > 0, 1.0 / repetition_penalty, repetition_penalty)
            scaled_logits = raw_logits * (1.0 - active_mask) + (raw_logits * rep_scale) * active_mask
            adapted_logits = scaled_logits - tier1_penalty

            decay = torch.sigmoid(self.refractory_decay)
            with torch.no_grad():
                current_prob = F.softmax(raw_logits, dim=-1)
                top1_idx = torch.argmax(adapted_logits, dim=-1)
                top1_pulse = F.one_hot(top1_idx, num_classes=raw_logits.shape[-1]).to(raw_logits.dtype)
                next_trace = decay * refractory_trace + (1.0 - decay) * current_prob + 1.0 * top1_pulse
        else:
            adapted_logits, next_trace = raw_logits, None

        return adapted_logits, efference_copy, next_trace, next_plan


class CerebroSpheroidV4(nn.Module):
    """
    CerebroSpheroid V4: Continuous Volumetric Manifold.
    Pure systolic architecture bitwise aligned with pure JAX v4_model.py.
    """
    def __init__(
        self,
        vocab_size: int = 32000,
        surface_dim: int = 640,
        num_shells: int = 16,
        radial_dim: int = 256,
        num_shortcut_tracts: int = 40,
        max_len: int = 2048,
        dropout: float = 0.0,
        max_phase_steps: int = 1024,
        use_temporal: bool = False,
        use_residual_core: bool = True,
        use_plan: bool = True,
        use_shell_tap: bool = True,
        enable_multimodal: bool = False,
        use_causal_ingestion: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.surface_dim = surface_dim
        self.num_shells = num_shells
        self.radial_dim = radial_dim
        self.num_shortcut_tracts = num_shortcut_tracts
        self.max_len = max_len
        self.max_phase_steps = max_phase_steps

        # Front-end, Core, and Decoder matching pure JAX V4 systolic architecture
        self.sensory_dnn = SensoryInputDNNV4(vocab_size=vocab_size, surface_dim=surface_dim)
        self.volumetric_core = ResidualVolumetricCoreV4(
            num_shells=num_shells,
            shell_dim=surface_dim,
            radial_dim=radial_dim,
            num_shortcut_tracts=num_shortcut_tracts,
        )
        self.motor_dnn = PremotorMotorDecoderV4(
            num_shells=num_shells,
            surface_dim=surface_dim,
            vocab_size=vocab_size,
            max_phase_steps=max_phase_steps,
            dropout=dropout,
            use_plan=use_plan,
            use_shell_tap=use_shell_tap,
        )

    @property
    def sensory(self):
        return self.sensory_dnn

    @property
    def core(self):
        return self.volumetric_core

    @property
    def motor(self):
        return self.motor_dnn

    @property
    def boundary_transceiver(self):
        return self.sensory_dnn

    @property
    def shell_zones(self) -> Dict[str, range]:
        return get_shell_zones(self.num_shells)

    def init_brain_state(self, batch_size: int, device: torch.device, living: bool = False) -> torch.Tensor:
        return self.volumetric_core.init_state(batch_size, device)

    def init_living_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.init_brain_state(batch_size, device, living=True)

    def encode_sensory(
        self, token_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        sensory_mem = self.sensory_dnn(token_ids)
        sensory_kv = self.motor_dnn.prime_sensory(sensory_mem)
        return sensory_mem, sensory_kv

    def deliberate(self, state: torch.Tensor, thinking_cycles: int = 3) -> torch.Tensor:
        return self.volumetric_core.circulate_mid_tensors(state, num_cycles=thinking_cycles)

    def emit_step(
        self,
        state: torch.Tensor,
        sensory_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        plan: Optional[torch.Tensor] = None,
        refractory_trace: Optional[torch.Tensor] = None,
        step_idx: int = 0,
        auditory_feedback: Optional[torch.Tensor] = None,
        repetition_penalty: float = 1.25,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        logits, efference, next_trace, next_plan = self.motor_dnn(
            state=state,
            sensory_kv=sensory_kv,
            plan=plan,
            refractory_trace=refractory_trace,
            step_idx=step_idx,
            auditory_feedback=auditory_feedback,
            repetition_penalty=repetition_penalty,
        )
        state_eff = self.volumetric_core.inject_efference_copy(state, efference)
        next_state = self.volumetric_core.deliberate_step(state_eff)
        return logits, next_state, next_trace, next_plan

    def compile_deliberate(self):
        try:
            self.volumetric_core.deliberate_step = torch.compile(self.volumetric_core.deliberate_step)
            print("[V4] volumetric_core.deliberate_step compiled with torch.compile")
        except Exception as e:
            print(f"[V4] torch.compile unavailable or failed: {e}")

    def training_logits(
        self,
        prompt_tokens: torch.Tensor,
        response_steps: int,
        thinking_cycles: int = 3,
        pad_token_id: Optional[int] = 0,
        targets: Optional[torch.Tensor] = None,
        feedback_tokens: Optional[torch.Tensor] = None,
        deliberation_mode: str = "scheduled",
        thalamic_anchor_weight: float = 0.1,
        sampling_p: float = 1.0,
        persona_id: Optional[int] = None,
        repetition_penalty: float = 1.25,
        **kwargs,
    ) -> torch.Tensor:
        """
        Systolic V4 differentiable rollout engine bitwise aligned with pure JAX training_rollout.
        """
        B, L = prompt_tokens.shape
        sensory_mem, sensory_kv = self.encode_sensory(prompt_tokens)

        # 1. Masked prompt centroid pooling (zero padding contamination)
        if pad_token_id is not None:
            mask = (prompt_tokens != pad_token_id).unsqueeze(-1).to(sensory_mem.dtype)
        else:
            mask = torch.ones(B, L, 1, device=prompt_tokens.device, dtype=sensory_mem.dtype)
        prompt_flux = (sensory_mem * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        # 2. Subconscious deliberation cycles before verbalization
        state = self.volumetric_core.init_state(B, prompt_tokens.device)
        state[:, -1, :] = prompt_flux
        state = self.deliberate(state, thinking_cycles=thinking_cycles)

        # 3. Thalamic memory basin anchoring (shells 0 .. K//3)
        thalamic_end = max(1, self.num_shells // 3)
        thalamic_anchor = state[:, :thalamic_end, :].clone()

        # 4. Pre-verbal executive plan synthesis
        delib_summary = torch.cat([state[:, 0, :], prompt_flux], dim=-1)
        plan = self.motor_dnn.init_plan(delib_summary, persona_id=persona_id)

        # 5. Pre-embed teacher feedback tokens if available
        feed = feedback_tokens if feedback_tokens is not None else (
            targets if targets is not None else None
        )
        if feed is not None:
            safe_feed = feed.clamp(min=0)
            valid_mask = (feed >= 0).unsqueeze(-1).to(sensory_mem.dtype)
            tf_sensory = self.sensory_dnn(safe_feed) * valid_mask
        else:
            tf_sensory = torch.zeros(B, response_steps, self.surface_dim, device=prompt_tokens.device, dtype=sensory_mem.dtype)

        refractory_trace = torch.zeros(B, self.vocab_size, device=prompt_tokens.device, dtype=torch.float32)
        prev_thought = torch.zeros(B, self.surface_dim, device=prompt_tokens.device, dtype=sensory_mem.dtype)

        all_logits = []
        for step_idx in range(response_steps):
            tf_aud = tf_sensory[:, step_idx, :] if step_idx < tf_sensory.shape[1] else torch.zeros_like(prev_thought)
            if deliberation_mode == "teacher_forcing":
                aud_feed = tf_aud
            elif deliberation_mode == "internal":
                aud_feed = torch.zeros_like(prev_thought) if step_idx == 0 else prev_thought
            else:  # "scheduled"
                internal_feed = tf_aud if step_idx == 0 else prev_thought
                aud_feed = sampling_p * tf_aud + (1.0 - sampling_p) * internal_feed

            logits, efference, refractory_trace, plan = self.motor_dnn(
                state=state,
                sensory_kv=sensory_kv,
                plan=plan,
                refractory_trace=refractory_trace,
                step_idx=step_idx,
                auditory_feedback=aud_feed,
                repetition_penalty=repetition_penalty,
            )
            all_logits.append(logits)

            # Efference copy injected back into subcortical core + deliberate step
            state_eff = self.volumetric_core.inject_efference_copy(state, efference)
            state = self.volumetric_core.deliberate_step(state_eff)

            # Deep thalamic basin retention
            if thalamic_anchor_weight > 0.0:
                state[:, :thalamic_end, :] = (
                    (1.0 - thalamic_anchor_weight) * state[:, :thalamic_end, :]
                    + thalamic_anchor_weight * thalamic_anchor
                )

            # Thought re-afference for step t+1 (isolated with detached gradient)
            if deliberation_mode != "teacher_forcing":
                with torch.no_grad():
                    soft_p = F.softmax(logits, dim=-1)
                    soft_emb = soft_p @ self.sensory_dnn.embedding.weight
                    prev_thought = self.sensory_dnn(soft_emb)

        return torch.stack(all_logits, dim=1)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        tokens: Optional[torch.Tensor] = None,
        response_steps: Optional[int] = None,
        thinking_cycles: int = 3,
        pad_token_id: Optional[int] = 0,
        targets: Optional[torch.Tensor] = None,
        feedback_tokens: Optional[torch.Tensor] = None,
        deliberation_mode: str = "scheduled",
        thalamic_anchor_weight: float = 0.1,
        sampling_p: float = 1.0,
        persona_id: Optional[int] = None,
        repetition_penalty: float = 1.25,
        **kwargs,
    ) -> torch.Tensor:
        toks = input_ids if input_ids is not None else tokens
        if toks is None:
            raise ValueError("CerebroSpheroidV4.forward requires input_ids or tokens.")
        steps = response_steps if response_steps is not None else toks.shape[1]
        return self.training_logits(
            toks,
            response_steps=steps,
            thinking_cycles=thinking_cycles,
            pad_token_id=pad_token_id,
            targets=targets,
            feedback_tokens=feedback_tokens,
            deliberation_mode=deliberation_mode,
            thalamic_anchor_weight=thalamic_anchor_weight,
            sampling_p=sampling_p,
            persona_id=persona_id,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        thinking_cycles: int = 3,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        repetition_penalty: float = 1.25,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = 0,
        persona_id: Optional[int] = 0,
        thalamic_anchor_weight: float = 0.1,
        **kwargs,
    ) -> torch.Tensor:
        """Autoregressive generative rollout matching JAX generate_v4."""
        self.eval()
        B, L = prompt_tokens.shape
        sensory_mem, sensory_kv = self.encode_sensory(prompt_tokens)

        if pad_token_id is not None:
            mask = (prompt_tokens != pad_token_id).unsqueeze(-1).to(sensory_mem.dtype)
        else:
            mask = torch.ones(B, L, 1, device=prompt_tokens.device, dtype=sensory_mem.dtype)
        prompt_flux = (sensory_mem * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        state = self.volumetric_core.init_state(B, prompt_tokens.device)
        state[:, -1, :] = prompt_flux
        state = self.deliberate(state, thinking_cycles=thinking_cycles)

        thalamic_end = max(1, self.num_shells // 3)
        thalamic_anchor = state[:, :thalamic_end, :].clone()

        delib_summary = torch.cat([state[:, 0, :], prompt_flux], dim=-1)
        plan = self.motor_dnn.init_plan(delib_summary, persona_id=persona_id)

        refractory_trace = torch.zeros(B, self.vocab_size, device=prompt_tokens.device, dtype=torch.float32)
        prev_thought = torch.zeros(B, self.surface_dim, device=prompt_tokens.device, dtype=sensory_mem.dtype)

        emitted: List[torch.Tensor] = []
        finished = torch.zeros(B, dtype=torch.bool, device=prompt_tokens.device)

        for step_idx in range(max_new_tokens):
            aud_feed = torch.zeros_like(prev_thought) if step_idx == 0 else prev_thought
            logits, efference, refractory_trace, plan = self.motor_dnn(
                state=state,
                sensory_kv=sensory_kv,
                plan=plan,
                refractory_trace=refractory_trace,
                step_idx=step_idx,
                auditory_feedback=aud_feed,
                repetition_penalty=repetition_penalty,
            )

            if temperature <= 0.05:
                next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / max(0.1, temperature), dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)

            emitted.append(next_tok)

            if eos_token_id is not None:
                finished |= (next_tok.squeeze(-1) == eos_token_id)
                if finished.all():
                    break

            # Thought re-afference
            soft_p = F.softmax(logits, dim=-1)
            soft_emb = soft_p @ self.sensory_dnn.embedding.weight
            prev_thought = self.sensory_dnn(soft_emb)

            # State propagation
            state_eff = self.volumetric_core.inject_efference_copy(state, efference)
            state = self.volumetric_core.deliberate_step(state_eff)

            if thalamic_anchor_weight > 0.0:
                state[:, :thalamic_end, :] = (
                    (1.0 - thalamic_anchor_weight) * state[:, :thalamic_end, :]
                    + thalamic_anchor_weight * thalamic_anchor
                )

        return torch.cat(emitted, dim=-1) if emitted else prompt_tokens.new_zeros((B, 0))

    def state_dict_compatible(self, include_jax_keys: bool = True) -> Dict[str, torch.Tensor]:
        """Returns dual-keyed state dictionary containing both PyTorch and JAX key conventions."""
        sd = self.state_dict()
        res = dict(sd)
        if include_jax_keys:
            for k, v in sd.items():
                jax_k = (
                    k.replace("sensory_dnn.", "sensory.")
                    .replace("volumetric_core.", "core.")
                    .replace("motor_dnn.", "motor.")
                )
                if "motor.phase_embed." in jax_k or "motor.motor_phase_embed." in jax_k:
                    jax_k = jax_k.replace("motor.motor_phase_embed.", "motor.phase_embed.")
                res[jax_k] = v
            res["core.src_indices"] = self.volumetric_core.src_indices
            res["core.dst_indices"] = self.volumetric_core.dst_indices
        return res

    def load_compatible_checkpoint(self, checkpoint_path: str, device: torch.device = torch.device("cpu")) -> bool:
        """Loads V4 checkpoints from TPU JAX or GPU PyTorch with 100% parameter parity."""
        if not os.path.exists(checkpoint_path):
            return False
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        weights = state.get("model", state.get("model_state_dict", state)) if isinstance(state, dict) else state
        if not isinstance(weights, dict):
            return False

        remapped = {}
        for k, v in weights.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("sensory."):
                k = "sensory_dnn." + k[len("sensory."):]
            elif k.startswith("core."):
                k = "volumetric_core." + k[len("core."):]
            elif k.startswith("motor."):
                k = "motor_dnn." + k[len("motor."):]

            if "motor_dnn.motor_phase_embed." in k:
                k = k.replace("motor_dnn.motor_phase_embed.", "motor_dnn.phase_embed.")

            remapped[k] = v

        my_state = self.state_dict()
        matched = {}
        for k, v in remapped.items():
            if k in my_state:
                target_shape = my_state[k].shape
                if v.numel() == my_state[k].numel() and v.shape != target_shape:
                    v = v.view(target_shape)
                if v.shape == target_shape:
                    matched[k] = v.to(device=device, dtype=my_state[k].dtype)
                else:
                    print(f"⚠️ [checkpoint] Shape mismatch for {k}: {v.shape} vs {target_shape}")

        missing = set(my_state.keys()) - set(matched.keys())
        if missing:
            print(f"⚠️ [checkpoint] Note: Initializing {len(missing)} missing parameter(s) from reference defaults: {list(missing)[:5]}")
            if len(missing) > 15:
                lift_checkpoint_v3_to_v4(weights, self)
                return True

        self.load_state_dict(matched, strict=False)
        print(f"✅ [checkpoint] Successfully loaded {len(matched)}/{len(my_state)} parameters from {os.path.basename(checkpoint_path)}")
        return True

    def save_compatible_checkpoint(
        self,
        checkpoint_path: str,
        step: int = 0,
        best: float = float("inf"),
        loss: float = 0.0,
    ) -> None:
        """Saves checkpoint in dual-key format for seamless cross-framework resume."""
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        payload = {
            "model": self.state_dict_compatible(include_jax_keys=True),
            "step": step,
            "best": best,
            "loss": loss,
            "arch": "v4",
        }
        tmp_path = checkpoint_path + ".tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, checkpoint_path)


# =========================================================================
# Manifold Diagnostics
# =========================================================================

@torch.no_grad()
def compute_manifold_diagnostics(
    model: CerebroSpheroidV4,
    state: torch.Tensor,
) -> Dict[str, Union[float, List[float]]]:
    """
    Computes non-invasive geometric diagnostics of the recurrent manifold state.
    """
    batch_size, num_shells, shell_dim = state.shape
    state_f = state.float()

    effective_ranks = []
    rms_per_shell = []
    for k in range(num_shells):
        s_k = state_f[:, k, :]
        rms = s_k.pow(2).mean().sqrt().item()
        rms_per_shell.append(rms)

        if batch_size > 1:
            s_centered = s_k - s_k.mean(dim=0, keepdim=True)
            try:
                s_vals = torch.linalg.svdvals(s_centered)
                s2 = s_vals.pow(2)
                rank = (s2.sum().pow(2) / (s2.pow(2).sum() + 1e-9)).item() if s2.sum().item() >= 1e-6 else 1.0
            except Exception:
                rank = 1.0
        else:
            rank = 1.0
        effective_ranks.append(rank)

    adjacent_cosines = []
    for k in range(num_shells - 1):
        s_curr = state_f[:, k, :]
        s_next = state_f[:, k + 1, :]
        cos = F.cosine_similarity(s_curr, s_next, dim=-1).mean().item()
        adjacent_cosines.append(cos)

    write_saturation = 0.0
    if hasattr(model, "volumetric_core") and hasattr(model.volumetric_core, "log_write"):
        write_vals = F.softplus(model.volumetric_core.log_write)
        write_saturation = (write_vals > 0.95).float().mean().item()

    return {
        "mean_effective_rank": float(sum(effective_ranks) / len(effective_ranks)),
        "min_effective_rank": float(min(effective_ranks)),
        "effective_ranks": effective_ranks,
        "mean_shell_rms": float(sum(rms_per_shell) / len(rms_per_shell)),
        "shell_rms": rms_per_shell,
        "mean_adjacent_cosine": float(sum(adjacent_cosines) / len(adjacent_cosines)) if adjacent_cosines else 1.0,
        "write_saturation": write_saturation,
    }


# =========================================================================
# Presets and Model Factory
# =========================================================================

PRESETS = {
    "v4_small": dict(
        vocab_size=32000,
        surface_dim=640,
        num_shells=16,
        radial_dim=256,
        num_shortcut_tracts=40,
        max_len=2048,
    ),
    "v4_wide": dict(
        vocab_size=32000,
        surface_dim=896,
        num_shells=24,
        radial_dim=384,
        num_shortcut_tracts=48,
        max_len=2048,
    ),
    "v4_probe": dict(
        vocab_size=32000,
        surface_dim=148,
        num_shells=8,
        radial_dim=74,
        num_shortcut_tracts=16,
        max_len=1024,
    ),
    "v4_gpt2_small": dict(
        vocab_size=50257,
        surface_dim=640,
        num_shells=16,
        radial_dim=256,
        num_shortcut_tracts=40,
        max_len=2048,
    ),
    "v4_gpt2_wide": dict(
        vocab_size=50257,
        surface_dim=896,
        num_shells=24,
        radial_dim=384,
        num_shortcut_tracts=48,
        max_len=2048,
    ),
    "test": dict(
        vocab_size=1000,
        surface_dim=128,
        num_shells=8,
        radial_dim=64,
        num_shortcut_tracts=4,
        max_len=2048,
    ),
}


def build_spheroid_v4(preset: str = "v4_small", **kwargs) -> CerebroSpheroidV4:
    """Builds a CerebroSpheroidV4 model from a named preset with optional overrides."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; available: {list(PRESETS.keys())}")
    cfg = dict(PRESETS[preset])
    cfg.update(kwargs)
    return CerebroSpheroidV4(**cfg)


def build_v4_model(args, vocab_size: int = 32000) -> CerebroSpheroidV4:
    """CLI-aware factory that respects explicit dimension flags."""
    preset = getattr(args, "preset", "v4_small")
    cfg = dict(PRESETS.get(preset, PRESETS["v4_small"]))
    cfg["vocab_size"] = vocab_size

    if getattr(args, "surface_dim", None):
        cfg["surface_dim"] = args.surface_dim
    if getattr(args, "num_shells", None):
        cfg["num_shells"] = args.num_shells
    if getattr(args, "radial_dim", None):
        cfg["radial_dim"] = args.radial_dim
    if getattr(args, "num_shortcut_tracts", None):
        cfg["num_shortcut_tracts"] = args.num_shortcut_tracts
    if getattr(args, "max_len", None):
        cfg["max_len"] = args.max_len

    return CerebroSpheroidV4(**cfg)


def lift_checkpoint_v3_to_v4(
    checkpoint_state: Dict[str, torch.Tensor],
    target_model: CerebroSpheroidV4,
) -> Tuple[Dict[str, torch.Tensor], List[str], List[str]]:
    """Transplants a V3 checkpoint into a V4 model."""
    own_state = target_model.state_dict()
    accepted, lifted, dropped = {}, [], []

    for k, v in checkpoint_state.items():
        if k in own_state:
            if v.shape == own_state[k].shape:
                accepted[k] = v
                lifted.append(k)
            else:
                dropped.append(k)
        else:
            dropped.append(k)

    target_model.load_state_dict(accepted, strict=False)
    return accepted, lifted, dropped
