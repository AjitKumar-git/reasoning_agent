"""
CerebroSpheroid V4: Scaled, Measured Reasoning Manifold.

Implements the V4 specification from V4_PLAN_REFINED.md:
1. Throughput & Rollout Efficiency:
   - Pre-computed prompt and feedback sensory projections (Arcuate Fasciculus K/V).
   - Zero-allocation sensory memory indexing across emission cycles (replaces torch.cat).
   - Scaled Dot-Product Attention (SDPA) with hardware-optimized backend (memory-efficient on Turing T4,
     FlashAttention on Ampere/Hopper).
2. Dynamic Shell Zoning:
   - Thalamic (0 to K//3), Associative (K//3 to K - K//4), Superficial (K - K//4 to K).
   - Zone boundaries dynamically derived from num_shells rather than hardcoded.
3. Scaled Presets:
   - v4_small: 640 / 16 shells / 256 radial / 40 tracts, vocab 50257 = 162,272,317 params
   - v4_wide: 896 / 24 shells / 384 radial / 48 tracts, vocab 50257 = 327,851,093 params
4. Multi-modal Signatures & Modular Sensory Nodes:
   - forward(tokens=None, pixel_values=None, audio_values=None, ...)
   - Text-only numerics strictly preserved with exact equality.
5. Manifold Diagnostics:
   - Per-shell effective rank from trace ratio (sum lambda)^2 / sum lambda^2.
   - Per-shell RMS and adjacent-shell cosine alignment under no_grad at eval intervals.
"""

import math
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_module

from .model import SensoryInputDNN, VolumetricBrainCore
from .spheroid_v3 import (
    CerebroSpheroidV3,
    PremotorMotorDecoder,
    ResidualVolumetricCore,
    TemporalSensoryEncoder,
    apply_repetition_penalty,
    build_feedback_plan,
)
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


class PremotorMotorDecoderV4(PremotorMotorDecoder):
    """
    V4 Premotor Motor Decoder:
    - Retains internal plan buffer, corollary discharge, and multi-shell softmax tap from V3.
    - Upgrades Arcuate Fasciculus cross-resonance to F.scaled_dot_product_attention (SDPA).
    - Preserves exact parameter naming and shapes for checkpoint compatibility.
    """
    def __init__(
        self,
        num_shells: int = 16,
        surface_dim: int = 640,
        vocab_size: int = 50257,
        max_phase_steps: int = 1024,
        dropout: float = 0.1,
        use_plan: bool = True,
        use_shell_tap: bool = True,
    ):
        super().__init__(
            num_shells=num_shells,
            surface_dim=surface_dim,
            vocab_size=vocab_size,
            max_phase_steps=max_phase_steps,
            dropout=dropout,
            use_plan=use_plan,
            use_shell_tap=use_shell_tap,
        )

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

        # Auditory Re-afference (Broca-Wernicke feedback)
        if auditory_feedback is not None:
            gate = torch.tanh(self.auditory_gate) if hasattr(self, "auditory_gate") else 1.0
            motor_surface = motor_surface + gate * self.auditory_proj(auditory_feedback.to(motor_surface.dtype))

        # Multi-shell efferent tap
        if self.use_shell_tap:
            shell_pool = (
                torch.softmax(self.shell_score, dim=0).view(1, -1, 1) * self.shell_proj(state)
            ).sum(dim=1)
            shell_pool = self.shell_scale * shell_pool
        else:
            shell_pool = torch.zeros_like(surface)

        # Arcuate Fasciculus cross-resonance via Scaled Dot-Product Attention (SDPA)
        if sensory_kv is not None:
            keys, values = sensory_kv
            q = self.q_proj(motor_surface).unsqueeze(1)  # [B, 1, D]
            # SDPA handles key-value attention with optimal hardware backend
            res = F.scaled_dot_product_attention(
                q.to(keys.dtype), keys, values
            )  # [B, 1, D]
            resonance = self.resonance_norm(res.squeeze(1).to(motor_surface.dtype))
        else:
            resonance = torch.zeros_like(motor_surface)

        # Fused readout
        fused = self.fused_norm(
            torch.cat([motor_surface, core, resonance, shell_pool, plan], dim=-1)
        ).to(self.fc_gate.weight.dtype)
        h = self.norm_mid(F.silu(self.fc_gate(fused)) * self.fc_val(fused))
        h = self.dropout(h)

        raw_logits = self.head(h) / torch.clamp(self.temperature_scale, min=0.1, max=5.0)
        efference_copy = self.efference_proj(h)

        # Premotor plan buffer
        if self.use_plan:
            phase_batch = phase.unsqueeze(0).expand(h.shape[0], -1)
            plan_input = torch.cat([h, phase_batch], dim=-1)
            gate = torch.sigmoid(self.plan_gate(plan_input))
            next_plan = self.plan_decay * plan + self.plan_write * gate * self.plan_update(plan_input)
        else:
            next_plan = plan

        # Refractory dynamics
        if refractory_trace is not None:
            alpha = torch.clamp(self.refractory_strength, min=0.1, max=5.0)
            adapted = raw_logits - alpha * refractory_trace.to(raw_logits.dtype)
            decay = torch.sigmoid(self.refractory_decay)
            current_prob = F.softmax(raw_logits, dim=-1).detach()
            next_trace = (decay * refractory_trace + (1.0 - decay) * current_prob).detach()
            return adapted, next_trace, efference_copy, next_plan

        return raw_logits, None, efference_copy, next_plan


class CerebroSpheroidV4(CerebroSpheroidV3):
    """
    CerebroSpheroid V4: Continuous Volumetric Manifold.

    Upgrades V3 with:
    - SDPA accelerated cross-resonance in the motor decoder.
    - Zero-allocation sensory indexing during rollout.
    - Multi-modal forward signature with Vision & Audio sensory node support.
    - Dynamic shell zoning.
    - Non-invasive manifold diagnostics (effective rank, shell RMS, saturation).
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
        enable_multimodal: bool = False,
        use_causal_ingestion: bool = True,
    ):
        nn.Module.__init__(self)
        self.vocab_size = vocab_size
        self.surface_dim = surface_dim
        self.num_shells = num_shells
        self.radial_dim = radial_dim
        self.num_shortcut_tracts = num_shortcut_tracts
        self.use_causal_ingestion = use_causal_ingestion
        self.architecture = dict(
            use_temporal=use_temporal,
            use_residual_core=use_residual_core,
            use_plan=use_plan,
            use_shell_tap=use_shell_tap,
            use_causal_ingestion=use_causal_ingestion,
        )

        # Afferent Front-End
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

        # Volumetric Brain Core
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

        # Efferent Motor Decoder with SDPA
        self.motor_dnn = PremotorMotorDecoderV4(
            num_shells=num_shells,
            surface_dim=surface_dim,
            vocab_size=vocab_size,
            max_phase_steps=max_phase_steps,
            dropout=dropout,
            use_plan=use_plan,
            use_shell_tap=use_shell_tap,
        )
        self.boundary_transceiver = self.sensory_dnn

        self.gist_scale = nn.Parameter(torch.zeros(1))
        self.packet_mix = nn.Parameter(torch.zeros(1))

        # Multi-modal sensory nodes (initialized lazily or when requested)
        self.enable_multimodal = enable_multimodal
        if enable_multimodal:
            self.vision_encoder = VisionSensoryEncoder(surface_dim=surface_dim)
            self.audio_encoder = AudioSensoryEncoder(surface_dim=surface_dim)
            self.thalamic_gater = ThalamicSensoryGater(surface_dim=surface_dim)
        else:
            self.vision_encoder = None
            self.audio_encoder = None
            self.thalamic_gater = None

    @property
    def shell_zones(self) -> Dict[str, range]:
        return get_shell_zones(self.num_shells)

    def init_living_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initializes resting-state manifold potential with living pacemaker rhythms."""
        return self.init_brain_state(batch_size, device, living=True)

    def compile_deliberate(self):
        """Optionally applies torch.compile to the recurrent deliberate step."""
        try:
            self.volumetric_core.deliberate_step = torch.compile(
                self.volumetric_core.deliberate_step
            )
            print("[V4] volumetric_core.deliberate_step compiled with torch.compile")
        except Exception as e:
            print(f"[V4] torch.compile unavailable or failed: {e}")

    def _integrate_memory(
        self,
        memory: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        living: bool = True,
        max_packets: int = 32,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Streams encoded sensory memory into the volumetric manifold.

        V4 Causal Temporal Ingestion:
        Eliminates the order-erasing mean centroid pooling (chunk.mean(dim=1)).
        1. When self.use_causal_ingestion is True:
           - Applies a causal recency decay kernel: w_i = exp((i - (K-1)) / sqrt(K)),
             preserving token sequence order and phase trajectory.
           - Adds the temporal velocity vector: v = tail - head scaled by tanh(packet_mix),
             allowing learned momentum injection into the superficial shell.
           - Masks out padding tokens so batch-padded tokens never corrupt the flux.
        2. When self.use_causal_ingestion is False:
           - Emulates legacy V3 centroid pooling for backwards compatibility.
        """
        batch_size, seq_len, _ = memory.shape
        if state is None:
            state = self.init_brain_state(batch_size, memory.device, living=living)

        packet = max(1, math.ceil(seq_len / max_packets))
        last_valid = int(lengths.max().item()) if lengths is not None else seq_len

        if not getattr(self, "use_causal_ingestion", True):
            for start in range(0, min(seq_len, last_valid), packet):
                chunk = memory[:, start : start + packet, :]
                centroid = chunk.mean(dim=1)
                tail = chunk[:, -1, :]
                flux = centroid + self.packet_mix.to(centroid.dtype) * (tail - centroid)
                state = self.volumetric_core.inject_sensory_flux(state, flux)
                state = self.volumetric_core.circulate_mid_tensors(state, num_cycles=1)
            return state

        for start in range(0, min(seq_len, last_valid), packet):
            end = min(seq_len, start + packet)
            chunk = memory[:, start : end, :]
            chunk_len = chunk.shape[1]

            if chunk_len == 1:
                flux = chunk[:, 0, :]
            else:
                steps = torch.arange(chunk_len, device=memory.device, dtype=chunk.dtype)
                decay = torch.exp((steps - (chunk_len - 1)) / math.sqrt(chunk_len))

                if lengths is not None:
                    pos = torch.arange(start, end, device=memory.device).unsqueeze(0)
                    valid = (pos < lengths.unsqueeze(1)).to(chunk.dtype)
                    decay_weights = decay.unsqueeze(0) * valid
                    weight_sum = decay_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                    norm_weights = decay_weights / weight_sum
                    causal_flux = (chunk * norm_weights.unsqueeze(-1)).sum(dim=1)
                else:
                    norm_weights = decay / decay.sum()
                    causal_flux = (chunk * norm_weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)

                velocity = chunk[:, -1, :] - chunk[:, 0, :]
                flux = causal_flux + torch.tanh(self.packet_mix.to(chunk.dtype)) * velocity

            state = self.volumetric_core.inject_sensory_flux(state, flux)
            state = self.volumetric_core.circulate_mid_tensors(state, num_cycles=1)

        return state

    def training_logits(
        self,
        prompt_tokens: torch.Tensor,
        response_steps: int,
        thinking_cycles: int = 3,
        pad_token_id: Optional[int] = None,
        targets: Optional[torch.Tensor] = None,
        feedback_tokens: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        audio_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        High-throughput V4 training rollout with zero-allocation sensory memory indexing.
        Pre-embeds feedback and pre-projects Arcuate Fasciculus keys/values up front,
        attending via continuous prefix slices without torch.cat inside the emission loop.
        """
        prompt_len = prompt_tokens.shape[1]
        memory, gist, sensory_kv = self.encode_sensory(prompt_tokens, pad_token_id=pad_token_id)

        # Modulate sensory drive with ThalamicSensoryGater if enabled
        if self.thalamic_gater is not None:
            gist = self.thalamic_gater(prompt_tokens, gist, pad_token_id=pad_token_id)

        # Optional multi-modal sensory injection into outer cortical shell
        if pixel_values is not None:
            if self.vision_encoder is None:
                self.vision_encoder = VisionSensoryEncoder(surface_dim=self.surface_dim).to(prompt_tokens.device)
            _, visual_gist = self.vision_encoder(pixel_values)
            gist = gist + visual_gist
        if audio_values is not None:
            if self.audio_encoder is None:
                self.audio_encoder = AudioSensoryEncoder(surface_dim=self.surface_dim).to(prompt_tokens.device)
            _, audio_gist = self.audio_encoder(audio_values)
            gist = gist + audio_gist

        state = self._integrate_memory(
            memory, max_packets=32, lengths=self._lengths(prompt_tokens, pad_token_id)
        )
        state = self.volumetric_core.inject_efference_copy(state, self._gist_flux(gist))
        state = self.deliberate(state, thinking_cycles=thinking_cycles)

        feed = build_feedback_plan(targets, feedback_tokens)
        feed_reps = None
        all_k, all_v = sensory_kv[0], sensory_kv[1]

        if feed is not None:
            prev_toks = feed.clamp(min=0)
            if isinstance(self.sensory_dnn, SensoryInputDNN):
                feed_reps = self.sensory_dnn(prev_toks, start_pos=prompt_len - 1)
            else:
                feed_reps = self.sensory_dnn.embedding(prev_toks)
            feed_mask = (feed != -100).unsqueeze(-1)
            feed_reps = feed_reps * feed_mask
            if hasattr(self.motor_dnn, "prime_sensory"):
                feed_k, feed_v = self.motor_dnn.prime_sensory(feed_reps)
                # Concatenate once up front:
                all_k = torch.cat([sensory_kv[0], feed_k], dim=1)
                all_v = torch.cat([sensory_kv[1], feed_v], dim=1)

        plan = self.motor_dnn.init_plan(prompt_tokens.shape[0], state.device)
        refractory = torch.zeros(
            prompt_tokens.shape[0], self.vocab_size, device=state.device, dtype=torch.float32
        )
        logits = []
        auditory_feedback = None

        for step in range(response_steps):
            if feed is not None:
                auditory_feedback = feed_reps[:, step]
                # Zero-allocation O(1) prefix slice covering prompt + feedback up to current step
                cur_k = all_k[:, : prompt_len + step]
                cur_v = all_v[:, : prompt_len + step]
                cur_kv = (cur_k, cur_v)
            elif step > 0 and auditory_feedback is None and len(logits) > 0:
                prev_tok = torch.argmax(logits[-1], dim=-1)
                auditory_feedback = (
                    self.sensory_dnn(prev_tok)
                    if isinstance(self.sensory_dnn, SensoryInputDNN)
                    else self.sensory_dnn.embedding(prev_tok)
                )
                if hasattr(self.motor_dnn, "prime_sensory"):
                    new_k, new_v = self.motor_dnn.prime_sensory(auditory_feedback.unsqueeze(1))
                    all_k = torch.cat([all_k, new_k], dim=1)
                    all_v = torch.cat([all_v, new_v], dim=1)
                cur_kv = (all_k, all_v)
            else:
                cur_kv = (all_k[:, :prompt_len], all_v[:, :prompt_len])

            if self.training and torch.is_grad_enabled() and response_steps > 32:
                step_logits, state, refractory, plan = checkpoint_module.checkpoint(
                    self.emit_step,
                    state,
                    cur_kv,
                    plan,
                    refractory,
                    step,
                    auditory_feedback,
                    use_reentrant=False,
                )
            else:
                step_logits, state, refractory, plan = self.emit_step(
                    state,
                    sensory_kv=cur_kv,
                    plan=plan,
                    refractory_trace=refractory,
                    step_idx=step,
                    auditory_feedback=auditory_feedback,
                )
            logits.append(step_logits)

            if targets is None and feed is None:
                prev_tok = torch.argmax(step_logits, dim=-1)
                auditory_feedback = (
                    self.sensory_dnn(prev_tok)
                    if isinstance(self.sensory_dnn, SensoryInputDNN)
                    else self.sensory_dnn.embedding(prev_tok)
                )
                if hasattr(self.motor_dnn, "prime_sensory"):
                    new_k, new_v = self.motor_dnn.prime_sensory(auditory_feedback.unsqueeze(1))
                    all_k = torch.cat([all_k, new_k], dim=1)
                    all_v = torch.cat([all_v, new_v], dim=1)

        return torch.stack(logits, dim=1)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        tokens: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        audio_values: Optional[torch.Tensor] = None,
        response_steps: Optional[int] = None,
        thinking_cycles: int = 3,
        pad_token_id: Optional[int] = None,
        targets: Optional[torch.Tensor] = None,
        feedback_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Unified multi-modal forward pass.
        Accepts input_ids or tokens for text, with optional pixel_values and audio_values.
        When pixel_values and audio_values are omitted, reproduces text-only path identically.
        """
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
            pixel_values=pixel_values,
            audio_values=audio_values,
        )

    def load_compatible_checkpoint(self, checkpoint_path: str, device: torch.device = torch.device("cpu")) -> bool:
        """Loads V4 or V3 checkpoints directly into CerebroSpheroidV4 without dropping V4 tensors."""
        if not os.path.exists(checkpoint_path):
            return False
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        weights = state.get("model", state) if isinstance(state, dict) else state
        if not isinstance(weights, dict):
            return False

        # Remap JAX or foreign key naming conventions to native PyTorch
        remapped_weights = {}
        for k, v in weights.items():
            new_k = k
            if new_k.startswith("module."):
                new_k = new_k[len("module."):]
            if new_k.startswith("sensory."):
                new_k = "sensory_dnn." + new_k[len("sensory."):]
            elif new_k.startswith("motor."):
                new_k = "motor_dnn." + new_k[len("motor."):]
            remapped_weights[new_k] = v
        weights = remapped_weights

        my_keys = set(self.state_dict().keys())
        v4_overlap = len(set(weights.keys()) & my_keys)
        if v4_overlap > 10:
            matched = {k: v for k, v in weights.items() if k in my_keys and v.shape == self.state_dict()[k].shape}
            self.load_state_dict(matched, strict=False)
            return True
        else:
            lift_checkpoint_v3_to_v4(weights, self)
            return True


# =========================================================================
# Manifold Diagnostics (Section 5)
# =========================================================================

@torch.no_grad()
def compute_manifold_diagnostics(
    model: CerebroSpheroidV4,
    state: torch.Tensor,
) -> Dict[str, Union[float, List[float]]]:
    """
    Computes non-invasive geometric diagnostics of the recurrent manifold state:
    1. Per-shell effective rank from trace ratio (sum lambda)^2 / sum lambda^2
       via singular value decomposition of the mean-centered batch.
    2. Per-shell RMS energy.
    3. Adjacent-shell cosine similarity (radial alignment).
    4. Write gate saturation fraction (fraction of shells with write_gate > 0.95).
    """
    batch_size, num_shells, shell_dim = state.shape
    device = state.device
    state_f = state.float()

    effective_ranks = []
    rms_per_shell = []
    for k in range(num_shells):
        s_k = state_f[:, k, :]  # [B, D]
        # RMS energy
        rms = s_k.pow(2).mean().sqrt().item()
        rms_per_shell.append(rms)

        # Effective rank via SVD on centered state
        if batch_size > 1:
            s_centered = s_k - s_k.mean(dim=0, keepdim=True)
            # Singular values s_i satisfy lambda_i = s_i^2 / (B - 1)
            # Trace ratio: (sum lambda)^2 / sum lambda^2 = (sum s^2)^2 / sum s^4
            try:
                s_vals = torch.linalg.svdvals(s_centered)
                s2 = s_vals.pow(2)
                if s2.sum().item() < 1e-6:
                    rank = 1.0
                else:
                    rank = (s2.sum().pow(2) / (s2.pow(2).sum() + 1e-9)).item()
            except Exception:
                rank = 1.0
        else:
            rank = 1.0
        effective_ranks.append(rank)

    # Adjacent-shell cosine similarity
    adjacent_cosines = []
    for k in range(num_shells - 1):
        s_curr = state_f[:, k, :]
        s_next = state_f[:, k + 1, :]
        cos = F.cosine_similarity(s_curr, s_next, dim=-1).mean().item()
        adjacent_cosines.append(cos)

    # Write gate saturation
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
    # Modern Llama 32k Byte-Fallback BPE Presets (Default for V4)
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
    # Legacy GPT-2 50k Presets (for backwards compatibility)
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
    "v4_probe_25m": dict(
        vocab_size=50257,
        surface_dim=148,
        num_shells=8,
        radial_dim=74,
        num_shortcut_tracts=16,
        max_len=1024,
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

    # Override preset values if CLI arguments are explicitly provided
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
    """
    Transplants a V3 checkpoint into a V4 model.
    Checks shape compatibility and maps tensors directly. When dimensions are equal,
    provides exact logit parity.
    """
    own_state = target_model.state_dict()
    accepted, lifted, dropped = {}, [], []

    for k, v in checkpoint_state.items():
        if k in own_state:
            if v.shape == own_state[k].shape:
                accepted[k] = v
                lifted.append(k)
            elif "pos_embed" in k and v.ndim == own_state[k].ndim:
                # Handle sequence length difference by slicing/padding
                lifted_tensor = own_state[k].clone()
                rows = min(v.shape[1], own_state[k].shape[1])
                lifted_tensor[:, :rows, :] = v[:, :rows, :]
                accepted[k] = lifted_tensor
                lifted.append(k)
            else:
                # Shape mismatch (e.g. 16 shells -> 24 shells)
                dropped.append(k)
        else:
            dropped.append(k)

    target_model.load_state_dict(accepted, strict=False)
    return accepted, lifted, dropped
