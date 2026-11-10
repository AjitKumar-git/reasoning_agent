"""
CerebroSpheroid V4: Scaled, Measured Reasoning Manifold (Pure JAX Implementation).

Biological & Mathematical Principles:
1. Afferent Sensory Input (SwiGLU) -> Continuous Riemannian boundary flux & Arcuate Fasciculus KV.
2. Continuous Concentric Volumetric Core (B^3) -> Contractive unit-gain dynamics,
   radial wave diffusion, and subcortical axonal shortcut tracts.
3. Efferent Motor Output (Broca's Area) -> Multi-Shell Softmax with Arcuate Fasciculus
   cross-resonance, Premotor Phase Clock, Recurrent Plan, and Refractory Dynamics.
4. Zero-Feedback Autonomous Reasoning:
   Rollout trajectory unrolls via internal state dynamics without token teacher-forcing.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union, List
import math

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax


DEFAULT_RMS_EPS = 1e-6


@dataclass
class ModelConfigV4:
    vocab_size: int = 32000          # LLaMA 32k Byte-Fallback BPE
    surface_dim: int = 640           # Spheroid boundary dimension
    num_shells: int = 16             # Concentric spherical laminae
    radial_dim: int = 256            # Radial wave channel depth
    num_shortcut_tracts: int = 40    # Axonal shortcut tracts
    max_phase_steps: int = 1024      # Premotor phase clock
    thinking_cycles: int = 3         # Subconscious deliberation steps
    use_shell_tap: bool = True       # Multi-shell efferent tap
    use_plan: bool = True            # Recurrent premotor plan


# Register ModelConfigV4 as a static JAX PyTree so remat / jit never treat it as an array
jax.tree_util.register_pytree_node(
    ModelConfigV4,
    lambda c: ((), (c.vocab_size, c.surface_dim, c.num_shells, c.radial_dim, c.num_shortcut_tracts, c.max_phase_steps, c.thinking_cycles, c.use_shell_tap, c.use_plan)),
    lambda aux, _: ModelConfigV4(*aux),
)


PRESETS_V4: Dict[str, Dict[str, Any]] = {
    "small": {
        "surface_dim": 640,
        "num_shells": 16,
        "radial_dim": 256,
        "num_shortcut_tracts": 40,
        "thinking_cycles": 3,
    },
    "wide": {
        "surface_dim": 896,
        "num_shells": 24,
        "radial_dim": 384,
        "num_shortcut_tracts": 48,
        "thinking_cycles": 3,
    },
}


def rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = DEFAULT_RMS_EPS) -> jnp.ndarray:
    """Root Mean Square Layer Normalization."""
    rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def precompute_axonal_indices(num_shells: int, num_shortcut_tracts: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Precomputes small-world geodesic shortcut endpoints across spherical shells."""
    src = jnp.arange(num_shortcut_tracts) % num_shells
    dst = num_shells - 1 - (jnp.arange(num_shortcut_tracts) * 3) % num_shells
    return src.astype(jnp.int32), dst.astype(jnp.int32)


def init_v4_params(
    config: ModelConfigV4, key: jax.Array
) -> Tuple[Dict[str, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
    """Initializes CerebroSpheroid V4 parameters in pure JAX matching PyTorch key naming."""
    keys = jax.random.split(key, 25)
    vocab = config.vocab_size
    s_dim = config.surface_dim
    shells = config.num_shells
    r_dim = config.radial_dim
    n_tracts = config.num_shortcut_tracts
    fused_dim = s_dim * 5
    src_idx, dst_idx = precompute_axonal_indices(shells, n_tracts)

    p: Dict[str, jnp.ndarray] = {
        # Recurrent Volumetric Core (B^3)
        "core.intra_shell": jax.random.normal(keys[0], (shells, s_dim, s_dim)) * 0.015,
        "core.reentrant_field": jax.random.normal(keys[1], (shells, shells, r_dim, r_dim)) * 0.008,
        "core.shortcut_tracts": jax.random.normal(keys[2], (n_tracts, s_dim, s_dim)) * 0.01,
        "core.radial_proj_in.weight": jax.random.normal(keys[3], (r_dim, s_dim)) * (1.0 / math.sqrt(s_dim)),
        "core.radial_proj_out.weight": jax.random.normal(keys[4], (s_dim, r_dim)) * (1.0 / math.sqrt(r_dim)),
        "core.norm.weight": jnp.ones(s_dim, dtype=jnp.float32),
        "core.flow_norm.weight": jnp.ones(s_dim, dtype=jnp.float32),
        "core.log_write": jnp.full((shells,), math.log(math.expm1(0.35)), dtype=jnp.float32),
        "core.log_leak": jnp.full((shells,), math.log(0.35 / 0.65), dtype=jnp.float32),

        # Sensory Front-End
        "sensory.embedding.weight": jax.random.normal(keys[5], (vocab, s_dim)) * 0.02,
        "sensory.fc1_gate.weight": jax.random.normal(keys[6], (s_dim * 2, s_dim)) * (1.0 / math.sqrt(s_dim)),
        "sensory.fc1_val.weight": jax.random.normal(keys[7], (s_dim * 2, s_dim)) * (1.0 / math.sqrt(s_dim)),
        "sensory.norm1.weight": jnp.ones(s_dim * 2, dtype=jnp.float32),
        "sensory.fc2.weight": jax.random.normal(keys[8], (s_dim, s_dim * 2)) * (1.0 / math.sqrt(s_dim * 2)),
        "sensory.norm_out.weight": jnp.ones(s_dim, dtype=jnp.float32),

        # Motor Output Decoder (Broca's Area)
        "motor.phase_embed.weight": jax.random.normal(keys[9], (config.max_phase_steps, s_dim)) * 0.02,
        "motor.q_proj.weight": jax.random.normal(keys[10], (s_dim, s_dim)) * (1.0 / math.sqrt(s_dim)),
        "motor.k_proj.weight": jax.random.normal(keys[11], (s_dim, s_dim)) * (1.0 / math.sqrt(s_dim)),
        "motor.v_proj.weight": jax.random.normal(keys[12], (s_dim, s_dim)) * (1.0 / math.sqrt(s_dim)),
        "motor.resonance_norm.weight": jnp.ones(s_dim, dtype=jnp.float32),
        "motor.shell_proj.weight": jax.random.normal(keys[13], (s_dim, s_dim)) * (1.0 / math.sqrt(s_dim)),
        "motor.shell_score": jnp.zeros((shells,), dtype=jnp.float32),
        "motor.shell_scale": jnp.zeros((1,), dtype=jnp.float32),
        "motor.plan_gate.weight": jax.random.normal(keys[14], (s_dim, s_dim * 3)) * (1.0 / math.sqrt(s_dim * 3)),
        "motor.plan_gate.bias": jnp.zeros((s_dim,), dtype=jnp.float32),
        "motor.plan_update.weight": jax.random.normal(keys[15], (s_dim, s_dim * 3)) * (1.0 / math.sqrt(s_dim * 3)),
        "motor.plan_write": jnp.zeros((1,), dtype=jnp.float32),
        "motor.plan_decay": jnp.array(0.90, dtype=jnp.float32),
        "motor.fused_norm.weight": jnp.ones(fused_dim, dtype=jnp.float32),
        "motor.fc_gate.weight": jax.random.normal(keys[16], (s_dim * 2, fused_dim)) * (1.0 / math.sqrt(fused_dim)),
        "motor.fc_val.weight": jax.random.normal(keys[17], (s_dim * 2, fused_dim)) * (1.0 / math.sqrt(fused_dim)),
        "motor.norm_mid.weight": jnp.ones(s_dim * 2, dtype=jnp.float32),
        "motor.head.weight": jax.random.normal(keys[18], (vocab, s_dim * 2)) * (1.0 / math.sqrt(s_dim * 2)),
        "motor.temperature_scale": jnp.array(1.0, dtype=jnp.float32),
        "motor.refractory_strength": jnp.array(1.2, dtype=jnp.float32),
        "motor.refractory_decay": jnp.array(0.75, dtype=jnp.float32),
        "motor.efference_proj.weight": jax.random.normal(keys[19], (s_dim, s_dim * 2)) * 0.01,

        # Agentic Flow & Persona Steering Basis (0=Reasoner, 1=Explorer, 2=Concise, 3=Verifier)
        "motor.persona_embeddings": jax.random.normal(keys[20], (4, s_dim)) * 0.05,
        "motor.persona_scale": jnp.array(0.5, dtype=jnp.float32),
        "motor.plan_init.weight": jax.random.normal(keys[21], (s_dim, s_dim * 2)) * (1.0 / math.sqrt(s_dim * 2)),
        "motor.plan_init.bias": jnp.zeros((s_dim,), dtype=jnp.float32),
    }
    buffers = (src_idx, dst_idx)
    return p, buffers


def encode_sensory(
    params: Dict[str, jnp.ndarray], token_ids: jnp.ndarray, config: Optional[ModelConfigV4] = None
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """Encodes token IDs into continuous sensory memory and precomputed Arcuate Fasciculus K/V."""
    x = params["sensory.embedding.weight"][token_ids]
    gate = jax.nn.silu(jnp.matmul(x, params["sensory.fc1_gate.weight"].T))
    val = jnp.matmul(x, params["sensory.fc1_val.weight"].T)
    h = rms_norm(gate * val, params["sensory.norm1.weight"])
    flux = jnp.matmul(h, params["sensory.fc2.weight"].T) + x
    sensory_memory = rms_norm(flux, params["sensory.norm_out.weight"])
    keys = jnp.matmul(sensory_memory, params["motor.k_proj.weight"].T)
    values = jnp.matmul(sensory_memory, params["motor.v_proj.weight"].T)
    return sensory_memory, (keys, values)


def deliberate_step(
    params: Dict[str, jnp.ndarray],
    state: jnp.ndarray,
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    config: Optional[ModelConfigV4] = None,
) -> jnp.ndarray:
    """Single subconscious deliberation tick across concentric spherical shells."""
    B, K, D = state.shape
    src, dst = buffers

    # 1. Tangential flow (Laplace-Beltrami intra-shell diffusion)
    intra = jnp.einsum("bki,kij->bkj", state, params["core.intra_shell"])

    # 2. Axonal shortcut tracts
    state_src = state[:, src, :]
    tracts_out = jnp.einsum("bti,tij->btj", state_src, params["core.shortcut_tracts"])
    tract_acc = jnp.zeros_like(state).at[:, dst, :].add(tracts_out)

    # 3. Radial re-entrant wave coupling
    rad_in = jnp.matmul(state, params["core.radial_proj_in.weight"].T)
    reentrant = jnp.einsum("bki,kjio->bjo", rad_in, params["core.reentrant_field"])
    reentrant = jnp.matmul(jax.nn.silu(reentrant), params["core.radial_proj_out.weight"].T)

    flow = rms_norm(intra + tract_acc + reentrant, params["core.flow_norm.weight"])
    leak = jax.nn.sigmoid(params["core.log_leak"])[None, :, None]
    write = jax.nn.softplus(params["core.log_write"])[None, :, None]

    raw = (1.0 - leak) * state + write * jnp.tanh(flow)
    return rms_norm(raw, params["core.norm.weight"])


def emit_step(
    params: Dict[str, jnp.ndarray],
    state: jnp.ndarray,
    sensory_kv: Tuple[jnp.ndarray, jnp.ndarray],
    plan: jnp.ndarray,
    refractory_trace: Optional[jnp.ndarray],
    step_idx: Union[int, jnp.ndarray],
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    config: Optional[ModelConfigV4] = None,
    auditory_feedback: Optional[jnp.ndarray] = None,
    repetition_penalty: float = 1.25,
) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray]:
    """Single autoregressive / deliberative motor emission step with auditory feedback support."""
    B, K, D = state.shape
    surface, core = state[:, -1, :], state[:, 0, :]
    max_phase = params["motor.phase_embed.weight"].shape[0]
    phase_idx = jnp.clip(step_idx, 0, max_phase - 1)
    phase = params["motor.phase_embed.weight"][phase_idx]
    # Scaled phase injection: scale temporal clock signal to prevent static attractor well traps
    p_norm = jnp.maximum(jnp.linalg.norm(phase, axis=-1, keepdims=True), 1e-6)
    p_scaled = phase * (2.0 / p_norm)
    motor_surface = surface + p_scaled

    if auditory_feedback is not None:
        motor_surface = motor_surface + auditory_feedback

    # Cross-resonance with sensory prompt (Arcuate Fasciculus)
    keys, values = sensory_kv
    query = jnp.matmul(motor_surface, params["motor.q_proj.weight"].T)[:, None, :]
    scores = jnp.matmul(query, jnp.swapaxes(keys, -1, -2)) / math.sqrt(D)
    attn_weights = jax.nn.softmax(scores, axis=-1)
    resonance = rms_norm(jnp.squeeze(jnp.matmul(attn_weights, values), axis=1), params["motor.resonance_norm.weight"])

    # Multi-shell pooling
    shell_weights = jax.nn.softmax(params["motor.shell_score"], axis=0)[None, :, None]
    shell_proj = jnp.matmul(state, params["motor.shell_proj.weight"].T)
    shell_pool = jnp.sum(shell_weights * shell_proj, axis=1) * params["motor.shell_scale"]

    fused = rms_norm(
        jnp.concatenate([motor_surface, core, resonance, shell_pool, plan], axis=-1),
        params["motor.fused_norm.weight"],
    )
    gate = jax.nn.silu(jnp.matmul(fused, params["motor.fc_gate.weight"].T))
    val = jnp.matmul(fused, params["motor.fc_val.weight"].T)
    h = rms_norm(gate * val, params["motor.norm_mid.weight"])

    # Premotor recurrent plan update
    phase_batch = jnp.repeat(p_scaled[None, :], B, axis=0)
    plan_input = jnp.concatenate([h, phase_batch], axis=-1)
    p_gate = jax.nn.sigmoid(jnp.matmul(plan_input, params["motor.plan_gate.weight"].T) + params["motor.plan_gate.bias"])
    plan_decay = jnp.clip(params["motor.plan_decay"], 0.0, 0.92)
    plan_update = jnp.matmul(plan_input, params["motor.plan_update.weight"].T)
    next_plan = jnp.clip(
        plan_decay * plan + params["motor.plan_write"] * p_gate * plan_update,
        -50.0,
        50.0,
    )

    temp = jnp.clip(params["motor.temperature_scale"], 0.1, 5.0)
    raw_logits = jnp.matmul(h, params["motor.head.weight"].T) / temp
    efference = jnp.matmul(h, params["motor.efference_proj.weight"].T)

    # Two-Tier Biological Refractory Adaptation & Repetition Penalties
    if refractory_trace is not None:
        alpha = jnp.clip(params.get("motor.refractory_strength", 1.2), 0.1, 5.0)
        # Tier 1: Continuous probability trace subtraction
        tier1_penalty = 3.0 * alpha * refractory_trace

        # Tier 2: Multiplicative repetition discount for tokens with active recent trace
        active_mask = (refractory_trace > 0.02).astype(jnp.float32)
        rep_scale = jnp.where(raw_logits > 0, 1.0 / repetition_penalty, repetition_penalty)
        scaled_logits = raw_logits * (1.0 - active_mask) + (raw_logits * rep_scale) * active_mask
        adapted_logits = scaled_logits - tier1_penalty

        # Compute next trace with soft probability EMA + hard top-1 refractory pulse
        decay = jax.nn.sigmoid(params.get("motor.refractory_decay", 0.75))
        current_prob = lax.stop_gradient(jax.nn.softmax(raw_logits, axis=-1))
        top1_idx = jnp.argmax(adapted_logits, axis=-1)
        top1_pulse = lax.stop_gradient(jax.nn.one_hot(top1_idx, raw_logits.shape[-1]))
        next_trace = lax.stop_gradient(decay * refractory_trace + (1.0 - decay) * current_prob + 1.0 * top1_pulse)
    else:
        adapted_logits, next_trace = raw_logits, None

    # Efference copy injected back into subcortical core
    state_eff = state.at[:, 0, :].add(efference)
    next_state = deliberate_step(params, state_eff, buffers)
    return adapted_logits, next_state, next_trace, next_plan


def training_rollout(
    params: Dict[str, jnp.ndarray],
    prompts: jnp.ndarray,
    target_len: int,
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    config: Optional[ModelConfigV4] = None,
    feedback_tokens: Optional[jnp.ndarray] = None,
    deliberation_mode: str = "scheduled",
    sampling_p: float = 1.0,
    thalamic_anchor_weight: float = 0.1,
    pad_id: int = 0,
    persona_id: Optional[Union[int, jnp.ndarray]] = None,
    repetition_penalty: float = 1.25,
    targets: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Production-grade differentiable rollout engine with:
    1. Masked prompt centroid pooling (zero padding contamination).
    2. Subconscious deliberation cycles before verbalization.
    3. Plan-First pre-verbal executive synthesis from subcortical core + prompt flux.
    4. Agentic persona steering in the latent manifold.
    5. Scheduled sampling bridging teacher-forcing to internal thought re-afference.
    6. Thalamic memory basin anchoring deep shells (0 .. K//3) to prevent prompt erasure.
    7. Two-tier refractory adaptation & repetition suppression.
    """
    B, L = prompts.shape
    sensory_mem, sensory_kv = encode_sensory(params, prompts)
    num_shells = config.num_shells if config is not None else params["core.intra_shell"].shape[0]
    surface_dim = config.surface_dim if config is not None else params["sensory.fc2.weight"].shape[0]
    vocab_size = config.vocab_size if config is not None else params["sensory.embedding.weight"].shape[0]
    thinking_cycles = config.thinking_cycles if config is not None else 3

    # Masked mean centroid so padding tokens never dilute prompt semantics
    mask = (prompts != pad_id)[:, :, None].astype(jnp.float32)
    prompt_flux = jnp.sum(sensory_mem * mask, axis=1) / jnp.maximum(jnp.sum(mask, axis=1), 1.0)

    init_state = jnp.zeros((B, num_shells, surface_dim), dtype=jnp.float32)
    state = init_state.at[:, -1, :].set(prompt_flux)

    # Subconscious deliberation cycles before verbalization
    deliberate_remat = jax.checkpoint(deliberate_step)
    def _delib(s, _):
        return deliberate_remat(params, s, buffers), None

    state, _ = lax.scan(_delib, state, None, length=thinking_cycles)

    # Thalamic memory basin anchoring
    thalamic_end = max(1, num_shells // 3)
    thalamic_anchor = state[:, :thalamic_end, :]

    # Pre-Verbal Executive Plan Synthesis (Shell 0 Core State + Prompt Flux)
    if "motor.plan_init.weight" in params:
        delib_summary = jnp.concatenate([state[:, 0, :], prompt_flux], axis=-1)
        plan_raw = jnp.matmul(delib_summary, params["motor.plan_init.weight"].T)
        if "motor.plan_init.bias" in params:
            plan_raw = plan_raw + params["motor.plan_init.bias"]
        init_plan = jax.nn.tanh(plan_raw)
    else:
        init_plan = jnp.zeros((B, surface_dim), dtype=jnp.float32)

    # Built-in Agentic Persona Steering Injection (0=Reasoner, 1=Explorer, 2=Concise, 3=Verifier)
    if "motor.persona_embeddings" in params and persona_id is not None:
        p_scale = params.get("motor.persona_scale", 0.5)
        if isinstance(persona_id, int):
            persona_vec = params["motor.persona_embeddings"][persona_id][None, :]
        else:
            persona_vec = params["motor.persona_embeddings"][persona_id]
        init_plan = init_plan + p_scale * persona_vec

    init_refractory = jnp.zeros((B, vocab_size), dtype=jnp.float32)
    init_thought = jnp.zeros((B, surface_dim), dtype=jnp.float32)

    # Pre-embed teacher feedback tokens if available
    if feedback_tokens is not None:
        safe_feed = jnp.maximum(feedback_tokens, 0)
        valid_feed = (feedback_tokens >= 0)[:, :, None].astype(jnp.float32)
        tf_sensory, _ = encode_sensory(params, safe_feed, config=config)
        tf_sensory = tf_sensory * valid_feed
    else:
        tf_sensory = jnp.zeros((B, target_len, surface_dim), dtype=jnp.float32)

    emit_step_remat = jax.checkpoint(emit_step)

    def _emit(carry, step_idx):
        cur_state, cur_trace, cur_plan, prev_thought = carry

        # Determine auditory feed based on deliberation mode and sampling probability
        tf_aud = tf_sensory[:, step_idx, :]
        if deliberation_mode == "teacher_forcing":
            feed = tf_aud
        elif deliberation_mode == "internal":
            feed = jnp.where(step_idx == 0, jnp.zeros_like(prev_thought), prev_thought)
        else:  # "scheduled"
            internal_feed = jnp.where(step_idx == 0, tf_aud, prev_thought)
            feed = sampling_p * tf_aud + (1.0 - sampling_p) * internal_feed

        logits, nxt_s, nxt_t, nxt_p = emit_step_remat(
            params,
            cur_state,
            sensory_kv,
            cur_plan,
            cur_trace,
            step_idx,
            buffers,
            config=config,
            auditory_feedback=feed,
            repetition_penalty=repetition_penalty,
        )

        # Thought re-afference for step t+1 (stop-gradient isolates recurrent input to prevent 512-step BPTT explosion)
        soft_p = lax.stop_gradient(jax.nn.softmax(logits, axis=-1))
        soft_emb = jnp.matmul(soft_p, params["sensory.embedding.weight"])
        g = jax.nn.silu(jnp.matmul(soft_emb, params["sensory.fc1_gate.weight"].T))
        v = jnp.matmul(soft_emb, params["sensory.fc1_val.weight"].T)
        h_norm = rms_norm(g * v, params["sensory.norm1.weight"])
        nxt_thought = lax.stop_gradient(rms_norm(jnp.matmul(h_norm, params["sensory.fc2.weight"].T) + soft_emb, params["sensory.norm_out.weight"]))

        # Deep thalamic basin retention
        if thalamic_anchor_weight > 0.0:
            nxt_s = nxt_s.at[:, :thalamic_end, :].set(
                (1.0 - thalamic_anchor_weight) * nxt_s[:, :thalamic_end, :] + thalamic_anchor_weight * thalamic_anchor
            )

        if targets is not None:
            cur_target = targets[:, step_idx]
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            safe_targets = jnp.maximum(cur_target, 0)
            target_log_probs = jnp.take_along_axis(log_probs, safe_targets[:, None], axis=-1).squeeze(-1)
            mask_t = (cur_target >= 0).astype(jnp.float32)
            step_loss = -target_log_probs * mask_t
            return (nxt_s, nxt_t, nxt_p, nxt_thought), (step_loss, mask_t)

        return (nxt_s, nxt_t, nxt_p, nxt_thought), logits

    if targets is not None:
        _, (all_loss, all_mask) = lax.scan(_emit, (state, init_refractory, init_plan, init_thought), jnp.arange(target_len))
        return jnp.sum(all_loss) / jnp.maximum(jnp.sum(all_mask), 1.0)

    _, all_logits = lax.scan(_emit, (state, init_refractory, init_plan, init_thought), jnp.arange(target_len))
    return jnp.swapaxes(all_logits, 0, 1)


def generate_v4(
    params: Dict[str, jnp.ndarray],
    prompts: jnp.ndarray,
    max_new_tokens: int,
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    config: Optional[ModelConfigV4] = None,
    persona_id: Optional[int] = 0,
    temperature: float = 0.7,
    repetition_penalty: float = 1.25,
    thalamic_anchor_weight: float = 0.15,
    pad_id: int = 0,
    eos_token_id: Optional[int] = None,
) -> List[List[int]]:
    """
    High-fidelity autoregressive generation engine for CerebroSpheroid V4.
    """
    B, L = prompts.shape
    sensory_mem, sensory_kv = encode_sensory(params, prompts)
    num_shells = config.num_shells if config is not None else params["core.intra_shell"].shape[0]
    surface_dim = config.surface_dim if config is not None else params["sensory.fc2.weight"].shape[0]
    vocab_size = config.vocab_size if config is not None else params["sensory.embedding.weight"].shape[0]
    thinking_cycles = config.thinking_cycles if config is not None else 3

    # Masked prompt centroid
    mask = (prompts != pad_id)[:, :, None].astype(jnp.float32)
    prompt_flux = jnp.sum(sensory_mem * mask, axis=1) / jnp.maximum(jnp.sum(mask, axis=1), 1.0)

    init_state = jnp.zeros((B, num_shells, surface_dim), dtype=jnp.float32)
    state = init_state.at[:, -1, :].set(prompt_flux)

    # Subconscious deliberation cycles before verbalization
    for _ in range(thinking_cycles):
        state = deliberate_step(params, state, buffers)

    # Thalamic memory basin anchoring
    thalamic_end = max(1, num_shells // 3)
    thalamic_anchor = state[:, :thalamic_end, :]

    # Pre-verbal plan initialization
    if "motor.plan_init.weight" in params:
        delib_summary = jnp.concatenate([state[:, 0, :], prompt_flux], axis=-1)
        plan_raw = jnp.matmul(delib_summary, params["motor.plan_init.weight"].T)
        if "motor.plan_init.bias" in params:
            plan_raw = plan_raw + params["motor.plan_init.bias"]
        plan = jax.nn.tanh(plan_raw)
    else:
        plan = jnp.zeros((B, surface_dim), dtype=jnp.float32)

    # Persona steering vector
    if "motor.persona_embeddings" in params and persona_id is not None:
        p_scale = params.get("motor.persona_scale", 0.5)
        persona_vec = params["motor.persona_embeddings"][persona_id][None, :]
        plan = plan + p_scale * persona_vec

    refractory = jnp.zeros((B, vocab_size), dtype=jnp.float32)
    prev_thought = jnp.zeros((B, surface_dim), dtype=jnp.float32)
    results = [[] for _ in range(B)]
    completed = [False] * B

    for step_idx in range(max_new_tokens):
        logits, state, refractory, plan = emit_step(
            params,
            state,
            sensory_kv,
            plan,
            refractory,
            step_idx,
            buffers,
            config=config,
            auditory_feedback=prev_thought,
            repetition_penalty=repetition_penalty,
        )

        # Thought re-afference for step t+1
        soft_p = lax.stop_gradient(jax.nn.softmax(logits, axis=-1))
        soft_emb = jnp.matmul(soft_p, params["sensory.embedding.weight"])
        g = jax.nn.silu(jnp.matmul(soft_emb, params["sensory.fc1_gate.weight"].T))
        v = jnp.matmul(soft_emb, params["sensory.fc1_val.weight"].T)
        h_norm = rms_norm(g * v, params["sensory.norm1.weight"])
        prev_thought = lax.stop_gradient(rms_norm(jnp.matmul(h_norm, params["sensory.fc2.weight"].T) + soft_emb, params["sensory.norm_out.weight"]))

        # Thalamic retention
        if thalamic_anchor_weight > 0.0:
            state = state.at[:, :thalamic_end, :].set(
                (1.0 - thalamic_anchor_weight) * state[:, :thalamic_end, :] + thalamic_anchor_weight * thalamic_anchor
            )

        # Greedy / temperature selection
        if temperature <= 0.01:
            next_tokens = jnp.argmax(logits, axis=-1)
        else:
            scaled_logits = logits / temperature
            probs = jax.nn.softmax(scaled_logits, axis=-1)
            # Greedy argmax of tempered/penalized distribution
            next_tokens = jnp.argmax(probs, axis=-1)

        for b in range(B):
            if not completed[b]:
                tok_id = int(next_tokens[b])
                if eos_token_id is not None and tok_id == eos_token_id:
                    completed[b] = True
                else:
                    results[b].append(tok_id)

        if all(completed):
            break

    return results
