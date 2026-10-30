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
from typing import Optional, Tuple, Dict, Any
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
    keys = jax.random.split(key, 20)
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
    }
    buffers = (src_idx, dst_idx)
    return p, buffers


def encode_sensory(
    params: Dict[str, jnp.ndarray], token_ids: jnp.ndarray, config: ModelConfigV4
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
    config: ModelConfigV4,
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
    step_idx: int,
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    config: ModelConfigV4,
) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray]:
    """Single autoregressive/zero-feedback motor emission step."""
    B, K, D = state.shape
    surface, core = state[:, -1, :], state[:, 0, :]
    phase_idx = jnp.clip(step_idx, 0, config.max_phase_steps - 1)
    phase = params["motor.phase_embed.weight"][phase_idx]
    motor_surface = surface + phase

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
    phase_batch = jnp.repeat(phase[None, :], B, axis=0)
    plan_input = jnp.concatenate([h, phase_batch], axis=-1)
    p_gate = jax.nn.sigmoid(jnp.matmul(plan_input, params["motor.plan_gate.weight"].T) + params["motor.plan_gate.bias"])
    next_plan = (
        params["motor.plan_decay"] * plan
        + params["motor.plan_write"] * p_gate * jnp.matmul(plan_input, params["motor.plan_update.weight"].T)
    )

    temp = jnp.clip(params["motor.temperature_scale"], 0.1, 5.0)
    raw_logits = jnp.matmul(h, params["motor.head.weight"].T) / temp
    efference = jnp.matmul(h, params["motor.efference_proj.weight"].T)

    # Biological refractory adaptation (penalizes consecutive repetitive tokens)
    if refractory_trace is not None:
        alpha = jnp.clip(params["motor.refractory_strength"], 0.1, 5.0)
        adapted_logits = raw_logits - alpha * refractory_trace
        decay = jax.nn.sigmoid(params["motor.refractory_decay"])
        current_prob = lax.stop_gradient(jax.nn.softmax(raw_logits, axis=-1))
        next_trace = lax.stop_gradient(decay * refractory_trace + (1.0 - decay) * current_prob)
    else:
        adapted_logits, next_trace = raw_logits, None

    # Efference copy injected back into subcortical core
    state_eff = state.at[:, 0, :].add(efference)
    next_state = deliberate_step(params, state_eff, buffers, config)
    return adapted_logits, next_state, next_trace, next_plan


def training_rollout(
    params: Dict[str, jnp.ndarray],
    prompts: jnp.ndarray,
    target_len: int,
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    config: ModelConfigV4,
) -> jnp.ndarray:
    """Full unrolled zero-feedback training rollout over target_len steps."""
    B, L = prompts.shape
    sensory_mem, sensory_kv = encode_sensory(params, prompts, config)
    init_state = jnp.zeros((B, config.num_shells, config.surface_dim), dtype=jnp.float32)
    state = init_state.at[:, -1, :].set(jnp.mean(sensory_mem, axis=1))

    # Subconscious deliberation cycles before verbalization
    deliberate_remat = jax.checkpoint(deliberate_step)
    def _delib(s, _):
        return deliberate_remat(params, s, buffers, config), None

    state, _ = lax.scan(_delib, state, None, length=config.thinking_cycles)

    init_plan = jnp.zeros((B, config.surface_dim), dtype=jnp.float32)
    init_refractory = jnp.zeros((B, config.vocab_size), dtype=jnp.float32)

    # Gradient checkpointing on sequential emission reduces HBM temporaries from O(T) to O(1)
    emit_step_remat = jax.checkpoint(emit_step)

    def _emit(carry, step_idx):
        cur_state, cur_trace, cur_plan = carry
        logits, nxt_s, nxt_t, nxt_p = emit_step_remat(
            params, cur_state, sensory_kv, cur_plan, cur_trace, step_idx, buffers, config
        )
        return (nxt_s, nxt_t, nxt_p), logits

    _, all_logits = lax.scan(_emit, (state, init_refractory, init_plan), jnp.arange(target_len))
    return jnp.swapaxes(all_logits, 0, 1)
