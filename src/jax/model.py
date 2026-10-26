"""
CerebroSpheroid JAX Implementation:
High-Performance Volumetric Brain Manifold Architecture on JAX / XLA.

Biological Principles:
1. Afferent Sensory Input (SwiGLU / Causal Wavelets) -> continuous Riemannian boundary flux.
2. Continuous Concentric Volumetric Core (B^3) -> contractive unit-gain dynamics,
   radial wave diffusion, and subcortical axonal shortcut tracts.
3. Efferent Motor Output (Broca's Area) -> Dual-Tap (V2) or Multi-Shell Softmax (V3)
   with Arcuate Fasciculus cross-resonance, Premotor Phase Clock, and Refractory Dynamics.
4. Zero-Feedback Autonomous Reasoning:
   x_{t+1} != [x_{1..t}, y_t]. Internal state alone evolves and emits tokens.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List
import math

import jax
import jax.numpy as jnp
from jax import lax


# PyTorch default eps for fp32 RMSNorm is machine epsilon:
DEFAULT_RMS_EPS = 1.1920928955078125e-07


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    surface_dim: int = 640
    num_shells: int = 16
    radial_dim: int = 256
    num_shortcut_tracts: int = 40
    arch: str = "v2"  # "v2" or "v3"
    max_len: int = 2048
    max_phase_steps: int = 1024
    # V3 features
    use_temporal: bool = True
    use_residual_core: bool = True
    use_plan: bool = True
    use_shell_tap: bool = True
    use_flow_norm: bool = True
    energy_target: float = 1.0
    energy_clamp: Tuple[float, float] = (0.5, 2.0)
    dropout: float = 0.0

    @classmethod
    def from_scale(cls, scale: str = "laptop", arch: str = "v2") -> "ModelConfig":
        scales = {
            "laptop": dict(vocab_size=50257, surface_dim=640, num_shells=16, radial_dim=256, num_shortcut_tracts=40),
            "small": dict(vocab_size=50257, surface_dim=192, num_shells=8, radial_dim=64, num_shortcut_tracts=16),
            "2b": dict(vocab_size=32000, surface_dim=2048, num_shells=32, radial_dim=1024, num_shortcut_tracts=173),
            "260m": dict(vocab_size=32000, surface_dim=1024, num_shells=16, radial_dim=256, num_shortcut_tracts=48),
            "test": dict(vocab_size=1000, surface_dim=128, num_shells=8, radial_dim=64, num_shortcut_tracts=4),
        }
        if scale not in scales:
            raise ValueError(f"Unknown scale: {scale}. Choose from {list(scales.keys())}")
        return cls(arch=arch, **scales[scale])


def rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = DEFAULT_RMS_EPS) -> jnp.ndarray:
    """Root Mean Square Layer Normalization exactly matching PyTorch nn.RMSNorm."""
    rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def precompute_axonal_indices(num_shells: int, num_shortcut_tracts: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Precompute axonal tract wiring between non-adjacent shells."""
    src = jnp.arange(num_shortcut_tracts) % num_shells
    dst = num_shells - 1 - (jnp.arange(num_shortcut_tracts) * 3) % num_shells
    return src.astype(jnp.int32), dst.astype(jnp.int32)


# =========================================================================
# 1. Parameter Initialization (Pure JAX)
# =========================================================================

def init_params(config: ModelConfig, key: Optional[jax.Array] = None) -> Dict[str, jnp.ndarray]:
    """Initializes parameter dict matching PyTorch checkpoint keys and distributions."""
    if key is None:
        key = jax.random.PRNGKey(42)

    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13 = jax.random.split(key, 13)
    vocab = config.vocab_size
    s_dim = config.surface_dim
    shells = config.num_shells
    r_dim = config.radial_dim
    n_tracts = config.num_shortcut_tracts
    h_dim = s_dim

    src_idx, dst_idx = precompute_axonal_indices(shells, n_tracts)

    p: Dict[str, jnp.ndarray] = {
        # Core buffers
        "volumetric_core.src_indices": src_idx,
        "volumetric_core.dst_indices": dst_idx,

        # Volumetric Core
        "volumetric_core.intra_shell": jax.random.normal(k1, (shells, s_dim, s_dim)) * 0.015,
        "volumetric_core.reentrant_field": jax.random.normal(k2, (shells, shells, r_dim, r_dim)) * 0.008,
        "volumetric_core.shortcut_tracts": jax.random.normal(k3, (n_tracts, s_dim, s_dim)) * 0.01,
        "volumetric_core.inertia_gate": jnp.array(0.80, dtype=jnp.float32),
        "volumetric_core.rhythm_freq": jnp.array([0.25, 0.50], dtype=jnp.float32),
        "volumetric_core.spontaneous_amp": jnp.array(0.08, dtype=jnp.float32),
        "volumetric_core.radial_proj_in.weight": jax.random.normal(k4, (r_dim, s_dim)) * (1.0 / math.sqrt(s_dim)),
        "volumetric_core.radial_proj_out.weight": jax.random.normal(k5, (s_dim, r_dim)) * (1.0 / math.sqrt(r_dim)),
        "volumetric_core.norm.weight": jnp.ones(s_dim, dtype=jnp.float32),
    }

    # Sensory Input DNN
    p["sensory_dnn.embedding.weight"] = jax.random.normal(k6, (vocab, h_dim)) * 0.02
    p["sensory_dnn.fc1_gate.weight"] = jax.random.normal(k7, (h_dim * 2, h_dim)) * (1.0 / math.sqrt(h_dim))
    p["sensory_dnn.fc1_val.weight"] = jax.random.normal(k8, (h_dim * 2, h_dim)) * (1.0 / math.sqrt(h_dim))
    p["sensory_dnn.norm1.weight"] = jnp.ones(h_dim * 2, dtype=jnp.float32)
    p["sensory_dnn.fc2.weight"] = jax.random.normal(k9, (s_dim, h_dim * 2)) * (1.0 / math.sqrt(h_dim * 2))
    p["sensory_dnn.norm_out.weight"] = jnp.ones(s_dim, dtype=jnp.float32)

    # Motor Output DNN
    p["motor_dnn.motor_phase_embed.weight"] = jax.random.normal(k10, (config.max_phase_steps, s_dim)) * 0.02
    p["motor_dnn.q_proj.weight"] = jax.random.normal(k11, (s_dim, s_dim)) * (1.0 / math.sqrt(s_dim))
    p["motor_dnn.k_proj.weight"] = jax.random.normal(k12, (s_dim, s_dim)) * (1.0 / math.sqrt(s_dim))
    p["motor_dnn.v_proj.weight"] = jax.random.normal(k13, (s_dim, s_dim)) * (1.0 / math.sqrt(s_dim))
    p["motor_dnn.resonance_norm.weight"] = jnp.ones(s_dim, dtype=jnp.float32)

    p["motor_dnn.temperature_scale"] = jnp.array(1.0, dtype=jnp.float32)
    p["motor_dnn.refractory_strength"] = jnp.array(1.2, dtype=jnp.float32)
    p["motor_dnn.refractory_decay"] = jnp.array(0.75, dtype=jnp.float32)
    p["motor_dnn.efference_proj.weight"] = jax.random.normal(key, (s_dim, s_dim * 2)) * 0.01

    if config.arch == "v2":
        fused_dim = s_dim * 3
        p["motor_dnn.fused_norm.weight"] = jnp.ones(fused_dim, dtype=jnp.float32)
        p["motor_dnn.fc_gate.weight"] = jax.random.normal(k6, (s_dim * 2, fused_dim)) * (1.0 / math.sqrt(fused_dim))
        p["motor_dnn.fc_val.weight"] = jax.random.normal(k7, (s_dim * 2, fused_dim)) * (1.0 / math.sqrt(fused_dim))
        p["motor_dnn.norm_mid.weight"] = jnp.ones(s_dim * 2, dtype=jnp.float32)
        p["motor_dnn.head.weight"] = jax.random.normal(k8, (vocab, s_dim * 2)) * (1.0 / math.sqrt(s_dim * 2))
    else:
        # V3 additions
        fused_dim = s_dim * 5
        inherit = 1.0 - (1.0 / (1.0 + math.exp(-0.80)))
        p["volumetric_core.flow_norm.weight"] = jnp.ones(s_dim, dtype=jnp.float32)
        p["volumetric_core.log_write"] = jnp.full((shells,), math.log(math.expm1(inherit)), dtype=jnp.float32)
        p["volumetric_core.log_leak"] = jnp.full((shells,), math.log(inherit / (1.0 - inherit)), dtype=jnp.float32)
        p["volumetric_core.injection_gate.weight"] = jnp.zeros((s_dim, s_dim * 2), dtype=jnp.float32)
        p["volumetric_core.injection_gate.bias"] = jnp.full((s_dim,), 4.0, dtype=jnp.float32)

        p["gist_scale"] = jnp.zeros((1,), dtype=jnp.float32)
        p["packet_mix"] = jnp.zeros((1,), dtype=jnp.float32)

        p["sensory_dnn.pos_embed.weight"] = jnp.zeros((config.max_len, h_dim), dtype=jnp.float32)
        p["sensory_dnn.conv_local.weight"] = jax.random.normal(k9, (h_dim * 2, h_dim, 3)) * (1.0 / math.sqrt(h_dim * 3))
        p["sensory_dnn.conv_local.bias"] = jnp.zeros((h_dim * 2,), dtype=jnp.float32)
        p["sensory_dnn.conv_wide.weight"] = jax.random.normal(k10, (h_dim * 2, h_dim, 5)) * (1.0 / math.sqrt(h_dim * 5))
        p["sensory_dnn.conv_wide.bias"] = jnp.zeros((h_dim * 2,), dtype=jnp.float32)
        p["sensory_dnn.temporal_norm.weight"] = jnp.ones(h_dim, dtype=jnp.float32)
        p["sensory_dnn.conv_scale"] = jnp.zeros((1,), dtype=jnp.float32)
        p["sensory_dnn.gist_key.weight"] = jax.random.normal(k11, (s_dim, s_dim)) * (1.0 / math.sqrt(s_dim))
        p["sensory_dnn.gist_query"] = jax.random.normal(k12, (s_dim,)) * 0.02

        p["motor_dnn.shell_proj.weight"] = jax.random.normal(k13, (s_dim, s_dim)) * (1.0 / math.sqrt(s_dim))
        p["motor_dnn.shell_score"] = jnp.zeros((shells,), dtype=jnp.float32)
        p["motor_dnn.shell_scale"] = jnp.zeros((1,), dtype=jnp.float32)
        p["motor_dnn.plan_gate.weight"] = jax.random.normal(k1, (s_dim, s_dim * 3)) * (1.0 / math.sqrt(s_dim * 3))
        p["motor_dnn.plan_gate.bias"] = jnp.zeros((s_dim,), dtype=jnp.float32)
        p["motor_dnn.plan_update.weight"] = jax.random.normal(k2, (s_dim, s_dim * 3)) * (1.0 / math.sqrt(s_dim * 3))
        p["motor_dnn.plan_write"] = jnp.zeros((1,), dtype=jnp.float32)
        p["motor_dnn.plan_decay"] = jnp.array(0.9, dtype=jnp.float32)

        p["motor_dnn.fused_norm.weight"] = jnp.ones(fused_dim, dtype=jnp.float32)
        p["motor_dnn.fc_gate.weight"] = jax.random.normal(k3, (s_dim * 2, fused_dim)) * (1.0 / math.sqrt(fused_dim))
        p["motor_dnn.fc_val.weight"] = jax.random.normal(k4, (s_dim * 2, fused_dim)) * (1.0 / math.sqrt(fused_dim))
        p["motor_dnn.norm_mid.weight"] = jnp.ones(s_dim * 2, dtype=jnp.float32)
        p["motor_dnn.head.weight"] = jax.random.normal(k5, (vocab, s_dim * 2)) * (1.0 / math.sqrt(s_dim * 2))

    return p


# =========================================================================
# 2. Stage 1: Afferent Sensory Front-End
# =========================================================================

def encode_sensory(
    params: Dict[str, jnp.ndarray],
    token_ids: jnp.ndarray,
    config: ModelConfig,
    lengths: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Encodes discrete token sequence into continuous Riemannian boundary flux.
    Returns: (sensory_memory [B, L, D], gist [B, D], (keys [B, L, D], values [B, L, D]))
    """
    B, L = token_ids.shape
    s_dim = config.surface_dim

    if config.arch == "v2":
        x = params["sensory_dnn.embedding.weight"][token_ids]
        gate = jax.nn.silu(jnp.matmul(x, params["sensory_dnn.fc1_gate.weight"].T))
        val = jnp.matmul(x, params["sensory_dnn.fc1_val.weight"].T)
        h = rms_norm(gate * val, params["sensory_dnn.norm1.weight"])
        flux = jnp.matmul(h, params["sensory_dnn.fc2.weight"].T) + x
        memory = rms_norm(flux, params["sensory_dnn.norm_out.weight"])
        gist = jnp.mean(memory, axis=1)
    else:
        # V3 Causal Wavelets + Positional Embeddings
        pos = jnp.clip(jnp.arange(L), 0, config.max_len - 1)
        x = params["sensory_dnn.embedding.weight"][token_ids] + params["sensory_dnn.pos_embed.weight"][pos][None, :, :]

        h = jnp.swapaxes(x, 1, 2)
        h_pad2 = jnp.pad(h, ((0, 0), (0, 0), (2, 0)))
        short = lax.conv_general_dilated(
            h_pad2, params["sensory_dnn.conv_local.weight"], (1,), "VALID", dimension_numbers=("NCH", "OIH", "NCH")
        ) + params["sensory_dnn.conv_local.bias"][:, None]

        h_pad4 = jnp.pad(h, ((0, 0), (0, 0), (4, 0)))
        wide = lax.conv_general_dilated(
            h_pad4, params["sensory_dnn.conv_wide.weight"], (1,), "VALID", dimension_numbers=("NCH", "OIH", "NCH")
        ) + params["sensory_dnn.conv_wide.bias"][:, None]

        c_mid = short.shape[1] // 2
        g_short, v_short = short[:, :c_mid, :], short[:, c_mid:, :]
        g_wide, v_wide = wide[:, :c_mid, :], wide[:, c_mid:, :]

        delta = 0.5 * (jax.nn.silu(g_short) * v_short + jax.nn.silu(g_wide) * v_wide)
        delta = rms_norm(jnp.swapaxes(delta, 1, 2), params["sensory_dnn.temporal_norm.weight"])
        trunk_rms = jnp.clip(jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True)), 1e-6, None)
        x = x + params["sensory_dnn.conv_scale"] * delta * trunk_rms

        gate = jax.nn.silu(jnp.matmul(x, params["sensory_dnn.fc1_gate.weight"].T))
        val = jnp.matmul(x, params["sensory_dnn.fc1_val.weight"].T)
        flux = jnp.matmul(rms_norm(gate * val, params["sensory_dnn.norm1.weight"]), params["sensory_dnn.fc2.weight"].T) + x
        memory = rms_norm(flux, params["sensory_dnn.norm_out.weight"])

        scores = jnp.matmul(jnp.matmul(memory, params["sensory_dnn.gist_key.weight"].T), params["sensory_dnn.gist_query"]) / math.sqrt(s_dim)
        if lengths is not None:
            valid = jnp.arange(L)[None, :] < lengths[:, None]
            scores = jnp.where(valid, scores, -1e9)
        gist = jnp.sum(jax.nn.softmax(scores, axis=-1)[:, :, None] * memory, axis=1)

    k = jnp.matmul(memory, params["motor_dnn.k_proj.weight"].T)
    v = jnp.matmul(memory, params["motor_dnn.v_proj.weight"].T)
    return memory, gist, (k, v)


# =========================================================================
# 2. Stage 2: Volumetric Brain Core (B^3 Manifold)
# =========================================================================

def deliberate_step(
    params: Dict[str, jnp.ndarray],
    state: jnp.ndarray,
    config: ModelConfig,
) -> jnp.ndarray:
    """
    One cycle of continuous wave propagation across the 3D spherical core:
    - Intra-shell Laplace-Beltrami diffusion
    - 4D Radial wave transmission
    - White-matter axonal shortcut tracts
    - Contractive transition
    """
    src_idx = params["volumetric_core.src_indices"]
    dst_idx = params["volumetric_core.dst_indices"]

    # 1. Intra-shell tangential diffusion
    intra_flow = jnp.einsum("bkd,kdo->bko", state, params["volumetric_core.intra_shell"])

    # 2. 4D Radial wave diffusion
    radial_compact = jnp.matmul(state, params["volumetric_core.radial_proj_in.weight"].T)
    radial_flow = jnp.einsum("bki,kjio->bjo", radial_compact, params["volumetric_core.reentrant_field"])
    radial_flow = jnp.matmul(radial_flow, params["volumetric_core.radial_proj_out.weight"].T)

    # 3. White-matter axonal shortcut tracts
    src_state = state[:, src_idx, :]
    tract_flow = jnp.einsum("btd,tdo->bto", src_state, params["volumetric_core.shortcut_tracts"])
    shortcut_flow = jnp.zeros_like(state).at[:, dst_idx, :].add(tract_flow)

    velocity = intra_flow + radial_flow + 0.3 * shortcut_flow

    if config.arch == "v2":
        gamma = jax.nn.sigmoid(params["volumetric_core.inertia_gate"])
        next_state = gamma * state + (1.0 - gamma) * jnp.tanh(velocity)
        return rms_norm(next_state, params["volumetric_core.norm.weight"])
    else:
        if config.use_flow_norm:
            velocity = rms_norm(velocity, params["volumetric_core.flow_norm.weight"])
        write = jax.nn.softplus(params["volumetric_core.log_write"])[None, :, None]
        leak = jax.nn.sigmoid(params["volumetric_core.log_leak"])[None, :, None]
        next_state = (1.0 - leak) * state + write * jnp.tanh(velocity)

        # Homeostatic energy regulation
        rms = jnp.clip(jnp.sqrt(jnp.mean(jnp.square(next_state), axis=(1, 2), keepdims=True)), 1e-6, None)
        scale = jnp.clip(config.energy_target / rms, config.energy_clamp[0], config.energy_clamp[1])
        return next_state * scale


def circulate(
    params: Dict[str, jnp.ndarray],
    state: jnp.ndarray,
    config: ModelConfig,
    num_cycles: int = 1,
) -> jnp.ndarray:
    """Circulates internal mid-tensors across all shells without external input."""
    if num_cycles <= 0:
        return state

    def step_fn(s, _):
        return deliberate_step(params, s, config), None

    final_s, _ = lax.scan(step_fn, state, None, length=num_cycles)
    return final_s


def tick_living_dynamics(
    params: Dict[str, jnp.ndarray],
    state: jnp.ndarray,
    config: ModelConfig,
    t_step: int = 0,
    spontaneous_noise: bool = False,
    key: Optional[jax.Array] = None,
) -> jnp.ndarray:
    """Endogenous theta/alpha resting-state pacemaker tick."""
    batch_size, num_shells, shell_dim = state.shape
    depths = jnp.linspace(0.0, 1.0, num_shells)[None, :, None]
    w1, w2 = params["volumetric_core.rhythm_freq"][0], params["volumetric_core.rhythm_freq"][1]
    t_val = float(t_step)
    osc = jnp.sin(w1 * t_val + depths * math.pi) + 0.5 * jnp.cos(w2 * t_val + depths * 2.0 * math.pi)
    rhythm = osc * jnp.clip(params["volumetric_core.spontaneous_amp"], 0.01, 0.5)

    if spontaneous_noise and key is not None:
        noise = jax.random.normal(key, state.shape) * 0.01
        rhythm = rhythm + noise

    perturbed = state + rhythm
    return deliberate_step(params, perturbed, config)


def init_brain_state(
    params: Dict[str, jnp.ndarray],
    config: ModelConfig,
    batch_size: int,
    living: bool = True,
) -> jnp.ndarray:
    """Initializes quiescent or living resting-state manifold tensor [B, K, D]."""
    quiescent = jnp.zeros((batch_size, config.num_shells, config.surface_dim), dtype=jnp.float32)
    if living:
        return tick_living_dynamics(params, quiescent, config, t_step=0, spontaneous_noise=False)
    return quiescent


def inject_sensory_flux(
    params: Dict[str, jnp.ndarray],
    state: jnp.ndarray,
    config: ModelConfig,
    boundary_flux: jnp.ndarray,
) -> jnp.ndarray:
    """Injects sensory excitation into outermost cortical shell (K-1)."""
    if config.arch == "v2":
        new_state = state.at[:, -1, :].add(boundary_flux)
        return rms_norm(new_state, params["volumetric_core.norm.weight"])
    else:
        flux = boundary_flux
        gate_inp = jnp.concatenate([flux, state[:, -1, :]], axis=-1)
        gate = jax.nn.sigmoid(jnp.matmul(gate_inp, params["volumetric_core.injection_gate.weight"].T) + params["volumetric_core.injection_gate.bias"])
        return state.at[:, -1, :].add(gate * flux)


def inject_efference_copy(
    params: Dict[str, jnp.ndarray],
    state: jnp.ndarray,
    config: ModelConfig,
    efference_flux: jnp.ndarray,
) -> jnp.ndarray:
    """Injects motor corollary discharge into deep subcortical shell 0."""
    new_state = state.at[:, 0, :].add(efference_flux)
    if config.arch == "v2":
        return rms_norm(new_state, params["volumetric_core.norm.weight"])
    return new_state


def ingest_stimulus(
    params: Dict[str, jnp.ndarray],
    token_ids: jnp.ndarray,
    config: ModelConfig,
    state: Optional[jnp.ndarray] = None,
    living: bool = True,
    pad_token_id: Optional[int] = None,
    max_packets: int = 32,
) -> Tuple[jnp.ndarray, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Ingests sensory token prompt into the manifold using temporal gamma-wave packets.
    Returns: (final_state, sensory_memory, sensory_kv)
    """
    B, L = token_ids.shape
    if state is None:
        state = init_brain_state(params, config, B, living=living)

    lengths = None
    if pad_token_id is not None:
        lengths = jnp.clip(jnp.sum(token_ids != pad_token_id, axis=-1), 1, None)

    memory, gist, kv = encode_sensory(params, token_ids, config, lengths=lengths)

    packet_size = max(1, (L + max_packets - 1) // max_packets) if L > max_packets else 1
    num_packets = (L + packet_size - 1) // packet_size

    # Loop over packets
    for i in range(num_packets):
        start = i * packet_size
        end = min(L, start + packet_size)
        chunk = memory[:, start:end, :]

        if config.arch == "v2":
            flux = jnp.mean(chunk, axis=1)
        else:
            centroid = jnp.mean(chunk, axis=1)
            tail = chunk[:, -1, :]
            flux = centroid + params["packet_mix"] * (tail - centroid)

        state = inject_sensory_flux(params, state, config, flux)
        state = deliberate_step(params, state, config)

    if config.arch == "v3":
        gist_flux = params["gist_scale"] * gist
        state = inject_efference_copy(params, state, config, gist_flux)

    return state, memory, kv


# =========================================================================
# 3. Stage 3: Efferent Motor Output (Broca's Area)
# =========================================================================

def emit_step(
    params: Dict[str, jnp.ndarray],
    state: jnp.ndarray,
    config: ModelConfig,
    sensory_kv: Tuple[jnp.ndarray, jnp.ndarray],
    refractory_trace: Optional[jnp.ndarray] = None,
    plan: Optional[jnp.ndarray] = None,
    step_idx: int = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    """
    Reads out next-token distribution and advances the manifold internally.
    Returns: (adapted_logits, next_state, next_refractory_trace, next_plan)
    """
    B = state.shape[0]
    s_dim = config.surface_dim
    surface = state[:, -1, :]
    core = state[:, 0, :]

    phase_idx = jnp.clip(step_idx, 0, config.max_phase_steps - 1)
    phase = params["motor_dnn.motor_phase_embed.weight"][phase_idx]
    motor_surface = surface + phase

    k, v = sensory_kv
    q = jnp.expand_dims(jnp.matmul(motor_surface, params["motor_dnn.q_proj.weight"].T), 1)
    scale = 1.0 / math.sqrt(s_dim)
    scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) * scale
    weights = jax.nn.softmax(scores, axis=-1)
    resonance = rms_norm(jnp.squeeze(jnp.matmul(weights, v), 1), params["motor_dnn.resonance_norm.weight"])

    if config.arch == "v2":
        fused = rms_norm(
            jnp.concatenate([motor_surface, core, resonance], axis=-1),
            params["motor_dnn.fused_norm.weight"],
        )
        gate = jax.nn.silu(jnp.matmul(fused, params["motor_dnn.fc_gate.weight"].T))
        val = jnp.matmul(fused, params["motor_dnn.fc_val.weight"].T)
        h = rms_norm(gate * val, params["motor_dnn.norm_mid.weight"])
        next_plan = plan
    else:
        if plan is None:
            plan = jnp.zeros((B, s_dim), dtype=jnp.float32)

        if config.use_shell_tap:
            shell_weights = jax.nn.softmax(params["motor_dnn.shell_score"], axis=0)[None, :, None]
            shell_proj = jnp.matmul(state, params["motor_dnn.shell_proj.weight"].T)
            shell_pool = jnp.sum(shell_weights * shell_proj, axis=1) * params["motor_dnn.shell_scale"]
        else:
            shell_pool = jnp.zeros_like(surface)

        fused = rms_norm(
            jnp.concatenate([motor_surface, core, resonance, shell_pool, plan], axis=-1),
            params["motor_dnn.fused_norm.weight"],
        )
        gate = jax.nn.silu(jnp.matmul(fused, params["motor_dnn.fc_gate.weight"].T))
        val = jnp.matmul(fused, params["motor_dnn.fc_val.weight"].T)
        h = rms_norm(gate * val, params["motor_dnn.norm_mid.weight"])

        # Premotor plan recurrent buffer
        if config.use_plan:
            phase_batch = jnp.repeat(phase[None, :], B, axis=0)
            plan_input = jnp.concatenate([h, phase_batch], axis=-1)
            p_gate = jax.nn.sigmoid(jnp.matmul(plan_input, params["motor_dnn.plan_gate.weight"].T) + params["motor_dnn.plan_gate.bias"])
            next_plan = params["motor_dnn.plan_decay"] * plan + params["motor_dnn.plan_write"] * p_gate * jnp.matmul(plan_input, params["motor_dnn.plan_update.weight"].T)
        else:
            next_plan = plan

    temp = jnp.clip(params["motor_dnn.temperature_scale"], 0.1, 5.0)
    raw_logits = jnp.matmul(h, params["motor_dnn.head.weight"].T) / temp
    efference_copy = jnp.matmul(h, params["motor_dnn.efference_proj.weight"].T)

    if refractory_trace is not None:
        alpha = jnp.clip(params["motor_dnn.refractory_strength"], 0.1, 5.0)
        adapted_logits = raw_logits - alpha * refractory_trace
        decay = jax.nn.sigmoid(params["motor_dnn.refractory_decay"])
        current_prob = lax.stop_gradient(jax.nn.softmax(raw_logits, axis=-1))
        next_trace = lax.stop_gradient(decay * refractory_trace + (1.0 - decay) * current_prob)
    else:
        adapted_logits = raw_logits
        next_trace = None

    # Internal corollary discharge into shell 0
    state_eff = inject_efference_copy(params, state, config, efference_copy)
    next_state = deliberate_step(params, state_eff, config)

    return adapted_logits, next_state, next_trace, next_plan


# =========================================================================
# 4. Zero-Feedback Rollout & Objective
# =========================================================================

def training_rollout(
    params: Dict[str, jnp.ndarray],
    config: ModelConfig,
    prompts: jnp.ndarray,
    response_steps: int,
    thinking_cycles: int = 3,
    pad_token_id: Optional[int] = None,
) -> jnp.ndarray:
    """
    Teacher-Free Autonomous Rollout using jax.lax.scan:
    Returns: [Batch, response_steps, vocab_size] logits.
    """
    B = prompts.shape[0]
    vocab = config.vocab_size

    state, _, kv = ingest_stimulus(
        params, prompts, config, living=True, pad_token_id=pad_token_id
    )
    state = circulate(params, state, config, num_cycles=thinking_cycles)

    refractory = jnp.zeros((B, vocab), dtype=jnp.float32)
    plan = jnp.zeros((B, config.surface_dim), dtype=jnp.float32) if config.arch == "v3" else None

    def scan_fn(carry, step_idx):
        s, r, p = carry
        logits, next_s, next_r, next_p = emit_step(
            params, s, config, kv, refractory_trace=r, plan=p, step_idx=step_idx
        )
        return (next_s, next_r, next_p), logits

    init_carry = (state, refractory, plan)
    _, stacked_logits = lax.scan(scan_fn, init_carry, jnp.arange(response_steps))
    # Shape from scan is [response_steps, B, vocab] -> transpose to [B, response_steps, vocab]
    return jnp.swapaxes(stacked_logits, 0, 1)


def compute_loss(
    params: Dict[str, jnp.ndarray],
    config: ModelConfig,
    prompts: jnp.ndarray,
    targets: jnp.ndarray,
    thinking_cycles: int = 3,
    pad_token_id: Optional[int] = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Cross-entropy loss over token-weighted response horizon."""
    response_steps = targets.shape[1]
    logits = training_rollout(
        params, config, prompts, response_steps=response_steps,
        thinking_cycles=thinking_cycles, pad_token_id=pad_token_id
    )

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(targets, config.vocab_size)
    nll = -jnp.sum(one_hot * log_probs, axis=-1)  # [B, response_steps]

    if pad_token_id is not None:
        mask = (targets != pad_token_id).astype(jnp.float32)
        valid_count = jnp.clip(jnp.sum(mask), 1.0, None)
        loss = jnp.sum(nll * mask) / valid_count
    else:
        loss = jnp.mean(nll)

    metrics = {"loss": loss, "perplexity": jnp.exp(loss)}
    return loss, metrics


# =========================================================================
# 5. Autonomous Deliberate-Then-Emit Generation
# =========================================================================

def apply_repetition_penalty_jax(
    logits: jnp.ndarray,
    emitted: List[jnp.ndarray],
    penalty: float,
) -> jnp.ndarray:
    """Batch-safe multiplicative repetition penalty matching PyTorch apply_repetition_penalty."""
    for tok in emitted:
        idx = tok.squeeze(-1)  # [B]
        current = jnp.take_along_axis(logits, idx[:, None], axis=1).squeeze(1)
        penalised = jnp.where(current > 0, current / penalty, current * penalty)
        logits = logits.at[jnp.arange(logits.shape[0]), idx].set(penalised)
    return logits


def generate_zero_feedback(
    params: Dict[str, jnp.ndarray],
    config: ModelConfig,
    prompt_tokens: jnp.ndarray,
    thinking_cycles: int = 4,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    repetition_penalty: float = 1.25,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    min_new_tokens: int = 3,
    key: Optional[jax.Array] = None,
) -> jnp.ndarray:
    """
    Deliberate-then-emit generation with zero external output feedback.
    Supports greedy argmax or temperature sampling with repetition penalty.
    """
    B = prompt_tokens.shape[0]
    vocab = config.vocab_size

    state, _, kv = ingest_stimulus(
        params, prompt_tokens, config, living=True, pad_token_id=pad_token_id
    )
    state = circulate(params, state, config, num_cycles=thinking_cycles)

    refractory = jnp.zeros((B, vocab), dtype=jnp.float32)
    plan = jnp.zeros((B, config.surface_dim), dtype=jnp.float32) if config.arch == "v3" else None

    emitted: List[jnp.ndarray] = []
    finished = jnp.zeros((B,), dtype=jnp.bool_)
    rng = key if key is not None else jax.random.PRNGKey(0)

    for step in range(max_new_tokens):
        logits, state, refractory, plan = emit_step(
            params, state, config, kv, refractory_trace=refractory, plan=plan, step_idx=step
        )

        if eos_token_id is not None and step < max(min_new_tokens, 1):
            logits = logits.at[:, eos_token_id].set(-1e4)

        if repetition_penalty > 0 and emitted:
            if config.arch == "v2":
                for tok in emitted:
                    logits = logits.at[jnp.arange(B), tok.squeeze(-1)].add(-repetition_penalty)
            elif repetition_penalty > 1.0:
                logits = apply_repetition_penalty_jax(logits, emitted, repetition_penalty)

        if temperature <= 0.05:
            next_tok = jnp.argmax(logits, axis=-1, keepdims=True)
        else:
            rng, step_key = jax.random.split(rng)
            scaled = logits / max(temperature, 1e-5)
            next_tok = jax.random.categorical(step_key, scaled, axis=-1)[:, None]

        if eos_token_id is not None:
            is_eos = (next_tok.squeeze(-1) == eos_token_id)
            if bool(jnp.all(is_eos)):
                break

        emitted.append(next_tok)

    if not emitted:
        return jnp.zeros((B, 0), dtype=jnp.int32)
    return jnp.concatenate(emitted, axis=-1)


# =========================================================================
# 6. Object-Oriented Interface: CerebroSpheroidJAX
# =========================================================================

class CerebroSpheroidJAX:
    """
    Unified CerebroSpheroid Brain Model in JAX.
    Provides identical public API to PyTorch CerebroSpheroid / CerebroSpheroidV3.
    """
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        params: Optional[Dict[str, jnp.ndarray]] = None,
        key: Optional[jax.Array] = None,
        **kwargs,
    ):
        if config is None:
            if kwargs:
                self.config = ModelConfig(**kwargs)
            else:
                self.config = ModelConfig.from_scale("laptop", arch="v2")
        else:
            self.config = config

        self.vocab_size = self.config.vocab_size
        self.surface_dim = self.config.surface_dim
        self.num_shells = self.config.num_shells
        self.arch = self.config.arch

        if params is not None:
            self.params = params
        else:
            self.params = init_params(self.config, key=key)

    def init_brain_state(self, batch_size: int, living: bool = True) -> jnp.ndarray:
        return init_brain_state(self.params, self.config, batch_size, living=living)

    def tick_living_dynamics(self, state: jnp.ndarray, t_step: int = 0) -> jnp.ndarray:
        return tick_living_dynamics(self.params, state, self.config, t_step=t_step)

    def ingest_stimulus(
        self,
        token_ids: jnp.ndarray,
        state: Optional[jnp.ndarray] = None,
        living: bool = True,
        return_memory: bool = False,
        pad_token_id: Optional[int] = None,
    ):
        final_state, mem, _ = ingest_stimulus(
            self.params, token_ids, self.config, state=state, living=living, pad_token_id=pad_token_id
        )
        if return_memory:
            return final_state, mem
        return final_state

    def ingest_stimulus_with_memory(
        self,
        token_ids: jnp.ndarray,
        state: Optional[jnp.ndarray] = None,
        living: bool = True,
        pad_token_id: Optional[int] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.ingest_stimulus(
            token_ids, state=state, living=living, return_memory=True, pad_token_id=pad_token_id
        )

    def deliberate(self, state: jnp.ndarray, thinking_cycles: int = 4) -> jnp.ndarray:
        return circulate(self.params, state, self.config, num_cycles=thinking_cycles)

    def emit_next_token(
        self,
        state: jnp.ndarray,
        sensory_memory: Optional[jnp.ndarray] = None,
        refractory_trace: Optional[jnp.ndarray] = None,
        return_trace: bool = False,
        step_idx: int = 0,
    ):
        if sensory_memory is not None:
            k = jnp.matmul(sensory_memory, self.params["motor_dnn.k_proj.weight"].T)
            v = jnp.matmul(sensory_memory, self.params["motor_dnn.v_proj.weight"].T)
            kv = (k, v)
        else:
            kv = (jnp.zeros_like(state[:, :1, :]), jnp.zeros_like(state[:, :1, :]))

        logits, next_state, next_trace, _ = emit_step(
            self.params, state, self.config, kv, refractory_trace=refractory_trace, step_idx=step_idx
        )
        if return_trace or refractory_trace is not None:
            return logits, next_state, next_trace
        return logits, next_state

    def training_logits(
        self,
        prompts: jnp.ndarray,
        response_steps: int,
        thinking_cycles: int = 3,
        pad_token_id: Optional[int] = None,
    ) -> jnp.ndarray:
        return training_rollout(
            self.params, self.config, prompts, response_steps=response_steps,
            thinking_cycles=thinking_cycles, pad_token_id=pad_token_id
        )

    def generate(
        self,
        prompt_tokens: jnp.ndarray,
        thinking_cycles: int = 4,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        repetition_penalty: float = 1.25,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        min_new_tokens: int = 3,
        key: Optional[jax.Array] = None,
    ) -> jnp.ndarray:
        return generate_zero_feedback(
            self.params, self.config, prompt_tokens, thinking_cycles=thinking_cycles,
            max_new_tokens=max_new_tokens, temperature=temperature,
            repetition_penalty=repetition_penalty, eos_token_id=eos_token_id,
            pad_token_id=pad_token_id, min_new_tokens=min_new_tokens, key=key
        )

    def solve_with_active_thinking(
        self,
        prompt_tokens: jnp.ndarray,
        thinking_cycles: int = 4,
        max_answer_tokens: Optional[int] = None,
        temperature: float = 0.0,
        repetition_penalty: float = 1.25,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> jnp.ndarray:
        return self.generate(
            prompt_tokens,
            thinking_cycles=thinking_cycles,
            max_new_tokens=max_answer_tokens or 64,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
