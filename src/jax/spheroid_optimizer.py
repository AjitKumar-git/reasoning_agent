"""
SpheroidOptimizer (Pure JAX Implementation):
Manifold & Spectral-Aware Optimizer for CerebroSpheroid Architectures on Cloud TPUs and GPUs.

Architecture Parity with PyTorch / GPU SpheroidOptimizer:
1. Manifold Matrix Operators (rank >= 2):
   Updated along nearest orthogonal manifold via Quintic Newton-Schulz iterations (Muon style).
   Preserves continuous Riemannian rotational isometry without AdamW coordinate-wise distortion.
   Maintains a single momentum buffer (50% memory saving).
2. Recurrent Manifold Core (intra_shell, reentrant_field, shortcut_tracts):
   Orthogonal polar updates with persistent power-iteration contractive spectral clamp (sigma_max <= 1.0).
3. Large Vocabulary Tables & Output Heads:
   Optimized with AdamW (first and second moments) to gracefully handle sparse tokens and frequency skew.
4. 1D Parameters (Scalars, Pacemakers, Biases, RMSNorm weights):
   Optimized with AdamW (zero weight decay) to handle heterogeneous parameter scales stably.
"""

from typing import Dict, Tuple, Set, Optional, Callable, Any
import math

import jax
import jax.numpy as jnp
from jax import lax

from .v4_model import ModelConfigV4, training_rollout


RECURRENT_KEYS: Set[str] = {"core.intra_shell", "core.reentrant_field", "core.shortcut_tracts"}
VOCAB_KEYS: Set[str] = {"sensory.embedding.weight", "motor.head.weight"}


def is_manifold_operator(k: str, p: jnp.ndarray) -> bool:
    """Returns True for 2D, 3D, and 4D operator matrices updated via Polar Newton-Schulz."""
    if k in VOCAB_KEYS:
        return False
    return p.ndim >= 2


def zeropower_via_newtonschulz5_jax(G: jnp.ndarray, steps: int = 3, eps: float = 1e-7) -> jnp.ndarray:
    """Vectorized Quintic Newton-Schulz iteration mapping update matrices to the orthogonal manifold."""
    orig_shape = G.shape
    if G.ndim > 2:
        G_mat = G.reshape(-1, orig_shape[-2], orig_shape[-1])
    else:
        G_mat = G[None, ...]

    rows, cols = G_mat.shape[-2], G_mat.shape[-1]
    transposed = rows > cols
    if transposed:
        G_mat = jnp.swapaxes(G_mat, -2, -1)

    norm = jnp.sqrt(jnp.sum(jnp.square(G_mat), axis=(-2, -1), keepdims=True) + eps)
    X = G_mat / norm

    # Quintic Newton-Schulz polynomial coefficients
    a, b, c = 3.4445, -4.7750, 2.0315
    def _ns_body(i, cur_x):
        A = jnp.matmul(cur_x, jnp.swapaxes(cur_x, -2, -1))
        B = b * A + c * jnp.matmul(A, A)
        return a * cur_x + jnp.matmul(B, cur_x)

    X = lax.fori_loop(0, steps, _ns_body, X)

    if transposed:
        X = jnp.swapaxes(X, -2, -1)
    return X.reshape(orig_shape)


def clamp_spectral_norm_jax(
    W: jnp.ndarray, v: jnp.ndarray, max_spec_norm: float = 1.0, power_steps: int = 2, eps: float = 1e-6
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Contractive spectral guard via persistent power iteration. Enforces sigma_max <= 1.0."""
    orig_shape = W.shape
    if W.ndim > 3:
        W_mat = W.reshape(-1, orig_shape[-2], orig_shape[-1])
    elif W.ndim == 2:
        W_mat = W[None, ...]
    else:
        W_mat = W

    def _power_body(i, cur_v):
        u = jnp.matmul(W_mat, cur_v)
        u_norm = jnp.sqrt(jnp.sum(jnp.square(u), axis=1, keepdims=True) + eps)
        u = u / u_norm
        v_next = jnp.matmul(jnp.swapaxes(W_mat, -2, -1), u)
        v_norm = jnp.sqrt(jnp.sum(jnp.square(v_next), axis=1, keepdims=True) + eps)
        return v_next / v_norm

    v_final = lax.fori_loop(0, power_steps, _power_body, v)
    u_final = jnp.matmul(W_mat, v_final)
    sigma = jnp.sqrt(jnp.sum(jnp.square(u_final), axis=1, keepdims=True) + eps)
    scale = jnp.minimum(1.0, max_spec_norm / (sigma * 1.08 + eps))
    W_clamped = (W_mat * scale).reshape(orig_shape)
    return W_clamped, v_final


def init_spheroid_opt_state(params: Dict[str, jnp.ndarray], key: Optional[jax.Array] = None) -> Dict[str, Any]:
    """Initializes single-moment buffers for matrices, AdamW buffers for vocab/1D, and power vectors."""
    if key is None:
        key = jax.random.PRNGKey(42)

    opt_m: Dict[str, jnp.ndarray] = {}
    opt_v: Dict[str, jnp.ndarray] = {}
    power_v: Dict[str, jnp.ndarray] = {}

    for k, p in params.items():
        if not jnp.issubdtype(p.dtype, jnp.inexact):
            continue
        opt_m[k] = jnp.zeros_like(p)
        if not is_manifold_operator(k, p):
            opt_v[k] = jnp.zeros_like(p)
        if k in RECURRENT_KEYS:
            key, subkey = jax.random.split(key)
            orig_shape = p.shape
            mat_b = orig_shape[0] if p.ndim == 3 else (orig_shape[0] * orig_shape[1] if p.ndim == 4 else 1)
            mat_n = orig_shape[-1]
            v_init = jax.random.normal(subkey, (mat_b, mat_n, 1), dtype=jnp.float32)
            v_init = v_init / jnp.sqrt(jnp.sum(jnp.square(v_init), axis=1, keepdims=True) + 1e-6)
            power_v[k] = v_init

    return {
        "step": jnp.array(0, dtype=jnp.int32),
        "m": opt_m,
        "v": opt_v,
        "power_v": power_v,
    }


def build_train_step_spheroid(
    config: ModelConfigV4,
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    pad_id: int,
    weight_decay: float = 0.01,
) -> Callable:
    """Builds a JIT-compiled train_step function utilizing SpheroidOptimizer."""

    @jax.jit
    def train_step(p, opt_s, q_batch, r_batch, base_lr):
        target_len = r_batch.shape[1]

        def _loss_fn(param_dict):
            logits = training_rollout(param_dict, q_batch, target_len, buffers, config)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            # Memory optimization: Gather target log_probs directly without allocating 2.1 GB one-hot tensor
            safe_targets = jnp.maximum(r_batch, 0)
            target_log_probs = jnp.take_along_axis(log_probs, safe_targets[..., None], axis=-1).squeeze(-1)
            token_loss = -target_log_probs
            mask = ((r_batch != pad_id) & (r_batch >= 0)).astype(jnp.float32)
            return jnp.sum(token_loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)

        loss_val, grads = jax.value_and_grad(_loss_fn, allow_int=True)(p)

        # Gradient clipping
        grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in grads.values() if jnp.issubdtype(g.dtype, jnp.inexact)))
        clip_scale = jnp.minimum(1.0, 1.0 / (grad_norm + 1e-6))
        grads = jax.tree_util.tree_map(lambda g: g * clip_scale if jnp.issubdtype(g.dtype, jnp.inexact) else g, grads)

        cur_step = opt_s["step"] + 1
        new_params = {}
        new_m = {}
        new_v = {}
        new_power_v = dict(opt_s["power_v"])

        lr_1d = 0.2 * base_lr
        lr_vocab = jnp.minimum(base_lr, 6.0e-4)

        for k, param in p.items():
            if not jnp.issubdtype(param.dtype, jnp.inexact):
                new_params[k] = param
                continue

            g = grads[k]
            m_prev = opt_s["m"][k]

            # -----------------------------------------------------------------
            # 1. Manifold Matrix Operators (Quintic Newton-Schulz Polar Update)
            # -----------------------------------------------------------------
            if is_manifold_operator(k, param):
                m_new = 0.95 * m_prev + 0.05 * g
                new_m[k] = m_new

                # Decoupled Riemannian Weight Decay
                p_decayed = param * (1.0 - base_lr * weight_decay)

                # Polar orthogonalization
                O = zeropower_via_newtonschulz5_jax(m_new, steps=3)

                # Natural aspect-ratio scaling
                rows = param.shape[-2]
                cols = param.shape[-1]
                aspect_scale = math.sqrt(max(1.0, float(rows) / float(cols)))

                # Update along orthogonal manifold
                p_updated = p_decayed - (base_lr * aspect_scale) * O

                # Recurrent Manifold Contractive Spectral Guard (rho <= 1.0)
                if k in RECURRENT_KEYS:
                    p_updated, v_next = clamp_spectral_norm_jax(p_updated, opt_s["power_v"][k], max_spec_norm=1.0)
                    new_power_v[k] = v_next

                new_params[k] = p_updated

            # -----------------------------------------------------------------
            # 2. Vocabulary Embeddings & Heads (AdamW)
            # -----------------------------------------------------------------
            elif k in VOCAB_KEYS:
                v_prev = opt_s["v"][k]
                m_new = 0.9 * m_prev + 0.1 * g
                v_new = 0.95 * v_prev + 0.05 * jnp.square(g)
                new_m[k] = m_new
                new_v[k] = v_new

                m_hat = m_new / (1.0 - jnp.power(0.9, cur_step))
                v_hat = v_new / (1.0 - jnp.power(0.95, cur_step))
                p_decayed = param * (1.0 - lr_vocab * weight_decay)
                new_params[k] = p_decayed - lr_vocab * (m_hat / (jnp.sqrt(v_hat) + 1e-8))

            # -----------------------------------------------------------------
            # 3. 1D Parameters, Scalars, Pacemakers & Norms (AdamW, 0 Decay)
            # -----------------------------------------------------------------
            else:
                v_prev = opt_s["v"][k]
                m_new = 0.9 * m_prev + 0.1 * g
                v_new = 0.95 * v_prev + 0.05 * jnp.square(g)
                new_m[k] = m_new
                new_v[k] = v_new

                m_hat = m_new / (1.0 - jnp.power(0.9, cur_step))
                v_hat = v_new / (1.0 - jnp.power(0.95, cur_step))
                new_params[k] = param - lr_1d * (m_hat / (jnp.sqrt(v_hat) + 1e-8))

        new_opt_state = {
            "step": cur_step,
            "m": new_m,
            "v": new_v,
            "power_v": new_power_v,
        }
        return new_params, new_opt_state, loss_val

    return train_step
