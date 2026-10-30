"""
CerebroSpheroid JAX Package:
Non-Autoregressive Volumetric Reasoning Architecture on JAX.
"""

# V4 Architecture
from .v4_model import (
    ModelConfigV4,
    PRESETS_V4,
    rms_norm as rms_norm_v4,
    init_v4_params,
    precompute_axonal_indices,
    encode_sensory,
    deliberate_step,
    emit_step,
    training_rollout as training_rollout_v4,
)

# SpheroidOptimizer (Newton-Schulz + Spectral Guard)
from .spheroid_optimizer import (
    zeropower_via_newtonschulz5_jax,
    clamp_spectral_norm_jax,
    is_manifold_operator,
    init_spheroid_opt_state,
    build_train_step_spheroid,
)

# Checkpointing & Synchronization
from .checkpoint import (
    save_jax_to_torch_pt,
    load_torch_pt_to_jax,
    discover_latest_checkpoint,
    AsyncCheckpointManager,
)

# V3 Legacy Architecture
from .model import (
    ModelConfig,
    CerebroSpheroidJAX,
    rms_norm,
    init_params,
    init_brain_state,
    training_rollout,
    compute_loss,
    generate_zero_feedback,
)
from .convert import (
    torch_to_jax_params,
    load_torch_checkpoint_to_jax,
    save_jax_checkpoint,
    load_jax_checkpoint,
    detect_checkpoint_architecture,
)
from .inference import ReasoningAgentJAX

__all__ = [
    # V4
    "ModelConfigV4",
    "PRESETS_V4",
    "init_v4_params",
    "precompute_axonal_indices",
    "training_rollout_v4",
    # Optimizer
    "zeropower_via_newtonschulz5_jax",
    "clamp_spectral_norm_jax",
    "is_manifold_operator",
    "init_spheroid_opt_state",
    "build_train_step_spheroid",
    # Checkpoint
    "save_jax_to_torch_pt",
    "load_torch_pt_to_jax",
    "discover_latest_checkpoint",
    "AsyncCheckpointManager",
    # V3
    "ModelConfig",
    "CerebroSpheroidJAX",
    "ReasoningAgentJAX",
    "init_params",
    "training_rollout",
    "compute_loss",
    "generate_zero_feedback",
    "torch_to_jax_params",
    "load_torch_checkpoint_to_jax",
    "save_jax_checkpoint",
    "load_jax_checkpoint",
    "detect_checkpoint_architecture",
]
