"""
CerebroSpheroid Reasoning Agent Package (V4 Architecture)
"""

# Kaggle / Python 3.12 compatibility shim: Newer pyarrow (>=19) removed pa.PyExtensionType,
# which causes older HuggingFace datasets (<3.0.0) to raise AttributeError
try:
    import pyarrow as pa
    if not hasattr(pa, "PyExtensionType") and hasattr(pa, "ExtensionType"):
        pa.PyExtensionType = pa.ExtensionType
except Exception:
    pass

# Framework-agnostic utilities
from .corpus import (
    build_tokenizer,
    format_stimulus_prompt,
    format_motor_response,
    clean_motor_response,
)
from .elastic_window import (
    get_default_elastic_buckets,
    select_best_bucket,
    ElasticBucketBatcher,
)
from .cloud_sync import (
    check_remote_manifest,
    pull_checkpoint,
    push_checkpoint,
    push_checkpoint_async,
    auto_sync_checkpoints,
)

__all__ = [
    "build_tokenizer",
    "format_stimulus_prompt",
    "format_motor_response",
    "clean_motor_response",
    "get_default_elastic_buckets",
    "select_best_bucket",
    "ElasticBucketBatcher",
    "check_remote_manifest",
    "pull_checkpoint",
    "push_checkpoint",
    "push_checkpoint_async",
    "auto_sync_checkpoints",
]

# PyTorch V4 Architecture (Guarded to prevent import failure if torch has issues or in pure JAX)
try:
    from .spheroid_v4 import (
        PremotorMotorDecoderV4,
        CerebroSpheroidV4,
        build_spheroid_v4,
        build_v4_model,
        compute_manifold_diagnostics,
        get_shell_zones,
        lift_checkpoint_v3_to_v4,
    )
    from .spheroid_optimizer import (
        SpheroidOptimizer,
        build_spheroid_optimizer,
        create_spheroid_param_groups,
    )
    from .sensory_nodes import (
        VisionSensoryEncoder,
        AudioSensoryEncoder,
        ThalamicSensoryGater,
    )

    __all__ += [
        "PremotorMotorDecoderV4",
        "CerebroSpheroidV4",
        "build_spheroid_v4",
        "build_v4_model",
        "compute_manifold_diagnostics",
        "get_shell_zones",
        "lift_checkpoint_v3_to_v4",
        "VisionSensoryEncoder",
        "AudioSensoryEncoder",
        "ThalamicSensoryGater",
        "SpheroidOptimizer",
        "build_spheroid_optimizer",
        "create_spheroid_param_groups",
    ]
    HAS_TORCH = True
except (ImportError, SystemError, Exception) as _torch_err:
    HAS_TORCH = False
    _TORCH_IMPORT_ERROR = str(_torch_err)

# Lazy-load JAX submodule to prevent expensive TPU/GPU device initialization on import
_JAX_ATTRS = {
    "ModelConfigV4": "ModelConfigV4",
    "PRESETS_V4": "PRESETS_V4",
    "init_v4_params": "init_v4_params",
    "init_spheroid_opt_state": "init_spheroid_opt_state",
    "build_spheroid_train_engine": "build_spheroid_train_engine",
    "build_train_step_spheroid": "build_train_step_spheroid",
    "load_torch_pt_to_jax": "load_torch_pt_to_jax",
    "save_jax_to_torch_pt": "save_jax_to_torch_pt",
    "discover_latest_checkpoint": "discover_latest_checkpoint",
    "AsyncCheckpointManager": "AsyncCheckpointManager",
    "training_rollout_v4": "training_rollout_v4",
    "generate_v4": "generate_v4",
}

__all__ += list(_JAX_ATTRS.keys())


def __getattr__(name):
    if name in _JAX_ATTRS:
        try:
            from . import jax as _jax
            target_name = _JAX_ATTRS[name]
            return getattr(_jax, target_name)
        except ImportError as e:
            raise AttributeError(f"JAX components not available: {e}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
