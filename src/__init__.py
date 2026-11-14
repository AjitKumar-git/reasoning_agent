"""
CerebroSpheroid Reasoning Agent Package (V5 Architecture)
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

# PyTorch V5 Architecture
try:
    from .spheroid_v5 import (
        CerebroSpheroidV5,
        ModelConfigV5,
        PRESETS_V5,
        build_spheroid_v5,
        compute_manifold_diagnostics,
    )
    from .spheroid_optimizer import (
        SpheroidOptimizer,
        build_spheroid_optimizer,
        create_spheroid_param_groups,
    )
    from .emergence_detector import (
        AGIEmergenceAuditor,
        EmergenceReport,
    )

    __all__ += [
        "CerebroSpheroidV5",
        "ModelConfigV5",
        "PRESETS_V5",
        "build_spheroid_v5",
        "compute_manifold_diagnostics",
        "SpheroidOptimizer",
        "build_spheroid_optimizer",
        "create_spheroid_param_groups",
        "AGIEmergenceAuditor",
        "EmergenceReport",
    ]

    HAS_TORCH = True
except (ImportError, SystemError, Exception) as _torch_err:
    HAS_TORCH = False
    _TORCH_IMPORT_ERROR = str(_torch_err)

# Lazy-load JAX submodule to prevent expensive TPU/GPU device initialization on import
_JAX_ATTRS = {
    "ModelConfigV5": "ModelConfigV5",
    "PRESETS_V5": "PRESETS_V5",
    "init_params_v5": "init_params_v5",
    "generate_small_world_indices": "generate_small_world_indices",
    "encode_sensory": "encode_sensory",
    "deliberate_step": "deliberate_step",
    "emit_step": "emit_step",
    "training_rollout": "training_rollout",
    "generate_text_jax": "generate_text_jax",
    "init_spheroid_opt_state": "init_spheroid_opt_state",
    "build_spheroid_train_engine": "build_spheroid_train_engine",
    "build_train_step_spheroid": "build_train_step_spheroid",
    "load_torch_pt_to_jax": "load_torch_pt_to_jax",
    "save_jax_to_torch_pt": "save_jax_to_torch_pt",
    "discover_latest_checkpoint": "discover_latest_checkpoint",
    "AsyncCheckpointManager": "AsyncCheckpointManager",
    "LinguisticDataStreamer": "LinguisticDataStreamer",
    "run_auto_tuner": "run_auto_tuner",
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
