"""
CerebroSpheroid Reasoning Agent Package
"""

from .model import (
    SensoryInputDNN,
    VolumetricBrainCore,
    MotorOutputDNN,
    CerebroSpheroid,
    build_cerebro_model,
)
from .spheroid_v3 import (
    TemporalSensoryEncoder,
    ResidualVolumetricCore,
    PremotorMotorDecoder,
    CerebroSpheroidV3,
    build_spheroid_v3,
    training_rollout,
)
from .spheroid_v4 import (
    PremotorMotorDecoderV4,
    CerebroSpheroidV4,
    build_spheroid_v4,
    build_v4_model,
    compute_manifold_diagnostics,
    get_shell_zones,
    lift_checkpoint_v3_to_v4,
)
from .sensory_nodes import (
    VisionSensoryEncoder,
    AudioSensoryEncoder,
    ThalamicSensoryGater,
)
from .spheroid_optimizer import (
    SpheroidOptimizer,
    build_spheroid_optimizer,
    create_spheroid_param_groups,
)

__all__ = [
    "SensoryInputDNN",
    "VolumetricBrainCore",
    "MotorOutputDNN",
    "CerebroSpheroid",
    "build_cerebro_model",
    "TemporalSensoryEncoder",
    "ResidualVolumetricCore",
    "PremotorMotorDecoder",
    "CerebroSpheroidV3",
    "build_spheroid_v3",
    "training_rollout",
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

# Lazy-load JAX submodule to prevent expensive GPU device initialization on import
_JAX_ATTRS = {
    "ModelConfigJAX": "ModelConfig",
    "CerebroSpheroidJAX": "CerebroSpheroidJAX",
    "ReasoningAgentJAX": "ReasoningAgentJAX",
    "training_rollout_jax": "training_rollout",
    "compute_loss_jax": "compute_loss",
    "load_torch_checkpoint_to_jax": "load_torch_checkpoint_to_jax",
    "save_jax_checkpoint": "save_jax_checkpoint",
    "load_jax_checkpoint": "load_jax_checkpoint",
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


