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
]

try:
    from .jax import (
        ModelConfig as ModelConfigJAX,
        CerebroSpheroidJAX,
        ReasoningAgentJAX,
        training_rollout as training_rollout_jax,
        compute_loss as compute_loss_jax,
        load_torch_checkpoint_to_jax,
        save_jax_checkpoint,
        load_jax_checkpoint,
    )
    __all__ += [
        "ModelConfigJAX",
        "CerebroSpheroidJAX",
        "ReasoningAgentJAX",
        "training_rollout_jax",
        "compute_loss_jax",
        "load_torch_checkpoint_to_jax",
        "save_jax_checkpoint",
        "load_jax_checkpoint",
    ]
except ImportError:
    pass

