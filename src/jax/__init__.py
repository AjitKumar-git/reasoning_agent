"""
CerebroSpheroid JAX Package:
Non-Autoregressive Volumetric Reasoning Architecture on JAX.
"""

from .model import (
    ModelConfig,
    CerebroSpheroidJAX,
    rms_norm,
    init_params,
    init_brain_state,
    deliberate_step,
    circulate,
    tick_living_dynamics,
    inject_sensory_flux,
    inject_efference_copy,
    encode_sensory,
    ingest_stimulus,
    emit_step,
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
    "ModelConfig",
    "CerebroSpheroidJAX",
    "ReasoningAgentJAX",
    "rms_norm",
    "init_params",
    "init_brain_state",
    "deliberate_step",
    "circulate",
    "tick_living_dynamics",
    "inject_sensory_flux",
    "inject_efference_copy",
    "encode_sensory",
    "ingest_stimulus",
    "emit_step",
    "training_rollout",
    "compute_loss",
    "generate_zero_feedback",
    "torch_to_jax_params",
    "load_torch_checkpoint_to_jax",
    "save_jax_checkpoint",
    "load_jax_checkpoint",
    "detect_checkpoint_architecture",
]
