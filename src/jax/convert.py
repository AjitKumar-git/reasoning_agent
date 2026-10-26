"""
PyTorch <-> JAX Checkpoint and State Conversion Utilities.

Supports:
1. Converting PyTorch state_dict into JAX parameter dict (pure jnp.ndarray).
2. Loading `.pt` checkpoints directly into JAX parameters with architecture auto-detection.
3. V2-to-V3 width-lifting surgery (lossless transplantation of motor tensors & inertia gate).
4. Saving and loading standalone JAX `.npz` checkpoints.
"""

import os
import math
from typing import Dict, Tuple, Optional, Any
import numpy as np

import jax
import jax.numpy as jnp
import torch

from .model import ModelConfig, init_params


def detect_checkpoint_architecture(state_dict: Dict[str, Any]) -> str:
    """Inspects tensor keys to identify whether a checkpoint was trained on V3 or V2."""
    keys = {k.replace("module.", "") for k in state_dict.keys()}
    v3_keys = (
        "sensory_dnn.conv_scale",
        "sensory_dnn.pos_embed.weight",
        "motor_dnn.plan_write",
        "motor_dnn.shell_scale",
        "volumetric_core.log_write",
    )
    return "v3" if any(k in keys for k in v3_keys) else "v2"


def lift_v2_to_v3_tensor(key: str, value: np.ndarray, target_shape: Tuple[int, ...]) -> Optional[np.ndarray]:
    """
    Performs exact V2 -> V3 width-lifting surgery:
    - Phase clock: repeat last beat for extra slots
    - Motor SwiGLU projections: zero-pad extra channels
    - Fused RMSNorm: rescale gain by sqrt(d_old / d_new) so pre-activations remain invariant
    """
    if key == "motor_dnn.motor_phase_embed.weight":
        if value.shape[1] != target_shape[1]:
            return None
        lifted = np.zeros(target_shape, dtype=value.dtype)
        rows = min(value.shape[0], target_shape[0])
        lifted[:rows] = value[:rows]
        if target_shape[0] > rows:
            lifted[rows:] = value[rows - 1]
        return lifted

    if key in ("motor_dnn.fc_gate.weight", "motor_dnn.fc_val.weight"):
        if value.shape[0] != target_shape[0] or value.shape[1] > target_shape[1]:
            return None
        lifted = np.zeros(target_shape, dtype=value.dtype)
        lifted[:, : value.shape[1]] = value
        return lifted

    if key == "motor_dnn.fused_norm.weight":
        gain = math.sqrt(value.size / np.prod(target_shape))
        lifted = np.zeros(target_shape, dtype=value.dtype)
        lifted[: value.shape[0]] = value * gain
        return lifted

    return None


def torch_to_jax_params(
    torch_state: Dict[str, Any],
    config: Optional[ModelConfig] = None,
    arch: str = "auto",
) -> Tuple[Dict[str, jnp.ndarray], ModelConfig]:
    """
    Translates a PyTorch state_dict into JAX arrays.
    Returns: (jax_params_dict, model_config)
    """
    # Clean up prefixes
    clean_state: Dict[str, np.ndarray] = {}
    for k, v in torch_state.items():
        k_clean = k.replace("module.", "")
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.bfloat16:
                clean_state[k_clean] = v.detach().to(torch.float32).cpu().numpy()
            else:
                clean_state[k_clean] = v.detach().cpu().numpy()
        else:
            clean_state[k_clean] = np.array(v)

    detected_arch = detect_checkpoint_architecture(clean_state)
    target_arch = detected_arch if arch == "auto" else arch

    if config is None:
        vocab_size = clean_state["sensory_dnn.embedding.weight"].shape[0] if "sensory_dnn.embedding.weight" in clean_state else 50257
        surface_dim = clean_state["sensory_dnn.embedding.weight"].shape[1] if "sensory_dnn.embedding.weight" in clean_state else 640
        num_shells = clean_state["volumetric_core.intra_shell"].shape[0] if "volumetric_core.intra_shell" in clean_state else 16
        radial_dim = clean_state["volumetric_core.radial_proj_in.weight"].shape[0] if "volumetric_core.radial_proj_in.weight" in clean_state else 256
        num_shortcut_tracts = clean_state["volumetric_core.shortcut_tracts"].shape[0] if "volumetric_core.shortcut_tracts" in clean_state else 40
        config = ModelConfig(
            vocab_size=vocab_size,
            surface_dim=surface_dim,
            num_shells=num_shells,
            radial_dim=radial_dim,
            num_shortcut_tracts=num_shortcut_tracts,
            arch=target_arch,
        )

    # Initialize template params
    jax_params = init_params(config)

    # Copy / lift parameters
    for k, v_np in clean_state.items():
        if k in jax_params:
            target_shape = jax_params[k].shape
            if v_np.shape == target_shape:
                jax_params[k] = jnp.array(v_np)
            else:
                lifted = lift_v2_to_v3_tensor(k, v_np, target_shape)
                if lifted is not None:
                    jax_params[k] = jnp.array(lifted)

    # Handle V2 inertia gate inheritance in V3
    if target_arch == "v3" and detected_arch == "v2" and "volumetric_core.inertia_gate" in clean_state:
        gate_val = float(clean_state["volumetric_core.inertia_gate"])
        gamma = 1.0 / (1.0 + math.exp(-gate_val))
        inherit = min(max(1.0 - gamma, 1e-3), 1.0 - 1e-3)
        jax_params["volumetric_core.log_write"] = jnp.full((num_shells,), math.log(math.expm1(inherit)), dtype=jnp.float32)
        jax_params["volumetric_core.log_leak"] = jnp.full((num_shells,), math.log(inherit / (1.0 - inherit)), dtype=jnp.float32)

    return jax_params, config


def load_torch_checkpoint_to_jax(
    checkpoint_path: str,
    arch: str = "auto",
) -> Tuple[Dict[str, jnp.ndarray], ModelConfig]:
    """Reads a .pt checkpoint file directly and converts to JAX parameter dict."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(blob, dict) and "state_dict" in blob:
        blob = blob["state_dict"]

    return torch_to_jax_params(blob, arch=arch)


def save_jax_checkpoint(
    checkpoint_path: str,
    params: Dict[str, jnp.ndarray],
    config: ModelConfig,
    extra_metadata: Optional[Dict[str, Any]] = None,
):
    """Saves JAX parameters and configuration into a compressed .npz archive."""
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    arrays_to_save: Dict[str, np.ndarray] = {
        k: np.array(v) for k, v in params.items()
    }
    # Save config attributes as scalar metadata
    cfg_dict = {
        "__config_vocab_size": config.vocab_size,
        "__config_surface_dim": config.surface_dim,
        "__config_num_shells": config.num_shells,
        "__config_radial_dim": config.radial_dim,
        "__config_num_shortcut_tracts": config.num_shortcut_tracts,
        "__config_arch": config.arch,
        "__config_max_len": config.max_len,
        "__config_max_phase_steps": config.max_phase_steps,
    }
    arrays_to_save.update(cfg_dict)

    if extra_metadata:
        for k, v in extra_metadata.items():
            arrays_to_save[f"__meta_{k}"] = np.array(v)

    np.savez_compressed(checkpoint_path, **arrays_to_save)


def load_jax_checkpoint(checkpoint_path: str) -> Tuple[Dict[str, jnp.ndarray], ModelConfig]:
    """Loads a .npz JAX checkpoint archive."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    data = np.load(checkpoint_path)
    params: Dict[str, jnp.ndarray] = {}

    config = ModelConfig(
        vocab_size=int(data.get("__config_vocab_size", 50257)),
        surface_dim=int(data.get("__config_surface_dim", 640)),
        num_shells=int(data.get("__config_num_shells", 16)),
        radial_dim=int(data.get("__config_radial_dim", 256)),
        num_shortcut_tracts=int(data.get("__config_num_shortcut_tracts", 40)),
        arch=str(data.get("__config_arch", "v2")),
        max_len=int(data.get("__config_max_len", 2048)),
        max_phase_steps=int(data.get("__config_max_phase_steps", 1024)),
    )

    for k in data.files:
        if not k.startswith("__"):
            params[k] = jnp.array(data[k])

    return params, config
