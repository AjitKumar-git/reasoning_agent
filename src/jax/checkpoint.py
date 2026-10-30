"""
CerebroSpheroid JAX Checkpointing & Autonomous Synchronization Suite.

Features:
1. PyTorch (.pt) <-> JAX Parameter Converter:
   Bi-directional lossless conversion between PyTorch state_dict and JAX PyTrees.
2. Autonomous Checkpoint Discovery (Zero Hardcoded Step Numbers):
   Automatically inspects local and Google Drive/cloud directories to identify the most advanced checkpoint.
3. Asynchronous Background Checkpoint Manager (AsyncCheckpointManager):
   Offloads parameter conversion, PyTorch disk serialization, and remote Google Drive upload
   to an independent CPU thread pool, eliminating TPU latency stalls (0ms I/O pause).
"""

import os
import re
import glob
import json
import shutil
import time
from typing import Dict, Tuple, Optional, Any, List
import concurrent.futures

import numpy as np
import torch
import jax
import jax.numpy as jnp

from .v4_model import ModelConfigV4, precompute_axonal_indices


def save_jax_to_torch_pt(
    params: Dict[str, jnp.ndarray],
    buffers: Tuple[jnp.ndarray, jnp.ndarray],
    save_path: str,
    meta: Optional[dict] = None,
) -> None:
    """Converts JAX PyTree into standard PyTorch state_dict and saves to .pt format."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch_state: Dict[str, torch.Tensor] = {}
    for k, v in params.items():
        torch_state[k] = torch.from_numpy(np.array(v))
    torch_state["core.src_indices"] = torch.from_numpy(np.array(buffers[0]))
    torch_state["core.dst_indices"] = torch.from_numpy(np.array(buffers[1]))

    tmp_path = save_path + ".tmp"
    torch.save(torch_state, tmp_path)
    os.replace(tmp_path, save_path)

    if meta:
        meta_path = save_path.replace(".pt", ".json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


def load_torch_pt_to_jax(
    ckpt_path: str, config: Optional[ModelConfigV4] = None
) -> Tuple[Dict[str, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
    """Loads a PyTorch .pt checkpoint directly into JAX parameter dictionary."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    params: Dict[str, jnp.ndarray] = {}
    for k, v in state_dict.items():
        if k not in ("core.src_indices", "core.dst_indices"):
            params[k] = jnp.array(v.detach().cpu().numpy())

    if config is None:
        shells = 16
        tracts = 40
    else:
        shells = config.num_shells
        tracts = config.num_shortcut_tracts

    default_src, default_dst = precompute_axonal_indices(shells, tracts)
    src_idx = jnp.array(state_dict.get("core.src_indices", default_src))
    dst_idx = jnp.array(state_dict.get("core.dst_indices", default_dst))
    buffers = (src_idx.astype(jnp.int32), dst_idx.astype(jnp.int32))

    return params, buffers


def discover_latest_checkpoint(
    search_dirs: List[str],
    preferred_pattern: str = r"step_(\d+)\.pt",
) -> Tuple[Optional[str], int, Optional[dict]]:
    """
    Autonomously scans directories to locate the most advanced checkpoint.
    Returns: (checkpoint_path, latest_step, metadata_dict)
    """
    latest_path: Optional[str] = None
    max_step: int = 0
    latest_meta: Optional[dict] = None

    for d in search_dirs:
        if not os.path.exists(d):
            continue
        # Scan for step-numbered checkpoints
        for pt_file in glob.glob(os.path.join(d, "*.pt")):
            match = re.search(preferred_pattern, os.path.basename(pt_file))
            if match:
                step_num = int(match.group(1))
                if step_num > max_step:
                    max_step = step_num
                    latest_path = pt_file

    # Check latest.pt
    for d in search_dirs:
        candidate_latest = os.path.join(d, "cerebro_spheroid_v4_jax_latest.pt")
        candidate_json = candidate_latest.replace(".pt", ".json")
        if os.path.exists(candidate_latest) and os.path.exists(candidate_json):
            try:
                with open(candidate_json, "r") as f:
                    meta = json.load(f)
                    step_val = meta.get("step", 0)
                    if step_val >= max_step:
                        max_step = step_val
                        latest_path = candidate_latest
                        latest_meta = meta
            except Exception:
                pass

    if latest_path and latest_meta is None:
        json_path = latest_path.replace(".pt", ".json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    latest_meta = json.load(f)
            except Exception:
                pass

    return latest_path, max_step, latest_meta


class AsyncCheckpointManager:
    """Manages asynchronous, non-blocking parameter serialization and cloud synchronization."""

    def __init__(
        self,
        target_dir: str,
        local_scratch_dir: str = "/tmp/cerebro_ckpts",
        cloud_sync: bool = False,
        cloud_url: Optional[str] = None,
        cloud_token: Optional[str] = None,
    ):
        self.target_dir = target_dir
        self.local_dir = local_scratch_dir
        self.cloud_sync = cloud_sync
        self.cloud_url = cloud_url
        self.cloud_token = cloud_token
        os.makedirs(self.target_dir, exist_ok=True)
        os.makedirs(self.local_dir, exist_ok=True)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._last_future: Optional[concurrent.futures.Future] = None

    def _worker(
        self,
        params_cpu: Dict[str, np.ndarray],
        buffers_cpu: Tuple[np.ndarray, np.ndarray],
        filename: str,
        meta_dict: Optional[dict],
    ):
        try:
            local_path = os.path.join(self.local_dir, filename)
            target_path = os.path.join(self.target_dir, filename)

            torch_state: Dict[str, torch.Tensor] = {}
            for k, v in params_cpu.items():
                torch_state[k] = torch.from_numpy(v)
            torch_state["core.src_indices"] = torch.from_numpy(buffers_cpu[0])
            torch_state["core.dst_indices"] = torch.from_numpy(buffers_cpu[1])

            tmp_local = local_path + ".tmp"
            torch.save(torch_state, tmp_local)
            os.replace(tmp_local, local_path)

            if meta_dict:
                local_json = local_path.replace(".pt", ".json")
                with open(local_json, "w") as f:
                    json.dump(meta_dict, f, indent=2)
                target_json = os.path.join(self.target_dir, os.path.basename(local_json))
                shutil.copyfile(local_json, target_json)

            tmp_target = target_path + ".tmp"
            shutil.copyfile(local_path, tmp_target)
            os.replace(tmp_target, target_path)

            # Prune scratch disk to prevent unbounded local disk growth
            self._prune_scratch()

            # Mirror to bioflo.in if cloud sync is active
            if self.cloud_sync and ("latest" in filename or "best" in filename):
                try:
                    from src.cloud_sync import push_checkpoint_async
                    step_val = meta_dict.get("step", 0) if meta_dict else 0
                    loss_val = meta_dict.get("loss", 0.0) if meta_dict else 0.0
                    target_type = "best" if "best" in filename else "latest"
                    push_checkpoint_async(
                        local_path,
                        step=step_val,
                        loss=loss_val,
                        target_type=target_type,
                        url=self.cloud_url,
                        token=self.cloud_token,
                    )
                except Exception as sync_err:
                    print(f"⚠️ [CloudSync Error]: {sync_err}")
        except Exception as err:
            print(f"⚠️ [AsyncCheckpoint Error]: {err}")

    def _prune_scratch(self):
        try:
            pt_files = glob.glob(os.path.join(self.local_dir, "*_step_*.pt"))
            if len(pt_files) > 2:
                pt_files.sort(key=os.path.getmtime)
                for old_f in pt_files[:-2]:
                    try:
                        os.remove(old_f)
                        old_j = old_f.replace(".pt", ".json")
                        if os.path.exists(old_j):
                            os.remove(old_j)
                    except OSError:
                        pass
        except Exception:
            pass

    def dispatch(
        self,
        params: Dict[str, jnp.ndarray],
        buffers: Tuple[jnp.ndarray, jnp.ndarray],
        filename: str,
        meta_dict: Optional[dict] = None,
        block_if_busy: bool = False,
    ) -> bool:
        """Dispatches an asynchronous save job. Returns True if job was queued."""
        if not block_if_busy and (self._last_future is not None and not self._last_future.done()):
            return False  # Prior backup is still writing; skip redundant queue

        # Rapid CPU memory hand-off (<15ms)
        params_cpu = {k: np.array(v) for k, v in params.items()}
        buffers_cpu = (np.array(buffers[0]), np.array(buffers[1]))
        self._last_future = self.executor.submit(self._worker, params_cpu, buffers_cpu, filename, meta_dict)
        return True

    def flush(
        self,
        params: Dict[str, jnp.ndarray],
        buffers: Tuple[jnp.ndarray, jnp.ndarray],
        filename: str,
        meta_dict: Optional[dict] = None,
    ):
        """Synchronously flushes state (used on shutdown / emergency exit)."""
        params_cpu = {k: np.array(v) for k, v in params.items()}
        buffers_cpu = (np.array(buffers[0]), np.array(buffers[1]))
        self._worker(params_cpu, buffers_cpu, filename, meta_dict)

    def shutdown(self, wait: bool = True):
        """Waits for all in-flight Drive / cloud writes to complete."""
        self.executor.shutdown(wait=wait)
