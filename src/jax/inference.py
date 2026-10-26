"""
Inference and Deliberation Engine for CerebroSpheroid Reasoning Agent in JAX.

Biological Principles Preserved:
1. Biological Zero-Feedback Deliberate-Then-Emit Active Reasoning:
   x_{t+1} != [x_{1..t}, y_t]. Internal state alone evolves and emits tokens.
2. Refractory Motor Readout:
   Suppresses mode collapse and token looping without autoregressive feedback.
3. Living Core Heartbeat Daemon:
   Continuous background resting-state homeostasis.
"""

import os
os.environ.setdefault("HF_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".hf_cache")))

import time
import threading
from typing import Optional, Dict, Any
import numpy as np

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

from .model import ModelConfig, CerebroSpheroidJAX, generate_zero_feedback
from .convert import load_torch_checkpoint_to_jax, load_jax_checkpoint


class ReasoningAgentJAX:
    """
    High-level Inference Interface for CerebroSpheroid running on JAX:
    1. Sensory Ingestion -> Continuous boundary flux fields.
    2. Active Deliberation -> Multi-cycle internal processing (Zero token emission).
    3. Motor Readout -> Zero-Feedback Efferent Output (x_{t+1} != [x_{1..t}, y_t]).
    4. Living Core Daemon -> Continuous spontaneous heartbeat keeping the core alive.
    """
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/cerebro_spheroid_1m_best.pt",
        arch: str = "auto",
        key: Optional[jax.Array] = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = self.tokenizer.vocab_size

        self.rng_key = key if key is not None else jax.random.PRNGKey(42)

        # Fallback preference list matching PyTorch agent
        target_ckpt = checkpoint_path if checkpoint_path and os.path.exists(checkpoint_path) else None
        if target_ckpt is None:
            for candidate in [
                "checkpoints/cerebro_spheroid_1m_best.pt",
                "checkpoints/cerebro_spheroid_gsm8k_100m.pt",
                "checkpoints/cerebro_spheroid_100m.pt",
            ]:
                if os.path.exists(candidate):
                    target_ckpt = candidate
                    break

        if target_ckpt and target_ckpt.endswith(".npz"):
            params, config = load_jax_checkpoint(target_ckpt)
            self.checkpoint_loaded = target_ckpt
        elif target_ckpt and (target_ckpt.endswith(".pt") or target_ckpt.endswith(".pth")):
            params, config = load_torch_checkpoint_to_jax(target_ckpt, arch=arch)
            self.checkpoint_loaded = target_ckpt
        else:
            config = ModelConfig.from_scale("laptop", arch=("v2" if arch == "auto" else arch))
            config.vocab_size = self.vocab_size
            params = None
            self.checkpoint_loaded = None

        self.config = config
        self.arch = config.arch
        self.model = CerebroSpheroidJAX(config=config, params=params, key=self.rng_key)

        # Living System State
        self.living_state = None
        self._heartbeat_thread = None
        self._stop_heartbeat = threading.Event()
        self._heartbeat_ticks = 0
        self._state_lock = threading.Lock()

    def start_living_heartbeat(self, frequency_hz: float = 4.0):
        """Starts continuous biological pacemaker thread in background."""
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return

        with self._state_lock:
            if self.living_state is None:
                self.living_state = self.model.init_brain_state(batch_size=1, living=True)

        self._stop_heartbeat.clear()
        interval = 1.0 / max(0.1, frequency_hz)

        def _pacemaker_loop():
            step = 0
            while not self._stop_heartbeat.is_set():
                time.sleep(interval)
                with self._state_lock:
                    if self.living_state is not None:
                        self.living_state = self.model.tick_living_dynamics(self.living_state, t_step=step)
                step += 1
                self._heartbeat_ticks = step

        self._heartbeat_thread = threading.Thread(target=_pacemaker_loop, daemon=True)
        self._heartbeat_thread.start()

    def stop_living_heartbeat(self):
        """Stops the living pacemaker background thread."""
        self._stop_heartbeat.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)
            self._heartbeat_thread = None

    def get_living_status(self) -> Dict[str, Any]:
        """Queries the current status of the living core manifold."""
        with self._state_lock:
            is_active = self._heartbeat_thread is not None and self._heartbeat_thread.is_alive()
            energy = 0.0
            if self.living_state is not None:
                energy = float(jnp.mean(jnp.square(self.living_state)))
            return {
                "is_alive": is_active,
                "alive": is_active,
                "resting_ticks": self._heartbeat_ticks,
                "has_living_state": self.living_state is not None,
                "manifold_energy": energy,
                "shells": self.config.num_shells,
                "surface_dim": self.config.surface_dim,
                "arch": self.arch,
            }

    def generate(
        self,
        prompt_text: str,
        thinking_cycles: int = 4,
        max_tokens: int = 64,
        temperature: float = 0.0,
        repetition_penalty: float = 1.25,
    ) -> str:
        """
        Deliberate-then-emit reasoning in JAX with zero external output feedback.
        """
        encoded = self.tokenizer(prompt_text, return_tensors="np")
        prompt_ids = jnp.array(encoded["input_ids"], dtype=jnp.int32)

        self.rng_key, subkey = jax.random.split(self.rng_key)

        output_tokens = self.model.generate(
            prompt_tokens=prompt_ids,
            thinking_cycles=thinking_cycles,
            max_new_tokens=max_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            key=subkey,
        )

        tokens_np = np.array(output_tokens[0])
        decoded = self.tokenizer.decode(tokens_np, skip_special_tokens=True).strip()
        return decoded

    def solve(
        self,
        prompt_text: str,
        thinking_cycles: int = 4,
        max_answer_tokens: Optional[int] = 64,
        temperature: float = 0.0,
    ) -> str:
        """Drop-in convenience alias for solving with active thinking."""
        return self.generate(
            prompt_text,
            thinking_cycles=thinking_cycles,
            max_tokens=max_answer_tokens or 64,
            temperature=temperature,
        )
