"""
Inference and Deliberation Engine for CerebroSpheroid Reasoning Agent
Supports:
1. Biological Zero-Feedback Deliberate-Then-Emit Active Reasoning.
2. Refractory Motor Readout (suppresses mode collapse without autoregressive feedback).
3. Living Core Heartbeat Daemon (continuous background resting-state homeostasis).
"""

import os
os.environ.setdefault("HF_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".hf_cache")))

import time
import threading
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from .model import CerebroSpheroid


class ReasoningAgent:
    """
    High-level Inference Interface for CerebroSpheroid:
    1. Sensory Ingestion -> Continuous boundary flux fields.
    2. Active Deliberation -> Multi-cycle internal processing (Zero token emission).
    3. Motor Readout -> Zero-Feedback Efferent Output (x_{t+1} != [x_{1..t}, y_t]).
    4. Living Core Daemon -> Continuous spontaneous heartbeat keeping the core alive.
    """
    def __init__(self, checkpoint_path: str = "checkpoints/cerebro_spheroid_100m.pt", device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = self.tokenizer.vocab_size

        self.model = CerebroSpheroid(
            vocab_size=self.vocab_size,
            surface_dim=640,
            num_shells=16,
            radial_dim=256,
            num_shortcut_tracts=40,
        )

        target_ckpt = None
        if checkpoint_path and checkpoint_path != "checkpoints/cerebro_spheroid_100m.pt" and os.path.exists(checkpoint_path):
            target_ckpt = checkpoint_path
        else:
            for candidate in [
                checkpoint_path,
                "checkpoints/cerebro_spheroid_1m_best.pt",
                "checkpoints/cerebro_spheroid_gsm8k_100m.pt",
                "checkpoints/cerebro_spheroid_svamp_100m.pt",
                "checkpoints/cerebro_spheroid_100m.pt",
            ]:
                if candidate and os.path.exists(candidate):
                    target_ckpt = candidate
                    break

        if target_ckpt:
            self.model.load_compatible_checkpoint(target_ckpt, device=self.device)
            self.checkpoint_loaded = target_ckpt
        else:
            self.checkpoint_loaded = None

        self.model.to(device=self.device, dtype=torch.bfloat16)
        self.model.eval()

        # Living System State
        self.living_state = None
        self._heartbeat_thread = None
        self._stop_heartbeat = threading.Event()
        self._heartbeat_ticks = 0
        self._state_lock = threading.Lock()

    def start_living_heartbeat(self, frequency_hz: float = 4.0):
        """
        Starts a continuous biological pacemaker thread in the background.
        The brain core ticks spontaneously at the specified frequency, maintaining
        an active resting-state default mode network (DMN).
        """
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return

        self._stop_heartbeat.clear()

        def _heartbeat_worker():
            interval = 1.0 / max(frequency_hz, 0.1)
            with torch.no_grad():
                with self._state_lock:
                    if self.living_state is None:
                        self.living_state = self.model.init_brain_state(1, self.device, living=True)

                while not self._stop_heartbeat.is_set():
                    time.sleep(interval)
                    with self._state_lock:
                        self._heartbeat_ticks += 1
                        self.living_state = self.model.live_heartbeat_tick(
                            self.living_state, t_step=self._heartbeat_ticks
                        )

        self._heartbeat_thread = threading.Thread(target=_heartbeat_worker, daemon=True)
        self._heartbeat_thread.start()

    def stop_living_heartbeat(self):
        """Safely pauses the background living heartbeat thread."""
        if self._heartbeat_thread is not None:
            self._stop_heartbeat.set()
            self._heartbeat_thread.join(timeout=1.0)
            self._heartbeat_thread = None

    def get_living_status(self) -> dict:
        is_active = self._heartbeat_thread is not None and self._heartbeat_thread.is_alive()
        return {
            "is_alive": is_active,
            "resting_ticks": self._heartbeat_ticks,
            "has_living_state": self.living_state is not None
        }

    def generate(
        self,
        prompt: str,
        thinking_cycles: int = 4,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        repetition_penalty: float = 1.8
    ) -> str:
        """
        Generates deliberated output with zero feedback.
        Unbounded dynamic generation (like GPT-4 / o1): emits tokens dynamically until
        the model naturally outputs the EOS token (or hits safety ceiling 4096).
        """
        formatted_prompt = prompt.strip()
        tokens = self.tokenizer.encode(formatted_prompt)
        prompt_tensor = torch.tensor([tokens], device=self.device)

        with torch.no_grad():
            # Check if living background state is active
            initial_state = None
            if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
                with self._state_lock:
                    if self.living_state is not None:
                        initial_state = self.living_state.clone()

            # Phase 1: Sensory Ingestion (Perturbing living state or fresh living manifold)
            state, sensory_memory = self.model.ingest_stimulus_with_memory(
                prompt_tensor, state=initial_state, living=True
            )

            # Phase 2: Active Subconscious Deliberation (No emission)
            state = self.model.deliberate(state, thinking_cycles=thinking_cycles)

            # Phase 3: Zero-Feedback Efferent Readout with Arcuate Fasciculus Sensory Resonance
            refractory_trace = torch.zeros(1, self.vocab_size, device=self.device)
            generated_ids = []

            ceiling = max_tokens if max_tokens is not None else 4096
            for step_idx in range(ceiling):
                logits, state, refractory_trace = self.model.emit_next_token(
                    state, sensory_memory=sensory_memory, refractory_trace=refractory_trace, step_idx=step_idx
                )

                # Suppress premature EOS on early tokens so output cannot be blank
                if len(generated_ids) < 3:
                    logits[..., self.tokenizer.eos_token_id] -= 50.0

                if repetition_penalty > 0 and len(generated_ids) > 0:
                    for prev_id in generated_ids:
                        logits[0, prev_id] -= repetition_penalty

                if temperature <= 0.05:
                    next_tok = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    probs = F.softmax(logits / max(temperature, 1e-4), dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1)

                tok_item = next_tok.item()
                if tok_item == self.tokenizer.eos_token_id:
                    break
                generated_ids.append(tok_item)

            # Homeostatic loop: update living state with current cognitive trace
            if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
                with self._state_lock:
                    self.living_state = state

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def run_benchmark(agent: ReasoningAgent, test_queries: list = None):
    if test_queries is None:
        test_queries = [
            "Maya has 8 markers. She buys 4 more and gives 3 away. How many markers does she have?",
            "A train travels 60 miles in 1 hour. How many miles does it travel in 3 hours?",
            "If 1 book costs $5, how much do 4 books cost?",
            "There are 20 apples in a box. You take 6 apples. How many apples remain?",
            "How can identity protection services help protect me against identity theft?",
        ]

    print("=" * 80)
    print("      CEREBRO-SPHEROID: REASONING BENCHMARK SUITE                       ")
    print("=" * 80)
    print(f"Loaded Checkpoint : {agent.checkpoint_loaded or 'Random Initialization'}")
    print(f"Target Device     : {agent.device}")
    status = agent.get_living_status()
    print(f"Living System     : {'ACTIVE (Pulsing at resting state)' if status['is_alive'] else 'QUIESCENT'}")
    print("=" * 80)

    for i, q in enumerate(test_queries, 1):
        t0 = time.time()
        ans = agent.generate(q, thinking_cycles=4, max_tokens=64, temperature=0.0)
        dt = time.time() - t0
        print(f"\n[{i}] Query  : \"{q}\"")
        print(f"    Answer : \"{ans}\"")
        print(f"    Latency: {dt*1000:.1f}ms | Deliberation: 4 cycles | Output Feedback: 0")

    print("\n" + "=" * 80)
