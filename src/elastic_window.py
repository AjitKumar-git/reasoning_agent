"""
Elastic Sliding Context Window & Complementary Bucket Harmonizer.
Framework-agnostic engine supporting PyTorch (GPU) and JAX / XLA (TPU).

Core Principles:
1. Fixed Total Sequence Budget: W_total = L_prompt + L_target is strictly invariant.
2. Complementary Inverse Length Buckets:
   - Deep Reasoning (CoT) : Short Question (64)     -> Massive Deliberation Output (W - 64)
   - Standard CoT         : Typical Question (256)   -> Long Derivation Output (W - 256)
   - Balanced             : Moderate Context (W//2)  -> Moderate Generation (W//2)
   - Long Context         : Long Input (W - 128)     -> Concise Answer (128)
3. Zero Hardware Recompilation:
   Fixed static shapes compile once under XLA / CUDA graphs, eliminating recompilation freezes.
4. Zero Padding Waste:
   Maximizes useful token throughput and compute density across heterogeneous datasets.
"""

from typing import Dict, List, Tuple, Optional, Any, Iterator
import collections
import numpy as np


def get_default_elastic_buckets(total_window: int = 1024) -> List[Tuple[str, int, int]]:
    """Returns the 4 canonical complementary sliding window buckets summing to total_window."""
    assert total_window >= 256, "total_window must be at least 256 tokens"
    return [
        ("deep_reasoning", 64, total_window - 64),
        ("standard_cot", 256, total_window - 256),
        ("balanced", total_window // 2, total_window // 2),
        ("long_context", total_window - 128, 128),
    ]


def select_best_bucket(p_len: int, t_len: int, buckets: List[Tuple[str, int, int]]) -> int:
    """Selects the bucket that minimizes wasted padding while preserving target reasoning tokens."""
    best_idx = 2  # default balanced
    min_penalty = float("inf")

    for idx, (_, bp, bt) in enumerate(buckets):
        if p_len <= bp and t_len <= bt:
            # Fits completely: penalty is wasted padding
            penalty = (bp - p_len) + (bt - t_len)
            if penalty < min_penalty:
                min_penalty = penalty
                best_idx = idx
        elif p_len <= bp:
            # Fits prompt, mild target truncation
            penalty = (bp - p_len) + (t_len - bt) * 3
            if penalty < min_penalty:
                min_penalty = penalty
                best_idx = idx

    return best_idx


class ElasticBucketBatcher:
    """
    Asynchronous / Streaming Bucket Batcher for Elastic Context Windows.
    Buffers incoming dataset records into complementary (L_p, L_t) queues
    and yields batches of exact fixed shapes for GPU and TPU execution.
    """

    def __init__(
        self,
        tokenizer: Any,
        data_iterator: Iterator[Any],
        batch_size: int = 16,
        total_window: int = 1024,
        buckets: Optional[List[Tuple[str, int, int]]] = None,
        framework: str = "jax",  # 'jax' or 'torch'
        target_pad_id: int = -100,
        return_feedback: bool = False,
    ):
        self.tokenizer = tokenizer
        self.data_iter = data_iterator
        self.batch_size = batch_size
        self.total_window = total_window
        self.buckets = buckets if buckets is not None else get_default_elastic_buckets(total_window)
        self.framework = framework.lower()
        self.target_pad_id = target_pad_id
        self.return_feedback = return_feedback

        self.pad_id = tokenizer.pad_token_id if (hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None) else 0
        # Queues per bucket index
        self.queues: Dict[int, collections.deque] = {i: collections.deque() for i in range(len(self.buckets))}

    def _extract_text_pair(self, row: Any) -> Tuple[str, str]:
        """Extracts (prompt, response) from diverse dataset formats (OpenHermes, ShareGPT, LMSYS, tuple pairs, etc.)."""
        if isinstance(row, (tuple, list)):
            if len(row) == 3:
                return str(row[1]), str(row[2])
            elif len(row) >= 2:
                return str(row[0]), str(row[1])
            return "", ""

        if isinstance(row, dict):
            # 1. OpenHermes / ShareGPT format
            convs = row.get("conversations", [])
            if convs:
                q_txt, r_txt = "", ""
                for turn in convs:
                    sender = str(turn.get("from", "")).lower()
                    val = turn.get("value", "")
                    if ("human" in sender or "user" in sender) and not q_txt:
                        q_txt = val
                    elif ("gpt" in sender or "assistant" in sender or "model" in sender) and not r_txt:
                        r_txt = val
                if q_txt and r_txt:
                    return q_txt, r_txt

            # 2. GSM8K / Q&A format
            q_txt = row.get("question", row.get("prompt", row.get("input", "")))
            r_txt = row.get("answer", row.get("response", row.get("output", "")))
            return str(q_txt), str(r_txt)
        return "", ""

    def _format_batch(self, b_idx: int, name: str, p_list: list, t_list: list):
        if self.framework == "torch":
            import torch
            prompts = torch.tensor(p_list, dtype=torch.long)
            targets = torch.tensor(t_list, dtype=torch.long)
            if self.return_feedback:
                feedback = torch.full_like(targets, self.target_pad_id)
                if targets.shape[1] > 1:
                    feedback[:, 1:] = targets[:, :-1]
                return b_idx, name, prompts, targets, feedback
            return b_idx, name, prompts, targets
        elif self.framework == "jax":
            import jax.numpy as jnp
            prompts = jnp.array(p_list, dtype=jnp.int32)
            targets = jnp.array(t_list, dtype=jnp.int32)
            if self.return_feedback:
                feedback = jnp.full_like(targets, self.target_pad_id)
                feedback = feedback.at[:, 1:].set(targets[:, :-1])
                return b_idx, name, prompts, targets, feedback
            return b_idx, name, prompts, targets
        else:
            return b_idx, name, np.array(p_list, dtype=np.int32), np.array(t_list, dtype=np.int32)

    def _pull_record(self) -> Any:
        if hasattr(self.data_iter, "__next__"):
            return next(self.data_iter)
        elif hasattr(self.data_iter, "next"):
            return self.data_iter.next()
        elif callable(self.data_iter):
            return self.data_iter()
        else:
            raise TypeError("data_iterator is not iterable or callable")

    def next_batch(self) -> Any:
        """
        Pulls from dataset stream until any bucket has batch_size examples.
        Returns: (bucket_idx, bucket_name, prompt_batch, target_batch[, feedback_batch])
        """
        while True:
            # Check if any bucket queue is ready to pop a full batch
            for b_idx in range(len(self.buckets)):
                if len(self.queues[b_idx]) >= self.batch_size:
                    name, p_len, t_len = self.buckets[b_idx]
                    batch_items = [self.queues[b_idx].popleft() for _ in range(self.batch_size)]
                    p_list, t_list = zip(*batch_items)
                    return self._format_batch(b_idx, name, p_list, t_list)

            # Ingest more records from stream
            try:
                row = self._pull_record()
            except (StopIteration, RuntimeError):
                # If stream exhausted, flush whatever bucket has the most items
                best_b = max(self.queues.keys(), key=lambda k: len(self.queues[k]))
                if len(self.queues[best_b]) == 0:
                    raise StopIteration("Data stream exhausted.")
                # Pad to batch_size with repeated items
                items = list(self.queues[best_b])
                self.queues[best_b].clear()
                while len(items) < self.batch_size:
                    items.append(items[0])
                name, p_len, t_len = self.buckets[best_b]
                p_list, t_list = zip(*items)
                return self._format_batch(best_b, name, p_list, t_list)

            q_txt, r_txt = self._extract_text_pair(row)
            if not q_txt or not r_txt:
                continue

            # Tokenize raw
            q_ids = self.tokenizer.encode(q_txt, truncation=False)
            r_ids = self.tokenizer.encode(r_txt, truncation=False)
            q_len, r_len = len(q_ids), len(r_ids)

            # Route to best complementary bucket
            b_idx = select_best_bucket(q_len, r_len, self.buckets)
            name, bp, bt = self.buckets[b_idx]

            # Pad / truncate to the exact bucket dimensions
            if len(q_ids) > bp:
                q_padded = q_ids[:bp]
            else:
                q_padded = q_ids + [self.pad_id] * (bp - len(q_ids))

            if len(r_ids) > bt:
                r_padded = r_ids[:bt]
            else:
                r_padded = r_ids + [self.target_pad_id] * (bt - len(r_ids))

            self.queues[b_idx].append((q_padded, r_padded))
