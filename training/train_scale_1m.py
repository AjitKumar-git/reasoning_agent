"""
CerebroSpheroid: 1-Million Step Scaling Engine with Adaptive Plateau Early-Stopping
Key Features:
1. Target Scale: Up to 1,000,000 gradient steps over full datasets.
2. Plateau Early Stopping: Monitors a FIXED held-out loss; terminates automatically if no
   progress is recorded for `patience` consecutive evaluation intervals.
3. Hybrid Dual-Curriculum Stream: 70% GSM8K/SVAMP reasoning chains + 30% LMSYS conversational fluency.
4. Refractory Trace Backpropagation: Trains biological refractory dynamics directly to eliminate unigram mode collapse.
5. Living Core Heartbeat Stimulation: Trains over active oscillating resting-state manifolds.
6. 4GB VRAM Safety: AdamW/SGD + BFloat16/FP16 autocast maintaining <3.2 GB allocation.

A note on why (2) reads "FIXED". This script used to select `*_best.pt` on the current step's
mean loss over `batch_size * grad_accum` freshly drawn records and save when it beat the record
by 0.005. That estimator measures about 16 records; on weights held *frozen* it has a standard
deviation of 0.35 nats and a 1.7-2.0 nat range, so the 0.005 threshold sits roughly 1.4% inside
one sigma of its own noise and printed "New Best Checkpoint" on one evaluation in six without
the model changing at all. The shipped `cerebro_spheroid_1m_best.pt` consequently scores WORSE
(7.5468) than `cerebro_spheroid_1m_latest.pt` (7.3595) on an identical token set. Selection and
early stopping now both read a frozen, materialised held-out batch list scored token-weighted,
so a recorded improvement is a property of the weights rather than of the draw. The same draws
used to reset `patience_counter`, which is how a ~20-step run could look like a plateau; that
path is gone too. See RESEARCH_NOTES 20.12.
"""

import os
os.environ.setdefault(
    "HF_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".hf_cache"))
)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import math
import random
import hashlib
import warnings
import argparse
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.model import CerebroSpheroid
from src.spheroid_v3 import CerebroSpheroidV3, training_rollout
from src.data_loaders import (
    get_hybrid_reasoning_stream,
    get_lmsys_stream,
    get_gsm8k_stream,
    get_svamp_stream,
    get_specialized_cognitive_stream,
    get_world_knowledge_stream,
    get_general_llm_stream,
    get_openhermes_stream,
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASETS = ["openhermes", "general", "knowledge", "specialized", "hybrid", "lmsys", "gsm8k", "svamp"]


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_stream(name: str, tokenizer, batch_size: int, max_prompt_len: int,
                 max_resp_len: int, parquet_path: Optional[str], shuffle_seed: Optional[int]):
    """Instantiates one curriculum stream. `shuffle_seed` is honoured by the specialized
    loader only; the others shuffle from the global RNG, which is seeded once in `main()`."""
    data_dir = os.path.abspath(os.path.join(ROOT_DIR, "data", "lmsys_chat_1m", "data"))
    if name == "openhermes":
        openhermes_path = parquet_path or os.path.abspath(os.path.join(ROOT_DIR, "data", "openhermes", "openhermes_2_5.parquet"))
        return get_openhermes_stream(
            tokenizer,
            parquet_path=openhermes_path,
            batch_size=batch_size,
            max_prompt_len=max_prompt_len,
            max_resp_len=max_resp_len,
            shuffle_seed=shuffle_seed,
        )
    if name == "general":
        return get_general_llm_stream(
            tokenizer,
            batch_size=batch_size,
            max_prompt_len=max_prompt_len,
            max_resp_len=max_resp_len,
            lmsys_dir=data_dir,
            shuffle_seed=shuffle_seed,
        )
    if name == "knowledge":
        kwargs = {}
        if parquet_path is not None:
            kwargs["parquet_path"] = parquet_path
        if shuffle_seed is not None:
            kwargs["shuffle_seed"] = shuffle_seed
        return get_world_knowledge_stream(
            tokenizer,
            batch_size=batch_size,
            max_prompt_len=max_prompt_len,
            max_resp_len=max_resp_len,
            **kwargs,
        )
    if name == "specialized":
        kwargs = {}
        if parquet_path is not None:
            kwargs["parquet_path"] = parquet_path
            kwargs["allow_synthesis_fallback"] = False
        if shuffle_seed is not None:
            kwargs["shuffle_seed"] = shuffle_seed
        return get_specialized_cognitive_stream(
            tokenizer, batch_size=batch_size,
            max_prompt_len=max_prompt_len, max_resp_len=max_resp_len, **kwargs
        )
    if name == "hybrid":
        return get_hybrid_reasoning_stream(
            data_dir, tokenizer, batch_size=batch_size, reasoning_ratio=0.70,
            max_prompt_len=max_prompt_len, max_resp_len=max_resp_len
        )
    if name == "lmsys":
        return get_lmsys_stream(
            data_dir, tokenizer, batch_size=batch_size,
            max_prompt_len=max_prompt_len, max_resp_len=max_resp_len
        )
    if name == "gsm8k":
        return get_gsm8k_stream(
            tokenizer, batch_size=batch_size,
            max_prompt_len=max_prompt_len, max_resp_len=max_resp_len
        )
    if name == "svamp":
        return get_svamp_stream(
            tokenizer, batch_size=batch_size,
            max_prompt_len=max_prompt_len, max_resp_len=max_resp_len
        )
    raise ValueError(f"unknown dataset: {name}")


def collect_eval_batches(stream, n_batches: int, max_tokens: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Materialises a frozen list of batches to serve as the evaluation set.

    Drawn once, at startup, and never refreshed: every later evaluation scores exactly these
    tensors, so a delta between two evaluations is attributable to the weights alone. Bounded
    by both a batch count and a token count because response length varies a lot across these
    curricula, and the batch count on its own would make the set's statistical size a function
    of which records happened to be drawn.
    """
    collected: List[Tuple[torch.Tensor, torch.Tensor]] = []
    tokens = 0
    while len(collected) < n_batches and tokens < max_tokens:
        prompts, targets = next(stream)
        collected.append((prompts.clone(), targets.clone()))
        tokens += int((targets != -100).sum())
    return collected, tokens


@torch.no_grad()
def evaluate_fixed(model, eval_batches, device, dtype, thinking_cycles: int):
    """Token-weighted cross-entropy over the frozen batch list, under the trained objective.

    Weighting matters: per-batch mean CE gives a 2-token response the same say as an 82-token
    one, which reads optimistically against any token-weighted baseline and moves with batch
    composition rather than with the model.

    `model.eval()` is numerically the trained function here, not a reduced one: V2 has no
    dropout or batchnorm, and the only training-mode-dependent paths are the gradient
    checkpointing in `circulate_mid_tensors` (identical arithmetic) and `spontaneous_noise`,
    which no caller in this repository passes as True.
    """
    model.eval()
    loss_sum = 0.0
    n_tokens = 0
    for prompts, targets in eval_batches:
        prompts = prompts.to(device)
        targets = targets.to(device)
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
            logits = training_rollout(model, prompts, targets.shape[1],
                                      thinking_cycles=thinking_cycles, targets=targets)
            per_batch = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100
            )
        valid = int((targets != -100).sum())
        loss_sum += float(per_batch) * valid
        n_tokens += valid
        del prompts, targets, logits, per_batch
    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return loss_sum / max(n_tokens, 1), n_tokens


def meta_path_for(ckpt_path: str) -> str:
    return ckpt_path[: -len(".pt")] + ".meta.json" if ckpt_path.endswith(".pt") else ckpt_path + ".meta.json"


def write_sidecar(ckpt_path: str, payload: dict) -> None:
    """Records run provenance next to the weights instead of inside them.

    A bare `state_dict()` carries no step count, no seed and no config, and that silence is how
    a roughly-twentieth-step snapshot came to be documented as a converged 5.4002-loss model:
    nothing in the artifact could contradict the claim. The tensor payload stays bare so the
    seven existing `torch.load(..., weights_only=True)` call sites keep working unchanged.
    """
    try:
        with open(meta_path_for(ckpt_path), "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
    except OSError as exc:
        print(f"  [warn] could not write provenance sidecar: {exc}")


def read_sidecar(ckpt_path: str) -> Optional[dict]:
    path = meta_path_for(ckpt_path)
    if os.path.exists(path):
        try:
            with open(path) as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None
    return None


def main():
    parser = argparse.ArgumentParser(description="CerebroSpheroid 1-Million Step Scaling Engine")
    parser.add_argument("--dataset", type=str, default="specialized", choices=DATASETS,
                        help="Curriculum dataset (default: specialized cognitive stream)")
    parser.add_argument("--parquet", type=str, default=None,
                        help="Train parquet for --dataset specialized (default: cerebro_cognitive_train.parquet)")
    parser.add_argument("--max_steps", type=int, default=1_000_000, help="Maximum gradient steps")
    parser.add_argument("--eval_interval", type=int, default=100, help="Steps between plateau evaluation checks")
    parser.add_argument("--patience", type=int, default=100, help="Patience (number of eval intervals without improvement)")
    parser.add_argument("--batch_size", type=int, default=2, help="Micro-batch size per forward pass")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation micro-batches (effective batch = batch_size * grad_accum)")
    parser.add_argument("--max_prompt_len", type=int, default=1024, help="Max input prompt tokens")
    parser.add_argument("--max_resp_len", type=int, default=128, help="Max output answer tokens")
    parser.add_argument("--thinking_cycles", type=int, default=3, help="Subconscious internal deliberation cycles")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"], help="Optimizer (adamw or sgd)")
    parser.add_argument("--lr", type=float, default=None, help="Base learning rate (default: 3e-4 for AdamW, 0.015 for SGD)")
    parser.add_argument("--seed", type=int, default=1337, help="Seeds python/torch RNGs and the eval set draw")
    parser.add_argument("--resume", type=str, default=None, help="Explicit checkpoint to resume from")
    parser.add_argument("--save_prefix", type=str, default="cerebro_spheroid_1m", help="Checkpoint filename prefix")
    parser.add_argument("--arch", type=str, default="v2", choices=["v2", "v3"], help="Manifold generation to train")
    parser.add_argument("--eval_dataset", type=str, default=None, choices=DATASETS,
                        help="Curriculum to draw the fixed eval set from (default: --dataset)")
    parser.add_argument("--eval_parquet", type=str, default=None,
                        help="Held-out parquet for the eval set (default: cerebro_cognitive_val.parquet)")
    parser.add_argument("--eval_batches", type=int, default=128,
                        help="Batches frozen into the eval set")
    parser.add_argument("--eval_max_tokens", type=int, default=20000,
                        help="Stop freezing the eval set once it holds this many scored tokens")
    parser.add_argument("--min_improvement", type=float, default=0.002,
                        help="Held-out CE drop required to claim a new best (fixed set, so noise is ~0)")
    parser.add_argument("--select_on", type=str, default="val", choices=["val", "train_loss"],
                        help="Selection signal. 'train_loss' restores the old noise-dominated rule")
    parser.add_argument("--save_optimizer", action="store_true",
                        help="Also write optimizer/scheduler state next to *_latest.pt so a resume keeps momentum")
    parser.add_argument("--skip_inspection", action="store_true",
                        help="Disable periodic text-generation cognitive inspections during training")
    args = parser.parse_args()

    if args.lr is None:
        args.lr = 3e-4 if args.optimizer == "adamw" else 0.015
    if args.eval_dataset is None:
        args.eval_dataset = args.dataset
    default_val_parquet = (
        os.path.join(ROOT_DIR, "data", "general_knowledge", "world_knowledge_val.parquet")
        if args.eval_dataset in ["general", "knowledge", "openhermes"]
        else os.path.join(ROOT_DIR, "data", "cerebro_cognitive", "cerebro_cognitive_val.parquet")
    )
    args.eval_parquet = args.eval_parquet or default_val_parquet
    default_train_parquet = (
        os.path.join(ROOT_DIR, "data", "general_knowledge", "world_knowledge_train.parquet")
        if args.dataset in ["general", "knowledge"]
        else os.path.join(ROOT_DIR, "data", "cerebro_cognitive", "cerebro_cognitive_train.parquet")
    )
    train_parquet = args.parquet or default_train_parquet

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    effective_batch = args.batch_size * args.grad_accum

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    print("=" * 80)
    print("      CEREBRO-SPHEROID: 1-MILLION STEP LIVING COGNITIVE TRAINING ENGINE  ")
    print("=" * 80)
    print(f"Curriculum Stream  : {args.dataset.upper()} (Expanded Prompt: {args.max_prompt_len} / Resp: {args.max_resp_len} Tokens)")
    print(f"Target Accelerator : {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Hardware       : {gpu_name} ({vram_gb:.2f} GB VRAM)")
    print(f"Max Scale          : {args.max_steps:,} steps | Micro-Batch: {args.batch_size} | Grad Accum: {args.grad_accum}")
    print(f"Effective Batch    : {effective_batch} sequences averaged per gradient step")
    print(f"Optimizer          : {args.optimizer.upper()} (lr={args.lr:.2e})")
    print(f"Seed               : {args.seed}")
    print(f"Selection Signal   : {args.select_on}")
    print(f"Plateau Detector   : Patience {args.patience} checks (eval every {args.eval_interval} steps)")
    print(f"Active Latency     : {args.thinking_cycles} subconscious cycles (Living Manifold)")
    print(f"Manifold           : {args.arch.upper()}")
    print("=" * 80)

    # 1. Instantiate Architecture
    model_cls = CerebroSpheroidV3 if args.arch == "v3" else CerebroSpheroid
    model = model_cls(
        vocab_size=vocab_size,
        surface_dim=640,
        num_shells=16,
        radial_dim=256,
        num_shortcut_tracts=40,
    )

    best_ckpt_path = os.path.join("checkpoints", f"{args.save_prefix}_best.pt")
    latest_ckpt_path = os.path.join("checkpoints", f"{args.save_prefix}_latest.pt")
    optim_path = os.path.join("checkpoints", f"{args.save_prefix}_optim.pt")

    ckpt_candidates = []
    if args.resume:
        ckpt_candidates.append(args.resume)
    ckpt_candidates.extend([
        os.path.join("checkpoints", "cerebro_spheroid_gsm8k_100m.pt"),
        best_ckpt_path,
        os.path.join("checkpoints", "cerebro_spheroid_1m_latest.pt"),
        os.path.join("checkpoints", "cerebro_spheroid_1m_best.pt"),
        os.path.join("checkpoints", "cerebro_spheroid_100m.pt")
    ])

    loaded = None
    for candidate in ckpt_candidates:
        if os.path.exists(candidate):
            print(f"Resuming weights from checkpoint: {candidate}...")
            model.load_compatible_checkpoint(candidate, device="cpu")
            loaded = candidate
            break
    if loaded is None:
        print("No prior checkpoint found; initializing fresh weights.")

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model.to(device=device, dtype=dtype)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, eps=1e-8)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=args.lr * 0.05)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 2. Freeze the held-out evaluation set BEFORE the training stream exists, so neither the
    #    tokeniser cache nor the RNG has been perturbed by training progress.
    print(f"\nFreezing evaluation set: {args.eval_batches} batches from {args.eval_dataset}"
          f"{' @ ' + os.path.relpath(args.eval_parquet, ROOT_DIR) if args.eval_dataset == 'specialized' else ''}")
    eval_stream = build_stream(
        args.eval_dataset, tokenizer, args.batch_size, args.max_prompt_len, args.max_resp_len,
        args.eval_parquet if args.eval_dataset == "specialized" else None,
        shuffle_seed=args.seed,
    )
    eval_batches, eval_tokens = collect_eval_batches(eval_stream, args.eval_batches, args.eval_max_tokens)
    if len(eval_batches) < 8 or eval_tokens < 2000:
        raise RuntimeError(
            f"evaluation set froze only {len(eval_batches)} batches / {eval_tokens:,} tokens. "
            f"A selection signal this small is dominated by which records were drawn -- the "
            f"exact failure this rewrite exists to remove. Raise --eval_batches."
        )
    if eval_tokens < 5000:
        print(f"  [warn] only {eval_tokens:,} eval tokens; standard error on the reported CE is "
              f"roughly {1.5 / math.sqrt(eval_tokens) * 8:.3f} nats. Fine for a smoke test, too "
              f"noisy to compare two runs on.")
    print(f"  frozen: {len(eval_batches)} batches, {eval_tokens:,} scored tokens")
    train_md5 = md5_of(train_parquet) if args.dataset == "specialized" and os.path.exists(train_parquet) else None
    if args.eval_dataset == "specialized" and os.path.exists(args.eval_parquet):
        val_md5 = md5_of(args.eval_parquet)
        if train_md5 and train_md5 == val_md5:
            raise RuntimeError(
                f"--eval_parquet and --parquet name the same file ({train_md5}); that is not a "
                f"holdout. Build one with: python3 src/generate_specialized_dataset.py"
            )
        print(f"  eval md5  {val_md5}")
        print(f"  train md5 {train_md5 or 'n/a (train set is not a parquet)'}")
    else:
        print("  [warn] this eval source has no held-out file; the set is FIXED within the run "
              "(so comparisons across steps are valid) but the records may overlap training.")

    # 3. Select Curriculum Stream
    data_stream = build_stream(
        args.dataset, tokenizer, args.batch_size, args.max_prompt_len, args.max_resp_len,
        args.parquet, shuffle_seed=None,
    )

    test_queries = [
        "What is the capital of France?",
        "What is the capital of Japan?",
        "What is the chemical formula for water?",
        "Maya has 8 markers. She buys 4 more and gives 3 away. How many markers does she have?",
        "If 4 books cost $20, how much do 6 books cost?",
    ]

    start_step = 0
    best_loss = float("inf")
    if loaded is not None:
        side = read_sidecar(loaded)
        if side:
            start_step = int(side.get("step", 0))
            if side.get("eval_dataset") == args.eval_dataset:
                recorded = side.get("eval_ce", side.get("best_loss"))
                best_loss = float(recorded) if recorded is not None and math.isfinite(float(recorded)) else float("inf")
            else:
                best_loss = float("inf")
                print(f"  sidecar eval_dataset ({side.get('eval_dataset')}) != current eval_dataset ({args.eval_dataset}); resetting best_loss to +inf")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(start_step):
                    scheduler.step()
            print(f"  sidecar: resuming at step {start_step:,}, best held-out CE {best_loss:.4f}")
            if args.save_optimizer and os.path.exists(optim_path):
                state = torch.load(optim_path, map_location=device)
                optimizer.load_state_dict(state["optimizer"])
                scheduler.load_state_dict(state["scheduler"])
                print("  optimizer and scheduler state restored (momentum preserved)")
            else:
                print("  [warn] optimizer moments NOT restored -- the first ~100 steps after a "
                      "resume re-warm Adam/SGD statistics and may briefly raise the loss.")
        else:
            print(f"  [warn] no sidecar beside {loaded}; step numbering restarts at 0 and the "
                  f"best-loss record restarts at +inf. This checkpoint predates provenance logging.")

    print("\n" + "-" * 80)
    print("Initiating Scaled Training with Plateau Early-Stopping...")
    print("-" * 80)

    t_start = time.time()
    patience_counter = 0
    step = start_step
    last_eval_ce = best_loss if math.isfinite(best_loss) else float("nan")
    interval_loss = 0.0
    interval_count = 0
    last_eval_seconds = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    def provenance(kind: str, ckpt: str, ce: float, train_loss: Optional[float]) -> dict:
        return {
            "kind": kind,
            "step": step,
            "eval_ce": ce,
            "eval_ppl": math.exp(min(ce, 15)),
            "train_loss_at_save": train_loss,
            "eval_tokens": eval_tokens,
            "eval_batches": len(eval_batches),
            "eval_dataset": args.eval_dataset,
            "eval_parquet": os.path.relpath(args.eval_parquet, ROOT_DIR) if args.eval_dataset == "specialized" else None,
            "eval_parquet_md5": md5_of(args.eval_parquet) if args.eval_dataset == "specialized" and os.path.exists(args.eval_parquet) else None,
            "train_parquet_md5": train_md5,
            "seed": args.seed,
            "arch": args.arch,
            "dataset": args.dataset,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "optimizer": args.optimizer,
            "thinking_cycles": args.thinking_cycles,
            "dtype": str(dtype),
            "wall_minutes_at_save": (time.time() - t_start) / 60.0,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    while step < args.max_steps:
        step += 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        try:
            for _ in range(args.grad_accum):
                prompts, targets = next(data_stream)
                prompts = prompts.to(device)
                targets = targets.to(device)
                batch_size, ans_len = targets.shape

                with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
                    # Ingest stimulus into living manifold, deliberate, then emit with Auditory Re-afference
                    predicted_logits = training_rollout(
                        model, prompts, ans_len, thinking_cycles=args.thinking_cycles, targets=targets
                    )
                    loss = criterion(predicted_logits.reshape(-1, vocab_size), targets.reshape(-1))
                    scaled_loss = loss / args.grad_accum

                scaled_loss.backward()
                accum_loss += scaled_loss.item()
                # Report the unscaled CE. `scaled_loss` is loss/grad_accum, so accumulating it here
                # would divide the interval mean by 8 and make a 7.1-nat step read as 0.89.
                interval_loss += loss.item()
                interval_count += 1

                # Explicitly free micro-batch computation graph tensors to prevent memory inflation
                del prompts, targets, predicted_logits, loss, scaled_loss

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        except torch.OutOfMemoryError as oom_exc:
            print(f"  [warn] CUDA OOM caught at step {step}; flushing VRAM cache and skipping micro-batch.")
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        # Release fragmented cached blocks back to CUDA driver
        if device.type == "cuda" and (step % 2 == 0 or torch.cuda.memory_allocated() > 1.8 * 1024**3):
            torch.cuda.empty_cache()

        loss_val = accum_loss

        # Step Progress Logging. The number here is one step's 16-record mean: keep it as a
        # divergence watchdog only. It is not, and must not be read as, a progress signal.
        if step % 10 == 0 or step == start_step + 1 or step == 1:
            elapsed = time.time() - t_start
            ppl = math.exp(min(loss_val, 15))
            vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            cur_lr = scheduler.get_last_lr()[0]
            train_mean = interval_loss / max(interval_count, 1)
            print(f"Step [{step:7,d}/{args.max_steps:,}] | Loss(step): {loss_val:.4f} | "
                  f"Loss(interval mean): {train_mean:.4f} | PPL: {ppl:8.2f} | LR: {cur_lr:.2e} | "
                  f"VRAM: {vram:.0f} MB | Patience: {patience_counter}/{args.patience}", flush=True)

        # Plateau & Checkpoint Inspection Interval
        if step % args.eval_interval == 0:
            t_eval = time.time()
            if args.select_on == "val":
                eval_ce, _ = evaluate_fixed(model, eval_batches, device, dtype,
                                            args.thinking_cycles)
            else:
                eval_ce = interval_loss / max(interval_count, 1)
            eval_seconds = time.time() - t_eval
            last_eval_seconds = eval_seconds
            last_eval_ce = eval_ce
            sec_per_step = (time.time() - t_start - eval_seconds) / max(step - start_step, 1)
            train_mean = interval_loss / max(interval_count, 1)
            interval_loss, interval_count = 0.0, 0

            if eval_ce < (best_loss - args.min_improvement):
                improvement = best_loss - eval_ce
                was_opening_record = not math.isfinite(best_loss)
                best_loss = eval_ce
                patience_counter = 0
                torch.save(model.state_dict(), best_ckpt_path)
                write_sidecar(best_ckpt_path, provenance("best", best_ckpt_path, eval_ce, train_mean))
                tag = "held-out CE" if args.select_on == "val" else "train-loss mean (NOISY)"
                print(f"  [Eval {step:,}] {tag}: {eval_ce:.4f} (PPL {math.exp(min(eval_ce, 15)):.2f}) "
                      f"| train mean {train_mean:.4f}", flush=True)
                if was_opening_record:
                    print(f"  ★ [First Best Checkpoint] Saved to {best_ckpt_path}", flush=True)
                else:
                    print(f"  ★ [New Best Checkpoint] Saved to {best_ckpt_path} "
                          f"({tag}: {best_loss:.4f}, Improvement: -{improvement:.4f})", flush=True)
            else:
                patience_counter += 1
                tag = "held-out CE" if args.select_on == "val" else "train-loss mean (NOISY)"
                print(f"  [Eval {step:,}] {tag}: {eval_ce:.4f} | train mean {train_mean:.4f}", flush=True)
                print(f"  [Plateau Check] No improvement for {patience_counter}/{args.patience} "
                      f"intervals (Best {tag}: {best_loss:.4f})", flush=True)

            # Periodic latest checkpoint save
            if step % (args.eval_interval * 5) == 0:
                torch.save(model.state_dict(), latest_ckpt_path)
                write_sidecar(latest_ckpt_path, provenance("latest", latest_ckpt_path, eval_ce, train_mean))
                if args.save_optimizer:
                    torch.save({"optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "step": step}, optim_path)

            # Probe cognitive inspection
            if step % (args.eval_interval * 2) == 0:
                model.eval()
                print("\n" + "·" * 60)
                print(f"  [Cognitive Inspection at Step {step:,} | Held-out CE: {eval_ce:.4f}]")
                with torch.no_grad():
                    for q in test_queries:
                        pt = torch.tensor([tokenizer.encode(q)], device=device)
                        ans_toks = model.solve_with_active_thinking(
                            pt, thinking_cycles=args.thinking_cycles, max_answer_tokens=None, eos_token_id=tokenizer.eos_token_id, temperature=0.0
                        )
                        ans_text = tokenizer.decode(ans_toks[0].tolist(), skip_special_tokens=True)
                        print(f"  Q: \"{q[:60]}...\"")
                        print(f"  A: \"{ans_text.strip()}\"")
                print("·" * 60 + "\n")
                model.train()

            if eval_seconds > 0.10 * sec_per_step * args.eval_interval:
                print(f"  [note] this evaluation took {eval_seconds:.1f}s against "
                      f"{sec_per_step * args.eval_interval:.1f}s of training per interval; "
                      f"raise --eval_interval or lower --eval_batches if it dominates the run.")

            # Check early stopping condition
            if patience_counter >= args.patience:
                print("\n" + "!" * 80)
                print(f"  EARLY STOPPING TRIGGERED: No progress for {args.patience} consecutive evaluation intervals ({args.patience * args.eval_interval:,} steps).")
                print(f"  Final Optimum Loss Achieved: {best_loss:.4f}")
                print("!" * 80)
                break

    # Save final weights
    torch.save(model.state_dict(), latest_ckpt_path)
    write_sidecar(latest_ckpt_path, provenance("latest_final", latest_ckpt_path, last_eval_ce, None))
    if args.save_optimizer:
        # Also here, not only on the periodic branch: a run that stops between two of those
        # boundaries would otherwise report --save_optimizer and leave nothing to resume from.
        torch.save({"optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step}, optim_path)
    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"Training Concluded after {step:,} steps in {total_time/60:.2f} minutes "
          f"(evaluated {max((step - start_step) // args.eval_interval, 1)} times, "
          f"~{last_eval_seconds:.1f}s per eval).")
    print(f"Best Weights  : {best_ckpt_path} (Held-out CE: {best_loss:.4f})")
    print(f"Latest Weights: {latest_ckpt_path}")
    print(f"Provenance    : {meta_path_for(best_ckpt_path)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
