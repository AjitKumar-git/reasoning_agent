"""
Full-corpus trainer for CerebroSpheroidV3, built to be paused and resumed at will.

Three things separate this from `train_scale_1m.py`:

1. **Corpus.** One weighted stream over everything on disk instead of one curriculum at a
   time: OpenHermes, LMSYS, the canonicalised world-knowledge file and the cognitive file.
   The knowledge file is the rebuilt one from `training/build_corpus.py`, where phrasing
   variants are collapsed to one target and ten countries are held out by entity, so the
   knowledge number is a generalisation measurement rather than a lookup score.

2. **Long answers.** A V3 rollout is a Python loop over response steps, so a 900-token
   OpenHermes answer costs 900 sequential forward passes and a longer graph than a 4 GB card
   holds. Each record is trained on a random `--window`-token slice of its answer instead,
   with the genuine preceding token fed back as context and excluded from the loss. A quarter
   of windows are taken from the end of the answer so end-of-text and EOS stay represented.
   Cost per step is then fixed by `--window`, not by the corpus.

3. **Pause and resume.** SIGINT, SIGTERM, a sentinel file, and a wall-clock budget all end
   the run at a step boundary after writing `*_state.pt` (weights, optimizer, scheduler, step,
   best score, RNG state, corpus position) plus `*_state.json` for reading without torch. The
   same command picks it up again. Resume is exact: the corpus resumes at the same record and
   the source sequence replays identically.

Selection is the token-weighted held-out CE over every eval slice at once, printed with a
per-source breakdown. Early stopping stays on the frozen set, so an improvement is a property
of the weights rather than of the draw.

On memory: 162M parameters cost 1.2 GiB as fp32 weights plus gradients, which fits a 4 GB card.
AdamW adds two more fp32 buffers per parameter (another 1.2 GiB) and does not fit: step 1
succeeds because those buffers do not exist yet, then step 2 OOMs. SGD is therefore the default,
and the script prints the arithmetic at start-up and refuses to begin when the configuration
cannot fit.

Usage:
  python3 training/build_corpus.py                     # once, to rebuild the knowledge split
  python3 training/train_v3_full.py --dry_run          # build the eval set, print, exit
  python3 training/train_v3_full.py                    # train, resuming if a state file exists
  python3 training/train_v3_full.py --max_minutes 90   # train for at most 90 minutes
  touch checkpoints/PAUSE                              # stop cleanly after the current step
  python3 training/train_v3_full.py --resume none      # start over, ignoring saved state
"""

import os
os.environ.setdefault(
    "HF_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".hf_cache"))
)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import contextlib
import glob
import hashlib
import json
import math
import random
import signal
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.corpus import (
    DATA_DIR,
    Mixture,
    default_paths,
    parse_mix,
    small_parquet_pairs,
    tokenize_pair,
)
from src.spheroid_v3 import build_spheroid_v3, training_rollout

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints")

STOP_REASON: Optional[str] = None


# --------------------------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------------------------

def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def meta_path_for(ckpt_path: str) -> str:
    return ckpt_path[: -len(".pt")] + ".meta.json" if ckpt_path.endswith(".pt") else ckpt_path + ".meta.json"


def write_json(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)


def read_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------------------------
# batch assembly
# --------------------------------------------------------------------------------------------

def sample_window(answer_ids: List[int], window: int, rng: random.Random,
                  tail_probability: float = 0.25) -> Tuple[Optional[int], List[int]]:
    """Picks the scored slice of one answer and the token heard before it.

    Returns `(context_token, scored_tokens)`. The context token is fed to the auditory channel
    at step 0 and never scored, so a mid-answer window is conditioned on real text instead of
    on silence. `tail_probability` of windows are the final `window` tokens, which keeps
    end-of-text and EOS in the objective even though most answers are longer than a window.
    """
    if len(answer_ids) <= window:
        return None, list(answer_ids)
    max_start = len(answer_ids) - window - 1
    if max_start <= 0:
        start = 0
    elif rng.random() < tail_probability:
        start = max_start
    else:
        start = rng.randint(0, max_start)
    return answer_ids[start], answer_ids[start + 1 : start + 1 + window]


def build_batch(tokenizer, records: Sequence[Tuple[str, str, str]], window: Optional[int],
                rng: random.Random, max_prompt_len: int, max_answer_tokens: int,
                pad_token_id: int):
    """Materialises one batch. `window=None` scores the answer whole, which is what eval uses.

    Returns `(prompts, targets, feedback)`, or None when every record was unusable.
    """
    prompt_rows, target_rows, feedback_rows = [], [], []
    for _, prompt, answer in records:
        prompt_ids, answer_ids = tokenize_pair(
            tokenizer, prompt, answer, max_prompt_len, max_answer_tokens
        )
        if len(prompt_ids) < 2 or len(answer_ids) < 2:
            continue
        if window is None:
            context, scored = None, answer_ids
        elif len(answer_ids) > window:
            context, scored = sample_window(answer_ids, window, rng)
        else:
            context, scored = None, answer_ids
        prompt_rows.append(prompt_ids)
        target_rows.append(scored)
        feedback_rows.append((context, scored))

    if not prompt_rows:
        return None

    max_p = max(len(p) for p in prompt_rows)
    max_t = max(len(t) for t in target_rows)
    prompts = torch.full((len(prompt_rows), max_p), pad_token_id, dtype=torch.long)
    targets = torch.full((len(prompt_rows), max_t), -100, dtype=torch.long)
    feedback = torch.full((len(prompt_rows), max_t), -100, dtype=torch.long)

    for i, (p, t) in enumerate(zip(prompt_rows, target_rows)):
        prompts[i, : len(p)] = torch.tensor(p, dtype=torch.long)
        targets[i, : len(t)] = torch.tensor(t, dtype=torch.long)
        context, _ = feedback_rows[i]
        if context is not None:
            feedback[i, 0] = context
        if len(t) > 1:
            feedback[i, 1: len(t)] = targets[i, : len(t) - 1]
    return prompts, targets, feedback


# --------------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------------

def build_eval_set(args, tokenizer, pad_token_id: int) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
    """Freezes one batch per source. Knowledge comes from the entity-disjoint val file."""
    batches = []

    if args.knowledge_val and any(n == "knowledge" for n, _ in args.mix):
        path = os.path.join(DATA_DIR, "corpus", "world_knowledge_canonical_val.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing; run training/build_corpus.py first")
        records = [("knowledge", q, a) for q, a in small_parquet_pairs(path)]
        if args.eval_knowledge_records > 0:
            records = records[: args.eval_knowledge_records]
        for start in range(0, len(records), args.eval_batch_size):
            built = build_batch(tokenizer, records[start: start + args.eval_batch_size], None,
                                random.Random(args.seed), args.max_prompt_len,
                                args.eval_window, pad_token_id)
            if built:
                batches.append(("knowledge", built[0], built[1]))
        print(f"  eval knowledge   {len(records):,} records from the held-out entity split")

    paths = args.paths
    for name, _ in args.mix:
        if name == "knowledge":
            continue
        if name not in ("openhermes", "lmsys", "gsm8k"):
            continue
        iterator = Mixture([(name, 1.0)], paths, seed=args.seed + 101,
                           holdout_every=args.lm_holdout_every, want_heldout=True).next
        records = []
        for _ in range(args.eval_lm_records):
            try:
                records.append(iterator())
            except RuntimeError:
                break
        for start in range(0, len(records), args.eval_batch_size):
            built = build_batch(tokenizer, records[start: start + args.eval_batch_size], None,
                                random.Random(args.seed), args.max_prompt_len,
                                args.eval_window, pad_token_id)
            if built:
                batches.append((name, built[0], built[1]))
        print(f"  eval {name:<9} {len(records):,} records from held-out row groups "
              f"(every {args.lm_holdout_every}th)")

    if not batches:
        raise SystemExit("no eval batches could be built")
    return batches


def autocast_ctx(device, dtype):
    """Autocast only when there is something to cast.

    torch.amp.autocast on CPU with a float32 target warns and disables itself, which turns
    every step into a warning line in the log for no numerical benefit.
    """
    if dtype == torch.float32:
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype)


@torch.no_grad()
def evaluate(model, batches, device, dtype, thinking_cycles: int) -> Tuple[float, Dict[str, float]]:
    """Token-weighted held-out CE overall and per source, under the trained objective."""
    model.eval()
    totals: Dict[str, List[float]] = {}
    for source, prompts, targets in batches:
        prompts = prompts.to(device)
        targets = targets.to(device)
        with autocast_ctx(device, dtype):
            logits = training_rollout(model, prompts, targets.shape[1],
                                      thinking_cycles=thinking_cycles, targets=targets)
            per_batch = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100
            )
        valid = int((targets != -100).sum())
        bucket = totals.setdefault(source, [0.0, 0])
        bucket[0] += float(per_batch) * valid
        bucket[1] += valid
    model.train()

    per_source = {k: v[0] / max(v[1], 1) for k, v in totals.items()}
    tokens = sum(v[1] for v in totals.values())
    total = sum(v[0] for v in totals.values()) / max(tokens, 1)
    return total, per_source


# --------------------------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------------------------

def state_paths(prefix: str) -> Tuple[str, str]:
    base = os.path.join(CKPT_DIR, prefix)
    return base + "_state.pt", base + "_state.json"


def save_state(prefix: str, model, optimizer, scheduler, step: int, best: float, patience: int,
               consumed: Dict[str, int], elapsed: float, history: List[dict], args,
               kind: str) -> None:
    raw_model = model.module if hasattr(model, "module") else model
    pt_path, json_path = state_paths(prefix)
    os.makedirs(CKPT_DIR, exist_ok=True)
    payload = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "best": best,
        "patience": patience,
        "corpus": consumed,
        "elapsed_seconds": elapsed,
        "history": history[-40:],
        "args": vars(args),
        "arch": args.arch,
        "kind": kind,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng": random.getstate(),
    }
    torch.save(payload, pt_path)
    write_json(json_path, {k: v for k, v in payload.items() if not k.endswith("rng") and k != "model"
                           and k != "optimizer" and k != "scheduler"})
    drive_dir = getattr(args, "drive_dir", None)
    if drive_dir and os.path.exists(drive_dir):
        import shutil
        try:
            shutil.copy2(pt_path, os.path.join(drive_dir, os.path.basename(pt_path)))
            shutil.copy2(json_path, os.path.join(drive_dir, os.path.basename(json_path)))
        except Exception as e:
            print(f"  [warn] failed to mirror state to drive_dir: {e}", flush=True)


def load_state(path: str, model, optimizer, scheduler, device, dtype, args) -> Optional[dict]:
    print(f"  resuming from {os.path.relpath(path, ROOT_DIR)}")
    # Load to host memory, not to `device`. With map_location=device the payload itself keeps a
    # second copy of every weight and optimizer buffer resident on the GPU for the rest of the
    # run: 1.3 GiB at the laptop scale, which is enough that a fresh run fits and a resumed run
    # OOMs on its first forward pass.
    payload = torch.load(path, map_location="cpu", weights_only=False)

    model_param_dtype = torch.float32 if dtype == torch.float16 else dtype

    if not isinstance(payload, dict) or "model" not in payload:
        # Bare weights checkpoint (e.g. *_best.pt or raw model weights)
        print(f"  [checkpoint] {os.path.basename(path)} is a model weights checkpoint. Loading weights...")
        raw_m = model.module if hasattr(model, "module") else model
        if hasattr(raw_m, "load_compatible_checkpoint"):
            raw_m.load_compatible_checkpoint(path, device=device)
        else:
            raw_m.load_state_dict(payload, strict=False)
        model.to(device=device, dtype=model_param_dtype)
        meta = read_json(meta_path_for(path)) or {}
        step = int(meta.get("step", 0))
        best = float(meta.get("selected", meta.get("best", float("inf"))))
        return {
            "step": step,
            "best": best,
            "patience": 0,
            "elapsed_seconds": float(meta.get("elapsed_minutes", 0.0)) * 60,
            "history": [],
            "corpus": {},
        }

    stored_arch = payload.get("arch", "v2")
    if stored_arch != args.arch:
        raise SystemExit(f"state file holds an {stored_arch} run but --arch is {args.arch}")

    model.load_state_dict(payload.pop("model"))
    model.to(device=device, dtype=model_param_dtype)
    stored_optimizer = payload.get("args", {}).get("optimizer", "adamw")
    if stored_optimizer != args.optimizer:
        # The parameter groups differ (AdamW stores betas/eps, SGD stores momentum), and
        # loading one into the other fails deep inside the optimizer with a bare KeyError.
        raise SystemExit(
            f"state file was written with --optimizer {stored_optimizer} and this run asks for "
            f"{args.optimizer}. Re-run with --optimizer {stored_optimizer}, or start over with "
            f"--resume none."
        )
    optimizer.load_state_dict(payload.pop("optimizer"))
    # load_state_dict restores the buffers as CPU tensors because the payload came from the host;
    # the first step would then fail on a device mismatch between a moment buffer and its param.
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    scheduler.load_state_dict(payload.pop("scheduler"))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if payload.get("torch_rng") is not None:
        # set_rng_state requires a CPU byte tensor. The payload now comes from the host, but the
        # dtype is pinned anyway: it costs nothing and a uint8 view is what the generator wants.
        torch.set_rng_state(payload.pop("torch_rng").detach().cpu().to(torch.uint8))
    cuda_rng = payload.pop("cuda_rng", None)
    if cuda_rng and torch.cuda.is_available():
        if len(cuda_rng) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.detach().cpu().to(torch.uint8) for s in cuda_rng])
        elif len(cuda_rng) > 0:
            torch.cuda.set_rng_state(cuda_rng[0].detach().cpu().to(torch.uint8))
    python_rng = payload.pop("python_rng", None)
    if python_rng:
        random.setstate(tuple(python_rng))

    saved_args = payload.get("args", {})
    for key in ("max_prompt_len", "max_answer_tokens", "window", "mix", "seed", "max_steps"):
        if key not in saved_args:
            continue
        old, new = saved_args[key], getattr(args, key, None)
        if key == "mix":
            old, new = sorted((str(a), float(b)) for a, b in old), sorted((str(a), float(b)) for a, b in new)
        if old != new:
            print(f"  [warn] {key} was {old!r} when this state was written and is {new!r} now; "
                  f"the run continues under the new value")
    print(f"  step {payload.get('step', 0):,} | best held-out CE {payload.get('best', float('inf')):.4f} "
          f"| {payload.get('elapsed_seconds', 0) / 60:.1f} prior minutes "
          f"| corpus {sum(payload.get('corpus', {}).values()):,} records consumed")
    return payload


# --------------------------------------------------------------------------------------------
# signals
# --------------------------------------------------------------------------------------------

def install_signal_handlers() -> None:
    def handler(signum, _frame):
        global STOP_REASON
        name = signal.Signals(signum).name
        if STOP_REASON is None:
            STOP_REASON = f"received {name}"
            print(f"\n  [pause] {name} received; finishing the current step, then saving. "
                  f"Send it again to abort immediately.", flush=True)
        else:
            # Be truthful about "abort immediately": a second signal must actually abort, or the
            # message above is a promise the handler does not keep.
            print(f"\n  [abort] second {name}; dropping the run without saving state.", flush=True)
            raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, handler)
        except (ValueError, AttributeError):
            pass


def stop_requested(args) -> Optional[str]:
    """Returns the reason to stop, or None. Cheap enough to call every step."""
    global STOP_REASON
    if STOP_REASON:
        return STOP_REASON
    if os.path.exists(args.pause_file):
        try:
            os.remove(args.pause_file)
        except OSError:
            pass
        return f"pause file {args.pause_file} was present (removed)"
    return None


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------

BYTES_PER_PARAM = {"sgd": 8, "spheroid": 12, "adamw": 16}


def memory_preflight(device, param_count: int, optimizer_name: str, window: int,
                     batch_size: int, dtype: torch.dtype = torch.float32) -> None:
    """Prints the static cost before it turns into an OOM at step 2.

    Weights and gradients are fp32 and resident for the whole run: 4 bytes each, plus 4 per
    optimizer moment buffer per parameter. SpheroidOptimizer allocates only 1 moment buffer
    (12 B/param), saving 50% state memory compared to AdamW (16 B/param).
    """
    gib = 1024 ** 3
    static = param_count * BYTES_PER_PARAM[optimizer_name] / gib
    print(f"memory {static:.2f} GiB static for {optimizer_name} "
          f"({BYTES_PER_PARAM[optimizer_name]} B/param: weights, grads"
          + (", one moment)" if optimizer_name == "spheroid" else (", two moments)" if optimizer_name == "adamw" else ")")))
    if device.type != "cuda":
        return
    free, total = (x / gib for x in torch.cuda.mem_get_info())
    headroom = free - static
    # Empirically calibrated on A100 / laptop: in bf16/fp16, activation memory per
    # micro-batch item per 64-token window is ~0.13 GiB (0.26 GiB in fp32).
    coeff = 0.13 if dtype in (torch.bfloat16, torch.float16) else 0.26
    need = 0.6 + coeff * batch_size * (window / 64.0)
    print(f"       {free:.2f} GiB free of {total:.2f} GiB | headroom {headroom:.2f} GiB | "
          f"estimated need {need:.2f} GiB at window {window}, batch {batch_size}")
    if headroom >= need:
        return
    if optimizer_name == "adamw" and headroom < 1.0:
        raise SystemExit(
            f"{headroom:.2f} GiB of headroom is insufficient for AdamW. Use --optimizer sgd or lower --batch_size."
        )
    print("       [notice] memory headroom is tight. If a step OOMs it will be dropped cleanly and retried.")


def main():
    parser = argparse.ArgumentParser(
        description="Full-corpus CerebroSpheroidV4 trainer with pause and resume",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # corpus
    parser.add_argument("--mix", default="openhermes=0.45,lmsys=0.35,gsm8k=0.15,cognitive=0.05",
                        help="Weighted sources, e.g. openhermes=0.45,lmsys=0.35,gsm8k=0.15,cognitive=0.05")
    parser.add_argument("--max_prompt_len", type=int, default=256, help="Prompt token cap")
    parser.add_argument("--max_answer_tokens", type=int, default=1024,
                        help="Answers longer than this are truncated for storage, not scored in full")
    parser.add_argument("--window", type=int, default=64,
                        help="Scored answer tokens per record. Fixed step cost; raise if VRAM allows")
    parser.add_argument("--lm_holdout_every", type=int, default=50,
                        help="Hold out every Nth parquet row group of OpenHermes/LMSYS for eval")
    # optimisation
    parser.add_argument("--arch", default="v4", choices=["v4", "v3", "v2"], help="Manifold generation")
    parser.add_argument("--tokenizer", default="llama", choices=["llama", "gpt2"],
                        help="Tokenizer vocabulary: 'llama' (32,000 Byte-Fallback BPE, default) or 'gpt2' (legacy 50,257)")
    parser.add_argument("--preset", default="v4_small",
                        choices=["v4_small", "v4_wide", "v4_probe", "v4_gpt2_small", "v4_gpt2_wide", "v4_probe_25m"],
                        help="Architecture preset: v4_small (127M Llama), v4_wide (279M Llama), v4_probe (16.8M), or legacy v4_gpt2_* presets")
    parser.add_argument("--surface_dim", type=int, default=None, help="Explicit surface dimension override")
    parser.add_argument("--num_shells", type=int, default=None, help="Explicit shell count override")
    parser.add_argument("--radial_dim", type=int, default=None, help="Explicit radial dimension override")
    parser.add_argument("--num_shortcut_tracts", type=int, default=None, help="Explicit shortcut tracts override")
    parser.add_argument("--compile", action="store_true", help="Compile deliberate step using torch.compile")
    parser.add_argument("--batch_size", type=int, default=1, help="Records per micro-batch")
    parser.add_argument("--grad_accum", type=int, default=8, help="Micro-batches per gradient step")
    parser.add_argument("--thinking_cycles", type=int, default=1, help="Internal deliberation cycles")
    parser.add_argument("--lr", type=float, default=None,
                        help="Base learning rate (default: 0.005 for Spheroid, 0.015 for SGD, 3e-4 for AdamW)")
    parser.add_argument("--optimizer", default="spheroid", choices=["spheroid", "sgd", "adamw"],
                        help="Optimizer: 'spheroid' (default) uses isometric Newton-Schulz polar updates "
                             "and recurrent spectral radius guards; 'sgd' uses momentum SGD; "
                             "'adamw' uses coordinate-wise moments (16 B/param, high VRAM).")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Cosine floor as a fraction of --lr")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    # run control
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--max_minutes", type=float, default=0.0,
                        help="Stop cleanly after this many minutes in this session (0 = no limit)")
    parser.add_argument("--eval_interval", type=int, default=400)
    parser.add_argument("--patience", type=int, default=25, help="Evals without improvement before stopping")
    parser.add_argument("--min_improvement", type=float, default=0.002)
    parser.add_argument("--save_every", type=int, default=40, help="Steps between state writes (default: 40)")
    parser.add_argument("--log_every", type=int, default=10, help="Steps between progress lines")
    parser.add_argument("--save_prefix", default="cerebro_spheroid_v4_state")
    parser.add_argument("--drive_dir", default=None,
                        help="Optional Google Drive directory to continuously mirror best checkpoints and state to")
    parser.add_argument("--resume", default="auto",
                        help="'auto' (default), 'none', or a path to a *_state.pt")
    parser.add_argument("--init_from", default="none",
                        help="'none' (default, random), 'auto' (newest trained checkpoint), or a .pt path. "
                             "Existing checkpoints were trained on the pre-split knowledge file, so "
                             "initialising from one turns the knowledge eval into a retention check")
    parser.add_argument("--pause_file", default=os.path.join(CKPT_DIR, "PAUSE"),
                        help="Create this file to stop cleanly; it is consumed (deleted) on pause")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--gpus", default="auto",
                        help="Number of GPUs to use: 'auto' (all available if > 1), 'all', '1', '2', etc. "
                             "Automatically launches torch.distributed.run when > 1 GPU.")
    # eval sizing
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--eval_window", type=int, default=128,
                        help="Eval scores the first N answer tokens. Deterministic, unlike --window")
    parser.add_argument("--eval_lm_records", type=int, default=64)
    parser.add_argument("--eval_knowledge_records", type=int, default=0,
                        help="0 uses the whole held-out knowledge split")
    parser.add_argument("--knowledge_val", dest="knowledge_val", action="store_true",
                        help="Include the held-out knowledge split in eval if knowledge is in mix")
    parser.set_defaults(knowledge_val=False)
    parser.add_argument("--select_on", default="total", choices=["total", "knowledge", "openhermes", "gsm8k"],
                        help="Which held-out number decides 'best'")
    parser.add_argument("--probe_tokens", type=int, default=0, help="0 disables the sample generation (default: 0)")
    parser.add_argument("--dry_run", action="store_true", help="Build the eval set and exit")
    args = parser.parse_args()

    # Multi-GPU auto-launcher: if multiple GPUs are available and requested,
    # and we are not already inside torch.distributed.run, launch it automatically.
    if "LOCAL_RANK" not in os.environ and not getattr(args, "dry_run", False) and torch.cuda.is_available():
        num_avail = torch.cuda.device_count()
        target_gpus = 1
        if args.gpus in ("auto", "all"):
            target_gpus = num_avail
        else:
            try:
                target_gpus = int(args.gpus)
            except ValueError:
                target_gpus = 1
        if target_gpus > 1 and num_avail >= target_gpus:
            print(f"[multi-gpu] Detected {num_avail} GPUs. Auto-launching torch.distributed.run with {target_gpus} GPUs...\n", flush=True)
            import subprocess
            clean_args = []
            skip_next = False
            for a in sys.argv[1:]:
                if skip_next:
                    skip_next = False
                    continue
                if a == "--gpus":
                    skip_next = True
                    continue
                if a.startswith("--gpus="):
                    continue
                clean_args.append(a)

            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={target_gpus}",
                os.path.abspath(__file__),
            ] + clean_args + [f"--gpus={target_gpus}"]
            ret = subprocess.call(cmd)
            sys.exit(ret)

    install_signal_handlers()

    # Distributed environment setup
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if is_distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)

    def rank_print(*p_args, **p_kwargs):
        if rank == 0:
            print(*p_args, **p_kwargs)

    args.mix = parse_mix(args.mix)
    args.paths = default_paths()

    # Download / build missing sources safely on rank 0
    if rank == 0:
        for name, weight in args.mix:
            path = args.paths.get(name)

            if name == "knowledge" and (not path or not os.path.exists(path)):
                print("[corpus notice] 'knowledge' split not found on disk; auto-building via training/build_corpus.py...")
                import subprocess
                build_script = os.path.join(ROOT_DIR, "training", "build_corpus.py")
                subprocess.run([sys.executable, build_script], check=True)
                path = args.paths.get(name)

            if name == "openhermes" and (not path or not os.path.exists(path)):
                print("[corpus notice] 'openhermes' not found on disk; auto-downloading directly from Hugging Face (~759 MB)...")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                oh_url = "https://huggingface.co/api/datasets/teknium/OpenHermes-2.5/parquet/default/train/0.parquet"
                ret = os.system(f'curl -L --create-dirs -o "{path}" "{oh_url}"')
                if ret != 0 or not os.path.exists(path) or os.path.getsize(path) < 100_000_000:
                    import urllib.request
                    print("[corpus notice] downloading via urllib fallback...")
                    urllib.request.urlretrieve(oh_url, path)
                print(f"[corpus notice] Downloaded openhermes: {os.path.getsize(path) / (1024**2):.1f} MB")

            if name == "gsm8k" and (not path or not os.path.exists(path)):
                print("[corpus notice] 'gsm8k' not found on disk; auto-downloading directly from Hugging Face (~2.3 MB)...")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                gsm_url = "https://huggingface.co/api/datasets/openai/gsm8k/parquet/main/train/0.parquet"
                ret = os.system(f'curl -L --create-dirs -o "{path}" "{gsm_url}"')
                if ret != 0 or not os.path.exists(path) or os.path.getsize(path) < 1_000_000:
                    import urllib.request
                    print("[corpus notice] downloading gsm8k via urllib fallback...")
                    urllib.request.urlretrieve(gsm_url, path)
                print(f"[corpus notice] Downloaded gsm8k: {os.path.getsize(path) / (1024**2):.1f} MB")

    if is_distributed:
        torch.distributed.barrier()

    available_mix = []
    for name, weight in args.mix:
        path = args.paths.get(name)
        if path and os.path.exists(path):
            available_mix.append((name, weight))
        else:
            rank_print(f"[corpus notice] source {name!r} not found at {path}; omitting from mix")
    if not available_mix:
        raise FileNotFoundError(f"None of the sources in mix were found on disk: {args.paths}")
    total_w = sum(w for _, w in available_mix)
    args.mix = [(n, w / total_w) for n, w in available_mix]
    rank_print("Active training mix:", ", ".join(f"{n}={w:.2f}" for n, w in args.mix))
    if args.lr is None:
        if args.optimizer == "spheroid":
            args.lr = 0.002
        elif args.optimizer == "sgd":
            args.lr = 0.015
        else:
            args.lr = 3e-4
    if args.window < 8:
        raise SystemExit("--window must be at least 8 scored tokens")
    if args.eval_window < 8:
        raise SystemExit("--eval_window must be at least 8 scored tokens")
    os.makedirs(CKPT_DIR, exist_ok=True)

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    if args.dtype == "auto":
        if device.type == "cuda":
            # T4 (sm_75 Turing) does not have native bf16 hardware; fp16 uses high-speed tensor cores
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    from src.corpus import build_tokenizer
    tokenizer = build_tokenizer(getattr(args, "tokenizer", "llama"))
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    rank_print("=" * 88)
    rank_print("  CEREBRO-SPHEROID V4  |  FULL-CORPUS TRAINER  |  PAUSE / RESUME")
    rank_print("=" * 88)
    rank_print(f"device {device} ({world_size} GPU{'s' if world_size > 1 else ''}) | dtype {dtype} | arch {args.arch} | preset {getattr(args, 'preset', 'v4_small')} | seed {args.seed}")
    weight_summary = "  ".join(f"{n}={w:.2f}" for n, w in args.mix)
    rank_print(f"mix    {weight_summary}")
    rank_print(f"window {args.window} scored tokens/record | prompt cap {args.max_prompt_len} "
               f"| answer cap {args.max_answer_tokens}")
    rank_print(f"batch  {args.batch_size} x {args.grad_accum} x {world_size} GPUs = {args.batch_size * args.grad_accum * world_size} records/step")

    if args.arch == "v4":
        from src.spheroid_v4 import build_v4_model, compute_manifold_diagnostics
        model = build_v4_model(args, vocab_size=tokenizer.vocab_size)
        if getattr(args, "compile", False):
            model.compile_deliberate()
    elif args.arch == "v3":
        from src.spheroid_v3 import build_spheroid_v3
        model = build_spheroid_v3("laptop")
    elif args.arch == "v2":
        from src.model import CerebroSpheroid
        model = CerebroSpheroid(
            vocab_size=tokenizer.vocab_size,
            surface_dim=args.surface_dim or 640,
            num_shells=args.num_shells or 16,
            radial_dim=args.radial_dim or 256,
            num_shortcut_tracts=args.num_shortcut_tracts or 40,
        )
    model_param_dtype = torch.float32 if dtype == torch.float16 else dtype
    model.to(device=device, dtype=model_param_dtype)
    param_count = sum(p.numel() for p in model.parameters())
    rank_print(f"model  {type(model).__name__} | {param_count:,} parameters")
    if rank == 0:
        memory_preflight(device, param_count, args.optimizer, args.window, args.batch_size, dtype)

    # Optimizer: decay matrices only. Norms, biases and embeddings take none, which is the
    # usual split and keeps the 380-answer knowledge table from being shrunk toward zero.
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "embed" in name or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    optimizer_groups = [{"params": decay, "weight_decay": args.weight_decay},
                        {"params": no_decay, "weight_decay": 0.0}]
    if args.optimizer == "spheroid":
        from src.spheroid_optimizer import build_spheroid_optimizer
        optimizer = build_spheroid_optimizer(
            model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            spectral_clamp=1.0,
        )
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(optimizer_groups, lr=args.lr, momentum=0.9, nesterov=True)
    else:
        optimizer = torch.optim.AdamW(optimizer_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, args.warmup_steps, args.max_steps, args.min_lr_ratio)
    )

    # --- resume or fresh start -------------------------------------------------------------
    state_file = None
    if args.dry_run:
        rank_print("resume skipped (--dry_run)")
    elif args.resume == "auto":
        candidate, _ = state_paths(args.save_prefix)
        if not os.path.exists(candidate) and getattr(args, "drive_dir", None):
            drive_cand = os.path.join(args.drive_dir, os.path.basename(candidate))
            if os.path.exists(drive_cand):
                import shutil
                rank_print(f"Found saved state in Google Drive: {drive_cand}. Restoring locally...")
                os.makedirs(CKPT_DIR, exist_ok=True)
                shutil.copy2(drive_cand, candidate)
                drive_json = drive_cand.replace("_state.pt", "_state.json")
                if os.path.exists(drive_json):
                    shutil.copy2(drive_json, candidate.replace("_state.pt", "_state.json"))

        if os.path.exists(candidate):
            state_file = candidate
        else:
            # 1. Search for other state files on disk, newest first
            state_candidates = [
                os.path.join(CKPT_DIR, "cerebro_spheroid_v3_kaggle_state.pt"),
                os.path.join(CKPT_DIR, "cerebro_spheroid_v3_colab_state.pt"),
                os.path.join(CKPT_DIR, "cerebro_spheroid_v3full_state.pt"),
            ]
            all_states = [f for f in glob.glob(os.path.join(CKPT_DIR, "*_state.pt")) if os.path.isfile(f)]
            all_states.sort(key=os.path.getmtime, reverse=True)
            for sc in state_candidates + all_states:
                if os.path.exists(sc):
                    state_file = sc
                    break

            # 2. If no full state file found, look for best / latest model checkpoints
            if not state_file:
                weight_candidates = [
                    os.path.join(CKPT_DIR, "cerebro_spheroid_v3_kaggle_best.pt"),
                    os.path.join(CKPT_DIR, "cerebro_spheroid_v3_colab_best.pt"),
                    os.path.join(CKPT_DIR, "cerebro_spheroid_v3full_best.pt"),
                    os.path.join(CKPT_DIR, "cerebro_spheroid_distill_v2_best.pt"),
                ]
                all_weights = [f for f in glob.glob(os.path.join(CKPT_DIR, "*.pt")) if not f.endswith("_state.pt")]
                all_weights.sort(key=os.path.getmtime, reverse=True)
                for wc in weight_candidates + all_weights:
                    if os.path.exists(wc):
                        state_file = wc
                        break

            # 3. If still nothing found, auto-download cerebro_spheroid_v3_colab_best.pt from Google Drive
            if not state_file:
                colab_best = os.path.join(CKPT_DIR, "cerebro_spheroid_v3_colab_best.pt")
                if rank == 0:
                    from scripts.download_checkpoints import download_drive_file, CHECKPOINTS
                    download_drive_file(CHECKPOINTS["v3_colab_best"]["file_id"], colab_best, CHECKPOINTS["v3_colab_best"]["size_bytes"])
                if is_distributed:
                    torch.distributed.barrier()
                if os.path.exists(colab_best):
                    state_file = colab_best

    elif args.resume != "none":
        state_file = args.resume
        if not os.path.exists(state_file) and os.path.basename(state_file) == "cerebro_spheroid_v3_colab_best.pt":
            if rank == 0:
                from scripts.download_checkpoints import download_drive_file, CHECKPOINTS
                download_drive_file(CHECKPOINTS["v3_colab_best"]["file_id"], state_file, CHECKPOINTS["v3_colab_best"]["size_bytes"])
            if is_distributed:
                torch.distributed.barrier()

    payload = None
    if state_file:
        payload = load_state(state_file, model, optimizer, scheduler, device, dtype, args)
    else:
        if args.init_from != "none":
            init_path = args.init_from
            if init_path == "auto":
                for candidate in ("cerebro_spheroid_v3_colab_best.pt",
                                  "cerebro_spheroid_distill_v2_best.pt",
                                  "cerebro_spheroid_1m_latest.pt",
                                  "cerebro_spheroid_100m.pt"):
                    full = os.path.join(CKPT_DIR, candidate)
                    if os.path.exists(full):
                        init_path = full
                        break
                else:
                    init_path = "none"
            elif os.path.basename(init_path) == "cerebro_spheroid_v3_colab_best.pt" and not os.path.exists(init_path):
                if rank == 0:
                    from scripts.download_checkpoints import download_drive_file, CHECKPOINTS
                    download_drive_file(CHECKPOINTS["v3_colab_best"]["file_id"], init_path, CHECKPOINTS["v3_colab_best"]["size_bytes"])
                if is_distributed:
                    torch.distributed.barrier()

            if init_path != "none" and os.path.exists(init_path):
                raw_m = model.module if hasattr(model, "module") else model
                loaded = raw_m.load_compatible_checkpoint(init_path, device=device)
                rank_print(f"init   {os.path.relpath(init_path, ROOT_DIR)} "
                           f"({'loaded' if loaded else 'refused; random init'})")
                if loaded and args.knowledge_val:
                    rank_print("       note: that checkpoint was trained on the pre-split knowledge file, "
                               "which contained every one of the held-out entities. The knowledge eval "
                               "will measure retention of those countries, not generalisation to them.")
            else:
                rank_print("init   random")
        else:
            rank_print("init   random (--init_from none)")

    # Wrap model with DistributedDataParallel if running multi-GPU
    if is_distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    use_scaler = (dtype == torch.float16 and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    step = int(payload["step"]) if payload else 0
    best = float(payload["best"]) if payload else float("inf")
    patience = int(payload["patience"]) if payload else 0
    elapsed = float(payload.get("elapsed_seconds", 0.0)) if payload else 0.0
    history = list(payload.get("history", [])) if payload else []
    consumed = payload.get("corpus") if payload else None

    ckpt_best = os.path.join(CKPT_DIR, f"{args.save_prefix}_best.pt")
    ckpt_latest = os.path.join(CKPT_DIR, f"{args.save_prefix}_latest.pt")

    # --- eval set -------------------------------------------------------------------------
    rank_print("-" * 88)
    rank_print("Freezing the held-out eval set")
    t0 = time.time()
    eval_batches = build_eval_set(args, tokenizer, pad_token_id)
    eval_tokens = sum(int((t != -100).sum()) for _, _, t in eval_batches)
    rank_print(f"  {len(eval_batches)} batches | {eval_tokens:,} scored tokens | "
               f"{time.time() - t0:.1f}s to build")
    if args.dry_run:
        for source, prompts, targets in eval_batches:
            rank_print(f"  {source:<10} batch {tuple(prompts.shape)} -> targets {tuple(targets.shape)}")
        rank_print("\n[dry_run] exiting before training")
        return

    # --- data ----------------------------------------------------------------------------
    rank_print("-" * 88)
    rank_print("Opening the corpus")
    t0 = time.time()
    if consumed and world_size > 1:
        rank_consumed = {n: count // world_size for n, count in consumed.items()}
    else:
        rank_consumed = consumed
    rank_seed = args.seed + rank * 10007
    mixture = Mixture(args.mix, args.paths, seed=rank_seed, holdout_every=args.lm_holdout_every,
                      want_heldout=False, consumed=rank_consumed)
    total_open_consumed = sum(mixture.consumed.values()) * world_size
    rank_print(f"  positioned at {total_open_consumed:,} records "
               f"({time.time() - t0:.1f}s to open)")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    vocab_size = tokenizer.vocab_size
    session_start = time.time()
    session_base = session_start
    start_step = step
    call_rng = random.Random(rank_seed * 7 + 3)

    rank_print("-" * 88)
    rank_print(f"Training from step {step + 1:,} to {args.max_steps:,} "
               f"| eval every {args.eval_interval} steps | pause file {os.path.relpath(args.pause_file, ROOT_DIR)}")
    rank_print("-" * 88)

    reason = None
    while step < args.max_steps:
        step += 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_loss_tensor = torch.zeros(1, device=device)
        step_count = 0

        oom = False
        accum_steps = args.grad_accum
        for accum_idx in range(accum_steps):
            records = []
            for _ in range(args.batch_size):
                record = mixture.next()
                records.append(record)
            built = build_batch(tokenizer, records, args.window, call_rng,
                                args.max_prompt_len, args.max_answer_tokens, pad_token_id)
            if built is None:
                continue
            prompts, targets, feedback = (t.to(device) for t in built)
            try:
                sync_ctx = model.no_sync() if (is_distributed and accum_idx < accum_steps - 1 and hasattr(model, "no_sync")) else contextlib.nullcontext()
                with sync_ctx:
                    with autocast_ctx(device, dtype):
                        logits = training_rollout(model, prompts, targets.shape[1],
                                                  thinking_cycles=args.thinking_cycles,
                                                  pad_token_id=pad_token_id,
                                                  targets=targets, feedback_tokens=feedback)
                        loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
                    scaler.scale(loss / accum_steps).backward()
                step_loss_tensor = step_loss_tensor + (loss.detach() / accum_steps)
                step_count += 1
            except torch.OutOfMemoryError:
                rank_print(f"  [warn] OOM at step {step:,}; dropping this micro-batch. "
                           f"Lower --window or --batch_size.", flush=True)
                oom = True
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                break
            finally:
                del prompts, targets, feedback, built

        if oom or step_count == 0:
            step -= 1  # Do not advance step on OOM without update
            reason = stop_requested(args)
            if reason:
                break
            continue

        if use_scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
        scheduler.step()

        if device.type == "cuda" and step % 20 == 0:
            torch.cuda.empty_cache()

        train_mean = 0.0
        if rank == 0 and (step % args.log_every == 0 or step == start_step + 1):
            train_mean = step_loss_tensor.item()
            lr_now = scheduler.get_last_lr()[0]
            vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == "cuda" else 0.0
            sec_per_step = (time.time() - session_base) / max(step - start_step, 1)
            passes_per_step = args.batch_size * accum_steps * world_size * args.window
            passes_per_sec = passes_per_step / max(sec_per_step, 1e-4)
            total_consumed = sum(mixture.consumed.values()) * world_size
            gpu_str = f" (x{world_size} gpus)" if world_size > 1 else ""
            print(f"step {step:6,d}/{args.max_steps:,} | loss {train_mean:6.4f} "
                  f"(ppl {math.exp(min(train_mean, 15)):7.1f}) | lr {lr_now:.2e} | "
                  f"{passes_per_sec:5.1f} passes/s | vram {vram:5.0f} MB | {sec_per_step:5.2f}s/step | "
                  f"corpus {total_consumed:,}{gpu_str}", flush=True)

        # --- evaluation -------------------------------------------------------------------
        if step % args.eval_interval == 0:
            t_eval = time.time()
            total_ce, per_source = evaluate(model, eval_batches, device, dtype, args.thinking_cycles)
            eval_seconds = time.time() - t_eval
            breakdown = "  ".join(f"{k} {v:.4f}" for k, v in sorted(per_source.items()))

            diagnostics = {}
            diag_str = ""
            if args.arch == "v4":
                from src.spheroid_v4 import compute_manifold_diagnostics
                raw_m = model.module if hasattr(model, "module") else model
                diag_state = raw_m.init_living_state(min(4, args.eval_batch_size), device)
                diagnostics = compute_manifold_diagnostics(raw_m, diag_state)
                diag_str = f" | eff_rank {diagnostics['mean_effective_rank']:.2f} | rms {diagnostics['mean_shell_rms']:.2f} | write_sat {diagnostics['write_saturation']:.1%}"

            if args.select_on == "knowledge":
                score = per_source.get("knowledge", total_ce)
            elif args.select_on == "openhermes":
                score = per_source.get("openhermes", total_ce)
            elif args.select_on == "gsm8k":
                score = per_source.get("gsm8k", total_ce)
            else:
                score = total_ce

            record = {"step": step, "eval_total": total_ce, "eval_seconds": eval_seconds,
                      "selected": score, **{f"eval_{k}": v for k, v in per_source.items()},
                      **diagnostics,
                      "train_mean": train_mean, "elapsed_minutes": elapsed / 60 + (time.time() - session_start) / 60}
            history.append(record)

            if rank == 0:
                print(f"  [eval {step:,}] total {total_ce:.4f} | {breakdown}{diag_str} | {eval_seconds:.1f}s", flush=True)

                if score < best - args.min_improvement:
                    best, patience = score, 0
                    raw_model = model.module if hasattr(model, "module") else model
                    torch.save(raw_model.state_dict(), ckpt_best)
                    write_json(meta_path_for(ckpt_best), {
                        "kind": "best", "step": step, "arch": args.arch,
                        "select_on": args.select_on, "selected": score,
                        "eval_total": total_ce, "eval_by_source": per_source,
                        "eval_tokens": eval_tokens, "eval_batches": len(eval_batches),
                        "train_mean_at_save": train_mean,
                        "mix": dict(args.mix), "window": args.window, "seed": args.seed,
                        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                        "world_size": world_size,
                        "lr": args.lr, "thinking_cycles": args.thinking_cycles,
                        "cursor_md5": {k: md5_of(v) for k, v in args.paths.items()
                                       if os.path.isfile(v)},
                        "elapsed_minutes": record["elapsed_minutes"],
                        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    })
                    print(f"  new best ({args.select_on}: {score:.4f}) -> "
                          f"{os.path.relpath(ckpt_best, ROOT_DIR)}", flush=True)
                    drive_dir = getattr(args, "drive_dir", None)
                    if drive_dir and os.path.exists(drive_dir):
                        import shutil
                        try:
                            shutil.copy2(ckpt_best, os.path.join(drive_dir, os.path.basename(ckpt_best)))
                            meta_p = meta_path_for(ckpt_best)
                            if os.path.exists(meta_p):
                                shutil.copy2(meta_p, os.path.join(drive_dir, os.path.basename(meta_p)))
                            print(f"  mirrored best checkpoint to {drive_dir}", flush=True)
                        except Exception as e:
                            print(f"  [warn] failed to mirror best to drive_dir: {e}", flush=True)
                else:
                    patience += 1
                    print(f"  no improvement for {patience}/{args.patience} evals "
                          f"(best {best:.4f})", flush=True)

                if args.probe_tokens > 0:
                    probe(model, tokenizer, eval_batches, device, args)
                    model.train()

            if is_distributed:
                torch.distributed.barrier()

            if patience >= args.patience:
                rank_print(f"\n  early stopping: nothing better than {best:.4f} in "
                           f"{args.patience} evaluations")
                reason = f"plateau (best {best:.4f})"
                break

        # --- pause checks -----------------------------------------------------------------
        reason = stop_requested(args)
        if is_distributed:
            stop_val = 1 if reason else 0
            stop_tensor = torch.tensor([stop_val], device=device)
            torch.distributed.all_reduce(stop_tensor, op=torch.distributed.ReduceOp.MAX)
            if stop_tensor.item() == 1 and not reason:
                reason = "peer rank requested stop"

        if reason is None and args.max_minutes > 0:
            used = (time.time() - session_start) / 60
            if used >= args.max_minutes:
                reason = f"time budget reached ({used:.1f}/{args.max_minutes:.0f} min)"
        if reason is None and step % args.save_every == 0:
            elapsed += time.time() - session_start
            session_start = time.time()
            if rank == 0:
                total_consumed_state = {n: count * world_size for n, count in mixture.state().items()}
                save_state(args.save_prefix, model, optimizer, scheduler, step, best, patience,
                           total_consumed_state, elapsed, history, args, "periodic")
                print(f"  state written at step {step:,}", flush=True)
            if is_distributed:
                torch.distributed.barrier()
        if reason:
            break

    # --- wind down ------------------------------------------------------------------------
    elapsed += time.time() - session_start
    raw_model = model.module if hasattr(model, "module") else model
    if rank == 0:
        torch.save(raw_model.state_dict(), ckpt_latest)
        last = history[-1] if history else {}
        write_json(meta_path_for(ckpt_latest), {
            "kind": "latest", "step": step, "arch": args.arch,
            "eval_total": last.get("eval_total"),
            "eval_by_source": {k[5:]: v for k, v in last.items()
                               if k.startswith("eval_") and k not in ("eval_total", "eval_seconds")},
            "best": best, "select_on": args.select_on, "reason": reason,
            "mix": dict(args.mix), "window": args.window, "seed": args.seed,
            "elapsed_minutes": elapsed / 60,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        total_consumed_state = {n: count * world_size for n, count in mixture.state().items()}
        save_state(args.save_prefix, model, optimizer, scheduler, step, best, patience,
                   total_consumed_state, elapsed, history, args, "pause" if reason else "final")

        pt_path, _ = state_paths(args.save_prefix)
        print("=" * 88)
        print(f"stopped at step {step:,} after {elapsed / 60:.1f} minutes"
              + (f" ({reason})" if reason else " (--max_steps reached)"))
        print(f"best held-out {args.select_on} CE : {best:.4f}  -> "
              f"{os.path.relpath(ckpt_best, ROOT_DIR)}")
        print(f"resume state              : {os.path.relpath(pt_path, ROOT_DIR)}")
        print(f"rerun the same command to continue; --resume none to start over")
        print("=" * 88)

        # Automatic Kaggle export: mirror checkpoints to /kaggle/working/ for instant UI access
        if os.path.exists("/kaggle/working") and os.path.abspath(CKPT_DIR) != "/kaggle/working/checkpoints":
            import shutil
            try:
                os.makedirs("/kaggle/working/checkpoints", exist_ok=True)
                for f in os.listdir(CKPT_DIR):
                    src_f = os.path.join(CKPT_DIR, f)
                    dst_f = os.path.join("/kaggle/working/checkpoints", f)
                    if os.path.isfile(src_f):
                        shutil.copy2(src_f, dst_f)
                print("  [kaggle] mirrored checkpoints to /kaggle/working/checkpoints/", flush=True)
            except Exception as e:
                print(f"  [kaggle warn] failed to mirror to /kaggle/working: {e}", flush=True)

    if is_distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def lr_lambda(step: int, warmup: int, total: int, min_ratio: float) -> float:
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if total <= warmup:
        return 1.0
    progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def probe(model, tokenizer, eval_batches, device, args) -> None:
    """Generates from a few held-out prompts so the log shows behaviour, not just a number."""
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()
    shown = 0
    for source, prompts, _ in eval_batches:
        for row in range(min(2, prompts.shape[0])):
            if shown >= 3:
                return
            prompt_ids = prompts[row: row + 1].to(device)
            text = tokenizer.decode(prompt_ids[0].tolist(), skip_special_tokens=True)[:70]
            out = raw_model.solve_with_active_thinking(
                prompt_ids, thinking_cycles=args.thinking_cycles,
                max_answer_tokens=args.probe_tokens, temperature=0.0,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            )
            answer = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
            print(f"    [{source}] {text!r}\n             -> {answer[:110]!r}")
            shown += 1


if __name__ == "__main__":
    main()
