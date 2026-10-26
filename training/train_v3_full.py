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
        if name not in ("openhermes", "lmsys"):
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
    pt_path, json_path = state_paths(prefix)
    os.makedirs(CKPT_DIR, exist_ok=True)
    payload = {
        "model": model.state_dict(),
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

    stored_arch = payload.get("arch", "v2")
    if stored_arch != args.arch:
        raise SystemExit(f"state file holds an {stored_arch} run but --arch is {args.arch}")

    model.load_state_dict(payload.pop("model"))
    model.to(device=device, dtype=dtype)
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
        torch.cuda.set_rng_state_all([s.detach().cpu().to(torch.uint8) for s in cuda_rng])
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

BYTES_PER_PARAM = {"sgd": 8, "adamw": 16}


def memory_preflight(device, param_count: int, optimizer_name: str, window: int,
                     batch_size: int, dtype: torch.dtype = torch.float32) -> None:
    """Prints the static cost before it turns into an OOM at step 2.

    Weights and gradients are fp32 and resident for the whole run: 4 bytes each, plus 4 per
    AdamW moment buffer per parameter. The arithmetic matters because the failure is not at
    start-up: step 1 runs with no optimizer buffers, allocates them in `optimizer.step()`, and
    step 2 is the first one that has to fit alongside them. On a 4 GB card the AdamW total for
    162M parameters is 2.4 GiB, which leaves under 600 MiB, so the run dies at step 2 with a
    message about the allocator rather than about the choice of optimizer.
    """
    gib = 1024 ** 3
    static = param_count * BYTES_PER_PARAM[optimizer_name] / gib
    print(f"memory {static:.2f} GiB static for {optimizer_name} "
          f"({BYTES_PER_PARAM[optimizer_name]} B/param: weights, grads"
          + (", two moments)" if optimizer_name == "adamw" else ")"))
    if device.type != "cuda":
        return
    free, total = (x / gib for x in torch.cuda.mem_get_info())
    headroom = free - static
    dtype_scale = 0.5 if dtype in (torch.bfloat16, torch.float16) else 1.0
    need = 0.6 + (0.55 * dtype_scale) * batch_size * (window / 64.0)
    print(f"       {free:.2f} GiB free of {total:.2f} GiB | headroom {headroom:.2f} GiB | "
          f"estimated need {need:.2f} GiB at window {window}, batch {batch_size}")
    if headroom >= need:
        return
    if optimizer_name == "adamw":
        raise SystemExit(
            f"{headroom:.2f} GiB of headroom against an estimated {need:.2f} GiB need. AdamW's "
            f"buffers are the reason the shortfall exists and cannot be traded away by shrinking "
            f"the window: use --optimizer sgd, which frees "
            f"{param_count * 8 / gib:.2f} GiB, or --arch v2 to reduce the parameter count."
        )
    print("       warning: this is tight. If a step OOMs it is dropped and the step is retried, "
          "so the run continues, but lower --window or --batch_size for a stable rate.")


def main():
    parser = argparse.ArgumentParser(
        description="Full-corpus CerebroSpheroidV3 trainer with pause and resume",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # corpus
    parser.add_argument("--mix", default="openhermes=0.45,lmsys=0.35,knowledge=0.15,cognitive=0.05",
                        help="Weighted sources, e.g. openhermes=0.45,lmsys=0.35,knowledge=0.15")
    parser.add_argument("--max_prompt_len", type=int, default=256, help="Prompt token cap")
    parser.add_argument("--max_answer_tokens", type=int, default=1024,
                        help="Answers longer than this are truncated for storage, not scored in full")
    parser.add_argument("--window", type=int, default=64,
                        help="Scored answer tokens per record. Fixed step cost; raise if VRAM allows")
    parser.add_argument("--lm_holdout_every", type=int, default=50,
                        help="Hold out every Nth parquet row group of OpenHermes/LMSYS for eval")
    # optimisation
    parser.add_argument("--arch", default="v3", choices=["v3", "v2"], help="Manifold generation")
    parser.add_argument("--batch_size", type=int, default=1, help="Records per micro-batch")
    parser.add_argument("--grad_accum", type=int, default=8, help="Micro-batches per gradient step")
    parser.add_argument("--thinking_cycles", type=int, default=1, help="Internal deliberation cycles")
    parser.add_argument("--lr", type=float, default=None,
                        help="Base learning rate (default: 0.015 for SGD, 3e-4 for AdamW)")
    parser.add_argument("--optimizer", default="sgd", choices=["adamw", "sgd"],
                        help="SGD by default because AdamW's two moment buffers per parameter "
                             "add 1.2 GB at the laptop scale, and on a 4 GB card that is the "
                             "difference between running and OOM-ing from step 2. See the "
                             "preflight line the script prints at start-up")
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
    parser.add_argument("--save_every", type=int, default=200, help="Steps between state writes")
    parser.add_argument("--log_every", type=int, default=10, help="Steps between progress lines")
    parser.add_argument("--save_prefix", default="cerebro_spheroid_v3full")
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
    # eval sizing
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--eval_window", type=int, default=128,
                        help="Eval scores the first N answer tokens. Deterministic, unlike --window")
    parser.add_argument("--eval_lm_records", type=int, default=64)
    parser.add_argument("--eval_knowledge_records", type=int, default=0,
                        help="0 uses the whole held-out knowledge split")
    parser.add_argument("--no_knowledge_val", dest="knowledge_val", action="store_false",
                        help="Skip the held-out knowledge split in eval")
    parser.set_defaults(knowledge_val=True)
    parser.add_argument("--select_on", default="total", choices=["total", "knowledge", "openhermes"],
                        help="Which held-out number decides 'best'")
    parser.add_argument("--probe_tokens", type=int, default=24, help="0 disables the sample generation")
    parser.add_argument("--dry_run", action="store_true", help="Build the eval set and exit")
    args = parser.parse_args()

    install_signal_handlers()

    args.mix = parse_mix(args.mix)
    args.paths = default_paths()

    available_mix = []
    for name, weight in args.mix:
        path = args.paths.get(name)
        if path and os.path.exists(path):
            available_mix.append((name, weight))
        else:
            print(f"[corpus notice] source {name!r} not found at {path}; omitting from mix")
    if not available_mix:
        raise FileNotFoundError(f"None of the sources in mix were found on disk: {args.paths}")
    total_w = sum(w for _, w in available_mix)
    args.mix = [(n, w / total_w) for n, w in available_mix]
    print("Active training mix:", ", ".join(f"{n}={w:.2f}" for n, w in args.mix))
    if args.lr is None:
        args.lr = 0.015 if args.optimizer == "sgd" else 3e-4
    if args.window < 8:
        raise SystemExit("--window must be at least 8 scored tokens")
    if args.eval_window < 8:
        raise SystemExit("--eval_window must be at least 8 scored tokens")
    os.makedirs(CKPT_DIR, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.dtype == "auto":
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    else:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.eos_token_id

    print("=" * 88)
    print("  CEREBRO-SPHEROID V3  |  FULL-CORPUS TRAINER  |  PAUSE / RESUME")
    print("=" * 88)
    print(f"device {device} | dtype {dtype} | arch {args.arch} | seed {args.seed}")
    weight_summary = "  ".join(f"{n}={w:.2f}" for n, w in args.mix)
    print(f"mix    {weight_summary}")
    print(f"window {args.window} scored tokens/record | prompt cap {args.max_prompt_len} "
          f"| answer cap {args.max_answer_tokens}")
    print(f"batch  {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum} records/step")

    model = build_spheroid_v3("laptop")
    if args.arch == "v2":
        from src.model import CerebroSpheroid
        model = CerebroSpheroid(vocab_size=tokenizer.vocab_size, surface_dim=640, num_shells=16,
                                radial_dim=256, num_shortcut_tracts=40)
    model.to(device=device, dtype=dtype)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"model  {type(model).__name__} | {param_count:,} parameters")
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
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(optimizer_groups, lr=args.lr, momentum=0.9, nesterov=True)
    else:
        optimizer = torch.optim.AdamW(optimizer_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, args.warmup_steps, args.max_steps, args.min_lr_ratio)
    )

    # --- resume or fresh start -------------------------------------------------------------
    state_file = None
    if args.dry_run:
        # A dry run only builds the eval set, so it must not depend on a state file existing,
        # nor fail because that file was written with a different optimizer.
        print("resume skipped (--dry_run)")
    elif args.resume == "auto":
        candidate, _ = state_paths(args.save_prefix)
        if not os.path.exists(candidate) and getattr(args, "drive_dir", None):
            drive_cand = os.path.join(args.drive_dir, os.path.basename(candidate))
            if os.path.exists(drive_cand):
                import shutil
                print(f"Found saved state in Google Drive: {drive_cand}. Restoring locally...")
                os.makedirs(CKPT_DIR, exist_ok=True)
                shutil.copy2(drive_cand, candidate)
                drive_json = drive_cand.replace("_state.pt", "_state.json")
                if os.path.exists(drive_json):
                    shutil.copy2(drive_json, candidate.replace("_state.pt", "_state.json"))
        state_file = candidate if os.path.exists(candidate) else None
    elif args.resume != "none":
        state_file = args.resume

    payload = None
    if state_file:
        payload = load_state(state_file, model, optimizer, scheduler, device, dtype, args)
    else:
        if args.init_from != "none":
            init_path = args.init_from
            if init_path == "auto":
                for candidate in ("cerebro_spheroid_distill_v2_best.pt",
                                  "cerebro_spheroid_1m_latest.pt",
                                  "cerebro_spheroid_100m.pt"):
                    full = os.path.join(CKPT_DIR, candidate)
                    if os.path.exists(full):
                        init_path = full
                        break
                else:
                    init_path = "none"
            if init_path != "none" and os.path.exists(init_path):
                loaded = model.load_compatible_checkpoint(init_path, device=device)
                print(f"init   {os.path.relpath(init_path, ROOT_DIR)} "
                      f"({'loaded' if loaded else 'refused; random init'})")
                if loaded and args.knowledge_val:
                    print("       note: that checkpoint was trained on the pre-split knowledge file, "
                          "which contained every one of the held-out entities. The knowledge eval "
                          "will measure retention of those countries, not generalisation to them.")
            else:
                print("init   random")
        else:
            print("init   random (--init_from none)")

    step = int(payload["step"]) if payload else 0
    best = float(payload["best"]) if payload else float("inf")
    patience = int(payload["patience"]) if payload else 0
    elapsed = float(payload.get("elapsed_seconds", 0.0)) if payload else 0.0
    history = list(payload.get("history", [])) if payload else []
    consumed = payload.get("corpus") if payload else None

    ckpt_best = os.path.join(CKPT_DIR, f"{args.save_prefix}_best.pt")
    ckpt_latest = os.path.join(CKPT_DIR, f"{args.save_prefix}_latest.pt")

    # --- eval set -------------------------------------------------------------------------
    print("-" * 88)
    print("Freezing the held-out eval set")
    t0 = time.time()
    eval_batches = build_eval_set(args, tokenizer, pad_token_id)
    eval_tokens = sum(int((t != -100).sum()) for _, _, t in eval_batches)
    print(f"  {len(eval_batches)} batches | {eval_tokens:,} scored tokens | "
          f"{time.time() - t0:.1f}s to build")
    if args.dry_run:
        for source, prompts, targets in eval_batches:
            print(f"  {source:<10} batch {tuple(prompts.shape)} -> targets {tuple(targets.shape)}")
        print("\n[dry_run] exiting before training")
        return

    # --- data ----------------------------------------------------------------------------
    print("-" * 88)
    print("Opening the corpus")
    t0 = time.time()
    mixture = Mixture(args.mix, args.paths, seed=args.seed, holdout_every=args.lm_holdout_every,
                      want_heldout=False, consumed=consumed)
    print(f"  positioned at {sum(mixture.consumed.values()):,} records "
          f"({time.time() - t0:.1f}s to open)")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    vocab_size = tokenizer.vocab_size
    session_start = time.time()
    session_base = session_start
    start_step = step
    call_rng = random.Random(args.seed * 7 + 3)

    print("-" * 88)
    print(f"Training from step {step + 1:,} to {args.max_steps:,} "
          f"| eval every {args.eval_interval} steps | pause file {os.path.relpath(args.pause_file, ROOT_DIR)}")
    print("-" * 88)

    reason = None
    while step < args.max_steps:
        step += 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_loss, step_count = 0.0, 0

        oom = False
        for _ in range(args.grad_accum):
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
                with autocast_ctx(device, dtype):
                    logits = training_rollout(model, prompts, targets.shape[1],
                                              thinking_cycles=args.thinking_cycles,
                                              pad_token_id=pad_token_id,
                                              targets=targets, feedback_tokens=feedback)
                    loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
                (loss / args.grad_accum).backward()
                step_loss += loss.detach().item()
                step_count += 1
            except torch.OutOfMemoryError:
                print(f"  [warn] OOM at step {step:,}; dropping this micro-batch. "
                      f"Lower --window or --batch_size.", flush=True)
                oom = True
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                break
            finally:
                del prompts, targets, feedback, built

        if oom or step_count == 0:
            # A step that produced no gradient still has to honour a stop request. Without this
            # check a run stuck in an OOM loop would never pause, however often the sentinel
            # file appeared.
            reason = stop_requested(args)
            if reason:
                break
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        scheduler.step()

        if device.type == "cuda" and step % 20 == 0:
            torch.cuda.empty_cache()

        train_mean = step_loss / step_count
        if step % args.log_every == 0 or step == start_step + 1:
            lr_now = scheduler.get_last_lr()[0]
            vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == "cuda" else 0.0
            sec_per_step = (time.time() - session_base) / max(step - start_step, 1)
            print(f"step {step:6,d}/{args.max_steps:,} | loss {train_mean:6.4f} "
                  f"(ppl {math.exp(min(train_mean, 15)):7.1f}) | lr {lr_now:.2e} | "
                  f"vram {vram:5.0f} MB | {sec_per_step:5.2f}s/step | "
                  f"corpus {sum(mixture.consumed.values()):,}", flush=True)

        # --- evaluation -------------------------------------------------------------------
        if step % args.eval_interval == 0:
            t_eval = time.time()
            total_ce, per_source = evaluate(model, eval_batches, device, dtype, args.thinking_cycles)
            eval_seconds = time.time() - t_eval
            breakdown = "  ".join(f"{k} {v:.4f}" for k, v in sorted(per_source.items()))

            if args.select_on == "knowledge":
                score = per_source.get("knowledge", total_ce)
            elif args.select_on == "openhermes":
                score = per_source.get("openhermes", total_ce)
            else:
                score = total_ce

            record = {"step": step, "eval_total": total_ce, "eval_seconds": eval_seconds,
                      "selected": score, **{f"eval_{k}": v for k, v in per_source.items()},
                      "train_mean": train_mean, "elapsed_minutes": elapsed / 60 + (time.time() - session_start) / 60}
            history.append(record)

            print(f"  [eval {step:,}] total {total_ce:.4f} | {breakdown} | {eval_seconds:.1f}s", flush=True)

            if score < best - args.min_improvement:
                best, patience = score, 0
                torch.save(model.state_dict(), ckpt_best)
                write_json(meta_path_for(ckpt_best), {
                    "kind": "best", "step": step, "arch": args.arch,
                    "select_on": args.select_on, "selected": score,
                    "eval_total": total_ce, "eval_by_source": per_source,
                    "eval_tokens": eval_tokens, "eval_batches": len(eval_batches),
                    "train_mean_at_save": train_mean,
                    "mix": dict(args.mix), "window": args.window, "seed": args.seed,
                    "batch_size": args.batch_size, "grad_accum": args.grad_accum,
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

            if patience >= args.patience:
                print(f"\n  early stopping: nothing better than {best:.4f} in "
                      f"{args.patience} evaluations")
                reason = f"plateau (best {best:.4f})"
                break

        # --- pause checks -----------------------------------------------------------------
        reason = stop_requested(args)
        if reason is None and args.max_minutes > 0:
            used = (time.time() - session_start) / 60
            if used >= args.max_minutes:
                reason = f"time budget reached ({used:.1f}/{args.max_minutes:.0f} min)"
        if reason is None and step % args.save_every == 0:
            elapsed += time.time() - session_start
            session_start = time.time()
            save_state(args.save_prefix, model, optimizer, scheduler, step, best, patience,
                       mixture.state(), elapsed, history, args, "periodic")
            print(f"  state written at step {step:,}", flush=True)
        if reason:
            break

    # --- wind down ------------------------------------------------------------------------
    elapsed += time.time() - session_start
    torch.save(model.state_dict(), ckpt_latest)
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
    save_state(args.save_prefix, model, optimizer, scheduler, step, best, patience,
               mixture.state(), elapsed, history, args, "pause" if reason else "final")

    pt_path, _ = state_paths(args.save_prefix)
    print("=" * 88)
    print(f"stopped at step {step:,} after {elapsed / 60:.1f} minutes"
          + (f" ({reason})" if reason else " (--max_steps reached)"))
    print(f"best held-out {args.select_on} CE : {best:.4f}  -> "
          f"{os.path.relpath(ckpt_best, ROOT_DIR)}")
    print(f"resume state              : {os.path.relpath(pt_path, ROOT_DIR)}")
    print(f"rerun the same command to continue; --resume none to start over")
    print("=" * 88)


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
    model.eval()
    shown = 0
    for source, prompts, _ in eval_batches:
        for row in range(min(2, prompts.shape[0])):
            if shown >= 3:
                return
            prompt_ids = prompts[row: row + 1].to(device)
            text = tokenizer.decode(prompt_ids[0].tolist(), skip_special_tokens=True)[:70]
            out = model.solve_with_active_thinking(
                prompt_ids, thinking_cycles=args.thinking_cycles,
                max_answer_tokens=args.probe_tokens, temperature=0.0,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            )
            answer = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
            print(f"    [{source}] {text!r}\n             -> {answer[:110]!r}")
            shown += 1


if __name__ == "__main__":
    main()
