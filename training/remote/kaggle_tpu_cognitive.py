"""
CerebroSpheroid cognitive-curriculum training on Kaggle's free Cloud TPU (v3-8).

Why this file exists
--------------------
The previous `kaggle_tpu.py` could not produce a trainable model. It redefined its own copy
of the architecture and ran ingestion under `torch.no_grad()`, so no gradient ever reached
the encoder: it optimized a strictly weaker objective than the one the production trainer
uses. It also set `vocab_size=32000` while tokenizing with gpt2 (50,257 ids) and clamped
every token id with `min(t, vocab_size-1)`, silently colliding roughly 18,000 distinct
tokens onto a handful of embeddings. That is data corruption, not truncation. This script
imports the real `src.model.CerebroSpheroid` and the real trained objective
(`src.spheroid_v3.training_rollout`, the one `training/overfit_sanity.py` proved can reach
0.0003 nats), and it reads the seeded `cerebro_cognitive` split with the correct vocabulary.

Kaggle TPU facts this is built around
-------------------------------------
- Kaggle does offer TPUs for free. Docs: 20 hours/week, 9 hours maximum per session.
- The instance is a v3-8 TPU VM: 8 cores (4 chips x 2). PyTorch/XLA is supported; Kaggle
  runs it under the TPU VM runtime and pytorch/xla ships a Kaggle PJRT notebook.
- This script deliberately uses ONE core, not all 8. The laptop-scale model is 146,618,503
  parameters: about 0.29 GB in bf16 plus ~1.2 GB of AdamW moments, which fits a single
  core's HBM with enormous room. Multi-core data parallelism would add replica-group and
  sharding complexity while the model is this small. `xm.mark_step()` is still required.

Fixed shapes are not optional
-----------------------------
`get_specialized_cognitive_stream` pads each batch to the longest row *inside that batch*,
so its widths vary. XLA recompiles its whole graph whenever an input shape changes, and one
training step unrolls roughly 100 core cycles (see `prompt_len`/`resp_len` below), so a
varying width means recompiling that graph every single step. Every batch here is therefore
re-padded to a constant (`PROMPT_LEN`, `RESP_LEN`), measured from the corpus: prompt p99 is
32 and max 37, response p99 is 54 and max 73, so 40 x 80 covers the whole split with no
truncation.

Usage
-----
    # On Kaggle: Settings -> Accelerator -> TPU v3-8, upload the three parquets as a Dataset,
    # then run in a notebook cell:
    !python3 kaggle_tpu_cognitive.py --scale laptop --steps 4000 --batch_size 16

    # Portable: the same file runs on GPU or CPU, which is how it gets smoke-tested.
    python3 training/remote/kaggle_tpu_cognitive.py --scale small --steps 6 \
        --eval_batches 8 --dtype float32 --device cpu
"""

import os
import sys
import time
import json
import math
import hashlib
import argparse
import contextlib

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, ".hf_cache"))
sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F

IS_KAGGLE = os.path.exists("/kaggle")
OUTPUT_DIR = "/kaggle/working" if IS_KAGGLE else os.path.join(ROOT_DIR, "checkpoints")
DATA_DIR = os.path.join(ROOT_DIR, "data", "cerebro_cognitive")

try:
    import torch_xla.core.xla_model as xm
    IS_TPU = True
except Exception:
    xm = None
    IS_TPU = False

from src.model import CerebroSpheroid
from src.data_loaders import get_specialized_cognitive_stream
from src.spheroid_v3 import training_rollout

CONFIGS = {
    # The shipped architecture, 146,618,503 params. Same dims the diagnostics measured.
    "laptop": dict(surface_dim=640, num_shells=16, radial_dim=256, num_shortcut_tracts=40),
    # 31M, used by overfit_sanity.py; handy for a fast end-to-end check on any device.
    "small": dict(surface_dim=192, num_shells=8, radial_dim=64, num_shortcut_tracts=16),
}

VOCAB_SIZE = 50257  # gpt2. The old 32000 here silently corrupted the token stream.
PROMPT_LEN = 40     # corpus max is 37
RESP_LEN = 80       # corpus max is 73


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_device(requested):
    if requested:
        return torch.device(requested)
    if IS_TPU:
        dev = xm.xla_device()
        print(f"Cloud TPU active via PyTorch/XLA: {dev}")
        return dev
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def disable_gradient_checkpointing(model):
    """Replaces `circulate_mid_tensors` with a plain loop.

    The shipped core wraps each cycle in `torch.utils.checkpoint(..., use_reentrant=False)`
    when `self.training` is set, which exists to fit activations into a 3.7 GB laptop GPU.
    A TPU has no such pressure, and non-reentrant checkpointing is an unnecessary thing to
    hope XLA handles. Substituting the identical arithmetic removes the risk without
    changing a number: the checkpointed path is documented as arithmetically equal.
    """
    core = model.volumetric_core

    def circulate(state, num_cycles=1):
        for _ in range(num_cycles):
            state = core.deliberate_step(state)
        return state

    core.circulate_mid_tensors = circulate


def to_fixed_shape(prompts, targets, eos_id, prompt_len, resp_len, dropped):
    """Re-pads a streamed batch to constant widths.

    Returns None when a row is longer than the cap rather than silently truncating it, since
    on this curriculum the tail of the response is where the numeric answer sits; cropping it
    would teach a wrong answer. With the defaults (40 x 80) nothing exceeds the caps.
    """
    p = prompts[:, :prompt_len]
    t = targets[:, :resp_len]
    if prompts.shape[1] > prompt_len or targets.shape[1] > resp_len:
        dropped[0] += 1
        return None
    pad_p = prompt_len - p.shape[1]
    pad_t = resp_len - t.shape[1]
    if pad_p:
        p = F.pad(p, (0, pad_p), value=eos_id)
    if pad_t:
        t = F.pad(t, (0, pad_t), value=-100)
    return p, t


def next_fixed(stream, eos_id, prompt_len, resp_len, dropped):
    while True:
        prompts, targets = next(stream)
        got = to_fixed_shape(prompts, targets, eos_id, prompt_len, resp_len, dropped)
        if got is not None:
            return got


@torch.no_grad()
def evaluate_fixed(model, eval_batches, device, thinking_cycles):
    """Token-weighted CE over the frozen batch list, matching `train_scale_1m.py`.

    Selection reads this, never the training step's own loss. The step's loss is a
    batch_size-sized draw and its standard deviation measured 0.35 nats on weights that
    could not improve, which is what let the shipped `*_best.pt` turn out worse than
    `*_latest.pt` (RESEARCH_NOTES 20.12).
    """
    model.eval()
    loss_sum, n_tokens = 0.0, 0
    for prompts, targets in eval_batches:
        logits = training_rollout(model, prompts, targets.shape[1], thinking_cycles=thinking_cycles)
        # float32 for the reduction: a 50,257-way softmax accumulator in bf16 loses ground.
        per_batch = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1), ignore_index=-100
        )
        valid = int((targets != -100).sum())
        loss_sum += float(per_batch) * valid
        n_tokens += valid
    model.train()
    return loss_sum / max(n_tokens, 1), n_tokens


def collect_eval_batches(stream, eos_id, n_batches, prompt_len, resp_len, dropped):
    """Draws the frozen held-out set once. Never refreshed, so a delta is the weights' doing."""
    out = []
    tokens = 0
    while len(out) < n_batches and tokens < 20000:
        p, t = next_fixed(stream, eos_id, prompt_len, resp_len, dropped)
        out.append((p, t))
        tokens += int((t != -100).sum())
    return out, tokens


def write_sidecar(ckpt_path, payload):
    """Provenance beside the weights, not inside them.

    The tensor payload stays a bare state_dict so the existing `weights_only=True` readers
    keep working. A bare state_dict carrying no step count is how a twenty-step snapshot came
    to be documented as converged.
    """
    with open(ckpt_path + ".meta.json", "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def save_torch(obj, path):
    if IS_TPU:
        xm.save(obj, path)
    else:
        torch.save(obj, path)


def main():
    ap = argparse.ArgumentParser(description="Train CerebroSpheroid on a Kaggle Cloud TPU")
    ap.add_argument("--scale", default="laptop", choices=list(CONFIGS))
    ap.add_argument("--steps", type=int, default=4000, help="Optimizer steps. Kaggle caps a session at 9h.")
    ap.add_argument("--batch_size", type=int, default=16, help="TPUs prefer larger batches than the laptop 2")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--thinking_cycles", type=int, default=2)
    ap.add_argument("--eval_interval", type=int, default=100)
    ap.add_argument("--eval_batches", type=int, default=64)
    ap.add_argument("--min_improvement", type=float, default=0.002)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--prompt_len", type=int, default=PROMPT_LEN)
    ap.add_argument("--resp_len", type=int, default=RESP_LEN)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dtype", default=None, choices=["bfloat16", "float32"],
                    help="Default bfloat16 on TPU (native MXU), float32 elsewhere.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--save_prefix", default="cerebro_spheroid_tpu")
    ap.add_argument("--train_parquet", default=os.path.join(DATA_DIR, "cerebro_cognitive_train.parquet"))
    ap.add_argument("--val_parquet", default=os.path.join(DATA_DIR, "cerebro_cognitive_val.parquet"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    dtype = torch.bfloat16 if (args.dtype is None and IS_TPU) else (
        torch.bfloat16 if args.dtype == "bfloat16" else torch.float32)

    for path in (args.train_parquet, args.val_parquet):
        if not os.path.exists(path):
            raise SystemExit(
                f"missing {path}\nOn Kaggle, attach the cognitive parquets as a Dataset (they are "
                f"gitignored, so they do not travel with the code) and pass --train_parquet/"
                f"--val_parquet at its mount path."
            )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.vocab_size != VOCAB_SIZE:
        raise SystemExit(f"expected gpt2 vocab {VOCAB_SIZE}, got {tokenizer.vocab_size}")
    eos_id = tokenizer.eos_token_id
    train_md5, val_md5 = md5_of(args.train_parquet), md5_of(args.val_parquet)
    if train_md5 == val_md5:
        raise SystemExit("train and val parquets are the same file; that is not a holdout")

    dropped = [0]
    train_stream = get_specialized_cognitive_stream(
        tokenizer, batch_size=args.batch_size, parquet_path=args.train_parquet,
        max_prompt_len=args.prompt_len, max_resp_len=args.resp_len, shuffle_seed=args.seed)
    val_stream = get_specialized_cognitive_stream(
        tokenizer, batch_size=args.batch_size, parquet_path=args.val_parquet,
        max_prompt_len=args.prompt_len, max_resp_len=args.resp_len,
        shuffle_seed=args.seed + 1, allow_synthesis_fallback=False)

    model = CerebroSpheroid(vocab_size=VOCAB_SIZE, **CONFIGS[args.scale])
    if device.type == "xla" or IS_TPU:
        disable_gradient_checkpointing(model)
    model.to(device=device, dtype=dtype)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())

    print("=" * 80)
    print("  CEREBROSPHEROID ON KAGGLE CLOUD TPU")
    print("=" * 80)
    print(f"Device        : {device} | TPU={IS_TPU} | dtype={dtype} | scale={args.scale}")
    print(f"Parameters    : {n_params:,}")
    print(f"Objective     : training_rollout (zero-feedback, trace and state carried)")
    print(f"Checkpointing : {'off (XLA path)' if IS_TPU else 'core default (memory-saving)'}")
    print(f"Train / Val   : {os.path.basename(args.train_parquet)} ({train_md5[:8]}) / "
          f"{os.path.basename(args.val_parquet)} ({val_md5[:8]})")
    print(f"Fixed shapes  : prompt {args.prompt_len} x response {args.resp_len} "
          f"(one XLA compile; skipped-as-too-long batches so far: {dropped[0]})")

    eval_batches, eval_tokens = collect_eval_batches(
        val_stream, eos_id, args.eval_batches, args.prompt_len, args.resp_len, dropped)
    eval_batches = [(p.to(device), t.to(device)) for p, t in eval_batches]
    if len(eval_batches) < 8 or eval_tokens < 2000:
        raise SystemExit(
            f"held-out set froze only {len(eval_batches)} batches / {eval_tokens} tokens; too small "
            f"to select on. Raise --eval_batches.")
    print(f"Held-out set  : {len(eval_batches)} batches, {eval_tokens:,} response tokens (frozen)")

    # One timed pre-flight also forces the XLA compile before the budget clock starts.
    p0, t0b = next_fixed(train_stream, eos_id, args.prompt_len, args.resp_len, dropped)
    p0, t0b = p0.to(device), t0b.to(device)
    c_start = time.time()
    probe = training_rollout(model, p0, t0b.shape[1], thinking_cycles=args.thinking_cycles)
    loss0 = F.cross_entropy(probe.reshape(-1, probe.shape[-1]).float(),
                            t0b.reshape(-1), ignore_index=-100)
    loss0.backward()
    if IS_TPU:
        xm.mark_step()
    compile_s = time.time() - c_start
    print(f"First step    : compiled and ran in {compile_s:.1f}s (CE {float(loss0.detach()):.4f}). "
          f"XLA builds the graph here once.")
    model.zero_grad(set_to_none=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best_ckpt = os.path.join(OUTPUT_DIR, f"{args.save_prefix}_best.pt")
    latest_ckpt = os.path.join(OUTPUT_DIR, f"{args.save_prefix}_latest.pt")
    best_loss = float("inf")
    patience = 0
    t_start = time.time()

    for step in range(1, args.steps + 1):
        prompts, targets = next_fixed(train_stream, eos_id, args.prompt_len, args.resp_len, dropped)
        prompts, targets = prompts.to(device), targets.to(device)

        logits = training_rollout(model, prompts, targets.shape[1],
                                  thinking_cycles=args.thinking_cycles)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                               targets.reshape(-1), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Native optimizer.step() is the supported path on current PyTorch/XLA; the legacy
        # xm.optimizer_step is deprecated and varies by Kaggle's XLA build. mark_step flushes
        # the traced graph so the step actually executes on the device.
        optimizer.step()
        if IS_TPU:
            xm.mark_step()

        if step % args.eval_interval == 0 or step == 1:
            eval_ce, scored = evaluate_fixed(model, eval_batches, device, args.thinking_cycles)
            elapsed = time.time() - t_start
            per_step = elapsed / step
            tag = ""
            if eval_ce < best_loss - args.min_improvement:
                opening = not math.isfinite(best_loss)
                drop = 0.0 if opening else best_loss - eval_ce
                best_loss, patience = eval_ce, 0
                save_torch(model.state_dict(), best_ckpt)
                write_sidecar(best_ckpt, {
                    "event": "first_best" if opening else "new_best", "step": step,
                    "seed": args.seed, "scale": args.scale, "dtype": str(dtype),
                    "n_params": n_params, "thinking_cycles": args.thinking_cycles,
                    "prompt_len": args.prompt_len, "resp_len": args.resp_len,
                    "batch_size": args.batch_size, "lr": args.lr, "device": str(device),
                    "train_parquet_md5": train_md5, "val_parquet_md5": val_md5,
                    "eval_ce_heldout": eval_ce, "eval_tokens": scored,
                    "eval_ppl_heldout": math.exp(min(eval_ce, 20)),
                })
                tag = " | * first best" if opening else f" | * new best (-{drop:.4f})"
            else:
                patience += 1
            print(f"step {step:5d}/{args.steps} | train CE {float(loss.detach()):.4f} | "
                  f"HELD-OUT CE {eval_ce:.4f} (PPL {math.exp(min(eval_ce, 20)):.2f}) | "
                  f"{per_step:.2f}s/step{tag}")
            if patience >= args.patience:
                print(f"Plateau: no held-out improvement for {args.patience} evals. Stopping.")
                break
            if IS_TPU:
                xm.mark_step()

        if step % max(args.eval_interval * 5, 1) == 0:
            save_torch(model.state_dict(), latest_ckpt)
            elapsed = time.time() - t_start
            remaining = (args.steps - step) * (elapsed / step)
            write_sidecar(latest_ckpt, {"step": step, "seed": args.seed, "scale": args.scale,
                                        "dtype": str(dtype), "n_params": n_params,
                                        "thinking_cycles": args.thinking_cycles,
                                        "prompt_len": args.prompt_len, "resp_len": args.resp_len,
                                        "batch_size": args.batch_size, "lr": args.lr,
                                        "device": str(device), "train_parquet_md5": train_md5,
                                        "val_parquet_md5": val_md5, "best_heldout_ce": best_loss,
                                        "elapsed_s": elapsed, "eta_s": remaining})

    save_torch(model.state_dict(), latest_ckpt)
    elapsed = time.time() - t_start
    write_sidecar(latest_ckpt, {"step": min(step, args.steps), "seed": args.seed, "scale": args.scale,
                                "dtype": str(dtype), "n_params": n_params,
                                "thinking_cycles": args.thinking_cycles,
                                "prompt_len": args.prompt_len, "resp_len": args.resp_len,
                                "batch_size": args.batch_size, "lr": args.lr,
                                "device": str(device), "train_parquet_md5": train_md5,
                                "val_parquet_md5": val_md5, "best_heldout_ce": best_loss,
                                "elapsed_s": elapsed, "final": True})
    print("-" * 80)
    print(f"Done in {elapsed / 3600:.2f} h. Best held-out CE {best_loss:.4f} | "
          f"skipped-too-long batches {dropped[0]}")
    print(f"Kaggle caps a TPU session at 9 h and allows 20 h/week: measured {elapsed/3600:.2f} h "
          f"for {args.steps} steps at {elapsed/max(args.steps,1):.2f} s/step.")
    print(f"Checkpoints in {OUTPUT_DIR}: {os.path.basename(best_ckpt)}, {os.path.basename(latest_ckpt)} "
          f"(+ .meta.json provenance)")
    if dropped[0]:
        print(f"[!] {dropped[0]} batches exceeded the fixed shape and were skipped; raise "
              f"--prompt_len/--resp_len to stop losing supervision.")
    print("=" * 80)


if __name__ == "__main__":
    main()
