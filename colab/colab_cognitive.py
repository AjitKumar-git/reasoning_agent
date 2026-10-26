"""
CerebroSpheroid cognitive-curriculum training on a Google Colab GPU (T4 / V100 / A100).

Why this file exists
--------------------
It replaces `colab_run.py` for the seeded cognitive curriculum. `colab_run.py` decides which
checkpoint is "best" from the *current training batch's own loss* (colab_run.py:219) and keeps
no held-out set. On this objective that number is noise: its standard deviation measures about
0.35 nats across steps on weights that cannot improve, which is exactly how the shipped
`cerebro_spheroid_1m_best.pt` ended up worse than `_latest.pt` (RESEARCH_NOTES 20.12). This
trainer selects on a frozen held-out batch list drawn from `cerebro_cognitive_val.parquet`,
writes a `.meta.json` provenance sidecar beside every checkpoint, and uses the real trained
objective `src.spheroid_v3.training_rollout` (the one `training/overfit_sanity.py` proved can
reach 0.0003 nats), the same three properties the laptop trainer and `kaggle_tpu_cognitive.py`
already have.

GPU, not TPU
------------
This targets a CUDA runtime, so it keeps the core's gradient checkpointing (it exists to fit
the unrolled activations into 16 GB) and never touches `torch_xla`. Fixed 40 x 80 shapes are
retained anyway: the corpus max is 37 prompts / 73 responses, so nothing truncates, and a
constant shape keeps CUDA memory planning stable across the ~80-step unroll.

Persistence
-----------
Best and latest checkpoints are written to a local `checkpoints/` dir and mirrored to
`--drive_dir` when a mounted Google Drive path is given, so a Colab disconnect does not lose
progress. `--resume` (or an automatic warm-start from `cerebro_spheroid_1m_best.pt`) loads prior
weights through `load_compatible_checkpoint`, which reports shape-skipped tensors instead of
silently leaving them random.

Usage
-----
    # In a Colab notebook with a T4/A100 runtime and the bundled zip unpacked:
    !python3 colab/colab_cognitive.py --scale laptop --steps 2000 --batch_size 4 \
        --eval_interval 50 --drive_dir /content/drive/MyDrive/cerebro_checkpoints

    # Portable smoke test (proves the path on any machine, no GPU needed):
    python3 colab/colab_cognitive.py --scale small --steps 4 --eval_batches 20 \
        --batch_size 4 --dtype float32 --device cpu
"""

import os
import sys
import time
import json
import math
import hashlib
import argparse

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, ".hf_cache"))
sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn.functional as F

from src.model import CerebroSpheroid
from src.data_loaders import get_specialized_cognitive_stream
from src.spheroid_v3 import training_rollout

CONFIGS = {
    # The shipped architecture, 146,618,503 params.
    "laptop": dict(surface_dim=640, num_shells=16, radial_dim=256, num_shortcut_tracts=40),
    # 31M, used by overfit_sanity.py; for a fast end-to-end check on any device.
    "small": dict(surface_dim=192, num_shells=8, radial_dim=64, num_shortcut_tracts=16),
}

VOCAB_SIZE = 50257  # gpt2
PROMPT_LEN = 40     # corpus max is 37
RESP_LEN = 80       # corpus max is 73


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_device(requested, auto_dtype):
    """Pick a CUDA device when present. Never resolve an XLA device here."""
    if requested:
        dev = torch.device(requested)
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    if auto_dtype:
        if dev.type == "cuda" and torch.cuda.is_bf16_supported():
            return dev, torch.bfloat16
        return dev, torch.float32
    return dev, None


def to_fixed_shape(prompts, targets, eos_id, prompt_len, resp_len, dropped):
    """Re-pad a streamed batch to constant widths; refuse to silently truncate."""
    if prompts.shape[1] > prompt_len or targets.shape[1] > resp_len:
        dropped[0] += 1
        return None
    p = prompts[:, :prompt_len]
    t = targets[:, :resp_len]
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
def evaluate_fixed(model, eval_batches, thinking_cycles):
    """Token-weighted held-out CE over the frozen batch list. Selection reads this, never the
    training step's own tiny-batch loss."""
    model.eval()
    loss_sum, n_tokens = 0.0, 0
    for prompts, targets in eval_batches:
        logits = training_rollout(model, prompts, targets.shape[1], thinking_cycles=thinking_cycles)
        per_batch = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1), ignore_index=-100
        )
        valid = int((targets != -100).sum())
        loss_sum += float(per_batch) * valid
        n_tokens += valid
    model.train()
    return loss_sum / max(n_tokens, 1), n_tokens


def collect_eval_batches(stream, eos_id, n_batches, prompt_len, resp_len, dropped, device):
    """Draws the frozen held-out set once and moves it to device. Never refreshed."""
    out = []
    tokens = 0
    while len(out) < n_batches and tokens < 40000:
        p, t = next_fixed(stream, eos_id, prompt_len, resp_len, dropped)
        out.append((p.to(device), t.to(device)))
        tokens += int((t != -100).sum())
    return out, tokens


def write_sidecar(ckpt_path, payload):
    """Provenance beside the weights, not inside them, so the bare state_dict stays
    `weights_only=True` loadable."""
    with open(ckpt_path + ".meta.json", "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def save_all(state, local_path, drive_dir, provenance):
    torch.save(state, local_path)
    write_sidecar(local_path, provenance)
    if drive_dir:
        dpath = os.path.join(drive_dir, os.path.basename(local_path))
        torch.save(state, dpath)
        write_sidecar(dpath, provenance)


def find_warm_start(explicit, local_ckpt, drive_best):
    if explicit:
        return explicit if os.path.exists(explicit) else None
    for cand in (local_ckpt, drive_best,
                 os.path.join(ROOT_DIR, "checkpoints", "cerebro_spheroid_1m_best.pt")):
        if cand and os.path.exists(cand):
            return cand
    return None


def main():
    ap = argparse.ArgumentParser(description="Train CerebroSpheroid on a Colab GPU")
    ap.add_argument("--scale", default="laptop", choices=list(CONFIGS))
    ap.add_argument("--steps", type=int, default=2000, help="Optimizer steps for this session")
    ap.add_argument("--batch_size", type=int, default=4, help="Micro-batch; 4 fits a T4 with the 80-step unroll")
    ap.add_argument("--grad_accum", type=int, default=2, help="Effective batch = batch_size * grad_accum")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--thinking_cycles", type=int, default=2)
    ap.add_argument("--eval_interval", type=int, default=50)
    ap.add_argument("--eval_batches", type=int, default=64)
    ap.add_argument("--min_improvement", type=float, default=0.002)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--prompt_len", type=int, default=PROMPT_LEN)
    ap.add_argument("--resp_len", type=int, default=RESP_LEN)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float32"],
                    help="auto = bfloat16 on bf16-capable GPUs (A100/V100s), float32 elsewhere (T4)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--drive_dir", default=None,
                    help="Mounted Google Drive dir to mirror checkpoints into")
    ap.add_argument("--resume", default=None, help="Checkpoint to warm-start weights from")
    ap.add_argument("--save_prefix", default="cerebro_spheroid_colab")
    ap.add_argument("--train_parquet", default=os.path.join(ROOT_DIR, "data", "cerebro_cognitive", "cerebro_cognitive_train.parquet"))
    ap.add_argument("--val_parquet", default=os.path.join(ROOT_DIR, "data", "cerebro_cognitive", "cerebro_cognitive_val.parquet"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    torch.manual_seed(args.seed)

    # Dtype resolution only when --dtype auto; an explicit choice is honoured verbatim.
    device, auto_dtype = None, (args.dtype == "auto")
    if args.device:
        device = torch.device(args.device)
        auto_dtype = False
    if auto_dtype:
        device, dtype = resolve_device(None, True)
    else:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    for path in (args.train_parquet, args.val_parquet):
        if not os.path.exists(path):
            raise SystemExit(
                f"missing {path}\nUpload the cognitive parquets (they are in the bundled zip under "
                f"data/cerebro_cognitive/) or pass --train_parquet/--val_parquet explicitly."
            )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.vocab_size != VOCAB_SIZE:
        raise SystemExit(f"expected gpt2 vocab {VOCAB_SIZE}, got {tokenizer.vocab_size}")
    eos_id = tokenizer.eos_token_id
    train_md5, val_md5 = md5_of(args.train_parquet), md5_of(args.val_parquet)
    if train_md5 == val_md5:
        raise SystemExit("train and val parquets are the same file; that is not a holdout")

    drive_dir = None
    if args.drive_dir and os.path.isdir(os.path.dirname(args.drive_dir.rstrip("/"))):
        drive_dir = args.drive_dir.rstrip("/")
        os.makedirs(drive_dir, exist_ok=True)
    local_dir = os.path.join(ROOT_DIR, "checkpoints")
    os.makedirs(local_dir, exist_ok=True)

    dropped = [0]
    train_stream = get_specialized_cognitive_stream(
        tokenizer, batch_size=args.batch_size, parquet_path=args.train_parquet,
        max_prompt_len=args.prompt_len, max_resp_len=args.resp_len, shuffle_seed=args.seed)
    val_stream = get_specialized_cognitive_stream(
        tokenizer, batch_size=args.batch_size, parquet_path=args.val_parquet,
        max_prompt_len=args.prompt_len, max_resp_len=args.resp_len,
        shuffle_seed=args.seed + 1, allow_synthesis_fallback=False)

    model = CerebroSpheroid(vocab_size=VOCAB_SIZE, **CONFIGS[args.scale])
    warm = find_warm_start(args.resume, os.path.join(local_dir, f"{args.save_prefix}_best.pt"),
                           os.path.join(drive_dir, f"{args.save_prefix}_best.pt") if drive_dir else None)
    warm_note = "none (fresh init)"
    if warm:
        ok = model.load_compatible_checkpoint(warm, device="cpu")
        warm_note = f"{warm} (compatible={ok})"
    model.to(device=device, dtype=dtype)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())

    print("=" * 80)
    print("  CEREBROSPHEROID COGNITIVE TRAINING ON COLAB GPU")
    print("=" * 80)
    print(f"Device        : {device} | dtype={dtype} | scale={args.scale}")
    print(f"Parameters    : {n_params:,}")
    print(f"Objective     : training_rollout (zero-feedback) | grad_accum={args.grad_accum} "
          f"(effective batch {args.batch_size * args.grad_accum})")
    print(f"Warm start    : {warm_note}")
    print(f"Train / Val   : {os.path.basename(args.train_parquet)} ({train_md5[:8]}) / "
          f"{os.path.basename(args.val_parquet)} ({val_md5[:8]})")
    print(f"Fixed shapes  : prompt {args.prompt_len} x response {args.resp_len} | "
          f"Drive mirror: {drive_dir or 'off'}")

    eval_batches, eval_tokens = collect_eval_batches(
        val_stream, eos_id, args.eval_batches, args.prompt_len, args.resp_len, dropped, device)
    if len(eval_batches) < 8 or eval_tokens < 2000:
        raise SystemExit(
            f"held-out set froze only {len(eval_batches)} batches / {eval_tokens} tokens; too small "
            f"to select on. Raise --eval_batches.")
    print(f"Held-out set  : {len(eval_batches)} batches, {eval_tokens:,} response tokens (frozen)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best_local = os.path.join(local_dir, f"{args.save_prefix}_best.pt")
    latest_local = os.path.join(local_dir, f"{args.save_prefix}_latest.pt")
    best_loss = float("inf")
    patience = 0
    t_start = time.time()
    global_step = 0

    def base_prov(step_now):
        return {"step": step_now, "seed": args.seed, "scale": args.scale, "dtype": str(dtype),
                "n_params": n_params, "thinking_cycles": args.thinking_cycles,
                "prompt_len": args.prompt_len, "resp_len": args.resp_len,
                "batch_size": args.batch_size, "grad_accum": args.grad_accum, "lr": args.lr,
                "device": str(device), "train_parquet_md5": train_md5, "val_parquet_md5": val_md5,
                "warm_start": warm_note}

    for step in range(1, args.steps + 1):
        global_step = step
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for _ in range(args.grad_accum):
            p, t = next_fixed(train_stream, eos_id, args.prompt_len, args.resp_len, dropped)
            p, t = p.to(device), t.to(device)
            logits = training_rollout(model, p, t.shape[1], thinking_cycles=args.thinking_cycles)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                   t.reshape(-1), ignore_index=-100) / args.grad_accum
            loss.backward()
            running += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_interval == 0 or step == 1:
            eval_ce, scored = evaluate_fixed(model, eval_batches, args.thinking_cycles)
            elapsed = time.time() - t_start
            per_step = elapsed / step
            tag = ""
            if eval_ce < best_loss - args.min_improvement:
                opening = not math.isfinite(best_loss)
                drop = 0.0 if opening else best_loss - eval_ce
                best_loss, patience = eval_ce, 0
                prov = base_prov(step)
                prov.update({"event": "first_best" if opening else "new_best",
                             "eval_ce_heldout": eval_ce, "eval_tokens": scored,
                             "eval_ppl_heldout": math.exp(min(eval_ce, 20))})
                save_all(model.state_dict(), best_local, drive_dir, prov)
                tag = " | * first best" if opening else f" | * new best (-{drop:.4f})"
            else:
                patience += 1
            print(f"step {step:5d}/{args.steps} | train CE {running:.4f} | "
                  f"HELD-OUT CE {eval_ce:.4f} (PPL {math.exp(min(eval_ce, 20)):.2f}) | "
                  f"{per_step:.2f}s/step{tag}")
            if patience >= args.patience:
                print(f"Plateau: no held-out improvement for {args.patience} evals. Stopping.")
                break

        if step % max(args.eval_interval * 5, 1) == 0:
            prov = base_prov(step)
            prov.update({"best_heldout_ce": best_loss, "elapsed_s": time.time() - t_start,
                         "eta_s": (args.steps - step) * ((time.time() - t_start) / step)})
            save_all(model.state_dict(), latest_local, drive_dir, prov)

    prov = base_prov(global_step)
    prov.update({"best_heldout_ce": best_loss, "final": True, "elapsed_s": time.time() - t_start})
    save_all(model.state_dict(), latest_local, drive_dir, prov)
    elapsed = time.time() - t_start
    print("-" * 80)
    print(f"Done in {elapsed / 3600:.2f} h. Best held-out CE {best_loss:.4f} | "
          f"skipped-too-long batches {dropped[0]}")
    print(f"Checkpoints: local {local_dir} ({os.path.basename(best_local)}, "
          f"{os.path.basename(latest_local)})" + (f" + Drive mirror {drive_dir}" if drive_dir else ""))
    print(f"Each has a .meta.json provenance sidecar. Free Colab disconnects around 90 min idle "
          f"and ~12 h; relaunch with --resume to continue from {os.path.basename(best_local)}.")
    print("=" * 80)


if __name__ == "__main__":
    main()
