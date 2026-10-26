"""
Overfit sanity check: can CerebroSpheroid fit a handful of examples at all?

A 146M checkpoint scores 7.55 nats on data it has already seen dozens of times, which is
barely under the 7.82-nats counting floor. That is underfitting, not overfitting, so the
first question is not "which architecture refinement helps" but "does the training path
work at all". This script trains a fresh small model on a fixed, tiny dataset two ways:

  emission -- the production objective: ingest the prompt, deliberate, then predict the
              whole response from the state, with no emitted token fed back.
  lm      -- the control: `CerebroSpheroid.forward`, which injects each real token's flux
              at its own step, so predicting response position t sees response tokens <t.

Both arms see identical examples, model, optimiser and step budget. If `lm` drives CE to
~0 while `emission` stalls, the bottleneck is the zero-feedback readout itself rather than
capacity, data, or the optimiser.

Usage:
    python3 training/overfit_sanity.py --examples 1 --steps 800
    python3 training/overfit_sanity.py --examples 8 --steps 1500 --arms lm,emission
"""

import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, ".hf_cache"))

import sys
sys.path.insert(0, ROOT_DIR)

import argparse
import contextlib
import math
import random

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.model import CerebroSpheroid
from src.data_loaders import get_specialized_cognitive_stream
from src.spheroid_v3 import training_rollout

CONFIGS = {
    "small": dict(surface_dim=192, num_shells=8, radial_dim=64, num_shortcut_tracts=16),
    "laptop": dict(surface_dim=640, num_shells=16, radial_dim=256, num_shortcut_tracts=40),
}


def valid_tokens(targets_row, eos_id):
    """Response tokens without the -100 padding, keeping the trailing EOS."""
    keep = [int(t) for t in targets_row if int(t) != -100]
    return keep


def emission_loss(model, batch, thinking_cycles):
    prompts, targets = batch
    logits = training_rollout(model, prompts, targets.shape[1], thinking_cycles=thinking_cycles)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                           ignore_index=-100)


def lm_loss(model, batch):
    """Teacher-forced next-token loss over prompt+response, scored on response positions.

    `forward` emits logits[t] after consuming token t, so logits[t] predicts token t+1.
    """
    prompts, targets = batch
    losses = []
    for p_row, t_row in zip(prompts, targets):
        prompt = [int(x) for x in p_row if int(x) != model_eos]
        resp = valid_tokens(t_row, model_eos)
        if not resp:
            continue
        seq = torch.tensor([prompt + resp], device=prompts.device)
        logits = model.forward(seq)[:, :-1]                      # predict tokens 1..n-1
        nxt = seq[0, 1:]
        # Only positions whose successor lies in the response region are supervised.
        boundary = len(prompt) - 1
        mask = torch.arange(nxt.numel(), device=nxt.device) >= boundary
        if int(mask.sum()) == 0:
            continue
        sel = F.cross_entropy(logits[0][mask], nxt[mask], reduction="sum")
        losses.append(sel / int(mask.sum()))
    if not losses:
        raise RuntimeError("no scoreable response positions")
    return torch.stack(losses).mean()


def train_arm(arm, model_factory, batches, args, device):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = model_factory().to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # `train_scale_1m.py` runs its forward pass inside bf16 autocast; matching that here is
    # the point of the experiment, since fp32 fitting proves little about the shipped run.
    amp_dtype = {"float32": None, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    def autocast():
        if amp_dtype is None or device.type == "cpu":
            return contextlib.nullcontext()
        return torch.amp.autocast(device_type=device.type, dtype=amp_dtype)

    trajectory = []
    for step in range(1, args.steps + 1):
        # Cycle the whole fixed set so --examples N trains on N rather than on batch 0.
        b = batches[(step - 1) % len(batches)]
        with autocast():
            loss = (emission_loss(model, b, args.thinking_cycles) if arm == "emission"
                    else lm_loss(model, b))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % max(args.steps // 10, 1) == 0:
            trajectory.append((step, float(loss.item()), float(grad_norm)))

    model.eval()
    with torch.no_grad():
        vals = []
        for b in batches:
            with autocast():
                l = emission_loss(model, b, args.thinking_cycles) if arm == "emission" else lm_loss(model, b)
            vals.append(float(l.item()))
    return sum(vals) / len(vals), trajectory, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=1, help="Batches to overfit (each is 2 pairs)")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--config", type=str, default="small", choices=list(CONFIGS))
    parser.add_argument("--thinking_cycles", type=int, default=3)
    parser.add_argument("--arms", type=str, default="lm,emission")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "bfloat16", "float16"],
                        help="bfloat16 reproduces the shipped trainer's autocast setting.")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model_eos = tokenizer.eos_token_id

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    stream = get_specialized_cognitive_stream(tokenizer, batch_size=2)
    # Move once here: `forward` derives its state device from input_ids.device, so a CPU
    # batch against CUDA parameters raises a device mismatch deep in the pacemaker.
    batches = [(p.to(device), t.to(device)) for p, t in
               (next(stream) for _ in range(args.examples))]

    def factory():
        return CerebroSpheroid(vocab_size=tokenizer.vocab_size, **CONFIGS[args.config])

    n_params = sum(p.numel() for p in factory().parameters())
    print("=" * 78)
    print("  OVERFIT SANITY CHECK")
    print("=" * 78)
    print(f"Config: {args.config} ({n_params:,} params) | {args.examples} batch(es) "
          f"= {2 * args.examples} pairs | {args.steps} steps | lr {args.lr} | dtype {args.dtype}")
    print("A model that cannot drive this loss to ~0 on data it sees every step is not")
    print("limited by data or capacity -- the training path or the objective is at fault.\n")

    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        final, traj, _ = train_arm(arm, factory, batches, args, device)
        print(f"[{arm}] final CE on the overfit set: {final:.4f} (PPL {math.exp(min(final, 20)):.2f})")
        print("      step |    loss   grad_norm")
        for s, l, g in traj:
            print(f"    {s:6d} | {l:7.4f} {g:10.4f}")
        verdict = "FIT (memorised)" if final < 0.10 else "DID NOT FIT"
        print(f"    => {verdict}\n")

    # Reference: the same fixed set scored by an input-independent counter, so "FIT" has a
    # number to compare against rather than an absolute.
    counts = torch.zeros(tokenizer.vocab_size)
    for _, t in batches:
        flat = t.reshape(-1).cpu()
        counts += torch.bincount(flat[flat != -100], minlength=tokenizer.vocab_size).float()
    logp = torch.log((counts + 1.0) / (counts.sum() + counts.numel()))
    tot, n = 0.0, 0
    for _, t in batches:
        flat = t.reshape(-1).cpu()
        keep = flat != -100
        tot += float(-logp[flat[keep]].sum())
        n += int(keep.sum())
    print(f"unigram floor on this fixed set: {tot / n:.4f} nats")
    print("=" * 78)
