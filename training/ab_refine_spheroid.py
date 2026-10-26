"""
Controlled A/B evaluation of the spheroid refinements.

Trains CerebroSpheroid (V2) and CerebroSpheroidV3 on the *identical* batch sequence,
with the same initialization seed, optimizer, LR schedule and step budget, then reports:

  - held-out cross-entropy / perplexity
  - working-memory retention vs deliberation depth
  - word-order sensitivity of the efferent readout
  - wallclock + peak VRAM

Usage:
    python3 training/ab_refine_spheroid.py --steps 150 --config small
    python3 training/ab_refine_spheroid.py --steps 60 --config laptop --resume-v2
"""

import os
os.environ.setdefault("HF_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".hf_cache")))

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import math
import random
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.model import CerebroSpheroid
from src.spheroid_v3 import CerebroSpheroidV3
from src.data_loaders import get_specialized_cognitive_stream


CONFIGS = {
    # dims, vocab, tract count  -- identical between V2 and V3
    "small": dict(surface_dim=192, num_shells=8, radial_dim=64, num_shortcut_tracts=16),
    "laptop": dict(surface_dim=640, num_shells=16, radial_dim=256, num_shortcut_tracts=40),
}

# Each ablation enables exactly one refinement on top of the V2 baseline.
# New mechanisms initialize as exact no-ops, so every variant starts at V2 behaviour.
VARIANTS = {
    "v2": (CerebroSpheroid, {}),
    "order": (
        CerebroSpheroidV3,
        dict(use_residual_core=False, use_plan=False, use_shell_tap=False),
    ),
    "core": (
        CerebroSpheroidV3,
        dict(use_temporal=False, use_plan=False, use_shell_tap=False),
    ),
    "plan": (
        CerebroSpheroidV3,
        dict(use_temporal=False, use_residual_core=False, use_shell_tap=False),
    ),
    # Shipped default: order-aware encoder + residual core + premotor plan +
    # multi-shell tap.
    "v3": (CerebroSpheroidV3, {}),
    # Ablation arm: drop the leaky residual potential, keep everything else. Order
    # sensitivity collapses to V2's level here (0.0579 vs 0.1726), which is how the
    # core was identified as the component actually carrying the refinement.
    "v3_no_core": (CerebroSpheroidV3, dict(use_residual_core=False)),
}


# ---------------------------------------------------------------- data

def take_batches(stream, count, prompt_len, resp_len):
    batches = []
    while len(batches) < count:
        prompts, targets = next(stream)
        prompts = prompts[:, :prompt_len]
        targets = targets[:, :resp_len]
        if targets.shape[1] < 4 or (targets != -100).sum() < 8:
            continue
        batches.append((prompts, targets))
    return batches


# ---------------------------------------------------------------- rollouts

def rollout_v2(model, prompts, steps):
    state, sensory_memory = model.ingest_stimulus_with_memory(prompts)
    state = model.deliberate(state, thinking_cycles=3)
    logits = []
    for t in range(steps):
        step_logits, state = model.emit_next_token(state, sensory_memory=sensory_memory, step_idx=t)
        logits.append(step_logits)
    return torch.stack(logits, dim=1)


def rollout_v3(model, prompts, steps):
    return model.training_logits(prompts, steps, thinking_cycles=3)


def rollout_for(kind):
    """Both arms must score the same objective; V3 rewrites it internally only."""
    return rollout_v2 if kind == "v2" else rollout_v3


def evaluate(model, kind, batches):
    rollout = rollout_for(kind)
    device = next(model.parameters()).device
    total, count = 0.0, 0
    with torch.no_grad():
        for prompts, targets in batches:
            prompts, targets = prompts.to(device), targets.to(device)
            logits = rollout(model, prompts, targets.shape[1])
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100
            )
            total += loss.item()
            count += 1
    return total / max(count, 1)


# ---------------------------------------------------------------- diagnostics

def state_similarity(a, b):
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=-1).mean().item()


def retention(model, prompt_ids, cycles=(0, 1, 2, 4, 8, 16, 32)):
    with torch.no_grad():
        base, _ = model.ingest_stimulus_with_memory(prompt_ids)
        curve = []
        for k in cycles:
            state = model.deliberate(base, thinking_cycles=k) if k else base
            curve.append((k, round(state_similarity(base, state), 4)))
    return curve


def order_sensitivity(model, kind, tokenizer, text, n=4):
    """Mean KL(original || permuted) between efferent next-token distributions, in nats."""
    device = next(model.parameters()).device
    base = tokenizer.encode(text)[:24]
    with torch.no_grad():
        def readout(ids):
            t = torch.tensor([ids], device=device)
            state, memory = model.ingest_stimulus_with_memory(t)
            if kind == "v2":
                logits, _, _ = model.motor_dnn(state[:, -1], state[:, 0], sensory_memory=memory)
            else:
                keys, values = model.motor_dnn.prime_sensory(memory)
                logits, _, _, _ = model.motor_dnn(state, sensory_kv=(keys, values))
            return F.log_softmax(logits.float(), dim=-1)

        ref = readout(base)
        deltas = []
        rng = random.Random(0)
        for _ in range(n):
            shuffled = base[:]
            rng.shuffle(shuffled)
            alt = readout(shuffled)
            deltas.append((ref.exp() * (ref - alt)).sum(dim=-1).mean().item())
    return sum(deltas) / len(deltas)


# ---------------------------------------------------------------- training

def train(variant, dims, vocab, train_batches, eval_batches, args, device, warm_start=None):
    kind = "v2" if variant == "v2" else "v3"
    rollout = rollout_for(kind)
    cls, extra = VARIANTS[variant]
    torch.manual_seed(args.seed)
    model = cls(vocab_size=vocab, **dims, **extra)
    if warm_start:
        model.load_compatible_checkpoint(warm_start, device="cpu")
        print(f"  warm-started from {warm_start}")

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model.to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    losses = []
    model.train()
    for step in range(1, args.steps + 1):
        prompts, targets = train_batches[(step - 1) % len(train_batches)]
        prompts, targets = prompts.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
            logits = rollout(model, prompts, targets.shape[1])
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100
            )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            print(f"    [{variant}] step {step}: non-finite loss/grad, skipping update", flush=True)
            optimizer.zero_grad(set_to_none=True)
            losses.append(losses[-1] if losses else 0.0)
            scheduler.step()
            continue
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        if step % max(1, args.steps // 6) == 0 or step == 1:
            window = losses[-min(20, len(losses)):]
            print(f"    [{variant}] step {step:4d}/{args.steps} | loss {sum(window)/len(window):.4f} "
                  f"| {time.time()-t0:6.1f}s", flush=True)

    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
    model.eval()
    with torch.no_grad():
        held_out = evaluate(model, kind, eval_batches)
        recalled = evaluate(model, kind, train_batches[:8])
    tail = sum(losses[-min(20, len(losses)):]) / min(20, len(losses))
    return model, dict(
        held_out_loss=held_out,
        train_recall_loss=recalled,
        tail_train_loss=tail,
        wallclock_s=wall,
        peak_vram_mb=peak,
        params=sum(p.numel() for p in model.parameters()),
    )


def sample(model, kind, tokenizer, prompts, device, cycles=3, max_tokens=24):
    outs = []
    for text in prompts:
        ids = torch.tensor([tokenizer.encode(text)[:96]], device=device)
        with torch.no_grad():
            if kind == "v2":
                toks = model.solve_with_active_thinking(
                    ids, thinking_cycles=cycles, max_answer_tokens=max_tokens,
                    temperature=0.0, repetition_penalty=1.3, eos_token_id=tokenizer.eos_token_id,
                )
            else:
                toks = model.generate(
                    ids, thinking_cycles=cycles, max_new_tokens=max_tokens, temperature=0.0,
                    repetition_penalty=1.3, eos_token_id=tokenizer.eos_token_id,
                )
        outs.append(tokenizer.decode(toks[0].tolist(), skip_special_tokens=True).strip())
    return outs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--prompt_len", type=int, default=64)
    parser.add_argument("--resp_len", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--config", type=str, default="small", choices=list(CONFIGS))
    parser.add_argument("--train_batches", type=int, default=64)
    parser.add_argument("--eval_batches", type=int, default=16)
    parser.add_argument("--resume_v2", action="store_true", help="Warm-start from the laptop V2 checkpoint")
    parser.add_argument("--variants", type=str, default="v2,v3",
                        help=f"Comma-separated subset of {list(VARIANTS)}")
    parser.add_argument("--out", type=str, default="checkpoints/ab_refinement.json")
    args = parser.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        parser.error(f"unknown variants {unknown}; choose from {list(VARIANTS)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab = tokenizer.vocab_size
    dims = CONFIGS[args.config]

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    stream = get_specialized_cognitive_stream(tokenizer, batch_size=args.batch_size)
    train_batches = take_batches(stream, args.train_batches, args.prompt_len, args.resp_len)
    eval_batches = take_batches(stream, args.eval_batches, args.prompt_len, args.resp_len)

    print("=" * 84)
    print("  SPHEROID REFINEMENT A/B  (identical data order, init seed, optimizer, budget)")
    print("=" * 84)
    print(f"Config {args.config}: {dims} | vocab {vocab}")
    print(f"Steps {args.steps} | batch {args.batch_size} | lr {args.lr:.1e} | "
          f"prompt {args.prompt_len} | response {args.resp_len} | device {device}")
    print(f"Train batches {len(train_batches)} | held-out batches {len(eval_batches)}")
    print("-" * 84)

    warm = "checkpoints/cerebro_spheroid_1m_best.pt" if args.resume_v2 and dims == CONFIGS["laptop"] else None
    results, models = {}, {}
    for variant in variants:
        print(f"\n>>> {variant.upper()}")
        model, metrics = train(variant, dims, vocab, train_batches, eval_batches, args, device, warm_start=warm)
        models[variant] = model
        results[variant] = metrics

    # ---- diagnostics ----------------------------------------------------
    probe = tokenizer.encode(
        "Maya buys 6 notebooks priced at $4 each. She pays with a $50 bill. "
        "Exactly how much change does she receive?"
    )[:32]
    probe_t = torch.tensor([probe], device=device)
    test_prompts = [
        "Maya has 8 markers. She buys 4 more and gives 3 away. How many markers does she have?",
        "A train travels 60 miles in 1 hour. How many miles does it travel in 3 hours?",
    ]
    for variant, model in models.items():
        kind = "v2" if variant == "v2" else "v3"
        results[variant]["retention"] = retention(model, probe_t)
        results[variant]["order_sensitivity"] = round(
            order_sensitivity(model, kind, tokenizer, "The box holds 12 red apples and 5 green pears."), 5
        )
        results[variant]["samples"] = sample(model, kind, tokenizer, test_prompts, device)

    # ---- report ---------------------------------------------------------
    print("\n" + "=" * 96)
    print("  RESULTS (identical data order, init seed, optimizer, step budget)")
    print("=" * 96)
    header = f"{'metric':<28}" + "".join(f"{v:>13}" for v in variants)
    print(header)
    for key, label, fmt in [
        ("params", "parameters", "{:,.0f}"),
        ("held_out_loss", "held-out CE", "{:.4f}"),
        ("train_recall_loss", "train recall CE", "{:.4f}"),
        ("tail_train_loss", "tail train CE", "{:.4f}"),
        ("order_sensitivity", "order KL (nats)", "{:.3e}"),
        ("wallclock_s", "train wallclock (s)", "{:.1f}"),
        ("peak_vram_mb", "peak VRAM (MB)", "{:.0f}"),
    ]:
        print(f"{label:<28}" + "".join(f"{fmt.format(results[v][key]):>13}" for v in variants))

    print("\nWorking-memory retention (cosine to post-ingestion state)")
    print(f"{'cycles':<10}" + "".join(f"{v:>13}" for v in variants))
    cycles = [c for c, _ in results[variants[0]]["retention"]]
    for i, cycle in enumerate(cycles):
        print(f"{cycle:<10}" + "".join(f"{results[v]['retention'][i][1]:>13.4f}" for v in variants))

    for variant in variants:
        print(f"\n{variant.upper()} samples:")
        for prompt, out in zip(test_prompts, results[variant]["samples"]):
            print(f"  Q: {prompt[:70]}")
            print(f"  A: {out}")

    if "v2" in results and "v3" in results:
        gain = (results["v2"]["held_out_loss"] - results["v3"]["held_out_loss"]) / results["v2"]["held_out_loss"] * 100
        print(f"\nHeld-out cross-entropy improvement (v3 vs v2): {gain:+.2f}%")
    print("=" * 96)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"config": args.config, "args": vars(args), "results": results}, fh, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
