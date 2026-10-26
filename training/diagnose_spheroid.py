"""
Diagnostic probe for CerebroSpheroid: quantifies the failure modes that bound reasoning quality.

Measures:
1. Held-out cross-entropy / perplexity of a checkpoint on the cognitive curriculum, against
   two input-independent floors: the global token marginal and the position-wise marginal.
   Both are fitted on the training pool (--floor_parquet) and scored on the eval pool; a floor
   fitted on the records it predicts has memorised them and is not a floor.
2. Working-memory retention: cosine similarity between the post-ingestion manifold state
   and the state after K deliberation cycles (tests whether prompt information survives).
3. Word-order sensitivity: does permuting prompt tokens change the efferent readout?
4. Readout input-dependence: cross-question KL against the same-question-scrambled
   reference, which is the scale that says whether the answer is being read at all.
5. Verbal report: rank of the correct answer token in the readout distribution.
6. Selection-criterion noise: the spread of the loss estimate `train_scale_1m.py` uses to
   decide whether to keep a checkpoint, measured on a model that cannot improve at all.

Note: item 4 of the original list (effective context length) was never implemented; the
harness measures 1, 2, 3 as written and adds 4-6, which is what running the model surfaced.
"""

import os
os.environ.setdefault("HF_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".hf_cache")))

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import hashlib
import math
import random
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.model import CerebroSpheroid
from src.data_loaders import get_specialized_cognitive_stream


def build(scale_dims, vocab_size, device):
    model = CerebroSpheroid(vocab_size=vocab_size, **scale_dims)
    return model.to(device=device, dtype=torch.float32).eval()


def state_similarity(a, b):
    a = a.flatten(1)
    b = b.flatten(1)
    return F.cosine_similarity(a, b, dim=-1).mean().item()


def retention_curve(model, prompt, cycles=(0, 1, 2, 4, 8, 16, 32), living=True):
    """Cosine similarity of the manifold state after ingestion vs after N deliberation cycles."""
    ids = torch.tensor([prompt], device=next(model.parameters()).device)
    with torch.no_grad():
        base, _ = model.ingest_stimulus_with_memory(ids, living=living)
        norms = []
        for k in cycles:
            state = model.deliberate(base, thinking_cycles=k) if k else base
            norms.append((k, state_similarity(base, state), state.norm(dim=-1).mean().item()))
    return norms


def order_sensitivity(model, tokenizer, prompt, n_permutations=16, living=True):
    """Tests whether the efferent readout depends on token order at all.

    Averages over `n_permutations` shuffles: the per-shuffle spread is large relative to
    the effect being measured, so a 5-shuffle estimate was mostly estimator noise.
    """
    device = next(model.parameters()).device
    base = tokenizer.encode(prompt)
    with torch.no_grad():
        ids = torch.tensor([base], device=device)
        state, mem = model.ingest_stimulus_with_memory(ids, living=living)
        logits_a, _, _ = model.motor_dnn(state[:, -1], state[:, 0], sensory_memory=mem)
        embs = []
        for _ in range(n_permutations):
            idx = torch.randperm(len(base)).tolist()
            shuffled = [base[i] for i in idx]
            ids_p = torch.tensor([shuffled], device=device)
            state_p, mem_p = model.ingest_stimulus_with_memory(ids_p, living=living)
            logits_p, _, _ = model.motor_dnn(state_p[:, -1], state_p[:, 0], sensory_memory=mem_p)
            embs.append(1.0 - F.cosine_similarity(logits_a, logits_p, dim=-1).mean().item())
    return sum(embs) / len(embs)


def readout_logprobs(model, ids, cycles=4, living=True):
    """Log-probs of the first emitted token after ingest -> deliberate -> emit."""
    device = next(model.parameters()).device
    with torch.no_grad():
        state, mem = model.ingest_stimulus_with_memory(torch.tensor([ids], device=device),
                                                      living=living)
        if cycles:
            state = model.deliberate(state, thinking_cycles=cycles)
        trace = torch.zeros(1, model.vocab_size, device=device)
        logits, _, _ = model.emit_next_token(state, sensory_memory=mem, refractory_trace=trace,
                                             step_idx=0)
        return F.log_softmax(logits[0].float(), dim=-1)


def _kl(a, b):
    """KL(a||b) for two log-prob vectors, computed by hand.

    F.kl_div(..., log_target=True, batchfirst=True) divides by the leading dimension, and a
    1-D vocab vector's leading dimension *is* the vocab size -- so it silently returns the
    true KL divided by 50257 and reports ~0.0000 for distributions that plainly differ.
    """
    return float((a.exp() * (a - b)).sum())


def input_dependence(model, tokenizer, prompts, cycles=4, n_permutations=6, living=True):
    """Compares divergence between different questions against divergence under scrambling.

    Cross-prompt KL is what a reader means by "the model responds to the question". The
    shuffled reference holds the word bag fixed and perturbs only order, so if the two
    numbers land in the same range the readout distinguishes questions about as well as it
    distinguishes a question from its own anagram.
    """
    dists = [readout_logprobs(model, tokenizer.encode(p.strip()), cycles, living) for p in prompts]
    pairs = [(i, j) for i in range(len(dists)) for j in range(i + 1, len(dists))]
    cross = [_kl(dists[i], dists[j]) for i, j in pairs]

    base = tokenizer.encode(prompts[0].strip())
    perm = []
    for seed in range(n_permutations):
        g = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(base), generator=g).tolist()
        perm.append(readout_logprobs(model, [base[i] for i in order], cycles, living))
    shuffled = [_kl(dists[0], pd) for pd in perm]
    return sum(cross) / len(cross), sum(shuffled) / len(shuffled)


def verbal_report(model, tokenizer, qa_pairs, cycles=(0, 4, 16), living=True, depth=100):
    """Is the correct answer token reachable from the deliberated state, and how well ranked?

    Localises the failure: a high-but-finite rank says the vocabulary head works and the
    state does not carry the answer; "not in top-k" says the answer is unreachable.
    """
    rows = []
    for prompt, answer in qa_pairs:
        ids = tokenizer.encode(prompt.strip())
        ans_ids = set(tokenizer.encode(" " + answer)) | set(tokenizer.encode(answer))
        per_cycle = []
        for c in cycles:
            lp = readout_logprobs(model, ids, c, living)
            ranked = torch.argsort(lp, descending=True)[:depth].tolist()
            hits = [ranked.index(t) + 1 for t in ans_ids if t in ranked]
            per_cycle.append(min(hits) if hits else None)
        rows.append((answer, per_cycle))
    return rows


def emit_all_steps(model, prompts, steps, thinking_cycles=3, living=True):
    """Rollout matching `spheroid_v3.training_rollout`: the refractory trace and the state
    both carry across emission steps.

    Re-passing the same post-deliberation state with a fresh trace scores a different
    function from the one the trainer optimised, and it is strictly harder to predict, so
    the reported cross-entropy is inflated relative to the trained objective.
    """
    state, sensory_memory = model.ingest_stimulus_with_memory(prompts, living=living)
    state = model.deliberate(state, thinking_cycles=thinking_cycles)
    refractory = torch.zeros(prompts.shape[0], model.vocab_size,
                             device=prompts.device, dtype=torch.float32)
    rows = []
    for step in range(steps):
        step_logits, state, refractory = model.emit_next_token(
            state, sensory_memory=sensory_memory, refractory_trace=refractory, step_idx=step
        )
        rows.append(step_logits)
    return torch.stack(rows, dim=1)


def unigram_baseline(model, stream, held_out, n_batches, width, ignore_index=-100):
    """Log-probs of the best input-independent predictor, counted from *training* targets.

    Fitting the marginal on held-out targets would flatter the baseline; fitting it on a
    disjoint draw keeps it a fair floor for what "the model learned the answer distribution
    and nothing else" would cost in cross-entropy.

    The fit draw count is a property of the *baseline*, not of the score set, and it has to be
    large: this histogram has 50,257 cells, so fitting it on a few hundred tokens leaves most
    of them at the smoothing floor and reports a ceiling the model clears for free. An
    under-fitted floor flatters the model -- the opposite direction to the one people worry
    about. Returns the number of tokens actually fitted, so the print can say so.
    """
    counts = torch.zeros(model.vocab_size, dtype=torch.float64)
    fit_tokens = 0
    for _ in range(n_batches):
        _, targets = next(stream)
        flat = targets.reshape(-1)
        flat = flat[flat != ignore_index]
        counts += torch.bincount(flat, minlength=model.vocab_size).to(torch.float64).cpu()
        fit_tokens += int(flat.numel())
    logp = torch.log((counts + 1.0) / (counts.sum() + counts.numel()))

    total, n = 0.0, 0
    pos_tot = torch.zeros(width, dtype=torch.float64)
    pos_n = torch.zeros(width, dtype=torch.float64)
    for _, targets in held_out:
        tc = targets.cpu()
        valid2d = tc != ignore_index
        loss2d = -logp[tc.clamp(min=0)] * valid2d          # padded slots contribute nothing
        total += float(loss2d.sum())
        n += int(valid2d.sum())
        w = tc.shape[1]
        pos_tot[:w] += loss2d.sum(dim=0).double()
        pos_n[:w] += valid2d.sum(dim=0).double()
    return total / max(n, 1), (pos_tot / pos_n.clamp(min=1)), logp, fit_tokens


def positional_unigram_baseline(model, stream, held_out, n_batches, width, ignore_index=-100):
    """Position-wise marginal: the best predictor allowed to know *where* it is, not *what*.

    A global unigram cannot express "slot 0 is nearly always a space and the tail is nearly
    always EOS", because that is a property of the answer template rather than of the
    question. This floor absorbs it. Whatever advantage the model keeps over this one, and
    only that, is attributable to the input rather than to response shape -- so the gap
    between the two floors measures how much of the model's headline advantage over counting
    was really just learning where answers start and stop.
    """
    counts = torch.zeros(width, model.vocab_size, dtype=torch.float64)
    seen = torch.zeros(width, dtype=torch.float64)
    for _ in range(n_batches):
        _, targets = next(stream)
        tc = targets.cpu()
        w = min(tc.shape[1], width)
        for i in range(w):
            col = tc[:, i]
            col = col[col != ignore_index]
            if col.numel():
                counts[i] += torch.bincount(col, minlength=model.vocab_size).to(torch.float64)
                seen[i] += col.numel()
    total = counts.sum(dim=1, keepdim=True)
    logp = torch.log((counts + 1.0) / (total + counts.shape[1]))   # rows never seen go uniform

    tot, n = 0.0, 0
    pos_tot = torch.zeros(width, dtype=torch.float64)
    pos_n = torch.zeros(width, dtype=torch.float64)
    for _, targets in held_out:
        tc = targets.cpu()
        w = min(tc.shape[1], width)
        valid2d = tc[:, :w] != ignore_index
        loss2d = -logp[torch.arange(w).unsqueeze(0).expand(tc.shape[0], w),
                       tc[:, :w].clamp(min=0)] * valid2d
        tot += float(loss2d.sum())
        n += int(valid2d.sum())
        pos_tot[:w] += loss2d.sum(dim=0).double()
        pos_n[:w] += valid2d.sum(dim=0).double()
    return tot / max(n, 1), (pos_tot / pos_n.clamp(min=1)), seen


def selection_noise(model, stream, grad_accum, thinking_cycles, n_trials, device, living=True):
    """Draws of `loss_val` exactly as `train_scale_1m.py` forms it, against a frozen model.

    The trainer saves a new best when the current step's `loss_val` beats the running best by
    0.005 nats. `loss_val` is the mean cross-entropy of `grad_accum` micro-batches of
    `batch_size` samples -- here 16 records, re-drawn every step. On a model held fixed, the
    spread across draws is pure sampling noise, so it bounds what that rule can detect: a
    frozen model that triggers "New Best Checkpoint" at rate r is a rule whose threshold sits
    inside r's noise band.
    """
    vals = []
    with torch.no_grad():
        for _ in range(n_trials):
            accum = 0.0
            for _ in range(grad_accum):
                prompts, targets = next(stream)
                prompts, targets = prompts.to(device), targets.to(device)
                logits = emit_all_steps(model, prompts, targets.shape[1],
                                        thinking_cycles=thinking_cycles, living=living)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       targets.reshape(-1), ignore_index=-100)
                accum += float(loss) / grad_accum
            vals.append(accum)
    return vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cerebro_spheroid_1m_best.pt")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337,
                        help="Seeds the data shuffle and permutation draws. Without it the "
                             "held-out CE and order figures drift between runs.")
    parser.add_argument("--living", action="store_true", default=True,
                        help="Ingest from the living resting state (training default).")
    parser.add_argument("--quiescent", dest="living", action="store_false")
    parser.add_argument("--thinking_cycles", type=int, default=3,
                        help="Deliberation cycles before emission. Trainers default to 3, "
                             "so this must track the checkpoint's training config.")
    parser.add_argument("--noise_trials", type=int, default=24,
                        help="Draws for section [6]. Each trial costs grad_accum rollouts, "
                             "so this is the slowest part of the instrument.")
    parser.add_argument("--pos_floor_batches", type=int, default=512,
                        help="Batches used to fit the position-wise marginal in [1c]. This "
                             "floor has width x vocab cells, so fitting it on the same 128 "
                             "responses as the global unigram leaves tail positions near "
                             "uniform and hands the model a free advantage there.")
    parser.add_argument("--uni_floor_batches", type=int, default=0,
                        help="Batches used to fit the global unigram in [1b]. 0 means the whole "
                             "floor pool. Used to default to --batches, which tied a baseline's "
                             "fit size to the score-set size and left a 50,257-cell histogram "
                             "fitted on ~1,300 tokens -- a starved floor, and therefore one that "
                             "flatters the model. The 20.10 ladder's 7.8244 was measured that "
                             "way; pass --uni_floor_batches 64 with --batches 64 to reproduce it.")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="Micro-batches per optimizer step, matching the trainer; sets "
                             "the sample size of the selection estimator under test.")
    parser.add_argument("--parquet", type=str, default=None,
                        help="Curriculum parquet to score against. Defaults to the held-out "
                             "cerebro_cognitive_val.parquet when it exists, else the train file. "
                             "Every headline number in RESEARCH_NOTES 20.10/20.12 was measured on "
                             "the unseeded 50,000-row draw that happened to be on disk; pass "
                             "data/cerebro_cognitive/legacy_train_unseeded_e16c6f47.parquet to "
                             "reproduce those exactly.")
    parser.add_argument("--floor_parquet", type=str, default=None,
                        help="Pool the input-independent floors in [1b]/[1c] are fitted from. "
                             "Defaults to the sibling *_train.parquet when scoring a *_val file, "
                             "and to the scored file otherwise. A floor fitted on the targets it "
                             "then predicts memorises them: it reads strong, and the model's gap "
                             "over it shrinks by construction.")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dims = dict(surface_dim=640, num_shells=16, radial_dim=256, num_shortcut_tracts=40)
    model = build(dims, tokenizer.vocab_size, device)
    loaded = model.load_compatible_checkpoint(args.checkpoint, device=device)
    n_params = sum(p.numel() for p in model.parameters())

    print("=" * 78)
    print("  CEREBROSPHEROID DIAGNOSTIC")
    print("=" * 78)
    print(f"Checkpoint       : {args.checkpoint} (loaded={loaded})")
    print(f"Parameters       : {n_params:,}")
    trivially_untrained = [n for n, _ in model.named_parameters()
                           if n not in torch.load(args.checkpoint, map_location="cpu", weights_only=True)]
    print(f"Untrained params : {len(trivially_untrained)} tensors "
          f"({sum(p.numel() for n, p in model.named_parameters() if n in trivially_untrained):,} values)")
    print(f"Ingest mode      : {'living resting state' if args.living else 'quiescent'} "
          f"| seed {args.seed}")

    # ---- 1. Held-out loss -------------------------------------------------
    # NOTE: model.forward() emits one logit row per *prompt* token, so it cannot be scored
    # against response targets. The trained objective is ingest -> deliberate -> emit loop.
    # The evaluation set is pulled once and frozen: the stream shuffles, so scoring whatever
    # batches come next would report a different cross-entropy on every consecutive run.
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    eval_parquet = args.parquet
    if eval_parquet is None:
        val_default = os.path.join(root, "data", "cerebro_cognitive", "cerebro_cognitive_val.parquet")
        eval_parquet = val_default if os.path.exists(val_default) else os.path.join(
            root, "data", "cerebro_cognitive", "cerebro_cognitive_train.parquet")
    eval_md5 = hashlib.md5(open(eval_parquet, "rb").read()).hexdigest() if os.path.exists(eval_parquet) else "absent"
    print(f"Eval file        : {os.path.relpath(eval_parquet, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))}")
    print(f"Eval file md5    : {eval_md5}")
    if args.parquet is None and "_val." not in os.path.basename(eval_parquet):
        print("                     [!] no holdout on disk -- this is IN-SET for a model trained"
              " on this file. Build one: python3 src/generate_specialized_dataset.py")
    stream = get_specialized_cognitive_stream(
        tokenizer, batch_size=args.batch_size, parquet_path=eval_parquet,
        shuffle_seed=args.seed, allow_synthesis_fallback=False)

    # ---- Floor fit source --------------------------------------------------
    # A baseline is entitled to see the training distribution; it is not entitled to see the
    # records it will be scored on. Fitting the histograms on the scored targets memorises
    # them, which makes the floor optimistically strong and the model look worse by
    # construction -- the exact asymmetry the whole ladder exists to avoid.
    #
    # When floors come from the same file being scored (the legacy reproduction path), the fit
    # stream is deliberately the *same generator object*, not a fresh one: its draws continue
    # past the frozen batch list, so fit and score records stay disjoint within one epoch.
    # Two independently seeded generators over a small file would hand the floor the scored
    # records verbatim.
    floor_parquet = args.floor_parquet
    if floor_parquet is None:
        if "_val." in os.path.basename(eval_parquet):
            sibling = eval_parquet.replace("_val.", "_train.")
            floor_parquet = sibling if os.path.exists(sibling) else eval_parquet
        else:
            floor_parquet = eval_parquet
    floors_same_file = os.path.abspath(floor_parquet) == os.path.abspath(eval_parquet)
    if floors_same_file:
        fit_stream = stream
    else:
        fit_stream = get_specialized_cognitive_stream(
            tokenizer, batch_size=args.batch_size, parquet_path=floor_parquet,
            shuffle_seed=args.seed, allow_synthesis_fallback=False)
    floor_md5 = (hashlib.md5(open(floor_parquet, "rb").read()).hexdigest()
                 if os.path.exists(floor_parquet) else "absent")
    try:
        floor_pool_rows = pq.ParquetFile(floor_parquet).metadata.num_rows
    except Exception:
        floor_pool_rows = None
    # Unigram fit + positional fit + the scored batch list, all drawn from this pool.
    uni_fit_batches = args.uni_floor_batches
    if uni_fit_batches == 0:
        # One full pass of the floor pool, and not a record more: wrapping into a second epoch
        # would double-count identical targets in the histogram for no information gain.
        uni_fit_batches = (args.pos_floor_batches if floor_pool_rows is None
                           else max(1, floor_pool_rows // args.batch_size))
    records_from_floor_pool = (uni_fit_batches + args.batches + args.pos_floor_batches) * args.batch_size

    held_out = [next(stream) for _ in range(args.batches)]
    total, count = 0.0, 0
    loss_sum, loss_n = 0.0, 0
    # Response width is per-batch padding, so the position accumulators must be sized to the
    # widest batch in the set rather than the first one.
    width = max(t.shape[1] for _, t in held_out)
    pos_tot = torch.zeros(width, dtype=torch.float64)
    pos_n = torch.zeros(width, dtype=torch.float64)
    with torch.no_grad():
        for prompts, targets in held_out:
            prompts, targets = prompts.to(device), targets.to(device)
            logits = emit_all_steps(model, prompts, targets.shape[1],
                                    thinking_cycles=args.thinking_cycles, living=args.living)
            per_token = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                        targets.reshape(-1), ignore_index=-100,
                                        reduction="none").view(targets.shape)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                   ignore_index=-100)
            total += loss.item()
            count += 1
            valid = (targets != -100)
            masked = per_token.masked_fill(~valid, 0)
            loss_sum += float(masked.sum())
            loss_n += int(valid.sum())
            w = targets.shape[1]
            pos_tot[:w] += masked.sum(dim=0).double().cpu()
            pos_n[:w] += valid.sum(dim=0).double().cpu()
    mean_loss = total / max(count, 1)
    # Token-weighted, so it is comparable with the unigram floor below. The per-batch mean
    # is short responses averaging the same as long ones, which reads optimistically.
    token_ce = loss_sum / max(loss_n, 1)
    print(f"\n[1] Cognitive-curriculum CE : {token_ce:.4f} ({loss_n:,} tokens) | "
          f"PPL: {math.exp(min(token_ce, 20)):.2f}")
    base = os.path.basename(eval_parquet)
    if "_val." in base:
        print("    HELD-OUT: scored on a seeded split that is disjoint from the training pool by")
        print("    construction (RESEARCH_NOTES 20.13). That is a real holdout only for weights")
        print("    trained after the split existed -- the shipped 1m checkpoints were fitted to a")
        print("    different, unseeded pool, so for them this is out-of-sample records from the")
        print("    same generator rather than a designed holdout.")
    elif "legacy" in base:
        print("    FROZEN LEGACY POOL: the unseeded 50,000-row draw the figures in RESEARCH_NOTES")
        print("    20.10 and 20.12 were measured on, pinned here by md5 so they still reproduce.")
        print("    Out-of-sample for the shipped weights by accident, not by protocol; the")
        print("    canonical path now holds a seeded split instead. The state dict carries no step")
        print("    count; the run behind these weights was tens of steps, so this is early-")
        print("    training loss, not a plateau.")
    else:
        print("    IN-SET RISK: this is a training parquet. If the checkpoint was fitted to it,")
        print("    the figure below is in-sample. Pass --parquet pointing at *_val.parquet.")
    print(f"    per-batch mean (the older, optimistic convention): {mean_loss:.4f}")
    print(f"    (objective matched to training: trace and state carried across steps, "
          f"{args.thinking_cycles} thinking cycles)")
    per_pos = (pos_tot / pos_n.clamp(min=1))
    shown = "  ".join(f"t{i}:{v:.2f}" for i, v in enumerate(per_pos[:8]))
    print(f"    per-position CE: {shown} ... over {int((pos_n > 0).sum())} scored positions")
    first = float(per_pos[0])
    rest = float(pos_tot[1:].sum() / pos_n[1:].sum().clamp(min=1))
    print(f"    first response token {first:.4f} | later positions {rest:.4f} nats")

    # ---- 1b. Floor ladders ------------------------------------------------
    uni_ce, uni_pos, _, uni_fit_tokens = unigram_baseline(
        model, fit_stream, held_out, n_batches=uni_fit_batches, width=width)
    print(f"\n[1b] Cross-entropy floors on the same held-out targets (both token-weighted)")
    print(f"    floors fitted on : {os.path.relpath(floor_parquet, root)} (md5 {floor_md5})")
    if floors_same_file:
        pool = f"{floor_pool_rows:,} rows" if floor_pool_rows is not None else "size unknown"
        print(f"                     same pool as the score set ({pool}); the fit draws continue "
              f"past the {args.batches} frozen batches, so they are disjoint as long as "
              f"{records_from_floor_pool:,} records fit in one epoch.")
        if floor_pool_rows is not None and records_from_floor_pool > floor_pool_rows:
            print(f"                     [!] THEY DO NOT. The fit stream wrapped and scored records "
              f"it had already fitted on -- floors below are optimistically strong and the "
              f"model's gap is understated. Pass --floor_parquet pointing at a larger pool.")
    else:
        print("                     disjoint from the scored records: a fair floor.")
    print(f"    uniform over vocab : {math.log(model.vocab_size):.4f}")
    thin = "  <- FIT ON A HANDFUL; this floor is under-fitted and flatters the model" \
        if uni_fit_tokens < 20_000 else ""
    print(f"    training unigram   : {uni_ce:.4f}  (best input-independent predictor; "
          f"{uni_fit_tokens:,} fit tokens{thin})")
    print(f"    model              : {token_ce:.4f}")
    gap = uni_ce - token_ce
    verdict = ("the readout carries task signal" if gap > 0.05
               else "NOISE-LEVEL: the readout is indistinguishable from counting tokens")
    print(f"    model beats unigram by {gap:+.4f} nats ({verdict})")

    # The aggregate can be won on filler positions while losing on the answer itself, so
    # compare floor and model position by position -- t0 is the first answer token.
    print("    per-position CE (model vs unigram floor, lower is better):")
    print("      pos |   model  unigram |    gain")
    for i in range(min(6, width)):
        m, u = float(per_pos[i]), float(uni_pos[i])
        print(f"      t{i:<3d}| {m:7.3f} {u:8.3f} | {u - m:+7.3f}")

    # ---- 1c. The stronger input-independent floor --------------------------
    pos_ce, pos_floor, pos_seen = positional_unigram_baseline(
        model, fit_stream, held_out, n_batches=args.pos_floor_batches, width=width)
    gain_pos = pos_ce - token_ce
    absorbed = uni_ce - pos_ce
    verdict2 = ("the readout carries signal beyond response shape" if gain_pos > 0.05
                else "NOISE-LEVEL once response shape is priced in")
    share = f"{absorbed / gap:.0%}" if abs(gap) > 1e-9 else "n/a"
    print(f"\n[1c] Position-wise marginal floor (knows the slot, not the question)")
    n_scored = int((pos_n > 0).sum())
    weak = int((pos_seen[:n_scored] < 100).sum())
    print(f"    global unigram : {uni_ce:.4f}")
    print(f"    positional     : {pos_ce:.4f}  ({int(pos_seen[0])} fit tokens at t0; "
          f"{weak}/{n_scored} scored slots have <100, where this floor is under-fitted and "
          f"flatters the model)")
    print(f"    model          : {token_ce:.4f}")
    print(f"    model beats the positional floor by {gain_pos:+.4f} nats ({verdict2})")
    print(f"    of the {gap:+.4f} nats won over the global unigram, {absorbed:+.4f} ({share}) "
          f"is reproducible without reading the input")
    print("      pos |   model  unigram position |    gain")
    for i in range(min(6, width)):
        m, u, p = float(per_pos[i]), float(uni_pos[i]), float(pos_floor[i])
        print(f"      t{i:<3d}| {m:7.3f} {u:8.3f} {p:9.3f} | {p - m:+7.3f}")

    # The articulations are chain-of-thought, so the numeric answer sits in the tail -- and
    # the tail is where this floor has the least fitting data. Splitting the slots by support
    # shows whether the model's residual advantage is real or an artefact of a starved floor.
    print("    gain by floor support (slots with <100 fit tokens have a floor biased high):")
    for label, idxs in (("well-fitted", torch.nonzero((pos_seen >= 100) & (pos_n > 0)).flatten()),
                        ("under-fitted", torch.nonzero((pos_seen < 100) & (pos_n > 0)).flatten())):
        if idxs.numel() == 0:
            continue
        tokens = pos_n[idxs].sum()
        m = float(pos_tot[idxs].sum() / tokens)
        p = float((pos_floor[idxs] * pos_n[idxs]).sum() / tokens)
        print(f"      {label:12} | {idxs.numel():3d} slots | {int(tokens):6d} tokens | "
              f"model {m:7.3f} floor {p:7.3f} | gain {p - m:+7.3f}")

    # ---- 2. Working memory retention --------------------------------------
    prompt = tokenizer.encode(
        "Maya buys 6 notebooks priced at $4 each. She pays with a $50 bill. "
        "Exactly how much change does she receive?", max_length=64, truncation=True)
    print("\n[2] Working-memory retention after K deliberation cycles")
    print("    cycles | state cosine | mean shell norm")
    for k, cos, norm in retention_curve(model, prompt, living=args.living):
        print(f"    {k:6d} | {cos:12.4f} | {norm:14.4f}")

    # ---- 3. Word-order sensitivity ----------------------------------------
    delta = order_sensitivity(model, tokenizer, "The box holds 12 red apples and 5 green pears.",
                              living=args.living)
    print(f"\n[3] Word-order sensitivity (1 - cos(logits) under permutation): {delta:.6f}")

    # ---- 4. Does the readout respond to the question at all? ---------------
    qa_pairs = [
        ("Maya has 8 markers. She buys 4 more and gives 3 away. How many markers does she have?", "9"),
        ("A train travels 60 miles in 1 hour. How many miles does it travel in 3 hours?", "180"),
        ("If 1 book costs $5, how much do 4 books cost?", "20"),
    ]
    cross, shuffled = input_dependence(model, tokenizer, [q for q, _ in qa_pairs],
                                       living=args.living)
    print(f"\n[4] Readout input-dependence (KL, nats)  |  scale reference: token-order "
          f"sensitivity of one prompt")
    print(f"    mean KL(question_a || question_b)        : {cross:.4f}")
    print(f"    mean KL(question || its own anagram)     : {shuffled:.4f}")
    print("    If these are the same order of magnitude, changing the question moves the")
    print("    readout about as little as scrambling its words does.")

    print("\n[5] Verbal report: best rank of the correct answer token in the readout")
    print("    answer | rank @ cycles 0, 4, 16   (searched top-100)")
    for answer, ranks in verbal_report(model, tokenizer, qa_pairs, living=args.living):
        cells = ", ".join(str(r) if r else "out" for r in ranks)
        print(f"    {answer:>6} | {cells}")

    # ---- 6. Can the checkpoint rule see a 0.005-nat improvement? -----------
    # Frozen model, so every draw below is the same weights. Any "improvement" is the
    # 16-record estimator disagreeing with itself.
    vals = selection_noise(model, stream, grad_accum=args.grad_accum,
                           thinking_cycles=args.thinking_cycles, n_trials=args.noise_trials,
                           device=device, living=args.living)
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(var)
    running_min, best_events = float("inf"), 0
    for v in vals:
        if v < running_min - 0.005:
            best_events += 1
        running_min = min(running_min, v)
    print(f"\n[6] Selection-criterion noise on a FROZEN model "
          f"({args.grad_accum} micro-batches x {args.batch_size} = "
          f"{args.grad_accum * args.batch_size} records per draw, {n} draws)")
    print(f"    loss_val: mean {mean:.4f} | std {std:.4f} | range [{min(vals):.4f}, {max(vals):.4f}]")
    threshold_note = f"({0.005 / std:.1%} of one std)" if std > 0 else "(estimator is degenerate)"
    print(f"    the old rule's improvement threshold   : 0.0050 nats {threshold_note}")
    print(f"    'New Best Checkpoint' events on frozen weights: {best_events}/{n} "
          f"({best_events / n:.1%} of evals)")
    print("    These draws are against weights that never change, so any event above is the")
    print("    estimator re-drawing easier records rather than the model improving. Each event")
    print("    also reset the patience counter, which is how a run of tens of steps was stopped.")
    print("    HISTORICAL MEASUREMENT: train_scale_1m.py no longer uses this estimator. Since")
    print("    RESEARCH_NOTES 20.13 it selects on a frozen held-out batch list, and --min_improvement")
    print("    defaults to 0.002 nats against a token-weighted holdout CE instead. Read the numbers")
    print("    above as the audit that motivated the change, not as the current rule.")

    print("=" * 78)


if __name__ == "__main__":
    main()
