# CerebroSpheroid V4: Revised Specification and Implementation Plan

Refined from `cerebro_spheroid_v4_plan.md` (antigravity brain dir). Every figure below was measured on
this repo or fetched from the Hugging Face dataset APIs, not estimated.

## 0. What was wrong in the incoming plan

| # | Claim in the old plan | What the check shows |
|---|---|---|
| 1 | Scale 162M to 280M because it is "ideal for a 16 GB T4" | The GPU is not the binding limit. A 50 unit Colab budget buys ~25 T4 hours, which at the measured rate in section 2 yields single-digit millions of tokens. Chinchilla (20 tokens/param) sets the largest model that budget can train at well under 1M parameters. The old plan sized the model orders of magnitude past what its own compute can feed. |
| 2 | "100% World Knowledge ingestion, zero withheld entities" | Removes the only generalisation test the project has. Withheld entities are what make a held-out CE and a nearest-name probe mean anything. Kept as an optional flag, default unchanged. |
| 3 | SmolLM mix `cosmopedia=0.35, fineweb_edu=0.30` | `HuggingFaceTB/cosmopedia-v2` is not a repo id; it is config `cosmopedia-v2` inside `HuggingFaceTB/smollm-corpus`. Sizes from the datasets-server API: cosmopedia-v2 122.4 GB / 39,134,000 rows; fineweb-edu-dedup 550.1 GB / 190,168,005 rows; whole corpus 673 GB / 237M rows. Colab gives ~100 GB ephemeral disk and drops it on disconnect. `train_v3_full.py:786` already tells the user to download `cosmopedia-v2` for 100M+ step runs, which is unachievable advice at this budget. |
| 4 | Memory budget "~3.92 GiB static, 12 GiB headroom, B=16/32 at window 256" | The table omits fp32 gradients (1.12 GB at 280M) and the CUDA context plus cuDNN workspace (~0.5-1.0 GB). It also sizes only *static* memory. Measured activation cost is 0.26 GiB per batch item per 64-token window in fp32, 0.13 GiB in bf16 (`train_v3_full.py:475`), so B=32 at window 256 needs 4.2-8.4 GiB of activations alone. |
| 5 | "FP16 AMP with float32 master weights" listed as a V3 learning | V3 is bf16-first (`--dtype bf16` in the Colab notebook), and bf16 needs no GradScaler. The fp16 path does exist now (scaler at `train_v3_full.py:860`, `unscale_` before clip), so the "gradient unscale failure" the bullet guards against is already fixed. Keep bf16 as the default and state the fp16 branch as separate. |
| 6 | Shell zones "20-31", "8-19", "0-7" with K=24 | Those ranges describe 32 shells. With K=24 they are 0-7 (thalamic), 8-19 (associative), 20-23 (superficial). Zone bounds must be derived from `num_shells`, not hard-coded. |
| 7 | Vocabulary "32,000 (BPE), kept compatible with GPT-2" | GPT-2 BPE is 50257, and `train_v3_full.py:698` loads `"gpt2"`. V3 laptop is 50257 for that reason. Setting 32000 while keeping that tokenizer produces out-of-range embedding indices. The `2b`/`260m` presets do declare 32000, which is exactly why they cannot be used with the current tokenizer. |
| 8 | "SDPA gives ~2x speedup and ~40% VRAM reduction" | V3 uses no `scaled_dot_product_attention` anywhere, so it is a new change, but the cost is not attention. It is the sequential rollout: one Python decoder pass per scored token, over a short attended sequence at `window=64`. The advertised gain lands on the component that is not the bottleneck. Also, on a T4 (sm_75) the FlashAttention backend is unavailable; only the memory-efficient backend applies. |
| 9 | Anti-collapse rectifier inserted between every thinking cycle | Unmeasured, and expensive: variance-axis expansion needs a covariance eigendecomposition per cycle. It collides with the existing `_homeostasis` RMS clamp and the per-shell `log_leak` that were tuned together. See section 5. |
| 10 | Multi-modal (text, vision, audio) as a build-now feature | No image or audio corpus exists in `src/corpus.py` (`SOURCES = openhermes, lmsys, knowledge, cognitive, gsm8k`), nothing is downloaded or sized, and the plan defines no loss for those streams and no output head for them. Input-only sensory nodes with no data are untestable. Deferred; see section 6. |
| 11 | Phase 5 acceptance is "parameter count & VRAM benchmark test" | This project has already been fooled by a shape-level check passing while lifted weights were mis-scaled. Acceptance must include logit parity and the existing invariance tests. |
| 12 | `--save_every 200` carried over from the A100 notebook | A free-tier session is cut near 12 hours and a preemption is a kill, not a pause. 200 steps at ~9 s is 30 minutes of unmirrored compute lost per disconnect. Set `--save_every 40`. |

## 1. What V4 should actually be

V4 is V3 plus the things measurement said was missing, at a size the available compute can train, with the
pause/resume and holdout machinery already in `train_v3_full.py` carried over unchanged.

Ordered by what limits the result:

1. **Token throughput per GPU-hour.** The rollout is a Python loop, so one backward pass covers `window`
   sequential decoder calls. This is the bottleneck, not layer count. V4's first contribution is a cached
   prompt/feedback encoding and a preallocated working-memory buffer (section 4.2), targeting 2x or more
   tokens/s at equal loss. If that does not move, no architecture change matters.
2. **Size matched to budget.** Two presets, both explicit about what feeds them:
   - `v4_small`: 640 / 16 shells / 256 radial / 40 tracts, vocab 50257 = **162,272,317 params** (measured).
     Identical to V3 laptop, so it is the control run and the one the free-T4 budget can actually train.
   - `v4_wide`: 896 / 24 / 384 / 48, vocab 50257 = **327,851,093 params** (measured). Requires a rented
     A100-class budget with `--max_steps` set so total tokens exceed several billion; otherwise do not run it.
   The old plan's "~280M" is not reachable as stated. At 896/24/384/48 it is 327.9M with the GPT-2 vocab;
   the 278,776,277 figure only appears if you also swap to a 32k tokenizer, which changes the embedding
   matrix and the whole data pipeline. Pick one configuration and state its vocab.
3. **One corpus, honest splits.** Keep the row-group holdout (every 50th group) and the 10-of-50 entity
   holdout. Add SmolLM sources only as bounded subsets (section 3).
4. **Re-derived LR and schedule.** Surface 640 to 896 and shells 16 to 24 change residual gain and depth;
   the SGD 0.015 that works on V3 does not transfer. Sweep LR on the new preset before spending a session.

## 2. Budget arithmetic (do this before choosing anything else)

Measured on the laptop RTX 3050 4GB, bf16, SGD, `--batch_size 1 --window 64 --grad_accum 8`: **~8.5 s per
step.** Each gradient step runs 1 x 8 = 8 records, each costing `window` = 64 sequential decoder passes,
so **512 passes per step, ~60 passes/s.** Use your own first log line to convert, since the scaling is
what matters, not the constant:

```
passes_per_step = batch_size * grad_accum * window        # 8 * 2 = 16 records x 256 = 4096 in the notebook
s_per_step      = read from the log after 10 steps
passes_per_hour  = 3600 * passes_per_step / s_per_step
```

- s/step grows with `batch_size x grad_accum x window` because each record costs `window` sequential
  passes, and V3's per-pass cost grows with the window as well, because the working-memory buffer is
  extended by one entry per emitted token (`train_v3_full.py:1046`) and each pass re-embeds feedback
  (line 1036). Doubling the window costs more than doubling the time.
- Free Colab: 50 units at ~1.96 units/hour on a T4 is ~25 GPU-hours, and a session is cut near 12 hours,
  so those 25 hours arrive as 3-4 sessions, each of which must resume cleanly.
- The old plan's `B=8, window=256, grad_accum=2` is the A100 notebook's live cell: 4,096 sequential passes
  per gradient step. At 3-5x the laptop's passes/s a T4 needs roughly 75-125 hours for 20,000 such steps,
  three to five times the whole budget.

Consequence: **the free-T4 budget is an ablation budget at V3 scale, not a training budget for V4 scale.**
At 25 hours the arithmetic gives order 10-20M tokens, for which Chinchilla-optimal is well under 1M
parameters; V3's 162M is already far past that, and 280M is 1.7x further past it. Use the T4 hours for
throughput and architecture comparisons at `v4_small`, where the comparison is cheap and the result is
interpretable. Train `v4_wide` on rented A100/H100 time or Kaggle dual-T4, and say so in the plan rather
than sizing for hardware that cannot run it.

## 3. Data plan (corrected)

| Source | Real id and size | What V4 does with it |
|---|---|---|
| OpenHermes 2.5 | `teknium/OpenHermes-2.5`, 1M rows, local + auto-download | Keep, weight 0.45 |
| LMSYS-Chat-1M | local parquet, 6 shards | Keep, weight 0.35 |
| World Knowledge (canonical) | `data/corpus/world_knowledge_train.parquet` 3,045 train / 708 val, 50 withheld entities | Keep, weight 0.15, entity holdout intact |
| Cognitive | `cerebro_cognitive_train.parquet` | Keep, weight 0.05 |
| GSM8K | `openai/gsm8k` main/train, 7.47K rows, auto-download at `train_v3_full.py:650` | Optional, weight <= 0.10. Too small to carry more without memorising |
| Cosmopedia-v2 | config `cosmopedia-v2` of `HuggingFaceTB/smollm-corpus`: 122.4 GB, 39.1M rows | **Subset only.** Read a fixed number of shards into a bounded tokenized cache (200-500M tokens). Never the whole config |
| FineWeb-Edu | `fineweb-edu-dedup`: 550.1 GB, 190.2M rows | Skip at this budget. It only pays off above ~10B tokens, which needs rented compute |

Rules for `src/corpus.py`:
1. Add `cosmopedia` / `fineweb_edu` to `SOURCES` **with a `max_shards` / `max_records` cap**, so a mix
   weight cannot silently start pulling a 122 GB stream. The default cap must keep the whole mix inside
   the disk limit. `parse_mix` (`corpus.py:175`) raises on unknown names, so new sources must land in
   `SOURCES` before any notebook or script references them.
2. Streaming and cached loading are different modes and the plan must pick one per source. With
   `streaming=True` there is no row-group random access, so the row-group holdout plan cannot be built
   from it; either download shards, or hold out by a hash of record text, and record which in the manifest.
3. Exact resume is currently O(skipped records): `Mixture` advances each source iterator by
   `consumed[name]` records on restart (`corpus.py:219`), which means a long streamed run rereads and
   discards everything it already trained on, every session. Add a shard/row-group cursor to the state
   file so resume cost stays bounded. With 12-hour session cuts this is not a nicety.
4. Tokenize once, train many times: write a tokenized cache keyed by (source, tokenizer, max_len) plus a
   content hash, so re-tokenizing 1M rows is not paid per session.
5. Every run writes dataset file hashes and the mix string into the state sidecar, so a later loss number
   can be labelled held-out or in-set from evidence rather than from reading pipeline code.
6. Distilled-corpus caution: Cosmopedia is Mixtral-generated textbook-style text. It raises information
   density per token and it also teaches one generator's register. Keep at least one human-conversation
   source (OpenHermes/LMSYS) in the mix so probes measure transfer, not imitation of a single style.

## 4. Architecture changes worth making

### 4.1 Keep from V3, unchanged
`TemporalSensoryEncoder` causal wavelets (k=3, k=5), `ResidualVolumetricCore` leaky update with per-shell
`log_write`/`log_leak`, thalamic `injection_gate`, `_homeostasis`, multi-shell softmax tap, premotor plan
buffer. These have ablation evidence behind them. The volumetric core stays config-switchable because
`--no-volumetric-core` produced the best held-out CE in the V3 ablation; V4 must not re-tie that knot by
assuming the core helps.

### 4.2 Throughput fixes (highest value, do first)
- The rollout re-embeds feedback tokens on every step (line 1036). Encode prompt/feedback once and index.
- `sensory_memory = torch.cat([...])` per emitted token (line 1046) makes cost superlinear in window. Use
  a preallocated buffer of size `window` with a length mask.
- Put the per-step `.item()` loss accumulation behind the existing `--probe_tokens` gate so no CUDA sync
  happens every step.
- Optional `torch.compile` on `deliberate_step`, measured rather than assumed: compile time must be
  amortised across the session.

### 4.3 SDPA
Replace manual `Q@K^T` attention with `F.scaled_dot_product_attention`. Expected gain is small at
`window=64` and larger at 256. On Turing (T4) the flash backend is unavailable, so benchmark on the
actual target device. Accept on measured passes/s at fixed seed, not on vendor multipliers.

### 4.4 SwiGLU
Converting the two `nn.Linear` feedforward layers to SwiGLU costs 1.5x params at equal hidden width, so
shrink hidden width by 2/3 to stay parameter-neutral. It is a quality-per-parameter bet, not a speedup,
and it breaks checkpoint transplant (4.6).

### 4.5 Refractory dynamics already exist
The old plan's "neuronal refractory dynamics (anti-repetition trace)" is the existing `refractory` state
threaded through `emit_next_token`. V4 work is tuning its decay and gain, and stating current values in
the spec so the change is measurable.

### 4.6 Zoning must be transplant-safe
Zones are index ranges over `num_shells`. V3 checkpoints are 16 shells; a 24-shell model cannot load them
by index. Either keep K=16 and re-label zones, or write an explicit lifting step that remaps shell indices
and re-initialises the added shells, then verify it the way this project requires: **logit parity on a
fixed input, not a shape check**.

## 5. Manifold rectification: measure first, then decide

Do not add `AntiCollapseRectifier` as specified.
1. Expanding variance along under-used principal axes needs the covariance eigendecomposition of a
   `[B, K, D]` state per thinking cycle. At K=24, D=896 that dominates step time, and it is inside the
   sequential loop, so it is paid `window` times per record.
2. Mean-centering plus axis rescaling is a normalization layer placed inside a recurrent loop. It changes
   what `_homeostasis` and `log_leak` were tuned against, and both act on the same tensor.
3. "False manifold collapse" is an unmeasured hypothesis about this model.

Cheaper and decisive:
1. **Diagnostics only, no gradient path.** Per-shell effective rank from the covariance trace ratio
   `(sum lambda)^2 / sum lambda^2`, per-shell RMS, adjacent-shell cosine similarity, and the fraction of
   shells whose `write_gate` has saturated. Compute at eval intervals under `no_grad`, log to the status
   JSON next to CE.
2. If the diagnostics show collapse (effective rank near 1, RMS pinned at the clamp), the first lever is
   the one the ablation already supports: disable the volumetric core, or raise `leak` / lower `write`.
3. If a regularizer is still needed, add a **soft loss term** (rank or entropy bonus, weight tuned on the
   probe) instead of a forward-path transform. A loss term cannot destabilise inference; a forward-path
   transform can, and it also silently changes eval-time behaviour, which is the class of bug this
   project has already been bitten by.

## 6. Multi-modal: design now, build later

The thalamic-gate concept is right; the ordering is wrong. Vision and audio need a corpus, a token budget,
and a loss. None exist here, and a 50-unit T4 budget cannot feed an image-text mix.

What to do in V4 at near-zero cost:
1. Fix the forward signature so later modalities do not require surgery: `forward(tokens=None,
   pixel_values=None, audio_values=None)`, all-absent taking the existing `init_state` path. Text-only
   numerics must not shift; verify with an exact-equality test.
2. Write `src/sensory_nodes.py` with `VisionSensoryEncoder` (16x16 conv, stride 16, 196 patches, project
   to `surface_dim`, inject into a designated outer-shell sector) and `AudioSensoryEncoder` (strided 1D
   depthwise convs k=7/11 down to 25-50 Hz frames), each with an input-shape contract and a test that
   zero injection reproduces text-only logits exactly.
3. Build `ThalamicSensoryGater` on **text first**: gate on prompt quality (padding ratio, tokenizer
   round-trip mismatch, perplexity band). That is testable with data already on disk and is the same
   mechanism the modal gates will need.
4. Only then add a modality, with its own milestone and its own budget: gated cross-attention per
   modality plus a next-token loss on aligned captions. Until a caption corpus is downloaded and
   tokenized this is a design, not a phase.

## 7. Trainer requirements carried over (do not regress)

Create `train_v4_full.py` by renaming `train_v3_full.py` and adding an `--arch v4` branch, not by
rewriting, so these keep working:
- pause sentinel, SIGINT-then-second-SIGINT-abort, `--max_minutes` so a session exits before the cut;
- exact resume: per-source consumed counts, cursor replay, RNG state (load to CPU and
  `detach().cpu().to(torch.uint8)` before `set_rng_state`; keep the payload from holding GPU copies),
  best CE and patience across sessions;
- `--save_every 40`, and `--drive_dir` mirroring, since only mirrored state survives a preemption;
- token-weighted held-out CE with per-source breakdown as the only checkpoint criterion. `--probe_tokens`
  must default off for long runs; selecting best on one 3-batch probe was already shown to prefer worse
  weights;
- memory preflight with the calibrated activation coefficient (0.13 GiB bf16 / 0.26 GiB fp32 per batch
  item per 64-token window), which prints the arithmetic and refuses configurations that cannot fit, and
  the OOM branch must not advance `step` without a completed update, or the cosine schedule moves on for
  a step that never trained;
- `build_v4_model` must take dims from CLI args. The `--arch v2` branch (`train_v3_full.py:714-716`)
  hard-codes 640/16/256/40 whatever the arguments say, and line 712 builds `build_spheroid_v3("laptop")`
  with the preset fixed in source. That is how a preset flag ends up silently unused; also note the
  `2b`/`260m` presets declare `vocab_size=32000` while the trainer loads the 50257-token gpt2 tokenizer,
  so those presets cannot be selected without changing the tokenizer too.
- DDP: rank 0 writes state only, per-rank seeds, `consumed` counts are per-rank not global, and
  `find_unused_parameters=True` must be paired with a note that it tolerates entire modality branches
  receiving no gradient on a given step.

## 8. Phases, each with the check that ends it

**Phase 0 (controlling, no new code).** Run V3 laptop to a loss plateau on the current mix with throughput
logged. Exit: passes/s (or tokens/hour), held-out CE at plateau, status JSON recorded. This is the
baseline for every claim after it.

**Phase 1: throughput.** Apply 4.2. Exit: same seed and steps, >=2x passes/s versus Phase 0, and held-out
CE within 0.01 of Phase 0. A speedup that changes the loss is a different model, not a faster one.

**Phase 2: data.** Capped SmolLM subset, tokenized cache, provenance manifest, bounded resume cursor.
Exit: mix builds; leakage assertion reports 0 duplicated questions and 0 withheld entities in train; peak
disk under the Colab limit; a 200-step run showing the new sources in the per-source CE breakdown.

**Phase 3: V4 preset.** `v4_wide` (896/24/384/48 at vocab 50257, 327.9M) with SDPA, on a budget that can
feed it. Exit: measured param count matches the table, transplant verified by logit parity or declared
impossible, LR swept, preflight accepts the intended batch and window.

**Phase 4: collapse diagnostics.** Section 5 steps 1-2 only. Exit: a report of effective rank and shell
RMS over a real run plus a written decision on whether any rectifier is needed.

**Phase 5: modality interfaces.** Section 6 steps 1-3. Exit: text-only parity exact, gater tested on
text, encoders unit-tested for zero-injection equivalence.

**Phase 6 (deferred, needs its own budget):** real vision/audio training, with the corpus named and paid
for before code is written.

## 9. Acceptance criteria (replacing "parameter count and VRAM benchmark")

1. `python3 training/test_feedback_channel.py`: the shift-invariance invariants must stay 0.0. That test
   catches the alignment-bug class that silently corrupts the feedback channel.
2. Transplant logit-parity test on a fixed input. Shape checks are not sufficient here.
3. Held-out leakage assertion from the corpus manifest: 0 duplicated questions across train and val, 0
   withheld entities present in train.
4. Throughput at fixed seed versus Phase 0, and passes/s versus the ~60 passes/s laptop anchor in
   section 2 (re-measured per device, since a T4 and a 3050 differ here).
5. Generation probes on withheld entities, scored on whether the *pattern* transfers. That is the only
   knowledge test that stays meaningful if a future version trains on 100% of a dataset.
