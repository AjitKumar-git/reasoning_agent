# CerebroSpheroid

A non-autoregressive language model that holds state in a volumetric manifold rather than in a
token sequence. It ingests a prompt, deliberates for a fixed number of cycles, and emits tokens
one at a time from the resulting 3-D state.

The design constraint that separates it from a Transformer: the emission loop never receives an
external output token. Each step feeds back the model's own refractory trace and its own evolving
manifold state. Nothing from outside the skull is written into its input stream, so
`x_{t+1} != [x_{1..t}, y_t]`.

## Status, stated before any architecture claim

The shipped checkpoints have barely been trained. `cerebro_spheroid_1m_best.pt` scores 7.5468 nats
on the cognitive curriculum, against 7.7165 for a position-wise token histogram, and it is behind
that histogram at each of the first five answer slots. The state dict carries no step count; the
run behind it was a few tens of optimizer steps.

What is solid: the model loads with full tensor coverage, runs, and is measured by an instrument
that now states its own errors. What is unproven: that any of these weights reason.

`RESEARCH_NOTES.md` accounts for the detail: §20.10 and §20.12 explain how a 5.4002-loss claim came
to be attached to these weights, §20.13 records the selection-rule and dataset fixes, and §20.14
lists the reproduce commands.

## Model

| | |
|---|---|
| Trainable parameters | 146,618,503 |
| Tensors in the saved state dict | 37 |
| Values in the state dict | 181,242,583 (parameters plus buffers) |
| Checkpoint size on disk | 293,251,003 bytes (~0.29 GB); ~0.73 GB once loaded as float32 |
| `surface_dim` | 640 |
| `num_shells` | 16 concentric shells |
| `radial_dim` | 256 |
| `num_shortcut_tracts` | 40 axonal tracts |
| Vocabulary | 50,257 (gpt2 tokenizer, `pad_token = eos_token`) |
| Parameter coverage of `cerebro_spheroid_1m_best.pt` | 0 tensors missing, verified on load |

### Stages

Three stages, in `src/model.py`:

1. `SensoryInputDNN`. SwiGLU projections, no positional encoding (a bag of tokens).
2. `VolumetricBrainCore`. 16 concentric shells. `intra_shell` is an einsum over each shell,
   `reentrant_field` propagates a radial wave outward and back, and shortcut tracts move state
   between non-adjacent shells by `index_add_`.
3. `MotorOutputDNN`. Reads the outermost shell plus the sensory memory trace and produces logits.

The core update law is

```
S_{t+1} = RMSNorm( gamma * S_t + (1 - gamma) * tanh(Flow(S_t)) )
```

which pins the state RMS at `sqrt(640) = 25.2982`. The diagnostic prints that number at every
deliberation depth, and it does not move: the manifold cannot inflate, so any information the model
carries is in direction rather than magnitude.

`CerebroSpheroid.load_compatible_checkpoint()` (`src/model.py`) reads a bare `state_dict()` with
`weights_only=True` and maps by name, so a V2 file loads into a V3 model where the tensor shapes
agree.

There is no dropout and no batchnorm anywhere in V2. The only `self.training` branches are the
gradient checkpointing in `circulate_mid_tensors`, which performs identical arithmetic, and
`tick_living_dynamics(spontaneous_noise=...)`, which no caller in this repository passes as `True`.
`model.eval()` is therefore numerically the trained function, and evaluation-mode measurement is
valid rather than a reduced model.

### Manifold generations

Two generations ship side by side. `--arch auto` reads the checkpoint and picks one.

| | V2 (`src/model.py`) | V3 (`src/spheroid_v3.py`) |
|---|---|---|
| Sensory front-end | position-free MLP (bag of tokens) | causal depthwise conv (k=3/k=5) + learned position embedding |
| Core update | `RMSNorm(γS + (1-γ)tanh(Flow))`, norm re-pinned every cycle | `(1-λ)S + α·tanh(Flow)` with per-shell write/leak and a wide homeostatic clamp |
| Efferent tap | shell 0 and shell K-1 only | learned softmax over all shells |
| Sequence memory | phase clock + detached refractory trace | adds a premotor plan buffer driven by corollary discharge |
| Checkpoints | native | loads V2 checkpoints with 0 tensors lost via width-lifting surgery |

Every added V3 branch is zero-initialized (`conv_scale`, `pos_embed`, `gist_scale`, `packet_mix`,
`shell_scale`, `plan_write`), so a warm start is a genuine no-op rather than a random perturbation.
`training/verify_warmstart.py` asserts this end to end: on identical weights, V3 reproduces V2's
encoder field, post-ingestion state, and logits to float noise (~2e-6).

The trained objective in both generations is `training_rollout(model, prompts, response_steps,
thinking_cycles)` (`src/spheroid_v3.py`). The V2 branch carries both the refractory trace and the
evolving state across emission steps, so it is not teacher-forced on prior answer tokens.

Measured on a 1500-step equal-budget ablation (`training/ab_refine_spheroid.py`, small config, seed
1337, 256 train / 16 held-out batches):

| | held-out CE | train recall | order KL (nats) |
|---|---|---|---|
| V2 | 2.5961 | 1.9288 | 0.0644 |
| V3 | 2.5938 | 1.9261 | **0.1726** |

V3 matches V2's cross-entropy within single-seed noise while raising order sensitivity 2.7x, the
defect it exists to fix. The gain lives in the residual core: with `use_residual_core=False`, order
KL falls back to 0.0579. That table's "held-out" batches come from the same curriculum generator
the arms trained on, which is defensible for models that saw only 256 of the records but is not a
designed split; the split that ships now (§Datasets) is a separate, later fix.

**Warm-start caveat.** A V2 checkpoint gives V3 valid weights but not a valid *contract*: the
residual core integrates differently by design, so V2's trained decoder sees states it never trained
on. Exact numerical parity is asserted only with `use_residual_core=False`; the shipped
configuration should be trained from scratch. See RESEARCH_NOTES §20.7.

## Measured behaviour

Produced by `training/diagnose_spheroid.py --batches 64` on
`data/cerebro_cognitive/legacy_train_unseeded_e16c6f47.parquet` (md5 `e16c6f47…`), seed 1337, 3
thinking cycles, living ingest, 6,705 scored response tokens.

| predictor | CE (nats) | perplexity |
|---|---|---|
| uniform over vocabulary | 10.8249 | 50,257 |
| global unigram | 7.8244 | 2,501 |
| position-wise marginal | 7.7165 | 2,245 |
| `cerebro_spheroid_1m_best` | 7.5468 | 1,895 |
| `cerebro_spheroid_1m_latest` | 7.3595 | 1,571 |

The position-wise marginal is allowed to know which slot of the answer it is predicting but not what
the question was. Of the 0.2776 nats the shipped model wins over the global unigram, 0.1079 (39%) is
reproduced by that counter alone, leaving 0.1697 attributable to anything prompt-specific. Slot by
slot the residue is worse than the aggregate:

```
 pos |   model  unigram positionwise |    gain
  t0 |   8.179    8.511      6.882   |  -1.297
  t1 |   8.621    8.217      7.163   |  -1.458
  t2 |   8.141    7.961      7.014   |  -1.128
  t3 |   7.735    8.045      7.375   |  -0.360
  t4 |   8.202    8.239      7.506   |  -0.696
  t5 |   7.209    7.819      7.344   |  +0.134
```

Two caveats printed by the instrument itself. The global unigram value above was fitted on ~6,654
tokens spread over 50,257 cells, which leaves it under-fitted and therefore too weak a floor;
`--uni_floor_batches 0` refits it on the whole pool. The tail slots carry the numeric answer, and 14
of 82 scored slots have fewer than 100 fit tokens, so the floor is biased high exactly there; the 68
well-fitted slots hold 6,650 of the 6,705 tokens and still show a gain of only +0.154.

Other sections of the same run:

| probe | result | reading |
|---|---|---|
| Working-memory retention | cosine 1.0000, 0.9925, 0.9569, 0.7020, -0.3992, 0.4107, -0.0575 at 0/1/2/4/8/16/32 cycles | prompt information is largely gone by 4 cycles |
| Word-order sensitivity | 1 - cos(logits) = 0.000276 over 16 permutations | the readout is order-blind |
| Input dependence | mean KL between questions 0.1221, question vs its own anagram 0.0999 | changing the question moves the readout about as much as scrambling it |
| Verbal report | rank of "9" is 29, 78, 30 at 0, 4, 16 cycles; "180" and "20" outside the top 100 | the answer is not the most reachable token |
| Selection-rule noise | mean 7.2460, std 0.3402, range [6.3464, 8.0941] on frozen weights | the old 0.005-nat rule fired "New Best" on 4 of 24 evals of a model that could not improve |

## Datasets

Primary curriculum: `cerebro_cognitive`, generated by `src/generate_specialized_dataset.py`. Each
record is a chain-of-thought articulation of an arithmetic or counting word problem.

The generator drew from `random` with no seed until this project's last revision. Rebuilding it now
is deterministic, and the audit output (`data/cerebro_cognitive/cerebro_cognitive_manifest.json`)
records what the draw contained:

| | |
|---|---|
| Draws requested | 50,000 |
| Exact repeated stimuli folded | 29,643 (59.3%) |
| Stimuli with conflicting answers | 0 |
| Distinct stimuli | 20,357 |
| Train rows | 19,339 (md5 `777126b6…`) |
| Validation rows | 1,018 (md5 `b0c51907…`) |

The 59.3% repeat rate matters: the pool that looked like 50,000 curriculum items is about 20,400
questions replayed 2.4 times, which skews both training weight and any frequency baseline. The split
is by distinct stimulus, so no train/val pair shares a question. It is not template-disjoint and does
not claim to be: the generators reuse sentence forms deliberately.

`data/cerebro_cognitive/legacy_train_unseeded_e16c6f47.parquet` is a frozen `cp -p` copy of the
unseeded 50,000-row draw that sections 20.10 and 20.12 measured. It is the only reason those figures
still reproduce. Note that the shipped checkpoints were fitted before this split existed, so the
val file is out-of-sample records from the same generator for them, not a designed holdout.

`src/data_loaders.py` also exposes LMSYS chat, GSM8K, and SVAMP streams, and
`src/advanced_datasets.py` specialised variants. Two loader arguments exist because of past failures:
`shuffle_seed` draws epoch orderings from a private `random.Random` rather than the process global,
and `allow_synthesis_fallback=False` turns a missing parquet into an error instead of an infinite
stream of freshly invented samples, which is the difference between validating a file and validating
your own generator.

`data/` and `archive/` are gitignored, so the parquets above are local. Regenerate them from the seed
rather than assuming they are present.

## Training

```bash
python3 main.py train-1m --patience 20 --max_steps 2000 --save_optimizer
```

`train-1m` maps to `training/train_scale_1m.py` (default `--dataset specialized`), which now:

- freezes a held-out batch list at startup (`--eval_batches 128`, `--eval_max_tokens 20000`) and
  scores it token-weighted under the trained objective. Selection and early stopping both read that
  number, never the step's own loss. The step loss, which is `batch_size * grad_accum` = 16 freshly
  drawn records, is demoted to a divergence watchdog printed as `Loss(step)` next to
  `Loss(interval mean)`.
- refuses to start a meaningless run: fewer than 8 batches or 2,000 eval tokens aborts, and under
  5,000 tokens it prints the standard error on the reported CE.
- refuses `--eval_parquet` and `--parquet` naming the same file, since that is not a holdout.
- records a sidecar `<ckpt>.meta.json` with step, seed, both dataset md5s, eval CE, perplexity,
  dtype, and config. The tensor payload stays bare so the existing `weights_only=True` readers keep
  working unchanged. The shipped checkpoints predate this and carry no sidecar.
- resumes from the sidecar: step numbering continues and the best record carries over.
  `--save_optimizer` writes a separate `*_optim.pt` (measured 586 MB on a throwaway prefix) that
  restores Adam moments and the scheduler; without that file the run warns that moments were not
  restored instead of pretending.
- selects on `--min_improvement 0.002`. `--select_on train_loss` keeps the other datasets usable and
  labels itself NOISY in every print.

Timing budget, measured on a laptop RTX 3050 with 3.73 GB usable: one optimizer step (8 micro-batches
of 2, `--thinking_cycles 1`) costs about 14 s. A thousand steps is 4 to 6 hours. The default
`--max_steps 1,000,000` is on the order of five months on that hardware.

`training/overfit_sanity.py` addresses learnability separately: it trains a fresh 31M model on a
fixed example set under the production objective and reaches CE 0.0001 on 2 examples and 0.021 on 64,
matching a teacher-forced control arm. Gradient flow, capacity at that scale, and training precision
are not the bottleneck; the zero-feedback invariant costs convergence speed, not learnability (§20.11).

## CLI

`python3 main.py <command> [flags]`

| command | purpose |
|---|---|
| `infer` | one prompt through the selected architecture (`--backend torch` or `--backend jax`) |
| `benchmark` | the multi-domain reasoning suite (`--backend torch` or `--backend jax`) |
| `interactive` | REPL over the ingest-deliberate-emit loop |
| `living` | continuous resting-state heartbeat (`--hz`) |
| `train-1m` | scaled trainer (`--dataset specialized` default) |
| `train` | quick laptop trainer (LMSYS / GSM8K / SVAMP) |
| `train-jax` | high-throughput fused XLA trainer for GPU / TPU / CPU |
| `verify-jax` | 6-gate analytical numerical and gradient parity suite |

```bash
python3 main.py infer --arch v3 --prompt "What is 12 minus 5?"   # choose an architecture
python3 main.py infer --prompt "What is 12 minus 5?" --backend jax # run via JAX / XLA
python3 main.py verify-jax                                       # assert PyTorch <-> JAX bit-exact parity
python3 main.py train-jax --scale laptop --max_steps 1000        # JAX high-throughput trainer
python3 main.py train-1m --dataset specialized --patience 20 --max_steps 2000
```

## Directory layout

```
src/
  model.py                        V2: three-stage spheroid, checkpoint loader
  spheroid_v3.py                  V3 mechanisms and training_rollout (the trained objective)
  jax/                            JAX / XLA implementation (model, convert, inference)
  hemispheric_spheroid.py         two-hemisphere variant
  organoid_layers.py, multimodal_layers.py, real_geodesic_sphere.py,
  modular_kissing_spheres.py, self_replicating_spheroid.py, ...   research variants
  data_loaders.py                 streams for every curriculum
  generate_specialized_dataset.py seeded generator, dedupe, train/val split, manifest
  inference.py                    ReasoningAgent class and benchmark suite (dual torch/jax)
training/
  train_scale_1m.py               primary trainer
  train_jax.py                    high-throughput JAX / XLA trainer (GPU / TPU / CPU)
  verify_jax_parity.py            6-gate analytical numerical and gradient parity verifier
  diagnose_spheroid.py            the measurement instrument described above
  check_eval_stream_determinism.py  guards the seeded shuffle against the recorded ordering
  run_both_checkpoints.sh         scores both shipped snapshots under one seed
  overfit_sanity.py, ab_refine_spheroid.py, verify_warmstart.py, train_*.py, test_*.py
  standalone/                     self-contained single-dataset trainers
  remote/                         Kaggle, Colab, and TPU jobs (including kaggle_tpu_jax.py)
checkpoints/                      shipped weights, provenance sidecars, colony genomes
data/                             parquets and the cognitive manifest (gitignored)
archive/                          the v1 concentric spheroid, full tarball (gitignored)
RESEARCH_NOTES.md                 the running record, including 20.7 to 20.14
PLAN_REAL_SPHERE_MULTIMODAL.md
main.py                           unified CLI entrypoint
cerebro_spheroid.py               root compatibility wrapper
```

## Reproduce

```bash
# Seeded curriculum and its split. Deterministic: the md5s above come out again.
PYTHONPATH=. python3 src/generate_specialized_dataset.py --seed 1337

# The 20.10 ladder, bit-exact, on the frozen pool it was measured on.
PYTHONPATH=. python3 training/diagnose_spheroid.py --batches 64 \
    --parquet data/cerebro_cognitive/legacy_train_unseeded_e16c6f47.parquet \
    --uni_floor_batches 64

# The honest configuration: holdout scored, floors fitted on the whole training pool.
PYTHONPATH=. python3 training/diagnose_spheroid.py --batches 64

# Both shipped snapshots under one seed.
bash training/run_both_checkpoints.sh

# That the loader's seeded shuffle is the permutation the recorded figures used.
PYTHONPATH=. python3 training/check_eval_stream_determinism.py

# Trainer smoke test on a throwaway prefix, then delete the artifacts.
PYTHONPATH=. python3 training/train_scale_1m.py --dataset specialized \
    --save_prefix smoke_1m --max_steps 8 --eval_interval 4 --eval_batches 48 \
    --thinking_cycles 1 --save_optimizer
```

## Known defects and open questions

- No measurement in this repository supports calling the current output reasoning. The order
  sensitivity, the retention curve, and the slot-by-slot loss agree that the readout barely depends
  on the prompt.
- A properly fitted global-unigram floor has not been recorded yet. The 7.8244 figure is a starved
  fit and will move down, which shrinks the model's apparent advantage.
- The shipped checkpoints' first properly fitted cross-entropy on the val split is pending. A single
  thin pass measured 6.86 on 750 tokens, too small to quote, and is not a holdout for these weights
  in the useful sense: they were fitted before the split existed.
- V3's order-sensitivity gain (0.0644 to 0.1726 nats KL) is a valid controlled comparison between two
  undertrained models and says nothing about whether either can reason.
- Whether the zero-feedback objective can reach low cross-entropy at all is a separate question from
  whether it has been given the compute. `training/overfit_sanity.py` addresses learnability on 1 and
  32 examples; large-scale training on this architecture and dataset has never been run.

## Environment

PyTorch 2.14.0+cu132, Python 3.14, transformers gpt2 tokenizer. Developed on an NVIDIA RTX 3050 Laptop
(3.73 GB) with `HF_HOME` pointed at `.hf_cache/`. The day-to-day shell is fish, where `$?` is
`$status` and heredocs do not exist; the reproduce commands above use bash.
