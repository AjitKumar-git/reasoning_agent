# Google Colab training guide

Trains the 146,618,503-parameter CerebroSpheroid on the seeded cognitive curriculum
(`data/cerebro_cognitive/`) using the zero-feedback objective in `src/spheroid_v3.py`.

The entry point is `colab/colab_cognitive.py`. It replaces `colab_run.py`, which chose the
"best" checkpoint from the current training batch's own loss with no held-out set. On this
objective that number varies by roughly 0.35 nats between steps that cannot improve, which is
how `cerebro_spheroid_1m_best.pt` came to score worse than `_latest.pt`. `colab_cognitive.py`
selects on a frozen held-out batch list, writes a `.meta.json` provenance sidecar beside every
checkpoint, and keeps the gradient checkpointing the unrolled activations need.

## What the self-contained package is

`reasoning_agent_colab_selfcontained.zip` bundles everything a Colab run needs so you upload one
file:

- `src/`, `main.py`, `cerebro_spheroid.py`, `colab/` (including this notebook)
- `training/train_scale_1m.py`, `training/overfit_sanity.py`,
  `training/remote/kaggle_tpu_cognitive.py`
- `data/cerebro_cognitive/` with `cerebro_cognitive_train.parquet`,
  `cerebro_cognitive_val.parquet`, and the manifest
- `checkpoints/cerebro_spheroid_1m_best.pt` and `cerebro_spheroid_1m_latest.pt` for warm start

The datasets and checkpoints are gitignored, so they are not in the repository and only travel
inside this zip.

## Steps

1. Upload `reasoning_agent_colab_selfcontained.zip` and `cerebro_spheroid_colab.ipynb` to Colab.
2. Runtime → Change runtime type → **T4 GPU** (A100 on Colab Pro). A T4 has 16 GB; the unrolled
   objective does not fit the 3.73 GB laptop GPU, which is the reason to run here.
3. Run the notebook cells in order: GPU/deps, Drive mount (optional), unpack the zip, train.

The training cell runs:

```bash
python3 colab/colab_cognitive.py \
    --scale laptop \
    --steps 2000 \
    --batch_size 4 \
    --grad_accum 4 \
    --lr 3e-4 \
    --eval_interval 50 \
    --drive_dir /content/drive/MyDrive/cerebro_checkpoints \
    --resume checkpoints/cerebro_spheroid_1m_best.pt
```

`--batch_size 4` is the memory-bound micro-batch that fits a T4 across the ~80-step response
unroll; `--grad_accum 4` raises the effective batch to 16. On an A100 you can lift `--batch_size`
to 8 or 16. `--resume` warm-starts from the bundled `1m_best` weights; drop it to train from a
fresh initialization.

## Output and persistence

Best and latest checkpoints are written to `checkpoints/` and, when Drive is mounted, mirrored to
`--drive_dir`. Each `.pt` has a `.meta.json` sidecar recording step, seed, scale, dtype, parameter
count, the held-out CE and perplexity it was selected on, and the md5 of both parquet splits. The
tensor file stays a bare `state_dict` so `weights_only=True` loading keeps working.

Free Colab disconnects after roughly 90 minutes idle and around a 12-hour session. Progress saved
to Drive survives that; relaunch with `--resume` pointing at your `cerebro_spheroid_colab_best.pt`
to continue.

## Interactive check

The last notebook cell loads the best checkpoint from this session (or the shipped `1m_best`)
through `src.inference.ReasoningAgent` and runs the deliberate-then-emit loop on a sample prompt.

## Honest caveat

A few-thousand-step run is not enough to fit this curriculum. The learnability probe
(`training/overfit_sanity.py`) shows the objective and architecture can drive cross-entropy to
0.0646 on 64 examples, so capacity and gradient flow are not the limit; the shipped `1m_best`
sits near 7.5 nats because it had only tens of optimizer steps. A single Colab session shortens
that gap, it does not close it.
