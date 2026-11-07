"""
CerebroSpheroid V4: Full Production JAX Training Script.
Supports Google Colab Cloud TPU, Kaggle TPU v3-8 (PJRT), and Google Cloud Instance Groups.

Features:
- Pure JAX V4 Model Architecture
- SpheroidOptimizer (Quintic Newton-Schulz Polar Updates + Contractive Spectral Clamp)
- Autonomous Checkpoint Discovery & Seamless Resumption
- Asynchronous Non-blocking Cloud / Google Drive Backup (Zero TPU Stall)
- Streaming OpenHermes-2.5 & GSM8K Reasoning Data
"""

import argparse
import math
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

# Kaggle / Python 3.12 compatibility shim: Newer pyarrow (>=19) deprecated and removed pa.PyExtensionType,
# which causes older HuggingFace datasets (<3.0.0) to raise AttributeError
try:
    import pyarrow as pa
    if not hasattr(pa, "PyExtensionType") and hasattr(pa, "ExtensionType"):
        pa.PyExtensionType = pa.ExtensionType
except Exception:
    pass

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.jax import (
    ModelConfigV4,
    PRESETS_V4,
    init_v4_params,
    init_spheroid_opt_state,
    build_spheroid_train_engine,
    build_train_step_spheroid,
    load_torch_pt_to_jax,
    discover_latest_checkpoint,
    AsyncCheckpointManager,
    training_rollout_v4,
)


def parse_args(args_list=None):
    if args_list is None:
        args_list = [a for a in sys.argv[1:] if a != "n"]
    parser = argparse.ArgumentParser(description="CerebroSpheroid V4 JAX High-Throughput Trainer")
    parser.add_argument("--steps", "--max_steps", dest="steps", type=int, default=100000, help="Total target steps (default: 100000)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per TPU device (default: 16)")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation micro-steps per optimizer update (default: 4)")
    parser.add_argument("--total_window", type=int, default=1024, help="Total sequence budget for elastic sliding window (default: 1024)")
    parser.add_argument("--prompt_len", type=int, default=256, help="Prompt context length (used when --elastic_window is disabled)")
    parser.add_argument("--target_len", type=int, default=768, help="Deliberated target rollout length (used when --elastic_window is disabled)")
    parser.add_argument("--elastic_window", action="store_true", default=True, help="Enable elastic sliding context window")
    parser.add_argument("--lr", type=float, default=2.0e-3, help="Peak SpheroidOptimizer learning rate")
    parser.add_argument("--min_lr", type=float, default=1.0e-4, help="Final annealed learning rate")
    parser.add_argument("--preset", type=str, default="small", choices=["small", "wide"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"],
                        help="Matrix execution precision ('auto' selects fp16 on Turing GPUs to enable 65 TFLOPS Tensor Cores, and bf16 on TPU)")
    parser.add_argument("--max_hours", type=float, default=None, help="Maximum training wall-clock time in hours before graceful shutdown")
    parser.add_argument("--cloud_save_hours", type=float, default=None, help="Hours between asynchronous cloud checkpoint uploads (e.g. 1.0 for hourly uploads; overrides --save_every for cloud sync)")
    parser.add_argument("--source_node", type=str, default=None, help="Identifier for training environment (e.g. kaggle_tpu, colab_tpu, workstation)")
    parser.add_argument("--ckpt_dir", "--drive_dir", dest="ckpt_dir", type=str, default=os.path.join(ROOT_DIR, "checkpoints"), help="Destination checkpoint directory")
    parser.add_argument("--scratch_dir", type=str, default="/tmp/cerebro_ckpts", help="Fast local disk for staging")
    parser.add_argument("--save_every", type=int, default=1000, help="Steps between latest.pt async saves (default: 1000)")
    parser.add_argument("--periodic_every", type=int, default=500, help="Steps between milestone archive saves")
    parser.add_argument("--log_every", type=int, default=25, help="Steps between console progress lines")
    parser.add_argument("--schedule_steps", type=int, default=5000, help="Steps to anneal from teacher-forcing to internal deliberation")
    parser.add_argument("--min_sampling_p", type=float, default=0.35, help="Minimum teacher sampling floor for scheduled sampling (default: 0.35)")
    parser.add_argument("--eval_interval", type=int, default=200, help="Steps between held-out validation evaluations")
    parser.add_argument("--thalamic_anchor_weight", type=float, default=0.1, help="Thalamic memory basin retention strength (0.0 to 0.3)")
    parser.add_argument("--deliberation_mode", type=str, default="scheduled", choices=["scheduled", "teacher_forcing", "internal"])
    parser.add_argument("--repetition_penalty", type=float, default=1.25, help="Two-tier refractory repetition penalty multiplier (default: 1.25)")
    parser.add_argument("--persona_id", type=int, default=None, help="Agentic persona ID (0=Reasoner, 1=Explorer, 2=Concise, 3=Verifier, or None for round-robin curriculum)")
    parser.add_argument("--dataset_source", type=str, default="dense_edu",
                        choices=["openhermes", "fineweb_edu", "balanced_mix", "dense_edu"],
                        help="Streaming training dataset: 'dense_edu' (40%% Cosmopedia-v2 + 40%% FineWeb-Edu [score>=4.0] + 20%% OpenHermes-2.5), "
                             "'balanced_mix' (50%% OpenHermes, 50%% FineWeb-Edu), 'fineweb_edu', or 'openhermes'")
    parser.add_argument("--cloud_sync", action="store_true", default=False, help="Enable automatic checkpoint syncing with bioflo.in")
    parser.add_argument("--resume_target", type=str, default="latest", choices=["latest", "best"], help="Target checkpoint to pull from bioflo.in ('latest' or 'best')")
    parser.add_argument("--cloud_url", type=str, default="https://bioflo.in/api/cerebro/sync.php", help="bioflo.in sync endpoint")
    parser.add_argument("--cloud_token", type=str, default="CerebroSpheroid_Sync_Secure_2026", help="Secret sync token")
    return parser.parse_args(args_list)


def run_training(**kwargs):
    """Programmatic in-process entrypoint for Jupyter / Colab notebooks (avoids TPU process lock conflicts)."""
    args = parse_args([])
    for k, v in kwargs.items():
        if hasattr(args, k):
            setattr(args, k, v)
        elif k == "drive_dir":
            setattr(args, "ckpt_dir", v)
        elif k == "max_steps":
            setattr(args, "steps", v)
    return main(args)


def main(args=None):
    if args is None:
        args = parse_args()

    # If max_hours is set and steps wasn't customized from default 100k, run continuously for the full time window
    if args.max_hours is not None and args.steps == 100000:
        args.steps = 50_000_000

    devices = jax.devices()
    num_devices = len(devices)
    dev_kind = devices[0].device_kind.lower() if devices else ""
    backend = jax.default_backend().lower()

    if args.dtype == "auto":
        if "tpu" in dev_kind or "tpu" in backend:
            target_dtype = jnp.bfloat16
            dtype_desc = "bfloat16 (Native TPU MXU)"
        elif "gpu" in backend or "cuda" in backend or "nvidia" in dev_kind:
            # On Tesla T4 GPUs, float32 master parameters guarantee zero exponent overflow (takes only 472 MB out of 32 GB VRAM)
            target_dtype = jnp.float32
            dtype_desc = "float32 (Stable GPU Master Weights)"
        else:
            target_dtype = jnp.float32
            dtype_desc = "float32 (CPU)"
    elif args.dtype == "fp16":
        target_dtype = jnp.float16
        dtype_desc = "float16"
    elif args.dtype == "bf16":
        target_dtype = jnp.bfloat16
        dtype_desc = "bfloat16"
    else:
        target_dtype = jnp.float32
        dtype_desc = "float32"

    print("=" * 80)
    print("🧠 CEREBRO-SPHEROID V4: JAX HIGH-THROUGHPUT ENGINE")
    print(f"Devices        : {devices} ({num_devices} active core(s))")
    print(f"Backend Kind   : {devices[0].device_kind}")
    print(f"Precision      : {dtype_desc}")
    print(f"Batch Size     : {args.batch_size} | Window: {args.prompt_len + args.target_len} tokens")
    if args.max_hours is not None:
        print(f"Target Horizon : Continuous Ingestion ({args.max_hours:.1f} hours max budget)")
    else:
        print(f"Target Horizon : {args.steps:,} steps")
    print(f"Checkpoint Dir : {args.ckpt_dir}")
    print("=" * 80)

    # 1. Initialize Model Configuration
    preset_kwargs = PRESETS_V4.get(args.preset, PRESETS_V4["small"]).copy()
    config = ModelConfigV4(vocab_size=32000, **preset_kwargs)

    # 2. Tokenizer
    from src.corpus import build_tokenizer
    tokenizer = build_tokenizer("llama")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # 3. Autonomous Checkpoint Discovery & Cloud Synchronization
    synced_ckpt = None
    if args.cloud_sync:
        try:
            from src.cloud_sync import auto_sync_checkpoints
            synced_ckpt = auto_sync_checkpoints(
                local_dir=args.ckpt_dir,
                url=args.cloud_url,
                token=args.cloud_token,
                resume_target=args.resume_target
            )
        except Exception as sync_err:
            print(f"⚠️ [cloud_sync] Check failed: {sync_err}")

    search_dirs = [
        args.ckpt_dir,
        "/content/checkpoints",
        args.scratch_dir,
        "/content/drive/MyDrive/cerebro_checkpoints",
        "/kaggle/working/checkpoints",
        "/kaggle/working/reasoning_agent/checkpoints",
        "/kaggle/input",
        "checkpoints",
        "/content",
    ]
    latest_ckpt, max_step, meta = discover_latest_checkpoint(search_dirs, target_preference=args.resume_target)

    # Prioritize cloud-synced checkpoint if present
    if synced_ckpt and os.path.exists(synced_ckpt):
        s_step = 0
        s_meta = {}
        for m_ext in [".meta.json", ".json"]:
            cand = synced_ckpt.replace(".pt", m_ext)
            if os.path.exists(cand):
                try:
                    with open(cand, "r") as mf:
                        s_meta = json.load(mf)
                        s_step = int(s_meta.get("step", 0))
                        if s_step > 0:
                            break
                except Exception:
                    pass
        if s_step >= max_step or latest_ckpt is None:
            latest_ckpt = synced_ckpt
            max_step = s_step
            meta = s_meta

    if latest_ckpt and max_step > 0:
        print(f"\n🔍 Auto-Resume: Discovered existing checkpoint: {latest_ckpt}")
        print(f"🔄 Automatically resuming from Step {max_step + 1:,}...")
        params, buffers = load_torch_pt_to_jax(latest_ckpt, config)
        start_step = max_step + 1

        # Synchronize initial restored model to bioflo.in asynchronously
        if args.cloud_sync:
            try:
                from src.cloud_sync import check_remote_manifest, push_checkpoint_async
                m_info = check_remote_manifest(args.cloud_url, args.cloud_token)
                rem_m = m_info.get("manifest", {})
                if not isinstance(rem_m, dict):
                    rem_m = {}
                rem_latest = rem_m.get("latest", {})
                rem_step = rem_latest.get("step", 0) if isinstance(rem_latest, dict) else 0
                if max_step > rem_step:
                    print(f"🌐 [cloud_sync] Initial sync: Uploading restored Step {max_step:,} to bioflo.in asynchronously...")
                    push_checkpoint_async(
                        latest_ckpt,
                        step=max_step,
                        loss=float(meta.get("loss", meta.get("eval_loss", 0.0))) if meta else 0.0,
                        target_type="latest",
                        url=args.cloud_url,
                        token=args.cloud_token,
                    )
            except Exception as e:
                print(f"⚠️ [cloud_sync] Initial upload notice: {e}")
    else:
        print("\n🆕 No previous checkpoints found. Initializing fresh parameters at Step 1...")
        key = jax.random.PRNGKey(1337)
        params, buffers = init_v4_params(config, key)
        start_step = 1

    params = {k: v.astype(target_dtype) if jnp.issubdtype(v.dtype, jnp.floating) else v for k, v in params.items()}
    param_count = sum(p.size for p in params.values() if jnp.issubdtype(p.dtype, jnp.inexact))
    print(f"Model Scale    : {param_count:,} parameters ({param_count * 2 / (1024**2):.1f} MB in {target_dtype})")

    # Notify website / user that training has officially commenced
    if args.cloud_sync:
        try:
            from src.cloud_sync import notify_training_start, detect_source_environment
            src_env = args.source_node or detect_source_environment()
            hours_budget = args.max_hours or 8.5
            notify_training_start(
                source=src_env,
                step=start_step,
                hours=hours_budget,
                message=f"CerebroSpheroid V4 training session active on {src_env}. Resuming from Step {start_step:,} ({hours_budget}h budget).",
                url=args.cloud_url,
                token=args.cloud_token,
            )
        except Exception as notify_err:
            print(f"⚠️ [cloud_sync] Startup notification notice: {notify_err}")

    # 4. Initialize SpheroidOptimizer & Build JIT Train Engine
    opt_state = init_spheroid_opt_state(params)
    compute_grad_step, apply_opt_step = build_spheroid_train_engine(
        config,
        buffers,
        pad_id=pad_id,
        weight_decay=0.01,
        thalamic_anchor_weight=args.thalamic_anchor_weight,
        repetition_penalty=args.repetition_penalty,
    )
    ckpt_mgr = AsyncCheckpointManager(
        target_dir=args.ckpt_dir,
        local_scratch_dir=args.scratch_dir,
        cloud_sync=args.cloud_sync,
        cloud_url=args.cloud_url,
        cloud_token=args.cloud_token,
    )

    # 4b. Multi-Device SPMD Mesh Sharding (8 TPU Cores or Multi-GPU)
    num_devices = len(devices)
    batch_sharding = None
    replicate_sharding = None
    if num_devices > 1:
        try:
            from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
            mesh = Mesh(np.array(devices), axis_names=('data',))
            batch_sharding = NamedSharding(mesh, P('data', None))
            replicate_sharding = NamedSharding(mesh, P())
            print(f"⚡ JAX SPMD Multi-Device Mesh Active: Sharding batch across {num_devices} devices in parallel!")
            params = jax.device_put(params, replicate_sharding)
            buffers = jax.device_put(buffers, replicate_sharding)
            opt_state = jax.device_put(opt_state, replicate_sharding)
        except Exception as mesh_err:
            print(f"⚠️ Multi-device mesh initialization notice: {mesh_err}")

    # 5. Data Stream Generator with Elastic Sliding Buckets & Feedback
    try:
        import pyarrow as pa
        if not hasattr(pa, "PyExtensionType") and hasattr(pa, "ExtensionType"):
            pa.PyExtensionType = pa.ExtensionType
    except Exception:
        pass
    from datasets import load_dataset
    from src.elastic_window import ElasticBucketBatcher, get_default_elastic_buckets

    ds_choice = getattr(args, "dataset_source", "dense_edu")
    if ds_choice == "dense_edu":
        print("\n📦 Initializing 3B Dense Knowledge Stream (40% Cosmopedia-v2 + 40% FineWeb-Edu [score>=4.0] + 20% OpenHermes)...")
        cosmo_raw = iter(load_dataset("HuggingFaceTB/cosmopedia-v2", "cosmopedia-v2", split="train", streaming=True))
        fw_raw = iter(load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True))
        oh_raw = iter(load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True))

        def _dense_edu_stream():
            nonlocal cosmo_raw, fw_raw, oh_raw
            turn = 0
            while True:
                mod5 = turn % 5
                try:
                    if mod5 == 0 or mod5 == 2:
                        try:
                            row = next(cosmo_raw)
                        except StopIteration:
                            cosmo_raw = iter(load_dataset("HuggingFaceTB/cosmopedia-v2", "cosmopedia-v2", split="train", streaming=True))
                            row = next(cosmo_raw)
                        p = (row.get("prompt") or "").strip()
                        r = (row.get("text") or "").strip()
                        if len(p) >= 10 and len(r) >= 50:
                            yield {"conversations": [{"from": "human", "value": p}, {"from": "gpt", "value": r}]}
                    elif mod5 == 1 or mod5 == 4:
                        for _ in range(50):
                            try:
                                row = next(fw_raw)
                            except StopIteration:
                                fw_raw = iter(load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True))
                                row = next(fw_raw)
                            score = float(row.get("score", 0.0) or row.get("educational_score", 0.0))
                            if score >= 4.0:
                                text = (row.get("text") or "").strip()
                                if len(text) >= 50:
                                    if "\n\n" in text:
                                        p, r = text.split("\n\n", 1)
                                    else:
                                        lines = text.split(". ", 1)
                                        p = lines[0] + "."
                                        r = lines[1] if len(lines) > 1 else text
                                    if len(p) < 15:
                                        p = "Explain the scientific/educational concept behind this:\n" + p
                                    yield {"conversations": [{"from": "human", "value": p.strip()}, {"from": "gpt", "value": r.strip()}]}
                                    break
                    else:  # mod5 == 3: OpenHermes reasoning
                        try:
                            yield next(oh_raw)
                        except StopIteration:
                            oh_raw = iter(load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True))
                            yield next(oh_raw)
                except Exception as stream_err:
                    time.sleep(1.0)
                turn += 1

        data_iter = _dense_edu_stream()
    elif ds_choice == "fineweb_edu":
        print("\n📦 Initializing streaming educational data (HuggingFaceFW/fineweb-edu: sample-10BT)...")
        raw_ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        def _fineweb_stream():
            for row in raw_ds:
                score = float(row.get("score", 0.0) or row.get("educational_score", 0.0))
                if score < 3.0:
                    continue
                text = (row.get("text") or "").strip()
                if len(text) < 50:
                    continue
                if "\n\n" in text:
                    p, r = text.split("\n\n", 1)
                else:
                    lines = text.split(". ", 1)
                    p = lines[0] + "."
                    r = lines[1] if len(lines) > 1 else text
                if len(p) < 15:
                    p = "Explain the educational context of the following topic:\n" + p
                yield {"conversations": [{"from": "human", "value": p.strip()}, {"from": "gpt", "value": r.strip()}]}
        data_iter = _fineweb_stream()
    elif ds_choice == "balanced_mix":
        print("\n📦 Initializing balanced streaming mix (50% OpenHermes-2.5 + 50% FineWeb-Edu)...")
        oh_ds = iter(load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True))
        fw_raw = iter(load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True))
        def _balanced_stream():
            turn = 0
            while True:
                if turn % 2 == 0:
                    try:
                        yield next(oh_ds)
                    except StopIteration:
                        break
                else:
                    try:
                        for row in fw_raw:
                            score = float(row.get("score", 0.0) or row.get("educational_score", 0.0))
                            if score >= 3.0:
                                text = (row.get("text") or "").strip()
                                if len(text) >= 50:
                                    if "\n\n" in text:
                                        p, r = text.split("\n\n", 1)
                                    else:
                                        lines = text.split(". ", 1)
                                        p = lines[0] + "."
                                        r = lines[1] if len(lines) > 1 else text
                                    if len(p) < 15:
                                        p = "Explain the educational context of the following topic:\n" + p
                                    yield {"conversations": [{"from": "human", "value": p.strip()}, {"from": "gpt", "value": r.strip()}]}
                                    break
                    except StopIteration:
                        break
                turn += 1
        data_iter = _balanced_stream()
    else:
        print("\n📦 Initializing streaming reasoning data (teknium/OpenHermes-2.5)...")
        dataset = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
        data_iter = iter(dataset)

    global_batch_size = args.batch_size * num_devices if num_devices > 1 else args.batch_size
    if args.elastic_window:
        buckets = get_default_elastic_buckets(args.total_window)
        print(f"🌊 Elastic Sliding Context Window Active (Budget: {args.total_window} tokens | Global Batch: {global_batch_size}):")
        for idx, (b_name, bp, bt) in enumerate(buckets):
            print(f"   Bucket {idx}: {b_name:<18} (Prompt: {bp:4d} tok -> Target: {bt:4d} tok)")
        batcher = ElasticBucketBatcher(
            tokenizer, data_iter, batch_size=global_batch_size, total_window=args.total_window, buckets=buckets, framework="jax", return_feedback=True
        )
    else:
        buckets = [("static", args.prompt_len, args.target_len)]
        batcher = ElasticBucketBatcher(
            tokenizer, data_iter, batch_size=global_batch_size, total_window=args.prompt_len + args.target_len, buckets=buckets, framework="jax", return_feedback=True
        )

    # Asynchronous batch prefetching (prevents GPU/TPU starvation during I/O)
    from src.elastic_window import PrefetchedBatcher
    batcher = PrefetchedBatcher(batcher, queue_size=12)
    print("⚡ Background Asynchronous Batch Prefetcher Active (Queue Depth: 12 batches)")

    # 6. Fixed Held-Out Validation Split (Guards against fake low training loss)
    eval_batches = []
    try:
        from scripts.eval_latest_checkpoint_loss import load_openhermes_eval_records, load_gsm8k_eval_records
        from training.train_v4_full import build_batch
        import random
        eval_records = load_openhermes_eval_records(count=32) + load_gsm8k_eval_records(count=32)
        rng_eval = random.Random(1337)
        for i in range(0, len(eval_records), args.batch_size):
            chunk = eval_records[i : i + args.batch_size]
            b_eval = build_batch(tokenizer, chunk, window=128, rng=rng_eval, max_prompt_len=256, max_answer_tokens=128, pad_token_id=pad_id)
            if b_eval is not None:
                if len(b_eval) == 3:
                    p_eval, t_eval, f_eval = b_eval
                    eval_batches.append((jnp.array(p_eval.numpy()), jnp.array(t_eval.numpy()), jnp.array(f_eval.numpy())))
                else:
                    p_eval, t_eval = b_eval
                    t_arr = jnp.array(t_eval.numpy())
                    f_arr = jnp.full_like(t_arr, -100)
                    f_arr = f_arr.at[:, 1:].set(t_arr[:, :-1])
                    eval_batches.append((jnp.array(p_eval.numpy()), t_arr, f_arr))
        print(f"✅ Held-Out Validation: Prepared {len(eval_batches)} evaluation batches ({len(eval_records)} records).")
    except Exception as e:
        print(f"⚠️ Validation setup notice: {e} (will fallback to training EMA for checkpoint selection)")

    @jax.jit
    def eval_step_loss(p, q_b, r_b, f_b):
        mask = (r_b >= 0).astype(jnp.float32)
        total_tokens = jnp.sum(mask)
        mean_loss = training_rollout_v4(
            p,
            q_b,
            r_b.shape[1],
            buffers,
            config=config,
            feedback_tokens=f_b,
            deliberation_mode="teacher_forcing",
            sampling_p=1.0,
            thalamic_anchor_weight=args.thalamic_anchor_weight,
            pad_id=pad_id,
            persona_id=0,
            repetition_penalty=1.0,
            targets=r_b,
        )
        return mean_loss * total_tokens, total_tokens

    # 7. Training Loop with Scheduled Sampling and SpheroidOptimizer Updates
    init_loss = float(meta.get("eval_loss", meta.get("loss", meta.get("best", float("inf"))))) if meta else float("inf")
    best_eval_loss = init_loss if (not math.isnan(init_loss) and init_loss > 0.0) else float("inf")
    best_ema_loss = best_eval_loss
    ema_loss = None
    t_start = time.time()
    last_cloud_save_time = time.time()
    total_tokens_ingested = 0
    current_step = start_step

    print("\n" + "=" * 80)
    if args.max_hours is not None:
        print(f"🚀 Training Commenced: Resuming Step {start_step:,} -> Continuous Ingestion ({args.max_hours:.1f}h Budget)")
    else:
        print(f"🚀 Training Commenced: Steps {start_step:,} -> {args.steps:,}")
    print(f"💎 Matrix Operators: Quintic Newton-Schulz Polar (Single-Moment Memory)")
    print(f"🛡️ Recurrent Deliberation: Contractive Spectral Guard (rho <= 1.0)")
    print(f"🧭 Scheduled Sampling Curriculum: {args.deliberation_mode} over {args.schedule_steps:,} steps")
    if args.cloud_save_hours is not None:
        print(f"⚡ Hourly Cloud Sync: Uploading every {args.cloud_save_hours:.1f} hour(s) to bioflo.in (0ms TPU Pause)")
    else:
        print(f"⚡ Asynchronous Cloud Sync: 0ms TPU Pause")
    print("=" * 80 + "\n")

    try:
        for step in range(start_step, args.steps + 1):
            current_step = step

            # Learning Rate Schedule (Gentle warmup on start/resume, global cosine decay)
            step_offset = step - start_step
            progress = float(step) / max(1.0, float(args.steps))
            target_lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * min(1.0, progress)))
            if step_offset < 100:
                # 100-step ramp from min_lr to target_lr on session start/resume
                lr = args.min_lr + (target_lr - args.min_lr) * (float(step_offset) / 100.0)
            else:
                lr = target_lr

            # Scheduled sampling curriculum (Smooth Cosine Transition down to min_sampling_p)
            if args.deliberation_mode == "teacher_forcing":
                sampling_p = 1.0
            elif args.deliberation_mode == "internal":
                sampling_p = 0.0
            else:  # "scheduled"
                if args.schedule_steps > 0:
                    prog = min(1.0, max(0.0, float(step) / float(args.schedule_steps)))
                    min_p = getattr(args, "min_sampling_p", 0.35)
                    sampling_p = float(min_p + 0.5 * (1.0 - min_p) * (1.0 + math.cos(math.pi * prog)))
                else:
                    sampling_p = float(getattr(args, "min_sampling_p", 0.35))

            # Dynamic Persona Curriculum (0: Reasoner, 1: Explorer, 2: Concise, 3: Verifier)
            if args.persona_id is not None:
                step_persona = jnp.int32(args.persona_id)
            else:
                step_persona = jnp.int32(step % 4)

            t0 = time.time()
            accum_grads = None
            accum_loss = 0.0
            last_bucket_name = ""
            total_step_tokens = 0

            # Gradient accumulation across diverse elastic sliding window buckets
            valid_micro_steps = 0
            for _ in range(args.grad_accum):
                batch_data = batcher.next_batch()
                if len(batch_data) == 5:
                    b_idx, b_name, q_batch, r_batch, f_batch = batch_data
                else:
                    b_idx, b_name, q_batch, r_batch = batch_data
                    f_batch = None
                last_bucket_name = b_name
                total_step_tokens += int(q_batch.shape[0] * (q_batch.shape[1] + r_batch.shape[1]))

                if batch_sharding is not None and (q_batch.shape[0] % num_devices == 0):
                    q_batch = jax.device_put(q_batch, batch_sharding)
                    r_batch = jax.device_put(r_batch, batch_sharding)
                    if f_batch is not None:
                        f_batch = jax.device_put(f_batch, batch_sharding)

                micro_loss, micro_grads = compute_grad_step(
                    params,
                    q_batch,
                    r_batch,
                    feedback_tokens=f_batch,
                    sampling_p=sampling_p,
                    persona_id=step_persona,
                )
                m_loss_f = float(micro_loss)

                # Skip corrupted or non-finite micro-batches
                if math.isnan(m_loss_f) or math.isinf(m_loss_f):
                    continue

                accum_loss += m_loss_f
                valid_micro_steps += 1

                scale = 1.0 / float(args.grad_accum)
                if accum_grads is None:
                    accum_grads = jax.tree_util.tree_map(lambda g: g * scale, micro_grads)
                else:
                    accum_grads = jax.tree_util.tree_map(lambda a, g: a + g * scale, accum_grads, micro_grads)

            if valid_micro_steps > 0 and accum_grads is not None:
                if valid_micro_steps < args.grad_accum:
                    renorm = float(args.grad_accum) / float(valid_micro_steps)
                    accum_grads = jax.tree_util.tree_map(lambda g: g * renorm, accum_grads)
                    loss_val = accum_loss / float(valid_micro_steps)
                else:
                    loss_val = accum_loss / float(args.grad_accum)

                params, opt_state = apply_opt_step(params, opt_state, accum_grads, lr)
            else:
                loss_val = float("nan")

            dt = time.time() - t0
            total_tokens_ingested += total_step_tokens

            if not math.isnan(loss_val):
                if ema_loss is None:
                    ema_loss = loss_val
                else:
                    ema_loss = 0.96 * ema_loss + 0.04 * loss_val

            if step % args.log_every == 0 or step == start_step:
                loss_disp = loss_val if not math.isnan(loss_val) else float("nan")
                ema_disp = ema_loss if (ema_loss is not None and not math.isnan(ema_loss)) else float("nan")
                ppl = math.exp(min(loss_disp, 15.0)) if not math.isnan(loss_disp) else float("nan")
                ema_ppl = math.exp(min(ema_disp, 15.0)) if not math.isnan(ema_disp) else float("nan")
                tok_s = total_step_tokens / max(dt, 1e-4)

                if args.max_hours is not None:
                    elapsed_sec = time.time() - t_start
                    budget_sec = args.max_hours * 3600.0
                    el_h = int(elapsed_sec // 3600)
                    el_m = int((elapsed_sec % 3600) // 60)
                    tot_h = int(budget_sec // 3600)
                    tot_m = int((budget_sec % 3600) // 60)
                    time_pct = min(100.0, (elapsed_sec / budget_sec) * 100.0)

                    if total_tokens_ingested >= 1e6:
                        tok_str = f"{total_tokens_ingested / 1e6:.1f}M"
                    elif total_tokens_ingested >= 1e3:
                        tok_str = f"{total_tokens_ingested / 1e3:.1f}k"
                    else:
                        tok_str = f"{total_tokens_ingested}"

                    print(
                        f"[T+{el_h:02d}h {el_m:02d}m / {tot_h:02d}h {tot_m:02d}m] ({time_pct:4.1f}%) | "
                        f"Ingested: {tok_str:>6s} tok ({tok_s:5.0f} tok/s) | "
                        f"Step {step:,} | {last_bucket_name:<14} | p={sampling_p:.2f} | "
                        f"loss {loss_disp:6.4f} (ema {ema_disp:6.4f}) | ema_ppl {ema_ppl:6.1f} | lr {lr:.2e}"
                    )
                else:
                    eta_h = ((args.steps - step) * dt) / 3600.0
                    print(
                        f"step {step:5d}/{args.steps:,} | {last_bucket_name:<14} | p={sampling_p:.2f} | loss {loss_disp:6.4f} (ema {ema_disp:6.4f}) | "
                        f"ema_ppl {ema_ppl:6.1f} | lr {lr:.2e} | {dt*1000:5.1f} ms/step | {tok_s:7.1f} tok/s | ETA: {eta_h:4.2f}h"
                    )

            # Held-Out Validation Evaluation (Every args.eval_interval steps, Aligned Teacher-Forced PPL)
            if eval_batches and (step % args.eval_interval == 0):
                eval_tot_loss, eval_tot_tokens = 0.0, 0
                for eq, er, ef in eval_batches:
                    el, et = eval_step_loss(params, eq, er, ef)
                    eval_tot_loss += float(el)
                    eval_tot_tokens += int(et)
                held_out_loss = eval_tot_loss / max(eval_tot_tokens, 1)
                held_out_ppl = math.exp(min(held_out_loss, 20.0))
                print(
                    f"\n🔍 [EVAL @ Step {step:,}] Aligned Held-Out Loss: {held_out_loss:.4f} | "
                    f"Held-Out PPL: {held_out_ppl:.2f} ({eval_tot_tokens:,} tokens)"
                )

                # Record-breaking checkpoint is selected on genuine held-out validation
                if held_out_loss < best_eval_loss - 0.005:
                    best_eval_loss = held_out_loss
                    print(f"  ⭐ GENUINE RECORD (Held-Out): Loss {best_eval_loss:.4f} (PPL {held_out_ppl:.1f}) -> Syncing best.pt\n")
                    ckpt_mgr.dispatch(
                        params,
                        buffers,
                        "cerebro_spheroid_v4_jax_best.pt",
                        meta_dict={
                            "step": step,
                            "loss": float(best_eval_loss),
                            "eval_loss": float(best_eval_loss),
                            "eval_ppl": float(held_out_ppl),
                            "tokens_ingested": total_tokens_ingested,
                            "elapsed_hours": (time.time() - t_start) / 3600.0,
                        },
                    )

            # Async checkpointing (strictly guarded against NaN / Inf corruption)
            if not math.isnan(loss_val) and not math.isinf(loss_val):
                save_latest = False
                if args.cloud_save_hours is not None:
                    # Hourly cloud upload schedule
                    if (time.time() - last_cloud_save_time) >= (args.cloud_save_hours * 3600.0):
                        save_latest = True
                        last_cloud_save_time = time.time()
                else:
                    if step % args.save_every == 0:
                        save_latest = True

                if save_latest:
                    elapsed_hours = (time.time() - t_start) / 3600.0
                    if args.cloud_save_hours is not None:
                        print(f"\n💾 [Hourly Cloud Sync @ Step {step:,}] Archiving and uploading checkpoint ({elapsed_hours:.2f}h elapsed, {total_tokens_ingested/1e6:.1f}M tokens ingested)...")
                    ckpt_mgr.dispatch(
                        params,
                        buffers,
                        "cerebro_spheroid_v4_jax_latest.pt",
                        meta_dict={
                            "step": step,
                            "loss": loss_val,
                            "ema_loss": ema_loss,
                            "best_eval_loss": best_eval_loss,
                            "tokens_ingested": total_tokens_ingested,
                            "elapsed_hours": elapsed_hours,
                        },
                    )

                if step % args.periodic_every == 0:
                    ckpt_mgr.dispatch(
                        params,
                        buffers,
                        f"cerebro_spheroid_v4_jax_step_{step}.pt",
                        meta_dict={"step": step, "loss": loss_val, "ema_loss": ema_loss},
                    )

            # Max wall-clock time limit guard (e.g. Kaggle 8.5-hour safety limit)
            if args.max_hours is not None:
                elapsed_hours = (time.time() - t_start) / 3600.0
                if elapsed_hours >= args.max_hours:
                    print(f"\n⏰ Reached session time limit of {args.max_hours:.2f}h (elapsed: {elapsed_hours:.2f}h).")
                    print(f"💾 Gracefully flushing final state for Step {step:,} ({total_tokens_ingested/1e6:.1f}M tokens ingested)...")
                    ckpt_mgr.flush(
                        params,
                        buffers,
                        "cerebro_spheroid_v4_jax_latest.pt",
                        meta_dict={
                            "step": step,
                            "loss": loss_val,
                            "ema_loss": ema_loss,
                            "best_ema_loss": best_ema_loss,
                            "tokens_ingested": total_tokens_ingested,
                            "elapsed_hours": elapsed_hours,
                        },
                    )
                    break

    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user.")
        print(f"💾 Synchronously saving emergency state for step {current_step}...")
        ckpt_mgr.flush(
            params,
            buffers,
            "cerebro_spheroid_v4_jax_latest.pt",
            meta_dict={
                "step": current_step,
                "loss": loss_val if "loss_val" in locals() else 0,
                "ema_loss": ema_loss if "ema_loss" in locals() else 0,
            },
        )
        print("✅ Emergency backup completed!")
    except Exception as e:
        import traceback
        print(f"\n⚠️ Training exception encountered: {e}")
        traceback.print_exc()
        print(f"💾 Synchronously saving emergency state for step {current_step}...")
        ckpt_mgr.flush(
            params,
            buffers,
            "cerebro_spheroid_v4_jax_latest.pt",
            meta_dict={
                "step": current_step,
                "loss": loss_val if "loss_val" in locals() else 0,
                "ema_loss": ema_loss if "ema_loss" in locals() else 0,
            },
        )
        print("✅ Emergency backup completed!")

    finally:
        if "batcher" in locals() and hasattr(batcher, "shutdown"):
            try:
                batcher.shutdown()
            except Exception:
                pass
        ckpt_mgr.shutdown(wait=True)

    print("=" * 80)
    print(f"🎉 Training session finished at step {current_step:,}.")
    print(f"Checkpoints preserved in: {args.ckpt_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
