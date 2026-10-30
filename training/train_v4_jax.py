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

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.jax import (
    ModelConfigV4,
    PRESETS_V4,
    init_v4_params,
    init_spheroid_opt_state,
    build_train_step_spheroid,
    load_torch_pt_to_jax,
    discover_latest_checkpoint,
    AsyncCheckpointManager,
)


def parse_args():
    parser = argparse.ArgumentParser(description="CerebroSpheroid V4 JAX High-Throughput Trainer")
    parser.add_argument("--steps", "--max_steps", dest="steps", type=int, default=50000, help="Total target steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per TPU device")
    parser.add_argument("--total_window", type=int, default=1536, help="Total sequence budget for elastic sliding window")
    parser.add_argument("--prompt_len", type=int, default=256, help="Prompt context length (used when --elastic_window is disabled)")
    parser.add_argument("--target_len", type=int, default=768, help="Deliberated target rollout length (used when --elastic_window is disabled)")
    parser.add_argument("--elastic_window", action="store_true", default=True, help="Enable elastic sliding context window")
    parser.add_argument("--lr", type=float, default=2.0e-3, help="Peak SpheroidOptimizer learning rate")
    parser.add_argument("--min_lr", type=float, default=1.0e-4, help="Final annealed learning rate")
    parser.add_argument("--preset", type=str, default="small", choices=["small", "wide"])
    parser.add_argument("--ckpt_dir", "--drive_dir", dest="ckpt_dir", type=str, default="/content/drive/MyDrive/cerebro_checkpoints", help="Destination checkpoint directory")
    parser.add_argument("--scratch_dir", type=str, default="/tmp/cerebro_ckpts", help="Fast local disk for staging")
    parser.add_argument("--save_every", type=int, default=100, help="Steps between latest.pt async saves")
    parser.add_argument("--periodic_every", type=int, default=500, help="Steps between milestone archive saves")
    parser.add_argument("--log_every", type=int, default=25, help="Steps between console progress lines")
    parser.add_argument("--cloud_sync", action="store_true", default=True, help="Enable automatic checkpoint syncing with bioflo.in")
    parser.add_argument("--cloud_url", type=str, default="https://bioflo.in/api/cerebro/sync.php", help="bioflo.in sync endpoint")
    parser.add_argument("--cloud_token", type=str, default="CerebroSpheroid_Sync_Secure_2026", help="Secret sync token")
    return parser.parse_args()


def main():
    args = parse_args()
    devices = jax.devices()
    print("=" * 80)
    print("🧠 CEREBRO-SPHEROID V4: JAX HIGH-THROUGHPUT ENGINE")
    print(f"Devices        : {devices} ({len(devices)} active core(s))")
    print(f"Backend Kind   : {devices[0].device_kind}")
    print(f"Batch Size     : {args.batch_size} | Window: {args.prompt_len + args.target_len} tokens")
    print(f"Target Horizon : {args.steps:,} steps")
    print(f"Checkpoint Dir : {args.ckpt_dir}")
    print("=" * 80)

    # 1. Initialize Model Configuration
    preset_kwargs = PRESETS_V4.get(args.preset, PRESETS_V4["small"]).copy()
    config = ModelConfigV4(vocab_size=32000, **preset_kwargs)

    # 2. Tokenizer
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b_v2")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # 3. Autonomous Checkpoint Discovery & Cloud Synchronization
    if args.cloud_sync:
        try:
            from src.cloud_sync import auto_sync_checkpoints
            auto_sync_checkpoints(local_dir=args.ckpt_dir, url=args.cloud_url, token=args.cloud_token)
        except Exception as sync_err:
            print(f"⚠️ [cloud_sync] Check failed: {sync_err}")

    search_dirs = [args.ckpt_dir, args.scratch_dir, "/content"]
    latest_ckpt, max_step, meta = discover_latest_checkpoint(search_dirs)

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
                rem_step = m_info.get("manifest", {}).get("latest", {}).get("step", 0)
                if max_step > rem_step:
                    print(f"🌐 [cloud_sync] Initial sync: Uploading restored Step {max_step:,} to bioflo.in asynchronously...")
                    push_checkpoint_async(
                        latest_ckpt,
                        step=max_step,
                        loss=float(meta.get("loss", 0.0)) if meta else 0.0,
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

    param_count = sum(p.size for p in params.values() if jnp.issubdtype(p.dtype, jnp.inexact))
    print(f"Model Scale    : {param_count:,} parameters ({param_count * 2 / (1024**2):.1f} MB in bf16)")

    # 4. Initialize SpheroidOptimizer & Build JIT Train Step
    opt_state = init_spheroid_opt_state(params)
    train_step = build_train_step_spheroid(config, buffers, pad_id=pad_id, weight_decay=0.01)
    ckpt_mgr = AsyncCheckpointManager(
        target_dir=args.ckpt_dir,
        local_scratch_dir=args.scratch_dir,
        cloud_sync=args.cloud_sync,
        cloud_url=args.cloud_url,
        cloud_token=args.cloud_token,
    )

    # 5. Data Stream Generator (OpenHermes-2.5) with Elastic Sliding Buckets
    from datasets import load_dataset
    from src.elastic_window import ElasticBucketBatcher, get_default_elastic_buckets

    print("\n📦 Initializing streaming reasoning data (teknium/OpenHermes-2.5)...")
    dataset = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
    data_iter = iter(dataset)

    if args.elastic_window:
        buckets = get_default_elastic_buckets(args.total_window)
        print(f"🌊 Elastic Sliding Context Window Active (Budget: {args.total_window} tokens):")
        for idx, (b_name, bp, bt) in enumerate(buckets):
            print(f"   Bucket {idx}: {b_name:<18} (Prompt: {bp:4d} tok -> Target: {bt:4d} tok)")
        batcher = ElasticBucketBatcher(
            tokenizer, data_iter, batch_size=args.batch_size, total_window=args.total_window, buckets=buckets, framework="jax"
        )
    else:
        buckets = [("static", args.prompt_len, args.target_len)]
        batcher = ElasticBucketBatcher(
            tokenizer, data_iter, batch_size=args.batch_size, total_window=args.prompt_len + args.target_len, buckets=buckets, framework="jax"
        )

    # 6. Training Loop with Smooth Warmup and SpheroidOptimizer Updates
    best_ema_loss = float("inf")
    ema_loss = None
    t_start = time.time()
    current_step = start_step

    print("\n" + "=" * 80)
    print(f"🚀 Training Commenced: Steps {start_step:,} -> {args.steps:,}")
    print(f"💎 Matrix Operators: Quintic Newton-Schulz Polar (Single-Moment Memory)")
    print(f"🛡️ Recurrent Deliberation: Contractive Spectral Guard (rho <= 1.0)")
    print(f"⚡ Asynchronous Cloud Sync: 0ms TPU Pause")
    print("=" * 80 + "\n")

    try:
        for step in range(start_step, args.steps + 1):
            current_step = step
            b_idx, b_name, q_batch, r_batch = batcher.next_batch()

            # Schedule
            step_offset = step - start_step
            total_rem = max(1, args.steps - start_step)
            if step_offset < 100:
                lr = args.min_lr + (args.lr - args.min_lr) * (float(step_offset) / 100.0)
            else:
                progress = float(step_offset - 100) / max(1, total_rem - 100)
                lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))

            t0 = time.time()
            params, opt_state, loss = train_step(params, opt_state, q_batch, r_batch, lr)
            loss_val = float(loss)
            dt = time.time() - t0

            if ema_loss is None:
                ema_loss = loss_val
            else:
                ema_loss = 0.96 * ema_loss + 0.04 * loss_val

            if step % args.log_every == 0 or step == start_step:
                ppl = math.exp(min(loss_val, 15.0))
                ema_ppl = math.exp(min(ema_loss, 15.0))
                seq_len = q_batch.shape[1] + r_batch.shape[1]
                tok_s = (args.batch_size * seq_len) / max(dt, 1e-4)
                eta_h = ((args.steps - step) * dt) / 3600.0
                print(
                    f"step {step:5d}/{args.steps:,} | {b_name:<16} | loss {loss_val:6.4f} (ema {ema_loss:6.4f}) | "
                    f"ema_ppl {ema_ppl:6.1f} | lr {lr:.2e} | {dt*1000:5.1f} ms/step | {tok_s:7.1f} tok/s | ETA: {eta_h:4.2f}h"
                )

            # Async checkpointing
            if step % args.save_every == 0:
                ckpt_mgr.dispatch(
                    params,
                    buffers,
                    "cerebro_spheroid_v4_jax_latest.pt",
                    meta_dict={"step": step, "loss": loss_val, "ema_loss": ema_loss, "best_ema_loss": best_ema_loss},
                )

            if step_offset > 150 and ema_loss < best_ema_loss - 0.01:
                best_ema_loss = ema_loss
                print(f"  ⭐ GENUINE RECORD (EMA): Loss {best_ema_loss:.4f} (PPL {math.exp(best_ema_loss):.1f}) -> Syncing best.pt")
                ckpt_mgr.dispatch(
                    params,
                    buffers,
                    "cerebro_spheroid_v4_jax_best.pt",
                    meta_dict={"step": step, "loss": loss_val, "ema_loss": best_ema_loss},
                )

            if step % args.periodic_every == 0:
                ckpt_mgr.dispatch(
                    params,
                    buffers,
                    f"cerebro_spheroid_v4_jax_step_{step}.pt",
                    meta_dict={"step": step, "loss": loss_val, "ema_loss": ema_loss},
                )

    except (KeyboardInterrupt, Exception) as e:
        print(f"\n⚠️ Interrupted or stopped: {e}")
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
        ckpt_mgr.shutdown(wait=True)

    print("=" * 80)
    print(f"🎉 Training session finished at step {current_step:,}.")
    print(f"Checkpoints preserved in: {args.ckpt_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
