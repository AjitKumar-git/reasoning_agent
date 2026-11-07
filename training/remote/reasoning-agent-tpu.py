"""
CerebroSpheroid V4: Kaggle TPU High-Throughput Headless Execution Script.
Continuously trains for 8.5 hours with hourly cloud synchronization to bioflo.in.
"""

import os
import sys
import subprocess

# PyArrow compatibility monkey-patch for Kaggle Python 3.12 / pyarrow 19+
try:
    import pyarrow as pa
    if not hasattr(pa, "PyExtensionType") and hasattr(pa, "ExtensionType"):
        pa.PyExtensionType = pa.ExtensionType
except Exception:
    pass

print("=" * 80)
print("🚀 CEREBRO-SPHEROID V4: KAGGLE TPU JAX ENGINE LAUNCH")
print("================================================================================")

# 1. Clone or pull latest repository
WORKDIR = "/kaggle/working/reasoning_agent"
if not os.path.exists(WORKDIR):
    print(f"📦 Cloning latest repository from GitHub...")
    subprocess.run(["git", "clone", "https://github.com/AjitKumar-git/reasoning_agent.git", WORKDIR], check=True)
else:
    print(f"🔄 Updating existing repository at {WORKDIR}...")

os.chdir(WORKDIR)
subprocess.run(["git", "fetch", "origin", "main"], check=True)
subprocess.run(["git", "reset", "--hard", "origin/main"], check=True)

# 2. Verify dependencies
print("\n📦 Checking / installing Python dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "datasets>=3.0.0", "pyarrow", "transformers", "sentencepiece", "optax", "torch"], check=True)

# 3. Create checkpoints directory
os.makedirs("/kaggle/working/checkpoints", exist_ok=True)

# 4. Launch train_v4_jax.py
cmd = [
    sys.executable,
    "training/train_v4_jax.py",
    "--ckpt_dir", "/kaggle/working/checkpoints",
    "--resume_target", "latest",
    "--dataset_source", "dense_edu",
    "--dtype", "auto",
    "--elastic_window",
    "--total_window", "1024",
    "--batch_size", "16",
    "--grad_accum", "4",
    "--lr", "0.0005",
    "--deliberation_mode", "scheduled",
    "--schedule_steps", "20000",
    "--min_sampling_p", "0.35",
    "--repetition_penalty", "1.25",
    "--eval_interval", "200",
    "--thalamic_anchor_weight", "0.1",
    "--max_hours", "8.5",
    "--cloud_save_hours", "1.0",
    "--log_every", "10",
    "--cloud_sync",
    "--source_node", "kaggle_tpu",
]

print("\n🚀 Executing trainer:")
print(" ".join(cmd))
print("=" * 80, flush=True)

res = subprocess.run(cmd)
sys.exit(res.returncode)
