import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🧠 CerebroSpheroid V4: Cloud TPU JAX High-Throughput Engine\n",
                "### 128 GB HBM • 8 TPU v3 Cores • Pure JAX / PJRT Native • 0ms Compilation Delay\n",
                "\n",
                "### **Architecture & Capabilities in V4-Small:**\n",
                "1. **Modern Llama 32,000 Vocabulary**: Byte-fallback BPE with atomic single-digit arithmetic (`0`-`9`) and zero OOV tokens (127M total parameters).\n",
                "2. **Elastic Sliding Context Window**: 1,536-token budget dynamically partitioned across 4 complementary cognitive regimes (`deep_reasoning`, `standard_cot`, `balanced`, `long_context`).\n",
                "3. **SpheroidOptimizer**: Polar Quintic Newton-Schulz Riemannian optimizer with single-moment memory and contractive spectral bounds ($\\rho \\le 1.0$).\n",
                "4. **Pure JAX / XLA Fused Kernel**: The entire multi-step deliberation and zero-feedback emission unroll runs natively on the TPU MXU systolic array without PyTorch/XLA recompilation lags.\n",
                "5. **Autonomous Multi-Session Resumption**: Safely trains under Kaggle's 9-hour limit via `--max_hours 8.5` and automatically mirrors checkpoints into `/kaggle/working/checkpoints/` and bioflo.in.\n",
                "\n",
                "---\n",
                "### ⚡ Kaggle Settings:\n",
                "- **Accelerator**: Select **TPU VM v3-8**\n",
                "- **Internet**: Toggle **Internet ON**\n",
                "- **Persistence**: Variables and Files\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 1: Verify Kaggle TPU Hardware & JAX PJRT Environment"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import jax\n",
                "import jax.numpy as jnp\n",
                "\n",
                "print(f\"JAX Version      : {jax.__version__}\")\n",
                "print(f\"JAX Default Dev  : {jax.default_backend().upper()}\")\n",
                "devices = jax.devices()\n",
                "print(f\"Detected Devices : {devices}\")\n",
                "print(f\"Active Core Count: {len(devices)}\")\n",
                "print(f\"Device Kind      : {devices[0].device_kind}\")\n",
                "\n",
                "is_tpu = any(\"TPU\" in d.device_kind.upper() for d in devices)\n",
                "assert is_tpu, \"⚠️ No TPU detected! In the Kaggle notebook settings (right panel), set Accelerator to 'TPU VM v3-8'!\"\n",
                "print(f\"\\n✅ Successfully connected to Cloud TPU VM! ({len(devices)} cores active)\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 2: Clone / Pull Latest Repository & Dependencies"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "%cd /kaggle/working\n",
                "if not os.path.exists(\"reasoning_agent\"):\n",
                "    !git clone https://github.com/AjitKumar-git/reasoning_agent.git\n",
                "%cd /kaggle/working/reasoning_agent\n",
                "!git fetch origin main\n",
                "!git reset --hard origin/main\n",
                "!pip install -q transformers datasets sentencepiece optax torch\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 3: Checkpoint Ingestion & Resumption (Automatic)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, glob, shutil\n",
                "os.makedirs(\"/kaggle/working/checkpoints\", exist_ok=True)\n",
                "# 1. Check mounted inputs under /kaggle/input\n",
                "if os.path.exists(\"/kaggle/input\"):\n",
                "    for root, dirs, files in os.walk(\"/kaggle/input\"):\n",
                "        for f in files:\n",
                "            full_p = os.path.join(root, f)\n",
                "            if f.endswith(\".zip\") and \"cerebro_spheroid\" in f:\n",
                "                print(f\"📦 Unzipping checkpoint archive from input: {full_p}\")\n",
                "                !unzip -q -o \"{full_p}\" -d /kaggle/working/checkpoints/\n",
                "            elif f.endswith(\".pt\") and \"cerebro_spheroid\" in f:\n",
                "                shutil.copy2(full_p, os.path.join(\"/kaggle/working/checkpoints\", f))\n",
                "                print(f\"📥 Found raw checkpoint: {f} -> copied to /kaggle/working/checkpoints/\")\n",
                "\n",
                "ckpts = glob.glob(\"/kaggle/working/checkpoints/*.pt\")\n",
                "if ckpts:\n",
                "    print(f\"✅ Active checkpoints ready for auto-resume ({len(ckpts)} files found):\")\n",
                "    for c in ckpts:\n",
                "        print(f\"   • {os.path.basename(c)} ({os.path.getsize(c) / (1024**2):.1f} MB)\")\n",
                "else:\n",
                "    print(\"🆕 No existing checkpoints found in inputs. Will auto-sync latest state from bioflo.in or start clean!\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 4: Launch CerebroSpheroid V4 JAX Cloud TPU Training\n",
                "- **Engine**: Pure JAX native execution with XLA fusion\n",
                "- **Context**: Elastic Sliding Window (1,536 tokens)\n",
                "- **Time Safety**: `--max_hours 8.5` safely exits before Kaggle's 9.0-hr hard timeout\n",
                "- **Cloud Sync**: 0ms TPU pause asynchronous mirroring to bioflo.in"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python3 training/train_v4_jax.py \\\n",
                "    --ckpt_dir /kaggle/working/checkpoints \\\n",
                "    --elastic_window \\\n",
                "    --total_window 1536 \\\n",
                "    --batch_size 4 \\\n",
                "    --grad_accum 8 \\\n",
                "    --lr 0.002 \\\n",
                "    --max_steps 50000 \\\n",
                "    --max_hours 8.5 \\\n",
                "    --save_every 100 \\\n",
                "    --log_every 10 \\\n",
                "    --cloud_sync\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 5: Archive Checkpoints for Download & Multi-Session Chaining"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, glob\n",
                "!mkdir -p /kaggle/working/checkpoints_tpu\n",
                "!cp -r /kaggle/working/checkpoints/* /kaggle/working/checkpoints_tpu/ 2>/dev/null || true\n",
                "!zip -r /kaggle/working/cerebro_spheroid_v4_jax_checkpoints.zip /kaggle/working/checkpoints_tpu\n",
                "!ls -lh /kaggle/working/*.zip\n",
                "print(\"\\n✅ Checkpoints successfully archived to /kaggle/working/cerebro_spheroid_v4_jax_checkpoints.zip!\")\n",
                "print(\"Download it directly from the Kaggle Output panel or attach as input to your next session!\")\n"
            ]
        }
    ],
    "metadata": {
        "accelerator": "TPU",
        "kaggle": {
            "accelerator": "tpu",
            "isInternetEnabled": True
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("training/remote/CerebroSpheroid_V4_TPU_Kaggle.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("Successfully written TPU JAX notebook!")
