import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🧠 CerebroSpheroid V4: Cloud TPU Training on Kaggle\n",
                "### 128 GB High-Bandwidth Memory (HBM) • PyTorch/XLA • Native bfloat16\n",
                "\n",
                "### **Architecture & Capabilities in V4-Small:**\n",
                "1. **Modern Llama 32,000 Vocabulary**: Byte-fallback BPE with atomic single-digit arithmetic (`0`-`9`) and zero OOV tokens (saving 35M parameters: 127M total).\n",
                "2. **Causal Temporal Sensory Ingestion**: Exponential recency decay + phase velocity momentum vectors replacing order-erasing centroid pooling.\n",
                "3. **SpheroidOptimizer**: Riemannian polar Newton-Schulz with persistent power iteration spectral radius guard (<1.5 ms) and decoupled row-wise RMS token momentum (<0.5 ms).\n",
                "4. **Reasoning-Focused Mix**: Multi-step instruction and math reasoning (`OpenHermes=0.45, LMSYS=0.35, GSM8K=0.15, Cognitive=0.05`).\n",
                "\n",
                "---\n",
                "### ⚡ Kaggle Settings:\n",
                "- **Accelerator**: Select **TPU VM v3-8** or **TPU v5e**\n",
                "- **Internet**: Toggle **Internet ON**\n",
                "- **Persistence**: Variables and Files\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 1: TPU Hardware & PyTorch/XLA Verification"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys, os\n",
                "try:\n",
                "    import torch_xla.core.xla_model as xm\n",
                "    device = xm.xla_device()\n",
                "    print(f\"✅ Successfully connected to Cloud TPU device: {device}\")\n",
                "except ImportError:\n",
                "    print(\"Installing torch_xla for TPU VM...\")\n",
                "    !pip install -q torch~=2.5.0 torch_xla[tpu]~=2.5.0 -f https://storage.googleapis.com/libtpu-releases/index.html\n",
                "    import torch_xla.core.xla_model as xm\n",
                "    device = xm.xla_device()\n",
                "    print(f\"✅ Successfully installed and connected to Cloud TPU: {device}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 2: Clone / Pull Latest Repository"]
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
                "!pip install -q pyarrow fastparquet transformers datasets sentencepiece\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 3: Checkpoint Restoration (Optional / Automatic)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, glob, shutil\n",
                "os.makedirs(\"checkpoints\", exist_ok=True)\n",
                "if os.path.exists(\"/kaggle/input\"):\n",
                "    for root, dirs, files in os.walk(\"/kaggle/input\"):\n",
                "        for f in files:\n",
                "            full_p = os.path.join(root, f)\n",
                "            if f.endswith(\".zip\") and \"cerebro_spheroid\" in f:\n",
                "                print(f\"Unpacking checkpoint archive: {f}\")\n",
                "                os.system(f'unzip -n -q \"{full_p}\" -d checkpoints/')\n",
                "            elif f.endswith(\".pt\") and \"cerebro_spheroid_v4\" in f:\n",
                "                print(f\"Found checkpoint file: {f}\")\n",
                "                shutil.copy2(full_p, os.path.join(\"checkpoints\", f))\n",
                "\n",
                "# Flatten nested checkpoints_out if present\n",
                "if os.path.exists(\"checkpoints/checkpoints_out\"):\n",
                "    for item in os.listdir(\"checkpoints/checkpoints_out\"):\n",
                "        shutil.move(os.path.join(\"checkpoints/checkpoints_out\", item), os.path.join(\"checkpoints\", item))\n",
                "    shutil.rmtree(\"checkpoints/checkpoints_out\", ignore_errors=True)\n",
                "\n",
                "print(\"\\nCheckpoints available:\")\n",
                "!ls -lh checkpoints/\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 4: Acceptance Verification Tests on TPU VM"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python3 -u training/test_v4_architecture.py\n",
                "!python3 -u training/test_spheroid_optimizer.py\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 5: Launch Training on Cloud TPU (Native bfloat16, SpheroidOptimizer)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python3 -u training/train_v4_full.py \\\n",
                "    --arch v4 \\\n",
                "    --preset v4_small \\\n",
                "    --tokenizer llama \\\n",
                "    --mix \"openhermes=0.45,lmsys=0.35,gsm8k=0.15,cognitive=0.05\" \\\n",
                "    --optimizer spheroid \\\n",
                "    --device xla \\\n",
                "    --dtype bf16 \\\n",
                "    --lr 0.002 \\\n",
                "    --max_steps 3000 \\\n",
                "    --batch_size 16 \\\n",
                "    --grad_accum 2 \\\n",
                "    --window 128 \\\n",
                "    --eval_window 128 \\\n",
                "    --eval_interval 100 \\\n",
                "    --save_every 50 \\\n",
                "    --save_prefix cerebro_spheroid_v4_small_tpu\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 6: Archive Checkpoints for Download"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, glob\n",
                "!mkdir -p /kaggle/working/checkpoints_tpu\n",
                "!cp -r checkpoints/* /kaggle/working/checkpoints_tpu/ 2>/dev/null || true\n",
                "!zip -r /kaggle/working/cerebro_spheroid_v4_small_tpu_checkpoints.zip /kaggle/working/checkpoints_tpu\n",
                "!ls -lh /kaggle/working/*.zip\n",
                "print(\"\\n✅ Checkpoints successfully archived to /kaggle/working/cerebro_spheroid_v4_small_tpu_checkpoints.zip!\")\n",
                "print(\"Download it directly from the right-hand Kaggle Output panel.\")\n"
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

with open("/tmp/kaggle_tpu_pull/reasoning-agent-tpu.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("Successfully written TPU notebooks!")
