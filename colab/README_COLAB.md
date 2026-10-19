# 🚀 Google Colab High-Speed Training Guide

Training CerebroSpheroid on Google Colab will accelerate training by **10x to 30x** compared to a laptop RTX 3050.

---

### Step 1: Open Google Colab & Select GPU
1. Go to [colab.research.google.com](https://colab.research.google.com).
2. Click **File > Upload Notebook** and select:
   `colab/cerebro_spheroid_colab.ipynb`
3. In the top-right menu, select:
   **Runtime > Change runtime type > T4 GPU** (or **A100 / V100** if you have Colab Pro).

---

### Step 2: Upload Your Code & Checkpoints
In the left sidebar of Colab, click the **Folder icon (Files)**:
1. Drag and drop `colab/reasoning_agent_colab.zip` into the `/content/` area.
2. In Cell 3 of the notebook, run the unzip command:
   ```bash
   !unzip -q /content/reasoning_agent_colab.zip -d /content/
   ```
   *(This instantly unpacks `src/`, `colab/`, `main.py`, and your latest trained checkpoint `cerebro_spheroid_1m_best.pt`)*.

---

### Step 3: Run Training with 1 Click
Run Cell 4 to launch high-speed training:
```bash
!python colab/colab_run.py \
    --dataset hybrid \
    --batch_size 8 \
    --grad_accum 2 \
    --max_prompt_len 1024 \
    --max_resp_len 128 \
    --lr 5e-4 \
    --max_steps 50000 \
    --eval_interval 50 \
    --drive_dir /content/drive/MyDrive/cerebro_checkpoints
```

### Why Colab is Much Faster:
| Metric | Laptop RTX 3050 (4 GB) | Colab T4 (16 GB) | Colab A100 (40/80 GB) |
| :--- | :--- | :--- | :--- |
| **VRAM Capacity** | 3.73 GB | 16 GB | 40 GB - 80 GB |
| **Memory Bandwidth** | ~192 GB/s | 320 GB/s | **1,555 GB/s** |
| **Micro-Batch Size** | 2 | 8 | 16 - 32 |
| **Step Speed** | ~4.0s / step | **~0.3s / step** | **~0.05s / step** |
| **Speedup vs Laptop** | 1x (Baseline) | **~13x Faster** | **~80x Faster** |
| **10,000 Steps Time** | ~11 hours | **~50 minutes** | **~8 minutes** |

---

### Step 4: Test Your Brain Interactively
Run the final cell in the notebook to interactively test Maya's problem, multi-step math, and conversational reasoning directly inside Colab!
