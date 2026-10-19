# CerebroSpheroid Reasoning Agent

A biological brain-inspired AI architecture driven by a continuous 3D spherical core ($B^3$), contractive unit-gain dynamics, and **zero output token feedback** ($x_{t+1} \neq [x_{1..t}, y_t]$).

---

## Architecture Overview

1. **Afferent Sensory Input DNN**: 2-layer SwiGLU ResNet encoding discrete token sequences into smooth continuous boundary flux on $S^2$.
2. **Volumetric Manifold Core ($B^3$)**: 16 concentric spherical shells coupled through 4D radial wave diffusion and 40 subcortical axonal shortcut tracts.
3. **Subconscious Deliberation**: Internal cognitive thinking latency unrolling $K$ cycles through contractive unit-gain dynamics before speech emission.
4. **Efferent Motor Output DNN**: Dual-tap decoder reading superficial cortex (Shell 15) and deep thalamic attractor states (Shell 0) without feeding emitted tokens back into inputs.

---

## Directory Organization

```
reasoning_agent/
├── main.py                     # Unified CLI entrypoint (infer, benchmark, train, interactive)
├── cerebro_spheroid.py         # Root compatibility wrapper exporting model architecture
│
├── src/                        # Core Python Engine
│   ├── model.py                # SwiGLU Sensory DNN, Volumetric Core, Motor Output DNN
│   ├── inference.py            # ReasoningAgent class & benchmark suite
│   ├── data_loaders.py         # Unified streaming data loaders (LMSYS, GSM8K, SVAMP)
│   └── __init__.py
│
├── training/                   # Training Engines
│   ├── train_laptop.py         # Unified multi-dataset laptop trainer
│   ├── standalone/             # Self-contained individual trainers (GSM8K, LMSYS, SVAMP)
│   └── remote/                 # Dual-T4 & Cloud TPU v3-8 Kaggle / Colab scripts & notebooks
│       ├── kaggle_dual_gpu.py
│       ├── kaggle_tpu.py
│       ├── colab_2b.py
│       ├── CerebroSpheroid_2B_Kaggle_Training.ipynb
│       └── CerebroSpheroid_2B_Kaggle_TPU.ipynb
│
├── checkpoints/                # Model Checkpoints
│   ├── cerebro_spheroid_100m.pt       # Trained cognitive reasoning checkpoint (107.2M params)
│   └── cerebro_spheroid_lmsys_100m.pt # Trained LMSYS conversational checkpoint
│
└── data/
    └── lmsys_chat_1m/          # 6 local Parquet partitions (1,000,000 conversations, 1.42 GB)
```

---

## Quickstart Commands

### 1. Run Active Reasoning on a Single Prompt
```bash
python3 main.py infer --prompt "Problem: Maya has 8 markers. She buys 4 more and gives 3 away. How many markers does she have? Answer:"
```

### 2. Run Comprehensive Reasoning Benchmark Suite
```bash
python3 main.py benchmark
```

### 3. Start Living System Interactive Session (Continuous Resting Heartbeat)
Keeps the 3D core continuously pulsating in the background like a living neural system (Default Mode Network):
```bash
python3 main.py living --hz 4.0
```

### 4. Start Standard Interactive Thinking REPL
```bash
python3 main.py interactive
```

### 5. Scaled 1-Million Step Training with Plateau Early-Stopping
Trains up to 1,000,000 steps using Truncated Backpropagation Through Time (TBPTT) and hybrid curriculum (70% reasoning + 30% LMSYS). Automatically stops when no loss improvement occurs across 100 consecutive evaluation checks:
```bash
# Scaled 1M training on Hybrid Reasoning Curriculum
python3 main.py train-1m --dataset hybrid --patience 100 --batch_size 2

# Scaled 1M training on LMSYS-Chat-1M full dataset
python3 main.py train-1m --dataset lmsys --patience 100 --batch_size 2
```

### 6. Quick Laptop Training
```bash
# Train on local 1M conversation dataset
python3 main.py train --dataset lmsys --steps 300 --batch_size 2

# Train on GSM8K multi-step reasoning
python3 main.py train --dataset gsm8k --steps 300 --batch_size 2

# Train on SVAMP adversarial edge cases
python3 main.py train --dataset svamp --steps 300 --batch_size 2
```
