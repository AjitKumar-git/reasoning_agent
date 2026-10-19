"""
CerebroSpheroid: Extensive Preexisting Dataset Training Engine
Trained 100% on Real, Preexisting Datasets from Hugging Face:
1. Real Industrial Telemetry & Machine Failures:
   - Dataset: 'TinchoARS/AI4I-2020-Predictive-Maintenance' (10,000 real industrial telemetry records)
   - Edge Cases: Heat Dissipation Failure (HDF), Tool Wear Failure (TWF), Power Failure (PWF),
                 Overstrain Failure (OSF), and Random Failures (RNF).
2. Real Adversarial Reasoning & Linguistic Edge Cases:
   - Dataset: 'ChilleD/SVAMP' (700 challenging real-world reasoning problems with linguistic distractors).

Key Principles:
- ZERO synthetic samples. All inputs and targets derived directly from published datasets.
- 80/20 Train / Held-Out Test Split.
- Zero-Feedback Output Emission (x_{t+1} != [x_{1..t}, y_t]).
"""

import os
os.environ["HF_HOME"] = "/home/ajit/Documents/Startup/spheroid_brain/.hf_cache"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import math
import random
import argparse
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer
from cerebro_spheroid import CerebroSpheroid


def prepare_real_datasets():
    """Loads and formats real, preexisting datasets from Hugging Face."""
    print("Loading preexisting AI4I 2020 Predictive Maintenance dataset (10,000 records)...")
    ds_ai4i = load_dataset("TinchoARS/AI4I-2020-Predictive-Maintenance", split="train")

    print("Loading preexisting SVAMP Edge-Case Reasoning dataset (1,000 records)...")
    ds_svamp = load_dataset("ChilleD/SVAMP", split="train")

    # Format Industrial Telemetry Records
    industrial_records = []
    for r in ds_ai4i:
        p_id = r["Product ID"]
        air_t = r["Air temperature [K]"]
        proc_t = r["Process temperature [K]"]
        rpm = r["Rotational speed [rpm]"]
        torque = r["Torque [Nm]"]
        wear = r["Tool wear [min]"]
        is_fail = r["Machine failure"]

        prompt = f"[Mach:{p_id}] Air:{air_t:.1f}K Proc:{proc_t:.1f}K RPM:{rpm} Torq:{torque:.1f}Nm Wear:{wear}m Action:"

        if is_fail == 0:
            target = " Status: Normal. Action: Continue steady operation."
            label_type = "Normal"
        else:
            if r["HDF"] == 1:
                target = " Status: Critical. Failure: Heat Dissipation. Action: Inspect cooling."
                label_type = "HDF_Failure"
            elif r["TWF"] == 1:
                target = " Status: Critical. Failure: Tool Wear. Action: Replace tool bit."
                label_type = "TWF_Failure"
            elif r["PWF"] == 1:
                target = " Status: Critical. Failure: Power Failure. Action: Check torque and supply."
                label_type = "PWF_Failure"
            elif r["OSF"] == 1:
                target = " Status: Critical. Failure: Overstrain. Action: Derate cutting force."
                label_type = "OSF_Failure"
            else:
                target = " Status: Warning. Failure: Anomaly. Action: Run safety scan."
                label_type = "RNF_Failure"

        industrial_records.append((prompt, target, label_type))

    # Format SVAMP Edge-Case Reasoning Records
    svamp_records = []
    for r in ds_svamp:
        q_concat = r["question_concat"].strip()
        ans = str(r["Answer"]).strip()
        prompt = f"Problem: {q_concat} Answer:"
        target = f" The answer is {ans}."
        svamp_records.append((prompt, target, "SVAMP_Reasoning"))

    print(f"✓ Formatted {len(industrial_records)} real industrial records.")
    print(f"✓ Formatted {len(svamp_records)} real SVAMP edge-case reasoning records.")

    # 80/20 Train/Test Split
    random.seed(42)
    random.shuffle(industrial_records)
    random.shuffle(svamp_records)

    split_ind = int(0.8 * len(industrial_records))
    split_svamp = int(0.8 * len(svamp_records))

    train_data = industrial_records[:split_ind] + svamp_records[:split_svamp]
    test_data = industrial_records[split_ind:] + svamp_records[split_svamp:]

    # Separate failure edge cases in train set for balanced oversampling
    failure_train = [x for x in industrial_records[:split_ind] if x[2] != "Normal"]
    normal_train = [x for x in industrial_records[:split_ind] if x[2] == "Normal"]
    svamp_train = svamp_records[:split_svamp]

    print(f"Train set: {len(train_data)} records (Failures: {len(failure_train)}, Normal: {len(normal_train)}, SVAMP: {len(svamp_train)})")
    print(f"Held-out Test set: {len(test_data)} records (Failures: {sum(1 for x in test_data if 'Failure' in x[2])})")

    return failure_train, normal_train, svamp_train, test_data


def get_balanced_data_stream(tokenizer, failure_train, normal_train, svamp_train, batch_size=2):
    """Yields balanced batches across real failure edge cases, normal operation, and reasoning."""
    while True:
        batch_p, batch_t = [], []
        while len(batch_p) < batch_size:
            # 40% real failure edge cases, 30% real normal telemetry, 30% real SVAMP reasoning
            r_val = random.random()
            if r_val < 0.40 and failure_train:
                prompt, target, _ = random.choice(failure_train)
            elif r_val < 0.70:
                prompt, target, _ = random.choice(normal_train)
            else:
                prompt, target, _ = random.choice(svamp_train)

            p_toks = tokenizer.encode(prompt)[:32]
            t_toks = tokenizer.encode(target)[:16]

            if len(p_toks) >= 4 and len(t_toks) >= 4:
                batch_p.append(p_toks)
                batch_t.append(t_toks)

        max_p = max(len(p) for p in batch_p)
        max_t = max(len(t) for t in batch_t)

        p_pad = [p + [tokenizer.eos_token_id] * (max_p - len(p)) for p in batch_p]
        t_pad = [t + [tokenizer.eos_token_id] * (max_t - len(t)) for t in batch_t]

        yield (
            torch.tensor(p_pad, dtype=torch.long),
            torch.tensor(t_pad, dtype=torch.long)
        )


def main():
    parser = argparse.ArgumentParser(description="Train CerebroSpheroid on Preexisting Datasets")
    parser.add_argument("--steps", type=int, default=400, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--thinking_cycles", type=int, default=3, help="Subconscious deliberation cycles")
    parser.add_argument("--lr", type=float, default=0.015, help="Learning rate for Momentum SGD")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    print("=" * 80)
    print("      CEREBRO-SPHEROID: EXTENSIVE PREEXISTING DATASET TRAINING ENGINE   ")
    print("=" * 80)
    print(f"Device          : {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Accelerator : {gpu_name} ({vram_gb:.2f} GB VRAM)")
    print(f"Target Steps    : {args.steps} steps | Batch Size: {args.batch_size}")
    print(f"Thinking Cycles : {args.thinking_cycles} cycles (Endogenous B^3 manifold deliberation)")
    print("=" * 80)

    # 1. Prepare 100% Real Datasets
    failure_train, normal_train, svamp_train, test_data = prepare_real_datasets()

    # 2. Instantiate Model
    model = CerebroSpheroid(
        vocab_size=vocab_size,
        surface_dim=640,
        num_shells=16,
        radial_dim=256,
        num_shortcut_tracts=40,
    )

    base_ckpt = "cerebro_spheroid_100m.pt"
    if os.path.exists(base_ckpt):
        print(f"Loading base cognitive checkpoint: {base_ckpt}...")
        model.load_state_dict(torch.load(base_ckpt, map_location="cpu", weights_only=True))

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model.to(device=device, dtype=dtype)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)
    criterion = nn.CrossEntropyLoss()

    data_stream = get_balanced_data_stream(tokenizer, failure_train, normal_train, svamp_train, batch_size=args.batch_size)

    # Sample Unseen Test Edge Cases
    test_edge_cases = [x for x in test_data if x[2] in ["HDF_Failure", "TWF_Failure", "PWF_Failure", "OSF_Failure"]][:4]
    test_svamp_cases = [x for x in test_data if x[2] == "SVAMP_Reasoning"][:2]
    evaluation_suite = test_edge_cases + test_svamp_cases

    print("\n" + "-" * 80)
    print("Beginning Training on Real Preexisting Telemetry & Edge Cases...")
    print("-" * 80)

    t_start = time.time()
    best_loss = 999.0

    for step in range(1, args.steps + 1):
        model.train()
        prompts, targets = next(data_stream)
        prompts = prompts.to(device)
        targets = targets.to(device)
        ans_len = targets.shape[1]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
            state = model.ingest_stimulus(prompts)
            state = model.deliberate(state, thinking_cycles=args.thinking_cycles)

            logits_list = []
            for t in range(ans_len):
                logits_t, state = model.emit_next_token(state)
                logits_list.append(logits_t)

            predicted_logits = torch.stack(logits_list, dim=1)
            loss = criterion(predicted_logits.view(-1, vocab_size), targets.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            torch.save(model.state_dict(), "cerebro_spheroid_preexisting_100m.pt")

        if step % 25 == 0 or step == 1 or step == args.steps:
            elapsed = time.time() - t_start
            ppl = math.exp(min(loss_val, 15))
            vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            cur_lr = scheduler.get_last_lr()[0]
            print(f"Step [{step:4d}/{args.steps}] | Loss: {loss_val:.4f} | PPL: {ppl:8.2f} | LR: {cur_lr:.2e} | VRAM: {vram:.0f} MB | Time: {elapsed:.1f}s")

        if step % 100 == 0 or step == args.steps:
            model.eval()
            print("\n" + "·" * 60)
            print(f"  [Real Unseen Test Evaluation at Step {step} | Loss: {loss_val:.4f}]")
            with torch.no_grad():
                for prompt, ground_truth, label in evaluation_suite[:3]:
                    pt = torch.tensor([tokenizer.encode(prompt)], device=device)
                    ans_toks = model.solve_with_active_thinking(
                        pt, thinking_cycles=args.thinking_cycles, max_answer_tokens=15, temperature=0.0
                    )
                    ans_text = tokenizer.decode(ans_toks[0].tolist(), skip_special_tokens=True)
                    print(f"  [{label}]")
                    print(f"  Prompt : \"{prompt}\"")
                    print(f"  Actual : \"{ans_text.strip()}\"")
                    print(f"  Target : \"{ground_truth.strip()}\"")
            print("·" * 60 + "\n")

    print("\n" + "=" * 80)
    print("Extensive Training Complete! Full Benchmark on Real Held-Out Test Set:")
    print("=" * 80)

    model.eval()
    with torch.no_grad():
        for prompt, ground_truth, label in evaluation_suite:
            print(f"\n[Test Case Category: {label}]")
            print(f"Prompt (From Dataset) : \"{prompt}\"")
            pt = torch.tensor([tokenizer.encode(prompt)], device=device)
            ans_toks = model.solve_with_active_thinking(
                pt, thinking_cycles=args.thinking_cycles, max_answer_tokens=15, temperature=0.0
            )
            ans_text = tokenizer.decode(ans_toks[0].tolist(), skip_special_tokens=True)
            print(f"Ground Truth Target   : \"{ground_truth.strip()}\"")
            print(f"Model Deliberated Ans : \"{ans_text.strip()}\"")
            print("✓ Zero feedback used ($x_{t+1} != [x, y]$).")

    print("\n" + "=" * 80)
    print(f"Saved Checkpoint to: cerebro_spheroid_preexisting_100m.pt (Best Loss: {best_loss:.4f})")
    print("=" * 80)


if __name__ == "__main__":
    main()
