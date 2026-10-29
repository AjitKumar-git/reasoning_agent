"""
Testing and Evaluation Suite for CerebroSpheroid V4 Small Trained Checkpoint:
checkpoints/cerebro_spheroid_v4_small_kaggle_best.pt (127M, Llama 32k BPE).

Trained for 1,000 steps on Kaggle Dual Tesla T4 GPUs on OpenHermes-2.5 and GSM8K.
Final held-out total loss: 3.8504 (GSM8K: 2.6188, OpenHermes: 5.5228).
"""

import json
import math
import os
import sys
import time
import torch
import torch.nn.functional as F

os.environ["HF_HOME"] = os.path.expanduser("~/.hf_cache")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.corpus import build_tokenizer
from src.spheroid_v3 import training_rollout
from src.spheroid_v4 import CerebroSpheroidV4, PRESETS


def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.abspath("checkpoints/cerebro_spheroid_v4_small_kaggle_best.pt")
    meta_path = os.path.abspath("checkpoints/cerebro_spheroid_v4_small_kaggle_best.meta.json")

    print("=" * 80)
    print(" CEREBRO-SPHEROID V4 SMALL KAGGLE CHECKPOINT EVALUATION")
    print(f" Device    : {device}")
    print(f" Checkpoint: {ckpt_path}")
    print("=" * 80)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # Read metadata if present
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        print("\n[Metadata from Kaggle Training Run]")
        print(f"  Step Count       : {meta.get('step')}")
        print(f"  Batch Config     : batch_size={meta.get('batch_size')}, grad_accum={meta.get('grad_accum')}, window={meta.get('window')}")
        print(f"  Training Runtime : {meta.get('elapsed_minutes', 0):.1f} minutes (~{meta.get('elapsed_minutes', 0)/60:.1f} hours)")
        print(f"  World Size       : {meta.get('world_size', 1)} GPU(s)")
        print(f"  Train Loss Mean  : {meta.get('train_mean_at_save', 0):.4f}")
        print(f"  Held-out Total CE: {meta.get('eval_total', 0):.4f}")
        by_src = meta.get("eval_by_source", {})
        print(f"    - GSM8K Loss   : {by_src.get('gsm8k', 0):.4f} (PPL: {math.exp(by_src.get('gsm8k', 0)):.2f})")
        print(f"    - OpenHermes   : {by_src.get('openhermes', 0):.4f} (PPL: {math.exp(by_src.get('openhermes', 0)):.2f})")

    # 1. Load Tokenizer & Model
    print("\n" + "-" * 80)
    print(" [1/5] Loading Llama 32k Tokenizer & V4 Small Architecture...")
    print("-" * 80)
    tokenizer = build_tokenizer("llama")

    cfg = PRESETS["v4_small"].copy()
    model = CerebroSpheroidV4(**cfg).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    load_res = model.load_state_dict(state, strict=True)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded model successfully with {total_params:,} parameters (strict=True: {load_res})")
    assert total_params == 127_218_877, f"Expected 127,218,877 parameters, got {total_params}"

    # 2. Volumetric Manifold Diagnostics
    print("\n" + "-" * 80)
    print(" [2/5] Volumetric Manifold Geometric Diagnostics (16 Concentric Shells)")
    print("-" * 80)
    test_prompt = "Question: If a car travels at 60 mph, how far does it travel in 3.5 hours? Answer:"
    toks = tokenizer.encode(test_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        brain_state, _ = model.ingest_stimulus_with_memory(toks)
        K = brain_state.shape[1]
        rms_energies = [brain_state[0, k].pow(2).mean().sqrt().item() for k in range(K)]
        mean_rms = sum(rms_energies) / len(rms_energies)

        cos_sims = []
        for k in range(K - 1):
            cs = F.cosine_similarity(brain_state[0, k:k+1], brain_state[0, k+1:k+2]).item()
            cos_sims.append(cs)
        mean_cos = sum(cos_sims) / len(cos_sims)

        # Spectral Effective Rank
        flat_state = brain_state[0].reshape(K, -1)
        U, S, V = torch.linalg.svd(flat_state, full_matrices=False)
        p = S / (S.sum() + 1e-12)
        eff_rank = torch.exp(-torch.sum(p * torch.log(p + 1e-12))).item()

    print(f"  Mean Shell RMS Energy      : {mean_rms:.4f}")
    print(f"  Manifold Effective Rank    : {eff_rank:.2f} / {K} shells")
    print(f"  Adjacent Shell Cosine Sim  : {mean_cos:.4f}")
    print("  Per-Shell RMS Energy Profile (Shell 0 [Subcortical Core] -> Shell 15 [Superficial Mantle]):")
    for i, val in enumerate(rms_energies):
        bar = "#" * int(val * 25)
        zone = "Core" if i < 5 else ("Associative" if i < 12 else "Superficial")
        print(f"    Shell {i:2d} ({zone:<11}): {val:.4f} | {bar}")

    # 3. Core Intervention Probe
    print("\n" + "-" * 80)
    print(" [3/5] Volumetric Core Structural Intervention Probe")
    print("-" * 80)
    prompt_str = "Question: Janet has 16 eggs. She bakes 3 cakes using 4 eggs each. How many eggs are left?\nAnswer:"
    answer_str = " Janet uses 3 * 4 = 12 eggs. She had 16 eggs, so 16 - 12 = 4 eggs are left. #### 4"

    p_toks = tokenizer.encode(prompt_str, return_tensors="pt").to(device)
    a_toks = tokenizer.encode(answer_str, return_tensors="pt").to(device)

    conditions = [
        ("Intact Core (Baseline)", None),
        ("Noise sigma=0.2", lambda s: s + torch.randn_like(s) * 0.2),
        ("Noise sigma=0.5", lambda s: s + torch.randn_like(s) * 0.5),
        ("Noise sigma=1.0", lambda s: s + torch.randn_like(s) * 1.0),
        ("Zeroed Core (state = 0)", lambda s: torch.zeros_like(s)),
    ]

    print(f"  {'Condition':30s} | {'NLL Loss':10s} | {'Delta NLL':10s} | {'Perplexity':12s} | {'Verdict'}")
    print("  " + "-" * 75)

    base_loss = None
    orig_circulate = model.volumetric_core.circulate_mid_tensors

    for cond_name, intervention_fn in conditions:
        if intervention_fn is not None:
            def wrapped_circulate(st, num_cycles=1, fn=intervention_fn):
                out_st = orig_circulate(st, num_cycles=num_cycles)
                return fn(out_st)
            model.volumetric_core.circulate_mid_tensors = wrapped_circulate
        else:
            model.volumetric_core.circulate_mid_tensors = orig_circulate

        with torch.no_grad():
            logits = training_rollout(model, p_toks, a_toks.shape[1], targets=a_toks, thinking_cycles=1)
            loss_val = F.cross_entropy(logits.reshape(-1, logits.size(-1)), a_toks.reshape(-1)).item()
            ppl_val = math.exp(min(loss_val, 20.0))

        model.volumetric_core.circulate_mid_tensors = orig_circulate

        if base_loss is None:
            base_loss = loss_val
            delta_str = "Baseline"
            verdict = "HEALTHY BASELINE"
        else:
            delta = loss_val - base_loss
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            verdict = "ACTIVE RECRUITMENT" if delta > 0.0 else "UNRECRUITED"

        print(f"  {cond_name:30s} | {loss_val:10.4f} | {delta_str:10s} | {ppl_val:12.2f} | {verdict}")

    # 4. Multi-Step Mathematical Reasoning (GSM8K)
    print("\n" + "-" * 80)
    print(" [4/5] Multi-Step Mathematical Reasoning (GSM8K Prompts)")
    print("-" * 80)

    gsm8k_prompts = [
        "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer: Let's solve this step by step:",
        "Question: A farmer has 12 cows and 8 sheep. If he buys 6 more cows and sells 3 sheep, how many animals does he have in total?\nAnswer:",
        "Question: A train travels 60 miles per hour. How long will it take to travel 240 miles?\nAnswer:",
    ]

    for p_idx, prompt in enumerate(gsm8k_prompts, 1):
        print(f"\n--- [GSM8K Case {p_idx}] ---")
        print(f"Prompt: {prompt.strip()}")
        p_toks = tokenizer.encode(prompt, return_tensors="pt").to(device)

        for cycles in [0, 1, 3]:
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    p_toks,
                    thinking_cycles=cycles,
                    max_new_tokens=48,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            dt = time.time() - t0
            output_tokens = out[0, p_toks.shape[1]:]
            text = tokenizer.decode(output_tokens, skip_special_tokens=True)
            print(f"\n[Thinking Cycles T={cycles}] ({dt:.2f}s):")
            print(f"{text.strip() or '<empty>'}")

    # 5. Open-Ended Instruction Following & Code Generation
    print("\n" + "-" * 80)
    print(" [5/5] Open-Ended Instruction Following & Code Generation (OpenHermes)")
    print("-" * 80)

    instruct_prompts = [
        "<|im_start|>user\nWrite a Python function to check if a string is a palindrome.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain why the sky is blue in two simple sentences.<|im_end|>\n<|im_start|>assistant\n",
    ]

    for p_idx, prompt in enumerate(instruct_prompts, 1):
        print(f"\n--- [Instruction {p_idx}] ---")
        print(f"Prompt: {prompt.strip()[:60]}...")
        p_toks = tokenizer.encode(prompt, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                p_toks,
                thinking_cycles=1,
                max_new_tokens=64,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        dt = time.time() - t0
        output_tokens = out[0, p_toks.shape[1]:]
        text = tokenizer.decode(output_tokens, skip_special_tokens=True)
        print(f"\n[Response] ({dt:.2f}s):\n{text.strip()}")

    print("\n" + "=" * 80)
    print(" EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()
