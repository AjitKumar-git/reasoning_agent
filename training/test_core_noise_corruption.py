"""
Core Noise Corruption & Intervention Probe for cerebro_spheroid_v3full_best.pt (162M).

Tests whether the 162M model trained on Kaggle is genuinely reading from and
dependent on the Spheroid Volumetric Core, or if the core is just an unused spectator.

Interventions:
1. Intact Model (Baseline)
2. Zeroed Spheroid Core (state = 0 across all 16 shells)
3. Spheroid Core with Gaussian Noise (sigma = 0.2, 0.5, 1.0, 3.0)
4. Scrambled Core (Randomly permuting shell ordering across the 16 laminae)
5. Layer-by-Layer Shell Ablation (Subcortical Shell 0 vs Cortical Shell 15)

Measures:
- Next-Token Prediction Perplexity (CE Loss)
- Top-1 Token Agreement with Intact Model
- Qualitative Text Generation Breakdown
"""

import math
import os
import sys
import torch
import torch.nn.functional as F

os.environ["HF_HOME"] = os.path.expanduser("~/.hf_cache")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoTokenizer
from src.spheroid_v3 import build_spheroid_v3, training_rollout


def run_corruption_probe():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(" SPHEROID CORE NOISE CORRUPTION & DEPENDENCY PROBE")
    print(f" Model: checkpoints/cerebro_spheroid_v3full_best.pt (162M) on {device}")
    print("=" * 75)

    ckpt_path = os.path.abspath("checkpoints/cerebro_spheroid_v3full_best.pt")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = build_spheroid_v3("laptop").to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False), strict=True)
    model.eval()

    # Evaluation test batch (GSM8K & OpenHermes style prompts & target completions)
    test_data = [
        ("Question: If a train leaves Chicago at 2:00 PM traveling at 50 mph, how many miles has it traveled by 5:00 PM? Answer:",
         " The train travels for 3 hours. 50 * 3 = 150 miles."),
        ("Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether? Answer:",
         " In May Natalia sold 48 / 2 = 24 clips. Altogether she sold 48 + 24 = 72 clips."),
        ("Question: What is photosynthesis? Answer:",
         " Photosynthesis is the biological process by which green plants use sunlight to synthesize food from carbon dioxide and water."),
        ("Question: Janet has 16 eggs. She uses 3 to make a cake and 4 for pancakes. How many eggs are left? Answer:",
         " Janet used 3 + 4 = 7 eggs. She has 16 - 7 = 9 eggs left."),
    ]

    prompts = [p for p, _ in test_data]
    answers = [a for _, a in test_data]

    # Tokenize
    max_p = max(len(tokenizer.encode(p)) for p in prompts)
    max_a = max(len(tokenizer.encode(a)) for a in answers)

    prompt_toks = torch.zeros(len(prompts), max_p, dtype=torch.long, device=device)
    target_toks = torch.full((len(answers), max_a), -100, dtype=torch.long, device=device)

    for i, (p, a) in enumerate(zip(prompts, answers)):
        p_ids = tokenizer.encode(p)
        a_ids = tokenizer.encode(a)
        prompt_toks[i, :len(p_ids)] = torch.tensor(p_ids, device=device)
        target_toks[i, :len(a_ids)] = torch.tensor(a_ids, device=device)

    # -------------------------------------------------------------
    # 1. Quantitative CE Loss & Degradation Under Noise
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print(" 1. QUANTITATIVE IMPACT OF CORE CORRUPTION (Perplexity & Token Agreement)")
    print("=" * 75)

    def eval_with_core_intervention(intervention_fn):
        orig_circulate = model.volumetric_core.circulate_mid_tensors

        def wrapped_circulate(state, num_cycles=1):
            out_state = orig_circulate(state, num_cycles=num_cycles)
            return intervention_fn(out_state)

        model.volumetric_core.circulate_mid_tensors = wrapped_circulate
        with torch.no_grad():
            logits = training_rollout(
                model,
                prompt_toks,
                target_toks.shape[1],
                targets=target_toks,
                thinking_cycles=1,
            )
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_toks.reshape(-1), ignore_index=-100)
            ppl = math.exp(min(loss.item(), 20.0))
        model.volumetric_core.circulate_mid_tensors = orig_circulate
        return loss.item(), ppl, logits

    # Baseline (Intact Core)
    loss_base, ppl_base, logits_base = eval_with_core_intervention(lambda s: s)
    preds_base = logits_base.argmax(dim=-1)

    interventions = [
        ("Intact Model (Baseline)", lambda s: s),
        ("Noise: sigma = 0.2", lambda s: s + torch.randn_like(s) * 0.2),
        ("Noise: sigma = 0.5", lambda s: s + torch.randn_like(s) * 0.5),
        ("Noise: sigma = 1.0", lambda s: s + torch.randn_like(s) * 1.0),
        ("Noise: sigma = 2.0", lambda s: s + torch.randn_like(s) * 2.0),
        ("Noise: sigma = 5.0", lambda s: s + torch.randn_like(s) * 5.0),
        ("Zeroed Core (state = 0)", lambda s: torch.zeros_like(s)),
        ("Inverted Shell Order (K-1 -> 0)", lambda s: torch.flip(s, dims=[1])),
        ("Random Shell Permutation", lambda s: s[:, torch.randperm(s.shape[1]), :]),
    ]

    print(f"{'Condition':<32} | {'Loss (NLL)':<10} | {'Delta NLL':<10} | {'Perplexity':<10} | {'Agreement %'}")
    print("-" * 75)

    for name, fn in interventions:
        loss, ppl, logits = eval_with_core_intervention(fn)
        preds = logits.argmax(dim=-1)
        valid_mask = target_toks != -100
        agreement = (preds[valid_mask] == preds_base[valid_mask]).float().mean().item() * 100.0
        delta = loss - loss_base
        print(f"{name:<32} | {loss:<10.4f} | {delta:<+10.4f} | {ppl:<10.2f} | {agreement:<5.1f}%")

    # -------------------------------------------------------------
    # 2. Qualitative Text Generation Under Progressive Noise
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print(" 2. QUALITATIVE GENERATIVE BREAKDOWN UNDER CORE NOISE")
    print("=" * 75)

    sample_prompt = "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer: Let's solve this step by step:"
    toks = tokenizer.encode(sample_prompt, return_tensors="pt").to(device)

    noise_levels = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0]

    for sigma in noise_levels:
        orig_circulate = model.volumetric_core.circulate_mid_tensors

        def noisy_circulate(state, num_cycles=1):
            out = orig_circulate(state, num_cycles=num_cycles)
            if sigma > 0.0:
                out = out + torch.randn_like(out) * sigma
            return out

        model.volumetric_core.circulate_mid_tensors = noisy_circulate
        with torch.no_grad():
            gen_toks = model.solve_with_active_thinking(
                toks,
                thinking_cycles=1,
                max_answer_tokens=32,
                temperature=0.1,
                repetition_penalty=1.2,
            )
            text = tokenizer.decode(gen_toks[0].cpu().tolist(), skip_special_tokens=True)
            label = "Baseline (No Noise)" if sigma == 0.0 else f"Noise sigma = {sigma}"
            print(f"[{label}]:")
            print(f"  \"{text.strip()}\"\n")
        model.volumetric_core.circulate_mid_tensors = orig_circulate

    # -------------------------------------------------------------
    # 3. Scientific Verdict
    # -------------------------------------------------------------
    print("=" * 75)
    print(" 3. SCIENTIFIC VERDICT")
    print("=" * 75)


if __name__ == "__main__":
    run_corruption_probe()
