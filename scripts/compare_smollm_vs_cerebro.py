#!/usr/bin/env python3
"""
Head-to-Head Comparative Benchmark:
CerebroSpheroid V4 (118M) vs. Hugging Face SmolLM2-135M-Instruct (135M).
"""

import os
import sys
import torch

os.environ["HF_HOME"] = os.path.expanduser("~/.cache/hf_home")
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/hf_home")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.corpus import build_tokenizer
from src.spheroid_v4 import build_spheroid_v4


def run_head_to_head(cerebro_ckpt: str = "checkpoints/cerebro_spheroid_v4_step149200_best.pt", device: str = "cuda:0"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("🥊 HEAD-TO-HEAD BENCHMARK: CerebroSpheroid V4 (118M) vs. SmolLM2-135M-Instruct (135M)")
    print("=" * 80)

    # 1. Load SmolLM2-135M-Instruct
    print("\n📦 Loading SmolLM2-135M-Instruct...")
    smol_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    smol_tok = AutoTokenizer.from_pretrained(smol_id)
    smol_model = AutoModelForCausalLM.from_pretrained(smol_id).to(device)
    smol_model.eval()
    smol_params = sum(p.numel() for p in smol_model.parameters())
    print(f"✅ SmolLM2-135M Loaded: {smol_params:,} parameters.")

    # 2. Load CerebroSpheroid V4
    print(f"\n🧠 Loading CerebroSpheroid V4 ({cerebro_ckpt})...")
    cerebro_tok = build_tokenizer("llama")
    cerebro_model = build_spheroid_v4("v4_small", vocab_size=len(cerebro_tok))
    cerebro_model.load_compatible_checkpoint(cerebro_ckpt, device=device)
    cerebro_model.to(device=device, dtype=torch.float32)
    cerebro_model.eval()
    cerebro_params = sum(p.numel() for p in cerebro_model.parameters())
    print(f"✅ CerebroSpheroid V4 Loaded: {cerebro_params:,} parameters.")

    test_queries = [
        # Spelling / Tokenizer blindness
        "How many letters 'r' are in the word strawberry?",
        
        # Temporal reasoning
        "If yesterday was Tuesday, what day is tomorrow?",
        "A train leaves at 2:00 PM and travels for 3 hours. What time does it arrive?",
        
        # Hallucination & False Premise Traps
        "Who was the first person to walk on the Sun?",
        "What is the capital city of Mars?",
        "Can a human breathe underwater without equipment?",
        
        # World Facts & Coding
        "What is the capital of France?",
        "What keyword is used in Python to define a function?"
    ]

    print("\n" + "=" * 80)
    print("🔬 COMPARATIVE GENERATION RESULTS")
    print("=" * 80)

    for idx, q in enumerate(test_queries, 1):
        print(f"\n[Test {idx}] Prompt: \"{q}\"")
        print("-" * 80)

        # SmolLM2 Generation
        smol_chat = [{"role": "user", "content": q}]
        smol_input_text = smol_tok.apply_chat_template(smol_chat, tokenize=False, add_generation_prompt=True)
        smol_inputs = smol_tok(smol_input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            smol_out = smol_model.generate(
                **smol_inputs,
                max_new_tokens=48,
                temperature=0.2,
                do_sample=True,
                pad_token_id=smol_tok.eos_token_id
            )
        smol_gen_ids = smol_out[0][smol_inputs["input_ids"].shape[1]:]
        smol_ans = smol_tok.decode(smol_gen_ids, skip_special_tokens=True).strip()

        # CerebroSpheroid Generation
        with torch.no_grad():
            cerebro_ans = cerebro_model.generate_text(
                q,
                tokenizer=cerebro_tok,
                thinking_cycles=3,
                max_new_tokens=48,
                temperature=0.2,
                repetition_penalty=1.1,
                device=device,
                strip_bio_prefix=True
            ).strip()

        print(f"🤖 SmolLM2-135M : {smol_ans}")
        print(f"🧠 Cerebro-118M : {cerebro_ans}")
        print("-" * 80)


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/cerebro_spheroid_v4_step149200_best.pt"
    run_head_to_head(ckpt)
