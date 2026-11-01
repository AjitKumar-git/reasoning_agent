"""
Test Generation Outputs from CerebroSpheroid Models:
1. cerebro_spheroid_v4_small_kaggle_best.pt (127M params, Llama 32k vocab)
2. cerebro_spheroid_v3_colab_best.pt (162M params, GPT-2 50k vocab)
"""

import os
import sys
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.corpus import build_tokenizer
from src.spheroid_v4 import build_spheroid_v4
from src.spheroid_v3 import build_spheroid_v3

PROMPTS = [
    "What is the capital of France?",
    "If you have 3 apples and you eat one, how many are left?",
    "Explain what photosynthesis is in simple terms.",
    "Write a short haiku about the ocean.",
]

def load_v4_small(ckpt_path, device):
    tok = build_tokenizer("llama")
    model = build_spheroid_v4("v4_small", vocab_size=len(tok))
    model.load_compatible_checkpoint(ckpt_path, device=device)
    model.to(device=device, dtype=torch.float32)
    model.eval()
    return model, tok

def load_v3_colab(ckpt_path, device):
    tok = build_tokenizer("gpt2")
    model = build_spheroid_v4("v4_small", vocab_size=len(tok)) # in benchmark, v4 was built for it
    model.load_compatible_checkpoint(ckpt_path, device=device)
    model.to(device=device, dtype=torch.float32)
    model.eval()
    return model, tok

def run_tests():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    models = [
        ("CerebroSpheroid V4 Small (Kaggle Champion)", "checkpoints/cerebro_spheroid_v4_small_kaggle_best.pt", load_v4_small),
        ("CerebroSpheroid V3/V4 (Colab GPT-2)", "/home/ajit/Documents/cerebro_spheroid_v3_colab_best.pt", load_v3_colab),
    ]

    for model_name, ckpt_path, loader in models:
        print(f"\n{'='*70}\nMODEL: {model_name}\nPATH:  {ckpt_path}\n{'='*70}")
        if not os.path.exists(ckpt_path):
            print(f"File not found: {ckpt_path}")
            continue

        try:
            model, tok = loader(ckpt_path, device)
        except Exception as e:
            print(f"Failed to load: {e}")
            continue

        pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
        eos_id = tok.eos_token_id

        for prompt in PROMPTS:
            formatted_prompt = f"User: {prompt}\nAssistant:"
            input_ids = tok.encode(formatted_prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                out_tokens = model.generate(
                    input_ids,
                    thinking_cycles=4,
                    max_new_tokens=40,
                    temperature=0.7,
                    top_k=40,
                    repetition_penalty=1.2,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                )

            # Decode only newly generated tokens
            output_text = tok.decode(out_tokens[0], skip_special_tokens=True).strip()
            print(f"\n[Prompt]: {prompt}")
            print(f"[Generated]: {output_text}")

if __name__ == "__main__":
    run_tests()
