"""
Interactive Reasoning & Cognitive Evaluation for Bipolar Hemispheric Wave Spheroid
Evaluates deliberation depth (0, 2, 4, 8 cycles) and refractory zero-feedback readout.
"""

import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, ".hf_cache"))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from src.hemispheric_spheroid import build_minimal_hemispheric_spheroid


def test_hemispheric_cognition(checkpoint_path="checkpoints/hemispheric_minimal_best.pt", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    print(f"Loading checkpoint from {checkpoint_path} onto {device} ({dtype})...")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Minimal config matching train_hemispheric_minimal
    model = build_minimal_hemispheric_spheroid()
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device=device, dtype=dtype)
    model.eval()

    test_queries = [
        "If 4 books cost $20, how much do 6 books cost?",
        "Maya has 8 markers. She buys 4 more and gives 3 away. How many markers does she have left?",
        "Why does ice float on liquid water?",
        "A car travels 60 miles per hour. How far does it go in 3 hours?",
        "What happens when you mix an acid and a base?",
        "Is 17 a prime number? Explain why.",
    ]

    print("\n" + "="*80)
    print("      BIPOLAR HEMISPHERIC WAVE SPHEROID: MULTI-CYCLE COGNITIVE TEST     ")
    print("="*80)

    for query in test_queries:
        print(f"\n[QUERY]: {query}")
        q_toks = tokenizer.encode(query, return_tensors="pt").to(device)
        for deliberation_cycles in [1, 3, 6]:
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
                    out_toks = model.solve_with_active_thinking(
                        q_toks,
                        thinking_cycles=deliberation_cycles,
                        max_answer_tokens=25,
                        temperature=0.7,
                        repetition_penalty=1.5,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            response = tokenizer.decode(out_toks[0]) if len(out_toks[0]) > 0 else "<empty>"
            print(f"  Cycles={deliberation_cycles:02d} -> {response.strip()}")

    print("\n" + "="*80)


if __name__ == "__main__":
    test_hemispheric_cognition()
