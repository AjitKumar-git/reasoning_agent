"""
Test Bipolar Hemispheric Wave Spheroid on completely out-of-distribution (non-trained) questions.
"""

import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, ".hf_cache"))

import torch
from transformers import AutoTokenizer
from src.hemispheric_spheroid import build_minimal_hemispheric_spheroid


def evaluate_unseen_questions(checkpoint_path="checkpoints/hemispheric_minimal_best.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    print(f"Loading checkpoint {checkpoint_path} onto {device}...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = build_minimal_hemispheric_spheroid()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device=device, dtype=dtype)
    model.eval()

    unseen_queries = [
        # Domain 1: Unseen Word Math / Operations
        {"domain": "Math / Baker", "q": "A baker bakes 48 cupcakes. He sells 30 in the morning and 12 in the afternoon. How many cupcakes are left?"},
        {"domain": "Math / Speed", "q": "A train travels at 75 miles per hour for 4 hours. What is the total distance covered?"},
        {"domain": "Geometry", "q": "A garden is 15 meters long and 8 meters wide. What is its total perimeter?"},

        # Domain 2: Unseen Science / Natural Phenomena
        {"domain": "Science / Biology", "q": "What is photosynthesis and why do green plants need sunlight?"},
        {"domain": "Science / Heat", "q": "Why does a metal spoon feel colder than a wooden spoon at room temperature?"},
        {"domain": "Science / Astronomy", "q": "What causes day and night on planet Earth?"},

        # Domain 3: Unseen Logic / Common Sense
        {"domain": "Logic / Deduction", "q": "If all cats are mammals and all mammals have hearts, do cats have hearts?"},
        {"domain": "Logic / Calendar", "q": "Yesterday was Tuesday. What day will tomorrow be?"},
    ]

    print("=" * 85)
    print("      OUT-OF-DISTRIBUTION (NON-TRAINED) COGNITIVE BENCHMARK           ")
    print("=" * 85)

    results = []

    for item in unseen_queries:
        domain = item["domain"]
        q = item["q"]
        print(f"\n[{domain.upper()}] Q: {q}")
        q_toks = tokenizer.encode(q, return_tensors="pt").to(device)

        for cycles in [2, 6]:
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
                    out_toks = model.solve_with_active_thinking(
                        q_toks,
                        thinking_cycles=cycles,
                        max_answer_tokens=32,
                        temperature=0.7,
                        repetition_penalty=1.5,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            ans = tokenizer.decode(out_toks[0]) if len(out_toks[0]) > 0 else "<empty>"
            print(f"  Deliberation Cycles={cycles:02d} -> {ans.strip()}")
            results.append({"domain": domain, "q": q, "cycles": cycles, "ans": ans.strip()})

    print("\n" + "=" * 85)
    return results


if __name__ == "__main__":
    evaluate_unseen_questions()
