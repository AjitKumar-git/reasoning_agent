"""
Evaluation Benchmark for Continuous Real Geodesic Spheroid (B^3)
================================================================
Evaluates checkpoints/real_sphere_best.pt across multi-domain prompts:
- Formal deductive logic and categorical syllogisms
- Physical mechanisms (thermal equilibrium, convection, conduction)
- Biological pathways (photosynthesis)
- Mathematical reasoning (percentages, rates, geometry)
- Tests varying internal deliberation cycles (N = 1, 3, 5 wave pulses)
- Strict Zero External Token Feedback (x_{t+1} != [x, y]).
"""

import os
import sys
import time
import torch
from transformers import AutoTokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, ".hf_cache"))

from src.real_geodesic_sphere import build_real_geodesic_spheroid


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "real_sphere_best.pt")

    print("=" * 85)
    print("      CONTINUOUS REAL GEODESIC SPHEROID (B^3): EVALUATION BENCHMARK          ")
    print("=" * 85)
    print(f"Device        : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Weights       : {ckpt_path}")
    print(f"Topology      : B^3 Unit Ball, 8 Fibonacci Shells, 1,024 Nodes on S^2")
    print(f"Operator      : Laplace-Beltrami + Cubic Soliton + Noise Reduction Filter")
    print("=" * 85)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = build_real_geodesic_spheroid()
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        return

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device=device, dtype=dtype)
    model.eval()

    test_cases = [
        ("Deductive Syllogism", "All birds have feathers. A robin is a bird. Does a robin have feathers?"),
        ("Transitive Logic", "A rectangle is larger than a square. A square is larger than a triangle. Is a rectangle larger than a triangle?"),
        ("Thermodynamics", "Why does hot air rise while cold air sinks?"),
        ("Thermal Conduction", "Why does a metal spoon feel colder than a wooden spoon at room temperature?"),
        ("Biochemical Kinetics", "What happens to the rate of photosynthesis if carbon dioxide concentration increases?"),
        ("Unit Rate Arithmetic", "If 5 pencils cost $15, how much do 8 pencils cost?"),
        ("Speed & Distance", "A cyclist rides at a speed of 12 miles per hour. How far does the cyclist travel in 5 hours?"),
        ("Percentage Discount", "A store offers a 25% discount on a $60 jacket. What is the final price?"),
    ]

    print("\n--- Testing Wave Deliberation Cycles (N = 1, 3) ---\n")
    for category, prompt in test_cases:
        print(f"[{category}]")
        print(f"Prompt: \"{prompt}\"")
        q_toks = tokenizer.encode(prompt, return_tensors="pt").to(device)

        for cycles in [1, 3]:
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
                    out_toks = model.solve_with_active_thinking(
                        q_toks,
                        thinking_cycles=cycles,
                        max_answer_tokens=32,
                        temperature=0.65,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            ans = tokenizer.decode(out_toks[0]).strip() if len(out_toks[0]) > 0 else "<empty>"
            print(f"  Cycles={cycles}: \"{ans}\"")
        print("-" * 75)


if __name__ == "__main__":
    main()
