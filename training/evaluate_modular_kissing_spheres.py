"""
Comparative Evaluation Benchmark for Modular Adjacent Kissing-Sphere Network
=============================================================================
Evaluates checkpoints/modular_kissing_best.pt:
1. Multi-sphere boundary wave transmission across tangent contact disks:
   S_sensory (0, 0, -2) -> S_cognitive (0, 0, 0) -> S_motor (0, 0, +2)
2. Compares Deliberation Cycles (N = 2, 4, 8)
3. Evaluates with vs without Retrograde Standing-Wave Interference
"""

import os
import sys
import torch
from transformers import AutoTokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, ".hf_cache"))

from src.modular_kissing_spheres import build_modular_kissing_lattice


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "modular_kissing_best.pt")

    print("=" * 86)
    print("      MODULAR ADJACENT KISSING-SPHERE NETWORK (B^3): EVALUATION BENCHMARK     ")
    print("=" * 86)
    print(f"Device        : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Weights       : {ckpt_path}")
    print(f"Geometry      : 4 Kissing Unit Balls (B^3), Tangent Contact Aperture 32°")
    print(f"Contact Disks : Sensory <-> Cognitive <-> Motor + Auxiliary")
    print("=" * 86)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = build_modular_kissing_lattice()
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} not found.")
        return

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device=device, dtype=dtype)
    model.eval()

    test_queries = [
        ("Deductive Syllogism", "All birds have feathers. A robin is a bird. Does a robin have feathers?"),
        ("Transitive Logic", "A rectangle is larger than a square. A square is larger than a triangle. Is a rectangle larger than a triangle?"),
        ("Unit Rate Arithmetic", "If 5 pencils cost $15, how much do 8 pencils cost?"),
        ("Speed & Distance", "A cyclist rides at a speed of 12 miles per hour. How far does the cyclist travel in 5 hours?"),
        ("Percentage Discount", "A store offers a 25% discount on a $60 jacket. What is the final price?"),
    ]

    print("\n--- Comparative Thinking Cycles & Retrograde Wave Analysis ---\n")
    for category, prompt in test_queries:
        print(f"[{category}]")
        print(f"Prompt: \"{prompt}\"")
        q_toks = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # 1. Deliberation Cycles N=2 without Retrograde
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
                out_n2 = model.solve_with_active_thinking(
                    q_toks, thinking_cycles=2, max_answer_tokens=24, temperature=0.65, enable_retrograde=False
                )
        ans_n2 = tokenizer.decode(out_n2[0]).strip() if len(out_n2[0]) > 0 else "<empty>"
        print(f"  Cycles N=2 (Forward Only)       : \"{ans_n2}\"")

        # 2. Deliberation Cycles N=4 with Retrograde Interference
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
                out_n4 = model.solve_with_active_thinking(
                    q_toks, thinking_cycles=4, max_answer_tokens=24, temperature=0.65, enable_retrograde=True
                )
        ans_n4 = tokenizer.decode(out_n4[0]).strip() if len(out_n4[0]) > 0 else "<empty>"
        print(f"  Cycles N=4 (Retrograde Fringes) : \"{ans_n4}\"")

        # 3. Deliberation Cycles N=8 Deep Resonance
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
                out_n8 = model.solve_with_active_thinking(
                    q_toks, thinking_cycles=8, max_answer_tokens=24, temperature=0.65, enable_retrograde=True
                )
        ans_n8 = tokenizer.decode(out_n8[0]).strip() if len(out_n8[0]) > 0 else "<empty>"
        print(f"  Cycles N=8 (Deep Standing Wave) : \"{ans_n8}\"")
        print("-" * 75)


if __name__ == "__main__":
    main()
