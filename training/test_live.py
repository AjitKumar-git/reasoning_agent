"""
Live Cognitive Testing Harness for Bipolar Hemispheric Wave Spheroid
Evaluates fresh, non-trained questions across varying deliberation cycles.
Strict Zero-Feedback Biological Readout (x_{t+1} != [x, y]).
"""

import os
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, ".hf_cache"))

import torch
from transformers import AutoTokenizer
from src.hemispheric_spheroid import build_minimal_hemispheric_spheroid


def run_live_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    ckpt_path = "checkpoints/hemispheric_minimal_best.pt"
    print("=" * 80)
    print("       BIPOLAR HEMISPHERIC WAVE SPHEROID: LIVE COGNITIVE BENCHMARK       ")
    print("=" * 80)
    print(f"Device        : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Precision     : {dtype}")
    print(f"Checkpoint    : {ckpt_path}")
    print("Zero-Feedback : Strict (x_{t+1} != [x, y], Motor -> Deep Core Corollary Discharge)")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = build_minimal_hemispheric_spheroid()
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device=device, dtype=dtype)
    model.eval()

    test_suite = [
        {
            "category": "Arithmetic & Finance",
            "prompt": "If 5 pencils cost $15, how much do 8 pencils cost?",
        },
        {
            "category": "Multi-Step Subtraction & Inventory",
            "prompt": "Liam has 50 marbles. He gives 18 to Alex and 12 to Noah. How many marbles does Liam have left?",
        },
        {
            "category": "Kinematics & Rate Equations",
            "prompt": "A cyclist rides at a speed of 12 miles per hour. How far does the cyclist travel in 5 hours?",
        },
        {
            "category": "Geometry & Dimensional Analysis",
            "prompt": "A rectangular swimming pool is 20 meters long and 10 meters wide. What is its perimeter and area?",
        },
        {
            "category": "Physical Mechanics & Temperature",
            "prompt": "Why does hot air rise while cold air sinks?",
        },
        {
            "category": "Deductive Syllogism",
            "prompt": "All birds have feathers. A robin is a bird. Does a robin have feathers?",
        },
    ]

    for idx, item in enumerate(test_suite, 1):
        category = item["category"]
        prompt = item["prompt"]
        print(f"\n[{idx}/{len(test_suite)}] CATEGORY: {category}")
        print(f"PROMPT: \"{prompt}\"")
        print("-" * 75)

        q_toks = tokenizer.encode(prompt, return_tensors="pt").to(device)

        for cycles in [1, 3, 6]:
            t0 = time.time()
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
                    out_toks = model.solve_with_active_thinking(
                        q_toks,
                        thinking_cycles=cycles,
                        max_answer_tokens=35,
                        temperature=0.65,
                        repetition_penalty=1.4,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            elapsed_ms = (time.time() - t0) * 1000
            ans = tokenizer.decode(out_toks[0]) if len(out_toks[0]) > 0 else "<empty>"
            clean_ans = ans.strip()
            print(f"  Deliberation Depth {cycles:02d} ({elapsed_ms:5.1f} ms):")
            print(f"    --> {clean_ans}\n")

    print("=" * 80)
    print("Benchmark complete.")
    print("=" * 80)


if __name__ == "__main__":
    run_live_test()
