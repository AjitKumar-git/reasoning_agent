"""
Comprehensive Multi-Domain Cognitive Benchmark for Bipolar Hemispheric Wave Spheroid
Evaluates latest checkpoint (Loss: 0.4854 / PPL: 1.62) trained on 50k rich multi-domain dataset.
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


def run_master_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    ckpt_path = "checkpoints/hemispheric_minimal_best.pt"

    print("=" * 85)
    print("      BIPOLAR HEMISPHERIC WAVE SPHEROID: MASTER MULTI-DOMAIN EVALUATION       ")
    print("=" * 85)
    print(f"Device        : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Precision     : {dtype}")
    print(f"Weights       : {ckpt_path}")
    print("Zero-Feedback : Strict x_{t+1} != [x, y] (Sensory -> 3D Wave -> Motor Output)")
    print("=" * 85)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = build_minimal_hemispheric_spheroid()
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device=device, dtype=dtype)
    model.eval()

    test_domains = [
        {
            "domain": "Deductive Syllogism (Birds)",
            "prompt": "All birds have feathers. A robin is a bird. Does a robin have feathers?",
            "expected_key": "universal categorical deduction, possesses feathers, yes",
        },
        {
            "domain": "Deductive Syllogism (Mammals)",
            "prompt": "All mammals have fur. A cat is a mammal. Does a cat have fur?",
            "expected_key": "universal categorical deduction, possesses fur, yes",
        },
        {
            "domain": "Thermodynamics (Convection)",
            "prompt": "Why does hot air rise while cold air sinks?",
            "expected_key": "heated, thermal energy, molecules move apart, density decreases, buoyant force",
        },
        {
            "domain": "Thermal Conductivity",
            "prompt": "Why does a metal spoon feel colder than a wooden spoon at room temperature?",
            "expected_key": "thermal equilibrium, high thermal conductivity, delocalized electrons, rate of heat loss",
        },
        {
            "domain": "Astronomy (Day / Night)",
            "prompt": "What causes day and night on planet Earth?",
            "expected_key": "rotation of Earth on its axis, 24 hours, facing Sun daylight",
        },
        {
            "domain": "Biology (Photosynthesis)",
            "prompt": "What is photosynthesis and why do green plants need sunlight?",
            "expected_key": "chloroplasts, chlorophyll, sunlight, glucose, oxygen, carbon dioxide",
        },
        {
            "domain": "Transitive Logic (Speed)",
            "prompt": "cheetah is faster than horse. horse is faster than lion. Is cheetah faster or slower than lion?",
            "expected_key": "transitive, cheetah > lion, faster",
        },
        {
            "domain": "Kinematics (Cyclist)",
            "prompt": "A cyclist travels at a steady speed of 12 miles per hour. How far does the cyclist travel in 5 hours?",
            "expected_key": "Distance equals speed multiplied by time, 12 * 5 = 60 miles",
        },
        {
            "domain": "Kinematics (Car)",
            "prompt": "A car travels at a steady speed of 60 miles per hour. How far does the car travel in 3 hours?",
            "expected_key": "Distance equals speed multiplied by time, 60 * 3 = 180 miles",
        },
        {
            "domain": "Geometry (Pool)",
            "prompt": "A rectangular swimming pool is 20 meters long and 10 meters wide. What is its perimeter and area?",
            "expected_key": "Area is L * W = 200 sq meters, Perimeter is 2 * (L + W) = 60 meters",
        },
        {
            "domain": "Multi-Step Subtraction (Bakery)",
            "prompt": "Elena bakes 50 cupcakes at a bakery. Elena sells 20 in the morning and 15 in the afternoon. How many cupcakes are left?",
            "expected_key": "50 total, sold 20 + 15 = 35, remaining 50 - 35 = 15 cupcakes left",
        },
        {
            "domain": "Finance / Discount",
            "prompt": "A watch normally costs $60. A store offers a 25% discount. What is the final sale price?",
            "expected_key": "25% of 60 = 15 discount, 60 - 15 = $45 final price",
        },
    ]

    for idx, item in enumerate(test_domains, 1):
        domain = item["domain"]
        prompt = item["prompt"]
        expected = item["expected_key"]

        print(f"\n[{idx:02d}/{len(test_domains):02d}] DOMAIN: {domain.upper()}")
        print(f"PROMPT   : \"{prompt}\"")
        print(f"KEY TARGET: [{expected}]")
        print("-" * 75)

        q_toks = tokenizer.encode(prompt, return_tensors="pt").to(device)

        for cycles in [1, 4]:
            t0 = time.time()
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
                    out_toks = model.solve_with_active_thinking(
                        q_toks,
                        thinking_cycles=cycles,
                        max_answer_tokens=42,
                        temperature=0.6,
                        repetition_penalty=1.4,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            elapsed_ms = (time.time() - t0) * 1000
            ans = tokenizer.decode(out_toks[0]) if len(out_toks[0]) > 0 else "<empty>"
            clean_ans = ans.strip()
            print(f"  Cycles={cycles:02d} ({elapsed_ms:5.1f} ms) -> {clean_ans}")

    print("\n" + "=" * 85)
    print("Master Multi-Domain Evaluation Complete.")
    print("=" * 85)


if __name__ == "__main__":
    run_master_evaluation()
