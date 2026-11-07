"""
Comprehensive Testing Script for CerebroSpheroid V4 Latest/Best Checkpoints.
Probes:
1. Manifold Geometry & Shell Energy Distribution
2. Reasoning & Cognitive QA across multiple categories (Math, Code, Factual, Commonsense)
3. Internal Deliberation Impact (0 vs 3 vs 6 thinking cycles)
"""

import os
import sys
import json
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.corpus import build_tokenizer, format_stimulus_prompt
from src.spheroid_v4 import build_spheroid_v4, compute_manifold_diagnostics

TEST_PROMPTS = [
    # 1. Identity & Alignment
    ("Identity", "Hello! Who are you?"),
    ("Identity", "Hi there"),
    
    # 2. Arithmetic & Word Problems
    ("Math", "What is 2 + 3?"),
    ("Math", "If you have 3 apples and eat 1, how many are left?"),
    ("Math", "What is 15 * 4?"),
    ("Math", "Solve for x: 2x - 6 = 10"),
    
    # 3. Science & Explanations
    ("Science", "Why is the sky blue?"),
    ("Science", "Explain photosynthesis in simple terms."),
    ("Science", "What is the capital of France?"),
    
    # 4. Code & Structured Logic
    ("Code", "Write a Python function to check if a number is prime:"),
    ("Logic", "All humans are mortal. Socrates is human. Therefore:"),
]

def test_checkpoint(ckpt_path: str, device: torch.device):
    print("=" * 80)
    print(f"🔬 EVALUATING CEREBRO-SPHEROID V4 CHECKPOINT: {ckpt_path}")
    print("=" * 80)
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint file does not exist: {ckpt_path}")
        return

    file_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"📁 Checkpoint Size: {file_size_mb:.2f} MB")

    tok = build_tokenizer("llama")
    print(f"🔤 Tokenizer: Modern Llama BPE (Vocab Size: {len(tok):,})")

    model = build_spheroid_v4("v4_small", vocab_size=len(tok))
    meta = model.load_compatible_checkpoint(ckpt_path, device=device)
    model.to(device=device, dtype=torch.float32)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model Scale: {total_params:,} parameters")
    if meta:
        print(f"📊 Checkpoint Metadata: {json.dumps(meta, indent=2)}")

    # 1. Manifold Diagnostics
    print("\n" + "-" * 50)
    print("🌐 1. Manifold Geometry & Shell Spectral Diagnostics")
    print("-" * 50)
    with torch.no_grad():
        sample_ids = tok.encode(format_stimulus_prompt("Hello! Who are you?"), add_special_tokens=False)
        tok_tensor = torch.tensor([sample_ids], device=device)
        sensory_mem, _ = model.encode_sensory(tok_tensor)
        init_s = model.volumetric_core.init_state(1, device)
        init_s[:, -1, :] = sensory_mem.mean(dim=1)
        state = model.deliberate(init_s, thinking_cycles=3)
        diag = compute_manifold_diagnostics(model, state)
        for k, v in diag.items():
            if isinstance(v, float):
                print(f"  • {k:<25}: {v:.4f}")
            elif isinstance(v, list) and len(v) <= 16:
                formatted_v = [f"{x:.3f}" for x in v]
                print(f"  • {k:<25}: {formatted_v}")
            else:
                print(f"  • {k:<25}: {v}")

    # 2. Generative Quality Across Diverse Prompts
    print("\n" + "-" * 50)
    print("💬 2. Cognitive & Reasoning Generation Probing (3 Deliberation Cycles)")
    print("-" * 50)
    for category, prompt in TEST_PROMPTS:
        with torch.no_grad():
            ans = model.generate_text(
                prompt,
                tokenizer=tok,
                thinking_cycles=3,
                max_new_tokens=48,
                temperature=0.6,
                repetition_penalty=1.35,
                device=device,
                strip_bio_prefix=True,
            )
        print(f"[{category:<8}] Prompt: {prompt}")
        print(f"           Answer: {ans.strip()}")
        print("-" * 50)

    # 3. Deliberation Impact Analysis (0 vs 3 vs 6 Cycles)
    print("\n" + "-" * 50)
    print("🧠 3. Internal Deliberation Impact on Reasoning Quality")
    print("-" * 50)
    probe_prompt = "If you have 3 apples and eat 1, how many are left?"
    for cycles in [0, 1, 3, 6]:
        with torch.no_grad():
            ans = model.generate_text(
                probe_prompt,
                tokenizer=tok,
                thinking_cycles=cycles,
                max_new_tokens=40,
                temperature=0.5,
                repetition_penalty=1.35,
                device=device,
                strip_bio_prefix=True,
            )
        print(f"  • Cycles = {cycles}: {ans.strip()}")
    print("=" * 80)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Compute Device: {device}")
    
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/cerebro_spheroid_v4_step_72482_latest.pt"
    test_checkpoint(ckpt, device)
