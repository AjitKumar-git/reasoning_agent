"""
Reasoning Agent: Non-Autoregressive Cognitive Brain Interface
Ingests complex multi-step reasoning problems, performs subconscious deliberation
across a 3D concentric manifold (B^3), and emits answers with ZERO token feedback.
"""

import os
os.environ["HF_HOME"] = "/home/ajit/Documents/Startup/reasoning_agent/.hf_cache"

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from cerebro_spheroid import CerebroSpheroid


def main():
    parser = argparse.ArgumentParser(description="Run CerebroSpheroid Reasoning Agent")
    parser.add_argument("--checkpoint", type=str, default="cerebro_spheroid_100m.pt", help="Checkpoint file")
    parser.add_argument("--thinking_cycles", type=int, default=4, help="Internal subconscious thinking cycles")
    parser.add_argument("--max_tokens", type=int, default=24, help="Maximum answer tokens")
    parser.add_argument("--prompt", type=str, default=None, help="Direct prompt to evaluate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("=" * 80)
    print("      CEREBRO-SPHEROID: REASONING AGENT (DELIBERATE-THEN-EMIT)          ")
    print("=" * 80)
    print(f"Device          : {device}")
    print(f"Thinking Latency: {args.thinking_cycles} cycles (Subconscious deliberation)")
    print(f"Feedback Mode   : Strictly ZERO feedback (x_{{t+1}} != [x_{{1..t}}, y_t])")
    print("=" * 80)

    model = CerebroSpheroid(
        vocab_size=tokenizer.vocab_size,
        surface_dim=640,
        num_shells=16,
        radial_dim=256,
        num_shortcut_tracts=40,
    )

    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        alt = "cerebro_spheroid_lmsys_100m.pt"
        if os.path.exists(alt): ckpt_path = alt

    if os.path.exists(ckpt_path):
        print(f"Loading weights from {ckpt_path}...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print("✓ Model weights loaded successfully.")
    else:
        print(f"[Notice] Checkpoint {ckpt_path} not found. Running with random initialization.")

    model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    default_queries = [
        "Problem: Maya has 8 markers. She buys 4 more and gives 3 away. How many markers does she have? Answer:",
        "Problem: A train travels 60 miles in 1 hour. How many miles does it travel in 3 hours? Answer:",
        "Problem: If 1 book costs $5, how much do 4 books cost? Answer:",
        "Problem: There are 20 apples in a box. You eat 6 apples. How many apples remain? Answer:",
    ]

    queries = [args.prompt] if args.prompt else default_queries

    with torch.no_grad():
        for q in queries:
            print(f"\n[Problem Input]: \"{q}\"")
            pt = torch.tensor([tokenizer.encode(q)], device=device)

            # 1. Afferent Sensory Ingestion
            state = model.ingest_stimulus(pt)

            # 2. Subconscious Deliberation (Zero Emission)
            state = model.deliberate(state, thinking_cycles=args.thinking_cycles)

            # 3. Efferent Motor Readout (Zero Feedback)
            generated = []
            for _ in range(args.max_tokens):
                logits, state = model.emit_next_token(state)
                for prev in generated:
                    logits[0, prev] -= 2.0
                next_tok = torch.argmax(logits, dim=-1, keepdim=True)
                generated.append(next_tok.item())

            ans_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            print(f"  ▶ [Reasoning Readout]: \"{ans_text}\"")
            print("  ✓ Zero emitted tokens fed back into model inputs.")

    print("\n" + "=" * 80)
    print("Reasoning Evaluation Complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
