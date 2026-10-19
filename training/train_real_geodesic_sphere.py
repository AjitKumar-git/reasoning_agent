"""
Training Pipeline for Continuous Real Geodesic Spheroid (B^3)
============================================================
Physical 3D Ball & Spherical Geometry:
1. 8 Concentric Fibonacci Geodesic Shells (1,024 true 3D spatial points on B^3).
2. Continuous Laplace-Beltrami Diffusion on Curved Manifold S^2.
3. Active Noise-Reduction Autoregressor (damping thermal wave dispersion).
4. Dual-end Wave Ingestion/Readout:
   - Hemisphere A (South Pole, z < -0.05): Sensory Sheet
   - Hemisphere B (North Pole, z > 0.05): Motor Sheet
   - Corollary Discharge into Subcortical Core (r < 0.25)
5. Zero External Token Feedback (x_{t+1} != [x, y]).
6. Trained on 50,000 Multi-Domain Cognitive Samples at Maximum Stable VRAM (~2.3 GB).
"""

import os
import sys
import time
import math
import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Set HF cache dir
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.path.join(repo_root, ".hf_cache")

from src.real_geodesic_sphere import build_real_geodesic_spheroid
from src.data_loaders import get_specialized_cognitive_stream


def main():
    parser = argparse.ArgumentParser(description="Train Continuous Real Geodesic Spheroid (B^3)")
    parser.add_argument("--steps", type=int, default=500, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=12, help="Micro-batch size (12 targets ~2.3 GB VRAM)")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate (AdamW)")
    parser.add_argument("--thinking_cycles", type=int, default=2, help="Internal wave deliberation steps")
    parser.add_argument("--eval_interval", type=int, default=50, help="Evaluation and probe interval")
    parser.add_argument("--save_prefix", type=str, default="real_sphere", help="Checkpoint prefix")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # VRAM allocation configuration
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    # Instantiate Real Geodesic Spheroid (B^3)
    model = build_real_geodesic_spheroid()
    best_ckpt_path = os.path.join(repo_root, "checkpoints", f"{args.save_prefix}_best.pt")
    latest_ckpt_path = os.path.join(repo_root, "checkpoints", f"{args.save_prefix}_latest.pt")
    os.makedirs(os.path.join(repo_root, "checkpoints"), exist_ok=True)

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming weights from {args.resume}...")
        model.load_state_dict(torch.load(args.resume, map_location="cpu"))
    elif os.path.exists(best_ckpt_path) and not args.resume:
        print(f"Auto-resuming weights from {best_ckpt_path}...")
        try:
            model.load_state_dict(torch.load(best_ckpt_path, map_location="cpu"))
        except Exception as e:
            print(f"Could not load {best_ckpt_path} ({e}), starting fresh.")

    model.to(device=device, dtype=dtype)
    total_p = sum(p.numel() for p in model.parameters())

    print("=" * 82)
    print("      CONTINUOUS REAL GEODESIC SPHEROID (B^3): NEURAL WAVE TRAINING        ")
    print("=" * 82)
    print(f"Model Parameters   : {total_p:,} ({total_p/1e6:.2f}M params)")
    print(f"Target Accelerator : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Spatial Topology   : B^3 Unit Ball, 8 Fibonacci Shells, 1,024 Nodes on S^2")
    print(f"Laplace-Beltrami   : Continuous Tangential Diffusion + Radial Conduction + Cubic Soliton")
    print(f"Wave Regulation    : Active Noise-Reduction Autoregressor")
    print(f"Training Batch     : Micro {args.batch_size} × Accum {args.grad_accum} = Effective {args.batch_size * args.grad_accum}")
    print(f"Total Steps        : {args.steps} | Deliberation Cycles: {args.thinking_cycles}")
    print(f"Optimizer          : ADAMW (lr={args.lr:.2e}, weight_decay=1e-2)")
    print("=" * 82)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.05)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 50,000 Sample Diverse Multi-Domain Cognitive Stream
    data_stream = get_specialized_cognitive_stream(
        tokenizer, batch_size=args.batch_size, max_prompt_len=256, max_resp_len=48
    )

    test_queries = [
        "If 5 pencils cost $15, how much do 8 pencils cost?",
        "A cyclist rides at a speed of 12 miles per hour. How far does the cyclist travel in 5 hours?",
        "A rectangular swimming pool is 20 meters long and 10 meters wide. What is its perimeter and area?",
        "Why does hot air rise while cold air sinks?",
        "All birds have feathers. A robin is a bird. Does a robin have feathers?",
        "Why does a metal spoon feel colder than a wooden spoon at room temperature?",
        "What happens to the rate of photosynthesis if carbon dioxide concentration increases?",
        "A store offers a 25% discount on a $60 jacket. What is the final price?",
    ]

    t_start = time.time()
    best_loss = 999.0
    step = 0

    while step < args.steps:
        step += 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(args.grad_accum):
            prompts, targets = next(data_stream)
            prompts = prompts.to(device)
            targets = targets.to(device)

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
                # Forward through Real Geodesic Sphere
                predicted_logits = model(prompts, targets, thinking_cycles=args.thinking_cycles)
                loss = criterion(predicted_logits.view(-1, vocab_size), targets.view(-1))
                scaled_loss = loss / args.grad_accum

            scaled_loss.backward()
            accum_loss += scaled_loss.item()
            del prompts, targets, predicted_logits, loss, scaled_loss

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if device.type == "cuda" and step % 2 == 0:
            torch.cuda.empty_cache()

        loss_val = accum_loss

        # Step Progress Logging
        if step % 10 == 0 or step == 1:
            ppl = math.exp(min(loss_val, 15))
            vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            cur_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - t_start
            print(f"Step [{step:4d}/{args.steps}] | Loss: {loss_val:.4f} | PPL: {ppl:8.2f} | LR: {cur_lr:.2e} | VRAM: {vram:.0f} MB | Elapsed: {elapsed:.1f}s")

        # Evaluation & Checkpoint Interval
        if step % args.eval_interval == 0:
            if loss_val < best_loss:
                best_loss = loss_val
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"  ★ [New Best Checkpoint] Saved to {best_ckpt_path} (Loss: {best_loss:.4f})")

            # Cognitive Probe Inspection
            model.eval()
            print("\n" + "·" * 70)
            print(f"  [Cognitive Inspection at Step {step} | Loss: {loss_val:.4f}]")
            for q in test_queries:
                q_toks = tokenizer.encode(q, return_tensors="pt").to(device)
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=dtype):
                        out_toks = model.solve_with_active_thinking(
                            q_toks,
                            thinking_cycles=args.thinking_cycles,
                            max_answer_tokens=24,
                            temperature=0.7,
                            eos_token_id=tokenizer.eos_token_id
                        )
                resp = tokenizer.decode(out_toks[0]) if len(out_toks[0]) > 0 else "<empty>"
                print(f"  Q: \"{q[:50]}...\"")
                print(f"  A: \"{resp.strip()}\"")
            print("·" * 70 + "\n")

    torch.save(model.state_dict(), latest_ckpt_path)
    total_time = time.time() - t_start
    print("=" * 82)
    print(f"Training Concluded in {total_time/60:.2f} minutes.")
    print(f"Best Weights  : {best_ckpt_path} (Loss: {best_loss:.4f})")
    print(f"Latest Weights: {latest_ckpt_path}")
    print("=" * 82)


if __name__ == "__main__":
    main()
