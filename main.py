"""
CerebroSpheroid Reasoning Agent: Unified CLI Interface
Usage:
  # 1. Single Prompt Active Deliberation
  python3 main.py infer --prompt "Problem: If 4 books cost $20, how much do 6 books cost? Answer:"

  # 2. Run Comprehensive Reasoning Benchmark Suite
  python3 main.py benchmark

  # 3. Interactive Living Cognitive Session (Core pulses continuously in background)
  python3 main.py living

  # 4. Standard Interactive Session
  python3 main.py interactive

  # 5. Scaled 1-Million Step Training with Plateau Early-Stopping
  python3 main.py train-1m --dataset hybrid --patience 100

  # 6. Quick Laptop Training
  python3 main.py train --dataset gsm8k --steps 300
"""

import os
os.environ["HF_HOME"] = "/home/ajit/Documents/Startup/reasoning_agent/.hf_cache"

import sys
import argparse
from src.inference import ReasoningAgent, run_benchmark


def main():
    parser = argparse.ArgumentParser(description="CerebroSpheroid Non-Autoregressive Reasoning Agent CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: infer
    infer_parser = subparsers.add_parser("infer", help="Run active reasoning on a single prompt")
    infer_parser.add_argument("--prompt", type=str, required=True, help="Input query/problem")
    infer_parser.add_argument("--thinking_cycles", type=int, default=4, help="Internal deliberation cycles")
    infer_parser.add_argument("--max_tokens", type=int, default=None, help="Max answer tokens to emit (default: unbounded until EOS)")
    infer_parser.add_argument("--checkpoint", type=str, default="checkpoints/cerebro_spheroid_100m.pt", help="Checkpoint path")

    # Command: benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run standard reasoning benchmark suite")
    bench_parser.add_argument("--checkpoint", type=str, default="checkpoints/cerebro_spheroid_100m.pt", help="Checkpoint path")

    # Command: interactive
    interactive_parser = subparsers.add_parser("interactive", help="Start interactive thinking session")
    interactive_parser.add_argument("--checkpoint", type=str, default="checkpoints/cerebro_spheroid_100m.pt")

    # Command: living (Living system with continuous background resting-state heartbeat)
    living_parser = subparsers.add_parser("living", help="Start living interactive session with continuous background heartbeat")
    living_parser.add_argument("--checkpoint", type=str, default="checkpoints/cerebro_spheroid_100m.pt")
    living_parser.add_argument("--hz", type=float, default=4.0, help="Heartbeat pacemaker frequency in Hz")

    # Command: train (quick run)
    train_parser = subparsers.add_parser("train", help="Launch local training on preexisting datasets")
    train_parser.add_argument("--dataset", type=str, default="lmsys", choices=["lmsys", "gsm8k", "svamp"], help="Dataset")
    train_parser.add_argument("--steps", type=int, default=300, help="Training steps")
    train_parser.add_argument("--batch_size", type=int, default=1, help="Batch size per micro-step")
    train_parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    train_parser.add_argument("--thinking_cycles", type=int, default=3, help="Subconscious thinking cycles")
    train_parser.add_argument("--lr", type=float, default=0.015, help="Learning rate")

    # Command: train-1m (1-million step scaling engine with plateau detection)
    scale_parser = subparsers.add_parser("train-1m", help="Launch 1-million step scaled training with plateau early-stopping")
    scale_parser.add_argument("--dataset", type=str, default="hybrid", choices=["hybrid", "lmsys", "gsm8k", "svamp"], help="Curriculum")
    scale_parser.add_argument("--max_steps", type=int, default=1_000_000, help="Max gradient steps")
    scale_parser.add_argument("--eval_interval", type=int, default=100, help="Evaluation interval")
    scale_parser.add_argument("--patience", type=int, default=100, help="Patience intervals without improvement")
    scale_parser.add_argument("--batch_size", type=int, default=2, help="Micro-batch size")
    scale_parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation micro-batches")
    scale_parser.add_argument("--max_prompt_len", type=int, default=1024, help="Max prompt tokens")
    scale_parser.add_argument("--max_resp_len", type=int, default=128, help="Max answer tokens")
    scale_parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"], help="Optimizer")
    scale_parser.add_argument("--lr", type=float, default=None, help="Learning rate")

    args = parser.parse_args()

    if args.command == "infer":
        agent = ReasoningAgent(checkpoint_path=args.checkpoint)
        print(f"\n[Problem]: \"{args.prompt}\"")
        ans = agent.generate(args.prompt, thinking_cycles=args.thinking_cycles, max_tokens=args.max_tokens)
        print(f"▶ [Deliberated Answer]: \"{ans}\"")
        print("✓ Zero emitted tokens fed back into model input ($x_{t+1} != [x, y]$).")

    elif args.command == "benchmark":
        agent = ReasoningAgent(checkpoint_path=args.checkpoint)
        run_benchmark(agent)

    elif args.command == "interactive":
        agent = ReasoningAgent(checkpoint_path=args.checkpoint)
        print("=" * 80)
        print("  CerebroSpheroid Interactive Session (Type 'exit' to quit)")
        print("  Zero-Feedback Deliberate-Then-Emit Active Cognitive Engine")
        print("=" * 80)
        while True:
            try:
                user_prompt = input("\nEnter Prompt > ").strip()
                if not user_prompt or user_prompt.lower() in ["exit", "quit"]:
                    break
                ans = agent.generate(user_prompt, thinking_cycles=4, max_tokens=64)
                print(f"Deliberated Response: {ans}")
            except (KeyboardInterrupt, EOFError):
                break
        print("\nSession ended.")

    elif args.command == "living":
        agent = ReasoningAgent(checkpoint_path=args.checkpoint)
        print("=" * 80)
        print("  CEREBRO-SPHEROID: LIVING ORGANISM CONTINUOUS HOMEOTIC ENGINE       ")
        print("=" * 80)
        print(f"Starting continuous biological pacemaker ({args.hz} Hz resting rhythm)...")
        agent.start_living_heartbeat(frequency_hz=args.hz)
        print("✓ Brain core is now actively pulsating in background resting state (DMN).")
        print("  Type any question to perturb the living manifold. Type 'exit' to quit.")
        print("=" * 80)
        while True:
            try:
                user_prompt = input("\nEnter Stimulus > ").strip()
                if not user_prompt or user_prompt.lower() in ["exit", "quit"]:
                    break
                status = agent.get_living_status()
                print(f"  [Living Manifold State: {status['resting_ticks']} resting heartbeats accumulated]")
                ans = agent.generate(user_prompt, thinking_cycles=4, max_tokens=64)
                print(f"▶ [Living Brain Response]: {ans}")
            except (KeyboardInterrupt, EOFError):
                break
        agent.stop_living_heartbeat()
        print("\nLiving pacemaker terminated. Session ended.")

    elif args.command == "train":
        from training.train_laptop import main as train_main
        sys.argv = [
            "train_laptop.py",
            "--dataset", args.dataset,
            "--steps", str(args.steps),
            "--batch_size", str(args.batch_size),
            "--grad_accum", str(args.grad_accum),
            "--thinking_cycles", str(args.thinking_cycles),
            "--lr", str(args.lr),
        ]
        train_main()

    elif args.command == "train-1m":
        from training.train_scale_1m import main as scale_main
        cmd_args = [
            "train_scale_1m.py",
            "--dataset", args.dataset,
            "--max_steps", str(args.max_steps),
            "--eval_interval", str(args.eval_interval),
            "--patience", str(args.patience),
            "--batch_size", str(args.batch_size),
            "--grad_accum", str(args.grad_accum),
            "--max_prompt_len", str(args.max_prompt_len),
            "--max_resp_len", str(args.max_resp_len),
            "--optimizer", args.optimizer,
        ]
        if args.lr is not None:
            cmd_args.extend(["--lr", str(args.lr)])
        sys.argv = cmd_args
        scale_main()

    else:
        agent = ReasoningAgent()
        run_benchmark(agent)


if __name__ == "__main__":
    main()
