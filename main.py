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
  python3 main.py train-1m --dataset specialized --patience 100

  # 6. Quick Laptop Training
  python3 main.py train --dataset gsm8k --steps 300

  # 7. Select the manifold generation explicitly (default: auto-detect from checkpoint)
  python3 main.py infer --prompt "..." --arch v3
"""

import os
os.environ.setdefault(
    "HF_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), ".hf_cache"))
)

import sys
import argparse
from src.inference import ReasoningAgent, run_benchmark

ARCH_HELP = "Manifold generation: auto reads the checkpoint, v2 is the original core, v3 the refined core"


def main():
    parser = argparse.ArgumentParser(description="CerebroSpheroid Non-Autoregressive Reasoning Agent CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: infer
    infer_parser = subparsers.add_parser("infer", help="Run active reasoning on a single prompt")
    infer_parser.add_argument("--prompt", type=str, required=True, help="Input query/problem")
    infer_parser.add_argument("--thinking_cycles", type=int, default=4, help="Internal deliberation cycles")
    infer_parser.add_argument(
        "--max_tokens", type=int, default=64,
        help="Max answer tokens to emit. Pass 0 for unbounded generation (runs to the 4096 "
             "safety ceiling). Default 64 matches the interactive and living sessions; the "
             "shipped checkpoints do not reliably emit EOS on their own, so an unbounded "
             "default prints thousands of tokens of word salad.")
    infer_parser.add_argument("--checkpoint", type=str, default="checkpoints/cerebro_spheroid_1m_best.pt", help="Checkpoint path")
    infer_parser.add_argument("--arch", type=str, default="auto", choices=["auto", "v2", "v3"], help=ARCH_HELP)
    infer_parser.add_argument("--backend", type=str, default="torch", choices=["torch", "jax"], help="Computation backend")

    # Command: benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run standard reasoning benchmark suite")
    bench_parser.add_argument("--checkpoint", type=str, default="checkpoints/cerebro_spheroid_1m_best.pt", help="Checkpoint path")
    bench_parser.add_argument("--arch", type=str, default="auto", choices=["auto", "v2", "v3"], help=ARCH_HELP)
    bench_parser.add_argument("--backend", type=str, default="torch", choices=["torch", "jax"], help="Computation backend")

    # Command: interactive
    interactive_parser = subparsers.add_parser("interactive", help="Start interactive thinking session")
    interactive_parser.add_argument("--checkpoint", type=str, default="checkpoints/cerebro_spheroid_1m_best.pt")
    interactive_parser.add_argument("--arch", type=str, default="auto", choices=["auto", "v2", "v3"], help=ARCH_HELP)
    interactive_parser.add_argument("--backend", type=str, default="torch", choices=["torch", "jax"], help="Computation backend")

    # Command: living (Living system with continuous background resting-state heartbeat)
    living_parser = subparsers.add_parser("living", help="Start living interactive session with continuous background heartbeat")
    living_parser.add_argument("--checkpoint", type=str, default="checkpoints/cerebro_spheroid_1m_best.pt")
    living_parser.add_argument("--hz", type=float, default=4.0, help="Heartbeat pacemaker frequency in Hz")
    living_parser.add_argument("--arch", type=str, default="auto", choices=["auto", "v2", "v3"], help=ARCH_HELP)
    living_parser.add_argument("--backend", type=str, default="torch", choices=["torch", "jax"], help="Computation backend")

    # Command: train (quick run)
    train_parser = subparsers.add_parser("train", help="Launch local training on preexisting datasets")
    train_parser.add_argument("--dataset", type=str, default="lmsys", choices=["lmsys", "gsm8k", "svamp"], help="Dataset")
    train_parser.add_argument("--steps", type=int, default=300, help="Training steps")
    train_parser.add_argument("--batch_size", type=int, default=1, help="Batch size per micro-step")
    train_parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    train_parser.add_argument("--thinking_cycles", type=int, default=3, help="Subconscious thinking cycles")
    train_parser.add_argument("--lr", type=float, default=0.015, help="Learning rate")
    train_parser.add_argument("--arch", type=str, default="v2", choices=["v2", "v3"], help=ARCH_HELP)

    # Command: train-1m (1-million step scaling engine with plateau detection)
    scale_parser = subparsers.add_parser("train-1m", help="Launch 1-million step scaled training with plateau early-stopping")
    scale_parser.add_argument("--dataset", type=str, default="general", choices=["general", "knowledge", "specialized", "hybrid", "lmsys", "gsm8k", "svamp"],
                              help="Curriculum. 'general' trains broad LLM discourse + world knowledge (capitals, geography, science) + reasoning")
    scale_parser.add_argument("--max_steps", type=int, default=1_000_000, help="Max gradient steps")
    scale_parser.add_argument("--eval_interval", type=int, default=100, help="Evaluation interval")
    scale_parser.add_argument("--patience", type=int, default=100, help="Patience intervals without improvement")
    scale_parser.add_argument("--batch_size", type=int, default=2, help="Micro-batch size")
    scale_parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation micro-batches")
    scale_parser.add_argument("--max_prompt_len", type=int, default=1024, help="Max prompt tokens")
    scale_parser.add_argument("--max_resp_len", type=int, default=128, help="Max answer tokens")
    scale_parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"], help="Optimizer")
    scale_parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    scale_parser.add_argument("--thinking_cycles", type=int, default=3, help="Subconscious internal deliberation cycles")
    scale_parser.add_argument("--resume", type=str, default=None, help="Explicit checkpoint to resume from")
    scale_parser.add_argument("--save_prefix", type=str, default="cerebro_spheroid_1m", help="Checkpoint filename prefix")
    scale_parser.add_argument("--select_on", type=str, default="val", choices=["val", "train_loss"], help="Selection signal")
    scale_parser.add_argument("--arch", type=str, default="v2", choices=["v2", "v3"], help=ARCH_HELP)
    scale_parser.add_argument("--seed", type=int, default=1337, help="Seed for RNGs and the frozen eval set")
    scale_parser.add_argument("--eval_dataset", type=str, default=None, help="Held-out eval dataset")
    scale_parser.add_argument("--eval_parquet", type=str, default=None, help="Held-out parquet path")
    scale_parser.add_argument("--eval_batches", type=int, default=128, help="Batches frozen into the held-out eval set")
    scale_parser.add_argument("--save_optimizer", action="store_true",
                              help="Persist optimizer/scheduler state so a resume keeps momentum")

    # Command: train-jax (High-throughput JAX / TPU training)
    train_jax_parser = subparsers.add_parser("train-jax", help="Launch high-performance JAX / XLA training")
    train_jax_parser.add_argument("--dataset", type=str, default="specialized", help="Dataset")
    train_jax_parser.add_argument("--scale", type=str, default="laptop", choices=["laptop", "small", "2b", "260m", "test"], help="Model scale")
    train_jax_parser.add_argument("--max_steps", type=int, default=1000, help="Max gradient steps")
    train_jax_parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    train_jax_parser.add_argument("--thinking_cycles", type=int, default=3, help="Subconscious thinking cycles")
    train_jax_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_jax_parser.add_argument("--arch", type=str, default="v2", choices=["v2", "v3"], help=ARCH_HELP)
    train_jax_parser.add_argument("--seed", type=int, default=1337, help="Seed")
    train_jax_parser.add_argument("--eval_interval", type=int, default=50, help="Evaluation interval")
    train_jax_parser.add_argument("--save_path", type=str, default="checkpoints/cerebro_spheroid_jax.npz", help="Save checkpoint path")
    train_jax_parser.add_argument("--init_from_pt", type=str, default=None, help="Initial weights from .pt checkpoint")

    # Command: verify-jax
    subparsers.add_parser("verify-jax", help="Run analytical JAX vs PyTorch parity and gradient verification")

    args = parser.parse_args()

    if args.command == "infer":
        agent = ReasoningAgent(checkpoint_path=args.checkpoint, arch=args.arch, backend=args.backend)
        print(f"\n[Backend] : {args.backend.upper()}")
        print(f"[Manifold]: {agent.arch}")
        print(f"[Problem]: \"{args.prompt}\"")
        ans = agent.generate(args.prompt, thinking_cycles=args.thinking_cycles,
                             max_tokens=(None if args.max_tokens == 0 else args.max_tokens))
        print(f"▶ [Deliberated Answer]: \"{ans}\"")
        print("✓ Zero emitted tokens fed back into model input ($x_{t+1} != [x, y]$).")

    elif args.command == "benchmark":
        agent = ReasoningAgent(checkpoint_path=args.checkpoint, arch=args.arch, backend=args.backend)
        run_benchmark(agent)

    elif args.command == "interactive":
        agent = ReasoningAgent(checkpoint_path=args.checkpoint, arch=args.arch, backend=args.backend)
        print("=" * 80)
        print("  CerebroSpheroid Interactive Session (Type 'exit' to quit)")
        print(f"  Zero-Feedback Deliberate-Then-Emit Active Cognitive Engine [{args.backend.upper()}]")
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
        agent = ReasoningAgent(checkpoint_path=args.checkpoint, arch=args.arch, backend=args.backend)
        print("=" * 80)
        print(f"  CEREBRO-SPHEROID: LIVING ORGANISM CONTINUOUS HOMEOTIC ENGINE [{args.backend.upper()}]")
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
            "--arch", args.arch,
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
            "--thinking_cycles", str(args.thinking_cycles),
            "--save_prefix", args.save_prefix,
            "--select_on", args.select_on,
            "--arch", args.arch,
            "--seed", str(args.seed),
            "--eval_batches", str(args.eval_batches),
        ]
        if args.eval_dataset is not None:
            cmd_args.extend(["--eval_dataset", args.eval_dataset])
        if args.eval_parquet is not None:
            cmd_args.extend(["--eval_parquet", args.eval_parquet])
        if args.resume is not None:
            cmd_args.extend(["--resume", args.resume])
        if args.save_optimizer:
            cmd_args.append("--save_optimizer")
        if args.lr is not None:
            cmd_args.extend(["--lr", str(args.lr)])
        sys.argv = cmd_args
        scale_main()

    elif args.command == "train-jax":
        from training.train_jax import run_training_jax
        run_training_jax(
            dataset=args.dataset,
            scale=args.scale,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            thinking_cycles=args.thinking_cycles,
            lr=args.lr,
            arch=args.arch,
            seed=args.seed,
            eval_interval=args.eval_interval,
            save_path=args.save_path,
            init_from_pt=args.init_from_pt,
        )

    elif args.command == "verify-jax":
        from training.verify_jax_parity import main as verify_main
        verify_main()

    else:
        agent = ReasoningAgent()
        run_benchmark(agent)


if __name__ == "__main__":
    main()
