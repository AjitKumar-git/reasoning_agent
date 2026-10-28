#!/usr/bin/env python3
"""
CerebroSpheroid Interactive Chat Session Manager.
Maintains persistent conversation history and writes timestamped logs to chat_logs/.
"""

import os
os.environ.setdefault("HF_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), ".hf_cache")))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys
import json
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = os.path.join(ROOT_DIR, "chat_logs")
SESSION_HISTORY_FILE = os.path.join(LOG_DIR, "session_history.json")
SESSION_LOG_MD = os.path.join(LOG_DIR, "chat_session.md")
SESSION_LOG_JSONL = os.path.join(LOG_DIR, "chat_session.jsonl")

DEFAULT_CKPT = None
for cand in [
    os.path.join(ROOT_DIR, "checkpoints", "cerebro_spheroid_v3_kaggle_best.pt"),
    os.path.join(ROOT_DIR, "checkpoints", "cerebro_spheroid_v3_colab_best.pt"),
    os.path.join(ROOT_DIR, "checkpoints", "cerebro_spheroid_v3full_best.pt"),
    os.path.join(ROOT_DIR, "checkpoints", "cerebro_spheroid_distill_v2_best.pt"),
]:
    if os.path.exists(cand):
        DEFAULT_CKPT = cand
        break
if not DEFAULT_CKPT:
    DEFAULT_CKPT = os.path.join(ROOT_DIR, "checkpoints", "cerebro_spheroid_v3_colab_best.pt")

from src.inference import ReasoningAgent


def load_history():
    if os.path.exists(SESSION_HISTORY_FILE):
        try:
            with open(SESSION_HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(history):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(SESSION_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def append_to_logs(entry):
    os.makedirs(LOG_DIR, exist_ok=True)
    # 1. JSONL log
    with open(SESSION_LOG_JSONL, "a") as f:
        f.write(json.dumps(entry) + "\n")

    # 2. Markdown log
    first_time = not os.path.exists(SESSION_LOG_MD)
    with open(SESSION_LOG_MD, "a") as f:
        if first_time:
            f.write("# CerebroSpheroid Chat Session Log\n\n")
            f.write(f"- Model Checkpoint: `{entry.get('checkpoint')}`\n")
            f.write(f"- Session Started: `{entry.get('timestamp')}`\n\n---\n\n")
        f.write(f"### Turn {entry.get('turn', 1)} — {entry.get('timestamp')}\n\n")
        f.write(f"**User**: {entry.get('user')}\n\n")
        f.write(f"**CerebroSpheroid**: {entry.get('assistant')}\n\n")
        f.write(f"> *Latency: {entry.get('latency_sec', 0.0):.2f}s | Cycles: {entry.get('thinking_cycles')} | Tokens: {entry.get('tokens_emitted')}*\n\n---\n\n")


def build_full_prompt(history, new_message):
    # Direct query prompt matches the model's prompt-locking training format
    return new_message.strip()


def chat_turn(agent, message, max_tokens=64, thinking_cycles=1, temperature=0.0, stream=False):
    history = load_history()
    prompt = build_full_prompt(history, message)

    t0 = time.time()
    if stream:
        final_text = ""
        prev_len = 0
        for partial in agent.generate_stream(
            prompt,
            thinking_cycles=thinking_cycles,
            max_tokens=max_tokens,
            temperature=temperature
        ):
            new_chunk = partial[prev_len:]
            sys.stdout.write(new_chunk)
            sys.stdout.flush()
            prev_len = len(partial)
            final_text = partial
        response = final_text
    else:
        response = agent.generate(
            prompt,
            thinking_cycles=thinking_cycles,
            max_tokens=max_tokens,
            temperature=temperature
        )
    latency = time.time() - t0

    # Clean up artifacts if any
    cleaned_resp = response.strip()

    turn_number = len(history) + 1
    log_entry = {
        "turn": turn_number,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "user": message,
        "assistant": cleaned_resp,
        "raw_prompt": prompt,
        "latency_sec": latency,
        "thinking_cycles": thinking_cycles,
        "tokens_emitted": len(cleaned_resp.split()),
        "checkpoint": os.path.basename(agent.checkpoint_loaded or "default"),
    }

    history.append({"user": message, "assistant": cleaned_resp})
    save_history(history)
    append_to_logs(log_entry)

    return cleaned_resp, log_entry


def main():
    parser = argparse.ArgumentParser(description="Chat with CerebroSpheroid")
    parser.add_argument("--message", type=str, default=None, help="Send a single message turn")
    parser.add_argument("--reset", action="store_true", help="Reset chat history")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT, help="Path to checkpoint")
    parser.add_argument("--arch", default=None, choices=["v2", "v3"], help="Architecture version (auto-detected if omitted)")
    parser.add_argument("--cycles", type=int, default=1, help="Thinking cycles (default: 1)")
    parser.add_argument("--max_tokens", type=int, default=64, help="Max tokens (default: 64)")
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature")
    args = parser.parse_args()

    if args.reset:
        if os.path.exists(SESSION_HISTORY_FILE):
            os.remove(SESSION_HISTORY_FILE)
        print("Chat history reset.")
        if not args.message:
            return

    arch = args.arch
    if not arch:
        arch = "v3" if "v3" in os.path.basename(args.checkpoint or "") else "v2"

    agent = ReasoningAgent(checkpoint_path=args.checkpoint, arch=arch)

    if args.message:
        print(f"User: {args.message}")
        print("Assistant: ", end="", flush=True)
        ans, entry = chat_turn(agent, args.message, max_tokens=args.max_tokens, thinking_cycles=args.cycles, temperature=args.temp, stream=True)
        print(f"\n\n[Logged to {SESSION_LOG_MD}]")
    else:
        print("=" * 80)
        print(f"  CerebroSpheroid Continuous Stream Session (Checkpoint: {os.path.basename(args.checkpoint)})")
        print(f"  Logging to: {SESSION_LOG_MD}")
        print("  Type 'exit' or 'quit' to end session.")
        print("=" * 80)
        while True:
            try:
                msg = input("\nYou > ").strip()
                if not msg or msg.lower() in ["exit", "quit"]:
                    break
                print("Assistant > ", end="", flush=True)
                ans, _ = chat_turn(agent, msg, max_tokens=args.max_tokens, thinking_cycles=args.cycles, temperature=args.temp, stream=True)
                print()
            except (KeyboardInterrupt, EOFError):
                break
        print("\nSession ended.")

    agent.stop_living_heartbeat()


if __name__ == "__main__":
    main()
