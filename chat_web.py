#!/usr/bin/env python3
"""
CerebroSpheroid Interactive Web Chat Application.
Runs a Gradio interface with continuous token-by-token streaming, active thinking cycles,
persistent volumetric state, and live logging to chat_logs/chat_session.md.
"""

import os
os.environ.setdefault("HF_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), ".hf_cache")))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys
import time
import json
from datetime import datetime, timezone
import gradio as gr
import torch

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = os.path.join(ROOT_DIR, "chat_logs")
SESSION_HISTORY_FILE = os.path.join(LOG_DIR, "session_history.json")
SESSION_LOG_MD = os.path.join(LOG_DIR, "chat_session.md")
SESSION_LOG_JSONL = os.path.join(LOG_DIR, "chat_session.jsonl")

DEFAULT_CKPT = os.path.join(ROOT_DIR, "checkpoints", "cerebro_spheroid_distill_v2_best.pt")
if not os.path.exists(DEFAULT_CKPT):
    DEFAULT_CKPT = os.path.join(ROOT_DIR, "checkpoints", "cerebro_spheroid_locked_best.pt")

from src.inference import ReasoningAgent

print("Loading CerebroSpheroid Reasoning Engine...")
agent = ReasoningAgent(checkpoint_path=DEFAULT_CKPT, arch="v2")
print(f"Loaded: {DEFAULT_CKPT} on {agent.device}")


def append_to_logs(entry):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(SESSION_LOG_JSONL, "a") as f:
        f.write(json.dumps(entry) + "\n")

    first_time = not os.path.exists(SESSION_LOG_MD)
    with open(SESSION_LOG_MD, "a") as f:
        if first_time:
            f.write("# CerebroSpheroid Chat Session Log\n\n")
            f.write(f"- Model Checkpoint: `{entry.get('checkpoint')}`\n")
            f.write(f"- Session Started: `{entry.get('timestamp')}`\n\n---\n\n")
        f.write(f"### Turn {entry.get('turn', 1)} — {entry.get('timestamp')}\n\n")
        f.write(f"**User**: {entry.get('user')}\n\n")
        f.write(f"**CerebroSpheroid**: {entry.get('assistant')}\n\n")
        f.write(f"> *Latency: {entry.get('latency_sec', 0.0):.2f}s | Cycles: {entry.get('thinking_cycles')} | Tokens: {entry.get('tokens_emitted')} | Mode: Continuous Stream*\n\n---\n\n")


def toggle_pacemaker(is_enabled):
    if is_enabled:
        agent.start_living_heartbeat(frequency_hz=4.0)
        return "⚡ Continuous Pacemaker: ACTIVE (4.0 Hz resting pulse)"
    else:
        agent.stop_living_heartbeat()
        return "💤 Continuous Pacemaker: QUIESCENT"


custom_css = """
.gradio-container { max-width: 960px !important; margin: auto !important; }
.chat-title { text-align: center; margin-bottom: 5px; }
.chat-subtitle { text-align: center; color: #666; margin-bottom: 20px; }
.pulse-indicator { font-weight: bold; color: #007acc; font-size: 0.9em; margin-top: 5px; }
"""

with gr.Blocks(title="CerebroSpheroid Reasoning Agent") as demo:
    gr.Markdown("# 🧠 CerebroSpheroid Continuous Cognitive Engine", elem_classes=["chat-title"])
    gr.Markdown(
        f"**Active Checkpoint**: `{os.path.basename(DEFAULT_CKPT)}` | "
        f"**Held-out CE**: `0.2863` (PPL: `1.33`) | "
        f"**Readout**: Fluid Continuous Emission Stream",
        elem_classes=["chat-subtitle"]
    )

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="Continuous Deliberative Stream", height=520)
            msg_input = gr.Textbox(
                placeholder="Ask CerebroSpheroid anything (science, geography, arithmetic, logic)...",
                label="Stimulus Input",
                lines=1
            )
            with gr.Row():
                submit_btn = gr.Button("Emit Stimulus", variant="primary")
                clear_btn = gr.Button("Clear Session")

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Cognitive Controls")
            thinking_slider = gr.Slider(
                minimum=1, maximum=8, value=1, step=1,
                label="Thinking Cycles",
                info="Subconscious deliberation loops before motor emission"
            )
            tokens_slider = gr.Slider(
                minimum=16, maximum=128, value=64, step=8,
                label="Max Emitted Tokens"
            )
            temp_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.0, step=0.05,
                label="Sampling Temperature",
                info="0.0 = Greedy argmax determinism"
            )
            living_check = gr.Checkbox(
                value=False,
                label="Living Pacemaker (DMN)",
                info="Continuously pulses manifold state at resting rhythm"
            )
            pulse_status = gr.Markdown("💤 Continuous Pacemaker: QUIESCENT", elem_classes=["pulse-indicator"])
            gr.Markdown(
                f"---\n"
                f"**Live Logs:**\n"
                f"- `chat_logs/chat_session.md`\n"
                f"- `chat_logs/chat_session.jsonl`"
            )

    living_check.change(toggle_pacemaker, [living_check], [pulse_status])

    def stream_respond(user_message, chat_history, cycles, max_toks, temp):
        if not user_message or not user_message.strip():
            yield "", chat_history or []
            return

        full_prompt = user_message.strip()
        history_list = list(chat_history) if chat_history else []
        history_list.append({"role": "user", "content": user_message})
        history_list.append({"role": "assistant", "content": ""})
        yield "", history_list

        t0 = time.time()
        final_text = ""

        # Continuous fluid token stream from efferent motor readout
        for partial_text in agent.generate_stream(
            full_prompt,
            thinking_cycles=int(cycles),
            max_tokens=int(max_toks),
            temperature=float(temp)
        ):
            final_text = partial_text
            history_list[-1]["content"] = partial_text
            yield "", history_list

        latency = time.time() - t0
        cleaned_resp = final_text.strip()
        history_list[-1]["content"] = cleaned_resp

        turn_num = len(history_list) // 2
        log_entry = {
            "turn": turn_num,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "user": user_message,
            "assistant": cleaned_resp,
            "raw_prompt": full_prompt,
            "latency_sec": latency,
            "thinking_cycles": int(cycles),
            "tokens_emitted": len(cleaned_resp.split()),
            "checkpoint": os.path.basename(agent.checkpoint_loaded or "default"),
        }
        append_to_logs(log_entry)
        yield "", history_list

    msg_input.submit(stream_respond, [msg_input, chatbot, thinking_slider, tokens_slider, temp_slider], [msg_input, chatbot])
    submit_btn.click(stream_respond, [msg_input, chatbot, thinking_slider, tokens_slider, temp_slider], [msg_input, chatbot])
    clear_btn.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=custom_css)
