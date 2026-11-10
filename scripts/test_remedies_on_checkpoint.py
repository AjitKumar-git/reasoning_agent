#!/usr/bin/env python3
"""
Diagnostic & Remedy Lab:
Systematically tests the 4 architectural & decoding remedies on the 10B-token checkpoint:
  1. Baseline (Current default: refractory on, rep_penalty=1.25, temp=0.7, reafference on)
  2. Remedy A: Disable refractory penalty during rollout (stops number vomiting)
  3. Remedy B: Eliminate noisy re-afference drift (stops positive feedback runaway)
  4. Remedy C: Greedy / low-temperature decoding (temp=0.1)
  5. Remedy D: Clean token biological delimiters (replaces broken string '<stim>')
"""

import os
import sys
import torch
import torch.nn.functional as F

os.environ["HF_HOME"] = os.path.expanduser("~/.cache/hf_home")
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/hf_home")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.corpus import build_tokenizer
from src.spheroid_v4 import build_spheroid_v4


def run_remedy_tests(ckpt_path: str = "checkpoints/cerebro_spheroid_v4_step149200_best.pt", device: str = "cuda:0"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"🧪 CEREBRO-SPHEROID V4: REMEDY VALIDATION ON 10B-TOKEN CHECKPOINT")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Device    : {device}")
    print("=" * 80)

    tok = build_tokenizer("llama")
    model = build_spheroid_v4("v4_small", vocab_size=len(tok))
    model.load_compatible_checkpoint(ckpt_path, device=device)
    model.to(device=device, dtype=torch.float32)
    model.eval()

    prompts = [
        "What is the capital of France?",
        "Why is the sky blue?",
        "What is 2 + 3?",
    ]

    STIM_ID = 20436  # True stimulus token

    configs = [
        ("1. BASELINE (Default: Refractory ON, RepPenalty=1.25, Temp=0.7, Re-Afference ON)", {
            "use_refractory": True,
            "rep_penalty": 1.25,
            "temp": 0.7,
            "reafference_scale": 1.0,
            "clean_stim": False,
        }),
        ("2. REMEDY A (Turn OFF Refractory Penalty on Function Words)", {
            "use_refractory": False,
            "rep_penalty": 1.05,
            "temp": 0.7,
            "reafference_scale": 1.0,
            "clean_stim": False,
        }),
        ("3. REMEDY B (Disable Noisy Re-Afference Loop + Low Temp 0.2)", {
            "use_refractory": False,
            "rep_penalty": 1.05,
            "temp": 0.2,
            "reafference_scale": 0.0,
            "clean_stim": False,
        }),
        ("4. REMEDY C (Full Remedy: No Refractory + Clean [20436] Token + Greedy Temp 0.1)", {
            "use_refractory": False,
            "rep_penalty": 1.02,
            "temp": 0.1,
            "reafference_scale": 0.0,
            "clean_stim": True,
        }),
    ]

    for config_name, cfg in configs:
        print("\n" + "=" * 80)
        print(f"⚙️  Testing Configuration: {config_name}")
        print("=" * 80)

        for p in prompts:
            # Prepare input
            if cfg["clean_stim"]:
                p_ids = [STIM_ID] + tok.encode(p, add_special_tokens=False)
            else:
                from src.corpus import format_stimulus_prompt
                p_ids = tok.encode(format_stimulus_prompt(p), add_special_tokens=False)

            inp = torch.tensor([p_ids], device=device, dtype=torch.long)

            # Custom rollout respecting the remedy flags
            with torch.no_grad():
                B, L = inp.shape
                sensory_mem, sensory_kv = model.encode_sensory(inp)
                prompt_flux = sensory_mem.mean(dim=1)

                state = model.volumetric_core.init_state(B, inp.device)
                state[:, -1, :] = prompt_flux
                state = model.deliberate(state, thinking_cycles=3)

                thalamic_end = max(1, model.num_shells // 3)
                thalamic_anchor = state[:, :thalamic_end, :].clone()
                delib_summary = torch.cat([state[:, 0, :], prompt_flux], dim=-1)
                plan = model.motor_dnn.init_plan(delib_summary, persona_id=0)

                refractory_trace = torch.zeros(B, model.vocab_size, device=inp.device, dtype=torch.float32)
                prev_thought = torch.zeros(B, model.surface_dim, device=inp.device, dtype=sensory_mem.dtype)

                emitted = []
                for step_idx in range(40):
                    aud_feed = prev_thought * cfg["reafference_scale"]
                    active_ref_trace = refractory_trace if cfg["use_refractory"] else None

                    logits, efference, next_trace, plan = model.motor_dnn(
                        state=state,
                        sensory_kv=sensory_kv,
                        plan=plan,
                        refractory_trace=active_ref_trace,
                        step_idx=step_idx,
                        auditory_feedback=aud_feed,
                        repetition_penalty=cfg["rep_penalty"],
                    )

                    t = cfg["temp"]
                    if t <= 0.05:
                        next_tok = torch.argmax(logits, dim=-1, keepdim=True)
                    else:
                        probs = F.softmax(logits / max(0.1, t), dim=-1)
                        next_tok = torch.multinomial(probs, num_samples=1)

                    tok_id = int(next_tok[0, 0].item())
                    if tok_id == tok.eos_token_id:
                        break
                    emitted.append(next_tok)

                    # Update trace
                    if cfg["use_refractory"] and next_trace is not None:
                        refractory_trace = next_trace

                    # Re-afference
                    if cfg["reafference_scale"] > 0.0:
                        soft_p = F.softmax(logits, dim=-1)
                        soft_emb = soft_p @ model.sensory_dnn.embedding.weight
                        prev_thought = model.sensory_dnn(soft_emb)

                    # Core propagation
                    state_eff = model.volumetric_core.inject_efference_copy(state, efference)
                    state = model.volumetric_core.deliberate_step(state_eff)
                    state[:, :thalamic_end, :] = 0.9 * state[:, :thalamic_end, :] + 0.1 * thalamic_anchor

                if emitted:
                    out_tensor = torch.cat(emitted, dim=-1)
                    raw_text = tok.decode(out_tensor[0].tolist(), skip_special_tokens=True).strip()
                else:
                    raw_text = "<empty>"

            print(f"  Q: \"{p}\"")
            print(f"  A: {raw_text}")
            print("-" * 60)


if __name__ == "__main__":
    run_remedy_tests()
