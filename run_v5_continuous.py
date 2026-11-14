#!/usr/bin/env python3
"""
Continuous Output / Streaming Inspector for CerebroSpheroid V5.
Streams autoregressive token emissions continuously without premature stops.
"""

import os
import sys
import time
import math
import argparse
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), ".hf_cache")))

from transformers import AutoTokenizer
from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5, apply_rope


def continuous_generate_stream(
    model: CerebroSpheroidV5,
    prompt_ids: torch.Tensor,
    tokenizer,
    max_tokens: int = 150,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.25,
    thinking_cycles: int = 4,
    ignore_eos: bool = True,
):
    model.eval()
    B, L = prompt_ids.shape
    dev = prompt_ids.device

    print(f"\n[Sensory Ingestion] Ingesting {L} prompt tokens into continuous cortical boundary...")
    sensory_flux, sensory_kv = model.sensory_dnn(prompt_ids)
    prompt_centroid = sensory_flux.mean(dim=1)

    # State initialization
    subcortical = torch.zeros(B, model.config.num_shells - 1, model.config.surface_dim, device=dev)
    state = torch.cat([subcortical, prompt_centroid.unsqueeze(1)], dim=1)

    print(f"[Subconscious Deliberation] Running {thinking_cycles} cycles in B^3 volumetric core...")
    for cycle in range(thinking_cycles):
        if sensory_kv is not None and model.deliberation_probe is not None:
            keys, values = sensory_kv
            L_kv = keys.shape[2]
            probe_q = model.deliberation_probe(state.mean(dim=1)).view(B, model.config.num_heads, 1, model.sensory_dnn.head_dim)
            q_rot = apply_rope(probe_q, model.motor_decoder.rope_cos, model.motor_decoder.rope_sin, offset=cycle)
            k_rot = apply_rope(keys, model.motor_decoder.rope_cos, model.motor_decoder.rope_sin, offset=0)
            scores = torch.matmul(q_rot.to(keys.dtype), k_rot.transpose(-1, -2)) / math.sqrt(model.sensory_dnn.head_dim)
            if model.config.top_k_resonance > 0 and L_kv > model.config.top_k_resonance:
                topk_v, _ = torch.topk(scores, model.config.top_k_resonance, dim=-1)
                scores = torch.where(scores < topk_v[..., -1, None], torch.full_like(scores, -1e9), scores)
            attn_p = F.softmax(scores, dim=-1)
            retrieved = torch.matmul(attn_p, values).transpose(1, 2).contiguous().view(B, model.config.surface_dim)
            state_mantle = state[:, -1:, :] + 0.1 * model.deliberation_probe_norm(retrieved).unsqueeze(1)
            state = torch.cat([state[:, :-1, :], state_mantle], dim=1)

        state = model.volumetric_core.deliberate_step(state)

    print("\n" + "=" * 60)
    print(" CONTINUOUS SPHEROID V5 MOTOR EMISSION STREAM")
    print("=" * 60 + "\n")
    sys.stdout.flush()

    plan = model.motor_decoder.init_plan(B, dev) if model.config.use_plan else None
    refractory = torch.zeros(B, model.config.vocab_size, device=dev)
    prev_thought = torch.zeros(B, model.config.surface_dim, device=dev)

    causal_keys_list = []
    causal_values_list = []
    eos_id = getattr(tokenizer, "eos_token_id", 2)

    generated_ids = []
    
    with torch.no_grad():
        for t in range(max_tokens):
            auditory_in = prev_thought if t > 0 else None
            causal_kv = None
            if len(causal_keys_list) > 0:
                causal_kv = (torch.cat(causal_keys_list, dim=2), torch.cat(causal_values_list, dim=2))

            logits, efference, refractory, plan = model.motor_decoder(
                state=state,
                sensory_kv=sensory_kv,
                tied_embedding_weight=model.sensory_dnn.embedding.weight,
                plan=plan,
                refractory_trace=refractory,
                step_idx=t,
                auditory_feedback=auditory_in,
                repetition_penalty=repetition_penalty,
                causal_kv=causal_kv,
            )

            # Restrict logits to vocab size
            if logits.size(-1) > model.config.vocab_size:
                logits[:, model.config.vocab_size:] = -float("Inf")

            logits = logits / max(temperature, 1e-4)

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1) # [B, 1]
            tok_id = next_tok.item()
            generated_ids.append(tok_id)

            piece = tokenizer.decode([tok_id], skip_special_tokens=False)
            sys.stdout.write(piece)
            sys.stdout.flush()

            if not ignore_eos and tok_id == eos_id:
                print("\n[EOS reached]")
                break

            prev_thought, (tok_k, tok_v) = model.sensory_dnn(next_tok.squeeze(-1))
            causal_keys_list.append(tok_k)
            causal_values_list.append(tok_v)

            state_eff = model.volumetric_core.inject_efference_copy(state, efference)
            state = model.volumetric_core.deliberate_step(state_eff)

    print("\n\n" + "=" * 60)
    print(" END OF CONTINUOUS EMISSION STREAM")
    print("=" * 60)
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/salvaged/cerebro_spheroid_v5_small_salvaged_step158500.pt")
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--max_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--thinking_cycles", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading LLaMA tokenizer & checkpoint: {args.checkpoint} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b", local_files_only=True)
    data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = data["config"]
    if isinstance(cfg, dict):
        cfg = ModelConfigV5(**cfg)

    model = CerebroSpheroidV5(cfg)
    state_dict = data.get("model_state", data.get("model", data))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Checkpoint loaded! Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    model.to(device)
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    print(f"Prompt: \"{args.prompt}\"")
    continuous_generate_stream(
        model=model,
        prompt_ids=prompt_tensor,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        thinking_cycles=args.thinking_cycles,
    )


if __name__ == "__main__":
    main()
