"""
Evaluate Loss of Checkpoint under the New Architecture.

Supports:
- Checkpoints from bioflo.in (Cloud TPU JAX checkpoints: Step 9600, Loss ~5.4 on TPU batches).
- Checkpoints from Kaggle (GPU PyTorch checkpoints).

Evaluates:
1. Teacher-Forcing Loss (Standard baseline).
2. Autonomous Internal Deliberation Loss (New Architecture: zero external feedback).
3. Effect of Thalamic Anchor (weight=0.0, 0.2, 0.5).
4. Breakdown across OpenHermes-2.5 and GSM8K evaluation splits.
"""

import argparse
import math
import os
import random
import sys
import time
import torch
import torch.nn.functional as F
import pyarrow.parquet as pq

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.corpus import build_tokenizer, _conversation_pairs
from training.train_v4_full import build_batch


def load_openhermes_eval_records(count: int = 50):
    """Loads held-out prompt-target pairs from OpenHermes-2.5."""
    path = os.path.join(ROOT_DIR, "data/openhermes/openhermes_2_5.parquet")
    pf = pq.ParquetFile(path)
    table = pf.read_row_group(0)
    convs = table["conversations"].to_pylist()
    records = []
    for conv in convs:
        pairs = _conversation_pairs(conv)
        for p, a in pairs:
            if len(p.strip()) > 10 and len(a.strip()) > 10:
                records.append(("openhermes", p.strip(), a.strip()))
                if len(records) >= count:
                    return records
    return records


def load_gsm8k_eval_records(count: int = 50):
    """Loads held-out prompt-target pairs from GSM8K."""
    path = os.path.join(ROOT_DIR, "data/gsm8k/gsm8k_train.parquet")
    pf = pq.ParquetFile(path)
    table = pf.read()
    questions = table["question"].to_pylist()
    answers = table["answer"].to_pylist()
    records = []
    for q, a in zip(questions[-count:], answers[-count:]):
        records.append(("gsm8k", f"Question: {q.strip()}\nAnswer:", f" {a.strip()}"))
    return records


class BitwiseJAXEngine:
    """Exact bitwise PyTorch implementation of the JAX V4 CerebroSpheroid architecture."""
    def __init__(self, ckpt_dict, device):
        self.device = device
        self.p = {k: v.to(device) for k, v in ckpt_dict.items()}
        self.num_shells = 16
        self.num_shortcut_tracts = 40
        self.src_idx = (torch.arange(self.num_shortcut_tracts) % self.num_shells).to(device)
        self.dst_idx = (self.num_shells - 1 - (torch.arange(self.num_shortcut_tracts) * 3) % self.num_shells).to(device)

    @staticmethod
    def rms_norm(x, w, eps=1e-6):
        return (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)) * w

    def encode_sensory(self, tokens):
        x = self.p['sensory.embedding.weight'][tokens]
        gate = F.silu(F.linear(x, self.p['sensory.fc1_gate.weight']))
        val = F.linear(x, self.p['sensory.fc1_val.weight'])
        h = self.rms_norm(gate * val, self.p['sensory.norm1.weight'])
        flux = F.linear(h, self.p['sensory.fc2.weight']) + x
        sens = self.rms_norm(flux, self.p['sensory.norm_out.weight'])
        keys = F.linear(sens, self.p['motor.k_proj.weight'])
        vals = F.linear(sens, self.p['motor.v_proj.weight'])
        return sens, (keys, vals)

    def deliberate_step(self, state):
        B, K, D = state.shape
        intra = torch.einsum('bkd,kdo->bko', state, self.p['core.intra_shell'])
        state_src = state[:, self.src_idx, :]
        tracts_out = torch.einsum('bti,tij->btj', state_src, self.p['core.shortcut_tracts'])
        tract_acc = torch.zeros_like(state)
        tract_acc.index_add_(1, self.dst_idx, tracts_out)

        rad_in = F.linear(state, self.p['core.radial_proj_in.weight'])
        reentrant = torch.einsum('bki,kjio->bjo', rad_in, self.p['core.reentrant_field'])
        reentrant = F.linear(F.silu(reentrant), self.p['core.radial_proj_out.weight'])

        flow = self.rms_norm(intra + tract_acc + reentrant, self.p['core.flow_norm.weight'])
        leak = torch.sigmoid(self.p['core.log_leak']).view(1, -1, 1)
        write = F.softplus(self.p['core.log_write']).view(1, -1, 1)

        raw = (1.0 - leak) * state + write * torch.tanh(flow)
        return self.rms_norm(raw, self.p['core.norm.weight'])

    def emit_step(self, state, sensory_kv, plan, refractory, step_idx, auditory_feedback=None):
        B, K, D = state.shape
        surface, core = state[:, -1, :], state[:, 0, :]
        max_phase = self.p['motor.phase_embed.weight'].shape[0]
        phase_idx = min(max(step_idx, 0), max_phase - 1)
        phase = self.p['motor.phase_embed.weight'][phase_idx]
        motor_surface = surface + phase

        if auditory_feedback is not None:
            motor_surface = motor_surface + auditory_feedback

        keys, values = sensory_kv
        query = F.linear(motor_surface, self.p['motor.q_proj.weight']).unsqueeze(1)
        scores = torch.matmul(query, keys.transpose(-1, -2)) / (D ** 0.5)
        attn = F.softmax(scores, dim=-1)
        resonance = self.rms_norm(torch.matmul(attn, values).squeeze(1), self.p['motor.resonance_norm.weight'])

        shell_weights = F.softmax(self.p['motor.shell_score'], dim=0).view(1, -1, 1)
        shell_proj = F.linear(state, self.p['motor.shell_proj.weight'])
        shell_pool = (shell_weights * shell_proj).sum(dim=1) * self.p['motor.shell_scale']

        fused = self.rms_norm(
            torch.cat([motor_surface, core, resonance, shell_pool, plan], dim=-1),
            self.p['motor.fused_norm.weight']
        )
        gate = F.silu(F.linear(fused, self.p['motor.fc_gate.weight']))
        val = F.linear(fused, self.p['motor.fc_val.weight'])
        h = self.rms_norm(gate * val, self.p['motor.norm_mid.weight'])

        phase_batch = phase.unsqueeze(0).expand(B, -1)
        plan_input = torch.cat([h, phase_batch], dim=-1)
        p_gate = torch.sigmoid(F.linear(plan_input, self.p['motor.plan_gate.weight'], self.p['motor.plan_gate.bias']))
        plan_decay = torch.clamp(self.p['motor.plan_decay'], 0.0, 0.92)
        plan_update = F.linear(plan_input, self.p['motor.plan_update.weight'])
        next_plan = torch.clamp(plan_decay * plan + self.p['motor.plan_write'] * p_gate * plan_update, -50.0, 50.0)

        temp = torch.clamp(self.p['motor.temperature_scale'], 0.1, 5.0)
        raw_logits = F.linear(h, self.p['motor.head.weight']) / temp
        efference = F.linear(h, self.p['motor.efference_proj.weight'])

        alpha = torch.clamp(self.p['motor.refractory_strength'], 0.1, 5.0)
        adapted_logits = raw_logits - alpha * refractory
        decay = torch.sigmoid(self.p['motor.refractory_decay'])
        current_prob = F.softmax(raw_logits, dim=-1).detach()
        next_trace = (decay * refractory + (1.0 - decay) * current_prob).detach()

        state_eff = state.clone()
        state_eff[:, 0, :] = state_eff[:, 0, :] + efference
        next_state = self.deliberate_step(state_eff)
        return adapted_logits, next_state, next_trace, next_plan


@torch.no_grad()
def evaluate_dataset(engine, records, tokenizer, device, deliberation_mode="internal", thalamic_anchor_weight=0.0, batch_size=8, window=128):
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
    total_loss = 0.0
    total_tokens = 0
    rng = random.Random(1337)

    for i in range(0, len(records), batch_size):
        batch_records = records[i : i + batch_size]
        built = build_batch(
            tokenizer,
            batch_records,
            window=window,
            rng=rng,
            max_prompt_len=256,
            max_answer_tokens=window,
            pad_token_id=pad_id,
        )
        if built is None:
            continue

        prompts, targets, feedback = (t.to(device) for t in built)
        B, T = targets.shape

        sens, kv = engine.encode_sensory(prompts)
        state = torch.zeros(B, engine.num_shells, 640, device=device)
        state[:, -1, :] = sens.mean(dim=1)
        for _ in range(3):
            state = engine.deliberate_step(state)

        plan = torch.zeros(B, 640, device=device)
        ref = torch.zeros(B, 32000, device=device)
        thalamic_end = max(1, engine.num_shells // 3)
        thalamic_anchor = state[:, :thalamic_end, :].clone()
        logits_list = []

        for step in range(T):
            feed_step = None
            if deliberation_mode == "teacher_forcing":
                prev_tok = feedback[:, step]
                valid = (prev_tok >= 0)
                if valid.any():
                    safe_prev = torch.where(valid, prev_tok, torch.zeros_like(prev_tok))
                    f_sens, _ = engine.encode_sensory(safe_prev.unsqueeze(1))
                    feed_step = f_sens.squeeze(1) * valid.unsqueeze(-1)

            step_logits, state, ref, plan = engine.emit_step(
                state, kv, plan, ref, step, auditory_feedback=feed_step
            )

            if thalamic_anchor_weight > 0.0:
                state[:, :thalamic_end, :] = (
                    (1.0 - thalamic_anchor_weight) * state[:, :thalamic_end, :]
                    + thalamic_anchor_weight * thalamic_anchor
                )

            logits_list.append(step_logits)

        all_logits = torch.stack(logits_list, dim=1)
        loss = F.cross_entropy(
            all_logits.reshape(-1, 32000),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        n_tokens = (targets != -100).sum().item()
        total_loss += loss.item()
        total_tokens += n_tokens

    if total_tokens == 0:
        return float("nan"), float("nan"), 0

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl, total_tokens


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint loss under the new architecture.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(ROOT_DIR, "checkpoints/cerebro_spheroid_v4_bioflo_latest.pt"),
        help="Path to checkpoint .pt file",
    )
    parser.add_argument("--count", type=int, default=50, help="Number of records per dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--window", type=int, default=128, help="Token response window")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint

    print("=" * 80)
    print(" 🧠 LOSS EVALUATION: CHECKPOINT FROM BIOFLO.IN UNDER NEW ARCHITECTURE")
    print(f" Device    : {device}")
    print(f" Checkpoint: {ckpt_path}")
    print("=" * 80)

    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        return

    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    weights = raw_ckpt.get("model", raw_ckpt) if isinstance(raw_ckpt, dict) else raw_ckpt

    # Load Tokenizer
    tok = build_tokenizer("llama")
    print(f"\n[1/3] Initializing Bitwise-Faithful Architecture Engine...")
    engine = BitwiseJAXEngine(weights, device=device)
    print(f"  Initialized {len(engine.p)} tensors successfully.")

    # Load Evaluation Records
    print(f"\n[2/3] Loading Held-Out Evaluation Data...")
    oh_records = load_openhermes_eval_records(count=args.count)
    gsm_records = load_gsm8k_eval_records(count=args.count)
    print(f"  Loaded {len(oh_records)} OpenHermes-2.5 held-out records.")
    print(f"  Loaded {len(gsm_records)} GSM8K held-out records.")

    # 3. Evaluate under different architecture modes
    print(f"\n[3/3] Evaluating Held-Out Loss Across Architectural Regimes...")

    modes = [
        ("Teacher-Forcing (Standard Baseline)", "teacher_forcing", 0.0),
        ("Autonomous Internal Deliberation (Unanchored)", "internal", 0.0),
        ("Autonomous Internal Deliberation (Anchored α=0.2)", "internal", 0.2),
        ("Autonomous Internal Deliberation (Anchored α=0.5)", "internal", 0.5),
    ]

    results = []
    for mode_name, delib_mode, anchor_wt in modes:
        print(f"\nEvaluating: {mode_name}...")
        t0 = time.time()
        loss_oh, ppl_oh, tok_oh = evaluate_dataset(
            engine, oh_records, tok, device,
            deliberation_mode=delib_mode,
            thalamic_anchor_weight=anchor_wt,
            batch_size=args.batch_size,
            window=args.window,
        )
        loss_gsm, ppl_gsm, tok_gsm = evaluate_dataset(
            engine, gsm_records, tok, device,
            deliberation_mode=delib_mode,
            thalamic_anchor_weight=anchor_wt,
            batch_size=args.batch_size,
            window=args.window,
        )
        dt = time.time() - t0

        tot_tokens = tok_oh + tok_gsm
        tot_loss = (loss_oh * tok_oh + loss_gsm * tok_gsm) / max(tot_tokens, 1)
        tot_ppl = math.exp(min(tot_loss, 20.0))

        results.append({
            "mode": mode_name,
            "loss_total": tot_loss,
            "ppl_total": tot_ppl,
            "loss_oh": loss_oh,
            "ppl_oh": ppl_oh,
            "loss_gsm": loss_gsm,
            "ppl_gsm": ppl_gsm,
            "time": dt,
            "tokens": tot_tokens,
        })
        print(f"  Total Loss: {tot_loss:.4f} | Total PPL: {tot_ppl:.2f} (Time: {dt:.1f}s, Tokens: {tot_tokens:,})")
        print(f"    - OpenHermes: Loss {loss_oh:.4f} | PPL {ppl_oh:.2f}")
        print(f"    - GSM8K     : Loss {loss_gsm:.4f} | PPL {ppl_gsm:.2f}")

    print("\n" + "=" * 80)
    print(" 🏆 FINAL LOSS COMPARISON: BIOFLO.IN CHECKPOINT UNDER NEW ARCHITECTURE")
    print("=" * 80)
    print(f"{'Evaluation Regime':<48} | {'Total CE':<9} | {'Total PPL':<10} | {'OpenHermes':<10} | {'GSM8K'}")
    print("-" * 80)
    for r in results:
        print(f"{r['mode']:<48} | {r['loss_total']:<9.4f} | {r['ppl_total']:<10.2f} | {r['loss_oh']:<10.4f} | {r['loss_gsm']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
