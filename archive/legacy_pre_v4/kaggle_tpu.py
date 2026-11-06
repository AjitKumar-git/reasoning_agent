"""
CerebroSpheroid-2B Cognitive Architecture - Google Cloud TPU v3-8 Training Engine
Optimized for Kaggle Cloud TPU v3-8 (128 GB HBM, 8 Cores, Native bfloat16 Matrix Units).

Why TPU v3-8 is ideal for CerebroSpheroid-2B:
1. 128 GB High-Bandwidth Memory:
   - Full 2.19B model (4.18 GB) + Full AdamW moments (17.54 GB) + Large Batches fit with >95 GB to spare!
2. Native Bfloat16 Systolic Matrix Units (MXUs):
   - Flawless acceleration of 4D radial wave diffusion fields and einsum tensor contractions.
3. PyTorch/XLA Optimization:
   - Fuses deliberation loops into optimized XLA hardware graphs.
"""

import os
import sys
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

# Detect TPU via PyTorch/XLA
try:
    import torch_xla.core.xla_model as xm
    DEVICE = xm.xla_device()
    IS_TPU = True
    print(f"Cloud TPU Accelerator Active via PyTorch/XLA: {DEVICE}")
except Exception as e:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IS_TPU = False
    print(f"TPU not detected ({e}). Falling back to: {DEVICE}")

IS_KAGGLE = os.path.exists("/kaggle")
OUTPUT_DIR = "/kaggle/working" if IS_KAGGLE else "./checkpoints"
CACHE_DIR = os.path.join(OUTPUT_DIR, ".hf_cache")
os.environ["HF_HOME"] = CACHE_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# =========================================================================
# 1. Complete Cognitive Brain Architecture
# =========================================================================

class SensoryInputDNN(nn.Module):
    def __init__(self, vocab_size: int = 32000, hidden_dim: int = 2048, surface_dim: int = 2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.fc1_gate = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.fc1_val = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.norm1 = nn.RMSNorm(hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, surface_dim, bias=False)
        self.norm_out = nn.RMSNorm(surface_dim)
        self.residual_proj = nn.Linear(hidden_dim, surface_dim, bias=False) if hidden_dim != surface_dim else nn.Identity()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        gate = F.silu(self.fc1_gate(x))
        val = self.fc1_val(x)
        h = self.norm1(gate * val)
        flux = self.fc2(h) + self.residual_proj(x)
        return self.norm_out(flux)


class MotorOutputDNN(nn.Module):
    def __init__(self, surface_dim: int = 2048, core_dim: int = 2048, vocab_size: int = 32000):
        super().__init__()
        fused_dim = surface_dim + core_dim
        hidden_dim = surface_dim * 2
        self.fused_norm = nn.RMSNorm(fused_dim)
        self.fc_gate = nn.Linear(fused_dim, hidden_dim, bias=False)
        self.fc_val = nn.Linear(fused_dim, hidden_dim, bias=False)
        self.norm_mid = nn.RMSNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.temperature_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, surface_state: torch.Tensor, core_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        if core_state is None:
            core_state = surface_state
        fused = self.fused_norm(torch.cat([surface_state, core_state], dim=-1)).to(self.fc_gate.weight.dtype)
        gate = F.silu(self.fc_gate(fused))
        val = self.fc_val(fused)
        h = self.norm_mid(gate * val)
        logits = self.head(h) / torch.clamp(self.temperature_scale, min=0.1, max=5.0)
        return logits


class VolumetricBrainCore(nn.Module):
    def __init__(self, num_shells: int = 32, shell_dim: int = 2048, radial_dim: int = 1024, num_shortcut_tracts: int = 173):
        super().__init__()
        self.num_shells = num_shells
        self.shell_dim = shell_dim
        self.radial_dim = radial_dim
        self.num_shortcut_tracts = num_shortcut_tracts

        src_indices = torch.arange(num_shortcut_tracts) % num_shells
        dst_indices = (num_shells - 1 - (torch.arange(num_shortcut_tracts) * 3) % num_shells)
        self.register_buffer("src_indices", src_indices)
        self.register_buffer("dst_indices", dst_indices)

        self.intra_shell = nn.Parameter(torch.empty(num_shells, shell_dim, shell_dim).normal_(mean=0.0, std=0.015))
        self.radial_proj_in = nn.Linear(shell_dim, radial_dim, bias=False)
        self.reentrant_field = nn.Parameter(torch.empty(num_shells, num_shells, radial_dim, radial_dim).normal_(mean=0.0, std=0.008))
        self.radial_proj_out = nn.Linear(radial_dim, shell_dim, bias=False)
        self.shortcut_tracts = nn.Parameter(torch.empty(num_shortcut_tracts, shell_dim, shell_dim).normal_(mean=0.0, std=0.01))
        self.inertia_gate = nn.Parameter(torch.tensor(0.80))
        self.norm = nn.RMSNorm(shell_dim)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.num_shells, self.shell_dim, device=device)

    def inject_sensory_flux(self, state: torch.Tensor, boundary_flux: torch.Tensor) -> torch.Tensor:
        new_state = state.clone()
        new_state[:, -1, :] = new_state[:, -1, :] + boundary_flux.to(state.dtype)
        return self.norm(new_state).to(state.dtype)

    def deliberate_step(self, state: torch.Tensor) -> torch.Tensor:
        curr = state.to(self.intra_shell.dtype)
        intra_flow = torch.einsum('bkd,kdo->bko', curr, self.intra_shell)
        radial_compact = self.radial_proj_in(curr)
        radial_flow = self.radial_proj_out(torch.einsum('bki,kjio->bjo', radial_compact, self.reentrant_field))
        src_state = curr[:, self.src_indices, :]
        tract_flow = torch.einsum('btd,tdo->bto', src_state.to(self.shortcut_tracts.dtype), self.shortcut_tracts)
        shortcut_flow = torch.zeros(curr.shape, dtype=tract_flow.dtype, device=curr.device)
        shortcut_flow.index_add_(1, self.dst_indices, tract_flow)
        total_velocity = intra_flow + radial_flow + 0.3 * shortcut_flow
        gamma = torch.sigmoid(self.inertia_gate)
        next_state = gamma * curr + (1.0 - gamma) * torch.tanh(total_velocity)
        return self.norm(next_state).to(state.dtype)

    def circulate_mid_tensors(self, state: torch.Tensor, num_cycles: int = 1) -> torch.Tensor:
        curr = state
        for _ in range(num_cycles):
            curr = self.deliberate_step(curr)
        return curr

    def tap_efferents(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return state[:, -1, :], state[:, 0, :]


class CerebroSpheroid(nn.Module):
    def __init__(self, vocab_size: int = 32000, surface_dim: int = 2048, num_shells: int = 32, radial_dim: int = 1024, num_shortcut_tracts: int = 173):
        super().__init__()
        self.vocab_size = vocab_size
        self.sensory_dnn = SensoryInputDNN(vocab_size=vocab_size, hidden_dim=surface_dim, surface_dim=surface_dim)
        self.volumetric_core = VolumetricBrainCore(num_shells=num_shells, shell_dim=surface_dim, radial_dim=radial_dim, num_shortcut_tracts=num_shortcut_tracts)
        self.motor_dnn = MotorOutputDNN(surface_dim=surface_dim, core_dim=surface_dim, vocab_size=vocab_size)

    def ingest_stimulus(self, token_ids: torch.Tensor, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        if state is None:
            state = self.volumetric_core.init_state(batch_size, token_ids.device)
        for t in range(seq_len):
            flux = self.sensory_dnn(token_ids[:, t])
            state = self.volumetric_core.inject_sensory_flux(state, flux)
            state = self.volumetric_core.circulate_mid_tensors(state, num_cycles=1)
        return state

    def deliberate(self, state: torch.Tensor, thinking_cycles: int = 4) -> torch.Tensor:
        return self.volumetric_core.circulate_mid_tensors(state, num_cycles=thinking_cycles)

    def emit_next_token(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        surface_efferent, core_attractor = self.volumetric_core.tap_efferents(state)
        logits = self.motor_dnn(surface_efferent, core_attractor)
        next_state = self.volumetric_core.circulate_mid_tensors(state, num_cycles=1)
        return logits, next_state

    @torch.no_grad()
    def solve_with_active_thinking(self, prompt_tokens: torch.Tensor, thinking_cycles: int = 4, max_answer_tokens: int = 25, temperature: float = 0.6) -> torch.Tensor:
        self.eval()
        state = self.ingest_stimulus(prompt_tokens)
        state = self.deliberate(state, thinking_cycles=thinking_cycles)
        generated = []
        for _ in range(max_answer_tokens):
            logits, state = self.emit_next_token(state)
            probs = F.softmax(logits / max(temperature, 1e-5), dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            generated.append(next_tok)
        return torch.cat(generated, dim=-1)


# =========================================================================
# 2. Data Stream Loader (GSM8k)
# =========================================================================

def get_data_stream(tokenizer, vocab_size, batch_size=4):
    try:
        from datasets import load_dataset
        print("Connecting to Hugging Face GSM8K reasoning dataset...")
        ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
        def stream():
            batch_p, batch_t = [], []
            for item in ds:
                q = "Problem: " + item.get("question", "").strip() + " Solution:"
                a = item.get("answer", "").strip()
                if not q or not a: continue
                q_toks = [min(t, vocab_size - 1) for t in tokenizer.encode(q)[:24]]
                a_toks = [min(t, vocab_size - 1) for t in tokenizer.encode(a)[:16]]
                if len(q_toks) >= 6 and len(a_toks) >= 4:
                    batch_p.append(q_toks)
                    batch_t.append(a_toks)
                    if len(batch_p) == batch_size:
                        max_p = max(len(p) for p in batch_p)
                        max_t = max(len(t) for t in batch_t)
                        p_pad = [p + [tokenizer.eos_token_id] * (max_p - len(p)) for p in batch_p]
                        t_pad = [t + [tokenizer.eos_token_id] * (max_t - len(t)) for t in batch_t]
                        yield torch.tensor(p_pad, dtype=torch.long), torch.tensor(t_pad, dtype=torch.long)
                        batch_p, batch_t = [], []
        return stream()
    except Exception as e:
        print(f"[Notice] Streaming unavailable ({e}). Using algorithmic multi-step stream.")
        def synthetic_stream():
            while True:
                p = torch.randint(100, 2000, (batch_size, 16), dtype=torch.long)
                t = torch.randint(100, 2000, (batch_size, 8), dtype=torch.long)
                yield p, t
        return synthetic_stream()


# =========================================================================
# 3. Main TPU Training Loop
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train CerebroSpheroid-2B on Google Cloud TPU v3-8")
    parser.add_argument("--scale", type=str, default="2b", choices=["2b", "260m"])
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (TPUs excel at larger batches)")
    parser.add_argument("--thinking_cycles", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    vocab_size = 32000

    print("=" * 80)
    print("      CEREBRO-SPHEROID-2B: CLOUD TPU v3-8 HIGH-THROUGHPUT ENGINE        ")
    print("=" * 80)
    print(f"Target Accelerator : {'Google Cloud TPU v3-8 (128 GB HBM)' if IS_TPU else str(DEVICE)}")
    print(f"Target Scale       : {args.scale.upper()}")
    print(f"Batch Size         : {args.batch_size} (Massive memory headroom on TPU)")
    print(f"Thinking Cycles    : {args.thinking_cycles} cycles (Subconscious deliberation)")
    print(f"Output Directory   : {OUTPUT_DIR}")
    print("=" * 80)

    if args.scale == "2b":
        model = CerebroSpheroid(vocab_size=vocab_size, surface_dim=2048, num_shells=32, radial_dim=1024, num_shortcut_tracts=173)
    else:
        model = CerebroSpheroid(vocab_size=vocab_size, surface_dim=1024, num_shells=16, radial_dim=256, num_shortcut_tracts=48)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params:,} ({total_params/1e9:.3f} Billion)")

    # Load to TPU in native bfloat16
    dtype = torch.bfloat16
    print(f"Loading model onto {DEVICE} in hardware-native {dtype}...")
    model.to(device=DEVICE, dtype=dtype)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # On TPU (128 GB HBM), we can easily run full AdamW!
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    print("\n" + "-" * 80)
    print(f"Starting Deliberate-Then-Emit Training: {args.steps} Steps on Cloud TPU...")
    print("-" * 80)

    data_stream = get_data_stream(tokenizer, vocab_size, batch_size=args.batch_size)
    t_start = time.time()
    losses = []

    model.train()
    for step in range(1, args.steps + 1):
        prompts, targets = next(data_stream)
        prompts = prompts.to(DEVICE)
        targets = targets.to(DEVICE)
        ans_len = targets.shape[1]

        optimizer.zero_grad()

        # Ingestion -> Deliberation -> Answer Emission
        with torch.no_grad():
            state = model.ingest_stimulus(prompts)
        state = state.detach().requires_grad_(True)
        state = model.deliberate(state, thinking_cycles=args.thinking_cycles)

        logits_list = []
        for t in range(ans_len):
            logits_t, state = model.emit_next_token(state)
            logits_list.append(logits_t)

        predicted_logits = torch.stack(logits_list, dim=1)
        loss = criterion(predicted_logits.view(-1, vocab_size), targets.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # TPU PyTorch/XLA step execution
        if IS_TPU:
            xm.optimizer_step(optimizer)
            xm.mark_step()
        else:
            optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if step % 20 == 0 or step == 1:
            elapsed = time.time() - t_start
            tok_s = (step * args.batch_size * (prompts.shape[1] + ans_len)) / elapsed
            print(f"Step [{step:03d}/{args.steps:03d}] | Loss: {loss_val:.4f} | PPL: {math.exp(min(loss_val, 20)):.2f} | Speed: {tok_s:.0f} tok/s | Elapsed: {elapsed:.1f}s")

    train_time = time.time() - t_start
    print("-" * 80)
    print(f"TPU Training Completed in {train_time:.2f}s! Final Loss: {losses[-1]:.4f}")

    ckpt_path = os.path.join(OUTPUT_DIR, f"cerebro_spheroid_{args.scale}_tpu.pt")
    if IS_TPU:
        xm.save(model.state_dict(), ckpt_path)
    else:
        torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint successfully saved to: {ckpt_path}")

    # Active Thinking Demo
    print("\n" + "=" * 80)
    print("TPU ACTIVE DELIBERATION TEST: INGESTION -> 4 THINKING CYCLES -> ZERO-FEEDBACK ANSWER")
    print("=" * 80)

    test_queries = [
        "Problem: Maya has 8 markers. She buys 4 more and gives 3 away. How many markers does she have? Solution:",
        "Problem: A train travels 60 miles in 1 hour. How many miles does it travel in 3 hours? Solution:",
    ]

    model.eval()
    with torch.no_grad():
        for q in test_queries:
            print(f"\n[Problem]: \"{q}\"")
            toks = [min(t, vocab_size - 1) for t in tokenizer.encode(q)]
            prompt_tensor = torch.tensor([toks], device=DEVICE)
            ans_toks = model.solve_with_active_thinking(prompt_tensor, thinking_cycles=4, max_answer_tokens=20, temperature=0.6)
            ans_text = tokenizer.decode(ans_toks[0].tolist(), skip_special_tokens=True)
            print(f"  [Cognitive Answer]: \"{ans_text}\"")
            print("  ✓ 4 internal thinking cycles executed before speech.")
            print("  ✓ Zero emitted tokens fed back into the model!")

    print("\n" + "=" * 80)
    print(f"TPU Job Finished Successfully! Download model from: {ckpt_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
