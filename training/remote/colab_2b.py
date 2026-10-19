"""
Turnkey Standalone Training Script for Google Colab (GPU: T4 16GB, A100, V100, or L4).
Trains the exact CerebroSpheroid-2B (2,007,506,977 parameters) using PyTorch CUDA AMP.
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset


# =====================================================================
# 1. CerebroSpheroid-2B Exact Architecture (2,007,506,977 Parameters)
# =====================================================================

class SurfaceTransceiverBoundary(nn.Module):
    """Dual-Mode Boundary Transceiver on S^2 (65.5M parameters)."""
    def __init__(self, vocab_size: int = 32000, surface_dim: int = 2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.surface_dim = surface_dim
        self.transceiver_matrix = nn.Parameter(
            torch.empty(vocab_size, surface_dim).normal_(mean=0.0, std=0.02)
        )
        self.input_norm = nn.RMSNorm(surface_dim)
        self.output_norm = nn.RMSNorm(surface_dim)

    def ingest_sensory_input(self, token_ids: torch.Tensor) -> torch.Tensor:
        boundary_flux = F.embedding(token_ids, self.transceiver_matrix)
        return self.input_norm(boundary_flux)

    def emit_motor_output(self, surface_state: torch.Tensor) -> torch.Tensor:
        return F.linear(self.output_norm(surface_state), self.transceiver_matrix)


class ReentrantMidTensorCirculator(nn.Module):
    """Re-entrant Thalamocortical Loop Tensor Field (1.07B parameters)."""
    def __init__(self, num_shells: int = 32, shell_dim: int = 2048, radial_dim: int = 1024):
        super().__init__()
        self.radial_proj_in = nn.Linear(shell_dim, radial_dim, bias=False)
        self.radial_proj_out = nn.Linear(radial_dim, shell_dim, bias=False)
        self.reentrant_field = nn.Parameter(
            torch.empty(num_shells, num_shells, radial_dim, radial_dim).normal_(mean=0.0, std=0.008)
        )
        self.gate_proj = nn.Linear(shell_dim, shell_dim)
        self.leaky_inertia = nn.Parameter(torch.ones(num_shells, 1) * 0.85)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        radial_compact = self.radial_proj_in(state)  # [B, K, radial_dim]
        reentrant_flow = torch.einsum('bki,kjio->bjo', radial_compact, self.reentrant_field)
        reentrant_flow = self.radial_proj_out(F.silu(reentrant_flow))
        alpha = torch.sigmoid(self.gate_proj(state)) * torch.sigmoid(self.leaky_inertia)
        return alpha * state + (1.0 - alpha) * reentrant_flow


class VolumetricBrainCore(nn.Module):
    """32-Shell Continuous Volumetric Ball Core B^3 (~1.94B parameters)."""
    def __init__(
        self,
        num_shells: int = 32,
        shell_dim: int = 2048,
        radial_dim: int = 1024,
        num_shortcut_tracts: int = 173,
    ):
        super().__init__()
        self.num_shells = num_shells
        self.shell_dim = shell_dim
        self.radial_dim = radial_dim
        self.num_shortcut_tracts = num_shortcut_tracts

        # Precompute indices for 173 axonal tract bundles connecting shells
        src_indices = torch.arange(num_shortcut_tracts) % num_shells
        dst_indices = (num_shells - 1 - (torch.arange(num_shortcut_tracts) * 3) % num_shells)
        self.register_buffer("src_indices", src_indices)
        self.register_buffer("dst_indices", dst_indices)

        # 1. Concentric Intra-Shell Laplace-Beltrami Diffusion (134.2M params)
        self.intra_shell_operators = nn.Parameter(
            torch.empty(num_shells, shell_dim, shell_dim).normal_(mean=0.0, std=0.015)
        )

        # 2. Re-entrant Mid-Tensor Circulation Operator (1.07B params)
        self.mid_tensor_circulator = ReentrantMidTensorCirculator(
            num_shells=num_shells, shell_dim=shell_dim, radial_dim=radial_dim
        )

        # 3. White-Matter Shortcut Tracts (173 tracts = 725.6M params)
        self.shortcut_tracts = nn.Parameter(
            torch.empty(num_shortcut_tracts, shell_dim, shell_dim).normal_(mean=0.0, std=0.01)
        )

        self.damping_coefficient = nn.Parameter(torch.tensor(0.88))
        self.shell_norm = nn.RMSNorm(shell_dim)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.num_shells, self.shell_dim, device=device)

    def inject_sensory_flux(self, state: torch.Tensor, boundary_flux: torch.Tensor) -> torch.Tensor:
        new_state = state.clone()
        new_state[:, -1, :] = new_state[:, -1, :] + boundary_flux
        return self.shell_norm(new_state)

    def circulate_mid_tensors(self, state: torch.Tensor, num_cycles: int = 1) -> torch.Tensor:
        curr_state = state
        for _ in range(num_cycles):
            # Intra-shell diffusion
            intra_flow = torch.einsum('bkd,kdo->bko', curr_state, self.intra_shell_operators)
            intra_flow = F.gelu(intra_flow)

            # Re-entrant mid-tensor circulation
            reentrant_flow = self.mid_tensor_circulator(curr_state)

            # 173 White-matter shortcut tracts
            src_state = curr_state[:, self.src_indices, :]
            tract_flow = torch.einsum('btd,tdo->bto', src_state, self.shortcut_tracts)
            shortcut_flow = torch.zeros_like(curr_state)
            shortcut_flow.index_add_(1, self.dst_indices, tract_flow)

            updated_state = curr_state + self.damping_coefficient * (intra_flow + reentrant_flow) + 0.25 * shortcut_flow
            curr_state = self.shell_norm(updated_state)

        return curr_state

    def tap_surface_efferent(self, state: torch.Tensor) -> torch.Tensor:
        return state[:, -1, :]


class CerebroSpheroid2B(nn.Module):
    """CerebroSpheroid-2B Unified Brain Architecture (Exact 2,007,506,977 Parameters)."""
    def __init__(self, vocab_size: int = 32000, surface_dim: int = 2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.surface_dim = surface_dim
        self.boundary_transceiver = SurfaceTransceiverBoundary(vocab_size, surface_dim)
        self.volumetric_core = VolumetricBrainCore(
            num_shells=32,
            shell_dim=surface_dim,
            radial_dim=1024,
            num_shortcut_tracts=173,
        )

    def init_brain_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.volumetric_core.init_state(batch_size, device)

    def ingest_prompt(self, input_ids: torch.Tensor, internal_cycles_per_token: int = 1) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        state = self.init_brain_state(batch_size, device)
        for t in range(seq_len):
            flux = self.boundary_transceiver.ingest_sensory_input(input_ids[:, t])
            state = self.volumetric_core.inject_sensory_flux(state, flux)
            state = self.volumetric_core.circulate_mid_tensors(state, num_cycles=internal_cycles_per_token)
        return state

    def deliberate(self, state: torch.Tensor, num_cycles: int = 3) -> torch.Tensor:
        return self.volumetric_core.circulate_mid_tensors(state, num_cycles=num_cycles)

    def readout_logits(self, state: torch.Tensor) -> torch.Tensor:
        surface_echo = self.volumetric_core.tap_surface_efferent(state)
        return self.boundary_transceiver.emit_motor_output(surface_echo)

    @torch.no_grad()
    def generate_autonomous(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 25,
        deliberation_cycles: int = 1,
        mid_circulation_cycles_per_step: int = 1,
        temperature: float = 0.8,
    ):
        """Zero-feedback autonomous generation (output is NEVER fed back to input)."""
        self.eval()
        state = self.ingest_prompt(prompt_ids, internal_cycles_per_token=1)
        if deliberation_cycles > 0:
            state = self.deliberate(state, num_cycles=deliberation_cycles)

        generated_tokens = []
        for _ in range(max_new_tokens):
            logits = self.readout_logits(state)
            probs = F.softmax(logits / max(temperature, 1e-5), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens.append(next_token)
            # Mid-tensors circulate internally without feeding next_token back:
            state = self.volumetric_core.circulate_mid_tensors(state, num_cycles=mid_circulation_cycles_per_step)

        return torch.cat(generated_tokens, dim=-1)


# =====================================================================
# 2. Google Colab Training Pipeline
# =====================================================================

def main():
    print("=" * 75)
    print("CEREBRO-SPHEROID-2B: GOOGLE COLAB TRAINING HARNESS")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Accelerated by GPU: {gpu_name} ({gpu_vram:.2f} GB VRAM)")
    else:
        print("Warning: Running on CPU. For best performance, switch runtime to T4/A100 GPU!")

    # 1. Instantiate 2.01B Model
    print("\nInstantiating CerebroSpheroid-2B Architecture...")
    t0 = time.time()
    # In Colab T4 (16GB), instantiate in float16/bfloat16 for instant training:
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    
    model = CerebroSpheroid2B(vocab_size=32000, surface_dim=2048).to(device=device, dtype=dtype)
    total_params = sum(p.numel() for p in model.parameters())
    model_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
    
    print(f"Model Initialized in {time.time()-t0:.2f}s!")
    print(f"  • Total Parameters : {total_params:,} ({total_params/1e9:.6f} Billion)")
    print(f"  • VRAM Footprint   : {model_size_gb:.2f} GB in {dtype}")
    print(f"  • Non-Layered Core : 32 Concentric Shells | 173 White-Matter Tracts")

    # 2. Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print("\nConnecting to TinyStories streaming dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    # 3. Optimizer with Memory-Efficient 16-bit or SGD
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    batch_size = 4
    seq_len = 16
    grad_accum_steps = 2
    max_steps = 100

    def batch_stream():
        batch = []
        for item in dataset:
            text = item.get("text", "").strip()
            if not text:
                continue
            tokens = tokenizer.encode(text)
            if len(tokens) >= seq_len + 1:
                # Clamp to vocab range 32,000
                tokens = [min(t, 31999) for t in tokens[:seq_len + 1]]
                batch.append(tokens)
                if len(batch) == batch_size:
                    yield torch.tensor(batch, dtype=torch.long, device=device)
                    batch = []

    print("-" * 75)
    print(f"Starting Training: {max_steps} Steps | Batch: {batch_size} | Seq Len: {seq_len}")
    print("Tensors circulate through all 32 shells during each step (Backprop Through Deliberation)...")
    print("-" * 75)

    model.train()
    stream = batch_stream()
    train_start = time.time()
    optimizer.zero_grad()

    for step in range(1, max_steps + 1):
        batch = next(stream)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=dtype):
            state = model.init_brain_state(batch_size, device).to(dtype)
            step_loss = 0.0

            for t in range(seq_len):
                flux = model.boundary_transceiver.ingest_sensory_input(inputs[:, t])
                state = model.volumetric_core.inject_sensory_flux(state, flux)
                state = model.volumetric_core.circulate_mid_tensors(state, num_cycles=1)
                logits = model.readout_logits(state).float()
                step_loss = step_loss + criterion(logits, targets[:, t])

            step_loss = (step_loss / seq_len) / grad_accum_steps

        scaler.scale(step_loss).backward()

        if step % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if step % 10 == 0 or step == 1:
            raw_loss = step_loss.item() * grad_accum_steps
            elapsed = time.time() - train_start
            print(f"Step [{step:03d}/{max_steps}] | Loss: {raw_loss:.4f} | Perplexity: {math.exp(min(raw_loss, 20)):.2f} | Time: {elapsed:.1f}s")

    print("-" * 75)
    print(f"Training completed in {time.time()-train_start:.2f} seconds!")

    # 4. Save Checkpoint
    checkpoint_file = "cerebro_spheroid_2b_colab.pt"
    torch.save(model.state_dict(), checkpoint_file)
    print(f"Saved 2B model checkpoint to: {checkpoint_file}")

    # 5. Autonomous Zero-Feedback Generation Demo
    print("\n" + "=" * 75)
    print("AUTONOMOUS ZERO-FEEDBACK GENERATION (EXACT 2.01B MODEL)")
    print("=" * 75)
    model.eval()

    prompts = [
        "Once upon a time, a little rabbit named Jack",
        "The clever scientist discovered a secret inside the",
    ]

    for p in prompts:
        print(f"\n[Prompt]: \"{p}\"")
        prompt_tokens = [min(t, 31999) for t in tokenizer.encode(p)]
        prompt_tensor = torch.tensor([prompt_tokens], device=device)

        gen_tokens = model.generate_autonomous(
            prompt_tensor, max_new_tokens=20, deliberation_cycles=1, mid_circulation_cycles_per_step=1, temperature=0.7
        )
        gen_text = tokenizer.decode(gen_tokens[0].cpu().tolist(), skip_special_tokens=True)
        print(f"  [Zero-Feedback 2B Continuation]: \"{gen_text}\"")
        print("  ✓ Zero output token was fed back into the model!")

    print("\n" + "=" * 75)
    print("COLAB TRAINING & INFERENCE DEMO FINISHED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    main()
