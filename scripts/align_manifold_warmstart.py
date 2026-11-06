#!/usr/bin/env python3
"""
Manifold-Accelerated Weight Grafting for CerebroSpheroid V4.

Supports:
- JackFram/llama-160m (160M params)
- TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T (1.1B params, 3T tokens)
- openlm-research/open_llama_3b_v2 (3B params)

Mathematical Principle:
Pretrained LLaMA models share the exact 100% bit-for-bit 32,000-token BPE vocabulary
with CerebroSpheroid V4. Their learned weight matrices lie on a compact Riemannian manifold.
This engine performs GPU-accelerated Orthogonal Procrustes / SVD Subspace Projections
to transfer converged lexical, syntactic, and attention representations into CerebroSpheroid.
"""

import argparse
import math
import os
import sys
import time
import pickle
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/tmp/hf_cache"


def compute_orthogonal_projection_gpu(
    src_matrix: torch.Tensor, target_dim: int, device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Computes the optimal rank-k orthogonal projection matrix using SVD / Eigendecomposition on GPU.
    """
    d_src = src_matrix.shape[1]
    if target_dim > d_src:
        raise ValueError(f"Target dim ({target_dim}) cannot exceed source dim ({d_src})")
    
    mat = src_matrix.to(device=device, dtype=torch.float32)
    gram = mat.T @ mat  # (D_src, D_src)
    eigvals, eigvecs = torch.linalg.eigh(gram)
    
    # Sort descending
    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    P = eigvecs[:, :target_dim]  # (D_src, target_dim)
    
    total_energy = torch.sum(torch.clamp(eigvals, min=0.0)).item()
    captured_energy = torch.sum(torch.clamp(eigvals[:target_dim], min=0.0)).item()
    energy_ratio = float(captured_energy / max(total_energy, 1e-12))
    
    projected = mat @ P
    return P, projected, energy_ratio


def init_default_cerebro_params(surface_dim: int = 640, num_shells: int = 16, radial_dim: int = 256,
                               num_shortcut_tracts: int = 40, vocab_size: int = 32000) -> Dict[str, torch.Tensor]:
    """Initializes CerebroSpheroid V4 parameters in PyTorch matching JAX key naming."""
    fused_dim = surface_dim * 5
    p: Dict[str, torch.Tensor] = {
        # Recurrent Volumetric Core (B^3)
        "core.intra_shell": torch.randn(num_shells, surface_dim, surface_dim) * 0.015,
        "core.reentrant_field": torch.randn(num_shells, num_shells, radial_dim, radial_dim) * 0.008,
        "core.shortcut_tracts": torch.randn(num_shortcut_tracts, surface_dim, surface_dim) * 0.01,
        "core.radial_proj_in.weight": torch.randn(radial_dim, surface_dim) * (1.0 / math.sqrt(surface_dim)),
        "core.radial_proj_out.weight": torch.randn(surface_dim, radial_dim) * (1.0 / math.sqrt(radial_dim)),
        "core.norm.weight": torch.ones(surface_dim),
        "core.flow_norm.weight": torch.ones(surface_dim),
        "core.log_write": torch.full((num_shells,), math.log(math.expm1(0.35))),
        "core.log_leak": torch.full((num_shells,), math.log(0.35 / 0.65)),

        # Sensory Front-End
        "sensory.embedding.weight": torch.randn(vocab_size, surface_dim) * 0.02,
        "sensory.fc1_gate.weight": torch.randn(surface_dim * 2, surface_dim) * (1.0 / math.sqrt(surface_dim)),
        "sensory.fc1_val.weight": torch.randn(surface_dim * 2, surface_dim) * (1.0 / math.sqrt(surface_dim)),
        "sensory.norm1.weight": torch.ones(surface_dim * 2),
        "sensory.fc2.weight": torch.randn(surface_dim, surface_dim * 2) * (1.0 / math.sqrt(surface_dim * 2)),
        "sensory.norm_out.weight": torch.ones(surface_dim),

        # Motor Output Decoder (Broca's Area)
        "motor.phase_embed.weight": torch.randn(1024, surface_dim) * 0.02,
        "motor.q_proj.weight": torch.randn(surface_dim, surface_dim) * (1.0 / math.sqrt(surface_dim)),
        "motor.k_proj.weight": torch.randn(surface_dim, surface_dim) * (1.0 / math.sqrt(surface_dim)),
        "motor.v_proj.weight": torch.randn(surface_dim, surface_dim) * (1.0 / math.sqrt(surface_dim)),
        "motor.resonance_norm.weight": torch.ones(surface_dim),
        "motor.shell_proj.weight": torch.randn(surface_dim, surface_dim) * (1.0 / math.sqrt(surface_dim)),
        "motor.shell_score": torch.zeros(num_shells),
        "motor.shell_scale": torch.zeros(1),
        "motor.plan_gate.weight": torch.randn(surface_dim, surface_dim * 3) * (1.0 / math.sqrt(surface_dim * 3)),
        "motor.plan_gate.bias": torch.zeros(surface_dim),
        "motor.plan_update.weight": torch.randn(surface_dim, surface_dim * 3) * (1.0 / math.sqrt(surface_dim * 3)),
        "motor.plan_write": torch.zeros(1),
        "motor.plan_decay": torch.tensor(0.90),
        "motor.fused_norm.weight": torch.ones(fused_dim),
        "motor.fc_gate.weight": torch.randn(surface_dim * 2, fused_dim) * (1.0 / math.sqrt(fused_dim)),
        "motor.fc_val.weight": torch.randn(surface_dim * 2, fused_dim) * (1.0 / math.sqrt(fused_dim)),
        "motor.norm_mid.weight": torch.ones(surface_dim * 2),
        "motor.head.weight": torch.randn(vocab_size, surface_dim * 2) * (1.0 / math.sqrt(surface_dim * 2)),
        "motor.temperature_scale": torch.tensor(1.0),
        "motor.refractory_strength": torch.tensor(1.2),
        "motor.refractory_decay": torch.tensor(0.75),
        "motor.efference_proj.weight": torch.randn(surface_dim, surface_dim * 2) * 0.01,

        # Agentic Flow & Persona Steering Basis
        "motor.persona_embeddings": torch.randn(4, surface_dim) * 0.05,
        "motor.persona_scale": torch.tensor(0.5),
        "motor.plan_init.weight": torch.randn(surface_dim, surface_dim * 2) * (1.0 / math.sqrt(surface_dim * 2)),
        "motor.plan_init.bias": torch.zeros(surface_dim),
    }
    return p


def graft_manifold_weights(
    cerebro_params: Dict[str, torch.Tensor],
    source_model_name: str = "JackFram/llama-160m",
    target_dim: int = 640,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Downloads source LLaMA model and grafts its converged manifold representations
    into CerebroSpheroid V4 parameters via GPU SVD / Orthogonal Procrustes projection.
    """
    from transformers import AutoModelForCausalLM
    
    print(f"\n=================================================================")
    print(f"  Manifold Grafting Engine: {source_model_name} -> CerebroSpheroid V4")
    print(f"  Compute Device: {device.upper()}")
    print(f"=================================================================")
    print(f"[*] Downloading / Loading source model: {source_model_name}...")
    
    src_model = AutoModelForCausalLM.from_pretrained(source_model_name, torch_dtype=torch.float32)
    src_dict = src_model.state_dict()
    
    d_src = src_model.config.hidden_size
    d_inter = src_model.config.intermediate_size
    vocab = src_model.config.vocab_size
    print(f"[*] Source architecture: hidden_size={d_src}, intermediate_size={d_inter}, vocab={vocab}")
    print(f"[*] Target architecture: surface_dim={target_dim}, fused_dim={target_dim * 5}, vocab=32000")
    
    stats = {}
    
    # -------------------------------------------------------------
    # 1. Lexical Semantic Manifold (Embedding Projection)
    # -------------------------------------------------------------
    print(f"\n[1/5] Projecting Lexical Semantic Manifold (Embedding)...")
    src_embed = src_dict["model.embed_tokens.weight"].to(device=device, dtype=torch.float32)
    P_embed, embed_projected, energy_embed = compute_orthogonal_projection_gpu(src_embed, target_dim, device=device)
    print(f"  -> GPU SVD Orthogonal Projection ({d_src} -> {target_dim})")
    print(f"  -> Semantic Energy Retained: {energy_embed * 100:.2f}%")
    
    scale_factor = 0.035 / (torch.std(embed_projected).item() + 1e-8)
    aligned_embed = (embed_projected * scale_factor).cpu()
    cerebro_params["sensory.embedding.weight"] = aligned_embed
    stats["embed_energy_retained"] = energy_embed
    
    # -------------------------------------------------------------
    # 2. Afferent Sensory SwiGLU Front-End
    # -------------------------------------------------------------
    print(f"\n[2/5] Projecting Afferent Sensory SwiGLU Front-End...")
    gate_w = src_dict["model.layers.0.mlp.gate_proj.weight"].to(device=device, dtype=torch.float32)  # (d_inter, d_src)
    up_w = src_dict["model.layers.0.mlp.up_proj.weight"].to(device=device, dtype=torch.float32)      # (d_inter, d_src)
    down_w = src_dict["model.layers.0.mlp.down_proj.weight"].to(device=device, dtype=torch.float32)  # (d_src, d_inter)
    
    target_inter = target_dim * 2
    if d_inter >= target_inter:
        # SVD on gate_w on GPU to get top target_inter left singular vectors
        u_inter, s_inter, _ = torch.linalg.svd(gate_w, full_matrices=False)
        P_inter = u_inter[:, :target_inter]  # (d_inter, target_inter)
        energy_inter = float(torch.sum(s_inter[:target_inter]**2) / max(torch.sum(s_inter**2), 1e-12))
    else:
        P_inter = torch.eye(d_inter, target_inter, device=device)
        energy_inter = 1.0
        
    print(f"  -> Intermediate MLP Subspace Projection ({d_inter} -> {target_inter})")
    print(f"  -> Intermediate Energy Retained: {energy_inter * 100:.2f}%")
    
    fc1_gate = P_inter.T @ gate_w @ P_embed
    fc1_val = P_inter.T @ up_w @ P_embed
    fc2 = P_embed.T @ down_w @ P_inter
    
    cerebro_params["sensory.fc1_gate.weight"] = fc1_gate.cpu()
    cerebro_params["sensory.fc1_val.weight"] = fc1_val.cpu()
    cerebro_params["sensory.fc2.weight"] = fc2.cpu()
    
    if "model.layers.0.input_layernorm.weight" in src_dict:
        norm_w = src_dict["model.layers.0.input_layernorm.weight"].to(device=device, dtype=torch.float32)
        P_norm = torch.abs(P_embed).T @ norm_w
        cerebro_params["sensory.norm_out.weight"] = P_norm.cpu()
        
    stats["mlp_energy_retained"] = energy_inter
    
    # -------------------------------------------------------------
    # 3. Arcuate Fasciculus Attention Projections
    # -------------------------------------------------------------
    print(f"\n[3/5] Projecting Arcuate Fasciculus Attention KV Projections...")
    q_w = src_dict["model.layers.0.self_attn.q_proj.weight"].to(device=device, dtype=torch.float32)
    k_w = src_dict["model.layers.0.self_attn.k_proj.weight"].to(device=device, dtype=torch.float32)
    v_w = src_dict["model.layers.0.self_attn.v_proj.weight"].to(device=device, dtype=torch.float32)
    
    cerebro_params["motor.q_proj.weight"] = (P_embed.T @ q_w @ P_embed).cpu()
    cerebro_params["motor.k_proj.weight"] = (P_embed.T @ k_w @ P_embed).cpu()
    cerebro_params["motor.v_proj.weight"] = (P_embed.T @ v_w @ P_embed).cpu()
    print(f"  -> Projected Q, K, V matrices onto ({target_dim} x {target_dim}) manifold")
    
    # -------------------------------------------------------------
    # 4. Efferent Motor Decoder (Broca's Area Head)
    # -------------------------------------------------------------
    print(f"\n[4/5] Projecting Efferent Motor Softmax Decoder Head...")
    lm_head = src_dict["lm_head.weight"].to(device=device, dtype=torch.float32)
    head_proj_base = lm_head @ P_embed  # (32000, 640)
    
    head_1280 = torch.cat([head_proj_base * 0.707, head_proj_base * 0.707], dim=-1)
    scale_head = (1.0 / math.sqrt(target_dim * 2)) / (torch.std(head_1280).item() + 1e-8)
    cerebro_params["motor.head.weight"] = (head_1280 * scale_head * 0.8).cpu()
    print(f"  -> Synthesized dual-channel efferent head (32000 x {target_inter})")
    
    # -------------------------------------------------------------
    # 5. Volumetric Core Stability Check
    # -------------------------------------------------------------
    print(f"\n[5/5] Enforcing Contractive Spectral Bounds on Volumetric Core...")
    intra = cerebro_params["core.intra_shell"]
    for k in range(intra.shape[0]):
        mat = intra[k].to(device=device, dtype=torch.float32)
        s = torch.linalg.svdvals(mat)
        max_s = s[0].item()
        if max_s > 0.98:
            intra[k] = (mat * (0.95 / max_s)).cpu()
    print(f"  -> Volumetric Core contractive bounds verified (rho <= 0.98)")
    
    print(f"\n[+] Manifold Grafting Complete! All sensory, lexical, and projection manifolds aligned.")
    return cerebro_params, stats


def evaluate_cerebro_loss(params: Dict[str, torch.Tensor], tokenizer, device: str = "cpu") -> float:
    """Evaluates cross-entropy loss on standard validation sentences."""
    from scripts.eval_latest_checkpoint_loss import BitwiseJAXEngine
    
    engine = BitwiseJAXEngine(params, device=device)
    sentences = [
        "The fundamental law of calculus connects derivatives with integrals.",
        "Deep learning models compress high dimensional data onto low dimensional manifolds.",
        "To solve the equation three x minus nine equals zero, we isolate the variable.",
        "Artificial intelligence architectures mimic cortical microcircuits in the human brain.",
    ]
    
    total_loss = 0.0
    total_tokens = 0
    
    for sent in sentences:
        tokens = tokenizer.encode(sent, return_tensors="pt").to(device)
        if tokens.shape[1] < 4:
            continue
        
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        
        sens, (keys, vals) = engine.encode_sensory(inputs)
        B, S, D = sens.shape
        K = 16
        state = torch.zeros(B, K, D, device=device)
        plan = torch.zeros(B, D, device=device)
        refractory = torch.zeros(B, tokenizer.vocab_size, device=device)
        
        logits_list = []
        for t in range(S):
            state = engine.deliberate_step(state + sens[:, t:t+1, :].expand(-1, K, -1) * 0.2)
            logits, state, refractory, plan = engine.emit_step(
                state, (keys[:, :t+1], vals[:, :t+1]), plan, refractory, step_idx=t
            )
            logits_list.append(logits.unsqueeze(1))
            
        all_logits = torch.cat(logits_list, dim=1)
        loss = F.cross_entropy(all_logits.view(-1, tokenizer.vocab_size), targets.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += targets.numel()
        
    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser(description="Manifold-Accelerated Weight Grafting Engine")
    parser.add_argument("--source_model", type=str, default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
                        help="Pretrained LLaMA model sharing 32k vocab (e.g. TinyLlama-1.1B or JackFram/llama-160m)")
    parser.add_argument("--checkpoint_in", type=str, default="",
                        help="Optional existing Cerebro checkpoint to graft into")
    parser.add_argument("--output_pt", type=str, default="checkpoints/cerebro_v4_tinyllama_aligned.pt",
                        help="Output path for PyTorch checkpoint")
    parser.add_argument("--output_jax", type=str, default="checkpoints/cerebro_v4_tinyllama_aligned_jax.pkl",
                        help="Output path for JAX pickle checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Compute device (cuda or cpu)")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output_pt)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_jax)), exist_ok=True)
    
    from src.corpus import build_tokenizer
    tokenizer = build_tokenizer("llama")
    
    if args.checkpoint_in and os.path.exists(args.checkpoint_in):
        print(f"[*] Loading base checkpoint: {args.checkpoint_in}")
        with open(args.checkpoint_in, "rb") as f:
            base_params = pickle.load(f)
        if "params" in base_params:
            base_params = base_params["params"]
        base_params = {k: torch.from_numpy(np.array(v)).float() if not isinstance(v, torch.Tensor) else v for k, v in base_params.items()}
    else:
        print("[*] Initializing fresh CerebroSpheroid V4 base architecture...")
        base_params = init_default_cerebro_params()
        
    print("[*] Evaluating baseline loss before manifold grafting...")
    base_loss = evaluate_cerebro_loss(base_params, tokenizer, device=args.device)
    base_ppl = math.exp(min(base_loss, 20.0))
    print(f"  -> Baseline Random Initialization Loss: {base_loss:.4f} (Perplexity: {base_ppl:.1f})")
    
    t0 = time.time()
    grafted_params, stats = graft_manifold_weights(base_params, source_model_name=args.source_model, device=args.device)
    elapsed = time.time() - t0
    
    print(f"\n[*] Evaluating loss after manifold grafting...")
    grafted_loss = evaluate_cerebro_loss(grafted_params, tokenizer, device=args.device)
    grafted_ppl = math.exp(min(grafted_loss, 20.0))
    print(f"  -> Grafted Manifold Loss: {grafted_loss:.4f} (Perplexity: {grafted_ppl:.1f})")
    
    loss_reduction = base_loss - grafted_loss
    print(f"\n=================================================================")
    print(f"  RESULTS SUMMARY: 1.1B / 2B MANIFOLD TRANSFER")
    print(f"=================================================================")
    print(f"  Initial Loss:       {base_loss:.4f}  (PPL: {base_ppl:.1f})")
    print(f"  Grafted Loss:       {grafted_loss:.4f}  (PPL: {grafted_ppl:.1f})")
    print(f"  Loss Improvement:   -{loss_reduction:.4f} points instantly!")
    print(f"  Elapsed Time:       {elapsed:.2f} seconds")
    print(f"=================================================================")
    
    print(f"\n[*] Saving PyTorch checkpoint to: {args.output_pt}")
    torch.save(grafted_params, args.output_pt)
    
    print(f"[*] Saving JAX checkpoint to: {args.output_jax}")
    jax_params = {k: np.array(v.cpu().numpy(), dtype=np.float32) for k, v in grafted_params.items()}
    jax_checkpoint = {
        "params": jax_params,
        "step": 0,
        "tokens_seen": 0,
        "loss": grafted_loss,
        "source": f"manifold_grafted_from_{args.source_model}"
    }
    with open(args.output_jax, "wb") as f:
        pickle.dump(jax_checkpoint, f)
        
    print(f"[+] Done! Checkpoints successfully generated.")


if __name__ == "__main__":
    main()
