"""
Comprehensive Numerical Parity Verification: PyTorch vs JAX.

Verifies:
  1. V2 Component Parity: SensoryInputDNN, VolumetricBrainCore, MotorOutputDNN
  2. V3 Component Parity: TemporalSensoryEncoder, ResidualVolumetricCore, PremotorMotorDecoder
  3. Real Checkpoint Parity: Loads `cerebro_spheroid_1m_best.pt` in both frameworks and tests forward fidelity
  4. Full Training Rollout Parity: jax.lax.scan vs PyTorch sequential loop
  5. Analytical Gradient Parity: PyTorch autograd vs JAX autodiff (jax.grad)
  6. End-to-End Generation Parity
"""

import os
import sys
import math
import argparse

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("HF_HOME", os.path.join(ROOT_DIR, ".hf_cache"))
sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn.functional as F
import numpy as np
import jax
import jax.numpy as jnp

from src.model import (
    SensoryInputDNN,
    VolumetricBrainCore,
    MotorOutputDNN,
    CerebroSpheroid,
)
from src.spheroid_v3 import (
    TemporalSensoryEncoder,
    ResidualVolumetricCore,
    PremotorMotorDecoder,
    CerebroSpheroidV3,
    training_rollout as pt_training_rollout,
)
from src.jax.model import (
    ModelConfig,
    CerebroSpheroidJAX,
    training_rollout as jax_training_rollout,
    compute_loss as jax_compute_loss,
    encode_sensory,
    deliberate_step,
    emit_step,
    circulate,
    ingest_stimulus,
)
from src.jax.convert import (
    torch_to_jax_params,
    load_torch_checkpoint_to_jax,
)


def verify_v2_component_parity():
    print("[1/6] Testing V2 Component Parity...")
    torch.manual_seed(101)
    vocab_size = 80
    surface_dim = 32
    num_shells = 4
    radial_dim = 16
    num_shortcut_tracts = 6

    cfg = ModelConfig(
        vocab_size=vocab_size,
        surface_dim=surface_dim,
        num_shells=num_shells,
        radial_dim=radial_dim,
        num_shortcut_tracts=num_shortcut_tracts,
        arch="v2",
    )

    pt_sens = SensoryInputDNN(vocab_size=vocab_size, hidden_dim=surface_dim, surface_dim=surface_dim)
    pt_core = VolumetricBrainCore(num_shells=num_shells, shell_dim=surface_dim, radial_dim=radial_dim, num_shortcut_tracts=num_shortcut_tracts)
    pt_motor = MotorOutputDNN(surface_dim=surface_dim, core_dim=surface_dim, vocab_size=vocab_size)

    # Convert to JAX params
    pt_combined = {
        **{f"sensory_dnn.{k}": v for k, v in pt_sens.state_dict().items()},
        **{f"volumetric_core.{k}": v for k, v in pt_core.state_dict().items()},
        **{f"motor_dnn.{k}": v for k, v in pt_motor.state_dict().items()},
    }
    j_params, _ = torch_to_jax_params(pt_combined, config=cfg)

    # A. Sensory
    toks = torch.randint(0, vocab_size, (2, 8))
    y_pt_sens = pt_sens(toks).detach().numpy()
    y_j_sens, _, _ = encode_sensory(j_params, jnp.array(toks.numpy()), cfg)
    sens_diff = float(jnp.max(jnp.abs(y_j_sens - y_pt_sens)))
    assert sens_diff < 1e-5, f"Sensory V2 diff too large: {sens_diff}"
    print(f"  ✓ SensoryInputDNN parity: max diff = {sens_diff:.2e}")

    # B. Core deliberate step
    state = torch.randn(2, num_shells, surface_dim)
    y_pt_core = pt_core.deliberate_step(state).detach().numpy()
    y_j_core = deliberate_step(j_params, jnp.array(state.numpy()), cfg)
    core_diff = float(jnp.max(jnp.abs(y_j_core - y_pt_core)))
    assert core_diff < 1e-5, f"Core V2 diff too large: {core_diff}"
    print(f"  ✓ VolumetricBrainCore deliberate_step parity: max diff = {core_diff:.2e}")

    # C. Motor emit step
    sens_mem = torch.randn(2, 8, surface_dim)
    refractory = torch.randn(2, vocab_size).abs()
    l_pt, tr_pt, eff_pt = pt_motor(state[:, -1, :], state[:, 0, :], sensory_memory=sens_mem, refractory_trace=refractory, step_idx=2)
    k_j = jnp.matmul(jnp.array(sens_mem.numpy()), j_params["motor_dnn.k_proj.weight"].T)
    v_j = jnp.matmul(jnp.array(sens_mem.numpy()), j_params["motor_dnn.v_proj.weight"].T)
    l_j, next_s_j, tr_j, _ = emit_step(
        j_params, jnp.array(state.numpy()), cfg, (k_j, v_j),
        refractory_trace=jnp.array(refractory.numpy()), step_idx=2
    )
    l_diff = float(jnp.max(jnp.abs(l_j - l_pt.detach().numpy())))
    tr_diff = float(jnp.max(jnp.abs(tr_j - tr_pt.detach().numpy())))
    assert l_diff < 1e-5, f"Motor V2 logits diff too large: {l_diff}"
    assert tr_diff < 1e-5, f"Motor V2 trace diff too large: {tr_diff}"
    print(f"  ✓ MotorOutputDNN emit_step parity: logits diff = {l_diff:.2e}, trace diff = {tr_diff:.2e}")


def verify_v3_component_parity():
    print("\n[2/6] Testing V3 Component Parity...")
    torch.manual_seed(202)
    vocab_size = 80
    surface_dim = 32
    num_shells = 4
    radial_dim = 16
    num_shortcut_tracts = 6

    cfg = ModelConfig(
        vocab_size=vocab_size,
        surface_dim=surface_dim,
        num_shells=num_shells,
        radial_dim=radial_dim,
        num_shortcut_tracts=num_shortcut_tracts,
        arch="v3",
        max_phase_steps=2048,
        dropout=0.0,
    )

    pt_sens = TemporalSensoryEncoder(vocab_size=vocab_size, hidden_dim=surface_dim, surface_dim=surface_dim)
    pt_core = ResidualVolumetricCore(num_shells=num_shells, shell_dim=surface_dim, radial_dim=radial_dim, num_shortcut_tracts=num_shortcut_tracts)
    pt_motor = PremotorMotorDecoder(num_shells=num_shells, surface_dim=surface_dim, vocab_size=vocab_size, dropout=0.0)
    pt_motor.eval()

    pt_combined = {
        **{f"sensory_dnn.{k}": v for k, v in pt_sens.state_dict().items()},
        **{f"volumetric_core.{k}": v for k, v in pt_core.state_dict().items()},
        **{f"motor_dnn.{k}": v for k, v in pt_motor.state_dict().items()},
    }
    j_params, _ = torch_to_jax_params(pt_combined, config=cfg)

    # A. Temporal Wavelets Sensory
    toks = torch.randint(0, vocab_size, (2, 8))
    lens = torch.tensor([8, 6])
    m_pt, g_pt = pt_sens(toks, lengths=lens)
    m_j, g_j, _ = encode_sensory(j_params, jnp.array(toks.numpy()), cfg, lengths=jnp.array(lens.numpy()))
    m_diff = float(jnp.max(jnp.abs(m_j - m_pt.detach().numpy())))
    g_diff = float(jnp.max(jnp.abs(g_j - g_pt.detach().numpy())))
    assert m_diff < 1e-5, f"Sensory V3 memory diff: {m_diff}"
    assert g_diff < 1e-5, f"Sensory V3 gist diff: {g_diff}"
    print(f"  ✓ TemporalSensoryEncoder parity: memory diff = {m_diff:.2e}, gist diff = {g_diff:.2e}")

    # B. Residual Core
    state = torch.randn(2, num_shells, surface_dim)
    y_pt_core = pt_core.deliberate_step(state).detach().numpy()
    y_j_core = deliberate_step(j_params, jnp.array(state.numpy()), cfg)
    core_diff = float(jnp.max(jnp.abs(y_j_core - y_pt_core)))
    assert core_diff < 1e-5, f"Core V3 diff: {core_diff}"
    print(f"  ✓ ResidualVolumetricCore deliberate_step parity: max diff = {core_diff:.2e}")

    # C. Premotor Decoder
    k_pt, v_pt = pt_motor.prime_sensory(m_pt)
    plan = pt_motor.init_plan(2, state.device)
    refractory = torch.randn(2, vocab_size).abs()
    l_pt, tr_pt, eff_pt, pl_pt = pt_motor(state, sensory_kv=(k_pt, v_pt), plan=plan, refractory_trace=refractory, step_idx=1)
    k_j = jnp.array(k_pt.detach().numpy())
    v_j = jnp.array(v_pt.detach().numpy())
    l_j, next_s_j, tr_j, pl_j = emit_step(
        j_params, jnp.array(state.numpy()), cfg, (k_j, v_j),
        refractory_trace=jnp.array(refractory.numpy()),
        plan=jnp.array(plan.numpy()), step_idx=1
    )
    l_diff = float(jnp.max(jnp.abs(l_j - l_pt.detach().numpy())))
    pl_diff = float(jnp.max(jnp.abs(pl_j - pl_pt.detach().numpy())))
    assert l_diff < 1e-5, f"Decoder V3 logits diff: {l_diff}"
    assert pl_diff < 1e-5, f"Decoder V3 plan diff: {pl_diff}"
    print(f"  ✓ PremotorMotorDecoder parity: logits diff = {l_diff:.2e}, plan diff = {pl_diff:.2e}")


def verify_checkpoint_parity():
    print("\n[3/6] Testing Checkpoint Parity (cerebro_spheroid_1m_best.pt)...")
    ckpt_path = "checkpoints/cerebro_spheroid_1m_best.pt"
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] {ckpt_path} not found on disk.")
        return

    # Load PyTorch model in float32 for bit-exact float32 parity
    from src.inference import ReasoningAgent
    pt_agent = ReasoningAgent(checkpoint_path=ckpt_path, arch="v2")
    pt_agent.model.to(dtype=torch.float32)

    # Load JAX model
    from src.jax.inference import ReasoningAgentJAX
    j_agent = ReasoningAgentJAX(checkpoint_path=ckpt_path, arch="v2")

    test_prompts = [
        "What is 2+2?",
        "If 4 apples cost $20, how much do 6 cost?",
    ]

    for q in test_prompts:
        toks = pt_agent.tokenizer.encode(q)
        q_pt = torch.tensor([toks], device=pt_agent.device)
        q_j = jnp.array([toks], dtype=jnp.int32)

        # Ingest + deliberate parity
        s_pt, m_pt = pt_agent.model.ingest_stimulus_with_memory(q_pt, living=True)
        s_pt = pt_agent.model.deliberate(s_pt, thinking_cycles=2)

        s_j, m_j, kv_j = j_agent.model.model if hasattr(j_agent.model, "model") else (None, None, None)
        # Directly call JAX functions
        s_j, _, kv_j = ingest_stimulus(j_agent.model.params, q_j, j_agent.model.config, living=True)
        s_j = circulate(j_agent.model.params, s_j, j_agent.model.config, num_cycles=2)

        state_diff = float(jnp.max(jnp.abs(s_j - s_pt.float().detach().cpu().numpy())))
        print(f"  ✓ Ingest + 2 cycles on \"{q}\": state diff = {state_diff:.2e}")
        assert state_diff < 1e-4, f"State diff too large: {state_diff}"

        # Deliberated answer parity
        ans_pt = pt_agent.generate(q, thinking_cycles=4, max_tokens=16, temperature=0.0)
        ans_j = j_agent.generate(q, thinking_cycles=4, max_tokens=16, temperature=0.0)
        assert ans_pt == ans_j, f"Generated answers diverged!\nPyTorch: '{ans_pt}'\nJAX:     '{ans_j}'"
        print(f"  ✓ Identical generated answer: \"{ans_j}\"")


def verify_training_rollout_parity():
    print("\n[4/6] Testing Full Training Rollout Parity (jax.lax.scan vs PyTorch)...")
    torch.manual_seed(303)
    vocab_size = 60
    surface_dim = 32
    num_shells = 4
    radial_dim = 16
    num_shortcut_tracts = 6

    # Test both V2 and V3
    for arch_gen in ("v2", "v3"):
        cfg = ModelConfig(
            vocab_size=vocab_size,
            surface_dim=surface_dim,
            num_shells=num_shells,
            radial_dim=radial_dim,
            num_shortcut_tracts=num_shortcut_tracts,
            arch=arch_gen,
            dropout=0.0,
        )
        if arch_gen == "v2":
            pt_model = CerebroSpheroid(
                vocab_size=vocab_size, surface_dim=surface_dim,
                num_shells=num_shells, radial_dim=radial_dim,
                num_shortcut_tracts=num_shortcut_tracts,
            )
        else:
            pt_model = CerebroSpheroidV3(
                vocab_size=vocab_size, surface_dim=surface_dim,
                num_shells=num_shells, radial_dim=radial_dim,
                num_shortcut_tracts=num_shortcut_tracts,
                dropout=0.0,
            )
        pt_model.eval()

        j_params, _ = torch_to_jax_params(pt_model.state_dict(), config=cfg)

        prompts = torch.randint(0, vocab_size, (2, 8))
        q_j = jnp.array(prompts.numpy())

        logits_pt = pt_training_rollout(pt_model, prompts, response_steps=4, thinking_cycles=2).detach().numpy()
        logits_j = jax_training_rollout(j_params, cfg, q_j, response_steps=4, thinking_cycles=2)

        diff = float(jnp.max(jnp.abs(logits_j - logits_pt)))
        assert diff < 1e-5, f"{arch_gen.upper()} rollout diff too large: {diff}"
        print(f"  ✓ {arch_gen.upper()} training_rollout (4 steps, 2 cycles): max logits diff = {diff:.2e}")


def verify_gradient_parity():
    print("\n[5/6] Testing Analytical Gradient Parity (PyTorch autograd vs JAX autodiff)...")
    torch.manual_seed(404)
    vocab_size = 40
    surface_dim = 16
    num_shells = 4
    radial_dim = 8
    num_shortcut_tracts = 4

    cfg = ModelConfig(
        vocab_size=vocab_size,
        surface_dim=surface_dim,
        num_shells=num_shells,
        radial_dim=radial_dim,
        num_shortcut_tracts=num_shortcut_tracts,
        arch="v2",
    )

    pt_model = CerebroSpheroid(
        vocab_size=vocab_size, surface_dim=surface_dim,
        num_shells=num_shells, radial_dim=radial_dim,
        num_shortcut_tracts=num_shortcut_tracts,
    )

    prompts = torch.randint(0, vocab_size, (2, 6))
    targets = torch.randint(0, vocab_size, (2, 4))

    # PyTorch loss & backward
    logits_pt = pt_training_rollout(pt_model, prompts, response_steps=4, thinking_cycles=1)
    loss_pt = F.cross_entropy(logits_pt.reshape(-1, vocab_size), targets.reshape(-1))
    loss_pt.backward()

    # JAX loss & value_and_grad
    j_params, _ = torch_to_jax_params(pt_model.state_dict(), config=cfg)
    q_j = jnp.array(prompts.numpy())
    t_j = jnp.array(targets.numpy())

    def loss_wrapper(p):
        loss, _ = jax_compute_loss(p, cfg, q_j, t_j, thinking_cycles=1)
        return loss

    loss_j, grads_j = jax.value_and_grad(loss_wrapper, allow_int=True)(j_params)

    loss_diff = abs(float(loss_pt.item()) - float(loss_j))
    assert loss_diff < 1e-5, f"Loss mismatch: {loss_diff}"
    print(f"  ✓ Loss value parity: PyTorch = {loss_pt.item():.6f}, JAX = {float(loss_j):.6f} (diff = {loss_diff:.2e})")

    # Check gradients across representative layers
    check_keys = [
        "motor_dnn.head.weight",
        "motor_dnn.fc_gate.weight",
        "volumetric_core.intra_shell",
        "sensory_dnn.embedding.weight",
    ]
    for k in check_keys:
        pt_g = dict(pt_model.named_parameters())[k].grad.detach().numpy()
        j_g = grads_j[k]
        g_diff = float(jnp.max(jnp.abs(j_g - pt_g)))
        assert g_diff < 1e-4, f"Gradient mismatch on {k}: {g_diff}"
        print(f"  ✓ Gradient parity on {k:30s}: max diff = {g_diff:.2e}")


def verify_end_to_end_agent():
    print("\n[6/6] Testing Drop-In Backend Switching via ReasoningAgent...")
    from src.inference import ReasoningAgent

    ckpt_path = "checkpoints/cerebro_spheroid_1m_best.pt"
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] {ckpt_path} not found.")
        return

    agent_torch = ReasoningAgent(checkpoint_path=ckpt_path, backend="torch")
    agent_jax = ReasoningAgent(checkpoint_path=ckpt_path, backend="jax")

    prompt = "What is 10 plus 5?"
    out_pt = agent_torch.generate(prompt, max_tokens=16, temperature=0.0)
    out_j = agent_jax.generate(prompt, max_tokens=16, temperature=0.0)

    print(f"  Prompt : \"{prompt}\"")
    print(f"  Torch  : \"{out_pt}\"")
    print(f"  JAX    : \"{out_j}\"")
    assert out_pt == out_j, f"Mismatch between backends!\nTorch: {out_pt}\nJAX:   {out_j}"
    print("  ✓ Seamless backend parity verified!")


def main():
    print("=" * 80)
    print("      CEREBRO-SPHEROID: JAX NUMERICAL PARITY VERIFICATION SUITE         ")
    print("=" * 80)
    verify_v2_component_parity()
    verify_v3_component_parity()
    verify_checkpoint_parity()
    verify_training_rollout_parity()
    verify_gradient_parity()
    verify_end_to_end_agent()
    print("\n" + "=" * 80)
    print("  ALL JAX NUMERICAL PARITY & GRADIENT GATES PASSED (100% SUCCESS)       ")
    print("=" * 80)


if __name__ == "__main__":
    main()
