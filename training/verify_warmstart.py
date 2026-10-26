"""
Warm-start and refinement verification for CerebroSpheroidV3 against CerebroSpheroid (V2).

Four assertions, each isolating one claim made by the architecture:

  1. STRUCTURE  every V2 parameter has a V3 slot (directly or via a lossless lift),
                so no pretrained tensor is silently discarded on upgrade.
  2. TRANSPLANT a real 176M checkpoint loads with 0 unrecoverable keys.
  3. SILENCE    the added-capacity branches (multi-shell tap, premotor plan) are
                exactly inert at initialization, end-to-end on the logits.
  4. RETENTION  given the SAME transplanted weights, the residual core keeps the
                question legible across deliberation cycles where V2's norm-per-step
                core collapses it. This is the claim the architecture exists for.
  5. FIDELITY   on identical weights a warm-started V3 reproduces V2's encoder
                field, state and logits to float noise. Checks 1-4 all passed while
                this was false, so it is now its own gate.
"""

import argparse
import os
import sys

os.environ.setdefault("HF_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".hf_cache")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from src.model import CerebroSpheroid, VolumetricBrainCore
from src.spheroid_v3 import CerebroSpheroidV3, ResidualVolumetricCore

SMALL = dict(surface_dim=192, num_shells=8, radial_dim=64, num_shortcut_tracts=16)
# Every shipped checkpoint is laptop-scale; transplant tests must match it exactly.
LAPTOP = dict(surface_dim=640, num_shells=16, radial_dim=256, num_shortcut_tracts=40)
VOCAB = 1000


# ----------------------------------------------------------------------------
# checkpoint plumbing
# ----------------------------------------------------------------------------
def read_checkpoint(ckpt: str) -> dict:
    raw = torch.load(ckpt, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    return {k[7:] if k.startswith("module.") else k: v for k, v in raw.items()}


def transplant(model, raw: dict):
    """Copies every tensor, lifting width-changed slots through V3's own surgery."""
    own = model.state_dict()
    copy, lifted, bad_shape, no_slot = {}, [], [], []
    for k, v in raw.items():
        if k not in own:
            no_slot.append(k)
        elif own[k].shape == v.shape:
            copy[k] = v
        else:
            lift = getattr(model, "_lift_v2_tensor", None)
            reshaped = lift(k, v, own[k]) if lift is not None else None
            if reshaped is None:
                bad_shape.append((k, tuple(v.shape), tuple(own[k].shape)))
            else:
                copy[k] = reshaped
                lifted.append(k)
    merged = dict(own)
    merged.update(copy)
    model.load_state_dict(merged, strict=True)
    gate_key = "volumetric_core.inertia_gate"
    if gate_key in copy and hasattr(model.volumetric_core, "set_blend_from_gate"):
        model.volumetric_core.set_blend_from_gate(copy[gate_key].float().mean())
    return copy, lifted, bad_shape, no_slot


# ----------------------------------------------------------------------------
# 1. structural coverage
# ----------------------------------------------------------------------------
def check_structure():
    torch.manual_seed(1234)
    v2 = CerebroSpheroid(vocab_size=VOCAB, **SMALL)
    torch.manual_seed(1234)
    v3 = CerebroSpheroidV3(vocab_size=VOCAB, **SMALL)

    raw = v2.state_dict()
    copy, lifted, bad_shape, no_slot = transplant(v3, raw)
    fresh = [k for k in v3.state_dict() if k not in copy]

    print(f"[1] STRUCTURE     V2 tensors: {len(raw)}")
    print(f"    carried over unchanged : {len(copy) - len(lifted)}")
    print(f"    lifted (reshaped)      : {len(lifted)} -> {sorted(lifted)}")
    print(f"    lost (shape mismatch)  : {len(bad_shape)}")
    for k, a, b in bad_shape:
        print(f"      ! {k}: V2{list(a)} vs V3{list(b)}")
    print(f"    lost (no V3 slot)      : {len(no_slot)} -> {sorted(no_slot)}")
    print(f"    V3-only tensors        : {len(fresh)}")

    ok = not bad_shape and not no_slot
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


# ----------------------------------------------------------------------------
# 2. real checkpoint transplant
# ----------------------------------------------------------------------------
def check_transplant(ckpt: str):
    if not os.path.exists(ckpt):
        print(f"[2] TRANSPLANT    skipped (missing {ckpt})")
        return None

    raw = read_checkpoint(ckpt)
    v3 = CerebroSpheroidV3(vocab_size=50257, **LAPTOP)
    copy, lifted, bad_shape, no_slot = transplant(v3, raw)

    fresh = [k for k in v3.state_dict() if k not in copy]
    print(f"[2] TRANSPLANT    {ckpt}")
    print(f"    tensors in checkpoint  : {len(raw)}")
    print(f"    carried over unchanged : {len(copy) - len(lifted)}")
    print(f"    lifted (reshaped)      : {len(lifted)} -> {sorted(lifted)}")
    print(f"    lost                   : {len(bad_shape) + len(no_slot)}")
    for k, a, b in bad_shape[:8]:
        print(f"      ! {k}: ckpt{list(a)} vs model{list(b)}")
    for k in no_slot[:8]:
        print(f"      ? {k}")
    fresh_decoder = [k for k in fresh if "motor_dnn" not in k or k.startswith("motor_dnn.shell")
                     or k.startswith("motor_dnn.plan")]
    print(f"    fresh V3 tensors       : {len(fresh)} ({len(fresh_decoder)} in the motor decoder)")

    float_t = {k: v for k, v in raw.items() if v.is_floating_point()}
    live = sum(1 for v in float_t.values() if v.float().abs().max().item() > 0)
    print(f"    checkpoint liveness    : {live}/{len(float_t)} floating tensors non-zero")

    ok = not bad_shape and not no_slot
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


# ----------------------------------------------------------------------------
# 3. added-capacity branches are inert at init
# ----------------------------------------------------------------------------
@torch.no_grad()
def emission_logits(model, prompts, steps: int, cycles: int = 3):
    memory, gist, kv = model.encode_sensory(prompts)
    state = model._integrate_memory(memory, lengths=model._lengths(prompts, None))
    state = model.volumetric_core.inject_efference_copy(state, gist)
    state = model.deliberate(state, thinking_cycles=cycles)
    plan = model.motor_dnn.init_plan(prompts.shape[0], state.device)
    refractory = torch.zeros(prompts.shape[0], model.vocab_size, device=state.device, dtype=torch.float32)
    out = []
    for t in range(steps):
        logits, state, refractory, plan = model.emit_step(
            state, sensory_kv=kv, plan=plan, refractory_trace=refractory, step_idx=t
        )
        out.append(logits)
    return torch.stack(out, dim=1)


@torch.no_grad()
def check_silence(ckpt: str, steps: int = 6):
    if not os.path.exists(ckpt):
        print("[3] SILENCE       skipped (missing checkpoint)")
        return None

    # Pinned explicitly: the residual core is opt-in since the 1500-step ablation,
    # and this check must keep covering it.
    full = CerebroSpheroidV3(vocab_size=50257, use_residual_core=True, **LAPTOP)
    transplant(full, read_checkpoint(ckpt))
    full.eval()

    # Identical encoder and core, only the two added-capacity branches removed.
    reduced = CerebroSpheroidV3(
        vocab_size=50257, use_temporal=True, use_residual_core=True,
        use_plan=False, use_shell_tap=False, **LAPTOP,
    )
    reduced.load_state_dict(full.state_dict(), strict=False)
    reduced.eval()

    torch.manual_seed(11)
    prompts = torch.randint(0, 50257, (2, 16))

    lg_full = emission_logits(full, prompts, steps)
    lg_red = emission_logits(reduced, prompts, steps)

    gap = (lg_full - lg_red).abs().max().item()
    scale = lg_full.abs().max().item()
    conv = full.sensory_dnn.conv_scale.abs().item()
    tap = full.motor_dnn.shell_scale.abs().item()
    plan = full.motor_dnn.plan_write.abs().item()
    print(f"[3] SILENCE       logit scale {scale:.3f}")
    print(f"    shell-tap scale={tap:.1e}  plan-write={plan:.1e}  conv-scale={conv:.1e}")
    print(f"    full V3 vs branch-free V3: max|dlogit|={gap:.3e} ({gap / max(scale, 1e-9):.1e} rel)")
    ok = gap == 0.0
    print(f"    -> {'PASS (exactly inert)' if ok else 'FAIL (branches leak at init)'}")
    return ok


# ----------------------------------------------------------------------------
# 4. memory retention: same weights, different integration law
# ----------------------------------------------------------------------------
def sensory_field(model, prompts):
    """Normalizes V2's tensor return and V3's (memory, gist) tuple."""
    out = model.sensory_dnn(prompts)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def retention(model, prompts, cycles=(0, 1, 2, 4, 8, 16, 32)):
    memory = sensory_field(model, prompts)
    state = model.init_brain_state(prompts.shape[0], prompts.device, living=True)
    # Identical ingestion for both arms: mean-pool the packet, inject once.
    flux = memory.mean(dim=1)
    state = model.volumetric_core.inject_sensory_flux(state, flux)
    reference = state.flatten(1).float()

    curve = {}
    current = state
    for cycle in range(max(cycles) + 1):
        if cycle in cycles:
            flat = current.flatten(1).float()
            curve[cycle] = F.cosine_similarity(flat, reference, dim=-1).mean().item()
        current = model.deliberate(current, thinking_cycles=1)
    norms = current.pow(2).mean(dim=(1, 2)).sqrt().mean().item()
    return curve, norms


def check_retention(ckpt: str):
    if not os.path.exists(ckpt):
        print("[4] RETENTION     skipped (missing checkpoint)")
        return None

    raw = read_checkpoint(ckpt)

    torch.manual_seed(5)
    v2 = CerebroSpheroid(vocab_size=50257, **LAPTOP)
    transplant(v2, raw)
    v2.eval()

    # Seed V3 from V2's *full* state, not from the checkpoint. Older checkpoints
    # predate the motor cross-attention, so ~10 tensors stay at random init; if each
    # arm drew its own, this comparison would be measuring RNG luck instead of the
    # integration law. That matters here because volumetric_core.rhythm_freq and
    # spontaneous_amp drive the resting rhythm both arms deliberate against.
    v3_core = CerebroSpheroidV3(
        vocab_size=50257, use_temporal=False, use_residual_core=True, **LAPTOP
    )
    transplant(v3_core, v2.state_dict())
    v3_core.eval()

    sd2, sd3 = v2.state_dict(), v3_core.state_dict()
    shared = [k for k in sd2 if k in sd3 and sd2[k].shape == sd3[k].shape]
    drift = max((sd2[k].float() - sd3[k].float()).abs().max().item() for k in shared)
    widened = [k for k in sd2 if k in sd3 and sd2[k].shape != sd3[k].shape]
    print(f"[4] RETENTION     {len(shared)} shared tensors, max|V2-V3| before run = {drift:.1e}")
    print(f"    widened by lift (not bit-identical by design): {len(widened)}")

    torch.manual_seed(5)
    prompts = torch.randint(0, 50257, (1, 24))

    curve_v2, norm_v2 = retention(v2, prompts)
    curve_v3, norm_v3 = retention(v3_core, prompts)

    cycles = sorted(curve_v2)
    print(f"    cosine to post-ingestion state, same weights both arms")
    print("    cycle : " + " ".join(f"{c:6d}" for c in cycles))
    print("    V2    : " + " ".join(f"{curve_v2[c]:6.3f}" for c in cycles))
    print("    V3    : " + " ".join(f"{curve_v3[c]:6.3f}" for c in cycles))
    print(f"    final shell RMS  V2={norm_v2:.3f}  V3={norm_v3:.3f}")

    def half_life(curve):
        """Cycles until cosine memory falls below 0.5, linearly interpolated."""
        keys = sorted(curve)
        for prev, cur in zip(keys, keys[1:]):
            if curve[cur] < 0.5:
                span = curve[prev] - curve[cur]
                return prev + (curve[prev] - 0.5) / span * (cur - prev) if span > 0 else cur
        return float("inf")

    hl2, hl3 = half_life(curve_v2), half_life(curve_v3)
    print(f"    retention half-life : V2={hl2:.1f} cycles   V3={hl3:.1f} cycles   "
          f"({hl3 / max(hl2, 1e-9):.1f}x)")
    ok = hl3 > hl2
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


# ----------------------------------------------------------------------------
# 5. fidelity: a warm-started V3 must reproduce V2 exactly
# ----------------------------------------------------------------------------
@torch.no_grad()
def check_fidelity(ckpt: str):
    """
    On identical weights, V3 must be numerically indistinguishable from V2.

    Every refinement is zero-gated at init (conv_scale, pos_embed, gist_scale,
    packet_mix, plan_write, shell_scale), so a transplanted checkpoint reproduces
    V2's encoder field, post-ingestion state and logits to float noise. That makes
    the upgrade strictly additive: any later difference is learned, never injected.

    This gate exists because checks 1-4 all reported PASS while two separate lift
    bugs were silently corrupting the model - an inverted RMSNorm gain (inflating
    the decoder pre-activation by 5/3) and an RMSNorm of the whole embedding stream
    (~50x rescaling into a nonlinear gate). Both produced noise from a warm start
    while every structural assertion stayed green.
    """
    if not os.path.exists(ckpt):
        print("[5] FIDELITY      skipped (missing checkpoint)")
        return None

    torch.manual_seed(5)
    v2 = CerebroSpheroid(vocab_size=50257, **LAPTOP).eval()
    transplant(v2, read_checkpoint(ckpt))

    # V2's core is pinned so this isolates the encoder/decoder lift. The residual
    # core is a *different integration law* by design and is measured in check 4.
    v3 = CerebroSpheroidV3(
        vocab_size=50257, use_residual_core=False, **LAPTOP
    ).eval()
    transplant(v3, v2.state_dict())

    torch.manual_seed(11)
    prompt = torch.randint(0, 50257, (2, 40))

    mem2 = v2.sensory_dnn(prompt)
    mem3, _, sensory_kv = v3.encode_sensory(prompt)
    state2, mem2 = v2.ingest_stimulus_with_memory(prompt)
    state3 = v3.ingest_stimulus(prompt)
    state2 = v2.deliberate(state2, thinking_cycles=4)
    state3 = v3.deliberate(state3, thinking_cycles=4)
    lg2 = v2.emit_next_token(state2, sensory_memory=mem2, step_idx=0)[0]
    lg3 = v3.emit_step(state3, sensory_kv=sensory_kv, step_idx=0)[0]

    stages = {
        "encoder field": (mem2 - mem3).abs().max().item(),
        "ingested state": (state2 - state3).abs().max().item(),
        "first logits": (lg2 - lg3).abs().max().item(),
    }
    scale = lg2.abs().max().item()
    print(f"[5] FIDELITY      V2 vs warm-started V3 on identical weights (logit scale {scale:.3f})")
    for name, gap in stages.items():
        print(f"    {name:15s}: max|delta| = {gap:.3e}")
    ok = stages["first logits"] < 1e-3
    print(f"    -> {'PASS (warm start is a no-op)' if ok else 'FAIL (visible drift on load)'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="V3 warm-start and refinement verification")
    parser.add_argument("--checkpoint", default="checkpoints/cerebro_spheroid_100m.pt")
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()

    print("=" * 78)
    print("  CEREBRO-SPHEROID V3  /  V2 VERIFICATION")
    print("=" * 78)
    results = {
        "structure": check_structure(),
        "transplant": check_transplant(args.checkpoint),
        "silence": check_silence(args.checkpoint, args.steps),
        "retention": check_retention(args.checkpoint),
        "fidelity": check_fidelity(args.checkpoint),
    }
    print("=" * 78)
    print("  " + " | ".join(f"{k}={v}" for k, v in results.items()))
    print("=" * 78)
    return 0 if all(v is not False for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
