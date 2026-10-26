"""Guards the auditory-feedback channel that windowed training depends on.

Run directly: python3 training/test_feedback_channel.py

The channel has to satisfy three things for `train_v3_full.py` to be sound:

1. Passing the shifted targets explicitly must equal letting the model build the shift.
2. The refactored loop must reproduce the pre-refactor loop on the plain `targets` path, so
   the older training script keeps producing the same numbers.
3. A window's first scored position must actually hear the token before the window, and that
   token must not enter the loss. Position 0's logits are therefore sensitive to the context,
   and so is everything after it, because the manifold state is recurrent.

The last print is a record, not an assertion: the context propagates, which is intended.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.spheroid_v3 import build_spheroid_v3


def make(seed=0):
    torch.manual_seed(seed)
    m = build_spheroid_v3("test")
    m.eval()
    return m


EOS = 1
B, P, R = 2, 6, 7
prompt = torch.randint(5, 900, (B, P))
targets = torch.randint(5, 900, (B, R))
targets[1, -2:] = -100

m = make()
with torch.no_grad():
    torch.manual_seed(99)
    a = m.training_logits(prompt, R, thinking_cycles=1, pad_token_id=EOS, targets=targets)

    torch.manual_seed(99)
    shift = torch.cat([targets.new_full((B, 1), -100), targets[:, :-1]], dim=1)
    b = m.training_logits(prompt, R, thinking_cycles=1, pad_token_id=EOS,
                          targets=targets, feedback_tokens=shift)
print("implicit vs explicit shift  max|diff| =", (a - b).abs().max().item())

# Replica of the pre-refactor loop, to prove the default path did not move.
with torch.no_grad():
    torch.manual_seed(99)
    memory, gist, sensory_kv = m.encode_sensory(prompt, pad_token_id=EOS)
    state = m._integrate_memory(memory, max_packets=32, lengths=m._lengths(prompt, EOS))
    state = m.volumetric_core.inject_efference_copy(state, m._gist_flux(gist))
    state = m.deliberate(state, thinking_cycles=1)
    plan = m.motor_dnn.init_plan(B, state.device)
    refractory = torch.zeros(B, m.vocab_size, device=state.device)
    old = []
    fb = None
    for step in range(R):
        if step > 0:
            prev = targets[:, step - 1].clamp(min=0)
            rep = m.sensory_dnn.embedding(prev)
            mask = (targets[:, step - 1] != -100).unsqueeze(-1)
            fb = rep * mask
            if sensory_kv is not None and hasattr(m.motor_dnn, "prime_sensory"):
                nk, nv = m.motor_dnn.prime_sensory(fb.unsqueeze(1))
                sensory_kv = (torch.cat([sensory_kv[0], nk], 1), torch.cat([sensory_kv[1], nv], 1))
        lg, state, refractory, plan = m.emit_step(
            state, sensory_kv=sensory_kv, plan=plan, refractory_trace=refractory,
            step_idx=step, auditory_feedback=fb)
        old.append(lg)
    old = torch.stack(old, 1)
print("refactor vs pre-refactor    max|diff| =", (a - old).abs().max().item())

# A feedback tensor that is entirely padding must collapse to the no-feedback path.
blank = torch.full((B, R), -100)
with torch.no_grad():
    torch.manual_seed(99)
    d = m.training_logits(prompt, R, thinking_cycles=1, pad_token_id=EOS, targets=blank)
    torch.manual_seed(99)
    e = m.training_logits(prompt, R, thinking_cycles=1, pad_token_id=EOS,
                          targets=blank, feedback_tokens=blank)
print("all-padding feedback vs none max|diff| =", (d - e).abs().max().item())

# The first scored position must actually hear its context token.
with torch.no_grad():
    torch.manual_seed(99)
    ctx = torch.randint(5, 900, (B, 1))
    with_ctx = m.training_logits(prompt, R, thinking_cycles=1, pad_token_id=EOS,
                                 targets=targets, feedback_tokens=torch.cat([ctx, targets[:, :-1]], 1))
print("context changes step-0 logits   max|diff| =", (a[:, 0] - with_ctx[:, 0]).abs().max().item())
print("context propagates to step 1+   max|diff| =", (a[:, 1:] - with_ctx[:, 1:]).abs().max().item())

assert (a - b).abs().max().item() == 0.0, "explicit shift must equal the implicit one"
assert (a - old).abs().max().item() == 0.0, "refactor changed the plain targets path"
assert (d - e).abs().max().item() == 0.0, "all-padding feedback must equal no feedback"
assert (a[:, 0] - with_ctx[:, 0]).abs().max().item() > 0.0, "context never reached the decoder"
print("\nOK: feedback channel is aligned and position 0 hears its context")
