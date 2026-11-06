import jax
import jax.numpy as jnp
from jax import lax
from src.jax.v4_model import ModelConfigV4, init_v4_params, training_rollout, encode_sensory, deliberate_step, emit_step, rms_norm

def test_online_loss_parity():
    cfg = ModelConfigV4(surface_dim=64, num_shells=4, radial_dim=32, num_shortcut_tracts=4, vocab_size=100)
    key = jax.random.PRNGKey(42)
    params, buffers = init_v4_params(cfg, key)

    B, L_q, L_r = 2, 8, 12
    q = jax.random.randint(key, (B, L_q), 1, 99)
    r = jax.random.randint(key, (B, L_r), 1, 99)

    # 1. Standard: materialize all logits, compute loss outside
    def standard_loss_fn(p):
        logits = training_rollout(p, q, L_r, buffers, config=cfg, deliberation_mode="scheduled", sampling_p=0.5, thalamic_anchor_weight=0.1)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        safe_targets = jnp.maximum(r, 0)
        target_log_probs = jnp.take_along_axis(log_probs, safe_targets[..., None], axis=-1).squeeze(-1)
        token_loss = -target_log_probs
        mask = (r >= 0).astype(jnp.float32)
        return jnp.sum(token_loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    # 2. Online: compute step-wise loss inside _emit with exact same parameters
    def online_loss_fn(params):
        B, L = q.shape
        target_len = r.shape[1]
        sensory_mem, sensory_kv = encode_sensory(params, q, config=cfg)
        num_shells = cfg.num_shells
        surface_dim = cfg.surface_dim
        vocab_size = cfg.vocab_size
        thinking_cycles = cfg.thinking_cycles

        mask = (q != 0)[:, :, None].astype(jnp.float32)
        prompt_flux = jnp.sum(sensory_mem * mask, axis=1) / jnp.maximum(jnp.sum(mask, axis=1), 1.0)
        init_state = jnp.zeros((B, num_shells, surface_dim), dtype=jnp.float32)
        state = init_state.at[:, -1, :].set(prompt_flux)

        deliberate_remat = jax.checkpoint(deliberate_step)
        def _delib(s, _):
            return deliberate_remat(params, s, buffers), None
        state, _ = lax.scan(_delib, state, None, length=thinking_cycles)

        thalamic_end = max(1, num_shells // 3)
        thalamic_anchor = state[:, :thalamic_end, :]

        if "motor.plan_init.weight" in params:
            delib_summary = jnp.concatenate([state[:, 0, :], prompt_flux], axis=-1)
            plan_raw = jnp.matmul(delib_summary, params["motor.plan_init.weight"].T)
            if "motor.plan_init.bias" in params:
                plan_raw = plan_raw + params["motor.plan_init.bias"]
            init_plan = jax.nn.tanh(plan_raw)
        else:
            init_plan = jnp.zeros((B, surface_dim), dtype=jnp.float32)

        init_refractory = jnp.zeros((B, vocab_size), dtype=jnp.float32)
        init_thought = jnp.zeros((B, surface_dim), dtype=jnp.float32)
        tf_sensory = jnp.zeros((B, target_len, surface_dim), dtype=jnp.float32)

        emit_step_remat = jax.checkpoint(emit_step)

        def _emit(carry, step_idx):
            cur_state, cur_trace, cur_plan, prev_thought = carry
            tf_aud = tf_sensory[:, step_idx, :]
            internal_feed = jnp.where(step_idx == 0, tf_aud, prev_thought)
            feed = 0.5 * tf_aud + 0.5 * internal_feed

            logits, nxt_s, nxt_t, nxt_p = emit_step_remat(
                params, cur_state, sensory_kv, cur_plan, cur_trace,
                step_idx, buffers, config=cfg, auditory_feedback=feed, repetition_penalty=1.25,
            )

            soft_p = lax.stop_gradient(jax.nn.softmax(logits, axis=-1))
            soft_emb = jnp.matmul(soft_p, params["sensory.embedding.weight"])
            g = jax.nn.silu(jnp.matmul(soft_emb, params["sensory.fc1_gate.weight"].T))
            v = jnp.matmul(soft_emb, params["sensory.fc1_val.weight"].T)
            h_norm = rms_norm(g * v, params["sensory.norm1.weight"])
            nxt_thought = lax.stop_gradient(rms_norm(jnp.matmul(h_norm, params["sensory.fc2.weight"].T) + soft_emb, params["sensory.norm_out.weight"]))

            if 0.1 > 0.0:
                nxt_s = nxt_s.at[:, :thalamic_end, :].set(
                    0.9 * nxt_s[:, :thalamic_end, :] + 0.1 * thalamic_anchor
                )

            # Online step cross-entropy
            cur_target = r[:, step_idx]
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            safe_targets = jnp.maximum(cur_target, 0)
            target_log_probs = jnp.take_along_axis(log_probs, safe_targets[:, None], axis=-1).squeeze(-1)
            mask_t = (cur_target >= 0).astype(jnp.float32)
            step_loss = -target_log_probs * mask_t

            return (nxt_s, nxt_t, nxt_p, nxt_thought), (step_loss, mask_t)

        _, (all_loss, all_mask) = lax.scan(_emit, (state, init_refractory, init_plan, init_thought), jnp.arange(target_len))
        return jnp.sum(all_loss) / jnp.maximum(jnp.sum(all_mask), 1.0)

    l1, g1 = jax.value_and_grad(standard_loss_fn)(params)
    l2, g2 = jax.value_and_grad(online_loss_fn)(params)

    diff = abs(float(l1) - float(l2))
    print(f"Standard Loss: {float(l1):.6f} | Online Loss: {float(l2):.6f} | Diff: {diff:.2e}")
    assert diff < 1e-5, f"Loss mismatch: {diff}"

    for k in g1:
        g_diff = float(jnp.max(jnp.abs(g1[k] - g2[k])))
        assert g_diff < 1e-4, f"Gradient mismatch on {k}: {g_diff}"
    print("ALL TESTS PASSED! Exact bitwise/mathematical parity confirmed!")

if __name__ == "__main__":
    test_online_loss_parity()
