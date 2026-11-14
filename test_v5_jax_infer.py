import sys, os
sys.path.insert(0, os.path.abspath("../cerebro_spheroid_v5"))
import jax
import jax.numpy as jnp
import torch
from transformers import AutoTokenizer
from src.jax.v5_model import ModelConfigV5, PRESETS_V5, training_rollout, generate_small_world_indices

tok = AutoTokenizer.from_pretrained("huggyllama/llama-7b", cache_dir=os.path.abspath(".hf_cache/hub"))
ckpt_path = "../cerebro_spheroid_v5/checkpoints/v5_small_cloud_latest.pt"
data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
step = data.get("step", "unknown")
loss = data.get("loss", "unknown")
print(f"Loaded cloud checkpoint step {step}, loss {loss}")

jax_params = {k: jnp.array(v.float().numpy()) for k, v in data["model_state"].items()}
cfg = ModelConfigV5(**PRESETS_V5["v5_small"])
buffers = generate_small_world_indices(cfg.num_shells, cfg.num_shortcut_tracts)

prompt = "The capital of France is"
p_ids = jnp.array([tok.encode(prompt)], dtype=jnp.int32)
print("Prompt tokens:", p_ids)

# Run training_rollout in closed-loop generation mode (sampling_p=1.0 uses model's own predictions as feedback)
logits = training_rollout(
    params=jax_params,
    prompts=p_ids,
    target_len=60,
    buffers=buffers,
    config=cfg,
    feedback_tokens=None,
    sampling_p=1.0,
    repetition_penalty=1.2,
)

print("Logits shape:", logits.shape)
tokens = []
for t in range(logits.shape[1]):
    tok_id = int(jnp.argmax(logits[0, t]))
    tokens.append(tok_id)

print("\n--- CONTINUOUS OUTPUT FROM V5 CLOUD (TPU/JAX) ---")
print(tok.decode(tokens))
print("--- END OF OUTPUT ---")
