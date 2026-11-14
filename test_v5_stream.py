import os
import sys
import torch
from transformers import AutoTokenizer

from src.spheroid_v5 import CerebroSpheroidV5, ModelConfigV5

ckpt_path = "checkpoints/salvaged/cerebro_spheroid_v5_small_salvaged_step158500.pt"
print(f"Loading checkpoint: {ckpt_path}")
data = torch.load(ckpt_path, map_location="cpu", weights_only=False)

cfg = data["config"]
print("Config:", cfg)
model = CerebroSpheroidV5(cfg)
state_dict = data["model_state"]
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("Loaded state dict! Missing:", len(missing), "Unexpected:", len(unexpected))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Let's inspect tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
except Exception as e:
    print("Could not load gpt2 tokenizer:", e)
    tokenizer = None

print("Model successfully ready on", device)
