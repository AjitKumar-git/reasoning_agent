"""Does a seeded private RNG shuffle the curriculum the same way the global RNG did?

`diagnose_spheroid.py` used to rely on `random.seed()` in the harness plus an unseeded
`random.shuffle()` inside the loader, so the eval batches were reproducible only because the
harness happened to seed the process and nothing else consumed `random` in between. That is a
fragile accident: any model init or tokenizer path that draws a python random number shifts
every recorded metric. The loader now takes an explicit `shuffle_seed`.

This checks the two are the same permutation, because the loss ladder in RESEARCH_NOTES 20.10
was measured under the old one and has to keep reproducing against the same parquet.
"""

import os
os.environ.setdefault("HF_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".hf_cache")))

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import torch
from transformers import AutoTokenizer

from src.data_loaders import get_specialized_cognitive_stream

SEED = 1337
N_BATCHES = 6


def pull(**kwargs):
    # One generator, many draws -- exactly how diagnose_spheroid.py freezes its eval set.
    # Constructing it per draw would re-shuffle every time and compare nothing.
    stream = get_specialized_cognitive_stream(TOKENIZER, batch_size=2, **kwargs)
    return [next(stream) for _ in range(N_BATCHES)]


def tokens_of(batches):
    return [tuple(t.flatten().tolist()) for _, t in batches]


TOKENIZER = AutoTokenizer.from_pretrained("gpt2")
TOKENIZER.pad_token = TOKENIZER.eos_token
PARQUET = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cerebro_cognitive",
                                      "legacy_train_unseeded_e16c6f47.parquet"))
if not os.path.exists(PARQUET):
    raise SystemExit(f"missing {PARQUET}")

random.seed(SEED)
torch.manual_seed(SEED)
old = pull(parquet_path=PARQUET)
new = pull(parquet_path=PARQUET, shuffle_seed=SEED)

old_t, new_t = tokens_of(old), tokens_of(new)
print(f"batches compared : {N_BATCHES}")
print(f"scored tokens    : {sum(len(t) for t in old_t)}")
match = old_t == new_t
for i, (a, b) in enumerate(zip(old_t, new_t)):
    print(f"  batch {i}: {'identical' if a == b else 'DIFFERENT'}")
print(f"\npermutations agree: {match}")
if not match:
    print("The documented loss ladder will NOT reproduce under --shuffle_seed. Either pin the")
    print("loader back to the global RNG or re-measure and re-record the figures.")
else:
    print("A private Random(seed) shuffles identically to seed()+global shuffle here, so the")
    print("loader no longer depends on what else touched the global RNG while still")
    print("reproducing the recorded numbers.")
