"""One-shot reasoning probe for the shipped CerebroSpheroid checkpoint.

Loads the default 1M checkpoint once, runs a spread of tasks through the production
generate() path (ingest -> deliberate -> zero-feedback emit), and for two of them dumps the
first emitted token's top-5 so the readout's actual preference distribution is visible.
"""
import os
os.environ.setdefault("HF_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), ".hf_cache")))
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
from src.inference import ReasoningAgent

TASKS = [
    ("in-domain arithmetic", "Maya has 8 markers. She buys 4 more and gives 3 away. How many markers does she have? Answer:"),
    ("in-domain arithmetic", "If 4 books cost $20, how much do 6 books cost? Answer:"),
    ("counting", "There are 20 apples in a box. You take 6 apples. How many apples remain? Answer:"),
    ("commonsense", "What is the primary function of the heart in the human body? Answer:"),
    ("logic", "All roses are flowers. Some flowers fade quickly. Do all roses fade quickly? Answer:"),
    ("out-of-distribution", "Write a haiku about autumn leaves. Answer:"),
]

agent = ReasoningAgent(checkpoint_path="checkpoints/cerebro_spheroid_1m_best.pt", arch="auto")
print(f"arch={agent.arch}  ckpt={agent.checkpoint_loaded}  device={agent.device}  dtype=bfloat16")
print("=" * 84)

for label, prompt in TASKS:
    ans = agent.generate(prompt, thinking_cycles=4, max_tokens=24, temperature=0.0)
    print(f"\n[{label}]")
    print(f"  Q: {prompt}")
    print(f"  A: {ans!r}")

# First-token preference for two prompts: shows whether the answer is even reachable.
print("\n" + "=" * 84)
print("First emitted token, top-5 (greedy readout, 4 deliberation cycles):")
for prompt in [TASKS[0][1], TASKS[3][1]]:
    tokens = agent.tokenizer.encode(prompt.strip())
    pt = torch.tensor([tokens], device=agent.device)
    with torch.no_grad():
        state, mem = agent.model.ingest_stimulus_with_memory(pt, state=None, living=True)
        state = agent.model.deliberate(state, thinking_cycles=4)
        trace = torch.zeros(1, agent.vocab_size, device=agent.device)
        logits, _, _ = agent.model.emit_next_token(state, sensory_memory=mem,
                                                   refractory_trace=trace, step_idx=0)
        topk = torch.topk(logits[0].float(), 5)
    toks = [repr(agent.tokenizer.decode([i])) for i in topk.indices.tolist()]
    probs = (F.softmax(logits[0].float(), dim=-1)[topk.indices]).tolist()
    print(f"\n  Q: {prompt}")
    print("  top5: " + ", ".join(f"{t}({p:.3f})" for t, p in zip(toks, probs)))

agent.stop_living_heartbeat()
