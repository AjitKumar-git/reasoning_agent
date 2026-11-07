"""
Natural Conversational Capability Probe for CerebroSpheroid V4.

Tests open-ended conversational fluency, natural dialogue, tone, persona,
coherence, empathy, and everyday reasoning across:
1. Casual Greetings & Chit-chat
2. Self-Identity & Persona
3. Empathy & Emotional Support
4. Everyday Advice & Simple Planning
5. Storytelling & Imagination
6. Friendly Explanations

Tests both raw stimulus input (<stim> ... </stim>) and primed act prefix (<stim> ... </stim> <act>).
Compares thinking cycles K=1, K=3, K=5 and temperatures T=0.4, 0.7.
"""

import os
import sys
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.corpus import build_tokenizer, format_stimulus_prompt, clean_motor_response
from src.spheroid_v4 import build_spheroid_v4

CONVERSATIONAL_PROMPTS = [
    # 1. Casual Greetings & Chit-chat
    ("Greeting", "Hello! How are you doing today?"),
    ("Greeting", "Good morning! Did you sleep well?"),
    
    # 2. Self-Identity & Persona
    ("Persona", "Who are you and what can you do?"),
    ("Persona", "Are you an AI or a human?"),
    
    # 3. Empathy & Support
    ("Empathy", "I had a really exhausting and stressful day today."),
    ("Empathy", "I just adopted a cute little puppy and I'm so happy!"),
    
    # 4. Everyday Advice & Cooking
    ("Advice", "What's a simple, healthy breakfast I can make in 5 minutes?"),
    ("Advice", "Can you recommend a fun activity for a rainy afternoon?"),
    
    # 5. Creativity & Storytelling
    ("Story", "Tell me a short bedtime story about a curious little robot who loved stars."),
    ("Humor", "Tell me a lighthearted joke."),
    
    # 6. Friendly Explanations
    ("Explanation", "Can you explain why the sky is blue in a warm, simple way for a 7-year-old?"),
    ("Explanation", "Why do dogs wag their tails when they are happy?")
]

def run_probe(ckpt_path: str, device: str = "cuda:0"):
    print("=" * 80)
    print(f"🗣️  PROBING NATURAL CONVERSATION CAPABILITY: {ckpt_path}")
    print("=" * 80)
    
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Compute Device: {dev}")
    
    tok = build_tokenizer("llama")
    print(f"🔤 Tokenizer: Modern Llama BPE (Vocab: {len(tok):,})")
    
    model = build_spheroid_v4("v4_small", vocab_size=len(tok))
    meta = model.load_compatible_checkpoint(ckpt_path, device=dev)
    model.to(device=dev, dtype=torch.float32)
    model.eval()
    
    step_info = meta.get("step", "unknown") if isinstance(meta, dict) else "84,800"
    loss_info = meta.get("loss", "unknown") if isinstance(meta, dict) else "unknown"
    print(f"📌 Checkpoint Step: {step_info} | Loss: {loss_info}")
    
    # We test with K=3 thinking cycles (the standard operational mode) and temperature=0.7
    print("\n" + "=" * 80)
    print("--- EVALUATION SUITE: NATURAL DIALOGUE TRANSCRIPTS (K=3, T=0.7) ---")
    print("=" * 80)
    
    for category, prompt in CONVERSATIONAL_PROMPTS:
        print(f"\n[{category.upper()}] User: \"{prompt}\"")
        
        # Mode A: Standard stimulus formatting
        with torch.no_grad():
            ans_stim = model.generate_text(
                prompt,
                tokenizer=tok,
                thinking_cycles=3,
                max_new_tokens=96,
                temperature=0.7,
                repetition_penalty=1.25,
                device=dev,
                strip_bio_prefix=True,
            )
            print(f"  🤖 Model (<stim>):\n    {ans_stim.strip()}")
            
        # Mode B: Primed with <stim> ... </stim> <act>
        with torch.no_grad():
            primed_prompt = f"{format_stimulus_prompt(prompt)} <act>"
            p_ids = tok.encode(primed_prompt, add_special_tokens=False)
            tok_tensor = torch.tensor([p_ids], dtype=torch.long, device=dev)
            out_ids = model.generate(
                tok_tensor,
                thinking_cycles=3,
                max_new_tokens=96,
                temperature=0.7,
                repetition_penalty=1.25,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )
            raw = tok.decode(out_ids[0].tolist(), skip_special_tokens=True)
            # Remove the primed prompt
            if raw.startswith(primed_prompt):
                gen_part = raw[len(primed_prompt):]
            else:
                gen_part = clean_motor_response(raw)
            print(f"  🤖 Model (primed <act>):\n    {gen_part.strip()}")

if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/cerebro_spheroid_v4_step_84800_latest.pt"
    run_probe(ckpt)
