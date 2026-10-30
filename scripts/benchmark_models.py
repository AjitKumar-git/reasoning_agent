"""
Benchmark All Candidate Spheroid Models on 100 Test OpenHermes Lines.

Evaluates every model under identical, fair conditions:
- 100 standardized test conversation pairs from OpenHermes-2.5 held-out row group.
- Exact Cross-Entropy Loss and Perplexity (PPL) ignoring pad tokens (-100).
- Excludes probe models (*probe*).
- Outputs ranked leaderboard from lowest loss to highest.
"""

import os
import sys
import glob
import math
import random
import torch
import torch.nn.functional as F
import pyarrow.parquet as pq

# Add repo root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.corpus import build_tokenizer, _conversation_pairs
from src.spheroid_v4 import build_spheroid_v4, CerebroSpheroidV4
from src.spheroid_v3 import build_spheroid_v3, CerebroSpheroidV3, training_rollout
from training.train_v4_full import build_batch


def get_test_records(parquet_path: str, count: int = 100):
    """Loads exactly `count` prompt-answer pairs from OpenHermes."""
    pf = pq.ParquetFile(parquet_path)
    table = pf.read_row_group(0)
    convs = table["conversations"].to_pylist()
    langs = table["language"].to_pylist() if "language" in table.column_names else [None] * len(convs)

    records = []
    for lang, conv in zip(langs, convs):
        if lang and str(lang).lower() not in ("english", "en"):
            continue
        pairs = _conversation_pairs(conv)
        for p, a in pairs:
            if len(p.strip()) > 10 and len(a.strip()) > 10:
                records.append(("openhermes", p.strip(), a.strip()))
                if len(records) >= count:
                    return records
    return records


@torch.no_grad()
def evaluate_checkpoint(ckpt_path: str, records, device: torch.device, batch_size: int = 10):
    """Evaluates cross-entropy loss on the test records using batched rollout."""
    filename = os.path.basename(ckpt_path)

    try:
        raw_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return {"error": f"Failed to load: {e}"}

    weights = raw_state.get("model", raw_state) if isinstance(raw_state, dict) else raw_state
    if not isinstance(weights, dict):
        return {"error": "Invalid state dict"}

    # Detect vocabulary size from embedding weight
    vocab_size = 32000
    for k in ("sensory_dnn.embedding.weight", "sensory.embedding.weight", "embedding.weight"):
        if k in weights:
            vocab_size = weights[k].shape[0]
            break

    tokenizer_type = "llama" if vocab_size == 32000 else "gpt2"
    tokenizer = build_tokenizer(tokenizer_type)
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    # Detect architecture V4 vs V3
    is_v4 = any("shell_proj" in k or "intra_shell" in k or "v4" in filename.lower() for k in weights.keys()) or ("v4" in filename.lower())

    try:
        if is_v4:
            model = build_spheroid_v4("v4_small", vocab_size=vocab_size)
            model.load_compatible_checkpoint(ckpt_path, device=device)
        else:
            model = build_spheroid_v3(vocab_size=vocab_size)
            model.load_compatible_checkpoint(ckpt_path, device=device)
    except Exception as e:
        try:
            matched = {k: v for k, v in weights.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
            model.load_state_dict(matched, strict=False)
        except Exception as e2:
            return {"error": f"Model build/load error: {e2}"}

    model = model.to(device=device, dtype=torch.float32)
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    rng = random.Random(1337)

    for i in range(0, len(records), batch_size):
        batch_records = records[i : i + batch_size]
        built = build_batch(
            tokenizer,
            batch_records,
            window=128,
            rng=rng,
            max_prompt_len=256,
            max_answer_tokens=128,
            pad_token_id=pad_id,
        )
        if built is None:
            continue

        prompts, targets, feedback = (t.to(device) for t in built)
        try:
            logits = training_rollout(model, prompts, targets.shape[1], targets=targets, feedback_tokens=feedback)
            # Cross entropy ignoring -100 padding
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100, reduction="sum")
            valid_mask = (targets != -100)
            n_tokens = valid_mask.sum().item()
            total_loss += loss.item()
            total_tokens += n_tokens
        except Exception as e:
            continue

    if total_tokens == 0:
        return {"error": "No tokens evaluated"}

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20.0))

    param_count = sum(p.numel() for p in model.parameters())
    return {
        "filename": filename,
        "path": ckpt_path,
        "loss": avg_loss,
        "ppl": ppl,
        "arch": "v4" if is_v4 else "v3",
        "vocab": vocab_size,
        "params": param_count,
        "size_mb": os.path.getsize(ckpt_path) / (1024**2),
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"======================================================================")
    print(f" CEREBRO-SPHEROID MODEL BENCHMARK (100 OPENHERMES TEST LINES)")
    print(f" Device: {device} | Evaluation Metric: Mean Cross-Entropy & Perplexity")
    print(f"======================================================================")

    parquet_path = os.path.join(ROOT_DIR, "data/openhermes/openhermes_2_5.parquet")
    if not os.path.exists(parquet_path):
        print(f"Error: {parquet_path} not found.")
        return

    print("Loading 100 test records from OpenHermes...")
    records = get_test_records(parquet_path, count=100)
    print(f"Loaded {len(records)} test dialogue pairs.\n")

    # Discover candidate models (accept custom CLI paths or scan checkpoints/)
    candidates = []
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if os.path.isdir(arg):
                candidates.extend(sorted(glob.glob(os.path.join(arg, "*.pt"))))
            elif os.path.isfile(arg):
                candidates.append(arg)
    else:
        ckpt_dir = os.path.join(ROOT_DIR, "checkpoints")
        candidates = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))

    # Filter out probe and optimizer state dumps
    filtered = []
    for c in candidates:
        fname = os.path.basename(c).lower()
        if "probe" in fname:
            continue
        if "_state.pt" in fname:
            continue
        filtered.append(c)
    candidates = sorted(list(set(filtered)))

    print(f"Evaluating {len(candidates)} candidate model checkpoints:\n")
    results = []
    for c in candidates:
        fname = os.path.basename(c)
        print(f"-> Testing {fname} ({os.path.getsize(c)/(1024**2):.1f} MB)... ", end="", flush=True)
        res = evaluate_checkpoint(c, records, device, batch_size=10)
        if "error" in res:
            print(f"[FAILED: {res['error']}]")
        else:
            print(f"Loss: {res['loss']:.4f} | PPL: {res['ppl']:.2f} ({res['arch']}, {res['vocab']} vocab)")
            results.append(res)

    results.sort(key=lambda x: x["loss"])

    print("\n" + "=" * 82)
    print(" 🏆 FINAL MODEL BENCHMARK LEADERBOARD (100 TEST LINES)")
    print("=" * 82)
    print(f"{'Rank':<5} | {'Model File':<42} | {'Loss':<7} | {'PPL':<8} | {'Arch':<5} | {'Size'}")
    print("-" * 82)
    for i, r in enumerate(results, 1):
        print(f"{i:<5} | {r['filename']:<42} | {r['loss']:<7.4f} | {r['ppl']:<8.2f} | {r['arch']:<5} | {r['size_mb']:5.1f} MB")
    print("=" * 82)

    if results:
        best = results[0]
        print(f"\n🥇 UNDISPUTED BEST MODEL: {best['filename']}")
        print(f"   Cross-Entropy Loss : {best['loss']:.4f}")
        print(f"   Perplexity         : {best['ppl']:.2f}")
        print(f"   Architecture       : {best['arch']} ({best['params']:,} parameters)")
        print(f"   Vocabulary         : {best['vocab']} tokens")


if __name__ == "__main__":
    main()
