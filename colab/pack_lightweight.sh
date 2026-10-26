#!/usr/bin/env bash
# Packs ONLY code and tiny metadata parquets into a featherweight zip (< 300 KB).
# Excludes all heavy datasets, checkpoints, caches, and logs.
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

OUT_ZIP="colab/reasoning_agent_code_only.zip"
rm -f "$OUT_ZIP"

echo "Creating lightweight zip: $OUT_ZIP..."
zip -q -r "$OUT_ZIP" \
    src/ \
    training/ \
    colab/ \
    data/corpus/ \
    data/general_knowledge/ \
    data/cerebro_cognitive/cerebro_cognitive_manifest.json \
    main.py \
    chat_session.py \
    chat_web.py \
    test_v3_trained.py \
    README.md \
    -x "*.pt" \
    -x "*.bin" \
    -x "*.parquet" \
    -x "*.log" \
    -x "*__pycache__*" \
    -x "*.hf_cache*" \
    -x "*colab/*.zip*"

echo "Done! Package size:"
ls -lh "$OUT_ZIP"
