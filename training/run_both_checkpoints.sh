#!/usr/bin/env bash
# Scores both shipped 1m checkpoints with the same seeded instrument, sequentially, so they
# contend for nothing. Results land in /tmp/v2_<name>.txt.
#
# Pinned to the frozen unseeded pool that sections 20.10 and 20.12 measured, because the
# canonical path now holds the rebuilt seeded split. Set PARQUET to
# data/cerebro_cognitive/cerebro_cognitive_val.parquet to score them on a real holdout instead
# -- the numbers will differ, and neither set is the pool these weights were fitted to.
#
# FLOOR_PARQUET controls where the [1b]/[1c] input-independent floors are fitted. Left empty, the
# instrument picks the honest source for the eval file in use: fitting the legacy pool reproduces
# the 20.10/20.12 ladder exactly (fit draws continue the same generator past the frozen batch list,
# so they stay disjoint within its 50,000-row epoch), while scoring a *_val file fits floors from
# the sibling *_train file. Set FLOOR_PARQUET explicitly to force a source and watch the floors move.
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PARQUET="${PARQUET:-data/cerebro_cognitive/legacy_train_unseeded_e16c6f47.parquet}"
FLOOR_PARQUET="${FLOOR_PARQUET-}"
floor_args=()
if [[ -n "$FLOOR_PARQUET" ]]; then floor_args=(--floor_parquet "$FLOOR_PARQUET"); fi
: > /tmp/v2_status.txt
for ck in best latest; do
    timeout 2400 python3 training/diagnose_spheroid.py \
        --checkpoint "checkpoints/cerebro_spheroid_1m_${ck}.pt" --batches 64 \
        --parquet "$PARQUET" "${floor_args[@]}" \
        > "/tmp/v2_${ck}.txt" 2>&1
    echo "$ck exit=$?" >> /tmp/v2_status.txt
done
