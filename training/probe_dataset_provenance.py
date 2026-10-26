"""Compares the dataset on disk now with the archived copy that predates the checkpoints.

The generator is unseeded, so `data/cerebro_cognitive/cerebro_cognitive_train.parquet` is not
stable across regenerations. Whether the records a checkpoint trained on are the records a
diagnostic scores is a question about these two files, not about the pipeline.
"""
import hashlib
import os

import pandas as pd

LIVE = "data/cerebro_cognitive/cerebro_cognitive_train.parquet"
ARCH = "archive/cerebro_spheroid_v1_concentric/data/cerebro_cognitive/cerebro_cognitive_train.parquet"


def digest(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


for label, path in (("live", LIVE), ("archived", ARCH)):
    st = os.stat(path)
    print(f"{label:9} {path}\n          md5 {digest(path)} | {st.st_size:,} bytes "
          f"| mtime {st.st_mtime:.0f}")

live = pd.read_parquet(LIVE)
arch = pd.read_parquet(ARCH)
print(f"\nrows           live {len(live):,} | archived {len(arch):,}")
print(f"columns        live {list(live.columns)} | archived {list(arch.columns)}")

sl, sa = set(live.stimulus), set(arch.stimulus)
print(f"\nunique stimuli live {len(sl):,} | archived {len(sa):,} | shared {len(sl & sa):,}")
print(f"identical pools: {sl == sa}")
al = dict(zip(live.stimulus, live.articulation))
aa = dict(zip(arch.stimulus, arch.articulation))
shared = list(sl & sa)[:2000]
disagree = sum(1 for s in shared if al[s] != aa[s])
print(f"answer disagreements over the first {len(shared)} shared stimuli: {disagree}")

print("\nsample records, same generator, different draw:")
for name, df in (("live", live), ("archived", arch)):
    print(f"  {name:9} {df.stimulus.iloc[0][:78]!r} -> {df.articulation.iloc[0]!r}")
    print(f"  {name:9} {df.stimulus.iloc[1][:78]!r} -> {df.articulation.iloc[1]!r}")
