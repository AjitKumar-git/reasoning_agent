"""What the shipped 1m checkpoints contain, and how far apart the two saved snapshots are.

`train_scale_1m.py` writes bare state dicts: no step count, no config, no RNG state. So the
length of the run behind `cerebro_spheroid_1m_best.pt` cannot be read off the artifact, and
which of the two files is the later model cannot either. This prints what is recoverable --
timestamps, key sets, and the magnitude of the weight difference -- and leaves the inference
to the notes. A max-abs difference near the optimiser's per-step displacement is the only
hint about how many steps separate them.
"""
import os
import torch

BEST = "checkpoints/cerebro_spheroid_1m_best.pt"
LATEST = "checkpoints/cerebro_spheroid_1m_latest.pt"

for path in (BEST, LATEST):
    st = os.stat(path)
    print(f"{path}\n  size {st.st_size:,} | mtime {st.st_mtime:.6f}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    print(f"  container {type(ck).__name__} | {len(ck)} top-level keys")
    meta = [k for k in ck if not torch.is_tensor(ck[k])] if isinstance(ck, dict) else []
    print(f"  non-tensor entries (step/config would live here): {meta}")
    values = sum(v.numel() for v in ck.values() if torch.is_tensor(v))
    print(f"  tensor values: {values:,}")

a = torch.load(BEST, map_location="cpu", weights_only=True)
b = torch.load(LATEST, map_location="cpu", weights_only=True)
print(f"\nmtime gap: {os.stat(LATEST).st_mtime - os.stat(BEST).st_mtime:+.1f} s")
print("key sets equal:", set(a) == set(b))
tot = mx = 0.0
worst = None
for k in a:
    d = (a[k].float() - b[k].float()).abs()
    tot += float(d.sum())
    if float(d.max()) > mx:
        mx, worst = float(d.max()), k
n_values = sum(x.numel() for x in a.values())
print(f"total abs diff {tot:.4f} | max abs diff {mx:.6f} (on {worst})")
print(f"mean abs diff per value {tot / n_values:.3e}")
print("bitwise identical:", all(torch.equal(a[k], b[k]) for k in a))
