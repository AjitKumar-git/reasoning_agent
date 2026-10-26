import os
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(__file__), ".hf_cache"))
from transformers import AutoTokenizer
import pyarrow.parquet as pq
import statistics

tok = AutoTokenizer.from_pretrained("gpt2")
for f in ("cerebro_cognitive_train", "cerebro_cognitive_val"):
    t = pq.read_table("data/cerebro_cognitive/%s.parquet" % f).to_pydict()
    n = len(t["stimulus"])
    pl = sorted(len(tok.encode(str(r))) for r in t["stimulus"][:6000])
    rl = sorted(len(tok.encode(str(r))) for r in t["articulation"][:6000])
    print("--- %s (%d rows)" % (f, n))
    for name, x in (("prompt", pl), ("response", rl)):
        print("  %-8s mean %6.1f  p50 %4d  p95 %4d  p99 %4d  max %4d" % (
            name, statistics.mean(x), x[len(x) // 2], x[int(.95 * len(x))],
            x[int(.99 * len(x))], x[-1]))
