"""
Rebuilds the world-knowledge corpus into a leak-free train/val pair.

The shipped `data/general_knowledge/world_knowledge_{train,val}.parquet` cannot support a
meaningful number: 20,000 rows collapse to 3,753 distinct questions and 470 distinct
answers, 59.6% of rows share their question with 2-5 differently phrased answers, and
1,914 of the 2,000 "val" questions also appear verbatim in train. Cross-entropy measured
on that file is a memorisation score, not a generalisation score.

This script produces:

  world_knowledge_canonical_train.parquet   one row per question, one canonical answer
  world_knowledge_canonical_val.parquet     questions about entities never seen in train
  manifest.json                             counts, md5s, held-out entity list

Canonicalisation removes the phrasing lottery: when a question accepts "Canberra." and
"The capital city of Australia is Canberra, located in Oceania.", asking the model to
choose between them puts an irreducible ln(k) floor under the loss and muddles every
comparison. One question, one target.

The val split is by ENTITY, not by row: the chosen countries and their capitals are
removed from every train question *and* every train answer, so no train row can leak a
held-out answer through a neighbouring fact.

Usage:
  python3 training/build_corpus.py                       # build with defaults
  python3 training/build_corpus.py --dry_run             # report, write nothing
  python3 training/build_corpus.py --holdout_countries 10
"""

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT_DIR, "data", "general_knowledge", "world_knowledge_train.parquet")
OUT_DIR = os.path.join(ROOT_DIR, "data", "corpus")

# "What is the capital city of Australia?" -> Australia
COUNTRY_RE = re.compile(r"capital (?:city )?of ([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*)")
# "The capital of Australia is Canberra, located in Oceania." -> (Australia, Canberra)
ANSWER_RE = re.compile(
    r"capital (?:city )?of ([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*) is "
    r"([A-Z][a-zA-Zé\-']+(?: [A-Z][a-zA-Zé\-']+)*)"
)


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_answers(pairs):
    """Collapses the phrasing variants of one question to a single target.

    Most frequent exact string wins; ties break on shortest, then lexicographic. Frequency
    is the corpus's own preference, and the tie-breaks are deterministic so two runs on the
    same input produce byte-identical output.
    """
    counts = defaultdict(Counter)
    for q, a in pairs:
        counts[q][a] += 1
    canonical = {}
    for q, c in counts.items():
        best = sorted(c.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))[0][0]
        canonical[q] = best
    return canonical


def derive_country_capitals(pairs):
    """Builds the country -> capital table from the corpus itself, keeping only 1:1 matches."""
    found = defaultdict(set)
    for q, a in pairs:
        country = COUNTRY_RE.search(q)
        answer = ANSWER_RE.search(a)
        if country and answer and country.group(1) == answer.group(1):
            found[country.group(1)].add(answer.group(2))
    clean = {c: next(iter(v)) for c, v in found.items() if len(v) == 1}
    conflicted = {c: sorted(v) for c, v in found.items() if len(v) > 1}
    return clean, conflicted


def mentions(text: str, terms) -> bool:
    return any(term in text for term in terms)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=SRC)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--holdout_countries", type=int, default=10,
                        help="Countries removed from train entirely and used as the val split")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dry_run", action="store_true", help="Print the plan without writing files")
    args = parser.parse_args()

    table = pq.read_table(args.source)
    pairs = list(zip(table["stimulus"].to_pylist(), table["articulation"].to_pylist()))
    print(f"source             {os.path.relpath(args.source, ROOT_DIR)}")
    print(f"rows               {len(pairs):,}")
    print(f"distinct questions {len({q for q, _ in pairs}):,}")
    print(f"distinct answers   {len({a for _, a in pairs}):,}")

    canonical = canonical_answers(pairs)
    print(f"canonical rows     {len(canonical):,}  (one target per question)")
    print()

    capitals, conflicts = derive_country_capitals(pairs)
    print(f"countries parsed   {len(capitals)} with a 1:1 capital")
    if conflicts:
        print(f"  [warn] {len(conflicts)} countries had conflicting capitals and were excluded: {sorted(conflicts)[:6]}")
    if len(capitals) < args.holdout_countries:
        raise SystemExit(f"only {len(capitals)} clean countries; cannot hold out {args.holdout_countries}")

    names = sorted(capitals)
    rng = random.Random(args.seed)
    held = sorted(rng.sample(names, args.holdout_countries))
    held_terms = []
    for c in held:
        held_terms += [c, capitals[c]]
    print(f"held-out entities  {len(held)}: {', '.join(held)}")
    print()

    # A row belongs to val when either field names a held-out country or capital. The check
    # covers the answer too, so a held-out capital cannot survive inside a neighbouring fact.
    train_rows, val_rows = [], []
    for q, a in canonical.items():
        if mentions(q, held_terms) or mentions(a, held_terms):
            val_rows.append((q, a))
        else:
            train_rows.append((q, a))

    train_q = {q for q, _ in train_rows}
    overlap = [q for q, _ in val_rows if q in train_q]
    if overlap:
        raise SystemExit(f"split leaked: {len(overlap)} val questions also in train")

    # Entity-level guarantee: no held-out name may survive anywhere in train.
    for q, a in train_rows:
        assert not mentions(q + " " + a, held_terms), f"held-out entity leaked through: {q}"

    print(f"train              {len(train_rows):,} rows")
    print(f"val                {len(val_rows):,} rows")
    print(f"train answers      {len({a for _, a in train_rows}):,} distinct")
    print()

    for name, rows in (("train", train_rows), ("val", val_rows)):
        lens = sorted(len(a.split()) for _, a in rows)
        print(f"  {name} answer words: median {lens[len(lens) // 2]}, max {lens[-1]}")

    if args.dry_run:
        print("\n[dry_run] nothing written")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    paths = {}
    for name, rows in (("train", train_rows), ("val", val_rows)):
        path = os.path.join(args.out_dir, f"world_knowledge_canonical_{name}.parquet")
        table_out = pa.table({
            "stimulus": pa.array([r[0] for r in rows], type=pa.string()),
            "articulation": pa.array([r[1] for r in rows], type=pa.string()),
        })
        pq.write_table(table_out, path)
        paths[name] = path

    manifest = {
        "built_by": "training/build_corpus.py",
        "source": os.path.relpath(args.source, ROOT_DIR),
        "source_md5": md5_of(args.source),
        "source_rows": len(pairs),
        "canonicalisation": "most frequent exact answer per question; ties by length then lexicographic",
        "holdout_entities": {c: capitals[c] for c in held},
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_distinct_answers": len({a for _, a in train_rows}),
        "val_distinct_answers": len({a for _, a in val_rows}),
        "train_md5": md5_of(paths["train"]),
        "val_md5": md5_of(paths["val"]),
        "seed": args.seed,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    print()
    for name, path in paths.items():
        print(f"wrote {os.path.relpath(path, ROOT_DIR)}  ({os.path.getsize(path):,} B)")
    print(f"wrote {os.path.relpath(os.path.join(args.out_dir, 'manifest.json'), ROOT_DIR)}")


if __name__ == "__main__":
    main()
