"""
Fixes the tiny val set.

dataset.py splits val off before balance_data.py runs, so val gets balanced
against its own 4-hop count and comes out at ~57 rows. Useless. At n=57 the
error bar on F1 is about +-6.6%, so two runs have to differ by ~18 points
before you can say anything. Can't rank ablations on that.

Merges train+val back together and re-splits properly.

    python resplit.py --val_per_hop 120
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict

DATA = "data"


def load(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def show(label: str, rows: list[dict]) -> None:
    hops = dict(sorted(Counter(r.get("hops") for r in rows).items(),
                       key=lambda x: (x[0] is None, x[0])))
    # rough binomial SE at p=0.5, just to see if the split is big enough
    se = (0.25 / len(rows)) ** 0.5 * 100 if rows else 0
    print(f"  {label:14} {len(rows):6}  hops={hops}"
          + (f"   +-{se:.1f}% noise" if rows else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_per_hop", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    train_p = os.path.join(DATA, "train.jsonl")
    val_p = os.path.join(DATA, "val.jsonl")

    pool = load(train_p) + load(val_p)
    if not pool:
        print("no data. run dataset.py + balance_data.py first")
        return

    print("BEFORE")
    show("train", load(train_p))
    show("val", load(val_p))

    # train and val might overlap if dataset.py was rerun, so dedupe on question
    seen: set[str] = set()
    uniq = []
    for r in pool:
        q = r["question"].strip().lower()
        if q not in seen:
            seen.add(q)
            uniq.append(r)
    if len(uniq) < len(pool):
        print(f"\n  dropped {len(pool) - len(uniq)} dupes")

    by_hop: dict = defaultdict(list)
    for r in uniq:
        by_hop[r.get("hops")].append(r)

    # don't take more than a third of the smallest bucket
    n_val = min(a.val_per_hop, min(len(v) for v in by_hop.values()) // 3)
    if n_val < a.val_per_hop:
        print(f"\n  only {n_val}/hop available (wanted {a.val_per_hop}) "
              "- 4-hop is the bottleneck")

    train, val = [], []
    for h in sorted(by_hop, key=lambda x: (x is None, x)):
        rows = by_hop[h]
        rng.shuffle(rows)
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)

    # keep a copy in case this goes wrong
    for p in (train_p, val_p):
        bak = p.replace(".jsonl", ".presplit.jsonl")
        if os.path.exists(p) and not os.path.exists(bak):
            shutil.copy(p, bak)

    save(train_p, train)
    save(val_p, val)

    print("\nAFTER")
    show("train", train)
    show("val", val)

    print(f"\nval is {len(val)} rows now. anything under ~"
          f"{3 * (0.25 / len(val)) ** 0.5 * 100:.0f}% is still noise but that's "
          "workable.")
    print("compare runs on val. dev_benchmark is only 54 rows - use it once at "
          "the end to sanity check, not for picking configs.")


if __name__ == "__main__":
    main()