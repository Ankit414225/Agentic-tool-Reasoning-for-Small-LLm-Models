"""
balance_data.py -- match the training hop distribution to the benchmark.

    python balance_data.py

The dev split is exactly 18 x 2-hop, 18 x 3-hop, 18 x 4-hop -- deliberately
uniform. MuSiQue's train split is not: it skews heavily to 2-hop, and dataset.py
also mixes in HotpotQA which is 100% 2-hop. Training on that distribution while
being evaluated on a uniform one wastes a third of the score.

This rewrites data/train.jsonl and data/val.jsonl with equal counts per hop
depth, and drops sources that cannot supply deep questions.

Run AFTER dataset.py. Keeps a backup at data/train.raw.jsonl.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from collections import Counter, defaultdict

DATA = "data"
HOPS_WANTED = (2, 3, 4)
SEED = 0


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def report(label: str, rows: list[dict]) -> None:
    hc = Counter(r.get("hops") for r in rows)
    sc = Counter(r.get("source") for r in rows)
    print(f"\n{label}: {len(rows)} rows")
    print("  hops   :", dict(sorted(hc.items(), key=lambda x: (x[0] is None, x[0]))))
    print("  sources:", dict(sc))


def balance(rows: list[dict], rng: random.Random) -> list[dict]:
    by_hop: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        h = r.get("hops")
        if h in HOPS_WANTED:
            by_hop[h].append(r)

    missing = [h for h in HOPS_WANTED if not by_hop.get(h)]
    if missing:
        print(f"\n!! no {missing}-hop rows available at all.")
        print("   MuSiQue is the only source with 3/4-hop questions -- check that")
        print("   it loaded in dataset.py, and that --min_hops did not filter them.")

    avail = {h: len(v) for h, v in sorted(by_hop.items())}
    print(f"\navailable per hop: {avail}")
    if not by_hop:
        return []

    n = min(len(v) for v in by_hop.values())
    print(f"balancing to {n} per hop depth "
          f"(limited by {min(by_hop, key=lambda h: len(by_hop[h]))}-hop)")

    out = []
    for h in sorted(by_hop):
        pool = by_hop[h]
        rng.shuffle(pool)
        # The benchmark is 100% MuSiQue, so prefer MuSiQue rows within each
        # bucket and use HotpotQA only to top up. Same hop count, but MuSiQue's
        # question phrasing and distractor style match what we're scored on.
        pool.sort(key=lambda r: 0 if r.get("source") == "musique" else 1)
        out.extend(pool[:n])
    rng.shuffle(out)
    return out


def main() -> None:
    rng = random.Random(SEED)

    for name in ("train", "val"):
        path = os.path.join(DATA, f"{name}.jsonl")
        if not os.path.exists(path):
            print(f"{path} not found -- run dataset.py first")
            continue

        rows = load(path)
        report(f"{name}.jsonl BEFORE", rows)

        raw = os.path.join(DATA, f"{name}.raw.jsonl")
        if not os.path.exists(raw):
            shutil.copy(path, raw)
            print(f"  backup -> {raw}")

        out = balance(rows, rng)
        if not out:
            print(f"  nothing to write for {name}, leaving it alone")
            continue

        save(path, out)
        report(f"{name}.jsonl AFTER", out)

    print("\ndone. the training mix now matches the benchmark's 2/3/4-hop split.")
    print("\nNote: 4-hop is the scarce bucket in MuSiQue, so it caps the total.")
    print("If you got fewer than ~3000 train rows, rerun dataset.py with a bigger")
    print("--n_train (e.g. 30000) so nothing is trimmed before balancing.")
    print("RL needs far fewer prompts than SFT -- 3-5k is plenty for 400 steps.")


if __name__ == "__main__":
    main()