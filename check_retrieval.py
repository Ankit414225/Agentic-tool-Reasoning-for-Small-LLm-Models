"""
check_retrieval.py -- can BM25 find the answer in one shot?

    python check_retrieval.py

Run after dataset.py. The dev_benchmark number is the one that matters:
  10-40%  healthy. one search is not enough, so multi-hop querying earns reward
  > 70%   task is mostly single-hop; RL will gain little, raise it with the team
  < 5%    retrieval may be broken, or the candidate sets are too large
"""

import json
import sys

sys.path.insert(0, ".")

from env import BM25Tool

for path in ["data/dev_benchmark.jsonl", "data/train.jsonl", "data/val.jsonl"]:
    try:
        rows = [json.loads(l) for l in open(path, encoding="utf-8")][:200]
    except FileNotFoundError:
        print(f"{path:32} not found (run dataset.py first)")
        continue
    if not rows:
        print(f"{path:32} empty")
        continue

    tool = BM25Tool()
    hit = 0
    by_hops: dict = {}
    for r in rows:
        got = tool.search(r["question"], docs=r["docs"], corpus_id=r["id"])
        found = any(r["answer"].lower() in d.lower() for d in got)
        hit += found
        h = r.get("hops")
        d = by_hops.setdefault(h, [0, 0])
        d[0] += found
        d[1] += 1

    print(f"{path:32} {hit:3}/{len(rows):3}  ({100*hit/len(rows):5.1f}%)")
    if len(by_hops) > 1:
        for h in sorted(by_hops, key=lambda x: (x is None, x)):
            f, n = by_hops[h]
            print(f"    {str(h) + '-hop':10} {f:3}/{n:3}  ({100*f/n:5.1f}%)")