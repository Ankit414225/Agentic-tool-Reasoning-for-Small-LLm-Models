"""
ablate_retrieval.py -- is the gain from CHAINING or just from seeing more docs?

    python ablate_retrieval.py

estimate_headroom.py showed 1 round (top_k=3, 3 docs seen) = 33% and 3 rounds
(top_k=3, up to 9 docs seen) = 59%. But those two strategies do not see the same
number of documents, so the comparison confounds two different things:

  (a) chaining      -- using what you retrieved to build a better next query
  (b) volume        -- simply looking at more documents

This holds the document budget FIXED and varies only the strategy:

  wide    one search, top_k = budget          (no chaining, all volume)
  chain   budget/3 searches at top_k = 3      (chaining, same volume)

If `wide` matches `chain`, raise TOP_K in env.py and the task stops being about
multi-hop reasoning. If `chain` clearly wins at equal budget, the gain is real
reasoning and RL is the right lever.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

sys.path.insert(0, ".")

from env import BM25Tool
from estimate_headroom import bridge_entities, contains_answer, content_terms, titles_of

BUDGETS = (3, 6, 9, 12, 15)


def wide(tool: BM25Tool, row: dict, budget: int) -> list[str]:
    """One search, big top_k. No chaining at all."""
    return tool.search(row["question"], docs=row["docs"],
                       corpus_id=row["id"], top_k=budget)


def chain(tool: BM25Tool, row: dict, budget: int, per_round: int = 3) -> list[str]:
    """budget/per_round searches, expanding on entities found in retrieved text."""
    rounds = max(1, budget // per_round)
    seen: dict[str, None] = {}
    q = row["question"]
    used: set[str] = set()
    for _ in range(rounds):
        hits = tool.search(q, docs=row["docs"], corpus_id=row["id"],
                           top_k=per_round)
        for h in hits:
            seen.setdefault(h, None)
        cands = bridge_entities(hits, row["question"]) + titles_of(hits)
        fresh = [t for t in cands if t not in used]
        if not fresh:
            break
        used.update(fresh[:3])
        q = " ".join(fresh[:3] + content_terms(row["question"])[:6])
    return list(seen)


def main() -> None:
    for path in ("data/dev_benchmark.jsonl", "data/val.jsonl"):
        try:
            rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        except FileNotFoundError:
            print(f"{path} not found")
            continue
        if not rows:
            continue

        tool = BM25Tool()
        res: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])

        for r in rows:
            for b in BUDGETS:
                for name, fn in (("wide", wide), ("chain", chain)):
                    got = contains_answer(fn(tool, r, b), r["answer"])
                    res[(name, b)][0] += got
                    res[(name, b)][1] += 1

        print(f"\n{'=' * 64}\n{path}  ({len(rows)} rows)\n{'=' * 64}")
        print(f"  {'docs seen':<17}" + "".join(f"{b:>9}" for b in BUDGETS))
        print("  " + "-" * 58)
        for name, label in (("wide", "wide (1 search)"), ("chain", "chain (k=3)")):
            line = f"  {label:<17}"
            for b in BUDGETS:
                hit, n = res[(name, b)]
                line += f"{100*hit/n:>8.1f}%"
            print(line)

        line = f"  {'difference':<17}"
        for b in BUDGETS:
            w = res[("wide", b)][0] / res[("wide", b)][1]
            c = res[("chain", b)][0] / res[("chain", b)][1]
            line += f"{100*(c-w):>+8.1f} "
        print(line)

        # verdict at the largest shared budget
        b = BUDGETS[-1]
        w = res[("wide", b)][0] / res[("wide", b)][1]
        c = res[("chain", b)][0] / res[("chain", b)][1]
        print(f"\n  at {b} docs seen: wide {100*w:.1f}%  vs  chain {100*c:.1f}%")
        if c - w > 0.08:
            print("  -> CHAINING WINS at equal budget. The gain is real multi-hop")
            print("     reasoning, not volume. RL on query formulation is the")
            print("     right lever, and TOP_K should stay small.")
        elif w - c > 0.08:
            print("  -> VOLUME WINS. Just raise TOP_K in env.py; chaining is not")
            print("     buying anything here.")
        else:
            print("  -> ROUGHLY EQUAL. Volume explains most of the headroom.")
            print("     Raise TOP_K (cheap, no training needed), then re-measure")
            print("     how much chaining still adds on top.")
        print("\n  Note: bigger top_k costs context. Check the token budget in")
        print("  env.py before raising it -- see inspect_bench.py.")


if __name__ == "__main__":
    main()