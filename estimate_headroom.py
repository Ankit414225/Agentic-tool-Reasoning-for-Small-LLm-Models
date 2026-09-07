"""
estimate_headroom.py -- how much is multi-hop querying actually worth?

    python estimate_headroom.py

Builds a ladder of retrieval strategies, none of which use a language model:

  oracle       is the answer anywhere in the 20 candidate docs?   (hard ceiling)
  3 rounds     question -> expand on retrieved titles -> expand again
  2 rounds     question -> expand on retrieved titles
  1 round      question only                    (what a non-chaining model does)

The gap between "1 round" and "3 rounds" is the headroom RL is chasing. If that
gap is large, the task rewards learning to chain queries and the 60% target is
reachable. If it's small, BM25 itself is the bottleneck and no amount of RL on
the policy will fix it.

Expansion here is deliberately dumb -- it just appends retrieved document titles
to the query. A trained model should beat it comfortably, so treat "3 rounds" as
a soft floor for what good chaining can reach, not a ceiling.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict

sys.path.insert(0, ".")

from env import BM25Tool

TOP_K = 3
STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is",
    "was", "were", "which", "who", "what", "where", "when", "how", "that",
    "this", "with", "by", "from", "as", "it", "its", "be", "been", "are",
    "city", "state", "country", "county", "person", "did", "does", "do",
}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


def contains_answer(docs: list[str], answer: str) -> bool:
    a = norm(answer).strip()
    if not a:
        return False
    return any(a in norm(d) for d in docs)


def titles_of(docs: list[str]) -> list[str]:
    out = []
    for d in docs:
        first = d.splitlines()[0]
        out.append(first[len("Title: "):].strip() if first.startswith("Title: ")
                   else first[:60].strip())
    return out


def content_terms(question: str) -> list[str]:
    return [w for w in norm(question).split() if w not in STOP and len(w) > 2]


def bridge_entities(docs: list[str], question: str) -> list[str]:
    """Pull candidate bridge entities out of retrieved PASSAGE TEXT.

    This is the crux of multi-hop retrieval: the entity you need for the next
    query sits in the body of what you just retrieved, not in its title.
    Expanding on titles alone can never chain -- it just re-finds the same doc.

    Heuristic: capitalised multi-word spans not already in the question.
    A trained model does this far better; this is a floor.
    """
    qwords = set(norm(question).split())
    found: list[str] = []
    for d in docs:
        body = d.split("\n", 1)[1] if "\n" in d else d
        for span in re.findall(r"\b([A-Z][\w']*(?:\s+[A-Z][\w']*)*)", body):
            span = span.strip()
            if len(span) < 4:
                continue
            toks = set(norm(span).split())
            if toks & qwords:          # already in the question, not a bridge
                continue
            if span not in found:
                found.append(span)
    return found


def iterative(tool: BM25Tool, row: dict, rounds: int) -> list[str]:
    """Retrieve for `rounds` turns, expanding the query on retrieved titles."""
    docs, cid = row["docs"], row["id"]
    seen: dict[str, None] = {}
    q = row["question"]
    used_titles: set[str] = set()

    for _ in range(rounds):
        hits = tool.search(q, docs=docs, corpus_id=cid, top_k=TOP_K)
        for h in hits:
            seen.setdefault(h, None)

        # Build the next query from bridge entities found in the retrieved
        # TEXT (plus titles as a fallback), skipping ones already used.
        cands = bridge_entities(hits, row["question"]) + titles_of(hits)
        fresh = [t for t in cands if t not in used_titles]
        if not fresh:
            break
        used_titles.update(fresh[:3])
        q = " ".join(fresh[:3] + content_terms(row["question"])[:6])

    return list(seen)


def main() -> None:
    for path in ("data/dev_benchmark.jsonl", "data/val.jsonl"):
        try:
            rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        except FileNotFoundError:
            print(f"{path} not found -- run dataset.py first")
            continue
        if not rows:
            continue

        tool = BM25Tool()
        tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        per_hop: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])

        for r in rows:
            ans, hops = r["answer"], r.get("hops")

            results = {
                "oracle (all 20 docs)": contains_answer(r["docs"], ans),
                "3 rounds": contains_answer(iterative(tool, r, 3), ans),
                "2 rounds": contains_answer(iterative(tool, r, 2), ans),
                "1 round": contains_answer(iterative(tool, r, 1), ans),
            }
            for k, v in results.items():
                tally[k][0] += v
                tally[k][1] += 1
                per_hop[(k, hops)][0] += v
                per_hop[(k, hops)][1] += 1

        print(f"\n{'=' * 60}\n{path}  ({len(rows)} rows)\n{'=' * 60}")
        order = ["oracle (all 20 docs)", "3 rounds", "2 rounds", "1 round"]
        for k in order:
            hit, n = tally[k]
            bar = "#" * int(40 * hit / n)
            print(f"  {k:22} {hit:4}/{n:<4} {100*hit/n:5.1f}%  {bar}")

        hop_vals = sorted({h for (_, h) in per_hop if h is not None})
        if hop_vals:
            print(f"\n  {'strategy':22}" + "".join(f"{h}-hop".rjust(9)
                                                   for h in hop_vals))
            for k in order:
                line = f"  {k:22}"
                for h in hop_vals:
                    hit, n = per_hop[(k, h)]
                    line += f"{(100*hit/n if n else 0):8.0f}%"
                print(line)

        one = tally["1 round"][0] / tally["1 round"][1]
        three = tally["3 rounds"][0] / tally["3 rounds"][1]
        orc = tally["oracle (all 20 docs)"][0] / tally["oracle (all 20 docs)"][1]
        print(f"\n  chaining headroom (1 -> 3 rounds): {100*(three-one):+.1f} points")
        print(f"  remaining to oracle:               {100*(orc-three):+.1f} points")
        if three - one > 0.12:
            print("\n  -> Large. The task genuinely rewards learning to chain")
            print("     queries, which is exactly what RL can teach. Good sign.")
        else:
            print("\n  -> Small. BM25 recall may be the bottleneck rather than")
            print("     query formulation. Consider raising TOP_K before blaming")
            print("     the policy, and check whether the gold docs are reachable.")
        if orc < 0.9:
            print(f"\n  !! oracle is only {100*orc:.0f}% -- for the rest, the answer")
            print("     string never appears literally in any candidate doc.")
            print("     Those are paraphrase cases (or my substring check is too")
            print("     strict); they cap what ANY retrieval strategy can score.")


if __name__ == "__main__":
    main()