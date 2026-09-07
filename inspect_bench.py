"""
inspect_bench.py -- what is actually in the benchmark dev split?

    python inspect_bench.py

Prints the category / subcategory / hop distribution and the size stats that
decide MAX_TURNS, TOP_K and PASSAGE_MAX_CHARS in env.py.
"""

import sys
from collections import Counter

sys.path.insert(0, ".")

from datasets import load_dataset

NAME = "YashBhamare123/tools-benchmark"

b = load_dataset(NAME)
split = "public" if "public" in b else list(b.keys())[0]
d = b[split]
rows = [dict(r) for r in d]
print(f"{NAME}  split={split}  rows={len(rows)}\n")


def dist(label, values):
    c = Counter(values)
    print(f"--- {label} ---")
    for k, v in c.most_common():
        bar = "#" * int(40 * v / len(rows))
        print(f"  {str(k):24} {v:4}  {100*v/len(rows):5.1f}%  {bar}")
    print()


dist("category", [r.get("category") for r in rows])
dist("subcategory", [r.get("subcategory") for r in rows])
dist("metadata.hops", [(r.get("metadata") or {}).get("hops") for r in rows])
dist("metadata.dataset", [(r.get("metadata") or {}).get("dataset") for r in rows])

# ---- size stats: these set the token budget ----
n_docs = [len(r["context"]) for r in rows]
doc_chars = [len(p["paragraph_text"]) for r in rows for p in r["context"]]
ans_words = [len(str(r["answer"]).split()) for r in rows]
q_words = [len(str(r["question"]).split()) for r in rows]


def stats(name, xs, note=""):
    xs = sorted(xs)
    print(f"  {name:16} min {xs[0]:5}  med {xs[len(xs)//2]:5}  "
          f"p90 {xs[int(len(xs)*0.9)]:5}  max {xs[-1]:6}   {note}")


print("--- size stats ---")
stats("docs/question", n_docs)
stats("doc chars", doc_chars, "<- sets PASSAGE_MAX_CHARS")
stats("answer words", ans_words, "<- if med > 3, F1 will punish verbosity")
stats("question words", q_words)

# ---- turn budget ----
max_hops = max((r.get("metadata") or {}).get("hops") or 0 for r in rows)
print(f"\n--- turn budget ---")
print(f"  deepest question is {max_hops}-hop")
print(f"  minimum turns needed: {max_hops} searches + 1 answer = {max_hops + 1}")
print(f"  recommended MAX_TURNS: {max_hops + 3}  (slack for bad queries)")

import env
print(f"  env.py currently has MAX_TURNS = {env.MAX_TURNS}", end="")
print("   <-- TOO LOW, FIX THIS" if env.MAX_TURNS < max_hops + 1 else "   ok")

#  context budget at that many turns 
tok_per_passage = sorted(doc_chars)[len(doc_chars)//2] / 4
obs = env.TOP_K * min(tok_per_passage, env.PASSAGE_MAX_CHARS / 4) + 20
total = 400 + (max_hops + 3) * (obs + 80)
print(f"\n--- context budget with TOP_K={env.TOP_K}, "
      f"PASSAGE_MAX_CHARS={env.PASSAGE_MAX_CHARS} ---")
print(f"  ~{obs:.0f} tokens per observation")
print(f"  ~{total:.0f} tokens for a full {max_hops + 3}-turn episode")
print(f"  env.py MAX_TOTAL_TOKENS = {env.MAX_TOTAL_TOKENS}", end="")
print("   <-- TOO LOW" if total > env.MAX_TOTAL_TOKENS else "   ok")

# non-multihop examples
others = [r for r in rows if r.get("category") != "multi_hop_reasoning"]
if others:
    print(f"\n--- {len(others)} non-multihop rows, first 3 ---")
    for r in others[:3]:
        print(f"  [{r.get('category')}/{r.get('subcategory')}] {r['question'][:90]}")
        print(f"      answer: {r['answer']!r}  docs: {len(r['context'])}")
else:
    print("\n  every row is multi_hop_reasoning -- no no-tool cases in the dev split")

print(f"\n--- eval noise warning ---")
print(f"  {len(rows)} rows means 1 question = {100/len(rows):.1f}% F1.")
print(f"  Differences under ~{300/len(rows):.0f}% between runs are noise.")
print(f"  Use n_samples>1 and compare on your own val split too.")