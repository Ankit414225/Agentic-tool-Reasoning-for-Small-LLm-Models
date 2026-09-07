"""
dataset.py -- benchmark row schema -> Task rows for GRPO.

Schema confirmed from the team's eval.py:

    row['id']
    row['question']
    row['answer']
    row['context']            -> [{'title': str, 'paragraph_text': str}, ...]
    row['metadata']['hops']

This is MuSiQue's schema, which means:
  * candidate docs are PER QUESTION (row['context']), not one global corpus
  * we can filter training data by hop count
  * MuSiQue's train split is a drop-in proxy training set, same fields

Build everything:
    python dataset.py --bench YashBhamare123/tools-benchmark --out data
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections.abc import Iterator
from typing import Any

from datasets import load_dataset


# The eval harness formats passages as "Title: X\n<body>". We match that byte
# for byte so the model sees identical passage formatting at train and eval.
def format_doc(title: str, body: str) -> str:
    return f"Title: {title}\n{body}".strip()


def docs_from_context(context) -> list[str]:
    """Handle every shape these datasets show up in."""
    out = []
    for p in context:
        if isinstance(p, dict):
            title = p.get("title") or p.get("paragraph_title") or ""
            body = (p.get("paragraph_text") or p.get("text")
                    or p.get("paragraph") or "")
            if isinstance(body, list):          # HotpotQA-style sentence lists
                body = " ".join(body)
            out.append(format_doc(title, body))
        elif isinstance(p, (list, tuple)) and len(p) == 2:   # (title, sentences)
            body = " ".join(p[1]) if isinstance(p[1], list) else str(p[1])
            out.append(format_doc(str(p[0]), body))
        else:
            out.append(str(p))
    return [d for d in out if d]


def convert(row, source: str) -> dict | None:
    q = row.get("question")
    a = row.get("answer")
    if not q or not a:
        return None
    ctx = row.get("context") or row.get("paragraphs") or []
    docs = docs_from_context(ctx)
    if len(docs) < 2:
        return None
    meta = row.get("metadata") or {}
    hops = meta.get("hops") or row.get("hops")
    if hops is None and row.get("question_decomposition"):
        hops = len(row["question_decomposition"])
    return {
        "id": str(row.get("id", "")),
        "question": q.strip(),
        "answer": str(a).strip(),
        "aliases": [x for x in (row.get("answer_aliases") or []) if x],
        "docs": docs,
        "hops": hops,
        "source": source,
    }


def ngrams(s: str, n: int = 8) -> set[str]:
    w = re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def iter_rows(ds: Any) -> Iterator[dict]:
    """Yield dataset rows as plain dicts.

    Purely for the type checker. Pylance types `for r in hf_dataset` as yielding
    a Dataset rather than a row, so `r["id"]` looks like slice indexing and it
    reports reportArgumentType / reportCallIssue. At runtime the rows are
    already dicts; this makes that explicit and clears the warnings without
    scattering `# type: ignore` around.
    """
    for row in ds:
        yield dict(row)


def first_row(ds: Any) -> dict:
    return dict(ds[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="YashBhamare123/tools-benchmark")
    ap.add_argument("--out", default="data")
    ap.add_argument("--n_train", type=int, default=12000)
    ap.add_argument("--min_hops", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    # ---------- 1. the public dev split: eval only, NEVER trained on ----------
    bench = []
    try:
        b = load_dataset(a.bench)
        split = "train" if "train" in b else list(b.keys())[0]
        print(f"benchmark split '{split}': {len(b[split])} rows")
        print(f"columns: {b[split].column_names}")
        print("\n--- first row ---")
        r0 = first_row(b[split])
        for k, v in r0.items():
            print(f"{k}: {str(v)[:220]}")
        print("--- end ---\n")
        for r in iter_rows(b[split]):
            c = convert(r, "benchmark")
            if c:
                bench.append(c)
        with open(os.path.join(a.out, "dev_benchmark.jsonl"), "w") as f:
            for r in bench:
                f.write(json.dumps(r) + "\n")
        print(f"wrote dev_benchmark.jsonl: {len(bench)}  (EVAL ONLY)")
        nd = [len(r["docs"]) for r in bench]
        cl = [len(d) for r in bench for d in r["docs"]]
        al = [len(r["answer"].split()) for r in bench]
        print(f"  docs/question: min {min(nd)} med {sorted(nd)[len(nd)//2]} max {max(nd)}")
        print(f"  doc chars:     med {sorted(cl)[len(cl)//2]} max {max(cl)}")
        print(f"  answer words:  med {sorted(al)[len(al)//2]} max {max(al)}"
              f"   <-- if med > 3, tell the data team")
        hp = {}
        for r in bench:
            hp[r["hops"]] = hp.get(r["hops"], 0) + 1
        print(f"  hops: {hp}")
    except Exception as e:
        print(f"!! could not load {a.bench}: {e}")
        print("   get access to it before training -- everything below is guesswork without it")

    # ---------- 2. proxy training set from MuSiQue (same schema) ----------
    train = []
    try:
        m = load_dataset("dgslibisey/MuSiQue", split="train")
        print(f"\nMuSiQue train: {len(m)} rows")
        for r in iter_rows(m):
            if not r.get("answerable", True):
                continue
            c = convert(r, "musique")
            if c and (c["hops"] is None or c["hops"] >= a.min_hops):
                train.append(c)
    except Exception as e:
        print(f"!! MuSiQue unavailable: {e}")

    # HotpotQA as filler for volume
    try:
        h = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train",
                         trust_remote_code=True)
        for r in iter_rows(h.select(range(min(15000, len(h))))):
            ctx = r["context"]
            row = {"id": r["id"], "question": r["question"], "answer": r["answer"],
                   "context": list(zip(ctx["title"], ctx["sentences"]))}
            c = convert(row, "hotpotqa")
            if c:
                c["hops"] = 2
                train.append(c)
    except Exception as e:
        print(f"!! HotpotQA unavailable: {e}")

    # ---------- 3. decontaminate against the dev split ----------
    if bench:
        seen = set()
        for r in bench:
            seen |= ngrams(r["question"])
        before = len(train)
        train = [r for r in train if not (ngrams(r["question"]) & seen)]
        print(f"\ndecontamination: dropped {before - len(train)} / {before}")
    else:
        print("\n!! DECONTAMINATION SKIPPED -- no dev split loaded. Do not ship like this.")

    # ---------- 4. drop rows that give no training signal ----------
    train = [r for r in train
             if r["answer"] and len(r["answer"].split()) <= 8
             and r["answer"].lower() not in ("yes", "no")]

    random.shuffle(train)
    train = train[: a.n_train]
    n_val = min(500, len(train) // 10)
    for name, part in (("val.jsonl", train[:n_val]), ("train.jsonl", train[n_val:])):
        with open(os.path.join(a.out, name), "w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {name}: {len(part)}")

    src = {}
    for r in train:
        src[r["source"]] = src.get(r["source"], 0) + 1
    print("by source:", src)


if __name__ == "__main__":
    main()
