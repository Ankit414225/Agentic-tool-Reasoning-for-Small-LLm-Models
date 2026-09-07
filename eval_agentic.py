"""
eval.py -- run a checkpoint against the frozen env. Same env.py as training.

  python eval.py --model Qwen/Qwen3-1.7B --data data/val.jsonl
  python eval.py --model ckpt/final --data data/dev_benchmark.jsonl --dump runs/x.jsonl

Run this on the BASE model first. You need that number before you touch RL,
otherwise you cannot tell whether anything you did helped.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from transformers import AutoTokenizer
from vllm import LLM

from env import BM25Tool
from reward import best_over_aliases, em_score, f1_score
from rollout import Task, generate_episodes, rollout_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)   # greedy for eval
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--dump", default="")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=0.85,
              enable_prefix_caching=True, max_model_len=4096)

    tasks = []
    with open(a.data) as f:
        for line in f:
            d = json.loads(line)
            tasks.append(Task(d["question"], d["answer"], d.get("aliases"),
                              d.get("docs"), d.get("corpus_id")))
    if a.limit:
        tasks = tasks[: a.limit]

    groups = generate_episodes(llm, tok, tasks, BM25Tool(),
                               group_size=a.n_samples, temperature=a.temperature)

    f1s, ems, recs = [], [], []
    stop = Counter()
    for g, t in zip(groups, tasks):
        golds = [t.gold] + (t.aliases or [])
        ep = g[0]
        pred = ep.answer or ""
        f1 = best_over_aliases(pred, golds, f1_score)
        em = best_over_aliases(pred, golds, em_score)
        f1s.append(f1); ems.append(em); stop[ep.stop_reason] += 1
        recs.append({"q": t.question, "gold": t.gold, "pred": pred,
                     "f1": round(f1, 3), "calls": ep.n_tool_calls,
                     "stop": ep.stop_reason, "turns": ep.text_log})

    n = len(f1s)
    print(json.dumps({
        "model": a.model, "n": n,
        "F1": round(sum(f1s) / n, 4), "EM": round(sum(ems) / n, 4),
        "stop_reasons": dict(stop),
        **{k: round(v, 3) for k, v in rollout_stats(groups).items()},
    }, indent=2))

    if a.dump:
        with open(a.dump, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"\nwrote {a.dump} -- READ 20 OF THESE BY HAND. "
              "The loss curve will not tell you the model is cheating.")


if __name__ == "__main__":
    main()
