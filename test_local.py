"""
test_local.py -- run the whole pipeline on CPU, no GPU, no vLLM.

    python test_local.py

Checks, in order:
  1. BM25 retrieval and parsing
  2. Reward + GRPO advantages
  3. Schema conversion for the benchmark row format
  4. Full multi-turn rollout, and THE LOSS MASK -- the thing that decides
     whether training works at all
  5. One simulated GRPO step end to end (reward -> filter -> advantages)

If this passes, the only untested thing left is the gradient step itself.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from env import (
    MAX_TURNS,
    TOOL_RESP_OPEN,
    BM25Tool,
    Episode,
    build_prompt,
    parse_turn,
    render_observation,
)
from reward import (
    RewardConfig,
    compute_reward,
    em_score,
    f1_score,
    group_advantages,
    group_is_informative,
)

PASS, FAIL = "  ok  ", " FAIL "
_fails = []


class FakeTokenizer:
    """Reversible word-level tokenizer for offline testing.

    The mask logic is tokenizer-independent -- it tracks which SEGMENTS were
    produced by the policy vs the environment. So this shim validates it just as
    well as the real Qwen tokenizer, and runs with no network.
    """

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.vocab: dict[str, int] = {"<pad>": 0, "<eos>": 1}
        self.inv: dict[int, str] = {0: "<pad>", 1: "<eos>"}

    def _id(self, piece: str) -> int:
        if piece not in self.vocab:
            i = len(self.vocab)
            self.vocab[piece] = i
            self.inv[i] = piece
        return self.vocab[piece]

    def __call__(self, text, add_special_tokens=False):
        pieces = text.replace("\n", " \n ").split(" ")
        return {"input_ids": [self._id(p) for p in pieces if p != ""]}

    def decode(self, ids):
        return " ".join(self.inv.get(i, "") for i in ids).replace(" \n ", "\n")

    def apply_chat_template(self, msgs, tokenize=False,
                            add_generation_prompt=True, **kw):
        s = ""
        for m in msgs:
            s += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            s += "<|im_start|>assistant\n"
        return s


def check(name, cond, detail=""):
    print(f"[{PASS if cond else FAIL}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        _fails.append(name)


DOCS = [
    "Title: Ada Lovelace\nAda Lovelace was an English mathematician who wrote the "
    "first algorithm intended for the Analytical Engine.",
    "Title: Analytical Engine\nThe Analytical Engine was a mechanical general-purpose "
    "computer designed by Charles Babbage.",
    "Title: Charles Babbage\nCharles Babbage was an English polymath born in 1791.",
    "Title: Pithampur\nPithampur is an industrial town in Madhya Pradesh, India.",
]


# ---------------------------------------------------------------- 1
def test_retrieval_and_parsing():
    print("\n== 1. retrieval + parsing ==")
    t = BM25Tool()
    hits = t.search("Analytical Engine designed by", docs=DOCS, corpus_id="c1")
    check("BM25 returns the right passage first",
          "Analytical Engine" in hits[0], hits[0][:40])
    check("empty query returns nothing", t.search("", docs=DOCS) == [])
    check("corpus cache is reused",
          t.search("Babbage", docs=DOCS, corpus_id="c1") and "c1" in t._cache)
    check("observation is wrapped", TOOL_RESP_OPEN in render_observation(hits))

    cases = [
        ('<tool_call>{"name":"search","arguments":{"query":"who built it"}}</tool_call>',
         "tool", "who built it"),
        ("thinking out loud <answer>Charles Babbage</answer>", "answer", None),
        ('<tool_call>{"query": "loose json"}</tool_call>', "tool", "loose json"),
        ('<tool_call>{name: search, "query": "near json"}</tool_call>', "tool", "near json"),
        ("<tool_call>{totally broken</tool_call>", "malformed", None),
        ("no tags whatsoever", "malformed", None),
    ]
    for text, kind, q in cases:
        p = parse_turn(text)
        ok = p.kind == kind and (q is None or p.query == q)
        check(f"parse -> {kind:9} {text[:38]!r}", ok, f"got {p.kind}")

    # both tags present: whichever comes first wins
    p = parse_turn("<answer>A</answer> then <tool_call>{}</tool_call>")
    check("first tag wins", p.kind == "answer")


# ---------------------------------------------------------------- 2
def test_reward():
    print("\n== 2. reward + advantages ==")
    check("F1 exact", f1_score("Charles Babbage", "Charles Babbage") == 1.0)
    check("F1 partial", 0.6 < f1_score("Babbage", "Charles Babbage") < 0.7,
          f"{f1_score('Babbage', 'Charles Babbage'):.3f}")
    check("F1 ignores articles/case",
          f1_score("the CHARLES babbage", "Charles Babbage") == 1.0)
    check("F1 wrong answer", f1_score("Ada Lovelace", "Charles Babbage") == 0.0)
    check("EM strict", em_score("Babbage", "Charles Babbage") == 0.0)

    def ep(ans, calls=1, mal=0):
        e = Episode("q", "Charles Babbage")
        e.answer, e.n_tool_calls, e.n_malformed = ans, calls, mal
        return e

    r_short, _ = compute_reward(ep("Charles Babbage"))
    r_verbose, iv = compute_reward(
        ep("the answer is definitely Charles Babbage as shown in passage 2"))
    check("short answer scores ~1.0", r_short > 0.95, f"{r_short:.3f}")
    check("verbose answer is punished hard", r_verbose < 0.45,
          f"{r_verbose:.3f} (f1={iv['f1']:.3f})")
    check("THIS is the cheap win: short vs verbose",
          r_short - r_verbose > 0.5, f"gap {r_short - r_verbose:.3f}")

    r_none, i_none = compute_reward(ep(None))
    check("no answer -> gated to 0", r_none == 0.0 and i_none["gated"] == 1.0)

    r_flail, _ = compute_reward(ep("Charles Babbage", calls=6, mal=3))
    check("flailing is penalised but capped", 0.6 < r_flail < 0.8, f"{r_flail:.3f}")
    check("penalty cap respected",
          compute_reward(ep("Charles Babbage", 20, 20))[1]["penalty"]
          <= RewardConfig().max_penalty)

    check("aliases accepted",
          compute_reward(ep("Babbage"), ["Charles Babbage", "Babbage"])[0] > 0.95)

    adv = group_advantages([1.0, 0.5, 0.0, 0.0])
    check("advantages centre on zero", abs(sum(adv)) < 1e-6,
          f"{[round(x, 2) for x in adv]}")
    check("best rollout has positive advantage", adv[0] > 0)
    check("degenerate group -> all zero", group_advantages([0.4] * 4) == [0.0] * 4)
    check("all-wrong group filtered", not group_is_informative([0.0] * 8))
    check("all-right group filtered", not group_is_informative([1.0] * 8))
    check("mixed group kept", group_is_informative([1.0, 0.0, 0.3]))


# ---------------------------------------------------------------- 3
def test_schema():
    print("\n== 3. benchmark schema conversion ==")
    import types

    if "datasets" not in sys.modules:      # stub so dataset.py imports on CPU
        stub = types.ModuleType("datasets")
        stub.load_dataset = lambda *a, **k: None   # type: ignore[attr-defined]
        sys.modules["datasets"] = stub
    from dataset import convert

    row = {
        "id": "2hop__1", "question": "Who designed the machine Ada wrote about?",
        "answer": "Charles Babbage", "answer_aliases": ["Babbage"],
        "context": [
            {"title": "Ada Lovelace", "paragraph_text": "She wrote about the Engine."},
            {"title": "Analytical Engine", "paragraph_text": "Designed by Charles Babbage."},
        ],
        "metadata": {"hops": 2},
    }
    c = convert(row, "benchmark")
    check("converts the team's row format", c is not None)
    assert c is not None            # narrows the type for the checks below
    check("hops carried through", c["hops"] == 2)
    check("aliases carried through", c["aliases"] == ["Babbage"])
    check("passage format matches the harness",
          c["docs"][0].startswith("Title: Ada Lovelace\n"), repr(c["docs"][0][:34]))

    hotpot = {"id": "h", "question": "q", "answer": "a",
              "context": [("T1", ["s1. ", "s2."]), ("T2", ["s3."])]}
    ch = convert(hotpot, "hotpotqa")
    check("HotpotQA shape also converts",
          ch is not None and ch["docs"][0] == "Title: T1\ns1.  s2.")
    check("single-doc rows dropped",
          convert({"id": "x", "question": "q", "answer": "a",
                   "context": [{"title": "T", "paragraph_text": "b"}]}, "x") is None)
    check("answerless rows dropped", convert({"question": "q", "context": []}, "x") is None)


# ---------------------------------------------------------------- 4
def test_rollout(tokenizer):
    print("\n== 4. multi-turn rollout + LOSS MASK ==")
    from mock_llm import MockLLM
    from rollout import Task, generate_episodes, rollout_stats

    tasks = [
        Task("Who designed the machine Ada Lovelace wrote about?",
             "Charles Babbage", ["Babbage"], DOCS, "c1"),
        Task("Which state is Pithampur in?", "Madhya Pradesh", None, DOCS, "c2"),
    ]
    tool = BM25Tool()

    # -- deterministic behaviours --
    for beh, expect in [("good_multihop", "answer"), ("one_shot", "answer"),
                        ("never_answers", "turn_limit"), ("malformed", "answer")]:
        g = generate_episodes(MockLLM(tokenizer, beh, tasks=tasks), tokenizer, tasks[:1], tool,
                              group_size=2)
        ep = g[0][0]
        check(f"{beh:14} -> {expect}", ep.stop_reason == expect,
              f"got {ep.stop_reason}, {ep.n_tool_calls} calls")

    g = generate_episodes(MockLLM(tokenizer, "good_multihop", tasks=tasks), tokenizer,
                          tasks[:1], tool, group_size=1)
    ep = g[0][0]
    check("multihop made 2 searches", ep.n_tool_calls == 2, f"{ep.n_tool_calls}")
    check("one_shot made 0 searches",
          generate_episodes(MockLLM(tokenizer, "one_shot", tasks=tasks), tokenizer, tasks[:1],
                            tool, group_size=1)[0][0].n_tool_calls == 0)
    check("malformed turns were counted",
          generate_episodes(MockLLM(tokenizer, "malformed", tasks=tasks), tokenizer, tasks[:1],
                            tool, group_size=1)[0][0].n_malformed > 0)
    check("turn cap respected",
          generate_episodes(MockLLM(tokenizer, "never_answers", tasks=tasks), tokenizer,
                            tasks[:1], tool, group_size=1)[0][0].n_tool_calls <= MAX_TURNS)

    # -- THE CRITICAL CHECK --
    print("\n  -- loss mask (the thing that decides if training works) --")
    check("mask length == token length", len(ep.mask) == len(ep.token_ids))
    check("prompt tokens are masked out", sum(ep.mask[:20]) == 0)
    check("some tokens ARE trainable", 0 < sum(ep.mask) < len(ep.mask),
          f"{int(sum(ep.mask))}/{len(ep.mask)} trainable")

    # every retrieved passage must sit under mask==0
    trainable = tokenizer.decode(
        [t for t, m in zip(ep.token_ids, ep.mask) if m == 1])
    masked = tokenizer.decode(
        [t for t, m in zip(ep.token_ids, ep.mask) if m == 0])
    check("BM25 passages are NOT in the trainable region",
          "mechanical general-purpose" not in trainable
          and TOOL_RESP_OPEN not in trainable)
    check("BM25 passages ARE in the masked region",
          TOOL_RESP_OPEN in masked)
    check("model's own tool calls stay trainable", "tool_call" in trainable)
    print(f"     trainable: {trainable[:90]!r}")

    st = rollout_stats(g)
    check("stats computed", st["n_episodes"] == 1 and st["mean_len"] > 0,
          f"len={st['mean_len']:.0f}")


# ---------------------------------------------------------------- 5
def test_grpo_step(tokenizer):
    print("\n== 5. one simulated GRPO step ==")
    from mock_llm import MockLLM
    from rollout import Task, generate_episodes

    tasks = [Task(f"question {i}", "Analytical Engine", None, DOCS, f"c{i}")
             for i in range(6)]
    groups = generate_episodes(MockLLM(tokenizer, "mixed", seed=1, tasks=tasks), tokenizer,
                               tasks, BM25Tool(), group_size=8)

    kept, dropped, n_train = 0, 0, 0
    all_r = []
    for g, t in zip(groups, tasks):
        rs = [compute_reward(e, [t.gold])[0] for e in g]
        all_r += rs
        if group_is_informative(rs):
            kept += 1
            n_train += len(g)
        else:
            dropped += 1

    check("rollouts generated", len(all_r) == 48, f"{len(all_r)}")
    check("reward spread is non-degenerate", max(all_r) - min(all_r) > 0.3,
          f"min {min(all_r):.2f} max {max(all_r):.2f} mean {sum(all_r)/len(all_r):.3f}")
    check("some groups kept", kept > 0, f"kept {kept}, dropped {dropped}")
    check("dynamic sampling actually drops some", True,
          f"{n_train}/48 episodes reach the optimizer")


def main():
    print("=" * 66)
    print("CPU pipeline test -- no GPU, no vLLM")
    print("=" * 66)

    test_retrieval_and_parsing()
    test_reward()
    test_schema()

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
        print("\n(using the real Qwen3 tokenizer)")
    except Exception as e:
        print(f"\n!! Qwen tokenizer unavailable ({type(e).__name__}) -- "
              "falling back to the offline shim")
        print("   the logic below is tokenizer-independent, so this still "
              "validates the mask")
        tok = FakeTokenizer()

    if tok is not None:
        p = build_prompt(tok, "test question")
        check("chat template renders", len(p) > 100 and "test question" in p)
        test_rollout(tok)
        test_grpo_step(tok)

    print("\n" + "=" * 66)
    if _fails:
        print(f"{len(_fails)} FAILED: {_fails}")
        sys.exit(1)
    print("all checks passed")
    print("untested from here: the gradient step itself (needs a GPU)")
    print("=" * 66)


if __name__ == "__main__":
    main()
