"""
demo_rollout.py -- watch one episode happen, turn by turn.

    python demo_rollout.py

No GPU. Uses the mock LLM. The point is to make three things concrete:
  1. what a multi-turn tool-use episode actually looks like
  2. which tokens are trainable and which are masked (the whole algorithm)
  3. how a group of rollouts turns into GRPO advantages
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from env import BM25Tool, parse_turn
from mock_llm import MockLLM
from reward import compute_reward, group_advantages, group_is_informative
from rollout import Task, generate_episodes
from test_local import DOCS, FakeTokenizer

W = 72


def rule(title=""):
    print("\n" + ("=" * W if not title else f"== {title} " + "=" * (W - len(title) - 4)))


def get_tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B"), "Qwen3"
    except Exception:
        return FakeTokenizer(), "offline shim"


def main():
    tok, which = get_tokenizer()
    task = Task(
        question="Who designed the machine that Ada Lovelace wrote an algorithm for?",
        gold="Charles Babbage",
        aliases=["Babbage"],
        docs=DOCS,
        corpus_id="demo",
    )

    rule("THE SETUP")
    print(f"tokenizer : {which}")
    print(f"question  : {task.question}")
    print(f"gold      : {task.gold}   (aliases: {task.aliases})")
    print(f"\ncandidate set the BM25 tool can see ({len(DOCS)} docs):")
    for d in DOCS:
        print(f"  - {d.splitlines()[0]}")
    print("\nNote the question never names the Analytical Engine. The model has to")
    print("find it in doc 1, THEN use it to query for its designer in doc 2.")
    print("That is what 'multi-hop' means, and why one search is not enough.")

    # ---------------- one episode, turn by turn ----------------
    rule("ONE EPISODE, TURN BY TURN")
    ep = generate_episodes(
        MockLLM(tok, "good_multihop", tasks=[task]), tok, [task], BM25Tool(),
        group_size=1,
    )[0][0]

    tool = BM25Tool()
    for i, text in enumerate(ep.text_log):
        p = parse_turn(text)
        print(f"\n--- turn {i}: model emits ---")
        print(f"  {text.strip()[:200]}")
        print(f"  parsed as: {p.kind}", end="")
        if p.kind == "tool" and p.query:
            print(f"   query={p.query!r}")
            hits = tool.search(p.query, docs=DOCS, corpus_id="demo")
            print("  --- environment returns (MASKED from the loss) ---")
            for j, h in enumerate(hits[:2]):
                print(f"    [{j+1}] {h[:88]}")
        elif p.kind == "answer":
            print(f"   answer={p.answer!r}")
        else:
            print(f"   error={p.error}")

    print(f"\nepisode ended: stop_reason={ep.stop_reason}, "
          f"tool_calls={ep.n_tool_calls}, malformed={ep.n_malformed}")

    #  the mask
    rule("THE LOSS MASK -- the part that decides if training works")
    trainable = tok.decode([t for t, m in zip(ep.token_ids, ep.mask) if m == 1])
    n_tr, n_tot = int(sum(ep.mask)), len(ep.mask)
    print(f"\ntotal tokens in the episode : {n_tot}")
    print(f"trainable (mask == 1)       : {n_tr}   <- the policy wrote these")
    print(f"masked    (mask == 0)       : {n_tot - n_tr}   "
          f"<- prompt + BM25 passages")
    print(f"\nonly {100*n_tr/n_tot:.0f}% of the sequence gets a gradient.")

    print("\nvisual map (each char = one token):")
    strip = "".join("#" if m else "." for m in ep.mask)
    for i in range(0, len(strip), W):
        print(f"  {strip[i:i+W]}")
    print("  '.' = masked (prompt / retrieved passages)   '#' = trainable")

    print(f"\nwhat the gradient actually sees:\n  {trainable[:240]!r}")
    print("\nWHY: the passages were written by BM25, not by the model. Training on")
    print("them optimises the model to predict search results. Reward climbs for a")
    print("while, then the run rots. This is the single most common bug in")
    print("tool-use RL, so test_local.py asserts on it.")

    #  a group -> advantages
    rule("A GROUP OF 8 ROLLOUTS -> GRPO ADVANTAGES")
    groups = generate_episodes(
        MockLLM(tok, "mixed", seed=3, tasks=[task]), tok, [task], BM25Tool(),
        group_size=8,
    )
    rewards, rows = [], []
    for e in groups[0]:
        r, info = compute_reward(e, [task.gold] + (task.aliases or []))
        rewards.append(r)
        rows.append((e.stop_reason, e.n_tool_calls, e.answer, r, info["f1"]))

    adv = group_advantages(rewards)
    print(f"\n{'stop':<11}{'calls':<7}{'answer':<34}{'F1':>6}{'reward':>8}{'adv':>7}")
    print("-" * W)
    for (stop, calls, ans, r, f1), a in zip(rows, adv):
        print(f"{stop:<11}{calls:<7}{str(ans)[:32]:<34}{f1:>6.2f}{r:>8.2f}{a:>+7.2f}")

    mean = sum(rewards) / len(rewards)
    print("-" * W)
    print(f"group mean reward: {mean:.3f}   (this IS the baseline -- no critic needed)")
    print(f"advantages sum to {sum(adv):+.4f}, i.e. zero-centred")
    print(f"informative group? {group_is_informative(rewards)}")

    rule("WHAT THE UPDATE DOES")
    print("\npositive advantage -> make those tokens MORE likely")
    print("negative advantage -> make those tokens LESS likely")
    print("\nNo one told the model to search twice. It just happens that rollouts")
    print("which searched twice landed the right short answer and scored above the")
    print("group mean, so their tokens get reinforced. Multi-hop behaviour is an")
    print("emergent consequence of the reward, not an instruction.")
    print("\nThen: discard every rollout, resample from the updated weights.")
    print("That is 'on-policy'. There is no trajectory dataset anywhere.")
    print("=" * W)


if __name__ == "__main__":
    main()
