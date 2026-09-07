"""
rollout.py -- ON-POLICY DATA GENERATION.

This is the answer to "RL on-policy data kaise generate karenge":
there is no pre-generated trajectory dataset. The only static asset is a table
of (question, gold_answer, candidate_docs). Trajectories are produced HERE, by
the CURRENT policy weights, against the LIVE BM25 tool, consumed by one gradient
step, and thrown away. Next step's rollouts come from the updated weights.
That is what makes it on-policy.

Turn-synchronous batching: instead of running episodes one at a time, we advance
every live episode by one turn in a single vLLM call, run all their tool calls,
then advance again. With G=8 and 64 prompts that is 512 sequences per vLLM call
and ~4 calls per batch, which is the difference between minutes and hours.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from vllm import SamplingParams
except ImportError:  # local CPU dev without vLLM installed
    @dataclass
    class SamplingParams:  # type: ignore[no-redef]
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        max_tokens: int = 256
        stop: list[str] | None = None
        include_stop_str_in_output: bool = True
        n: int = 1

from env import (
    MAX_NEW_TOKENS_PER_TURN,
    MAX_TOTAL_TOKENS,
    MAX_TURNS,
    STOP_STRINGS,
    BM25Tool,
    Episode,
    build_prompt,
    parse_turn,
    render_error,
    render_observation,
)


@dataclass
class Task:
    question: str
    gold: str
    aliases: list[str] | None = None
    docs: list[str] | None = None      # per-question candidate set
    corpus_id: str | None = None


def _sampling(temp: float, max_tok: int) -> SamplingParams:
    return SamplingParams(
        temperature=temp,
        top_p=1.0,                     # keep the full distribution: truncating
        top_k=-1,                      # top_p/top_k makes rollouts off-policy
        max_tokens=max_tok,
        stop=STOP_STRINGS,
        include_stop_str_in_output=True,
        n=1,
    )


def generate_episodes(
    llm,
    tokenizer,
    tasks: list[Task],
    tool: BM25Tool,
    group_size: int = 8,
    temperature: float = 1.0,
) -> list[list[Episode]]:
    """Return one list of `group_size` episodes per task."""

    groups: list[list[Episode]] = []
    flat: list[Episode] = []
    owners: list[Task] = []

    for t in tasks:
        prompt_ids = tokenizer(build_prompt(tokenizer, t.question),
                               add_special_tokens=False)["input_ids"]
        g = []
        for _ in range(group_size):
            ep = Episode(question=t.question, gold=t.gold)
            ep.extend(list(prompt_ids), trainable=False)   # prompt: no loss
            g.append(ep)
            flat.append(ep)
            owners.append(t)
        groups.append(g)

    for turn in range(MAX_TURNS):
        live = [i for i, e in enumerate(flat) if not e.finished]
        if not live:
            break

        budget = min(
            MAX_NEW_TOKENS_PER_TURN,
            max(16, MAX_TOTAL_TOKENS - max(len(flat[i].token_ids) for i in live)),
        )
        outs = llm.generate(
            prompt_token_ids=[flat[i].token_ids for i in live],
            sampling_params=_sampling(temperature, budget),
            use_tqdm=False,
        )

        for slot, out in zip(live, outs):
            ep = flat[slot]
            task = owners[slot]
            comp = out.outputs[0]

            # policy tokens -> trainable
            ep.extend(list(comp.token_ids), trainable=True)
            ep.text_log.append(comp.text)

            if len(ep.token_ids) >= MAX_TOTAL_TOKENS:
                ep.finished, ep.stop_reason = True, "length"
                continue

            p = parse_turn(comp.text)

            if p.kind == "answer":
                ep.answer = p.answer
                ep.finished, ep.stop_reason = True, "answer"
                continue

            last_turn = turn == MAX_TURNS - 1

            if p.kind == "tool":
                ep.n_tool_calls += 1
                try:
                    hits = tool.search(p.query or "", docs=task.docs,
                                       corpus_id=task.corpus_id)
                    obs = render_observation(hits)
                except Exception as exc:                    # never kill an episode
                    obs = render_error(str(exc)[:120])
            else:
                ep.n_malformed += 1
                obs = render_error(p.error or "malformed output")

            if last_turn:
                ep.finished, ep.stop_reason = True, "turn_limit"
                continue

            # >>> THE CRITICAL LINE <<<
            # Environment tokens get trainable=False. If you train on these you
            # are teaching the model to predict BM25 output, and the run rots.
            obs_ids = tokenizer(_wrap_observation(tokenizer, obs),
                                add_special_tokens=False)["input_ids"]
            ep.extend(obs_ids, trainable=False)

    for ep in flat:
        if not ep.finished:
            ep.finished, ep.stop_reason = True, "turn_limit"
        if ep.answer is None and ep.stop_reason == "answer":
            ep.stop_reason = "no_answer"

    return groups


def _wrap_observation(tokenizer, obs: str) -> str:
    """Feed the observation back as a user turn so the chat template stays valid."""
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": obs}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return f"\n{obs}\n"
    # strip a duplicated BOS/system block if the template injects one
    return rendered


def rollout_stats(groups: list[list[Episode]]) -> dict:
    eps = [e for g in groups for e in g]
    n = len(eps)
    return {
        "n_episodes": n,
        "answered_frac": sum(e.answer is not None for e in eps) / n,
        "mean_tool_calls": sum(e.n_tool_calls for e in eps) / n,
        "zero_call_frac": sum(e.n_tool_calls == 0 for e in eps) / n,
        "malformed_frac": sum(e.n_malformed > 0 for e in eps) / n,
        "trunc_frac": sum(e.stop_reason == "length" for e in eps) / n,
        "mean_len": sum(len(e.token_ids) for e in eps) / n,
    }
