"""
mock_llm.py -- a fake LLM with vLLM's `generate()` interface.

Purpose: exercise the whole rollout loop on a laptop with no GPU. It lets you
verify the parts that actually break in practice -- the loss mask alignment,
multi-turn state, malformed-output recovery, turn limits, reward wiring --
without waiting for a cloud GPU.

It is NOT a language model. It replays scripted behaviours so you can assert on
them deterministically.
"""

from __future__ import annotations

import json
import random
import re

from env import (ANSWER_CLOSE, ANSWER_OPEN, TOOL_CALL_CLOSE, TOOL_CALL_OPEN,
                 TOOL_RESP_OPEN)


class _Out:
    def __init__(self, text: str, token_ids: list[int]):
        self.text = text
        self.token_ids = token_ids


class _Req:
    def __init__(self, outputs):
        self.outputs = outputs


class MockLLM:
    """Behaviours:
      'good_multihop' : search, search again with a term from the passages, answer
      'one_shot'      : answer immediately, no search
      'malformed'     : emit garbage first, then recover
      'never_answers' : search forever, hits the turn limit
      'verbose'       : correct answer wrapped in a sentence (should lose F1)
      'mixed'         : random pick per rollout -- use this for reward spread
    """

    def __init__(self, tokenizer, behaviour: str = "mixed", seed: int = 0,
                 tasks=None):
        self.tok = tokenizer
        self.behaviour = behaviour
        self.rng = random.Random(seed)
        # question -> gold, so scripted rollouts can actually score. A test
        # double is allowed to cheat; we need a non-degenerate reward spread.
        self.golds = {t.question: t.gold for t in (tasks or [])}

    # ---- vLLM-compatible entry point ----
    def generate(self, prompt_token_ids, sampling_params, use_tqdm=False):
        reqs = []
        for ids in prompt_token_ids:
            ctx = self.tok.decode(ids)
            b = self.behaviour
            if b == "mixed":
                b = self.rng.choice(
                    ["good_multihop", "one_shot", "malformed", "verbose", "never_answers"]
                )
            text = self._turn(ctx, b)
            out_ids = self.tok(text, add_special_tokens=False)["input_ids"]
            reqs.append(_Req([_Out(text, out_ids)]))
        return reqs

    # ---- scripted policy ----
    def _turn(self, ctx: str, b: str) -> str:
        # Count ENVIRONMENT observations, not '<tool_call>' occurrences: the
        # system prompt contains '<tool_call>' in its format instructions, so
        # counting that is off by one from turn zero.
        n_searches = ctx.count(TOOL_RESP_OPEN)
        gold = self._gold_for(ctx)

        if b == "one_shot":
            return f"I know this one. {ANSWER_OPEN}{gold}{ANSWER_CLOSE}"

        if b == "verbose":
            if n_searches == 0:
                return self._search("first hop keywords")
            return (f"Based on the passages above, "
                    f"{ANSWER_OPEN}the answer is definitely {gold}, "
                    f"as shown in passage 2{ANSWER_CLOSE}")

        if b == "never_answers":
            return self._search(f"more keywords attempt {n_searches + 1}")

        if b == "malformed":
            if n_searches == 0 and "ERROR" not in ctx:
                return "let me look this up"                      # no tags at all
            if "ERROR" in ctx and n_searches == 0:
                return f"{TOOL_CALL_OPEN}{{query: broken json{TOOL_CALL_CLOSE}"
            if n_searches == 0:
                return self._search("recovering now")
            return f"{ANSWER_OPEN}{gold}{ANSWER_CLOSE}"

        # good_multihop
        if n_searches == 0:
            return f"I need the first fact. {self._search('first hop entity')}"
        if n_searches == 1:
            hint = self._term_from_passages(ctx)
            return f"Now I can chain from that. {self._search(f'{hint} second hop')}"
        return f"{ANSWER_OPEN}{gold}{ANSWER_CLOSE}"

    # ---- helpers ----
    @staticmethod
    def _search(query: str) -> str:
        payload = json.dumps({"name": "search", "arguments": {"query": query}})
        return f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}"

    @staticmethod
    def _term_from_passages(ctx: str) -> str:
        titles = re.findall(r"Title: ([^\n]+)", ctx)
        return titles[-1] if titles else "entity"

    def _gold_for(self, ctx: str) -> str:
        """Recover the gold answer for whichever question is in this context.

        Deliberate cheating: some rollouts must succeed or every GRPO group is
        degenerate and we cannot verify that advantages are computed correctly.
        Falls back to a retrieved title, which usually scores 0 -- that gives us
        the failing half of the spread.
        """
        for q, g in self.golds.items():
            if q[:40] in ctx:
                return g
        m = re.findall(r"Title: ([^\n]+)", ctx)
        return m[-1].strip() if m else "unknown"
