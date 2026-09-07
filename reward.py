"""
reward.py -- outcome reward for a finished Episode.

Design notes you can defend in the writeup:

  * Primary signal is token-level F1 against the gold answer, because F1 IS the
    eval metric. Binary EM throws away the partial-credit gradient that makes
    early training work at all.
  * Format is a HARD GATE, not a bonus. An unparseable trajectory scores 0.
    Bonuses for format get farmed; gates do not.
  * We do NOT reward tool use. Rewarding search teaches the model to search on
    every question, which destroys the "no tool needed" cases the PS tests.
  * We do NOT reward intermediate retrieval hits. That teaches query-stuffing.
  * Small length penalty on the answer: F1 is diluted by extra tokens, so long
    answers already lose points -- this just sharpens it early on.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass

from env import Episode

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.U)
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(s: str) -> str:
    s = s.lower()
    s = s.translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def em_score(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def best_over_aliases(pred: str, golds: list[str], fn) -> float:
    return max((fn(pred, g) for g in golds), default=0.0)


@dataclass
class RewardConfig:
    length_penalty_per_token: float = 0.02   # on answer tokens beyond `free_tokens`
    free_tokens: int = 6
    malformed_penalty: float = 0.05          # per malformed turn
    extra_call_penalty: float = 0.02         # per tool call beyond 2
    max_penalty: float = 0.3                 # penalties can never exceed this
    format_gate: bool = True                 # set False late in training


def compute_reward(
    ep: Episode,
    golds: list[str] | None = None,
    cfg: RewardConfig = RewardConfig(),
) -> tuple[float, dict]:
    golds = golds or [ep.gold]
    info: dict[str, float] = {}

    if ep.answer is None:
        info.update(f1=0.0, em=0.0, penalty=0.0, gated=1.0)
        return 0.0, info

    f1 = best_over_aliases(ep.answer, golds, f1_score)
    em = best_over_aliases(ep.answer, golds, em_score)

    pen = 0.0
    pen += cfg.malformed_penalty * ep.n_malformed
    pen += cfg.extra_call_penalty * max(0, ep.n_tool_calls - 2)
    n_ans_tok = len(ep.answer.split())
    pen += cfg.length_penalty_per_token * max(0, n_ans_tok - cfg.free_tokens)
    pen = min(pen, cfg.max_penalty)

    r = max(0.0, f1 - pen)
    info.update(f1=f1, em=em, penalty=pen, gated=0.0,
                n_calls=float(ep.n_tool_calls), ans_len=float(n_ans_tok))
    return r, info


def group_advantages(rewards: list[float], eps: float = 1e-4) -> list[float]:
    """GRPO: standardise within the group of G rollouts for one prompt."""
    n = len(rewards)
    mu = sum(rewards) / n
    var = sum((r - mu) ** 2 for r in rewards) / n
    sd = var ** 0.5
    if sd < eps:
        return [0.0] * n            # degenerate group -> no signal
    return [(r - mu) / sd for r in rewards]


def group_is_informative(rewards: list[float], lo: float = 1e-3) -> bool:
    """DAPO dynamic sampling: drop all-right and all-wrong groups.

    Typically discards 30-50% of prompts and roughly doubles learning per
    GPU-hour, because a degenerate group contributes exactly zero gradient but
    costs a full set of rollouts.
    """
    return (max(rewards) - min(rewards)) > lo
