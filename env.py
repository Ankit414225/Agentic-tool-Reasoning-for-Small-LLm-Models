"""
env.py -- THE FROZEN ENVIRONMENT.

This module is the single source of truth for:
  - the system prompt / tool schema
  - the wire format for tool calls and final answers
  - BM25 retrieval behaviour (k1, b, top_k, passage truncation)
  - the turn/token budget

RULE: training and evaluation both import from HERE. Never fork this file.
If the organizers' harness uses a different wire format, change TOOL_CALL_OPEN /
ANSWER_OPEN / render_observation below and NOTHING ELSE needs to move.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import bm25s

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESP_OPEN = "<tool_response>"
TOOL_RESP_CLOSE = "</tool_response>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"

# vLLM stop strings. We stop the instant a turn is complete so we can run the
# tool and hand control back to the model.
STOP_STRINGS = [TOOL_CALL_CLOSE, ANSWER_CLOSE]


MAX_TURNS = 7           # assistant turns per episode
TOP_K = 3               # passages returned per search
PASSAGE_MAX_CHARS = 100    # truncate each passage
MAX_NEW_TOKENS_PER_TURN = 320
MAX_TOTAL_TOKENS = 8192

SYSTEM_PROMPT = f"""You answer questions using a search tool over a document collection.

You have exactly one tool:
  search(query: str) -> passages
  Runs BM25 keyword retrieval and returns the top {TOP_K} passages.

On each turn you must emit EXACTLY ONE of:

1. A tool call:
{TOOL_CALL_OPEN}{{"name": "search", "arguments": {{"query": "<keywords>"}}}}{TOOL_CALL_CLOSE}

2. A final answer:
{ANSWER_OPEN}<answer>{ANSWER_CLOSE}

Guidelines:
- You may write one or two short sentences of reasoning before the tag.
- Search only when you need information you do not already have. If you can
  answer from the question itself or from what you already retrieved, answer.
- Multi-hop questions need multiple searches: find the first fact, then use it
  to build the next query. Do not put the whole question in one query -- BM25
  matches keywords, not meaning.
- The final answer must be as SHORT as possible: a name, a date, a number, a
  noun phrase. No sentences, no restating the question, no explanation.
- You have at most {MAX_TURNS} turns. If the last turn arrives, answer with your
  best guess rather than searching."""



class BM25Tool:
    """BM25 over a candidate set.

    Two modes:
      - per-question corpora: build_for(doc_list) each episode (cheap, small sets)
      - one global corpus: build_global(doc_list) once, reused for all episodes

    ASK THE ORGANIZERS which one the benchmark uses. It changes everything about
    how hard retrieval is.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._global: bm25s.BM25 | None = None
        self._global_docs: list[str] = []
        self._cache: dict[str, tuple[bm25s.BM25, list[str]]] = {}

    @staticmethod
    def _tok(texts: list[str]):
        return bm25s.tokenize(texts, stopwords="en", show_progress=False)

    def build_global(self, docs: list[str]) -> None:
        idx = bm25s.BM25(k1=self.k1, b=self.b)
        idx.index(self._tok(docs), show_progress=False)
        self._global, self._global_docs = idx, docs

    def _index_for(self, corpus_id: str | None, docs: list[str] | None):
        if docs is None:
            if self._global is None:
                raise RuntimeError("no global index built and no docs given")
            return self._global, self._global_docs
        if corpus_id is not None and corpus_id in self._cache:
            return self._cache[corpus_id]
        idx = bm25s.BM25(k1=self.k1, b=self.b)
        idx.index(self._tok(docs), show_progress=False)
        if corpus_id is not None:
            self._cache[corpus_id] = (idx, docs)
        return idx, docs

    def search(
        self,
        query: str,
        docs: list[str] | None = None,
        corpus_id: str | None = None,
        top_k: int = TOP_K,
    ) -> list[str]:
        query = (query or "").strip()
        if not query:
            return []
        idx, corpus = self._index_for(corpus_id, docs)
        qt = self._tok([query])
        k = min(top_k, len(corpus))
        if k == 0:
            return []
        res, _ = idx.retrieve(qt, k=k, show_progress=False)
        return [corpus[i] for i in res[0]]


def render_observation(passages: list[str]) -> str:
    """Environment -> model. These tokens are MASKED OUT of the RL loss."""
    if not passages:
        body = "No results."
    else:
        body = "\n".join(
            f"[{i + 1}] {p[:PASSAGE_MAX_CHARS].strip()}" for i, p in enumerate(passages)
        )
    return f"{TOOL_RESP_OPEN}\n{body}\n{TOOL_RESP_CLOSE}"


def render_error(msg: str) -> str:
    return f"{TOOL_RESP_OPEN}\nERROR: {msg}\n{TOOL_RESP_CLOSE}"



_TOOL_RE = re.compile(
    re.escape(TOOL_CALL_OPEN) + r"(.*?)" + re.escape(TOOL_CALL_CLOSE), re.S
)
_ANS_RE = re.compile(re.escape(ANSWER_OPEN) + r"(.*?)" + re.escape(ANSWER_CLOSE), re.S)


@dataclass
class TurnParse:
    kind: str                      # "tool" | "answer" | "malformed"
    query: str | None = None
    answer: str | None = None
    error: str | None = None


def parse_turn(text: str) -> TurnParse:
    """Parse one assistant turn. Lenient on the JSON, strict on the tags."""
    a = _ANS_RE.search(text)
    t = _TOOL_RE.search(text)

    # Whichever tag closes first is the action for this turn.
    if a and (not t or a.start() < t.start()):
        return TurnParse("answer", answer=a.group(1).strip())

    if t:
        raw = t.group(1).strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            # fall back to grabbing a "query" field out of near-JSON
            m = re.search(r'"query"\s*:\s*"([^"]*)"', raw)
            if m:
                return TurnParse("tool", query=m.group(1))
            return TurnParse("malformed", error="tool call is not valid JSON")
        if isinstance(obj, dict):
            args = obj.get("arguments", obj)
            if isinstance(args, dict) and isinstance(args.get("query"), str):
                return TurnParse("tool", query=args["query"])
        return TurnParse("malformed", error='tool call needs arguments.query as a string')

    return TurnParse("malformed", error="no <tool_call> or <answer> tag found")


# Prompt construction

def build_prompt(tokenizer, question: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]
    kwargs: dict[str, Any] = dict(tokenize=False, add_generation_prompt=True)
    try:  # Qwen3: keep thinking off, it eats the token budget
        return tokenizer.apply_chat_template(msgs, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, **kwargs)


@dataclass
class Episode:
    """One trajectory. `mask` is 1 for tokens the POLICY produced, 0 otherwise."""

    question: str
    gold: str
    token_ids: list[int] = field(default_factory=list)
    mask: list[int] = field(default_factory=list)
    n_tool_calls: int = 0
    n_malformed: int = 0
    finished: bool = False
    answer: str | None = None
    stop_reason: str = "running"   # answer | turn_limit | length | no_answer
    text_log: list[str] = field(default_factory=list)

    def extend(self, ids: list[int], trainable: bool) -> None:
        self.token_ids.extend(ids)
        self.mask.extend([1 if trainable else 0] * len(ids))
