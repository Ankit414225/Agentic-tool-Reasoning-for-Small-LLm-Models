# Local setup (VS Code, no GPU)

## What runs where

| Task | Laptop | GPU needed |
|---|---|---|
| Inspect the benchmark schema | yes | no |
| Build + decontaminate training data | yes | no |
| BM25 retrieval, tool-call parsing | yes | no |
| Reward function, GRPO advantages | yes | no |
| Full multi-turn rollout loop (mock LLM) | yes | no |
| Loss-mask verification | yes | no |
| Baseline eval on the real model | no | yes |
| GRPO training | no | 40GB+ |

Everything in the first block is real work and it's most of the debugging.
`torch` and `vllm` are not needed for any of it.

---

## Step 1 — Project

```bash
mkdir toolrl && cd toolrl
# copy the .py files + requirements-*.txt in here
code .
```

## Step 2 — Virtual environment

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

In VS Code: `Ctrl+Shift+P` → **Python: Select Interpreter** → pick the one
inside `.venv`. If you skip this, the terminal and the editor will use different
Pythons and the imports will look broken for no reason.

## Step 3 — Install (CPU only)

```bash
pip install -r requirements-local.txt
```

Do **not** `pip install vllm` locally. It pulls CUDA wheels, takes ~10 minutes,
and fails or silently installs a CPU build that can't run anything. `rollout.py`
already falls back to a local `SamplingParams` shim when vLLM is absent.

## Step 4 — Verify the pipeline

```bash
python test_local.py
```

Expect 40 checks and `all checks passed`. This covers BM25, tool-call parsing
(including malformed JSON recovery), F1 reward, GRPO advantages, benchmark
schema conversion, the multi-turn rollout loop, and the loss mask.

The block to actually read is this one:

```
-- loss mask (the thing that decides if training works) --
[  ok  ] BM25 passages are NOT in the trainable region
[  ok  ] BM25 passages ARE in the masked region
[  ok  ] model's own tool calls stay trainable
```

Retrieved passages must sit under `mask == 0`. BM25 wrote them, not the policy.
If they leak into the trainable region you're training the model to predict
search results, reward climbs for a while and then rots. This test is the guard
against silently reintroducing that.

If the Qwen tokenizer can't download, the suite falls back to an offline
word-level shim. The mask logic tracks *segments*, not tokens, so the shim
validates it just as well.

## Step 5 — Real work: inspect the benchmark

```bash
huggingface-cli login          # if the dataset is gated
python dataset.py --bench YashBhamare123/tools-benchmark --out data
```

This prints the first row and the column names, then:

```
docs/question: min .. med .. max ..
doc chars:     med .. max ..
answer words:  med .. max ..     <-- if med > 3, tell the data team
hops: {...}
```

Read those four lines carefully — they answer the questions you were going to
ask the team, and each one changes a config value:

| Output | What to change |
|---|---|
| `answer words` median > 3 | gold answers are too long; raise this with the data team before training, or your reward teaches verbosity |
| `doc chars` max > 2000 | lower `TOP_K` or `PASSAGE_MAX_CHARS` in `env.py`, or one search fills the context window |
| `docs/question` > 20 | BM25 is doing real work; retrieval quality matters more, consider raising `TOP_K` |
| mostly 2-hop | lower `MAX_TURNS` to 3 and save rollout time |
| `hops` missing | can't difficulty-filter; sample by pass rate instead once you have a GPU |

Outputs: `data/dev_benchmark.jsonl` (eval only, never trained on),
`data/train.jsonl`, `data/val.jsonl`.

Confirm the decontamination line printed a real number. If it says
`DECONTAMINATION SKIPPED`, the dataset didn't load and everything downstream is
guesswork.

## Step 6 — Sanity-check your own data by hand

```bash
python -c "
import json
rows=[json.loads(l) for l in open('data/train.jsonl')][:3]
for r in rows:
    print(r['question']); print(' gold:',repr(r['answer']),'hops:',r['hops'])
    print(' docs:',len(r['docs']),'| first:',r['docs'][0][:80]); print()
"
```

Then check BM25 can actually find the answer — if it can't, no amount of RL will:

```bash
python -c "
import json
from env import BM25Tool
t=BM25Tool(); hit=0; rows=[json.loads(l) for l in open('data/train.jsonl')][:200]
for r in rows:
    got=t.search(r['question'], docs=r['docs'], corpus_id=r['id'])
    hit += any(r['answer'].lower() in d.lower() for d in got)
print(f'naive one-shot retrieval finds the answer in {hit}/{len(rows)}')
"
```

If that number is above ~80%, the task is mostly single-hop and RL will gain
little — push back on the data mix. Around 30–50% is the healthy range: BM25
alone is not enough, so multi-hop querying is what earns the reward.

## Step 7 — Debugging in VS Code

`.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "test_local",
      "type": "debugpy",
      "request": "launch",
      "program": "test_local.py",
      "console": "integratedTerminal",
      "justMyCode": false
    },
    {
      "name": "dataset.py",
      "type": "debugpy",
      "request": "launch",
      "program": "dataset.py",
      "args": ["--bench", "YashBhamare123/tools-benchmark", "--out", "data"],
      "console": "integratedTerminal"
    }
  ]
}
```

Best breakpoint to actually learn from: `rollout.py`, on the
`ep.extend(obs_ids, trainable=False)` line. Step through one episode and watch
`ep.mask` fill in. That's the whole algorithm in one variable.

## Step 8 — When you get a GPU

```bash
pip install -r requirements-gpu.txt
python eval.py --model Qwen/Qwen3-1.7B --data data/dev_benchmark.jsonl \
  --limit 200 --dump runs/base.jsonl
```

Baseline first, always. Then read 20 rows of `runs/base.jsonl` by hand before
training anything.

## Files

```
env.py         frozen environment -- shared with eval, single source of truth
dataset.py     benchmark schema -> tasks, + decontamination
rollout.py     on-policy multi-turn rollout
reward.py      F1 + format gate + advantages
grpo.py        training loop (GPU)
eval.py        agentic eval (GPU)
mock_llm.py    fake LLM, vLLM-compatible interface, for CPU testing
test_local.py  the CPU test suite
```
