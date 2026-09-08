# Agentic Tool Reasoning (RL part)

RL training pipeline for the tool-reasoning problem statement.

The model gets a question and one tool: BM25 search over a set of 20 candidate
documents. It decides when to search, what to search for, and when to answer.
Questions need 2 to 4 hops, so one search is usually not enough.

We train it with GRPO. There is no trajectory dataset. The model generates its
own rollouts during training, gets scored on answer F1, and updates from that.

## Setup

You need Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\activate
pip install -r requirements-local.txt
```

Do not install vllm or torch on a laptop. They are GPU only and listed
separately in requirements-gpu.txt.

## Check it works

```bash
python test_local.py
```

You should see 40 checks and "all checks passed". This runs on CPU with a fake
model, so it needs no GPU.

```bash
python demo_rollout.py
```

This prints one full episode: the model searching, the passages coming back, the
final answer, and how a group of rollouts turns into training signal. Read this
if you want to understand what the code does.

## Get the data

Log in to Hugging Face first if the benchmark dataset is private:

```bash
pip install -U huggingface_hub
hf auth login
```

Then:

```bash
python dataset.py --out data --n_train 30000
python balance_data.py
```

This downloads MuSiQue and HotpotQA (a few GB, 10 to 20 minutes the first time),
converts them into training format, removes anything that overlaps the benchmark,
and balances the mix to match the benchmark's 2/3/4 hop split.

Output goes to `data/`. It is gitignored, so regenerate it rather than looking
for it in the repo.

## Look at the data

```bash
python inspect_bench.py        # what is in the benchmark dev split
python check_retrieval.py      # can BM25 find the answer in one search
python estimate_headroom.py    # how much does searching multiple times help
python ablate_retrieval.py     # is it chaining that helps, or just more docs
```

These are read only and need no GPU. What they found so far:

- Every answer appears somewhere in the 20 candidate documents, so no question
  is impossible.
- One search with top_k=3 finds the answer 33% of the time on the dev split.
- Retrieving more documents in one search beats searching several times with a
  simple keyword expansion.

## Train (needs a GPU)

40GB or more of VRAM for the 1.7B model.

```bash
pip install -r requirements-gpu.txt

# baseline first, before training anything
python eval_agentic.py --model Qwen/Qwen3-1.7B --data data/dev_benchmark.jsonl \
  --limit 200 --dump runs/base.jsonl

python grpo.py --model Qwen/Qwen3-1.7B --data data/train.jsonl --out ckpt/run1
```

Get the baseline number first. Without it there is no way to tell if training
helped. After that, open `runs/base.jsonl` and read about 20 trajectories by
hand. The loss curve will not tell you when the model is doing something silly.

Record every run in `runs/results.md`.

## Files

```
env.py              the environment: prompt, tool format, BM25, turn limits
dataset.py          benchmark and MuSiQue into training format
balance_data.py     match the training hop mix to the benchmark
rollout.py          runs episodes, model talks to the tool
reward.py           F1 scoring and GRPO advantages
grpo.py             training loop
sft.py              optional cold start (not written yet)
eval_agentic.py     evaluation through the tool loop
launch_vllm.py      starts a vLLM server

mock_llm.py         fake model so tests run without a GPU
test_local.py       test suite
demo_rollout.py     prints one episode start to finish

inspect_bench.py    dev split stats
check_retrieval.py  BM25 one shot recall
estimate_headroom.py    recall with repeated searching
ablate_retrieval.py     chaining vs more documents

configs/            frozen config, share with whoever owns eval
data/               gitignored, regenerate with dataset.py
runs/               gitignored except results.md
ckpt/               gitignored, checkpoints
```

## One thing to be careful about

`env.py` is shared by training and evaluation. If you copy it or change it on
one side only, the model trains against one environment and gets scored on a
different one, and the training gains disappear at submission. Keep one copy.

Inside `rollout.py` there is a line marked as critical:

```python
ep.extend(obs_ids, trainable=False)
```

Retrieved passages are masked out of the loss because BM25 wrote them, not the
model. If you remove that mask, reward goes up for a while and then the run
falls apart. `test_local.py` checks for this.
