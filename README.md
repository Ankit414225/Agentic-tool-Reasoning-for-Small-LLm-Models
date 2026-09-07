# RL for agentic BM25 tool use (Qwen3-1.7B → 4B)

On-policy GRPO over multi-turn retrieval trajectories. No pre-generated
trajectory dataset — rollouts are produced by the current policy inside the
training loop and discarded after one gradient step.

```
env.py            FROZEN environment: prompt, wire format, BM25, budgets.
                  Shared by training AND eval. Never fork this file.
dataset.py        benchmark schema -> tasks, + decontamination
rollout.py        on-policy multi-turn rollout (turn-synchronous, via vLLM)
reward.py         F1 + format gate + anti-hack penalties, group advantages
grpo.py           training loop                              (GPU)
sft.py            cold-start rejection sampling              (GPU, TODO)
eval.py           agentic eval against the SAME env          (GPU)
launch_vllm.py    server launcher, polls until ready

demo_rollout.py   watch one episode + the mask + advantages  (START HERE)
mock_llm.py       fake LLM with vLLM's interface, for CPU testing
test_local.py     CPU test suite -- 40 checks, no GPU needed

configs/          frozen benchmark configs (committed)
data/             gitignored -- regenerate with dataset.py
runs/             gitignored except results.md (COMMIT that)
ckpt/             gitignored -- checkpoints + tensorboard
```

Start here: **LOCAL_SETUP.md** for laptop setup, **REPO_REVIEW.md** for the
issues found in the team's existing eval harness.

## Setup

```bash
pip install -r requirements.txt
```
Needs one 40GB+ GPU for 1.7B (vLLM engine and the training model share it via
`--gpu_mem_util`). For the 4B final run use 80GB, or split rollout and training
across two GPUs.

## Order of operations — do not skip step 1

**1. Baseline.** Before any training, get the number you are trying to beat.
```bash
python data_prep.py --out data --n 12000
python eval.py --model Qwen/Qwen3-1.7B --data data/val.jsonl --limit 300 --dump runs/base.jsonl
```
Then open `runs/base.jsonl` and read 20 trajectories. You are looking for: does
it emit the tags at all, does it dump the whole question into one BM25 query,
does it answer in full sentences (which halves F1 on its own).

**2. Cold start if format compliance is under ~80%.** GRPO needs some rollouts
in each group to succeed or every advantage is zero and nothing moves. Cheapest
fix: sample k=8 rollouts per training question with the base model at temp 1.0,
keep trajectories with F1 > 0.8, SFT on them for 1–2 epochs (masking observation
tokens exactly as `rollout.py` does). A few thousand kept trajectories is enough.

**3. GRPO on 1.7B.**
```bash
python grpo.py --model <cold_start_or_base> --data data/train.jsonl --out ckpt/run1 \
  --prompts_per_step 48 --group_size 8 --lr 1e-6 --steps 400
```

**4. Ablations on 1.7B, then one long run on 4B.** The ablation table is what
makes this presentable. Minimum set: observation masking on/off, reward with and
without the length penalty, group size 4 vs 8 vs 16, MAX_TURNS 2 vs 4.

## Config decisions and why

| Choice | Value | Reason |
|---|---|---|
| Algorithm | GRPO | no critic; group mean is the baseline. At 4B a value net doubles optimizer memory to fit something a sparse outcome reward barely trains anyway |
| Loss normalization | token-level | sequence-level down-weights long trajectories, and multi-hop trajectories are the ones we want to learn |
| Clip | 0.2 / 0.28 | clip-higher keeps entropy alive; symmetric clipping collapses to greedy in ~100 steps |
| KL coef | 0.0 | no reference model to hold in memory; the task is verifiable so drift toward the reward is what we want |
| Sampling | temp 1.0, top_p 1.0, top_k −1 | truncated sampling makes rollouts off-policy w.r.t. the ratio you then compute |
| Group filtering | drop all-right / all-wrong | zero gradient at full rollout cost; usually removes 30–50% of prompts |
| LR | 1e-6 | RL on a small model is not SFT. 1e-5 will diverge |

## The one line that decides whether this works

`rollout.py`, in the tool branch:

```python
ep.extend(obs_ids, trainable=False)
```

Retrieved passages were written by BM25, not by the policy. Train on them and
you are optimizing the model to predict search results — reward climbs briefly
then rots. This is the single most commonly reported bug in tool-use RL.

## What to watch each step

- `f1` — should rise slowly and unevenly. Flat for 30 steps is normal.
- `kept_groups / n_groups` — if it drops near 0 the task is too hard (fix the
  cold start) or too easy (get harder data).
- `neg_logp` — entropy proxy. A sharp drop means the model found a degenerate
  strategy. Stop and read rollouts, not the loss curve.
- `zero_call_frac` — should sit near your no-tool slice fraction. Rising toward
  1.0 means the model has learned to skip search and guess.
- `mean_tool_calls` — rising past ~3 with flat F1 is flailing, not reasoning.
- `clipfrac` — above ~0.3 means the step size is too large.

## Known failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Reward climbs, eval F1 flat | verbose answers gaming token overlap | tighten `free_tokens`, check `ans_len` |
| Model searches on every question | rewarded tool use somewhere | never reward calls; keep the no-tool slice in the mix |
| All groups degenerate at step 0 | no cold start | step 2 above |
| Ratio explodes after step 1 | old logprobs taken from vLLM | they are recomputed with the training model in `grpo.py`; keep it that way |
| Train F1 ≫ dev F1 | contamination | rerun `data_prep.py` and confirm decontamination did not silently skip |

## Open questions for the club head

1. Is the BM25 candidate set **per question** or one shared index across the
   benchmark? Per-question makes retrieval far easier and changes the data build.
2. What exact tool-call wire format does the eval harness parse? If it differs
   from `env.py`, change `TOOL_CALL_OPEN`/`ANSWER_OPEN` there and nowhere else.
3. Is F1 computed on the final answer only, or over the whole output? If the
   whole output, the length penalty needs to be far more aggressive.
