"""
grpo.py -- minimal, readable GRPO for multi-turn tool use.

Why GRPO and not PPO: at 1.7B/4B a separate value network doubles optimizer
memory to estimate a baseline that the group mean already gives us for free,
and with a single sparse outcome reward the critic is hard to fit anyway.

Why write the loop instead of using a framework: the multi-turn loss mask is the
part that decides whether this works, and it is the part frameworks hide.

Run:
  python grpo.py --model Qwen/Qwen3-1.7B --data data/train.jsonl --out ckpt/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM

from env import BM25Tool
from reward import RewardConfig, compute_reward, group_advantages, group_is_informative
from rollout import Task, generate_episodes, rollout_stats


# ---------------------------------------------------------------- data
def load_tasks(path: str) -> list[Task]:
    tasks = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            tasks.append(
                Task(
                    question=d["question"],
                    gold=d["answer"],
                    aliases=d.get("aliases"),
                    docs=d.get("docs"),
                    corpus_id=d.get("corpus_id"),
                )
            )
    return tasks


# ---------------------------------------------------------------- batching
def collate(episodes, pad_id, device):
    L = max(len(e.token_ids) for e in episodes)
    ids = torch.full((len(episodes), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(episodes), L), dtype=torch.long)
    msk = torch.zeros((len(episodes), L), dtype=torch.float)
    for i, e in enumerate(episodes):
        n = len(e.token_ids)
        ids[i, :n] = torch.tensor(e.token_ids)
        att[i, :n] = 1
        msk[i, :n] = torch.tensor(e.mask, dtype=torch.float)
    return ids.to(device), att.to(device), msk.to(device)


def token_logprobs(model, ids, att):
    """log p(token_t | tokens_<t), aligned so index t-1 holds the score of token t."""
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1]
    logits = logits.float() / 1.0
    tgt = ids[:, 1:]
    return torch.gather(F.log_softmax(logits, -1), 2, tgt.unsqueeze(-1)).squeeze(-1)


# ---------------------------------------------------------------- vLLM sync
def sync_weights(llm, model):
    """Push updated HF weights into the vLLM engine.

    The private-attribute path below moves between vLLM versions. If it breaks,
    the safe fallback is to save the checkpoint and re-instantiate LLM(), which
    costs ~60s per step -- fine for a first run, worth fixing after.
    """
    sd = {k: v.detach().to(torch.bfloat16) for k, v in model.state_dict().items()}
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    runner.model.load_weights(sd.items())
    del sd


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--out", default="ckpt")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--prompts_per_step", type=int, default=48)
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--micro_bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--clip_low", type=float, default=0.2)
    ap.add_argument("--clip_high", type=float, default=0.28)   # clip-higher (DAPO)
    ap.add_argument("--kl_coef", type=float, default=0.0)      # 0 = no ref model
    ap.add_argument("--inner_epochs", type=int, default=1)
    ap.add_argument("--gpu_mem_util", type=float, default=0.35)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.out, "tb"))

    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).cuda()   # type: ignore[union-attr]  # phantom Pylance error: transformers'
    #            model types can't resolve without torch installed (GPU-only dep)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        enable_prefix_caching=True,     # big win: shared prompt across the group
        max_model_len=4096,
    )

    tasks = load_tasks(args.data)
    tool = BM25Tool()
    rcfg = RewardConfig()
    print(f"{len(tasks)} training tasks")

    for step in range(args.steps):
        t0 = time.time()

        # ---------- 1. ON-POLICY ROLLOUT ----------
        batch_tasks = random.sample(tasks, min(args.prompts_per_step, len(tasks)))
        model.eval()
        with torch.no_grad():
            groups = generate_episodes(
                llm, tok, batch_tasks, tool,
                group_size=args.group_size, temperature=args.temperature,
            )
        t_roll = time.time() - t0

        # ---------- 2. REWARD ----------
        train_eps, train_adv, all_r, all_f1 = [], [], [], []
        kept_groups = 0
        for g, task in zip(groups, batch_tasks):
            golds = [task.gold] + (task.aliases or [])
            scored = [compute_reward(e, golds, rcfg) for e in g]
            rs = [s[0] for s in scored]
            all_r += rs
            all_f1 += [s[1]["f1"] for s in scored]

            # DAPO dynamic sampling: a degenerate group has zero gradient but
            # costs a full set of rollouts. Drop it.
            if not group_is_informative(rs):
                continue
            kept_groups += 1
            for e, a in zip(g, group_advantages(rs)):
                train_eps.append(e)
                train_adv.append(a)

        stats = rollout_stats(groups)
        if not train_eps:
            print(f"step {step}: all groups degenerate, skipping")
            continue

        # ---------- 3. OLD LOGPROBS ----------
        # Recompute with the TRAINING model, not vLLM: the two differ
        # numerically and importing vLLM logprobs poisons the ratio.
        order = sorted(range(len(train_eps)), key=lambda i: len(train_eps[i].token_ids))
        train_eps = [train_eps[i] for i in order]
        train_adv = [train_adv[i] for i in order]

        micros = [
            (train_eps[i:i + args.micro_bs], train_adv[i:i + args.micro_bs])
            for i in range(0, len(train_eps), args.micro_bs)
        ]

        old_lp = []
        with torch.no_grad():
            for eps, _ in micros:
                ids, att, _ = collate(eps, pad_id, "cuda")
                old_lp.append(token_logprobs(model, ids, att))

        # ---------- 4. UPDATE ----------
        model.train()
        # token-level normalisation: every policy token gets equal weight, so
        # long trajectories are not down-weighted relative to short ones
        total_tokens = sum(sum(e.mask) for e in train_eps)
        losses, clipfracs, ents = [], [], []
        gnorm = 0.0        # stays 0 if inner_epochs == 0; keeps logging safe

        for _ in range(args.inner_epochs):
            opt.zero_grad(set_to_none=True)
            for (eps, advs), olp in zip(micros, old_lp):
                ids, att, msk = collate(eps, pad_id, "cuda")
                m = msk[:, 1:]                                # align with logprobs
                lp = token_logprobs(model, ids, att)
                A = torch.tensor(advs, device="cuda", dtype=torch.float).unsqueeze(1)

                ratio = torch.exp(lp - olp)
                unclipped = ratio * A
                clipped = torch.clamp(ratio, 1 - args.clip_low, 1 + args.clip_high) * A
                pg = -torch.min(unclipped, clipped)

                loss = (pg * m).sum() / max(total_tokens, 1.0)
                loss.backward()

                losses.append(loss.item())
                clipfracs.append((((ratio - 1).abs() > args.clip_low) * m).sum().item()
                                 / max(m.sum().item(), 1))
                ents.append((-(lp * m).sum() / max(m.sum().item(), 1)).item())

            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        sync_weights(llm, model)

        # ---------- 5. LOG ----------
        mean_r = sum(all_r) / len(all_r)
        mean_f1 = sum(all_f1) / len(all_f1)
        log = {
            "step": step, "reward": round(mean_r, 4), "f1": round(mean_f1, 4),
            "kept_groups": kept_groups, "n_groups": len(groups),
            "grad_norm": round(float(gnorm), 3),
            "clipfrac": round(sum(clipfracs) / len(clipfracs), 4),
            "neg_logp": round(sum(ents) / len(ents), 3),
            "t_rollout": round(t_roll, 1), "t_total": round(time.time() - t0, 1),
            **{k: round(v, 3) for k, v in stats.items()},
        }
        print(json.dumps(log))
        for k, v in log.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(k, v, step)

        if step and step % args.save_every == 0:
            p = os.path.join(args.out, f"step{step}")
            model.save_pretrained(p); tok.save_pretrained(p)

    model.save_pretrained(os.path.join(args.out, "final"))
    tok.save_pretrained(os.path.join(args.out, "final"))


if __name__ == "__main__":
    main()
