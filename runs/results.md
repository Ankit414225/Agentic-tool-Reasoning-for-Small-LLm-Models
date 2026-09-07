# Results

Every row = one run. Change ONE thing per row, or the table proves nothing.
Dev F1 is measured on `data/dev_benchmark.jsonl`, which is never trained on.

Reference points: Qwen2.5-32B scores ~34% F1 on this benchmark. Target is ~60%.

| run | model | change from previous | dev F1 | dev EM | tool calls/q | notes |
|-----|-------|---------------------|--------|--------|--------------|-------|
| base | Qwen3-1.7B | none, naive prompting | | | | baseline. fill this in FIRST |
| sft | Qwen3-1.7B | + rejection-sampling cold start | | | | only if base format compliance < 80% |
| run1 | Qwen3-1.7B | + GRPO, 400 steps | | | | |
| run2 | Qwen3-1.7B | − observation masking | | | | expected to be WORSE; that's the point |
| run3 | Qwen3-1.7B | − length penalty | | | | |
| run4 | Qwen3-1.7B | group_size 8 → 16 | | | | |
| run5 | Qwen3-1.7B | MAX_TURNS 4 → 2 | | | | |
| final | Qwen3-4B | best config from above | | | | |

## Notes per run

### base
- date:
- command:
- observations from reading 20 trajectories by hand:

