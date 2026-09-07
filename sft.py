"""
sft.py -- cold start via rejection sampling (STaR / RFT).  [TODO: not written yet]

WHY THIS EXISTS
GRPO needs at least one successful rollout per group. If it has none, every
reward in the group is identical, every advantage is zero, and the model learns
nothing while burning full rollout cost. On a 1.7B doing multi-turn tool use,
that is the default outcome from a cold base model.

WHEN TO SKIP IT
Run `eval.py` on the base model first. If format compliance is already above
~80% and dev F1 is clearly non-zero, go straight to grpo.py.

THE RECIPE
  1. sample k=8 rollouts per training question at temperature 1.0
     -> reuse rollout.generate_episodes, it already does this
  2. keep trajectories with F1 > 0.8
     -> reuse reward.compute_reward
  3. SFT on the kept trajectories for 1-2 epochs, lr ~1e-5
     -> CRITICAL: mask the loss exactly as rollout.py does. Episode.mask is
        already built for this; use it directly, do not recompute it.
  4. optional: repeat from step 1 with the new checkpoint (that is the "iterated"
     part of STaR, and it is the cheap fallback if GRPO won't stabilise in time)

A few thousand kept trajectories is enough. This is a format/competence primer,
not the main event.
"""
