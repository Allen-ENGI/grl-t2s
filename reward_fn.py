"""
Reward arithmetic — the single source of truth.

Deliberately dependency-free (no gymnasium, no torch, no SB3) so that BOTH
the live training wrapper (policy_env.Time2SuccessRewardWrapper) and the
offline preview (reward_preview.compute_step_rewards) can import it, and so
it stays unit-testable without a sim installed.

This module exists because an earlier version duplicated the formula in both
places: editing the wrapper changed training but silently did nothing to the
preview. Same bug class as the X_mean/x_mean normalization mismatch — two
copies of one contract, drifting apart.

REWARD_MODES:
  "absolute"    r = -pred(s')
      Every step costs the current predicted time-to-success. Depends on the
      prediction's absolute LEVEL, which varies a lot between T2S models
      (62 vs 200 on the same failure trajectory in observed runs). Long
      episodes accumulate huge negative return, so a 500-step failure can
      dominate a 73-step success purely by length.

  "difference"  r = pred(s) - pred(s')
      Potential-based shaping (Ng et al.): preserves the optimal policy, and
      uses only the prediction's SHAPE, which is far more consistent across
      models than its level. But it telescopes to pred(first) - pred(last),
      so standing still earns exactly 0 — and when moving risks large
      negative spikes, freezing becomes a safe local optimum. Observed
      directly in the reward preview: a long zero plateau on failure
      trajectories alongside dips to -40 for moving.

  "difference_timed"  r = pred(s) - pred(s') - time_penalty
      Restores the task reward that "difference" dropped. For a minimum-time
      objective, T(s_t) = dt + T(s_{t+1}) means the per-step cost IS -1 and
      the value difference is the shaping term; plain "difference" kept the
      shaping and discarded the objective. With an accurate prediction a
      genuinely progressing step gives pred(s)-pred(s') ~ 1 so r ~ 0 (free),
      while hovering gives r = -time_penalty (costly). Strictly this breaks
      policy-invariance, but -1/step is the true objective here, so it is
      restoring the objective rather than adding a bias.
"""

REWARD_MODES = ("absolute", "difference", "difference_timed")
DEFAULT_TIME_PENALTY = 1.0


def step_reward(prev_pred, pred_now, reward_mode="absolute", time_penalty=DEFAULT_TIME_PENALTY):
    """
    Reward for one transition, given the T2S prediction before and after it.
    prev_pred is unused by "absolute" (pass None if unavailable).
    """
    if reward_mode == "absolute":
        return -pred_now
    if reward_mode == "difference":
        return prev_pred - pred_now
    if reward_mode == "difference_timed":
        return prev_pred - pred_now - time_penalty
    raise ValueError(f"unknown reward_mode {reward_mode!r}, expected one of {REWARD_MODES}")
