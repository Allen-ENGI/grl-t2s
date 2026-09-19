from config import RL_GAMMA, TIME_PENALTY, T2S_PRED_CLIP_MAX

REWARD_MODES = ("absolute", "difference", "difference_timed")

# kept as module-level names because callers pass time_penalty=None to mean
# "use the default"; the values themselves live in config.
DEFAULT_TIME_PENALTY = TIME_PENALTY
DEFAULT_GAMMA = RL_GAMMA

# modes whose arithmetic involves the shaping potential, and therefore gamma
SHAPED_MODES = ("difference", "difference_timed")


def step_reward(prev_pred, pred_now, reward_mode="difference_timed",
                time_penalty=DEFAULT_TIME_PENALTY, gamma=DEFAULT_GAMMA):
    
    if reward_mode == "absolute":
        return -pred_now                                         # no gamma: real per-step cost
    if reward_mode == "difference":
        return prev_pred - gamma * pred_now                      # Ng shaping, Phi = -pred
    if reward_mode == "difference_timed":
        return prev_pred - gamma * pred_now - time_penalty       # shaping + the objective
    raise ValueError(f"unknown reward_mode {reward_mode!r}, expected one of {REWARD_MODES}")


def hover_reward(pred_level, reward_mode="difference_timed",
                 time_penalty=DEFAULT_TIME_PENALTY, gamma=DEFAULT_GAMMA):
    """
    What this reward pays for a step that changes nothing, at a constant
    prediction of `pred_level`. Must be negative, or freezing is an optimum.
    """
    return step_reward(pred_level, pred_level, reward_mode=reward_mode,
                       time_penalty=time_penalty, gamma=gamma)

def assert_shaping_gamma_safe(gamma=DEFAULT_GAMMA, time_penalty=DEFAULT_TIME_PENALTY,
                              pred_max=T2S_PRED_CLIP_MAX, reward_mode="difference_timed"):
    
    """
    Refuse a (gamma, time_penalty, pred_max) combination that pays the policy
    to stand still. Raises ValueError with the numbers rather than letting a
    sweep spend hours discovering it as a hovering policy.

    Call with pred_max set to the LARGEST prediction the chosen T2S model
    actually emits (reward_preview reports it), not just the clip ceiling —
    a bootstrap-trained model saturating near 100 is safe at gamma=0.995,
    while a censor-trained model reaching 300 is not.
    """
    
    if reward_mode not in SHAPED_MODES:
        return True
    r = hover_reward(pred_max, reward_mode=reward_mode,
                     time_penalty=time_penalty, gamma=gamma)
    if r >= 0:
        raise ValueError(
            f"reward_mode={reward_mode!r} with gamma={gamma}, time_penalty={time_penalty}, "
            f"max prediction {pred_max} pays r={r:+.3f} per step for standing still, so "
            f"freezing is a local optimum (this is the empty-handed hovering failure).\n"
            f"Need (1-gamma)*pred_max < time_penalty, i.e. gamma > "
            f"{1 - time_penalty / pred_max:.5f} at this pred_max, or a larger time_penalty "
            f"(> {(1 - gamma) * pred_max:.3f})."
        )
    return True
