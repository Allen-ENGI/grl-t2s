"""
Single source of truth for constants shared across every pipeline stage.

Nothing else in this package should hardcode OBS_DIM, index positions, gammas,
or path conventions — import them from here. This is what stopped the
`X_mean` vs `x_mean` class of bug: one definition, everyone imports it.
"""
import os

# ---- task / environment ----------------------------------------------
TASK_NAME = "peg-insert-side-v3"
SUCCESS_KEY = "success"
OBS_DIM = 39

HAND_POS_IDX = [0, 1, 2]
GRIPPER_IDX = 3          # validation only — never used to define a stage
PEG_POS_IDX = [4, 5, 6]
GOAL_POS_IDX = [36, 37, 38]

MAX_STEPS = 500
CENSOR_LABEL = 300.0      # ceiling assigned to steps_remaining for episodes that never succeed

# ---- SUCCESS INDEX CONTRACT ------------------------------------------
# ONE convention, everywhere in this package:
#
#   success_step == index of the first state that IS successful.
#
# An env step from s_t returns the flag that describes s_{t+1}. So code that
# loops `obs -> step -> flag` must record `success_step = t + 1`, NOT `t`.
#
# This used to differ between modules: data_collection.collect_episode used
# t+1 (correct) to build training labels, while t2s_eval, visualize and
# rl_common all used t. That made every Stage-4 ground-truth curve one frame
# short, so a model reproducing its own training labels perfectly still
# scored a nonzero MAE (~0.8 frames on a 73-step success) — a large share of
# the best observed MAE of 2.54. It also made reward_preview's success latch
# zero out the success-achieving transition, the single most important reward
# in the episode.
SUCCESS_STEP_IS_FIRST_SUCCESSFUL_STATE = True


def steps_remaining_curve(n_states, success_step, censor_label=CENSOR_LABEL):
    """
    Ground-truth steps_remaining for a trajectory of `n_states` states.

    The ONE definition of this curve. success_step follows the contract above
    (index of the first successful state), so the value at success_step is 0.
    success_step=None (never succeeded) gives the censored ceiling.
    """
    import numpy as np
    if success_step is None:
        return np.full(n_states, censor_label, dtype=np.float32)
    return np.array([max(0, success_step - i) for i in range(n_states)], dtype=np.float32)


# ---- discounting: TWO gammas, different jobs, do NOT unify -----------
# RL_GAMMA — the discount SAC uses for its critic, and therefore also the
#   gamma that must appear in the potential-based shaping term in reward_fn
#   and in VecNormalize's return statistics. Ng et al. shaping is only
#   policy-invariant when the gamma inside the shaping term equals the one
#   the learner discounts with; three copies of this number (SAC, reward,
#   VecNormalize) used to be able to drift apart independently.
RL_GAMMA = 0.999

# T2S_BOOTSTRAP_GAMMA — unrelated to the above. Used only inside
#   t2s_targets/t2s_train, where the recursion V = dt + gamma*V needs a finite
#   fixed point dt/(1-gamma) for states that never succeed (100 at 0.99).
#   Pushing this to RL_GAMMA=0.999 would put the fixed point at 1000, far above
#   T2S_PRED_CLIP_MAX, reinstating the arbitrary ceiling the bootstrap scheme
#   exists to remove. Keep them separate.
T2S_BOOTSTRAP_GAMMA = 0.995

TIME_PENALTY = 1.0        # per-step cost for the minimum-time objective
T2S_PRED_CLIP_MAX = 300.0  # predictions clipped to [0, this] in t2s_predict

# ---- reference trajectories -----------------------------------------
# Rolled once at the END of data collection and stored on disk, so every
# downstream tool (t2s_eval, reward_preview, t2s_ordering, make_videos,
# manual_drive) scores the SAME states instead of each re-rolling its own.
REFERENCE_SEEDS = (900_001, 900_002, 900_003)   # far from any collection seed
REFERENCE_KINDS = ("clean_success", "clean_failure",
                   "perturbed_recovery", "perturbed_failure")
# perturbation: this many random actions before the policy takes over. Expert
# rollouts never leave the demonstration manifold, so expert-only references
# cannot exercise the regime where the models were measured to misbehave.
PERTURB_STEPS = 25
REFERENCE_TAIL = 15       # frames kept after success; the rest is a finished task

# ---- downstream RL ---------------------------------------------------
RL_SEEDS = (0, 1, 2)      # 3 seeds: mean AND std across seeds is the point

# ---- directory layout ------------------------------------------------
# Everything is rooted here so the whole pipeline is relocatable.
ROOT_DIR = os.environ.get("T2S_ROOT", os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT_DIR, "data")
TRAIN_RES_DIR = os.path.join(ROOT_DIR, "train_res")

EXPERT_RUN_NAME = "peg-insert"
EXPERT_POLICY_DIR = os.path.join(TRAIN_RES_DIR, "expertPolicy", EXPERT_RUN_NAME)

TASK_SLUG = TASK_NAME.replace("-", "_")

# Fixed scene shared by data collection, T2S training, and policy training —
# all three MUST use the same scene or predictions/rewards are meaningless.
FIXED_TASK_PATH = os.path.join(DATA_DIR, "fixed_task.pkl")


def ensure_dirs():
    for d in (DATA_DIR, TRAIN_RES_DIR, EXPERT_POLICY_DIR):
        os.makedirs(d, exist_ok=True)