"""
Single source of truth for constants shared across every pipeline stage.

Nothing else in this package should hardcode OBS_DIM, index positions, or
path conventions — import them from here. This is what stopped the
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

# ---- directory layout ---------------------------------------------------
# Everything is rooted here so the whole pipeline is relocatable.
ROOT_DIR = os.environ.get("T2S_ROOT", os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT_DIR, "data")
TRAIN_RES_DIR = os.path.join(ROOT_DIR, "train_res")

# Expert policy pretraining is currently OUT of scope for this pipeline.
# We assume checkpoints already exist here (as per current run):
#   train_res/expertPolicy/<EXPERT_RUN_NAME>/{TASK_SLUG}_<steps>.zip
#   train_res/expertPolicy/<EXPERT_RUN_NAME>/{TASK_SLUG}_final.zip
#   train_res/expertPolicy/<EXPERT_RUN_NAME>/eval_history.json
# EXPERT_RUN_NAME is just a folder label for organizing runs by task/experiment
# (e.g. "peg-insert") — it does not need to match TASK_SLUG.
EXPERT_RUN_NAME = "peg-insert"
EXPERT_POLICY_DIR = os.path.join(TRAIN_RES_DIR, "expertPolicy", EXPERT_RUN_NAME)

# Derived, task-agnostic naming — used for checkpoint/dir names instead of
# hardcoding "peg_insert" as PolicyTrain.ipynb did. Change TASK_NAME above
# and every downstream name (checkpoint files, run dirs) follows automatically.
TASK_SLUG = TASK_NAME.replace("-", "_")

# Fixed scene shared by data collection, T2S training, and policy training —
# all three MUST use the same scene or predictions/rewards are meaningless.
FIXED_TASK_PATH = os.path.join(DATA_DIR, "fixed_task.pkl")

def ensure_dirs():
    for d in (DATA_DIR, TRAIN_RES_DIR, EXPERT_POLICY_DIR):
        os.makedirs(d, exist_ok=True)
