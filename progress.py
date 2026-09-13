"""
Model-independent task-progress metrics.

Why these exist: when no policy succeeds, success_rate is 0 everywhere and
tells you nothing. Cumulative T2S reward CANNOT substitute, because each T2S
model defines its own reward scale — comparing them would be comparing
rulers, not progress (and VecNormalize rescales each run anyway).

Everything here is computed from raw OBSERVATIONS via fixed indices, so it
is identical in scale and meaning across every combo, seed, and reward mode.
That makes it the only valid basis for ranking runs that all score 0% success.

Metrics:
  - hand_peg_distance(obs): Euclidean distance between the end effector
    (HAND_POS_IDX) and the object (PEG_POS_IDX). Tracks the REACH phase,
    which peg-to-goal distance structurally cannot see: an arm that reaches
    the peg perfectly but never grasps leaves the peg at its reset position,
    so min_peg_goal_distance stays bit-identical to reset whether the arm
    reached it or never approached. Observed runs showed exactly this
    (0.41354388902050304 in 5 of 6 runs and in the random baseline) while
    the arm was visually reaching the peg.
  - min_hand_peg_distance: closest the effector ever got to the object.
    The milestone BEFORE grasp, and the one that distinguishes "not even
    reaching" from "reaching but failing to grasp" — two failures that need
    completely different fixes.
  - peg_goal_distance(obs): Euclidean distance between the manipulated
    object (PEG_POS_IDX) and the goal (GOAL_POS_IDX), in sim units.
  - min_peg_goal_distance: the closest the peg ever got during an episode.
    The key partial-credit signal — distinguishes "flailing far away" from
    "got the peg to the hole but failed to insert", which success_rate
    cannot distinguish at all.
  - final_peg_goal_distance: where it ended up (vs. where it got closest —
    a large gap means the policy approached then drifted away).
  - peg_moved: whether the peg's position changed at all from its reset
    value. Cheap, unambiguous check — if False, the object was never
    touched regardless of what any distance metric suggests.
  - achieved_control: whether the object ever came under the effector's
    control, reusing data_collection.detect_control_onset. A meaningful
    milestone on the way to insertion.
  - obs_movement: total observation change over the episode. Near-zero
    means the arm became static — which matters for interpreting flat T2S
    predictions (a constant prediction on a constant observation is
    correct model behavior, not model failure).
"""
import numpy as np

from config import HAND_POS_IDX, PEG_POS_IDX, GOAL_POS_IDX

PEG_MOVED_EPS = 1e-6   # peg position change below this counts as "never touched"


def hand_peg_distance(obs):
    """Euclidean end-effector-to-object distance for a single observation."""
    obs = np.asarray(obs, dtype=np.float64)
    return float(np.linalg.norm(obs[HAND_POS_IDX] - obs[PEG_POS_IDX]))


def hand_peg_distance_curve(obs_seq):
    """Per-step end-effector-to-object distance for a whole trajectory."""
    obs_seq = np.asarray(obs_seq, dtype=np.float64)
    return np.linalg.norm(obs_seq[:, HAND_POS_IDX] - obs_seq[:, PEG_POS_IDX], axis=1)


def peg_goal_distance(obs):
    """Euclidean peg-to-goal distance for a single observation."""
    obs = np.asarray(obs, dtype=np.float64)
    return float(np.linalg.norm(obs[PEG_POS_IDX] - obs[GOAL_POS_IDX]))


def peg_goal_distance_curve(obs_seq):
    """Per-step peg-to-goal distance for a whole trajectory."""
    obs_seq = np.asarray(obs_seq, dtype=np.float64)
    return np.linalg.norm(obs_seq[:, PEG_POS_IDX] - obs_seq[:, GOAL_POS_IDX], axis=1)


def obs_movement(obs_seq):
    """
    Total per-step observation change, summed over the episode. Used to tell
    "the model outputs a constant because the scene is static" apart from
    "the model outputs a constant despite the scene changing" — only the
    latter is a model problem.
    """
    obs_seq = np.asarray(obs_seq, dtype=np.float64)
    if len(obs_seq) < 2:
        return 0.0
    return float(np.abs(np.diff(obs_seq, axis=0)).sum())


def compute_progress_metrics(obs_seq, success_step=None, include_control_onset=True):
    """
    All model-independent progress metrics for one trajectory.
    obs_seq: (T, obs_dim) array of raw observations.
    """
    obs_seq = np.asarray(obs_seq, dtype=np.float32)
    if len(obs_seq) == 0:
        return dict(n_steps=0, min_hand_peg_distance=None, final_hand_peg_distance=None,
                     min_peg_goal_distance=None, final_peg_goal_distance=None,
                     mean_peg_goal_distance=None, achieved_control=None,
                     peg_moved=False, peg_max_drift=0.0,
                     obs_movement=0.0, succeeded=False, success_step=None)

    dist = peg_goal_distance_curve(obs_seq)
    hand_dist = hand_peg_distance_curve(obs_seq)
    peg_positions = np.asarray(obs_seq, dtype=np.float64)[:, PEG_POS_IDX]
    peg_drift = float(np.abs(peg_positions - peg_positions[0]).max())

    out = dict(
        n_steps=int(len(obs_seq)),
        # reach phase
        min_hand_peg_distance=float(hand_dist.min()),
        final_hand_peg_distance=float(hand_dist[-1]),
        # manipulation phase
        min_peg_goal_distance=float(dist.min()),
        final_peg_goal_distance=float(dist[-1]),
        mean_peg_goal_distance=float(dist.mean()),
        # was the object touched at all?
        peg_moved=bool(peg_drift > PEG_MOVED_EPS),
        peg_max_drift=peg_drift,
        obs_movement=obs_movement(obs_seq),
        succeeded=success_step is not None,
        success_step=int(success_step) if success_step is not None else None,
    )

    if include_control_onset:
        try:
            from data_collection import detect_control_onset
            onset = detect_control_onset(obs_seq)
            out["achieved_control"] = onset is not None
            out["control_onset_step"] = int(onset) if onset is not None else None
        except Exception:
            # detector is heuristic and task-tuned; never let it break metric
            # collection during a long training run
            out["achieved_control"] = None
            out["control_onset_step"] = None

    return out


def aggregate_progress_metrics(metric_dicts):
    """
    Averages a list of compute_progress_metrics outputs (e.g. across eval
    episodes), skipping None values rather than propagating them.
    """
    if not metric_dicts:
        return {}
    numeric_keys = ("min_hand_peg_distance", "final_hand_peg_distance",
                     "min_peg_goal_distance", "final_peg_goal_distance",
                     "mean_peg_goal_distance", "peg_max_drift",
                     "obs_movement", "n_steps")
    out = {}
    for k in numeric_keys:
        vals = [m[k] for m in metric_dicts if m.get(k) is not None]
        out[f"mean_{k}"] = float(np.mean(vals)) if vals else None
    out["success_rate"] = float(np.mean([bool(m.get("succeeded")) for m in metric_dicts]))
    # fraction of episodes in which the object was touched at all — the
    # clearest single indicator of whether the reach phase is working
    out["peg_moved_rate"] = float(np.mean([bool(m.get("peg_moved")) for m in metric_dicts]))
    control = [m["achieved_control"] for m in metric_dicts if m.get("achieved_control") is not None]
    out["control_rate"] = float(np.mean(control)) if control else None
    # best (closest) single episode, not just the average — shows whether ANY
    # episode got close, which the mean can hide
    mins = [m["min_peg_goal_distance"] for m in metric_dicts if m.get("min_peg_goal_distance") is not None]
    out["best_min_peg_goal_distance"] = float(np.min(mins)) if mins else None
    hand_mins = [m["min_hand_peg_distance"] for m in metric_dicts if m.get("min_hand_peg_distance") is not None]
    out["best_min_hand_peg_distance"] = float(np.min(hand_mins)) if hand_mins else None
    return out