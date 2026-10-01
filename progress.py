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
    control, reusing data_collection.detect_control_onset. CAUTION: this fires
    on PUSHING as well as grasping — a hand shoving the peg across the table
    moves with it at a constant offset, which is exactly the signature it
    tests. Read it with peg_lifted, never alone.
  - peg_lift_max / peg_lifted: how far the peg rose above its reset height.
    The one metric that separates a grasp from a shove, because pushing along
    the table leaves height unchanged.
  - obs_movement: total observation change over the episode. Near-zero
    means the arm became static — which matters for interpreting flat T2S
    predictions (a constant prediction on a constant observation is
    correct model behavior, not model failure).
"""
import os

import numpy as np

from config import HAND_POS_IDX, PEG_POS_IDX, GOAL_POS_IDX

# Threshold for "the peg was displaced". 1e-6 was far too low: a random-action
# rollout measured a peg-z range of 0.0074, which is 7,400x that — so
# peg_moved_rate read 1.00 for every policy including a random one, and the
# column was saturated rather than informative. 1cm is about the scale of a
# deliberate nudge and well above settling jitter.
PEG_MOVED_EPS = 0.01
PEG_JITTER_EPS = 1e-6  # the old value, kept to report "touched at all"

# A grasp LIFTS the peg. Pushing it along the table does not. Both satisfy
# detect_control_onset (hand and peg move together with a constant offset) and
# both reduce peg-to-goal distance, so neither of those metrics can tell them
# apart — a run reporting control_rate 1.00 and 74% of the way to the hole can
# still be a policy shoving the peg across the table. Height is the one signal
# that separates them.
PEG_LIFT_EPS = 0.01    # metres above the reset height that counts as lifted


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
    # PEG_POS_IDX is [x, y, z]; the third component is height
    peg_lift = float((peg_positions[:, 2] - peg_positions[0, 2]).max())

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
        peg_jittered=bool(peg_drift > PEG_JITTER_EPS),
        # vertical displacement only: the discriminator between a grasp and a shove
        peg_lift_max=peg_lift,
        peg_lifted=bool(peg_lift > PEG_LIFT_EPS),
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


def peg_goal_progress_fraction(distance, reset_distance, success_distance):
    """
    Converts a raw peg-to-goal distance into a 0-1 progress fraction.

    Raw distance is misleading on its own because success does NOT occur at
    distance 0: PEG_POS_IDX tracks the peg's body centre while GOAL_POS_IDX
    is the hole, so at insertion the centre is still offset by roughly half
    the peg length (measured ~0.192 in this task, vs ~0.4135 at reset).
    A bar chart of raw distance therefore compresses the entire meaningful
    range into the top half of the axis and makes "no progress at all" look
    close to "solved".

    Returns 0.0 at the reset distance and 1.0 at the success distance.
    """
    span = reset_distance - success_distance
    if span <= 0:
        return None
    return float(np.clip((reset_distance - distance) / span, 0.0, 1.0))


def load_run_histories(pattern):
    """
    Reads eval_history.json for every run directory matching a glob pattern.
    Returns {run_name: history_list}, skipping runs with no history.
    """
    import glob
    import json

    out = {}
    for path in sorted(glob.glob(os.path.join(pattern, "eval_history.json"))):
        with open(path) as f:
            hist = json.load(f)
        if hist:
            out[os.path.basename(os.path.dirname(path))] = hist
    return out


def summarize_run_progress(history, reset_distance=None, success_distance=None):
    """
    Best-over-training values for one run's eval history. Uses the BEST
    (not final) value because observed runs reached peak progress early and
    then regressed — reporting only the final checkpoint hides that entirely.
    """
    def best(key, reducer=min):
        vals = [h[key] for h in history if h.get(key) is not None]
        return reducer(vals) if vals else None

    out = dict(
        best_hand_peg=best("best_min_hand_peg_distance"),
        final_hand_peg=(history[-1].get("best_min_hand_peg_distance")),
        best_peg_goal=best("best_min_peg_goal_distance"),
        final_peg_goal=(history[-1].get("best_min_peg_goal_distance")),
        best_success=best("success_rate", max),
        final_success=history[-1].get("success_rate"),
        peg_moved_rate=best("peg_moved_rate", max),
        mean_obs_movement=best("mean_obs_movement", max),
        n_evals=len(history),
    )
    if reset_distance is not None and success_distance is not None and out["best_peg_goal"] is not None:
        out["progress_fraction"] = peg_goal_progress_fraction(
            out["best_peg_goal"], reset_distance, success_distance)
    return out


def plot_progress_comparison(histories, reset_distance=0.4135, success_distance=0.192,
                              save_path=None):
    """
    Bar comparison across runs for every model-independent progress metric.

    histories: {run_name: eval_history list}, e.g. from load_run_histories.
    reset_distance / success_distance: calibration for the peg-goal axis —
        measure these for your task rather than trusting the defaults
        (see peg_goal_progress_fraction for why they matter).
    """
    import matplotlib.pyplot as plt

    names = list(histories.keys())
    rows = [summarize_run_progress(h, reset_distance, success_distance)
            for h in histories.values()]
    x = np.arange(len(names))

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    (ax1, ax2), (ax3, ax4) = axes

    # --- hand -> peg: the REACH phase. 0 means contact.
    best_h = [r["best_hand_peg"] if r["best_hand_peg"] is not None else np.nan for r in rows]
    fin_h = [r["final_hand_peg"] if r["final_hand_peg"] is not None else np.nan for r in rows]
    w = 0.38
    ax1.bar(x - w/2, best_h, w, label="best during run", color="tab:green")
    ax1.bar(x + w/2, fin_h, w, label="final", color="tab:green", alpha=0.45)
    ax1.set_ylabel("hand-to-peg distance")
    ax1.set_title("Reach phase (lower = closer; 0 = contact)")
    ax1.legend(fontsize=8)

    # --- peg -> goal, with the two reference lines that make it readable
    best_p = [r["best_peg_goal"] if r["best_peg_goal"] is not None else np.nan for r in rows]
    fin_p = [r["final_peg_goal"] if r["final_peg_goal"] is not None else np.nan for r in rows]
    ax2.bar(x - w/2, best_p, w, label="best during run", color="tab:orange")
    ax2.bar(x + w/2, fin_p, w, label="final", color="tab:orange", alpha=0.45)
    ax2.axhline(reset_distance, color="red", ls="--", lw=1.2, label=f"reset ({reset_distance:.3f})")
    ax2.axhline(success_distance, color="green", ls="--", lw=1.2,
                 label=f"success ({success_distance:.3f})")
    ax2.set_ylabel("peg-to-goal distance")
    ax2.set_title("Manipulation phase (success is NOT at 0)")
    ax2.legend(fontsize=7)

    # --- normalized progress: 0 = untouched, 1 = solved
    frac = [(r.get("progress_fraction") or 0.0) for r in rows]
    colors = ["tab:blue" if f > 0 else "lightgray" for f in frac]
    ax3.bar(x, frac, color=colors)
    ax3.set_ylim(0, 1)
    ax3.set_ylabel("fraction of the way to success")
    ax3.set_title("Normalized progress (0 = peg untouched, 1 = solved)")
    for xi, f in zip(x, frac):
        ax3.text(xi, f + 0.02, f"{f:.0%}", ha="center", fontsize=8)

    # --- did the object move at all, and was the arm active
    moved = [(r.get("peg_moved_rate") or 0.0) for r in rows]
    ax4.bar(x - w/2, moved, w, label="peg moved rate", color="tab:purple")
    mv = [r.get("mean_obs_movement") or 0.0 for r in rows]
    mv_norm = [m / max(mv) if max(mv) > 0 else 0 for m in mv]
    ax4.bar(x + w/2, mv_norm, w, label="obs movement (normalized)",
             color="tab:gray", alpha=0.6)
    ax4.set_ylim(0, 1.05)
    ax4.set_title("Object contact and arm activity")
    ax4.legend(fontsize=8)

    for ax in (ax1, ax2, ax3, ax4):
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def format_progress_table(histories, reset_distance=0.4135, success_distance=0.192):
    """Pure-string version of plot_progress_comparison, for logging."""
    header = (f"{'run':<34}{'hand->peg':>11}{'peg->goal':>11}{'progress':>10}"
              f"{'moved':>8}{'succ':>7}")
    lines = [header, "-" * len(header)]
    for name, hist in histories.items():
        r = summarize_run_progress(hist, reset_distance, success_distance)
        def f(v, p=4):
            return f"{v:.{p}f}" if v is not None else "--"
        frac = r.get("progress_fraction")
        lines.append(
            f"{name:<34}{f(r['best_hand_peg']):>11}{f(r['best_peg_goal']):>11}"
            f"{(f'{frac:.0%}' if frac is not None else '--'):>10}"
            f"{f(r['peg_moved_rate'], 2):>8}{f(r['best_success'], 2):>7}"
        )
    return "\n".join(lines)
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
                     "obs_movement", "n_steps",
                     # reward-side keys the RL eval now records; averaged the
                     # same way, and skipped when absent (they only exist when
                     # the eval env is the T2S-wrapped one)
                     "peg_lift_max",
                     "t2s_pred_min", "t2s_pred_final", "false_optimism",
                     "reward_mean", "reward_total", "reward_nonneg_frac",
                     "dead_step_frac")
    out = {}
    for k in numeric_keys:
        vals = [m[k] for m in metric_dicts if m.get(k) is not None]
        out[f"mean_{k}"] = float(np.mean(vals)) if vals else None
    out["success_rate"] = float(np.mean([bool(m.get("succeeded")) for m in metric_dicts]))
    # fraction of episodes in which the object was touched at all — the
    # clearest single indicator of whether the reach phase is working
    out["peg_moved_rate"] = float(np.mean([bool(m.get("peg_moved")) for m in metric_dicts]))
    out["peg_jitter_rate"] = float(np.mean([bool(m.get("peg_jittered")) for m in metric_dicts]))
    # the honest "did it pick the peg up" rate
    out["peg_lifted_rate"] = float(np.mean([bool(m.get("peg_lifted")) for m in metric_dicts]))
    control = [m["achieved_control"] for m in metric_dicts if m.get("achieved_control") is not None]
    out["control_rate"] = float(np.mean(control)) if control else None
    # best (closest) single episode, not just the average — shows whether ANY
    # episode got close, which the mean can hide
    mins = [m["min_peg_goal_distance"] for m in metric_dicts if m.get("min_peg_goal_distance") is not None]
    out["best_min_peg_goal_distance"] = float(np.min(mins)) if mins else None
    hand_mins = [m["min_hand_peg_distance"] for m in metric_dicts if m.get("min_hand_peg_distance") is not None]
    out["best_min_hand_peg_distance"] = float(np.min(hand_mins)) if hand_mins else None
    return out