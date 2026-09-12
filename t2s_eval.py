"""
Stage 4: T2S model analysis.

Critical requirement from the spec: evaluate on data the model was NOT
trained on. Two distinct kinds of "held out" are supported and should not
be conflated:
  1. held-out ROWS from the same dataset (the val split from t2s_train) —
     tests interpolation within the same policies/trajectories used to build
     the dataset.
  2. held-out POLICIES never used to generate any training data at all
     (eval_success_pol / eval_failure_pol in the original notebook) —
     tests generalization to genuinely unseen behavior. This is the
     stronger and more meaningful test and is what run_full_evaluation
     focuses on.

Metrics, defined explicitly (spec asked for this):
  - MAE (mean absolute error): mean(|prediction - true_steps_remaining|),
    in the same units as steps_remaining (frames). Robust to outliers,
    easy to interpret ("predictions are off by N frames on average").
  - RMSE: sqrt(mean((prediction - true)^2)). Same units as MAE but
    penalizes large errors more heavily — a few badly-wrong predictions
    move RMSE much more than MAE.
  - Pearson correlation: linear correlation between predicted and true
    steps_remaining. Sensitive to the overall linear trend, not to scale
    or offset errors.
  - Spearman correlation: rank correlation between predicted and true.
    Answers "does the model correctly order which states are closer to
    success", independent of whether the absolute scale is right — this is
    usually the metric that matters most for reward shaping, since SAC only
    cares about relative ordering of predicted value along a trajectory.
  - max_wrong_direction: the largest single-frame *increase* in a predicted
    countdown along a successful rollout — a countdown that goes up when it
    should be going down is the failure mode that most directly breaks
    reward shaping (see policy_env.py).
"""
import numpy as np
from scipy.stats import pearsonr, spearmanr

from config import CENSOR_LABEL, SUCCESS_KEY

CONFIRM_BUFFER_DEFAULT = 20  # frames to keep recording after success, matching data_collection


# ---- row-level metrics on any (pred, true) pair --------------------------

def compute_metrics(preds, truths):
    """
    preds, truths: 1D arrays of predicted / true steps_remaining, same length.
    Returns a dict with every metric named and defined in the module docstring.
    """
    preds = np.asarray(preds, dtype=np.float64)
    truths = np.asarray(truths, dtype=np.float64)
    err = preds - truths
    out = dict(
        n=len(preds),
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err ** 2))),
    )
    if len(preds) >= 2 and np.std(preds) > 0 and np.std(truths) > 0:
        out["pearson_r"] = float(pearsonr(preds, truths)[0])
        out["spearman_rho"] = float(spearmanr(preds, truths)[0])
    else:
        out["pearson_r"] = None
        out["spearman_rho"] = None
    return out


def max_wrong_direction(pred_curve):
    """Largest single-frame *increase* in a predicted countdown — the only
    direction that's actually concerning for a countdown signal."""
    if len(pred_curve) < 2:
        return 0.0
    return float(np.max(np.diff(pred_curve)))


# ---- rollout-based evaluation against held-out POLICIES ------------------

def eval_rollout(predict_fn, policy, seed, max_steps=500, random_policy=False,
                  stop_after_success=CONFIRM_BUFFER_DEFAULT):
    """
    Runs one rollout on the fixed scene using `policy` (never one used to
    generate training data), recording the model's prediction at every step.
    predict_fn(obs) -> float prediction, already normalized/clipped by caller.
    Returns (preds: np.ndarray, success_step: int|None).
    """
    from env_utils import make_fixed_scene_env  # lazy: sim dependency

    env = make_fixed_scene_env()
    obs, _ = env.reset(seed=seed)
    preds, success = [], None
    for t in range(max_steps):
        preds.append(predict_fn(obs))
        action = env.action_space.sample() if random_policy else policy.predict(obs, deterministic=True)[0]
        obs, _, term, trunc, info = env.step(action)
        if info.get(SUCCESS_KEY, 0) and success is None:
            success = t
        if success is not None and t >= success + stop_after_success:
            break
        if term or trunc:
            break
    env.close()
    return np.array(preds), success


def run_full_evaluation(predict_fn, eval_success_policy, eval_failure_policy,
                         n_seeds=8, seed_offset=500, max_steps=500):
    """
    The stage-4 deliverable: evaluate against policies that generated NONE
    of the training data. Returns a report dict with per-scenario metrics.

    - "success" scenario: predictions vs. ground-truth steps_remaining
      (computed from the actual success frame), on rollouts from
      eval_success_policy, which reliably succeeds but was held out of
      data collection entirely.
    - "failure" scenario: same policy family but the held-out policy that
      reliably fails — there is no ground-truth steps_remaining here (it
      never reaches 0), so only max_wrong_direction / prediction range are
      reported, not MAE/correlation.
    """
    report = {"success_scenario": [], "failure_scenario": []}

    for seed in range(seed_offset, seed_offset + n_seeds):
        preds, success = eval_rollout(predict_fn, eval_success_policy, seed=seed, max_steps=max_steps)
        if success is None:
            continue  # this "reliable" policy failed on this seed; skip rather than fabricate ground truth
        truths = np.array([max(0, success - t) for t in range(len(preds))], dtype=np.float32)
        m = compute_metrics(preds, truths)
        m["seed"] = seed
        m["max_wrong_direction"] = max_wrong_direction(preds)
        report["success_scenario"].append(m)

    for seed in range(seed_offset, seed_offset + n_seeds):
        preds, success = eval_rollout(predict_fn, eval_failure_policy, seed=seed, max_steps=max_steps)
        report["failure_scenario"].append(dict(
            seed=seed, succeeded_unexpectedly=success is not None,
            pred_min=float(preds.min()), pred_max=float(preds.max()),
            max_wrong_direction=max_wrong_direction(preds),
        ))

    succ_rows = report["success_scenario"]
    if succ_rows:
        report["success_scenario_summary"] = {
            k: float(np.mean([r[k] for r in succ_rows if r.get(k) is not None]))
            for k in ("mae", "rmse", "pearson_r", "spearman_rho", "max_wrong_direction")
            if any(r.get(k) is not None for r in succ_rows)
        }
    report["failure_scenario_summary"] = dict(
        unexpected_success_rate=float(np.mean([r["succeeded_unexpectedly"] for r in report["failure_scenario"]])),
    )
    return report


def plot_combo_comparison(reports, metrics=("mae", "rmse", "spearman_rho", "max_wrong_direction"),
                           save_path=None):
    """
    Bar chart comparing every combo (e.g. mc_succ, td0_succ, ...) across the
    given metrics, one subplot per metric. Reads reports[combo]['success_scenario_summary'].
    Skips a metric entirely if no combo has it (e.g. correlation undefined for
    degenerate predictions) rather than plotting an empty/misleading panel.
    """
    import matplotlib.pyplot as plt

    combos = list(reports.keys())
    available_metrics = [
        m for m in metrics
        if any(m in reports[c].get("success_scenario_summary", {}) for c in combos)
    ]
    if not available_metrics:
        raise ValueError("none of the requested metrics are present in any report's success_scenario_summary")

    fig, axes = plt.subplots(1, len(available_metrics), figsize=(5 * len(available_metrics), 4))
    if len(available_metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available_metrics):
        values = [reports[c].get("success_scenario_summary", {}).get(metric, np.nan) for c in combos]
        bars = ax.bar(combos, values, color="tab:blue")
        # highlight the best combo for this metric (lower is better, except correlations)
        higher_is_better = metric in ("pearson_r", "spearman_rho")
        finite = [v for v in values if not np.isnan(v)]
        if finite:
            best_val = max(finite) if higher_is_better else min(finite)
            for bar, v in zip(bars, values):
                if v == best_val:
                    bar.set_color("tab:green")
        ax.set_title(metric)
        ax.set_xticklabels(combos, rotation=45, ha="right")
        ax.axhline(0, color="gray", lw=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def format_report_table(reports, metrics=("mae", "rmse", "pearson_r", "spearman_rho", "max_wrong_direction")):
    """
    Pure-string comparison table across combos and metrics — no matplotlib,
    so this is unit-testable and also useful for logging/console output
    where a plot isn't practical. Returns the table as a single string;
    print it yourself (kept as a pure function rather than printing directly
    so it's testable and reusable for writing to a log file).
    """
    combos = list(reports.keys())
    available_metrics = [
        m for m in metrics
        if any(m in reports[c].get("success_scenario_summary", {}) for c in combos)
    ]
    if not available_metrics:
        return "(no metrics available in any report's success_scenario_summary)"

    col_w = 14
    header = f"{'combo':<16}" + "".join(f"{m:>{col_w}}" for m in available_metrics)
    lines = [header, "-" * len(header)]
    for c in combos:
        summary = reports[c].get("success_scenario_summary", {})
        row = f"{c:<16}"
        for m in available_metrics:
            v = summary.get(m)
            row += f"{v:>{col_w}.4f}" if v is not None else f"{'--':>{col_w}}"
        lines.append(row)
    return "\n".join(lines)

