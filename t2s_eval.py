"""
Stage 4: T2S model analysis.

Six public functions. The previous version had sixteen, ten of which nothing
outside this module ever called, plus a `plot_sample_trajectories` that
t2s_video now does better and that nothing called at all.

    assert_policies_held_out   verify the evaluation policies were reserved
    evaluate_runs              evaluate every combo -> {name: report}
    format_report_table        report -> string table
    plot_combo_comparison      report -> bar chart
    collect_eval_trajectories  reference rollouts for reward_preview / video
    run_full_evaluation        one model's report (used by t2s_video)

Two kinds of "held out", not to be conflated:
  1. held-out ROWS — the `val` split assigned at collection time and stored in
     dataset.npz. Interpolation within the same policies.
  2. held-out POLICIES — checkpoints reserved via Stage 1's `holdout`, never
     rolled out during collection. Generalization to unseen behaviour.
`assert_policies_held_out` checks (2) against the recorded manifest rather
than trusting that whoever ran Stage 1 remembered which files it touched.

WHICH CORRELATIONS MEAN ANYTHING
--------------------------------
Within one success rollout the truth is max(0, S - i): exactly linear in the
frame index. So correlating predictions against it is correlating against -i.
Measured on the observed 73-step-success + 20-frame-tail shape:

    perfect model                    pearson +1.000  spearman +1.000  MAE   0.0
    predictions x 3                  pearson +1.000  spearman +1.000  MAE  58.1
    predictions + 100                pearson +1.000  spearman +1.000  MAE 100.0
    straight ramp ignoring S         pearson +0.989  spearman +0.995  MAE 121.0
    truth vs. the frame index        pearson +0.989

NO correlation fixes this, because Pearson and Spearman are both invariant to
a global affine transform:

    error class                   within-traj rho   pooled rho
    +100 global offset                     1.0000       1.0000
    x3 global scale                        1.0000       1.0000
    per-trajectory normalisation           1.0000       0.7997
    level drifts with trajectory           1.0000       0.8610

Hence the division of labour in the reported metrics:
  bias, calibration_slope  the ONLY metrics that see a global offset or scale
                           error. bias is signed; NEGATIVE means the model
                           claims success is nearer than it is, the dangerous
                           direction for a reward. calibration_slope is 1.0
                           when correct.
  pooled_spearman          across rollouts. The only one that sees a level
                           consistent within each trajectory but drifting
                           between them — the observed 62-vs-200 spread.
  step_slope_error         mean |predicted decrement - 1|. Literally the
                           quantity reward_fn's difference modes consume, so
                           the most directly predictive number here.
  monotonicity_rho         the old correlation, renamed. Shape only. Read it
                           with max_wrong_direction.

Other corrections: ground truth was off by one (success recorded as the step
index t rather than the first successful state t+1, so a model reproducing its
training labels perfectly still scored MAE ~0.8); MAE now excludes the
post-success tail by default; max_wrong_direction is pre-success only and
clamped at 0; correlations average in Fisher z, since a plain mean of r is
biased toward 0 badly near |r|~1.
"""
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr

from config import CENSOR_LABEL, SUCCESS_KEY, steps_remaining_curve

CONFIRM_BUFFER = 20          # frames recorded after success
HIGHER_IS_BETTER = ("pooled_spearman", "monotonicity_rho")
CORRELATIONS = ("pooled_pearson", "pooled_spearman", "monotonicity_rho")
REPORT_METRICS = ("mae", "bias", "calibration_slope", "step_slope_error",
                  "pooled_spearman", "monotonicity_rho", "max_wrong_direction")


# ---- held-out bookkeeping ----------------------------------------------

def assert_policies_held_out(dataset_summary_path, policy_paths, strict=True):
    """
    Verify every policy about to be evaluated was RESERVED during collection,
    not collected from. Stage 4's whole claim rests on this, and it used to
    rest on memory. Raises on overlap when strict.
    """
    import json

    with open(dataset_summary_path) as f:
        summary = json.load(f)
    collected = set(summary.get("collected_checkpoints") or [])
    holdout = set(summary.get("holdout_checkpoints") or [])
    names = [os.path.basename(p) for p in policy_paths]

    leaked = [n for n in names if n in collected]
    unreserved = [n for n in names if n not in holdout]
    if strict and leaked:
        raise AssertionError(
            f"evaluation policies {leaked} WERE collected from (sources: {sorted(collected)}). "
            "The held-out-policy claim does not hold. Re-collect with these in --holdout.")
    if strict and unreserved:
        raise AssertionError(
            f"evaluation policies {unreserved} are not in the declared holdout "
            f"({sorted(holdout)}). Pass them to Stage 1's --holdout so it is recorded.")
    return dict(evaluated=names, leaked=leaked, not_declared_holdout=unreserved,
                ok=not leaked and not unreserved)


# ---- metrics -------------------------------------------------------------

def _corr(fn, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    v = fn(a, b)[0]
    return None if not np.isfinite(v) else float(v)


def fisher_mean(values):
    """Mean of correlations via Fisher z; a plain mean of r is biased toward 0."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(v):
        return None
    return float(np.tanh(np.mean(np.arctanh(np.clip(v, -0.999999, 0.999999)))))


def _mean(values, metric):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return fisher_mean(vals) if metric in CORRELATIONS else float(np.mean(vals))


def compute_metrics(preds, truths, success_step=None):
    """
    Per-rollout metrics. `success_step` (index of the first successful state)
    splits the curve: error metrics use the PRE-SUCCESS portion, since the
    confirm buffer after success has truth == 0 throughout and would blend
    "can it count down" with "does it decay to zero when finished".
    """
    preds, truths = np.asarray(preds, float), np.asarray(truths, float)
    n_pre = len(preds) if success_step is None else max(0, min(success_step, len(preds)))
    p, t = preds[:n_pre], truths[:n_pre]
    err_all = preds - truths

    out = dict(n=len(preds), n_pre_success=len(p),
               mae_with_tail=float(np.mean(np.abs(err_all))),
               pred_min=float(preds.min()), pred_max=float(preds.max()))
    if not len(p):
        out.update({k: None for k in ("mae", "rmse", "bias", "calibration_slope",
                                      "monotonicity_rho", "step_slope_error",
                                      "max_wrong_direction")})
        return out

    err = p - t
    out["mae"] = float(np.mean(np.abs(err)))
    out["rmse"] = float(np.sqrt(np.mean(err ** 2)))
    out["bias"] = float(np.mean(err))                       # signed: sees an offset
    out["calibration_slope"] = float(np.polyfit(t, p, 1)[0]) if np.std(t) > 0 else None
    # honest name: Spearman against the frame index, i.e. "is the curve
    # decreasing", NOT ranking quality
    out["monotonicity_rho"] = _corr(spearmanr, p, -np.arange(len(p), dtype=float))
    if len(p) >= 2:
        d = -np.diff(p)                                     # predicted decrement
        out["step_slope_error"] = float(np.mean(np.abs(d - 1.0)))
        out["max_wrong_direction"] = float(max(np.max(-d), 0.0))
    else:
        out["step_slope_error"] = None
        out["max_wrong_direction"] = 0.0
    return out


def max_wrong_direction(pred_curve, success_step=None):
    """
    Magnitude of the worst UPWARD jump in a predicted countdown; 0 if it never
    rises. Pre-success only — the confirm buffer has no countdown left to get
    wrong, so wobble there used to contaminate this.
    """
    c = np.asarray(pred_curve, float)
    if success_step is not None:
        c = c[:max(0, min(int(success_step), len(c)))]
    return 0.0 if len(c) < 2 else float(max(np.max(np.diff(c)), 0.0))


def pooled_ranking_metrics(pairs):
    """
    Correlations across trajectories, on pooled pre-success (pred, truth).

    Sees a level consistent within each rollout but drifting between them.
    Does NOT see a global affine error — nothing does; use bias and
    calibration_slope for that. pairs: (preds, truths, success_step) each.
    """
    P, T = [], []
    for preds, truths, ss in pairs:
        preds, truths = np.asarray(preds, float), np.asarray(truths, float)
        n = len(preds) if ss is None else max(0, min(ss, len(preds)))
        if n:
            P.append(preds[:n]); T.append(truths[:n])
    if not P:
        return dict(pooled_pearson=None, pooled_spearman=None, n_pooled=0)
    P, T = np.concatenate(P), np.concatenate(T)
    return dict(pooled_pearson=_corr(pearsonr, P, T),
                pooled_spearman=_corr(spearmanr, P, T), n_pooled=int(len(P)))


# ---- rollouts ------------------------------------------------------------

def _rollout(predict_fn, policy, seed, max_steps=500, stop_after_success=CONFIRM_BUFFER,
             record_obs=False, record_frames=False):
    """
    One rollout on the fixed scene. Returns (preds, success_step, obs, frames).

    success_step is the index of the first state that IS successful (t+1 for a
    flag returned by the step out of s_t); it used to be recorded as t, making
    every ground-truth curve a frame short.

    The policy is always SAMPLED: with the scene pinned and MuJoCo
    deterministic, taking the policy mean made every eval seed produce the
    SAME trajectory, so "spread across 8 seeds" measured one rollout eight
    times. Both RNGs are seeded per rollout, so rollouts differ across seeds
    but are identical across runs of one seed.
    """
    import torch

    from env_utils import make_fixed_scene_env

    env = make_fixed_scene_env(render_mode="rgb_array" if record_frames else None)
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)

    preds, obs_list, frames, success_step = [], [], [], None
    for t in range(max_steps):
        if predict_fn is not None:
            preds.append(float(predict_fn(obs)))
        if record_obs:
            obs_list.append(np.asarray(obs, dtype=np.float32).copy())
        if record_frames:
            from t2s_video import render_frame  # MuJoCo bottom-up fix
            frames.append(render_frame(env))
        action, _ = policy.predict(obs, deterministic=False)
        obs, _r, term, trunc, info = env.step(action)
        if info.get(SUCCESS_KEY, 0) and success_step is None:
            success_step = t + 1
        if success_step is not None and t >= success_step + stop_after_success:
            break
        if term or trunc:
            break
    env.close()
    return (np.array(preds), success_step,
            np.array(obs_list, dtype=np.float32) if record_obs else None,
            frames if record_frames else None)


def run_full_evaluation(predict_fn, success_policy, failure_policy, n_seeds=8,
                        seed_offset=500, max_steps=500):
    """
    One model's report: per-seed metrics on rollouts of the held-out success
    policy, prediction ranges on the held-out failure policy (no ground truth
    exists there), and pooled cross-rollout correlations.
    """
    report = {"success_scenario": [], "failure_scenario": []}
    pairs = []

    for seed in range(seed_offset, seed_offset + n_seeds):
        preds, ss, _o, _f = _rollout(predict_fn, success_policy, seed, max_steps)
        if ss is None:
            continue        # this "reliable" policy failed here; don't fabricate truth
        truths = steps_remaining_curve(len(preds), ss)
        m = compute_metrics(preds, truths, success_step=ss)
        m.update(seed=seed, success_step=int(ss))
        report["success_scenario"].append(m)
        pairs.append((preds, truths, ss))

    for seed in range(seed_offset, seed_offset + n_seeds):
        preds, ss, _o, _f = _rollout(predict_fn, failure_policy, seed, max_steps)
        report["failure_scenario"].append(dict(
            seed=seed, succeeded_unexpectedly=ss is not None,
            pred_min=float(preds.min()), pred_max=float(preds.max()),
            # how far below the censor level it drifts on a trajectory that
            # never succeeds: the false-optimism magnitude
            optimism_vs_censor=float(CENSOR_LABEL - preds.min()),
            max_wrong_direction=max_wrong_direction(preds)))

    report["pooled_ranking"] = pooled_ranking_metrics(pairs)
    rows = report["success_scenario"]
    if rows:
        keys = ("mae", "rmse", "mae_with_tail", "bias", "calibration_slope",
                "monotonicity_rho", "step_slope_error", "max_wrong_direction", "pred_max")
        report["summary"] = {k: _mean([r.get(k) for r in rows], k) for k in keys}
        report["summary"].update({k: report["pooled_ranking"][k]
                                  for k in ("pooled_pearson", "pooled_spearman")})
    fails = report["failure_scenario"]
    report["failure_summary"] = dict(
        unexpected_success_rate=float(np.mean([r["succeeded_unexpectedly"] for r in fails]))
        if fails else None,
        mean_pred_min=float(np.mean([r["pred_min"] for r in fails])) if fails else None)
    return report


def evaluate_on_val_split(predict_fn, dataset_path, max_rows=50_000, seed=0):
    """
    Held-out-ROWS evaluation on the split stored by data_collection. Thousands
    of states against eight rollouts, and its correlation spans many episodes
    by construction, so it is the pooled kind.
    """
    d = np.load(dataset_path)
    if "split" not in d.files:
        raise KeyError(f"{dataset_path} has no 'split' array — re-collect with the "
                       "current data_collection.")
    X, y, split = d["X"], d["y_steps"], d["split"].astype(str)
    # rows from failed episodes carry the censored ceiling, not a true label
    idx = np.flatnonzero((split == "val") & (y != CENSOR_LABEL))
    if not len(idx):
        return dict(n=0, note="val split empty after filtering")
    if len(idx) > max_rows:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_rows, replace=False))

    preds = np.array([predict_fn(o) for o in X[idx]], dtype=float)
    truths = y[idx].astype(float)
    err = preds - truths
    return dict(n=int(len(idx)), mae=float(np.mean(np.abs(err))),
                bias=float(np.mean(err)),
                calibration_slope=float(np.polyfit(truths, preds, 1)[0])
                if np.std(truths) > 0 else None,
                pooled_pearson=_corr(pearsonr, preds, truths),
                pooled_spearman=_corr(spearmanr, preds, truths))


def collect_eval_trajectories(success_policy, failure_policy, seeds=(500, 501, 502),
                              max_steps=500, record_frames=False):
    """
    Reference rollouts from the held-out policies, for reward_preview and
    t2s_video. Returns {"success": [traj, ...], "failure": [traj, ...]}.

    SEVERAL seeds, not one. This used to roll out exactly two trajectories
    with a deterministic policy, and reward_preview computed its go/no-go
    return_gap from that single pair — model selection on a sample of two.
    Observations are collected ONCE and every model predicts on the same
    stored states, so differences between prediction curves are the model.
    """
    def roll(policy, seed):
        _p, ss, obs, frames = _rollout(None, policy, seed, max_steps,
                                       record_obs=True, record_frames=record_frames)
        return dict(obs=obs, success_step=ss, seed=seed, frames=frames,
                    truth=steps_remaining_curve(len(obs), ss) if ss is not None else None)

    return {"success": [roll(success_policy, s) for s in seeds],
            "failure": [roll(failure_policy, s) for s in seeds]}


# ---- reporting -----------------------------------------------------------

def metric_mean_std(report, metric):
    """(mean, std) across per-seed rollouts. Pooled metrics have no spread."""
    if metric.startswith("pooled_"):
        return (report.get("pooled_ranking", {}).get(metric)
                or report.get("summary", {}).get(metric), None)
    vals = [r[metric] for r in report.get("success_scenario", [])
            if r.get(metric) is not None]
    if not vals:
        return (report.get("summary", {}).get(metric), None)
    return (_mean(vals, metric), float(np.std(vals)))


def _ordered(reports, sort_by):
    names = list(reports)
    if not sort_by:
        return names
    higher = sort_by in HIGHER_IS_BETTER or sort_by == "calibration_slope"

    def key(c):
        m = metric_mean_std(reports[c], sort_by)[0]
        if m is None:
            return float("-inf") if higher else float("inf")
        return -abs(m - 1.0) if sort_by == "calibration_slope" else m    # best is 1.0
    return sorted(names, key=key, reverse=higher)


LEGEND = """
mae/bias          : pre-success frames only (mae_with_tail keeps the confirm buffer)
bias              : signed; NEGATIVE = claims success is nearer than it is
calibration_slope : 1.0 is correct; no correlation can see a wrong scale
step_slope_error  : mean |predicted decrement - 1|; what the shaped reward consumes
pooled_spearman   : ACROSS trajectories — sees level drift between rollouts
monotonicity_rho  : WITHIN trajectory vs frame index; ~0.99 for any decreasing curve"""


def format_report_table(reports, metrics=REPORT_METRICS, sort_by="mae", show_std=True):
    """Pure-string comparison table. No matplotlib, so it works headless."""
    if not reports:
        return "(no reports)"
    names = _ordered(reports, sort_by)
    cols = [m for m in metrics
            if any(metric_mean_std(reports[c], m)[0] is not None for c in names)]
    if not cols:
        return "(no metrics available)"

    w = 18 if show_std else 14
    header = f"{'combo':<20}" + "".join(f"{m[:w]:>{w}}" for m in cols)
    lines = [header, "-" * len(header)]
    for c in names:
        row = f"{c:<20}"
        for m in cols:
            mean, sd = metric_mean_std(reports[c], m)
            if mean is None:
                row += f"{'--':>{w}}"
            elif show_std and sd is not None:
                row += f"{mean:>{w - 7}.3f} ±{sd:<5.3f}"
            else:
                row += f"{mean:>{w}.4f}"
        lines.append(row)
    return "\n".join(lines) + "\n" + LEGEND


def plot_combo_comparison(reports, metrics=REPORT_METRICS, save_path=None, sort_by="mae"):
    """
    One subplot per metric. Every seed is drawn as a dot as well as an error
    bar, because error bars alone are invisible when the spread is small
    relative to the bar height — a bar with +/-0.02 must not look like one
    with +/-2.0.
    """
    import matplotlib.pyplot as plt

    names = _ordered(reports, sort_by)
    cols = [m for m in metrics
            if any(metric_mean_std(reports[c], m)[0] is not None for c in names)]
    if not cols:
        raise ValueError("none of the requested metrics are present")

    fig, axes = plt.subplots(1, len(cols), figsize=(4.6 * len(cols), 4.2))
    axes = np.atleast_1d(axes)

    for ax, metric in zip(axes, cols):
        stats = [metric_mean_std(reports[c], metric) for c in names]
        values = [np.nan if m is None else m for m, _ in stats]
        errs = [sd or 0.0 for _, sd in stats]
        bars = ax.bar(range(len(names)), values, yerr=errs, capsize=4, color="tab:blue")

        for xi, c in enumerate(names):
            pts = [r[metric] for r in reports[c].get("success_scenario", [])
                   if r.get(metric) is not None]
            if pts:
                ax.scatter([xi] * len(pts), pts, s=14, color="black", zorder=3, alpha=.75)

        finite = [v for v in values if not np.isnan(v)]
        if finite:
            best = (min(finite, key=lambda v: abs(v - 1.0)) if metric == "calibration_slope"
                    else max(finite) if metric in HIGHER_IS_BETTER else min(finite))
            # a metric where every combo ties has no winner; colouring them all
            # green implies a distinction the numbers do not support
            if sum(v == best for v in values) < len(finite):
                for bar, v in zip(bars, values):
                    if v == best:
                        bar.set_color("tab:green")
        if metric == "calibration_slope":
            ax.axhline(1.0, color="tab:red", ls="--", lw=1)
        ax.set_title(metric, fontsize=9)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def evaluate_runs(t2s_run_dir, summary_rows, success_policy, failure_policy,
                  n_seeds=8, dataset_path=None, verbose=True):
    """
    Evaluate every combo in one t2s_model run. Returns {combo: report}.

    dataset_path also runs the held-out-ROWS evaluation and attaches it as
    report["val_split"] — worth doing, since 8 rollouts is thin.
    """
    import t2s_predict

    reports = {}
    for row in summary_rows:
        method, condition = row["combo"].rsplit("_", 1)
        if verbose:
            print(f"  evaluating {row['combo']} ...", flush=True)
        fn = t2s_predict.load_t2s_predictor(t2s_run_dir, method, condition,
                                            seed=row["best_seed"])
        rep = run_full_evaluation(fn, success_policy, failure_policy, n_seeds=n_seeds)
        if dataset_path:
            rep["val_split"] = evaluate_on_val_split(fn, dataset_path)
        reports[row["combo"]] = rep
    return reports