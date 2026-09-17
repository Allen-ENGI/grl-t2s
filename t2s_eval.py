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
                  stop_after_success=CONFIRM_BUFFER_DEFAULT, deterministic=False):
    """
    Runs one rollout on the fixed scene using `policy` (never one used to
    generate training data), recording the model's prediction at every step.
    predict_fn(obs) -> float prediction, already normalized/clipped by caller.
    Returns (preds: np.ndarray, success_step: int|None).

    deterministic=False (default) SAMPLES the policy. With the scene pinned
    and MuJoCo deterministic, deterministic=True made every eval seed produce
    the SAME trajectory — so "spread across 8 seeds" measured one rollout
    eight times and the reported std was ~0 by construction, not because the
    models were stable. Sampling gives genuinely different rollouts, so the
    std means something.
    """
    from env_utils import make_fixed_scene_env  # lazy: sim dependency

    env = make_fixed_scene_env()
    # Seed the POLICY's action sampling, not just the env reset. With
    # deterministic=False the actions come from torch's global RNG, so without
    # this an evaluation is not reproducible: re-running the same models gave
    # materially different numbers (one combo moved 1.25 -> 2.34 MAE between
    # two runs). Seeding here keeps rollouts different ACROSS seeds within a
    # run, but identical across runs of the same seed.
    if not deterministic:
        import torch
        torch.manual_seed(seed)
        np.random.seed(seed)

    obs, _ = env.reset(seed=seed)
    preds, success = [], None
    for t in range(max_steps):
        preds.append(predict_fn(obs))
        action = (env.action_space.sample() if random_policy
                  else policy.predict(obs, deterministic=deterministic)[0])
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
                         n_seeds=8, seed_offset=500, max_steps=500, deterministic=False):
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
        preds, success = eval_rollout(predict_fn, eval_success_policy, seed=seed,
                                       max_steps=max_steps, deterministic=deterministic)
        if success is None:
            continue  # this "reliable" policy failed on this seed; skip rather than fabricate ground truth
        truths = np.array([max(0, success - t) for t in range(len(preds))], dtype=np.float32)
        m = compute_metrics(preds, truths)
        m["seed"] = seed
        m["max_wrong_direction"] = max_wrong_direction(preds)
        report["success_scenario"].append(m)

    for seed in range(seed_offset, seed_offset + n_seeds):
        preds, success = eval_rollout(predict_fn, eval_failure_policy, seed=seed,
                                       max_steps=max_steps, deterministic=deterministic)
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


def metric_mean_std(report, metric):
    """
    (mean, std) for a metric across the per-seed evaluation rollouts.

    success_scenario_summary holds only means, but report["success_scenario"]
    keeps one entry per seed — so the spread IS recoverable and should be
    shown. A combo whose MAE is 5.0 +/- 0.3 is a different claim from one at
    5.0 +/- 4.0, and the bar chart previously hid that distinction entirely.
    """
    rows = report.get("success_scenario", [])
    vals = [r[metric] for r in rows if r.get(metric) is not None]
    if not vals:
        # fall back to the summary mean when per-seed detail is unavailable
        m = report.get("success_scenario_summary", {}).get(metric)
        return (m, None)
    return (float(np.mean(vals)), float(np.std(vals)))


def metric_values(report, metric):
    """Per-seed values for a metric (empty list if only a summary exists)."""
    return [r[metric] for r in report.get("success_scenario", [])
            if r.get(metric) is not None]


def _select_combos(reports, combos=None, exclude=None):
    """Resolve an explicit include list / exclude list against the report keys."""
    names = list(reports.keys())
    if combos is not None:
        missing = [c for c in combos if c not in reports]
        if missing:
            raise KeyError(f"combos not in reports: {missing}. Available: {names}")
        names = list(combos)
    if exclude:
        names = [c for c in names if c not in exclude]
    if not names:
        raise ValueError("no combos left to plot after filtering")
    return names


def plot_combo_comparison(reports, metrics=("mae", "rmse", "spearman_rho", "max_wrong_direction"),
                           save_path=None, combos=None, exclude=None, show_std=True,
                           sort_by=None):
    """
    Bar chart comparing every combo (e.g. mc_succ, td0_succ, ...) across the
    given metrics, one subplot per metric. Reads reports[combo]['success_scenario_summary'].
    Skips a metric entirely if no combo has it (e.g. correlation undefined for
    degenerate predictions) rather than plotting an empty/misleading panel.
    """
    import matplotlib.pyplot as plt

    names = _select_combos(reports, combos, exclude)
    if sort_by:
        higher = sort_by in ("pearson_r", "spearman_rho")
        names.sort(key=lambda c: (metric_mean_std(reports[c], sort_by)[0]
                                   if metric_mean_std(reports[c], sort_by)[0] is not None
                                   else (float("-inf") if higher else float("inf"))),
                    reverse=higher)
    available_metrics = [
        m for m in metrics
        if any(m in reports[c].get("success_scenario_summary", {}) for c in names)
    ]
    if not available_metrics:
        raise ValueError("none of the requested metrics are present in any report's success_scenario_summary")

    fig, axes = plt.subplots(1, len(available_metrics), figsize=(5 * len(available_metrics), 4))
    if len(available_metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available_metrics):
        stats = [metric_mean_std(reports[c], metric) for c in names]
        values = [np.nan if m is None else m for m, _sd in stats]
        errs = [0.0 if sd is None else sd for _m, sd in stats] if show_std else None

        bars = ax.bar(names, values, yerr=errs, capsize=4, color="tab:blue",
                       error_kw=dict(ecolor="0.25", lw=1.2))
        if show_std:
            # error bars alone are invisible when the spread is small relative
            # to the bar height, so also draw every seed as a dot and print the
            # number — a tiny bar with +/-0.02 and one with +/-2.0 must not look
            # the same.
            for xi, c in enumerate(names):
                pts = metric_values(reports[c], metric)
                if pts:
                    ax.scatter([xi] * len(pts), pts, s=14, color="black",
                                zorder=3, alpha=0.75)
            tops = []
            for xi, ((mean, sd), c) in enumerate(zip(stats, names)):
                if mean is None:
                    continue
                pts = metric_values(reports[c], metric)
                # clear the error bar AND the highest seed dot
                y = max([mean + (sd or 0.0)] + pts)
                tops.append(y)
            headroom = max(tops, default=1.0)
            for xi, ((mean, sd), c) in enumerate(zip(stats, names)):
                if mean is None:
                    continue
                pts = metric_values(reports[c], metric)
                y = max([mean + (sd or 0.0)] + pts)
                label = f"{mean:.2f}" + (f"\n±{sd:.2f}" if sd is not None else "")
                ax.text(xi, y + 0.03 * headroom, label, ha="center", va="bottom",
                         fontsize=7.5)
            ax.set_ylim(top=headroom * 1.30)
        # highlight the best combo for this metric (lower is better, except correlations)
        higher_is_better = metric in ("pearson_r", "spearman_rho")
        finite = [v for v in values if not np.isnan(v)]
        if finite:
            best_val = max(finite) if higher_is_better else min(finite)
            winners = [v for v in values if v == best_val]
            # a metric where every combo ties (e.g. correlations all at 0.98)
            # has no winner — colouring them all green implies a distinction
            # the numbers do not support
            if len(winners) < len(finite):
                for bar, v in zip(bars, values):
                    if v == best_val:
                        bar.set_color("tab:green")
        ax.set_title(metric + ("  (mean ± std over seeds)" if show_std else ""))
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.axhline(0, color="gray", lw=0.5)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def collect_eval_trajectories(eval_success_policy, eval_failure_policy, seed=500, max_steps=500,
                               stop_after_success=CONFIRM_BUFFER_DEFAULT):
    """
    Rolls out exactly TWO reference trajectories — one from the held-out
    success policy, one from the held-out failure policy — and returns their
    raw observations plus ground truth.

    Deliberately separated from prediction: the observations are collected
    ONCE, then every model predicts on these same stored states (see
    plot_sample_trajectories). If each model rolled out its own trajectory,
    differences between the prediction curves would be confounded by
    differences in the trajectories themselves, and the comparison would be
    meaningless.

    Returns {"success": traj, "failure": traj} where traj is a dict with
    "obs" (T, obs_dim), "success_step" (int|None), "truth" (T,) or None.
    """
    from env_utils import make_fixed_scene_env

    def _roll(policy, seed):
        env = make_fixed_scene_env()
        obs, _ = env.reset(seed=seed)
        obs_list, success_step = [], None
        for t in range(max_steps):
            obs_list.append(obs.copy())
            action, _ = policy.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(action)
            if info.get(SUCCESS_KEY, 0) and success_step is None:
                success_step = t
            if success_step is not None and t >= success_step + stop_after_success:
                break
            if term or trunc:
                break
        env.close()
        obs_arr = np.array(obs_list, dtype=np.float32)
        truth = (np.array([max(0, success_step - t) for t in range(len(obs_arr))], dtype=np.float32)
                 if success_step is not None else None)
        return dict(obs=obs_arr, success_step=success_step, truth=truth)

    return {
        "success": _roll(eval_success_policy, seed),
        "failure": _roll(eval_failure_policy, seed),
    }


def plot_sample_trajectories(trajectories, predictors, save_path=None, censor_label=300.0):
    """
    Two side-by-side panels (success trajectory, failure trajectory), with
    EVERY model's prediction overlaid on the same states, plus ground truth.

    trajectories: output of collect_eval_trajectories.
    predictors: {combo_name: predict_fn} — each predict_fn(obs) -> float,
        e.g. from t2s_predict.load_t2s_predictor.

    What to look for:
      - Success panel: a usable model tracks the dashed ground-truth line
        downward and reaches ~0 at the success marker. Flat lines, or curves
        that never descend, mean the model isn't tracking progress at all.
      - Failure panel: there is no ground truth (success never happens), so
        the useful signal is whether predictions stay HIGH. A model that
        confidently predicts "almost there" throughout a failing trajectory
        is actively misleading as a reward, even with a good MAE on
        successful rollouts.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    colors = plt.cm.tab10.colors

    for ax, (scenario, traj) in zip(axes, trajectories.items()):
        obs = traj["obs"]
        for i, (combo, predict_fn) in enumerate(predictors.items()):
            preds = np.array([predict_fn(o) for o in obs], dtype=np.float32)
            ax.plot(preds, color=colors[i % len(colors)], label=combo, linewidth=1.6)

        if traj["truth"] is not None:
            ax.plot(traj["truth"], color="black", linestyle="--", linewidth=2, label="ground truth")
        if traj["success_step"] is not None:
            ax.axvline(traj["success_step"], color="black", alpha=0.3, linewidth=1)
            ax.annotate("success", (traj["success_step"], ax.get_ylim()[1] * 0.9),
                         fontsize=8, rotation=90, va="top")
        else:
            ax.axhline(censor_label, color="gray", linestyle=":", linewidth=1,
                        label=f"censor label ({censor_label:.0f})")

        n_steps = len(obs)
        ax.set_title(f"{scenario} trajectory "
                      f"({'succeeded at ' + str(traj['success_step']) if traj['success_step'] is not None else 'never succeeded'}"
                      f", {n_steps} steps)")
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("predicted steps remaining")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def format_report_table(reports, metrics=("mae", "rmse", "pearson_r", "spearman_rho", "max_wrong_direction"),
                         combos=None, exclude=None, show_std=True, sort_by=None):
    """
    Pure-string comparison table across combos and metrics — no matplotlib,
    so this is unit-testable and also useful for logging/console output
    where a plot isn't practical. Returns the table as a single string;
    print it yourself (kept as a pure function rather than printing directly
    so it's testable and reusable for writing to a log file).
    """
    if not reports:
        return "(no reports)"
    names = _select_combos(reports, combos, exclude)
    if sort_by:
        higher = sort_by in ("pearson_r", "spearman_rho")
        names.sort(key=lambda c: (metric_mean_std(reports[c], sort_by)[0]
                                   if metric_mean_std(reports[c], sort_by)[0] is not None
                                   else (float("-inf") if higher else float("inf"))),
                    reverse=higher)

    available_metrics = [
        m for m in metrics
        if any(m in reports[c].get("success_scenario_summary", {}) for c in names)
    ]
    if not available_metrics:
        return "(no metrics available in any report's success_scenario_summary)"

    col_w = 18 if show_std else 14
    header = f"{'combo':<18}" + "".join(f"{m[:col_w]:>{col_w}}" for m in available_metrics)
    lines = [header, "-" * len(header)]
    for c in names:
        row = f"{c:<18}"
        for m in available_metrics:
            mean, sd = metric_mean_std(reports[c], m)
            if mean is None:
                row += f"{'--':>{col_w}}"
            elif show_std and sd is not None:
                row += f"{mean:>{col_w - 7}.3f} ±{sd:<5.3f}"
            else:
                row += f"{mean:>{col_w}.4f}"
        lines.append(row)
    return "\n".join(lines)



def evaluate_runs(t2s_runs, eval_success_policy, eval_failure_policy,
                   combos=None, n_seeds=8, seed_offset=500, deterministic=False,
                   verbose=True, **eval_kwargs):
    """
    Evaluate the SAME combo names across SEVERAL t2s_model runs, in one table.

    Needed because combo names repeat across runs: a gamma probe trained with
    censor_schemes=('censor',) produces "tdlambda_succ", exactly the name the
    main run already uses. Without a per-run label the second would silently
    overwrite the first in the reports dict.

    t2s_runs: {label: (run_dir, summary_rows)}. The label is appended to each
        combo name, so an empty label leaves names unchanged. Example:

            evaluate_runs({
                '':     (t2s_run_dir, summary_rows),   # main run, gamma=1.0 censor
                '_g99': (probe_dir,   probe_rows),     # same combos at gamma=0.99
            }, eval_success_pol, eval_failure_pol)

        gives 'tdlambda_succ' and 'tdlambda_succ_g99' side by side.

    combos: optional list of combo names to evaluate from each run; None = all.
    """
    import t2s_predict

    reports = {}
    for label, (run_dir, rows) in t2s_runs.items():
        wanted = rows if combos is None else [r for r in rows if r["combo"] in combos]
        if combos is not None and len(wanted) != len(combos):
            missing = set(combos) - {r["combo"] for r in wanted}
            raise KeyError(f"run {label or '(main)'} is missing combos: {sorted(missing)}")
        for row in wanted:
            method, condition = row["combo"].rsplit("_", 1)
            key = row["combo"] + label
            if verbose:
                print(f"  evaluating {key} ...", flush=True)
            predict_fn = t2s_predict.load_t2s_predictor(
                run_dir, method, condition, seed=row["best_seed"])
            reports[key] = run_full_evaluation(
                predict_fn, eval_success_policy, eval_failure_policy,
                n_seeds=n_seeds, seed_offset=seed_offset,
                deterministic=deterministic, **eval_kwargs)
    return reports