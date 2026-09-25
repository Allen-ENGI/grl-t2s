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
WINDOW = 10                  # sliding window for the sustained-dead-signal metrics
HIGHER_IS_BETTER = ("pooled_spearman", "monotonicity_rho")
CORRELATIONS = ("pooled_pearson", "pooled_spearman", "monotonicity_rho")
# FOUR metrics by default, one per question the reward has to answer:
#   mae                    does it predict TIME correctly (success rollouts)
#   step_slope_error       is the per-step DECREMENT right — the quantity the
#                          shaped reward literally consumes
#   dead_window_fraction   is there any gradient at all, or sustained flat runs
#   fail_pred_min          does it stay pessimistic on a trajectory that never
#                          succeeds — the false-optimism defect
#
# Dropped from the default, with reasons:
#   fail_optimism      = CENSOR_LABEL - fail_pred_min exactly. A mirrored
#                        duplicate; two panels carried one fact.
#   calibration_slope  a wrong scale IS a wrong per-step decrement, which
#                      step_slope_error already measures in the units the
#                      reward uses.
#   bias               a constant offset nearly CANCELS in difference shaping
#                      (it survives only as c*(1-gamma) per step), so it is the
#                      least reward-relevant of the error terms.
#   worst_window_*     keeps the same information as dead_window_fraction in a
#                      less interpretable unit.
#   the correlations   read ~1.00 for every model on the observed runs.
# All remain available: metrics=REPORT_METRICS + EXTRA_METRICS.
REPORT_METRICS = ("mae", "bias", "fail_pred_min")
# REPORT_METRICS = ("mae", "bias", "decrement_mean", "decrement_std", "fail_pred_min")
EXTRA_METRICS = ("calibration_slope", "step_slope_error", "dead_window_fraction",
                 "worst_window_slope_error", "fail_pred_spread", "fail_optimism",
                 "max_wrong_direction")


# ---- held-out bookkeeping ----------------------------------------------

def assert_policies_held_out(dataset_summary_path, policy_paths, strict=False):
    """
    Report whether the evaluation policies were collected from.

    strict now defaults to FALSE, and the claim it supports has been weakened
    deliberately. Holding out whole checkpoints cost every one of their failure
    modes — failure modes are checkpoint-specific, a 100k policy fails
    differently from a 500k one — in exchange for a generalisation claim that
    did not probe the regime that matters: neither held-out expert resembles an
    untrained SAC policy, and the states that actually broke the models belong
    to neither. Episode-level holdout (assign_splits, stratified per source)
    gives validation power from EVERY checkpoint instead.

    So the honest framing is held-out EPISODES, not held-out POLICIES. Pass
    strict=True to restore the old behaviour if a run does reserve checkpoints.
    """
    import json

    with open(dataset_summary_path) as f:
        summary = json.load(f)
    collected = set(summary.get("collected_checkpoints") or [])
    holdout = set(summary.get("holdout_checkpoints") or [])
    names = [os.path.basename(p) for p in policy_paths]

    leaked = [n for n in names if n in collected]
    unreserved = [n for n in names if n not in holdout]
    if leaked and not strict:
        print(f"  note: evaluation policies {leaked} were also collected from. "
              "Stage 3 measures held-out EPISODES, not held-out policies.")
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


def compute_metrics(preds, truths, success_step=None, window=WINDOW):
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
                                      "decrement_mean", "decrement_std",
                                      "max_wrong_direction", "worst_window_slope_error",
                                      "dead_window_fraction", "worst_window_start")})
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
        err = np.abs(d - 1.0)
        # step_slope_error = mean|d - 1| CONFLATES two different faults, and for
        # an unbiased model it is mostly a noise measure: mean|d-1| ~= 1.13*sigma.
        # A perfectly calibrated model with 0.9 of per-step jitter scores 1.00 —
        # identical to a FLAT prediction, which is the opposite failure. Split it:
        #   decrement_mean  systematic. 1.0 correct, <1 compressed, 0 flat.
        #                   The reward is decrement - c, so this IS the reward's
        #                   mean, shifted.
        #   decrement_std   noise. Differencing amplifies prediction jitter, and
        #                   this is the noise the policy receives per step.
        out["decrement_mean"] = float(np.mean(d))
        out["decrement_std"] = float(np.std(d))
        out["step_slope_error"] = float(np.mean(err))
        out["max_wrong_direction"] = float(max(np.max(-d), 0.0))
        out.update(window_metrics(err, window=window))
    else:
        out["step_slope_error"] = None
        out["decrement_mean"] = out["decrement_std"] = None
        out["max_wrong_direction"] = 0.0
        out.update(worst_window_slope_error=None, dead_window_fraction=None,
                   worst_window_start=None)
    return out


def window_metrics(step_err, window=WINDOW, dead=1.0 - 1e-9):
    """
    Sliding-window view of the per-step slope error.

    The episode MEAN hides where the signal dies, and max_wrong_direction is a
    single frame, so there was nothing in between. Measured on a model that is
    perfect for 45 steps and then flat for 28:

        step_slope_error (mean)  0.38   <- looks BETTER than a noisy model's 1.20
        worst 10-step window     1.00   <- exactly the dead-signal line

    A mean can be dragged down by a long good stretch; a window cannot. 1.0 is
    the threshold because a perfectly flat prediction gives |0 - 1| = 1.0
    exactly, so a window at or above it carries no usable per-step gradient.

      worst_window_slope_error  the worst sustained stretch
      dead_window_fraction      share of windows at or above the dead line
      worst_window_start        where it is, as a fraction through the
                                pre-success span. Near 1.0 means the signal
                                dies in the APPROACH to success, which is the
                                worst place for it: that is where the shaped
                                reward most needs to discriminate.
    """
    step_err = np.asarray(step_err, float)
    if len(step_err) < window:
        window = max(1, len(step_err))
    if not len(step_err):
        return dict(worst_window_slope_error=None, dead_window_fraction=None,
                    worst_window_start=None)
    w = np.convolve(step_err, np.ones(window) / window, mode="valid")
    return dict(worst_window_slope_error=float(w.max()),
                dead_window_fraction=float(np.mean(w >= dead)),
                worst_window_start=float(np.argmax(w) / max(len(w) - 1, 1)))


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
            # how much the prediction MOVES on a failing rollout. A model
            # pinned at the clip ceiling scores a perfect fail_pred_min while
            # giving no gradient at all, so the two must be read together.
            pred_spread=float(preds.max() - preds.min()),
            # how far below the censor level it drifts on a trajectory that
            # never succeeds: the false-optimism magnitude
            optimism_vs_censor=float(CENSOR_LABEL - preds.min()),
            max_wrong_direction=max_wrong_direction(preds)))

    report["pooled_ranking"] = pooled_ranking_metrics(pairs)
    rows = report["success_scenario"]
    if rows:
        keys = ("mae", "rmse", "mae_with_tail", "bias", "calibration_slope",
                "monotonicity_rho", "step_slope_error", "decrement_mean",
                "decrement_std", "worst_window_slope_error",
                "dead_window_fraction", "worst_window_start",
                "max_wrong_direction", "pred_max")
        report["summary"] = {k: _mean([r.get(k) for r in rows], k) for k in keys}
        report["summary"].update({k: report["pooled_ranking"][k]
                                  for k in ("pooled_pearson", "pooled_spearman")})
    fails = report["failure_scenario"]
    report["failure_summary"] = dict(
        unexpected_success_rate=float(np.mean([r["succeeded_unexpectedly"] for r in fails]))
        if fails else None,
        mean_pred_min=float(np.mean([r["pred_min"] for r in fails])) if fails else None)
    # Promote the failure numbers into `summary` under a fail_ prefix so the
    # table and the chart can reach them. They were computed, stored, and then
    # never displayed — which is why a model could look fine on every plotted
    # panel while predicting "25 steps to success" throughout a trajectory that
    # never succeeds. The success rollouts come from an expert that goes
    # straight to the goal, so hovering NEVER OCCURS in them and no
    # success-scenario metric can see it.
    if fails:
        report.setdefault("summary", {}).update(
            # lowest prediction reached on a trajectory that never succeeds.
            # LOWER IS WORSE: it is the model's most confident false claim.
            fail_pred_min=float(np.mean([r["pred_min"] for r in fails])),
            fail_pred_spread=float(np.mean([r.get("pred_spread", 0.0) for r in fails])),
            # same thing as a distance below the censor level
            fail_optimism=float(np.mean([r["optimism_vs_censor"] for r in fails])),
            fail_pred_mean=float(np.mean([(r["pred_min"] + r["pred_max"]) / 2
                                          for r in fails])),
        )
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
    # Two exclusions, not one:
    #   y == CENSOR_LABEL  failed-episode rows carry the ceiling, not a label
    #   y == 0             the POST-SUCCESS TAIL. dataset.npz is untrimmed, so
    #                      ~89% of every successful episode is a peg already in
    #                      the hole. Keeping those rows made val MAE a measure
    #                      of "does it decay to zero when finished" rather than
    #                      "can it count down", and it dominated the number:
    #                      val bias came out nearly equal to val MAE, i.e. all
    #                      errors positive, which is over-prediction on y=0
    #                      rows. It also made val MAE incomparable with the
    #                      rollout MAE, which is pre-success only — 15.38 vs
    #                      1.21 for the same model.
    idx = np.flatnonzero((split == "val") & (y != CENSOR_LABEL) & (y > 0))
    if not len(idx):
        return dict(n=0, note="val split empty after filtering")
    if len(idx) > max_rows:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_rows, replace=False))

    preds = np.array([predict_fn(o) for o in X[idx]], dtype=float)
    truths = y[idx].astype(float)
    err = preds - truths
    return dict(n=int(len(idx)), note="pre-success rows only (0 < y < censor)",
                mae=float(np.mean(np.abs(err))),
                bias=float(np.mean(err)),
                calibration_slope=float(np.polyfit(truths, preds, 1)[0])
                if np.std(truths) > 0 else None,
                pooled_pearson=_corr(pearsonr, preds, truths),
                pooled_spearman=_corr(spearmanr, preds, truths))


def evaluate_on_val_episodes(predict_fn, dataset_path, max_episodes=None, seed=0):
    """
    The same report as run_full_evaluation, but scored on the held-out val
    EPISODES in dataset.npz instead of eight fresh rollouts.

    WHY THIS IS USUALLY THE BETTER NUMBER
    -------------------------------------
    `split` is assigned per EPISODE at collection time, so val rows form whole
    trajectories in frame order: every metric here is computable, temporal ones
    included. Against that, the live path rolls exactly n_seeds trajectories.
    Thousands of frames across dozens of episodes is a far larger sample, needs
    no simulator, and takes seconds.

    The live rollouts were justified when Stage 1 RESERVED checkpoints: they
    then came from policies absent from training, which the val split could not
    provide. That holdout was dropped, so the reference policies were collected
    from too — and the live rollouts are no longer a different distribution,
    just a smaller one.

    What neither covers is the regime that actually matters: both are expert
    generated, and the states an untrained RL policy visits appear in neither.
    That is what manual_drive and the perturbed references are for.

    Returns the same {"success_scenario", "failure_scenario", "summary"} shape,
    so format_report_table and plot_combo_comparison work unchanged.
    """
    d = np.load(dataset_path)
    for k in ("split", "episode_ids", "frame_idxs"):
        if k not in d.files:
            raise KeyError(f"{dataset_path} has no {k!r} — re-collect with the "
                           "current data_collection.")
    X, y = d["X"], d["y_steps"]
    ep, fr, split = d["episode_ids"], d["frame_idxs"], d["split"].astype(str)

    val = np.flatnonzero(split == "val")
    eps = np.unique(ep[val])
    if max_episodes and len(eps) > max_episodes:
        eps = np.sort(np.random.default_rng(seed).choice(eps, max_episodes,
                                                         replace=False))

    report = {"success_scenario": [], "failure_scenario": []}
    pairs = []
    for e in eps:
        idx = np.flatnonzero(ep == e)
        idx = idx[np.argsort(fr[idx])]              # frame order, not file order
        ye = y[idx].astype(float)
        preds = np.array([predict_fn(o) for o in X[idx]], dtype=float)

        if np.all(ye == CENSOR_LABEL):              # never succeeded
            report["failure_scenario"].append(dict(
                seed=int(e), succeeded_unexpectedly=False,
                pred_min=float(preds.min()), pred_max=float(preds.max()),
                pred_spread=float(preds.max() - preds.min()),
                optimism_vs_censor=float(CENSOR_LABEL - preds.min()),
                max_wrong_direction=max_wrong_direction(preds)))
            continue

        zero = np.flatnonzero(ye == 0.0)
        ss = int(zero[0]) if len(zero) else None
        if ss is None or ss < 2:
            continue                                # no countdown to score
        m = compute_metrics(preds, ye, success_step=ss)
        m.update(seed=int(e), success_step=ss)
        report["success_scenario"].append(m)
        pairs.append((preds, ye, ss))

    report["pooled_ranking"] = pooled_ranking_metrics(pairs)
    rows = report["success_scenario"]
    if rows:
        keys = ("mae", "rmse", "mae_with_tail", "bias", "calibration_slope",
                "monotonicity_rho", "step_slope_error", "decrement_mean",
                "decrement_std", "worst_window_slope_error",
                "dead_window_fraction", "worst_window_start",
                "max_wrong_direction", "pred_max")
        report["summary"] = {k: _mean([r.get(k) for r in rows], k) for k in keys}
        report["summary"].update({k: report["pooled_ranking"][k]
                                  for k in ("pooled_pearson", "pooled_spearman")})
    fails = report["failure_scenario"]
    if fails:
        report.setdefault("summary", {}).update(
            fail_pred_min=float(np.mean([r["pred_min"] for r in fails])),
            fail_pred_spread=float(np.mean([r.get("pred_spread", 0.0) for r in fails])),
            fail_optimism=float(np.mean([r["optimism_vs_censor"] for r in fails])),
            fail_pred_mean=float(np.mean([(r["pred_min"] + r["pred_max"]) / 2
                                          for r in fails])))
    report["n_episodes"] = dict(success=len(rows), failure=len(fails))
    return report


def load_references(data_run_dir, kinds=None):
    """
    Reference trajectories stored by data_collection.collect_references.

    Returns {"success": [traj, ...], "failure": [traj, ...]} in the shape
    reward_preview and the video tools already expect, so they need no change.
    The perturbed variants are folded in alongside the clean ones: a perturbed
    recovery IS a success trajectory, just one that starts somewhere an expert
    would never be, which is the regime the clean references cannot reach.

    Each traj carries "obs", "success_step", "truth", "seed", "kind", and
    "video" (a path, or None). Frames are NOT loaded — decode the mp4 only in
    the tools that draw pixels.
    """
    import json

    ref_dir = os.path.join(data_run_dir, "references")
    idx_path = os.path.join(ref_dir, "index.json")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(
            f"no references in {ref_dir}. Re-collect with the current "
            "data_collection (references=True), or call "
            "collect_eval_trajectories to roll them live.")
    with open(idx_path) as f:
        index = json.load(f)

    out = {"success": [], "failure": []}
    for row in index["trajectories"]:
        if kinds and row["kind"] not in kinds:
            continue
        d = np.load(os.path.join(ref_dir, row["name"] + ".npz"))
        ss = int(d["success_step"][0])
        ss = None if ss < 0 else ss
        mp4 = os.path.join(ref_dir, row["name"] + ".mp4")
        traj = dict(obs=d["obs"], success_step=ss, seed=row["seed"],
                    kind=row["kind"], frames=None,
                    video=mp4 if row.get("video") and os.path.exists(mp4) else None,
                    truth=(d["y_steps"] if ss is not None else None))
        out["success" if ss is not None else "failure"].append(traj)
    return out


def load_frames(traj):
    """Decode a reference trajectory's stored mp4 into frames, on demand."""
    import imageio.v3 as iio

    if not traj.get("video"):
        return None
    return list(iio.imiter(traj["video"]))


def collect_eval_trajectories(success_policy, failure_policy, seeds=(500, 501, 502),
                              max_steps=500, record_frames=False):
    """
    Reference rollouts from the held-out policies, for reward_preview and
    t2s_video. Returns {"success": [traj, ...], "failure": [traj, ...]}.

    PREFER load_references() when the data run has them: stored trajectories
    mean every tool scores identical states, and these eval seeds fall inside
    collection's own seed block anyway. This remains for runs without them.

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

# A model must stay at least this pessimistic on a trajectory that never
# succeeds before its MAE is worth ranking. The best-MAE model on the observed
# run scored fail_pred_min 4.11 — it claims four steps to success on a rollout
# that never succeeds — so ranking by MAE alone selects the reward most likely
# to pay for hovering. Stage 4 applies the same threshold; it lives here so the
# two stages cannot recommend different models.
MIN_FAIL_PRED = 30.0

FAIL_METRICS = ("fail_pred_min", "fail_pred_spread", "fail_optimism", "fail_pred_mean")
# Held-out EPISODES from dataset.npz (split == "val"): thousands of collected
# states the model never trained on. evaluate_on_val_split has always computed
# these and nothing ever displayed them, so the report looked as though it
# scored only the eight live rollouts. Thousands of rows is the larger sample;
# the rollouts are the more realistic one. Read both.
VAL_METRICS = ("val_mae", "val_bias", "val_calibration_slope", "val_n")
# fail_pred_min: LOWER is worse (a confident false claim of imminent success).
# fail_optimism: HIGHER is worse (further below the censor level).


def rank_models(reports, n=2, min_fail_pred=MIN_FAIL_PRED):
    """
    The models worth training against: reject false optimism FIRST, then rank
    the survivors by MAE. Returns (chosen, rejected, reason).

    MAE alone is the wrong order. It is measured only on success rollouts,
    where models trained on success data specialise, and it correlated +0.68
    with fail_pred_min across the observed grid — better success accuracy went
    with worse failure behaviour.
    """
    def m(c, k):
        v = metric_mean_std(reports[c], k)[0]
        return None if v is None else float(v)

    ok = [c for c in reports
          if (m(c, "fail_pred_min") or 0.0) >= min_fail_pred]
    rejected = [c for c in reports if c not in ok]
    reason = (f"{len(ok)} of {len(reports)} pass fail_pred_min >= {min_fail_pred}"
              if ok else
              f"NO model passes fail_pred_min >= {min_fail_pred}; ranking all by MAE")
    pool = ok or list(reports)
    chosen = sorted(pool, key=lambda c: m(c, "mae")
                    if m(c, "mae") is not None else float("inf"))[:n]
    return chosen, rejected, reason


def metric_mean_std(report, metric):
    """(mean, std) across per-seed rollouts. Pooled metrics have no spread."""
    if metric == "decrement_mean":
        # best is 1.0, like calibration_slope
        pass
    if metric in VAL_METRICS:
        v = report.get("val_split") or {}
        return (v.get(metric[len("val_"):] if metric != "val_n" else "n"), None)
    if metric in FAIL_METRICS:
        rows = report.get("failure_scenario", [])
        key = {"fail_pred_min": "pred_min", "fail_pred_spread": "pred_spread",
               "fail_optimism": "optimism_vs_censor"}.get(metric)
        vals = [r[key] for r in rows if key and r.get(key) is not None]
        if vals:
            return (float(np.mean(vals)), float(np.std(vals)))
        return (report.get("summary", {}).get(metric), None)
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
    unit_best = sort_by in ("calibration_slope", "decrement_mean")
    higher = sort_by in HIGHER_IS_BETTER or unit_best

    def key(c):
        m = metric_mean_std(reports[c], sort_by)[0]
        if m is None:
            return float("-inf") if higher else float("inf")
        return -abs(m - 1.0) if unit_best else m    # best is 1.0
    return sorted(names, key=key, reverse=higher)


LEGEND = """
mae                  pre-success frames of SUCCESS rollouts: does it predict
                     time correctly. NOT a selector on its own — it is measured
                     only where success-trained models specialise, and it
                     correlated +0.68 with fail_pred_min across the grid.
bias                 signed mean error. NEGATIVE = claims success is nearer
                     than it is. Note it barely reaches the reward: a constant
                     offset CANCELS in difference shaping, leaving only
                     c*(1-gamma) per step. Read it with mae — |bias|/mae near 1
                     means a harmless offset, near 0 means a shape error.
decrement_mean       mean predicted drop per step. 1.0 correct, <1 compressed,
                     0 flat. The reward IS decrement - c, so this is the
                     reward's mean, shifted.
decrement_std        noise in that drop, i.e. the per-step noise the policy
                     receives. Differencing amplifies prediction jitter.
                     These two replace step_slope_error = mean|d-1|, which
                     conflated them: a calibrated model with 0.9 of jitter and
                     a FLAT prediction both score exactly 1.00.
fail_pred_spread     how far the prediction MOVES on a failing rollout. Read
                     WITH fail_pred_min: a model pinned at the clip ceiling
                     scores a perfect fail_pred_min and gives no gradient at
                     all. mcboot_all sat at 300 for 500 steps.
fail_pred_min        lowest prediction on a trajectory that NEVER succeeds.
                     LOW IS BAD: the model's most confident false claim, and
                     the single best predictor of a reward that pays for
                     hovering. Reject below 30 before ranking by MAE.

val_mae / val_bias   the SAME questions asked on held-out EPISODES from
                     dataset.npz rather than on live rollouts: thousands of
                     collected states the model never trained on, against
                     eight rollouts. Add with metrics=REPORT_METRICS +
                     VAL_METRICS.

also available via metrics=REPORT_METRICS + EXTRA_METRICS:
bias, calibration_slope, worst_window_slope_error, fail_optimism,
max_wrong_direction, and the correlations."""


def format_report_table(reports, metrics=REPORT_METRICS, sort_by="mae", show_std=True):
    """Pure-string comparison table. No matplotlib, so it works headless."""
    if not reports:
        return "(no reports)"
    names = _ordered(reports, sort_by)
    cols = [m for m in metrics
            if any(metric_mean_std(reports[c], m)[0] is not None for c in names)]
    if not cols:
        return "(no metrics available)"

    # column width follows the LONGEST name present, so nothing is truncated:
    # "worst_window_slope_error" was cut to "worst_window_slope" at w=18, and
    # "dead_window_fraction" to "dead_window_fracti".
    w = max(18 if show_std else 14, max(len(m) for m in cols) + 2)
    header = f"{'combo':<20}" + "".join(f"{m:>{w}}" for m in cols)
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


def plot_combo_comparison(reports, metrics=REPORT_METRICS, save_path=None,
                          sort_by="mae", ncols=None):
    """
    Grid of metric panels, two rows by default, with the value printed on every
    bar. Seven metrics in one row made each panel ~180px wide, which is not
    enough to read a label or tell two similar bars apart.

    Every seed is also drawn as a dot, because error bars alone are invisible
    when the spread is small relative to the bar height — a bar with +/-0.02
    must not look like one with +/-2.0.

    The count under each bar is how many seeds actually contributed. It is
    printed because _corr returns None for a CONSTANT prediction and _mean
    drops those, so a model that flatlines on 5 of 8 seeds silently reports a
    correlation of 1.00 computed from the other 3. n<8 on a correlation panel
    means the model was degenerate on the missing seeds, which is worse than
    a low score, not better.
    """
    import matplotlib.pyplot as plt

    names = _ordered(reports, sort_by)
    cols = [m for m in metrics
            if any(metric_mean_std(reports[c], m)[0] is not None for c in names)]
    if not cols:
        raise ValueError("none of the requested metrics are present")

    nrows = 1 if len(cols) <= 2 else 2
    ncols = ncols or int(np.ceil(len(cols) / nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, metric in zip(axes, cols):
        stats = [metric_mean_std(reports[c], metric) for c in names]
        values = [np.nan if m is None else m for m, _ in stats]
        errs = [sd or 0.0 for _, sd in stats]
        bars = ax.bar(range(len(names)), values, yerr=errs, capsize=3, color="tab:blue",
                      error_kw=dict(ecolor="0.3", lw=1))

        scen = "failure_scenario" if metric in FAIL_METRICS else "success_scenario"
        key = {"fail_pred_min": "pred_min", "fail_pred_spread": "pred_spread",
               "fail_optimism": "optimism_vs_censor"}.get(metric, metric)
        per_seed = {c: [r[key] for r in reports[c].get(scen, [])
                        if r.get(key) is not None] for c in names}
        for xi, c in enumerate(names):
            if per_seed[c]:
                ax.scatter([xi] * len(per_seed[c]), per_seed[c], s=12, color="black",
                           zorder=3, alpha=.7)

        finite = [v for v in values if not np.isnan(v)]
        if finite:
            best = (min(finite, key=lambda v: abs(v - 1.0))
                    if metric in ("calibration_slope", "decrement_mean")
                    # bias is SIGNED and best at 0. Taking the minimum marked the
                    # most negative value green — tdlambda_all at -17.60 was
                    # highlighted as best when it is the worst in the panel.
                    else min(finite, key=abs) if metric == "bias"
                    # fail_pred_min: a HIGH minimum is good — the model stayed
                    # pessimistic on a trajectory that never succeeded
                    else max(finite) if metric in HIGHER_IS_BETTER + ("fail_pred_min",)
                    else min(finite))
            # a metric where every combo ties has no winner; colouring them all
            # green implies a distinction the numbers do not support
            if sum(v == best for v in values) < len(finite):
                for bar, v in zip(bars, values):
                    if v == best:
                        bar.set_color("tab:green")

        # ---- the value on every bar ----
        span = (max([v for v in values if not np.isnan(v)] + [0])
                - min([v for v in values if not np.isnan(v)] + [0])) or 1.0
        pad = 0.04 * span
        for xi, (v, e) in enumerate(zip(values, errs)):
            if np.isnan(v):
                continue
            top = max([v + e] + per_seed[names[xi]])
            bot = min([v - e] + per_seed[names[xi]])
            y, va = ((top + pad, "bottom") if v >= 0 else (bot - pad, "top"))
            ax.text(xi, y, f"{v:.2f}", ha="center", va=va, fontsize=7.5)
            n_seeds = len(per_seed[names[xi]])
            if metric in CORRELATIONS and n_seeds:
                ax.text(xi, bot - pad, f"n={n_seeds}", ha="center", va="top",
                        fontsize=6.5, color="0.45")

        if metric in ("calibration_slope", "decrement_mean"):
            ax.axhline(1.0, color="tab:red", ls="--", lw=1, label="correct = 1.0")
            if metric == "decrement_mean":
                ax.axhline(0.0, color="0.4", ls=":", lw=1, label="flat = 0.0")
            ax.legend(fontsize=7)
        if metric in ("step_slope_error", "worst_window_slope_error"):
            # a perfectly FLAT prediction gives |0 - 1| = 1.0 exactly, so this
            # line is the dead-signal threshold: at or above it the prediction
            # carries no usable per-step gradient
            ax.axhline(1.0, color="tab:red", ls="--", lw=1, label="flat = 1.0 (dead)")
            # green marks the best of the set, which is misleading when the
            # best is itself at the failure threshold — every model having a
            # fully dead window is not a distinction worth colouring
            if finite and min(finite) >= 1.0:
                for bar in bars:
                    bar.set_color("tab:red")
                ax.set_title(metric + "  — EVERY model has a dead window",
                             fontsize=9, color="tab:red")
            ax.legend(fontsize=7)
        ax.set_title(metric, fontsize=10)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7.5)
        ax.margins(y=0.18)
        ax.grid(axis="y", alpha=.3)

    for ax in axes[len(cols):]:
        ax.axis("off")

    fig.suptitle("T2S model comparison on held-out SUCCESS rollouts "
                 "(failure rollouts contribute no correlation)", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def evaluate_runs(t2s_run_dir, summary_rows, success_policy, failure_policy,
                  n_seeds=8, dataset_path=None, verbose=True, score_on="rollouts"):
    """
    Evaluate every combo in one t2s_model run. Returns {combo: report}.

    score_on="val" scores the held-out val EPISODES instead of live rollouts:
    dozens of complete trajectories against eight, no simulator, seconds
    instead of minutes. See evaluate_on_val_episodes for why that is now the
    better-justified default sample.

    Either way, dataset_path attaches the per-row val evaluation as
    report["val_split"], and score_on="val" also attaches the live report as
    report["rollouts"] only if policies are supplied.
    """
    import t2s_predict

    reports = {}
    for row in summary_rows:
        method, condition = row["combo"].rsplit("_", 1)
        if verbose:
            print(f"  evaluating {row['combo']} ...", flush=True)
        fn = t2s_predict.load_t2s_predictor(t2s_run_dir, method, condition,
                                            seed=row["best_seed"])
        if score_on == "val":
            if not dataset_path:
                raise ValueError("score_on='val' needs dataset_path")
            rep = evaluate_on_val_episodes(fn, dataset_path)
        else:
            rep = run_full_evaluation(fn, success_policy, failure_policy,
                                      n_seeds=n_seeds)
        if dataset_path:
            rep["val_split"] = evaluate_on_val_split(fn, dataset_path)
        reports[row["combo"]] = rep
    return reports