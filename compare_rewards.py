#!/usr/bin/env python3
"""
Compare the T2S reward with MetaWorld's reward over time, on many trajectories.

    python compare_rewards.py --t2s-run v8 --models tdlambdaboot_all
    python compare_rewards.py --t2s-run v8 --models tdlambdaboot_all --episodes 15

Trajectories are live rollouts of the two Stage 3 evaluation policies, grouped by
whether the episode succeeded. Each measure is computed on every trajectory, then
averaged over the successful and over the failed trajectories (with a 95% range).

Notation, over the steps t = 1..n up to success (or the whole failed episode):
    p_t   T2S prediction at the state after step t
    P_t   T2S progress = p_0 - p_t            (rises as the task progresses, starts at 0)
    m_t   MetaWorld reward at the state after step t
    r_t   T2S reward the policy is paid at step t, for a reward mode (training formula)
    dm_t  change in MetaWorld reward, m_t - m_{t-1}

Three kinds of measure:
    Spearman            rank correlation: do the two put the steps in the same order?
    direction           sum(a*b) / sum(|a|*|b|): do they move the same way, weighted by
                        how much both move (from -1 to +1; flat stretches count little)
    scale-fitted gap    each series divided by its own largest absolute value, then
                        mean |a/max|a| - b/max|b||: how different the patterns are once
                        both fit the same scale (0 = same pattern, up to 2)

Applied to:
    levels               P_t vs m_t
    <mode> per step      r_t vs dm_t, for difference and difference_timed
    changes over k steps (P_t+k - P_t) vs (m_t+k - m_t)
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

import config
import results
from reward_fn import step_reward

# ---- settings ---------------------------------------------------------------
MODES = ("difference", "difference_timed")
WINDOW_SIZES = (5, 10)              # extra windows for changes in progress
BOOTSTRAP_SAMPLES = 2000
STAGE_MANIFEST = "stage_manifest.json"


# ---- 1. trajectories --------------------------------------------------------

def rollout(policy, seed):
    """One episode with sampled actions -> observations, MetaWorld rewards, success step."""
    import torch
    from env_utils import make_fixed_scene_env

    env = make_fixed_scene_env()
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    observations, rewards, success_step = [np.asarray(obs, np.float32)], [], None
    for t in range(config.MAX_STEPS):
        action, _ = policy.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(np.asarray(obs, np.float32))
        rewards.append(float(reward))                     # rewards[i] belongs to observations[i + 1]
        if info.get(config.SUCCESS_KEY, 0) and success_step is None:
            success_step = t + 1
        if terminated or truncated or (success_step is not None
                                       and t >= success_step + config.REFERENCE_TAIL):
            break
    env.close()
    return np.array(observations), np.array(rewards), success_step


def series(predictions, metaworld_rewards, success_step):
    """Progress P_1..P_n, MetaWorld m_1..m_n, and the per-mode T2S rewards r_1..r_n."""
    n = len(metaworld_rewards) if success_step is None else min(success_step, len(metaworld_rewards))
    p = np.asarray(predictions, float)
    progress = p[0] - p[1:n + 1]
    metaworld = np.asarray(metaworld_rewards[:n], float)
    nxt = p[1:n + 1].copy()
    if success_step is not None and success_step <= n:
        nxt[success_step - 1] = 0.0                       # the wrapper's value at success
    paid = {m: np.array([step_reward(p[i], nxt[i], reward_mode=m, time_penalty=config.TIME_PENALTY,
                                     gamma=config.RL_GAMMA) for i in range(n)]) for m in MODES}
    return progress, metaworld, paid


# ---- 2. measures ------------------------------------------------------------

def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(spearmanr(a, b)[0])


def direction(a, b):
    """sum(a*b) / sum(|a|*|b|), from -1 (always opposite) to +1 (always the same way)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    both = np.sum(np.abs(a) * np.abs(b))
    return float(np.sum(a * b) / both) if len(a) >= 3 and both > 0 else None


def scale_fitted_gap(a, b):
    """Each series divided by its own largest absolute value, then the mean absolute gap."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ma, mb = np.abs(a).max() if len(a) else 0, np.abs(b).max() if len(b) else 0
    if len(a) < 3 or ma == 0 or mb == 0:
        return None
    return float(np.mean(np.abs(a / ma - b / mb)))


def trajectory_measures(progress, metaworld, paid):
    m = {"levels: Spearman": spearman(progress, metaworld),
         "levels: scale-fitted gap": scale_fitted_gap(progress, metaworld)}
    dm = np.diff(metaworld)
    for mode in MODES:
        r = paid[mode][1:]                                # aligned with dm (steps 2..n)
        m[f"{mode} per step: Spearman"] = spearman(r, dm)
        m[f"{mode} per step: direction"] = direction(r, dm)
        m[f"{mode} per step: scale-fitted gap"] = scale_fitted_gap(r, dm)
    for k in WINDOW_SIZES:
        if len(progress) > k + 2:
            dp, dmk = progress[k:] - progress[:-k], metaworld[k:] - metaworld[:-k]
            m[f"changes over {k} steps: Spearman"] = spearman(dp, dmk)
            m[f"changes over {k} steps: direction"] = direction(dp, dmk)
        else:
            m[f"changes over {k} steps: Spearman"] = m[f"changes over {k} steps: direction"] = None
    return m


def average_with_range(values):
    v = np.array([x for x in values if x is not None], float)
    if len(v) == 0:
        return None, None, None
    if len(v) == 1:
        return float(v[0]), None, None
    means = np.random.default_rng(0).choice(v, (BOOTSTRAP_SAMPLES, len(v))).mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def as_text(result):
    mean, low, high = result
    if mean is None:
        return "not available"
    return f"{mean:+.2f}" + (f"  ({low:+.2f} to {high:+.2f})" if low is not None else "")


# ---- 3. plots ---------------------------------------------------------------

def plot_example(title, predictions, metaworld_rewards, success_step, path):
    """Raw curves; levels on a common scale; per-step signals on a common scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    progress, metaworld, paid = series(predictions, metaworld_rewards, success_step)
    t = np.arange(1, len(metaworld) + 1)
    unit = lambda v: v / np.abs(v).max() if np.abs(v).max() > 0 else v

    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(t, np.asarray(predictions, float)[1:len(t) + 1], color="tab:orange", lw=2,
               label="T2S prediction")
    ax[0].set_ylabel("T2S prediction (steps)", color="tab:orange")
    twin = ax[0].twinx()
    twin.plot(t, metaworld, color="black", lw=1.5, label="MetaWorld reward")
    twin.set_ylabel("MetaWorld reward")
    ax[0].set_title(title, fontsize=10)
    ax[0].legend(handles=ax[0].get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0],
                 fontsize=8, loc="center right")

    ax[1].plot(t, unit(progress), color="tab:orange", lw=2, label="T2S progress / its largest |value|")
    ax[1].plot(t, unit(metaworld), color="black", lw=1.5, label="MetaWorld reward / its largest |value|")
    ax[1].set_ylabel("levels, common scale")
    rho = spearman(progress, metaworld)
    ax[1].text(0.01, 0.95, "Spearman " + ("--" if rho is None else format(rho, "+.2f")),
               transform=ax[1].transAxes, va="top", fontsize=8)

    r, dm = paid["difference"][1:], np.diff(metaworld)
    ax[2].plot(t[1:], unit(r), color="tab:orange", lw=1.2, label="T2S reward (difference) / largest |value|")
    ax[2].plot(t[1:], unit(dm), color="black", lw=1.2, label="change in MetaWorld / largest |value|")
    ax[2].axhline(0, color="0.6", lw=0.8)
    ax[2].set_ylabel("per step, common scale"); ax[2].set_xlabel("step")
    d = direction(r, dm)
    ax[2].text(0.01, 0.95, "direction " + ("--" if d is None else format(d, "+.2f")),
               transform=ax[2].transAxes, va="top", fontsize=8)
    for a in ax:
        if success_step is not None:
            a.axvline(success_step, color="tab:green", ls=":")
        a.grid(alpha=0.3)
    for a in ax[1:]:
        a.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_summary(summary, names, path):
    """Spearman and direction measures (-1..1) on the left, scale-fitted gaps (0..2) on the right."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = ([n for n in names if "gap" not in n], [n for n in names if "gap" in n])
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.42 * len(groups[0]) * len(summary) + 2),
                             gridspec_kw=dict(width_ratios=(1.4, 1)))
    for ax, group, xlim in zip(axes, groups, ((-1, 1), (0, 2))):
        y, labels = 0, []
        for model, per_outcome in summary.items():
            for name in group:
                for offset, (outcome, color) in enumerate((("successful", "tab:green"),
                                                           ("failed", "tab:red"))):
                    mean, low, high = per_outcome[outcome][name]
                    if mean is not None:
                        err = [[mean - (low if low is not None else mean)],
                               [(high if high is not None else mean) - mean]]
                        ax.errorbar(mean, y + 0.2 * offset, xerr=err, fmt="o", color=color,
                                    capsize=3, label=outcome if y == 0 else None)
                labels.append(f"{model}: {name}" if len(summary) > 1 else name)
                y += 1
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis(); ax.set_xlim(*xlim); ax.grid(alpha=0.3, axis="x")
        ax.legend(fontsize=8)
    axes[0].axvline(0, color="0.6", lw=0.8)
    axes[0].set_xlabel("Spearman / direction (mean and 95% range); higher = more agreement")
    axes[1].set_xlabel("scale-fitted gap; lower = more similar pattern")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---- 4. main ----------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--t2s-run", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--eval-run", default=None, help="t2s_eval run holding the policies (default: --t2s-run)")
    p.add_argument("--episodes", type=int, default=10, help="episodes per evaluation policy")
    p.add_argument("--first-seed", type=int, default=500)
    args = p.parse_args(argv)

    from stable_baselines3 import SAC
    from t2s_model import load_t2s_predictor

    t2s_dir = results.get_run_dir("t2s_model", args.t2s_run)
    combos = json.load(open(os.path.join(t2s_dir, "manifest.json")))["combos"]
    predictors = {m: load_t2s_predictor(t2s_dir, *m.rsplit("_", 1),
                                        seed=combos[m].get("best_seed", 0)) for m in args.models}
    eval_dir = results.get_run_dir("t2s_eval", args.eval_run or args.t2s_run)
    stage3 = json.load(open(os.path.join(eval_dir, STAGE_MANIFEST)))
    policies = {k: SAC.load(os.path.join(config.EXPERT_POLICY_DIR, stage3[f"{k}_ckpt"]))
                for k in ("success", "failure")}

    trajectories = []
    for name, policy in policies.items():
        for seed in range(args.first_seed, args.first_seed + args.episodes):
            obs, rewards, ss = rollout(policy, seed)
            trajectories.append(dict(policy=name, seed=seed, obs=obs, rewards=rewards, success_step=ss))
    n_success = sum(t["success_step"] is not None for t in trajectories)
    print(f"{len(trajectories)} episodes from {stage3['success_ckpt']} and {stage3['failure_ckpt']}: "
          f"{n_success} successful, {len(trajectories) - n_success} failed\n")

    out_dir = os.path.join(eval_dir, "reward_compare")
    os.makedirs(out_dir, exist_ok=True)
    summary, all_rows, names = {}, [], None
    for model, predict in predictors.items():
        rows = []
        for t in trajectories:
            predictions = np.array([predict(o) for o in t["obs"]], float)
            outcome = "successful" if t["success_step"] is not None else "failed"
            measures = trajectory_measures(*series(predictions, t["rewards"], t["success_step"]))
            rows.append(dict(model=model, policy=t["policy"], seed=t["seed"], outcome=outcome, **measures))
            if not any(r["outcome"] == outcome for r in rows[:-1]):
                plot_example(f"{model}, {outcome} episode (seed {t['seed']})", predictions,
                             t["rewards"], t["success_step"],
                             os.path.join(out_dir, f"{model}_{outcome}_example.png"))
        all_rows += rows
        names = [k for k in rows[0] if k not in ("model", "policy", "seed", "outcome")]
        summary[model] = {o: {n: average_with_range([r[n] for r in rows if r["outcome"] == o])
                              for n in names} for o in ("successful", "failed")}
        counts = {o: sum(r["outcome"] == o for r in rows) for o in ("successful", "failed")}

        print(f"Model {model}")
        print(f"  {'':<44}{'successful (' + str(counts['successful']) + ')':<30}"
              f"{'failed (' + str(counts['failed']) + ')':<30}")
        for n in names:
            print(f"  {n:<44}{as_text(summary[model]['successful'][n]):<30}"
                  f"{as_text(summary[model]['failed'][n]):<30}")
        print()

    plot_summary(summary, names, os.path.join(out_dir, "summary.png"))
    json.dump(dict(t2s_run=args.t2s_run, summary=summary, trajectories=all_rows),
              open(os.path.join(out_dir, "compare.json"), "w"), indent=2)
    print(f"wrote {out_dir}: summary.png, compare.json, and one example plot per model and outcome")
    return 0


if __name__ == "__main__":
    sys.exit(main())