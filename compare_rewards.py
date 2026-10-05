#!/usr/bin/env python3
"""
Compare the T2S reward with MetaWorld's reward over time, on many trajectories.

    python compare_rewards.py --t2s-run v8 --models tdlambdaboot_all
    python compare_rewards.py --t2s-run v8 --models tdlambdaboot_all --episodes 15

Trajectories are live rollouts of the two Stage 3 evaluation policies, grouped by
whether the episode succeeded. Each measure is computed on every trajectory, then
averaged over the successful and over the failed trajectories (with a 95% range).

Over the steps t = 1..n up to success (or the whole failed episode):
    p_t   T2S prediction at the state after step t
    m_t   MetaWorld reward at the state after step t
and the two series compared, as changes over WINDOW steps (both rise with progress):
    a_t = p_t - p_{t+k}        how much the T2S prediction dropped (= the T2S reward
                               the policy collects over those k steps, without discount)
    b_t = m_{t+k} - m_t        how much MetaWorld's reward rose

Three measures, each computed per trajectory, then averaged:
    Spearman           rank correlation of a and b: do the two put the steps in the same
                       order (the sequence pattern of the two rewards)?
    direction          sum(a_t * b_t) / sum(|a_t| * |b_t|): agreement of the SIGNS of a and b,
                       each step weighted by how much both move (-1 always opposite,
                       +1 always the same way)
    scale-fitted gap   mean |a_t / max|a| - b_t / max|b||: each series divided by its own
                       largest absolute value, then the average gap (0 = same pattern, up to 2)
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

import config
import results

# ---- settings ---------------------------------------------------------------
WINDOW = 5                          # compare changes over this many steps (1 = every step)
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


def series(predictions, metaworld_rewards, success_step, k=WINDOW):
    """a: drop in the T2S prediction over k steps; b: rise in MetaWorld's reward over k steps.
    Steps 1..n up to success; the prediction at the success state counts as 0, as in training."""
    n = len(metaworld_rewards) if success_step is None else min(success_step, len(metaworld_rewards))
    p = np.asarray(predictions, float)[1:n + 1].copy()
    if success_step is not None and success_step <= n:
        p[success_step - 1] = 0.0
    m = np.asarray(metaworld_rewards[:n], float)
    if n <= k:
        return np.array([]), np.array([])
    return p[:-k] - p[k:], m[k:] - m[:-k]


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


MEASURES = ("Spearman", "direction", "scale-fitted gap")


def trajectory_measures(a, b):
    return {"Spearman": spearman(a, b), "direction": direction(a, b),
            "scale-fitted gap": scale_fitted_gap(a, b)}


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
    """Top: the raw curves. Bottom: the two compared series, each over its largest |value|."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, b = series(predictions, metaworld_rewards, success_step)
    n = len(metaworld_rewards) if success_step is None else min(success_step, len(metaworld_rewards))
    unit = lambda v: v / np.abs(v).max() if len(v) and np.abs(v).max() > 0 else v
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    t = np.arange(1, n + 1)
    ax[0].plot(t, np.asarray(predictions, float)[1:n + 1], color="tab:orange", lw=2, label="T2S prediction")
    ax[0].set_ylabel("T2S prediction (steps)", color="tab:orange")
    twin = ax[0].twinx()
    twin.plot(t, np.asarray(metaworld_rewards[:n], float), color="black", lw=1.5, label="MetaWorld reward")
    twin.set_ylabel("MetaWorld reward")
    ax[0].legend(handles=ax[0].get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0],
                 fontsize=8, loc="center right")
    ax[0].set_title(title, fontsize=10)
    tt = np.arange(1, len(a) + 1)
    ax[1].plot(tt, unit(a), color="tab:orange", lw=1.3, label=f"drop in T2S prediction over {WINDOW} steps")
    ax[1].plot(tt, unit(b), color="black", lw=1.3, label=f"rise in MetaWorld reward over {WINDOW} steps")
    ax[1].axhline(0, color="0.6", lw=0.8)
    ax[1].set_ylabel("each / its largest |value|"); ax[1].set_xlabel("step")
    vals = trajectory_measures(a, b)
    ax[1].text(0.01, 0.97, "   ".join(f"{k} {'--' if v is None else format(v, '+.2f')}"
                                      for k, v in vals.items()),
               transform=ax[1].transAxes, va="top", fontsize=8)
    ax[1].legend(fontsize=8, loc="lower right")
    for x in ax:
        if success_step is not None:
            x.axvline(success_step, color="tab:green", ls=":")
        x.grid(alpha=0.3)
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

        print(f"Model {model}   (changes over {WINDOW} steps)")
        print(f"  {'':<20}{'successful (' + str(counts['successful']) + ')':<30}"
              f"{'failed (' + str(counts['failed']) + ')':<30}")
        for n in names:
            print(f"  {n:<20}{as_text(summary[model]['successful'][n]):<30}"
                  f"{as_text(summary[model]['failed'][n]):<30}")
        print()

    plot_summary(summary, names, os.path.join(out_dir, "summary.png"))
    json.dump(dict(t2s_run=args.t2s_run, summary=summary, trajectories=all_rows),
              open(os.path.join(out_dir, "compare.json"), "w"), indent=2)
    print(f"wrote {out_dir}: summary.png, compare.json, and one example plot per model and outcome")
    return 0


if __name__ == "__main__":
    sys.exit(main())