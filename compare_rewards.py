#!/usr/bin/env python3
"""
Compare the T2S prediction with MetaWorld's reward over time, on many trajectories.

    python compare_rewards.py --t2s-run v6 --models tdlambdaboot_all
    python compare_rewards.py --t2s-run v6 --models tdlambdaboot_all td0boot_all --episodes 15

Trajectories are live rollouts of the two Stage 3 evaluation policies, grouped by
whether the episode succeeded. Every measure is computed on each trajectory
separately, then averaged over the successful and over the failed trajectories.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

import config
import results
from stable_baselines3 import SAC
from t2s_model import load_t2s_predictor
import torch
from env_utils import make_fixed_scene_env

import matplotlib
  
import matplotlib.pyplot as plt


# ---- settings you may want to change ----------------------------------------
WINDOW_SIZES = (1, 5, 10, 20)   # compare changes over this many steps
MAX_SHIFT = 15                  # largest time shift (steps) tried when aligning the two series
BOOTSTRAP_SAMPLES = 2000        # used for the 95% range of each average
STAGE_MANIFEST = "stage_manifest.json"


# ---- 1. collecting trajectories ----------------------------------------------
def rollout(policy, seed, max_steps, steps_after_success, success_key):
    """One episode with sampled actions. Returns observations, MetaWorld rewards, success step."""

    env = make_fixed_scene_env()
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    observations, rewards, success_step = [np.asarray(obs, np.float32)], [], None
    for t in range(max_steps):
        action, _ = policy.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(np.asarray(obs, np.float32))
        rewards.append(float(reward))              # rewards[i] belongs to observations[i + 1]
        if info.get(success_key, 0) and success_step is None:
            success_step = t + 1
        if terminated or truncated or (success_step is not None
                                       and t >= success_step + steps_after_success):
            break
    env.close()
    return np.array(observations), np.array(rewards), success_step


def aligned_series(predictions, metaworld_rewards, success_step):
    """
    The two series compared, on the same time steps s_1 .. s_n, where n is the success
    step (or the last step of a failed episode). T2S is negated so both rise with progress.
    """
    n = len(metaworld_rewards) if success_step is None else min(success_step, len(metaworld_rewards))
    t2s_progress = -np.asarray(predictions[1:n + 1], float)
    metaworld = np.asarray(metaworld_rewards[:n], float)
    return t2s_progress, metaworld


# ---- 2. measures on one trajectory -------------------------------------------

def correlation(a, b):
    """Spearman correlation, or None when either series is constant or too short."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(spearmanr(a, b)[0])


def remove_time_trend(x):
    """x minus its best straight-line fit against time."""
    t = np.arange(len(x))
    slope, intercept = np.polyfit(t, x, 1)
    return x - (slope * t + intercept)


def best_time_shift(t2s_progress, metaworld):
    """Shift (steps) with the highest correlation. Positive: T2S changes before MetaWorld."""
    best_value, best_shift = None, None
    for shift in range(-MAX_SHIFT, MAX_SHIFT + 1):
        if shift >= 0:
            a, b = t2s_progress[:len(t2s_progress) - shift], metaworld[shift:]
        else:
            a, b = t2s_progress[-shift:], metaworld[:len(metaworld) + shift]
        value = correlation(a, b)
        if value is not None and (best_value is None or value > best_value):
            best_value, best_shift = value, shift
    return best_shift


def trajectory_measures(t2s_progress, metaworld):
    """Every per-trajectory measure, by readable name. Add or remove entries here."""
    n = len(t2s_progress)
    measures = {
        "Correlation of levels": correlation(t2s_progress, metaworld),
        "Correlation of levels, time trend removed": (
            correlation(remove_time_trend(t2s_progress), remove_time_trend(metaworld))
            if n >= 3 else None),
    }
    for k in WINDOW_SIZES:
        measures[f"Correlation of changes over {k} step{'s' if k > 1 else ''}"] = (
            correlation(t2s_progress[k:] - t2s_progress[:-k], metaworld[k:] - metaworld[:-k])
            if n > k + 2 else None)
    measures["Time shift with the best agreement (steps)"] = (
        best_time_shift(remove_time_trend(t2s_progress), remove_time_trend(metaworld))
        if n >= 3 else None)
    return measures


# ---- 3. averaging over trajectories ------------------------------------------

def average_with_range(values):
    """Mean and 95% bootstrap range over trajectories (each trajectory counts once)."""
    v = np.array([x for x in values if x is not None], float)
    if len(v) == 0:
        return None, None, None
    if len(v) == 1:
        return float(v[0]), None, None
    means = np.random.default_rng(0).choice(v, (BOOTSTRAP_SAMPLES, len(v))).mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def as_text(result, decimals=2):
    mean, low, high = result
    if mean is None:
        return "not available"
    text = f"{mean:.{decimals}f}"
    if low is not None:
        text += f"  ({low:.{decimals}f} to {high:.{decimals}f})"
    return text


# ---- 4. plots ----------------------------------------------------------------

def plot_example(title, predictions, metaworld_rewards, success_step, path):
    """One trajectory: T2S prediction and MetaWorld reward over time."""
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.arange(len(predictions)), predictions, color="tab:orange", lw=2,
            label="T2S prediction (steps to success)")
    ax.set_xlabel("step"); ax.set_ylabel("T2S prediction", color="tab:orange")
    ax2 = ax.twinx()
    ax2.plot(np.arange(1, len(metaworld_rewards) + 1), metaworld_rewards, color="black", lw=1.5,
             label="MetaWorld reward")
    ax2.set_ylabel("MetaWorld reward")
    if success_step is not None:
        ax.axvline(success_step, color="tab:green", ls=":", label=f"success at step {success_step}")
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax.legend(handles=handles, fontsize=8, loc="center right")
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_summary(summary, names, path):
    """Each correlation measure: average over successful vs failed trajectories, per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 0.5 * len(names) * len(summary) + 2))
    y, labels = 0, []
    for model, per_outcome in summary.items():
        for name in names:
            for offset, (outcome, color) in enumerate((("successful", "tab:green"),
                                                       ("failed", "tab:red"))):
                mean, low, high = per_outcome[outcome][name]
                if mean is not None:
                    err = [[mean - (low if low is not None else mean)],
                           [(high if high is not None else mean) - mean]]
                    ax.errorbar(mean, y + 0.2 * offset, xerr=err, fmt="o", color=color,
                                capsize=3, label=outcome if y == 0 else None)
            labels.append(f"{model}: {name}")
            y += 1
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xlim(-1, 1); ax.set_xlabel("Spearman correlation (mean and 95% range)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---- 5. main -----------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--t2s-run", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--eval-run", default=None, help="t2s_eval run holding the policies (default: --t2s-run)")
    p.add_argument("--episodes", type=int, default=10, help="episodes per evaluation policy")
    p.add_argument("--first-seed", type=int, default=500)
    args = p.parse_args(argv)



    t2s_dir = results.get_run_dir("t2s_model", args.t2s_run)
    combos = json.load(open(os.path.join(t2s_dir, "manifest.json")))["combos"]
    predictors = {m: load_t2s_predictor(t2s_dir, *m.rsplit("_", 1),
                                        seed=combos[m].get("best_seed", 0)) for m in args.models}

    eval_dir = results.get_run_dir("t2s_eval", args.eval_run or args.t2s_run)
    stage3 = json.load(open(os.path.join(eval_dir, STAGE_MANIFEST)))
    policies = {k: SAC.load(os.path.join(config.EXPERT_POLICY_DIR, stage3[f"{k}_ckpt"]))
                for k in ("success", "failure")}

    trajectories = []
    for policy in policies.values():
        for seed in range(args.first_seed, args.first_seed + args.episodes):
            obs, rewards, success_step = rollout(policy, seed, config.MAX_STEPS,
                                                 config.REFERENCE_TAIL, config.SUCCESS_KEY)
            trajectories.append(dict(seed=seed, obs=obs, rewards=rewards,
                                     success_step=success_step))
    n_success = sum(t["success_step"] is not None for t in trajectories)
    print(f"{len(trajectories)} episodes: {n_success} successful, "
          f"{len(trajectories) - n_success} failed\n")

    out_dir = os.path.join(eval_dir, "reward_compare")
    os.makedirs(out_dir, exist_ok=True)
    summary, all_rows = {}, []
    
    for model, predict in predictors.items():
        rows = []
        for t in trajectories:
            predictions = np.array([predict(o) for o in t["obs"]], float)
            outcome = "successful" if t["success_step"] is not None else "failed"
            t2s_progress, metaworld = aligned_series(predictions, t["rewards"], t["success_step"])
            n = len(metaworld)
            row = dict(model=model, seed=t["seed"], outcome=outcome,
                       **trajectory_measures(t2s_progress, metaworld))
            row["Prediction at the start"] = float(predictions[0])
            row["Prediction at the end"] = float(predictions[n])
            row["MetaWorld reward at the end"] = float(metaworld[-1])
            rows.append(row)
            if not any(r["outcome"] == outcome for r in rows[:-1]):
                plot_example(f"{model}, {outcome} episode (seed {t['seed']})", predictions,
                             t["rewards"], t["success_step"],
                             os.path.join(out_dir, f"{model}_{outcome}_example.png"))
        all_rows += rows

        names = [k for k in rows[0] if k not in ("model", "seed", "outcome")]
        summary[model] = {o: {name: average_with_range([r[name] for r in rows if r["outcome"] == o])
                              for name in names} for o in ("successful", "failed")}
        counts = {o: sum(r["outcome"] == o for r in rows) for o in ("successful", "failed")}

        print(f"Model {model}")
        print(f"  {'':<48}{'successful (' + str(counts['successful']) + ')':<30}"
              f"{'failed (' + str(counts['failed']) + ')':<30}")
        
        for name in names:
            decimals = 2 if name.startswith("Correlation") else 1
            print(f"  {name:<48}{as_text(summary[model]['successful'][name], decimals):<30}"
                  f"{as_text(summary[model]['failed'][name], decimals):<30}")
            
        for o in ("successful", "failed"):
            rs = [r for r in rows if r["outcome"] == o]
            across = correlation([r["Prediction at the start"] - r["Prediction at the end"] for r in rs],
                                 [r["MetaWorld reward at the end"] for r in rs])
            summary[model][o]["Correlation across episodes"] = across
            print(f"  Across {o} episodes, correlation between the T2S prediction's drop and "
                  f"MetaWorld's final reward: "
                  f"{'not available' if across is None else format(across, '.2f')}")
        print()

    correlation_names = [n for n in names if n.startswith("Correlation")]
    plot_summary(summary, correlation_names, os.path.join(out_dir, "summary.png"))
    json.dump(dict(t2s_run=args.t2s_run, summary=summary, trajectories=all_rows),
              open(os.path.join(out_dir, "compare.json"), "w"), indent=2)
    print(f"wrote {out_dir}: compare.json, summary.png, and one example plot per model and outcome")
    return 0


if __name__ == "__main__":
    sys.exit(main())