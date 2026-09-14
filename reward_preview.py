"""
Pre-training reward diagnostic.

Answers "what reward would this T2S model actually emit?" WITHOUT training
anything — it replays stored trajectories (from t2s_eval.collect_eval_trajectories)
through the same reward arithmetic policy_env.Time2SuccessRewardWrapper uses.
Minutes instead of hours, and it can rule out a reward before a sweep.

The four diagnostics, and why each one matters:

  1. return_gap — total return on the SUCCESS trajectory minus total return
     on the FAILURE trajectory. THE critical one. If a reward assigns
     similar returns to a trajectory that succeeds and one that never does,
     no RL algorithm can learn to prefer success, regardless of budget or
     hyperparameters. A large positive gap is the minimum bar for a usable
     reward.

  2. dead_step_fraction — fraction of steps whose |reward| is below eps.
     A dead step provides no gradient. The failure panel of the sample-
     trajectory plot showed every model flatlining to a constant after
     ~150 steps, which is exactly what produces dead steps: in 'absolute'
     mode a constant prediction is a constant penalty (no signal about
     which action is better), and in 'difference' mode it is ~0 reward.

  3. progress_alignment — Spearman correlation between per-step reward and
     per-step REDUCTION in peg-to-goal distance. Asks the question that
     matters most: when the policy makes real physical progress, does the
     reward actually go up? A reward can have a fine return_gap and still
     be misaligned step-to-step, which makes credit assignment very hard.

  4. reward_std / range — dynamic range. Near-zero std means the reward is
     effectively constant, i.e. uninformative even if its level looks
     reasonable.
"""
import numpy as np
from scipy.stats import spearmanr

from progress import peg_goal_distance_curve

DEAD_STEP_EPS = 1e-3


def compute_step_rewards(preds, reward_mode="absolute", time_penalty=None):
    """
    Per-step rewards for a stored prediction sequence (length T-1, since each
    reward corresponds to a transition).

    Delegates to policy_env.step_reward — the SAME function the live training
    wrapper calls — so a preview can never disagree with what training does.
    (An earlier version of this file reimplemented the arithmetic locally,
    which meant editing the wrapper silently had no effect on the preview.)
    """
    from reward_fn import step_reward, DEFAULT_TIME_PENALTY

    preds = np.asarray(preds, dtype=np.float64)
    if len(preds) < 2:
        return np.zeros(0)
    tp = DEFAULT_TIME_PENALTY if time_penalty is None else time_penalty
    return np.array([
        step_reward(preds[i], preds[i + 1], reward_mode=reward_mode, time_penalty=tp)
        for i in range(len(preds) - 1)
    ], dtype=np.float64)


def analyze_reward_signal(preds, obs_seq=None, reward_mode="absolute", eps=DEAD_STEP_EPS,
                           time_penalty=None, success_step=None):
    """
    Per-trajectory reward diagnostics. obs_seq (raw observations for the same
    trajectory) enables progress_alignment; omit it to skip that metric.
    """
    rewards = compute_step_rewards(preds, reward_mode, time_penalty=time_penalty)
    # Reproduce Time2SuccessRewardWrapper's success latch: once the episode has
    # succeeded the wrapper charges nothing (and normally terminates), so a
    # preview that keeps accruing -time_penalty past success misrepresents the
    # reward exactly at the most important frame. rewards[t] covers the
    # transition into state t+1, so transitions from success_step onward are
    # post-success.
    if success_step is not None and len(rewards):
        rewards = rewards.copy()
        rewards[success_step:] = 0.0
    if len(rewards) == 0:
        return dict(n_steps=0, total_return=0.0, mean_reward=None, reward_std=None,
                     reward_min=None, reward_max=None, dead_step_fraction=None,
                     progress_alignment=None)

    out = dict(
        n_steps=int(len(rewards)),
        total_return=float(rewards.sum()),
        mean_reward=float(rewards.mean()),
        reward_std=float(rewards.std()),
        reward_min=float(rewards.min()),
        reward_max=float(rewards.max()),
        dead_step_fraction=float(np.mean(np.abs(rewards) < eps)),
    )

    if obs_seq is not None and len(obs_seq) == len(preds):
        dist = peg_goal_distance_curve(obs_seq)
        progress = dist[:-1] - dist[1:]   # positive = peg moved closer to goal
        if np.std(rewards) > 0 and np.std(progress) > 0:
            out["progress_alignment"] = float(spearmanr(rewards, progress)[0])
        else:
            out["progress_alignment"] = None
    else:
        out["progress_alignment"] = None

    return out


def preview_reward(trajectories, predictors,
                    reward_modes=("absolute", "difference", "difference_timed"),
                    time_penalty=None):
    """
    Full pre-training preview across every (combo x reward_mode) pair.

    trajectories: t2s_eval.collect_eval_trajectories output — needs the
        "success" and "failure" entries, each with "obs".
    predictors: {combo_name: predict_fn}

    Returns {(combo, reward_mode): diagnostics}, where diagnostics includes
    per-scenario analysis plus the cross-scenario return_gap.
    """
    # predict once per combo on each stored trajectory, reused for both modes
    preds_cache = {}
    for combo, predict_fn in predictors.items():
        for scenario, traj in trajectories.items():
            preds_cache[(combo, scenario)] = np.array(
                [predict_fn(o) for o in traj["obs"]], dtype=np.float64)

    out = {}
    for combo in predictors:
        for mode in reward_modes:
            per_scenario = {}
            for scenario, traj in trajectories.items():
                per_scenario[scenario] = analyze_reward_signal(
                    preds_cache[(combo, scenario)], traj["obs"], reward_mode=mode,
                    time_penalty=time_penalty, success_step=traj.get("success_step"))

            entry = dict(per_scenario=per_scenario)
            s, f = per_scenario.get("success"), per_scenario.get("failure")
            if s and f:
                entry["return_gap"] = s["total_return"] - f["total_return"]
                # normalized by trajectory length so a long failure trajectory
                # doesn't dominate the gap purely by having more steps
                entry["mean_reward_gap"] = (
                    (s["mean_reward"] - f["mean_reward"])
                    if s["mean_reward"] is not None and f["mean_reward"] is not None else None)
            out[(combo, mode)] = entry
    return out


def format_preview_table(preview):
    """Pure-string table of preview_reward output, sorted by return_gap descending."""
    if not preview:
        return "(no preview results)"

    rows = []
    for (combo, mode), entry in preview.items():
        s = entry["per_scenario"].get("success", {})
        f = entry["per_scenario"].get("failure", {})
        rows.append(dict(
            combo=combo, mode=mode,
            return_gap=entry.get("return_gap"),
            succ_return=s.get("total_return"), fail_return=f.get("total_return"),
            succ_dead=s.get("dead_step_fraction"), fail_dead=f.get("dead_step_fraction"),
            align=s.get("progress_alignment"),
        ))
    rows.sort(key=lambda r: r["return_gap"] if r["return_gap"] is not None else -float("inf"),
               reverse=True)

    def fmt(v, prec=3):
        return f"{v:.{prec}f}" if v is not None else "--"

    header = (f"{'combo':<16}{'mode':<11}{'return gap':>12}{'succ ret':>11}{'fail ret':>11}"
              f"{'dead% s/f':>14}{'align':>8}")
    lines = [header, "-" * len(header)]
    for r in rows:
        dead = f"{fmt(r['succ_dead'], 2)}/{fmt(r['fail_dead'], 2)}"
        lines.append(
            f"{r['combo']:<16}{r['mode']:<11}{fmt(r['return_gap'], 1):>12}"
            f"{fmt(r['succ_return'], 1):>11}{fmt(r['fail_return'], 1):>11}"
            f"{dead:>14}{fmt(r['align'], 2):>8}"
        )
    return "\n".join(lines)


def plot_reward_preview(trajectories, predictors, reward_mode="absolute", save_path=None,
                         time_penalty=None):
    """
    Per-step reward curves for every combo, on both trajectories. Makes the
    dead-signal problem visually obvious: a flat line at any level means no
    gradient, regardless of what level it sits at.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    colors = plt.cm.tab10.colors

    for ax, (scenario, traj) in zip(axes, trajectories.items()):
        for i, (combo, predict_fn) in enumerate(predictors.items()):
            preds = np.array([predict_fn(o) for o in traj["obs"]], dtype=np.float64)
            rewards = compute_step_rewards(preds, reward_mode, time_penalty=time_penalty)
            ss = traj.get("success_step")
            if ss is not None and len(rewards):
                rewards = rewards.copy()
                rewards[ss:] = 0.0        # match the wrapper's success latch
            ax.plot(rewards, color=colors[i % len(colors)], label=combo, linewidth=1.4)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{scenario} trajectory — per-step reward ({reward_mode})")
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(f"reward ({reward_mode} mode)")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig