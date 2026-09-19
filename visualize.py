"""
Manual policy inspection: load a specific checkpoint, roll it out on the
fixed scene, and visualize what happened.

Deliberately decoupled from the sweep/training path — this is for looking at
ONE policy you name explicitly (an expert checkpoint, a mid-training policy
snapshot, a final downstream policy), not for batch evaluation. Useful for
sanity-checking a policy before committing it to a long sweep, and for
seeing whether the T2S countdown behaves sensibly along a real trajectory.
"""
import os

import numpy as np

from config import SUCCESS_KEY, MAX_STEPS, steps_remaining_curve


def load_policy(checkpoint_path, policy_dir=None):
    """
    Loads a SAC checkpoint by explicit path (or bare filename + policy_dir).
    Raises with the available options listed rather than a bare file error.
    """
    import glob
    from stable_baselines3 import SAC

    path = checkpoint_path
    if policy_dir and not (os.path.isabs(path) or os.path.exists(path)):
        path = os.path.join(policy_dir, checkpoint_path)
    if not os.path.exists(path):
        available = [os.path.basename(p) for p in glob.glob(os.path.join(policy_dir or ".", "*.zip"))]
        raise FileNotFoundError(f"{checkpoint_path!r} not found (resolved to {path}). Available: {available}")
    return SAC.load(path), path


def plot_rollouts(rollouts, labels=None, save_path=None, show_truth=True):
    """
    Plots T2S prediction curves for one or more rollouts (from `rollout`).

    For rollouts that succeeded, optionally overlays the ground-truth
    countdown (a straight line from success_step down to 0) as a dashed
    reference — the gap between predicted and true is the model's error,
    and any upward kink in the prediction is the countdown-reversal problem
    that `t2s_eval.max_wrong_direction` measures numerically.
    """
    import matplotlib.pyplot as plt

    rollouts = rollouts if isinstance(rollouts, (list, tuple)) else [rollouts]
    labels = labels or [f"rollout {i}" for i in range(len(rollouts))]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = plt.cm.tab10.colors

    for i, (r, label) in enumerate(zip(rollouts, labels)):
        c = colors[i % len(colors)]
        if r["t2s_preds"] is None:
            continue
        ax.plot(r["t2s_preds"], color=c, label=f"{label} (predicted)")
        if show_truth and r["success_step"] is not None:
            truth = steps_remaining_curve(len(r["t2s_preds"]), r["success_step"])
            ax.plot(truth, color=c, linestyle="--", alpha=0.5, label=f"{label} (true)")
            ax.axvline(r["success_step"], color=c, alpha=0.25, linewidth=1)

    ax.set_xlabel("step")
    ax.set_ylabel("steps remaining until success")
    ax.set_title("T2S predictions along rollout (dashed = ground truth, vline = success)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig