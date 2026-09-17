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

from config import SUCCESS_KEY, MAX_STEPS


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


def rollout(policy, predict_t2s=None, seed=0, render=False, camera_name="corner2",
            max_steps=MAX_STEPS, deterministic=True, noise_std=0.0, stop_after_success=20):
    """
    Rolls out `policy` on the fixed scene, recording observations, success
    flags, optional rendered frames, and optional per-step T2S predictions.

    policy=None gives a random-action rollout (useful as a visual baseline).
    predict_t2s=None skips prediction recording entirely.

    Returns a dict with: obs (T, obs_dim), success_step (int|None),
    t2s_preds (T,) or None, frames (list of RGB arrays) or None.
    """
    from env_utils import make_fixed_scene_env

    env = make_fixed_scene_env(render_mode="rgb_array" if render else None,
                                camera_name=camera_name)
    obs, _ = env.reset(seed=seed)
    obs_list, preds, frames = [], [], []
    success_step = None

    for t in range(max_steps):
        obs_list.append(obs.copy())
        if render:
            frames.append(env.render())
        if predict_t2s is not None:
            preds.append(predict_t2s(obs))

        if policy is None:
            action = env.action_space.sample()
        else:
            action, _ = policy.predict(obs, deterministic=deterministic)
            if noise_std > 0:
                action = np.clip(action + np.random.normal(0, noise_std, action.shape),
                                  env.action_space.low, env.action_space.high)

        obs, _r, terminated, truncated, info = env.step(action)
        if info.get(SUCCESS_KEY, 0) and success_step is None:
            success_step = t
        if success_step is not None and t >= success_step + stop_after_success:
            break
        if terminated or truncated:
            break

    env.close()
    return dict(
        obs=np.array(obs_list, dtype=np.float32),
        success_step=success_step,
        t2s_preds=np.array(preds, dtype=np.float32) if preds else None,
        frames=frames if render else None,
    )


def save_video(frames, path, fps=20):
    """Writes rendered frames to an mp4 (or gif, by extension). Needs imageio."""
    import imageio
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
    return path


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
            truth = [max(0, r["success_step"] - t) for t in range(len(r["t2s_preds"]))]
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


def summarize_rollout(r):
    """One-line text summary of a rollout dict, for quick console inspection."""
    n = len(r["obs"])
    if r["success_step"] is None:
        status = f"FAILED (ran {n} steps without success)"
    else:
        status = f"succeeded at step {r['success_step']} (ran {n} steps)"
    if r["t2s_preds"] is not None and len(r["t2s_preds"]):
        p = r["t2s_preds"]
        biggest_rise = float(np.max(np.diff(p))) if len(p) > 1 else 0.0
        status += (f" | T2S start={p[0]:.1f} end={p[-1]:.1f} "
                   f"min={p.min():.1f} max_upward_jump={biggest_rise:.2f}")
    return status


def plot_models_on_rollout(rollout_result, predictors, save_path=None,
                            danger_threshold=10.0):
    """
    Score ONE rollout with EVERY model and plot them together, alongside the
    physical evidence of what actually happened.

    Built to make the false-optimism failure visible rather than inferred: a
    policy that hovers near the goal WITHOUT the object can drive a
    success-only model's prediction toward 0, because in every successful
    demonstration "hand near hole" and "peg in hole" always co-occur — so the
    model has no evidence about which one causes low time-to-success.

    All models score the SAME stored observations, so any difference between
    the curves is the model and not the trajectory.

    Panels:
      top    — predicted steps-remaining per model, with a shaded danger band
               below `danger_threshold`. On a rollout that never succeeds, any
               curve entering that band is falsely optimistic.
      bottom — hand-to-peg and peg-to-goal distance over the same frames, so
               you can see whether the object was ever touched while a model
               was predicting imminent success.

    Returns (fig, {combo: minimum_prediction}).
    """
    import matplotlib.pyplot as plt

    from progress import hand_peg_distance_curve, peg_goal_distance_curve

    obs = rollout_result["obs"]
    succeeded = rollout_result["success_step"] is not None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    colors = plt.cm.tab10.colors

    worst = {}
    for i, (combo, fn) in enumerate(predictors.items()):
        preds = np.array([fn(o) for o in obs], dtype=np.float64)
        ax1.plot(preds, color=colors[i % len(colors)], label=combo, linewidth=1.5)
        worst[combo] = float(preds.min())

    if not succeeded:
        ax1.axhspan(0, danger_threshold, color="red", alpha=0.10)
        ax1.axhline(danger_threshold, color="red", ls="--", lw=1,
                     label=f"falsely optimistic (<{danger_threshold:g})")
    else:
        truth = [max(0, rollout_result["success_step"] - t) for t in range(len(obs))]
        ax1.plot(truth, color="black", ls="--", lw=2, label="ground truth")

    ax1.set_ylabel("predicted steps remaining")
    ax1.set_title("T2S predictions on "
                   + ("a SUCCESSFUL rollout" if succeeded else
                      "a rollout that NEVER succeeds — curves in the red band predict "
                      "imminent success that never comes"))
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)

    ax2.plot(hand_peg_distance_curve(obs), color="tab:green", label="hand -> peg")
    ax2.plot(peg_goal_distance_curve(obs), color="tab:orange", label="peg -> goal")
    ax2.set_xlabel("step")
    ax2.set_ylabel("distance")
    ax2.set_title("what physically happened (flat peg->goal = the object never moved)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig, worst


def annotate_video(frames, preds, obs=None, danger_threshold=10.0):
    """
    Burn the T2S prediction into each frame so the video itself is the
    evidence: you can watch the arm hover empty-handed while the number in
    the corner claims success is imminent.

    frames: list of RGB arrays from rollout(..., render=True)
    preds:  per-frame predictions (same length as frames)
    obs:    optional observations, to also show hand->peg distance
    """
    from PIL import Image, ImageDraw

    from progress import hand_peg_distance_curve

    hand = hand_peg_distance_curve(obs) if obs is not None else None
    out = []
    for i, f in enumerate(frames):
        img = Image.fromarray(np.asarray(f, dtype=np.uint8))
        d = ImageDraw.Draw(img)
        p = float(preds[i]) if i < len(preds) else float("nan")
        # red when the model claims success is close — the failure being shown
        color = (255, 60, 60) if p < danger_threshold else (255, 255, 255)
        d.text((8, 8), f"T2S {p:6.1f}", fill=color)
        if hand is not None and i < len(hand):
            d.text((8, 22), f"hand->peg {hand[i]:.3f}", fill=(200, 200, 200))
        d.text((8, 36), f"step {i}", fill=(160, 160, 160))
        out.append(np.array(img))
    return out
