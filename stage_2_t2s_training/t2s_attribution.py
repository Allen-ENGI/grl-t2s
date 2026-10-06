#!/usr/bin/env python3
"""
Which inputs drive the T2S prediction, and where?

    python t2s_attribution.py --t2s-run v8 --model tdlambdaboot_all
    python t2s_attribution.py --t2s-run v8 --model tdlambdaboot_all \
        --policy-run diff_time_v1_tdlambdaboot_all_seed0 --checkpoint policy_900000.zip

The 39 inputs are grouped (current and previous frame together):
    hand position, gripper opening, peg position, peg orientation,
    unused object slots, goal / hole

Slices of states:
    expert reaching / carrying / inserting   successful dataset episodes, split into the
                                             first, middle and last third of the time to success
    failed episodes                          dataset episodes that never succeed
    policy states                            rollouts of --policy-run/--checkpoint (optional)

Two methods, per slice and group:
    integrated gradients   how much each group's change since the start state contributed to
                           the prediction's change since the start state (signed, in steps);
                           the contributions add up to prediction(s) - prediction(start)
    permutation reliance   mean absolute change in the prediction when the group's values are
                           shuffled across the slice (how much the model leans on the group)

A randomly initialised model with the same architecture is analysed the same way, as a
baseline: rankings that also appear there reflect the input structure, not learning.

Writes into <t2s run>/attribution/: summary.png, timeline_expert.png,
timeline_policy.png (if a policy is given) and attribution.json.
"""
import argparse
import json
import os

import numpy as np
import torch

import core.config as config
import core.results as results
from stage_2_t2s_training.t2s_io import load_normalization, checkpoint_name
from stage_2_t2s_training.t2s_model import T2SModel

GROUPS = {
    "hand position":       [0, 1, 2, 18, 19, 20],
    "gripper opening":     [3, 21],
    "peg position":        [4, 5, 6, 22, 23, 24],
    "peg orientation":     [7, 8, 9, 10, 25, 26, 27, 28],
    "unused object slots": list(range(11, 18)) + list(range(29, 36)),
    "goal / hole":         [36, 37, 38],
}
IG_STEPS = 32


# ---- model -------------------------------------------------------------------

def load_model(t2s_dir, model_name, seed):
    method, condition = model_name.rsplit("_", 1)
    x_mean, x_std = load_normalization(t2s_dir)
    model = T2SModel(obs_dim=len(x_mean))
    model.load_state_dict(torch.load(os.path.join(t2s_dir, checkpoint_name(method, condition, seed)),
                                     map_location="cpu"))
    model.eval()
    return model, x_mean, x_std


def untrained_like(model, x_dim):
    torch.manual_seed(0)
    m = T2SModel(obs_dim=x_dim)
    m.eval()
    return m


def predict(model, xn):
    with torch.no_grad():
        return model(torch.as_tensor(xn, dtype=torch.float32)).numpy()


def integrated_gradients(model, xn, ref):
    """Per-input contributions to model(x) - model(ref), straight path, IG_STEPS points."""
    x = torch.as_tensor(xn, dtype=torch.float32)
    r = torch.as_tensor(ref, dtype=torch.float32).expand_as(x)
    total = torch.zeros_like(x)
    for a in torch.linspace(1.0 / IG_STEPS, 1.0, IG_STEPS):
        point = (r + a * (x - r)).requires_grad_(True)
        grad, = torch.autograd.grad(model(point).sum(), point)
        total += grad
    return ((x - r) * total / IG_STEPS).numpy()


def grouped(per_input):
    return {g: per_input[:, idx].sum(axis=1) for g, idx in GROUPS.items()}


def permutation_reliance(model, xn, rng):
    base = predict(model, xn)
    out = {}
    for g, idx in GROUPS.items():
        shuffled = xn.copy()
        shuffled[:, idx] = xn[rng.permutation(len(xn))][:, idx]
        out[g] = float(np.mean(np.abs(predict(model, shuffled) - base)))
    return out


# ---- slices ------------------------------------------------------------------

def dataset_slices(dataset_path, split, max_rows, rng):
    d = np.load(dataset_path)
    X, y, ep, fr = d["X"], d["y_steps"], d["episode_ids"], d["frame_idxs"]
    keep = np.ones(len(X), bool) if split == "all" else d["split"].astype(str) == split
    rows = np.flatnonzero(keep)
    rows = rows[np.lexsort((fr[rows], ep[rows]))]
    slices = {"expert reaching": [], "expert carrying": [], "expert inserting": [], "failed episodes": []}
    first_success = None
    for g in np.split(rows, np.flatnonzero(np.diff(ep[rows])) + 1):
        zero = np.flatnonzero(y[g] == 0)
        if len(zero) and zero[0] >= 3:
            ss = int(zero[0])
            phase = np.arange(ss) / ss
            slices["expert reaching"].append(g[:ss][phase < 1 / 3])
            slices["expert carrying"].append(g[:ss][(phase >= 1 / 3) & (phase < 2 / 3)])
            slices["expert inserting"].append(g[:ss][phase >= 2 / 3])
            if first_success is None:
                first_success = g[:ss + 3]
        elif np.all(y[g] == config.CENSOR_LABEL):
            slices["failed episodes"].append(g)
    out = {}
    for name, parts in slices.items():
        if parts:
            r = np.concatenate(parts)
            out[name] = X[rng.choice(r, min(max_rows, len(r)), replace=False)]
    start = X[rows[fr[rows] == 0][0]]
    return out, start, (X[first_success] if first_success is not None else None)


def policy_states(policy_run, checkpoint, episodes):
    from stable_baselines3 import SAC
    from core.env_utils import make_fixed_scene_env

    policy = SAC.load(os.path.join(results.get_run_dir("policy", policy_run), checkpoint), device="cpu")
    rollouts = []
    for seed in range(episodes):
        env = make_fixed_scene_env()
        torch.manual_seed(seed)
        obs, _ = env.reset(seed=seed)
        states = [np.asarray(obs, np.float32)]
        for _ in range(config.MAX_STEPS):
            obs, _r, term, trunc, info = env.step(policy.predict(obs, deterministic=False)[0])
            states.append(np.asarray(obs, np.float32))
            if info.get(config.SUCCESS_KEY, 0) or term or trunc:
                break
        env.close()
        rollouts.append(np.array(states))
    return np.concatenate(rollouts), rollouts[0]


# ---- plots -------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_summary(table, path, title):


    slices = list(table["trained"])
    groups = list(GROUPS)
    fig, axes = plt.subplots(2, 2, figsize=(16, 8.5))
    width = 0.8 / len(slices)
    for row, which in enumerate(("trained", "untrained")):
        for col, (key, label) in enumerate((("ig_mean_abs", "integrated gradients: mean |contribution| (steps)"),
                                            ("permutation", "permutation: mean |change in prediction| (steps)"))):
            ax = axes[row][col]
            for k, s in enumerate(slices):
                vals = [table[which][s][key][g] for g in groups]
                ax.bar(np.arange(len(groups)) + k * width, vals, width, label=s)
            ax.set_xticks(np.arange(len(groups)) + width * (len(slices) - 1) / 2)
            ax.set_xticklabels(groups, rotation=20, fontsize=8)
            ax.set_title(f"{which} model: {label}", fontsize=9)
            ax.grid(alpha=0.3, axis="y")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

def plot_timeline(model, x_mean, x_std, states, ref_n, path, title):

    xn = ((states - x_mean) / x_std).astype(np.float32)
    contrib = grouped(integrated_gradients(model, xn, ref_n))
    pred = predict(model, xn)
    mat = np.array([contrib[g] for g in GROUPS])
    lim = np.abs(mat).max() or 1.0
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                 gridspec_kw=dict(height_ratios=(1, 1.4)))
    a1.plot(pred, color="tab:orange", lw=2)
    a1.set_ylabel("T2S prediction"); a1.grid(alpha=0.3); a1.set_title(title, fontsize=10)
    im = a2.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim,
                   extent=(-0.5, len(pred) - 0.5, len(GROUPS) - 0.5, -0.5))
    a2.set_yticks(range(len(GROUPS))); a2.set_yticklabels(list(GROUPS), fontsize=8)
    a2.set_xlabel("step")
    fig.colorbar(im, ax=a2, label="contribution to prediction - start prediction (steps)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---- main --------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--t2s-run", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--split", default="val", choices=["val", "train", "all"],
                   help="which dataset rows to analyse")
    p.add_argument("--max-rows", type=int, default=2000, help="rows per slice")
    p.add_argument("--policy-run", default=None)
    p.add_argument("--checkpoint", default="policy_final.zip")
    p.add_argument("--policy-episodes", type=int, default=3)
    args = p.parse_args()

    rng = np.random.default_rng(0)
    t2s_dir = results.get_run_dir("t2s_model", args.t2s_run)
    seed = json.load(open(os.path.join(t2s_dir, "manifest.json")))["combos"][args.model].get("best_seed", 0)
    model, x_mean, x_std = load_model(t2s_dir, args.model, seed)
    baseline = untrained_like(model, len(x_mean))
    dataset = json.load(open(os.path.join(t2s_dir, "stage_manifest.json")))["dataset_path"]

    slices, start, expert_episode = dataset_slices(dataset, args.split, args.max_rows, rng)
    policy_episode = None
    if args.policy_run:
        states, policy_episode = policy_states(args.policy_run, args.checkpoint, args.policy_episodes)
        slices["policy states"] = states[rng.choice(len(states), min(args.max_rows, len(states)),
                                                    replace=False)]
    norm = lambda X: ((X - x_mean) / x_std).astype(np.float32)
    ref = norm(start[None])[0]

    out_dir = os.path.join(t2s_dir, "attribution")
    os.makedirs(out_dir, exist_ok=True)
    table = {"trained": {}, "untrained": {}}
    for which, m in (("trained", model), ("untrained", baseline)):
        for name, X in slices.items():
            xn = norm(X)
            ig = integrated_gradients(m, xn, ref)
            g = grouped(ig)
            gap = float(np.mean(np.abs(ig.sum(axis=1) - (predict(m, xn) - predict(m, ref[None])[0]))))
            table[which][name] = dict(
                rows=len(X), mean_prediction=float(predict(m, xn).mean()), ig_completeness_gap=gap,
                ig_mean_abs={k: float(np.mean(np.abs(v))) for k, v in g.items()},
                ig_mean_signed={k: float(np.mean(v)) for k, v in g.items()},
                permutation=permutation_reliance(m, xn, rng))

    start_pred = float(predict(model, ref[None])[0])
    print(f"model {args.model} ({args.t2s_run}); prediction at the start state {start_pred:.1f}\n")
    for name, r in table["trained"].items():
        print(f"{name}  ({r['rows']} states, mean prediction {r['mean_prediction']:.1f})")
        print(f"  {'group':<22}{'IG mean |contrib|':>19}{'IG mean signed':>16}{'permutation':>13}"
              f"{'untrained perm.':>17}")
        for grp in GROUPS:
            print(f"  {grp:<22}{r['ig_mean_abs'][grp]:>19.2f}{r['ig_mean_signed'][grp]:>+16.2f}"
                  f"{r['permutation'][grp]:>13.2f}{table['untrained'][name]['permutation'][grp]:>17.2f}")
        print(f"  (integrated-gradients check: contributions miss the prediction change by "
              f"{r['ig_completeness_gap']:.2f} steps on average)\n")

    plot_summary(table, os.path.join(out_dir, "summary.png"), f"{args.model} ({args.t2s_run})")
    if expert_episode is not None:
        plot_timeline(model, x_mean, x_std, expert_episode, ref, os.path.join(out_dir, "timeline_expert.png"),
                      "expert success: prediction (top) and each group's contribution (bottom)")
    if policy_episode is not None:
        plot_timeline(model, x_mean, x_std, policy_episode, ref, os.path.join(out_dir, "timeline_policy.png"),
                      f"{args.policy_run}/{args.checkpoint}: prediction and contributions")
    json.dump(dict(model=args.model, t2s_run=args.t2s_run, start_prediction=start_pred, table=table),
              open(os.path.join(out_dir, "attribution.json"), "w"), indent=2)
    print(f"wrote {out_dir}")


    d = np.load(results.get_run_dir("data_collection", "v3") + "/dataset.npz")
    q = d["X"][:, 7:11]
    print("orientation inputs, spread:", q.std(axis=0).round(6))
    print("example values:", q[:3].round(4), "...", q[-3:].round(4))
    

if __name__ == "__main__":
    main()