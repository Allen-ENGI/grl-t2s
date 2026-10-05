#!/usr/bin/env python3
"""
One graph: what drives the T2S prediction at each step of one trajectory.

    python t2s_saliency.py --t2s-run v12 --model tdlambdaboot_all
    python t2s_saliency.py --t2s-run v12 --model tdlambdaboot_all --reference clean_success_seed900002
    python t2s_saliency.py --t2s-run v12 --model tdlambdaboot_all \
        --policy-run diff_time_v2_tdlambdaboot_all_seed0 --checkpoint policy_600000.zip

Top:    the T2S prediction along the trajectory (and the true steps remaining, on the
        model's scale, when the trajectory has one).
Bottom: for each input group, how much it has moved the prediction away from the start
        state's prediction at that step (integrated gradients; red = pushes the prediction
        up, "further from success"; blue = pushes it down, "closer"). Groups that never
        contribute (inputs that do not change) are left out.

Uses load_model / integrated_gradients / GROUPS from t2s_attribution.py.
"""
import argparse
import json
import os

import numpy as np

import config
import results
from t2s_attribution import GROUPS, grouped, integrated_gradients, load_model, predict


def policy_episode(policy_run, checkpoint, seed):
    import torch
    from stable_baselines3 import SAC
    from env_utils import make_fixed_scene_env

    policy = SAC.load(os.path.join(results.get_run_dir("policy", policy_run), checkpoint), device="cpu")
    env = make_fixed_scene_env()
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    states, success = [np.asarray(obs, np.float32)], None
    for t in range(config.MAX_STEPS):
        obs, _r, term, trunc, info = env.step(policy.predict(obs, deterministic=False)[0])
        states.append(np.asarray(obs, np.float32))
        if info.get(config.SUCCESS_KEY, 0) and success is None:
            success = t + 1
        if term or trunc or (success is not None and t >= success + 10):
            break
    env.close()
    states = np.array(states)
    truth = config.steps_remaining_curve(len(states), success) if success is not None else None
    return states, truth, success


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--t2s-run", required=True)
    p.add_argument("--model", required=True)
    # clean_failure_seed900001 clean_success_seed900001
    p.add_argument("--reference", default="clean_failure_seed900001",
                   help="stored reference trajectory of the T2S run's data run")
    p.add_argument("--policy-run", default=None, help="film an episode of this policy instead")
    p.add_argument("--checkpoint", default="policy_final.zip")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    t2s_dir = results.get_run_dir("t2s_model", args.t2s_run)
    manifest = json.load(open(os.path.join(t2s_dir, "manifest.json")))["combos"][args.model]
    model, x_mean, x_std = load_model(t2s_dir, args.model, manifest.get("best_seed", 0))
    norm = lambda X: ((np.asarray(X) - x_mean) / x_std).astype(np.float32)

    if args.policy_run:
        states, truth, success = policy_episode(args.policy_run, args.checkpoint, args.seed)
        title = f"{args.policy_run}/{args.checkpoint}, seed {args.seed}"
    else:
        data_run = json.load(open(os.path.join(t2s_dir, "stage_manifest.json")))["data_run"]
        ref = np.load(os.path.join(results.get_run_dir("data_collection", data_run), "references",
                                   args.reference + ".npz"))
        states, truth = ref["obs"], ref["y_steps"]
        success = int(ref["success_step"][0]) if ref["success_step"][0] >= 0 else None
        title = f"reference {args.reference}"

    xn = norm(states)
    pred = predict(model, xn)
    contrib = grouped(integrated_gradients(model, xn, xn[0]))          # relative to this episode's start
    rows = [g for g in GROUPS if np.abs(contrib[g]).max() > 1e-3]
    mat = np.array([contrib[g] for g in rows])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True,
                                 gridspec_kw=dict(height_ratios=(1, 1.3)))
    a1.plot(pred, color="tab:orange", lw=2, label="T2S prediction")
    if truth is not None:
        g = config.T2S_BOOTSTRAP_GAMMA
        scaled = (1 - g ** truth) / (1 - g) if manifest.get("censor_scheme") == "bootstrap" else truth
        a1.plot(scaled, color="black", lw=1.2, ls="--", label="true steps remaining (model's scale)")
    if success is not None:
        for a in (a1, a2):
            a.axvline(success, color="tab:green", ls=":", lw=1.5)
    a1.set_ylabel("prediction"); a1.grid(alpha=0.3); a1.legend(fontsize=8)
    a1.set_title(f"{args.model} ({args.t2s_run}) on {title}", fontsize=10)
    lim = np.abs(mat).max() or 1.0
    im = a2.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim,
                   extent=(-0.5, len(pred) - 0.5, len(rows) - 0.5, -0.5))
    a2.set_yticks(range(len(rows))); a2.set_yticklabels(rows, fontsize=9)
    a2.set_xlabel("step")
    fig.colorbar(im, ax=a2, label="change in prediction caused since the start (steps)")
    fig.tight_layout()

    name = (f"saliency_{args.policy_run}_{args.checkpoint[:-4]}_seed{args.seed}" if args.policy_run
            else f"saliency_{args.reference}")
    out = os.path.join(t2s_dir, "attribution", name + ".png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()