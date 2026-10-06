#!/usr/bin/env python3
"""
Does the T2S prediction go DOWN as the task progresses?

    python eval_monotonic.py --t2s-run v5 --models tdlambdaboot_all td0boot_all
    python eval_monotonic.py --t2s-run v5 --models tdlambdaboot_all --stages

Scores successful held-out val episodes from dataset.npz (no simulator).
--stages also re-rolls one clean_success reference and prints the mean
prediction per MetaWorld stage.
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

import core.results as results
from stage_2_t2s_training.t2s_model import load_t2s_predictor


def ranking_metrics(pred, ss):
    """pred: predictions for frames 0..ss (ss = first successful frame)."""
    p = np.asarray(pred[:ss + 1], dtype=float)
    truth = ss - np.arange(ss + 1)                 # true steps remaining
    d = np.diff(p)
    upper = np.triu(np.ones((len(p), len(p)), dtype=bool), 1)
    below_1 = np.flatnonzero(p < 1.0)
    return dict(
        spearman=float(spearmanr(p, truth)[0]),                       # 1 = perfect order
        pair_acc=float((p[:, None] > p[None, :])[upper].mean()),      # earlier frame predicted higher
        rise_frac=float(np.mean(d > 0.5)),                            # steps going the wrong way
        drop_per_step=float(-d.mean()),                               # ideal = 1
        zero_early=int(ss - below_1[0]) if len(below_1) else 0,       # steps at ~0 before success
    )


def val_success_episodes(dataset_path):
    d = np.load(dataset_path)
    X, y, ep = d["X"], d["y_steps"], d["episode_ids"]
    val = d["split"].astype(str) == "val"
    for e in np.unique(ep[val]):
        idx = np.flatnonzero(ep == e)
        zero = np.flatnonzero(y[idx] == 0)
        if len(zero) and zero[0] >= 5:
            yield X[idx], int(zero[0])


def stage_table(pred, stages):
    """Mean prediction per MetaWorld stage; should fall stage by stage."""
    def stage(s):
        if s.get("in_place_reward", 0) > 0.5:
            return "3 placing"
        if s.get("grasp_reward", 0) > 0.4:
            return "2 grasped"
        if s.get("near_object", 0) > 0.5:
            return "1 near"
        return "0 reaching"
    labels = [stage(s) for s in stages]            # stages[i] describes frame i+1
    for name in sorted(set(labels)):
        v = [pred[i + 1] for i, l in enumerate(labels) if l == name]
        print(f"    {name:<12} n={len(v):>4}   mean pred {np.mean(v):7.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--t2s-run", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--stages", action="store_true")
    args = p.parse_args()

    t2s_dir = results.get_run_dir("t2s_model", args.t2s_run)
    stage2 = json.load(open(os.path.join(t2s_dir, "stage_manifest.json")))
    episodes = list(val_success_episodes(stage2["dataset_path"]))
    print(f"{len(episodes)} successful val episodes\n")

    keys = ("spearman", "pair_acc", "rise_frac", "drop_per_step", "zero_early")
    print(f"{'model':<22}" + "".join(f"{k:>14}" for k in keys))
    fns = {}
    for m in args.models:
        fns[m] = load_t2s_predictor(t2s_dir, *m.rsplit("_", 1),
                                    seed=stage2["combos"][m]["best_seed"])
        rows = [ranking_metrics([fns[m](o) for o in X[:ss + 1]], ss) for X, ss in episodes]
        print(f"{m:<22}" + "".join(f"{np.median([r[k] for r in rows]):>14.2f}" for k in keys))
    print("\nmedians over episodes. ideal: spearman 1, pair_acc 1, rise_frac 0, "
          "drop_per_step 1, zero_early 0")

    if args.stages:
        import core.config as config
        from stable_baselines3 import SAC
        from stage_3_t2s_eval.compare_rewards import reroll

        ref_dir = os.path.join(os.path.dirname(stage2["dataset_path"]), "references")
        index = json.load(open(os.path.join(ref_dir, "index.json")))
        ref = next(r for r in index["trajectories"] if r["kind"] == "clean_success")
        pol = SAC.load(os.path.join(config.EXPERT_POLICY_DIR, index["success_ckpt"]))
        obs, _mw, stages, ss = reroll(pol, ref["seed"], ref["perturb_steps"])
        print(f"\nstages on {ref['name']} (success @ {ss}):")
        for m, fn in fns.items():
            print(f"  {m}")
            stage_table([fn(o) for o in obs], stages)


if __name__ == "__main__":
    main()
