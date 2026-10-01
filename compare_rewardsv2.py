#!/usr/bin/env python3
"""
Does the T2S reward move with MetaWorld's reward on the same trajectories?

    python compare_rewards.py --t2s-run v5 --models tdlambdaboot_succ td0boot_all
    python compare_rewards.py --t2s-run v5 --models td0boot_all --reward-mode absolute
    python compare_rewards.py --t2s-run v5 --models td0boot_all --seed-index 1

One trajectory per kind (--seed-index picks which), one plot each. Spearman between the T2S reward the policy is paid
(for --reward-mode) and the matching MetaWorld quantity:

    absolute                  -T2S(s')            vs  MetaWorld reward        (both levels)
    difference(_timed)        T2S(s)-g*T2S(s')    vs  change in MetaWorld     (both per-step changes)

The time penalty is a constant shift, so it does not change Spearman.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

STAGE_MANIFEST = "stage_manifest.json"
MW_STAGES = ("near_object", "grasp_success", "grasp_reward", "in_place_reward")
STAGE_COLORS = ("tab:purple", "tab:red", "tab:brown", "tab:pink")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Spearman: T2S reward vs MetaWorld reward",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--t2s-run", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--data-run", default=None, help="default: the T2S run's data run")
    p.add_argument("--kinds", nargs="+", default=["clean_success", "clean_failure",
                                                  "perturbed_recovery", "perturbed_failure"])
    p.add_argument("--seed-index", type=int, default=1,
                   help="which stored trajectory of each kind to check (0 = first)")
    p.add_argument("--reward-mode", default="difference_timed",
                   choices=["absolute", "difference", "difference_timed"])
    p.add_argument("--gamma", type=float, default=None, help="default: config.RL_GAMMA")
    p.add_argument("--time-penalty", type=float, default=None, help="default: config.TIME_PENALTY")
    return p.parse_args(argv)


def reroll(policy, seed, perturb):
    """Replay a stored reference with data_collection's seeding, recording MetaWorld's reward and stages."""
    import torch
    from config import MAX_STEPS, SUCCESS_KEY, REFERENCE_TAIL
    from env_utils import make_fixed_scene_env

    env = make_fixed_scene_env()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    O, R, S, ss = [], [], [], None
    for t in range(MAX_STEPS):
        O.append(np.asarray(obs, dtype=np.float32).copy())
        if t < perturb:
            a = rng.uniform(env.action_space.low, env.action_space.high)
        else:
            a, _ = policy.predict(obs, deterministic=False)
        obs, r, term, trunc, info = env.step(a)
        R.append(float(r))
        S.append({k: float(info[k]) for k in MW_STAGES if k in info})
        if info.get(SUCCESS_KEY, 0) and ss is None:
            ss = t + 1
        if term or trunc or (ss is not None and t >= ss + REFERENCE_TAIL):
            break
    O.append(np.asarray(obs, dtype=np.float32).copy())
    env.close()
    return np.array(O), np.array(R), S, ss


def reward_rho(pred, mw_r, ss, reward_mode, gamma, tp):
    """(rho, x, y): Spearman between the T2S reward x and MetaWorld's y, up to success."""
    from reward_fn import step_reward

    n = len(mw_r) if ss is None else min(ss, len(mw_r))     # transition i: s_i -> s_{i+1}
    nxt = pred[1:n + 1].copy()
    if ss is not None and ss <= n:
        nxt[ss - 1] = 0.0                                   # the wrapper's terminal value
    t2s = np.array([step_reward(pred[i], nxt[i], reward_mode=reward_mode,
                                time_penalty=tp, gamma=gamma) for i in range(n)])
    if reward_mode == "absolute":
        x, y = t2s, mw_r[:n]                                # levels at s_{i+1}
    else:
        x, y = t2s[1:], np.diff(mw_r[:n])                   # changes s_i -> s_{i+1}, i >= 1
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None, x, y
    return float(spearmanr(x, y)[0]), x, y


def plot_one(title, pred, mw_r, stages, ss, rho, x, y, reward_mode, path):
    """Left: the correlation. Right: MetaWorld's reward stages against the T2S prediction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    a1.scatter(x, y, s=14, alpha=0.6, color="tab:orange", edgecolors="none")
    if len(x) >= 2 and np.std(x) > 0:
        k, b = np.polyfit(x, y, 1)
        xx = np.linspace(np.min(x), np.max(x), 50)
        a1.plot(xx, k * xx + b, color="black", lw=1.2)
    a1.text(0.03, 0.97, f"Spearman \u03c1 = {'--' if rho is None else f'{rho:+.2f}'}   n = {len(x)}",
            transform=a1.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8"))
    if reward_mode == "absolute":
        a1.set_xlabel("T2S reward  -T2S(s')"); a1.set_ylabel("MetaWorld reward")
    else:
        a1.set_xlabel(f"T2S reward ({reward_mode})"); a1.set_ylabel("\u0394 MetaWorld reward")
    a1.set_title("Correlation (transitions up to success)", fontsize=10)
    a1.grid(alpha=0.25)

    t = np.arange(1, len(mw_r) + 1)                         # stage values describe s_{i+1}
    lines = []
    for k, c in zip(MW_STAGES, STAGE_COLORS):
        vals = [d.get(k, np.nan) for d in stages]
        if not np.all(np.isnan(vals)):
            lines += a2.plot(t, vals, color=c, lw=1.4, label=k)
    a2.set_ylabel("MetaWorld stage value")
    a2.set_xlabel("step")
    a3 = a2.twinx()
    lines += a3.plot(np.arange(len(pred)), pred, color="tab:orange", lw=2, ls="--",
                     label="T2S prediction")
    a3.set_ylabel("T2S predicted steps", color="tab:orange")
    if ss is not None:
        lines.append(a2.axvline(ss, color="tab:green", ls=":", lw=1.5, label=f"success @ {ss}"))
    a2.legend(handles=lines, fontsize=7, loc="center right")
    a2.set_title("MetaWorld reward stages vs T2S prediction", fontsize=10)
    a2.grid(alpha=0.25)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)

    import config
    import results
    from stable_baselines3 import SAC
    from t2s_model import load_t2s_predictor

    gamma = config.RL_GAMMA if args.gamma is None else args.gamma
    tp = config.TIME_PENALTY if args.time_penalty is None else args.time_penalty

    t2s_dir = results.get_run_dir("t2s_model", args.t2s_run)
    with open(os.path.join(t2s_dir, "manifest.json")) as f:
        combos = json.load(f)["combos"]
    missing = [m for m in args.models if m not in combos]
    if missing:
        raise SystemExit(f"ERROR: {missing} not in {args.t2s_run}; available: {sorted(combos)}")
    fns = {m: load_t2s_predictor(t2s_dir, *m.rsplit("_", 1), seed=combos[m].get("best_seed", 0))
           for m in args.models}

    data_run = args.data_run
    if data_run is None:
        with open(os.path.join(t2s_dir, STAGE_MANIFEST)) as f:
            data_run = json.load(f)["data_run"]
    ref_dir = os.path.join(results.get_run_dir("data_collection", data_run), "references")
    with open(os.path.join(ref_dir, "index.json")) as f:
        index = json.load(f)
    pols = {k: SAC.load(os.path.join(config.EXPERT_POLICY_DIR, index[f"{k}_ckpt"]))
            for k in ("success", "failure")}

    out = os.path.join(ref_dir, "reward_compare")
    print(f"=== T2S vs MetaWorld: reward_mode={args.reward_mode}  gamma={gamma} ===\n")
    rows = []
    for kind in args.kinds:
        cands = [r for r in index["trajectories"] if r["kind"] == kind]
        if not cands:
            print(f"  {kind}: no stored reference")
            continue
        for ref in [cands[min(args.seed_index, len(cands) - 1)]]:
            obs, mw_r, stages, ss = reroll(pols[ref["policy"]], ref["seed"], ref["perturb_steps"])
            stored = np.load(os.path.join(ref_dir, ref["name"] + ".npz"))["obs"]
            match = len(stored) == len(obs) and np.allclose(stored, obs, atol=1e-4)
            for m, fn in fns.items():
                pred = np.array([fn(o) for o in obs], dtype=float)
                rho, x, y = reward_rho(pred, mw_r, ss, args.reward_mode, gamma, tp)
                os.makedirs(os.path.join(out, m), exist_ok=True)
                plot_one(f"{m}  |  {ref['name']}  |  {args.reward_mode}"
                         + ("" if match else "  (replay diverged)"),
                         pred, mw_r, stages, ss, rho, x, y, args.reward_mode,
                         os.path.join(out, m, f"{ref['name']}_{args.reward_mode}.png"))
                rows.append(dict(model=m, kind=kind, name=ref["name"], success=ss is not None,
                                 rho=rho, n=len(x), replay_match=bool(match)))

    f = lambda v: "--" if v is None else f"{v:+.2f}"
    print(f"{'model':<22}{'trajectory':<32}{'rho':>7}{'n':>6}")
    print("-" * 67)
    for r in rows:
        print(f"{r['model']:<22}{r['name']:<32}{f(r['rho']):>7}{r['n']:>6}"
              + ("" if r["replay_match"] else "  (replay diverged)"))

    mean = lambda m, s: (lambda v: float(np.mean(v)) if v else None)(
        [r["rho"] for r in rows if r["model"] == m and r["success"] == s and r["rho"] is not None])
    print(f"\n{'model':<22}{'rho succ':>10}{'rho fail':>10}")
    print("-" * 42)
    for m in fns:
        print(f"{m:<22}{f(mean(m, True)):>10}{f(mean(m, False)):>10}")

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"spearman_{args.reward_mode}.json"), "w") as fh:
        json.dump(dict(t2s_run=args.t2s_run, reward_mode=args.reward_mode, gamma=gamma,
                       rows=rows), fh, indent=2)
    print(f"\nwrote {out}/spearman_{args.reward_mode}.json and plots in {out}/<model>/")
    return 0


if __name__ == "__main__":
    sys.exit(main())