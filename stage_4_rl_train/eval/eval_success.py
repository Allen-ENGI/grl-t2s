#!/usr/bin/env python3
"""
Success rate of trained policies, over many seeds.

    python eval_success.py --policy-run diff_time_v2_tdlambdaboot_all_seed0 --checkpoints policy_600000.zip
    python eval_success.py --policy-run <run> --checkpoints policy_550000.zip policy_600000.zip --episodes 100
    python eval_success.py --policy-run <run> --all-checkpoints --every 50000 --episodes 50 --workers 6
    python eval_success.py --policy-run <run> --checkpoints policy_600000.zip --deterministic

Each episode runs the policy on the fixed scene from reset seed BASE + i (and the same
seed for the policy's action sampling), up to 500 steps, stopping at success. Reports per
checkpoint:
    success rate with a 95% interval (Wilson), number of successes, and the median and
    mean steps to success over the successful episodes.

With --all-checkpoints it evaluates every saved policy_<step>.zip (optionally every
--every steps) and plots success rate against training step.

Note: with --deterministic on the fixed scene, every episode starts from the same state
and takes the same actions, so all episodes are identical and the rate is 0% or 100%.
"""
import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

try:
    import config
except ImportError:
    import core.config as config
    
try:
    import core.results as results
except ImportError:
    from core import results

SEED_BASE = 20_000


def run_episodes(args):
    """Worker: (checkpoint path, seeds, deterministic) -> list of success steps (None = failed)."""
    path, seeds, deterministic = args
    import torch
    from stable_baselines3 import SAC
    try:
        from env_utils import make_fixed_scene_env
    except ImportError:
        from core.env_utils import make_fixed_scene_env

    torch.set_num_threads(1)
    policy = SAC.load(path, device="cpu")
    out = []
    for seed in seeds:
        env = make_fixed_scene_env()
        torch.manual_seed(seed)
        obs, _ = env.reset(seed=seed)
        success = None
        for t in range(config.MAX_STEPS):
            action, _ = policy.predict(obs, deterministic=deterministic)
            obs, _r, terminated, truncated, info = env.step(action)
            if info.get(config.SUCCESS_KEY, 0):
                success = t + 1
                break
            if terminated or truncated:
                break
        env.close()
        out.append(success)
    return out


def wilson(k, n, z=1.96):
    """95% interval for a success rate k/n (works well also near 0% and 100%)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def evaluate(path, n_episodes, deterministic, workers):
    seeds = [SEED_BASE + i for i in range(n_episodes)]
    if workers <= 1:
        steps = run_episodes((path, seeds, deterministic))
    else:
        chunks = [seeds[i::workers] for i in range(workers)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(run_episodes, [(path, c, deterministic) for c in chunks if c]))
        by_seed = {}
        for chunk, part in zip([c for c in chunks if c], parts):
            by_seed.update(dict(zip(chunk, part)))
        steps = [by_seed[s] for s in seeds]
    ok = [s for s in steps if s is not None]
    low, high = wilson(len(ok), len(steps))
    return dict(episodes=len(steps), successes=len(ok), success_rate=len(ok) / len(steps),
                interval_low=low, interval_high=high,
                median_steps=float(np.median(ok)) if ok else None,
                mean_steps=float(np.mean(ok)) if ok else None,
                steps=steps)


def checkpoint_step(name):
    m = re.search(r"_(\d+)\.zip$", name)
    return int(m.group(1)) if m else None


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--policy-run", required=True)
    p.add_argument("--checkpoints", nargs="+", default=None, help="checkpoint file names in the run")
    p.add_argument("--all-checkpoints", action="store_true", help="every saved policy_<step>.zip")
    p.add_argument("--every", type=int, default=None, help="with --all-checkpoints: only every N steps")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--deterministic", action="store_true", help="mean action instead of sampling")
    p.add_argument("--workers", type=int, default=1, help="parallel processes (CPU cores)")
    args = p.parse_args()

    run_dir = results.get_run_dir("policy", args.policy_run)
    if args.all_checkpoints:
        names = sorted((os.path.basename(f) for f in glob.glob(os.path.join(run_dir, "policy_*.zip"))
                        if checkpoint_step(os.path.basename(f)) is not None),
                       key=checkpoint_step)
        if args.every:
            names = [n for n in names if checkpoint_step(n) % args.every == 0]
        final = os.path.join(run_dir, "policy_final.zip")
        if os.path.exists(final):
            names.append("policy_final.zip")
    else:
        names = args.checkpoints or ["policy_final.zip"]
    if not names:
        raise SystemExit(f"no checkpoints found in {run_dir}")

    mode = "deterministic" if args.deterministic else "sampled"
    if args.deterministic:
        print("note: deterministic actions on the fixed scene make every episode identical\n")
    print(f"{args.policy_run}: {len(names)} checkpoint(s), {args.episodes} episodes each, {mode} actions\n")
    print(f"{'checkpoint':<26}{'success':>10}{'95% interval':>18}{'successes':>12}"
          f"{'median steps':>14}{'mean steps':>12}")
    rows = []
    for name in names:
        path = os.path.join(run_dir, name)
        if not os.path.exists(path):
            print(f"{name:<26}  (not found)")
            continue
        r = evaluate(path, args.episodes, args.deterministic, args.workers)
        r.update(checkpoint=name, step=checkpoint_step(name))
        rows.append(r)
        med = "--" if r["median_steps"] is None else f"{r['median_steps']:.0f}"
        mean = "--" if r["mean_steps"] is None else f"{r['mean_steps']:.1f}"
        interval = f"{r['interval_low']:.0%}-{r['interval_high']:.0%}"
        count = f"{r['successes']}/{r['episodes']}"
        print(f"{name:<26}{r['success_rate']:>9.0%}{interval:>18}{count:>12}{med:>14}{mean:>12}")

    out = os.path.join(run_dir, f"eval_success_{mode}.json")
    json.dump(dict(policy_run=args.policy_run, mode=mode, episodes=args.episodes, seed_base=SEED_BASE,
                   rows=rows), open(out, "w"), indent=2)
    print(f"\nwrote {out}")

    curve = [r for r in rows if r["step"] is not None]
    if len(curve) > 1:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = [r["step"] for r in curve]
        y = [100 * r["success_rate"] for r in curve]
        lo = [100 * r["interval_low"] for r in curve]
        hi = [100 * r["interval_high"] for r in curve]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.fill_between(x, lo, hi, alpha=0.2, color="tab:green", label="95% interval")
        ax.plot(x, y, marker="o", color="tab:green", label="success rate")
        ax.set_xlabel("training step"); ax.set_ylabel("success rate (%)"); ax.set_ylim(-2, 102)
        ax.set_title(f"{args.policy_run}: {args.episodes} episodes per checkpoint, {mode} actions",
                     fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout()
        plot = os.path.join(run_dir, f"eval_success_{mode}.png")
        fig.savefig(plot, dpi=140)
        print(f"wrote {plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())