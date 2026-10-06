#!/usr/bin/env python3
"""
Where can the expert pick up and insert the peg?

    python peg_range.py                                   # 7 x 7 grid across the training range
    python peg_range.py --grid 9 --episodes 8 --margin-cm 3
    python peg_range.py --expert peg_insert_side_v3_final.zip --deterministic
    python peg_range.py --x-range -0.10 0.30 --y-range 0.40 0.80   # test outside the training range

For each peg start on a grid (x, y), the expert tries --episodes times. Per position:
    grasped   share of episodes where MetaWorld's grasp-and-lift bonus appeared (reward >= 1)
    inserted  share of episodes with MetaWorld's success flag
    steps     mean steps to success, over the successful episodes

The grid covers the environment's own peg sampling range (the range the expert was
trained on), extended by --margin-cm on every side, so you can see where competence
ends. --x-range / --y-range replace that grid with any range you choose, e.g. to test
peg positions outside the training range. The hole (the target the success check uses)
and the fixed-scene peg start are marked. Writes peg_range.png and peg_range.json into the current t2s_eval run folder.
"""
import argparse
import json
import os

import numpy as np
import torch
from stable_baselines3 import SAC

import core.config as config
import core.results as results
from core.config import GOAL_POS_IDX
from core.env_utils import make_fixed_scene_env
from core.scene_edit import reset_with_peg_offset


def attempt(policy, seed, offset, deterministic):
    env = make_fixed_scene_env()
    torch.manual_seed(seed)
    obs = reset_with_peg_offset(env, seed, offset)
    grasped, success = False, None
    for t in range(config.MAX_STEPS):
        obs, reward, term, trunc, info = env.step(policy.predict(obs, deterministic=deterministic)[0])
        grasped = grasped or reward >= 1.0
        if info.get(config.SUCCESS_KEY, 0):
            success = t + 1
            break
        if term or trunc:
            break
    env.close()
    return grasped, success


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--expert", default=None, help="checkpoint file in EXPERT_POLICY_DIR "
                                                  "(default: the eval run's success checkpoint)")
    p.add_argument("--eval-run", default="v8", help="t2s_eval run whose success checkpoint to use")
    p.add_argument("--grid", type=int, default=7, help="points per axis")
    p.add_argument("--episodes", type=int, default=5, help="attempts per position")
    p.add_argument("--margin-cm", type=float, default=2.0,
                   help="extend the default grid past the training range (ignored for an axis "
                        "given with --x-range / --y-range)")
    p.add_argument("--x-range", type=float, nargs=2, default=None, metavar=("XMIN", "XMAX"),
                   help="peg x positions to test, in metres (default: training range +- margin)")
    p.add_argument("--y-range", type=float, nargs=2, default=None, metavar=("YMIN", "YMAX"),
                   help="peg y positions to test, in metres (default: training range +- margin)")
    p.add_argument("--deterministic", action="store_true", help="mean action instead of sampling")
    args = p.parse_args()

    eval_dir = results.get_run_dir("t2s_eval", args.eval_run)
    name = args.expert or json.load(open(os.path.join(eval_dir, "stage_manifest.json")))["success_ckpt"]
    expert = SAC.load(os.path.join(config.EXPERT_POLICY_DIR, name), device="cpu")

    env = make_fixed_scene_env()
    obs0, _ = env.reset(seed=0)
    u = env.unwrapped
    base = np.asarray(u._last_rand_vec, float)[:3]
    hole = np.asarray(obs0, float)[GOAL_POS_IDX]           # the target the success check uses
    low, high = np.asarray(u._random_reset_space.low[:3]), np.asarray(u._random_reset_space.high[:3])
    env.close()
    m = args.margin_cm / 100
    x0, x1 = args.x_range if args.x_range else (low[0] - m, high[0] + m)
    y0, y1 = args.y_range if args.y_range else (low[1] - m, high[1] + m)
    xs = np.linspace(x0, x1, args.grid)
    ys = np.linspace(y0, y1, args.grid)
    print(f"expert {name}; fixed peg start {base[:2].round(3)}; hole {hole[:2].round(3)}; "
          f"training range x {low[0]:.3f}-{high[0]:.3f}, y {low[1]:.3f}-{high[1]:.3f}")
    print(f"testing x {xs[0]:.3f}-{xs[-1]:.3f}, y {ys[0]:.3f}-{ys[-1]:.3f} "
          f"({args.grid} x {args.grid} positions, {args.episodes} attempts each)")

    grasp = np.zeros((len(ys), len(xs)))
    insert = np.zeros((len(ys), len(xs)))
    steps = np.full((len(ys), len(xs)), np.nan)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            offset = (x - base[0], y - base[1], 0.0)
            res = [attempt(expert, 1000 + k, offset, args.deterministic) for k in range(args.episodes)]
            grasp[j, i] = np.mean([g for g, _ in res])
            ok = [s for _, s in res if s is not None]
            insert[j, i] = len(ok) / len(res)
            if ok:
                steps[j, i] = np.mean(ok)
            inside = low[0] <= x <= high[0] and low[1] <= y <= high[1]
            print(f"  peg at x {x:.3f}, y {y:.3f} {'(inside)' if inside else '(outside)':<10}"
                  f" grasped {grasp[j, i]:.0%}  inserted {insert[j, i]:.0%}"
                  + (f"  steps {steps[j, i]:.0f}" if ok else ""))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, data, title, cmap, vmax in ((axes[0], grasp, "grasped and lifted", "Blues", 1),
                                        (axes[1], insert, "inserted (success)", "Greens", 1),
                                        (axes[2], steps, "steps to success", "viridis", None)):
        im = ax.imshow(data, origin="lower", cmap=cmap, vmin=0, vmax=vmax, aspect="auto",
                       extent=(xs[0], xs[-1], ys[0], ys[-1]))
        fig.colorbar(im, ax=ax)
        ax.add_patch(Rectangle((low[0], low[1]), high[0] - low[0], high[1] - low[1],
                               fill=False, ec="red", lw=1.5, label="training range"))
        ax.plot(*base[:2], "k*", ms=12, label="fixed-scene peg start")
        ax.plot(*hole[:2], marker="X", color="red", ms=12, ls="none", label="hole (target)")
        pad = 0.02
        ax.set_xlim(min(xs[0], low[0], hole[0]) - pad, max(xs[-1], high[0], hole[0]) + pad)
        ax.set_ylim(min(ys[0], low[1], hole[1]) - pad, max(ys[-1], high[1], hole[1]) + pad)
        ax.set_title(title); ax.set_xlabel("peg x (m)"); ax.set_ylabel("peg y (m)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"{name}, {args.episodes} attempts per position")
    fig.tight_layout()
    tag = f"x{xs[0]:.3f}_{xs[-1]:.3f}_y{ys[0]:.3f}_{ys[-1]:.3f}".replace("-", "m")
    out = os.path.join(eval_dir, f"peg_range_{tag}.png")
    fig.savefig(out, dpi=140)
    json.dump(dict(expert=name, xs=xs.tolist(), ys=ys.tolist(), grasped=grasp.tolist(),
                   inserted=insert.tolist(), steps=np.where(np.isnan(steps), None, steps).tolist(),
                   training_range=dict(low=low.tolist(), high=high.tolist()), fixed_start=base.tolist(),
                   hole=hole.tolist()),
              open(os.path.join(eval_dir, f"peg_range_{tag}.json"), "w"), indent=2)
    print(f"\nwrote {out} and peg_range_{tag}.json")


if __name__ == "__main__":
    main()