#!/usr/bin/env python3
"""
What is the policy doing when the training reward goes up or down?

    # all checkpoints, 3 seeds x 100 episodes each
    python training_eval.py --policy-run diff_v2_tdlambdaboot_all_seed1 \
        --t2s-run v6 --model tdlambdaboot_all

    # only specific checkpoints (much faster)
    python training_eval.py --policy-run diff_v2_tdlambdaboot_all_seed1 \
        --t2s-run v6 --model tdlambdaboot_all --ckpts 550000 600000 final

Each checkpoint is rolled out for --episodes episodes under EACH of --seeds
(default 3 seeds x 100 = 300 rollouts, sampled actions like training). The scene is
pinned, so variation comes from the policy's sampling: every episode gets its own
torch seed (seed * 100_000 + i). Re-using one torch seed would replay one episode.

For each checkpoint it reports, next to TensorBoard's ep_rew_mean:

  SUCCESS per evaluation seed, and mean +- std across seeds
  WHERE the return comes from (difference mode splits it exactly):
    start   pred(s0) - gamma * pred(end)      progress: ending lower than it started
    hover   everything else = (1-gamma)*sum(pred)   paid for time spent, higher when pred is high
  WHAT the policy does, one label per episode (furthest stage reached):
    success > lift > drag > near > wander > idle
  plus: peg moved toward/away from the goal, return before vs after first touching
  the peg, and how often the gripper is commanded closed away from the peg.

Writes timeline.json and timeline.png into the run folder.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import torch
from stable_baselines3 import SAC

import core.config as config
import core.results as results
from core.config import SUCCESS_KEY, HAND_POS_IDX, PEG_POS_IDX, GOAL_POS_IDX
from core.env_utils import make_fixed_scene_env
from stage_4_rl_train.reward_fn import step_reward
from stage_2_t2s_training.t2s_model import load_t2s_predictor

NEAR, MOVED, TOUCH, LIFT_Z, HELD_STEPS, IDLE, DIRECTION = 0.15, 0.02, 0.005, 0.01, 10, 0.05, 0.02
LABELS = ("success", "lift", "drag", "near", "wander", "idle")
TORCH_SEED_STRIDE = 100_000


def rollout(model, fn, env_seed, torch_seed, rw):
    env = make_fixed_scene_env()
    torch.manual_seed(torch_seed)
    obs, _ = env.reset(seed=env_seed)
    O, A, P, R, ss = [np.asarray(obs, np.float64)], [], [fn(obs)], [], None
    for t in range(config.MAX_STEPS):
        a, _ = model.predict(obs, deterministic=False)
        obs, _r, term, trunc, info = env.step(a)
        success = bool(info.get(SUCCESS_KEY, 0))
        p = fn(obs)
        R.append(step_reward(P[-1], 0.0 if success else p, **rw))
        O.append(np.asarray(obs, np.float64)); A.append(np.asarray(a, np.float64)); P.append(p)
        if success:
            ss = t + 1
        if success or term or trunc:
            break
    env.close()
    return np.array(O), np.array(A), np.array(P), np.array(R), ss


def summarize(O, A, P, R, ss, reward_mode, gamma):
    hand, peg, goal = O[:, HAND_POS_IDX], O[:, PEG_POS_IDX], O[:, GOAL_POS_IDX]
    hand_peg = np.linalg.norm(hand - peg, axis=1)
    peg_goal = np.linalg.norm(peg - goal, axis=1)
    drift = np.linalg.norm(peg - peg[0], axis=1)
    lift = peg[:, 2] - peg[0, 2]
    run, held = 0, False
    for up in lift > LIFT_Z:
        run = run + 1 if up else 0
        held = held or run >= HELD_STEPS
    moved = drift.max() > MOVED
    label = ("success" if ss is not None else "lift" if held else "drag" if moved else
             "near" if hand_peg.min() < NEAR else
             "wander" if np.linalg.norm(hand - hand[0], axis=1).max() > IDLE else "idle")
    touch = np.flatnonzero(drift > TOUCH)
    k = touch[0] if len(touch) else len(R)
    ret = float(R.sum())
    start = None
    if reward_mode in ("difference", "difference_timed"):
        start = float(P[0] - gamma * (0.0 if ss is not None else P[-1]))
    closing = A[:, 3] > 0
    d_goal = peg_goal[-1] - peg_goal[0]
    return dict(
        label=label, success=ss is not None, ret=ret, start=start,
        hover=None if start is None else ret - start,
        pred_start=float(P[0]), pred_end=float(0.0 if ss is not None else P[-1]),
        pred_min=float(P.min()), ret_before_touch=float(R[:k].sum()),
        ret_after_touch=float(R[k:].sum()),
        direction=("toward" if d_goal < -DIRECTION else "away" if d_goal > DIRECTION else "none")
        if moved else "none",
        close_frac=float(closing.mean()),
        close_far_frac=float((closing & (hand_peg[1:] > NEAR)).mean()),
        steps=len(R))


def tb_curve(run_dir, tag="rollout/ep_rew_mean"):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}
    out = {}
    for d in sorted(glob.glob(os.path.join(run_dir, "tb", "train*"))):
        ea = EventAccumulator(d)
        ea.Reload()
        if tag in ea.Tags().get("scalars", []):
            out.update({e.step: e.value for e in ea.Scalars(tag)})
    return out


def find_checkpoints(run_dir, wanted, every, final_step):
    """[(step, filename, display_name)], numbered checkpoints plus optional final."""
    found = sorted((int(m.group(1)), f) for f in os.listdir(run_dir)
                   if (m := re.fullmatch(r"policy_(\d+)\.zip", f)))
    out = [(s, f, f"{s:,}") for s, f in found]
    if os.path.exists(os.path.join(run_dir, "policy_final.zip")):
        out.append((final_step, "policy_final.zip", "final"))
    if wanted:
        keep = set(w.lower() for w in wanted)
        sel = [c for c in out if str(c[0]) in keep or (c[2] == "final" and "final" in keep)]
        missing = keep - {str(c[0]) for c in sel} - ({"final"} if any(
            c[2] == "final" for c in sel) else set())
        if missing:
            print(f"  (not found, skipped: {sorted(missing)}; available: "
                  f"{[c[2] for c in out]})")
        return sel
    return out[::every]


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--policy-run", required=True)
    p.add_argument("--t2s-run", required=True, help="the T2S run this policy was trained on")
    p.add_argument("--model", required=True)
    p.add_argument("--reward-mode", default=None, help="default: from the run's metadata")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="evaluation seeds (env reset seed + base of the sampling seeds)")
    p.add_argument("--episodes", type=int, default=30, help="rollouts PER SEED per checkpoint")
    p.add_argument("--ckpts", nargs="*", default=None,
                   help="only these checkpoints: step numbers and/or 'final' (default: all)")
    p.add_argument("--every", type=int, default=1,
                   help="use every N-th checkpoint (ignored with --ckpts)")
    args = p.parse_args()

    run_dir = results.get_run_dir("policy", args.policy_run)
    meta = json.load(open(os.path.join(config.TRAIN_RES_DIR, "runs_index.json"))
                     )["policy"][args.policy_run].get("meta", {})
    reward_mode = args.reward_mode or meta.get("reward_mode", "difference")
    gamma = meta.get("gamma", config.RL_GAMMA)
    rw = dict(reward_mode=reward_mode, time_penalty=config.TIME_PENALTY, gamma=gamma)

    t2s_dir = results.get_run_dir("t2s_model", args.t2s_run)
    man = json.load(open(os.path.join(t2s_dir, "manifest.json")))
    fn = load_t2s_predictor(t2s_dir, *args.model.rsplit("_", 1),
                            seed=man["combos"][args.model].get("best_seed", 0))

    tb = tb_curve(run_dir)
    final_step = int(meta.get("timesteps") or (max(tb) if tb else 10**9))
    ckpts = find_checkpoints(run_dir, args.ckpts, args.every, final_step)
    n_total = len(args.seeds) * args.episodes
    print(f"{args.policy_run}: {len(ckpts)} checkpoint(s), reward={reward_mode}, gamma={gamma}, "
          f"{len(args.seeds)} seeds x {args.episodes} episodes = {n_total} rollouts each"
          + ("" if tb else "  (no TensorBoard curve found)"))

    rows = []
    for step, f, name in ckpts:
        model = SAC.load(os.path.join(run_dir, f), device="cpu")
        eps, succ_by_seed = [], []
        for sd in args.seeds:
            mine = [summarize(*rollout(model, fn, sd, sd * TORCH_SEED_STRIDE + i, rw),
                              reward_mode, gamma)
                    for i in range(args.episodes)]
            succ_by_seed.append(float(np.mean([e["success"] for e in mine])))
            eps += mine
        mean = lambda k: (lambda v: float(np.mean(v)) if v else None)(
            [e[k] for e in eps if e[k] is not None])
        tb_near = tb[min(tb, key=lambda s: abs(s - step))] if tb else None
        rows.append(dict(step=step, name=name, tb_ep_rew=tb_near,
                         n_episodes=len(eps), seeds=args.seeds,
                         success_by_seed=succ_by_seed,
                         success_mean=float(np.mean(succ_by_seed)),
                         success_std=float(np.std(succ_by_seed)),
                         **{k: mean(k) for k in ("ret", "start", "hover", "pred_start", "pred_end",
                                                 "ret_before_touch", "ret_after_touch",
                                                 "close_frac", "close_far_frac")},
                         labels={l: sum(e["label"] == l for e in eps) for l in LABELS},
                         toward=sum(e["direction"] == "toward" for e in eps),
                         away=sum(e["direction"] == "away" for e in eps)))
        r = rows[-1]
        tb_txt = "--" if tb_near is None else f"{tb_near:.1f}"
        start_txt = "--" if r["start"] is None else f"{r['start']:.1f}"
        hover_txt = "--" if r["hover"] is None else f"{r['hover']:.1f}"
        per_seed = " ".join(f"{v:.2f}" for v in succ_by_seed)
        labels_txt = " ".join(f"{l}:{n}" for l, n in r["labels"].items() if n)
        print(f"  {name:>9}  success {r['success_mean']:.2f} +- {r['success_std']:.2f} "
              f"(per seed {per_seed})")
        print(f"             tb {tb_txt:>6}  ret {r['ret']:7.1f}  start {start_txt:>6}"
              f"  hover {hover_txt:>6}  pred {r['pred_start']:5.0f}->{r['pred_end']:5.0f}"
              f"  pre/post-touch {r['ret_before_touch']:6.1f}/{r['ret_after_touch']:6.1f}"
              f"  close-far {r['close_far_frac']:.2f}")
        print(f"             {labels_txt}  toward/away {r['toward']}/{r['away']}")

    json.dump(rows, open(os.path.join(run_dir, "timeline.json"), "w"), indent=2)
    plot(rows, tb, os.path.join(run_dir, "timeline.png"), args.policy_run)
    print(f"\nwrote {run_dir}/timeline.json and timeline.png")


def plot(rows, tb, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [r["step"] for r in rows]
    fig, ax = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

    sm = np.array([r["success_mean"] for r in rows])
    ss = np.array([r["success_std"] for r in rows])
    ax[0].errorbar(x, sm, yerr=ss, fmt="o-", color="tab:green", capsize=3,
                   label=f"success (mean +- std over {len(rows[0]['seeds'])} seeds)"
                   if rows else "success")
    ax[0].set_ylim(-0.05, 1.05); ax[0].set_ylabel("success rate")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    if tb:
        s = sorted(tb)
        ax[1].plot(s, [tb[k] for k in s], color="0.6", lw=1, label="TensorBoard ep_rew_mean")
    ax[1].plot(x, [r["ret"] for r in rows], "o-", color="black", label="eval return")
    if rows and rows[0]["start"] is not None:
        ax[1].plot(x, [r["start"] for r in rows], "s-", label="start term (progress)")
        ax[1].plot(x, [r["hover"] for r in rows], "^-", label="hover term (time in high-pred states)")
    ax[1].axhline(0, color="0.8", lw=0.8)
    ax[1].set_ylabel("return"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    # label shares as FRACTIONS so different episode counts stay comparable
    bottom = np.zeros(len(rows))
    colors = dict(success="tab:green", lift="tab:blue", drag="tab:orange", near="tab:purple",
                  wander="0.6", idle="0.85")
    width = (min(np.diff(sorted(x))) * 0.8) if len(x) > 1 else 1
    for l in LABELS:
        v = np.array([r["labels"][l] / r["n_episodes"] for r in rows], float)
        ax[2].bar(x, v, bottom=bottom, width=width, color=colors[l], label=l)
        bottom += v
    ax[2].set_ylim(0, 1); ax[2].set_ylabel("share of episodes")
    ax[2].legend(fontsize=8, ncol=6); ax[2].grid(alpha=0.3)

    ax[3].plot(x, [r["pred_end"] for r in rows], "o-", label="pred at end")
    ax[3].plot(x, [r["pred_start"] for r in rows], "--", color="0.5", label="pred at start")
    ax2 = ax[3].twinx()
    ax2.plot(x, [r["close_far_frac"] for r in rows], "x-", color="tab:red",
             label="gripper closed away from peg")
    ax2.set_ylim(0, 1); ax2.set_ylabel("fraction of steps", color="tab:red")
    ax[3].set_ylabel("T2S prediction"); ax[3].set_xlabel("training step")
    ax[3].legend(fontsize=8, loc="upper left"); ax2.legend(fontsize=8, loc="upper right")
    ax[3].grid(alpha=0.3)
    ax[3].set_xticks(x)
    ax[3].set_xticklabels([r["name"] for r in rows], rotation=45, ha="right", fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()