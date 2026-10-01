#!/usr/bin/env python3
"""
Does the T2S reward prefer the helpful action?

    python probe_actions.py --t2s-run v6 --model tdlambdaboot_all \
        --policy-run abs_v1_tdlambdaboot_all_seed0 --reward-mode absolute

States are taken every --every steps from rollouts of the trained policy and
of the expert. At each state the sim is saved, then two scripted branches of
--k steps are run from it and scored with the T2S reward:

  reach   toward vs away from the peg                 (any state)
  grip    close+lift vs open+lift                     (hand within --near of the peg)
  carry   peg toward vs away from the goal, gripper closed   (peg held up)

win   = share of states where the helpful branch earns more T2S reward
geo   = share where the helpful branch really was better physically
        (sanity check that the scripted action did what it should)
"""
import argparse
import json
import os

import numpy as np
import torch
from stable_baselines3 import SAC

import config
import results
from config import SUCCESS_KEY, HAND_POS_IDX, PEG_POS_IDX, GOAL_POS_IDX
from env_utils import make_fixed_scene_env
from eval_policy import snapshot, restore
from reward_fn import step_reward
from t2s_model import load_t2s_predictor

HELD_Z = 0.01


def act(xyz, grip):
    return np.r_[np.asarray(xyz, dtype=np.float32), grip].astype(np.float32)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def branch(env, snap, obs, actions, fn, rw):
    """Run `actions` from the saved state; return (T2S reward sum, final obs)."""
    restore(env, snap)
    prev, total = fn(obs), 0.0
    for a in actions:
        obs, _r, term, trunc, info = env.step(a)
        success = bool(info.get(SUCCESS_KEY, 0))
        pred = fn(obs)
        total += step_reward(prev, 0.0 if success else pred, **rw)
        prev = pred
        if success or term or trunc:
            break
    return total, obs


def tests(obs, peg_z0, k, near):
    hand, peg, goal = obs[HAND_POS_IDX], obs[PEG_POS_IDX], obs[GOAL_POS_IDX]
    hand_peg = lambda o: np.linalg.norm(o[HAND_POS_IDX] - o[PEG_POS_IDX])
    peg_goal = lambda o: np.linalg.norm(o[PEG_POS_IDX] - o[GOAL_POS_IDX])
    lift = lambda o: o[PEG_POS_IDX][2] - peg_z0
    d = unit(peg - hand)
    out = [("reach", [act(d, -1)] * k, [act(-d, -1)] * k, lambda g, b: hand_peg(g) < hand_peg(b))]
    if np.linalg.norm(hand - peg) < near:
        up = [0, 0, 1]
        out.append(("grip", [act([0, 0, 0], 1)] * k + [act(up, 1)] * k,
                    [act([0, 0, 0], -1)] * k + [act(up, -1)] * k,
                    lambda g, b: lift(g) > lift(b)))
    if lift(obs) > HELD_Z:
        d = unit(goal - peg)
        out.append(("carry", [act(d, 1)] * k, [act(-d, 1)] * k,
                    lambda g, b: peg_goal(g) < peg_goal(b)))
    return out


def probe_policy(policy, fn, rw, episodes, every, k, near, seed0=2000):
    rows = []
    for e in range(episodes):
        env = make_fixed_scene_env()
        torch.manual_seed(seed0 + e)
        obs, _ = env.reset(seed=seed0 + e)
        peg_z0 = float(obs[PEG_POS_IDX][2])
        for t in range(config.MAX_STEPS):
            if t % every == 0:
                snap = snapshot(env)
                for name, good, bad, geo_ok in tests(obs, peg_z0, k, near):
                    r_good, o_good = branch(env, snap, obs, good, fn, rw)
                    r_bad, o_bad = branch(env, snap, obs, bad, fn, rw)
                    rows.append(dict(test=name, t=t, r_good=r_good, r_bad=r_bad,
                                     win=float(r_good > r_bad) + 0.5 * float(r_good == r_bad), geo=bool(geo_ok(o_good, o_bad))))
                restore(env, snap)
            action, _ = policy.predict(obs, deterministic=False)
            obs, _r, term, trunc, info = env.step(action)
            if info.get(SUCCESS_KEY, 0) or term or trunc:
                break
        env.close()
    return rows


def report(source, rows):
    print(f"\n{source}")
    print(f"  {'test':<7}{'states':>8}{'win':>7}{'geo':>7}{'win|geo':>9}{'mean R good-bad':>18}")
    for name in ("reach", "grip", "carry"):
        rs = [r for r in rows if r["test"] == name]
        if not rs:
            print(f"  {name:<7}{0:>8}   (no qualifying states)")
            continue
        g = [r for r in rs if r["geo"]]
        win_geo = f"{np.mean([r['win'] for r in g]):.2f}" if g else "--"
        print(f"  {name:<7}{len(rs):>8}{np.mean([r['win'] for r in rs]):>7.2f}"
              f"{np.mean([r['geo'] for r in rs]):>7.2f}{win_geo:>9}"
              f"{np.mean([r['r_good'] - r['r_bad'] for r in rs]):>+18.2f}")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--t2s-run", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--policy-run", required=True)
    p.add_argument("--checkpoint", default="policy_final.zip")
    p.add_argument("--reward-mode", default="absolute",
                   choices=["absolute", "difference", "difference_timed"])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--every", type=int, default=10, help="probe a state every N steps")
    p.add_argument("--k", type=int, default=5, help="steps per scripted branch")
    p.add_argument("--near", type=float, default=0.08, help="hand-peg distance for the grip test")
    args = p.parse_args()

    t2s_dir = results.get_run_dir("t2s_model", args.t2s_run)
    man = json.load(open(os.path.join(t2s_dir, "manifest.json")))
    fn = load_t2s_predictor(t2s_dir, *args.model.rsplit("_", 1),
                            seed=man["combos"][args.model].get("best_seed", 0))
    rw = dict(reward_mode=args.reward_mode, time_penalty=config.TIME_PENALTY, gamma=config.RL_GAMMA)

    stage3 = json.load(open(os.path.join(results.get_run_dir("t2s_eval", args.t2s_run),
                                         "stage_manifest.json")))
    sources = {
        f"trained policy ({args.policy_run}/{args.checkpoint})":
            SAC.load(os.path.join(results.get_run_dir("policy", args.policy_run), args.checkpoint),
                     device="cpu"),
        f"expert ({stage3['success_ckpt']})":
            SAC.load(os.path.join(config.EXPERT_POLICY_DIR, stage3["success_ckpt"]), device="cpu"),
    }
    print(f"model={args.model}  reward={args.reward_mode}  k={args.k}  every={args.every}")
    for source, policy in sources.items():
        report(source, probe_policy(policy, fn, rw, args.episodes, args.every, args.k, args.near))
    print("\nwin|geo is the number to read: among states where the helpful branch really was "
          "helpful,\nhow often the T2S reward paid it more. ~1.0 = reward points the right way; "
          "~0.5 = no signal.")


if __name__ == "__main__":
    main()
