#!/usr/bin/env python3
"""
Why didn't the policy learn what the reward asks for?

    python eval_policy.py --t2s-run v5 --model tdlambdaboot_all \
        --runs sweep_v7_tdlambdaboot_all_seed0 sweep_v7_tdlambdaboot_all_seed1
    python eval_policy.py ... --checkpoints policy_200000.zip policy_final.zip --probe 16

Per rollout it records T2S, the T2S reward, MetaWorld's reward and stages, and
SAC's own critic Q(s, a), then reports:

  stages          P(reach) P(grasp) P(lift) P(align) P(success). reach/grasp come from
                  geometry, not MetaWorld's near_object/grasp_reward, which only count a
                  grasp at the peg's designated grasp point (a tip grasp reads as 0).
                  grasp = peg held >1 cm up for 10+ steps; align = MetaWorld in_place > 0.5.
  t2s_ret         total T2S reward; compare with the do-nothing/random baselines
  neg_frac        share of steps the reward itself scores as bad
  wrong_paid      after the grasp (lift): steps the peg moves AWAY from the goal but reward > 0
  right_punished  after the grasp (lift): steps the peg moves TOWARD the goal but reward < 0
  q_rho           Spearman(critic Q, actual discounted return): did the critic learn the reward?
  probe           share of random actions the policy's action beats on the reward (needs --probe)
  sat             share of action dims pinned at +-1

Reading it:
  wrong_paid high                      the reward pays for the wrong direction  -> reward/model problem
  neg_frac high, probe low, q_rho low  reward says bad, critic never learned it -> RL problem
  neg_frac high, probe low, q_rho ok   critic knows, actor does not follow     -> actor/entropy problem
  stages differ a lot across seeds     exploration / variance, not the reward
  t2s_ret below the zero-action row    SAC did not maximise its own reward     -> RL problem
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr
from stable_baselines3 import SAC

import config
import results
from config import SUCCESS_KEY, HAND_POS_IDX, PEG_POS_IDX, GOAL_POS_IDX
from env_utils import make_fixed_scene_env
from reward_fn import step_reward
from t2s_model import load_t2s_predictor

STAGES = ("reach", "grasp", "lift", "align", "success")
dist = lambda o, a, b: float(np.linalg.norm(np.asarray(o)[a] - np.asarray(o)[b]))


HELD_Z, HELD_STEPS, REACH_DIST = 0.01, 10, 0.08


class Baseline:
    """Zero-action or uniform-random policy, for a reward floor."""
    def __init__(self, kind, space, seed=0):
        self.kind, self.space, self.rng = kind, space, np.random.default_rng(seed)

    def predict(self, obs, deterministic=False):
        if self.kind == "zero":
            return np.zeros(self.space.shape, dtype=np.float32), None
        return self.rng.uniform(self.space.low, self.space.high).astype(np.float32), None


def q_value(model, obs, action):
    if not hasattr(model, "critic"):
        return float("nan")
    with torch.no_grad():
        o = model.policy.obs_to_tensor(obs)[0]
        a = torch.as_tensor(model.policy.scale_action(action)[None], dtype=torch.float32,
                            device=model.device)
        return float(torch.min(torch.cat(model.critic(o, a), dim=1)).item())


def snapshot(env):
    u = env.unwrapped
    prev = getattr(u, "_prev_obs", None)
    return u.get_env_state(), u.curr_path_length, None if prev is None else prev.copy()


def restore(env, snap):
    u = env.unwrapped
    u.set_env_state(snap[0])
    u.curr_path_length = snap[1]
    if snap[2] is not None:
        u._prev_obs = snap[2].copy()


def probe(env, fn, action, pred, k, rng, rw):
    """Try the policy's action and k random actions from the same state; restore afterwards."""
    snap = snapshot(env)
    cands = [action] + [rng.uniform(env.action_space.low, env.action_space.high) for _ in range(k)]
    rewards, first_obs = [], None
    for a in cands:
        restore(env, snap)
        o2, _r, _te, _tr, info = env.step(a)
        first_obs = o2 if first_obs is None else first_obs
        rewards.append(step_reward(pred, 0.0 if info.get(SUCCESS_KEY, 0) else fn(o2), **rw))
    restore(env, snap)
    return float(np.mean(np.array(rewards[1:]) < rewards[0])), first_obs


def rollout(model, fn, seed, rw, deterministic=False, n_probe=0, probe_every=10):
    env = make_fixed_scene_env()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    pred, rows, ss = fn(obs), [], None
    peg_z0 = float(obs[PEG_POS_IDX][2])
    for t in range(config.MAX_STEPS):
        action, _ = model.predict(obs, deterministic=deterministic)
        row = dict(t=t, pred=pred, q=q_value(model, obs, action),
                   sat=float(np.mean(np.abs(action) > 0.98)))
        probe_obs = None
        if n_probe and t % probe_every == 0:
            row["probe"], probe_obs = probe(env, fn, action, pred, n_probe, rng, rw)
        obs2, mw_r, term, trunc, info = env.step(action)
        if probe_obs is not None:
            row["probe_ok"] = bool(np.allclose(probe_obs, obs2, atol=1e-6))
        success = bool(info.get(SUCCESS_KEY, 0))
        pred2 = fn(obs2)
        row.update(r=step_reward(pred, 0.0 if success else pred2, **rw), mw=float(mw_r),
                   near=float(info.get("near_object", np.nan)),
                   grasp=float(info.get("grasp_reward", np.nan)),
                   in_place=float(info.get("in_place_reward", np.nan)),
                   peg_goal=dist(obs2, PEG_POS_IDX, GOAL_POS_IDX),
                   hand_peg=dist(obs2, HAND_POS_IDX, PEG_POS_IDX),
                   lift=float(obs2[PEG_POS_IDX][2]) - peg_z0)
        rows.append(row)
        obs, pred = obs2, pred2
        if success:
            ss = t + 1
        if success or term or trunc:           # the training wrapper ends at success
            break
    env.close()
    return rows, ss


def held_mask(lift):
    """True where the peg has been >HELD_Z up for at least HELD_STEPS consecutive steps."""
    up = np.asarray(lift) > HELD_Z
    out, run = np.zeros(len(up), dtype=bool), 0
    for i, u in enumerate(up):
        run = run + 1 if u else 0
        if run >= HELD_STEPS:
            out[i - HELD_STEPS + 1:i + 1] = True
    return out


def summarize(rows, ss, gamma):
    col = lambda k: np.array([r[k] for r in rows], dtype=float)
    r, q = col("r"), col("q")
    g = np.zeros(len(r))
    for i in range(len(r) - 1, -1, -1):        # discounted return actually received
        g[i] = r[i] + (gamma * g[i + 1] if i + 1 < len(r) else 0.0)
    held = held_mask(col("lift"))
    grasped = np.flatnonzero(held)
    after = np.arange(grasped[0] + 1, len(rows)) if len(grasped) else np.array([], int)
    d_goal = np.diff(np.r_[rows[0]["peg_goal"], col("peg_goal")])
    probes = [x["probe"] for x in rows if "probe" in x]
    return dict(
        reach=bool(col("hand_peg").min() < REACH_DIST or len(grasped)), grasp=bool(len(grasped)),
        lift=bool(col("lift").max() > 0.01), align=bool(np.nanmax(col("in_place")) > 0.5),
        success=ss is not None, steps=len(rows),
        t2s_return=float(r.sum()), mw_return=float(col("mw").sum()),
        neg_frac=float(np.mean(r < 0)),
        wrong_paid=float(np.mean((d_goal[after] > 1e-3) & (r[after] > 0))) if len(after) else None,
        right_punished=float(np.mean((d_goal[after] < -1e-3) & (r[after] < 0))) if len(after) else None,
        q_rho=(float(spearmanr(q, g)[0]) if len(r) >= 10 and np.all(np.isfinite(q))
               and np.std(q) > 0 and np.std(g) > 0 else None),
        probe=float(np.mean(probes)) if probes else None,
        probe_replay_ok=all(x.get("probe_ok", True) for x in rows),
        sat=float(col("sat").mean()),
    )


def print_trace(rows, every=5):
    prev = rows[0]["peg_goal"]
    held = held_mask([x["lift"] for x in rows])
    print(f"{'step':>5}{'T2S':>8}{'R':>8}{'MW_R':>7}{'lift':>7}{'place':>7}"
          f"{'hand->peg':>11}{'peg->goal':>11}{'Q':>9}")
    for x, h in zip(rows, held):
        away = h and x["peg_goal"] - prev > 1e-3
        flag = ("  <-- paid, peg moving away" if away and x["r"] > 0 else
                "  <-- peg moving away" if away else "")
        if x["t"] % every == 0 or flag:
            print(f"{x['t']:>5}{x['pred']:>8.1f}{x['r']:>+8.2f}{x['mw']:>7.2f}{x['lift']:>7.3f}"
                  f"{x['in_place']:>7.2f}{x['hand_peg']:>11.4f}{x['peg_goal']:>11.4f}{x['q']:>9.2f}{flag}")
        prev = x["peg_goal"]


def main():
    p = argparse.ArgumentParser(description="Evaluate trained policies against the T2S reward",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--t2s-run", required=True)
    p.add_argument("--model", required=True, help="the T2S combo the policies were trained on")
    p.add_argument("--runs", nargs="+", required=True, help="policy run names (e.g. one per seed)")
    p.add_argument("--checkpoints", nargs="+", default=["policy_final.zip"])
    p.add_argument("--rollouts", type=int, default=10, help="rollouts per checkpoint")
    p.add_argument("--reward-mode", default="difference",
                   choices=["absolute", "difference", "difference_timed"])
    p.add_argument("--deterministic", action="store_true", help="mean action instead of sampling")
    p.add_argument("--probe", type=int, default=0, help="random actions tried per probed state")
    p.add_argument("--trace-every", type=int, default=5)
    p.add_argument("--no-baselines", dest="baselines", action="store_false", default=True,
                   help="skip the zero-action and random-action reference rows")
    p.add_argument("--no-trace", dest="trace", action="store_false", default=True)
    args = p.parse_args()

    t2s_dir = results.get_run_dir("t2s_model", args.t2s_run)
    man = json.load(open(os.path.join(t2s_dir, "manifest.json")))
    fn = load_t2s_predictor(t2s_dir, *args.model.rsplit("_", 1),
                            seed=man["combos"][args.model].get("best_seed", 0))
    rw = dict(reward_mode=args.reward_mode, time_penalty=config.TIME_PENALTY, gamma=config.RL_GAMMA)

    jobs = [(run, ck) for run in args.runs for ck in args.checkpoints]
    if args.baselines:
        jobs += [("baseline", "zero-action"), ("baseline", "random")]
    table, traced = [], False
    for run, ck in jobs:
        if run == "baseline":
            env = make_fixed_scene_env()
            model, run_dir = Baseline(ck.split("-")[0], env.action_space), None
            env.close()
        else:
            run_dir = results.get_run_dir("policy", run)
            path = os.path.join(run_dir, ck)
            if not os.path.exists(path):
                print(f"  {run}/{ck}: missing")
                continue
            model = SAC.load(path, device="cpu")
        summ = []
        for i in range(args.rollouts):
            rows, ss = rollout(model, fn, seed=1000 + i, rw=rw,
                               deterministic=args.deterministic,
                               n_probe=0 if run == "baseline" else args.probe)
            if args.trace and not traced and run != "baseline":
                print(f"\n=== trace: {run}/{ck}, rollout 0 "
                      f"({'success @ ' + str(ss) if ss else 'no success'}) ===")
                print_trace(rows, args.trace_every)
                traced = True
            summ.append(summarize(rows, ss, config.RL_GAMMA))
        if run_dir:
            json.dump(summ, open(os.path.join(run_dir, f"policy_eval_{ck[:-4]}.json"), "w"),
                      indent=2)
        mean = lambda k: (lambda v: float(np.mean(v)) if v else None)(
            [s[k] for s in summ if s[k] is not None])
        table.append(dict(run=run, ckpt=ck,
                          replay_ok=all(s["probe_replay_ok"] for s in summ),
                          **{k: mean(k) for k in STAGES + (
                              "t2s_return", "neg_frac", "wrong_paid", "right_punished",
                              "q_rho", "probe", "sat")}))

    f = lambda v: "--" if v is None else (f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}")
    keys = STAGES + ("t2s_return", "neg_frac", "wrong_paid", "right_punished", "q_rho",
                     "probe", "sat")
    print(f"\n{'run / checkpoint':<48}" + "".join(f"{k[:9]:>10}" for k in keys))
    for t in table:
        print(f"{(t['run'] + '/' + t['ckpt'])[-47:]:<48}" + "".join(f"{f(t[k]):>10}" for k in keys)
              + ("" if t["replay_ok"] else "   (probe restore mismatch)"))
    print("\nstage columns are rates over rollouts; others are means. See the docstring for how to read them.")


if __name__ == "__main__":
    main()