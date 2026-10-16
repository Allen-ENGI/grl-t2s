#!/usr/bin/env python3
"""
Why does a WORSE T2S model give BETTER downstream RL?

    # 1) cheap: decompose rewards + on-policy predictor quality (no training)
    python diagnose_reward.py --eval-run v3 --models good_all bad_all \
        --sweep-name sweep_v6 --rl-seeds 0 1 2 --episodes 20

    # 2) decisive: train against CONSTANT predictors (pure success-bonus control)
    python diagnose_reward.py --eval-run v3 --models good_all bad_all \
        --train-controls --const 1 40 --timesteps 1000000

Reading the output:
  success_bonus   mean reward on the success step (= prev_pred - tp). This is the
                  env's success flag leaking through terminal_pred=0. Large for
                  models that over-predict near the goal.
  shaping_share   |shaping part| / |total return|. Small => T2S barely matters.
  spearman_ttg    rank corr(pred, true steps-to-go) on THIS policy's successful
                  episodes. What shaping needs, unlike MAE.
  sign_agree      fraction of steps where the shaping term is positive exactly when
                  the peg moved closer to the goal.
  hover@median    reward for standing still at the median prediction (must be < 0).
"""
import argparse
import json
import os

import numpy as np
from stable_baselines3 import SAC

import core.config as config
import core.results as results
from core.progress import peg_goal_distance
from stage_4_rl_train.policy_env import make_policy_train_env
from stage_4_rl_train.reward_fn import hover_reward
import stage_4_rl_train.policy_train as policy_train
from stage_4_rl_train.run_4_policy import _load_predictors, STAGE_MANIFEST


def spearman(a, b):
    if len(a) < 3:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def rollout(policy, predict, reward_mode, gamma, tp, seed, deterministic=False):
    env = make_policy_train_env(predict, reward_mode, gamma=gamma, time_penalty=tp)()
    obs, _ = env.reset(seed=seed)
    prev_pred = env._last_pred
    rec = dict(pred=[prev_pred], reward=[], shaping=[], dist=[peg_goal_distance(obs)],
               success_step=None, bonus=None)
    for t in range(500):
        action, _ = policy.predict(obs, deterministic=deterministic)
        obs, r, term, trunc, info = env.step(action)
        rec["reward"].append(r)
        rec["shaping"].append(r + (tp if reward_mode == "difference_timed" else 0.0))
        rec["pred"].append(info["t2s_pred"])
        rec["dist"].append(peg_goal_distance(obs))
        if info.get(config.SUCCESS_KEY, 0) and rec["success_step"] is None:
            rec["success_step"] = t + 1
            rec["bonus"] = r          # = prev_pred - tp : the leaked success signal
        if term or trunc:
            break
    env.close()
    return rec


def analyse(recs, reward_mode, gamma, tp):
    out = dict(n=len(recs), success_rate=np.mean([r["success_step"] is not None for r in recs]))
    bonuses = [r["bonus"] for r in recs if r["bonus"] is not None]
    out["success_bonus"] = float(np.mean(bonuses)) if bonuses else None
    tot = np.array([sum(r["reward"]) for r in recs])
    shp = np.array([sum(r["shaping"]) - (r["bonus"] + tp if r["bonus"] is not None else 0)
                    for r in recs])          # shaping without the terminal jump
    out["mean_return"] = float(tot.mean())
    out["shaping_share"] = float(np.mean(np.abs(shp)) / max(np.mean(np.abs(tot)), 1e-8))
    rhos, agree = [], []
    for r in recs:
        p, d = np.array(r["pred"]), np.array(r["dist"])
        if r["success_step"] is not None:
            k = r["success_step"]
            rhos.append(spearman(p[:k], (k - np.arange(k)).astype(float)))
        moved = np.diff(d)
        m = np.abs(moved) > 1e-4           # only steps where the peg actually moved
        if m.any():
            shaping_pos = (np.array(r["shaping"]) > 0)[: len(moved)]
            agree.append(np.mean(shaping_pos[m] == (moved[m] < 0)))
    rhos = [x for x in rhos if x is not None]
    out["spearman_ttg"] = float(np.mean(rhos)) if rhos else None
    out["sign_agree"] = float(np.mean(agree)) if agree else None
    allp = np.concatenate([r["pred"] for r in recs])
    out["pred_median"], out["pred_max"] = float(np.median(allp)), float(allp.max())
    out["hover@median"] = float(hover_reward(out["pred_median"], reward_mode=reward_mode,
                                             time_penalty=tp, gamma=gamma))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-run", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--sweep-name", default="sweep_v6")
    ap.add_argument("--rl-seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--reward-mode", default="difference_timed")
    ap.add_argument("--gamma", type=float, default=config.RL_GAMMA)
    ap.add_argument("--time-penalty", type=float, default=config.TIME_PENALTY)
    ap.add_argument("--const", type=float, nargs="*", default=[1.0],
                    help="constant-predictor controls")
    ap.add_argument("--train-controls", action="store_true")
    ap.add_argument("--zero-model", default=None,
                    help="the model that predicts 0 from mid-trajectory (one of --models)")
    ap.add_argument("--good-model", default=None, help="the better-MAE model (one of --models)")
    ap.add_argument("--zero-eps", type=float, default=0.5,
                    help="pred <= this counts as 'in the zero region'")
    ap.add_argument("--subgoal-c", type=float, default=50.0,
                    help="constant for the pure-subgoal control; use ~the zero model's "
                         "typical first-half prediction")
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    a = ap.parse_args()

    eval_dir = results.get_run_dir("t2s_eval", a.eval_run)
    with open(os.path.join(eval_dir, STAGE_MANIFEST)) as f:
        stage3 = json.load(f)
    _specs, predictors = _load_predictors(stage3["t2s_run_dir"], a.models, stage3)
    controls = {}
    for c in a.const:
        controls[f"CONST_{c:g}"] = (lambda c: (lambda obs: float(c)))(c)

    # splice controls: which half of the zero model's signal is doing the work?
    if a.zero_model and a.good_model:
        Z, G, eps = predictors[a.zero_model], predictors[a.good_model], a.zero_eps
        # good model's shape before the zero point, flat zero after it
        controls["SPLICE_good+zeroplateau"] = lambda obs: G(obs) if Z(obs) > eps else 0.0
    if a.zero_model:
        Z, eps = predictors[a.zero_model], a.zero_eps
        # pure subgoal: constant before the zero point, 0 after (no first-half shape)
        C = a.subgoal_c
        controls[f"SUBGOAL_{C:g}"] = lambda obs: C if Z(obs) > eps else 0.0
    predictors.update(controls)

    if a.train_controls:
        for cname, cpred in controls.items():
            for sd in a.rl_seeds:
                name = f"{a.sweep_name}_CONTROL_{cname}_seed{sd}"
                run_dir = results.new_run_dir("policy", name,
                                              meta=dict(control=cname, rl_seed=sd))
                print(f"training control {name}")
                policy_train.train_policy(
                    run_dir, cpred, reward_mode=a.reward_mode,
                    total_timesteps=a.timesteps, seed=sd, gamma=a.gamma,
                    time_penalty=a.time_penalty, norm_reward=False,
                    eval_freq=100_000, n_eval_episodes=20)
        return

    # policies to roll out: experts + every trained sweep policy found
    policies = {
        "expert_succ": SAC.load(os.path.join(config.EXPERT_POLICY_DIR, stage3["success_ckpt"])),
        "expert_fail": SAC.load(os.path.join(config.EXPERT_POLICY_DIR, stage3["failure_ckpt"])),
    }
    for m in a.models:
        for sd in a.rl_seeds:
            name = f"{a.sweep_name}_{m}_seed{sd}"
            try:
                rd = results.get_run_dir("policy", name)
                with open(os.path.join(rd, "meta.json")) as f:     # stale-run check
                    print(f"{name}: meta={json.load(f)}")
            except (KeyError, FileNotFoundError):
                pass
            try:
                policies[f"pol[{m}]s{sd}"] = SAC.load(os.path.join(rd, "policy_final.zip"))
            except Exception as e:
                print(f"skip {name}: {e}")

    rows = []
    for pname, pol in policies.items():
        for rname, pred in predictors.items():
            recs = [rollout(pol, pred, a.reward_mode, a.gamma, a.time_penalty, seed=20_000 + i)
                    for i in range(a.episodes)]
            row = dict(policy=pname, reward=rname,
                       **analyse(recs, a.reward_mode, a.gamma, a.time_penalty))
            rows.append(row)
            f = lambda v: "--" if v is None else f"{v:+.3f}"
            print(f"{pname:<22}{rname:<22} succ={row['success_rate']:.2f} "
                  f"bonus={f(row['success_bonus'])} shaping_share={row['shaping_share']:.2f} "
                  f"rho_ttg={f(row['spearman_ttg'])} sign_agree={f(row['sign_agree'])} "
                  f"hover@med={row['hover@median']:+.3f} pred[med/max]="
                  f"{row['pred_median']:.1f}/{row['pred_max']:.1f}")
    with open("diagnose_reward.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    print("\nwrote diagnose_reward.json")


if __name__ == "__main__":
    main()