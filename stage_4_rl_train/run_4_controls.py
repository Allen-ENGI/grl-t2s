#!/usr/bin/env python3
"""
Stage 4 baselines: train SAC with the SAME settings as run_4_policy.py, but with a
control predictor instead of a trained T2S model.

    python run_4_controls.py --control const   --seeds 0 1 2 3 4
    python run_4_controls.py --control randnet --seeds 0 1 2 3 4 --dataset v4
    python run_4_controls.py --control subgoal --seeds 0 1 2 3 4 \
        --zero-model train_res/t2s_model/v9 tdlambdaboot_all softplus

  const    prediction fixed at 1: no shaping at all. Reward = -1 per step until the
           success flag ends the episode (sparse-reward baseline).
  randnet  a randomly initialised, frozen T2S network, rescaled to [0, --rand-max]:
           state-dependent shaping with no meaning (random-potential baseline).
  subgoal  --subgoal-c until the given model's prediction reaches 0, then 0
           (optional: "reach the peg" subgoal with nothing else).

Run names: <sweep>_<CONTROL>_seed<N>, e.g. ctrl_CONST_seed0. Defaults match
run_4_policy.py (difference_timed, config gamma / time penalty, raw reward,
1M steps, eval every 100k, checkpoint every 50k). Pass the same overrides you
passed to run_4_policy.py, or the runs are not comparable.
"""
import argparse
import os

import numpy as np

import core.config as config
import core.results as results
import stage_4_rl_train.policy_train as policy_train

EVAL_FREQ, CKPT_FREQ, N_EVAL_EPISODES = 100_000, 50_000, 5      # as in run_4_policy.py


def random_net_predictor(dataset, rand_max, seed, n_probe=5000):
    import torch
    from stage_2_t2s_training.t2s_model import T2SModel
    path = dataset if os.path.isfile(dataset) else os.path.join(
        results.get_run_dir("data_collection", dataset), "dataset.npz")
    X = np.load(path)["X"].astype(np.float32)
    probe = X[np.random.default_rng(seed).choice(len(X), min(n_probe, len(X)), replace=False)]
    mu, sd = probe.mean(0), probe.std(0) + 1e-6
    torch.manual_seed(1_000 + seed)
    try:
        net = T2SModel(obs_dim=config.OBS_DIM, head="linear").eval()
    except TypeError:
        net = T2SModel(obs_dim=config.OBS_DIM).eval()
    with torch.no_grad():
        raw = net(torch.as_tensor((probe - mu) / sd)).numpy()
    lo, hi = np.percentile(raw, 1), np.percentile(raw, 99)

    @torch.no_grad()
    def predict(obs):
        z = net(torch.as_tensor(((np.asarray(obs, np.float32) - mu) / sd)[None]))[0].item()
        return float(np.clip(rand_max * (z - lo) / (hi - lo), 0.0, rand_max))
    return predict


def subgoal_predictor(spec, c, eps):
    import json
    from stage_2_t2s_training.t2s_model import load_t2s_predictor
    run_dir, combo, head = spec
    seed = json.load(open(os.path.join(run_dir, "manifest.json")))["combos"][combo].get("best_seed", 0)
    m, cond = combo.rsplit("_", 1)
    try:
        Z = load_t2s_predictor(run_dir, m, cond, seed=seed, head=head)
    except TypeError:
        Z = load_t2s_predictor(run_dir, m, cond, seed=seed)
    return lambda obs: c if Z(obs) > eps else 0.0


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--control", required=True, choices=["const", "randnet", "subgoal"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--sweep-name", default="ctrl")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--gamma", type=float, default=config.RL_GAMMA)
    p.add_argument("--time-penalty", type=float, default=config.TIME_PENALTY)
    p.add_argument("--reward-mode", default="difference_timed")
    p.add_argument("--eval-freq", type=int, default=EVAL_FREQ,
                   help="steps between training evals (run_4_policy.py uses 100k)")
    p.add_argument("--n-eval-episodes", type=int, default=N_EVAL_EPISODES)
    p.add_argument("--const", type=float, default=1.0)
    p.add_argument("--dataset", default="v4", help="randnet: data run or dataset.npz")
    p.add_argument("--rand-max", type=float, default=80.0,
                   help="randnet output range, ~ the T2S models' start-of-episode value")
    p.add_argument("--zero-model", nargs=3, metavar=("RUN_DIR", "COMBO", "HEAD"))
    p.add_argument("--subgoal-c", type=float, default=60.0)
    p.add_argument("--zero-eps", type=float, default=0.5)
    a = p.parse_args()

    label = a.control.upper()
    for sd in a.seeds:
        run = f"{a.sweep_name}_{label}_seed{sd}"
        try:
            rd = results.get_run_dir("policy", run)
            if os.path.exists(os.path.join(rd, "policy_final.zip")):
                print(f"{run}: already finished, skipping")
                continue
        except KeyError:
            pass
        if a.control == "const":
            pred = (lambda c: (lambda obs: c))(a.const)
        elif a.control == "randnet":
            pred = random_net_predictor(a.dataset, a.rand_max, seed=sd)   # new net per seed
        else:
            if not a.zero_model:
                p.error("--control subgoal needs --zero-model RUN_DIR COMBO HEAD")
            pred = subgoal_predictor(a.zero_model, a.subgoal_c, a.zero_eps)
        rd = results.new_run_dir("policy", run, meta=dict(
            sweep=a.sweep_name, label=label, control=a.control, rl_seed=sd,
            reward_mode=a.reward_mode, gamma=a.gamma, time_penalty=a.time_penalty,
            timesteps=a.timesteps, const=a.const, rand_max=a.rand_max,
            subgoal_c=a.subgoal_c, zero_model=a.zero_model))
        print(f"training {run}  ({a.timesteps:,} steps)")
        policy_train.train_policy(
            rd, pred, reward_mode=a.reward_mode, total_timesteps=a.timesteps, seed=sd,
            gamma=a.gamma, time_penalty=a.time_penalty, norm_reward=False,
            eval_freq=a.eval_freq, ckpt_freq=CKPT_FREQ, n_eval_episodes=a.n_eval_episodes)


if __name__ == "__main__":
    main()