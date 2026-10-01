#!/usr/bin/env python3
"""
Continue a downstream SAC run from one of its checkpoints.

    python resume_policy.py --policy-run diff_v3_tdlambdaboot_all_seed0 \
        --t2s-run v6 --model tdlambdaboot_all --extra-steps 400000
    python resume_policy.py ... --checkpoint policy_400000.zip

Keeps the run's step count, eval_history.json, TensorBoard curve and
VecNormalize statistics, so the continued part lines up with the original.
The replay buffer was not saved by the original run, so it restarts empty:
expect a short dip right after resuming (see --warmup).
"""
import argparse
import json
import os

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

import config
import results
from policy_env import make_policy_train_env
from rl_common import SuccessRateCallback
from t2s_model import load_t2s_predictor


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--policy-run", required=True)
    p.add_argument("--t2s-run", required=True, help="the T2S run the policy was trained on")
    p.add_argument("--model", required=True, help="the T2S combo, e.g. tdlambdaboot_all")
    p.add_argument("--checkpoint", default="policy_final.zip")
    p.add_argument("--extra-steps", type=int, default=400_000)
    p.add_argument("--reward-mode", default=None, help="default: from the run's metadata")
    p.add_argument("--n-envs", type=int, default=6)
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--ckpt-freq", type=int, default=10_000)
    p.add_argument("--n-eval-episodes", type=int, default=5)
    p.add_argument("--warmup", type=int, default=0,
                   help="steps to refill the empty replay buffer before updating "
                        "(SB3 takes random actions during these)")
    args = p.parse_args()

    run_dir = results.get_run_dir("policy", args.policy_run)
    index = json.load(open(os.path.join(config.TRAIN_RES_DIR, "runs_index.json")))
    meta = index["policy"][args.policy_run].get("meta", {})
    reward_mode = args.reward_mode or meta.get("reward_mode", "difference")
    gamma = meta.get("gamma", config.RL_GAMMA)

    t2s_dir = results.get_run_dir("t2s_model", args.t2s_run)
    man = json.load(open(os.path.join(t2s_dir, "manifest.json")))
    fn = load_t2s_predictor(t2s_dir, *args.model.rsplit("_", 1),
                            seed=man["combos"][args.model].get("best_seed", 0))

    make = make_policy_train_env(fn, reward_mode, gamma=gamma)
    venv = VecMonitor(DummyVecEnv([make for _ in range(args.n_envs)]))
    vn_path = os.path.join(run_dir, "vecnormalize.pkl")
    if os.path.exists(vn_path):
        venv = VecNormalize.load(vn_path, venv)
        venv.training = True                      # keep updating the statistics

    ckpt = os.path.join(run_dir, args.checkpoint)
    model = SAC.load(ckpt, env=venv)
    start = model.num_timesteps
    if args.warmup:
        model.learning_starts = start + args.warmup
    print(f"resuming {args.policy_run}/{args.checkpoint} at step {start:,}  "
          f"reward={reward_mode} gamma={gamma}  gradient_steps={model.gradient_steps}  "
          f"n_envs={args.n_envs}  vecnormalize={'yes' if os.path.exists(vn_path) else 'no'}")

    cb = SuccessRateCallback(make(), run_dir, eval_freq=args.eval_freq,
                             n_eval_episodes=args.n_eval_episodes, ckpt_freq=args.ckpt_freq,
                             ckpt_prefix="policy", verbose=1)
    hist_path = os.path.join(run_dir, "eval_history.json")
    if os.path.exists(hist_path):                 # keep the earlier evals, drop any past the resume point
        cb.history = [h for h in json.load(open(hist_path)) if h["step"] <= start]

    model.learn(total_timesteps=args.extra_steps, callback=cb, reset_num_timesteps=False,
                tb_log_name="train")

    model.save(os.path.join(run_dir, "policy_final_resumed"))
    model.save_replay_buffer(os.path.join(run_dir, "replay_buffer_resumed"))
    if isinstance(venv, VecNormalize):
        venv.save(vn_path)
    print(f"done at step {model.num_timesteps:,} -> policy_final_resumed.zip "
          f"(policy_final.zip left untouched)")


if __name__ == "__main__":
    main()