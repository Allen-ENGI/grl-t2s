"""
Stage 5 (part 2): downstream RL training against the frozen T2S reward.

Everything that decides *where output goes* is delegated to results.py —
this module takes a run_dir and writes into it, it never invents its own path.

The actual SAC loop + success-rate callback now live in rl_common.py, shared
with expert_train.py (Stage 1) — see rl_common's docstring for why.
"""
import os

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from policy_env import make_policy_train_env
from rl_common import train_sac


def train_policy(run_dir, predict_t2s, reward_mode="absolute", total_timesteps=1_000_000,
                  n_envs=6, eval_freq=10_000, ckpt_freq=50_000, n_eval_episodes=5,
                  seed=0, smoke_test_steps=3_000):
    """
    Trains SAC against the T2S reward. `predict_t2s` should come from
    t2s_predict.load_t2s_predictor so the frozen model / normalization
    match exactly what Stage 3/4 produced.
    """
    train_env = VecMonitor(DummyVecEnv(
        [make_policy_train_env(predict_t2s, reward_mode) for _ in range(n_envs)]))
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    eval_env = make_policy_train_env(predict_t2s, reward_mode)()  # single unwrapped-vec env for eval

    model, history = train_sac(
        train_env, eval_env, run_dir, total_timesteps=total_timesteps, ckpt_prefix="policy",
        eval_freq=eval_freq, ckpt_freq=ckpt_freq, n_eval_episodes=n_eval_episodes,
        smoke_test_steps=smoke_test_steps, seed=seed,
        sac_kwargs=dict(buffer_size=300_000),
    )
    train_env.save(os.path.join(run_dir, "vecnormalize.pkl"))
    return model, history

