"""
Stage 1: Expert policy pretraining.

Ported from PolicyTrain.ipynb. Domain-randomized on purpose: each parallel
env samples a DIFFERENT task instance (seed=rank), unlike every downstream
stage which pins to one env_utils.make_fixed_scene_env(). Do not swap this
for the fixed-scene env — an expert that only ever saw one goal placement
would be a weaker/overfit source of demonstrations for Stage 2's data
collection.
"""
import os

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from config import TASK_SLUG
from env_utils import make_env
from rl_common import train_sac


def train_expert_policy(run_dir, total_timesteps=3_000_000, n_envs=4,
                         eval_freq=10_000, ckpt_freq=100_000, n_eval_episodes=10,
                         smoke_test_steps=5_000, seed=0):
    """
    Trains a domain-randomized SAC expert against the environment's native
    reward. Writes checkpoints as f"{TASK_SLUG}_<step>.zip" plus
    f"{TASK_SLUG}_final.zip" and eval_history.json into run_dir.
    """
    def env_fn(rank):
        return lambda: make_env(seed=rank)

    train_env = VecMonitor(DummyVecEnv([env_fn(i) for i in range(n_envs)]))
    eval_env = make_env(seed=1000)  # held-out seed, never used by a training env

    model, history = train_sac(
        train_env, eval_env, run_dir, total_timesteps=total_timesteps,
        ckpt_prefix=TASK_SLUG, eval_freq=eval_freq, ckpt_freq=ckpt_freq,
        n_eval_episodes=n_eval_episodes, smoke_test_steps=smoke_test_steps, seed=seed,
    )
    return model, history
