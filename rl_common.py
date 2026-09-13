"""
Shared SAC training machinery, used by BOTH:
  - expert_train.py   (Stage 1: pretrain against the environment's own reward)
  - policy_train.py   (Stage 5: train against the frozen T2S reward)

These two notebooks in the original project each had their own copy of a
"checkpoint on a schedule + eval success-rate on a schedule + log history"
callback that were 95% identical (SuccessAndTimeToSuccessCallback in
PolicyTrain.ipynb vs SuccessCallback in the earlier policy-training
notebook). One copy here means a fix to eval logic never needs to be
applied twice.
"""
import json
import os
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from config import SUCCESS_KEY


class SuccessRateCallback(BaseCallback):
    """
    Periodically: (a) checkpoints the model, (b) runs `n_eval_episodes` on
    `eval_env` and records success rate + mean time-to-success against the
    REAL success flag (not any shaped/predicted reward), appending to
    `eval_history.json` in ckpt_dir.

    `eval_env` must be a single (non-vectorized) gym.Env exposing `.reset()`
    / `.step()` and `info["success"]`.
    """

    def __init__(self, eval_env, ckpt_dir, eval_freq=10_000, n_eval_episodes=10,
                 ckpt_freq=100_000, max_ep_steps=500, ckpt_prefix="policy", verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.ckpt_dir = ckpt_dir
        self.ckpt_freq = ckpt_freq
        self.max_ep_steps = max_ep_steps
        self.ckpt_prefix = ckpt_prefix
        os.makedirs(ckpt_dir, exist_ok=True)
        self.history = []

    def _run_eval_episode(self):
        obs, _ = self.eval_env.reset()
        success_step = None
        for t in range(self.max_ep_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = self.eval_env.step(action)
            if info.get(SUCCESS_KEY, 0) and success_step is None:
                success_step = t
            if terminated or truncated:
                break
        return success_step

    def _on_step(self) -> bool:
        # Use num_timesteps (total ENVIRONMENT steps across all parallel envs),
        # not n_calls (number of _on_step invocations). With n_envs>1 these
        # differ by a factor of n_envs, which would otherwise make eval_freq /
        # ckpt_freq mean different things in Stage 1 (n_envs=4) vs Stage 5
        # (n_envs=6), and skew every logged step number.
        # The modulo is a range check rather than == 0 because num_timesteps
        # jumps by n_envs each call and can step straight over an exact multiple.
        n_envs = getattr(self.training_env, "num_envs", 1)

        if self.num_timesteps % self.ckpt_freq < n_envs:
            path = os.path.join(self.ckpt_dir, f"{self.ckpt_prefix}_{self.num_timesteps}.zip")
            self.model.save(path)
            if self.verbose:
                print(f"[ckpt] saved {path}")

        if self.num_timesteps % self.eval_freq < n_envs:
            success_steps = [self._run_eval_episode() for _ in range(self.n_eval_episodes)]
            success_rate = sum(s is not None for s in success_steps) / self.n_eval_episodes
            times = [s for s in success_steps if s is not None]
            mean_t2s = float(np.mean(times)) if times else None

            self.logger.record("eval/success_rate", success_rate)
            if mean_t2s is not None:
                self.logger.record("eval/mean_time_to_success", mean_t2s)

            self.history.append(dict(step=self.num_timesteps, success_rate=success_rate,
                                      mean_time_to_success=mean_t2s,
                                      raw_time_to_success=success_steps, timestamp=time.time()))
            with open(os.path.join(self.ckpt_dir, "eval_history.json"), "w") as f:
                json.dump(self.history, f, indent=2)
            if self.verbose:
                print(f"[eval @ {self.num_timesteps}] success_rate={success_rate:.2f} mean_t2s={mean_t2s}")
        return True


def train_sac(train_env, eval_env, run_dir, total_timesteps, ckpt_prefix="policy",
              n_envs=None, eval_freq=10_000, ckpt_freq=100_000, n_eval_episodes=10,
              smoke_test_steps=5_000, seed=0, sac_kwargs=None, tb_subdir="tb"):
    """
    Generic SAC training loop: smoke test, then full run, saving into run_dir.
    `train_env` must already be wrapped as needed by the caller (VecMonitor,
    reward wrapper, VecNormalize, etc.) — this function is reward-source-agnostic,
    which is exactly what lets Stage 1 (real env reward) and Stage 5 (T2S reward)
    share it.
    """
    from stable_baselines3 import SAC

    # Build defaults, then let caller-supplied sac_kwargs override them.
    # (Doing this as dict(..., **sac_kwargs) raises TypeError on any key the
    # caller also sets — e.g. buffer_size — which defeats the point of the
    # override hook.)
    resolved_kwargs = dict(
        policy="MlpPolicy", learning_rate=3e-4, buffer_size=1_000_000,
        batch_size=256, tau=0.005, gamma=0.99, ent_coef="auto",
        policy_kwargs=dict(net_arch=[400, 400]),
    )
    resolved_kwargs.update(sac_kwargs or {})

    model = SAC(env=train_env, tensorboard_log=os.path.join(run_dir, tb_subdir),
                verbose=0, seed=seed, **resolved_kwargs)

    if smoke_test_steps:
        smoke_cb = SuccessRateCallback(eval_env, run_dir, eval_freq=min(500, smoke_test_steps),
                                        n_eval_episodes=3, ckpt_freq=smoke_test_steps + 1,
                                        ckpt_prefix=ckpt_prefix)
        model.learn(total_timesteps=smoke_test_steps, callback=smoke_cb, tb_log_name="smoke")

    main_cb = SuccessRateCallback(eval_env, run_dir, eval_freq=eval_freq,
                                   n_eval_episodes=n_eval_episodes, ckpt_freq=ckpt_freq,
                                   ckpt_prefix=ckpt_prefix, verbose=1)
    model.learn(total_timesteps=total_timesteps, callback=main_cb,
                tb_log_name="train", reset_num_timesteps=False)

    model.save(os.path.join(run_dir, f"{ckpt_prefix}_final"))
    return model, main_cb.history