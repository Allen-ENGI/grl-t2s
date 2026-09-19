"""
Shared SAC training machinery, used by BOTH:
  - expert_train.py   (Stage 1: pretrain against the environment's own reward)
  - policy_train.py   (Stage 5: train against the frozen T2S reward)

One copy here means a fix to eval logic never needs to be applied twice.
"""
import json
import os
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from config import SUCCESS_KEY, RL_GAMMA


class SuccessRateCallback(BaseCallback):
    """
    Periodically: (a) checkpoints the model, (b) runs `n_eval_episodes` on
    `eval_env` and records success rate + mean time-to-success against the
    REAL success flag (not any shaped/predicted reward), appending to
    `eval_history.json` in ckpt_dir.

    `eval_env` must be a single (non-vectorized) gym.Env exposing `.reset()`
    / `.step()` and `info["success"]`.

    EVAL EPISODES ARE STOCHASTIC ON PURPOSE. This used to call
    `eval_env.reset()` with no seed and `model.predict(deterministic=True)`.
    The scene is pinned (make_fixed_scene_env disables task resampling) and
    MuJoCo is deterministic, so all three sources of variation were gone:
    every one of `n_eval_episodes` was bit-identical. success_rate could only
    ever be 0.0 or 1.0, and every progress metric had exactly zero spread —
    the same defect dataset_audit.check_trajectory_uniqueness caught in the
    dataset (222 episodes -> 93 unique). n_eval_episodes=3 was paying 3x for
    one number. Now each episode gets its own reset seed and samples the
    policy, so the mean is an average over genuinely different rollouts.
    """

    def __init__(self, eval_env, ckpt_dir, eval_freq=10_000, n_eval_episodes=10,
                 ckpt_freq=100_000, max_ep_steps=500, ckpt_prefix="policy",
                 eval_seed_base=10_000, eval_deterministic=False, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.ckpt_dir = ckpt_dir
        self.ckpt_freq = ckpt_freq
        self.max_ep_steps = max_ep_steps
        self.ckpt_prefix = ckpt_prefix
        # eval seeds live in a block far from any training/collection seed so
        # they can never coincide with a trajectory the model was fit on
        self.eval_seed_base = eval_seed_base
        self.eval_deterministic = eval_deterministic
        os.makedirs(ckpt_dir, exist_ok=True)
        self.history = []

    def _run_eval_episode(self, seed):
        """
        Returns (success_step|None, progress_metrics_dict).

        success_step follows the package-wide contract in config: the index of
        the first state that IS successful, i.e. t+1 for a flag returned by the
        step out of s_t. It used to record t, one frame early, which shifted
        every reported time-to-success down by one and disagreed with the
        training labels built by data_collection.

        Progress metrics are model-independent (peg-to-goal distance, control
        onset) so runs that all score 0% success can still be ranked — see
        progress.py for why cumulative shaped reward cannot serve this role.
        """
        from progress import compute_progress_metrics

        obs, _ = self.eval_env.reset(seed=seed)
        if not self.eval_deterministic:
            # the policy's own sampling is the only source of episode-to-episode
            # variation on a pinned scene, so seed it explicitly to keep the
            # eval reproducible across runs while still varying within a run
            import torch
            torch.manual_seed(seed)
        obs_list, success_step = [], None
        for t in range(self.max_ep_steps):
            obs_list.append(np.asarray(obs, dtype=np.float32).copy())
            action, _ = self.model.predict(obs, deterministic=self.eval_deterministic)
            obs, _reward, terminated, truncated, info = self.eval_env.step(action)
            if info.get(SUCCESS_KEY, 0) and success_step is None:
                success_step = t + 1        # the state AFTER this step is the successful one
            if terminated or truncated:
                break
        # record the final state so index success_step always exists
        obs_list.append(np.asarray(obs, dtype=np.float32).copy())
        return success_step, compute_progress_metrics(np.array(obs_list), success_step)

    def _on_step(self) -> bool:
        # Use num_timesteps (total ENVIRONMENT steps across all parallel envs),
        # not n_calls. The modulo is a range check rather than == 0 because
        # num_timesteps jumps by n_envs each call and can step over an exact
        # multiple.
        n_envs = getattr(self.training_env, "num_envs", 1)

        if self.num_timesteps % self.ckpt_freq < n_envs:
            path = os.path.join(self.ckpt_dir, f"{self.ckpt_prefix}_{self.num_timesteps}.zip")
            self.model.save(path)
            if self.verbose:
                print(f"[ckpt] saved {path}")

        if self.num_timesteps % self.eval_freq < n_envs:
            from progress import aggregate_progress_metrics

            results = [self._run_eval_episode(self.eval_seed_base + len(self.history) * 1000 + i)
                       for i in range(self.n_eval_episodes)]
            success_steps = [s for s, _m in results]
            progress = aggregate_progress_metrics([m for _s, m in results])

            success_rate = sum(s is not None for s in success_steps) / self.n_eval_episodes
            times = [s for s in success_steps if s is not None]
            mean_t2s = float(np.mean(times)) if times else None

            self.logger.record("eval/success_rate", success_rate)
            if mean_t2s is not None:
                self.logger.record("eval/mean_time_to_success", mean_t2s)
            for key in ("mean_min_hand_peg_distance", "best_min_hand_peg_distance",
                        "mean_min_peg_goal_distance", "best_min_peg_goal_distance",
                        "mean_final_peg_goal_distance", "peg_moved_rate",
                        "control_rate", "mean_obs_movement"):
                if progress.get(key) is not None:
                    self.logger.record(f"eval/{key}", progress[key])

            entry = dict(step=self.num_timesteps, success_rate=success_rate,
                         mean_time_to_success=mean_t2s,
                         raw_time_to_success=success_steps, timestamp=time.time())
            entry.update(progress)
            self.history.append(entry)
            with open(os.path.join(self.ckpt_dir, "eval_history.json"), "w") as f:
                json.dump(self.history, f, indent=2)
            if self.verbose:
                hd = progress.get("mean_min_hand_peg_distance")
                pd = progress.get("mean_min_peg_goal_distance")
                print(f"[eval @ {self.num_timesteps}] succ={success_rate:.2f} "
                      f"hand->peg={f'{hd:.4f}' if hd is not None else '--'} "
                      f"peg->goal={f'{pd:.4f}' if pd is not None else '--'} "
                      f"peg_moved={progress.get('peg_moved_rate')} "
                      f"ctrl={progress.get('control_rate')}")
        return True


def train_sac(train_env, eval_env, run_dir, total_timesteps, ckpt_prefix="policy",
              eval_freq=10_000, ckpt_freq=100_000, n_eval_episodes=10,
              smoke_test_steps=5_000, seed=0, sac_kwargs=None, tb_subdir="tb",
              gamma=RL_GAMMA, n_eval_seeds_base=10_000):
    """
    Generic SAC training loop: smoke test, then full run, saving into run_dir.
    `train_env` must already be wrapped as needed by the caller (VecMonitor,
    reward wrapper, VecNormalize, etc.) — this function is reward-source-agnostic,
    which is what lets Stage 1 (real env reward) and Stage 5 (T2S reward) share it.

    gamma is an explicit argument rather than buried in the kwargs defaults
    because the T2S reward wrapper and VecNormalize need the SAME value, and
    `sac_kwargs.update()` silently let a caller change SAC's discount while
    the shaping term and the return statistics kept the old one.
    """
    from stable_baselines3 import SAC

    resolved_kwargs = dict(
        policy="MlpPolicy", learning_rate=3e-4, buffer_size=1_000_000,
        batch_size=256, tau=0.005, ent_coef="auto",
        policy_kwargs=dict(net_arch=[400, 400]),
    )
    resolved_kwargs.update(sac_kwargs or {})
    if "gamma" in resolved_kwargs and resolved_kwargs["gamma"] != gamma:
        raise ValueError(
            f"sac_kwargs sets gamma={resolved_kwargs['gamma']} but train_sac was called with "
            f"gamma={gamma}. The reward wrapper's shaping term and VecNormalize both use the "
            "train_sac value, so these must not differ — pass gamma= instead of sac_kwargs.")
    resolved_kwargs["gamma"] = gamma

    model = SAC(env=train_env, tensorboard_log=os.path.join(run_dir, tb_subdir),
                verbose=0, seed=seed, **resolved_kwargs)

    if smoke_test_steps:
        smoke_cb = SuccessRateCallback(eval_env, run_dir, eval_freq=min(500, smoke_test_steps),
                                       n_eval_episodes=3, ckpt_freq=smoke_test_steps + 1,
                                       ckpt_prefix=ckpt_prefix,
                                       eval_seed_base=n_eval_seeds_base + 500_000)
        model.learn(total_timesteps=smoke_test_steps, callback=smoke_cb, tb_log_name="smoke")

    main_cb = SuccessRateCallback(eval_env, run_dir, eval_freq=eval_freq,
                                  n_eval_episodes=n_eval_episodes, ckpt_freq=ckpt_freq,
                                  ckpt_prefix=ckpt_prefix, eval_seed_base=n_eval_seeds_base,
                                  verbose=1)
    model.learn(total_timesteps=total_timesteps, callback=main_cb,
                tb_log_name="train", reset_num_timesteps=False)

    model.save(os.path.join(run_dir, f"{ckpt_prefix}_final"))
    return model, main_cb.history
