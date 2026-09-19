"""
Stage 5 (part 1): T2S-driven reward wrapper.

The arithmetic itself lives in reward_fn.py (dependency-free) so the offline
preview imports the exact same function — see that module's docstring for the
reward modes and for why gamma belongs inside the shaping term.

This wrapper's only jobs are: call the frozen predictor, hand the pair of
predictions to reward_fn.step_reward, and emulate the absorbing state at
success that the environment itself does not provide.
"""
import gymnasium as gym

from config import SUCCESS_KEY, RL_GAMMA
from reward_fn import (REWARD_MODES, DEFAULT_TIME_PENALTY, step_reward,
                       assert_shaping_gamma_safe)


class Time2SuccessRewardWrapper(gym.Wrapper):
    def __init__(self, env, predict_t2s, reward_mode="difference_timed",
                 time_penalty=DEFAULT_TIME_PENALTY, gamma=RL_GAMMA,
                 terminate_on_success=True, check_gamma_pred_max=None):
        super().__init__(env)
        assert reward_mode in REWARD_MODES, f"reward_mode must be one of {REWARD_MODES}"
        self.predict_t2s = predict_t2s
        self.reward_mode = reward_mode
        self.time_penalty = time_penalty
        # gamma MUST equal the learner's discount — see reward_fn. Threaded in
        # rather than defaulted per-call so policy_train can pass the same
        # value it gives SAC and VecNormalize, instead of three copies drifting.
        self.gamma = gamma
        if check_gamma_pred_max is not None:
            assert_shaping_gamma_safe(gamma=gamma, time_penalty=time_penalty,
                                      pred_max=check_gamma_pred_max,
                                      reward_mode=reward_mode)
        # MetaWorld's peg-insert does NOT terminate on success — it runs to the
        # 500-step truncation. With a per-step time penalty that means the agent
        # keeps paying for time AFTER the task is complete, so in the post-success
        # tail succeeding and failing earn identical reward (visible as a -1.0
        # plateau on the SUCCESS trajectory in the reward preview).
        # Terminating on success emulates the absorbing state the MDP should
        # have, and matches how the T2S model was trained: t2s_targets sets
        # terminal_value = 0 at the success frame, i.e. T(s_T) = 0. It also
        # makes SAC bootstrap V = 0 after terminal, consistent with that target,
        # and stops burning ~450 steps per episode on an already-solved task.
        self.terminate_on_success = terminate_on_success
        self._last_pred = None
        self._succeeded = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_pred = self.predict_t2s(obs)
        self._succeeded = False
        return obs, info

    def step(self, action):
        obs, _env_reward, terminated, truncated, info = self.env.step(action)  # env reward discarded
        pred_now = self.predict_t2s(obs)
        success_now = bool(info.get(SUCCESS_KEY, 0))

        if self._succeeded:
            # already solved and (for some reason) still running: charge nothing
            reward = 0.0
        else:
            reward = step_reward(self._last_pred, pred_now, reward_mode=self.reward_mode,
                                 time_penalty=self.time_penalty, gamma=self.gamma)

        if success_now and not self._succeeded:
            self._succeeded = True
            if self.terminate_on_success:
                terminated = True

        self._last_pred = pred_now
        info["t2s_pred"] = pred_now
        info["t2s_reward"] = reward       # exposed so eval/video can show the live reward
        info["t2s_succeeded"] = self._succeeded
        return obs, reward, terminated, truncated, info


def make_policy_train_env(predict_t2s, reward_mode="difference_timed", camera_name="corner2",
                          time_penalty=DEFAULT_TIME_PENALTY, gamma=RL_GAMMA,
                          terminate_on_success=True, render_mode=None):
    """Factory for use with DummyVecEnv([...]) — one closure per env instance."""
    from env_utils import make_fixed_scene_env

    def _make():
        env = make_fixed_scene_env(camera_name=camera_name, render_mode=render_mode)
        return Time2SuccessRewardWrapper(env, predict_t2s, reward_mode=reward_mode,
                                         time_penalty=time_penalty, gamma=gamma,
                                         terminate_on_success=terminate_on_success)
    return _make
