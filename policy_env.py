"""
Stage 5 (part 1): T2S-driven reward wrapper.

REWARD_MODE:
  "absolute"   r = -pred(s')        every step not-done costs.
  "difference" r = pred(s) - pred(s')  telescopes to pred(first) - pred(last);
               standing still is free, which caused plateaus in earlier runs.
"""
import gymnasium as gym


class Time2SuccessRewardWrapper(gym.Wrapper):
    def __init__(self, env, predict_t2s, reward_mode="absolute"):
        super().__init__(env)
        assert reward_mode in ("absolute", "difference")
        self.predict_t2s = predict_t2s
        self.reward_mode = reward_mode
        self._last_pred = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_pred = self.predict_t2s(obs)
        return obs, info

    def step(self, action):
        obs, _env_reward, terminated, truncated, info = self.env.step(action)  # env reward discarded
        pred_now = self.predict_t2s(obs)
        if self.reward_mode == "absolute":
            reward = -pred_now
        else:
            reward = self._last_pred - pred_now
        self._last_pred = pred_now
        info["t2s_pred"] = pred_now
        return obs, reward, terminated, truncated, info


def make_policy_train_env(predict_t2s, reward_mode="absolute", camera_name="corner2"):
    """Factory for use with DummyVecEnv([...]) — one closure per env instance."""
    from env_utils import make_fixed_scene_env

    def _make():
        env = make_fixed_scene_env(camera_name=camera_name)
        return Time2SuccessRewardWrapper(env, predict_t2s, reward_mode=reward_mode)
    return _make
