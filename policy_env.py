"""
Stage 5 (part 1): T2S-driven reward wrapper.

REWARD_MODE:
  "absolute"    r = -pred(s')                  every step not-done costs.
  "difference"  r = pred(s) - pred(s')          potential-based shaping;
                standing still is free, which makes hovering a safe local
                optimum (the reward preview shows a long zero-reward plateau
                on failure trajectories, with large negative spikes for
                moving — so SAC can prefer to freeze).
  "difference_timed"
                r = pred(s) - pred(s') - time_penalty
                Restores the task reward that "difference" dropped. For a
                minimum-time objective, T(s_t) = dt + T(s_{t+1}) implies the
                per-step cost IS -1 and the value difference is the shaping
                term; plain "difference" keeps the shaping and discards the
                objective. With an accurate prediction, a genuinely
                progressing step gives pred(s)-pred(s') ~ 1, so r ~ 0 (free),
                while hovering gives r = -time_penalty (costly).

The arithmetic itself lives in reward_fn.py (dependency-free) so the offline
preview imports the exact same function — see that module's docstring for why.
"""
import gymnasium as gym

from config import SUCCESS_KEY
from reward_fn import REWARD_MODES, DEFAULT_TIME_PENALTY, step_reward


class Time2SuccessRewardWrapper(gym.Wrapper):
    def __init__(self, env, predict_t2s, reward_mode="absolute",
                 time_penalty=DEFAULT_TIME_PENALTY, terminate_on_success=True):
        super().__init__(env)
        assert reward_mode in REWARD_MODES, f"reward_mode must be one of {REWARD_MODES}"
        self.predict_t2s = predict_t2s
        self.reward_mode = reward_mode
        self.time_penalty = time_penalty
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
            reward = step_reward(self._last_pred, pred_now,
                                  reward_mode=self.reward_mode, time_penalty=self.time_penalty)

        if success_now and not self._succeeded:
            self._succeeded = True
            if self.terminate_on_success:
                terminated = True

        self._last_pred = pred_now
        info["t2s_pred"] = pred_now
        info["t2s_succeeded"] = self._succeeded
        return obs, reward, terminated, truncated, info


def make_policy_train_env(predict_t2s, reward_mode="absolute", camera_name="corner2",
                           time_penalty=DEFAULT_TIME_PENALTY, terminate_on_success=True):
    """Factory for use with DummyVecEnv([...]) — one closure per env instance."""
    from env_utils import make_fixed_scene_env

    def _make():
        env = make_fixed_scene_env(camera_name=camera_name)
        return Time2SuccessRewardWrapper(env, predict_t2s, reward_mode=reward_mode,
                                          time_penalty=time_penalty,
                                          terminate_on_success=terminate_on_success)
    return _make