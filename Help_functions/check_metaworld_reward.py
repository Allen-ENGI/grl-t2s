import numpy as np
from core.env_utils import make_fixed_scene_env

env = make_fixed_scene_env()
obs, _ = env.reset(seed=0)
a = env.action_space.sample()
obs, r, *_ , info = env.step(a)
print("info keys:", {k: v for k, v in info.items() if np.isscalar(v)})
out = env.unwrapped.compute_reward(a, obs)
print("compute_reward returns:", out)
env.close()