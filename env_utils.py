"""
Environment construction, shared by every stage that touches the sim.

Data collection, T2S model eval, and policy training all rolled their own
(near-identical) copies of `make_env` / `find_wrapper` / `make_fixed_scene_env`
across three notebooks. If one copy drifts (e.g. a different `camera_name`,
as happened between notebooks 1 and the compare notebook), predictions and
videos from different stages silently stop being comparable. There is now
exactly one definition.
"""
import os
import pickle

import gymnasium as gym
import metaworld  # noqa: F401  (registers the Meta-World envs)

from config import TASK_NAME, FIXED_TASK_PATH


def make_env(seed=0, render_mode=None, camera_name="corner2"):
    return gym.make(
        "Meta-World/MT1",
        env_name=TASK_NAME,
        seed=seed,
        render_mode=render_mode,
        camera_name=camera_name,
    )


def find_wrapper(env, attr_name):
    """Walk the gym.Wrapper chain to find the layer exposing `attr_name`."""
    w = env
    while not hasattr(w, attr_name):
        if not hasattr(w, "env"):
            raise AttributeError(f"no wrapper in the chain exposes {attr_name!r}")
        w = w.env
    return w


def ensure_fixed_task(path=FIXED_TASK_PATH, seed=0):
    """Create the fixed scene once, if it doesn't already exist."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    env = make_env(seed=seed)
    env.reset(seed=seed)
    with open(path, "wb") as f:
        pickle.dump(find_wrapper(env, "tasks").tasks[0], f)
    env.close()


def make_fixed_scene_env(render_mode=None, camera_name="corner2", fixed_task_path=FIXED_TASK_PATH):
    """The one fixed scene every stage of the pipeline must share."""
    env = make_env(seed=0, render_mode=render_mode, camera_name=camera_name)
    find_wrapper(env, "tasks").toggle_sample_tasks_on_reset(False)
    with open(fixed_task_path, "rb") as f:
        env.unwrapped.set_task(pickle.load(f))
    return env


def assert_scene_is_fixed(fixed_task_path=FIXED_TASK_PATH):
    """Sanity check used at the start of every stage's notebook cell."""
    import numpy as np
    from config import PEG_POS_IDX, GOAL_POS_IDX

    e = make_fixed_scene_env(fixed_task_path=fixed_task_path)
    o1, _ = e.reset(seed=1)
    o2, _ = e.reset(seed=2)
    e.close()
    idx = PEG_POS_IDX + GOAL_POS_IDX
    assert np.allclose(o1[idx], o2[idx]), "scene changed between resets — fixed_task did not take effect"
    return o1
