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


def check_obs_contains_history(n_steps=6, seed=0):
    """
    Does the observation already carry the previous timestep?

    MetaWorld's 39-dim observation is documented as
    [current(18) | previous(18) | goal(3)], which would mean indices 18..35 are
    a one-step history and velocity is already inferable from a single
    observation. This codebase names only 10 of the 39 indices, so that has
    never been verified here — and it decides whether feeding the T2S model a
    window of frames would add anything or merely duplicate what it is already
    given.

    Returns a dict with the verdict. Run it before building any frame-stacking:
    if history is already present, the question becomes whether the model USES
    it (see t2s_probe's `d/d other` column, which covers exactly these
    indices), not whether to supply it.
    """
    import numpy as np

    env = make_fixed_scene_env()
    obs, _ = env.reset(seed=seed)
    prev = np.asarray(obs, dtype=np.float64).copy()
    matches, diffs = [], []
    for _ in range(n_steps):
        obs, _r, term, trunc, _info = env.step(env.action_space.sample())
        cur = np.asarray(obs, dtype=np.float64)
        # the claim: cur[18:36] == prev[0:18]
        d = float(np.abs(cur[18:36] - prev[:18]).max())
        diffs.append(d)
        matches.append(d < 1e-6)
        prev = cur.copy()
        if term or trunc:
            break
    env.close()

    frac = float(np.mean(matches)) if matches else 0.0
    return dict(
        n_steps=len(matches), fraction_matching=frac,
        max_abs_difference=float(np.max(diffs)) if diffs else None,
        has_history=bool(frac > 0.9),
        verdict=("obs[18:36] IS the previous obs[0:18] — the observation is "
                 "already a 2-frame window; check whether the model uses it "
                 "(t2s_probe `d/d other`) before stacking more"
                 if frac > 0.9 else
                 "obs[18:36] is NOT the previous timestep — no history present, "
                 "so a windowed input would add genuinely new information"),
    )


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