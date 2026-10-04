"""
Start an episode with the peg moved, on the fixed scene.

    obs = reset_with_peg_offset(env, seed, (0.03, -0.02, 0.0))    # metres

MetaWorld places the peg and the box at reset from the task's start vector
(_last_rand_vec: peg position first, then the box). The fixed scene freezes that
vector. This shifts the peg part of it, resets normally, and puts the vector back,
so MetaWorld's own reset code sets every start-dependent value (peg start, peg head,
hole target) consistently. The box and hole are not moved.
"""
import numpy as np

from config import PEG_POS_IDX


def reset_with_peg_offset(env, seed, offset):
    u = env.unwrapped
    offset = np.asarray(offset, float)
    if not np.any(offset):
        obs, _ = env.reset(seed=seed)
        return np.asarray(obs, np.float32)
    if not hasattr(u, "_last_rand_vec"):
        raise RuntimeError("this MetaWorld version has no _last_rand_vec; see check_peg_reset() below")

    original = np.array(u._last_rand_vec, copy=True)
    frozen = getattr(u, "_freeze_rand_vec", None)
    base_obs, _ = env.reset(seed=seed)                       # where the peg normally starts
    shifted = original.copy()
    shifted[:3] += offset
    try:
        u._last_rand_vec = shifted
        u._freeze_rand_vec = True
        obs, _ = env.reset(seed=seed)
    finally:
        u._last_rand_vec = original                          # later resets are unaffected
        if frozen is not None:
            u._freeze_rand_vec = frozen

    obs = np.asarray(obs, np.float32)
    moved = obs[PEG_POS_IDX][:2] - np.asarray(base_obs)[PEG_POS_IDX][:2]
    if np.linalg.norm(moved - offset[:2]) > 0.01:
        raise RuntimeError(f"peg moved by {moved.round(3)} instead of {offset[:2].round(3)}; "
                           "run check_peg_reset() to see how this MetaWorld version places the peg")
    return obs


def check_peg_reset(env, seed=0):
    """Print what this MetaWorld version uses to place the peg, for debugging."""
    import inspect
    u = env.unwrapped
    obs, _ = env.reset(seed=seed)
    print("environment class:", type(u).__name__)
    print("peg position in obs:", np.asarray(obs)[PEG_POS_IDX].round(4))
    for name in ("_last_rand_vec", "_freeze_rand_vec", "obj_init_pos", "_target_pos"):
        print(f"{name}:", getattr(u, name, "(not present)"))
    print("\nreset_model source:\n", inspect.getsource(type(u).reset_model))