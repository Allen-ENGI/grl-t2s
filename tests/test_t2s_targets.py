import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from t2s_targets import build_targets, build_bookkeeping

CENSOR = 300.0


def _toy_success_episode():
    """A single 4-frame episode: steps_remaining = 3, 2, 1, 0 (success at frame 3)."""
    X = np.zeros((4, 2), dtype=np.float32)
    y = np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float32)
    ep = np.zeros(4, dtype=np.int32)
    frames = np.arange(4, dtype=np.int32)
    return X, y, ep, frames


def test_mc_target_is_just_y():
    X, y, ep, frames = _toy_success_episode()
    rows = np.arange(4)
    out = build_targets("mc", bootstrap_fn=None, rows=rows, y_all=y,
                         next_obs_n_all=X, is_terminal_all=np.zeros(4, bool),
                         terminal_value_all=np.zeros(4), episode_ids_all=ep,
                         frame_idxs_all=frames)
    np.testing.assert_allclose(out, y)


def test_td0_terminal_rows_use_terminal_value_not_bootstrap():
    X, y, ep, frames = _toy_success_episode()
    next_obs, is_terminal, terminal_value = build_bookkeeping(X, y, ep, frames, CENSOR)
    # frame 3 (y==0) must be terminal with terminal_value 0
    assert is_terminal[3] and terminal_value[3] == 0.0

    rows = np.arange(4)
    # bootstrap_fn returns a constant so we can isolate the terminal-row logic
    out = build_targets("td0", bootstrap_fn=lambda x: np.full(len(x), 5.0),
                         rows=rows, y_all=y, next_obs_n_all=next_obs,
                         is_terminal_all=is_terminal, terminal_value_all=terminal_value,
                         episode_ids_all=ep, frame_idxs_all=frames, gamma=1.0)
    # terminal row -> exactly terminal_value, not 1 + gamma * bootstrap
    assert out[3] == 0.0
    # non-terminal rows -> 1 + gamma * bootstrap(next_obs)
    np.testing.assert_allclose(out[:3], 1.0 + 1.0 * 5.0)


def test_failed_episode_gets_censor_label_as_terminal_value():
    X = np.zeros((3, 2), dtype=np.float32)
    y = np.full(3, CENSOR, dtype=np.float32)   # never succeeds
    ep = np.zeros(3, dtype=np.int32)
    frames = np.arange(3, dtype=np.int32)
    _, is_terminal, terminal_value = build_bookkeeping(X, y, ep, frames, CENSOR)
    # only the LAST frame of a failed episode is terminal, with value = CENSOR
    assert not is_terminal[0] and not is_terminal[1]
    assert is_terminal[2] and terminal_value[2] == CENSOR


def test_clip_max_bounds_td0_targets():
    X, y, ep, frames = _toy_success_episode()
    next_obs, is_terminal, terminal_value = build_bookkeeping(X, y, ep, frames, CENSOR)
    rows = np.arange(4)
    out = build_targets("td0", bootstrap_fn=lambda x: np.full(len(x), 1000.0),
                         rows=rows, y_all=y, next_obs_n_all=next_obs,
                         is_terminal_all=is_terminal, terminal_value_all=terminal_value,
                         episode_ids_all=ep, frame_idxs_all=frames, gamma=1.0, clip_max=50.0)
    assert out.max() <= 50.0


def test_unknown_method_raises():
    X, y, ep, frames = _toy_success_episode()
    try:
        build_targets("bogus", lambda x: np.zeros(len(x)), np.arange(4), y, X,
                       np.zeros(4, bool), np.zeros(4), ep, frames)
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    test_mc_target_is_just_y()
    test_td0_terminal_rows_use_terminal_value_not_bootstrap()
    test_failed_episode_gets_censor_label_as_terminal_value()
    test_clip_max_bounds_td0_targets()
    test_unknown_method_raises()
    print("t2s_targets: all tests passed")
