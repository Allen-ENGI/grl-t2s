import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from t2s_io import save_normalization, load_normalization, NORM_KEYS


def test_normalization_round_trip():
    """This is the exact regression test for the X_mean/x_mean KeyError bug:
    if save and load ever disagree on keys again, this fails immediately
    instead of surfacing three notebooks later as a cryptic KeyError."""
    with tempfile.TemporaryDirectory() as run_dir:
        X_mean = np.random.randn(39).astype(np.float32)
        X_std = np.abs(np.random.randn(39)).astype(np.float32) + 0.1
        y_mean, y_std = 42.0, 7.5

        save_normalization(run_dir, X_mean, X_std, y_mean, y_std)
        loaded_X_mean, loaded_X_std, loaded_y_mean, loaded_y_std = load_normalization(run_dir)

        np.testing.assert_allclose(loaded_X_mean, X_mean)
        np.testing.assert_allclose(loaded_X_std, X_std)
        assert loaded_y_mean == y_mean
        assert loaded_y_std == y_std


def test_load_missing_keys_raises_clear_error():
    """If a file is missing keys (old format, partial write, etc.) the error
    should name exactly which keys are missing — not a bare numpy KeyError."""
    with tempfile.TemporaryDirectory() as run_dir:
        np.savez(os.path.join(run_dir, "normalization.npz"), x_mean=np.zeros(3))
        try:
            load_normalization(run_dir)
            assert False, "expected a KeyError"
        except KeyError as e:
            msg = str(e)
            assert "X_mean" in msg and "y_mean" in msg


if __name__ == "__main__":
    test_normalization_round_trip()
    test_load_missing_keys_raises_clear_error()
    print("t2s_io: all tests passed")
