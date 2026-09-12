import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_collection import build_labels, detect_control_onset


def _synthetic_grasp_episode(n=40, onset=15, success=30, obs_dim=39):
    """Hand and peg drift independently before `onset`, then move together
    (peg 'grasped') from onset to success."""
    obs = np.zeros((n, obs_dim), dtype=np.float32)
    rng = np.random.RandomState(0)
    hand = np.cumsum(rng.normal(0, 0.001, size=(n, 3)), axis=0)
    peg = np.zeros((n, 3), dtype=np.float32)
    peg[:onset] = np.cumsum(rng.normal(0, 0.0005, size=(onset, 3)), axis=0)
    # from onset onward, peg tracks hand (simulating a firm grasp)
    offset = peg[onset - 1] - hand[onset - 1] if onset > 0 else np.zeros(3)
    peg[onset:] = hand[onset:] + offset
    obs[:, 0:3] = hand
    obs[:, 4:7] = peg
    return obs


def test_build_labels_success_episode_counts_down_to_zero():
    obs = _synthetic_grasp_episode()
    steps, stage, onset = build_labels(obs, first_success=30, include_stage_label=True)
    assert steps[30] == 0.0
    assert steps[0] == 30.0
    assert stage[30] == 2  # complete at/after success


def test_build_labels_failed_episode_gets_censor_ceiling():
    obs = _synthetic_grasp_episode()
    steps, stage, onset = build_labels(obs, first_success=None, censor_label=300.0)
    assert np.all(steps == 300.0)


def test_build_labels_default_skips_stage_label():
    obs = _synthetic_grasp_episode()
    steps, stage, onset = build_labels(obs, first_success=30)  # include_stage_label defaults False
    assert onset is None
    assert np.all(stage == 0)
    assert steps[30] == 0.0  # steps_remaining is unaffected by the toggle


def test_control_onset_detected_within_reasonable_window():
    obs = _synthetic_grasp_episode(onset=15, success=30)
    onset = detect_control_onset(obs)
    # detector shouldn't fire before the true onset, and shouldn't miss it by a lot
    assert onset is not None
    assert 15 <= onset <= 25


if __name__ == "__main__":
    test_build_labels_success_episode_counts_down_to_zero()
    test_build_labels_failed_episode_gets_censor_ceiling()
    test_build_labels_default_skips_stage_label()
    test_control_onset_detected_within_reasonable_window()
    print("data_collection: all tests passed")
