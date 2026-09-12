import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_collection import (build_labels, detect_control_onset, find_noise_for_target_success_rate,
                              probe_success_rate, _discover_checkpoints)


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


class _FakeActionSpace:
    low = np.array([-1.0, -1.0], dtype=np.float32)
    high = np.array([1.0, 1.0], dtype=np.float32)

    def sample(self):
        return np.random.uniform(self.low, self.high)


class _FakeEnv:
    """Minimal duck-typed env for probe_success_rate: single-step episodes,
    success iff the (possibly noise-perturbed) action stays small. Lets us
    test the sim-touching probe/escalation logic without gym or metaworld."""

    action_space = _FakeActionSpace()

    def reset(self, seed=None):
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        success = bool(np.linalg.norm(action) < 0.3)
        return np.zeros(2, dtype=np.float32), 0.0, True, False, {"success": success}


class _FakeZeroPolicy:
    def predict(self, obs, deterministic=True):
        return np.zeros(2, dtype=np.float32), None


def test_probe_success_rate_drops_as_noise_increases():
    np.random.seed(0)
    env, policy = _FakeEnv(), _FakeZeroPolicy()
    low_noise_rate = probe_success_rate(policy, env, noise_std=0.01, n_probe=30)
    high_noise_rate = probe_success_rate(policy, env, noise_std=2.0, n_probe=30)
    assert low_noise_rate > 0.8    # near-zero action, almost always under the success threshold
    assert high_noise_rate < 0.3   # large random action, rarely under the threshold


def test_find_noise_for_target_success_rate_picks_first_passing_candidate():
    # rate drops as noise increases: 0.9 -> 0.5 -> 0.2 -> 0.1
    rates = {0.2: 0.9, 0.4: 0.5, 0.6: 0.2, 0.8: 0.1}
    noise, rate = find_noise_for_target_success_rate(
        measure_success_rate=lambda n: rates[n],
        noise_candidates=(0.2, 0.4, 0.6, 0.8), max_success_rate=0.3,
    )
    assert noise == 0.6 and rate == 0.2   # first candidate at/under threshold, not the smallest rate overall


def test_find_noise_for_target_success_rate_raises_if_never_below_threshold():
    rates = {0.2: 0.95, 0.4: 0.9, 0.6: 0.85}
    try:
        find_noise_for_target_success_rate(
            measure_success_rate=lambda n: rates[n],
            noise_candidates=(0.2, 0.4, 0.6), max_success_rate=0.3,
        )
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "too strong" in str(e)


def test_discover_checkpoints_does_not_confuse_final_with_earliest():
    """
    Regression test for the exact bug hit in chat: ckpt_prefix containing a
    digit (e.g. "peg_insert_side_v3") must not corrupt the numeric sort key
    used to find the earliest checkpoint, and "final" must never appear in
    the numbered list at all (previously it could sort as if step=3, ahead
    of every real numbered checkpoint).
    """
    prefix = "peg_insert_side_v3"
    with tempfile.TemporaryDirectory() as d:
        for name in [f"{prefix}_100000.zip", f"{prefix}_400000.zip", f"{prefix}_final.zip"]:
            open(os.path.join(d, name), "w").close()

        numbered, final = _discover_checkpoints(d, ckpt_prefix=prefix)

        assert len(final) == 1 and final[0].endswith(f"{prefix}_final.zip")
        assert len(numbered) == 2
        # must be sorted ascending by actual step count, and "final" must be absent
        assert numbered[0].endswith(f"{prefix}_100000.zip")
        assert numbered[1].endswith(f"{prefix}_400000.zip")
        assert all("final" not in p for p in numbered)


if __name__ == "__main__":
    test_build_labels_success_episode_counts_down_to_zero()
    test_build_labels_failed_episode_gets_censor_ceiling()
    test_build_labels_default_skips_stage_label()
    test_control_onset_detected_within_reasonable_window()
    test_probe_success_rate_drops_as_noise_increases()
    test_find_noise_for_target_success_rate_picks_first_passing_candidate()
    test_find_noise_for_target_success_rate_raises_if_never_below_threshold()
    test_discover_checkpoints_does_not_confuse_final_with_earliest()
    print("data_collection: all tests passed")
