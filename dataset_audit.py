"""
Dataset quality audit.

Checks a collected dataset.npz for the specific problems that have actually
bitten this project, rather than generic statistics. Every check returns a
verdict plus the numbers behind it, so a bad dataset is caught before hours
of T2S training and RL sweeps are spent on it.

Checks, and why each exists:

  1. success-boundary labels — verifies successful episodes contain a
     1 -> 0 transition in steps_remaining. A 0 -> 0 boundary means the
     off-by-one in collect_episode is present (first_success pointing at
     the last PRE-success frame), which destroys TD signal at exactly the
     terminal state. This is the check that would have caught it.

  2. failure coverage — counts failed episodes and their share of rows.
     With zero failures, the "succ" and "all" training conditions in
     t2s_train collapse to the identical model (observed: byte-identical
     metrics across every combo), so the comparison measures nothing.

  3. action/state diversity — how many distinct sources contributed, and
     whether near-neighbour states carry varied actions. A Q-style T(s,a)
     model cannot learn an action gradient from near-deterministic expert
     data; without diversity it silently degenerates into V(s).

  4. exploration-region coverage — how far the dataset's states spread from
     the expert manifold, approximated by hand-to-object distance. If every
     state has the hand already near the object, the model has no data
     anywhere an untrained policy actually starts, which is what produces
     flat extrapolated predictions and a dead reward signal.

  5. label sanity — censoring ceiling vs episode lengths, monotonicity of
     steps_remaining within episodes, and duplicate/degenerate rows.
"""
import json
import os

import numpy as np

from config import (HAND_POS_IDX, PEG_POS_IDX, GOAL_POS_IDX, OBS_DIM,
                     CENSOR_LABEL, MAX_STEPS)


def _episode_slices(episode_ids):
    """Yield (episode_id, index_array) for each episode, in file order."""
    for ep in np.unique(episode_ids):
        yield ep, np.flatnonzero(episode_ids == ep)


def check_success_boundary(y, episode_ids, censor_label=CENSOR_LABEL):
    """
    For every successful episode, check that steps_remaining reaches 0 and
    that the frame before the first 0 is exactly 1.

    A boundary of 0 -> 0 (two consecutive zeros with no preceding 1) means
    the success frame was mislabelled by one step.
    """
    good, bad, no_zero = 0, [], 0
    for ep, idx in _episode_slices(episode_ids):
        y_ep = y[idx]
        if np.all(y_ep == censor_label):
            continue                       # failed episode, no boundary expected
        zeros = np.flatnonzero(y_ep == 0.0)
        if len(zeros) == 0:
            no_zero += 1
            continue
        first_zero = zeros[0]
        if first_zero == 0:
            bad.append((int(ep), "episode starts already successful"))
        elif y_ep[first_zero - 1] == 1.0:
            good += 1
        else:
            bad.append((int(ep), f"frame before first 0 is {y_ep[first_zero-1]}, expected 1"))
    return dict(
        episodes_with_correct_boundary=good,
        episodes_with_bad_boundary=len(bad),
        bad_examples=bad[:5],
        successful_episodes_without_zero=no_zero,
        verdict=("OK" if not bad and good > 0 else
                  "NO SUCCESSFUL EPISODES" if good == 0 and not bad else
                  "BAD — success frame likely off by one"),
    )


def check_failure_coverage(y, episode_ids, censor_label=CENSOR_LABEL):
    """Failed-episode count and row share. Zero failures collapses succ/all."""
    n_ep, n_failed, failed_rows = 0, 0, 0
    for ep, idx in _episode_slices(episode_ids):
        n_ep += 1
        if np.all(y[idx] == censor_label):
            n_failed += 1
            failed_rows += len(idx)
    frac_ep = n_failed / n_ep if n_ep else 0.0
    return dict(
        total_episodes=n_ep, failed_episodes=n_failed,
        failed_episode_fraction=frac_ep,
        failed_row_fraction=failed_rows / len(y) if len(y) else 0.0,
        verdict=("NONE — 'succ' and 'all' conditions will train identical models"
                  if n_failed == 0 else
                  "THIN — few failures; succ/all comparison will be weak"
                  if frac_ep < 0.1 else "OK"),
    )


def check_exploration_coverage(X, near_threshold=0.05, far_threshold=0.20):
    """
    Hand-to-object distance distribution. An untrained policy starts with the
    hand far from the object, so if the dataset has no far-hand states the
    model must extrapolate exactly where the reward matters most.
    """
    hand_peg = np.linalg.norm(X[:, HAND_POS_IDX] - X[:, PEG_POS_IDX], axis=1)
    frac_far = float(np.mean(hand_peg > far_threshold))
    return dict(
        hand_peg_min=float(hand_peg.min()), hand_peg_max=float(hand_peg.max()),
        hand_peg_median=float(np.median(hand_peg)),
        frac_hand_near_object=float(np.mean(hand_peg < near_threshold)),
        frac_hand_far_from_object=frac_far,
        verdict=("SPARSE — little data with the hand far from the object, so "
                  "predictions in the exploration region are extrapolation"
                  if frac_far < 0.15 else "OK"),
    )


def check_object_movement(X, episode_ids, eps=1e-6):
    """
    Fraction of episodes in which the object was ever displaced. Dataset
    trajectories where the object never moves teach nothing about
    manipulation, only about reaching.
    """
    moved = 0
    n_ep = 0
    for ep, idx in _episode_slices(episode_ids):
        n_ep += 1
        peg = X[idx][:, PEG_POS_IDX]
        if np.abs(peg - peg[0]).max() > eps:
            moved += 1
    return dict(
        episodes_with_object_moved=moved, total_episodes=n_ep,
        object_moved_fraction=moved / n_ep if n_ep else 0.0,
        verdict=("OK" if n_ep and moved / n_ep > 0.5 else
                  "LOW — most episodes never move the object"),
    )


def check_trajectory_uniqueness(X, episode_ids):
    """
    Counts distinct trajectories. Duplicates arise when the scene is pinned,
    MuJoCo is deterministic, and the policy is queried with
    deterministic=True — then every clean rollout from a checkpoint is
    bit-identical no matter what seed reset() got. Measured once at 222
    episodes -> 93 unique (58% duplicates), concentrated in exactly the
    expert rollouts the "succ" condition trains on.
    """
    sigs = [hash(X[np.flatnonzero(episode_ids == e)].tobytes())
            for e in np.unique(episode_ids)]
    n_ep, n_unique = len(sigs), len(set(sigs))
    frac = 1 - n_unique / n_ep if n_ep else 0.0
    return dict(
        total_episodes=n_ep, unique_trajectories=n_unique,
        duplicate_fraction=frac,
        verdict=("OK" if frac < 0.05 else
                  f"{frac:.0%} DUPLICATES — set deterministic=False and use "
                  "multiple collection_seeds"),
    )


def check_split_integrity(X, y, episode_ids, split, censor_label=CENSOR_LABEL):
    """
    Audit the train/val split stored by data_collection.

    Three ways a split can be useless here, all seen in this project:
      1. the same trajectory CONTENT on both sides. GroupShuffleSplit grouped
         by episode id, and this dataset's rollouts were 58% duplicates, so
         identical trajectories with different ids straddled the split and val
         MSE partly measured memorization. assign_splits routes duplicates to
         one side; this verifies it.
      2. an episode straddling the split (rows of one episode on both sides),
         which leaks adjacent near-identical states whose labels differ by 1.
      3. val with no FAILED episodes, which makes the succ/all condition
         comparison unvalidatable.
    """
    split = np.asarray(split).astype(str)
    straddling, sig_sides = [], {}
    val_failed = val_succ = 0
    for ep, idx in _episode_slices(episode_ids):
        sides = set(split[idx])
        if len(sides) > 1:
            straddling.append(int(ep))
        side = sorted(sides)[0]
        sig = hash(X[idx].tobytes())
        sig_sides.setdefault(sig, set()).add(side)
        if side == "val":
            if np.all(y[idx] == censor_label):
                val_failed += 1
            else:
                val_succ += 1
    leaked = [s for s, sides in sig_sides.items() if len(sides) > 1]
    n_val_rows = int(np.sum(split == "val"))
    problems = []
    if leaked:
        problems.append(f"{len(leaked)} duplicate trajectory group(s) on BOTH sides")
    if straddling:
        problems.append(f"{len(straddling)} episode(s) straddling the split")
    if n_val_rows == 0:
        problems.append("val split is empty")
    if val_failed == 0:
        problems.append("val has no failed episodes")
    return dict(
        val_rows=n_val_rows, train_rows=int(np.sum(split == "train")),
        val_row_fraction=float(n_val_rows / len(split)) if len(split) else 0.0,
        val_successful_episodes=val_succ, val_failed_episodes=val_failed,
        duplicate_groups_spanning_split=len(leaked),
        episodes_straddling_split=len(straddling),
        straddling_examples=straddling[:5],
        verdict="OK" if not problems else "LEAK/GAP — " + "; ".join(problems),
    )


def check_label_sanity(y, episode_ids, censor_label=CENSOR_LABEL, max_steps=MAX_STEPS):
    """Censoring ceiling vs episode length, and monotonicity within episodes."""
    non_monotonic = []
    max_len = 0
    for ep, idx in _episode_slices(episode_ids):
        y_ep = y[idx]
        max_len = max(max_len, len(idx))
        if np.all(y_ep == censor_label):
            continue
        # steps_remaining should be non-increasing within a successful episode
        if np.any(np.diff(y_ep) > 0):
            non_monotonic.append(int(ep))
    return dict(
        censor_label=float(censor_label), max_steps=int(max_steps),
        longest_episode=int(max_len),
        censor_below_longest_episode=bool(censor_label < max_len),
        non_monotonic_episodes=len(non_monotonic),
        non_monotonic_examples=non_monotonic[:5],
        y_min=float(y.min()), y_max=float(y.max()),
        verdict=("OK" if not non_monotonic else
                  "steps_remaining increases within some episodes"),
    )


def audit_dataset(dataset_path, summary_path=None, verbose=True):
    """
    Runs every check against a dataset.npz and returns a report dict.
    Pass summary_path (dataset_summary.json) to include collection metadata.
    """
    d = np.load(dataset_path)
    X, y = d["X"], d["y_steps"]
    episode_ids = d["episode_ids"]
    split = d["split"] if "split" in d.files else None

    report = dict(
        path=dataset_path,
        n_rows=int(len(X)), obs_dim=int(X.shape[1]),
        obs_dim_matches_config=bool(X.shape[1] == OBS_DIM),
        trajectory_uniqueness=check_trajectory_uniqueness(X, episode_ids),
        success_boundary=check_success_boundary(y, episode_ids),
        failure_coverage=check_failure_coverage(y, episode_ids),
        exploration_coverage=check_exploration_coverage(X),
        object_movement=check_object_movement(X, episode_ids),
        label_sanity=check_label_sanity(y, episode_ids),
    )
    report["split_integrity"] = (
        check_split_integrity(X, y, episode_ids, split)
        if split is not None else
        dict(verdict="MISSING — dataset has no 'split' array; re-collect with the "
                     "current data_collection so train/val is fixed at collection time"))

    if summary_path and os.path.exists(summary_path):
        with open(summary_path) as f:
            report["collection_summary"] = json.load(f)

    if verbose:
        print(format_audit(report))
    return report


def format_audit(report):
    """Human-readable audit report."""
    lines = [
        f"dataset: {report['path']}",
        f"  rows={report['n_rows']:,} obs_dim={report['obs_dim']} "
        f"(config match: {report['obs_dim_matches_config']})",
        "",
    ]
    order = [
        ("trajectory_uniqueness", "trajectory diversity"),
        ("success_boundary", "success-frame labelling"),
        ("failure_coverage", "failure coverage"),
        ("exploration_coverage", "exploration-region coverage"),
        ("object_movement", "object manipulation"),
        ("label_sanity", "label sanity"),
        ("split_integrity", "train/val split"),
    ]
    for key, label in order:
        sec = report[key]
        verdict = sec.get("verdict", "?")
        flag = "ok  " if verdict == "OK" else "WARN"
        lines.append(f"[{flag}] {label}: {verdict}")
        for k, v in sec.items():
            if k == "verdict":
                continue
            if isinstance(v, float):
                lines.append(f"         {k} = {v:.4f}")
            else:
                lines.append(f"         {k} = {v}")
        lines.append("")
    return "\n".join(lines)
