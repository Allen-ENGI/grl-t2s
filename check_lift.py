#!/usr/bin/env python3
"""
Does the DATASET teach that lifting is progress?

    python check_lift.py --data-run v2

Queries dataset.npz directly. No model, no rollout, no probe — so it cannot
be confounded by an off-manifold perturbation, which is what made the
"12 of 12 penalise lifting" result untrustworthy.

If lifted frames carry LOWER steps-remaining than non-lifted ones, the data
says lifting is progress and any model predicting otherwise contradicts its
training set. If they carry HIGHER labels, the models learned their data
correctly and the dataset is the problem.

Only SUCCESSFUL episodes are used: rows from failed episodes carry the
censored ceiling, not a real label.
"""
import argparse
import os
import sys

import numpy as np

from config import PEG_POS_IDX, HAND_POS_IDX, CENSOR_LABEL

LIFT_EPS = 0.01


def main(argv=None):
    p = argparse.ArgumentParser(description="Does the dataset reward lifting?")
    p.add_argument("--data-run", required=True)
    p.add_argument("--lift-eps", type=float, default=LIFT_EPS)
    p.add_argument("--include-post-success", action="store_true",
                   help="keep frames with y == 0. OFF by default: collect_episode "
                        "runs the full 500 steps because MetaWorld does not "
                        "terminate on success, so ~89%% of a successful episode is "
                        "a peg already sitting in the hole. The peg stays UP there, "
                        "which makes 'lifted' almost exactly 'post-success' and "
                        "turns this whole comparison into 'finished episodes have "
                        "0 steps left'.")
    args = p.parse_args(argv)

    import results
    d = np.load(os.path.join(results.get_run_dir("data_collection", args.data_run),
                             "dataset.npz"))
    X, y, ep = d["X"], d["y_steps"], d["episode_ids"]

    # successful episodes only
    good = np.zeros(len(y), bool)
    for e in np.unique(ep):
        idx = np.flatnonzero(ep == e)
        if not np.all(y[idx] == CENSOR_LABEL):
            good[idx] = True

    # lift is measured against each episode's OWN starting peg height
    lift = np.zeros(len(y))
    for e in np.unique(ep):
        idx = np.flatnonzero(ep == e)
        lift[idx] = X[idx, PEG_POS_IDX[2]] - X[idx[0], PEG_POS_IDX[2]]

    hp = np.linalg.norm(X[:, HAND_POS_IDX] - X[:, PEG_POS_IDX], axis=1)
    m = good & (y < CENSOR_LABEL)
    if not args.include_post_success:
        m &= y > 0                      # the countdown only; see --include-post-success
        print("(excluding post-success frames, where y == 0)")
    up, down = m & (lift > args.lift_eps), m & (lift <= args.lift_eps)

    print(f"=== {args.data_run}: {m.sum():,} labelled rows from successful episodes ===")
    print(f"  lifted     {up.sum():>8,} rows   mean steps-remaining {y[up].mean():7.1f}"
          if up.sum() else "  lifted            0 rows   <<< the peg is NEVER lifted in "
                           "this dataset")
    print(f"  not lifted {down.sum():>8,} rows   mean steps-remaining {y[down].mean():7.1f}")
    if not up.sum():
        print("\n  With no lifted frames, the model has no evidence that lifting is")
        print("  progress. It cannot have learned it, and the probe result is moot.")
        return 0

    diff = y[up].mean() - y[down].mean()
    print(f"\n  lifted minus not-lifted: {diff:+.1f} steps")
    print("  NEGATIVE = the data says lifting is progress\n")

    # Lifted frames occur LATER, so they naturally carry lower labels. Control
    # for phase by matching on hand-peg distance: compare lifted and non-lifted
    # rows that are equally close to the peg.
    print("  controlled for phase (matched on hand-peg distance):")
    print(f"  {'hand-peg band':>16}{'lifted':>16}{'not lifted':>16}{'diff':>9}")
    edges = np.percentile(hp[m], [0, 20, 40, 60, 80, 100])
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = (hp >= lo) & (hp < hi)
        a, b = up & band, down & band
        if a.sum() < 20 or b.sum() < 20:
            continue
        print(f"  {f'{lo:.3f}-{hi:.3f}':>16}{y[a].mean():>16.1f}{y[b].mean():>16.1f}"
              f"{y[a].mean() - y[b].mean():>+9.1f}")

    print(f"\n  lift range in the data: {lift[m].min():+.3f} to {lift[m].max():+.3f} m")
    print(f"  rows above {args.lift_eps} m: {up.sum() / m.sum():.1%} of the labelled set")
    if up.sum() / m.sum() < 0.05:
        print("  <<< under 5%: lifted states are nearly absent, so whatever the model")
        print("      predicts up there is extrapolation regardless of the label sign.")
    return 0


if __name__ == "__main__":
    sys.exit(main())