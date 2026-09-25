#!/usr/bin/env python3
"""
Interaction-stage labels, derived from the observations already in dataset.npz.

    python t2s_stages.py --data-run v3          # validate before training on them

WHY AN AUXILIARY STAGE HEAD
---------------------------
Every one of the twelve models measured `relative_score = 0.00`: they learned
pred = f(hand) + g(peg), additive in ABSOLUTE positions, rather than anything
about the RELATIONSHIP between hand and peg. An additive function provably
cannot represent "hand near peg", which is why the stale-target probe found
the prediction minimum sitting at the peg's ORIGINAL location rather than
following it.

Demonstrations cannot teach the difference: the peg is always at its start
when the hand arrives, so "near the peg" and "near where the peg lives" are
the same feature in every training frame. An auxiliary head that must classify
"am I holding it" forces the encoder to represent the relationship, because no
additive function of the two positions separates grasped from hovering.

The reward still comes from t2s_head. The stage head only shapes the shared
encoder — so success means the PROBES move, not that stage accuracy is high.
A head that classifies stages perfectly while `relative_score` stays at 0.00
has learned them in a corner of the representation the countdown ignores.

THE STAGES, AND WHICH BOUNDARIES ARE HONEST
-------------------------------------------
  0 approach       hand far from the peg
  1 reach          hand at the peg, but no reliable coupling evidence yet
  2 coupled        closed gripper and peg moving with the hand
  3 lifted         peg above its reset height
  4 aligned        lifted and close to the goal
  5 inserted       steps-remaining is 0

`coupled` (stage 2) deliberately merges GRASPING and DRAGGING. The observation
layout was measured and contains NO contact or grasp indicator, so on the
table those two produce the same signature: peg displaced, moving with the
hand. The only reliable evidence of a grasp is that the peg LEFT the table,
which is stage 3. Splitting stage 2 into grasp-vs-drag would invent a
boundary the sensors cannot support and feed the encoder noise.

WHY THERE IS NO SEPARATE `close` STAGE
--------------------------------------
The collected observations contain no frames that remain uniquely in a
"closed but peg untouched" class: those frames immediately become coupled,
lifted, aligned, or inserted. Keeping an empty class would make the auxiliary
classifier ill-defined. Gripper closure is still required by the `coupled`
label, so the encoder must still pay attention to the gripper.
"""
import argparse
import os
import sys

import numpy as np

from config import HAND_POS_IDX, PEG_POS_IDX, GOAL_POS_IDX, GRIPPER_IDX, CENSOR_LABEL

STAGE_NAMES = ("approach", "reach", "coupled", "lifted", "aligned", "inserted")
N_STAGES = len(STAGE_NAMES)

CONTACT_EPS = 0.06     # hand-to-peg distance counting as "at the peg"
MOVED_EPS = 0.01       # peg displacement from its reset position
LIFT_EPS = 0.01        # peg height above reset
ALIGN_EPS = 0.25       # peg-to-goal distance counting as aligned
GRIP_CLOSED = 0.35     # calibrated from this dataset: lower values are closed
PEG_SPEED_EPS = 0.001  # per-step peg motion counting as "moving"
COUPLE_EPS = 0.01      # hand and peg per-step motion must agree this closely


def label_episode(obs, y):
    """
    Stage index per frame for ONE episode. obs: (T, obs_dim), y: (T,) steps
    remaining. Returns int8 (T,).

    Order matters: the tests run from the most specific state outward, so a
    lifted-and-aligned frame is `aligned` rather than falling through to
    `lifted`.
    """
    obs = np.asarray(obs, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    peg0 = obs[0, PEG_POS_IDX]

    hand_peg = np.linalg.norm(obs[:, HAND_POS_IDX] - obs[:, PEG_POS_IDX], axis=1)
    peg_goal = np.linalg.norm(obs[:, PEG_POS_IDX] - obs[:, GOAL_POS_IDX], axis=1)
    moved = np.linalg.norm(obs[:, PEG_POS_IDX] - peg0, axis=1)
    lift = obs[:, PEG_POS_IDX[2]] - peg0[2]
    grip = obs[:, GRIPPER_IDX]

    # per-step motion; frame 0 has none
    peg_delta = np.zeros((len(obs), 3))
    hand_delta = np.zeros((len(obs), 3))
    peg_delta[1:] = obs[1:, PEG_POS_IDX] - obs[:-1, PEG_POS_IDX]
    hand_delta[1:] = obs[1:, HAND_POS_IDX] - obs[:-1, HAND_POS_IDX]
    peg_speed = np.linalg.norm(peg_delta, axis=1)
    motion_mismatch = np.linalg.norm(peg_delta - hand_delta, axis=1)

    at_peg = hand_peg < CONTACT_EPS
    co_moving = ((moved > MOVED_EPS) & at_peg & (grip < GRIP_CLOSED)
                 & (peg_speed > PEG_SPEED_EPS) & (motion_mismatch < COUPLE_EPS))

    st = np.zeros(len(obs), dtype=np.int8)                  # 0 approach
    st[at_peg] = 1                                          # 1 reach
    st[co_moving] = 2                                       # 2 coupled
    st[lift > LIFT_EPS] = 3                                 # 3 lifted
    st[(lift > LIFT_EPS) & (peg_goal < ALIGN_EPS)] = 4      # 4 aligned
    st[y <= 0] = 5                                          # 5 inserted
    return st


def label_dataset(X, y, episode_ids):
    """Stage labels for every row, episode by episode."""
    out = np.zeros(len(y), dtype=np.int8)
    for e in np.unique(episode_ids):
        idx = np.flatnonzero(episode_ids == e)
        out[idx] = label_episode(X[idx], y[idx])
    return out


def validate(stages, y, episode_ids, censor_label=CENSOR_LABEL):
    """
    Are these labels usable? A bad auxiliary target is worse than none: it
    spends encoder capacity on noise.

      coverage     every stage should appear
      monotonicity mean steps-remaining should FALL as the stage index rises,
                   or the stage order disagrees with the countdown and the two
                   heads fight
      progression  transitions should mostly go forward; heavy backward
                   traffic means the thresholds are chattering
    """
    real = y < censor_label
    rows = []
    for k, name in enumerate(STAGE_NAMES):
        m = real & (stages == k)
        rows.append(dict(stage=k, name=name, n=int(m.sum()),
                         frac=float(m.sum() / max(real.sum(), 1)),
                         mean_y=float(y[m].mean()) if m.sum() else None))

    means = [r["mean_y"] for r in rows if r["mean_y"] is not None]
    monotone = all(a > b for a, b in zip(means, means[1:]))

    fwd = back = skip = 0
    for e in np.unique(episode_ids):
        d = np.diff(stages[episode_ids == e].astype(int))
        fwd += int(np.sum(d == 1)); back += int(np.sum(d < 0)); skip += int(np.sum(d > 1))
    total = max(fwd + back + skip, 1)

    return dict(rows=rows, monotone=monotone,
                forward=fwd / total, backward=back / total, skip=skip / total,
                missing=[r["name"] for r in rows if r["n"] == 0],
                thin=[r["name"] for r in rows if 0 < r["frac"] < 0.01])


def format_validation(v):
    out = [f"{'stage':<12}{'rows':>10}{'share':>9}{'mean steps left':>18}", "-" * 49]
    for r in v["rows"]:
        my = f"{r['mean_y']:.1f}" if r["mean_y"] is not None else "--"
        out.append(f"{r['name']:<12}{r['n']:>10,}{r['frac']:>9.1%}{my:>18}")
    out += ["",
            "  monotone in steps-remaining : "
            + ("YES" if v["monotone"]
               else "NO — the stage order disagrees with the countdown"),
            f"  transitions forward / back / skip : {v['forward']:.0%} / "
            f"{v['backward']:.0%} / {v['skip']:.0%}"]
    if v["missing"]:
        out.append(f"  MISSING stages (no frames): {v['missing']}")
    if v["thin"]:
        out.append(f"  thin stages (<1% of rows): {v['thin']}")
    if v["backward"] > 0.25:
        out.append("  >>> heavy backward traffic: the thresholds are chattering "
                   "rather than describing phases. Widen them or merge stages.")
    return "\n".join(out)


def trim_mask(y, episode_ids, confirm_buffer, censor_label=CENSOR_LABEL):
    """
    The rows load_and_prepare keeps: failed episodes whole, successful ones up
    to `confirm_buffer` frames past the first zero. Validating on the UNTRIMMED
    file reported `inserted` at 87.5% and `coupled` at 0.4% — both artefacts of
    the ~447-frame post-success tail that training never sees.
    """
    keep = np.zeros(len(y), dtype=bool)
    for e in np.unique(episode_ids):
        idx = np.flatnonzero(episode_ids == e)
        yy = y[idx]
        if np.all(yy == censor_label):
            keep[idx] = True
            continue
        z = np.flatnonzero(yy == 0.0)
        end = len(idx) - 1 if not len(z) else min(z[0] + confirm_buffer, len(idx) - 1)
        keep[idx[:end + 1]] = True
    return keep


def main(argv=None):
    p = argparse.ArgumentParser(description="Derive and validate stage labels")
    p.add_argument("--data-run", required=True)
    p.add_argument("--save", action="store_true",
                   help="write stages.npy next to dataset.npz")
    p.add_argument("--confirm-buffer", type=int, default=None,
                   help="validate on the rows TRAINING will see (default: "
                        "t2s_train.CONFIRM_BUFFER). The labels written by --save "
                        "always cover the full file, since training indexes them "
                        "with its own trim mask.")
    args = p.parse_args(argv)

    import results
    run_dir = results.get_run_dir("data_collection", args.data_run)
    d = np.load(os.path.join(run_dir, "dataset.npz"))
    X, y, ep = d["X"], d["y_steps"], d["episode_ids"]

    print("gripper percentiles [0,10,25,50,75,90,100]:",
          np.round(np.percentile(X[:, GRIPPER_IDX], [0, 10, 25, 50, 75, 90, 100]), 3))
    stages = label_dataset(X, y, ep)          # full file: this is what --save writes

    cb = args.confirm_buffer
    if cb is None:
        try:
            import t2s_train
            cb = t2s_train.CONFIRM_BUFFER
        except Exception:
            cb = 20
    keep = trim_mask(y, ep, cb)
    v = validate(stages[keep], y[keep], ep[keep])
    print(f"\n=== stage labels for {args.data_run}: {keep.sum():,} of {len(y):,} rows "
          f"as TRAINING sees them (confirm_buffer={cb}) ===\n")
    print(format_validation(v))

    if args.save:
        path = os.path.join(run_dir, "stages.npy")
        np.save(path, stages)
        print(f"\nwrote {path}")
    else:
        print("\n(dry run; pass --save to write stages.npy)")
    return 0 if v["monotone"] and not v["missing"] else 1


if __name__ == "__main__":
    sys.exit(main())