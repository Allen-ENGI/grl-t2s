#!/usr/bin/env python3
"""
Document the 39 observation indices EMPIRICALLY, before changing any model.

    python obs_layout.py
    python obs_layout.py --steps 60 --markdown > OBS_LAYOUT.md

Nothing here trusts documentation or a remembered layout. Every claim is a
measurement on rollouts from this installation, because the only two index
facts this codebase ever asserted — config.PEG_POS_IDX and
config.GOAL_POS_IDX — were inherited unverified, and 29 of the 39 indices have
never been named at all.

WHAT IT CHECKS, AND HOW
-----------------------
  previous observation   obs[18:36] at step t vs obs[0:18] at t-1, exactly.
                         Already confirmed True with max difference 0.0.
  constant indices       zero variance across a randomised rollout. The goal
                         should be here; anything else constant is an unused
                         slot, and unused slots matter because they are dead
                         inputs the model still has to learn to ignore.
  gripper opening        correlation between an index and the commanded
                         gripper action, which is action[3].
  quaternions            consecutive 4-tuples whose norm stays ~1. MetaWorld's
                         18-dim current block is
                         [hand(3), gripper(1), obj1_pos(3), obj1_quat(4),
                          obj2_pos(3), obj2_quat(4)] — so quaternions occupy
                         8 of the 18, and there is NO velocity field.
  velocity               does any index track the finite difference of another?
                         If nothing does, velocity is only available by
                         subtracting the previous block — which is what makes
                         that block load-bearing rather than redundant.
  contact / grasp        any index that is binary, or that changes state
                         exactly when the gripper closes on the peg. MetaWorld
                         does not expose one; if that holds here, "is the peg
                         grasped" is NOT observable and the T2S model must
                         infer it from hand-peg geometry plus gripper opening.
                         That is worth knowing before concluding a model
                         "should" have learnt it.

Read the CONCLUSIONS block at the end. The per-index table is the evidence.
"""
import argparse
import sys

import numpy as np

from config import (OBS_DIM, HAND_POS_IDX, GRIPPER_IDX, PEG_POS_IDX,
                    GOAL_POS_IDX)

PREV_OFFSET = 18
CONST_EPS = 1e-9
QUAT_TOL = 0.02


def collect(steps=60, seed=0, gripper_phase=None, policy=None):
    """
    Rollout recording observations and actions.

    policy: an SB3 policy. STRONGLY PREFERRED over the random default, because
    random actions never grasp the peg — so the object stays put and every
    object-related index reads as constant. That is a property of the probe,
    not of the observation, and it makes the peg fields look unused when they
    are simply untouched. Run once with random actions to find the structural
    facts (previous-frame block, placeholders, goal) and once with an expert
    to find which fields actually carry signal during real manipulation.

    gripper_phase: if given, the gripper command is forced to -1 for the first
    half and +1 for the second, so an index that tracks gripper state is
    separable from one that merely correlates with time. Ignored when a policy
    is driving, since overriding its gripper would change what it does.
    """
    from env_utils import make_fixed_scene_env

    env = make_fixed_scene_env()
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    O, A = [np.asarray(obs, np.float64).copy()], []
    for t in range(steps):
        if policy is not None:
            a = np.asarray(policy.predict(obs, deterministic=False)[0], np.float64)
        else:
            a = rng.uniform(env.action_space.low, env.action_space.high)
            if gripper_phase:
                a[3] = -1.0 if t < steps // 2 else 1.0
        obs, _r, term, trunc, _i = env.step(a)
        O.append(np.asarray(obs, np.float64).copy())
        A.append(np.asarray(a, np.float64).copy())
        if term or trunc:
            break
    env.close()
    return np.array(O), np.array(A)


def analyse(O, A):
    """Per-index evidence, plus the structural checks."""
    T, D = O.shape
    std = O.std(axis=0)
    rng_ = O.max(axis=0) - O.min(axis=0)
    n_unique = np.array([len(np.unique(np.round(O[:, i], 9))) for i in range(D)])

    # previous-observation block
    prev_ok = (np.abs(O[1:, PREV_OFFSET:PREV_OFFSET + 18] - O[:-1, :18]).max()
               if D >= PREV_OFFSET + 18 else None)

    # gripper: correlate the index's LEVEL with the commanded gripper action.
    # Correlating the per-step DIFFERENCE was wrong: gripper opening tracks the
    # command as a level, so its difference is ~0 except at the switch, and the
    # detector scored the correct index at r=0.13 on data where it was exactly
    # the gripper.
    grip_corr = np.zeros(D)
    if len(A) > 2:
        a = A[:, 3]
        for i in range(D):
            lvl = O[1:len(A) + 1, i]
            if len(lvl) == len(a) and lvl.std() > 0 and a.std() > 0:
                grip_corr[i] = abs(np.corrcoef(lvl, a)[0, 1])

    # quaternions: consecutive 4-tuples with near-unit norm throughout
    # A unit-norm 4-tuple starting at i usually means the windows at i+1 and
    # i+2 are ALSO unit-norm, because the extra components are ~0 and one is
    # ~1. Reporting all of them implied three overlapping quaternions where
    # there is one. Keep the first of each consecutive run.
    hits = []
    for i in range(D - 3):
        n = np.linalg.norm(O[:, i:i + 4], axis=1)
        if np.abs(n - 1.0).max() < QUAT_TOL and std[i:i + 4].max() > CONST_EPS:
            hits.append(i)
    quats = [i for k, i in enumerate(hits) if k == 0 or i != hits[k - 1] + 1]

    # velocity: does index j track the per-step difference of index i?
    vel = []
    for i in range(D):
        di = np.diff(O[:, i])
        if di.std() < 1e-12:
            continue
        for j in range(D):
            if j == i or std[j] < 1e-12 or len(O) - 1 < 4:
                continue
            c = abs(np.corrcoef(di, O[1:, j])[0, 1])
            if c > 0.98:
                vel.append((i, j, float(c)))

    # contact/grasp: an indicator should take two values that look like a FLAG,
    # not merely two distinct floats. Requiring only "<= 2 unique values"
    # flagged a gripper command that happened to be driven with two levels,
    # which is a property of the probe, not of the observation.
    # A flag must take two WELL-SEPARATED values. The previous test rounded to
    # 6 decimals and accepted anything landing in {0,1} — so a quaternion
    # component wobbling at 1e-7 collapsed to two bins, and |1e-7| rounds to
    # 0.0, which is in {0,1}. That reported indices 8, 9, 26, 27 as possible
    # contact indicators when they are orientation components with no
    # meaningful variation at all. Require a real gap.
    binary = []
    for i in range(D):
        if std[i] <= 1e-3:                      # too small to be a state flag
            continue
        u = np.unique(np.round(O[:, i], 3))
        if len(u) == 2 and (u.max() - u.min()) > 0.5:
            binary.append(i)

    return dict(std=std, rng=rng_, n_unique=n_unique, prev_ok=prev_ok,
                grip_corr=grip_corr, quats=quats, vel=vel, binary=binary,
                const=[i for i in range(D) if std[i] <= CONST_EPS])


def label(i, a):
    """Best-supported name for one index, from the evidence only."""
    named = {**{k: "hand/TCP position" for k in HAND_POS_IDX},
             GRIPPER_IDX: "gripper (config says)",
             **{k: "peg position (config says)" for k in PEG_POS_IDX},
             **{k: "goal position (config says)" for k in GOAL_POS_IDX}}
    tags = []
    if i >= PREV_OFFSET and i < PREV_OFFSET + 18 and a["prev_ok"] is not None \
            and a["prev_ok"] < 1e-6:
        tags.append(f"PREV of idx {i - PREV_OFFSET}")
    if i in named:
        tags.append(named[i])
    if i in a["const"]:
        tags.append("CONSTANT")
    if i in a["binary"]:
        tags.append("BINARY (contact?)")
    if any(q <= i < q + 4 for q in a["quats"]):
        tags.append("quaternion component")
    if a["grip_corr"][i] > 0.5:
        tags.append(f"tracks gripper cmd (r={a['grip_corr'][i]:.2f})")
    return ", ".join(tags) or "unidentified"


def report(O, A, markdown=False, driver="random actions"):
    a = analyse(O, A)
    D = O.shape[1]
    L = []
    w = L.append

    if markdown:
        w("# Observation layout (measured)\n")
        w(f"`OBS_DIM = {OBS_DIM}`, {len(O)} steps driven by {driver} on the fixed scene.\n")
        w("| idx | std | range | uniq | identification |")
        w("|----:|----:|------:|-----:|:---------------|")
        for i in range(D):
            w(f"| {i} | {a['std'][i]:.4g} | {a['rng'][i]:.4g} | "
              f"{a['n_unique'][i]} | {label(i, a)} |")
    else:
        w(f"=== observation layout, {len(O)} steps driven by {driver} ===")
        w(f"{'idx':>4}{'std':>12}{'range':>12}{'uniq':>6}  identification")
        w("-" * 78)
        for i in range(D):
            w(f"{i:>4}{a['std'][i]:>12.4g}{a['rng'][i]:>12.4g}"
              f"{a['n_unique'][i]:>6}  {label(i, a)}")

    w("")
    w("## CONCLUSIONS" if markdown else "=== CONCLUSIONS ===")
    items = [
        ("TCP / hand position",
         f"indices {HAND_POS_IDX} (from config; they vary, std "
         f"{a['std'][HAND_POS_IDX].min():.3g}-{a['std'][HAND_POS_IDX].max():.3g})"),
        ("gripper opening",
         (f"index {GRIPPER_IDX}, correlation with the commanded gripper action "
          f"r={a['grip_corr'][GRIPPER_IDX]:.2f}"
          + ("  CONFIRMED" if a["grip_corr"][GRIPPER_IDX] > 0.5 else
             "  NOT confirmed — check which index does track it"))),
        ("peg position",
         f"indices {PEG_POS_IDX} (from config; std "
         f"{a['std'][PEG_POS_IDX].min():.3g}-{a['std'][PEG_POS_IDX].max():.3g})"
         + ("  — WARNING: the peg barely moved in this rollout, so these indices "
            "CANNOT be confirmed here and any 'constant' reading for them is an "
            "artefact of the driving policy. Re-run with --policy <expert.zip>."
            if a['std'][PEG_POS_IDX].max() < 1e-3 else "")),
        ("peg linear/angular velocity",
         ("NOT PRESENT as its own field"
          if not a["vel"] else
          f"candidate index pairs (diff-of-i tracks j): {a['vel'][:3]}")
         + ". Velocity is obtainable only as obs[0:18] - obs[18:36]."),
        ("goal position",
         f"indices {GOAL_POS_IDX} (from config)"
         + ("  — constant across the rollout, consistent with a pinned scene"
            if all(i in a["const"] for i in GOAL_POS_IDX) else
            "  — WARNING: NOT constant, which a pinned goal should be")),
        ("previous observation",
         (f"obs[18:36] == obs[0:18] of the previous step, max difference "
          f"{a['prev_ok']:.2e}  CONFIRMED" if a["prev_ok"] is not None
          and a["prev_ok"] < 1e-6 else
          f"NOT confirmed (max difference {a['prev_ok']})")),
        ("contact / grasp indicator",
         (f"binary-valued indices found: {a['binary']}" if a["binary"] else
          "NONE FOUND — no index is binary. Whether the peg is grasped is NOT "
          "directly observable; the model must infer it from hand-peg geometry "
          "plus gripper opening.")),
        ("quaternion blocks",
         f"start indices {a['quats']} (4-tuples with unit norm)" if a["quats"]
         else "none detected"),
        ("constant / dead inputs",
         f"{len(a['const'])} of {D} indices never change: {a['const']}"),
    ]
    for k, v in items:
        w(f"- **{k}**: {v}" if markdown else f"  {k:<28}{v}")

    w("")
    w("Unnamed-but-varying indices are real inputs the T2S model receives and")
    w("this codebase has never characterised. Before adding any input, check")
    w("t2s_probe's d/d other column: that group covers exactly these.")
    return "\n".join(L)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Measure the observation layout")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--markdown", action="store_true", help="emit a Markdown table")
    p.add_argument("--policy", default=None,
                   help="expert checkpoint to drive the rollout. Without it the "
                        "actions are random, which never grasps the peg, so every "
                        "object field reads as constant.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # forced gripper phases so a gripper-tracking index is separable from one
    # that merely drifts with time
    import os
    policy = None
    if args.policy:
        from stable_baselines3 import SAC
        from config import EXPERT_POLICY_DIR
        path = (args.policy if os.path.exists(args.policy)
                else os.path.join(EXPERT_POLICY_DIR, args.policy))
        policy = SAC.load(path)
        print(f"(driving with {os.path.basename(path)})\n")
    O, A = collect(steps=args.steps, seed=args.seed,
                   gripper_phase=policy is None, policy=policy)
    driver = os.path.basename(path) if args.policy else "random actions"
    print(report(O, A, markdown=args.markdown, driver=driver))
    return 0


if __name__ == "__main__":
    sys.exit(main())