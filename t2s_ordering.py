#!/usr/bin/env python3
"""
Does the T2S model order the interaction states correctly?

    python t2s_ordering.py --t2s-run v3
    python t2s_ordering.py --t2s-run v1 --models tdlambdaboot_all tdlambdaboot_succ

THE TEST
--------
    T(hovering) > T(grasped) > T(lifted) > T(aligned) > T(inserted) = 0

If that ordering fails, no reward mode, no gamma, and no RL algorithm will fix
it: the potential simply does not know that progress has been made, so there
is no gradient to follow between the stages it confuses.

WHY THIS BEATS EVERY OTHER METRIC HERE
--------------------------------------
MAE, correlation, calibration slope and step-slope error are all measured
ALONG A TRAJECTORY, where time advances and everything moves at once. They
answer "does the curve go down smoothly", which the expert rollouts make easy
and which a model can pass while conflating the states that matter.

This compares MATCHED states: the arm in the same region of space, differing
only in what it is doing with the peg. That is precisely the distinction the
hovering failure turns on — "hand at the hole, peg untouched" against "peg in
the hole" — and it is the one comparison a trajectory metric cannot isolate.

HOW THE STATES ARE FOUND
------------------------
From a real expert rollout, classified by geometry rather than by frame index:

  hovering   hand near the peg, peg NOT lifted, peg NOT moved toward the goal
  grasped    hand in contact, peg displaced, still at table height
  lifted     peg raised more than PEG_LIFT_EPS above its reset height
  aligned    peg lifted AND close to the goal
  inserted   the success flag is set

A stage with no matching frames is reported as absent rather than faked. If
the expert never hovers — likely, since it is an expert — the hovering state
is synthesised instead by taking a real grasped frame and putting the peg back
where it started, which is exactly the counterfactual the reward needs to
distinguish.
"""
import argparse
import json
import os
import sys

import numpy as np

from config import (HAND_POS_IDX, PEG_POS_IDX, GOAL_POS_IDX, SUCCESS_KEY,
                    steps_remaining_curve)
from progress import hand_peg_distance, peg_goal_distance

STAGE_MANIFEST = "stage_manifest.json"
PREV_OFFSET = 18
ORDER = ("hovering", "grasped", "lifted", "aligned", "inserted")

CONTACT_EPS = 0.06      # hand-to-peg distance counting as contact
LIFT_EPS = 0.01         # peg height above reset counting as lifted
ALIGN_EPS = 0.25        # peg-to-goal distance counting as aligned


def classify(obs_seq, success_step=None):
    """
    Assign each frame to an interaction stage, by geometry. Returns
    {stage: [frame indices]}.
    """
    obs = np.asarray(obs_seq, dtype=np.float64)
    peg0 = obs[0, PEG_POS_IDX]
    out = {k: [] for k in ORDER}
    for i, o in enumerate(obs):
        if success_step is not None and i >= success_step:
            out["inserted"].append(i)
            continue
        hp = hand_peg_distance(o)
        pg = peg_goal_distance(o)
        lift = float(o[PEG_POS_IDX[2]] - peg0[2])
        moved = float(np.linalg.norm(o[PEG_POS_IDX] - peg0))
        if lift > LIFT_EPS and pg < ALIGN_EPS:
            out["aligned"].append(i)
        elif lift > LIFT_EPS:
            out["lifted"].append(i)
        elif hp < CONTACT_EPS and moved > LIFT_EPS:
            out["grasped"].append(i)
        elif hp < CONTACT_EPS:
            out["hovering"].append(i)
    return out


def synth_hovering(obs_seq, stages):
    """
    A counterfactual hovering frame: take a real GRASPED or LIFTED frame and
    put the peg back at its reset position, leaving the arm where it is.

    This is the exact comparison the reward must get right — same arm pose,
    peg held versus peg untouched — and an expert rollout rarely contains a
    natural example, because an expert does not hover.

    CAUTION, and the reason the caller must report its z-score: the result is
    OFF-DISTRIBUTION by construction. An arm high near the goal with the peg
    still on the table never occurs in training, because the arm only goes
    there while carrying. So a large hovering-vs-holding gap may be the network
    extrapolating into a region it has never seen rather than understanding the
    counterfactual. Read it alongside the REAL hovering frames, which are
    in-distribution and therefore the honest evidence.
    """
    obs = np.asarray(obs_seq, dtype=np.float64)
    src = (stages.get("lifted") or stages.get("aligned") or stages.get("grasped"))
    if not src:
        return None
    o = obs[src[len(src) // 2]].copy()
    o[PEG_POS_IDX] = obs[0, PEG_POS_IDX]
    o[[i + PREV_OFFSET for i in PEG_POS_IDX]] = obs[0, PEG_POS_IDX]
    return o


def offdist_z(obs, run_dir):
    """Max |z| of a state against the training statistics. >3 = extrapolating."""
    from t2s_io import load_normalization
    try:
        mean, std = load_normalization(run_dir)
    except Exception:
        return None
    z = (np.asarray(obs, dtype=np.float32) - mean) / np.where(std < 1e-4, 1.0, std)
    return float(np.abs(z).max())


def score_model(fn, obs_seq, stages, synth=None):
    """Mean prediction per stage, plus the synthesised hovering counterfactual."""
    obs = np.asarray(obs_seq, dtype=np.float64)
    out = {}
    for k in ORDER:
        idx = stages.get(k) or []
        out[k] = float(np.mean([fn(obs[i]) for i in idx])) if idx else None
        out[f"n_{k}"] = len(idx)
    if synth is not None:
        out["hovering_synth"] = float(fn(synth))
    return out


def check_order(vals):
    """
    Which adjacent pairs of the required ordering hold. Returns
    (ok, [(a, b, va, vb, passed), ...]) over the stages that are present.
    """
    present = [k for k in ORDER if vals.get(k) is not None]
    pairs = []
    for a, b in zip(present, present[1:]):
        va, vb = vals[a], vals[b]
        pairs.append((a, b, va, vb, va > vb))
    return all(p[4] for p in pairs), pairs


def matched_pairs(succ_seq, fail_seq, max_hand_dist=0.04, min_peg_gap=0.05, limit=40):
    """
    Real frames from two rollouts with the SAME hand pose but DIFFERENT peg
    state. The non-circular version of the ordering test.

    WHY THIS IS NEEDED: on an expert rollout the stages are traversed in time
    order, and steps-remaining falls monotonically with time. So ANY model that
    tracks time accurately reproduces
        T(hover) > T(grasp) > T(lift) > T(align) > T(insert)
    whether or not it has any notion of grasping. Measured on run v3:

        stage       frames   true steps left   model says
        hovering     18-23              32.5         31.1
        grasped      24-33              24.5         23.4
        lifted       34-43              14.5         15.2
        aligned      44-52               5.0          3.0

    The model is accurate — and the ordering follows from that accuracy alone.
    The stage labels did no work.

    Pairing a SUCCESS frame against a FAILURE frame at the same hand pose
    removes time from the comparison: the arm is in the same place, and the
    only difference is whether the peg came with it. A model that merely
    tracks arm pose gives the same prediction to both. A model that
    understands grasping does not.
    """
    S = np.asarray(succ_seq, dtype=np.float64)
    F = np.asarray(fail_seq, dtype=np.float64)
    peg0 = S[0, PEG_POS_IDX]
    out = []
    for j, f in enumerate(F):
        d = np.linalg.norm(S[:, HAND_POS_IDX] - f[HAND_POS_IDX], axis=1)
        i = int(np.argmin(d))
        if d[i] > max_hand_dist:
            continue
        peg_gap = float(np.linalg.norm(S[i, PEG_POS_IDX] - f[PEG_POS_IDX]))
        if peg_gap < min_peg_gap:
            continue                       # peg in the same place: nothing to compare
        out.append(dict(succ_idx=i, fail_idx=j, hand_dist=float(d[i]),
                        peg_gap=peg_gap,
                        succ_lift=float(S[i, PEG_POS_IDX[2]] - peg0[2]),
                        fail_lift=float(F[j, PEG_POS_IDX[2]] - peg0[2])))
        if len(out) >= limit:
            break
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="T(hovering) > T(grasped) > T(lifted) > T(aligned) > T(inserted)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--t2s-run", required=True)
    p.add_argument("--models", nargs="+", default=None,
                   help="combos to test (default: all in the run)")
    p.add_argument("--seed", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=500)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    import results
    import t2s_predict
    from env_utils import make_fixed_scene_env
    from stable_baselines3 import SAC

    t2s_dir = results.get_run_dir("t2s_model", args.t2s_run)
    mpath = os.path.join(t2s_dir, STAGE_MANIFEST)
    if os.path.exists(mpath):
        with open(mpath) as f:
            man = json.load(f)
    else:
        import t2s_io
        man = t2s_io.load_manifest(t2s_dir)
    combos = args.models or sorted(man["combos"])

    # the success expert visits every stage except hovering
    try:
        ed = results.get_run_dir("t2s_eval", args.t2s_run)
        with open(os.path.join(ed, STAGE_MANIFEST)) as f:
            succ_ckpt = json.load(f)["success_ckpt"]
    except Exception:
        raise SystemExit("ERROR: need a t2s_eval manifest for the success policy; "
                         f"run  python run_3_eval.py --t2s-run {args.t2s_run}")
    import config
    pol = SAC.load(os.path.join(config.EXPERT_POLICY_DIR, succ_ckpt))

    # one rollout, shared by every model
    env = make_fixed_scene_env()
    obs, _ = env.reset(seed=args.seed)
    seq, success_step = [], None
    for t in range(args.max_steps):
        seq.append(np.asarray(obs, dtype=np.float32).copy())
        a, _ = pol.predict(obs, deterministic=False)
        obs, _r, term, trunc, info = env.step(a)
        if info.get(SUCCESS_KEY, 0) and success_step is None:
            success_step = t + 1
        if term or trunc or (success_step is not None and t > success_step + 10):
            break
    seq.append(np.asarray(obs, dtype=np.float32).copy())
    env.close()
    seq = np.array(seq)

    stages = classify(seq, success_step)
    synth = synth_hovering(seq, stages)
    print(f"=== interaction stages found in one expert rollout "
          f"({len(seq)} frames, success@{success_step}) ===")
    for k in ORDER:
        print(f"  {k:<12}{len(stages[k]):>4} frames"
              + ("   (none — an expert does not hover)"
                 if k == "hovering" and not stages[k] else ""))
    if synth is not None:
        print("  hovering_synth  1 frame   (a real grasped/lifted pose with the "
              "peg put back at its reset position)")

    print(f"\n{'combo':<22}" + "".join(f"{k[:9]:>11}" for k in ORDER)
          + f"{'hov_synth':>11}  ordering")
    print("-" * (22 + 11 * 6 + 12))
    verdicts = {}
    for c in combos:
        fn = t2s_predict.load_t2s_predictor(
            t2s_dir, *c.rsplit("_", 1), seed=man["combos"][c].get("best_seed", 0))
        v = score_model(fn, seq, stages, synth)
        ok, pairs = check_order(v)
        verdicts[c] = dict(values=v, ok=ok,
                           pairs=[(a, b, va, vb, p) for a, b, va, vb, p in pairs])
        row = f"{c:<22}"
        for k in ORDER:
            row += (f"{v[k]:>11.1f}" if v[k] is not None else f"{'--':>11}")
        row += (f"{v['hovering_synth']:>11.1f}" if "hovering_synth" in v
                else f"{'--':>11}")
        print(row + ("  PASS" if ok else "  FAIL"))

    print("\n--- where the ordering breaks ---")
    for c, d in verdicts.items():
        bad = [(a, b, va, vb) for a, b, va, vb, p in d["pairs"] if not p]
        if not bad:
            print(f"  {c}: ordering holds across every stage present")
            continue
        for a, b, va, vb in bad:
            print(f"  {c}: T({a})={va:.1f} is NOT greater than T({b})={vb:.1f} "
                  f"— the model does not know that {b} is progress over {a}")

    # The REAL-frame gap is the honest one; the synthetic counterfactual is
    # off-distribution by construction and is reported with its z-score so a
    # large gap there is not mistaken for understanding.
    print("\n--- hovering vs grasped, REAL frames (in-distribution) ---")
    for c, d in verdicts.items():
        v = d["values"]
        hv, gr = v.get("hovering"), v.get("grasped")
        if hv is None or gr is None:
            print(f"  {c}: one of the stages is absent from this rollout")
            continue
        n = max((v.get("n_hovering", 0) + v.get("n_grasped", 0)) / 2, 1)
        per = (hv - gr) / n
        # The right comparison is the reward for PROGRESSING against the reward
        # for HOVERING, not the decrement against some absolute bar. An earlier
        # version flagged "per < 1.1" as weak, which condemns 1.00/step — the
        # ideal decrement — as insufficient. The ideal is 1.0; what matters is
        # the gap it opens over standing still.
        import reward_fn as _R
        r_prog = _R.step_reward(hv, hv - per, "difference_timed",
                                gamma=_R.DEFAULT_GAMMA)
        r_hov = _R.hover_reward(hv, gamma=_R.DEFAULT_GAMMA)
        gap = r_prog - r_hov
        print(f"  {c}: hovering {hv:.1f} vs grasped {gr:.1f}  "
              f"decrement {per:.2f}/step (ideal 1.00)")
        print(f"      reward: grasping {r_prog:+.2f} vs hovering {r_hov:+.2f}  "
              f"-> gap {gap:+.2f}/step"
              + ("   <<< WEAK: little incentive to grasp" if gap < 0.3 else
                 "   (a clear incentive to grasp)"))

    zs = offdist_z(synth, t2s_dir) if synth is not None else None
    print("\n--- the synthetic counterfactual (arm pose held, peg put back) ---")
    if zs is not None:
        print(f"  max|z| of this state against training data: {zs:.1f}"
              + ("   <<< OFF-DISTRIBUTION: a large gap here may be extrapolation"
                 if zs > 3 else ""))
        print("  NOTE: z-scores are PER DIMENSION. A state whose every coordinate")
        print("  is typical but whose COMBINATION never occurs still scores low,")
        print("  so a low max|z| does not establish that this pose-with-peg-at-rest")
        print("  is something the model was ever trained on.")
        print("  Also: no policy traverses this transition. A real grasp goes")
        print("  hovering -> grasped (the row above); the peg does not teleport")
        print("  from the table into a raised gripper. Read this as evidence the")
        print("  POTENTIAL separates the states, not as the reward a policy earns.")
    for c, d in verdicts.items():
        v = d["values"]
        hs, gr = v.get("hovering_synth"), (v.get("lifted") or v.get("grasped"))
        if hs is None or gr is None:
            continue
        print(f"  {c}: hovering {hs:.1f} vs holding {gr:.1f}  gap {hs - gr:+.1f}")

    # ---- the non-circular test -------------------------------------------
    try:
        with open(os.path.join(ed, STAGE_MANIFEST)) as f:
            fail_ckpt = json.load(f)["failure_ckpt"]
        fpol = SAC.load(os.path.join(config.EXPERT_POLICY_DIR, fail_ckpt))
        env = make_fixed_scene_env()
        o, _ = env.reset(seed=args.seed)
        fseq = []
        for _t in range(args.max_steps):
            fseq.append(np.asarray(o, dtype=np.float32).copy())
            a, _ = fpol.predict(o, deterministic=False)
            o, _r, term, trunc, _i = env.step(a)
            if term or trunc:
                break
        env.close()
        pairs = matched_pairs(seq, np.array(fseq))
    except Exception as e:
        pairs = []
        print(f"\n(matched-pair test skipped: {e})")

    if pairs:
        print(f"\n--- MATCHED FRAMES: same hand pose, different peg state "
              f"({len(pairs)} pairs) ---")
        print("    this removes TIME from the comparison. The stage ordering above")
        print("    follows from temporal accuracy alone; this does not.")
        for c in combos:
            fn = t2s_predict.load_t2s_predictor(
                t2s_dir, *c.rsplit("_", 1), seed=man["combos"][c].get("best_seed", 0))
            ds = []
            for pr in pairs:
                ps = float(fn(seq[pr["succ_idx"]]))
                pf = float(fn(np.array(fseq)[pr["fail_idx"]]))
                ds.append(pf - ps)          # failure should be HIGHER
            ds = np.asarray(ds)
            frac = float(np.mean(ds > 0))
            print(f"  {c}: peg-not-carried minus peg-carried = {ds.mean():+.1f} "
                  f"+/- {ds.std():.1f}, higher in {frac:.0%} of pairs"
                  + ("   <<< the model gives the SAME prediction whether or not the "
                     "peg came along" if abs(ds.mean()) < 3 else
                     "   (the model does distinguish them)"))

    out = os.path.join(results.get_run_dir("t2s_eval", args.t2s_run), "ordering.json")
    with open(out, "w") as f:
        json.dump(dict(seed=args.seed, success_step=success_step,
                       stage_counts={k: len(stages[k]) for k in ORDER},
                       models=verdicts), f, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())