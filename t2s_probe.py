#!/usr/bin/env python3
"""
Why does a T2S model not react when the arm moves?

    python t2s_probe.py --t2s-run v3
    python t2s_probe.py --t2s-run v3 --models td0_all tdlambdaboot_succ

step_slope_error is measured ALONG EXPERT TRAJECTORIES, where the state is
almost a deterministic function of the frame index. A model can score well
there by learning "which frame of a demonstration does this look like" and
still be completely blind to a displacement no demonstration contains. This
script separates the two, and distinguishes four different reasons a
prediction can sit still while the arm moves:

  1. THE MODEL IGNORES THE INPUT DIMENSIONS THAT MOVED.
     `sensitivity` perturbs each observation group (hand, peg, goal, rest) and
     measures how much the prediction changes. A model whose d(pred)/d(hand)
     is ~0 does not use hand position, full stop. This is the direct answer,
     and it is the first thing to check.

  2. THE STATE IS OFF THE TRAINING DISTRIBUTION AND THE NETWORK EXTRAPOLATES
     FLAT. `coverage_z` reports the normalized input magnitude against the
     training statistics in normalization.npz. Past roughly |z| > 3 the
     encoder is being asked about a region it never saw; a ReLU MLP typically
     responds with something locally constant, which looks exactly like
     insensitivity but has a different fix (collect those states) than cause 1
     (change the inputs or the objective).

  3. THE PREDICTION SATURATES. `t2s_head` ends in Softplus and t2s_predict
     clips to [0, T2S_PRED_CLIP_MAX]. A model pinned at the clip ceiling, or
     driven deep into Softplus's flat region, cannot move regardless of input.
     `saturation` reports how close the raw output sits to either bound.

  4. THE OBJECTIVE NEVER REQUIRED IT. With the "succ" condition every training
     trajectory succeeds, so steps-remaining is almost perfectly predicted by
     progress along a stereotyped path — a model can fit it without ever
     learning what "further away" means, because the dataset contains no
     further-away states to be wrong about. `sweep_axis` tests this directly:
     it translates the hand along one axis and plots the prediction, so you
     see whether the model believes distance matters at all.

`sweep_axis` is the decisive one because it is causal rather than
correlational: the same base state, one coordinate changed, everything else
held fixed. A rollout can never isolate that.
"""
import argparse
import json
import os
import sys

import numpy as np

from config import HAND_POS_IDX, PEG_POS_IDX, GOAL_POS_IDX, OBS_DIM, T2S_PRED_CLIP_MAX

STAGE_MANIFEST = "stage_manifest.json"

# MetaWorld's 39-dim observation is [current(18) | previous(18) | goal(3)] —
# verified by env_utils.check_obs_contains_history, which found obs[18:36] at
# step t exactly equal to obs[0:18] at t-1. So velocity is inferable as
# obs[0:18] - obs[18:36], and a perturbation that touches only the CURRENT
# half changes the implied velocity as well as the position.
PREV_OFFSET = 18

# Perturbation modes, because "move the hand" is two different experiments:
#   static  — move current AND previous together. Position changes, implied
#             velocity stays zero. This is "the arm is somewhere else".
#   current — move only the current half, as the first version of this script
#             did. Position AND implied velocity change together, so a
#             displacement of 0.5 implies a one-step velocity of 0.5, far
#             outside anything in training. |velocity| also rises in BOTH
#             directions from zero, which produces a symmetric V shape that is
#             easy to misread as a distance-from-training-data artefact.
#             Kept, because velocity sensitivity is worth measuring — just not
#             worth confusing with position sensitivity.
def paired(idxs):
    """Current indices plus their previous-frame counterparts."""
    return list(idxs) + [i + PREV_OFFSET for i in idxs]


GROUPS = {
    "hand": paired(HAND_POS_IDX),
    "peg": paired(PEG_POS_IDX),
    "goal": GOAL_POS_IDX,                      # goal has no previous copy
    "other": [i for i in range(OBS_DIM)
              if i not in paired(HAND_POS_IDX) + paired(PEG_POS_IDX) + GOAL_POS_IDX],
}


def sensitivity(predict_fn, obs, eps=0.02, groups=GROUPS, x_std=None):
    """
    Finite-difference sensitivity of the prediction to each observation group.

    Returns {group: mean |pred(obs + eps) - pred(obs)|} over +/- perturbations
    of every index in the group. eps=0.02 is a couple of centimetres in sim
    units — a displacement the arm covers in a few steps, so the answer is
    "does a real movement change the prediction", not "is the function
    differentiable".

    A group scoring ~0 is not being used. That is a stronger statement than
    any trajectory metric can make, because everything else is held fixed.

    x_std: per-dimension training std from normalization.npz. Pass it. A fixed
    2cm step is meaningless on a dimension whose training std is 1e-9 — that is
    a z-score of 10^7, so the measurement is extrapolation into nowhere rather
    than sensitivity, and the index reads as "unused" when it was never
    plausibly perturbed. With x_std the step becomes eps_z * std_i, i.e. the
    same distance in units the model was actually trained on. Dimensions whose
    training std is ~0 are SKIPPED and reported, because no perturbation of
    them is in-distribution.
    """
    base = float(predict_fn(obs))
    out, skipped = {}, []
    for name, idxs in groups.items():
        deltas = []
        for i in idxs:
            step = eps if x_std is None else float(x_std[i]) * 2.0
            if step < 1e-6:                     # never varied in training
                skipped.append(i)
                continue
            for sign in (+1, -1):
                o = np.asarray(obs, dtype=np.float32).copy()
                o[i] += sign * step
                deltas.append(abs(float(predict_fn(o)) - base))
        out[name] = float(np.mean(deltas)) if deltas else None
    out["_base"] = base
    out["_skipped"] = sorted(skipped)
    return out


def sweep_axis(predict_fn, obs, idxs=HAND_POS_IDX, axis=1, span=0.5, n=25,
               mode="static"):
    """
    Translate one coordinate across +/- span and return (offsets, predictions).

    mode="static" (default) moves the CURRENT and PREVIOUS copies together, so
    the arm is genuinely somewhere else with zero implied velocity. This is the
    position test.

    mode="current" moves only the current copy, which is what the first version
    of this script did unintentionally. That implies a one-step velocity equal
    to the displacement — 0.5 sim units per step is far outside training — and
    since |velocity| grows in both directions, it produces a symmetric V around
    zero displacement whatever the model has learnt about position. Run both:
    a V under "current" that FLATTENS under "static" means the model is reading
    velocity, not distance.
    """
    offsets = np.linspace(-span, span, n)
    targets = ([idxs[axis]] if mode == "current"
               else [idxs[axis], idxs[axis] + PREV_OFFSET])
    preds = []
    for d in offsets:
        o = np.asarray(obs, dtype=np.float32).copy()
        for t in targets:
            o[t] += d
        preds.append(float(predict_fn(o)))
    return offsets, np.array(preds)


def sweep_geometry(obs, idxs=HAND_POS_IDX, axis=1, span=0.5, n=25):
    """
    The TRUE geometry along the same sweep: hand-to-peg and peg-to-goal
    distance as the hand is translated.

    This is the control the sweep was missing. A V-shaped prediction centred
    near zero displacement looks suspicious — until you check whether the hand
    starts roughly ALIGNED with the peg on this axis, in which case moving it
    either way increases hand-peg distance and a real time-to-success function
    SHOULD be V-shaped. Without this curve there is no way to tell a model
    reading distance-from-training-data from a model reading distance-to-peg,
    because both produce a V.

    Returns (offsets, hand_peg, peg_goal).
    """
    from progress import hand_peg_distance, peg_goal_distance

    offsets = np.linspace(-span, span, n)
    hp, pg = [], []
    for d in offsets:
        o = np.asarray(obs, dtype=np.float32).copy()
        o[idxs[axis]] += d
        o[idxs[axis] + PREV_OFFSET] += d
        hp.append(hand_peg_distance(o))
        pg.append(peg_goal_distance(o))
    return offsets, np.array(hp), np.array(pg)


def stale_target_test(predict_fn, obs, displace=0.20, axis=1, n=25):
    """
    Has the model learned "hand near the PEG", or "hand near where the peg
    STARTED"?

    The distinction is invisible on any demonstration, because in every
    demonstration the peg is at its start when the hand approaches it. It
    matters enormously during RL: if the model tracks the fixed start
    location, then a policy that shoves the peg away and then returns an
    EMPTY gripper to the peg's original spot still collects the approach
    reward, over and over.

    relative_vs_absolute already found the structure that implies this:
    pred = f(hand) + g(peg), additive in ABSOLUTE positions rather than a
    function of their difference. An additive model cannot express "hand near
    peg"; it can only express "hand near a fixed good place" plus "peg near a
    fixed good place".

    THE TEST: displace the peg, then sweep the hand between the peg's OLD
    position and its NEW one.
      pred lowest at the NEW peg position -> the model tracks the peg
      pred lowest at the OLD peg position -> the model tracks a stale location,
                                             and an empty gripper is rewarded
                                             for returning to it
    """
    o = np.asarray(obs, dtype=np.float32)
    peg_old = float(o[PEG_POS_IDX[axis]])
    peg_new = peg_old + displace

    moved = o.copy()                      # peg displaced, hand untouched
    moved[PEG_POS_IDX[axis]] = peg_new
    moved[PEG_POS_IDX[axis] + PREV_OFFSET] = peg_new

    xs = np.linspace(peg_old - 0.1, peg_new + 0.1, n)
    preds = []
    for x in xs:
        p = moved.copy()
        p[HAND_POS_IDX[axis]] = x
        p[HAND_POS_IDX[axis] + PREV_OFFSET] = x
        preds.append(float(predict_fn(p)))
    preds = np.array(preds)

    at_old = float(np.interp(peg_old, xs, preds))
    at_new = float(np.interp(peg_new, xs, preds))
    argmin_x = float(xs[int(np.argmin(preds))])
    # 0 = the minimum sits at the peg's old spot, 1 = at its new spot
    frac = float((argmin_x - peg_old) / displace) if displace else None
    return dict(peg_old=peg_old, peg_new=peg_new, pred_at_old=at_old,
                pred_at_new=at_new, argmin_at=argmin_x, tracks_peg_frac=frac,
                offsets=(xs - peg_old).tolist(), preds=preds.tolist(),
                verdict=("tracks the PEG (minimum follows it)" if frac is not None
                         and frac > 0.6 else
                         "STALE: minimum stays at the peg's ORIGINAL position — an "
                         "empty gripper returning there is rewarded"
                         if frac is not None and frac < 0.4 else "mixed"))


def lift_sensitivity(predict_fn, obs, span=0.15, n=21, coupled=True):
    """
    coupled=True (default) raises the HAND WITH THE PEG, as a real grasp does.

    The first version raised the peg alone. In every training trajectory the
    peg only leaves the table while the gripper holds it, so peg height and
    hand height are near-perfectly correlated, and "peg at 0.15 with the hand
    still at table height" is a state that cannot physically occur and never
    appears in the data. A network asked about it answers "unfamiliar", which
    reads as a high predicted time-to-success — so the uncoupled sweep
    measures off-manifold surprise, not an opinion about lifting.

    That is the same confound flagged for the hovering counterfactual, and it
    is likely the source of an 11-of-12 "lifting RAISES the prediction"
    verdict across a whole run. Set coupled=False to reproduce the old
    behaviour and compare.
    """
    """
    Does the prediction FALL when the peg is lifted off the table?

    peg-insert-side requires the peg to leave the table and enter a hole in the
    side of the box. A policy that shoves the peg along the table can reduce
    peg-to-goal distance without ever picking it up — and control_rate cannot
    tell the two apart, because a pushing hand also moves with the peg at a
    constant offset.

    So the question for the REWARD is narrower and sharper than "does it use
    the peg": does lifting the peg lower the predicted time-to-success? If it
    does, the shaping pays for lifting and the policy can find it. If the
    response is flat or rising, the model is indifferent to whether the peg is
    on the table and no amount of training will produce a grasp — the reward
    is not asking for one.

    Sweeps PEG_POS_IDX[2] (height) with everything else fixed, current and
    previous frame together.

    Returns lift_gradient: change in prediction per metre of lift, over the
    upward half. NEGATIVE is what you want.
    """
    if coupled:
        offs = np.linspace(-span, span, n)
        preds = []
        for d in offs:
            o = np.asarray(obs, dtype=np.float32).copy()
            for idxs in (PEG_POS_IDX, HAND_POS_IDX):
                o[idxs[2]] += d
                o[idxs[2] + PREV_OFFSET] += d
            preds.append(float(predict_fn(o)))
        preds = np.array(preds)
    else:
        offs, preds = sweep_axis(predict_fn, obs, idxs=PEG_POS_IDX, axis=2,
                                 span=span, n=n, mode="static")
    up = offs >= 0
    grad = (float(np.polyfit(offs[up], preds[up], 1)[0])
            if up.sum() >= 2 and np.std(preds[up]) > 0 else None)
    at0 = float(np.interp(0.0, offs, preds))
    at_top = float(preds[-1])
    return dict(lift_offsets=offs.tolist(), lift_preds=preds.tolist(),
                lift_gradient=grad, pred_at_table=at0, pred_at_top=at_top,
                # endpoint difference: a linear fit over a non-monotonic curve
                # can report a confident negative slope while the curve ends
                # where it started (mcboot_all: fit -251, endpoints -2.8)
                lift_endpoint_drop=at_top - at0,
                lift_monotonic=bool(np.all(np.diff(preds[up]) <= 1e-9)
                                    or np.all(np.diff(preds[up]) >= -1e-9)),
                coupled=coupled,
                lift_pays=bool(grad is not None and grad < 0),
                verdict=("lifting LOWERS the prediction — the reward pays for it"
                         if grad is not None and grad < -1 else
                         "lifting RAISES the prediction — the reward penalises "
                         "picking the peg up" if grad is not None and grad > 1 else
                         "indifferent to peg height — no grasp gradient exists"))


def relative_vs_absolute(predict_fn, obs, axis=1, span=0.3, n=21):
    """
    Does the model use hand-peg GEOMETRY, or absolute hand position?

    The hand sweep cannot answer this. Moving the hand alone changes both the
    absolute hand position and the hand-peg distance, so a V-shaped response
    is equally consistent with

        model A: pred = f(||p_hand - p_peg||)   knows relative geometry
        model B: pred = f(p_hand)               memorised where the hand sits
                                                 in a demonstration

    and both put their minimum where the demos start. They differ on exactly
    the question that matters for the hovering failure: model B cannot tell
    "hand in the hole, peg untouched" from "peg in the hole", because the peg
    never enters its computation.

    THE TEST: translate hand AND peg together by the same offset. The relative
    geometry is unchanged, so model A is invariant and model B is not.

    Returns:
      cotranslation_range  prediction span under joint translation.
                           ~0 => relative. Large => absolute position.
      peg_only_range       span when ONLY the peg moves. ~0 means the peg is
                           not used at all, which is the strongest form of the
                           same defect.
      relative_score       1.0 = purely relative, 0.0 = purely absolute.
    """
    offs = np.linspace(-span, span, n)
    base = float(predict_fn(obs))

    def sweep(move_hand, move_peg):
        out = []
        for d in offs:
            o = np.asarray(obs, dtype=np.float32).copy()
            if move_hand:
                o[HAND_POS_IDX[axis]] += d
                o[HAND_POS_IDX[axis] + PREV_OFFSET] += d
            if move_peg:
                o[PEG_POS_IDX[axis]] += d
                o[PEG_POS_IDX[axis] + PREV_OFFSET] += d
            out.append(float(predict_fn(o)))
        return np.array(out)

    both = sweep(True, True)         # geometry preserved
    hand = sweep(True, False)        # geometry changes
    peg = sweep(False, True)         # geometry changes, hand fixed

    r_both = float(both.max() - both.min())
    r_hand = float(hand.max() - hand.min())
    r_peg = float(peg.max() - peg.min())
    # a relative model has r_both ~ 0 while r_hand is large
    score = float(1.0 - min(r_both / r_hand, 1.0)) if r_hand > 1e-9 else None

    # "mixed" was too coarse. The three patterns are distinguishable:
    #   relative  f(hand - peg)        both ~ 0
    #   absolute  f(hand)              both ~ hand, peg ~ 0
    #   additive  f(hand) + g(peg)     both ~ hand + peg, BOTH terms real
    # The additive case matters because it still SEPARATES peg-in-hole from
    # hand-in-hole — the peg has its own term — even though the model has no
    # notion of relative geometry. That is fine on a pinned scene and would
    # break if the goal moved.
    if max(r_both, r_hand, r_peg) < 0.5:
        verdict = "no response to anything"
    elif r_peg < 0.2 * r_hand:
        verdict = "PEG UNUSED — absolute hand position only"
    elif score is not None and score > 0.7:
        verdict = "RELATIVE — uses hand-peg geometry"
    elif r_both > 0.7 * (r_hand + r_peg):
        verdict = "ADDITIVE in absolute positions (peg IS used)"
    else:
        verdict = "mixed"
    return dict(cotranslation_range=r_both, hand_only_range=r_hand,
                peg_only_range=r_peg, relative_score=score,
                additive_ratio=(float(r_both / (r_hand + r_peg))
                                if (r_hand + r_peg) > 1e-9 else None),
                base_pred=base, verdict=verdict)


def geometry_alignment(offsets, preds, hand_peg):
    """
    Score the sweep against the TRUE geometry instead of eyeballing it.

    With the control curve overlaid, "is there a V" stops being the question —
    the true hand-peg distance is V-shaped whenever the hand starts aligned
    with the peg, so a V is correct. What is left to check:

      geometry_rho   Spearman(prediction, true hand-peg distance) over the
                     sweep. 1.0 means the model orders these states exactly as
                     the geometry does. This is the ranking question the
                     trajectory-based correlations could never ask, because
                     along a rollout everything moves at once.
      argmin_error   how far the model's minimum sits from the true minimum.
      left_response  prediction span on the NEGATIVE side, as a fraction of
      right_response the true distance span on that side. 1.0 = proportionate.
                     These are reported separately because a model can be
                     blind on one side and saturated on the other, which
                     averages out to something that looks fine.
      asymmetry      |left - right|. The true curve is near-symmetric about
                     its minimum, so a large value is a real defect rather
                     than the geometry showing through.
    """
    from scipy.stats import spearmanr

    offsets = np.asarray(offsets, float)
    preds = np.asarray(preds, float)
    hp = np.asarray(hand_peg, float)
    imin = int(np.argmin(hp))

    def side(lo, hi):
        p, d = preds[lo:hi], hp[lo:hi]
        if len(p) < 2 or (d.max() - d.min()) <= 0:
            return None
        # prediction span per unit of true distance span
        return float((p.max() - p.min()) / (d.max() - d.min()))

    left, right = side(0, imin + 1), side(imin, len(preds))
    rho = (float(spearmanr(preds, hp)[0])
           if np.std(preds) > 0 and np.std(hp) > 0 else None)
    return dict(
        geometry_rho=rho,
        argmin_error=float(offsets[int(np.argmin(preds))] - offsets[imin]),
        left_response=left, right_response=right,
        asymmetry=(abs(left - right) if left is not None and right is not None else None),
    )


def coverage_z(obs, run_dir):
    """
    How far outside the training distribution this state sits, in units of the
    per-dimension training std stored in normalization.npz.

    |z| under ~2 is inside the data. Past ~3 the network is extrapolating, and
    a flat extrapolation is normal MLP behaviour rather than a broken model —
    different diagnosis, different fix.
    """
    from t2s_io import load_normalization
    X_mean, X_std = load_normalization(run_dir)
    z = (np.asarray(obs, dtype=np.float32) - X_mean) / X_std
    return dict(max_abs_z=float(np.abs(z).max()),
                mean_abs_z=float(np.abs(z).mean()),
                n_dims_beyond_3=int(np.sum(np.abs(z) > 3.0)),
                worst_dims=[int(i) for i in np.argsort(-np.abs(z))[:5]])


def saturation(predict_fn, obs, clip_max=T2S_PRED_CLIP_MAX):
    """
    Is the prediction pinned against a bound? clip=False asks the predictor for
    its raw output, so a value at or beyond the ceiling is visible rather than
    silently clamped.
    """
    raw = float(predict_fn(obs, clip=False))
    return dict(raw=raw, clipped=float(np.clip(raw, 0.0, clip_max)),
                at_ceiling=bool(raw >= clip_max * 0.99),
                near_zero=bool(raw <= 0.5))


def probe_model(predict_fn, obs, run_dir, label=""):
    """All four checks for one model at one state."""
    from t2s_io import load_normalization
    try:
        _m, x_std = load_normalization(run_dir)
    except Exception:
        x_std = None                            # fall back to a fixed step
    s = sensitivity(predict_fn, obs, x_std=x_std)
    offs, static = sweep_axis(predict_fn, obs, mode="static")
    _o, current = sweep_axis(predict_fn, obs, mode="current")
    return dict(
        label=label, sensitivity=s, coverage=coverage_z(obs, run_dir),
        saturation=saturation(predict_fn, obs),
        sweep_range=float(static.max() - static.min()),
        sweep_range_current=float(current.max() - current.min()),
        sweep_offsets=offs.tolist(), sweep_preds=static.tolist(),
        sweep_preds_current=current.tolist(),
        # where the model thinks it is closest, vs where it actually is
        argmin_offset=float(offs[int(np.argmin(static))]),
        **geometry_alignment(offs, static, sweep_geometry(obs)[1]),
        **{f"rel_{k}": v for k, v in relative_vs_absolute(predict_fn, obs).items()},
        **{f"lift_{k}" if not k.startswith("lift_") else k: v
           for k, v in lift_sensitivity(predict_fn, obs).items()},
        **{f"stale_{k}": v for k, v in stale_target_test(predict_fn, obs).items()},
    )


def format_probe(rows, flat_threshold=0.5):
    """One table. `flat` is the verdict the script exists to deliver."""
    hdr = (f"{'combo':<22}{'pred':>8}{'d/d hand':>10}{'d/d peg':>9}{'d/d other':>11}"
           f"{'span:static':>13}{'span:vel':>10}{'max|z|':>8}  verdict")
    lines = [hdr, "-" * (len(hdr) + 20)]
    for r in rows:
        s, c = r["sensitivity"], r["coverage"]
        rng = r["sweep_range"]
        responds_elsewhere = max(v for k, v in s.items()
                                 if not k.startswith("_") and k != "hand"
                                 and v is not None) > 0.05
        # order matters: a model that reacts to peg/goal but not to the hand is
        # cause 1 (wrong inputs), NOT cause 2 (extrapolation) — it is plainly
        # computing something from the state, just not from the hand. Testing
        # the off-distribution branch first mislabelled exactly that case.
        vel_rng = r.get("sweep_range_current", 0.0)
        if r["saturation"]["at_ceiling"]:
            verdict = "SATURATED at the clip ceiling"
        elif rng < flat_threshold and vel_rng > 5 * max(rng, flat_threshold):
            # not state-blind: it reacts strongly to the current-frame-only
            # perturbation and not at all to the static one, which is a model
            # keyed on implied VELOCITY. |velocity| grows in both directions
            # from zero, so this is also what produces a symmetric V in the
            # sweep plot without the model knowing anything about distance.
            verdict = "reads VELOCITY, not position"
        elif s["hand"] < 0.05 and responds_elsewhere:
            verdict = "IGNORES HAND — reacts to peg/goal only"
        elif rng < flat_threshold and not responds_elsewhere:
            verdict = ("STATE-BLIND, and off-distribution (collect these states)"
                       if c["max_abs_z"] > 3 else
                       "STATE-BLIND in-distribution — output barely uses the state")
        elif rng < flat_threshold:
            verdict = "flat along hand-y, but reacts to other inputs"
        else:
            verdict = "responsive"
        lines.append(
            f"{r['label']:<22}{s['_base']:>8.1f}{s['hand']:>10.3f}{s['peg']:>9.3f}"
            f"{(s['other'] if s['other'] is not None else float('nan')):>11.3f}{rng:>13.2f}"
            f"{r.get('sweep_range_current', float('nan')):>10.2f}"
            f"{c['max_abs_z']:>8.1f}  {verdict}")
    lines += [
        "",
        "d/d X       : mean |change| when that group (current AND previous frame)",
        "              moves 2cm. ~0 = unused. 'other' covers the history block.",
        "span:static : hand moved with its previous copy — position changes,",
        "              implied velocity stays 0. THE position test.",
        "span:vel    : only the current copy moved, so the displacement also",
        "              implies a one-step velocity. A large span here with a small",
        "              span:static means the model reads VELOCITY, not distance.",
        "max|z|      : distance from the training distribution. >3 = extrapolating,",
        "              so a flat response is expected MLP behaviour, not a bug.",
    ]
    return "\n".join(lines)


def format_stale(rows):
    """Does the model follow the peg, or a memorised location?"""
    hdr = (f"{'combo':<22}{'pred@old peg':>14}{'pred@new peg':>14}"
           f"{'min tracks peg':>16}  verdict")
    lines = [hdr, "-" * (len(hdr) + 28)]
    for r in sorted(rows, key=lambda r: -(r.get("stale_tracks_peg_frac") or -1)):
        f = r.get("stale_tracks_peg_frac")
        lines.append(f"{r['label']:<22}{(r.get('stale_pred_at_old') or 0):>14.1f}"
                     f"{(r.get('stale_pred_at_new') or 0):>14.1f}"
                     f"{(f'{f:.0%}' if f is not None else '--'):>16}"
                     f"  {r.get('stale_verdict', '--')}")
    lines += [
        "",
        "The peg is displaced 20cm, then the hand is swept between the peg's OLD",
        "and NEW positions. Every demonstration has the peg at its start when the",
        "hand arrives, so this distinction is invisible in training data — and it",
        "decides whether a policy can farm approach reward by pushing the peg away",
        "and returning an empty gripper to where it used to be.",
    ]
    return "\n".join(lines)


def format_lift(rows):
    """Does the reward pay for picking the peg UP? The grasp-specific question."""
    hdr = (f"{'combo':<22}{'pred on table':>15}{'pred lifted':>13}"
           f"{'endpoint':>10}{'d/d height':>12}{'mono':>6}  verdict")
    lines = [hdr, "-" * (len(hdr) + 30)]
    for r in sorted(rows, key=lambda r: (r.get("lift_gradient") or 0)):
        g = r.get("lift_gradient")
        # keys arrive prefixed from probe_model (lift_pred_at_table, ...)
        lines.append(f"{r['label']:<22}"
                     f"{(r.get('lift_pred_at_table', r.get('pred_at_table')) or 0):>15.1f}"
                     f"{(r.get('lift_pred_at_top', r.get('pred_at_top')) or 0):>13.1f}"
                     f"{(r.get('lift_endpoint_drop') or 0):>10.1f}"
                     f"{(f'{g:+.0f}' if g is not None else '--'):>12}"
                     f"{('y' if r.get('lift_monotonic') else 'N'):>6}"
                     f"  {r.get('lift_verdict', '--')}")
    lines += [
        "",
        "The hand is raised WITH the peg, as a grasp does. Raising the peg alone",
        "is physically impossible and off-manifold, and measures surprise rather",
        "than an opinion about lifting.",
        "endpoint : prediction at the top minus at the table. Read it WITH d/d",
        "           height — a linear fit over a non-monotonic curve (mono=N) can",
        "           report a confident slope while the curve ends where it began.",
        "peg-insert-side needs the peg LIFTED into a hole in the side of the box.",
        "Pushing it along the table reduces peg->goal distance without picking it",
        "up, and control_rate cannot tell those apart. d/d height must be NEGATIVE",
        "for the shaping reward to pay for a grasp at all.",
    ]
    return "\n".join(lines)


def format_relative(rows):
    """The hand-vs-geometry test. This is the one that speaks to hovering."""
    hdr = (f"{'combo':<22}{'hand only':>11}{'peg only':>10}{'both (want ~0)':>16}"
           f"{'relative':>10}  verdict")
    lines = [hdr, "-" * (len(hdr) + 18)]
    for r in sorted(rows, key=lambda r: -(r.get("rel_relative_score") or -1)):
        f = lambda k, p=2: ("--" if r.get(k) is None else f"{r[k]:.{p}f}")
        lines.append(f"{r['label']:<22}{f('rel_hand_only_range', 1):>11}"
                     f"{f('rel_peg_only_range', 1):>10}"
                     f"{f('rel_cotranslation_range', 1):>16}"
                     f"{f('rel_relative_score'):>10}  {r.get('rel_verdict', '--')}")
    lines += [
        "",
        "hand only : prediction span when only the hand moves (geometry changes)",
        "peg only  : span when only the PEG moves. ~0 = the peg is never used,",
        "            so the model cannot tell 'peg in hole' from 'hand in hole'.",
        "both      : span when hand AND peg translate together. The geometry is",
        "            UNCHANGED, so a relative model should read ~0 here. A large",
        "            value means the model tracks absolute hand position.",
        "relative  : 1.0 purely relative, 0.0 purely absolute.",
        "ADDITIVE means the model has separate terms for hand and peg rather than",
        "their difference. It still separates 'peg in hole' from 'hand in hole',",
        "so a grasp gradient EXISTS; it just encodes absolute layout, which is",
        "valid on a pinned scene and would fail if the goal moved.",
    ]
    return "\n".join(lines)


def format_geometry(rows):
    """Sweep scored against the true geometry. The eyeball test, as numbers."""
    hdr = (f"{'combo':<22}{'geom rho':>10}{'argmin err':>12}{'left resp':>11}"
           f"{'right resp':>12}{'asymmetry':>11}  reading")
    lines = [hdr, "-" * (len(hdr) + 22)]
    for r in sorted(rows, key=lambda r: -(r.get("geometry_rho") or -1)):
        f = lambda k, p=2: ("--" if r.get(k) is None else f"{r[k]:.{p}f}")
        lt, rt = r.get("left_response"), r.get("right_response")
        if lt is None or rt is None:
            reading = "--"
        elif max(lt, rt) < 60:
            reading = "under-responds on both sides"
        elif min(lt, rt) < 0.2 * max(lt, rt):
            reading = f"blind on the {'LEFT' if lt < rt else 'RIGHT'}"
        else:
            reading = "proportionate on both sides"
        lines.append(f"{r['label']:<22}{f('geometry_rho'):>10}{f('argmin_error'):>12}"
                     f"{f('left_response', 0):>11}{f('right_response', 0):>12}"
                     f"{f('asymmetry', 0):>11}  {reading}")
    lines += [
        "",
        "geom rho   : Spearman(prediction, TRUE hand-peg distance) over the sweep.",
        "             1.0 = orders these states exactly as the geometry does.",
        "argmin err : offset between the model's minimum and the true minimum.",
        "left/right : prediction span per unit of true distance span, each side of",
        "             the true minimum. Reported separately because blind-on-one-side",
        "             and saturated-on-the-other average out to something plausible.",
    ]
    return "\n".join(lines)


def plot_sweeps(rows, obs=None, save_path=None, show_velocity=False):
    """
    Prediction vs hand displacement, one line per model, with the TRUE
    hand-to-peg distance overlaid on a second axis.

    The geometry curve is the control. If it is itself V-shaped — which it is
    whenever the hand starts aligned with the peg on this axis — then a
    V-shaped prediction is CORRECT, not an artefact, and the thing to compare
    is where each model's minimum sits and how steeply it rises.

    show_velocity=True adds the current-frame-only sweep as dashed lines. Off
    by default: on the observed runs it tracked the static sweep closely, so
    it doubled the number of lines without adding information.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for r in rows:
        line, = ax.plot(r["sweep_offsets"], r["sweep_preds"], lw=1.8, label=r["label"])
        if show_velocity and "sweep_preds_current" in r:
            ax.plot(r["sweep_offsets"], r["sweep_preds_current"], lw=1.0, ls="--",
                    alpha=0.5, color=line.get_color())
    ax.axvline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("hand displacement along y (sim units), everything else fixed")
    ax.set_ylabel("predicted steps to success")
    ax.grid(alpha=0.3)

    subtitle = "a flat line means the prediction ignores hand position"
    if obs is not None:
        offs, hp, _pg = sweep_geometry(obs)
        ax2 = ax.twinx()
        ax2.plot(offs, hp, color="black", lw=2.5, ls=":", label="TRUE hand-peg distance")
        ax2.set_ylabel("true hand-to-peg distance (sim units)")
        ax2.legend(loc="upper center", fontsize=8)
        argmin = float(offs[int(np.argmin(hp))])
        ax.axvline(argmin, color="black", lw=1.2, ls="--", alpha=0.6)
        subtitle = (f"dotted black = TRUE hand-peg distance, minimum at "
                    f"{argmin:+.2f}; a V here means a V in the prediction is CORRECT")
    ax.set_title("Does the prediction depend on where the hand is?\n" + subtitle,
                 fontsize=10)
    ax.legend(fontsize=7.5, ncol=2, loc="upper left")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Probe why a T2S model does not react to motion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--t2s-run", default="v3")
    p.add_argument("--models", nargs="+", default=None,
                   help="combos to probe (default: all in the run)")
    p.add_argument("--at", default="reset", choices=["reset", "drive"],
                   help="state to probe at: the reset pose, or after a drive")
    p.add_argument("--drive", default="retreat",
                   help="drive script when --at drive (see manual_drive.py)")
    p.add_argument("--span", type=float, default=0.5, help="sweep half-width")
    p.add_argument("--no-plot", dest="plot", action="store_false", default=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    import results
    import t2s_predict

    t2s_dir = results.get_run_dir("t2s_model", args.t2s_run)
    with open(os.path.join(t2s_dir, STAGE_MANIFEST)) as f:
        stage2 = json.load(f)
    combos = args.models or sorted(stage2["combos"])

    # ---- the state to probe at ------------------------------------------
    from env_utils import make_fixed_scene_env
    if args.at == "reset":
        env = make_fixed_scene_env()
        obs, _ = env.reset(seed=0)
        env.close()
        where = "reset pose"
    else:
        from stable_baselines3 import SAC
        import config
        from manual_drive import parse_drive, drive_then_policy
        eval_dir = results.get_run_dir("t2s_eval", args.t2s_run)
        with open(os.path.join(eval_dir, STAGE_MANIFEST)) as f:
            stage3 = json.load(f)
        pol = SAC.load(os.path.join(config.EXPERT_POLICY_DIR, stage3["success_ckpt"]))
        roll = drive_then_policy(pol, parse_drive(args.drive), seed=0)
        obs = roll["obs"][roll["handover"]]
        where = f"after drive '{args.drive}' ({roll['handover']} steps)"

    from progress import hand_peg_distance, peg_goal_distance
    print(f"=== probing {len(combos)} model(s) at the {where} ===")
    print(f"    hand->peg {hand_peg_distance(obs):.4f}   "
          f"peg->goal {peg_goal_distance(obs):.4f}\n")

    rows = []
    for c in combos:
        fn = t2s_predict.load_t2s_predictor(t2s_dir, *c.rsplit("_", 1),
                                            seed=stage2["combos"][c]["best_seed"])
        rows.append(probe_model(fn, obs, t2s_dir, label=c))

    print(format_probe(rows))
    print()
    print(format_geometry(rows))
    print()
    print(format_relative(rows))
    print()
    print(format_lift(rows))
    print()
    print(format_stale(rows))

    out = os.path.join(results.get_run_dir("t2s_eval", args.t2s_run), "probe")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"probe_{args.at}.json"), "w") as f:
        json.dump(dict(where=where, rows=rows), f, indent=2)
    if args.plot:
        try:
            plot_sweeps(rows, obs=obs,
                        save_path=os.path.join(out, f"sweep_{args.at}.png"))
        except Exception as e:
            print(f"(plot skipped: {e})")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())