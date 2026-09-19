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

# observation groups worth perturbing separately
GROUPS = {
    "hand": HAND_POS_IDX,
    "peg": PEG_POS_IDX,
    "goal": GOAL_POS_IDX,
    "other": [i for i in range(OBS_DIM)
              if i not in HAND_POS_IDX + PEG_POS_IDX + GOAL_POS_IDX],
}


def sensitivity(predict_fn, obs, eps=0.02, groups=GROUPS):
    """
    Finite-difference sensitivity of the prediction to each observation group.

    Returns {group: mean |pred(obs + eps) - pred(obs)|} over +/- perturbations
    of every index in the group. eps=0.02 is a couple of centimetres in sim
    units — a displacement the arm covers in a few steps, so the answer is
    "does a real movement change the prediction", not "is the function
    differentiable".

    A group scoring ~0 is not being used. That is a stronger statement than
    any trajectory metric can make, because everything else is held fixed.
    """
    base = float(predict_fn(obs))
    out = {}
    for name, idxs in groups.items():
        deltas = []
        for i in idxs:
            for sign in (+1, -1):
                o = np.asarray(obs, dtype=np.float32).copy()
                o[i] += sign * eps
                deltas.append(abs(float(predict_fn(o)) - base))
        out[name] = float(np.mean(deltas)) if deltas else 0.0
    out["_base"] = base
    return out


def sweep_axis(predict_fn, obs, idxs=HAND_POS_IDX, axis=1, span=0.5, n=25):
    """
    Translate `idxs[axis]` across +/- span and return (offsets, predictions).

    The causal test. Same base state, one coordinate moved, everything else
    fixed — so any change in the prediction is attributable to that coordinate
    and nothing else. A flat line here means the model's output does not
    depend on where the hand is.
    """
    offsets = np.linspace(-span, span, n)
    preds = []
    for d in offsets:
        o = np.asarray(obs, dtype=np.float32).copy()
        o[idxs[axis]] += d
        preds.append(float(predict_fn(o)))
    return offsets, np.array(preds)


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
    s = sensitivity(predict_fn, obs)
    offs, curve = sweep_axis(predict_fn, obs)
    return dict(
        label=label, sensitivity=s, coverage=coverage_z(obs, run_dir),
        saturation=saturation(predict_fn, obs),
        sweep_range=float(curve.max() - curve.min()),
        sweep_offsets=offs.tolist(), sweep_preds=curve.tolist(),
    )


def format_probe(rows, flat_threshold=0.5):
    """One table. `flat` is the verdict the script exists to deliver."""
    hdr = (f"{'combo':<22}{'pred':>8}{'d/d hand':>10}{'d/d peg':>9}{'d/d goal':>10}"
           f"{'sweep range':>13}{'max|z|':>8}  verdict")
    lines = [hdr, "-" * (len(hdr) + 20)]
    for r in rows:
        s, c = r["sensitivity"], r["coverage"]
        rng = r["sweep_range"]
        responds_elsewhere = max(s["peg"], s["goal"], s["other"]) > 0.05
        # order matters: a model that reacts to peg/goal but not to the hand is
        # cause 1 (wrong inputs), NOT cause 2 (extrapolation) — it is plainly
        # computing something from the state, just not from the hand. Testing
        # the off-distribution branch first mislabelled exactly that case.
        if r["saturation"]["at_ceiling"]:
            verdict = "SATURATED at the clip ceiling"
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
            f"{s['goal']:>10.3f}{rng:>13.2f}{c['max_abs_z']:>8.1f}  {verdict}")
    lines += [
        "",
        "d/d X       : mean |change in prediction| when that group moves 2cm. ~0 = unused.",
        "sweep range : prediction span as the hand is translated +/-0.5 along y,",
        "              everything else held fixed. THE causal test — a flat line",
        "              means the output does not depend on where the hand is.",
        "max|z|      : distance from the training distribution. >3 = extrapolating,",
        "              so a flat response is expected MLP behaviour, not a bug.",
    ]
    return "\n".join(lines)


def plot_sweeps(rows, save_path=None):
    """Prediction vs hand displacement, one line per model."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for r in rows:
        ax.plot(r["sweep_offsets"], r["sweep_preds"], lw=1.8, label=r["label"])
    ax.axvline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("hand displacement along y (sim units), everything else fixed")
    ax.set_ylabel("predicted steps to success")
    ax.set_title("Does the prediction depend on where the hand is?\n"
                 "a flat line means no")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
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

    out = os.path.join(results.get_run_dir("t2s_eval", args.t2s_run), "probe")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"probe_{args.at}.json"), "w") as f:
        json.dump(dict(where=where, rows=rows), f, indent=2)
    if args.plot:
        try:
            plot_sweeps(rows, save_path=os.path.join(out, f"sweep_{args.at}.png"))
        except Exception as e:
            print(f"(plot skipped: {e})")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
