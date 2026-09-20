#!/usr/bin/env python3
"""
Compare the SAME T2S combo across model versions — before spending a sweep.

    python compare_t2s.py --runs v3 v4 --combo tdlambdaboot_succ
    python compare_t2s.py --runs v3 v4 v5 --combo tdlambdaboot_succ --drive retreat

Renders every version onto the SAME stored frames and prints the four numbers
that decide whether a new version is worth training against. A 400k-step sweep
is hours; this is minutes.

WHY A SEPARATE SCRIPT
---------------------
run_3_eval.py scores the combos WITHIN one training run. It cannot compare
`tdlambdaboot_succ` in run v3 against the same combo in v4, because a run's
manifest only knows its own models — and the interesting question after a
change to the data or the loss is exactly that comparison.

THE FOUR NUMBERS, AND WHY THESE FOUR
------------------------------------
Ranked by how much they predict downstream RL success, not by how familiar
they are. MAE is deliberately NOT among them: your twelve models spanned 25x
in MAE while the correlation columns read ~1.00 for all of them, and MAE is
measured only on expert rollouts where nothing interesting happens.

  fail_pred_min   lowest prediction on a rollout that NEVER succeeds. LOW IS
                  BAD — it is the model's most confident false claim, and the
                  empty-gripper hover made numeric. The single best predictor
                  of whether a shaped reward will pay for hovering.
  peg_sensitivity prediction span when only the PEG moves. ~0 means the model
                  cannot separate "peg in hole" from "hand in hole", so no
                  reward mode can build a grasp gradient from it.
  rose_on_drive   did the prediction RISE while the arm was driven away from
                  the peg? A model that does not is reading familiarity, not
                  distance.
  hover_at_ceil   what the reward pays per step for standing still, evaluated
                  at the reachable prediction ceiling rather than the expert's
                  range. Must be negative.

A new version is worth training against if fail_pred_min goes UP materially
and the other three do not regress. If only MAE improved, it probably is not.
"""
import argparse
import json
import os
import sys

import numpy as np

STAGE_MANIFEST = "stage_manifest.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Compare one T2S combo across model versions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # not required: --combos already names its runs, and demanding --runs too
    # invited passing run:combo pairs to the singular --combo, which argparse
    # rejects with an unhelpful "unrecognized arguments"
    p.add_argument("--runs", nargs="+", default=None,
                   help="t2s_model run names, used with --combo. Not needed with "
                        "--combos, which names its own runs.")
    p.add_argument("--combo", default=None,
                   help="one combo name, compared across every --runs entry")
    p.add_argument("--combos", nargs="+", default=None,
                   help="explicit run:combo pairs, for comparing DIFFERENT combos "
                        "across runs, e.g. v1:tdlambdaboot_all v3:tdlambdaboot_succ. "
                        "Use this when the model you trained RL against is not the "
                        "same combo as the new candidate — --combo cannot express "
                        "that, which is a real case: the 74%-of-the-way run used "
                        "tdlambdaboot_all while the new best was tdlambdaboot_succ.")
    p.add_argument("--drive", default="retreat",
                   help="drive script for the off-distribution test (see manual_drive)")
    p.add_argument("--seed", type=int, default=500)
    p.add_argument("--reward-mode", default="difference_timed",
                   choices=["absolute", "difference", "difference_timed"])
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--out", default=None, help="output dir for the videos")
    p.add_argument("--no-video", dest="video", action="store_false", default=True)
    return p.parse_args(argv)


def parse_pairs(combos, runs, combo):
    """
    Resolve --combo / --combos into [(run, combo), ...].

    --combo X --runs a b   ->  [(a, X), (b, X)]
    --combos a:X b:Y       ->  [(a, X), (b, Y)]
    """
    if combos:
        pairs = []
        for spec in combos:
            if ":" not in spec:
                raise SystemExit(f"ERROR: --combos entries must be run:combo, got "
                                 f"{spec!r}. Example: v1:tdlambdaboot_all")
            run, _, name = spec.partition(":")
            pairs.append((run.strip(), name.strip()))
        return pairs
    if not combo:
        raise SystemExit(
            "ERROR: pass one of\n"
            "  --runs v1 v3 --combo tdlambdaboot_all        (same combo, several runs)\n"
            "  --combos v1:tdlambdaboot_all v3:tdlambdaboot_succ   (different combos)")
    if ":" in combo:
        raise SystemExit(
            f"ERROR: --combo takes ONE plain combo name, got {combo!r}.\n"
            "       For run:combo pairs use the plural flag:\n"
            f"  python compare_t2s.py --combos {combo} <run>:<combo>")
    if not runs:
        raise SystemExit("ERROR: --combo needs --runs, e.g. --runs v1 v3")
    return [(r, combo) for r in runs]


def load_versions(pairs):
    """{label: (run_dir, predict_fn)} for each (run, combo) pair."""
    import results
    import t2s_predict

    out = {}
    for run, combo in pairs:
        try:
            d = results.get_run_dir("t2s_model", run)
        except KeyError:
            raise SystemExit(f"ERROR: no t2s_model run named {run!r}. "
                             f"Registered: {results.list_runs('t2s_model')}")
        # Prefer the staged manifest, but fall back to the run's own
        # manifest.json, which every t2s_model run has written since long
        # before the staged scripts existed. The whole point of this script is
        # comparing an OLD version against a new one, so refusing to read old
        # runs would defeat it.
        mpath = os.path.join(d, STAGE_MANIFEST)
        if os.path.exists(mpath):
            with open(mpath) as f:
                man = json.load(f)
        else:
            import t2s_io
            try:
                man = t2s_io.load_manifest(d)
            except FileNotFoundError:
                raise SystemExit(f"ERROR: {run} has neither {STAGE_MANIFEST} nor "
                                 "manifest.json; nothing identifies its models.")
            print(f"  ({run}: no {STAGE_MANIFEST}, using legacy manifest.json)")
        if combo not in man["combos"]:
            raise SystemExit(f"ERROR: {combo!r} not in run {run!r}; available: "
                             f"{sorted(man['combos'])}")
        seed = man["combos"][combo].get("best_seed", 0)
        fn = t2s_predict.load_t2s_predictor(d, *combo.rsplit("_", 1), seed=seed)
        out[f"{combo}@{run}"] = (d, fn)
    return out


def score(fn, run_dir, fail_roll, drive_roll, reward_mode, gamma):
    """The four decisive numbers for one version."""
    from reward_fn import hover_reward, SHAPED_MODES
    from config import T2S_PRED_CLIP_MAX
    import t2s_probe

    fail_preds = np.array([fn(o) for o in fail_roll["obs"]], dtype=float)
    ho = drive_roll["handover"]
    dp = np.array([fn(o) for o in drive_roll["obs"]], dtype=float)
    rel = t2s_probe.relative_vs_absolute(fn, drive_roll["obs"][0])

    return dict(
        # lowest prediction on a trajectory that never succeeds
        fail_pred_min=float(fail_preds.min()),
        fail_pred_mean=float(fail_preds.mean()),
        peg_sensitivity=float(rel["peg_only_range"]),
        rose_on_drive=bool(dp[min(ho, len(dp) - 1)] > dp[0]),
        drive_rise=float(dp[min(ho, len(dp) - 1)] - dp[0]),
        hover_at_ceil=(float(hover_reward(T2S_PRED_CLIP_MAX, reward_mode=reward_mode,
                                          gamma=gamma))
                       if reward_mode in SHAPED_MODES else None),
        pred_max=float(max(fail_preds.max(), dp.max())),
    )


def normalise(r):
    """
    Range-relative versions of the two headline numbers.

    A model with a COMPRESSED prediction range scores a flattering absolute
    fail_pred_min simply because all of its outputs are small. Dividing by
    pred_max separates "stays pessimistic" from "never says much at all" —
    and pred_max itself matters, because the shaping reward is a DIFFERENCE of
    predictions, so a smaller range means a smaller reward signal everywhere.
    """
    pm = max(r.get("pred_max") or 0.0, 1e-9)
    return dict(fail_frac=r["fail_pred_min"] / pm,
                peg_frac=r["peg_sensitivity"] / pm)


def main(argv=None):
    args = parse_args(argv)

    import config
    import results
    import t2s_video
    from manual_drive import parse_drive, drive_then_policy
    from stable_baselines3 import SAC

    gamma = config.RL_GAMMA if args.gamma is None else args.gamma
    pairs = parse_pairs(args.combos, args.runs, args.combo)
    versions = load_versions(pairs)
    runs_used = [r for r, _c in pairs]

    # the eval policies come from whichever run has an eval manifest
    succ = fail = None
    for run in runs_used:
        try:
            ed = results.get_run_dir("t2s_eval", run)
            with open(os.path.join(ed, STAGE_MANIFEST)) as f:
                st = json.load(f)
            succ, fail = st["success_ckpt"], st["failure_ckpt"]
            break
        except (KeyError, FileNotFoundError):
            continue
    if succ is None:
        raise SystemExit("ERROR: no t2s_eval manifest found for any of "
                         f"{runs_used}; run run_3_eval.py first.")
    succ_pol = SAC.load(os.path.join(config.EXPERT_POLICY_DIR, succ))
    fail_pol = SAC.load(os.path.join(config.EXPERT_POLICY_DIR, fail))

    label_desc = args.combo or ", ".join(f"{r}:{c}" for r, c in pairs)
    print(f"=== comparing {len(versions)} model(s): {label_desc} ===")
    print(f"    reward_mode={args.reward_mode}  gamma={gamma}\n")

    # roll ONCE. Every version scores identical frames, so a difference between
    # versions is the model and not the rollout.
    print("rolling reference trajectories (shared by every version)...")
    fail_roll = t2s_video.rollout_with_frames(fail_pol, seed=args.seed)
    drive_roll = drive_then_policy(fail_pol, parse_drive(args.drive), seed=args.seed)

    rows = {lbl: score(fn, d, fail_roll, drive_roll, args.reward_mode, gamma)
            for lbl, (d, fn) in versions.items()}

    print(f"\n{'version':<26}{'fail_min':>10}{'/pmax':>8}{'peg sens':>10}{'/pmax':>8}"
          f"{'rose?':>7}{'pred_max':>10}")
    print("-" * 79)
    for lbl, r in sorted(rows.items(), key=lambda kv: -kv[1]["peg_sensitivity"]):
        nf = normalise(r)
        print(f"{lbl:<26}{r['fail_pred_min']:>10.1f}{nf['fail_frac']:>8.2f}"
              f"{r['peg_sensitivity']:>10.1f}{nf['peg_frac']:>8.2f}"
              f"{('yes' if r['rose_on_drive'] else 'NO'):>7}{r['pred_max']:>10.1f}")
    print("\n/pmax columns divide by the model's own range. A compressed model")
    print("scores a flattering absolute fail_min just by never predicting much.")
    print("pred_max matters on its own: the shaping reward is a DIFFERENCE of")
    print("predictions, so a smaller range is a weaker signal everywhere.")

    # ---- the verdict -----------------------------------------------------
    labels = list(rows)
    if len(labels) >= 2:
        base, new = labels[0], labels[-1]
        b, n = rows[base], rows[new]
        d_fail = n["fail_pred_min"] - b["fail_pred_min"]
        d_peg = n["peg_sensitivity"] - b["peg_sensitivity"]
        print(f"\n--- {new} vs {base} ---")
        print(f"  fail_pred_min   {b['fail_pred_min']:7.1f} -> {n['fail_pred_min']:7.1f}"
              f"  ({d_fail:+.1f})   higher is better")
        print(f"  peg sensitivity {b['peg_sensitivity']:7.1f} -> {n['peg_sensitivity']:7.1f}"
              f"  ({d_peg:+.1f})   higher is better")
        print(f"  rose on drive   {b['rose_on_drive']!s:>7} -> {n['rose_on_drive']!s:>7}")
        d_pmax = n["pred_max"] - b["pred_max"]
        print(f"  pred_max        {b['pred_max']:7.1f} -> {n['pred_max']:7.1f}"
              f"  ({d_pmax:+.1f})   a smaller range = a weaker reward everywhere")
        # peg sensitivity is weighted FIRST: this is a manipulation task, and a
        # model that cannot tell where the peg is cannot reward grasping it, no
        # matter how pessimistic it is on failure rollouts.
        peg_ok = d_peg >= -0.1 * max(b["peg_sensitivity"], 1.0)
        fail_ok = d_fail > 0.2 * max(b["fail_pred_min"], 1.0)
        if not peg_ok:
            print(f"\n  >>> NOT clearly better: peg sensitivity fell {-d_peg:.0f} "
                  f"({-d_peg / max(b['peg_sensitivity'], 1e-9):.0%}). For a "
                  "manipulation task that outranks a better fail_pred_min — a "
                  "model that cannot see the peg cannot reward grasping it.")
        elif fail_ok and n["rose_on_drive"]:
            print("\n  >>> WORTH TRAINING: fail_pred_min improved materially and "
                  "nothing regressed")
        else:
            print("\n  >>> NOT clearly better: fail_pred_min did not improve enough, "
                  "or the drive response regressed. A better MAE alone does not qualify.")
        if max(b["fail_pred_min"], n["fail_pred_min"]) < 100:
            print(f"\n  NOTE: BOTH versions claim under 100 steps-to-success on a "
                  "trajectory that never succeeds. The right answer is the censor "
                  "level (~300). Neither is pessimistic enough; this is a shared "
                  "defect, not a difference between them.")

    # ---- video: every version on the same frames -------------------------
    if args.video:
        out_dir = args.out or os.path.join(
            results.get_run_dir("t2s_eval", runs_used[0]), "compare")
        os.makedirs(out_dir, exist_ok=True)
        preds = {lbl: fn for lbl, (_d, fn) in versions.items()}
        for name, roll in (("failure", fail_roll), (f"drive_{args.drive}", drive_roll)):
            tag = args.combo or "mixed"
            path = os.path.join(out_dir, f"compare_{tag}_{name}.mp4")
            t2s_video.render_model_comparison_video(roll, preds, save_path=path)
            print(f"  wrote {os.path.basename(path)}")
        print("\nin the video, watch the FAILURE rollout: whichever version's readout")
        print("turns red while peg>goal stays flat is the one that would pay the")
        print("policy to hover.")

        with open(os.path.join(out_dir, f"compare_{args.combo or 'mixed'}.json"), "w") as f:
            json.dump(dict(pairs=pairs, gamma=gamma,
                           reward_mode=args.reward_mode, rows=rows), f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())