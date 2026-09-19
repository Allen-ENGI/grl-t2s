#!/usr/bin/env python3
"""
STAGE 3 of 4 — T2S evaluation.

    python run_3_eval.py --t2s-run v3

Scores every model Stage 2 trained against held-out POLICIES and held-out
ROWS, and writes the manifest run_4_policy.py reads. Trains nothing, so it is
cheap to re-run for a different metric or more rollout seeds.

NUMBERS AND CHARTS ONLY. Video lives in make_videos.py, which takes its own
model list and reward mode. They were one script, which meant re-rendering
footage to try a different reward mode also re-ran the whole evaluation, and
choosing which models to film meant editing an evaluation script. --video
still renders here for convenience; make_videos.py is the flexible path.

This script is orchestration only — every number comes from t2s_eval and
every frame from t2s_video, the same functions the notebook cells call.

TWO KINDS OF "HELD OUT"
-----------------------
  ROWS      the `val` split fixed at collection time. Thousands of states,
            but the same policies that generated training data. Tests
            interpolation. This is what Stage 2's val MSE already ranked.
  POLICIES  checkpoints reserved by Stage 1's --holdout and never rolled out.
            Tests generalization to unseen behaviour — the stronger claim,
            and it can rank differently from val MSE.
The evaluation policies are DERIVED from Stage 1's recorded holdout, not
retyped: strongest reserved checkpoint reliably succeeds, weakest reliably
fails. assert_policies_held_out then verifies neither was collected from.

READING THE TABLE
-----------------
Within one success rollout the truth is a straight ramp in the frame index,
so correlating predictions against it is correlating against -t: a model with
a 3x wrong scale or a +100 offset still scored a perfect 1.000, and any
decreasing curve scored ~0.99. What replaced that column:

  bias               signed. NEGATIVE = claims success is nearer than it is,
                     the dangerous direction for a reward.
  calibration_slope  1.0 is correct. With bias, the only metrics that see a
                     global offset or scale error — no correlation can.
  step_slope_error   mean |predicted decrement - 1|; literally the quantity
                     the shaped reward consumes. A model can have excellent
                     MAE and a useless step slope.
  pooled_spearman    across rollouts; sees a level consistent within each
                     trajectory but drifting between them.
  monotonicity_rho   the old correlation, renamed. Shape only.

Exit codes: 0 ok, 1 Stage 2's output is missing or the held-out claim failed.
"""
import argparse
import json
import os
import re
import sys

STAGE_MANIFEST = "stage_manifest.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Stage 3: T2S evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--t2s-run", required=True, help="run name from run_2_train.py")
    p.add_argument("--eval-seeds", type=int, default=8,
                   help="held-out-policy rollouts per model; these DO vary")
    p.add_argument("--models", nargs="+", default=None,
                   help="subset of combos to score (default: all trained)")
    p.add_argument("--video", action="store_true", default=False,
                   help="also render videos (make_videos.py is the flexible path)")
    p.add_argument("--video-seed", type=int, default=500)
    return p.parse_args(argv)


def pick_eval_policies(holdout):
    """
    Split Stage 1's holdout into (success_policy, failure_policy) by training
    step. Derived rather than typed, so it cannot disagree with what
    collection actually reserved — these two had to match --holdout exactly or
    the held-out assertion fails, and asking for the same filenames twice was
    an invitation to get it wrong.
    """
    if len(holdout) < 2:
        raise ValueError(f"need 2 held-out checkpoints, Stage 1 reserved {holdout}")

    def step(name):
        m = re.search(r"_(\d+)\.zip$", name)
        return int(m.group(1)) if m else float("inf")      # "final" is strongest
    ranked = sorted(holdout, key=step)
    return ranked[-1], ranked[0]


def main(argv=None):
    args = parse_args(argv)

    import config
    import results
    import t2s_eval
    import t2s_predict

    # ---- Stage 2's handoff ----------------------------------------------
    try:
        t2s_dir = results.get_run_dir("t2s_model", args.t2s_run)
    except KeyError:
        print(f"ERROR: no t2s_model run named {args.t2s_run!r}. "
              f"Registered: {results.list_runs('t2s_model')}\n"
              f"Run:  python run_2_train.py --data-run <name> --run-name {args.t2s_run}",
              file=sys.stderr)
        return 1
    mpath = os.path.join(t2s_dir, STAGE_MANIFEST)
    if not os.path.exists(mpath):
        print(f"ERROR: no {STAGE_MANIFEST} in {t2s_dir} — training did not finish, or "
              "that run predates the staged scripts.", file=sys.stderr)
        return 1
    with open(mpath) as f:
        stage2 = json.load(f)

    combos = args.models or sorted(stage2["combos"])
    missing = [c for c in combos if c not in stage2["combos"]]
    if missing:
        print(f"ERROR: {missing} not in {args.t2s_run}; available: "
              f"{sorted(stage2['combos'])}", file=sys.stderr)
        return 1
    rows = [dict(combo=c, best_seed=stage2["combos"][c]["best_seed"]) for c in combos]

    succ_name, fail_name = pick_eval_policies(stage2["holdout_checkpoints"])
    print(f"=== Stage 3: evaluating {len(rows)} model(s) from {args.t2s_run} ===")
    print(f"    held out: success={succ_name}  failure={fail_name}\n")

    # ---- verify the held-out claim --------------------------------------
    from stable_baselines3 import SAC

    succ_path = os.path.join(config.EXPERT_POLICY_DIR, succ_name)
    fail_path = os.path.join(config.EXPERT_POLICY_DIR, fail_name)
    try:
        t2s_eval.assert_policies_held_out(stage2["dataset_summary_path"],
                                          [succ_path, fail_path])
        print("held-out check: ok, neither policy was collected from")
    except AssertionError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    succ_pol, fail_pol = SAC.load(succ_path), SAC.load(fail_path)
    eval_dir = results.new_run_dir(
        "t2s_eval", args.t2s_run,
        meta=dict(stage="3_eval", t2s_run=args.t2s_run, data_run=stage2["data_run"]))

    # ---- score -----------------------------------------------------------
    print(f"\nscoring ({args.eval_seeds} rollouts each) -> {eval_dir}")
    reports = t2s_eval.evaluate_runs(
        t2s_dir, rows, succ_pol, fail_pol, n_seeds=args.eval_seeds,
        dataset_path=stage2["dataset_path"])       # also the held-out-ROWS check

    with open(os.path.join(eval_dir, "eval_report.json"), "w") as f:
        json.dump(reports, f, indent=2)

    print()
    print(t2s_eval.format_report_table(reports, sort_by="mae"))

    # val MSE ranked fit to held-out ROWS; this ranks held-out POLICIES.
    # Disagreement between them is informative, not a bug.
    print(f"\n{'combo':<22}{'val MSE (rows)':>16}{'MAE (policies)':>16}")
    print("-" * 54)
    for c in sorted(combos, key=lambda c: stage2["combos"][c]["val_mse"]):
        mae = reports[c].get("summary", {}).get("mae")
        print(f"{c:<22}{stage2['combos'][c]['val_mse']:>16.2f}"
              f"{(f'{mae:.2f}' if mae is not None else '--'):>16}")

    try:
        t2s_eval.plot_combo_comparison(
            reports, sort_by="mae",
            save_path=os.path.join(eval_dir, "combo_comparison.png"))
    except Exception as e:                     # headless: the numbers still stand
        print(f"(plot skipped: {e})")

    # ---- videos ----------------------------------------------------------
    videos = {}
    if args.video:
        import t2s_video

        video_dir = os.path.join(eval_dir, "videos")
        print(f"\n=== live prediction videos -> {video_dir} ===")
        predictors = {
            c: t2s_predict.load_t2s_predictor(t2s_dir, *c.rsplit("_", 1),
                                              seed=stage2["combos"][c]["best_seed"])
            for c in combos}

        for combo, fn in predictors.items():
            _rep, v = t2s_video.evaluate_with_video(
                fn, succ_pol, fail_pol, video_dir, label=combo,
                seeds=(args.video_seed,), n_metric_seeds=0, gamma=config.RL_GAMMA)
            videos[combo] = v

        # every model on the SAME frames, so a divergence is the model and not
        # the rollout. Whichever readout turns red is the model that would have
        # paid the policy to keep hovering.
        roll = t2s_video.rollout_with_frames(fail_pol, seed=args.video_seed)
        _f, cmp_info = t2s_video.render_model_comparison_video(
            roll, predictors, save_path=os.path.join(video_dir, "all_models_failure.mp4"))
        print(f"\n  {'combo':<24}{'min pred':>10}")
        for name, i in sorted(cmp_info.items(), key=lambda kv: kv[1]["pred_min"]):
            print(f"  {name:<24}{i['pred_min']:>10.1f}"
                  + ("   <<< FALSELY OPTIMISTIC" if i["falsely_optimistic"] else ""))

    # ---- handoff ---------------------------------------------------------
    manifest = dict(
        stage="3_eval", t2s_run=args.t2s_run, t2s_run_dir=t2s_dir, eval_dir=eval_dir,
        data_run=stage2["data_run"], dataset_path=stage2["dataset_path"],
        success_ckpt=succ_name, failure_ckpt=fail_name, eval_seeds=args.eval_seeds,
        combos={c: dict(best_seed=stage2["combos"][c]["best_seed"],
                        val_mse=stage2["combos"][c]["val_mse"],
                        eval={k: reports[c].get("summary", {}).get(k)
                              for k in ("mae", "bias", "calibration_slope",
                                        "step_slope_error", "pooled_spearman",
                                        "pred_max")})
                for c in combos},
        videos=videos, ok=True)
    with open(os.path.join(eval_dir, STAGE_MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2)

    best = sorted(combos, key=lambda c: reports[c].get("summary", {}).get("mae")
                  if reports[c].get("summary", {}).get("mae") is not None else float("inf"))[:2]
    print(f"\nwrote {os.path.join(eval_dir, STAGE_MANIFEST)}")
    print(f"next:  python run_4_policy.py --eval-run {args.t2s_run} "
          f"--models {' '.join(best)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())