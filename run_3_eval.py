#!/usr/bin/env python3
"""
STAGE 3 of 4 - T2S evaluation.

    python run_3_eval.py --t2s-run v3

Scores every model Stage 2 trained and writes the manifest run_4_policy.py
reads. Trains nothing, so it is cheap to re-run.

NUMBERS AND CHARTS ONLY. Video lives in make_videos.py; --video still renders
here for convenience, but make_videos.py is the flexible path.

WHAT IS HELD OUT
----------------
Held-out EPISODES: the `val` split fixed at collection time, stratified across
every source. Reserving whole CHECKPOINTS was dropped - it cost every one of
their failure modes in exchange for a generalisation claim that did not probe
the regime that matters, since neither reserved expert resembles an untrained
policy. assert_policies_held_out is now advisory, not a gate.

THE FOUR METRICS
----------------
  mae                   does it predict TIME correctly, on success rollouts
  step_slope_error      is the per-step DECREMENT right - literally what the
                        shaped reward consumes. 1.0 = no better than flat
  dead_window_fraction  any gradient at all, or sustained flat runs
  fail_pred_min         does it stay pessimistic on a trajectory that NEVER
                        succeeds. LOW IS BAD

MAE IS NOT THE SELECTOR
-----------------------
It is measured only on success rollouts, where models trained on success data
specialise, and across the observed grid it correlated +0.68 with
fail_pred_min: better success accuracy went with WORSE failure behaviour. The
best-MAE model scored fail_pred_min 4.11 - four steps to success on a rollout
that never succeeds. So models are ranked by t2s_eval.rank_models: reject
false optimism first, then rank the survivors by MAE. Stage 4 applies the same
threshold, so the two stages cannot recommend different models.

Exit codes: 0 ok, 1 Stage 2's output is missing.
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
                   help="rollouts per model; these DO vary")
    p.add_argument("--models", nargs="+", default=None,
                   help="subset of combos to score (default: all trained)")
    p.add_argument("--video", action="store_true", default=False,
                   help="also render videos (make_videos.py is the flexible path)")
    p.add_argument("--video-seed", type=int, default=500)
    p.add_argument("--score-on", default="rollouts", choices=["rollouts", "val"],
                   help="'val' scores the held-out val EPISODES from dataset.npz "
                        "instead of 8 live rollouts: dozens of complete "
                        "trajectories, no simulator, seconds. Live rolling was "
                        "justified when Stage 1 reserved checkpoints; that holdout "
                        "is gone, so the reference policies were collected from too "
                        "and the rollouts are no longer a different distribution, "
                        "just a smaller one.")
    return p.parse_args(argv)


def pick_eval_policies(holdout):
    """
    Split a recorded holdout into (success_policy, failure_policy) by training
    step. Only used when a run has no stored reference set; the references
    record which checkpoints produced them, so nothing has to be inferred.
    """
    if len(holdout) < 2:
        raise ValueError(f"need 2 held-out checkpoints, got {holdout}")

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
        print(f"ERROR: no {STAGE_MANIFEST} in {t2s_dir} - training did not finish.",
              file=sys.stderr)
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

    # prefer the stored reference set; fall back to the holdout-derived policies
    ref_dir = os.path.join(os.path.dirname(stage2["dataset_path"]), "references")
    use_refs = os.path.exists(os.path.join(ref_dir, "index.json"))
    if use_refs:
        with open(os.path.join(ref_dir, "index.json")) as f:
            ridx = json.load(f)
        succ_name, fail_name = ridx["success_ckpt"], ridx["failure_ckpt"]
    else:
        succ_name, fail_name = pick_eval_policies(stage2["holdout_checkpoints"])
    print(f"=== Stage 3: evaluating {len(rows)} model(s) from {args.t2s_run} ===")
    print(f"    reference policies: success={succ_name}  failure={fail_name}")
    # The reference index is used only to NAME the checkpoints. Scoring still
    # re-rolls them live (t2s_eval._rollout, seeds 500-507), so the earlier
    # "STORED (identical across every tool)" message was false: the stored
    # observations are used by reward_preview and the video tools, not here.
    print("    rollouts: re-rolled live, seeds 500-"
          f"{500 + args.eval_seeds - 1}"
          + ("  (checkpoints named by the stored reference index)" if use_refs
             else "  (checkpoints from the recorded holdout)") + "\n")

    from stable_baselines3 import SAC

    succ_path = os.path.join(config.EXPERT_POLICY_DIR, succ_name)
    fail_path = os.path.join(config.EXPERT_POLICY_DIR, fail_name)
    # advisory, not a gate: Stage 3 measures held-out EPISODES (the val split,
    # stratified across every source) rather than held-out policies
    held = t2s_eval.assert_policies_held_out(stage2["dataset_summary_path"],
                                             [succ_path, fail_path], strict=False)
    print("held-out episodes via the val split"
          + ("" if held["ok"] else "; reference policies were also collected from"))

    succ_pol, fail_pol = SAC.load(succ_path), SAC.load(fail_path)
    eval_dir = results.new_run_dir(
        "t2s_eval", args.t2s_run,
        meta=dict(stage="3_eval", t2s_run=args.t2s_run, data_run=stage2["data_run"]))

    # ---- score -----------------------------------------------------------
    print("\nscoring on "
          + (f"{args.eval_seeds} live rollouts each" if args.score_on == "rollouts"
             else "held-out val EPISODES from dataset.npz")
          + f" -> {eval_dir}")
    reports = t2s_eval.evaluate_runs(
        t2s_dir, rows, succ_pol, fail_pol, n_seeds=args.eval_seeds,
        dataset_path=stage2["dataset_path"], score_on=args.score_on)

    with open(os.path.join(eval_dir, "eval_report.json"), "w") as f:
        json.dump(reports, f, indent=2)

    print()
    print(t2s_eval.format_report_table(reports, sort_by="mae"))

    # THREE views of the same models, on different data:
    #   val MSE         Stage 2's own number, held-out ROWS at training time
    #   val MAE/bias    held-out EPISODES from dataset.npz: thousands of
    #                   collected states never trained on. Computed all along
    #                   by evaluate_on_val_split and never displayed.
    #   rollout MAE     eight LIVE rollouts of the reference policies
    # Disagreement between them is informative, not a bug: the val split is a
    # bigger sample, the rollouts are the more realistic one.
    n_val = (reports[combos[0]].get("val_split") or {}).get("n")
    print(f"\nheld-out EPISODES ({n_val:,} val rows) vs LIVE rollouts"
          if n_val else "\nheld-out rows vs live rollouts")
    print(f"{'combo':<22}{'val MSE':>10}{'val MAE':>10}{'val bias':>10}"
          f"{'rollout MAE':>13}")
    print("-" * 65)
    for c in sorted(combos, key=lambda c: stage2["combos"][c]["val_mse"]):
        v = reports[c].get("val_split") or {}
        mae = reports[c].get("summary", {}).get("mae")
        f = lambda x, d=2: "--" if x is None else f"{x:.{d}f}"
        print(f"{c:<22}{stage2['combos'][c]['val_mse']:>10.2f}"
              f"{f(v.get('mae')):>10}{f(v.get('bias')):>10}{f(mae):>13}")

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
        # the rollout
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
                              for k in t2s_eval.REPORT_METRICS
                              + ("bias", "calibration_slope", "pred_max")},
                        val_split=reports[c].get("val_split"))
                for c in combos},
        videos=videos, ok=True)
    with open(os.path.join(eval_dir, STAGE_MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2)

    # the SAME rule Stage 4 applies. Ranking by MAE here used to recommend
    # models Stage 4 then refused - and the top-MAE model was the most falsely
    # optimistic one in the grid.
    best, rejected, reason = t2s_eval.rank_models(reports, n=2)
    print(f"\nwrote {os.path.join(eval_dir, STAGE_MANIFEST)}")
    print(f"\nselection: {reason}")
    if rejected:
        print(f"  rejected for false optimism (fail_pred_min < "
              f"{t2s_eval.MIN_FAIL_PRED:.0f}): {sorted(rejected)}")
    print(f"next:  python run_4_policy.py --eval-run {args.t2s_run} "
          f"--models {' '.join(best)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())