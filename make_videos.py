#!/usr/bin/env python3
"""
Standalone video generation — separate from the graphs on purpose.

    python make_videos.py                      # uses the MODELS list below
    python make_videos.py --reward-mode absolute
    python make_videos.py --models td0boot_all mc_succ --policy-run sweep_v6_td0boot_all_seed0

"""
import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# EDIT THIS: which T2S models to film, as combo names from a t2s_model run.
# None means "every combo in the run".
# e.g. MODELS = ["tdlambdaboot_succ", "td0_all", "mc_all"]
# ---------------------------------------------------------------------------
MODELS = None

T2S_RUN = "v8"                      # t2s_model run name (run_2_train.py --run-name)
REWARD_MODE = "difference_timed"    # absolute | difference | difference_timed
SEEDS = (500,)                      # rollout seeds; one video per model per seed
FPS = None                          # None -> t2s_video.FPS (12)
# ---------------------------------------------------------------------------

STAGE_MANIFEST = "stage_manifest.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Render live T2S prediction videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--t2s-run", default=T2S_RUN, help="t2s_model run name")
    p.add_argument("--models", nargs="+", default=MODELS,
                   help="combo names to film (default: the MODELS list in this file, "
                        "or all combos in the run)")
    p.add_argument("--reward-mode", default=REWARD_MODE,
                   choices=["absolute", "difference", "difference_timed"],
                   help="which reward the overlay shows; use the one the sweep uses")
    p.add_argument("--gamma", type=float, default=None,
                   help="discount in the shaping term (default: config.RL_GAMMA)")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--fps", type=int, default=FPS, help="lower is slower")
    p.add_argument("--policy-run", default=None,
                   help="film a TRAINED policy from this policy run instead of the "
                        "expert checkpoints, e.g. sweep_v6_td0boot_all_seed0")
    p.add_argument("--checkpoint", default="policy_final.zip",
                   help="which checkpoint inside --policy-run to load. Bare "
                        "filename or absolute path. Checkpoints are written every "
                        "ckpt_freq steps as policy_<steps>.zip, so e.g. "
                        "policy_200000.zip films the policy mid-training — useful "
                        "when the final one regressed, which progress.py's own "
                        "docstring notes has happened in this project.")
    p.add_argument("--out", default=None, help="output dir (default: the eval run's videos/)")
    p.add_argument("--no-comparison", dest="comparison", action="store_false", default=True,
                   help="skip the all-models-on-one-rollout video")
    return p.parse_args(argv)


def _load_stage(kind, run_name):
    import results
    try:
        d = results.get_run_dir(kind, run_name)
    except KeyError:
        raise SystemExit(f"ERROR: no {kind} run named {run_name!r}. "
                         f"Registered: {results.list_runs(kind)}")
    path = os.path.join(d, STAGE_MANIFEST)
    if not os.path.exists(path):
        raise SystemExit(f"ERROR: no {STAGE_MANIFEST} in {d} — that stage did not finish.")
    with open(path) as f:
        return d, json.load(f)


def main(argv=None):
    args = parse_args(argv)

    import config
    from t2s_model import load_t2s_predictor
    import t2s_video

    from stable_baselines3 import SAC

    gamma = config.RL_GAMMA if args.gamma is None else args.gamma
    fps = t2s_video.FPS if args.fps is None else args.fps

    t2s_dir, stage2 = _load_stage("t2s_model", args.t2s_run)
    combos = args.models or sorted(stage2["combos"])
    missing = [c for c in combos if c not in stage2["combos"]]
    if missing:
        raise SystemExit(f"ERROR: {missing} not in {args.t2s_run}; "
                         f"available: {sorted(stage2['combos'])}")

    # the reward the overlay shows is only honest if it is the one that will be
    # (or was) used for training — warn rather than silently disagree
    print(f"=== videos: {len(combos)} model(s) from {args.t2s_run} ===")
    print(f"    reward_mode={args.reward_mode}  gamma={gamma}  fps={fps}")
    if args.reward_mode in ("difference", "difference_timed"):
        from reward_fn import hover_reward
        worst = max((stage2["combos"][c].get("eval", {}).get("pred_max") or 0) for c in combos)
        if worst:
            h = hover_reward(worst, reward_mode=args.reward_mode, gamma=gamma)
            print(f"    hovering at the largest prediction seen ({worst:.0f}) pays "
                  f"{h:+.2f}/step" + ("   <<< POSITIVE: freezing is profitable" if h >= 0 else ""))

    # ---- which rollouts --------------------------------------------------
    eval_dir, stage3 = _load_stage("t2s_eval", args.t2s_run)
    out_dir = args.out or os.path.join(eval_dir, "videos")
    os.makedirs(out_dir, exist_ok=True)

    predictors = {c: load_t2s_predictor(
        t2s_dir, *c.rsplit("_", 1), seed=stage2["combos"][c]["best_seed"]) for c in combos}

    if args.policy_run:
        import results
        import visualize
        try:
            pol_dir = results.get_run_dir("policy", args.policy_run)
        except KeyError:
            raise SystemExit(f"ERROR: no policy run named {args.policy_run!r}. "
                             f"Registered: {results.list_runs('policy')}")
        policy, path = visualize.load_policy(args.checkpoint, pol_dir)
        print(f"    filming TRAINED policy {path}")
        scenarios = {"policy": policy}
    else:
        scenarios = {
            "success": SAC.load(os.path.join(config.EXPERT_POLICY_DIR, stage3["success_ckpt"])),
            "failure": SAC.load(os.path.join(config.EXPERT_POLICY_DIR, stage3["failure_ckpt"])),
        }
        print(f"    held-out policies: success={stage3['success_ckpt']} "
              f"failure={stage3['failure_ckpt']}")

    # ---- one video per (model, scenario, seed) ---------------------------
    print(f"\n-> {out_dir}")
    results_rows = []
    for scenario, policy in scenarios.items():
        for seed in args.seeds:
            # roll ONCE per (scenario, seed); every model scores the SAME stored
            # frames, so a difference between videos is the model, not the rollout
            roll = t2s_video.rollout_with_frames(policy, seed=seed)
            for combo, fn in predictors.items():
                roll_c = dict(roll, t2s_preds=None)
                name = f"{combo}_{scenario}_seed{seed}.mp4"
                _f, info = t2s_video.render_t2s_video(
                    roll_c, predict_t2s=fn, save_path=os.path.join(out_dir, name),
                    reward_mode=args.reward_mode, gamma=gamma, fps=fps,
                    label=f"{combo} | {scenario}")
                results_rows.append(dict(combo=combo, scenario=scenario, seed=seed,
                                         file=name, **{k: info[k] for k in
                                                       ("pred_min", "pred_max", "total_return",
                                                        "success_step", "entered_danger_band")}))
                print(f"  {name:<46}pred {info['pred_min']:6.1f}-{info['pred_max']:6.1f}"
                      f"  return {info['total_return']:+9.1f}"
                      + ("  <<< FALSELY OPTIMISTIC" if info["entered_danger_band"] else ""))

            if args.comparison and len(predictors) > 1:
                name = f"ALL_{scenario}_seed{seed}.mp4"
                t2s_video.render_model_comparison_video(
                    roll, predictors, save_path=os.path.join(out_dir, name), fps=fps)
                print(f"  {name:<46}every model on the same frames")

    with open(os.path.join(out_dir, "video_index.json"), "w") as f:
        json.dump(dict(t2s_run=args.t2s_run, reward_mode=args.reward_mode, gamma=gamma,
                       fps=fps, policy_run=args.policy_run, videos=results_rows), f, indent=2)

    # on a rollout that never succeeds, total_return ranks the models by what
    # they would have paid the policy for that behaviour
    fail_rows = [r for r in results_rows if r["scenario"] == "failure"]
    if fail_rows:
        print(f"\non the FAILING rollout, what each reward would have paid:")
        for r in sorted(fail_rows, key=lambda r: -r["total_return"]):
            print(f"  {r['combo']:<24}return {r['total_return']:+9.1f}  "
                  f"min pred {r['pred_min']:6.1f}"
                  + ("   <<< claims imminent success" if r["entered_danger_band"] else ""))
        print("  (higher return on a trajectory that NEVER succeeds is worse:")
        print("   it means the reward is happy with behaviour that never solves the task)")

    print(f"\nwrote {len(results_rows)} video(s) + video_index.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())