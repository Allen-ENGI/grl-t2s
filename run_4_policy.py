#!/usr/bin/env python3
"""
STAGE 4 of 4 — downstream RL against the frozen T2S reward.

    python run_4_policy.py --eval-run v3 --models td0boot_all tdlambdaboot_all

Runs the reward preview as a GATE, then trains one SAC policy per
(model x seed), then summarises. `--preview-only` stops after the gate, which
costs minutes instead of hours and is the single cheapest check in the
pipeline.

THE GATE IS NOT DECORATION
--------------------------
`difference_timed` is potential-based shaping with Phi = -pred, so the shaping
term is pred(s) - gamma*pred(s'). On a CONSTANT prediction P — the
post-~150-step flatline this project observed — it pays

    r_hover = (1 - gamma) * P - time_penalty

for a step that changes nothing. That goes POSITIVE as soon as
(1-gamma)*P >= time_penalty, i.e. the policy is PAID TO FREEZE:

    gamma=0.99,  P=200  ->  +1.00 / step
    gamma=0.995, P=300  ->  +0.50 / step
    gamma=0.999, P=300  ->  -0.70 / step   (correct: freezing costs)

Measured with an ORACLE predictor (predictions exactly equal to ground truth),
three success and three failure trajectories:

    gamma=0.99   return_gap = -419 +/- 3    hover +1.99   FAIL
    gamma=0.999  return_gap = +520 +/- 11   hover -0.70   PASS

A perfect predictor fails the gate at 0.99. That is why config.RL_GAMMA is
0.999 and why --gamma-sweep exists: run it on your EXISTING models before
committing compute, because if the gap sign flips the way it did for the
oracle, the gamma explains the sweep results rather than the T2S models doing
so.

ONE GAMMA HERE
--------------
config.RL_GAMMA flows into SAC's critic, the shaping term inside
Time2SuccessRewardWrapper, and VecNormalize's discounted-return statistics.
Those were three independent defaults that could drift apart. It is NOT
config.T2S_BOOTSTRAP_GAMMA, which Stage 2 uses for a different job.

Exit codes: 0 ok, 1 the reward gate failed (see --force).
"""
import argparse
import json
import os
import sys

STAGE_MANIFEST = "stage_manifest.json"

# constants, not flags: nobody chooses these per run
PREVIEW_SEEDS = (500, 501, 502)   # reference trajectories for the reward gate
EVAL_FREQ = CKPT_FREQ = 10_000
N_EVAL_EPISODES = 5


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Stage 4: downstream RL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--eval-run", required=True,
                   help="t2s_eval run name from run_3_eval.py")
    p.add_argument("--models", nargs="+", default=None,
                   help="combo names to train against (default: best held-out MAE)")
    p.add_argument("--n-best", type=int, default=2,
                   help="how many combos to take when --models is omitted")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="RL seeds (default: config.RL_SEEDS = 0 1 2)")
    p.add_argument("--sweep-name", default="sweep_v6")
    p.add_argument("--reward-mode", default="difference",
                   choices=["absolute", "difference", "difference_timed"])
    p.add_argument("--gamma", type=float, default=None,
                   help="RL discount (default: config.RL_GAMMA)")
    p.add_argument("--timesteps", type=int, default=400_000)
    p.add_argument("--no-norm-reward", dest="norm_reward", action="store_false",
                   default=True,
                   help="train on the RAW T2S reward instead of passing it through "
                        "VecNormalize. VecNormalize divides by a RUNNING std of the "
                        "discounted return, which drifts as the policy improves, so "
                        "the reward the critic learns from changes scale underneath "
                        "it. The expert checkpoints were trained on a fixed-scale "
                        "reward with no normalisation. The T2S reward is already "
                        "small (std ~0.9 per step), so it does not need rescaling.")
    p.add_argument("--time-penalty", type=float, default=None,
                   help="per-step time penalty (default: config.TIME_PENALTY). 0 gives "
                        "pure difference shaping, pred(s) - gamma*pred(s'). Note the "
                        "gate will FAIL it: with no penalty a constant prediction pays "
                        "(1-gamma)*pred > 0 per step, i.e. hovering is rewarded.")
    p.add_argument("--stall-steps", type=int, default=0,
                   help="end a TRAINING episode (true terminal, V=0) if the peg has not "
                        "moved closer to the goal for this many steps. 0 = off. Fixes "
                        "the hovering tie in `difference` mode without a time penalty. "
                        "Must exceed the approach phase — the peg does not move until "
                        "grasped, ~20-30 steps into an expert success — so 100 is a "
                        "safe start. Does NOT fix a model that pays for moving the arm "
                        "away from the peg: the arm is not stalled there.")
    p.add_argument("--ent-coef", default="auto",
                   help="'auto' learns alpha toward --target-entropy; a float pins it")
    p.add_argument("--target-entropy", default="auto",
                   help="'auto' = -dim(action) = -4. A HIGHER value (e.g. -2) keeps "
                        "the policy more stochastic for longer, which is the knob "
                        "for a narrow action window like closing the gripper.")
    p.add_argument("--preview-only", action="store_true",
                   help="run the gate and stop; no training")
    p.add_argument("--gamma-sweep", type=float, nargs="*", default=None,
                   help="run the gate at several gammas and stop, e.g. 0.99 0.999")
    p.add_argument("--force", action="store_true",
                   help="train even if the reward gate fails")
    p.add_argument("--no-video", dest="video", action="store_false", default=True,
                   help="skip the post-training policy videos")
    return p.parse_args(argv)


def _load_predictors(t2s_run_dir, combos, manifest):
    import t2s_predict
    specs, predictors = {}, {}
    for combo in combos:
        if combo not in manifest["combos"]:
            raise KeyError(f"{combo!r} not in {t2s_run_dir}; available: "
                           f"{sorted(manifest['combos'])}")
        best_seed = manifest["combos"][combo]["best_seed"]
        method, condition = combo.rsplit("_", 1)
        label = combo
        predictors[label] = t2s_predict.load_t2s_predictor(
            t2s_run_dir, method, condition, seed=best_seed)
        specs[label] = (t2s_run_dir, combo, best_seed)
    return specs, predictors


def run_gate(trajectories, predictors, reward_mode, gamma, verbose=True,
             time_penalty=None):
    """Reward preview + pass/fail verdict. Returns (verdicts, pred_maxes)."""
    import reward_preview

    preview = reward_preview.preview_reward(
        trajectories, predictors, reward_modes=(reward_mode,), gamma=gamma,
        time_penalty=time_penalty)
    verdicts = reward_preview.check_preview(preview, reward_mode,
                                            time_penalty=time_penalty)
    if verbose:
        print(reward_preview.format_preview_table(preview))
    return verdicts, {k: v["pred_max"] for k, v in verdicts.items()}


def main(argv=None):
    args = parse_args(argv)

    import config
    import results
    import t2s_eval

    gamma = config.RL_GAMMA if args.gamma is None else args.gamma
    seeds = list(config.RL_SEEDS if args.seeds is None else args.seeds)

    # ---- read Stage 2's handoff -----------------------------------------
    try:
        eval_dir = results.get_run_dir("t2s_eval", args.eval_run)
    except KeyError:
        print(f"ERROR: no t2s_eval run named {args.eval_run!r}. "
              f"Registered: {results.list_runs('t2s_eval')}\nRun:\n"
              f"  python run_3_eval.py --t2s-run {args.eval_run}", file=sys.stderr)
        return 1
    manifest_path = os.path.join(eval_dir, STAGE_MANIFEST)
    if not os.path.exists(manifest_path):
        print(f"ERROR: no {STAGE_MANIFEST} in {eval_dir} — evaluation did not finish.\n"
              f"Run:  python run_3_eval.py --t2s-run {args.eval_run}", file=sys.stderr)
        return 1
    with open(manifest_path) as f:
        stage3 = json.load(f)
    t2s_run_dir = stage3["t2s_run_dir"]

    combos = args.models
    if not combos:
        # rank by held-out-POLICY MAE, which is what Stage 3 measured; val MSE
        # ranks held-out ROWS and can disagree
        # MAE alone is the wrong selector: the best predictor measured on
        # expert success rollouts was also the most falsely optimistic on
        # failures (fail_pred_min 4.11 — it claims 4 steps to success on a
        # trajectory that never succeeds). Filter on the reward-relevant number
        # FIRST, then rank the survivors by MAE.
        def ev(c, k, default):
            v = stage3["combos"][c]["eval"].get(k)
            return default if v is None else v

        # threshold lives in t2s_eval so Stage 3 and Stage 4 cannot disagree
        thr = t2s_eval.MIN_FAIL_PRED
        ok = [c for c in stage3["combos"] if ev(c, "fail_pred_min", 0.0) >= thr]
        if not ok:
            ok = list(stage3["combos"])
            print(f"(no model has fail_pred_min >= {thr}; ranking all by MAE)")
        else:
            print(f"({len(ok)} of {len(stage3['combos'])} models pass "
                  f"fail_pred_min >= {thr})")
        combos = sorted(ok, key=lambda c: ev(c, "mae", float("inf")))[:args.n_best]
        print(f"(no --models given; took best {len(combos)} by MAE among those: {combos})")

    specs, predictors = _load_predictors(t2s_run_dir, combos, stage3)
    print(f"\n=== Stage 4: {len(specs)} model(s) x {len(seeds)} seed(s) = "
          f"{len(specs)*len(seeds)} runs @ {args.timesteps:,} steps ===")
    print(f"    reward_mode={args.reward_mode}  RL gamma={gamma}  "
          f"reward normalisation={'VecNormalize' if args.norm_reward else 'OFF (raw)'}"
          f"  stall termination={args.stall_steps or 'off'}")
    for label, (_d, combo, sd) in specs.items():
        ev = stage3["combos"][combo]["eval"]
        print(f"    {label:<28} t2s seed {sd}  mae={ev.get('mae')}  "
              f"slope={ev.get('calibration_slope')}")

    # ---- reference trajectories -----------------------------------------
    from stable_baselines3 import SAC

    succ_pol = SAC.load(os.path.join(config.EXPERT_POLICY_DIR, stage3["success_ckpt"]))
    fail_pol = SAC.load(os.path.join(config.EXPERT_POLICY_DIR, stage3["failure_ckpt"]))
    # SEVERAL seeds per scenario: return_gap used to come from one
    # success/failure pair, and every model was ranked by it
    trajectories = t2s_eval.collect_eval_trajectories(
        succ_pol, fail_pol, seeds=PREVIEW_SEEDS)

    # ---- gamma sweep: the cheap check, then stop ------------------------
    if args.gamma_sweep is not None:
        gammas = args.gamma_sweep or [0.99, 0.995, 0.999]
        print(f"\n=== gamma sweep (no training) ===")
        print(f"\n{'gamma':>8}  {'model':<28}{'return gap':>12}"
              f"{'hover@obs':>11}{'hover@ceil':>12}   verdict")
        print("-" * 84)
        for g in gammas:
            v, _ = run_gate(trajectories, predictors, args.reward_mode, g, verbose=False,
                            time_penalty=args.time_penalty)
            for label, d in v.items():
                gap = f"{d['return_gap']:,.0f}" if d["return_gap"] is not None else "--"
                verdict = "PASS" if d["ok"] else "FAIL"
                print(f"{g:>8}  {label:<28}{gap:>12}{d['hover_reward']:>+11.2f}"
                      f"{(d.get('hover_reward_guard') or 0):>+12.2f}   {verdict}")
        print("\nBOTH hover columns must be NEGATIVE. 'obs' is the expert's range;")
        print("'ceil' is the reachable ceiling, which is what the guard uses — an")
        print("untrained policy leaves the expert's region on step one.")
        return 0

    # ---- the gate --------------------------------------------------------
    print(f"\n=== reward gate (gamma={gamma}) ===")
    for scenario, trajs in trajectories.items():
        print(f"  {scenario}: " + ", ".join(
            f"seed{t['seed']} {len(t['obs'])}st succ@{t['success_step']}" for t in trajs))
    print()
    verdicts, pred_maxes = run_gate(trajectories, predictors, args.reward_mode, gamma,
                                    time_penalty=args.time_penalty)

    print(f"\n--- verdict for {args.reward_mode} ---")
    failed = []
    for label, v in verdicts.items():
        hg = v.get("hover_reward_guard")
        print(f"  {label:<30}{'PASS' if v['ok'] else 'FAIL'}  gap={v['return_gap']:,.0f}"
              f"  hover@observed({v['pred_max']:.0f})={v['hover_reward']:+.2f}"
              + (f"  hover@ceiling({v['guard_pred_max']:.0f})={hg:+.2f}"
                 if hg is not None else ""))
        for r in v["reasons"]:
            print(f"      - {r}")
        if not v["ok"]:
            failed.append(label)

    if args.preview_only:
        return 0 if not failed else 1
    if failed and not args.force:
        print(f"\n>>> {len(failed)} model(s) failed the gate; not training.")
        print(">>> Try --gamma-sweep 0.99 0.995 0.999, or --force to train anyway.")
        return 1

    # ---- train -----------------------------------------------------------
    import policy_train

    print(f"\n=== training ===")
    sweep_results = policy_train.run_sweep(
        specs, results, seeds=seeds, sweep_name=args.sweep_name,
        reward_mode=args.reward_mode, total_timesteps=args.timesteps, gamma=gamma,
        pred_maxes=pred_maxes, force=args.force, norm_reward=args.norm_reward,
        stall_steps=args.stall_steps,
        **({} if args.time_penalty is None else {"time_penalty": args.time_penalty}),
        ent_coef=(float(args.ent_coef) if args.ent_coef != "auto" else "auto"),
        target_entropy=(float(args.target_entropy)
                        if args.target_entropy != "auto" else "auto"),
        skip_existing=True, eval_freq=EVAL_FREQ, ckpt_freq=CKPT_FREQ,
        n_eval_episodes=N_EVAL_EPISODES)

    sweep_rows = policy_train.summarize_sweep(sweep_results)
    print()
    print(policy_train.format_sweep_table(sweep_rows))

    sweep_dir = results.new_run_dir(
        "policy", args.sweep_name,
        meta=dict(kind="sweep_summary", stage="4_policy", eval_run=args.eval_run,
                  models=combos, seeds=seeds, reward_mode=args.reward_mode,
                  gamma=gamma, timesteps=args.timesteps))
    with open(os.path.join(sweep_dir, "sweep_summary.json"), "w") as f:
        json.dump({"rows": sweep_rows,
                   "raw": {f"{c}|{s}": h for (c, s), h in sweep_results.items()}},
                  f, indent=2)

    # ---- video of what each trained policy actually does -----------------
    # Stage 3 filmed the EXPERT policies to judge the T2S models. This films
    # the POLICIES THOSE REWARDS PRODUCED, which is the question Stage 4 is
    # actually asking: success_rate says whether it worked, the video says what
    # it did instead. A policy that hovers empty-handed while the prediction
    # falls toward zero is obvious in two seconds and invisible in a table.
    videos = {}
    if args.video:
        import t2s_video
        import visualize

        print("\n=== policy videos ===")
        for label, (_d, combo, _sd) in specs.items():
            for sd in seeds:
                run_name = f"{args.sweep_name}_{label}_seed{sd}"
                try:
                    run_dir = results.get_run_dir("policy", run_name)
                    policy, _path = visualize.load_policy("policy_final.zip", run_dir)
                except (KeyError, FileNotFoundError) as e:
                    print(f"  {run_name}: no final checkpoint ({e.__class__.__name__})")
                    continue
                roll = t2s_video.rollout_with_frames(
                    policy, predict_t2s=predictors[label], seed=0)
                path = os.path.join(run_dir, "policy_t2s_live.mp4")
                _f, info = t2s_video.render_t2s_video(
                    roll, save_path=path, reward_mode=args.reward_mode, gamma=gamma,
                    label=f"{label} seed{sd} | policy")
                videos[run_name] = info
                # build the status OUTSIDE the f-string: a replacement field
                # spanning several lines is PEP 701 syntax and only parses on
                # Python 3.12+, so it raised SyntaxError on older interpreters
                ss = info["success_step"]
                status = f"SUCCESS @{ss}" if ss is not None else "never succeeded"
                flag = "  <<< FALSELY OPTIMISTIC" if info["entered_danger_band"] else ""
                print(f"  {run_name}: {status}"
                      f"  pred {info['pred_min']:.0f}-{info['pred_max']:.0f}{flag}")

    manifest = dict(
        stage="4_policy", sweep_name=args.sweep_name, sweep_dir=sweep_dir,
        eval_run=args.eval_run, t2s_run_dir=t2s_run_dir, data_run=stage3["data_run"],
        reward_mode=args.reward_mode, gamma=gamma, seeds=seeds,
        timesteps=args.timesteps, models=combos,
        gate=({k: {kk: vv for kk, vv in v.items() if kk != "reasons"}
               for k, v in verdicts.items()}),
        rows=sweep_rows, videos=videos, ok=True)
    with open(os.path.join(sweep_dir, STAGE_MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nwrote {os.path.join(sweep_dir, STAGE_MANIFEST)}")
    print("\nin eval_history.json, read these together:")
    print("  success_rate        did it solve the task")
    print("  peg_moved_rate      0 means it never grasped -> exploration problem")
    print("  mean_false_optimism lowest prediction on episodes that did NOT succeed;")
    print("                      LOW is bad, it is the empty-gripper hover as a number")
    print("  mean_reward_nonneg  share of steps paid >= 0 while failing")
    print("  mean_dead_step_frac share of steps with no gradient at all")
    return 0


if __name__ == "__main__":
    sys.exit(main())