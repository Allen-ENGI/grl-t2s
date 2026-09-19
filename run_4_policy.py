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
        description="Stage 3: downstream RL",
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
    p.add_argument("--reward-mode", default="difference_timed",
                   choices=["absolute", "difference", "difference_timed"])
    p.add_argument("--gamma", type=float, default=None,
                   help="RL discount (default: config.RL_GAMMA)")
    p.add_argument("--timesteps", type=int, default=400_000)
    p.add_argument("--preview-only", action="store_true",
                   help="run the gate and stop; no training")
    p.add_argument("--gamma-sweep", type=float, nargs="*", default=None,
                   help="run the gate at several gammas and stop, e.g. 0.99 0.999")
    p.add_argument("--force", action="store_true",
                   help="train even if the reward gate fails")
    return p.parse_args(argv)


def _load_predictors(t2s_run_dir, combos, manifest, run_label=""):
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


def run_gate(trajectories, predictors, reward_mode, gamma, verbose=True):
    """Reward preview + pass/fail verdict. Returns (verdicts, pred_maxes)."""
    import reward_preview

    preview = reward_preview.preview_reward(
        trajectories, predictors, reward_modes=(reward_mode,), gamma=gamma)
    verdicts = reward_preview.check_preview(preview, reward_mode)
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
        def mae(c):
            v = stage3["combos"][c]["eval"].get("mae")
            return float("inf") if v is None else v
        combos = sorted(stage3["combos"], key=mae)[:args.n_best]
        print(f"(no --models given; took best {len(combos)} by held-out MAE: {combos})")

    specs, predictors = _load_predictors(t2s_run_dir, combos, stage3)
    print(f"\n=== Stage 3: {len(specs)} model(s) x {len(seeds)} seed(s) = "
          f"{len(specs)*len(seeds)} runs @ {args.timesteps:,} steps ===")
    print(f"    reward_mode={args.reward_mode}  RL gamma={gamma}")
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
        print(f"\n{'gamma':>8}{'model':<30}{'return gap':>16}{'hover':>9}  verdict")
        print("-" * 72)
        for g in gammas:
            v, _ = run_gate(trajectories, predictors, args.reward_mode, g, verbose=False)
            for label, d in v.items():
                gap = f"{d['return_gap']:,.0f}" if d["return_gap"] is not None else "--"
                print(f"{g:>8}{label:<30}{gap:>16}{d['hover_reward']:>+9.2f}"
                      f"  {'PASS' if d['ok'] else 'FAIL'}")
        print("\nhover must be NEGATIVE, or the policy is paid to stand still.")
        return 0

    # ---- the gate --------------------------------------------------------
    print(f"\n=== reward gate (gamma={gamma}) ===")
    for scenario, trajs in trajectories.items():
        print(f"  {scenario}: " + ", ".join(
            f"seed{t['seed']} {len(t['obs'])}st succ@{t['success_step']}" for t in trajs))
    print()
    verdicts, pred_maxes = run_gate(trajectories, predictors, args.reward_mode, gamma)

    print(f"\n--- verdict for {args.reward_mode} ---")
    failed = []
    for label, v in verdicts.items():
        print(f"  {label:<30}{'PASS' if v['ok'] else 'FAIL'}  "
              f"gap={v['return_gap']:,.0f}  hover={v['hover_reward']:+.2f}  "
              f"pred_max={v['pred_max']:.0f}")
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
        pred_maxes=pred_maxes,         # real prediction range -> real hovering check
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

    manifest = dict(
        stage="4_policy", sweep_name=args.sweep_name, sweep_dir=sweep_dir,
        eval_run=args.eval_run, t2s_run_dir=t2s_run_dir, data_run=stage3["data_run"],
        reward_mode=args.reward_mode, gamma=gamma, seeds=seeds,
        timesteps=args.timesteps, models=combos,
        gate=({k: {kk: vv for kk, vv in v.items() if kk != "reasons"}
               for k, v in verdicts.items()}),
        rows=sweep_rows, ok=True)
    with open(os.path.join(sweep_dir, STAGE_MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nwrote {os.path.join(sweep_dir, STAGE_MANIFEST)}")
    print("inspect a policy: see the Section 8 cell in pipeline_cells.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())