#!/usr/bin/env python3
"""
STAGE 2 of 4 — T2S model training.

    python run_2_train.py --data-run v2

Trains the full grid on Stage 1's dataset and stops. Evaluation is
run_3_eval.py; splitting them means a failed or rethought evaluation costs
minutes instead of a retrain, and the two manifests say plainly which models
exist and which have been scored.

THE GRID IS THE EXPERIMENT
--------------------------
    methods         mc | td0 | tdlambda    the three value targets
    conditions      succ | all             the two data sources
    censor_schemes  censor | bootstrap     value bootstrap, or not
    = 12 models, named "<method>[boot]_<condition>"

ONE SEED, DELIBERATELY. The scene is pinned, the dataset is fixed, and the
train/val split is fixed at collection time, so a second training seed varies
only network initialisation and batch order. That measures optimiser noise,
not which formulation is better, and averaging it into a `mean_val_mse ± std`
made the seed loop look like evidence it was not. Seeds earn their keep in
Stage 3 (rollouts genuinely differ) and in the Stage 4 RL sweep (exploration
differs). Override with --seed if you want a different draw.

TWO GAMMAS, DO NOT UNIFY
------------------------
config.T2S_BOOTSTRAP_GAMMA (0.99) sets the fixed point dt/(1-gamma)=100 for
the never-succeeding recursion. NOT config.RL_GAMMA (0.999), which Stage 4
uses for shaping. Raising this one to 0.999 puts the fixed point at 1000,
above the prediction clip, reinstating the arbitrary ceiling the bootstrap
scheme exists to remove.

Exit codes: 0 ok, 1 Stage 1's output is missing or nothing trained.
"""
import argparse
import json
import os
import sys

STAGE_MANIFEST = "stage_manifest.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Stage 2: T2S training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-run", required=True, help="run name from run_1_collect.py")
    p.add_argument("--run-name", default="v3", help="t2s_model run name")
    p.add_argument("--seed", type=int, default=0, help="single training seed")
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    p.add_argument("--quick", action="store_true",
                   help="one combo only — smoke test, not a result")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    import torch

    import config
    import results
    import t2s_train

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Stage 1's handoff ----------------------------------------------
    try:
        data_dir = results.get_run_dir("data_collection", args.data_run)
    except KeyError:
        print(f"ERROR: no data_collection run named {args.data_run!r}. "
              f"Registered: {results.list_runs('data_collection')}\n"
              f"Run:  python run_1_collect.py --run-name {args.data_run}", file=sys.stderr)
        return 1
    
    manifest_path = os.path.join(data_dir, STAGE_MANIFEST)
    if not os.path.exists(manifest_path):
        print(f"ERROR: no {STAGE_MANIFEST} in {data_dir} — that run predates the "
              "staged scripts, or collection did not finish.", file=sys.stderr)
        return 1
    
    with open(manifest_path) as f:
        stage1 = json.load(f)
    if not stage1.get("ok"):
        print(f"WARNING: Stage 1 reported blocking audit failures "
              f"{stage1.get('audit_blocking')}; results here may be meaningless.\n")

    grid = (dict(methods=("td0",), conditions=("succ",), censor_schemes=("bootstrap",))
            if args.quick else
            dict(methods=t2s_train.METHODS, conditions=t2s_train.CONDITIONS,
                 censor_schemes=t2s_train.CENSOR_SCHEMES))
    n = len(grid["methods"]) * len(grid["conditions"]) * len(grid["censor_schemes"])

    run_dir = results.new_run_dir(
        "t2s_model", args.run_name,
        meta=dict(stage="2_train", data_run=args.data_run, seed=args.seed,
                  quick=args.quick, bootstrap_gamma=config.T2S_BOOTSTRAP_GAMMA))
    print(f"=== Stage 2: training {n} model(s) on {args.data_run} "
          f"({stage1['total_episodes']} episodes) -> {run_dir} ===")
    print(f"    device={device}  seed={args.seed}  "
          f"bootstrap gamma={config.T2S_BOOTSTRAP_GAMMA}\n")

    _models, _hist, rows = t2s_train.run_all_combos(
        run_dir, stage1["dataset_path"], seed=args.seed, device=device,
        bootstrap_gamma=config.T2S_BOOTSTRAP_GAMMA, **grid)

    if not rows:
        print("ERROR: nothing trained", file=sys.stderr)
        return 1

    print(f"\n{'combo':<22}{'method':<10}{'data':<7}{'scheme':<11}{'gamma':>7}"
          f"{'val MSE':>11}{'succ only':>11}")
    print("-" * 79)
    for r in sorted(rows, key=lambda r: r["val_mse"]):
        print(f"{r['combo']:<22}{r['method']:<10}{r['condition']:<7}"
              f"{r['censor_scheme']:<11}{r['gamma']:>7.2f}"
              f"{r['val_mse']:>11.2f}{r['val_mse_succ_only']:>11.2f}")
    print("\nval MSE    : all val rows; failed-episode rows carry a censored label,")
    print("             so this partly scores agreement with a fiction")
    print("succ only  : val rows from SUCCESSFUL episodes — the honest number")
    print("These rank fit to held-out ROWS. Stage 3 evaluates held-out POLICIES,")
    print("which is the stronger test and can rank differently.")

    manifest = dict(
        stage="2_train", run_name=args.run_name, run_dir=run_dir,
        data_run=args.data_run, data_run_dir=data_dir,
        dataset_path=stage1["dataset_path"],
        dataset_summary_path=stage1["dataset_summary_path"],
        holdout_checkpoints=stage1["holdout_checkpoints"],
        seed=args.seed, device=device,
        t2s_bootstrap_gamma=config.T2S_BOOTSTRAP_GAMMA,
        combos={r["combo"]: {k: r[k] for k in
                             ("method", "condition", "censor_scheme", "gamma",
                              "val_mse", "val_mse_succ_only", "best_seed")}
                for r in rows},
        ok=True)
    with open(os.path.join(run_dir, STAGE_MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nwrote {os.path.join(run_dir, STAGE_MANIFEST)}")
    print(f"next:  python run_3_eval.py --t2s-run {args.run_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())