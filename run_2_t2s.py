#!/usr/bin/env python3
"""
STAGE 2 of 4 — T2S model training.

    python run_2_t2s.py --data-run v3                              # full grid
    python run_2_t2s.py --data-run v3 --model tdlambdaboot_all     # one model

Trains on Stage 1's dataset and stops. Evaluation is run_3_eval.py; splitting
them means a failed or rethought evaluation costs minutes instead of a retrain,
and the two manifests say plainly which models exist and which have been scored.
"""
import argparse
import json
import os
import sys
import torch

import config
import results
import t2s_train


STAGE_MANIFEST = "stage_manifest.json"

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Stage 2: T2S training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-run", required=True, help="run name from run_1_collect.py")
    p.add_argument("--run-name", required=True, help="t2s_model run name")
    p.add_argument("--seed", type=int, default=0, help="single training seed")
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    p.add_argument("--models", "--model", dest="models", nargs="+", default=None,
                   help="train just these combos, e.g. tdlambdaboot_all tdlambdaboot_succ "
                        "(default: the full grid)")
    return p.parse_args(argv)


def model_grid(name):
    """'tdlambdaboot_all' -> the one-model grid for run_all_combos."""
    method, cond = name.rsplit("_", 1)
    boot = method.endswith("boot")
    method = method[:-4] if boot else method
    if method not in t2s_train.METHODS or cond not in t2s_train.CONDITIONS:
        valid = [f"{m}{b}_{c}" for b in ("", "boot") for m in t2s_train.METHODS
                 for c in t2s_train.CONDITIONS]
        raise SystemExit(f"ERROR: unknown model {name!r}. Valid: {valid}")
    return dict(methods=(method,), conditions=(cond,),
                censor_schemes=("bootstrap" if boot else "censor",))


def main(argv=None):
    args = parse_args(argv)

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

    grids = ([model_grid(m) for m in args.models] if args.models else
             [dict(methods=t2s_train.METHODS, conditions=t2s_train.CONDITIONS,
                   censor_schemes=t2s_train.CENSOR_SCHEMES)])
    n = sum(len(g["methods"]) * len(g["conditions"]) * len(g["censor_schemes"]) for g in grids)

    if args.run_name in results.list_runs("t2s_model"):
        print(f"ERROR: T2S run {args.run_name!r} already exists; training into it would "
                "overwrite its checkpoints and manifests. Use a new --run-name.", file=sys.stderr)
        return 1

    run_dir = results.new_run_dir(
        "t2s_model", args.run_name,
        meta=dict(stage="2_train", data_run=args.data_run, seed=args.seed,
                  model=args.models, bootstrap_gamma=config.T2S_BOOTSTRAP_GAMMA))
    print(f"=== Stage 2: training {n} model(s) on {args.data_run} "
          f"({stage1['total_episodes']} episodes) -> {run_dir} ===")
    print(f"    device={device}  seed={args.seed}  "
          f"bootstrap gamma={config.T2S_BOOTSTRAP_GAMMA}\n")

    rows, merged = [], None
    for g in grids:
        _models, _hist, r = t2s_train.run_all_combos(
            run_dir, stage1["dataset_path"], seed=args.seed, device=device,
            bootstrap_gamma=config.T2S_BOOTSTRAP_GAMMA, **g)
        rows += r
        m = json.load(open(os.path.join(run_dir, "manifest.json")))
        if merged is None:
            merged = m
        else:
            merged["combos"].update(m["combos"])
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(merged, f, indent=2)
        
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