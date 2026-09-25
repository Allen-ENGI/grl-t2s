#!/usr/bin/env python3
"""
STAGE 1 of 4 — data collection.

    python run_1_collect.py --run-name v3

Collects from every expert checkpoint (minus any holdout) on the fixed scene,
with the policy sampled and no action noise. Assigns the train/val split,
audits the result, and writes a manifest for run_2_train.py.
"""
import argparse
import json
import os
import sys

import config
import data_collection
import dataset_audit
import results
from env_utils import ensure_fixed_task

BLOCKING = ("success_boundary", "failure_coverage", "split_integrity")
ADVISORY = ("exploration_coverage", "object_movement", "label_sanity",
            "trajectory_uniqueness")
STAGE_MANIFEST = "stage_manifest.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Stage 1: T2S data collection (checkpoints only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run-name", default="v3", help="run name; run_2 is pointed at this")
    p.add_argument("--holdout", nargs="*", default=[],
                   help="checkpoints to reserve and not collect from")
    p.add_argument("--reference-policies", nargs=2, default=None,
                   metavar=("SUCCESS_CKPT", "FAILURE_CKPT"),
                   help="policies for the stored reference trajectories "
                        "(default: strongest and weakest discovered)")
    p.add_argument("--no-references", dest="references", action="store_false",
                   default=True, help="skip the stored reference set")
    p.add_argument("--episodes", type=int, default=80,
                   help="episodes per checkpoint per seed")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="independent passes; the main diversity knob")
    p.add_argument("--val", type=float, default=0.2, help="validation share")
    p.add_argument("--allow-flagged", action="store_true",
                   help="exit 0 even if a blocking audit check fails")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config.ensure_dirs()
    ensure_fixed_task()

    run_dir = results.new_run_dir(
        "data_collection", args.run_name,
        meta=dict(stage="1_collect", seeds=args.seeds, episodes=args.episodes,
                  val_fraction=args.val, holdout=args.holdout,
                  policy_sampling="stochastic", action_noise=None, scene="fixed"))
    print(f"=== Stage 1: data collection -> {run_dir} ===\n")

    summary = data_collection.collect(
        run_dir, holdout=args.holdout, episodes=args.episodes,
        seeds=tuple(args.seeds), val_fraction=args.val,
        references=args.references,
        reference_policies=tuple(args.reference_policies)
        if args.reference_policies else None)

    print("\n=== audit ===")
    dataset_path = os.path.join(run_dir, "dataset.npz")
    summary_path = os.path.join(run_dir, "dataset_summary.json")
    audit = dataset_audit.audit_dataset(dataset_path, summary_path=summary_path)
    blocking = [k for k in BLOCKING if audit.get(k, {}).get("verdict") != "OK"]
    advisory = [k for k in ADVISORY if audit.get(k, {}).get("verdict") != "OK"]

    if advisory:
        print(f"\n[advisory] {advisory} — worth reading, not blocking")
    if blocking:
        print(f"\n>>> BLOCKING: {blocking}")
        for k in blocking:
            print(f"      {k}: {audit[k]['verdict']}")
        if "failure_coverage" in blocking:
            print(">>> no/few failures: save earlier checkpoints in expert_train.py; "
                  "'succ' and 'all' would otherwise train identical models")
    else:
        print("\n>>> all blocking checks passed")

    manifest = dict(
        stage="1_collect", run_name=args.run_name, run_dir=run_dir,
        dataset_path=dataset_path, dataset_summary_path=summary_path,
        holdout_checkpoints=summary["holdout_checkpoints"],
        collected_checkpoints=summary["collected_checkpoints"],
        per_checkpoint=summary["per_checkpoint"],
        references=summary.get("references"),
        references_dir=os.path.join(run_dir, "references"),
        failure_sources=summary["failure_sources"], split=summary["split"],
        total_episodes=summary["total_episodes"],
        duplicate_fraction=summary["duplicate_fraction"],
        audit_blocking=blocking, audit_advisory=advisory, ok=not blocking)
    with open(os.path.join(run_dir, STAGE_MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nwrote {os.path.join(run_dir, STAGE_MANIFEST)}")
    print(f"next:  python run_2_train.py --data-run {args.run_name} --run-name {args.run_name}")
    return 0 if (manifest["ok"] or args.allow_flagged) else 1


if __name__ == "__main__":
    sys.exit(main())