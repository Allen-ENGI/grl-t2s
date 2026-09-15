#!/usr/bin/env python3
"""
Command-line data collection, for shell scripts and parallel runs.

Every knob the notebook exposes is a flag here, so the same collection can be
reproduced from a script or fanned out across processes. Each invocation
writes to its own run directory, so parallel calls never collide.

Examples
--------
Single run, three seeds, double the episode counts:

    python collect_cli.py --run-name v9 --seeds 0 1 2 --multiplier 2

Fan out one process per seed (each gets its own run dir), then merge:

    for s in 0 1 2; do
        python collect_cli.py --run-name v9_s$s --seeds $s &
    done
    wait
    python collect_cli.py --merge v9 v9_s0 v9_s1 v9_s2

Reproduce an existing run exactly (same seeds, same flags) — collection is
deterministic given the master seeds, because each episode draws from a
np.random.Generator seeded from them rather than from global NumPy state.
"""
import argparse
import json
import os
import sys

import numpy as np


def build_parser():
    p = argparse.ArgumentParser(description="T2S data collection")

    p.add_argument("--run-name", default="cli", help="results run name under data_collection/")
    p.add_argument("--expert-dir", default=None,
                    help="expert checkpoint dir (default: config.EXPERT_POLICY_DIR)")

    # diversity
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="master collection seeds; each re-runs the whole plan independently")
    p.add_argument("--deterministic", action="store_true",
                    help="take the policy MEAN instead of sampling. NOT recommended: with a "
                         "pinned scene this makes every clean rollout bit-identical")
    p.add_argument("--multiplier", type=float, default=1.0,
                    help="scale every episode count in the plan")
    p.add_argument("--episodes-per-checkpoint", type=int, default=40)

    # what to collect from
    p.add_argument("--checkpoints", nargs="*", default=None,
                    help="explicit checkpoint filenames; omit to auto-select earliest/mid/final")
    p.add_argument("--noise-variants", action="store_true", help="add noise=0.15/0.4 batches")
    p.add_argument("--random-policy", action="store_true", help="add random-action episodes")

    # failure source
    p.add_argument("--no-forced-failure", action="store_true")
    p.add_argument("--failure-checkpoint", default=None)
    p.add_argument("--failure-episodes", type=int, default=30)
    p.add_argument("--failure-noise", type=float, nargs="+",
                    default=[0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0])
    p.add_argument("--failure-max-success-rate", type=float, default=0.4)
    p.add_argument("--failure-probes", type=int, default=5)

    p.add_argument("--stage-labels", action="store_true", help="compute stage labels")
    p.add_argument("--audit", action="store_true", help="run dataset_audit after collecting")

    p.add_argument("--merge", nargs="+", metavar=("TARGET", "SOURCE"),
                    help="merge existing runs: --merge TARGET SRC1 SRC2 ...")
    return p


def merge_runs(target_name, source_names):
    """Concatenate several collected datasets into one, renumbering episode ids."""
    import results

    parts = {k: [] for k in ("X", "y_steps", "stage", "frame_idxs", "collection_seeds")}
    ep_ids, offset = [], 0
    summaries = []

    for name in source_names:
        d = np.load(os.path.join(results.get_run_dir("data_collection", name), "dataset.npz"))
        for k in parts:
            if k in d.files:
                parts[k].append(d[k])
        ids = d["episode_ids"]
        ep_ids.append(ids + offset)          # keep episodes distinct across sources
        offset += int(ids.max()) + 1
        spath = os.path.join(results.get_run_dir("data_collection", name), "dataset_summary.json")
        if os.path.exists(spath):
            summaries.append(json.load(open(spath)))

    target_dir = results.new_run_dir("data_collection", target_name,
                                      meta={"merged_from": list(source_names)})
    out = {k: np.concatenate(v) for k, v in parts.items() if v}
    out["episode_ids"] = np.concatenate(ep_ids)
    np.savez(os.path.join(target_dir, "dataset.npz"), **out)

    n_ep = int(out["episode_ids"].max()) + 1
    merged = dict(
        total_rows=int(len(out["X"])), total_episodes=n_ep,
        obs_dim=int(out["X"].shape[1]),
        merged_from=list(source_names),
        successful_episodes=sum(s.get("successful_episodes", 0) for s in summaries),
        failed_episodes=sum(s.get("failed_episodes", 0) for s in summaries),
        censor_label=summaries[0].get("censor_label") if summaries else None,
    )
    with open(os.path.join(target_dir, "dataset_summary.json"), "w") as f:
        json.dump(merged, f, indent=2)

    print(f"merged {len(source_names)} run(s) -> {target_dir}")
    print(f"  {merged['total_episodes']} episodes, {merged['total_rows']:,} rows")
    return target_dir


def main(argv=None):
    args = build_parser().parse_args(argv)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config
    import results
    import data_collection
    from env_utils import ensure_fixed_task

    if args.merge:
        merge_runs(args.merge[0], args.merge[1:])
        return 0

    config.ensure_dirs()
    ensure_fixed_task()
    expert_dir = args.expert_dir or config.EXPERT_POLICY_DIR

    run_dir = results.new_run_dir(
        "data_collection", args.run_name,
        meta=dict(seeds=args.seeds, deterministic=args.deterministic,
                   multiplier=args.multiplier, expert_dir=expert_dir))

    plan = data_collection.default_collection_plan(
        expert_policy_dir=expert_dir,
        checkpoints=args.checkpoints,
        include_noise_variants=args.noise_variants,
        include_random_policy=args.random_policy,
        n_episodes_per_checkpoint=args.episodes_per_checkpoint,
    )

    print(f"collecting into {run_dir}")
    print(f"  seeds={args.seeds} deterministic={args.deterministic} "
          f"multiplier={args.multiplier}")

    summary = data_collection.collect_dataset(
        run_dir, plan=plan,
        expert_policy_dir=expert_dir,
        collection_seeds=args.seeds,
        deterministic=args.deterministic,
        episodes_multiplier=args.multiplier,
        include_stage_label=args.stage_labels,
        force_failure_source=not args.no_forced_failure,
        failure_checkpoint=args.failure_checkpoint,
        n_failure_episodes=args.failure_episodes,
        failure_noise_candidates=tuple(args.failure_noise),
        failure_max_success_rate=args.failure_max_success_rate,
        failure_n_probe=args.failure_probes,
    )

    if args.audit:
        import dataset_audit
        dataset_audit.audit_dataset(
            os.path.join(run_dir, "dataset.npz"),
            summary_path=os.path.join(run_dir, "dataset_summary.json"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
