#!/usr/bin/env python3
"""
Film the checkpoint where the peg got closest to the hole, or any checkpoint you name.

    python closest_video.py --policy-run diff_v3_tdlambdaboot_all_seed0 --t2s-run v7 --model tdlambdaboot_all
    python closest_video.py ... --metric best_min_peg_goal_distance
    python closest_video.py ... --checkpoint policy_200000.zip
    python closest_video.py ... --list                 # just show the ranking

Closest is read from the run's eval_history.json (peg-to-goal distance at each
eval), then matched to the nearest saved policy_<step>.zip.
"""
import argparse
import glob
import json
import os
import re

import results


def saved_checkpoints(run_dir):
    out = {}
    for p in glob.glob(os.path.join(run_dir, "policy_*.zip")):
        m = re.search(r"policy_(\d+)\.zip$", p)
        if m:
            out[int(m.group(1))] = os.path.basename(p)
    return out


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--policy-run", required=True)
    p.add_argument("--t2s-run", required=True, help="for the prediction overlay")
    p.add_argument("--model", required=True, help="T2S combo for the overlay")
    p.add_argument("--metric", default="mean_min_peg_goal_distance",
                   choices=["mean_min_peg_goal_distance", "best_min_peg_goal_distance",
                            "mean_final_peg_goal_distance", "best_min_hand_peg_distance"],
                   help="mean_* = averaged over eval episodes (robust); best_* = single closest episode")
    p.add_argument("--checkpoint", default=None, help="film this one instead of the closest")
    p.add_argument("--seeds", type=int, nargs="+", default=[500])
    p.add_argument("--reward-mode", default="difference_timed",
                   choices=["absolute", "difference", "difference_timed"])
    p.add_argument("--list", action="store_true", help="print the ranking and stop")
    args = p.parse_args()

    run_dir = results.get_run_dir("policy", args.policy_run)
    ckpts = saved_checkpoints(run_dir)
    if not ckpts:
        raise SystemExit(f"no policy_<step>.zip checkpoints in {run_dir}")

    history = json.load(open(os.path.join(run_dir, "eval_history.json")))
    scored = []
    for e in history:
        if e.get(args.metric) is None:
            continue
        step = min(ckpts, key=lambda s: abs(s - e["step"]))    # nearest saved checkpoint
        scored.append((e[args.metric], e["step"], step, e.get("success_rate")))
    scored.sort()
    if not scored and not args.checkpoint:
        raise SystemExit(f"eval_history.json has no {args.metric!r}; pass --checkpoint")

    print(f"{args.policy_run}: {len(ckpts)} checkpoints, ranked by {args.metric}")
    print(f"  {'distance':>10}{'eval step':>11}{'checkpoint':>22}{'success':>9}")
    for d, s, c, sr in scored[:10]:
        print(f"  {d:>10.4f}{s:>11,}{ckpts[c]:>22}{sr if sr is not None else '--':>9}")
    if args.list:
        return 0

    ck = args.checkpoint or ckpts[scored[0][2]]
    print(f"\nfilming {ck}")
    import make_videos
    return make_videos.main(["--t2s-run", args.t2s_run, "--models", args.model,
                             "--policy-run", args.policy_run, "--checkpoint", ck,
                             "--reward-mode", args.reward_mode,
                             "--seeds", *map(str, args.seeds),
                             "--out", os.path.join(run_dir, "videos", ck[:-4])])


if __name__ == "__main__":
    main()