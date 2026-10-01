#!/usr/bin/env python3
"""
Add episodes from your trained (failing) policies to an existing dataset, as a NEW data run.

    python collect_extra.py --base-data-run v3 --new-data-run v4 \
        --sources dt_fix_v1_tdlambdaboot_all_seed0/policy_150000.zip \
                  dt_fix_v1_tdlambdaboot_all_seed0/policy_final.zip \
                  dt_fix_v1_tdlambdaboot_all_seed1/policy_final.zip \
        --episodes 60 --save-video 3

Each source policy acts for the whole episode, exactly like the original collection:
fixed scene, up to 500 steps, sampled actions, no added noise, labels = steps to the
first success (300 if it never succeeds). Episodes that never succeed are treated as
failures by T2S training; the rare one that succeeds keeps its correct countdown.

The new run holds the base dataset unchanged plus the new episodes (all in the TRAIN
split, so validation matches the base run), in the same file format as the original.
The base run is only read, never written.
"""
import argparse
import json
import os
import shutil

import numpy as np
import torch
from stable_baselines3 import SAC

import results
from config import SUCCESS_KEY, MAX_STEPS, CENSOR_LABEL, steps_remaining_curve
from env_utils import make_fixed_scene_env

STAGE_MANIFEST = "stage_manifest.json"


def run_episode(policy, seed, save_video_path=None):
    """One episode of `policy` on the fixed scene -> (observations, first success index)."""
    env = make_fixed_scene_env(render_mode="rgb_array" if save_video_path else None)
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    observations, frames, first_success = [np.asarray(obs, np.float32)], [], None
    if save_video_path:
        frames.append(env.render())
    for t in range(MAX_STEPS):
        action, _ = policy.predict(obs, deterministic=False)
        obs, _r, terminated, truncated, info = env.step(action)
        observations.append(np.asarray(obs, np.float32))
        if save_video_path:
            frames.append(env.render())
        if info.get(SUCCESS_KEY, 0) and first_success is None:
            first_success = t + 1
        if terminated or truncated:
            break
    env.close()
    if save_video_path:
        from t2s_video import save_video
        save_video(frames, save_video_path)
    return np.array(observations), first_success


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--base-data-run", required=True, help="existing data run to extend")
    p.add_argument("--new-data-run", required=True, help="name for the extended run (must be new)")
    p.add_argument("--sources", nargs="+", required=True,
                   help="trained policies as <policy run>/<checkpoint>.zip")
    p.add_argument("--episodes", type=int, default=60, help="episodes per source checkpoint")
    p.add_argument("--first-seed", type=int, default=700_000)
    p.add_argument("--save-video", type=int, default=0,
                   help="save the first N episodes of each source as .mp4 "
                        "(videos/episode_<id>.mp4); 0 = none")
    args = p.parse_args()

    if args.new_data_run in results.list_runs("data_collection"):
        raise SystemExit(f"ERROR: data run {args.new_data_run!r} already exists; pick a new name")

    base_dir = results.get_run_dir("data_collection", args.base_data_run)
    base = dict(np.load(os.path.join(base_dir, "dataset.npz")))
    n_base = len(base["X"])
    next_id = int(base["episode_ids"].max()) + 1
    print(f"base {args.base_data_run}: {len(np.unique(base['episode_ids']))} episodes, "
          f"{n_base:,} rows")

    new_dir = results.new_run_dir("data_collection", args.new_data_run,
                                  meta=dict(stage="1_collect_extra", base=args.base_data_run,
                                            sources=args.sources))
    video_dir = os.path.join(new_dir, "videos")
    if args.save_video:
        os.makedirs(video_dir, exist_ok=True)

    seed, episodes, per_source = args.first_seed, [], []
    for src in args.sources:
        run, ckpt = src.split("/", 1)
        policy = SAC.load(os.path.join(results.get_run_dir("policy", run), ckpt), device="cpu")
        first_id = next_id + len(episodes)
        n_succ = 0
        for i in range(args.episodes):
            ep_id = next_id + len(episodes)
            path = (os.path.join(video_dir, f"episode_{ep_id}.mp4")
                    if i < args.save_video else None)
            obs, fs = run_episode(policy, seed, save_video_path=path)
            seed += 1
            episodes.append(dict(obs=obs, y=steps_remaining_curve(len(obs), fs)))
            n_succ += fs is not None
        per_source.append(dict(source=src, episodes=args.episodes, failed=args.episodes - n_succ,
                               successful=n_succ, episode_ids=[first_id, next_id + len(episodes) - 1]))
        print(f"  {src}: {args.episodes - n_succ} failed, {n_succ} succeeded "
              f"(episode ids {first_id}-{next_id + len(episodes) - 1})")

    # ---- append to the base dataset, same arrays and format as the original ----
    new = dict(
        X=np.concatenate([e["obs"] for e in episodes]),
        y_steps=np.concatenate([e["y"] for e in episodes]),
        episode_ids=np.concatenate([np.full(len(e["obs"]), next_id + i, np.int32)
                                    for i, e in enumerate(episodes)]),
        frame_idxs=np.concatenate([np.arange(len(e["obs"]), dtype=np.int32) for e in episodes]),
        collection_seeds=np.concatenate([np.full(len(e["obs"]), -1, np.int32) for e in episodes]),
        split=np.concatenate([np.full(len(e["obs"]), "train", dtype="U5") for e in episodes]),
    )
    merged = {}
    for k in new:
        old = base[k] if k in base else np.zeros(n_base, new[k].dtype)
        merged[k] = np.concatenate([old, new[k]])
    np.savez(os.path.join(new_dir, "dataset.npz"), **merged)

    if os.path.isdir(os.path.join(base_dir, "references")):
        shutil.copytree(os.path.join(base_dir, "references"), os.path.join(new_dir, "references"),
                        dirs_exist_ok=True)

    summary = json.load(open(os.path.join(base_dir, "dataset_summary.json")))
    n_fail = sum(s["failed"] for s in per_source)
    summary.update(
        total_rows=int(len(merged["X"])),
        total_episodes=int(len(np.unique(merged["episode_ids"]))),
        failed_episodes=int(summary.get("failed_episodes", 0)) + n_fail,
        successful_episodes=int(summary.get("successful_episodes", 0)) + len(episodes) - n_fail,
        extra=dict(base_data_run=args.base_data_run, new_episodes=len(episodes),
                   new_failed=n_fail, per_source=per_source, split="train",
                   videos=video_dir if args.save_video else None))
    summary_path = os.path.join(new_dir, "dataset_summary.json")
    json.dump(summary, open(summary_path, "w"), indent=2)

    base_manifest = json.load(open(os.path.join(base_dir, STAGE_MANIFEST)))
    manifest = dict(base_manifest, run_name=args.new_data_run, run_dir=new_dir,
                    dataset_path=os.path.join(new_dir, "dataset.npz"),
                    dataset_summary_path=summary_path,
                    references_dir=os.path.join(new_dir, "references"),
                    total_episodes=summary["total_episodes"], extended_from=args.base_data_run)
    json.dump(manifest, open(os.path.join(new_dir, STAGE_MANIFEST), "w"), indent=2)

    print(f"\nadded {len(episodes)} episodes ({n_fail} failed, {len(episodes) - n_fail} succeeded), "
          f"{len(new['X']):,} rows, all in the train split")
    print(f"wrote {new_dir}  (base {args.base_data_run} unchanged)")
    print(f"next:  python run_2_t2s.py --data-run {args.new_data_run} --run-name <new t2s run> "
          f"--model tdlambdaboot_all")


if __name__ == "__main__":
    main()