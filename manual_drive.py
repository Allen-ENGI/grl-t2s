#!/usr/bin/env python3
"""
Drive the arm manually, then hand over to a successful policy, and film the
T2S prediction across both phases.

    python manual_drive.py --drive "back:25,up:15"
    python manual_drive.py --drive retreat --models tdlambdaboot_succ td0_all
    python manual_drive.py --drive "back:40" --policy-run sweep_v6_td0boot_all_seed0

WHY THIS IS THE MOST INFORMATIVE VIDEO IN THE PIPELINE
------------------------------------------------------
Every other rollout in this project starts at the same pinned reset state and
is driven by an expert or a near-expert. So every state a T2S model is ever
scored on sits on or near the expert manifold — which is precisely the gap
dataset_audit.check_exploration_coverage flags: if the dataset has no states
with the hand far from the object, predictions out there are extrapolation,
and extrapolation is what a downstream RL policy hits from step one.

Driving the arm somewhere the expert never goes, then releasing a policy that
CAN solve the task from there, produces the one trajectory that separates the
models honestly:

  - during the MANUAL phase the prediction SHOULD RISE. The arm is being taken
    further from success, so steps-to-success genuinely increases. A model
    whose prediction stays flat, or falls, while the arm retreats has learnt
    "this looks like a demonstration frame" rather than "this is how far I am".
  - at HANDOVER the prediction should sit at roughly the number of steps the
    policy then actually takes. That is the only place in this pipeline where
    a prediction can be checked against a real, unrehearsed outcome.
  - during the POLICY phase it should count down at about 1 per step, which is
    what step_slope_error measures in aggregate — here you can watch it.

The video labels each phase and marks the handover frame, and the summary
prints predicted-at-handover against actual-steps-taken per model.

DRIVE SCRIPTS
-------------
Comma-separated "move:steps" pairs, applied in order, e.g. "back:25,up:15".
Moves are named directions in MetaWorld's 4-D action space
[dx, dy, dz, gripper], all in [-1, 1]:

    back / forward / left / right / up / down / open / close / still

Presets: --drive retreat   = back:30,up:10   (pull away from the table)
         --drive wander    = left:15,back:15,right:15,up:10
         --drive lift      = up:25
         --drive nudge     = back:10

Nothing here is interactive: a scripted drive is reproducible, works headless
over SSH, and can be replayed identically across models. If you want to feel
your way to an interesting pose, run with --probe to print the resulting
hand/peg/goal distances without rendering, then commit to a script.
"""
import argparse
import json
import os
import sys

import numpy as np

# MetaWorld action: [dx, dy, dz, gripper], each in [-1, 1]
MOVES = {
    "forward": (0.0,  1.0, 0.0, 0.0),
    "back":    (0.0, -1.0, 0.0, 0.0),
    "right":   (1.0,  0.0, 0.0, 0.0),
    "left":    (-1.0, 0.0, 0.0, 0.0),
    "up":      (0.0,  0.0, 1.0, 0.0),
    "down":    (0.0,  0.0, -1.0, 0.0),
    "open":    (0.0,  0.0, 0.0, -1.0),
    "close":   (0.0,  0.0, 0.0, 1.0),
    "still":   (0.0,  0.0, 0.0, 0.0),
}
PRESETS = {
    "retreat": "back:30,up:10",
    "wander":  "left:15,back:15,right:15,up:10",
    "lift":    "up:25",
    "nudge":   "back:10",
}
STAGE_MANIFEST = "stage_manifest.json"


def parse_drive(spec):
    """'back:25,up:15' -> [(name, action_vector, n_steps), ...]"""
    spec = PRESETS.get(spec, spec)
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, n = part.partition(":")
        name = name.strip().lower()
        if name not in MOVES:
            raise SystemExit(f"ERROR: unknown move {name!r}. Known: {sorted(MOVES)}\n"
                             f"Presets: {sorted(PRESETS)}")
        try:
            steps = int(n) if n else 10
        except ValueError:
            raise SystemExit(f"ERROR: bad step count in {part!r}; use e.g. back:25")
        out.append((name, np.array(MOVES[name], dtype=np.float32), steps))
    if not out:
        raise SystemExit("ERROR: empty drive script")
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Manual drive, then policy takeover, filmed with T2S predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--drive", default="retreat",
                   help="'move:steps,...' or a preset: " + ", ".join(sorted(PRESETS)))
    p.add_argument("--t2s-run", default="v3", help="t2s_model run name")
    p.add_argument("--models", nargs="+", default=None,
                   help="combo names to score (default: all in the run)")
    p.add_argument("--takeover", default="success", choices=["success", "failure"],
                   help="which held-out expert takes over after the drive. "
                        "'success' answers 'can a competent policy recover, and "
                        "did the model predict how long it would take'. 'failure' "
                        "answers 'what does the model predict while a policy "
                        "flounders somewhere it has never been' — the regime where "
                        "a shaped reward actually operates, and the one no "
                        "success-rollout metric can see.")
    p.add_argument("--policy-run", default=None,
                   help="take over with a TRAINED policy from this policy run "
                        "instead of an expert; overrides --takeover")
    p.add_argument("--checkpoint", default="policy_final.zip",
                   help="which checkpoint inside --policy-run to load. Bare "
                        "filename or absolute path. Checkpoints are written every "
                        "ckpt_freq steps as policy_<steps>.zip, so e.g. "
                        "policy_200000.zip films the policy mid-training — useful "
                        "when the final one regressed, which progress.py's own "
                        "docstring notes has happened in this project.")
    p.add_argument("--reward-mode", default="difference_timed",
                   choices=["absolute", "difference", "difference_timed"])
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--out", default=None, help="output dir (default: the eval run's videos/)")
    p.add_argument("--probe", action="store_true",
                   help="print distances after the drive and exit; no rendering")
    return p.parse_args(argv)


def drive_then_policy(policy, drive, seed=0, max_steps=500, stop_after_success=20):
    """
    Run the scripted drive, then hand control to `policy`. Returns a rollout
    dict plus `handover` (the frame index where the policy took over) and
    `phases` (a per-frame label).

    success_step follows the package contract: index of the first state that IS
    successful. It is reported RELATIVE TO THE START, so `success_step -
    handover` is what the policy actually took from the pose it inherited.
    """
    import torch

    from config import SUCCESS_KEY
    from env_utils import make_fixed_scene_env
    from t2s_video import render_frame

    env = make_fixed_scene_env(render_mode="rgb_array")
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)

    obs_list, frames, phases = [], [], []
    success_step = None
    t = 0

    # ---- manual phase ----------------------------------------------------
    for name, action, steps in drive:
        for _ in range(steps):
            if t >= max_steps:
                break
            obs_list.append(np.asarray(obs, dtype=np.float32).copy())
            frames.append(render_frame(env))
            phases.append(f"MANUAL {name}")
            obs, _r, term, trunc, info = env.step(
                np.clip(action, env.action_space.low, env.action_space.high))
            if info.get(SUCCESS_KEY, 0) and success_step is None:
                success_step = t + 1
            t += 1
            if term or trunc:
                break
    handover = t

    # ---- policy phase ----------------------------------------------------
    while t < max_steps:
        obs_list.append(np.asarray(obs, dtype=np.float32).copy())
        frames.append(render_frame(env))
        phases.append("POLICY")
        action, _ = policy.predict(obs, deterministic=False)
        obs, _r, term, trunc, info = env.step(action)
        if info.get(SUCCESS_KEY, 0) and success_step is None:
            success_step = t + 1
        t += 1
        if success_step is not None and t >= success_step + stop_after_success:
            break
        if term or trunc:
            break

    obs_list.append(np.asarray(obs, dtype=np.float32).copy())
    frames.append(render_frame(env))
    phases.append("POLICY")
    env.close()

    from config import steps_remaining_curve
    obs_arr = np.array(obs_list, dtype=np.float32)
    return dict(obs=obs_arr, frames=frames, seed=seed, success_step=success_step,
                handover=handover, phases=phases,
                truth=steps_remaining_curve(len(obs_arr), success_step)
                if success_step is not None else None)


def render_phased(roll, predict_fn, save_path, combo, reward_mode, gamma, fps):
    """
    Like t2s_video.render_t2s_video, but the per-frame label carries the phase
    and the handover marker, so the video says which part of the trajectory a
    given prediction belongs to.
    """
    import t2s_video
    from reward_preview import compute_step_rewards

    obs = roll["obs"]
    preds = np.array([predict_fn(o) for o in obs], dtype=np.float64)
    rewards = compute_step_rewards(preds, reward_mode, gamma=gamma)
    ss, ho = roll["success_step"], roll["handover"]
    if ss is not None and len(rewards):
        rewards = rewards.copy()
        rewards[ss:] = 0.0                       # match the wrapper's success latch

    n = min(len(roll["frames"]), len(preds), len(obs))
    out = [t2s_video.compose_frame(
               roll["frames"][i], i, preds[:n], truth=roll["truth"], obs=obs,
               rewards=rewards, success_step=ss, reward_mode=reward_mode,
               label=f"{combo} | {roll['phases'][i]}" + ("  <HANDOVER" if i == ho else ""))
           for i in range(n)]
    if save_path:
        t2s_video.save_video(out, save_path, fps=fps)

    return dict(
        file=os.path.basename(save_path) if save_path else None,
        pred_at_start=float(preds[0]),
        pred_at_handover=float(preds[min(ho, n - 1)]),
        pred_peak_during_drive=float(preds[:max(ho, 1)].max()),
        # did the prediction RISE while the arm was driven away? It should.
        rose_during_drive=bool(preds[min(ho, n - 1)] > preds[0]),
        actual_steps_after_handover=(int(ss - ho) if ss is not None else None),
        # lowest prediction once the takeover policy is driving: on a rollout
        # that never succeeds this is the model's most confident false claim
        pred_min_after_handover=float(preds[min(ho, n - 1):n].min()),
        succeeded=ss is not None,
        total_return=float(np.sum(rewards)),
    )


def main(argv=None):
    args = parse_args(argv)

    import config
    import results
    import t2s_predict
    import t2s_video

    from stable_baselines3 import SAC

    gamma = config.RL_GAMMA if args.gamma is None else args.gamma
    fps = t2s_video.FPS if args.fps is None else args.fps
    drive = parse_drive(args.drive)

    total_manual = sum(s for _n, _a, s in drive)
    print(f"=== manual drive: {', '.join(f'{n} x{s}' for n, _a, s in drive)} "
          f"({total_manual} steps) ===")

    # ---- the taking-over policy -----------------------------------------
    t2s_dir = results.get_run_dir("t2s_model", args.t2s_run)
    with open(os.path.join(t2s_dir, STAGE_MANIFEST)) as f:
        stage2 = json.load(f)
    eval_dir = results.get_run_dir("t2s_eval", args.t2s_run)
    with open(os.path.join(eval_dir, STAGE_MANIFEST)) as f:
        stage3 = json.load(f)

    if args.policy_run:
        import visualize
        pol_dir = results.get_run_dir("policy", args.policy_run)
        policy, path = visualize.load_policy(args.checkpoint, pol_dir)
        print(f"    takeover: trained policy {os.path.basename(path)}")
        takeover_name = os.path.basename(path)
    else:
        # the manifest key is "failure_ckpt", not "failed_ckpt"
        key = "success_ckpt" if args.takeover == "success" else "failure_ckpt"
        takeover_name = stage3[key]
        path = os.path.join(config.EXPERT_POLICY_DIR, takeover_name)
        policy = SAC.load(path)
        print(f"    takeover: held-out {args.takeover} expert {takeover_name}")

    # ---- roll ONCE; every model scores the same frames -------------------
    roll = drive_then_policy(policy, drive, seed=args.seed, max_steps=args.max_steps)
    from progress import hand_peg_distance, peg_goal_distance
    ho = roll["handover"]
    o0, oh = roll["obs"][0], roll["obs"][min(ho, len(roll["obs"]) - 1)]
    print(f"\nafter the drive ({ho} steps):")
    print(f"    hand->peg  {hand_peg_distance(o0):.4f} -> {hand_peg_distance(oh):.4f}")
    print(f"    peg->goal  {peg_goal_distance(o0):.4f} -> {peg_goal_distance(oh):.4f}")
    if roll["success_step"] is None:
        print(f"    no success in {len(roll['obs'])} steps"
              + ("  (expected — the failure expert rarely succeeds; the point is "
                 "what the MODEL predicts while it flounders)"
                 if args.takeover == "failure" and not args.policy_run
                 else "  — the policy did NOT recover"))
    else:
        print(f"    policy recovered: success at step {roll['success_step']} "
              f"({roll['success_step'] - ho} steps after handover)")

    if args.probe:
        print("\n(--probe: no video rendered)")
        return 0

    combos = args.models or sorted(stage2["combos"])
    missing = [c for c in combos if c not in stage2["combos"]]
    if missing:
        raise SystemExit(f"ERROR: {missing} not in {args.t2s_run}; "
                         f"available: {sorted(stage2['combos'])}")

    out_dir = args.out or os.path.join(eval_dir, "videos")
    os.makedirs(out_dir, exist_ok=True)
    tag = args.drive.replace(":", "").replace(",", "_")

    print(f"\n=== filming {len(combos)} model(s) -> {out_dir} ===")
    rows = {}
    predictors = {}
    for combo in combos:
        fn = t2s_predict.load_t2s_predictor(t2s_dir, *combo.rsplit("_", 1),
                                            seed=stage2["combos"][combo]["best_seed"])
        predictors[combo] = fn
        rows[combo] = render_phased(
            roll, fn, os.path.join(out_dir, f"drive_{tag}_{combo}.mp4"),
            combo, args.reward_mode, gamma, fps)

    if len(predictors) > 1:
        t2s_video.render_model_comparison_video(
            roll, predictors, save_path=os.path.join(out_dir, f"drive_{tag}_ALL.mp4"), fps=fps)
        print(f"  drive_{tag}_ALL.mp4  every model on the same frames")

    # ---- the table this script exists for --------------------------------
    actual = rows[combos[0]]["actual_steps_after_handover"]
    if actual is None:
        # no ground truth exists, so score what CAN be scored without it: did
        # the prediction rise as the arm was driven away, and how low did it
        # fall while the takeover policy failed to solve anything. A low
        # minimum here is the model's most confident false claim.
        print(f"\n{'combo':<24}{'pred@start':>11}{'pred@handover':>15}{'rose?':>7}"
              f"{'min pred':>10}{'verdict':>26}")
        print("-" * 93)
        for c in sorted(combos,
                        key=lambda c: -(rows[c].get("pred_min_after_handover") or 0)):
            r = rows[c]
            lo = r.get("pred_min_after_handover")
            v = ("falsely optimistic" if lo is not None and lo < 30 else "stays pessimistic")
            print(f"{c:<24}{r['pred_at_start']:>11.1f}{r['pred_at_handover']:>15.1f}"
                  f"{('yes' if r['rose_during_drive'] else 'NO'):>7}"
                  f"{(f'{lo:.1f}' if lo is not None else '--'):>10}{v:>26}")
        print("\nno success occurred, so there is no 'actual steps left' to compare")
        print("against. LOW min pred on a trajectory that never succeeds is the")
        print("failure this whole project is about.")
        with open(os.path.join(out_dir, f"drive_{tag}_index.json"), "w") as f:
            json.dump(dict(drive=args.drive, handover=ho, seed=args.seed,
                           takeover=args.takeover, reward_mode=args.reward_mode,
                           gamma=gamma, success_step=None, models=rows), f, indent=2)
        return 0
    print(f"\n{'combo':<24}{'pred@start':>11}{'pred@handover':>15}{'rose?':>7}"
          f"{'actual left':>13}{'error':>9}")
    print("-" * 79)
    for c in sorted(combos, key=lambda c: abs(
            (rows[c]["pred_at_handover"] - actual) if actual is not None else 0)):
        r = rows[c]
        err = (f"{r['pred_at_handover'] - actual:+.1f}" if actual is not None else "--")
        print(f"{c:<24}{r['pred_at_start']:>11.1f}{r['pred_at_handover']:>15.1f}"
              f"{('yes' if r['rose_during_drive'] else 'NO'):>7}"
              f"{(str(actual) if actual is not None else '--'):>13}{err:>9}")

    print("\npred@handover vs actual left is the only check in this pipeline where a")
    print("prediction meets an unrehearsed outcome — the drive put the arm somewhere")
    print("no training trajectory went, so nothing here was memorised.")
    print("'rose? NO' means the prediction did not increase while the arm was driven")
    print("AWAY from success: that model is reading familiarity, not distance.")

    with open(os.path.join(out_dir, f"drive_{tag}_index.json"), "w") as f:
        json.dump(dict(drive=args.drive, handover=ho, seed=args.seed,
                       reward_mode=args.reward_mode, gamma=gamma,
                       success_step=roll["success_step"],
                       actual_steps_after_handover=actual, models=rows), f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())