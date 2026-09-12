"""
Stage 2: Time2Success data collection.

Ported from 1_collect_v7_stages.ipynb. Pure functions here (control-onset
detection, label building) have no sim/torch dependency and are unit-tested
in tests/test_data_collection.py. Only `collect_dataset` touches the env.
"""
import glob
import os

import numpy as np

from config import (HAND_POS_IDX, PEG_POS_IDX, SUCCESS_KEY, MAX_STEPS,
                     CENSOR_LABEL, EXPERT_POLICY_DIR, TASK_SLUG)
# env_utils (and therefore gymnasium/metaworld) is only needed by the
# sim-touching functions below; imported lazily inside them so the pure
# label/detection logic stays unit-testable without those deps installed.

# ---- control-onset detection (pure, sim-free) --------------------------
VEL_WINDOW = 5
VEL_COS_THRESH = 0.7
VEL_MIN_MOTION = 3e-4
POSE_STD_THRESH = 2e-3
CONFIRM_FRAMES = 2
POSE_WINDOW = 5


def coupling_signals(obs_seq):
    """obs_seq: (T, obs_dim). Returns per-frame velocity-cosine and relative-pose std."""
    hand = obs_seq[:, HAND_POS_IDX]
    peg = obs_seq[:, PEG_POS_IDX]
    T = len(obs_seq)

    dh, dp = np.diff(hand, axis=0), np.diff(peg, axis=0)
    cos = np.zeros(T, dtype=np.float32)
    for t in range(T - 1 - VEL_WINDOW):
        h, p = dh[t:t + VEL_WINDOW].ravel(), dp[t:t + VEL_WINDOW].ravel()
        nh, npg = np.linalg.norm(h), np.linalg.norm(p)
        cos[t + VEL_WINDOW] = 0.0 if (nh < VEL_MIN_MOTION or npg < VEL_MIN_MOTION) \
            else float(h @ p / (nh * npg))

    rel = peg - hand
    pose_std = np.full(T, np.inf, dtype=np.float32)
    for t in range(POSE_WINDOW, T):
        pose_std[t] = rel[t - POSE_WINDOW:t].std(axis=0).mean()

    return cos, pose_std


def detect_control_onset(obs_seq):
    """First frame the object comes under the effector's control, or None."""
    cos, pose_std = coupling_signals(obs_seq)
    candidate = (cos > VEL_COS_THRESH) & (pose_std < POSE_STD_THRESH)
    run = 0
    for t in range(len(candidate)):
        run = run + 1 if candidate[t] else 0
        if run >= CONFIRM_FRAMES:
            return t - CONFIRM_FRAMES + 1
    return None


def build_labels(obs_seq, first_success, censor_label=CENSOR_LABEL, include_stage_label=False):
    """
    steps_remaining: first-success anchored, flat 0 from the anchor onward,
                      or a censored ceiling if the episode never succeeds.
    stage: 0 pre-control, 1 control, 2 complete. include_stage_label=False
           (current default) skips control-onset detection entirely and
           returns an all-zeros stage array + onset=None — steps_remaining
           is unaffected either way. Set True to restore the v7 stage
           labeling once you want the auxiliary stage head back.
    """
    n = len(obs_seq)
    if first_success is not None:
        steps = np.array([max(0, first_success - t) for t in range(n)], dtype=np.float32)
    else:
        steps = np.full(n, censor_label, dtype=np.float32)

    if not include_stage_label:
        return steps, np.zeros(n, dtype=np.int64), None

    onset = detect_control_onset(obs_seq)
    stage = np.zeros(n, dtype=np.int64)
    if onset is not None:
        stage[onset:] = 1
    if first_success is not None:
        stage[first_success:] = 2
    return steps, stage, onset


# ---- rollout collection (touches the sim) -------------------------------

def collect_episode(policy, env, deterministic=True, seed=None, noise_std=0.0):
    obs, _ = env.reset(seed=seed)
    obs_list, flags = [], []
    first_success = None
    for t in range(MAX_STEPS):
        obs_list.append(obs.copy())
        if policy is None:
            action = env.action_space.sample()
        else:
            action, _ = policy.predict(obs, deterministic=deterministic)
            if noise_std > 0:
                action = np.clip(action + np.random.normal(0, noise_std, action.shape),
                                  env.action_space.low, env.action_space.high)
        obs, _, terminated, truncated, info = env.step(action)
        flags.append(bool(info.get(SUCCESS_KEY, 0)))
        if flags[-1] and first_success is None:
            first_success = t
        if terminated or truncated:
            break
    return np.array(obs_list, dtype=np.float32), np.array(flags), first_success


def default_collection_plan(expert_policy_dir=EXPERT_POLICY_DIR, ckpt_prefix=TASK_SLUG,
                             include_noise_variants=False, include_random_policy=False):
    """
    (checkpoint_path, noise_std, n_episodes) plan spanning expert / mid / early
    checkpoints plus a random policy, matching notebook 1's PLAN.
    `ckpt_prefix` must match the ckpt_prefix used in expert_train.train_expert_policy
    (defaults to config.TASK_SLUG, derived from TASK_NAME) so this keeps working
    if you retarget the whole pipeline at a different Meta-World task.

    include_noise_variants=False (current default): skips the noise=0.15/0.4
    "off-path" variants for each checkpoint and only collects clean (noise=0.0)
    rollouts. Set True to restore the original v7 plan's off-path coverage —
    that coverage was the one change that meaningfully fixed the failing-rollout
    prediction floor in the original notebook, so turn it back on once the
    simpler pipeline is verified end-to-end.

    include_random_policy=False (current default): skips the 20 pure-random
    (env.action_space.sample()) episodes. Excluded so the state-space training
    distribution matches what a future video/real-robot dataset can realistically
    contain — random-action rollouts are cheap in sim but not practical to
    collect on real hardware, so keeping them here would bias any later
    state-space-vs-video comparison. Set True to restore them if you decide
    that off-path robustness matters more than train-set comparability.
    """
    ckpts = sorted(
        glob.glob(os.path.join(expert_policy_dir, f"{ckpt_prefix}_*.zip")),
        key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0),
    )
    numbered = [p for p in ckpts if any(c.isdigit() for c in os.path.basename(p))]
    final = [p for p in ckpts if "final" in os.path.basename(p)]
    if not numbered or not final:
        raise FileNotFoundError(
            f"no expert policy checkpoints found under {expert_policy_dir}. "
            "Expert pretraining is assumed already done — populate this "
            "directory with sac_peg_insert_<steps>.zip and sac_peg_insert_final.zip."
        )
    selected = list(dict.fromkeys([numbered[0], numbered[len(numbered) // 2], final[0]]))

    plan = []
    for ck in selected:
        plan.append((ck, 0.0, 40))
        if include_noise_variants:
            plan += [(ck, 0.15, 12), (ck, 0.4, 12)]
    if include_random_policy:
        plan += [(None, 0.0, 20)]  # random policy — see chat for why this is off by default
    return plan


def collect_dataset(run_dir, plan=None, censor_label=CENSOR_LABEL, include_stage_label=False):
    """
    Runs `plan` (see default_collection_plan) and writes dataset.npz +
    dataset_summary.json into run_dir (obtained from results.new_run_dir).
    Requires stable_baselines3's SAC — imported lazily so this module can be
    unit-tested (coupling_signals / build_labels) without SB3 installed.

    include_stage_label=False (current default): forwarded to build_labels;
    dataset.npz still contains a "stage" field for schema stability, but it
    will be all zeros. Flip to True once you want real stage labels again.
    """
    from stable_baselines3 import SAC
    from env_utils import make_fixed_scene_env
    import json

    plan = plan or default_collection_plan()
    episodes = []
    scene = make_fixed_scene_env()
    seed = 0
    for ck, ns, n in plan:
        pol = SAC.load(ck) if ck else None
        tag = f"{os.path.basename(ck) if ck else 'random'}_n{ns}"
        for _ in range(n):
            obs_seq, flags, fs = collect_episode(pol, scene, deterministic=True, seed=seed, noise_std=ns)
            seed += 1
            steps, stage, onset = build_labels(obs_seq, fs, censor_label, include_stage_label=include_stage_label)
            episodes.append(dict(obs=obs_seq, steps=steps, stage=stage, onset=onset,
                                  success=fs, source=tag))
    scene.close()

    X = np.concatenate([e["obs"] for e in episodes])
    y = np.concatenate([e["steps"] for e in episodes])
    stage = np.concatenate([e["stage"] for e in episodes])
    ep_ids = np.concatenate([np.full(len(e["obs"]), i, dtype=np.int32) for i, e in enumerate(episodes)])
    frames = np.concatenate([np.arange(len(e["obs"]), dtype=np.int32) for e in episodes])

    np.savez(os.path.join(run_dir, "dataset.npz"),
              X=X, y_steps=y, stage=stage, episode_ids=ep_ids, frame_idxs=frames)

    succ = sum(1 for e in episodes if e["success"] is not None)
    summary = dict(
        total_rows=int(len(X)), total_episodes=int(len(episodes)),
        obs_dim=int(X.shape[1]),
        successful_episodes=int(succ), failed_episodes=int(len(episodes) - succ),
        censor_label=float(censor_label),
        stage_fractions={int(k): float((stage == k).mean()) for k in (0, 1, 2)},
        sources=sorted({e["source"] for e in episodes}),
    )
    with open(os.path.join(run_dir, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
