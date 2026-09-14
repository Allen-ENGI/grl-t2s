"""
Stage 2: Time2Success data collection.

Ported from 1_collect_v7_stages.ipynb. Pure functions here (control-onset
detection, label building, noise-escalation search) have no sim/torch
dependency and are unit-tested in tests/test_data_collection.py. Only
`collect_dataset` / `select_failure_inducing_config` touch the sim.

CHANGE: forced failure-trajectory collection.
Relying on noise=0.4 applied to an already-competent checkpoint does NOT
reliably produce failures — a policy that's mostly solved the task often
still succeeds even with noise added, just less cleanly. If your dataset
ends up with ~0 failed episodes (check dataset_summary.json), this is why.

Instead, `select_failure_inducing_config` explicitly:
  1. always starts from the EARLIEST available checkpoint (weakest policy),
  2. empirically PROBES actual success rate at increasing noise levels,
  3. picks the first noise level that measurably fails often enough,
  4. raises loudly (rather than silently succeeding) if even the earliest
     checkpoint at maximum tested noise still succeeds too often — that's
     a signal to save an earlier checkpoint during expert_train.py, not
     something to paper over here.
`collect_dataset(..., force_failure_source=True)` (default) uses this to
add a batch of episodes it has actually verified fail some meaningful
fraction of the time, instead of just hoping.
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
    """
    Rolls out one episode, returning (obs_seq, flags, first_success).

    INDEXING CONTRACT (this was an off-by-one bug; see below):
      obs_seq has T+1 entries for T actions — states s_0 .. s_T.
      flags has T entries; flags[t] is the success flag returned by the step
      that took s_t -> s_{t+1}, so it describes state s_{t+1}, NOT s_t.
      first_success is therefore t+1, the index of the first state that IS
      successful — not t, the last state that isn't.

    The previous version recorded `first_success = t` and never stored the
    final post-step observation. That made build_labels assign
    steps_remaining=0 to the last PRE-success frame, so the most informative
    transition in every successful trajectory (1 -> 0) was flattened into
    0 -> 0, giving TD targets no signal exactly at the terminal boundary.
    """
    obs, _ = env.reset(seed=seed)
    obs_list, flags = [], []
    first_success = None
    for t in range(MAX_STEPS):
        obs_list.append(np.asarray(obs, dtype=np.float32).copy())
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
            first_success = t + 1      # the state AFTER this step is the successful one
        if terminated or truncated:
            break
    # store the final resulting state so index first_success always exists
    obs_list.append(np.asarray(obs, dtype=np.float32).copy())
    return np.array(obs_list, dtype=np.float32), np.array(flags), first_success


def _discover_checkpoints(expert_policy_dir, ckpt_prefix=TASK_SLUG):
    """
    Splits checkpoints under expert_policy_dir into (numbered_sorted_ascending, final_paths).

    BUG THIS FIXES: the old sort key concatenated every digit in the whole
    filename, including digits embedded in ckpt_prefix itself (TASK_SLUG=
    "peg_insert_side_v3" contains a "3"). That meant "..._v3_final.zip"
    (digits: just "3" -> key=3) could sort as EARLIER than
    "..._v3_100000.zip" (digits: "3100000" -> key=3100000) — silently
    selecting the final/strongest checkpoint as if it were the earliest/
    weakest one. Fixed by parsing only the filename SUFFIX after
    "{ckpt_prefix}_", and treating "final" as its own category with no
    numeric key at all rather than something to be digit-sorted.
    """
    all_ckpts = glob.glob(os.path.join(expert_policy_dir, f"{ckpt_prefix}_*.zip"))
    prefix_len = len(ckpt_prefix) + 1  # +1 for the underscore separator
    numbered, final = [], []
    for p in all_ckpts:
        suffix = os.path.basename(p)[prefix_len:].rsplit(".", 1)[0]  # "100000" or "final"
        if suffix == "final":
            final.append(p)
        elif suffix.isdigit():
            numbered.append((int(suffix), p))
        else:
            # Don't mis-sort an unrecognized name, but don't drop it silently
            # either — a typo'd checkpoint vanishing without a word has cost
            # real debugging time in this project.
            import warnings
            warnings.warn(
                f"ignoring checkpoint with unrecognized suffix {suffix!r}: "
                f"{os.path.basename(p)} (expected {{prefix}}_<steps>.zip or {{prefix}}_final.zip)",
                RuntimeWarning, stacklevel=2)
    numbered.sort(key=lambda t: t[0])
    return [p for _, p in numbered], final


def probe_success_rate(policy, env, noise_std, n_probe=5, seed_start=900):
    """
    Empirically measures how often `policy` succeeds at a given noise level,
    on seeds distinct from the main collection loop (seed_start=900+) so
    probing never overlaps with actual collected episodes.
    """
    successes = 0
    for i in range(n_probe):
        _, _, first_success = collect_episode(policy, env, deterministic=True,
                                               seed=seed_start + i, noise_std=noise_std)
        successes += first_success is not None
    return successes / n_probe


def find_noise_for_target_success_rate(measure_success_rate, noise_candidates, max_success_rate):
    """
    Pure escalation logic, sim-free and unit-testable: given a callable
    measure_success_rate(noise) -> rate in [0, 1], returns the first
    (noise, rate) pair from noise_candidates (tried in order, so smallest
    first) whose rate is at or below max_success_rate. Raises RuntimeError
    if none qualify — this is meant to raise, not silently pick the last
    candidate, because a policy that never fails is a real problem for
    downstream failure-trajectory coverage, not something to paper over.
    """
    last_rate = None
    for noise in noise_candidates:
        rate = measure_success_rate(noise)
        last_rate = rate
        if rate <= max_success_rate:
            return noise, rate
    raise RuntimeError(
        f"no noise level in {noise_candidates} brought success_rate <= {max_success_rate}; "
        f"best achieved was {last_rate:.0%} at noise={noise_candidates[-1]}. "
        "The checkpoint being probed is too strong — save an earlier checkpoint "
        "during expert_train.py (lower ckpt_freq) and retry."
    )


def select_failure_inducing_config(expert_policy_dir, scene, ckpt_prefix=TASK_SLUG,
                                    noise_candidates=(0.4, 0.6, 0.8, 1.0),
                                    n_probe=5, max_success_rate=0.3,
                                    failure_checkpoint=None):
    """
    Probes real success rate at increasing noise until it measurably fails
    often enough. Returns (checkpoint_path, chosen_noise_std, measured_success_rate).

    failure_checkpoint: explicit path (or bare filename inside expert_policy_dir)
        of the policy to use as the failure source. STRONGLY PREFERRED over the
        auto-discovery fallback — auto-discovery infers "weakest" from filename
        sort order, which cannot know whether your earliest saved checkpoint is
        actually weak (e.g. if your first save is already at 400K steps, it is
        not). Pass this explicitly whenever you know which policy you want.
        If None, falls back to the earliest numbered checkpoint found on disk.

    Requires stable_baselines3 (imported lazily) and a live `scene` env
    (e.g. from env_utils.make_fixed_scene_env) to actually roll out probes.
    """
    from stable_baselines3 import SAC

    if failure_checkpoint is not None:
        chosen_ckpt = (failure_checkpoint if os.path.isabs(failure_checkpoint)
                        or os.path.exists(failure_checkpoint)
                        else os.path.join(expert_policy_dir, failure_checkpoint))
        if not os.path.exists(chosen_ckpt):
            raise FileNotFoundError(
                f"failure_checkpoint {failure_checkpoint!r} not found (resolved to {chosen_ckpt}). "
                f"Available: {[os.path.basename(p) for p in glob.glob(os.path.join(expert_policy_dir, '*.zip'))]}"
            )
    else:
        numbered, _final = _discover_checkpoints(expert_policy_dir, ckpt_prefix)
        if not numbered:
            raise FileNotFoundError(f"no numbered expert checkpoints found under {expert_policy_dir}")
        chosen_ckpt = numbered[0]
        print(f"  (no failure_checkpoint given; auto-selected earliest of "
              f"{[os.path.basename(p) for p in numbered]})")

    policy = SAC.load(chosen_ckpt)

    noise, rate = find_noise_for_target_success_rate(
        measure_success_rate=lambda n: probe_success_rate(policy, scene, n, n_probe=n_probe),
        noise_candidates=noise_candidates, max_success_rate=max_success_rate,
    )
    return chosen_ckpt, noise, rate


def default_collection_plan(expert_policy_dir=EXPERT_POLICY_DIR, ckpt_prefix=TASK_SLUG,
                             include_noise_variants=False, include_random_policy=False,
                             checkpoints=None, n_episodes_per_checkpoint=40):
    """
    (checkpoint_path, noise_std, n_episodes) plan.

    checkpoints: explicit list of filenames (or paths) to collect from, e.g.
        ["peg_insert_side_v3_100000.zip", "peg_insert_side_v3_final.zip"].
        Recommended — pass exactly what you want rather than relying on the
        auto-selection below, which picks earliest/middle/final by filename
        order and cannot know which of your checkpoints are actually
        interesting. If None, auto-selects earliest + middle + final.

    NOTE: this plan alone is not a reliable source of FAILED episodes even
    with include_noise_variants=True — a competent checkpoint plus noise
    often still succeeds. Use collect_dataset(..., force_failure_source=True,
    failure_checkpoint=...) for verified failure coverage.

    include_noise_variants=False (current default): skips the noise=0.15/0.4
    variants for each checkpoint and only collects clean (noise=0.0) rollouts.

    include_random_policy=False (current default): skips the 20 pure-random
    (env.action_space.sample()) episodes — excluded so the state-space
    training distribution matches what a future video/real-robot dataset can
    realistically contain (see chat). Set True to restore them.
    """
    if checkpoints is not None:
        selected = []
        for ck in checkpoints:
            path = ck if os.path.isabs(ck) or os.path.exists(ck) else os.path.join(expert_policy_dir, ck)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"checkpoint {ck!r} not found (resolved to {path}). Available: "
                    f"{[os.path.basename(p) for p in glob.glob(os.path.join(expert_policy_dir, '*.zip'))]}"
                )
            selected.append(path)
    else:
        numbered, final = _discover_checkpoints(expert_policy_dir, ckpt_prefix)
        if not numbered or not final:
            raise FileNotFoundError(
                f"no expert policy checkpoints found under {expert_policy_dir}. "
                f"Expected files named {ckpt_prefix}_<steps>.zip and {ckpt_prefix}_final.zip."
            )
        selected = list(dict.fromkeys([numbered[0], numbered[len(numbered) // 2], final[0]]))
        print(f"  (no checkpoints given; auto-selected "
              f"{[os.path.basename(p) for p in selected]})")

    plan = []
    for ck in selected:
        plan.append((ck, 0.0, n_episodes_per_checkpoint))
        if include_noise_variants:
            plan += [(ck, 0.15, 12), (ck, 0.4, 12)]
    if include_random_policy:
        plan += [(None, 0.0, 20)]  # random policy — see chat for why this is off by default
    return plan


def collect_dataset(run_dir, plan=None, censor_label=CENSOR_LABEL, include_stage_label=False,
                     force_failure_source=True, n_failure_episodes=30,
                     failure_noise_candidates=(0.4, 0.6, 0.8, 1.0),
                     failure_n_probe=5, failure_max_success_rate=0.3,
                     failure_checkpoint=None,
                     expert_policy_dir=EXPERT_POLICY_DIR, ckpt_prefix=TASK_SLUG):
    """
    Runs `plan` (see default_collection_plan) and writes dataset.npz +
    dataset_summary.json into run_dir (obtained from results.new_run_dir).
    Requires stable_baselines3's SAC — imported lazily so this module can be
    unit-tested (coupling_signals / build_labels) without SB3 installed.

    force_failure_source=True (current default): before running `plan`, probes
    a policy at increasing noise (see select_failure_inducing_config) and adds
    n_failure_episodes genuinely-verified-to-often-fail episodes.

    failure_checkpoint: which policy to use as that failure source. Pass a
        filename (e.g. "peg_insert_side_v3_100000.zip") or full path to choose
        explicitly — recommended, since auto-discovery can only guess "weakest"
        from filename ordering and has no way to know whether your earliest
        saved checkpoint is genuinely weak. None = auto-select earliest found.
    """
    from stable_baselines3 import SAC
    from env_utils import make_fixed_scene_env
    import json

    plan = plan or default_collection_plan(expert_policy_dir=expert_policy_dir, ckpt_prefix=ckpt_prefix)
    episodes = []
    scene = make_fixed_scene_env()

    fail_ckpt = fail_noise = measured_rate = None
    if force_failure_source:
        fail_ckpt, fail_noise, measured_rate = select_failure_inducing_config(
            expert_policy_dir, scene, ckpt_prefix=ckpt_prefix,
            noise_candidates=failure_noise_candidates, n_probe=failure_n_probe,
            max_success_rate=failure_max_success_rate,
            failure_checkpoint=failure_checkpoint,
        )
        print(f"forced failure source: {os.path.basename(fail_ckpt)} @ noise={fail_noise} "
              f"(measured success rate {measured_rate:.0%} over {failure_n_probe} probes)")
        plan = [(fail_ckpt, fail_noise, n_failure_episodes)] + list(plan)

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
        forced_failure_source=(dict(checkpoint=os.path.basename(fail_ckpt), noise=fail_noise,
                                     measured_success_rate=measured_rate)
                                if force_failure_source else None),
    )
    with open(os.path.join(run_dir, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary