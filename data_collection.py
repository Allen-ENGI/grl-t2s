"""
Stage 2: Time2Success data collection.

    from data_collection import collect
    collect(run_dir, holdout=["..._200000.zip", "..._700000.zip"])

That is the whole API. Everything else in this module is either a pure helper
(unit-tested without a sim) or a tuning constant at the top of the file.

THE FIVE REAL DECISIONS
-----------------------
  holdout    which checkpoints are RESERVED for Stage 4's held-out-policy
             evaluation, and therefore not collected from. Only you know this,
             and getting it wrong silently invalidates Stage 4's central claim.
  episodes   how many clean episodes per checkpoint.
  seeds      how many independent passes; the main diversity knob.
  val        validation share.
  run_dir    where output goes.

WHAT THE MODULE GUARANTEES
--------------------------
  - every discovered checkpoint is collected from, minus the holdout
  - the policy is always SAMPLED, never its deterministic mean (a pinned scene
    plus a deterministic policy gave 222 episodes with only 93 unique
    trajectories)
  - failure coverage is verified by probing, not hoped for
  - the train/val split is assigned HERE and stored in dataset.npz, stratified
    by (source, succeeded) and duplicate-safe, so every downstream run
    validates on identical states
"""
import glob
import os
import warnings
import numpy as np

from config import (HAND_POS_IDX, PEG_POS_IDX, SUCCESS_KEY, MAX_STEPS,
                    CENSOR_LABEL, EXPERT_POLICY_DIR, TASK_SLUG,
                    steps_remaining_curve)

# ---- tuning constants: edit here, not via arguments ---------------------
NOISE_LEVELS = (0.0, 0.15, 0.4)      # action noise per checkpoint; 0.0 is the clean pass
NOISE_EPISODE_SHARE = 0.3            # episodes at each noisy level, as a share of `episodes`

FAILURE_NOISE_LADDER = (0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0)
FAILURE_MAX_SUCCESS_RATE = 0.4       # a source qualifies below this measured rate
FAILURE_PROBES = 5                   # rollouts per rung of the ladder
FAILURE_EPISODE_SHARE = 0.4          # verified-failing episodes, as a share of `episodes`
FAILURE_SOURCE_SHARE = 0.5           # probe the weakest half of eligible checkpoints

DEFAULT_VAL_FRACTION = 0.2
SPLITS = ("train", "val")

# ---- control-onset detection (pure, sim-free) ---------------------------
VEL_WINDOW, VEL_COS_THRESH, VEL_MIN_MOTION = 5, 0.7, 3e-4
POSE_WINDOW, POSE_STD_THRESH, CONFIRM_FRAMES = 5, 2e-3, 2


def coupling_signals(obs_seq):
    """obs_seq: (T, obs_dim). Per-frame velocity-cosine and relative-pose std."""
    hand, peg = obs_seq[:, HAND_POS_IDX], obs_seq[:, PEG_POS_IDX]
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


def build_labels(obs_seq, first_success, censor_label=CENSOR_LABEL):
    """
    steps_remaining for one episode, from the single definition in
    config.steps_remaining_curve. `first_success` is the index of the first
    state that IS successful (so its label is 0), or None.
    """
    return steps_remaining_curve(len(obs_seq), first_success, censor_label=censor_label)


# ---- train/val split (pure, sim-free) -----------------------------------

def assign_splits(episodes, val_fraction=DEFAULT_VAL_FRACTION, dedupe=True):
    """
    Assign each episode "train" or "val". Pure and deterministic, so two runs
    over the same episodes agree exactly.
    """
    n = len(episodes)
    splits = ["train"] * n
    if val_fraction <= 0 or n == 0:
        return np.array(splits, dtype=object), dict(
            achieved_val_fraction=0.0, n_val=0, n_train=n, duplicate_groups=0,
            stratum_count=0)

    sig_of = {}
    if dedupe:
        for i, e in enumerate(episodes):
            obs = e.get("obs")
            sig_of[i] = hash(np.asarray(obs).tobytes()) if obs is not None else i

    groups = {}
    for i, e in enumerate(episodes):
        groups.setdefault((e.get("source", ""), e.get("success") is not None), []).append(i)

    sig_split = {}
    for key in sorted(groups, key=lambda t: (str(t[0]), t[1])):
        idxs = groups[key]
        weight, pending = {}, []
        for i in idxs:
            sig = sig_of.get(i, i)
            if sig not in weight:
                weight[sig] = 0
                if sig not in sig_split:
                    pending.append(sig)
            weight[sig] += 1
        m = len(pending)
        if m:
            target = sum(weight[s] for s in pending) * val_fraction
            min_groups = 1 if m >= 2 else 0
            order = sorted(range(m), key=lambda pos: ((pos * 997) % m, pos))
            taken_eps = taken_groups = 0
            chosen = set()
            for pos in order:
                if taken_groups >= max(m - 1, 0):
                    break                      # never send a whole stratum to val
                w = weight[pending[pos]]
                if taken_groups < min_groups or taken_eps + w <= target:
                    chosen.add(pos)
                    taken_eps += w
                    taken_groups += 1
            for pos, sig in enumerate(pending):
                sig_split[sig] = "val" if pos in chosen else "train"
        for i in idxs:
            splits[i] = sig_split[sig_of.get(i, i)]

    n_val = sum(s == "val" for s in splits)
    return np.array(splits, dtype=object), dict(
        achieved_val_fraction=float(n_val / n), n_val=int(n_val), n_train=int(n - n_val),
        duplicate_groups=int((len(sig_of) - len(set(sig_of.values()))) if dedupe else 0),
        stratum_count=int(len(groups)))


# ---- rollout (touches the sim) ------------------------------------------

def collect_episode(policy, env, seed=None, noise_std=0.0, rng=None):
    """
    One episode -> (obs_seq, flags, first_success).
    """
    rng = rng if rng is not None else np.random
    obs, _ = env.reset(seed=seed)
    obs_list, flags, first_success = [], [], None
    for t in range(MAX_STEPS):
        obs_list.append(np.asarray(obs, dtype=np.float32).copy())
        if policy is None:
            action = env.action_space.sample()
        else:
            action, _ = policy.predict(obs, deterministic=False)
            if noise_std > 0:
                action = np.clip(action + rng.normal(0, noise_std, action.shape),
                                 env.action_space.low, env.action_space.high)
        obs, _, terminated, truncated, info = env.step(action)
        flags.append(bool(info.get(SUCCESS_KEY, 0)))
        if flags[-1] and first_success is None:
            first_success = t + 1
        if terminated or truncated:
            break
    obs_list.append(np.asarray(obs, dtype=np.float32).copy())
    return np.array(obs_list, dtype=np.float32), np.array(flags), first_success


# ---- checkpoint discovery -----------------------------------------------

def discover_checkpoints(expert_dir=EXPERT_POLICY_DIR, holdout=()):
    """
    (eligible, reserved) checkpoint paths, weakest first, final last.

    """
    reserved = {os.path.basename(_resolve(h, expert_dir)) for h in (holdout or ())}
    prefix_len = len(TASK_SLUG) + 1
    numbered, final = [], []
    for p in glob.glob(os.path.join(expert_dir, f"{TASK_SLUG}_*.zip")):
        suffix = os.path.basename(p)[prefix_len:].rsplit(".", 1)[0]
        if suffix == "final":
            final.append(p)
        elif suffix.isdigit():
            numbered.append((int(suffix), p))
        else:
            warnings.warn(f"ignoring checkpoint with unrecognized suffix {suffix!r}: "
                          f"{os.path.basename(p)}", RuntimeWarning, stacklevel=2)
    numbered.sort(key=lambda t: t[0])
    allp = [p for _, p in numbered] + final
    return ([p for p in allp if os.path.basename(p) not in reserved],
            [p for p in allp if os.path.basename(p) in reserved])


def _resolve(ck, expert_dir):
    path = ck if os.path.isabs(ck) or os.path.exists(ck) else os.path.join(expert_dir, ck)
    if not os.path.exists(path):
        avail = [os.path.basename(p) for p in glob.glob(os.path.join(expert_dir, "*.zip"))]
        raise FileNotFoundError(f"checkpoint {ck!r} not found (-> {path}). Available: {avail}")
    return path


# ---- failure sourcing ---------------------------------------------------

def probe_success_rate(policy, env, noise_std, n_probe=FAILURE_PROBES, seed_start=900, rng=None):
    """
    Measured success rate at a noise level, on seeds distinct from the main
    collection loop so probing never overlaps collected episodes.
    """
    return sum(collect_episode(policy, env, seed=seed_start + i, noise_std=noise_std,
                               rng=rng)[2] is not None
               for i in range(n_probe)) / n_probe


def find_failing_noise(measure, ladder=FAILURE_NOISE_LADDER,
                       max_success_rate=FAILURE_MAX_SUCCESS_RATE):
    """
    First (noise, rate) from `ladder` at or below max_success_rate, else None.

    Returns None rather than raising: with several candidate sources, one that
    refuses to fail is a reason to try the next, not to abort the whole
    collection. `collect` raises only if EVERY source declines.
    """
    for noise in ladder:
        rate = measure(noise)
        if rate <= max_success_rate:
            return noise, rate
    return None


def _failure_sources(scene, eligible, n_episodes, rng_seed=12345, verbose=True):
    """
    Probe the weakest share of eligible checkpoints and return the ones that
    measurably fail, as (path, noise, rate, episodes) tuples.
    """
    from stable_baselines3 import SAC

    candidates = eligible[:max(1, int(len(eligible) * FAILURE_SOURCE_SHARE))]
    found = []
    for k, ck in enumerate(candidates):
        hit = find_failing_noise(
            lambda n, _p=SAC.load(ck): probe_success_rate(
                _p, scene, n, rng=np.random.default_rng(rng_seed + k)))
        if hit is None:
            if verbose:
                print(f"  {os.path.basename(ck)}: never fails often enough, skipping")
            continue
        found.append((ck, hit[0], hit[1]))
        if verbose:
            print(f"  {os.path.basename(ck)}: fails at noise={hit[0]} "
                  f"(success rate {hit[1]:.0%})")
    if not found:
        raise RuntimeError(
            f"no checkpoint among {[os.path.basename(c) for c in candidates]} could be made "
            f"to fail at or below {FAILURE_MAX_SUCCESS_RATE:.0%} success, even at noise "
            f"{FAILURE_NOISE_LADDER[-1]}. Without failed episodes the 'succ' and 'all' "
            "training conditions produce identical models. Save an earlier checkpoint "
            "during expert_train.py, or raise FAILURE_MAX_SUCCESS_RATE.")
    per = max(1, n_episodes // len(found))
    return [(ck, ns, rate, per) for ck, ns, rate in found]


# ---- the entry point ----------------------------------------------------

def collect(run_dir, holdout=(), episodes=80, seeds=(0, 1, 2),
            val_fraction=DEFAULT_VAL_FRACTION, expert_dir=EXPERT_POLICY_DIR,
            verbose=True):
    """
    Collect a T2S dataset into run_dir. Writes dataset.npz + dataset_summary.json.
    """
    import json

    from stable_baselines3 import SAC
    from env_utils import make_fixed_scene_env

    eligible, reserved = discover_checkpoints(expert_dir, holdout)
    if not eligible:
        raise FileNotFoundError(
            f"no collectable checkpoints under {expert_dir} "
            f"(found {len(reserved)} but all are in holdout)")
    if verbose:
        print(f"collecting from {len(eligible)}: {[os.path.basename(p) for p in eligible]}")
        if reserved:
            print(f"reserved for Stage 4: {[os.path.basename(p) for p in reserved]}")

    scene = make_fixed_scene_env()

    if verbose:
        print(f"\nprobing for failure sources:")
    fail_sources = _failure_sources(scene, eligible,
                                    int(episodes * FAILURE_EPISODE_SHARE), verbose=verbose)

    # (checkpoint, noise, n_episodes) — every checkpoint at every noise level,
    # plus the verified-failing batches
    plan = [(ck, ns, episodes if ns == 0.0 else int(episodes * NOISE_EPISODE_SHARE))
            for ck in eligible for ns in NOISE_LEVELS]
    plan = [(ck, ns, n) for ck, ns, _r, n in fail_sources] + plan

    policies = {ck: SAC.load(ck) for ck, _n, _e in plan}

    episodes_out = []
    for master_seed in seeds:
        rng = np.random.default_rng(master_seed)
        seed = master_seed * 100_000        # disjoint reset block per seed
        for ck, ns, n in plan:
            tag = f"{os.path.basename(ck)}_n{ns}"
            for _ in range(n):
                obs_seq, _flags, fs = collect_episode(policies[ck], scene, seed=seed,
                                                      noise_std=ns, rng=rng)
                seed += 1
                episodes_out.append(dict(obs=obs_seq, steps=build_labels(obs_seq, fs),
                                         success=fs, source=tag, collection_seed=master_seed))
        if verbose:
            mine = [e for e in episodes_out if e["collection_seed"] == master_seed]
            print(f"  seed {master_seed}: {len(mine)} episodes "
                  f"({sum(1 for e in mine if e['success'] is not None)} successful)")
    scene.close()

    ep_splits, split_info = assign_splits(episodes_out, val_fraction=val_fraction)

    def stack(fn, dtype=None):
        return np.concatenate([fn(i, e) for i, e in enumerate(episodes_out)])

    X = np.concatenate([e["obs"] for e in episodes_out])
    np.savez(
        os.path.join(run_dir, "dataset.npz"),
        X=X,
        y_steps=np.concatenate([e["steps"] for e in episodes_out]),
        episode_ids=stack(lambda i, e: np.full(len(e["obs"]), i, np.int32)),
        frame_idxs=stack(lambda i, e: np.arange(len(e["obs"]), dtype=np.int32)),
        collection_seeds=stack(lambda i, e: np.full(len(e["obs"]), e["collection_seed"], np.int32)),
        split=stack(lambda i, e: np.full(len(e["obs"]), ep_splits[i], dtype="U5")),
    )

    n_unique = len({hash(e["obs"].tobytes()) for e in episodes_out})
    n_succ = sum(1 for e in episodes_out if e["success"] is not None)

    def split_stats(name):
        eps = [e for e, s in zip(episodes_out, ep_splits) if s == name]
        return dict(episodes=len(eps), rows=sum(len(e["obs"]) for e in eps),
                    successful=sum(1 for e in eps if e["success"] is not None),
                    failed=sum(1 for e in eps if e["success"] is None))

    summary = dict(
        total_rows=int(len(X)), total_episodes=len(episodes_out), obs_dim=int(X.shape[1]),
        unique_trajectories=n_unique,
        duplicate_fraction=float(1 - n_unique / len(episodes_out)),
        successful_episodes=n_succ, failed_episodes=len(episodes_out) - n_succ,
        collection_seeds=[int(s) for s in seeds], episodes_per_checkpoint=episodes,
        split=dict(requested_val_fraction=val_fraction, **split_info,
                   train=split_stats("train"), val=split_stats("val")),
        sources=sorted({e["source"] for e in episodes_out}),
        collected_checkpoints=sorted({os.path.basename(c) for c, _n, _e in plan}),
        holdout_checkpoints=sorted(os.path.basename(p) for p in reserved),
        failure_sources=[dict(checkpoint=os.path.basename(c), noise=ns,
                              measured_success_rate=r, episodes=n)
                         for c, ns, r, n in fail_sources],
    )
    with open(os.path.join(run_dir, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        tr, va = summary["split"]["train"], summary["split"]["val"]
        print(f"\n{len(episodes_out)} episodes -> {n_unique} unique "
              f"({summary['duplicate_fraction']:.1%} duplicates), {len(X):,} rows")
        print(f"{n_succ} successful / {summary['failed_episodes']} failed")
        print(f"split: train {tr['episodes']} eps ({tr['successful']}s/{tr['failed']}f) | "
              f"val {va['episodes']} eps ({va['successful']}s/{va['failed']}f)")
        if va["failed"] == 0:
            print(">>> WARNING: val has no FAILED episodes — succ/all cannot be compared")
        if summary["duplicate_fraction"] > 0.1:
            print(">>> WARNING: high duplicate fraction; add more seeds")
    return summary