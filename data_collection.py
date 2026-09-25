"""
Stage 2: Time2Success data collection.

    from data_collection import collect
    collect(run_dir, holdout=["..._200000.zip"])

Data comes only from the expert checkpoints, each rolled as-is on the fixed
scene with the policy sampled (deterministic=False). No action noise is
injected: every episode is labelled with the time-to-success of a real policy.
Failed episodes come from weak checkpoints, which are verified by probing
before collection starts.
"""
import glob
import os
import warnings

import numpy as np

from config import (HAND_POS_IDX, PEG_POS_IDX, SUCCESS_KEY, MAX_STEPS,
                    CENSOR_LABEL, EXPERT_POLICY_DIR, TASK_SLUG,
                    REFERENCE_SEEDS, REFERENCE_KINDS, PERTURB_STEPS,
                    REFERENCE_TAIL, steps_remaining_curve)

FAILURE_MAX_SUCCESS_RATE = 0.4
FAILURE_PROBES = 10
PROBE_SEED_START = 50_000_000
COLLECTION_SEED_BLOCK = 100_000

DEFAULT_VAL_FRACTION = 0.2
SPLITS = ("train", "val")

VEL_WINDOW, VEL_COS_THRESH, VEL_MIN_MOTION = 5, 0.7, 3e-4
POSE_WINDOW, POSE_STD_THRESH, CONFIRM_FRAMES = 5, 2e-3, 2


# ---- control-onset detection --------------------------------------------

def coupling_signals(obs_seq):
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
    cos, pose_std = coupling_signals(obs_seq)
    candidate = (cos > VEL_COS_THRESH) & (pose_std < POSE_STD_THRESH)
    run = 0
    for t in range(len(candidate)):
        run = run + 1 if candidate[t] else 0
        if run >= CONFIRM_FRAMES:
            return t - CONFIRM_FRAMES + 1
    return None


def build_labels(obs_seq, first_success, censor_label=CENSOR_LABEL):
    return steps_remaining_curve(len(obs_seq), first_success, censor_label=censor_label)


# ---- train/val split ----------------------------------------------------

def assign_splits(episodes, val_fraction=DEFAULT_VAL_FRACTION, dedupe=True):
    """Deterministic episode-level split, stratified by (source, succeeded), duplicate-safe."""
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
                    break
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


# ---- rollout ------------------------------------------------------------

def collect_episode(policy, env, seed=None):
    """One episode with the policy sampled -> (obs_seq, flags, first_success)."""
    obs, _ = env.reset(seed=seed)
    obs_list, flags, first_success = [], [], None
    for t in range(MAX_STEPS):
        obs_list.append(np.asarray(obs, dtype=np.float32).copy())
        action, _ = policy.predict(obs, deterministic=False)
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
    """(eligible, reserved) checkpoint paths, weakest first, final last."""
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


# ---- failure verification -----------------------------------------------

def probe_success_rate(policy, env, n_probe=FAILURE_PROBES, seed_start=PROBE_SEED_START):
    """Success rate on reset seeds disjoint from every collection block."""
    return sum(collect_episode(policy, env, seed=seed_start + i)[2] is not None
               for i in range(n_probe)) / n_probe


def _probe_checkpoints(scene, policies, verbose=True):
    """Measure every checkpoint as-is; raise if none fails often enough."""
    import torch

    rates = {}
    for k, (ck, pol) in enumerate(policies.items()):
        torch.manual_seed(PROBE_SEED_START + k)
        rates[ck] = probe_success_rate(pol, scene)
        if verbose:
            tag = "  <- failure source" if rates[ck] <= FAILURE_MAX_SUCCESS_RATE else ""
            print(f"  {os.path.basename(ck):<40} success {rates[ck]:.0%}{tag}")
    if not any(r <= FAILURE_MAX_SUCCESS_RATE for r in rates.values()):
        raise RuntimeError(
            f"no checkpoint succeeds at or below {FAILURE_MAX_SUCCESS_RATE:.0%} "
            f"(rates: { {os.path.basename(c): r for c, r in rates.items()} }). "
            "Without failed episodes 'succ' and 'all' train identical models. "
            "Save earlier checkpoints in expert_train.py.")
    return rates


# ---- reference trajectories ---------------------------------------------

def _roll_reference(policy, scene_factory, seed, perturb=0, tail=REFERENCE_TAIL):
    """One reference rollout with frames; `perturb` random actions first, then the policy."""
    import torch

    env = scene_factory(render_mode="rgb_array")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)

    obs_list, frames, success_step = [], [], None
    for t in range(MAX_STEPS):
        obs_list.append(np.asarray(obs, dtype=np.float32).copy())
        frames.append(env.render())
        if t < perturb:
            action = rng.uniform(env.action_space.low, env.action_space.high)
        else:
            action, _ = policy.predict(obs, deterministic=False)
        obs, _r, term, trunc, info = env.step(action)
        if info.get(SUCCESS_KEY, 0) and success_step is None:
            success_step = t + 1
        if success_step is not None and t >= success_step + tail:
            break
        if term or trunc:
            break
    obs_list.append(np.asarray(obs, dtype=np.float32).copy())
    frames.append(env.render())
    env.close()
    return (np.array(obs_list, dtype=np.float32), frames, success_step)


def collect_references(run_dir, success_ckpt, failure_ckpt, expert_dir=EXPERT_POLICY_DIR,
                       seeds=REFERENCE_SEEDS, verbose=True):
    """Roll and store the reference set once so every downstream tool scores identical states."""
    import json

    from stable_baselines3 import SAC
    from env_utils import make_fixed_scene_env
    from t2s_video import save_video

    out_dir = os.path.join(run_dir, "references")
    os.makedirs(out_dir, exist_ok=True)
    pol = {"success": SAC.load(_resolve(success_ckpt, expert_dir)),
           "failure": SAC.load(_resolve(failure_ckpt, expert_dir))}
    plan = {"clean_success": ("success", 0), "clean_failure": ("failure", 0),
            "perturbed_recovery": ("success", PERTURB_STEPS),
            "perturbed_failure": ("failure", PERTURB_STEPS)}

    index = []
    for kind in REFERENCE_KINDS:
        who, perturb = plan[kind]
        for sd in seeds:
            obs, frames, ss = _roll_reference(pol[who], make_fixed_scene_env, sd,
                                              perturb=perturb)
            name = f"{kind}_seed{sd}"
            np.savez(os.path.join(out_dir, name + ".npz"), obs=obs,
                     y_steps=steps_remaining_curve(len(obs), ss),
                     success_step=np.array([-1 if ss is None else ss]))
            try:
                save_video(frames, os.path.join(out_dir, name + ".mp4"))
                has_video = True
            except Exception as e:
                has_video = False
                if verbose:
                    print(f"    (no video for {name}: {e})")
            index.append(dict(name=name, kind=kind, seed=int(sd), policy=who,
                              perturb_steps=perturb, n_frames=len(obs),
                              success_step=None if ss is None else int(ss),
                              video=has_video))
            if verbose:
                print(f"  {name:<34}{len(obs):>4} frames  "
                      + (f"success@{ss}" if ss is not None else "never succeeded"))

    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(dict(success_ckpt=os.path.basename(_resolve(success_ckpt, expert_dir)),
                       failure_ckpt=os.path.basename(_resolve(failure_ckpt, expert_dir)),
                       seeds=[int(s) for s in seeds], perturb_steps=PERTURB_STEPS,
                       trajectories=index), f, indent=2)
    return index


# ---- entry point --------------------------------------------------------

def collect(run_dir, holdout=(), episodes=80, seeds=(0, 1, 2),
            val_fraction=DEFAULT_VAL_FRACTION, expert_dir=EXPERT_POLICY_DIR,
            references=True, reference_policies=None, verbose=True):
    """
    Collect `episodes` per checkpoint per seed from every non-holdout checkpoint.
    Writes dataset.npz (X, y_steps, episode_ids, frame_idxs, collection_seeds,
    split) and dataset_summary.json.
    """
    import json

    import torch
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
    policies = {ck: SAC.load(ck) for ck in eligible}

    if verbose:
        print("\nprobing checkpoints:")
    rates = _probe_checkpoints(scene, policies, verbose=verbose)

    episodes_out = []
    for master_seed in seeds:
        torch.manual_seed(master_seed)
        seed = master_seed * COLLECTION_SEED_BLOCK
        for ck in eligible:
            tag = os.path.basename(ck)
            for _ in range(episodes):
                obs_seq, _flags, fs = collect_episode(policies[ck], scene, seed=seed)
                seed += 1
                episodes_out.append(dict(obs=obs_seq, steps=build_labels(obs_seq, fs),
                                         success=fs, source=tag,
                                         collection_seed=master_seed))
        if verbose:
            mine = [e for e in episodes_out if e["collection_seed"] == master_seed]
            print(f"  seed {master_seed}: {len(mine)} episodes "
                  f"({sum(1 for e in mine if e['success'] is not None)} successful)")
    scene.close()

    ep_splits, split_info = assign_splits(episodes_out, val_fraction=val_fraction)

    def stack(fn):
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

    def source_stats(ck):
        tag = os.path.basename(ck)
        eps = [e for e in episodes_out if e["source"] == tag]
        succ = sum(1 for e in eps if e["success"] is not None)
        return dict(checkpoint=tag, probed_success_rate=rates[ck],
                    episodes=len(eps), successful=succ, failed=len(eps) - succ,
                    collected_success_rate=succ / len(eps) if eps else 0.0)

    per_source = [source_stats(ck) for ck in eligible]
    summary = dict(
        total_rows=int(len(X)), total_episodes=len(episodes_out), obs_dim=int(X.shape[1]),
        unique_trajectories=n_unique,
        duplicate_fraction=float(1 - n_unique / len(episodes_out)),
        successful_episodes=n_succ, failed_episodes=len(episodes_out) - n_succ,
        collection_seeds=[int(s) for s in seeds], episodes_per_checkpoint=episodes,
        policy_sampling="stochastic", action_noise=None, scene="fixed",
        split=dict(requested_val_fraction=val_fraction, **split_info,
                   train=split_stats("train"), val=split_stats("val")),
        sources=[os.path.basename(c) for c in eligible],
        collected_checkpoints=[os.path.basename(c) for c in eligible],
        holdout_checkpoints=sorted(os.path.basename(p) for p in reserved),
        per_checkpoint=per_source,
        failure_sources=[s for s in per_source
                         if s["probed_success_rate"] <= FAILURE_MAX_SUCCESS_RATE],
    )
    if references:
        if verbose:
            print("\nrolling reference trajectories:")
        sc, fc = reference_policies or (eligible[-1], eligible[0])
        summary["references"] = collect_references(run_dir, sc, fc,
                                                   expert_dir=expert_dir,
                                                   verbose=verbose)

    with open(os.path.join(run_dir, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        tr, va = summary["split"]["train"], summary["split"]["val"]
        print(f"\n{len(episodes_out)} episodes -> {n_unique} unique "
              f"({summary['duplicate_fraction']:.1%} duplicates), {len(X):,} rows")
        print(f"{n_succ} successful / {summary['failed_episodes']} failed")
        for s in per_source:
            print(f"  {s['checkpoint']:<40} {s['successful']:>4}s / {s['failed']:>4}f")
        print(f"split: train {tr['episodes']} eps ({tr['successful']}s/{tr['failed']}f) | "
              f"val {va['episodes']} eps ({va['successful']}s/{va['failed']}f)")
        if va["failed"] == 0:
            print(">>> WARNING: val has no FAILED episodes — succ/all cannot be compared")
        if summary["duplicate_fraction"] > 0.1:
            print(">>> WARNING: high duplicate fraction; add more seeds")
    return summary