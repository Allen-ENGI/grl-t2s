"""
Stage 5 (part 2): downstream RL training against the frozen T2S reward.

Everything that decides *where output goes* is delegated to results.py —
this module takes a run_dir and writes into it, it never invents its own path.

The actual SAC loop + success-rate callback now live in rl_common.py, shared
with expert_train.py (Stage 1) — see rl_common's docstring for why.
"""
import os

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from policy_env import make_policy_train_env
from rl_common import train_sac


def train_policy(run_dir, predict_t2s, reward_mode="absolute", total_timesteps=1_000_000,
                  n_envs=6, eval_freq=10_000, ckpt_freq=50_000, n_eval_episodes=5,
                  seed=0, smoke_test_steps=3_000, time_penalty=None,
                  terminate_on_success=True):
    """
    Trains SAC against the T2S reward. `predict_t2s` should come from
    t2s_predict.load_t2s_predictor so the frozen model / normalization
    match exactly what Stage 3/4 produced.
    """
    from reward_fn import DEFAULT_TIME_PENALTY
    tp = DEFAULT_TIME_PENALTY if time_penalty is None else time_penalty

    train_env = VecMonitor(DummyVecEnv(
        [make_policy_train_env(predict_t2s, reward_mode, time_penalty=tp,
                                 terminate_on_success=terminate_on_success)
         for _ in range(n_envs)]))
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    eval_env = make_policy_train_env(predict_t2s, reward_mode, time_penalty=tp,
                                       terminate_on_success=terminate_on_success)()

    model, history = train_sac(
        train_env, eval_env, run_dir, total_timesteps=total_timesteps, ckpt_prefix="policy",
        eval_freq=eval_freq, ckpt_freq=ckpt_freq, n_eval_episodes=n_eval_episodes,
        smoke_test_steps=smoke_test_steps, seed=seed,
        sac_kwargs=dict(buffer_size=300_000),
    )
    train_env.save(os.path.join(run_dir, "vecnormalize.pkl"))
    return model, history


def load_vecnormalize(run_dir, venv):
    """
    Restore the VecNormalize statistics a policy was trained with.

    train_policy saves vecnormalize.pkl, but nothing used to read it back —
    every eval/visualisation path built a raw env instead. With
    norm_reward=True the policy was trained against a rescaled reward
    stream, so any later stage that skips this is silently evaluating under
    different conditions than training. Call this whenever you reload a
    trained policy for anything reward-related.

    Returns the wrapped env with training=False and norm_reward=False, the
    standard configuration for evaluation.
    """
    from stable_baselines3.common.vec_env import VecNormalize

    path = os.path.join(run_dir, "vecnormalize.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no vecnormalize.pkl in {run_dir}; this run either predates the "
            "save or was trained without VecNormalize")
    venv = VecNormalize.load(path, venv)
    venv.training = False
    venv.norm_reward = False
    return venv


def run_sweep(t2s_run_dir, results_module, combos, seeds, t2s_seeds=None,
               sweep_name="sweep_v1", reward_mode="absolute", total_timesteps=300_000,
               skip_existing=True, **train_kwargs):
    """
    Trains one policy per (combo, seed) pair. This is the configurable entry
    point: pass however many combos and seeds you want.

    combos: list of T2S combo names, e.g. ['td0_succ'] or
        ['mc_succ','td0_succ','tdlambda_succ']. Total runs = len(combos)*len(seeds).
    seeds: list of RL seeds. The scene is FIXED, so the seed varies SAC's
        initialization, exploration noise and replay sampling — not the task.
        That is what reveals whether a T2S reward is reliably trainable
        rather than lucky once.
    t2s_seeds: optional {combo: t2s_seed} giving which T2S checkpoint seed to
        load per combo (normally the best_seed from the Stage 3 manifest).
        Defaults to 0 for any combo not listed.
    skip_existing: if True, a (combo, seed) whose run dir already has a
        non-empty eval_history.json is loaded from disk instead of retrained.
        This makes the sweep resumable — a crash on run 7 of 9 does not cost
        the first six.

    Returns {(combo, seed): history}, ready for summarize_sweep.
    """
    import json

    import t2s_predict

    t2s_seeds = t2s_seeds or {}
    sweep_results = {}
    total = len(combos) * len(seeds)
    i = 0

    for combo in combos:
        method, condition = combo.rsplit("_", 1)
        # one frozen predictor per combo, reused across every RL seed so the
        # only thing varying within a combo is the RL seed
        predict_fn = t2s_predict.load_t2s_predictor(
            t2s_run_dir, method, condition, seed=t2s_seeds.get(combo, 0))

        for sd in seeds:
            i += 1
            run_name = f"{sweep_name}_{combo}_seed{sd}"
            run_dir = results_module.new_run_dir(
                "policy", run_name,
                meta=dict(sweep=sweep_name, combo=combo, rl_seed=sd,
                           reward_mode=reward_mode, timesteps=total_timesteps))

            history_path = os.path.join(run_dir, "eval_history.json")
            if skip_existing and os.path.exists(history_path):
                with open(history_path) as f:
                    existing = json.load(f)
                if existing:
                    print(f"[{i}/{total}] {run_name}: already done ({len(existing)} evals), skipping")
                    sweep_results[(combo, sd)] = existing
                    continue

            print(f"[{i}/{total}] {run_name}: training {total_timesteps:,} steps")
            _model, history = train_policy(run_dir, predict_fn, reward_mode=reward_mode,
                                            total_timesteps=total_timesteps, seed=sd,
                                            **train_kwargs)
            sweep_results[(combo, sd)] = history

    return sweep_results


def summarize_sweep(sweep_results):
    """
    Aggregates a {(combo, seed): history} sweep into per-combo statistics.

    `history` is the eval_history list from SuccessRateCallback, i.e. a list of
    dicts with 'step', 'success_rate', 'mean_time_to_success'.

    Returns a list of dicts, one per combo, sorted by mean final success rate
    (descending). Reporting mean AND std across seeds is the whole point of
    running multiple seeds — a combo that reaches 0.8 on one seed and 0.1 on
    two others is not better than one that reliably reaches 0.5, and only the
    std reveals that.
    """
    import numpy as np
    from collections import defaultdict

    by_combo = defaultdict(dict)
    for (combo, seed), history in sweep_results.items():
        by_combo[combo][seed] = history

    rows = []
    for combo, seed_histories in by_combo.items():
        finals, bests, times = [], [], []
        min_dists, best_dists, control_rates = [], [], []
        hand_dists, moved_rates = [], []
        for seed, history in seed_histories.items():
            if not history:
                continue
            finals.append(history[-1]["success_rate"])
            bests.append(max(h["success_rate"] for h in history))
            t = [h["mean_time_to_success"] for h in history
                 if h.get("mean_time_to_success") is not None]
            if t:
                times.append(t[-1])
            # model-independent progress: the closest the peg EVER got across
            # the whole run, not just at the final eval — when success_rate is
            # 0 everywhere this is the only thing that separates the runs
            d = [h["mean_min_peg_goal_distance"] for h in history
                 if h.get("mean_min_peg_goal_distance") is not None]
            if d:
                min_dists.append(d[-1])
                best_dists.append(min(d))
            c = [h["control_rate"] for h in history if h.get("control_rate") is not None]
            if c:
                control_rates.append(max(c))
            # reach phase: peg-goal distance cannot see reaching (an arm that
            # reaches but never grasps leaves the peg untouched), so track
            # hand->peg separately
            hd = [h["best_min_hand_peg_distance"] for h in history
                  if h.get("best_min_hand_peg_distance") is not None]
            if hd:
                hand_dists.append(min(hd))
            mv = [h["peg_moved_rate"] for h in history if h.get("peg_moved_rate") is not None]
            if mv:
                moved_rates.append(max(mv))
        if not finals:
            continue
        rows.append(dict(
            combo=combo,
            seeds=sorted(seed_histories.keys()),
            n_seeds=len(finals),
            mean_final_success=float(np.mean(finals)),
            std_final_success=float(np.std(finals)),
            per_seed_final_success=finals,
            mean_best_success=float(np.mean(bests)),
            mean_final_time_to_success=float(np.mean(times)) if times else None,
            mean_final_min_peg_goal_distance=float(np.mean(min_dists)) if min_dists else None,
            best_min_peg_goal_distance=float(np.min(best_dists)) if best_dists else None,
            mean_control_rate=float(np.mean(control_rates)) if control_rates else None,
            best_min_hand_peg_distance=float(np.min(hand_dists)) if hand_dists else None,
            peg_moved_rate=float(np.max(moved_rates)) if moved_rates else None,
        ))
    # rank by success first, then by how close the peg got — so all-zero-success
    # sweeps still produce a meaningful ordering instead of an arbitrary one
    return sorted(rows, key=lambda r: (
        -r["mean_final_success"],
        r["best_min_peg_goal_distance"] if r["best_min_peg_goal_distance"] is not None else float("inf"),
    ))


def format_sweep_table(sweep_rows):
    """Pure-string table of summarize_sweep output, same spirit as t2s_eval.format_report_table."""
    if not sweep_rows:
        return "(no sweep results)"
    header = (f"{'combo':<16}{'n':>3}{'final success':>15}"
              f"{'hand->peg':>11}{'peg->goal':>11}{'peg moved':>11}{'ctrl':>7}")
    lines = [header, "-" * len(header)]
    for r in sweep_rows:
        def fmt(v, prec=4):
            return f"{v:.{prec}f}" if v is not None else "--"
        lines.append(
            f"{r['combo']:<16}{r['n_seeds']:>3}"
            f"{r['mean_final_success']:>8.2f} ±{r['std_final_success']:<5.2f}"
            f"{fmt(r.get('best_min_hand_peg_distance')):>11}"
            f"{fmt(r.get('best_min_peg_goal_distance')):>11}"
            f"{fmt(r.get('peg_moved_rate'), 2):>11}"
            f"{fmt(r.get('mean_control_rate'), 2):>7}"
        )
    return "\n".join(lines)