"""
Stage 5 (part 2): downstream RL training against the frozen T2S reward.

Everything that decides *where output goes* is delegated to results.py.
The SAC loop + success-rate callback live in rl_common.py, shared with
expert_train.py (Stage 1).

ONE gamma flows from config.RL_GAMMA into all three places that need it:
SAC's critic, the shaping term inside Time2SuccessRewardWrapper, and
VecNormalize's discounted-return statistics. They used to be three
independent defaults.
"""
import json
import os

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from config import RL_GAMMA, RL_SEEDS, TIME_PENALTY
from policy_env import make_policy_train_env
from reward_fn import assert_shaping_gamma_safe
from rl_common import train_sac


def train_policy(run_dir, predict_t2s, reward_mode="difference_timed", total_timesteps=400_000,
                 n_envs=6, eval_freq=10_000, ckpt_freq=50_000, n_eval_episodes=5,
                 seed=0, smoke_test_steps=3_000, time_penalty=TIME_PENALTY,
                 gamma=RL_GAMMA, terminate_on_success=True, norm_reward=True,
                 pred_max_for_gamma_check=None, ent_coef="auto",
                 target_entropy="auto", stall_steps=0):
    """
    Trains SAC against the T2S reward. `predict_t2s` should come from
    t2s_predict.load_t2s_predictor so the frozen model / normalization
    match exactly what Stage 3/4 produced.

    pred_max_for_gamma_check: the largest prediction this T2S model actually
        emits (reward_preview reports it as `pred_max`). Passing it turns the
        hovering check from a worst-case guess into a real one — see
        reward_fn.assert_shaping_gamma_safe.
    """
    if pred_max_for_gamma_check is not None:
        assert_shaping_gamma_safe(gamma=gamma, time_penalty=time_penalty,
                                  pred_max=pred_max_for_gamma_check, reward_mode=reward_mode)

    # stall termination is applied to TRAINING envs only. The eval env keeps the
    # full 500 steps so success_rate stays comparable across runs with and
    # without it — otherwise a slow-but-eventual success could be cut short and
    # scored as a failure, which would make the flag look worse than it is.
    train_env = VecMonitor(DummyVecEnv(
        [make_policy_train_env(predict_t2s, reward_mode, time_penalty=time_penalty,
                               gamma=gamma, terminate_on_success=terminate_on_success,
                               stall_steps=stall_steps)
         for _ in range(n_envs)]))
    # VecNormalize keeps a running std of the DISCOUNTED return, so it has its
    # own gamma; leaving it at the 0.99 default while SAC ran at another value
    # meant the reward was rescaled by statistics from a different MDP.
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=norm_reward,
                             clip_reward=10.0, gamma=gamma)

    eval_env = make_policy_train_env(predict_t2s, reward_mode, time_penalty=time_penalty,
                                     gamma=gamma,
                                     terminate_on_success=terminate_on_success)()

    model, history = train_sac(
        train_env, eval_env, run_dir, total_timesteps=total_timesteps, ckpt_prefix="policy",
        eval_freq=eval_freq, ckpt_freq=ckpt_freq, n_eval_episodes=n_eval_episodes,
        smoke_test_steps=smoke_test_steps, seed=seed, gamma=gamma,
        # ent_coef="auto" LEARNS alpha to hit target_entropy, which SB3
        # defaults to -dim(action) = -4. Raising the target (e.g. -2) keeps the
        # policy more stochastic for longer; a fixed float pins alpha instead.
        sac_kwargs=dict(buffer_size=300_000, ent_coef=ent_coef,
                        target_entropy=target_entropy),
    )
    train_env.save(os.path.join(run_dir, "vecnormalize.pkl"))
    return model, history


def load_vecnormalize(run_dir, venv):
    """
    Restore the VecNormalize statistics a policy was trained with, configured
    for evaluation (training=False, norm_reward=False). With norm_reward=True
    the policy was trained against a rescaled reward stream, so any later
    stage that skips this is evaluating under different conditions.
    """
    path = os.path.join(run_dir, "vecnormalize.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no vecnormalize.pkl in {run_dir}; this run either predates the "
            "save or was trained without VecNormalize")
    venv = VecNormalize.load(path, venv)
    venv.training = False
    venv.norm_reward = False
    return venv


def run_sweep(model_specs, results_module, seeds=RL_SEEDS, sweep_name="sweep_v1",
              reward_mode="difference_timed", total_timesteps=400_000,
              gamma=RL_GAMMA, skip_existing=True, pred_maxes=None, force=False,
              **train_kwargs):
    """
    Trains one policy per (model, seed) pair.

    model_specs: {label: (t2s_run_dir, combo, t2s_seed)} — exactly the dict the
        notebook already builds when resolving MODELS. Previously this function
        took (t2s_run_dir, combos, t2s_seeds) and had to be called once per
        model in a loop, with sweep_name carrying the label, which produced run
        dirs like `sweep_v5_td0_all_v2_td0_all_seed1` — the combo name twice.
        Run dirs are now `{sweep_name}_{label}_seed{seed}`.

    seeds: RL seeds, default config.RL_SEEDS = (0, 1, 2). The scene is FIXED,
        so the seed varies SAC's initialization, exploration noise and replay
        sampling — not the task. Three of them is what distinguishes a reward
        that is reliably trainable from one that got lucky once; reporting
        mean without std across them would hide exactly that.

    pred_maxes: optional {label: max_prediction} from reward_preview, used for
        the hovering-safety check on the shaping gamma.

    skip_existing: a (label, seed) whose run dir already has a non-empty
        eval_history.json is loaded from disk instead of retrained, so a crash
        on run 7 of 9 does not cost the first six.

    Returns {(label, seed): history}, ready for summarize_sweep.
    """
    import t2s_predict

    # --force bypasses the gate in run_4_policy, but train_policy re-ran the
    # same hovering assertion and raised anyway, so the flag did not work.
    pred_maxes = {} if force else (pred_maxes or {})
    sweep_results = {}
    total = len(model_specs) * len(seeds)
    i = 0

    for label, (t2s_run_dir, combo, t2s_seed) in model_specs.items():
        method, condition = combo.rsplit("_", 1)
        # one frozen predictor per model, reused across every RL seed so the
        # only thing varying within a model is the RL seed
        predict_fn = t2s_predict.load_t2s_predictor(t2s_run_dir, method, condition,
                                                    seed=t2s_seed)
        for sd in seeds:
            i += 1
            run_name = f"{sweep_name}_{label}_seed{sd}"
            run_dir = results_module.new_run_dir(
                "policy", run_name,
                meta=dict(sweep=sweep_name, label=label, combo=combo, t2s_seed=t2s_seed,
                          rl_seed=sd, reward_mode=reward_mode, gamma=gamma,
                          timesteps=total_timesteps))

            history_path = os.path.join(run_dir, "eval_history.json")
            # a non-empty history is NOT evidence of a finished run: a smoke
            # test writes one, and so does a crash at 20k of 400k. The final
            # checkpoint is the only thing train_sac writes on completion.
            finished = os.path.exists(os.path.join(run_dir, "policy_final.zip"))
            if skip_existing and os.path.exists(history_path) and finished:
                with open(history_path) as f:
                    existing = json.load(f)
                if existing:
                    print(f"[{i}/{total}] {run_name}: already done ({len(existing)} evals), skipping")
                    sweep_results[(label, sd)] = existing
                    continue

            print(f"[{i}/{total}] {run_name}: training {total_timesteps:,} steps")
            _model, history = train_policy(
                run_dir, predict_fn, reward_mode=reward_mode,
                total_timesteps=total_timesteps, seed=sd, gamma=gamma,
                pred_max_for_gamma_check=pred_maxes.get(label), **train_kwargs)
            sweep_results[(label, sd)] = history

    return sweep_results


def summarize_sweep(sweep_results):
    """
    Aggregates a {(label, seed): history} sweep into per-model statistics.

    Reporting mean AND std across seeds is the whole point of running several:
    a model that reaches 0.8 on one seed and 0.1 on two others is not better
    than one that reliably reaches 0.5, and only the std reveals that.
    """
    import numpy as np
    from collections import defaultdict

    by_label = defaultdict(dict)
    for (label, seed), history in sweep_results.items():
        by_label[label][seed] = history

    def series(history, key, pick):
        vals = [h[key] for h in history if h.get(key) is not None]
        return pick(vals) if vals else None

    rows = []
    for label, seed_histories in by_label.items():
        finals, bests, times = [], [], []
        peg_final, peg_best, hand_best, ctrl, moved = [], [], [], [], []
        for _seed, history in seed_histories.items():
            if not history:
                continue
            finals.append(history[-1]["success_rate"])
            bests.append(max(h["success_rate"] for h in history))
            for acc, key, pick in (
                (times,     "mean_time_to_success",        lambda v: v[-1]),
                # closest the peg EVER got across the whole run, not just at the
                # final eval — when success_rate is 0 everywhere this is the only
                # thing separating the runs
                (peg_final, "mean_min_peg_goal_distance",  lambda v: v[-1]),
                (peg_best,  "mean_min_peg_goal_distance",  min),
                # peg->goal is blind to reaching (an arm that reaches but never
                # grasps leaves the peg untouched), so track hand->peg separately
                (hand_best, "best_min_hand_peg_distance",  min),
                (ctrl,      "control_rate",                max),
                (moved,     "peg_moved_rate",              max),
            ):
                v = series(history, key, pick)
                if v is not None:
                    acc.append(v)
        if not finals:
            continue
        rows.append(dict(
            label=label, seeds=sorted(seed_histories.keys()), n_seeds=len(finals),
            mean_final_success=float(np.mean(finals)),
            std_final_success=float(np.std(finals)),
            per_seed_final_success=finals,
            mean_best_success=float(np.mean(bests)),
            mean_final_time_to_success=float(np.mean(times)) if times else None,
            mean_final_min_peg_goal_distance=float(np.mean(peg_final)) if peg_final else None,
            best_min_peg_goal_distance=float(np.min(peg_best)) if peg_best else None,
            best_min_hand_peg_distance=float(np.min(hand_best)) if hand_best else None,
            mean_control_rate=float(np.mean(ctrl)) if ctrl else None,
            # best-case across seeds, kept but NAMED as such. The mean is what
            # a sweep is for; a max over seeds is an estimate of luck.
            best_peg_moved_rate=float(np.max(moved)) if moved else None,
            mean_peg_moved_rate=float(np.mean(moved)) if moved else None,
            std_peg_moved_rate=float(np.std(moved)) if moved else None,
            peg_moved_rate=float(np.mean(moved)) if moved else None,
        ))
    # rank by success first, then by how close the peg got — so all-zero-success
    # sweeps still produce a meaningful ordering instead of an arbitrary one
    return sorted(rows, key=lambda r: (
        -r["mean_final_success"],
        r["best_min_peg_goal_distance"] if r["best_min_peg_goal_distance"] is not None
        else float("inf"),
    ))


def format_sweep_table(sweep_rows):
    """Pure-string table of summarize_sweep output."""
    if not sweep_rows:
        return "(no sweep results)"
    header = (f"{'model':<26}{'n':>3}{'final success':>15}"
              f"{'hand->peg':>11}{'peg->goal':>11}{'peg moved':>11}{'ctrl':>7}")
    lines = [header, "-" * len(header)]

    def fmt(v, prec=4):
        return f"{v:.{prec}f}" if v is not None else "--"

    for r in sweep_rows:
        lines.append(
            f"{r['label']:<26}{r['n_seeds']:>3}"
            f"{r['mean_final_success']:>8.2f} ±{r['std_final_success']:<5.2f}"
            f"{fmt(r.get('best_min_hand_peg_distance')):>11}"
            f"{fmt(r.get('best_min_peg_goal_distance')):>11}"
            f"{fmt(r.get('peg_moved_rate'), 2):>11}"
            f"{fmt(r.get('mean_control_rate'), 2):>7}"
        )
    return "\n".join(lines)