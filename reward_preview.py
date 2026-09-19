"""
Pre-training reward diagnostic.

Answers "what reward would this T2S model actually emit?" WITHOUT training
anything — it replays stored trajectories (from
t2s_eval.collect_eval_trajectories) through the same reward arithmetic
policy_env.Time2SuccessRewardWrapper uses. Minutes instead of hours, and it
can rule out a reward before a sweep.

CHANGES FROM THE PREVIOUS VERSION
---------------------------------
  - MORE THAN TWO TRAJECTORIES. `return_gap` was computed from exactly one
    success and one failure rollout, then format_preview_table sorted every
    (combo x mode) pair by it — model selection on a sample of two, with no
    variance estimate. collect_eval_trajectories now returns several seeds per
    scenario and the gap is reported as mean +/- std across success/failure
    pairs. A gap of 800 +/- 900 is not the same claim as 800 +/- 40, and the
    old table could not tell them apart.
  - THE SUCCESS LATCH NO LONGER EATS THE SUCCESS STEP. The latch zeroes
    post-success rewards to match the wrapper. rewards[i] covers the
    transition s_i -> s_{i+1}, so the post-success transitions are i >=
    success_step and the success-ACHIEVING transition is i = success_step - 1.
    That arithmetic was right; the input was not, because success_step was
    recorded one frame early (as the step index rather than the first
    successful state — see config), so zeroing rewards[success_step:] also
    deleted the success-achieving reward. Fixed at the source by the shared
    convention, so the latch here is unchanged.

    Magnitude, measured on a model with a realistic 2.5-frame error at the
    success frame: the lost reward is -2.50 in 'absolute', +1.15 in
    'difference', +0.15 in 'difference_timed'. It is exactly zero when the
    prediction reaches 0 at success, which is why it went unnoticed. So this
    is a correctness fix worth one step of reward, not a decisive one — set
    against return gaps in the hundreds it will not change a ranking. The
    consequential off-by-one is the same convention error in t2s_eval's ground
    truth, which shifted every reported MAE.
  - GAMMA. compute_step_rewards now takes gamma and passes it to step_reward,
    which needs it for the shaping modes.
  - `pred_max` and `hover_reward` ARE REPORTED. Feed pred_max to
    policy_train(pred_max_for_gamma_check=...) so the hovering guard uses this
    model's real prediction range rather than the clip ceiling.

The diagnostics, and why each one matters:

  1. return_gap — total return on a SUCCESS trajectory minus total return on
     a FAILURE trajectory. THE critical one. If a reward assigns similar
     returns to a trajectory that succeeds and one that never does, no RL
     algorithm can learn to prefer success, regardless of budget or
     hyperparameters. A large positive gap is the minimum bar.

     Note the gap is in RAW reward units, while training runs under
     VecNormalize(norm_reward=True), which rescales by the running std of the
     discounted return. So the gap's SIGN and its size relative to
     reward_std are meaningful; its absolute magnitude is not comparable to
     anything SAC sees.

  2. hover_reward — what the reward pays for a step that changes nothing, at
     this model's maximum prediction. MUST BE NEGATIVE. With the (correct)
     gamma in the shaping term, difference_timed pays
     (1-gamma)*pred - time_penalty for standing still, which goes POSITIVE
     once (1-gamma)*pred exceeds time_penalty — so at gamma=0.99 a model
     predicting 200 is paid +1.0 per step to freeze. This column is the
     go/no-go check on the gamma/time_penalty/pred_max combination, and it is
     why config.RL_GAMMA is 0.999.

  3. dead_step_fraction — fraction of steps whose |reward| is below eps. A
     dead step provides no gradient. Every model flatlines to a constant after
     ~150 steps on failure trajectories, which is exactly what produces dead
     steps: in 'absolute' mode a constant prediction is a constant penalty (no
     signal about which action is better), and in 'difference' mode ~0 reward.

  4. progress_alignment — Spearman correlation between per-step reward and
     per-step REDUCTION in peg-to-goal distance. When the policy makes real
     physical progress, does the reward go up? A reward can have a fine
     return_gap and still be misaligned step-to-step, which makes credit
     assignment very hard.

  5. reward_std / range — dynamic range. Near-zero std means the reward is
     effectively constant, i.e. uninformative even if its level looks fine.
"""
import numpy as np
from scipy.stats import spearmanr

from config import RL_GAMMA, TIME_PENALTY
from progress import peg_goal_distance_curve

DEAD_STEP_EPS = 1e-3


def _as_list(trajs):
    """collect_eval_trajectories now returns lists; tolerate a bare dict."""
    if trajs is None:
        return []
    return list(trajs) if isinstance(trajs, (list, tuple)) else [trajs]


def compute_step_rewards(preds, reward_mode="difference_timed", time_penalty=None,
                         gamma=RL_GAMMA):
    """
    Per-step rewards for a stored prediction sequence (length T-1, since each
    reward corresponds to a transition).

    Delegates to reward_fn.step_reward — the SAME function the live training
    wrapper calls — so a preview can never disagree with what training does.
    (An earlier version reimplemented the arithmetic locally, which meant
    editing the wrapper silently had no effect on the preview.)
    """
    from reward_fn import step_reward, DEFAULT_TIME_PENALTY

    preds = np.asarray(preds, dtype=np.float64)
    if len(preds) < 2:
        return np.zeros(0)
    tp = DEFAULT_TIME_PENALTY if time_penalty is None else time_penalty
    return np.array([
        step_reward(preds[i], preds[i + 1], reward_mode=reward_mode,
                    time_penalty=tp, gamma=gamma)
        for i in range(len(preds) - 1)
    ], dtype=np.float64)


def analyze_reward_signal(preds, obs_seq=None, reward_mode="difference_timed",
                          eps=DEAD_STEP_EPS, time_penalty=None, gamma=RL_GAMMA,
                          success_step=None):
    """
    Per-trajectory reward diagnostics. obs_seq (raw observations for the same
    trajectory) enables progress_alignment; omit it to skip that metric.
    """
    rewards = compute_step_rewards(preds, reward_mode, time_penalty=time_penalty, gamma=gamma)
    # Reproduce Time2SuccessRewardWrapper's success latch: once the episode has
    # succeeded the wrapper charges nothing (and normally terminates), so a
    # preview that keeps accruing -time_penalty past success misrepresents the
    # reward exactly at the most important frame. rewards[i] covers the
    # transition into state i+1, so transitions from success_step onward are
    # post-success and rewards[success_step - 1] is the success-achieving one,
    # which must survive. This relies on success_step being the index of the
    # first SUCCESSFUL STATE (config); it used to be one lower, and the slice
    # below then deleted the success reward itself.
    if success_step is not None and len(rewards):
        rewards = rewards.copy()
        rewards[success_step:] = 0.0

    preds_arr = np.asarray(preds, dtype=np.float64)
    base = dict(pred_min=float(preds_arr.min()) if len(preds_arr) else None,
                pred_max=float(preds_arr.max()) if len(preds_arr) else None)
    if len(rewards) == 0:
        return dict(n_steps=0, total_return=0.0, mean_reward=None, reward_std=None,
                    reward_min=None, reward_max=None, dead_step_fraction=None,
                    progress_alignment=None, **base)

    out = dict(
        n_steps=int(len(rewards)),
        total_return=float(rewards.sum()),
        mean_reward=float(rewards.mean()),
        reward_std=float(rewards.std()),
        reward_min=float(rewards.min()),
        reward_max=float(rewards.max()),
        dead_step_fraction=float(np.mean(np.abs(rewards) < eps)),
        **base,
    )

    if obs_seq is not None and len(obs_seq) == len(preds):
        dist = peg_goal_distance_curve(obs_seq)
        progress = dist[:-1] - dist[1:]   # positive = peg moved closer to goal
        if np.std(rewards) > 0 and np.std(progress) > 0:
            out["progress_alignment"] = float(spearmanr(rewards, progress)[0])
        else:
            out["progress_alignment"] = None
    else:
        out["progress_alignment"] = None

    return out


def preview_reward(trajectories, predictors,
                   reward_modes=("absolute", "difference", "difference_timed"),
                   time_penalty=None, gamma=RL_GAMMA):
    """
    Full pre-training preview across every (combo x reward_mode) pair.

    trajectories: t2s_eval.collect_eval_trajectories output —
        {"success": [traj, ...], "failure": [traj, ...]}, each traj with "obs"
        and "success_step".
    predictors: {combo_name: predict_fn}

    Returns {(combo, reward_mode): entry}. Each entry carries per-scenario
    per-trajectory diagnostics, plus return_gap statistics over all
    success x failure pairs and the hovering check.
    """
    from reward_fn import hover_reward, SHAPED_MODES

    tp = TIME_PENALTY if time_penalty is None else time_penalty
    scenarios = {k: _as_list(v) for k, v in trajectories.items()}

    # predict once per (combo, scenario, trajectory), reused across all modes
    preds_cache = {}
    for combo, predict_fn in predictors.items():
        for scenario, trajs in scenarios.items():
            for j, traj in enumerate(trajs):
                preds_cache[(combo, scenario, j)] = np.array(
                    [predict_fn(o) for o in traj["obs"]], dtype=np.float64)

    out = {}
    for combo in predictors:
        pred_max = max((preds_cache[k].max() for k in preds_cache if k[0] == combo),
                       default=0.0)
        for mode in reward_modes:
            per_scenario = {}
            for scenario, trajs in scenarios.items():
                per_scenario[scenario] = [
                    analyze_reward_signal(
                        preds_cache[(combo, scenario, j)], traj["obs"], reward_mode=mode,
                        time_penalty=tp, gamma=gamma, success_step=traj.get("success_step"))
                    for j, traj in enumerate(trajs)
                ]

            entry = dict(per_scenario=per_scenario, pred_max=float(pred_max))
            succ, fail = per_scenario.get("success", []), per_scenario.get("failure", [])
            if succ and fail:
                # every success x failure pair, so the spread reflects both
                # trajectory-to-trajectory sources of variation
                gaps = [s["total_return"] - f["total_return"] for s in succ for f in fail]
                entry.update(
                    return_gap=float(np.mean(gaps)),
                    return_gap_std=float(np.std(gaps)),
                    return_gap_min=float(np.min(gaps)),
                    n_gap_pairs=int(len(gaps)),
                    # a gap whose sign is not consistent across pairs is not a
                    # gap, however large its mean
                    gap_positive_fraction=float(np.mean([g > 0 for g in gaps])),
                )
                mr_s = [s["mean_reward"] for s in succ if s["mean_reward"] is not None]
                mr_f = [f["mean_reward"] for f in fail if f["mean_reward"] is not None]
                # normalized by trajectory length so a long failure trajectory
                # doesn't dominate the gap purely by having more steps
                entry["mean_reward_gap"] = (float(np.mean(mr_s) - np.mean(mr_f))
                                            if mr_s and mr_f else None)
            # what the policy is paid to stand still at this model's worst level
            entry["hover_reward"] = (float(hover_reward(pred_max, reward_mode=mode,
                                                        time_penalty=tp, gamma=gamma))
                                     if mode in SHAPED_MODES else None)
            entry["gamma"] = float(gamma)
            out[(combo, mode)] = entry
    return out


def _agg(rows, key, fn=np.mean):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(fn(vals)) if vals else None


def format_preview_table(preview):
    """Pure-string table of preview_reward output, sorted by return_gap descending."""
    if not preview:
        return "(no preview results)"

    rows = []
    for (combo, mode), entry in preview.items():
        succ = entry["per_scenario"].get("success", [])
        fail = entry["per_scenario"].get("failure", [])
        rows.append(dict(
            combo=combo, mode=mode,
            return_gap=entry.get("return_gap"), gap_std=entry.get("return_gap_std"),
            gap_pos=entry.get("gap_positive_fraction"),
            hover=entry.get("hover_reward"), pred_max=entry.get("pred_max"),
            succ_dead=_agg(succ, "dead_step_fraction"),
            fail_dead=_agg(fail, "dead_step_fraction"),
            align=_agg(succ, "progress_alignment"),
        ))
    rows.sort(key=lambda r: r["return_gap"] if r["return_gap"] is not None else -float("inf"),
              reverse=True)

    def fmt(v, prec=3):
        return f"{v:.{prec}f}" if v is not None else "--"

    header = (f"{'combo':<20}{'mode':<18}{'return gap (mean±std)':>24}{'gap>0':>7}"
              f"{'hover r':>9}{'predmax':>9}{'dead% s/f':>13}{'align':>7}")
    lines = [header, "-" * len(header)]
    for r in rows:
        gap = (f"{r['return_gap']:.0f} ±{r['gap_std']:.0f}"
               if r["return_gap"] is not None else "--")
        dead = f"{fmt(r['succ_dead'], 2)}/{fmt(r['fail_dead'], 2)}"
        lines.append(
            f"{r['combo']:<20}{r['mode']:<18}{gap:>24}{fmt(r['gap_pos'], 2):>7}"
            f"{fmt(r['hover'], 2):>9}{fmt(r['pred_max'], 0):>9}{dead:>13}"
            f"{fmt(r['align'], 2):>7}"
        )
    lines += [
        "",
        "return gap : success return - failure return, mean ± std over all pairs (want LARGE POSITIVE)",
        "gap>0      : fraction of success/failure pairs with a positive gap (want 1.00)",
        "hover r    : reward for standing still at predmax — MUST BE NEGATIVE or freezing pays",
        "dead% s/f  : fraction of ~zero-reward steps (want LOW, especially on failure)",
        "align      : Spearman(reward, peg-distance reduction) (want POSITIVE)",
    ]
    return "\n".join(lines)


def check_preview(preview, reward_mode):
    """
    Turn the preview into a pass/fail verdict per combo for one reward mode.
    Returns {combo: dict(ok=bool, reasons=[...])}. Call this before a sweep
    instead of eyeballing the table.
    """
    verdicts = {}
    for (combo, mode), entry in preview.items():
        if mode != reward_mode:
            continue
        reasons = []
        gap = entry.get("return_gap")
        if gap is None:
            reasons.append("no success/failure pair to compute a return gap")
        elif gap <= 0:
            reasons.append(f"return_gap {gap:.0f} <= 0: reward prefers the failing trajectory")
        elif entry.get("gap_positive_fraction", 1.0) < 1.0:
            reasons.append(
                f"return_gap sign flips on {(1 - entry['gap_positive_fraction']):.0%} of pairs "
                f"(mean {gap:.0f} ± {entry.get('return_gap_std', 0):.0f})")
        hov = entry.get("hover_reward")
        if hov is not None and hov >= 0:
            reasons.append(
                f"hover_reward {hov:+.2f} >= 0 at pred_max {entry.get('pred_max', 0):.0f}: "
                f"standing still is profitable at gamma={entry.get('gamma')}")
        fail_dead = _agg(entry["per_scenario"].get("failure", []), "dead_step_fraction")
        if fail_dead is not None and fail_dead > 0.8:
            reasons.append(f"{fail_dead:.0%} of failure-trajectory steps give no gradient")
        verdicts[combo] = dict(ok=not reasons, reasons=reasons,
                               return_gap=gap, hover_reward=hov,
                               pred_max=entry.get("pred_max"))
    return verdicts


def plot_reward_preview(trajectories, predictors, reward_mode="difference_timed",
                        save_path=None, time_penalty=None, gamma=RL_GAMMA, traj_index=0):
    """
    Per-step reward curves for every combo, on one success and one failure
    trajectory. Makes the dead-signal problem visually obvious: a flat line at
    any level means no gradient, regardless of what level it sits at. A flat
    line ABOVE zero means the policy is paid to hold that pose.
    """
    import matplotlib.pyplot as plt

    scenarios = {k: _as_list(v) for k, v in trajectories.items()}
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    colors = plt.cm.tab10.colors

    for ax, scenario in zip(axes, ("success", "failure")):
        trajs = scenarios.get(scenario) or []
        if not trajs:
            continue
        traj = trajs[min(traj_index, len(trajs) - 1)]
        for i, (combo, predict_fn) in enumerate(predictors.items()):
            preds = np.array([predict_fn(o) for o in traj["obs"]], dtype=np.float64)
            rewards = compute_step_rewards(preds, reward_mode, time_penalty=time_penalty,
                                           gamma=gamma)
            ss = traj.get("success_step")
            if ss is not None and len(rewards):
                rewards = rewards.copy()
                rewards[ss:] = 0.0        # match the wrapper's success latch
            ax.plot(rewards, color=colors[i % len(colors)], label=combo, linewidth=1.4)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{scenario} trajectory (seed {traj.get('seed')}) — "
                     f"per-step reward ({reward_mode}, gamma={gamma})")
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(f"reward ({reward_mode} mode)")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig
