#!/usr/bin/env python3
"""
Does the T2S reward have the same SHAPE as MetaWorld's native reward?

    python compare_rewards.py --t2s-run v5 --model tdlambdaboot_succ
    python compare_rewards.py --t2s-run v5 --model tdlambdaboot_succ --kinds clean_failure
    python compare_rewards.py --t2s-run v5 --model tdlambdaboot_all --seed-index 1

WHY METAWORLD'S REWARD IS THE RIGHT REFERENCE
---------------------------------------------
It is the reward the expert checkpoints were trained on (PolicyTrain.ipynb),
and those checkpoints are what every T2S model learned from. It is known to
work: SAC reached success on it. So a T2S reward whose shape tracks it on the
same trajectories is plausibly sufficient, and one that diverges shows WHERE.

COMPARE LIKE WITH LIKE — the easy mistake here
----------------------------------------------
MetaWorld's dense reward is a function of the STATE: it is high when the
gripper is near the peg, higher when grasped, highest when inserted. That makes
it a LEVEL — structurally a potential.

WHERE THE TRAJECTORIES COME FROM
--------------------------------
The stored references (data_collection/<run>/references/) do not keep
MetaWorld's reward, so each one is RE-ROLLED from its recorded seed and policy
with the same seeding as collection, and its observations are checked against
the stored ones. A mismatch is reported rather than silently compared — it
would mean the replay is not the trajectory the T2S model was scored on.

MetaWorld's per-step info also exposes its reward COMPONENTS (near_object,
grasp_success, grasp_reward, in_place_reward). Those are plotted against the
T2S prediction, which shows directly whether the T2S model reacts at the grasp.
"""
import argparse
import json
import os
import sys

import numpy as np

STAGE_MANIFEST = "stage_manifest.json"
MW_COMPONENTS = ("near_object", "grasp_success", "grasp_reward", "in_place_reward")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="T2S reward vs MetaWorld native reward, same trajectories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--t2s-run", required=True)
    p.add_argument("--model", "--models", dest="models", nargs="+", required=True,
                   help="one or more combos, e.g. tdlambda_succ tdlambdaboot_all. "
                        "Every model is scored on the SAME re-rolled trajectories, "
                        "so the model is the only thing that varies between rows.")
    p.add_argument("--data-run", default=None,
                   help="data run holding the references (default: from the T2S run)")
    p.add_argument("--kinds", nargs="+",
                   default=["clean_success", "clean_failure",
                            "perturbed_recovery", "perturbed_failure"])
    p.add_argument("--seed-index", type=int, default=0,
                   help="which stored seed of each kind (0 = first)")
    p.add_argument("--gamma", type=float, default=None,
                   help="shaping gamma (default: config.RL_GAMMA)")
    p.add_argument("--reward-mode", default="difference",
                   choices=["absolute", "difference", "difference_timed"],
                   help="which shaped reward to compare. Must match what you train "
                        "with (run_4_policy --reward-mode), or this compares a reward "
                        "the policy never receives. `difference` ignores "
                        "--time-penalty entirely.")
    p.add_argument("--time-penalty", type=float, default=None,
                   help="per-step penalty c (default: config.TIME_PENALTY). Only "
                        "difference_timed uses it; it is a constant shift, so it "
                        "moves the reward MEAN and the return but changes no ranks "
                        "and therefore no Spearman value.")
    p.add_argument("--out", default=None, help="output directory for the plots")
    return p.parse_args(argv)


def reroll(policy, seed, perturb):
    """
    Replay one reference with the SAME seeding as data_collection's
    _roll_reference, additionally recording MetaWorld's reward and info.
    """
    import torch
    from config import MAX_STEPS, SUCCESS_KEY, REFERENCE_TAIL
    from env_utils import make_fixed_scene_env

    env = make_fixed_scene_env()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    O, R, INFO, success_step = [], [], [], None
    for t in range(MAX_STEPS):
        O.append(np.asarray(obs, dtype=np.float32).copy())
        if t < perturb:
            a = rng.uniform(env.action_space.low, env.action_space.high)
        else:
            a, _ = policy.predict(obs, deterministic=False)
        obs, r, term, trunc, info = env.step(a)
        R.append(float(r))
        INFO.append({k: float(info[k]) for k in MW_COMPONENTS if k in info})
        if info.get(SUCCESS_KEY, 0) and success_step is None:
            success_step = t + 1
        if success_step is not None and t >= success_step + REFERENCE_TAIL:
            break
        if term or trunc:
            break
    O.append(np.asarray(obs, dtype=np.float32).copy())
    env.close()
    return np.array(O), np.array(R), INFO, success_step


# rank relationship
def spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(spearmanr(a, b)[0])

#Pearson: linear relationship
def pearson(a, b):
    from scipy.stats import pearsonr
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(pearsonr(a, b)[0])

def analyse(obs, mw_r, pred, success_step, gamma, tp, reward_mode="difference_timed"):
    """Level and difference comparisons for one trajectory."""
    import reward_fn

    n = len(mw_r)                          # one reward per transition
    # Match Time2SuccessRewardWrapper exactly: at the success transition the
    # next potential is forced to 0 (the true remaining time there is 0, while
    # the model predicts ~2-5), and after success the wrapper pays nothing.
    # Without this the plot showed a different reward at the one transition
    # that matters most than the one training actually pays.
    vals = []
    for i in range(n):
        if success_step is not None and i + 1 > success_step:
            vals.append(0.0)
            continue
        nxt = 0.0 if (success_step is not None and i + 1 == success_step) else pred[i + 1]
        vals.append(reward_fn.step_reward(pred[i], nxt, reward_mode,
                                          gamma=gamma, time_penalty=tp))
    t2s_diff = np.array(vals)
    # compare over the part of the episode where something is happening: up to
    # success, or the whole rollout if it never succeeds
    end = success_step if success_step is not None else n
    end = max(min(end, n), 3)
    mw_level = mw_r[:end]
    potential = -pred[1:end + 1]           # -T2S at the state AFTER each transition
    mw_delta = np.diff(np.concatenate([[mw_r[0]], mw_r]))[:end]

    # MAE: does the model predict TIME correctly? Only defined where a
    # countdown truth exists, i.e. on trajectories that succeed. On a failure
    # the true time-to-success is unknown, so MAE is reported as absent rather
    # than scored against the censor value.
    mae = None
    if success_step is not None and success_step > 0:
        from config import steps_remaining_curve
        truth = steps_remaining_curve(len(pred), success_step)
        mae = float(np.mean(np.abs(pred[:success_step] - truth[:success_step])))
    # FALSE PROGRESS: how fast the prediction falls while MetaWorld says
    # nothing is happening. Spearman cannot measure this: when the policy
    # stalls, MetaWorld's reward PLATEAUS, a plateau is a run of tied values,
    # and ties carry no rank information — so a model that keeps counting
    # down through a stall still scored level rho ~0.7 against one that
    # stays put at 1.0. Measured directly: mean per-step drop in the
    # prediction over frames where MetaWorld's reward is flat. Near 0 is
    # correct (nothing is happening, so time-to-success should not fall);
    # near 1.0 means the model thinks it is progressing at full speed while
    # the policy is stuck.
    d_mw = np.diff(mw_r)
    stalled = np.flatnonzero(np.abs(d_mw) < 1e-3)
    # Only on FAILURES. On a success trajectory MetaWorld's reward also goes
    # flat during real progress — its tolerance terms saturate mid-transport —
    # so a flat reference there does not mean "nothing is happening", and the
    # model counting down through it is CORRECT. Measured +0.70/step on a
    # clean success, which would have read as a defect.
    false_progress = (float(np.mean(pred[stalled] - pred[stalled + 1]))
                      if success_step is None and len(stalled) >= 5 else None)
    return dict(
        mae=mae, false_progress=false_progress,
        stalled_frac=float(len(stalled) / max(len(d_mw), 1)),
        t2s_diff=t2s_diff,
        level_rho=spearman(mw_level, potential),
        diff_rho=spearman(mw_delta, t2s_diff[:end]),
        mw_stats=(float(mw_r.mean()), float(mw_r.std()), float(mw_r.min()), float(mw_r.max())),
        t2s_stats=(float(t2s_diff.mean()), float(t2s_diff.std()),
                   float(t2s_diff.min()), float(t2s_diff.max())),
        mw_return=float(mw_r.sum()), t2s_return=float(t2s_diff.sum()),
    )


def _add_corr_scatter(axis, x, y, rho, title, xlabel, ylabel):
    """Scatter with a least-squares line and the Spearman value."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    axis.scatter(x, y, s=18, alpha=0.65, color="tab:orange",
                 edgecolors="none", label="T2S samples")
    if len(x) >= 2 and np.std(x) > 0:
        coef = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        axis.plot(xx, coef[0] * xx + coef[1], color="black", lw=1.5,
                  label="linear fit")

    rho_text = (f"Spearman $\\rho$ = {rho:+.2f}" if rho is not None
                else "Spearman $\\rho$ = --")
    axis.text(0.04, 0.96, rho_text, transform=axis.transAxes,
              va="top", ha="left", fontsize=9,
              bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85,
                        edgecolor="0.8"))
    axis.set_title(title, fontsize=10)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, loc="best")


def _normalise_curve(values):
    """Map a curve to [0, 1] by its own min/max, for SHAPE comparison only.

    The correlations are computed on the RAW values; this is purely so two
    quantities on different scales can be overlaid. Note both curves then span
    the full 0-1 range by construction, so their endpoints carry no meaning.
    """
    values = np.asarray(values, dtype=float)
    lo, hi = np.nanmin(values), np.nanmax(values)
    span = hi - lo
    if span < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / span


def plot_one(name, obs, mw_r, pred, info, success_step, a, path,
             reward_mode="difference_timed", tp=1.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # BLUE = MetaWorld reference, ORANGE = T2S, GREEN = success
    MW_COLOR, T2S_COLOR, SUCCESS_COLOR = "tab:blue", "tab:orange", "tab:green"
    FORMULA = {"difference_timed": r"$r_t=\hat{T}_t-\gamma\hat{T}_{t+1}-c$",
               "difference": r"$r_t=\hat{T}_t-\gamma\hat{T}_{t+1}$",
               "absolute": r"$r_t=-\hat{T}_{t+1}$"}[reward_mode]

    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    t = np.arange(len(mw_r))

    def mark(axis):
        if success_step is not None:
            axis.axvline(success_step, color=SUCCESS_COLOR, ls="--", lw=1.2,
                         alpha=0.8, label=f"success @ {success_step}")

    end = success_step if success_step is not None else len(mw_r)
    end = max(min(end, len(mw_r)), 3)
    mw_level = mw_r[:end]
    t2s_potential = -pred[1:end + 1]

    # 1. LEVEL shape, normalised only so two scales can share an axis
    ax[0, 0].plot(np.arange(end), _normalise_curve(mw_level),
                  color=MW_COLOR, lw=2.0, label="MetaWorld reward")
    ax[0, 0].plot(np.arange(end), _normalise_curve(t2s_potential),
                  color=T2S_COLOR, lw=2.0, label="-T2S prediction")
    mark(ax[0, 0])
    ax[0, 0].set_title("LEVEL: task-progress shape", fontsize=10)
    ax[0, 0].set_ylabel("normalized level"); ax[0, 0].set_xlabel("step")
    ax[0, 0].set_ylim(-0.05, 1.05); ax[0, 0].grid(alpha=0.25)
    ax[0, 0].legend(fontsize=7, loc="best")
    ax[0, 0].text(0.02, 0.02, r"$V_{T2S}(s)=-\hat{T}(s)$",
                  transform=ax[0, 0].transAxes, fontsize=9,
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                            alpha=0.85, edgecolor="0.8"))

    # 2. LEVEL correlation
    _add_corr_scatter(ax[0, 1], mw_level, t2s_potential, a["level_rho"],
                      "LEVEL CORRELATION", "MetaWorld reward", "-T2S prediction")
    ax[0, 1].text(0.04, 0.08,
                  "Each point = one transition\ncloser to a monotonic diagonal "
                  "trend = more similar ordering",
                  transform=ax[0, 1].transAxes, fontsize=8,
                  bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                            alpha=0.75, edgecolor="0.8"))

    # 3. the per-step signals SAC would actually receive
    mw_delta = np.diff(np.concatenate([[mw_r[0]], mw_r]))
    ax[0, 2].plot(t, mw_delta, color=MW_COLOR, lw=1.5, alpha=0.9,
                  label=r"$\Delta$ MetaWorld")
    ax[0, 2].plot(t, a["t2s_diff"], color=T2S_COLOR, lw=1.5, alpha=0.9,
                  label=f"T2S {reward_mode}")
    ax[0, 2].axhline(0.0, color="0.5", lw=0.8)
    mark(ax[0, 2])
    ax[0, 2].set_title("SHAPED SIGNAL: per-step incentive", fontsize=10)
    ax[0, 2].set_ylabel("reward change / shaped reward")
    ax[0, 2].set_xlabel("step"); ax[0, 2].grid(alpha=0.25)
    ax[0, 2].legend(fontsize=7, loc="best")
    ax[0, 2].text(0.02, 0.02, FORMULA, transform=ax[0, 2].transAxes, fontsize=9,
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                            alpha=0.85, edgecolor="0.8"))

    # 4. shaped-signal correlation
    _add_corr_scatter(ax[1, 0], mw_delta[:end], a["t2s_diff"][:end], a["diff_rho"],
                      "SHAPED-SIGNAL CORRELATION", r"$\Delta$ MetaWorld reward",
                      f"T2S {reward_mode}")
    ax[1, 0].text(0.04, 0.08,
                  "Positive association means both rewards\ntend to increase on "
                  "the same useful transitions",
                  transform=ax[1, 0].transAxes, fontsize=8,
                  bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                            alpha=0.75, edgecolor="0.8"))

    # 5. MetaWorld's own stage components against the raw prediction
    if info and any(info):
        cc = {"near_object": "tab:purple", "grasp_success": "tab:red",
              "grasp_reward": "tab:brown", "in_place_reward": "tab:pink"}
        for k in MW_COMPONENTS:
            vals = [d.get(k) for d in info]
            if any(v is not None for v in vals):
                ax[1, 1].plot(t, [v if v is not None else np.nan for v in vals],
                              lw=1.4, color=cc[k], label=k)
        ax2 = ax[1, 1].twinx()
        ax2.plot(np.arange(len(pred)), pred, color=T2S_COLOR, lw=2.0, ls="--",
                 label="T2S prediction")
        ax2.set_ylabel("T2S predicted steps", color=T2S_COLOR)
        ax2.tick_params(axis="y", labelcolor=T2S_COLOR)
        ax[1, 1].legend(fontsize=6.5, loc="upper left")
        ax2.legend(fontsize=6.5, loc="upper right")
    mark(ax[1, 1])
    ax[1, 1].set_title("MetaWorld stages vs T2S prediction", fontsize=10)
    ax[1, 1].set_ylabel("MetaWorld component value")
    ax[1, 1].set_xlabel("step"); ax[1, 1].grid(alpha=0.25)

    # 6. cumulative shape. NOT comparable across trajectory kinds: references
    # stop 15 frames after success while failures run all 500 steps.
    ax[1, 2].plot(t, _normalise_curve(np.cumsum(mw_r)), color=MW_COLOR, lw=2.0,
                  label="MetaWorld cumulative")
    ax[1, 2].plot(t, _normalise_curve(np.cumsum(a["t2s_diff"])), color=T2S_COLOR,
                  lw=2.0, label="T2S cumulative")
    mark(ax[1, 2])
    ax[1, 2].set_title("CUMULATIVE SHAPE (normalised; endpoints carry no meaning)",
                       fontsize=10)
    ax[1, 2].set_ylabel("normalized cumulative reward")
    ax[1, 2].set_xlabel("step"); ax[1, 2].set_ylim(-0.05, 1.05)
    ax[1, 2].grid(alpha=0.25); ax[1, 2].legend(fontsize=7, loc="best")

    fig.suptitle(f"{name}\nBLUE = MetaWorld reference   |   ORANGE = T2S   |   "
                 "GREEN = success", fontsize=12)
    fig.text(0.5, 0.005,
             f"Level: V_T2S(s) = -T_hat(s)   |   Shaped reward: {reward_mode}"
             + (f" (c={tp})" if reward_mode == "difference_timed" else "")
             + "   |   Correlation metric: Spearman rho",
             ha="center", fontsize=9)
    plt.tight_layout(rect=(0, 0.035, 1, 0.92))
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)

    import config
    import results
    import t2s_io
    import t2s_predict
    from stable_baselines3 import SAC

    gamma = config.RL_GAMMA if args.gamma is None else args.gamma
    tp = config.TIME_PENALTY if args.time_penalty is None else args.time_penalty

    t2s_dir = results.get_run_dir("t2s_model", args.t2s_run)
    man = t2s_io.load_manifest(t2s_dir)
    missing = [m for m in args.models if m not in man["combos"]]
    if missing:
        raise SystemExit(f"ERROR: {missing} not in {args.t2s_run}; available: "
                         f"{sorted(man['combos'])}")
    fns = {m: t2s_predict.load_t2s_predictor(
        t2s_dir, *m.rsplit("_", 1), seed=man["combos"][m].get("best_seed", 0))
        for m in args.models}

    data_run = args.data_run
    if data_run is None:
        sm = os.path.join(t2s_dir, STAGE_MANIFEST)
        if os.path.exists(sm):
            data_run = json.load(open(sm)).get("data_run")
    if data_run is None:
        raise SystemExit("ERROR: pass --data-run; could not find it in the T2S run")
    ref_dir = os.path.join(results.get_run_dir("data_collection", data_run), "references")
    idx_path = os.path.join(ref_dir, "index.json")
    if not os.path.exists(idx_path):
        raise SystemExit(f"ERROR: no references in {ref_dir}. Re-collect with the "
                         "current data_collection (it writes them automatically).")
    index = json.load(open(idx_path))
    pols = {"success": SAC.load(os.path.join(config.EXPERT_POLICY_DIR,
                                             index["success_ckpt"])),
            "failure": SAC.load(os.path.join(config.EXPERT_POLICY_DIR,
                                             index["failure_ckpt"]))}

    out_root = args.out or os.path.join(ref_dir, "reward_compare")
    print(f"=== T2S vs MetaWorld reference reward — {len(fns)} model(s) from "
          f"{args.t2s_run} ===")
    print(f"    reward_mode={args.reward_mode}  gamma={gamma}"
          + (f"  time_penalty={tp}" if args.reward_mode == "difference_timed"
             else "  (time_penalty unused by this mode)") + "\n")

    # Re-roll each reference ONCE, then score every model on those frames.
    rows = []
    for kind in args.kinds:
        cands = [r for r in index["trajectories"] if r["kind"] == kind]
        if not cands:
            print(f"  {kind}: no stored reference")
            continue
        ref = cands[min(args.seed_index, len(cands) - 1)]
        obs, mw_r, info, ss = reroll(pols[ref["policy"]], ref["seed"],
                                     ref["perturb_steps"])
        stored = np.load(os.path.join(ref_dir, ref["name"] + ".npz"))["obs"]
        n = min(len(stored), len(obs))
        drift = float(np.abs(stored[:n] - obs[:n]).max()) if n else float("inf")
        match = drift < 1e-4 and len(stored) == len(obs)
        print(f"  re-rolled {ref['name']:<32} {len(obs):>4} frames  "
              + (f"success@{ss}" if ss is not None else "no success")
              + ("" if match else "   REPLAY DIVERGED"))

        for m, fn in fns.items():
            pred = np.array([fn(o) for o in obs], dtype=float)
            a = analyse(obs, mw_r, pred, ss, gamma, tp, args.reward_mode)
            out_dir = os.path.join(out_root, m)
            os.makedirs(out_dir, exist_ok=True)
            plot_one(f"{m}  |  {ref['name']}"
                     + ("" if match else "   (REPLAY DIVERGED)"),
                     obs, mw_r, pred, info, ss, a,
                     os.path.join(out_dir, f"{ref['name']}.png"),
                     reward_mode=args.reward_mode, tp=tp)
            rows.append(dict(model=m, kind=kind, name=ref["name"], success_step=ss,
                             match=match, drift=drift,
                             **{k: v for k, v in a.items() if k != "t2s_diff"}))

    f2 = lambda v: "--" if v is None else f"{v:+.2f}"
    f1 = lambda v: "--" if v is None else f"{v:.2f}"

    # ---- per trajectory --------------------------------------------------
    print(f"\n{'model':<22}{'trajectory':<24}{'MAE':>7}{'level rho':>11}"
          f"{'diff rho':>10}{'false prog':>12}")
    print("-" * 86)
    for r in rows:
        print(f"{r['model']:<22}{r['kind']:<24}{f1(r['mae']):>7}"
              f"{f2(r['level_rho']):>11}{f2(r['diff_rho']):>10}"
              f"{f2(r['false_progress']):>12}")

    # ---- the headline: one row per model, success vs failure split ------
    def mean_of(model, key, success):
        v = [r[key] for r in rows if r["model"] == model and r[key] is not None
             and ((r["success_step"] is not None) == success)]
        return float(np.mean(v)) if v else None

    print(f"\n{'model':<22}{'MAE':>7}{'level rho':>20}{'diff rho':>20}{'false prog':>12}")
    print(f"{'':<22}{'(succ)':>7}{'succ':>10}{'fail':>10}{'succ':>10}{'fail':>10}"
          f"{'(fail)':>12}")
    print("-" * 81)
    for m in fns:
        print(f"{m:<22}{f1(mean_of(m, 'mae', True)):>7}"
              f"{f2(mean_of(m, 'level_rho', True)):>10}"
              f"{f2(mean_of(m, 'level_rho', False)):>10}"
              f"{f2(mean_of(m, 'diff_rho', True)):>10}"
              f"{f2(mean_of(m, 'diff_rho', False)):>10}"
              f"{f2(mean_of(m, 'false_progress', False)):>12}")

    print("""
  Two questions, two metrics:

  MAE        does the model predict TIME correctly?  |prediction - true steps
             remaining|, over pre-success frames. Only defined on trajectories
             that SUCCEED, since a failure has no true time-to-success.

  level rho  is the SHAPING good?  Spearman(MetaWorld reward, -T2S prediction):
             do the two agree on how far along the task is?
  diff rho   Spearman(change in MetaWorld reward, T2S difference_timed): do
             their per-step shaping signals agree? Lower than level rho is
             normal — differencing amplifies noise.

  MetaWorld's reward is the REFERENCE, not ground truth: hand-designed shaping
  that SAC is known to learn from. Unlike the countdown, it exists on EVERY
  trajectory, which is why the fail columns exist at all.

  This is NOT the old Spearman. That one compared the prediction with the
  countdown, a straight line in time, so any decreasing curve scored ~1.
  MetaWorld's reward moves in STAGES (reach, grasp, place), so agreeing with
  it is not automatic and the number can discriminate between models.

  false prog how fast the prediction FALLS while MetaWorld's reward is flat,
             i.e. while nothing is happening. Want ~0. FAILURES ONLY: on a
             success, MetaWorld's reward also flattens during real progress,
             so a flat reference there does not mean the policy is stuck. Near 1.0 means the model
             counts down at full speed while the policy is stuck. This is the
             false-optimism defect measured directly — and it is needed
             because Spearman CANNOT see it: a stall is a plateau, a plateau
             is a run of tied values, and ties carry no rank information, so
             a model counting down through a stall still scores level rho
             ~0.7 on failures.

  High on succ, low on fail = the false-optimism defect: the model agrees
  with the reference where the expert goes and diverges where an untrained
  policy actually spends its time.""")

    # ---- scale -----------------------------------------------------------
    if rows:
        first = rows[0]["model"]
        mine = [r for r in rows if r["model"] == first]
        mw_std = np.mean([r["mw_stats"][1] for r in mine])
        print("\n=== per-step reward SCALE ===")
        print(f"  MetaWorld          std {mw_std:7.3f}")
        for m in fns:
            t2_std = np.mean([r["t2s_stats"][1] for r in rows if r["model"] == m])
            print(f"  {m:<18} std {t2_std:7.3f}   "
                  f"(MetaWorld / T2S = {mw_std / max(t2_std, 1e-9):.1f}x)")

    for r in rows:
        if not r["match"] and r["model"] == rows[0]["model"]:
            print(f"\n  WARNING: {r['name']} replay diverged (max obs diff "
                  f"{r['drift']:.2e}). The comparison uses the REPLAYED trajectory; "
                  "a divergence suggests a different MetaWorld/torch version.")

    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(dict(models=list(fns), t2s_run=args.t2s_run, gamma=gamma,
                       time_penalty=tp, rows=rows), f, indent=2)
    print(f"\nplots: {out_root}/<model>/")
    return 0


if __name__ == "__main__":
    sys.exit(main())