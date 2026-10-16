#!/usr/bin/env python3
"""
Why does the saturated T2S model (v9) give better downstream RL than the accurate one (v10)?

Hypothesis: v9's prediction hits 0 around the grasp, so its reward is a strong
"get to the grasp" signal with a flat, never-misleading signal afterwards, while
v10 spreads a weaker and noisier signal over the whole episode.

  analyse   (no training) both rewards on the SAME expert successes, aligned at grasp:
            where the shaping reward is paid, how steep it is around the grasp, and
            whether it points the right way in each phase. Writes a figure.
  curves    (free) from existing eval_history.json: when does RL first move the peg
            (grasp learned) and first succeed, for each model's sweep?
  train     (RL) reward surgery:
              SUBGOAL_C     constant C until A's zero point, then 0   (grasp subgoal ONLY)
              B_ZEROAFTER   B's prediction, but 0 wherever A is 0     (B with A's flat end)
            SUBGOAL ~ v9  -> the grasp subgoal explains v9.
            B_ZEROAFTER >> v10 -> v10's post-grasp signal was what held it back.

    python why_v9_trains.py analyse --eval-run v9 \
        --a train_res/t2s_model/v9 tdlambdaboot_all softplus \
        --b train_res/t2s_model/v10 tdlambdaboot_all softplus --labels v9 v10

    python why_v9_trains.py curves --sweeps diff_time_v2_tdlambdaboot_all <v10 sweep prefix> \
        --labels v9 v10 --seeds 0 1 2

    python why_v9_trains.py train --eval-run v9 \
        --a train_res/t2s_model/v9 tdlambdaboot_all softplus \
        --b train_res/t2s_model/v10 tdlambdaboot_all softplus --seeds 0 1 2

`--a` is the saturated model (defines the zero point), `--b` the accurate one.
"""
import argparse
import json
import os
import warnings

import numpy as np

import core.config as config
import core.results as results
from core.config import HAND_POS_IDX, PEG_POS_IDX

GAMMA = config.RL_GAMMA
TP = config.TIME_PENALTY


# ----------------------------------------------------------------- loading
def load_predictor(run_dir, combo, head):
    from stage_2_t2s_training.t2s_model import load_t2s_predictor
    with open(os.path.join(run_dir, "manifest.json")) as f:
        seed = json.load(f)["combos"][combo].get("best_seed", 0)
    method, cond = combo.rsplit("_", 1)
    try:
        return load_t2s_predictor(run_dir, method, cond, seed=seed, head=head)
    except TypeError:        # loader predates the head argument -> softplus only
        if head != "softplus":
            raise
        return load_t2s_predictor(run_dir, method, cond, seed=seed)


def expert_successes(eval_run, n_seeds):
    from stable_baselines3 import SAC
    import stage_2_t2s_training.t2s_eval as t2s_eval
    with open(os.path.join(results.get_run_dir("t2s_eval", eval_run), "stage_manifest.json")) as f:
        st3 = json.load(f)
    succ = SAC.load(os.path.join(config.EXPERT_POLICY_DIR, st3["success_ckpt"]))
    fail = SAC.load(os.path.join(config.EXPERT_POLICY_DIR, st3["failure_ckpt"]))
    trajs = t2s_eval.collect_eval_trajectories(succ, fail, seeds=list(range(700, 700 + n_seeds)))
    out = [dict(obs=np.asarray(t["obs"], float), K=int(t["success_step"]))
           for ts in trajs.values() for t in ts if t["success_step"] is not None]
    if not out:
        raise SystemExit("no successful expert trajectories collected")
    return out


def grasp_step(obs):
    from stage_1_data_collection.data_collection import detect_control_onset
    g = detect_control_onset(obs)
    if g is None:                               # fallback: first step the peg is 1 cm up
        z = obs[:, PEG_POS_IDX][:, 2]
        up = np.flatnonzero(z - z[0] > 0.01)
        g = int(up[0]) if len(up) else None
    return g


# ----------------------------------------------------------------- analyse
def shaping(pred, K):
    """r_t = P_t - gamma * P_{t+1}, success state forced to 0 as in the RL wrapper."""
    p = np.array(pred[:K + 1], float)
    p[K] = 0.0
    return p, p[:-1] - GAMMA * p[1:]


def phase_stats(obs, r, g, K, thr=1e-3):
    hand, peg = obs[:, HAND_POS_IDX], obs[:, PEG_POS_IDX]
    from core.progress import peg_goal_distance
    hp = np.linalg.norm(hand - peg, axis=1)
    pg = np.array([peg_goal_distance(o) for o in obs])
    out = {}
    for name, lo, hi, prog in (("approach", 0, g, -np.diff(hp)), ("carry", g, K, -np.diff(pg))):
        if hi <= lo:
            continue
        rr, pp = r[lo:hi], prog[lo:hi]
        moved = pp > thr                         # steps that genuinely made progress
        out[name] = dict(
            steps=hi - lo,
            reward_sum=float(rr.sum()),
            reward_per_step=float(rr.mean()),
            progress_paid=float(np.mean(rr[moved] > 0)) if moved.any() else np.nan,
            progress_penalised=float(np.mean(rr[moved] < -0.05)) if moved.any() else np.nan,
            flat=float(np.mean(np.abs(rr) < 1e-3)))
    w = slice(max(g - 5, 0), min(g + 5, K))
    out["grasp_window_sum"] = float(r[w].sum())
    out["share_before_grasp"] = float(r[:g].sum() / r.sum()) if abs(r.sum()) > 1e-9 else np.nan
    return out


def cmd_analyse(a):
    trajs = expert_successes(a.eval_run, a.n_seeds)
    models = {lab: load_predictor(*spec) for lab, spec in zip(a.labels, (a.a, a.b))}
    rows = {lab: [] for lab in models}
    aligned = {lab: [] for lab in models}
    used = 0
    for tr in trajs:
        obs, K = tr["obs"], tr["K"]
        g = grasp_step(obs[:K + 1])
        if g is None or g < 3 or g > K - 3:
            continue
        used += 1
        for lab, fn in models.items():
            p, r = shaping([fn(o) for o in obs], K)
            rows[lab].append(phase_stats(obs[:K + 1], r, g, K))
            aligned[lab].append((np.arange(K + 1) - g, p, np.concatenate([[0], np.cumsum(r)])))
    print(f"{used} expert successes with a detected grasp (of {len(trajs)})\n")

    def agg(lab, *keys):
        v = [r for r in rows[lab]]
        for k in keys:
            v = [x.get(k, {}) if isinstance(x, dict) else np.nan for x in v]
        v = np.array([x if np.isscalar(x) else np.nan for x in v], float)
        return np.nanmean(v), 1.96 * np.nanstd(v) / np.sqrt(max(np.isfinite(v).sum(), 1))

    lines = [("share of shaping paid BEFORE grasp", ("share_before_grasp",), "{:.0%}"),
             ("shaping within +-5 steps of grasp", ("grasp_window_sum",), "{:+.1f}"),
             ("approach: shaping per step", ("approach", "reward_per_step"), "{:+.2f}"),
             ("approach: progress steps paid > 0", ("approach", "progress_paid"), "{:.0%}"),
             ("carry: shaping per step", ("carry", "reward_per_step"), "{:+.2f}"),
             ("carry: progress steps paid > 0", ("carry", "progress_paid"), "{:.0%}"),
             ("carry: progress steps PENALISED", ("carry", "progress_penalised"), "{:.0%}"),
             ("carry: flat steps (|r| < 1e-3)", ("carry", "flat"), "{:.0%}")]
    print(f"  {'':<38}" + "".join(f"{lab:>18}" for lab in models))
    report = {}
    for name, keys, fmt in lines:
        cells = []
        for lab in models:
            m, ci = agg(lab, *keys)
            report.setdefault(name, {})[lab] = dict(mean=float(m), ci95=float(ci))
            cells.append(f"{fmt.format(m)} +-{ci:.2g}".rjust(18))
        print(f"  {name:<38}" + "".join(cells))
    print("\n  (time penalty of -1 per step not included; it is identical for both rewards)")

    out = a.out or "why_v9_analyse"
    with open(out + ".json", "w") as f:
        json.dump(report, f, indent=2)
    plot_aligned(aligned, out + ".png")
    print(f"\nwrote {out}.json and {out}.png")


def plot_aligned(aligned, path, lo=-80, hi=60):
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # empty columns at the edges
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["tab:red", "tab:blue", "tab:green"]
    grid = np.arange(lo, hi + 1)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, (lab, trs) in enumerate(aligned.items()):
        for j, title in ((1, "prediction"), (2, "cumulative shaping reward")):
            M = np.full((len(trs), len(grid)), np.nan)
            for k, tr in enumerate(trs):
                x, y = tr[0], tr[j]
                m = (x >= lo) & (x <= hi)
                M[k, x[m] - lo] = y[m]
            mu = np.nanmean(M, 0)
            n = np.sum(np.isfinite(M), 0)
            ci = 1.96 * np.nanstd(M, 0) / np.sqrt(np.maximum(n, 1))
            ok = n >= 3
            ax[j - 1].plot(grid[ok], mu[ok], color=colors[i], label=lab)
            ax[j - 1].fill_between(grid[ok], (mu - ci)[ok], (mu + ci)[ok], color=colors[i], alpha=0.2)
            ax[j - 1].set_title(title)
    for A in ax:
        A.axvline(0, color="0.3", ls="--", lw=1)
        A.text(0, A.get_ylim()[1] * 0.95, " grasp", fontsize=9)
        A.set_xlabel("steps relative to grasp onset")
        A.grid(alpha=0.3)
        A.legend()
    fig.suptitle("expert successes aligned at the grasp (mean, 95% CI)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- curves
def cmd_curves(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["tab:red", "tab:blue", "tab:green", "tab:purple"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    print(f"  {'run':<44}{'first peg_moved>=0.5':>22}{'first success>0':>18}{'max success':>13}")
    for i, (prefix, lab) in enumerate(zip(a.sweeps, a.labels)):
        for sd in a.seeds:
            name = f"{prefix}_seed{sd}"
            try:
                rd = results.get_run_dir("policy", name)
                h = json.load(open(os.path.join(rd, "eval_history.json")))
            except (KeyError, FileNotFoundError):
                print(f"  {name:<44}  (not found)")
                continue
            st = np.array([e["step"] for e in h])
            pm = np.array([e.get("peg_moved_rate") or 0 for e in h], float)
            sr = np.array([e["success_rate"] for e in h], float)
            first = lambda m: f"{st[np.argmax(m)]:,}" if m.any() else "never"
            print(f"  {name:<44}{first(pm >= 0.5):>22}{first(sr > 0):>18}{sr.max():>13.2f}")
            kw = dict(color=colors[i % 4], alpha=0.8, label=lab if sd == a.seeds[0] else None)
            ax[0].plot(st, pm, **kw)
            ax[1].plot(st, sr, **kw)
    ax[0].set_title("peg moved (grasp learned)")
    ax[1].set_title("success rate")
    for A in ax:
        A.set_xlabel("RL step"); A.set_ylim(-0.05, 1.05); A.grid(alpha=0.3); A.legend()
    fig.tight_layout()
    out = a.out or "why_v9_curves.png"
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}  (each line is one RL seed)")


# ----------------------------------------------------------------- train
def cmd_train(a):
    import stage_4_rl_train.policy_train as policy_train
    A = load_predictor(*a.a)
    B = load_predictor(*a.b)
    eps, C = a.zero_eps, a.subgoal_c
    controls = {f"SUBGOAL_{C:g}": lambda o: C if A(o) > eps else 0.0,
                "B_ZEROAFTER": lambda o: B(o) if A(o) > eps else 0.0}
    for name in a.only or list(controls):
        for sd in a.seeds:
            run = f"{a.sweep_name}_{name}_seed{sd}"
            rd = results.new_run_dir("policy", run, meta=dict(
                control=name, a=a.a, b=a.b, zero_eps=eps, subgoal_c=C, rl_seed=sd,
                gamma=GAMMA, time_penalty=TP, reward_mode="difference_timed"))
            print(f"training {run}")
            policy_train.train_policy(
                rd, controls[name], reward_mode="difference_timed",
                total_timesteps=a.timesteps, seed=sd, gamma=GAMMA, time_penalty=TP,
                norm_reward=False, eval_freq=100_000, ckpt_freq=50_000, n_eval_episodes=20)
    print(f"\ncompare with:  python why_v9_trains.py curves --sweeps "
          f"<v9 prefix> <v10 prefix> {a.sweep_name}_SUBGOAL_{C:g} {a.sweep_name}_B_ZEROAFTER "
          f"--labels v9 v10 subgoal v10-flat-end")


# ----------------------------------------------------------------- cli
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    spec = dict(nargs=3, metavar=("RUN_DIR", "COMBO", "HEAD"))

    s = sub.add_parser("analyse")
    s.add_argument("--eval-run", required=True, help="t2s_eval run whose expert ckpts to roll")
    s.add_argument("--a", required=True, **spec, help="saturated model (e.g. v9)")
    s.add_argument("--b", required=True, **spec, help="accurate model (e.g. v10)")
    s.add_argument("--labels", nargs=2, default=["v9", "v10"])
    s.add_argument("--n-seeds", type=int, default=20)
    s.add_argument("--out", default=None)

    s = sub.add_parser("curves")
    s.add_argument("--sweeps", nargs="+", required=True,
                   help="run-name prefixes, i.e. everything before _seed<N>")
    s.add_argument("--labels", nargs="+", required=True)
    s.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    s.add_argument("--out", default=None)

    s = sub.add_parser("train")
    s.add_argument("--eval-run", default=None)
    s.add_argument("--a", required=True, **spec)
    s.add_argument("--b", required=True, **spec)
    s.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    s.add_argument("--sweep-name", default="surgery")
    s.add_argument("--only", nargs="*", default=None, help="SUBGOAL_<C> and/or B_ZEROAFTER")
    s.add_argument("--zero-eps", type=float, default=0.5)
    s.add_argument("--subgoal-c", type=float, default=60.0,
                   help="~ v9's prediction at the start of an episode")
    s.add_argument("--timesteps", type=int, default=1_000_000)

    a = p.parse_args()
    {"analyse": cmd_analyse, "curves": cmd_curves, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()