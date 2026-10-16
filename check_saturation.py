#!/usr/bin/env python3
"""
Is the T2S collapse-to-0 a dead Softplus region?

    # v9 (trained with NO peg-range limit)
    python check_saturation.py --t2s-run-dir train_res/t2s_model/v9 \
        --combo tdlambdaboot_all --dataset v4 --max-peg-distance -1

    # the range-limited model, for comparison (use its own run dir / combo / data)
    python check_saturation.py --t2s-run-dir train_res/t2s_model/<run> \
        --combo tdlambdaboot_all --dataset <data run> --max-peg-distance 0.06

Rebuilds the exact training rows (load_and_prepare with the SAME range limit the
model was trained with) and TD targets (build_targets, the model as its own target
net), then measures z = the input to the final Softplus:

  dead row     supervised, target >= 1, sigmoid(z) < 1e-4  (z < -9.2): loss is large
               but dL/dz ~ 0, so training cannot move it
  loss share   share of the total training MSE sitting in dead rows
  grad share   share of the total |dL/dz| coming from dead rows (should be ~0)
  steps left   raw steps-to-success of dead rows on successful episodes:
               if they cluster just before success, that IS the collapse region
"""
import argparse
import importlib
import os

import numpy as np
import torch

import core.config as config
import core.results as results
from stage_2_t2s_training.t2s_model import T2SModel
from stage_2_t2s_training.t2s_io import load_normalization, checkpoint_name
from stage_2_t2s_training.t2s_targets import build_targets


def resolve_dataset(arg):
    if os.path.isfile(arg):
        return arg
    return os.path.join(results.get_run_dir("data_collection", arg), "dataset.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t2s-run-dir", required=True)
    ap.add_argument("--combo", required=True, help="e.g. tdlambdaboot_all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", required=True, help="data run name (v4) or dataset.npz path")
    ap.add_argument("--max-peg-distance", type=float, default=-1,
                    help="the value the model was TRAINED with; -1 = no limit")
    ap.add_argument("--subsample", type=float, default=None,
                    help="the SUBSAMPLE_EPISODES fraction the model was trained with, if any")
    ap.add_argument("--gamma", type=float, default=config.T2S_BOOTSTRAP_GAMMA)
    ap.add_argument("--lam", type=float, default=0.9)
    ap.add_argument("--head", default="softplus", choices=["softplus", "linear"],
                    help="softplus for v9/v10 and older runs; linear for runs trained "
                         "with the fixed T2SModel")
    ap.add_argument("--alias", action="store_true",
                    help="nearest-neighbour check: do near-success states look like failures?")
    ap.add_argument("--label", default=None, help="name used in plots (default: run dir name)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--train-module", default="stage_2_t2s_training.t2s_train",
                    help="module that defines load_and_prepare")
    a = ap.parse_args()

    tm = importlib.import_module(a.train_module)
    tm.MAX_PEG_DISTANCE = None if a.max_peg_distance < 0 else a.max_peg_distance
    if a.subsample is not None:
        if not hasattr(tm, "SUBSAMPLE_EPISODES"):
            raise SystemExit("--subsample given, but the training module has no "
                             "SUBSAMPLE_EPISODES constant; add it to load_and_prepare first")
        tm.SUBSAMPLE_EPISODES = a.subsample

    ck_method, condition = a.combo.rsplit("_", 1)
    boot = ck_method.endswith("boot")
    method = ck_method[:-4] if boot else ck_method
    gamma = a.gamma if boot else 1.0

    prep = tm.load_and_prepare(resolve_dataset(a.dataset), censor_label=config.CENSOR_LABEL,
                               censor_bootstrap=boot)
    # use the normalisation the model was ACTUALLY trained with
    x_mean, x_std = load_normalization(a.t2s_run_dir)
    norm = lambda X: ((X - x_mean) / x_std).astype(np.float32)
    Xn = norm(prep["X_all"])
    next_raw = prep["next_obs_n_all"] * prep["x_std"] + prep["x_mean"]
    next_n = norm(next_raw)

    try:
        model = T2SModel(obs_dim=config.OBS_DIM, head=a.head)
    except TypeError:                     # T2SModel predates the head argument
        if a.head != "softplus":
            raise
        model = T2SModel(obs_dim=config.OBS_DIM)
    model.load_state_dict(torch.load(
        os.path.join(a.t2s_run_dir, checkpoint_name(ck_method, condition, a.seed)),
        map_location="cpu"))
    model.eval()

    rows = prep["data_conditions"][condition]["rows_tr"]
    out = build_targets(method, lambda o: model.bootstrap_values(o, "cpu"), rows,
                        prep["y_all"], next_n, prep["is_terminal_all"],
                        prep["terminal_value_all"], prep["episode_ids_all"],
                        prep["frame_idxs_all"], gamma=gamma, lam=a.lam,
                        clip_max=config.CENSOR_LABEL * 1.2,
                        is_truncated_all=prep["is_truncated_all"] if boot else None)
    target, sup = out if isinstance(out, tuple) else (out, np.ones(len(rows), bool))

    with torch.no_grad():
        xb = torch.as_tensor(Xn[rows])
        z = torch.cat([model.t2s_head[0](model.encoder(xb[i:i + 65536]))
                       for i in range(0, len(xb), 65536)]).squeeze(-1).numpy()
    if a.head == "softplus":
        pred = np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)  # stable softplus
        sig = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))       # d softplus / dz
    else:
        pred = z                     # what the loss sees during training
        sig = np.ones_like(z)        # linear head: gradient never vanishes
    loss = (pred - target) ** 2
    grad = np.abs(2 * (pred - target) * sig)

    failed = prep["is_failed_episode"][rows]
    y_raw = prep["y_all"][rows]
    dead = sup & (target >= 1.0) & (sig < 1e-4)

    print(f"\n=== {a.combo}  ({a.t2s_run_dir}, max_peg_distance={tm.MAX_PEG_DISTANCE}) ===")
    print(f"training rows: {len(rows):,} supervised {sup.sum():,} "
          f"(success-episode {(~failed & sup).sum():,}, failure-episode {(failed & sup).sum():,})")
    for name, m in (("success-episode rows", ~failed & sup), ("failure-episode rows", failed & sup)):
        if m.any():
            q = np.percentile(z[m], [1, 5, 25, 50, 75, 95])
            print(f"  z on {name:<22} p1 {q[0]:+7.1f}  p5 {q[1]:+7.1f}  p25 {q[2]:+7.1f}  "
                  f"median {q[3]:+7.1f}  p95 {q[5]:+7.1f}")
    tot_loss, tot_grad = loss[sup].sum(), grad[sup].sum()
    print(f"\n  DEAD rows (target>=1, softplus grad<1e-4): {dead.sum():,} = "
          f"{dead.sum() / max(sup.sum(), 1):.1%} of supervised rows")
    for name, m in (("success", ~failed), ("failure", failed)):
        d = dead & m
        print(f"    {name:<8} {d.sum():>8,} dead ({d.sum() / max((m & sup).sum(), 1):.1%} "
              f"of that set), mean target {target[d].mean() if d.any() else float('nan'):.1f}")
    print(f"  share of training LOSS in dead rows:     {loss[dead].sum() / tot_loss:.1%}")
    print(f"  share of training GRADIENT in dead rows: {grad[dead].sum() / tot_grad:.2e}")

    ds = dead & ~failed
    if ds.any():
        st = y_raw[ds]
        print(f"\n  dead success rows, raw steps-to-success: median {np.median(st):.0f}  "
              f"p10 {np.percentile(st, 10):.0f}  p90 {np.percentile(st, 90):.0f}")

    # per-band table, ALWAYS printed. true = discounted value of the real steps left;
    # target = the TD(lambda) target the loss regresses onto; pred = what the loss sees.
    # pred >> target  : the FIT is biased (selection / imbalance / aliasing)
    # target >> true  : the TARGETS are biased (bootstrapping inherits the inflation)
    to_val = ((lambda n: (1 - gamma ** n) / (1 - gamma)) if boot else (lambda n: n))
    print(f"\n  {'steps to success':<18}{'rows':>8}{'true':>9}{'target':>9}{'pred':>9}"
          f"{'dead':>8}")
    for lo, hi in ((0, 1), (1, 5), (5, 10), (10, 20), (20, 30), (30, 40), (40, 60),
                   (60, 100), (100, 10**6)):
        band = sup & ~failed & (y_raw >= lo) & (y_raw < hi)
        if band.any():
            lab = "0 (success frame)" if hi == 1 else f"{lo}-{hi if hi < 10**6 else 'inf'}"
            print(f"  {lab:<18}{band.sum():>8,}{to_val(y_raw[band].astype(float)).mean():>9.2f}"
                  f"{target[band].mean():>9.2f}{pred[band].mean():>9.2f}"
                  f"{np.mean(dead[band]):>8.1%}")

    # ---- monotonicity along successful episodes: what the SHAPING actually sees ----
    # reward ~ pred(t) - gamma*pred(t+1) - tp, so a step where pred RISES while the
    # task is genuinely progressing is a step where correct behaviour is PENALISED.
    ep_r, fr_r = prep["episode_ids_all"][rows], prep["frame_idxs_all"][rows]
    order = np.lexsort((fr_r, ep_r))
    i0, i1 = order[:-1], order[1:]
    pair = ((ep_r[i0] == ep_r[i1]) & (fr_r[i1] == fr_r[i0] + 1)
            & ~failed[i0] & sup[i0] & sup[i1] & (y_raw[i0] >= 1))
    dpred = pred[i1] - pred[i0]
    shaping = pred[i0] - gamma * pred[i1]
    print(f"\n  monotonicity on successful episodes (consecutive frames):")
    print(f"  {'steps to success':<18}{'pairs':>8}{'pred rises':>12}{'mean dpred':>12}"
          f"{'shaping<0':>11}")
    for lo, hi in ((1, 5), (5, 10), (10, 20), (20, 40), (40, 100), (100, 10**6)):
        m = pair & (y_raw[i0] >= lo) & (y_raw[i0] < hi)
        if m.any():
            lab = f"{lo}-{hi if hi < 10**6 else 'inf'}"
            print(f"  {lab:<18}{m.sum():>8,}{np.mean(dpred[m] > 0):>12.1%}"
                  f"{dpred[m].mean():>+12.2f}{np.mean(shaping[m] < 0):>11.1%}")
    # per-episode: does the prediction ever climb by a lot in the last 40 steps?
    climbs = []
    for e in np.unique(ep_r[~failed & sup]):
        ix = order[ep_r[order] == e]
        ix = ix[sup[ix] & (y_raw[ix] >= 1) & (y_raw[ix] <= 40)]
        if len(ix) > 2:
            p_ = pred[ix]
            climbs.append(float(np.max(p_ - np.minimum.accumulate(p_))))
    if climbs:
        c = np.array(climbs)
        print(f"  largest climb within the last 40 steps, per episode: median {np.median(c):.1f}"
              f"  p90 {np.percentile(c, 90):.1f}  ({np.mean(c > 5):.0%} of episodes climb > 5)")

    if a.alias:
        # are near-success states closer to FAILURE states than to other successes?
        rng = np.random.default_rng(0)
        ep = prep["episode_ids_all"][rows]
        succ_eps = np.unique(ep[~failed])
        half = set(rng.permutation(succ_eps)[: len(succ_eps) // 2].tolist())
        inA = np.array([e in half for e in ep])
        near = np.flatnonzero(sup & ~failed & inA & (y_raw >= 1) & (y_raw <= 20))
        refS = np.flatnonzero(sup & ~failed & ~inA)
        refF = np.flatnonzero(sup & failed)
        pick = lambda ix, n: ix[rng.permutation(len(ix))[:n]]
        q, rS, rF = pick(near, 2000), pick(refS, 50_000), pick(refF, 50_000)
        X = Xn[rows]

        def nn(Q, R, chunk=1024):
            R2 = (R ** 2).sum(1)
            d, j = np.empty(len(Q)), np.empty(len(Q), int)
            for i in range(0, len(Q), chunk):
                qq = Q[i:i + chunk]
                d2 = (qq ** 2).sum(1)[:, None] + R2[None] - 2 * qq @ R.T
                j[i:i + chunk] = d2.argmin(1)
                d[i:i + chunk] = np.sqrt(np.maximum(d2.min(1), 0))
            return d, j

        dS, _ = nn(X[q], X[rS])
        dF, jF = nn(X[q], X[rF])
        closerF = dF < dS
        print(f"\n  ALIAS check on {len(q)} success states 1-20 steps from success:")
        print(f"    nearest FAILURE state closer than nearest other-episode SUCCESS: "
              f"{closerF.mean():.1%}")
        print(f"    median distance to success {np.median(dS):.3f}, to failure {np.median(dF):.3f}")
        print(f"    TD target of those nearest failure states: mean "
              f"{target[rF[jF]].mean():.1f}  (vs the queries' own target "
              f"{target[q].mean():.1f})")

    if a.head == "linear":
        succ_sup = sup & ~failed & (y_raw > 0)
        print(f"\n  linear head: success rows with raw prediction < 0: "
              f"{np.mean(z[succ_sup] < 0):.1%}  (clamped to 0 when used; still trainable)")
    print("\nverdict: " + (
        "SATURATED — a block of near-success states sits in a dead Softplus region"
        if ds.sum() > 0.05 * max((~failed & sup).sum(), 1)
        else "no meaningful saturation on success rows"))

    if not a.no_plot:
        label = a.label or os.path.basename(os.path.normpath(a.t2s_run_dir))
        succ = sup & ~failed
        steps = y_raw[succ].astype(int)
        smax = int(min(150, steps.max())) if len(steps) else 0
        cnt = np.bincount(steps, minlength=smax + 1)[:smax + 1]
        sel = steps <= smax
        mp = np.bincount(steps[sel], weights=pred[succ][sel], minlength=smax + 1) / np.maximum(cnt, 1)
        mt = np.bincount(steps[sel], weights=target[succ][sel], minlength=smax + 1) / np.maximum(cnt, 1)
        s_axis = np.arange(smax + 1)
        rng_p = np.random.default_rng(0)
        pick = rng_p.permutation(int(succ.sum()))[:6000]
        bands = [(1, 10), (10, 20), (20, 30), (30, 40), (40, 60), (60, 100), (100, 10**6)]
        # dead share per band, computed PER EPISODE so the error bar reflects
        # episode-to-episode variation (states within one episode are not independent)
        ep_rows = prep["episode_ids_all"][rows]
        band_dead, band_ci, band_neps = [], [], []
        for lo, hi in bands:
            m = succ & (y_raw >= lo) & (y_raw < hi)
            if not m.any():
                band_dead.append(np.nan); band_ci.append(np.nan); band_neps.append(0)
                continue
            _, inv = np.unique(ep_rows[m], return_inverse=True)
            per_ep = np.bincount(inv, weights=dead[m]) / np.bincount(inv)
            n = len(per_ep)
            band_dead.append(float(per_ep.mean()))
            band_ci.append(float(1.96 * per_ep.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan)
            band_neps.append(n)
        arrays = dict(
            label=label, head=a.head, gamma=gamma,
            z_succ=z[succ][pick], steps_succ=y_raw[succ][pick].astype(float),
            dead_succ=dead[succ][pick],
            curve_steps=s_axis, curve_pred=mp, curve_target=mt,
            curve_true=to_val(s_axis.astype(float)), curve_count=cnt,
            band_labels=np.array([f"{lo}-{hi if hi < 10**6 else 'inf'}" for lo, hi in bands]),
            band_dead=np.array(band_dead), band_ci=np.array(band_ci),
            band_neps=np.array(band_neps),
            loss_share=float(loss[dead].sum() / max(tot_loss, 1e-12)),
            grad_share=float(grad[dead].sum() / max(tot_grad, 1e-12)))
        base = os.path.join(a.t2s_run_dir, f"saturation_{a.combo}")
        np.savez(base + ".npz", **arrays)
        plot_single(arrays, base + ".png")
        print(f"\nwrote {base}.png and {base}.npz")
        print("  compare two models:  python check_saturation.py --compare "
              "<v9 npz> <v10 npz> --out saturation_compare.png")


# ============================================================================ plots
COLORS = ["tab:red", "tab:blue", "tab:green", "tab:purple"]


def _dead_bars(ax, ds, labels, legend=True, stats=False):
    """Grouped bars: % of success-episode states with gradient multiplier < 1e-4,
    by true steps to success, mean over episodes with a 95% CI (1.96 x SE)."""
    bl = [str(x) for x in ds[0]["band_labels"]]
    w = 0.8 / len(ds)
    for i, (d, lab) in enumerate(zip(ds, labels)):
        xs = np.arange(len(bl)) + (i - (len(ds) - 1) / 2) * w
        mean = 100 * np.nan_to_num(np.asarray(d["band_dead"], float))
        ci = (100 * np.nan_to_num(np.asarray(d["band_ci"], float))
              if "band_ci" in d else None)
        lower = None if ci is None else np.minimum(ci, mean)       # bars cannot go below 0
        ax.bar(xs, mean, width=w, color=COLORS[i % 4],
               yerr=None if ci is None else [lower, ci], capsize=3,
               error_kw=dict(lw=1, ecolor="black"),
               label=(f"{lab}: {100 * float(d['loss_share']):.1f}% of loss, "
                      f"{float(d['grad_share']):.0e} of gradient") if stats else lab)
        top = mean if ci is None else mean + ci
        for x, t_, m_ in zip(xs, top, mean):
            ax.text(x, t_ + 1.5, f"{m_:.0f}%", ha="center", va="bottom", fontsize=8,
                    color=COLORS[i % 4])
    # episode counts differ between models, so only label them on single-model plots
    neps = ds[0].get("band_neps") if len(ds) == 1 else None
    ticks = bl if neps is None else [f"{b}\n(n={int(n)} eps)" for b, n in zip(bl, neps)]
    ax.set_xticks(np.arange(len(bl)))
    ax.set_xticklabels(ticks, fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_xlabel("true steps to success")
    ax.set_ylabel("% of states with ≈ 0 gradient\n(Softplus slope < 1e-4)")
    if legend:
        ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.3, axis="y")


def plot_single(d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    _dead_bars(ax, [d], [str(d["label"])])
    ax.set_title(f"{d['label']}: states that can no longer learn, by distance to success\n"
                 "(mean over successful episodes, error bars = 95% CI)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_compare(paths, labels, out, legend=True, stats=False, title=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = [dict(np.load(p, allow_pickle=True)) for p in paths]
    labels = labels or [str(d["label"]) for d in ds]
    fig, ax = plt.subplots(figsize=(10, 5))
    _dead_bars(ax, ds, labels, legend=legend, stats=stats)
    if title:
        ax.set_title("States that can no longer learn (Softplus gradient < 1e-4), "
                     "by distance to success\n(mean over successful episodes, "
                     "error bars = 95% CI)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def compare_main(argv):
    ap = argparse.ArgumentParser(description="overlay saturation plots of several models")
    ap.add_argument("--compare", nargs="+", required=True, help="saturation_*.npz files")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default="saturation_compare.png")
    ap.add_argument("--no-legend", action="store_true", help="drop the legend box")
    ap.add_argument("--legend-stats", action="store_true",
                    help="put the loss / gradient shares back into the legend")
    ap.add_argument("--no-title", action="store_true", help="leave the title to the slide")
    a = ap.parse_args(argv)
    plot_compare(a.compare, a.labels, a.out, legend=not a.no_legend, stats=a.legend_stats,
                 title=not a.no_title)


if __name__ == "__main__":
    import sys
    if "--compare" in sys.argv:
        compare_main(sys.argv[1:])
    else:
        main()