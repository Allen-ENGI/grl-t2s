#!/usr/bin/env python3
"""
Quantify WHY a T2S model that collapses to 0 at grasp still drives RL.

    python quantify_dataset.py --eval-run v9 --model tdlambdaboot_all \
        --sweep-name diff_time_v2 --rl-seed 0 \
        --ckpts 450000 500000 550000 600000 650000 \
        [--dataset-file path/to/t2s_train_trajs.pkl]

Without --dataset-file the expert success/failure policies are rolled out on
--proxy-seeds seeds as a stand-in for the T2S training data. Use the real file
if you can: the point is to measure THAT dataset.
Expected file format: pickle of a list of dicts {"obs": (T, D) array,
"success_step": int or None}. Edit load_dataset() if yours differs.

Assumes the Meta-World v2 obs layout: [0:3] hand, [3] gripper, [4:7] peg.

M1 counterexamples   failure trajectories that ever grasp / lift the peg
M2 error by phase    MAE pre-lift vs post-lift; zero-entry vs lift step
M3 coverage          RL states' NN distance to the data vs a held-out baseline
M4 pattern vs task   Spearman(pred, dist to success states) vs
                     Spearman(pred, peg->goal) and Spearman(pred, hand->peg)
M5 fixed scene       how typical the pinned peg start is among dataset starts
"""
import argparse
import json
import os
import pickle

import numpy as np
from stable_baselines3 import SAC

import core.config as config
import core.results as results
from core.progress import peg_goal_distance
import stage_2_t2s_training.t2s_eval as t2s_eval
from stage_4_rl_train.policy_env import make_policy_train_env
from stage_4_rl_train.run_4_policy import _load_predictors, STAGE_MANIFEST


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return None
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def nn_dist(Q, R, chunk=2048):
    R2 = (R ** 2).sum(1)
    out = np.empty(len(Q))
    for i in range(0, len(Q), chunk):
        q = Q[i:i + chunk]
        d2 = (q ** 2).sum(1)[:, None] + R2[None, :] - 2.0 * q @ R.T
        out[i:i + chunk] = np.sqrt(np.maximum(d2.min(1), 0.0))
    return out


def hand_peg(obs):
    return np.linalg.norm(obs[:, 0:3] - obs[:, 4:7], axis=1)


def load_dataset(arg):
    """Accepts a dataset.npz path, a data_collection run name (e.g. 'v4'), or a pickle."""
    path = arg
    if not os.path.isfile(path):
        try:
            path = os.path.join(results.get_run_dir("data_collection", arg), "dataset.npz")
        except Exception as e:
            raise SystemExit(f"--dataset-file {arg!r}: not a file and not a data_collection "
                             f"run ({e.__class__.__name__})")
    print(f"loading {path}")
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        ep, fr = z["episode_ids"], z["frame_idxs"]
        order = np.lexsort((fr, ep))
        X, y, ep = z["X"][order], z["y_steps"][order], ep[order]
        pos = z["position_ids"][order] if "position_ids" in z.files else None
        spl = z["split"][order] if "split" in z.files else None
        out = []
        for idx in np.split(np.arange(len(ep)), np.flatnonzero(np.diff(ep)) + 1):
            # labels are steps-remaining: the first frame labelled 0 IS the success state;
            # failed (censored) episodes never reach 0
            zeros = np.flatnonzero(y[idx] <= 0)
            out.append(dict(obs=X[idx].astype(np.float32),
                            success_step=int(zeros[0]) if len(zeros) else None,
                            position_id=int(pos[idx[0]]) if pos is not None else None,
                            split=str(spl[idx[0]]) if spl is not None else None))
        return out
    with open(path, "rb") as f:
        data = pickle.load(f)
    return [dict(obs=np.asarray(d["obs"], np.float32), success_step=d.get("success_step"))
            for d in data]


def proxy_dataset(stage3, seeds):
    succ = SAC.load(os.path.join(config.EXPERT_POLICY_DIR, stage3["success_ckpt"]))
    fail = SAC.load(os.path.join(config.EXPERT_POLICY_DIR, stage3["failure_ckpt"]))
    trajs = t2s_eval.collect_eval_trajectories(succ, fail, seeds=seeds)
    return [dict(obs=np.asarray(t["obs"], np.float32), success_step=t["success_step"])
            for ts in trajs.values() for t in ts]


def predict_all(predict, obs):
    return np.array([float(predict(o)) for o in obs])


def rollout(env_fn, policy, seed):
    env = env_fn()
    obs, _ = env.reset(seed=seed)
    O, P, s = [np.asarray(obs, np.float32)], [env._last_pred], None
    for t in range(500):
        obs, _r, term, trunc, info = env.step(policy.predict(obs, deterministic=False)[0])
        O.append(np.asarray(obs, np.float32))
        P.append(float(info["t2s_pred"]))
        if info.get(config.SUCCESS_KEY, 0) and s is None:
            s = t + 1
        if term or trunc:
            break
    env.close()
    return np.stack(O), np.array(P), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-run", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sweep-name", required=True)
    ap.add_argument("--rl-seed", type=int, default=0)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--dataset-file", default=None)
    ap.add_argument("--proxy-seeds", type=int, default=30)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--reward-mode", default="difference_timed")
    ap.add_argument("--gamma", type=float, default=config.RL_GAMMA)
    ap.add_argument("--time-penalty", type=float, default=config.TIME_PENALTY)
    ap.add_argument("--lift-thr", type=float, default=0.02, help="m above start = lifted")
    ap.add_argument("--grasp-thr", type=float, default=0.03, help="hand-peg m = grasp-near")
    ap.add_argument("--zero-eps", type=float, default=0.5)
    ap.add_argument("--max-ref", type=int, default=30_000)
    ap.add_argument("--m2-traj", type=int, default=300,
                    help="success trajectories scored in M2 (T2S is called per state)")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    with open(os.path.join(results.get_run_dir("t2s_eval", a.eval_run), STAGE_MANIFEST)) as f:
        stage3 = json.load(f)
    _s, preds = _load_predictors(stage3["t2s_run_dir"], [a.model], stage3)
    predict = preds[a.model]

    D = (load_dataset(a.dataset_file) if a.dataset_file
         else proxy_dataset(stage3, list(range(600, 600 + a.proxy_seeds))))
    src = a.dataset_file or f"PROXY: expert rollouts on {a.proxy_seeds} seeds"
    S = [d for d in D if d["success_step"] is not None]
    F = [d for d in D if d["success_step"] is None]
    print(f"dataset: {src}\n  {len(D)} trajectories ({len(S)} success, {len(F)} failure)")
    rep = dict(dataset=src, n_traj=len(D), n_success=len(S), n_failure=len(F))

    # ---- M1 counterexamples ------------------------------------------------
    def lifted(o):
        return (o[:, 6] - o[0, 6]) > a.lift_thr

    f_lift = [bool(lifted(d["obs"]).any()) for d in F]
    f_grasp = [bool((hand_peg(d["obs"]) < a.grasp_thr).any()) for d in F]
    n_lift_S = int(sum(lifted(d["obs"]).sum() for d in S))
    n_lift_F = int(sum(lifted(d["obs"]).sum() for d in F))
    rep["M1"] = dict(fail_ever_lifted=float(np.mean(f_lift)) if F else None,
                     fail_ever_grasp_near=float(np.mean(f_grasp)) if F else None,
                     lifted_states_success=n_lift_S, lifted_states_failure=n_lift_F)
    print(f"\n[M1] failures that ever lift the peg: {rep['M1']['fail_ever_lifted']}"
          f"   ever get hand near peg: {rep['M1']['fail_ever_grasp_near']}")
    if F and F[0].get("position_id") is not None:
        rep["M1"]["by_start"] = {}
        for kind, sel in (("fixed start", lambda d: d["position_id"] < 0),
                          ("random start", lambda d: d["position_id"] >= 0)):
            Fk = [d for d in F if sel(d)]
            Sk = [d for d in S if sel(d)]
            if not Fk:
                continue
            fl = float(np.mean([lifted(d["obs"]).any() for d in Fk]))
            fg = float(np.mean([(hand_peg(d["obs"]) < a.grasp_thr).any() for d in Fk]))
            rep["M1"]["by_start"][kind] = dict(success=len(Sk), failure=len(Fk),
                                               fail_ever_lifted=fl, fail_ever_grasp_near=fg)
            print(f"     {kind:<13} {len(Sk):>5}s/{len(Fk):>5}f   failures that lift {fl:.3f}"
                  f"   reach peg {fg:.3f}")
    if n_lift_F < 0.02 * max(n_lift_S, 1):
        m1_verdict = "NO counterexamples: lifted => success is all the data says"
    else:
        m1_verdict = "counterexamples present"
    print(f"     lifted states: {n_lift_S} from successes vs {n_lift_F} from failures"
          f"   -> {m1_verdict}")

    # ---- M2 error by phase -------------------------------------------------
    pre_err, post_err, gaps = [], [], []
    S_m2 = [S[i] for i in rng.permutation(len(S))[:a.m2_traj]]
    for d in S_m2:
        o, k = d["obs"], d["success_step"]
        p = predict_all(predict, o[:k + 1])
        ttg = (k - np.arange(k + 1)).astype(float)
        lf = lifted(o[:k + 1])
        pre_err += list(np.abs(p - ttg)[~lf])
        post_err += list(np.abs(p - ttg)[lf])
        z = np.argmax(p <= a.zero_eps) if (p <= a.zero_eps).any() else None
        l = np.argmax(lf) if lf.any() else None
        if z is not None and l is not None:
            gaps.append(dict(zero_entry=int(z), lift_step=int(l), success=int(k),
                             ttg_at_zero=int(k - z)))
    rep["M2"] = dict(mae_pre_lift=float(np.mean(pre_err)) if pre_err else None,
                     mae_post_lift=float(np.mean(post_err)) if post_err else None,
                     zero_minus_lift=float(np.mean([g["zero_entry"] - g["lift_step"]
                                                    for g in gaps])) if gaps else None,
                     true_steps_left_at_zero=float(np.mean([g["ttg_at_zero"] for g in gaps]))
                     if gaps else None)
    print(f"\n[M2] success trajectories: MAE pre-lift {rep['M2']['mae_pre_lift']}  "
          f"post-lift {rep['M2']['mae_post_lift']}")
    print(f"     pred hits 0 {rep['M2']['zero_minus_lift']} steps after lift, while "
          f"{rep['M2']['true_steps_left_at_zero']} steps actually remain")

    # ---- reference sets & normalisation -----------------------------------
    allD = np.concatenate([d["obs"] for d in D])
    mu, sd = allD.mean(0), allD.std(0) + 1e-6
    norm = lambda x: (x - mu) / sd
    idx = rng.permutation(len(D))
    hold = set(idx[: max(1, len(D) // 5)].tolist())
    trainD = np.concatenate([D[i]["obs"] for i in range(len(D)) if i not in hold])
    holdD = np.concatenate([D[i]["obs"] for i in hold])
    sub = lambda X, n: X[rng.choice(len(X), min(n, len(X)), replace=False)]
    ref_all = norm(sub(allD, a.max_ref))
    ref_succ = norm(sub(np.concatenate([d["obs"] for d in S]), a.max_ref))
    base = nn_dist(norm(sub(holdD, 5000)), norm(sub(trainD, a.max_ref)))
    thr = float(np.percentile(base, 95))
    predD_max = float(predict_all(predict, sub(allD, 5000)).max())
    print(f"\n[M3] in-distribution NN distance (held-out data): median "
          f"{np.median(base):.3f}, p95 {thr:.3f}; max pred on data {predD_max:.1f}")

    # ---- M5 fixed scene typicality ----------------------------------------
    env_fn = make_policy_train_env(predict, a.reward_mode, gamma=a.gamma,
                                   time_penalty=a.time_penalty)
    e = env_fn()
    o0, _ = e.reset(seed=0)
    e.close()
    starts = np.stack([d["obs"][0, 4:7] for d in D])
    dpeg = np.linalg.norm(starts - np.asarray(o0)[4:7], axis=1)
    spacing = np.median([np.sort(np.linalg.norm(starts - s, axis=1))[1] for s in starts])
    rep["M5"] = dict(nearest_start_m=float(dpeg.min()), median_start_spacing_m=float(spacing),
                     n_starts_within_2cm=int((dpeg < 0.02).sum()))
    print(f"\n[M5] pinned scene peg start: nearest dataset start {dpeg.min():.3f} m "
          f"(typical spacing {spacing:.3f} m), {rep['M5']['n_starts_within_2cm']} "
          f"dataset starts within 2 cm")

    # ---- M3 / M4 per checkpoint -------------------------------------------
    run_dir = results.get_run_dir("policy", f"{a.sweep_name}_{a.model}_seed{a.rl_seed}")
    print(f"\n{'ckpt':<10}{'succ':>6}{'OOD%':>7}{'pred>dataMax%':>15}"
          f"{'rho(pred,dSucc)':>17}{'rho(pred,peg->goal)':>21}{'rho(pred,hand->peg)':>21}")
    rep["per_ckpt"] = []
    for ck in a.ckpts:
        pol = SAC.load(os.path.join(run_dir, f"policy_{ck}.zip"))
        O, P, succ = [], [], []
        for i in range(a.episodes):
            o, p, s = rollout(env_fn, pol, 40_000 + i)
            O.append(o); P.append(p); succ.append(s is not None)
        O, P = np.concatenate(O), np.concatenate(P)
        keep = rng.choice(len(O), min(len(O), 5000), replace=False)
        O, P = O[keep], P[keep]
        d_all = nn_dist(norm(O), ref_all)
        d_succ = nn_dist(norm(O), ref_succ)
        pg = np.array([peg_goal_distance(o) for o in O])
        row = dict(ckpt=ck, success=float(np.mean(succ)),
                   ood_frac=float(np.mean(d_all > thr)),
                   pred_above_data_max=float(np.mean(P > predD_max)),
                   rho_pred_dist_success=spearman(P, d_succ),
                   rho_pred_peg_goal=spearman(P, pg),
                   rho_pred_hand_peg=spearman(P, hand_peg(O)),
                   mean_pred_in=float(P[d_all <= thr].mean()) if (d_all <= thr).any() else None,
                   mean_pred_ood=float(P[d_all > thr].mean()) if (d_all > thr).any() else None)
        rep["per_ckpt"].append(row)
        f = lambda v: "--" if v is None else f"{v:+.2f}"
        print(f"{ck:<10}{row['success']:>6.2f}{100 * row['ood_frac']:>6.0f}%"
              f"{100 * row['pred_above_data_max']:>14.0f}%{f(row['rho_pred_dist_success']):>17}"
              f"{f(row['rho_pred_peg_goal']):>21}{f(row['rho_pred_hand_peg']):>21}")

    out = os.path.join(run_dir, "quantify_dataset.json")
    with open(out, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()