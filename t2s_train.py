"""
Stage 3: T2S model training (formulations: MC, TD(0), TD(lambda)).

Ported from Compare_value_targets_v3.ipynb, decoupled into:
  - t2s_targets.py   target construction (already unit-tested standalone)
  - t2s_model.py     the network
  - t2s_io.py        normalization/checkpoint/manifest contract
  - this module      the training loop + multi-seed orchestration + split/normalize

Import torch lazily-at-module-level is fine here since this module is only
ever used where torch is installed (unlike t2s_targets/data_collection which
are also imported by lightweight unit tests).

THE TRAIN/VAL SPLIT IS NO LONGER DRAWN HERE. load_and_prepare used to call
GroupShuffleSplit(random_state=seed) on whatever pool it was handed. That
grouped by episode id rather than trajectory CONTENT, so the dataset's
bit-identical duplicates (222 episodes -> 93 unique) landed on both sides and
val MSE partly measured memorization; and it was re-drawn per call, so no two
runs validated on the same states. The split is now assigned once at
collection time and stored in dataset.npz; this module reads it. See
data_collection.assign_splits.
"""
import copy
import os
import numpy as np
import torch
import torch.nn.functional as F

from config import CENSOR_LABEL, OBS_DIM, T2S_BOOTSTRAP_GAMMA
from t2s_model import T2SModel
from t2s_io import save_normalization, save_manifest, checkpoint_name
from t2s_targets import build_targets, build_bookkeeping

# ---- tuning constants: edit here, not via arguments --------------------
# train_one used to take 15 parameters. Nine of them were these, threaded
# through run_all_combos and a CLI flag each, none of them a decision anyone
# made per run.
EPOCHS = 150                 # val MSE plateaus well before 400
PATIENCE = 40
TARGET_REFRESH = 5           # epochs between target-network refreshes
LAMBDA = 0.9                 # TD(lambda) trace
BATCH_SIZE = 1024            # 256 underuses the GPU
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
CONFIRM_BUFFER = 3          # frames kept after success when trimming the tail
CENSOR_GAMMA = 1.0           # censor scheme only; bootstrap forces < 1
SEED = 0                     # one seed: see run_all_combos for why not a sweep

REL_WEIGHT = 1.0
REL_WEIGHT = 0.0
MAX_GAP = 30

# the grid IS the experiment: three value targets x two data sources x
# bootstrap-or-not = 12 models
METHODS = ("mc", "td0", "tdlambda")
CONDITIONS = ("succ", "all")
CENSOR_SCHEMES = ("censor", "bootstrap")



MAX_PEG_DISTANCE = 0.08   # metres from the fixed start; None = use every random position

def load_and_prepare(dataset_path, censor_label=CENSOR_LABEL, confirm_buffer=CONFIRM_BUFFER,
                     censor_bootstrap=False):
    """
    Loads dataset.npz, trims the long flat poFst-success tail, reads the STORED
    train/val split, and returns everything build_targets/train_one need.

    Requires the `split` array written by the current data_collection. A
    dataset without it predates collection-time splitting and is rejected
    rather than silently re-split here, which is the behaviour that let two
    runs disagree about what "validation" meant.
    """
    raw = np.load(dataset_path)
    X_all, y_all = raw["X"], raw["y_steps"]
    episode_ids_all, frame_idxs_all = raw["episode_ids"], raw["frame_idxs"]
    if "split" not in raw.files:
        raise KeyError(
            f"{dataset_path} has no 'split' array (found {list(raw.files)}). Re-collect "
            "with the current data_collection, which assigns train/val at collection "
            "time — see its module docstring for why the split moved out of this file.")
    split_all = raw["split"].astype(str)
    
    pos_ids = raw["position_ids"] if "position_ids" in raw.files else np.full(len(y_all), -1)
    if MAX_PEG_DISTANCE is not None and (pos_ids >= 0).any():
        import json
        peg = json.load(open(os.path.join(os.path.dirname(dataset_path),
                                          "dataset_summary.json")))["peg_starts"]
        pool = np.array([[p["x"], p["y"]] for p in peg["positions"]])
        near = np.linalg.norm(pool - np.array(peg["fixed_start"][:2]), axis=1) <= MAX_PEG_DISTANCE
        rows = (pos_ids < 0) | near[np.clip(pos_ids, 0, None)]
        X_all, y_all = X_all[rows], y_all[rows]
        episode_ids_all, frame_idxs_all = episode_ids_all[rows], frame_idxs_all[rows]
        split_all = split_all[rows]
        print(f"  kept fixed start + {int(near.sum())} of {len(pool)} random positions "
              f"within {100 * MAX_PEG_DISTANCE:.0f} cm ({rows.mean():.0%} of rows)")

    # is_failed_episode = np.array([
    #     np.all(y_all[episode_ids_all == ep] == censor_label) for ep in episode_ids_all
    # ])
    
    _, inv = np.unique(episode_ids_all, return_inverse=True)
    is_failed_episode = (np.bincount(inv, weights=(y_all != censor_label)) == 0)[inv]
    

    # trim tail: keep failed episodes whole, keep successful episodes only up
    # to confirm_buffer frames past the first zero-T2S frame
    keep = np.zeros(len(y_all), dtype=bool)
    for ep in np.unique(episode_ids_all):
        idx = np.flatnonzero(episode_ids_all == ep)
        if is_failed_episode[idx[0]]:
            keep[idx] = True
            continue
        zero_pos = np.flatnonzero(y_all[idx] == 0.0)
        if len(zero_pos) == 0:
            keep[idx] = True
            continue
        end_pos = min(zero_pos[0] + confirm_buffer, len(idx) - 1)
        keep[idx[:end_pos + 1]] = True

    X_all, y_all = X_all[keep], y_all[keep]
    episode_ids_all, frame_idxs_all = episode_ids_all[keep], frame_idxs_all[keep]
    is_failed_episode = is_failed_episode[keep]
    split_all = split_all[keep]

    next_obs_all, is_terminal_all, terminal_value_all, is_truncated_all = build_bookkeeping(
        X_all, y_all, episode_ids_all, frame_idxs_all, censor_label,
        censor_bootstrap=censor_bootstrap)

    tr_all = np.flatnonzero(split_all == "train")
    va_all = np.flatnonzero(split_all == "val")
    if len(va_all) == 0:
        raise ValueError(
            f"{dataset_path} has no rows in the 'val' split; re-collect with "
            "val_fraction > 0.")

    succ_mask = ~is_failed_episode
    tr_succ = tr_all[succ_mask[tr_all]]
    va_succ = va_all[succ_mask[va_all]]

    x_mean = X_all[tr_succ].mean(axis=0)
    x_std = X_all[tr_succ].std(axis=0)
    x_std = np.where(x_std < 1e-4, 1.0, x_std)

    def normalize(X):
        return ((X - x_mean) / x_std).astype(np.float32)

    return dict(
        X_all=X_all, y_all=y_all, split_all=split_all, next_obs_n_all=normalize(next_obs_all),
        Xn_all=normalize(X_all), is_terminal_all=is_terminal_all,
        terminal_value_all=terminal_value_all, is_truncated_all=is_truncated_all,
        censor_bootstrap=censor_bootstrap, episode_ids_all=episode_ids_all,
        frame_idxs_all=frame_idxs_all, is_failed_episode=is_failed_episode,
        x_mean=x_mean, x_std=x_std, normalize=normalize,
        data_conditions={
            "succ": dict(rows_tr=tr_succ, rows_va=va_succ),
            "all": dict(rows_tr=tr_all, rows_va=va_all),
        },
    )


def train_one(prepared, method, condition, run_dir, seed=0, gamma=CENSOR_GAMMA,
              target_clip=None, device="cpu", censor_bootstrap=None):
    """
    Train one (method, condition) model. Tuning lives in the constants above;
    only the things that vary per combo are arguments.
    """
    obs_dim, epochs, patience = OBS_DIM, EPOCHS, PATIENCE
    target_refresh, lam, batch_size = TARGET_REFRESH, LAMBDA, BATCH_SIZE

    if censor_bootstrap is None:
        censor_bootstrap = prepared.get("censor_bootstrap", False)
    if censor_bootstrap and gamma >= 1.0:
        raise ValueError(
            f"censor_bootstrap=True requires gamma < 1 (got {gamma}). With gamma=1 the "
            "recursion V = dt + V has no finite fixed point, so values on trajectories "
            "that never succeed drift upward every target refresh until clip_max catches "
            "them — reinstating an arbitrary ceiling. Use gamma=0.99 (fixed point 100) "
            "or 0.95 (fixed point 20).")

    torch.manual_seed(seed)
    np.random.seed(seed)

    rows_tr = prepared["data_conditions"][condition]["rows_tr"]
    rows_va = prepared["data_conditions"][condition]["rows_va"]
    is_failed = prepared["is_failed_episode"]
    rows_va_succ = rows_va[~is_failed[rows_va]]

    # bootstrap variants are a DIFFERENT model and must not overwrite the
    # censor variant's file — checkpoint_name is keyed on method, so tag it
    ckpt_method = f"{method}boot" if censor_bootstrap else method
    ckpt_path = os.path.join(run_dir, checkpoint_name(ckpt_method, condition, seed))

    model = T2SModel(obs_dim=obs_dim).to(device)
    target_net = copy.deepcopy(model)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    Xn_all, y_all = prepared["Xn_all"], prepared["y_all"]
    Xn_tr = Xn_all[rows_tr]
    Xn_va, y_va = Xn_all[rows_va], y_all[rows_va]
    Xn_va_succ, y_va_succ = Xn_all[rows_va_succ], y_all[rows_va_succ]

    Xn_tr_t = torch.as_tensor(Xn_tr, dtype=torch.float32, device=device)
    Xn_va_t = torch.as_tensor(Xn_va, dtype=torch.float32, device=device)
    Xn_va_succ_t = torch.as_tensor(Xn_va_succ, dtype=torch.float32, device=device)
    
    
    # Add new changes 
    # relative-change loss: pairs from the same successful training episode
    # ep_ids = prepared["episode_ids_all"]
    # succ_tr = rows_tr[~is_failed[rows_tr]]
    
    to_value = ((lambda n: (1 - gamma ** n) / (1 - gamma)) if censor_bootstrap
                else (lambda n: n))
    # X_all_t = torch.as_tensor(Xn_all, dtype=torch.float32, device=device)
    # v_all_t = torch.as_tensor(to_value(y_all.astype(np.float64)), dtype=torch.float32,
    #                           device=device)

    # def sample_pairs(n):
    #     i = np.random.choice(succ_tr, n)
    #     j = np.minimum(i + np.random.randint(1, MAX_GAP + 1, n), len(ep_ids) - 1)
    #     keep = ep_ids[j] == ep_ids[i]
    #     return (torch.as_tensor(i[keep], device=device),
    #             torch.as_tensor(j[keep], device=device))
    
    y_va_succ_v = to_value(y_va_succ.astype(np.float64))      # the model's own target scale

    def bootstrap_fn(obs_n):
        return target_net.bootstrap_values(obs_n, device)
    
    fail_va = is_failed[rows_va]
    best, stale = float("inf"), 0
    hist = []
    yb_tr, sup_tr, yb_tr_t, train_pool = None, None, None, None
    for ep in range(epochs):
        trunc_arg = prepared["is_truncated_all"] if censor_bootstrap else None

        def _build(m, **kw):
            out = build_targets(m, bootstrap_fn, rows_tr, y_all,
                                 prepared["next_obs_n_all"], prepared["is_terminal_all"],
                                 prepared["terminal_value_all"], prepared["episode_ids_all"],
                                 prepared["frame_idxs_all"], is_truncated_all=trunc_arg, **kw)
            # with a truncation mask build_targets returns (targets, supervise)
            return out if isinstance(out, tuple) else (out, None)

        rebuilt = False
        if method == "mc":
            if yb_tr is None:
                yb_tr, sup_tr = _build("mc")
                rebuilt = True
        elif ep % target_refresh == 0:
            target_net.load_state_dict(model.state_dict())
            yb_tr, sup_tr = _build(method, gamma=gamma, lam=lam, clip_max=target_clip)
            rebuilt = True
        if rebuilt:
            yb_tr_t = torch.as_tensor(yb_tr, dtype=torch.float32, device=device)
            train_pool = (np.arange(len(rows_tr)) if sup_tr is None
                          else np.flatnonzero(sup_tr))

        model.train()
        perm = torch.as_tensor(np.random.permutation(train_pool), device=device)
        td_sum = pair_sum = 0.0
        n_td = n_pair = 0
        for s in range(0, len(perm), batch_size):
            b = perm[s:s + batch_size]
            opt.zero_grad()
            td_loss = F.mse_loss(model(Xn_tr_t[b]), yb_tr_t[b])
            loss = td_loss
            td_sum, n_td = td_sum + td_loss.item(), n_td + 1
            
            # if REL_WEIGHT > 0:
            #     pi, pj = sample_pairs(len(b))
            #     if pi.numel() > 0:
            #         d_pred = model(X_all_t[pi]) - model(X_all_t[pj])
            #         pair_loss = F.mse_loss(d_pred, v_all_t[pi] - v_all_t[pj])
            #         loss = loss + REL_WEIGHT * pair_loss
            #         pair_sum, n_pair = pair_sum + pair_loss.item(), n_pair + 1
                    
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_va = model(Xn_va_t).cpu().numpy()
            pred_va_succ = model(Xn_va_succ_t).cpu().numpy()
            
        val_mse = float(np.mean((pred_va - y_va) ** 2))
        val_mse_succ = float(np.mean((pred_va_succ - y_va_succ_v) ** 2))    # selection
        val_mse_succ_raw = float(np.mean((pred_va_succ - y_va_succ) ** 2))  # raw steps, report
        pred_fail = float(pred_va[fail_va].mean()) if fail_va.any() else float("nan")
        td_mean = td_sum / max(n_td, 1)
        pair_mean = pair_sum / n_pair if n_pair else float("nan")
        hist.append((ep, val_mse, val_mse_succ, pred_fail, val_mse_succ_raw, td_mean, pair_mean))

        if val_mse < best:
            best, stale = val_mse, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    return model, np.array(hist), best


def run_all_combos(run_dir, dataset_path, methods=METHODS, conditions=CONDITIONS,
                   censor_schemes=CENSOR_SCHEMES, bootstrap_gamma=T2S_BOOTSTRAP_GAMMA,
                   seed=SEED, device="cpu", verbose=True):
    
    assert all(cs in ("censor", "bootstrap") for cs in censor_schemes), censor_schemes
    target_clip = CENSOR_LABEL * 1.2

    prepared_by_scheme = {
        cs: load_and_prepare(dataset_path, censor_label=CENSOR_LABEL,
                             censor_bootstrap=(cs == "bootstrap"))
        for cs in censor_schemes}
    prepared = prepared_by_scheme[censor_schemes[0]]      # for normalization stats

    n_total = len(censor_schemes) * len(methods) * len(conditions)
    models, histories, summary_rows = {}, {}, []
    i = 0
    for scheme in censor_schemes:
        prep = prepared_by_scheme[scheme]
        boot = scheme == "bootstrap"
        # gamma=1 has no finite fixed point once the truncated anchor is removed
        gamma = bootstrap_gamma if boot else CENSOR_GAMMA
        for method in methods:
            for condition in conditions:
                i += 1
                combo = f"{method}{'boot' if boot else ''}_{condition}"
                if verbose:
                    print(f"  [{i}/{n_total}] {combo} (gamma={gamma})", flush=True)
                # model, hist, val_mse = train_one(
                #     prep, method, condition, run_dir, seed=seed,
                #     target_clip=target_clip, device=device, gamma=gamma,
                #     censor_bootstrap=boot)
                # models[combo] = model
                # histories[combo] = hist
                # summary_rows.append(dict(
                #     combo=combo, method=method, condition=condition,
                #     censor_scheme=scheme, gamma=gamma, seed=seed,
                #     val_mse=float(val_mse),
                #     # val MSE restricted to rows from SUCCESSFUL episodes, at the
                #     # epoch that minimised overall val MSE. Failed-episode rows
                #     # carry a censored label, so the overall number partly scores
                #     # agreement with a fiction.
                #     val_mse_succ_only=float(hist[np.argmin(hist[:, 1]), 2]),
                #     # every consumer needs a seed to build the checkpoint filename
                #     best_seed=seed))
                
                
                model, hist, _ = train_one(
                    prep, method, condition, run_dir, seed=seed,
                    target_clip=target_clip, device=device, gamma=gamma,
                    censor_bootstrap=boot)
                
                at = int(np.argmin(hist[:, 1]))          # the epoch that was saved
                np.save(os.path.join(run_dir, f"{combo}_hist.npy"), hist)
                models[combo] = model
                histories[combo] = hist
                summary_rows.append(dict(
                    combo=combo, method=method, condition=condition,
                    censor_scheme=scheme, gamma=gamma, seed=seed,
                    val_mse=float(hist[at, 1]),
                    val_mse_succ_only=float(hist[at, 4]),
                    pred_fail_mean=float(hist[at, 3]),
                    best_epoch=at,
                    best_seed=seed))

    save_normalization(run_dir, prepared["x_mean"], prepared["x_std"])

    manifest = dict(
        obs_dim=OBS_DIM, seed=seed, target_clip=target_clip,
        normalization_file="normalization.npz",
        censor_schemes=list(censor_schemes), bootstrap_gamma=bootstrap_gamma,
        combos={r["combo"]: dict(
            method=r["method"], condition=r["condition"],
            censor_scheme=r["censor_scheme"], gamma=r["gamma"],
            checkpoint=checkpoint_name(
                f"{r['method']}{'boot' if r['censor_scheme'] == 'bootstrap' else ''}",
                r["condition"], seed),
            val_mse=r["val_mse"], val_mse_succ_only=r["val_mse_succ_only"],
            best_seed=seed) for r in summary_rows})
    save_manifest(run_dir, manifest)
    return models, histories, summary_rows