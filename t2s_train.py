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
"""
import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit

from config import CENSOR_LABEL, OBS_DIM
from t2s_model import T2SModel
from t2s_io import save_normalization, save_manifest, checkpoint_name
from t2s_targets import build_targets, build_bookkeeping

DEFAULT_EPOCHS = 400
DEFAULT_PATIENCE = 100
DEFAULT_TARGET_REFRESH = 5
DEFAULT_GAMMA = 1.0
DEFAULT_LAMBDA = 0.9
DEFAULT_BATCH_SIZE = 256


def load_and_prepare(dataset_path, censor_label=CENSOR_LABEL, confirm_buffer=20, test_size=0.15,
                      seed=0, censor_bootstrap=False):
    """
    Loads dataset.npz, trims the long flat post-success tail, builds the
    train/val split (grouped by episode so no episode leaks across the
    split), and returns everything build_targets/train_one need.
    """
    raw = np.load(dataset_path)
    X_all, y_all = raw["X"], raw["y_steps"]
    stage_all, episode_ids_all, frame_idxs_all = raw["stage"], raw["episode_ids"], raw["frame_idxs"]

    is_failed_episode = np.array([
        np.all(y_all[episode_ids_all == ep] == censor_label) for ep in episode_ids_all
    ])

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

    next_obs_all, is_terminal_all, terminal_value_all, is_truncated_all = build_bookkeeping(
        X_all, y_all, episode_ids_all, frame_idxs_all, censor_label,
        censor_bootstrap=censor_bootstrap)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_all, va_all = next(gss.split(X_all, y_all, groups=episode_ids_all))

    succ_mask = ~is_failed_episode
    tr_succ = tr_all[succ_mask[tr_all]]
    va_succ = va_all[succ_mask[va_all]]

    x_mean = X_all[tr_succ].mean(axis=0)
    x_std = X_all[tr_succ].std(axis=0)
    x_std = np.where(x_std < 1e-4, 1.0, x_std)

    def normalize(X):
        return ((X - x_mean) / x_std).astype(np.float32)

    return dict(
        X_all=X_all, y_all=y_all, next_obs_n_all=normalize(next_obs_all),
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


def train_one(prepared, method, condition, run_dir, seed=0, obs_dim=OBS_DIM,
              censor_bootstrap=None,
              epochs=DEFAULT_EPOCHS, patience=DEFAULT_PATIENCE,
              target_refresh=DEFAULT_TARGET_REFRESH, gamma=DEFAULT_GAMMA, lam=DEFAULT_LAMBDA,
              target_clip=None, batch_size=DEFAULT_BATCH_SIZE, device="cpu"):
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
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    Xn_all, y_all = prepared["Xn_all"], prepared["y_all"]
    Xn_tr = Xn_all[rows_tr]
    Xn_va, y_va = Xn_all[rows_va], y_all[rows_va]
    Xn_va_succ, y_va_succ = Xn_all[rows_va_succ], y_all[rows_va_succ]

    # Convert ONCE rather than per batch. The inner loop previously did
    # torch.tensor(Xn_tr[b]) every batch of every epoch — with a large
    # dataset that is hundreds of thousands of redundant numpy->torch copies
    # and dominates CPU-only training time.
    Xn_tr_t = torch.as_tensor(Xn_tr, dtype=torch.float32, device=device)
    Xn_va_t = torch.as_tensor(Xn_va, dtype=torch.float32, device=device)
    Xn_va_succ_t = torch.as_tensor(Xn_va_succ, dtype=torch.float32, device=device)

    def bootstrap_fn(obs_n):
        return target_net.bootstrap_values(obs_n, device)

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
        # truncated frames have no known target -> excluded from the loss
        perm = torch.as_tensor(np.random.permutation(train_pool), device=device)
        for s in range(0, len(perm), batch_size):
            b = perm[s:s + batch_size]
            opt.zero_grad()
            loss = F.mse_loss(model(Xn_tr_t[b]), yb_tr_t[b])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_va = model(Xn_va_t).cpu().numpy()
            pred_va_succ = model(Xn_va_succ_t).cpu().numpy()
        val_mse = float(np.mean((pred_va - y_va) ** 2))
        val_mse_succ = float(np.mean((pred_va_succ - y_va_succ) ** 2))
        hist.append((ep, val_mse, val_mse_succ))

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


def run_all_combos(run_dir, dataset_path, methods=("mc", "td0", "tdlambda"),
                    conditions=("succ", "all"), seeds=(0,), obs_dim=OBS_DIM,
                    censor_label=CENSOR_LABEL, device="cpu",
                    censor_schemes=("censor",), bootstrap_gamma=0.99, **train_kwargs):
    """
    Full multi-combo, multi-seed sweep. Saves normalization.npz + manifest.json.

    censor_schemes: which failed-episode handling to train. Any of:
        "censor"    — truncated frame supervised with censor_label (original)
        "bootstrap" — truncated frame unsupervised, bootstrapped through
                      (value bootstrap; forces gamma=bootstrap_gamma < 1)
    Passing both trains each method/condition under both schemes and names the
    bootstrap variants "<method>boot_<condition>", so Stage 4 ranks them side
    by side instead of you having to trust that the change helped.

    Note the two schemes need DIFFERENT preprocessing (build_bookkeeping marks
    truncated frames differently), so each gets its own `prepared`.
    """
    assert all(cs in ("censor", "bootstrap") for cs in censor_schemes), censor_schemes
    target_clip = train_kwargs.pop("target_clip", censor_label * 1.2)
    base_gamma = train_kwargs.pop("gamma", DEFAULT_GAMMA)

    prepared_by_scheme = {
        cs: load_and_prepare(dataset_path, censor_label=censor_label,
                              censor_bootstrap=(cs == "bootstrap"))
        for cs in censor_schemes
    }
    prepared = prepared_by_scheme[censor_schemes[0]]   # for normalization stats

    models, histories, summary_rows = {}, {}, []
    for scheme in censor_schemes:
        prep = prepared_by_scheme[scheme]
        boot = scheme == "bootstrap"
        # gamma=1 has no finite fixed point once the truncated anchor is removed
        gamma = bootstrap_gamma if boot else base_gamma
        for method in methods:
            for condition in conditions:
                key = f"{method}{'boot' if boot else ''}_{condition}"
                seed_mses = []
                best_model, best_hist, best_mse = None, None, float("inf")
                for sd in seeds:
                    m, h, mse = train_one(prep, method, condition, run_dir, seed=sd,
                                           obs_dim=obs_dim, target_clip=target_clip,
                                           device=device, gamma=gamma,
                                           censor_bootstrap=boot, **train_kwargs)
                    seed_mses.append(mse)
                    if mse < best_mse:
                        best_model, best_hist, best_mse = m, h, mse
                models[key] = best_model
                histories[key] = best_hist
                succ_at_best = float(best_hist[np.argmin(best_hist[:, 1]), 2])
                summary_rows.append(dict(
                    combo=key, mean_val_mse=float(np.mean(seed_mses)),
                    std_val_mse=float(np.std(seed_mses)),
                    seeds_val_mse=[float(v) for v in seed_mses],
                    mean_val_mse_succ_only=succ_at_best,
                    best_seed=int(seeds[int(np.argmin(seed_mses))]),
                    censor_scheme=scheme, gamma=gamma,
                ))

    # NOTE on y_mean/y_std: T2SModel is trained directly against raw
    # steps-remaining targets (see build_targets — nothing normalizes y
    # before the MSE loss). The policy-training notebook's predict_t2s(),
    # however, unnormalizes the model's output via `* Y_STD + Y_MEAN`,
    # which only makes sense if y WAS normalized during training. It
    # wasn't (that's bug #3 — see chat). Until/unless target normalization
    # is actually added to train_one, y_mean/y_std must be the identity
    # (0, 1) so predict.py's rescale step is a no-op and matches what the
    # model actually learned.
    save_normalization(run_dir, prepared["x_mean"], prepared["x_std"], 0.0, 1.0)

    manifest = dict(
        obs_dim=obs_dim, seeds=list(seeds), target_clip=target_clip,
        normalization_file="normalization.npz",
        censor_schemes=list(censor_schemes), bootstrap_gamma=bootstrap_gamma,
        combos={row["combo"]: dict(
            method=row["combo"].rsplit("_", 1)[0], condition=row["combo"].rsplit("_", 1)[1],
            censor_scheme=row["censor_scheme"], gamma=row["gamma"],
            per_seed_checkpoints=[checkpoint_name(*row["combo"].rsplit("_", 1), sd) for sd in seeds],
            per_seed_val_mse=row["seeds_val_mse"], mean_val_mse=row["mean_val_mse"],
            std_val_mse=row["std_val_mse"], mean_val_mse_succ_only=row["mean_val_mse_succ_only"],
            best_seed=row["best_seed"],
        ) for row in summary_rows},
    )
    save_manifest(run_dir, manifest)
    return models, histories, summary_rows
