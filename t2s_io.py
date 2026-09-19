"""
Save/load contract for T2S model artifacts.

This is the ONE place that defines the normalization.npz schema. The
`KeyError: 'X_mean' is not a file in the archive` bug happened because the
save call (in a training notebook) and the load call (in a policy-training
notebook) each had their own copy of this logic and drifted: save wrote
`x_mean`/`x_std`, load expected `X_mean`/`X_std`/`y_mean`/`y_std`. With both
sides importing from here, that class of bug is structurally impossible.

y_mean/y_std ARE GONE. T2SModel is trained directly against raw
steps-remaining targets — nothing normalized y before the MSE loss — but
predict_t2s unnormalized its output via `* y_std + y_mean`, which only makes
sense if y HAD been normalized. The reconciliation was to write the identity
(0.0, 1.0) so the rescale became a no-op. Two stored constants whose only
correct values are the identity, feeding an arithmetic step that must not do
anything, is a trap: anyone "fixing" them to real statistics would silently
break every prediction. Target normalization was never added, so the keys are
removed and t2s_predict no longer rescales. Old runs are still readable —
load_normalization ignores extra keys — but it will refuse a file whose
y_mean/y_std are not the identity, rather than quietly changing behaviour.
"""
import json
import os

import numpy as np

NORM_KEYS = ("X_mean", "X_std")
LEGACY_KEYS = ("y_mean", "y_std")


def save_normalization(run_dir, X_mean, X_std):
    path = os.path.join(run_dir, "normalization.npz")
    np.savez(
        path,
        X_mean=np.asarray(X_mean, dtype=np.float32),
        X_std=np.asarray(X_std, dtype=np.float32),
    )
    return path


def load_normalization(run_dir):
    """Returns (X_mean, X_std). Tolerates legacy files that also stored y stats."""
    path = os.path.join(run_dir, "normalization.npz")
    d = np.load(path)
    missing = [k for k in NORM_KEYS if k not in d.files]
    if missing:
        raise KeyError(
            f"normalization file {path} is missing keys {missing}. "
            f"Found: {list(d.files)}. Did save_normalization/load_normalization "
            "get out of sync, or is this an old-format file?"
        )
    # a legacy file with non-identity y stats was trained under an assumption
    # this loader no longer honours; fail loudly rather than shift every
    # prediction by an unexplained offset
    if "y_mean" in d.files or "y_std" in d.files:
        y_mean = float(d["y_mean"]) if "y_mean" in d.files else 0.0
        y_std = float(d["y_std"]) if "y_std" in d.files else 1.0
        if not (abs(y_mean) < 1e-6 and abs(y_std - 1.0) < 1e-6):
            raise ValueError(
                f"{path} stores y_mean={y_mean}, y_std={y_std}, but target normalization "
                "was never implemented in t2s_train and t2s_predict no longer rescales. "
                "This checkpoint's predictions would be off by that transform. Retrain, "
                "or re-save normalization.npz with save_normalization(run_dir, X_mean, X_std).")
    return d["X_mean"], d["X_std"]


def checkpoint_name(method, condition, seed):
    return f"{method}_{condition}_seed{seed}.pt"


def save_checkpoint(run_dir, model, method, condition, seed):
    import torch  # lazy: keeps normalization/manifest helpers testable without torch
    path = os.path.join(run_dir, checkpoint_name(method, condition, seed))
    torch.save(model.state_dict(), path)
    return path


def load_checkpoint(run_dir, model, method, condition, seed, map_location=None):
    import torch
    path = os.path.join(run_dir, checkpoint_name(method, condition, seed))
    model.load_state_dict(torch.load(path, map_location=map_location))
    return model


def save_manifest(run_dir, manifest: dict):
    path = os.path.join(run_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def load_manifest(run_dir):
    with open(os.path.join(run_dir, "manifest.json")) as f:
        return json.load(f)
