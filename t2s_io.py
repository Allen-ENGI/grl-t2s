"""
Save/load contract for T2S model artifacts.

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
