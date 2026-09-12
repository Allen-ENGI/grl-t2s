"""
Save/load contract for T2S model artifacts.

This is the ONE place that defines the normalization.npz schema. The bug you
hit earlier (`KeyError: 'X_mean' is not a file in the archive`) happened
because the save call (in a training notebook) and the load call (in a
policy-training notebook) each wrote their own copy of this logic and drifted:
save used `x_mean`/`x_std` and never wrote `y_mean`/`y_std` at all, while load
expected `X_mean`/`X_std`/`y_mean`/`y_std`. With both sides importing from
here, that class of bug is structurally impossible — see tests/test_t2s_io.py.
"""
import json
import os

import numpy as np

NORM_KEYS = ("X_mean", "X_std", "y_mean", "y_std")


def save_normalization(run_dir, X_mean, X_std, y_mean, y_std):
    path = os.path.join(run_dir, "normalization.npz")
    np.savez(
        path,
        X_mean=np.asarray(X_mean, dtype=np.float32),
        X_std=np.asarray(X_std, dtype=np.float32),
        y_mean=np.float32(y_mean),
        y_std=np.float32(y_std),
    )
    return path


def load_normalization(run_dir):
    path = os.path.join(run_dir, "normalization.npz")
    d = np.load(path)
    missing = [k for k in NORM_KEYS if k not in d.files]
    if missing:
        raise KeyError(
            f"normalization file {path} is missing keys {missing}. "
            f"Found: {list(d.files)}. Did save_normalization/load_normalization "
            "get out of sync, or is this an old-format file?"
        )
    return d["X_mean"], d["X_std"], float(d["y_mean"]), float(d["y_std"])


def checkpoint_name(method, condition, seed):
    return f"{method}_{condition}_seed{seed}.pt"


def save_checkpoint(run_dir, model, method, condition, seed):
    import torch  # lazy import: keeps normalization/manifest helpers testable without torch installed
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
