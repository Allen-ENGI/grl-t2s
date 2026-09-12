"""
Loads a frozen, trained T2S model and exposes a single predict_t2s(obs)
function. This is the ONE place obs -> normalize -> forward -> rescale ->
clip happens, used identically by t2s_eval.py (Stage 4) and policy_env.py
(Stage 5 reward shaping) — previously each notebook re-derived this chain
slightly differently (see the normalization-key bug and the
with_stage_head / no-stage-head mismatch fixed in t2s_io.py / t2s_model.py).
"""
import os

import numpy as np
import torch

from config import OBS_DIM
from t2s_io import load_normalization
from t2s_model import T2SModel


def load_t2s_predictor(run_dir, method, condition, seed=0, obs_dim=OBS_DIM,
                        with_stage_head=False, pred_clip_max=300.0, device="cpu"):
    """
    Returns a predict_t2s(obs, clip=True) callable with the model frozen
    (requires_grad_(False)) — never trained further during policy learning.
    """
    from t2s_io import checkpoint_name

    X_mean, X_std, y_mean, y_std = load_normalization(run_dir)
    assert X_mean.shape[0] == obs_dim, (
        f"normalization is {X_mean.shape[0]}-dim but obs_dim is {obs_dim} — wrong run_dir?"
    )

    model = T2SModel(obs_dim=obs_dim, with_stage_head=with_stage_head).to(device)
    ckpt_path = os.path.join(run_dir, checkpoint_name(method, condition, seed))
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def predict_t2s(obs, clip=True):
        x = torch.tensor((np.asarray(obs, dtype=np.float32) - X_mean) / X_std,
                          dtype=torch.float32).unsqueeze(0).to(device)
        pred = model.predict_t2s_only(x).item() * y_std + y_mean
        return float(np.clip(pred, 0.0, pred_clip_max)) if clip else float(pred)

    return predict_t2s
