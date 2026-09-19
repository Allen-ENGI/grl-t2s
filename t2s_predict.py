"""
Loads a frozen, trained T2S model and exposes a single predict_t2s(obs)
function. This is the ONE place obs -> normalize -> forward -> clip happens,
used identically by t2s_eval.py (Stage 4), t2s_video.py and policy_env.py
(Stage 5 reward shaping) — each notebook used to re-derive this chain
slightly differently (see the normalization-key bug and the stage-head
mismatch).

The `* y_std + y_mean` rescale is gone: targets were never normalized during
training, so the only correct values for those constants were the identity
and the step was a no-op. See t2s_io.
"""
import os

import numpy as np
import torch

from config import OBS_DIM, T2S_PRED_CLIP_MAX
from t2s_io import load_normalization, checkpoint_name
from t2s_model import T2SModel


def load_t2s_predictor(run_dir, method, condition, seed=0, obs_dim=OBS_DIM,
                       pred_clip_max=T2S_PRED_CLIP_MAX, device="cpu"):
    """
    Returns a predict_t2s(obs, clip=True) callable with the model frozen
    (requires_grad_(False)) — never trained further during policy learning.

    pred_clip_max matters for the reward: reward_fn's hovering guard is
    checked against the largest prediction a model actually emits, and this
    ceiling is the worst case. Keep it in config so the guard and the
    predictor cannot disagree.
    """
    X_mean, X_std = load_normalization(run_dir)
    assert X_mean.shape[0] == obs_dim, (
        f"normalization is {X_mean.shape[0]}-dim but obs_dim is {obs_dim} — wrong run_dir?"
    )

    model = T2SModel(obs_dim=obs_dim).to(device)
    ckpt_path = os.path.join(run_dir, checkpoint_name(method, condition, seed))
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def predict_t2s(obs, clip=True):
        x = torch.tensor((np.asarray(obs, dtype=np.float32) - X_mean) / X_std,
                         dtype=torch.float32).unsqueeze(0).to(device)
        pred = float(model.predict_t2s_only(x).item())
        return float(np.clip(pred, 0.0, pred_clip_max)) if clip else pred

    return predict_t2s
