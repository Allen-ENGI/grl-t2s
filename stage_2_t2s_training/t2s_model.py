import torch
import torch.nn as nn
import numpy as np
import os
from core.config import OBS_DIM, T2S_PRED_CLIP_MAX
from stage_2_t2s_training.t2s_io import load_normalization, checkpoint_name
class T2SModel(nn.Module):
    """Encoder + T2S regression head. Softplus output, so predictions are >= 0."""

    def __init__(self, obs_dim, hidden=256, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.t2s_head = nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())

    def forward(self, x):
        return self.t2s_head(self.encoder(x)).squeeze(-1)

    def predict_t2s_only(self, x):
        return self.forward(x)

    @torch.no_grad()
    def bootstrap_values(self, obs_n, device):
        """Used by td0/td-lambda target construction — see t2s_targets.py."""
        self.eval()
        return self.predict_t2s_only(torch.tensor(obs_n, device=device)).cpu().numpy()


"""
Output prediction remaining timestamps based on the selected T2S model and input observations.
"""
def load_t2s_predictor(run_dir, method, condition, seed=0, obs_dim=OBS_DIM,
                       pred_clip_max=T2S_PRED_CLIP_MAX, device="cpu"):

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