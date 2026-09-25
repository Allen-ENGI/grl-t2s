import torch
import torch.nn as nn


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



