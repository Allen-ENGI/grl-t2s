"""
One canonical T2S model definition.

BUG FOUND WHILE READING YOUR NOTEBOOKS:
`Compare_value_targets_v3.ipynb` trains and saves checkpoints using `T2SModel`
(encoder + t2s_head only, no stage head). `3_train_policy_v7.ipynb` then loads
those same checkpoints into `StagedTime2SuccessModel`, which additionally
defines `self.stage_head = nn.Linear(hidden, n_stages)`. A strict
`load_state_dict` (the default) will raise a "Missing key(s) in state_dict:
stage_head.weight, stage_head.bias" error the moment a real (non-random)
checkpoint is loaded — this hasn't bitten you yet only because you haven't
run the policy notebook against a genuine `td0_succ_seed0.pt` end to end.

Fix: a single model class with an *optional* stage head, controlled by a
flag saved in the manifest, so training and inference always agree on
architecture the same way t2s_io now makes them agree on normalization keys.
"""
import torch
import torch.nn as nn


class T2SModel(nn.Module):
    """
    Encoder + T2S regression head, with an optional auxiliary stage-classification
    head (0=pre-control, 1=control, 2=complete — see data_collection.py).

    with_stage_head=False reproduces the plain `T2SModel` used in the value-target
    comparison. with_stage_head=True reproduces `StagedTime2SuccessModel` used
    downstream in policy training. Same class either way, so a manifest flag
    (not a second class definition) is what varies.
    """

    def __init__(self, obs_dim, hidden=256, dropout=0.2, with_stage_head=False, n_stages=3):
        super().__init__()
        self.with_stage_head = with_stage_head
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.t2s_head = nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())
        self.stage_head = nn.Linear(hidden, n_stages) if with_stage_head else None

    def forward(self, x):
        h = self.encoder(x)
        t2s = self.t2s_head(h).squeeze(-1)
        if self.with_stage_head:
            return t2s, self.stage_head(h)
        return t2s

    def predict_t2s_only(self, x):
        return self.t2s_head(self.encoder(x)).squeeze(-1)

    @torch.no_grad()
    def bootstrap_values(self, obs_n, device):
        """Used by td0/td-lambda target construction — see t2s_targets.py."""
        self.eval()
        return self.predict_t2s_only(torch.tensor(obs_n, device=device)).cpu().numpy()
