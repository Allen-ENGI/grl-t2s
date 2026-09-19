"""
One canonical T2S model definition.

THE STAGE HEAD IS GONE. `Compare_value_targets_v3.ipynb` trained and saved
checkpoints with an encoder + t2s_head only, while `3_train_policy_v7.ipynb`
loaded those same files into a class that also defined
`stage_head = nn.Linear(hidden, n_stages)` — so a strict load_state_dict
(the default) raised "Missing key(s): stage_head.weight, stage_head.bias" the
moment a real checkpoint was loaded. The earlier fix was an optional head
controlled by a manifest flag.

It is now removed outright instead, because nothing ever turned it on:
data_collection's include_stage_label defaulted to False in every caller, so
the stage column it fed was all zeros in every dataset on disk, and
t2s_predict always loaded with with_stage_head=False. An optional
architecture branch that is never taken is just a second way for training and
inference to disagree. If the auxiliary stage objective is wanted later, add
it back together with the labels that supervise it.
"""
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



