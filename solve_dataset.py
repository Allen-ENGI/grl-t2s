# import json

# import numpy as np, results
# d2 = results.get_run_dir("t2s_model", "v9")
# h = np.load(f"{d2}/tdlambdaboot_all_hist.npy")     # columns: epoch, val_mse, succ (discounted), pred_fail, succ (raw), ...
# sel, best = int(np.argmin(h[:, 1])), int(np.argmin(h[:, 2]))
# print(f"{len(h)} epochs; saved epoch {sel}: success MSE {h[sel, 2]:.1f}")
# print(f"best epoch for successes {best}: success MSE {h[best, 2]:.1f}")

# data = np.load(json.load(open(f"{d2}/stage_manifest.json"))["dataset_path"])
# X, y, ep = data["X"], data["y_steps"], data["episode_ids"]
# t, dist = [], []
# for e in np.unique(ep[y == 0]):
#     yy, xx = y[ep == e], X[ep == e]
#     s = int(np.flatnonzero(yy == 0)[0])
#     t.append(s); dist.append(np.linalg.norm(xx[s, 4:7] - xx[s, 36:39]))
# t, dist = np.array(t), 100 * np.array(dist)
# print(f"time to success: min {t.min()}, 5th percentile {np.percentile(t, 5):.0f}, median {np.median(t):.0f}")
# print(f"peg-to-goal distance at success (cm): median {np.median(dist):.1f}, max {dist.max():.1f}")


import json, numpy as np, results
from t2s_model import load_t2s_predictor
g = 0.995
d2 = results.get_run_dir("t2s_model", "v9")
data = np.load(json.load(open(f"{d2}/stage_manifest.json"))["dataset_path"])
X, y, ep, sp, pos = data["X"], data["y_steps"], data["episode_ids"], data["split"].astype(str), data["position_ids"]
succ = np.isin(ep, np.unique(ep[y == 0])) & (y > 0)
for model in ("tdlambdaboot_all", "tdlambdaboot_succ"):
    fn = load_t2s_predictor(d2, *model.rsplit("_", 1), seed=0)
    for name, m in (("train, any position", (sp == "train") & succ),
                    ("val, fixed start", (sp == "val") & succ & (pos < 0)),
                    ("val, unseen positions", (sp == "val") & succ & (pos >= 0))):
        r = np.random.default_rng(0).choice(np.flatnonzero(m), min(3000, m.sum()), replace=False)
        p = np.array([fn(x) for x in X[r]])
        target = (1 - g ** y[r]) / (1 - g)
        print(f"{model:<20}{name:<24} typical error {np.sqrt(np.mean((p - target) ** 2)):5.1f}")