import json, numpy as np, core.results as results
d2 = results.get_run_dir("t2s_model", "v8")
norm = np.load(f"{d2}/normalization.npz")
data = np.load(json.load(open(f"{d2}/stage_manifest.json"))["dataset_path"])
y = data["y_steps"]
failed = np.isin(data["episode_ids"],
                 np.unique(data["episode_ids"][y == 0]), invert=True)
z = np.abs((data["X"][failed] - norm["X_mean"]) / norm["X_std"])
worst = np.argsort(-z.max(axis=0))[:8]
for i in worst:
    print(f"input {i:>2}: largest |z| on failed states {z[:, i].max():8.1f}, "
          f"99th percentile {np.percentile(z[:, i], 99):6.1f}")