import json, numpy as np, torch, core.results as results
from stable_baselines3 import SAC
from core.env_utils import make_fixed_scene_env

d2 = results.get_run_dir("t2s_model", "v8")
norm = np.load(f"{d2}/normalization.npz")
run_dir = results.get_run_dir("policy", "diff_time_v1_tdlambdaboot_all_seed0")
states = []
for ck in ("policy_final.zip", "policy_300000.zip"):
    pol = SAC.load(f"{run_dir}/{ck}", device="cpu")
    for seed in range(3):
        env = make_fixed_scene_env(); torch.manual_seed(seed)
        obs, _ = env.reset(seed=seed)
        for _ in range(500):
            states.append(obs)
            obs, _r, term, trunc, _i = env.step(pol.predict(obs, deterministic=False)[0])
            if term or trunc:
                break
        env.close()
z = np.abs((np.array(states) - norm["X_mean"]) / norm["X_std"])
for i in np.argsort(-z.max(axis=0))[:8]:
    print(f"input {i:>2}: largest |z| {z[:, i].max():8.1f}, 99th percentile {np.percentile(z[:, i], 99):6.1f}")