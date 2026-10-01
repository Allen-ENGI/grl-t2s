import json, os
import numpy as np
import config, results, t2s_train

# dataset from your T2S run
t2s_dir = results.get_run_dir("t2s_model", "v5")
dataset = json.load(open(os.path.join(t2s_dir, "stage_manifest.json")))["dataset_path"]

scratch = "train_res/scratch_es_test"          # NOT your real run dir: train_one saves a checkpoint
os.makedirs(scratch, exist_ok=True)
t2s_train.PATIENCE = t2s_train.EPOCHS          # no early stopping, so both minima are reachable

prep = t2s_train.load_and_prepare(dataset, censor_bootstrap=True)
for cond in ("all", "succ"):
    _, hist, _ = t2s_train.train_one(
        prep, "tdlambda", cond, scratch, gamma=config.T2S_BOOTSTRAP_GAMMA,
        target_clip=config.CENSOR_LABEL * 1.2, censor_bootstrap=True, device="cuda")
    best_all, best_succ = np.argmin(hist[:, 1]), np.argmin(hist[:, 2])
    print(f"tdlambdaboot_{cond}")
    print("  picked by val_mse     :", hist[best_all])
    print("  picked by val_mse_succ:", hist[best_succ])
    print(f"  succ MSE lost by using val_mse: {hist[best_all, 2] - hist[best_succ, 2]:.2f}")