"""
Run naming / results registry.

Every stage of the pipeline (data collection, each T2S formulation, each
policy training run) gets ONE call into this module to obtain its output
directory. This is what "naming the results" means concretely: a run is
identified by (stage, run_name), and everything that stage produces —
checkpoints, manifest.json, logs, plots — lives under one predictable path
and is registered in a top-level index so later stages (and `pipeline.ipynb`)
can look up "what's the latest / best t2s run" without hardcoding a path.

Directory layout produced:
    train_res/
      runs_index.json                 <- registry of every run ever created
      data_collection/<run_name>/     <- dataset.npz, dataset_summary.json
      t2s_model/<run_name>/           <- {method}_{condition}_seed{k}.pt, normalization.npz, manifest.json
      t2s_eval/<run_name>/            <- eval_report.json, plots
      policy/<run_name>/              <- policy checkpoints, vecnormalize.pkl, eval_history.json
"""
import json
import os
import time

from config import TRAIN_RES_DIR

STAGES = ("expert_policy", "data_collection", "t2s_model", "t2s_eval", "policy")

INDEX_PATH = os.path.join(TRAIN_RES_DIR, "runs_index.json")


def _load_index():
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH) as f:
            return json.load(f)
    return {stage: {} for stage in STAGES}


def _save_index(index):
    os.makedirs(TRAIN_RES_DIR, exist_ok=True)
    with open(INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)


def new_run_dir(stage, run_name, meta=None, exist_ok=True):
    """
    Register (or re-open) a run and return its output directory.

    stage: one of STAGES.
    run_name: short identifier, e.g. "v7", "td0_succ", "reward_absolute_v1".
    meta: optional dict of freeform metadata recorded in the index
          (git-free provenance: what produced this run and when).
    """
    assert stage in STAGES, f"unknown stage {stage!r}, expected one of {STAGES}"
    index = _load_index()

    run_dir = os.path.join(TRAIN_RES_DIR, stage, run_name)
    if os.path.exists(run_dir) and not exist_ok:
        raise FileExistsError(f"{run_dir} already exists; pick a new run_name or pass exist_ok=True")
    os.makedirs(run_dir, exist_ok=True)

    index.setdefault(stage, {})
    index[stage][run_name] = {
        "dir": run_dir,
        "created_or_updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "meta": meta or {},
    }
    _save_index(index)
    return run_dir


def get_run_dir(stage, run_name):
    """Look up a previously registered run's directory."""
    index = _load_index()
    try:
        return index[stage][run_name]["dir"]
    except KeyError:
        raise KeyError(f"no run named {run_name!r} registered for stage {stage!r}")


def latest_run(stage):
    """Return (run_name, dir) for the most recently created/updated run in a stage."""
    index = _load_index()
    entries = index.get(stage, {})
    if not entries:
        raise KeyError(f"no runs registered yet for stage {stage!r}")
    name = max(entries, key=lambda k: entries[k]["created_or_updated"])
    return name, entries[name]["dir"]


def list_runs(stage):
    index = _load_index()
    return sorted(index.get(stage, {}).keys())
