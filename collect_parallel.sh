#!/usr/bin/env bash
# Parallel data collection: one process per seed, then merge.
#
# Each process writes to its own run directory, so they never collide on
# disk. Collection is CPU-bound in MuJoCo, so N processes on N cores gives
# close to an N-times speedup — much better than the sequential seed loop
# inside collect_dataset.
#
#   ./collect_parallel.sh            # seeds 0 1 2, run name v9
#   ./collect_parallel.sh v10 0 1 2 3 4
set -euo pipefail

RUN_NAME="${1:-v9}"
shift || true
SEEDS=("${@:-0 1 2}")
if [ ${#SEEDS[@]} -eq 1 ]; then read -ra SEEDS <<< "${SEEDS[0]}"; fi

MULTIPLIER="${MULTIPLIER:-2}"
EPISODES_PER_CKPT="${EPISODES_PER_CKPT:-40}"

echo "run=$RUN_NAME seeds=${SEEDS[*]} multiplier=$MULTIPLIER"

pids=()
for s in "${SEEDS[@]}"; do
    # NOTE: --seeds takes ONE seed here on purpose. Each process runs the
    # whole plan for its own seed; the merge step combines them. Passing all
    # seeds to one process would run them sequentially instead.
    python collect_cli.py \
        --run-name "${RUN_NAME}_s${s}" \
        --seeds "$s" \
        --multiplier "$MULTIPLIER" \
        --episodes-per-checkpoint "$EPISODES_PER_CKPT" \
        --noise-variants \
        --random-policy \
        > "collect_${RUN_NAME}_s${s}.log" 2>&1 &
    pids+=($!)
    echo "  seed $s -> pid ${pids[-1]}  (log: collect_${RUN_NAME}_s${s}.log)"
done

fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || { echo "process $pid FAILED"; fail=1; }
done
[ $fail -eq 0 ] || { echo "at least one collection failed; check the logs"; exit 1; }

# merge into one dataset, renumbering episode ids so they stay distinct
SOURCES=()
for s in "${SEEDS[@]}"; do SOURCES+=("${RUN_NAME}_s${s}"); done
python collect_cli.py --merge "$RUN_NAME" "${SOURCES[@]}"

python - <<PY
import os, sys
sys.path.insert(0, os.getcwd())
import results, dataset_audit
d = results.get_run_dir("data_collection", "$RUN_NAME")
dataset_audit.audit_dataset(os.path.join(d, "dataset.npz"),
                             summary_path=os.path.join(d, "dataset_summary.json"))
PY

echo "done -> data_collection/$RUN_NAME"
