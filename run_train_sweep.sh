#!/usr/bin/env bash
# Run every ex1/config_train*.txt (24 configs, ex1/old excluded) ALL at once,
# each as a background process.
# Each config declares its own gpu_ids, so configs sharing a GPU run
# concurrently on it (e.g. train1/train11/train21 together on GPU 0) —
# make sure GPU memory fits 3 jobs per GPU.
# Already-trained models (existing model{i}.pth) are skipped so the sweep can
# be resumed after an interruption; pass --force to retrain everything.
# Per-config stdout goes to ex1/parametric_sweep/sweep_logs/train{i}.out.
# Run from the repository root:  bash run_train_sweep.sh [--force]
set -u

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

LOG_DIR=ex1/parametric_sweep/sweep_logs
mkdir -p "$LOG_DIR"

get_key() {
    grep -E "^[[:space:]]*$2" "$1" | head -n1 | awk '{print $2}'
}

run_one() {
    local i=$1
    local cfg=$2

    echo ">>> train${i}: started  ($(date '+%Y-%m-%d %H:%M:%S'))  gpu_ids=$(get_key "$cfg" gpu_ids)  log: $LOG_DIR/train${i}.out"
    local t0 t1
    t0=$(date +%s)
    if python MeshGraphNets_main.py --config "$cfg" > "$LOG_DIR/train${i}.out" 2>&1; then
        t1=$(date +%s)
        echo ">>> train${i}: DONE in $(( (t1 - t0) / 60 )) min"
    else
        t1=$(date +%s)
        echo "!!! train${i}: FAILED after $(( (t1 - t0) / 60 )) min (see $LOG_DIR/train${i}.out)"
    fi
}

sweep_start=$(date +%s)
launched=0

# only configs directly inside ex1/ — ex1/old is not matched by this glob
for cfg in ex1/config_train*.txt; do
    i=$(basename "$cfg" .txt)
    i=${i#config_train}

    ckpt=$(get_key "$cfg" modelpath)
    if [ "$FORCE" -eq 0 ] && [ -n "$ckpt" ] && [ -f "$ckpt" ]; then
        echo "--- train${i}: checkpoint $ckpt already exists, skipping (use --force to retrain)"
        continue
    fi

    run_one "$i" "$cfg" &
    launched=$((launched + 1))
done

echo ""
echo "=== ${launched} training jobs launched in parallel, waiting for all to finish..."
wait

# final status from checkpoint existence
sweep_end=$(date +%s)
echo ""
echo "============================================================"
echo "=== Training sweep finished in $(( (sweep_end - sweep_start) / 3600 ))h $(( ((sweep_end - sweep_start) % 3600) / 60 ))m"
for cfg in ex1/config_train*.txt; do
    i=$(basename "$cfg" .txt)
    i=${i#config_train}
    ckpt=$(get_key "$cfg" modelpath)
    if [ -n "$ckpt" ] && [ -f "$ckpt" ]; then
        echo "  train${i}: OK   ($ckpt)"
    else
        echo "  train${i}: MISSING checkpoint ($ckpt)"
    fi
done
echo "============================================================"
