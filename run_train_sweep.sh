#!/usr/bin/env bash
# Run config_train1..28 sequentially (missing numbers are skipped).
# Already-trained models (existing model{i}.pth) are skipped so the sweep can
# be resumed after an interruption; pass --force to retrain everything.
# Run from the repository root:  bash run_train_sweep.sh [--force]
set -u

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

sweep_start=$(date +%s)
ok=0; skipped=0; failed=0

for i in $(seq 1 28); do
    cfg="ex1/config_train${i}.txt"
    if [ ! -f "$cfg" ]; then
        echo "--- train${i}: $cfg not found, skipping"
        continue
    fi

    ckpt=$(grep -E '^[[:space:]]*modelpath' "$cfg" | awk '{print $2}')
    if [ "$FORCE" -eq 0 ] && [ -n "$ckpt" ] && [ -f "$ckpt" ]; then
        echo "--- train${i}: checkpoint $ckpt already exists, skipping (use --force to retrain)"
        skipped=$((skipped + 1))
        continue
    fi

    echo ""
    echo "============================================================"
    echo "=== train${i}: running $cfg  ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "============================================================"
    t0=$(date +%s)
    if python MeshGraphNets_main.py --config "$cfg"; then
        t1=$(date +%s)
        echo ">>> train${i}: done in $(( (t1 - t0) / 60 )) min"
        ok=$((ok + 1))
    else
        echo "!!! train${i}: FAILED, continuing with next config"
        failed=$((failed + 1))
    fi
done

sweep_end=$(date +%s)
echo ""
echo "============================================================"
echo "=== Training sweep finished: ${ok} trained, ${skipped} skipped, ${failed} failed"
echo "=== Total time: $(( (sweep_end - sweep_start) / 3600 ))h $(( ((sweep_end - sweep_start) % 3600) / 60 ))m"
echo "============================================================"
