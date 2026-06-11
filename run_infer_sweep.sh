#!/usr/bin/env bash
# Run config_infer1..28 sequentially, compare each rollout against hex_GT,
# and rename the rollout file so the stress relative-L2 appears in the filename.
# Run from the repository root:  bash run_infer_sweep.sh
set -u

GT=dataset/hex_GT.h5
SUMMARY=outputs/rollout/parametric_sweep/l2_summary.csv

mkdir -p outputs/rollout/parametric_sweep
echo "model,rollout_file,stress_relL2,disp_relL2,all_relL2" > "$SUMMARY"

for i in $(seq 1 28); do
    cfg="ex1/config_infer${i}.txt"
    if [ ! -f "$cfg" ]; then
        echo "--- model${i}: $cfg not found, skipping"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "=== model${i}: running $cfg"
    echo "============================================================"
    if ! python MeshGraphNets_main.py --config "$cfg"; then
        echo "!!! model${i}: inference FAILED, skipping comparison"
        continue
    fi

    out_dir=$(grep -E '^[[:space:]]*inference_output_dir' "$cfg" | awk '{print $2}')
    # newest raw rollout file; exclude already-renamed (_L2_) files from earlier runs
    rollout=$(ls -t "$out_dir"/rollout_sample*_steps*.h5 2>/dev/null | grep -v '_L2_' | head -n1)
    if [ -z "$rollout" ]; then
        echo "!!! model${i}: no rollout output found in $out_dir"
        continue
    fi

    if ! l2_line=$(python compare_rollout_gt.py "$rollout" "$GT"); then
        echo "!!! model${i}: GT comparison failed for $rollout"
        continue
    fi
    read -r stress_l2 disp_l2 all_l2 <<< "$l2_line"

    renamed="${rollout%.h5}_L2_${stress_l2}.h5"
    mv "$rollout" "$renamed"
    echo ">>> model${i}: stress relL2=${stress_l2}  disp relL2=${disp_l2}  all relL2=${all_l2}"
    echo ">>> saved: $renamed"
    echo "model${i},${renamed},${stress_l2},${disp_l2},${all_l2}" >> "$SUMMARY"
done

echo ""
echo "============================================================"
echo "=== Summary (sorted by stress relL2, best first)"
echo "============================================================"
head -n1 "$SUMMARY"
tail -n +2 "$SUMMARY" | sort -t, -k3 -g
echo ""
echo "Summary CSV: $SUMMARY"
