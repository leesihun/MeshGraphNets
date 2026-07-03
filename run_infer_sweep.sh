#!/usr/bin/env bash
# Run config_infer1..4 sequentially, compare each rollout against the GT, and
# rename the rollout file so both stress and disp R^2 appear in the filename.
# Each comparison also validates the rollout's t=0 against the seed dataset
# (--init) and records a per-model status (ok / skipped:<reason>) in the CSV.
# Run from the repository root:  bash run_infer_sweep.sh
#
# Optional: set ANIMATE=1 to also render elemental rollout GIFs per model
# (beside each rollout .h5 via animate_h5.py). Off by default because rendering
# all views for a 50-step rollout is slow.  Usage:  ANIMATE=1 bash run_infer_sweep.sh
set -u

GT=dataset/rect_GT.h5
SUMMARY=outputs/rollout/parametric_sweep/l2_summary.csv
PLOT_DIR=outputs/rollout/parametric_sweep/plots
LOG_DIR=outputs/rollout/parametric_sweep/logs

mkdir -p outputs/rollout/parametric_sweep "$PLOT_DIR" "$LOG_DIR"
echo "model,rollout_file,stress_R2,disp_R2,status" > "$SUMMARY"

for i in $(seq 1 4); do
    cfg="ex2/config_infer${i}.txt"
    if [ ! -f "$cfg" ]; then
        echo "--- model${i}: $cfg not found, skipping"
        echo "model${i},,,,skipped: config not found" >> "$SUMMARY"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "=== model${i}: running $cfg"
    echo "============================================================"
    if ! python MeshGraphNets_main.py --config "$cfg"; then
        echo "!!! model${i}: inference FAILED, skipping comparison"
        echo "model${i},,,,skipped: inference failed" >> "$SUMMARY"
        continue
    fi

    out_dir=$(grep -E '^[[:space:]]*inference_output_dir' "$cfg" | awk '{print $2}')
    # newest raw rollout file; exclude already-renamed files from earlier runs
    rollout=$(ls -t "$out_dir"/rollout_sample*_steps*.h5 2>/dev/null | grep -v '_sR2_' | head -n1)
    if [ -z "$rollout" ]; then
        echo "!!! model${i}: no rollout output found in $out_dir"
        echo "model${i},,,,skipped: no rollout output" >> "$SUMMARY"
        continue
    fi

    # validate the rollout's t=0 against the dataset it was actually seeded from
    init_ds=$(grep -E '^[[:space:]]*infer_dataset' "$cfg" | awk '{print $2}')
    log="$LOG_DIR/model${i}_compare.log"
    if [ -n "$init_ds" ] && [ -f "$init_ds" ]; then
        r2_output=$(python compare_rollout_gt.py "$rollout" "$GT" \
            --plot-dir "$PLOT_DIR" --name "model${i}" --init "$init_ds" 2>"$log")
    else
        r2_output=$(python compare_rollout_gt.py "$rollout" "$GT" \
            --plot-dir "$PLOT_DIR" --name "model${i}" 2>"$log")
    fi
    status=$?
    cat "$log" >&2   # surface the safety report on the terminal

    if [ "$status" -ne 0 ]; then
        # first failed check (or file error) becomes the recorded reason
        reason=$(grep -E '\[[[:space:]]*FAIL[[:space:]]*\]' "$log" | head -n1 \
                 | sed -E 's/.*\][[:space:]]*//; s/,/;/g')
        [ -z "$reason" ] && reason=$(grep -E '^ERROR' "$log" | head -n1 | sed 's/,/;/g')
        [ -z "$reason" ] && reason="compare failed (exit $status)"
        echo "!!! model${i}: GT comparison failed -- $reason"
        echo "    log: $log"
        echo "model${i},${rollout},,,skipped: ${reason}" >> "$SUMMARY"
        continue
    fi

    stress_r2=$(echo "$r2_output" | awk '{print $1}')
    disp_r2=$(echo "$r2_output" | awk '{print $2}')

    renamed="${rollout%.h5}_sR2_${stress_r2}_dR2_${disp_r2}.h5"
    mv "$rollout" "$renamed"
    echo ">>> model${i}: stress R2=${stress_r2}  disp R2=${disp_r2}"
    echo ">>> saved: $renamed"
    echo "model${i},${renamed},${stress_r2},${disp_r2},ok" >> "$SUMMARY"

    # Optional elemental rollout GIFs (opt-in). Default out-dir puts the GIFs
    # next to "$renamed", so they stay associated with the R^2-tagged rollout.
    if [ "${ANIMATE:-0}" = "1" ]; then
        echo ">>> model${i}: rendering rollout GIFs (ANIMATE=1)"
        if ! python animate_h5.py "$renamed"; then
            echo "!!! model${i}: animation failed (non-fatal)"
        fi
    fi
done

echo ""
echo "============================================================"
echo "=== Summary (ok models sorted by stress R2, best first)"
echo "============================================================"
head -n1 "$SUMMARY"
tail -n +2 "$SUMMARY" | awk -F, '$5 == "ok"' | sort -t, -k3 -gr

skipped=$(tail -n +2 "$SUMMARY" | awk -F, '$5 != "ok" {print "  "$1": "$5}')
if [ -n "$skipped" ]; then
    echo ""
    echo "Skipped models:"
    echo "$skipped"
fi

echo ""
echo "Summary CSV: $SUMMARY"
echo "Plots:       $PLOT_DIR"
echo "Logs:        $LOG_DIR"
