#!/usr/bin/env bash
# Sweep launcher — pipeline: train -> infer -> plot rollout, per config.
# 16 configs across 8 GPUs (2 per GPU). All 16 pipelines run in parallel.
#
# For each config N:
#   1. Train using ex1/config_train${N}.txt -> writes ./outputs/ex1/model${N}.pth
#   2. When training finishes, run inference using ex1/config_infer${N}.txt
#                                       -> writes outputs/rollout/model${N}/
#   3. Generate rollout plots (vs ground truth) into outputs/rollout/model${N}/
#
# Within each pipeline, stages are sequential (infer waits for train to finish).
# Across pipelines, everything runs in parallel.
#
# Usage:
#   ./ex1/run_sweep.sh                  # all 16 pipelines
#   ./ex1/run_sweep.sh --dry-run        # print commands without launching
#   ./ex1/run_sweep.sh --plot-only      # plot existing rollouts + loss curves
#   ./ex1/run_sweep.sh --train-only     # only train, skip infer/plot
#   ./ex1/run_sweep.sh 1 5 11           # only specific configs
#
# Override python: PYTHON_BIN=/path/to/python ./ex1/run_sweep.sh
# Override truth dataset for plotting: TRUTH=./dataset/other.h5 ./ex1/run_sweep.sh --plot-only
#
# chmod +x ex1/run_sweep.sh once after editing.

set -uo pipefail

DEFAULT_CONFIGS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
CONFIGS=()
DRY_RUN=0
PLOT_ONLY=0
TRAIN_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=1;    shift ;;
        --plot-only)  PLOT_ONLY=1;  shift ;;
        --train-only) TRAIN_ONLY=1; shift ;;
        --help|-h)
            grep -E '^# ' "$0" | sed 's/^# //'
            exit 0
            ;;
        *)
            CONFIGS+=("$1")
            shift
            ;;
    esac
done

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TRUTH="${TRUTH:-./dataset/ex1.h5}"

# Resolve venv python (override with PYTHON_BIN=/path/to/python)
if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON="$PYTHON_BIN"
elif [[ -x "$PROJECT_ROOT/venv/bin/python" ]]; then
    PYTHON="$PROJECT_ROOT/venv/bin/python"
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
else
    echo "ERROR: no venv python found." >&2
    echo "  Set PYTHON_BIN to override." >&2
    exit 1
fi
echo "Using python: $PYTHON"
echo "Truth dataset for rollout plots: $TRUTH"
echo ""

cd "$PROJECT_ROOT"

# Plot-only mode: regenerate loss curves + rollout plots, then exit
if [[ $PLOT_ONLY -eq 1 ]]; then
    echo "Plot-only mode"
    "$PYTHON" "$SCRIPT_DIR/plot_sweep5.py" "${CONFIGS[@]}" --combined
    for n in "${CONFIGS[@]}"; do
        rollout_dir="outputs/rollout/model${n}"
        if [[ -d "$rollout_dir" ]]; then
            echo "Plotting rollouts in $rollout_dir"
            "$PYTHON" ex1/plot_rollout.py "$rollout_dir" --truth "$TRUTH"
        else
            echo "  SKIP model${n}: $rollout_dir not found"
        fi
    done
    exit 0
fi

# Per-config pipeline: train -> infer -> plot rollout
# Runs in a subshell, backgrounded by the caller.
run_pipeline() {
    local n=$1
    local train_cfg="ex1/config_train${n}.txt"
    local infer_cfg="ex1/config_infer${n}.txt"
    local train_out="ex1/train${n}.stdout.log"
    local infer_out="ex1/infer${n}.stdout.log"
    local plot_out="ex1/plot${n}.stdout.log"
    local rollout_dir="outputs/rollout/model${n}"
    local stage_log="ex1/pipeline.log"

    echo "[$(date '+%F %T')] train${n} STARTED" >> "$stage_log"
    "$PYTHON" MeshGraphNets_main.py --config "$train_cfg" \
        < /dev/null > "$train_out" 2>&1
    local rc=$?
    echo "[$(date '+%F %T')] train${n} FINISHED rc=$rc" >> "$stage_log"
    if [[ $rc -ne 0 ]]; then
        echo "[$(date '+%F %T')] train${n} FAILED — skipping infer/plot" >> "$stage_log"
        return $rc
    fi

    if [[ ${TRAIN_ONLY:-0} -eq 1 ]]; then
        return 0
    fi

    if [[ ! -f "$infer_cfg" ]]; then
        echo "[$(date '+%F %T')] infer${n} SKIP — $infer_cfg not found" >> "$stage_log"
        return 0
    fi

    echo "[$(date '+%F %T')] infer${n} STARTED" >> "$stage_log"
    "$PYTHON" MeshGraphNets_main.py --config "$infer_cfg" \
        < /dev/null > "$infer_out" 2>&1
    rc=$?
    echo "[$(date '+%F %T')] infer${n} FINISHED rc=$rc" >> "$stage_log"
    if [[ $rc -ne 0 ]]; then
        return $rc
    fi

    if [[ -d "$rollout_dir" ]]; then
        echo "[$(date '+%F %T')] plot${n} STARTED" >> "$stage_log"
        "$PYTHON" ex1/plot_rollout.py "$rollout_dir" --truth "$TRUTH" \
            < /dev/null > "$plot_out" 2>&1
        echo "[$(date '+%F %T')] plot${n} FINISHED" >> "$stage_log"
    else
        echo "[$(date '+%F %T')] plot${n} SKIP — $rollout_dir empty" >> "$stage_log"
    fi
}

# Make pipeline + TRAIN_ONLY visible to subshells
export -f run_pipeline
export PYTHON TRUTH TRAIN_ONLY

mkdir -p ex1 outputs/ex1 outputs/rollout
: > ex1/pipeline.log
echo "[$(date '+%F %T')] launching: ${CONFIGS[*]}" >> ex1/pipeline.log

LAUNCHED=0
MISSING=()
PIDS=()

for n in "${CONFIGS[@]}"; do
    train_cfg="ex1/config_train${n}.txt"
    if [[ ! -f "$train_cfg" ]]; then
        echo "  SKIP train${n}: $train_cfg not found"
        MISSING+=("$n")
        continue
    fi
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  [dry] pipeline: train${n} -> infer${n} -> plot rollout"
        continue
    fi
    echo "  launching pipeline train${n} (background)"
    ( run_pipeline "$n" ) &
    PIDS+=($!)
    LAUNCHED=$((LAUNCHED + 1))
done

echo ""
if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry run complete. $((${#CONFIGS[@]} - ${#MISSING[@]})) pipelines would launch."
    exit 0
fi

echo "Launched $LAUNCHED pipeline(s)."
if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "Missing train configs (skipped): ${MISSING[*]}"
fi
echo "PIDs: ${PIDS[*]}"
echo "${PIDS[*]}" > "$SCRIPT_DIR/sweep_pids.txt"
echo ""
echo "Monitor:"
echo "  tail -f ex1/pipeline.log               # high-level stage tracker"
echo "  tail -f ex1/train1.log                 # specific train log"
echo "  tail -f ex1/train1.stdout.log          # specific train stdout"
echo "  nvidia-smi -l 5"
echo ""
echo "Stop all pipelines:"
echo "  kill \$(cat ex1/sweep_pids.txt)"
echo ""
echo "Rollout plots will appear automatically after each pipeline finishes:"
echo "  outputs/rollout/model{1..16}/rollout_sample*_stress.png"
echo ""
echo "Force-replot anytime:"
echo "  ./ex1/run_sweep.sh --plot-only"
