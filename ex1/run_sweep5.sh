#!/usr/bin/env bash
# Sweep 5 launcher (Linux) — runs training configs simultaneously.
# Each config carries its own gpu_ids; 2 processes share each GPU.
#
# GPU assignment (gpu_ids comes from each config file):
#   GPU 0: train1  (anchor)               + train11 (LR 0.001)
#   GPU 1: train2  (LR 0.00005)           + train12 (grad_accum 4 + LR 0.0002)
#   GPU 2: train3  (LR 0.0002)            + train13 (epochs 20000)
#   GPU 3: train4  (LR 0.0005)            + train14 (std_noise 0.01)
#   GPU 4: train5  (grad_accum 4)         + train15 (L=3 aggressive 2500/500/100)
#   GPU 5: train6  (epochs 5000)          + train16 (L=3 + bigger mp 32 blocks)
#   GPU 6: train7  (std_noise 0.005)      + train17 (mp 28 blocks at L=2)
#   GPU 7: train8  (L=3 5000/1000/200)    + train18 (Latent 192)
#
# Usage:
#   ./ex1/run_sweep5.sh                   # launch all 16
#   ./ex1/run_sweep5.sh --dry-run         # print commands without launching
#   ./ex1/run_sweep5.sh --plot-only       # skip training, just plot existing logs
#   ./ex1/run_sweep5.sh 1 5 11            # launch only selected configs
#
# Make executable once:  chmod +x ex1/run_sweep5.sh

set -euo pipefail

DEFAULT_CONFIGS=(1 2 3 4 5 6 7 8 11 12 13 14 15 16 17 18)
CONFIGS=()
DRY_RUN=0
PLOT_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1;   shift ;;
        --plot-only) PLOT_ONLY=1; shift ;;
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

if [[ $PLOT_ONLY -eq 1 ]]; then
    echo "Plot-only mode: generating loss plots from existing logs"
    python "$SCRIPT_DIR/plot_sweep5.py" "${CONFIGS[@]}" --combined
    exit 0
fi

echo "Sweep 5 launcher"
echo "  Project root: $PROJECT_ROOT"
echo "  Configs:      ${CONFIGS[*]}"
echo "  Dry run:      $DRY_RUN"
echo ""

cd "$PROJECT_ROOT"

LAUNCHED=0
MISSING=()
PIDS=()

for n in "${CONFIGS[@]}"; do
    CONFIG_REL="ex1/config_train${n}.txt"
    STDOUT_LOG="ex1/train${n}.stdout.log"
    STDERR_LOG="ex1/train${n}.stderr.log"

    if [[ ! -f "$CONFIG_REL" ]]; then
        echo "  SKIP train${n}: $CONFIG_REL not found"
        MISSING+=("$n")
        continue
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  [dry] python MeshGraphNets_main.py --config $CONFIG_REL > $STDOUT_LOG 2> $STDERR_LOG &"
        continue
    fi

    echo "  launching train${n} -> $STDOUT_LOG"
    nohup python MeshGraphNets_main.py --config "$CONFIG_REL" \
        < /dev/null > "$STDOUT_LOG" 2> "$STDERR_LOG" &
    PIDS+=($!)
    LAUNCHED=$((LAUNCHED + 1))
done

echo ""
if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry run complete. ${#CONFIGS[@]} commands would launch ($((${#CONFIGS[@]} - ${#MISSING[@]})) valid)."
else
    echo "Launched $LAUNCHED job(s)."
    if [[ ${#MISSING[@]} -gt 0 ]]; then
        echo "Missing configs (skipped): ${MISSING[*]}"
    fi
    echo "PIDs: ${PIDS[*]}"
    # Save PIDs for later monitoring/killing
    echo "${PIDS[*]}" > "$SCRIPT_DIR/sweep5_pids.txt"
    echo ""
    echo "Monitor with:"
    echo "  nvidia-smi -l 5"
    echo "  tail -f ex1/train1.log"
    echo "  tail -f ex1/train1.stdout.log"
    echo ""
    echo "Stop all jobs:"
    echo "  kill \$(cat ex1/sweep5_pids.txt)"
    echo ""
    echo "Generate loss plots (anytime — re-run as runs progress):"
    echo "  python ex1/plot_sweep5.py --combined"
    echo "  python ex1/plot_sweep5.py 1 11             # specific configs"
    echo "  ./ex1/run_sweep5.sh --plot-only            # same, via this script"
fi
