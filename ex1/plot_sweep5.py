"""Generate train/validation log-loss plots for sweep 5 configs.

Reads ex1/trainN.log files, extracts epoch/train/val loss, and writes
a per-config PNG (ex1/trainN_loss.png) plus an optional combined val plot.

Usage:
    python ex1/plot_sweep5.py                   # all 16 configs
    python ex1/plot_sweep5.py 1 11              # only train1, train11
    python ex1/plot_sweep5.py --combined        # add a combined val-loss plot
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Matches lines like:
#   "Elapsed: 12.34s Epoch 100 TrainOpt 1.2345e-04 Valid 5.6789e-03 LR: 1.0000e-04"
#   "Elapsed: 12.34s Epoch 101 TrainOpt 1.1234e-04 Valid skipped LR: 1.0000e-04"
LOG_PATTERN = re.compile(
    r"Epoch\s+(\d+)\s+TrainOpt\s+([\d.eE+-]+)\s+Valid\s+([\d.eE+-]+|skipped)"
)

DEFAULT_CONFIGS = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18]


def parse_log(path: Path):
    epochs, trains, vals = [], [], []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LOG_PATTERN.search(line)
            if not m:
                continue
            epochs.append(int(m.group(1)))
            trains.append(float(m.group(2)))
            v = m.group(3)
            vals.append(float(v) if v != "skipped" else None)
    return epochs, trains, vals


def plot_one(config_num: int, ex1_dir: Path):
    log_path = ex1_dir / f"train{config_num}.log"
    if not log_path.exists():
        print(f"  SKIP train{config_num}: {log_path.name} not found")
        return

    epochs, trains, vals = parse_log(log_path)
    if not epochs:
        print(f"  SKIP train{config_num}: no parseable epoch lines in log")
        return

    val_epochs = [e for e, v in zip(epochs, vals) if v is not None and v > 0]
    val_vals = [v for v in vals if v is not None and v > 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(epochs, trains, label="train", alpha=0.55, linewidth=1.0)
    if val_vals:
        ax.semilogy(val_epochs, val_vals, label="val", linewidth=2.0, color="C1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title(f"train{config_num} loss curves")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path = ex1_dir / f"train{config_num}_loss.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    best_val = min(val_vals) if val_vals else float("nan")
    print(f"  wrote {out_path.name}  (epochs: {len(epochs)}, best val: {best_val:.3e})")


def plot_combined(config_nums, ex1_dir: Path):
    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = 0
    for n in config_nums:
        log_path = ex1_dir / f"train{n}.log"
        if not log_path.exists():
            continue
        epochs, trains, vals = parse_log(log_path)
        if not epochs:
            continue
        val_epochs = [e for e, v in zip(epochs, vals) if v is not None and v > 0]
        val_vals = [v for v in vals if v is not None and v > 0]
        if val_vals:
            ax.semilogy(val_epochs, val_vals, label=f"train{n}", alpha=0.9)
            plotted += 1
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val loss (log scale)")
    ax.set_title("Sweep 5 — validation loss across configs")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    out_path = ex1_dir / "sweep5_val_combined.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  wrote {out_path.name}  ({plotted} configs plotted)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "configs",
        type=int,
        nargs="*",
        help="Config numbers to plot (default: all 16)",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also generate a combined val-loss plot",
    )
    args = parser.parse_args()

    ex1_dir = Path(__file__).parent
    config_nums = args.configs or DEFAULT_CONFIGS

    print(f"Plotting {len(config_nums)} configs in {ex1_dir}")
    for n in config_nums:
        plot_one(n, ex1_dir)
    if args.combined:
        plot_combined(config_nums, ex1_dir)


if __name__ == "__main__":
    main()
