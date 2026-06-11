"""Compare a rollout HDF5 against a ground-truth HDF5: stress relative L2 + plot.

Usage:
    python compare_rollout_gt.py <rollout.h5> <gt.h5> [--plot-dir DIR] [--name NAME]

Prints one line to stdout: "<stress_relL2>"
Comparison is between the final rollout timestep and the final GT timestep,
on the stress channel only (nodal_data[6]).

With --plot-dir, saves "<NAME>_L2_<stress_relL2>.png": GT vs prediction vs
|error| panels for stress and displacement magnitude, stress L2 in the title.
"""
import argparse
import os
import sys

import h5py
import numpy as np


def _load(path):
    with h5py.File(path, 'r') as f:
        sample_id = sorted(f['data'].keys(), key=int)[0]
        nodal = f[f'data/{sample_id}/nodal_data'][:]
    # [features, time, nodes]: reference xy + physical channels at last timestep
    xy = nodal[0:2, -1, :].T
    state = nodal[3:7, -1, :]
    return xy, state


def _rel_l2(pred, gt):
    denom = np.linalg.norm(gt)
    err = np.linalg.norm(pred - gt)
    return float(err / denom) if denom > 0 else float(err)


def _panel(fig, ax, xy, vals, cmap, title, vmin=None, vmax=None):
    if vmin is None:
        vmin, vmax = vals.min(), vals.max()
        if vmin == vmax:
            vmin -= 1e-6
            vmax += 1e-6
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=vals, cmap=cmap,
                    s=0.8, linewidths=0, rasterized=True,
                    vmin=vmin, vmax=vmax)
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=9)


def _plot(xy, pred, gt, stress_l2, name, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = [
        ('stress (MPa)', gt[3], pred[3], 'hot'),
        ('|disp| (mm)', np.linalg.norm(gt[:3], axis=0),
         np.linalg.norm(pred[:3], axis=0), 'plasma'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"{name}  —  stress relL2 = {stress_l2:.6f}", fontsize=12)
    for ax_row, (label, gt_vals, pred_vals, cmap) in zip(axes, rows):
        vmin, vmax = gt_vals.min(), gt_vals.max()
        if vmin == vmax:
            vmin -= 1e-6
            vmax += 1e-6
        _panel(fig, ax_row[0], xy, gt_vals, cmap, f"GT {label}", vmin, vmax)
        _panel(fig, ax_row[1], xy, pred_vals, cmap, f"Prediction {label}", vmin, vmax)
        _panel(fig, ax_row[2], xy, np.abs(pred_vals - gt_vals), 'viridis',
               f"|error| {label}")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved: {out_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rollout')
    parser.add_argument('gt')
    parser.add_argument('--plot-dir', default=None,
                        help='directory to save the GT-vs-prediction PNG')
    parser.add_argument('--name', default='rollout',
                        help='label used in the plot title and PNG filename')
    args = parser.parse_args()

    xy, pred = _load(args.rollout)
    _, gt = _load(args.gt)

    if pred.shape != gt.shape:
        print(f"ERROR: shape mismatch pred {pred.shape} vs gt {gt.shape}", file=sys.stderr)
        sys.exit(1)

    stress_l2 = _rel_l2(pred[3], gt[3])
    print(f"{stress_l2:.6f}")

    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        out_path = os.path.join(args.plot_dir, f"{args.name}_L2_{stress_l2:.6f}.png")
        _plot(xy, pred, gt, stress_l2, args.name, out_path)


if __name__ == '__main__':
    main()
