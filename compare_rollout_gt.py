"""Compare a rollout HDF5 against a ground-truth HDF5: stress R^2 + elemental plot.

Usage:
    python compare_rollout_gt.py <rollout.h5> <gt.h5> [--plot-dir DIR] [--name NAME]

Prints one line to stdout: "<stress_R2>"
Comparison is between the final rollout timestep and the final GT timestep,
on the stress channel only (nodal_data[6]).
R^2 = 1 - sum((pred-gt)^2) / sum((gt-mean(gt))^2); 1.0 is perfect, 0.0 means
no better than predicting the GT mean everywhere, negative is worse than that.

With --plot-dir, saves "<NAME>_R2_<stress_R2>.png": GT vs prediction vs |error|
rendered as elemental (per-triangle) filled contours, stress R^2 in the title.
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


def _r2(pred, gt):
    ss_res = float(np.sum((pred - gt) ** 2))
    ss_tot = float(np.sum((gt - gt.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def _build_triangulation(xy):
    import matplotlib.tri as mtri
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
    # Delaunay bridges concave gaps/holes; mask triangles with abnormally long edges
    pts = xy[tri.triangles]                      # [n_tri, 3, 2]
    edge_len = np.stack([
        np.linalg.norm(pts[:, 0] - pts[:, 1], axis=1),
        np.linalg.norm(pts[:, 1] - pts[:, 2], axis=1),
        np.linalg.norm(pts[:, 2] - pts[:, 0], axis=1),
    ], axis=1)
    max_edge = edge_len.max(axis=1)
    tri.set_mask(max_edge > 3.0 * np.median(edge_len))
    return tri


def _panel(fig, ax, tri, vals, cmap, title, vmin=None, vmax=None):
    # elemental rendering: one flat color per triangle (mean of its 3 nodes)
    face_vals = vals[tri.triangles].mean(axis=1)
    if tri.mask is not None:
        face_vals = face_vals[~tri.mask]
    if vmin is None:
        vmin, vmax = face_vals.min(), face_vals.max()
        if vmin == vmax:
            vmin -= 1e-6
            vmax += 1e-6
    tpc = ax.tripcolor(tri, facecolors=face_vals, cmap=cmap,
                       vmin=vmin, vmax=vmax, rasterized=True)
    fig.colorbar(tpc, ax=ax, fraction=0.04, pad=0.02)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=9)


def _plot(xy, pred, gt, stress_r2, name, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tri = _build_triangulation(xy)

    rows = [
        ('stress (MPa)', gt[3], pred[3], 'jet'),
        ('|disp| (mm)', np.linalg.norm(gt[:3], axis=0),
         np.linalg.norm(pred[:3], axis=0), 'jet'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"{name}  —  stress R² = {stress_r2:.6f}", fontsize=12)
    for ax_row, (label, gt_vals, pred_vals, cmap) in zip(axes, rows):
        vmin, vmax = gt_vals.min(), gt_vals.max()
        if vmin == vmax:
            vmin -= 1e-6
            vmax += 1e-6
        _panel(fig, ax_row[0], tri, gt_vals, cmap, f"GT {label}", vmin, vmax)
        _panel(fig, ax_row[1], tri, pred_vals, cmap, f"Prediction {label}", vmin, vmax)
        _panel(fig, ax_row[2], tri, np.abs(pred_vals - gt_vals), 'viridis',
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

    stress_r2 = _r2(pred[3], gt[3])
    print(f"{stress_r2:.6f}")

    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        out_path = os.path.join(args.plot_dir, f"{args.name}_R2_{stress_r2:.6f}.png")
        _plot(xy, pred, gt, stress_r2, args.name, out_path)


if __name__ == '__main__':
    main()
