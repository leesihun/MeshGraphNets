"""Compare a generated rollout HDF5 against a ground-truth HDF5.

For the chosen feature it draws three panels — prediction, ground truth, and
their difference — with the L2 norm ||pred - GT|| (and the relative L2,
||pred - GT|| / ||GT||) shown in the figure title. A per-channel metric table
(MAE / RMSE / L2 / rel-L2 / Pearson corr) is also printed to stdout.

The "final state" of each file is used: the last timestep of the rollout
(t = -1, the most-evolved prediction) is compared against the last timestep of
the GT. Files must share the same mesh (identical node count); node ordering is
assumed to match, which holds when the rollout was run on the same mesh the GT
was generated from.

Usage:
    python ex1/compare_gt.py <rollout.h5> [--gt dataset/hex_GT.h5]
    python ex1/compare_gt.py outputs/rollout/hex/model1/rollout_sample0_steps1.h5
    python ex1/compare_gt.py <rollout.h5> --gt dataset/hex_GT.h5 -f 6 -o out.png
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_FEATURE_NAMES = ["x", "y", "z", "x_disp", "y_disp", "z_disp", "stress", "part"]
PHYSICAL_CHANNELS = [3, 4, 5, 6]  # x_disp, y_disp, z_disp, stress


def _decode(s):
    return s.decode("utf-8", "replace") if isinstance(s, bytes) else str(s)


def load_final(path: Path):
    """Return (sample_id, nodal_data_final [F, N], coords (x, y), feature_names).

    nodal_data_final is the last timestep of the first numeric sample.
    """
    with h5py.File(path, "r") as f:
        keys = sorted([k for k in f["data"].keys() if k.isdigit()], key=int)
        if not keys:
            raise ValueError(f"No numeric sample keys under data/ in {path}")
        key = keys[0]
        nd = f[f"data/{key}/nodal_data"][:]  # [F, T, N]
        names = list(DEFAULT_FEATURE_NAMES)
        if "metadata" in f and "feature_names" in f["metadata"]:
            names = [_decode(n) for n in f["metadata/feature_names"][:]]
    final = nd[:, -1, :]  # [F, N]
    coords = (nd[0, 0], nd[1, 0])  # reference coords from first timestep
    return int(key), final, coords, names


def metrics(pred, gt):
    diff = pred - gt
    mae = float(np.abs(diff).mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    l2 = float(np.linalg.norm(diff))
    denom = float(np.linalg.norm(gt))
    rel = l2 / denom if denom > 0 else float("nan")
    corr = (
        float(np.corrcoef(pred, gt)[0, 1])
        if pred.std() > 0 and gt.std() > 0
        else float("nan")
    )
    return mae, rmse, l2, rel, corr


def print_table(pred_all, gt_all, names):
    hdr = f"{'idx':>3} {'channel':10s} {'MAE':>11} {'RMSE':>11} {'L2':>11} {'relL2':>8} {'corr':>7}"
    print(hdr)
    print("-" * len(hdr))
    for c in PHYSICAL_CHANNELS:
        name = names[c] if c < len(names) else f"ch{c}"
        mae, rmse, l2, rel, corr = metrics(pred_all[c], gt_all[c])
        print(f"{c:>3} {name:10s} {mae:11.4g} {rmse:11.4g} {l2:11.4g} {rel:8.3f} {corr:7.3f}")


def plot(pred, gt, coords, feature_name, l2, rel, corr, title_prefix, out_path):
    x, y = coords
    diff = pred - gt
    vmin = min(float(pred.min()), float(gt.min()))
    vmax = max(float(pred.max()), float(gt.max()))
    dmax = float(np.abs(diff).max()) or 1.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
    panels = [
        (pred, "prediction", "jet", vmin, vmax),
        (gt, "ground truth", "jet", vmin, vmax),
        (diff, "pred - GT", "coolwarm", -dmax, dmax),
    ]
    for ax, (vals, sub, cmap, lo, hi) in zip(axes, panels):
        sc = ax.scatter(x, y, c=vals, cmap=cmap, s=4, vmin=lo, vmax=hi)
        ax.set_title(sub, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(sc, ax=ax, shrink=0.8)

    fig.suptitle(
        f"{title_prefix} — {feature_name}   "
        f"L2={l2:.4g}   relL2={rel:.3f}   corr={corr:.3f}",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rollout", type=Path, help="Generated rollout HDF5 file.")
    ap.add_argument("--gt", type=Path, default=Path("dataset/hex_GT.h5"),
                    help="Ground-truth HDF5 (default: dataset/hex_GT.h5).")
    ap.add_argument("-f", "--feature", type=int, default=6,
                    help="Feature index to plot (default 6 = stress).")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="Output PNG path (default: <rollout_stem>_vs_GT_<feat>.png "
                         "next to the rollout file).")
    args = ap.parse_args()

    if not args.rollout.exists():
        print(f"ERROR: rollout not found: {args.rollout}", file=sys.stderr)
        return 1
    if not args.gt.exists():
        print(f"ERROR: GT not found: {args.gt}", file=sys.stderr)
        return 1

    sid, pred_all, coords, names = load_final(args.rollout)
    _, gt_all, _, gt_names = load_final(args.gt)

    if pred_all.shape[1] != gt_all.shape[1]:
        print(f"ERROR: node-count mismatch — rollout has {pred_all.shape[1]} nodes, "
              f"GT has {gt_all.shape[1]}. Cannot compare node-wise (different mesh).",
              file=sys.stderr)
        return 2

    fname = names[args.feature] if args.feature < len(names) else f"feature_{args.feature}"
    print(f"Rollout: {args.rollout.name} (sample {sid}, {pred_all.shape[1]} nodes)")
    print(f"GT     : {args.gt.name}\n")
    print_table(pred_all, gt_all, names)

    mae, rmse, l2, rel, corr = metrics(pred_all[args.feature], gt_all[args.feature])
    out = args.out or args.rollout.with_name(f"{args.rollout.stem}_vs_GT_{fname}.png")
    plot(pred_all[args.feature], gt_all[args.feature], coords, fname,
         l2, rel, corr, f"{args.rollout.name} vs {args.gt.name}", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
