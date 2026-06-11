"""Plot rollout HDF5 files — spatial stress field at multiple timesteps.

Produces two PNG files per rollout:

  1. {stem}_stress.png          — rollout prediction only
  2. {stem}_stress_vs_GT.png    — rollout vs ground-truth side-by-side;
                                  each column title shows the L2 norm of
                                  (pred - GT) at that timestep.

GT defaults to dataset/hex_GT.h5 (relative to cwd). Override with --gt.

Usage:
    python ex1/plot_rollout.py outputs/rollout/*.h5
    python ex1/plot_rollout.py outputs/rollout
    python ex1/plot_rollout.py rollout_sample0_steps10.h5 -n 6
    python ex1/plot_rollout.py outputs/rollout --gt path/to/other_GT.h5
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
DEFAULT_GT_PATH = Path("dataset/hex_GT.h5")


def _decode(s):
    if isinstance(s, bytes):
        return s.decode("utf-8", errors="replace")
    return str(s)


def load_rollout(path: Path):
    """Return (sample_id, nodal_data [F, T, N], mesh_edge [2, E], feature_names)."""
    with h5py.File(path, "r") as f:
        sample_keys = sorted([k for k in f["data"].keys() if k.isdigit()], key=int)
        if not sample_keys:
            raise ValueError(f"No numeric sample keys under data/ in {path}")
        key = sample_keys[0]
        nodal_data = f[f"data/{key}/nodal_data"][:]
        mesh_edge = f[f"data/{key}/mesh_edge"][:]
        sample_id = int(key)
        feature_names = list(DEFAULT_FEATURE_NAMES)
        if "metadata" in f and "feature_names" in f["metadata"]:
            feature_names = [_decode(n) for n in f["metadata/feature_names"][:]]
    return sample_id, nodal_data, mesh_edge, feature_names


def load_gt_field(gt_path: Path, sample_id: int, feature_idx: int):
    """Return GT field [T_gt, N] or None."""
    if not gt_path or not gt_path.exists():
        return None
    try:
        with h5py.File(gt_path, "r") as f:
            grp = f.get(f"data/{sample_id}")
            if grp is None:
                # Try first available sample as fallback
                keys = sorted([k for k in f["data"].keys() if k.isdigit()], key=int)
                if not keys:
                    return None
                grp = f[f"data/{keys[0]}"]
            data = grp["nodal_data"][:]  # [F, T, N]
            if feature_idx >= data.shape[0]:
                return None
            return data[feature_idx]  # [T_gt, N]
    except Exception as exc:
        print(f"  (GT load skipped: {exc})", file=sys.stderr)
        return None


def _pick_timesteps(T: int, num_panels: int):
    if T <= num_panels:
        return np.arange(T)
    return np.unique(np.linspace(0, T - 1, num_panels, dtype=int))


def _gt_at(gt_field, t: int):
    """Return GT snapshot at timestep t; broadcasts single-step GT."""
    if gt_field is None:
        return None
    t_gt = min(t, gt_field.shape[0] - 1)
    return gt_field[t_gt]


# ---------------------------------------------------------------------------
# Plot 1: rollout only
# ---------------------------------------------------------------------------

def plot_rollout_only(path: Path, feature_idx: int = 6, num_panels: int = 6):
    sample_id, nodal_data, _edges, feature_names = load_rollout(path)
    F, T, N = nodal_data.shape

    feature_name = (
        feature_names[feature_idx] if feature_idx < len(feature_names)
        else f"feature_{feature_idx}"
    )
    field = nodal_data[feature_idx]  # [T, N]
    x, y = nodal_data[0, 0], nodal_data[1, 0]

    timesteps = _pick_timesteps(T, num_panels)
    ncols = len(timesteps)

    fig, axes = plt.subplots(
        1, ncols,
        figsize=(max(12.0, 2.6 * ncols), 4.0),
        squeeze=False,
        constrained_layout=True,
    )

    vmin, vmax = float(field.min()), float(field.max())
    sc = None
    for col, t in enumerate(timesteps):
        ax = axes[0, col]
        sc = ax.scatter(x, y, c=field[t], cmap="jet", s=4, vmin=vmin, vmax=vmax)
        ax.set_title(f"t = {t}", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(feature_name, fontsize=9)

    if sc is not None:
        fig.colorbar(sc, ax=axes[0, :].tolist(), shrink=0.85, pad=0.01,
                     location="right", label=feature_name)

    fig.suptitle(f"{path.name} — sample {sample_id}, {feature_name}", fontsize=11)

    out = path.with_name(f"{path.stem}_{feature_name}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------------------------------------------------------------------------
# Plot 2: rollout vs GT, per-column L2 title
# ---------------------------------------------------------------------------

def plot_comparison(
    path: Path,
    gt_path: Path,
    feature_idx: int = 6,
    num_panels: int = 6,
):
    sample_id, nodal_data, _edges, feature_names = load_rollout(path)
    F, T, N = nodal_data.shape

    feature_name = (
        feature_names[feature_idx] if feature_idx < len(feature_names)
        else f"feature_{feature_idx}"
    )
    field = nodal_data[feature_idx]  # [T, N]
    x, y = nodal_data[0, 0], nodal_data[1, 0]

    gt_field = load_gt_field(gt_path, sample_id, feature_idx)
    if gt_field is None:
        print(f"  SKIP comparison: GT not found in {gt_path}", file=sys.stderr)
        return

    timesteps = _pick_timesteps(T, num_panels)
    ncols = len(timesteps)

    # shared colour scale across both rows
    vmin = min(float(field.min()), float(gt_field.min()))
    vmax = max(float(field.max()), float(gt_field.max()))

    fig, axes = plt.subplots(
        2, ncols,
        figsize=(max(12.0, 2.6 * ncols), 8.0),
        squeeze=False,
        constrained_layout=True,
    )

    sc = None
    for col, t in enumerate(timesteps):
        gt_snap = _gt_at(gt_field, t)
        l2 = float(np.linalg.norm(field[t] - gt_snap))

        # Row 0: prediction
        ax_pred = axes[0, col]
        sc = ax_pred.scatter(x, y, c=field[t], cmap="jet", s=4, vmin=vmin, vmax=vmax)
        ax_pred.set_title(f"t={t}  L2={l2:.4g}", fontsize=9)
        ax_pred.set_aspect("equal")
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        if col == 0:
            ax_pred.set_ylabel(f"prediction\n{feature_name}", fontsize=9)

        # Row 1: GT
        ax_gt = axes[1, col]
        ax_gt.scatter(x, y, c=gt_snap, cmap="jet", s=4, vmin=vmin, vmax=vmax)
        ax_gt.set_aspect("equal")
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])
        if col == 0:
            ax_gt.set_ylabel(f"ground truth\n{feature_name}", fontsize=9)

    if sc is not None:
        fig.colorbar(sc, ax=axes[:, :].ravel().tolist(), shrink=0.85, pad=0.01,
                     location="right", label=feature_name)

    fig.suptitle(
        f"{path.name} vs {gt_path.name} — sample {sample_id}, {feature_name}",
        fontsize=11,
    )

    out = path.with_name(f"{path.stem}_{feature_name}_vs_GT.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_paths(inputs):
    paths = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            paths.extend(sorted(p.glob("rollout_sample*_steps*.h5")))
        elif p.exists():
            paths.append(p)
        else:
            print(f"  WARN: {p} not found", file=sys.stderr)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="+",
        help="Rollout HDF5 file(s) or directory containing rollout_sample*.h5"
    )
    parser.add_argument(
        "-f", "--feature", type=int, default=6,
        help="Feature index to plot (default 6 = stress)."
    )
    parser.add_argument(
        "-n", "--num-panels", type=int, default=6,
        help="Number of timestep panels (default 6)."
    )
    parser.add_argument(
        "--gt", type=Path, default=DEFAULT_GT_PATH,
        help=f"Path to ground-truth HDF5 (default: {DEFAULT_GT_PATH})."
    )
    # back-compat alias
    parser.add_argument("--truth", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    gt_path = args.truth if args.truth is not None else args.gt

    paths = collect_paths(args.paths)
    if not paths:
        print("No rollout HDF5 files found.")
        return 1

    print(f"Plotting {len(paths)} rollout file(s)")
    for path in paths:
        try:
            plot_rollout_only(path, feature_idx=args.feature, num_panels=args.num_panels)
            plot_comparison(path, gt_path, feature_idx=args.feature, num_panels=args.num_panels)
        except Exception as exc:
            print(f"  ERROR {path}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
