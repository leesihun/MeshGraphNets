"""Plot rollout HDF5 files — spatial field at multiple timesteps.

Reads rollout HDF5 files (written by inference_profiles/rollout.py) and
writes per-file PNGs showing the chosen feature (default: stress) at
evenly-spaced timesteps.

Rollout HDF5 layout (from rollout.py):
    data/{sample_id}/nodal_data   shape [F, T, N]   F = 3 + output_var + 1
                                                    (x, y, z, ...predictions, part)
    data/{sample_id}/mesh_edge    shape [2, E]
    metadata/feature_names

Usage:
    python ex1/plot_rollout.py outputs/rollout/*.h5
    python ex1/plot_rollout.py outputs/rollout         # all rollouts in dir
    python ex1/plot_rollout.py rollout_sample0_steps10.h5 -f 6 -n 6
    python ex1/plot_rollout.py outputs/rollout --truth dataset/ex1.h5   # compare to ground truth
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


def _decode(s):
    if isinstance(s, bytes):
        return s.decode("utf-8", errors="replace")
    return str(s)


def load_rollout(path: Path):
    """Return (sample_id, nodal_data [F, T, N], mesh_edge [2, E], feature_names)."""
    with h5py.File(path, "r") as f:
        sample_keys = sorted(
            [k for k in f["data"].keys() if k.isdigit()], key=int
        )
        if not sample_keys:
            raise ValueError(f"No numeric sample keys under data/ in {path}")
        sample_key = sample_keys[0]
        nodal_data = f[f"data/{sample_key}/nodal_data"][:]
        mesh_edge = f[f"data/{sample_key}/mesh_edge"][:]
        sample_id = int(sample_key)

        feature_names = list(DEFAULT_FEATURE_NAMES)
        if "metadata" in f and "feature_names" in f["metadata"]:
            feature_names = [_decode(n) for n in f["metadata/feature_names"][:]]

    return sample_id, nodal_data, mesh_edge, feature_names


def load_truth(truth_path: Path, sample_id: int, feature_idx: int, num_timesteps: int):
    """Load ground-truth feature trajectory from input dataset, if available."""
    if not truth_path or not truth_path.exists():
        return None
    try:
        with h5py.File(truth_path, "r") as f:
            grp = f.get(f"data/{sample_id}")
            if grp is None:
                return None
            data = grp["nodal_data"][:]  # [F, T, N]
            if feature_idx >= data.shape[0]:
                return None
            return data[feature_idx, :num_timesteps, :]
    except Exception as exc:
        print(f"  (truth load skipped: {exc})", file=sys.stderr)
        return None


def plot_field(
    path: Path,
    feature_idx: int = 6,
    num_panels: int = 6,
    truth_path: Path | None = None,
):
    sample_id, nodal_data, _mesh_edge, feature_names = load_rollout(path)
    F, T, N = nodal_data.shape

    if feature_idx < 0:
        feature_idx = F + feature_idx
    if not 0 <= feature_idx < F:
        raise IndexError(f"Feature index {feature_idx} out of range [0, {F})")

    pos_ref = nodal_data[:3, 0, :]  # [3, N] reference positions at t=0
    x, y = pos_ref[0], pos_ref[1]

    field = nodal_data[feature_idx]  # [T, N]
    feature_name = (
        feature_names[feature_idx]
        if feature_idx < len(feature_names)
        else f"feature_{feature_idx}"
    )

    timesteps = np.linspace(0, T - 1, num_panels, dtype=int)

    truth = load_truth(truth_path, sample_id, feature_idx, T) if truth_path else None
    has_truth = truth is not None

    # Layout: rows = (1 prediction) or (prediction + truth + error), cols = timesteps
    n_rows = 3 if has_truth else 1
    fig_h = 4.0 * n_rows
    fig_w = max(12.0, 2.6 * num_panels)
    fig, axes = plt.subplots(
        n_rows,
        num_panels,
        figsize=(fig_w, fig_h),
        squeeze=False,
        constrained_layout=True,
    )

    vmin = float(field.min())
    vmax = float(field.max())
    if has_truth:
        vmin = min(vmin, float(truth.min()))
        vmax = max(vmax, float(truth.max()))

    pred_sc = err_sc = None
    err_max = 0.0
    if has_truth:
        err_max = float(np.abs(field[:, :truth.shape[1]] - truth).max())

    for col, t in enumerate(timesteps):
        ax_pred = axes[0, col]
        pred_sc = ax_pred.scatter(
            x, y, c=field[t], cmap="jet", s=4, vmin=vmin, vmax=vmax
        )
        ax_pred.set_title(f"t = {t}/{T - 1}", fontsize=9)
        ax_pred.set_aspect("equal")
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        if col == 0:
            ax_pred.set_ylabel(f"prediction\n{feature_name}", fontsize=9)

        if has_truth:
            ax_truth = axes[1, col]
            ax_err = axes[2, col]
            t_truth = min(t, truth.shape[0] - 1)
            ax_truth.scatter(
                x, y, c=truth[t_truth], cmap="jet", s=4, vmin=vmin, vmax=vmax
            )
            ax_truth.set_aspect("equal")
            ax_truth.set_xticks([])
            ax_truth.set_yticks([])
            if col == 0:
                ax_truth.set_ylabel(f"ground truth\n{feature_name}", fontsize=9)

            err = field[t] - truth[t_truth]
            err_sc = ax_err.scatter(
                x, y, c=err, cmap="RdBu_r", s=4, vmin=-err_max, vmax=err_max
            )
            mse = float(np.mean(err ** 2))
            ax_err.text(
                0.02, 0.03, f"MSE {mse:.3g}",
                transform=ax_err.transAxes, fontsize=7,
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1.5},
            )
            ax_err.set_aspect("equal")
            ax_err.set_xticks([])
            ax_err.set_yticks([])
            if col == 0:
                ax_err.set_ylabel("error (pred - truth)", fontsize=9)

    if pred_sc is not None:
        cbar = fig.colorbar(
            pred_sc, ax=axes[0, :].tolist(), shrink=0.85, pad=0.01, location="right"
        )
        cbar.set_label(feature_name, fontsize=8)
    if err_sc is not None:
        cbar2 = fig.colorbar(
            err_sc, ax=axes[2, :].tolist(), shrink=0.85, pad=0.01, location="right"
        )
        cbar2.set_label(f"{feature_name} error", fontsize=8)

    fig.suptitle(f"{path.name} — sample {sample_id}, feature: {feature_name}", fontsize=11)

    out_path = path.with_name(f"{path.stem}_{feature_name}.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.name}")


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
        help="Feature index to plot. Default 6 = stress. Negative = from end."
    )
    parser.add_argument(
        "-n", "--num-panels", type=int, default=6,
        help="Number of timestep panels (default 6)."
    )
    parser.add_argument(
        "--truth", type=Path, default=None,
        help="Path to ground-truth dataset HDF5 (e.g. dataset/ex1.h5). "
             "When provided, plots ground truth and error rows."
    )
    args = parser.parse_args()

    paths = collect_paths(args.paths)
    if not paths:
        print("No rollout HDF5 files found.")
        return 1

    print(f"Plotting {len(paths)} rollout file(s)")
    for path in paths:
        try:
            plot_field(
                path,
                feature_idx=args.feature,
                num_panels=args.num_panels,
                truth_path=args.truth,
            )
        except Exception as exc:
            print(f"  ERROR {path}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
