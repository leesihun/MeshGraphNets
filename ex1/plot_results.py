"""
Plot normalized prediction and ground-truth test results in one figure.

By default this selects the latest numeric folder under outputs/test/<N> and the
latest numeric epoch inside it, then renders all sample*.h5 files from that
epoch. Only faces/predicted_norm and faces/target_norm are used.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation


FEATURE_NAMES = ("Delta X", "Delta Y", "Delta Z", "Stress")


@dataclass
class SampleResult:
    path: Path
    sample_id: int
    pos: np.ndarray
    faces: np.ndarray
    predicted_norm: np.ndarray
    target_norm: np.ndarray


def numeric_dirs(path: Path) -> list[Path]:
    dirs = [p for p in path.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(dirs, key=lambda p: int(p.name))


def latest_numeric_dir(path: Path, label: str) -> Path:
    dirs = numeric_dirs(path)
    if not dirs:
        raise FileNotFoundError(f"No numeric {label} folders found under {path}")
    return dirs[-1]


def sample_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.startswith("sample") and stem[6:].isdigit():
        return int(stem[6:]), stem
    return 10**12, stem


def resolve_test_dir(outputs_test: Path, test_id: str | None, epoch: str | None) -> Path:
    if test_id is None or test_id == "latest":
        test_dir = latest_numeric_dir(outputs_test, "test id")
    else:
        test_dir = outputs_test / str(test_id)

    if not test_dir.exists():
        raise FileNotFoundError(f"Test folder does not exist: {test_dir}")

    if epoch is None or epoch == "latest":
        epoch_dir = latest_numeric_dir(test_dir, "epoch")
    else:
        epoch_dir = test_dir / str(epoch)

    if not epoch_dir.exists():
        raise FileNotFoundError(f"Epoch folder does not exist: {epoch_dir}")
    return epoch_dir


def load_sample(path: Path) -> SampleResult:
    with h5py.File(path, "r") as f:
        missing = [
            name
            for name in ("nodes/pos", "faces/index", "faces/predicted_norm", "faces/target_norm")
            if name not in f
        ]
        if missing:
            raise KeyError(f"{path} is missing required datasets: {', '.join(missing)}")

        sample_id = int(f.attrs.get("sample_id", sample_sort_key(path)[0]))
        return SampleResult(
            path=path,
            sample_id=sample_id,
            pos=f["nodes/pos"][:],
            faces=f["faces/index"][:].astype(np.int64, copy=False),
            predicted_norm=f["faces/predicted_norm"][:],
            target_norm=f["faces/target_norm"][:],
        )


def resolve_features(num_features: int, feature_indices: str) -> list[int]:
    if feature_indices.lower() == "all":
        return list(range(num_features))

    selected = []
    for raw in feature_indices.split(","):
        idx = int(raw.strip())
        if idx < 0:
            idx += num_features
        if idx < 0 or idx >= num_features:
            raise ValueError(f"Feature index {raw} is out of range for {num_features} features")
        selected.append(idx)
    return selected


def feature_label(idx: int) -> str:
    if idx < len(FEATURE_NAMES):
        return FEATURE_NAMES[idx]
    return f"Feature {idx}"


def color_limits(samples: list[SampleResult], feature_idx: int) -> tuple[float, float]:
    values = []
    for sample in samples:
        values.append(sample.predicted_norm[:, feature_idx])
        values.append(sample.target_norm[:, feature_idx])

    vmin = min(float(np.nanmin(v)) for v in values)
    vmax = max(float(np.nanmax(v)) for v in values)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError(f"Feature {feature_idx} contains non-finite color limits")
    if vmax - vmin < 1e-12:
        vmax = vmin + 1e-12
    return vmin, vmax


def draw_mesh(ax, triangulation, values, vmin, vmax):
    image = ax.tripcolor(
        triangulation,
        facecolors=values,
        shading="flat",
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color("0.7")
    return image


def plot_combined(samples: list[SampleResult], features: list[int], output_path: Path, title: str):
    n_rows = len(samples)
    n_cols = len(features) * 2
    fig_w = max(8.0, 3.1 * n_cols)
    fig_h = max(4.0, 3.0 * n_rows + 1.2)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        constrained_layout=True,
    )

    limits = {feature_idx: color_limits(samples, feature_idx) for feature_idx in features}
    mappables = {}

    for row, sample in enumerate(samples):
        triangulation = Triangulation(sample.pos[:, 0], sample.pos[:, 1], sample.faces)
        for feat_pos, feature_idx in enumerate(features):
            vmin, vmax = limits[feature_idx]
            pred_ax = axes[row, feat_pos * 2]
            truth_ax = axes[row, feat_pos * 2 + 1]

            pred_values = sample.predicted_norm[:, feature_idx]
            truth_values = sample.target_norm[:, feature_idx]

            mappables[feature_idx] = draw_mesh(pred_ax, triangulation, pred_values, vmin, vmax)
            draw_mesh(truth_ax, triangulation, truth_values, vmin, vmax)

            mse = float(np.mean((pred_values - truth_values) ** 2))
            truth_ax.text(
                0.02,
                0.03,
                f"MSE {mse:.3g}",
                transform=truth_ax.transAxes,
                fontsize=7,
                color="0.05",
                bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.5},
            )

            if row == 0:
                label = feature_label(feature_idx)
                pred_ax.set_title(f"{label}\nPrediction", fontsize=9)
                truth_ax.set_title(f"{label}\nGround truth", fontsize=9)

        axes[row, 0].set_ylabel(f"Sample {sample.sample_id}", fontsize=9)

    for feat_pos, feature_idx in enumerate(features):
        cols = axes[:, feat_pos * 2 : feat_pos * 2 + 2].ravel()
        cbar = fig.colorbar(
            mappables[feature_idx],
            ax=cols.tolist(),
            shrink=0.72,
            pad=0.01,
            aspect=24,
        )
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(f"{feature_label(feature_idx)} normalized", fontsize=8)

    fig.suptitle(title, fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all samples from a test output folder using normalized prediction/target data."
    )
    parser.add_argument("--outputs-test", default="outputs/test", type=Path)
    parser.add_argument("--test-id", default="latest", help="Numeric test id under outputs/test, or latest")
    parser.add_argument("--epoch", default="latest", help="Numeric epoch under outputs/test/<test-id>, or latest")
    parser.add_argument(
        "--features",
        default="all",
        help="Comma-separated feature indices, -1 for last feature, or all",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    epoch_dir = resolve_test_dir(args.outputs_test, args.test_id, args.epoch)
    sample_paths = sorted(epoch_dir.glob("sample*.h5"), key=sample_sort_key)
    if not sample_paths:
        raise FileNotFoundError(f"No sample*.h5 files found in {epoch_dir}")

    samples = [load_sample(path) for path in sample_paths]
    num_features = samples[0].predicted_norm.shape[1]
    for sample in samples:
        if sample.predicted_norm.shape != sample.target_norm.shape:
            raise ValueError(f"Shape mismatch in {sample.path}: prediction and ground truth differ")
        if sample.predicted_norm.shape[1] != num_features:
            raise ValueError(f"Feature count mismatch in {sample.path}")

    features = resolve_features(num_features, args.features)
    output_path = args.output
    if output_path is None:
        feature_tag = "all_features" if args.features.lower() == "all" else "features_" + args.features.replace(",", "_")
        output_path = epoch_dir / f"all_samples_normalized_{feature_tag}.png"

    rel_epoch_dir = epoch_dir.as_posix()
    title = f"{rel_epoch_dir}: normalized prediction vs ground truth ({len(samples)} samples)"
    print(f"Rendering {len(samples)} samples from {epoch_dir}")
    print(f"Features: {', '.join(str(i) for i in features)}")
    plot_combined(samples, features, output_path, title)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()