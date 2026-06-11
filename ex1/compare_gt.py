"""Compare generated rollouts against a ground-truth HDF5.

Two modes:

1. Single file — compare one existing rollout .h5 to the GT:
       python ex1/compare_gt.py outputs/rollout/hex/model1/rollout_sample0_steps1.h5

2. --all — find every .pth under a models root (default: outputs/),
   run inference on each (on --infer-dataset, default dataset/hex_dataset.h5),
   compare each rollout to the GT, and print a ranked summary:
       python ex1/compare_gt.py --all
       python ex1/compare_gt.py --all --models-root outputs --gt dataset/hex_GT.h5

For the chosen feature it draws three panels — prediction, ground truth, and
their difference — with the L2 norm ||pred - GT|| (and relative L2 and Pearson
correlation) in the figure title. A per-channel metric table is printed too.

The "final state" of each file is used: the last timestep of the rollout is
compared against the last timestep of the GT. Files must share the same mesh
(identical node count); node ordering is assumed to match, which holds when the
rollout was run on the same mesh the GT was generated from.

Each checkpoint's architecture is taken from its own stored ``model_config`` by
the rollout, so a single base inference config drives every model regardless of
its multiscale settings.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FEATURE_NAMES = ["x", "y", "z", "x_disp", "y_disp", "z_disp", "stress", "part"]
PHYSICAL_CHANNELS = [3, 4, 5, 6]  # x_disp, y_disp, z_disp, stress


def _decode(s):
    return s.decode("utf-8", "replace") if isinstance(s, bytes) else str(s)


def load_final(path: Path):
    """Return (sample_id, nodal_data_final [F, N], coords (x, y), feature_names)."""
    with h5py.File(path, "r") as f:
        keys = sorted([k for k in f["data"].keys() if k.isdigit()], key=int)
        if not keys:
            raise ValueError(f"No numeric sample keys under data/ in {path}")
        key = keys[0]
        nd = f[f"data/{key}/nodal_data"][:]  # [F, T, N]
        names = list(DEFAULT_FEATURE_NAMES)
        if "metadata" in f and "feature_names" in f["metadata"]:
            names = [_decode(n) for n in f["metadata/feature_names"][:]]
    return int(key), nd[:, -1, :], (nd[0, 0], nd[1, 0]), names


def metrics(pred, gt):
    diff = pred - gt
    mae = float(np.abs(diff).mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    l2 = float(np.linalg.norm(diff))
    denom = float(np.linalg.norm(gt))
    rel = l2 / denom if denom > 0 else float("nan")
    corr = (
        float(np.corrcoef(pred, gt)[0, 1])
        if pred.std() > 0 and gt.std() > 0 else float("nan")
    )
    return mae, rmse, l2, rel, corr


def print_table(pred_all, gt_all, names):
    hdr = f"{'idx':>3} {'channel':12s} {'MAE':>11} {'RMSE':>11} {'L2':>11} {'relL2':>8} {'corr':>7}"
    print(hdr)
    print("-" * len(hdr))
    for c in PHYSICAL_CHANNELS:
        name = names[c] if c < len(names) else f"ch{c}"
        mae, rmse, l2, rel, corr = metrics(pred_all[c], gt_all[c])
        print(f"{c:>3} {name:12s} {mae:11.4g} {rmse:11.4g} {l2:11.4g} {rel:8.3f} {corr:7.3f}")


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
    return out_path


def compare_one(rollout_path: Path, gt_path: Path, feature_idx: int, out_path: Path):
    """Compare a single rollout to GT. Returns the stress-channel metrics dict or None."""
    sid, pred_all, coords, names = load_final(rollout_path)
    _, gt_all, _, _ = load_final(gt_path)

    if pred_all.shape[1] != gt_all.shape[1]:
        print(f"  ERROR: node-count mismatch — rollout {pred_all.shape[1]} vs "
              f"GT {gt_all.shape[1]} nodes (different mesh).", file=sys.stderr)
        return None

    fname = names[feature_idx] if feature_idx < len(names) else f"feature_{feature_idx}"
    print(f"  rollout: {rollout_path.name} (sample {sid}, {pred_all.shape[1]} nodes)")
    print_table(pred_all, gt_all, names)

    mae, rmse, l2, rel, corr = metrics(pred_all[feature_idx], gt_all[feature_idx])
    written = plot(pred_all[feature_idx], gt_all[feature_idx], coords, fname,
                   l2, rel, corr, f"{rollout_path.name} vs {gt_path.name}", out_path)
    print(f"  wrote {written}")
    return {"feature": fname, "mae": mae, "rmse": rmse, "l2": l2, "rel": rel, "corr": corr}


# ---------------------------------------------------------------------------
# --all: discover checkpoints, run inference, compare
# ---------------------------------------------------------------------------

CONFIG_TEMPLATE = """model   MeshGraphNets
mode    inference
gpu_ids {gpu}
log_file_dir    {out_dir}/infer.log
modelpath   {model_path}
dataset_dir {infer_dataset}
infer_dataset   {infer_dataset}
inference_output_dir   {out_dir}
infer_timesteps  {steps}
input_var   4
output_var  4
edge_var    8
positional_features  4
positional_encoding  rwpe
message_passing_num 15
Latent_dim  128
Batch_size  1
num_workers 2
std_noise   0.0
use_amp             True
use_ema             True
use_node_types  True
use_world_edges         False
% Architecture below is overridden by each checkpoint's model_config at load.
use_multiscale      True
coarsening_type     voronoi_inherit
voronoi_clusters    5000, 1000
multiscale_levels   2
mp_per_level        2, 4, 8, 4, 2
bipartite_unpool    True
"""


def find_checkpoints(root: Path):
    return sorted(root.rglob("*.pth"))


def run_inference(model_path: Path, out_dir: Path, infer_dataset: Path, steps: int, gpu: int):
    """Run MeshGraphNets_main.py inference for one checkpoint. Returns the output .h5 or None."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_text = CONFIG_TEMPLATE.format(
        gpu=gpu,
        out_dir=out_dir.as_posix(),
        model_path=model_path.as_posix(),
        infer_dataset=infer_dataset.as_posix(),
        steps=steps,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     dir=tempfile.gettempdir()) as tf:
        tf.write(cfg_text)
        cfg_path = Path(tf.name)

    env = dict(os.environ, PYTHONUTF8="1")
    try:
        proc = subprocess.run(
            [sys.executable, "MeshGraphNets_main.py", "--config", str(cfg_path)],
            cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
        )
    finally:
        cfg_path.unlink(missing_ok=True)

    if proc.returncode != 0:
        print(f"  INFERENCE FAILED (exit {proc.returncode}):", file=sys.stderr)
        print("\n".join(proc.stdout.splitlines()[-15:]), file=sys.stderr)
        print("\n".join(proc.stderr.splitlines()[-15:]), file=sys.stderr)
        return None

    outputs = sorted(out_dir.glob("rollout_sample*_steps*.h5"),
                     key=lambda p: p.stat().st_mtime)
    return outputs[-1] if outputs else None


def run_all(models_root: Path, gt_path: Path, infer_dataset: Path, steps: int,
            feature_idx: int, gpu: int):
    checkpoints = find_checkpoints(models_root)
    if not checkpoints:
        print(f"No .pth checkpoints found under {models_root}")
        return 1
    print(f"Found {len(checkpoints)} checkpoint(s) under {models_root}:")
    for c in checkpoints:
        print(f"  - {c.relative_to(REPO_ROOT) if c.is_relative_to(REPO_ROOT) else c}")
    print()

    rows = []
    for ck in checkpoints:
        rel = ck.relative_to(models_root)
        tag = rel.with_suffix("").as_posix().replace("/", "__")
        out_dir = REPO_ROOT / "outputs" / "rollout" / "compare_all" / tag
        print(f"=== {rel} ===")
        rollout = run_inference(ck, out_dir, infer_dataset, steps, gpu)
        if rollout is None:
            print("  (skipped — no rollout produced)\n")
            rows.append((tag, None))
            continue
        plot_path = out_dir / f"{rollout.stem}_vs_GT.png"
        m = compare_one(rollout, gt_path, feature_idx, plot_path)
        rows.append((tag, m))
        print()

    # Ranked summary on the chosen feature
    fname = (DEFAULT_FEATURE_NAMES[feature_idx]
             if feature_idx < len(DEFAULT_FEATURE_NAMES) else f"feature_{feature_idx}")
    print("=" * 72)
    print(f"SUMMARY — {fname} (best relL2 first)")
    print("=" * 72)
    print(f"{'model':40s} {'L2':>11} {'relL2':>8} {'corr':>7}")
    print("-" * 72)
    ok = [(t, m) for t, m in rows if m is not None]
    ok.sort(key=lambda r: (np.isnan(r[1]["rel"]), r[1]["rel"]))
    for t, m in ok:
        print(f"{t:40s} {m['l2']:11.4g} {m['rel']:8.3f} {m['corr']:7.3f}")
    for t, m in rows:
        if m is None:
            print(f"{t:40s} {'FAILED':>11}")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rollout", type=Path, nargs="?",
                    help="A rollout HDF5 to compare (single-file mode). "
                         "Omit and pass --all to run every checkpoint.")
    ap.add_argument("--all", action="store_true",
                    help="Find every .pth under --models-root, run inference, compare all.")
    ap.add_argument("--models-root", type=Path, default=REPO_ROOT / "outputs",
                    help="Root to search for .pth checkpoints (default: outputs/).")
    ap.add_argument("--gt", type=Path, default=Path("dataset/hex_GT.h5"),
                    help="Ground-truth HDF5 (default: dataset/hex_GT.h5).")
    ap.add_argument("--infer-dataset", type=Path, default=Path("dataset/hex_dataset.h5"),
                    help="Inference input HDF5 for --all (default: dataset/hex_dataset.h5).")
    ap.add_argument("--steps", type=int, default=1, help="Rollout timesteps (default 1).")
    ap.add_argument("-f", "--feature", type=int, default=6,
                    help="Feature index to plot/rank (default 6 = stress).")
    ap.add_argument("--gpu", type=int, default=0, help="GPU id for inference (default 0).")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="Output PNG path for single-file mode.")
    args = ap.parse_args()

    if not args.gt.exists():
        print(f"ERROR: GT not found: {args.gt}", file=sys.stderr)
        return 1

    if args.all or args.rollout is None:
        return run_all(args.models_root, args.gt, args.infer_dataset,
                       args.steps, args.feature, args.gpu)

    if not args.rollout.exists():
        print(f"ERROR: rollout not found: {args.rollout}", file=sys.stderr)
        return 1
    sid, pred_all, _, names = load_final(args.rollout)
    fname = names[args.feature] if args.feature < len(names) else f"feature_{args.feature}"
    out = args.out or args.rollout.with_name(f"{args.rollout.stem}_vs_GT_{fname}.png")
    print(f"GT: {args.gt.name}\n")
    return 0 if compare_one(args.rollout, args.gt, args.feature, out) else 2


if __name__ == "__main__":
    sys.exit(main())
