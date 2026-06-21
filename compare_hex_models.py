"""Compare all ex1 model rollouts against hex_GT.h5.

Produces:
  outputs/plots/ex1_comparison/
    model{N}_R2_<r2>.png  — GT | Pred | |Error| per model (shared GT color scale)
    center_stress_comparison.png — stress along y-centerline for all models + GT
    stress_R2_bar.png            — R² bar chart

Usage:
    python compare_hex_models.py
"""

import os
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GT_PATH = 'dataset/hex_GT.h5'

MODELS = {
    'model1': {
        'path': 'outputs/rollout/ex1_model1/rollout_sample0_steps1.h5',
        'label': 'HI-MGN-L4\n(10k/2k/400/80)',
        'color': '#e41a1c',
        'marker': 'o',
    },
    'model2': {
        'path': 'outputs/rollout/ex1_model2/rollout_sample0_steps1.h5',
        'label': 'HI-MGN',
        'color': '#377eb8',
        'marker': 'o',
    },
    'model3': {
        'path': 'outputs/rollout/ex1_model3/rollout_sample0_steps1.h5',
        'label': 'BSMS-GNN',
        'color': '#4daf4a',
        'marker': '^',
    },
    'model4': {
        'path': 'outputs/rollout/ex1_model4/rollout_sample0_steps1.h5',
        'label': 'MGN',
        'color': '#ff7f00',
        'marker': 's',
    },
    'model5': {
        'path': 'outputs/rollout/ex1_model5/rollout_sample0_steps1.h5',
        'label': 'Flat MP=20',
        'color': '#ff7f00',
        'marker': 's',
    },
    'model6': {
        'path': 'outputs/rollout/ex1_model6/rollout_sample0_steps1.h5',
        'label': 'BSMS-GNN-L4',
        'color': '#984ea3',
        'marker': 'D',
    },
}

OUT_DIR = 'outputs/plots/ex1_comparison'

# Stored stress channel is in MPa; multiply to report everything in Pa.
PA_PER_MPA = 1e6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load(path):
    with h5py.File(path, 'r') as f:
        key = sorted(f['data'].keys(), key=int)[0]
        nodal = f[f'data/{key}/nodal_data'][:]
    xy = nodal[0:2, 0, :].T          # ref coords, shape (N, 2)
    state = nodal[3:7, -1, :]        # [xd, yd, zd, stress] at last step, (4, N)
    return xy, state


def _r2(pred, gt):
    r = np.corrcoef(pred, gt)[0, 1]
    return float(r ** 2)


def _build_tri(xy):
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
    pts = xy[tri.triangles]
    edge_len = np.stack([
        np.linalg.norm(pts[:, 0] - pts[:, 1], axis=1),
        np.linalg.norm(pts[:, 1] - pts[:, 2], axis=1),
        np.linalg.norm(pts[:, 2] - pts[:, 0], axis=1),
    ], axis=1)
    max_edge = edge_len.max(axis=1)
    tri.set_mask(max_edge > 3.0 * np.median(edge_len))
    return tri


def _panel(fig, ax, tri, vals, cmap, title, vmin, vmax):
    face_vals = vals[tri.triangles].mean(axis=1)
    if tri.mask is not None:
        face_vals = face_vals[~tri.mask]
    tpc = ax.tripcolor(tri, facecolors=face_vals, cmap=cmap,
                       vmin=vmin, vmax=vmax, rasterized=True)
    fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.01)
    ax.set_aspect('equal')
    ax.margins(0)
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=9)


def _save_single_panel(tri, vals, cmap, vmin, vmax, out_path):
    """Render one field as a standalone PNG (no title, just the contour + colorbar)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    _panel(fig, ax, tri, vals, cmap, '', vmin, vmax)
    ax.margins(0)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def _centerline_mask(xy, tol_frac=0.01):
    y = xy[:, 1]
    cy = y.mean()
    tol = (y.max() - y.min()) * tol_frac
    mask = np.abs(y - cy) < tol
    return mask, xy[:, 0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Optional CLI filter: only plot the named models (e.g. `model2 model3`)
    selected = [a for a in sys.argv[1:] if not a.startswith('-')]
    if selected:
        unknown = [s for s in selected if s not in MODELS]
        if unknown:
            print(f"ERROR: unknown model(s): {unknown}", file=sys.stderr)
            sys.exit(1)
        for name in list(MODELS.keys()):
            if name not in selected:
                del MODELS[name]
        print(f"Filtering to models: {list(MODELS.keys())}")

    # --- Load GT ---
    xy_gt, state_gt = _load(GT_PATH)
    stress_gt = state_gt[3] * PA_PER_MPA          # (N,) in Pa

    gt_stress_min = float(stress_gt.min())
    gt_stress_max = float(stress_gt.max())
    if gt_stress_min == gt_stress_max:
        gt_stress_min -= 1e-12
        gt_stress_max += 1e-12
    print(f"GT stress range: {gt_stress_min:.4e} .. {gt_stress_max:.4e} Pa")

    # error color scale: symmetric about 0, range = GT range
    err_scale = gt_stress_max - gt_stress_min

    # Build Delaunay triangulation once (same mesh for all)
    tri = _build_tri(xy_gt)

    # Center cross-section mask
    cmask, x_all = _centerline_mask(xy_gt)
    x_center = x_all[cmask]
    order = np.argsort(x_center)
    x_sorted = x_center[order]
    gt_stress_center = stress_gt[cmask][order]

    # --- Per-model individual plots ---
    r2_values = {}
    centerline_preds = {}

    for mname, mcfg in MODELS.items():
        mpath = mcfg['path']
        if not os.path.exists(mpath):
            print(f"  WARNING: {mpath} not found, skipping", file=sys.stderr)
            continue

        xy_pred, state_pred = _load(mpath)
        stress_pred = state_pred[3] * PA_PER_MPA   # Pa

        r2 = _r2(stress_pred, stress_gt)
        r2_values[mname] = r2
        centerline_preds[mname] = stress_pred[cmask][order]

        label_oneline = mcfg['label'].replace('\n', ' ')
        print(f"  {mname} ({label_oneline}): stress R² = {r2:.6f}")

        # 3-panel: GT | Pred | |Error| (no titles/subtitles)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        _panel(fig, axes[0], tri, stress_gt, 'jet', '',
               gt_stress_min, gt_stress_max)
        _panel(fig, axes[1], tri, stress_pred, 'jet', '',
               gt_stress_min, gt_stress_max)
        _panel(fig, axes[2], tri, np.abs(stress_pred - stress_gt), 'viridis',
               '', 0.0, err_scale)

        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.05)
        fname = f"{mname}_R2_{r2:.6f}.png"
        out_path = os.path.join(OUT_DIR, fname)
        fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)
        print(f"    Saved: {out_path}")

        # Also save each subplot as a standalone file
        base = f"{mname}_R2_{r2:.6f}"
        panels = [
            (f"{base}_GT.png", stress_gt, 'jet', gt_stress_min, gt_stress_max),
            (f"{base}_pred.png", stress_pred, 'jet', gt_stress_min, gt_stress_max),
            (f"{base}_error.png", np.abs(stress_pred - stress_gt), 'viridis',
             0.0, err_scale),
        ]
        for pfname, pvals, pcmap, pvmin, pvmax in panels:
            ppath = os.path.join(OUT_DIR, pfname)
            _save_single_panel(tri, pvals, pcmap, pvmin, pvmax, ppath)
            print(f"      Saved subplot: {ppath}")

    # --- Combined centerline stress line plot ---
    fig, ax = plt.subplots(figsize=(13, 5))

    # GT as a solid line
    ax.plot(x_sorted, gt_stress_center, color='black', linewidth=2.0,
            linestyle='-', label='Ground Truth', zorder=10)

    # Models as lines with markers (circles for HI-MGN, triangles for BSMS-GNN, etc.)
    markevery = max(1, len(x_sorted) // 40)
    for mname, mcfg in MODELS.items():
        if mname not in centerline_preds:
            continue
        ax.plot(x_sorted, centerline_preds[mname],
                color=mcfg['color'],
                linestyle='-',
                linewidth=1.4,
                marker=mcfg.get('marker', 'o'),
                markersize=6,
                markerfacecolor='none',
                markeredgewidth=1.3,
                markevery=markevery,
                label=mcfg['label'].replace(chr(10), ' '))

    ax.set_xlabel('x (mm)', fontsize=18)
    ax.set_ylabel('Stress (Pa)', fontsize=18)
    ax.tick_params(axis='both', labelsize=18)
    ax.legend(fontsize=18, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_sorted.min(), x_sorted.max())   # no vacant left/right band
    ax.margins(y=0.02)
    plt.tight_layout(pad=0.3)
    out_path = os.path.join(OUT_DIR, 'center_stress_comparison.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"\nCenter stress comparison saved: {out_path}")

    # --- R² bar chart ---
    if r2_values:
        fig, ax = plt.subplots(figsize=(8, 4))
        names = list(r2_values.keys())
        r2s = [r2_values[n] for n in names]
        colors = [MODELS[n]['color'] for n in names]
        bars = ax.bar(
            [MODELS[n]['label'].replace('\n', '\n') for n in names],
            r2s, color=colors, edgecolor='black', linewidth=0.8
        )
        for bar, r2 in zip(bars, r2s):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{r2:.4f}', ha='center', va='bottom', fontsize=9)
        ax.set_ylim(min(0, min(r2s) - 0.05), 1.05)
        ax.axhline(1.0, color='black', linestyle=':', linewidth=0.8)
        ax.set_ylabel('Pearson r²', fontsize=11)
        ax.set_title('Stress prediction Pearson r² by model', fontsize=12)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, 'stress_R2_bar.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"R² bar chart saved: {out_path}")


if __name__ == '__main__':
    main()
