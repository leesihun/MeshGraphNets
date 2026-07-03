"""Compare a rollout HDF5 against a ground-truth HDF5: safety checks + stress/disp R^2 + plots.

Usage:
    python compare_rollout_gt.py <rollout.h5> <gt.h5> [options]

Options:
    --plot-dir DIR   directory to save the per-panel PNGs
    --name NAME      label used in plot titles and PNG filenames (default: rollout)
    --init PATH      dataset whose t=0 is the intended initial condition; the
                     rollout's t=0 is validated against it (default: use the GT).
    --step IDX       timestep index to compare (default -1, the final step)
    --atol FLOAT     absolute tolerance for mesh / IC equality (mm, default 1e-3)
    --rtol FLOAT     relative tolerance for mesh / IC equality (default 1e-4)
    --force          downgrade every FAIL to a warning and keep going

Comparison is between one rollout timestep and the matching GT timestep:
  disp magnitude : ||disp channels||
  stress channel : the channel whose feature name contains "stress" (if present)

stdout: exactly one line, "<stress_R2> <disp_R2>" (consumed by run_infer_sweep.sh).
All diagnostics, the safety report, and plot paths go to stderr.

Exit codes:
  0  comparison done, R^2 printed
  1  a file could not be read
  2  a safety check failed (and --force was not given); no R^2 printed
"""
import argparse
import os
import sys

import h5py
import numpy as np


# ----------------------------------------------------------------------------
# HDF5 reading
# ----------------------------------------------------------------------------

def _decode_names(raw):
    if raw is None:
        return None
    return [n.decode() if isinstance(n, (bytes, bytearray)) else str(n) for n in raw]


def _channel_layout(names, n_features):
    """Return (phys_count, disp_idx, stress_idx) as physical-channel indices.

    Physical channels start at feature index 3 (after x/y/z reference coords).
    A trailing "Part No." column is dropped when present. Indices are relative
    to the physical block, i.e. 0 == feature 3.
    """
    if names is not None and len(names) == n_features:
        has_part = 'part' in names[-1].lower()
        phys_count = n_features - 3 - (1 if has_part else 0)
        phys_names = names[3:3 + phys_count]
        stress_idx = next(
            (i for i, nm in enumerate(phys_names) if 'stress' in nm.lower()),
            None,
        )
    else:
        # No usable names: assume [x,y,z, phys..., Part No.] when wide enough.
        phys_count = max(1, n_features - 4)
        stress_idx = 3 if phys_count >= 4 else None  # disp x/y/z then stress
    disp_idx = [i for i in range(min(3, phys_count))]
    return phys_count, disp_idx, stress_idx


def _read(path, want_sample=None):
    """Read one sample from an HDF5 file into a plain dict.

    If ``want_sample`` is given and present, that sample is used; otherwise the
    lowest-numbered sample is used and ``matched`` is set to False.
    """
    with h5py.File(path, 'r') as f:
        if 'data' not in f:
            raise ValueError(f"{path}: no 'data' group")
        ids = sorted(f['data'].keys(), key=int)
        if not ids:
            raise ValueError(f"{path}: 'data' group is empty")

        matched = True
        if want_sample is not None and str(want_sample) in f['data']:
            sid = str(want_sample)
        else:
            sid = ids[0]
            matched = want_sample is None or str(want_sample) == sid

        nodal = f[f'data/{sid}/nodal_data'][:]  # [features, time, nodes]
        names = _decode_names(f.get('metadata/feature_names', np.array([]))[:]) \
            if 'metadata/feature_names' in f else None

    n_feat, n_time, n_nodes = nodal.shape
    phys_count, disp_idx, stress_idx = _channel_layout(names, n_feat)

    return {
        'path': path,
        'sample_id': int(sid),
        'n_samples': len(ids),
        'matched': matched,
        'n_time': int(n_time),
        'n_nodes': int(n_nodes),
        'coords': nodal[0:3, 0, :],                       # [3, nodes], static ref
        'first': nodal[3:3 + phys_count, 0, :],           # [phys, nodes] at t=0
        'last_all': nodal[3:3 + phys_count, :, :],        # [phys, time, nodes]
        'phys_count': phys_count,
        'disp_idx': disp_idx,
        'stress_idx': stress_idx,
    }


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def _r2(pred, gt):
    ss_res = float(np.sum((pred - gt) ** 2))
    ss_tot = float(np.sum((gt - gt.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def _diffstats(a, b):
    """Return (max_abs_diff, rmse) between two arrays."""
    d = a - b
    return float(np.abs(d).max()), float(np.sqrt(np.mean(d ** 2)))


# ----------------------------------------------------------------------------
# Safety-check bookkeeping
# ----------------------------------------------------------------------------

class Checks:
    """Prints a PASS/WARN/SKIP/FAIL line per check and remembers fatal failures."""

    def __init__(self, force):
        self.force = force
        self.failed = False

    def _line(self, tag, name, msg):
        detail = f": {msg}" if msg else ""
        print(f"  [{tag:^7}] {name}{detail}", file=sys.stderr)

    def ok(self, name, msg=''):
        self._line('PASS', name, msg)

    def warn(self, name, msg=''):
        self._line('WARN', name, msg)

    def skip(self, name, msg=''):
        self._line('SKIP', name, msg)

    def fail(self, name, msg=''):
        self._line('FORCED' if self.force else 'FAIL', name, msg)
        if not self.force:
            self.failed = True

    def require(self, cond, name, ok_msg='', fail_msg=''):
        (self.ok if cond else self.fail)(name, ok_msg if cond else fail_msg)
        return cond


# ----------------------------------------------------------------------------
# Plotting (per-panel PNGs, unchanged rendering)
# ----------------------------------------------------------------------------

def _build_triangulation(xy):
    import matplotlib.tri as mtri
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
    pts = xy[tri.triangles]                        # [n_tri, 3, 2]
    edge_len = np.stack([
        np.linalg.norm(pts[:, 0] - pts[:, 1], axis=1),
        np.linalg.norm(pts[:, 1] - pts[:, 2], axis=1),
        np.linalg.norm(pts[:, 2] - pts[:, 0], axis=1),
    ], axis=1)
    max_edge = edge_len.max(axis=1)
    tri.set_mask(max_edge > 3.0 * np.median(edge_len))
    return tri


def _save_panel(tri, vals, cmap, title, out_path, vmin=None, vmax=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    face_vals = vals[tri.triangles].mean(axis=1)
    if tri.mask is not None:
        face_vals = face_vals[~tri.mask]

    if vmin is None:
        vmin, vmax = face_vals.min(), face_vals.max()
    if vmin == vmax:
        vmin -= 1e-6
        vmax += 1e-6

    fig, ax = plt.subplots(figsize=(5, 4))
    tpc = ax.tripcolor(tri, facecolors=face_vals, cmap=cmap,
                       vmin=vmin, vmax=vmax, rasterized=True)
    fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {out_path}", file=sys.stderr)


def _plot_panels(xy, disp_pred, disp_gt, stress_pred, stress_gt,
                 stress_r2, disp_r2, name, out_dir):
    import matplotlib
    matplotlib.use('Agg')

    tri = _build_triangulation(xy)
    prefix = f"{name}_sR2_{stress_r2:.6f}_dR2_{disp_r2:.6f}"
    os.makedirs(out_dir, exist_ok=True)

    d_vmin, d_vmax = disp_gt.min(), disp_gt.max()
    panels = [
        (disp_gt,                          'jet',     d_vmin, d_vmax,
         f"Disp |mag| GT (mm)  [dR2={disp_r2:.4f}]", '04_disp_GT'),
        (disp_pred,                        'jet',     d_vmin, d_vmax,
         "Disp |mag| Prediction (mm)", '05_disp_pred'),
        (np.abs(disp_pred - disp_gt),      'viridis', None,   None,
         "|Disp error| (mm)", '06_disp_diff'),
    ]
    if stress_gt is not None:
        s_gt = stress_gt * 1e6      # MPa -> Pa
        s_pred = stress_pred * 1e6
        s_vmin, s_vmax = s_gt.min(), s_gt.max()
        panels = [
            (s_gt,                       'jet',     s_vmin, s_vmax,
             f"Stress GT (Pa)  [sR2={stress_r2:.4f}]", '01_stress_GT'),
            (s_pred,                     'jet',     s_vmin, s_vmax,
             "Stress Prediction (Pa)", '02_stress_pred'),
            (np.abs(s_pred - s_gt),      'viridis', None,   None,
             "|Stress error| (Pa)", '03_stress_diff'),
        ] + panels

    for vals, cmap, vmin, vmax, title, tag in panels:
        _save_panel(tri, vals, cmap, title, os.path.join(out_dir, f"{prefix}_{tag}.png"),
                    vmin, vmax)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('rollout')
    parser.add_argument('gt')
    parser.add_argument('--plot-dir', default=None,
                        help='directory to save the per-panel PNGs')
    parser.add_argument('--name', default='rollout',
                        help='label used in plot titles and PNG filenames')
    parser.add_argument('--init', default=None,
                        help='dataset whose t=0 is the intended initial condition')
    parser.add_argument('--step', type=int, default=-1,
                        help='timestep index to compare (default -1)')
    parser.add_argument('--atol', type=float, default=1e-3,
                        help='absolute tolerance for mesh / IC equality (mm)')
    parser.add_argument('--rtol', type=float, default=1e-4,
                        help='relative tolerance for mesh / IC equality')
    parser.add_argument('--force', action='store_true',
                        help='downgrade every failed check to a warning')
    args = parser.parse_args()

    print(f"\n=== compare_rollout_gt: {args.name} ===", file=sys.stderr)
    print(f"  rollout : {args.rollout}", file=sys.stderr)
    print(f"  gt      : {args.gt}", file=sys.stderr)

    # --- load (file-level errors -> exit 1) ------------------------------------
    try:
        pred = _read(args.rollout)
        gt = _read(args.gt, want_sample=pred['sample_id'])
    except (OSError, ValueError, KeyError) as e:
        print(f"ERROR: could not read inputs: {e}", file=sys.stderr)
        sys.exit(1)

    chk = Checks(args.force)
    print("\n  --- safety checks ---", file=sys.stderr)

    # 1) sample-id match ---------------------------------------------------------
    if gt['n_samples'] == 1 and not gt['matched']:
        chk.warn('sample id',
                 f"rollout sample {pred['sample_id']} not in GT; GT has a single "
                 f"sample {gt['sample_id']}, using it")
    else:
        chk.require(gt['matched'], 'sample id',
                    f"both sample {pred['sample_id']}",
                    f"rollout sample {pred['sample_id']} absent from GT "
                    f"({gt['n_samples']} samples); compared vs GT sample {gt['sample_id']}")

    # 2) node count --------------------------------------------------------------
    same_nodes = chk.require(
        pred['n_nodes'] == gt['n_nodes'], 'node count',
        f"{pred['n_nodes']} nodes",
        f"rollout {pred['n_nodes']} vs GT {gt['n_nodes']}")
    if not same_nodes and not args.force:
        _abort()

    # 3) reference mesh (coordinates) -------------------------------------------
    if same_nodes:
        cmax, crmse = _diffstats(pred['coords'], gt['coords'])
        mesh_ok = np.allclose(pred['coords'], gt['coords'],
                              atol=args.atol, rtol=args.rtol)
        chk.require(mesh_ok, 'reference mesh',
                    f"coords match (max|d|={cmax:.2e} mm)",
                    f"coords differ (max|d|={cmax:.2e} mm, rmse={crmse:.2e}); "
                    "different or reordered mesh")

    # 4) timestep counts ---------------------------------------------------------
    if pred['n_time'] > 1 and gt['n_time'] > 1:
        chk.require(pred['n_time'] == gt['n_time'], 'timestep count',
                    f"both {pred['n_time']} steps",
                    f"rollout {pred['n_time']} vs GT {gt['n_time']} steps; "
                    "final-vs-final compares different physical times")
    elif gt['n_time'] == 1:
        chk.ok('timestep count',
               f"GT is single-step (static target); rollout has {pred['n_time']} steps")
    else:
        chk.warn('timestep count',
                 f"rollout single-step, GT has {gt['n_time']} steps")

    # 5) initial condition -------------------------------------------------------
    _check_initial_condition(chk, pred, gt, args)

    # 6) finite values -----------------------------------------------------------
    pred_last = pred['last_all'][:, args.step, :]
    gt_last = gt['last_all'][:, args.step, :]
    n_bad = int(np.count_nonzero(~np.isfinite(pred_last)))
    chk.require(n_bad == 0, 'finite rollout',
                'no NaN/Inf in prediction',
                f"{n_bad} non-finite values (rollout likely diverged)")
    if not np.all(np.isfinite(gt_last)):
        chk.warn('finite GT', 'GT contains NaN/Inf')

    # --- stop here if any fatal check failed -----------------------------------
    if chk.failed:
        _abort()

    # --- metrics on the shared channels ----------------------------------------
    common_phys = min(pred['phys_count'], gt['phys_count'])
    if pred['phys_count'] != gt['phys_count']:
        print(f"  note: physical channels differ (rollout {pred['phys_count']}, "
              f"GT {gt['phys_count']}); comparing first {common_phys}", file=sys.stderr)

    disp_ch = [i for i in (pred['disp_idx']) if i < common_phys]
    disp_pred = np.linalg.norm(pred_last[disp_ch], axis=0)
    disp_gt = np.linalg.norm(gt_last[disp_ch], axis=0)
    disp_r2 = _r2(disp_pred, disp_gt)
    dmax, drmse = _diffstats(disp_pred, disp_gt)

    s_idx = pred['stress_idx']
    have_stress = (s_idx is not None and s_idx < common_phys
                   and gt['stress_idx'] == s_idx)
    if have_stress:
        stress_pred = pred_last[s_idx]
        stress_gt = gt_last[s_idx]
        stress_r2 = _r2(stress_pred, stress_gt)
        smax, srmse = _diffstats(stress_pred, stress_gt)
    else:
        stress_pred = stress_gt = None
        stress_r2 = float('nan')
        chk.warn('stress channel', 'not present in both files; stress R2 = nan')

    # --- report ----------------------------------------------------------------
    print("\n  --- metrics (timestep index "
          f"{args.step}) ---", file=sys.stderr)
    print(f"  disp   : R2={disp_r2:.6f}  rmse={drmse:.4e} mm  max|err|={dmax:.4e} mm",
          file=sys.stderr)
    if have_stress:
        print(f"  stress : R2={stress_r2:.6f}  rmse={srmse:.4e} MPa  "
              f"max|err|={smax:.4e} MPa", file=sys.stderr)

    # single stdout line consumed by run_infer_sweep.sh
    print(f"{stress_r2:.6f} {disp_r2:.6f}")

    if args.plot_dir:
        xy = pred['coords'][:2].T
        _plot_panels(xy, disp_pred, disp_gt, stress_pred, stress_gt,
                     stress_r2, disp_r2, args.name, args.plot_dir)


def _check_initial_condition(chk, pred, gt, args):
    """Validate that the rollout was seeded from the intended t=0 state."""
    if args.init:
        try:
            ref = _read(args.init, want_sample=pred['sample_id'])
        except (OSError, ValueError, KeyError) as e:
            chk.warn('initial condition', f"could not read --init file: {e}")
            return
        source, label = ref, f"--init {os.path.basename(args.init)}"
    elif gt['n_time'] > 1:
        source, label = gt, 'GT t=0'
    else:
        chk.skip('initial condition',
                 'GT is single-step; no shared t=0 to validate against '
                 '(pass --init to check)')
        return

    if source['n_nodes'] != pred['n_nodes']:
        chk.warn('initial condition',
                 f"node count mismatch vs {label}; skipped")
        return

    n = min(pred['phys_count'], source['phys_count'])
    a, b = pred['first'][:n], source['first'][:n]
    imax, irmse = _diffstats(a, b)
    chk.require(np.allclose(a, b, atol=args.atol, rtol=args.rtol),
                'initial condition',
                f"rollout t=0 matches {label} (max|d|={imax:.2e})",
                f"rollout t=0 differs from {label} "
                f"(max|d|={imax:.2e}, rmse={irmse:.2e}); wrong seed")


def _abort():
    print("\n  SAFETY CHECK FAILED - refusing to report R^2. "
          "Re-run with --force to override.", file=sys.stderr)
    sys.exit(2)


if __name__ == '__main__':
    main()
