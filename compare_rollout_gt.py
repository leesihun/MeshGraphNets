"""Compare a rollout HDF5 against a ground-truth HDF5 and print relative L2 errors.

Usage:
    python compare_rollout_gt.py <rollout.h5> <gt.h5>

Prints one line to stdout: "<stress_relL2> <disp_relL2> <all_relL2>"
Comparison is between the final rollout timestep and the final GT timestep,
over physical channels nodal_data[3:7] = [x_disp, y_disp, z_disp, stress].
"""
import sys

import h5py
import numpy as np


def _load_state(path):
    with h5py.File(path, 'r') as f:
        sample_id = sorted(f['data'].keys(), key=int)[0]
        nodal = f[f'data/{sample_id}/nodal_data'][:]
    # [features, time, nodes] -> physical channels at the last timestep
    return nodal[3:7, -1, :]


def _rel_l2(pred, gt):
    denom = np.linalg.norm(gt)
    err = np.linalg.norm(pred - gt)
    return float(err / denom) if denom > 0 else float(err)


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)

    pred = _load_state(sys.argv[1])
    gt = _load_state(sys.argv[2])

    if pred.shape != gt.shape:
        print(f"ERROR: shape mismatch pred {pred.shape} vs gt {gt.shape}", file=sys.stderr)
        sys.exit(1)

    stress_l2 = _rel_l2(pred[3], gt[3])
    disp_l2 = _rel_l2(pred[:3], gt[:3])
    all_l2 = _rel_l2(pred, gt)
    print(f"{stress_l2:.6f} {disp_l2:.6f} {all_l2:.6f}")


if __name__ == '__main__':
    main()
