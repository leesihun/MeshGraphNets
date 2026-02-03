"""
Quick check of delta values across different timesteps.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import h5py
import numpy as np


def check_timestep_deltas(dataset_path):
    """Check delta values at different timesteps."""
    print("Checking delta values across timesteps...")

    with h5py.File(dataset_path, 'r') as f:
        sample_ids = sorted([int(k) for k in f['data'].keys()])
        sid = sample_ids[0]
        data = f[f'data/{sid}/nodal_data'][:]  # [features, time, nodes]

        num_timesteps = data.shape[1]
        print(f"Sample {sid}, total timesteps: {num_timesteps}")

        feature_names = ['disp_x', 'disp_y', 'disp_z', 'stress']

        # Check deltas at various timesteps
        timesteps_to_check = [0, 1, 10, 50, 100, 200, 300, 398]

        print("\n--- DELTA MAGNITUDES AT DIFFERENT TIMESTEPS ---")
        print(f"{'t->t+1':<10} {'disp_x':<15} {'disp_y':<15} {'disp_z':<15} {'stress':<15}")
        print("-" * 70)

        for t in timesteps_to_check:
            if t >= num_timesteps - 1:
                continue

            deltas = []
            for i in range(4):
                delta = data[3+i, t+1, :] - data[3+i, t, :]
                # Report max absolute value
                max_abs = np.abs(delta).max()
                deltas.append(max_abs)

            print(f"{t}->{t+1:<5} {deltas[0]:<15.6e} {deltas[1]:<15.6e} {deltas[2]:<15.6e} {deltas[3]:<15.6e}")

        # Also check the actual values at timestep 0 vs 1
        print("\n--- RAW VALUES AT t=0 vs t=1 (first 5 nodes) ---")
        for i, name in enumerate(feature_names):
            val_t0 = data[3+i, 0, :5]
            val_t1 = data[3+i, 1, :5]
            print(f"\n{name}:")
            print(f"  t=0: {val_t0}")
            print(f"  t=1: {val_t1}")
            print(f"  Same? {np.allclose(val_t0, val_t1)}")


if __name__ == '__main__':
    dataset_path = './dataset/deforming_plate_100.h5'
    if os.path.exists(dataset_path):
        check_timestep_deltas(dataset_path)
    else:
        print(f"Dataset not found: {dataset_path}")
