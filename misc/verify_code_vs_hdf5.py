"""
Compare what the code ACTUALLY computes vs what's stored in HDF5.
This helps identify if there's a mismatch.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import h5py
import numpy as np


def simulate_stats_computation(dataset_path, input_dim=4, output_dim=4):
    """Replicate the exact logic from _compute_stats_serial to see what stats should be."""
    print("=" * 60)
    print("SIMULATING EXACT STATS COMPUTATION FROM CODE")
    print("=" * 60)

    with h5py.File(dataset_path, 'r') as f:
        sample_ids = sorted([int(k) for k in f['data'].keys()])
        first_sample = sample_ids[0]
        data_shape = f[f'data/{first_sample}/nodal_data'].shape
        num_timesteps = data_shape[1]

        print(f"\nDataset: {dataset_path}")
        print(f"Samples: {len(sample_ids)}, Timesteps: {num_timesteps}")

        # Replicate the exact sampling from _compute_stats_serial
        max_timesteps_for_stats = 500  # Same as in code

        all_node_features = []
        all_delta_features = [[] for _ in range(output_dim)]

        print(f"\nProcessing samples (replicating code logic)...")

        # Process all samples like the code does
        for i, sid in enumerate(sample_ids):
            if i % 20 == 0:
                print(f"  Sample {i}/{len(sample_ids)}")

            data = f[f'data/{sid}/nodal_data'][:]  # [features, time, nodes]

            # Sample timesteps for features (same as code)
            num_samples_t = min(max_timesteps_for_stats, num_timesteps)
            timesteps = np.linspace(0, num_timesteps - 1, num_samples_t, dtype=int)

            for t in timesteps:
                # Node features: [disp_x, disp_y, disp_z, stress] - indices 3:7
                node_feat = data[3:3+input_dim, t, :].T  # [N, 4]
                all_node_features.append(node_feat)

            # Compute delta features (same as code)
            num_delta_samples = min(max_timesteps_for_stats, num_timesteps - 1)
            delta_timesteps = np.linspace(0, num_timesteps - 2, num_delta_samples, dtype=int)

            for t in delta_timesteps:
                for feat_idx in range(output_dim):
                    feat_t = data[3 + feat_idx, t, :]
                    feat_t1 = data[3 + feat_idx, t + 1, :]
                    delta = feat_t1 - feat_t
                    all_delta_features[feat_idx].append(delta)

        # Compute statistics
        all_node_features = np.vstack(all_node_features)
        node_mean = np.mean(all_node_features, axis=0).astype(np.float32)
        node_std = np.std(all_node_features, axis=0).astype(np.float32)
        node_std = np.maximum(node_std, 1e-8)

        delta_mean = np.zeros(output_dim, dtype=np.float32)
        delta_std = np.zeros(output_dim, dtype=np.float32)

        for feat_idx in range(output_dim):
            deltas = np.concatenate(all_delta_features[feat_idx])
            delta_mean[feat_idx] = np.mean(deltas)
            delta_std[feat_idx] = np.std(deltas)
            delta_std[feat_idx] = max(delta_std[feat_idx], 1e-8)

        print("\n--- COMPUTED STATS (simulating code) ---")
        feature_names = ['disp_x', 'disp_y', 'disp_z', 'stress']
        print("\nNode stats:")
        for i, name in enumerate(feature_names):
            print(f"  {name}: mean={node_mean[i]:.6e}, std={node_std[i]:.6e}")

        print("\nDelta stats:")
        for i, name in enumerate(feature_names):
            print(f"  delta_{name}: mean={delta_mean[i]:.6e}, std={delta_std[i]:.6e}")

        # Compare with stored HDF5 values
        print("\n--- COMPARISON WITH HDF5 STORED VALUES ---")
        if 'metadata/normalization_params' in f:
            norm_params = f['metadata/normalization_params']
            stored_delta_mean = norm_params['delta_mean'][:]
            stored_delta_std = norm_params['delta_std'][:]

            print("\nDelta Mean comparison:")
            print(f"  {'Feature':<10} {'Computed':<15} {'Stored':<15} {'Match?':<10}")
            print("  " + "-" * 50)
            for i, name in enumerate(feature_names):
                match = np.isclose(delta_mean[i], stored_delta_mean[i], rtol=0.1)
                print(f"  {name:<10} {delta_mean[i]:<15.6e} {stored_delta_mean[i]:<15.6e} {match}")

            print("\nDelta Std comparison:")
            print(f"  {'Feature':<10} {'Computed':<15} {'Stored':<15} {'Ratio':<10}")
            print("  " + "-" * 50)
            for i, name in enumerate(feature_names):
                ratio = delta_std[i] / stored_delta_std[i] if stored_delta_std[i] > 1e-10 else float('inf')
                print(f"  {name:<10} {delta_std[i]:<15.6e} {stored_delta_std[i]:<15.6e} {ratio:.3f}")


if __name__ == '__main__':
    dataset_path = './dataset/deforming_plate_100.h5'
    if os.path.exists(dataset_path):
        simulate_stats_computation(dataset_path)
    else:
        print(f"Dataset not found: {dataset_path}")
