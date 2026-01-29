"""
Compute Delta Statistics for State Differences

This script computes normalization parameters for state differences (deltas)
between consecutive timesteps, which are needed for proper target normalization
when training MeshGraphNets on temporal data.

Usage:
    python misc/compute_delta_stats.py --dataset ./dataset/deforming_plate.h5
    python misc/compute_delta_stats.py --dataset ./dataset/deforming_plate.h5 --dry-run
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm


def compute_delta_statistics(h5_path: str, feature_indices: tuple = (3, 4, 5, 6)) -> dict:
    """
    Compute min, max, mean, std for state differences between consecutive timesteps.

    Args:
        h5_path: Path to HDF5 dataset
        feature_indices: Indices of features to compute deltas for
                        (default: 3-6 for disp_x, disp_y, disp_z, stress)

    Returns:
        Dictionary with delta_min, delta_max, delta_mean, delta_std arrays
    """
    print(f"Computing delta statistics for: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        sample_ids = sorted([int(k) for k in f['data'].keys()])
        print(f"  Found {len(sample_ids)} samples")

        # Check timesteps from first sample
        first_sample = f[f'data/{sample_ids[0]}/nodal_data']
        num_features, num_timesteps, num_nodes = first_sample.shape
        print(f"  Shape: {num_features} features, {num_timesteps} timesteps, {num_nodes} nodes")

        if num_timesteps < 2:
            raise ValueError(f"Dataset has only {num_timesteps} timestep(s), need at least 2 for delta computation")

        num_delta_features = len(feature_indices)

        # Initialize accumulators for vectorized computation
        global_min = np.full(num_delta_features, np.inf, dtype=np.float64)
        global_max = np.full(num_delta_features, -np.inf, dtype=np.float64)
        sum_vals = np.zeros(num_delta_features, dtype=np.float64)
        sum_sq_vals = np.zeros(num_delta_features, dtype=np.float64)
        n_total = 0

        for sid in tqdm(sample_ids, desc="Processing samples"):
            nodal_data = f[f'data/{sid}/nodal_data'][:]  # [features, timesteps, nodes]

            # Extract target features
            features = nodal_data[list(feature_indices), :, :]  # [num_delta_features, T, N]

            # Compute deltas: state[t+1] - state[t]
            deltas = features[:, 1:, :] - features[:, :-1, :]  # [num_delta_features, T-1, N]

            # Vectorized min/max/sum/sum_sq computation per feature
            for i in range(num_delta_features):
                delta_feature = deltas[i].ravel()  # Flatten to 1D
                global_min[i] = min(global_min[i], delta_feature.min())
                global_max[i] = max(global_max[i], delta_feature.max())
                sum_vals[i] += delta_feature.sum()
                sum_sq_vals[i] += (delta_feature ** 2).sum()

            n_total += deltas[0].size  # Number of elements per feature

        # Compute mean and std from accumulated sums
        mean = sum_vals / n_total
        variance = (sum_sq_vals / n_total) - (mean ** 2)
        std = np.sqrt(variance)

        # Ensure no zero std (add small epsilon)
        std = np.maximum(std, 1e-8)

        print(f"\n  Total data points per feature: {n_total:,}")

        return {
            'delta_min': global_min.astype(np.float32),
            'delta_max': global_max.astype(np.float32),
            'delta_mean': mean.astype(np.float32),
            'delta_std': std.astype(np.float32)
        }


def save_delta_stats(h5_path: str, stats: dict, dry_run: bool = False):
    """Save delta statistics to the HDF5 file."""

    print("\nDelta Statistics:")
    feature_names = ['disp_x', 'disp_y', 'disp_z', 'stress']
    for i, name in enumerate(feature_names):
        print(f"  {name}:")
        print(f"    min={stats['delta_min'][i]:.6e}, max={stats['delta_max'][i]:.6e}")
        print(f"    mean={stats['delta_mean'][i]:.6e}, std={stats['delta_std'][i]:.6e}")
        print(f"    range={stats['delta_max'][i] - stats['delta_min'][i]:.6e}")

    if dry_run:
        print("\n[DRY RUN] Would save to HDF5, but skipping.")
        return

    print(f"\nSaving to {h5_path}...")
    with h5py.File(h5_path, 'a') as f:
        norm_group = f['metadata/normalization_params']

        for key, value in stats.items():
            if key in norm_group:
                print(f"  Overwriting existing {key}")
                del norm_group[key]
            norm_group.create_dataset(key, data=value)
            print(f"  Saved {key}: {value.shape}")

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Compute delta statistics for MeshGraphNets dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Path to HDF5 dataset')
    parser.add_argument('--dry-run', action='store_true', help='Compute stats but do not save')
    args = parser.parse_args()

    h5_path = Path(args.dataset)
    if not h5_path.exists():
        raise FileNotFoundError(f"Dataset not found: {h5_path}")

    stats = compute_delta_statistics(str(h5_path))
    save_delta_stats(str(h5_path), stats, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
