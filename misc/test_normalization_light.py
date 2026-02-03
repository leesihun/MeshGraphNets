"""
Lightweight normalization verification - minimal memory usage.
Samples a few data points instead of loading the full dataset.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import h5py
import numpy as np


def analyze_raw_data_light(dataset_path):
    """Analyze raw data with minimal memory footprint."""
    print("=" * 60)
    print("LIGHTWEIGHT NORMALIZATION ANALYSIS")
    print("=" * 60)

    with h5py.File(dataset_path, 'r') as f:
        sample_ids = sorted([int(k) for k in f['data'].keys()])
        first_sample = sample_ids[0]
        data_shape = f[f'data/{first_sample}/nodal_data'].shape
        num_timesteps = data_shape[1]

        print(f"\nDataset: {dataset_path}")
        print(f"Samples: {len(sample_ids)}, Timesteps: {num_timesteps}")
        print(f"Data shape (features, time, nodes): {data_shape}")

        # Sample more timesteps including later ones
        feature_names = ['disp_x', 'disp_y', 'disp_z', 'stress']

        # Sample 3 samples and spread timesteps
        sample_subset = sample_ids[:3]

        print(f"\nSampling {len(sample_subset)} samples across multiple timesteps")

        # Collect stats incrementally
        feature_stats = {name: {'values': []} for name in feature_names}
        delta_stats = {name: {'values': []} for name in feature_names}

        for sid in sample_subset:
            data = f[f'data/{sid}/nodal_data'][:]  # [features, time, nodes]

            # Sample every 50th timestep for features
            timesteps = list(range(0, num_timesteps, 50))
            for t in timesteps:
                for i, name in enumerate(feature_names):
                    vals = data[3+i, t, :]
                    feature_stats[name]['values'].extend(vals.flatten()[:500])

            # Sample deltas spread across time
            delta_timesteps = list(range(0, num_timesteps - 1, 50))
            for t in delta_timesteps:
                for i, name in enumerate(feature_names):
                    delta = data[3+i, t+1, :] - data[3+i, t, :]
                    delta_stats[name]['values'].extend(delta.flatten()[:500])

        # Compute stats
        print("\n--- RAW FEATURE STATISTICS (sampled) ---")
        feat_means = []
        feat_stds = []
        for name in feature_names:
            vals = np.array(feature_stats[name]['values'])
            feat_means.append(vals.mean())
            feat_stds.append(vals.std())
            print(f"  {name}: mean={vals.mean():.6e}, std={vals.std():.6e}")

        print("\n--- RAW DELTA STATISTICS (sampled) ---")
        delta_means = []
        delta_stds = []
        for name in feature_names:
            vals = np.array(delta_stats[name]['values'])
            delta_means.append(vals.mean())
            delta_stds.append(vals.std())
            print(f"  delta_{name}: mean={vals.mean():.6e}, std={vals.std():.6e}")

        # Check scale differences
        print("\n--- SCALE COMPARISON (feature_std / delta_std) ---")
        for i, name in enumerate(feature_names):
            ratio = feat_stds[i] / delta_stds[i] if delta_stds[i] > 1e-10 else float('inf')
            warning = " *** LARGE SCALE DIFF! ***" if ratio > 10 or ratio < 0.1 else ""
            print(f"  {name}: {ratio:.2f}{warning}")

        # Check HDF5 stored normalization params
        print("\n--- CHECKING HDF5 STORED NORMALIZATION PARAMS ---")
        if 'metadata/normalization_params' in f:
            norm_params = f['metadata/normalization_params']
            for key in norm_params.keys():
                val = norm_params[key][:]
                print(f"  {key}: {val}")

            # Compare stored vs sampled
            if 'delta_mean' in norm_params and 'delta_std' in norm_params:
                stored_delta_mean = norm_params['delta_mean'][:]
                stored_delta_std = norm_params['delta_std'][:]

                print("\n--- COMPARISON: SAMPLED vs STORED ---")
                print("  Feature    | Sampled delta_mean | Stored delta_mean | Diff")
                print("  " + "-" * 60)
                for i, name in enumerate(feature_names):
                    diff = abs(delta_means[i] - stored_delta_mean[i])
                    print(f"  {name:10} | {delta_means[i]:+.6e}   | {stored_delta_mean[i]:+.6e}   | {diff:.2e}")

                print("\n  Feature    | Sampled delta_std  | Stored delta_std  | Ratio")
                print("  " + "-" * 60)
                for i, name in enumerate(feature_names):
                    ratio = delta_stds[i] / stored_delta_std[i] if stored_delta_std[i] > 1e-10 else float('inf')
                    print(f"  {name:10} | {delta_stds[i]:.6e}    | {stored_delta_std[i]:.6e}   | {ratio:.2f}")
        else:
            print("  No normalization params stored in HDF5 yet")


def verify_single_sample_roundtrip(dataset_path):
    """Manually verify normalization and denormalization for one sample."""
    print("\n" + "=" * 60)
    print("SINGLE SAMPLE ROUNDTRIP VERIFICATION")
    print("=" * 60)

    with h5py.File(dataset_path, 'r') as f:
        sample_ids = sorted([int(k) for k in f['data'].keys()])
        sid = sample_ids[0]
        data = f[f'data/{sid}/nodal_data'][:]  # [features, time, nodes]

        num_timesteps = data.shape[1]
        print(f"\nSample {sid}, timesteps: {num_timesteps}")

        # Get stored normalization params
        if 'metadata/normalization_params' not in f:
            print("No stored normalization params, skipping roundtrip test")
            return

        norm_params = f['metadata/normalization_params']
        delta_mean = norm_params['delta_mean'][:]
        delta_std = norm_params['delta_std'][:]

        # Pick timestep 0 -> 1 transition
        t = 0
        x_raw = data[3:7, t, :].T  # [N, 4] - state at t
        y_raw = data[3:7, t+1, :].T  # [N, 4] - state at t+1
        raw_delta = y_raw - x_raw  # [N, 4] - actual delta

        # Normalize delta using stored params
        normalized_delta = (raw_delta - delta_mean) / delta_std

        # Denormalize back
        denormalized_delta = normalized_delta * delta_std + delta_mean

        # Check roundtrip error
        roundtrip_error = np.abs(raw_delta - denormalized_delta)

        print(f"\n--- Timestep {t} -> {t+1} ---")
        print(f"Raw delta shape: {raw_delta.shape}")

        feature_names = ['disp_x', 'disp_y', 'disp_z', 'stress']
        print("\nPer-feature analysis:")
        for i, name in enumerate(feature_names):
            print(f"\n  {name}:")
            print(f"    Raw delta: min={raw_delta[:, i].min():.6e}, max={raw_delta[:, i].max():.6e}, mean={raw_delta[:, i].mean():.6e}")
            print(f"    Normalized: min={normalized_delta[:, i].min():.4f}, max={normalized_delta[:, i].max():.4f}, std={normalized_delta[:, i].std():.4f}")
            print(f"    Roundtrip error: max={roundtrip_error[:, i].max():.2e}")


def main():
    dataset_path = './dataset/deforming_plate_100.h5'

    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return

    # Lightweight data analysis
    analyze_raw_data_light(dataset_path)

    # Single sample roundtrip verification
    verify_single_sample_roundtrip(dataset_path)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
KEY FINDINGS TO CHECK:

1. If sampled stats differ significantly from stored stats:
   -> The stored HDF5 values may have been computed differently
   -> The code recomputes stats on each run, so stored values may be stale

2. If normalized values don't have std ~= 1:
   -> Normalization may not be working as expected

3. For this dataset (deforming_plate_100.h5):
   -> Check if delta_stress has reasonable normalized range
   -> Large stress values (e.g., thousands) are expected for FEM simulations
""")


if __name__ == '__main__':
    main()
