"""
Detailed test script to verify normalization in MeshGraphDataset.

This script performs deep inspection of:
1. Raw data ranges and distributions
2. How node_mean/node_std are computed vs used
3. How delta_mean/delta_std are computed vs used
4. Verification that input normalization and output normalization are truly separate
5. Check single vs multi-timestep scenarios
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import h5py
import numpy as np
from general_modules.mesh_dataset import MeshGraphDataset


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def analyze_raw_data(dataset_path, config):
    """Analyze raw data directly from HDF5 to understand the ground truth."""
    print_section("RAW DATA ANALYSIS")

    with h5py.File(dataset_path, 'r') as f:
        sample_ids = sorted([int(k) for k in f['data'].keys()])
        first_sample = sample_ids[0]
        data_shape = f[f'data/{first_sample}/nodal_data'].shape

        print(f"Dataset: {dataset_path}")
        print(f"Number of samples: {len(sample_ids)}")
        print(f"Data shape (features, time, nodes): {data_shape}")

        # Analyze feature ranges across all samples
        all_features = []
        all_deltas = []

        for sid in sample_ids[:5]:  # First 5 samples for quick analysis
            data = f[f'data/{sid}/nodal_data'][:]  # [features, time, nodes]
            num_timesteps = data.shape[1]

            # Collect features at each timestep
            for t in range(min(10, num_timesteps)):
                feats = data[3:7, t, :].T  # [N, 4] - disp_x, disp_y, disp_z, stress
                all_features.append(feats)

            # Collect deltas
            if num_timesteps > 1:
                for t in range(min(10, num_timesteps - 1)):
                    delta = data[3:7, t+1, :].T - data[3:7, t, :].T  # [N, 4]
                    all_deltas.append(delta)
            else:
                # Single timestep: the "delta" is the feature value itself
                all_deltas.append(data[3:7, 0, :].T)

        all_features = np.vstack(all_features)
        all_deltas = np.vstack(all_deltas)

        feature_names = ['disp_x', 'disp_y', 'disp_z', 'stress']

        print(f"\n--- RAW FEATURE VALUES (state at time t) ---")
        print(f"Shape: {all_features.shape}")
        for i, name in enumerate(feature_names):
            feat = all_features[:, i]
            print(f"  {name}: min={feat.min():.6e}, max={feat.max():.6e}, mean={feat.mean():.6e}, std={feat.std():.6e}")

        print(f"\n--- RAW DELTA VALUES (t+1 - t, or feature itself for single timestep) ---")
        print(f"Shape: {all_deltas.shape}")
        for i, name in enumerate(feature_names):
            delta = all_deltas[:, i]
            print(f"  delta_{name}: min={delta.min():.6e}, max={delta.max():.6e}, mean={delta.mean():.6e}, std={delta.std():.6e}")

        # KEY OBSERVATION: Check if features and deltas have different scales
        print(f"\n--- SCALE COMPARISON ---")
        for i, name in enumerate(feature_names):
            feat_std = all_features[:, i].std()
            delta_std = all_deltas[:, i].std()
            ratio = feat_std / delta_std if delta_std > 0 else float('inf')
            print(f"  {name}: feature_std/delta_std = {ratio:.2f}")
            if ratio > 10 or ratio < 0.1:
                print(f"    *** WARNING: Large scale difference between feature and delta! ***")

        return all_features, all_deltas


def verify_dataset_normalization(dataset_path, config):
    """Load dataset and verify normalization parameters and behavior."""
    print_section("DATASET NORMALIZATION VERIFICATION")

    # Create dataset (this computes normalization stats)
    dataset = MeshGraphDataset(dataset_path, config)

    print(f"\n--- COMPUTED NORMALIZATION PARAMETERS ---")
    print(f"node_mean:  {dataset.node_mean}")
    print(f"node_std:   {dataset.node_std}")
    print(f"delta_mean: {dataset.delta_mean}")
    print(f"delta_std:  {dataset.delta_std}")
    print(f"edge_mean:  {dataset.edge_mean}")
    print(f"edge_std:   {dataset.edge_std}")

    # KEY CHECK: Are node and delta normalization params different?
    print(f"\n--- CHECKING IF NODE AND DELTA PARAMS ARE TRULY DIFFERENT ---")
    mean_same = np.allclose(dataset.node_mean, dataset.delta_mean)
    std_same = np.allclose(dataset.node_std, dataset.delta_std)
    print(f"node_mean == delta_mean? {mean_same}")
    print(f"node_std == delta_std? {std_same}")

    if mean_same and std_same:
        print("*** CRITICAL ISSUE: Node and delta normalization are the same! ***")
        print("    This means the model is using state statistics to normalize deltas.")
        print("    This could cause training issues if states and deltas have different scales.")
    elif mean_same or std_same:
        print("*** WARNING: Some normalization parameters are the same between nodes and deltas. ***")
    else:
        print("GOOD: Node and delta normalization parameters are different.")

    return dataset


def verify_getitem_behavior(dataset, config):
    """Verify that __getitem__ applies normalization correctly."""
    print_section("__getitem__ BEHAVIOR VERIFICATION")

    # Get a sample
    sample = dataset[0]

    print(f"\n--- SAMPLE 0 SHAPES ---")
    print(f"x (input): {sample.x.shape}")
    print(f"y (target delta): {sample.y.shape}")
    print(f"edge_attr: {sample.edge_attr.shape}")

    x_np = sample.x.numpy()
    y_np = sample.y.numpy()

    # Handle node types
    if config.get('use_node_types', False):
        print(f"Node types are being used. Physical features are first {config['input_var']} columns.")
        x_physical = x_np[:, :config['input_var']]
    else:
        x_physical = x_np

    print(f"\n--- NORMALIZED INPUT (x) STATISTICS ---")
    feature_names = ['disp_x', 'disp_y', 'disp_z', 'stress']
    print(f"Expected: mean ~0, std ~1 if correctly normalized")
    for i, name in enumerate(feature_names[:x_physical.shape[1]]):
        print(f"  {name}: mean={x_physical[:, i].mean():.4f}, std={x_physical[:, i].std():.4f}")

    print(f"\n--- NORMALIZED TARGET (y = delta) STATISTICS ---")
    print(f"Expected: mean ~0, std ~1 if correctly normalized")
    for i, name in enumerate(feature_names[:y_np.shape[1]]):
        print(f"  delta_{name}: mean={y_np[:, i].mean():.4f}, std={y_np[:, i].std():.4f}")

    # Multiple samples to get better statistics
    print(f"\n--- AGGREGATED STATISTICS ACROSS MULTIPLE SAMPLES ---")
    all_x = []
    all_y = []
    num_samples = min(50, len(dataset))

    for i in range(num_samples):
        s = dataset[i]
        if config.get('use_node_types', False):
            all_x.append(s.x.numpy()[:, :config['input_var']])
        else:
            all_x.append(s.x.numpy())
        all_y.append(s.y.numpy())

    all_x = np.vstack(all_x)
    all_y = np.vstack(all_y)

    print(f"\nAggregated x (input) over {num_samples} samples:")
    print(f"  Overall: mean={all_x.mean():.4f}, std={all_x.std():.4f}")
    for i, name in enumerate(feature_names[:all_x.shape[1]]):
        print(f"  {name}: mean={all_x[:, i].mean():.4f}, std={all_x[:, i].std():.4f}")

    print(f"\nAggregated y (delta) over {num_samples} samples:")
    print(f"  Overall: mean={all_y.mean():.4f}, std={all_y.std():.4f}")
    for i, name in enumerate(feature_names[:all_y.shape[1]]):
        print(f"  delta_{name}: mean={all_y[:, i].mean():.4f}, std={all_y[:, i].std():.4f}")

    return all_x, all_y


def verify_denormalization_roundtrip(dataset, config):
    """Verify that denormalization correctly recovers the original data."""
    print_section("DENORMALIZATION ROUNDTRIP TEST")

    with h5py.File(dataset.h5_file, 'r') as f:
        sample_id = dataset.sample_ids[0]
        raw_data = f[f'data/{sample_id}/nodal_data'][:]  # [features, time, nodes]

    num_timesteps = raw_data.shape[1]
    print(f"Sample {sample_id}: {num_timesteps} timesteps")

    # Get the first sample from dataset
    sample = dataset[0]
    normalized_y = sample.y.numpy()
    normalized_x = sample.x.numpy()

    if config.get('use_node_types', False):
        normalized_x = normalized_x[:, :config['input_var']]

    if num_timesteps > 1:
        # Multi-timestep case
        time_idx = 0
        raw_x = raw_data[3:3+config['input_var'], time_idx, :].T  # [N, 4]
        raw_y = raw_data[3:3+config['output_var'], time_idx+1, :].T  # [N, 4]
        raw_delta = raw_y - raw_x

        # Denormalize
        denorm_x = normalized_x * dataset.node_std + dataset.node_mean
        denorm_y = normalized_y * dataset.delta_std + dataset.delta_mean  # Delta uses delta params!

        print(f"\n--- INPUT (x) ROUNDTRIP TEST ---")
        x_diff = np.abs(raw_x - denorm_x)
        print(f"Max difference: {x_diff.max():.2e}")
        print(f"Mean difference: {x_diff.mean():.2e}")
        if x_diff.max() < 1e-5:
            print("PASS: Input denormalization is correct")
        else:
            print("FAIL: Input denormalization has errors!")

        print(f"\n--- DELTA (y) ROUNDTRIP TEST ---")
        y_diff = np.abs(raw_delta - denorm_y)
        print(f"Max difference: {y_diff.max():.2e}")
        print(f"Mean difference: {y_diff.mean():.2e}")
        if y_diff.max() < 1e-5:
            print("PASS: Delta denormalization is correct")
        else:
            print("FAIL: Delta denormalization has errors!")

            # Detailed error analysis
            feature_names = ['disp_x', 'disp_y', 'disp_z', 'stress']
            for i, name in enumerate(feature_names):
                print(f"  {name}: max_diff={y_diff[:, i].max():.2e}")
    else:
        print("Single timestep dataset - skipping roundtrip test")


def check_potential_issues(dataset, all_x, all_y):
    """Check for common normalization issues."""
    print_section("POTENTIAL ISSUE DETECTION")

    issues = []

    # Issue 1: Mean not close to 0
    if abs(all_x.mean()) > 0.5:
        issues.append(f"Input (x) mean is far from 0: {all_x.mean():.4f}")
    if abs(all_y.mean()) > 0.5:
        issues.append(f"Target (y) mean is far from 0: {all_y.mean():.4f}")

    # Issue 2: Std not close to 1
    if abs(all_x.std() - 1.0) > 0.5:
        issues.append(f"Input (x) std is far from 1: {all_x.std():.4f}")
    if abs(all_y.std() - 1.0) > 0.5:
        issues.append(f"Target (y) std is far from 1: {all_y.std():.4f}")

    # Issue 3: Per-feature scale imbalance
    feature_names = ['disp_x', 'disp_y', 'disp_z', 'stress']
    for i, name in enumerate(feature_names[:all_x.shape[1]]):
        x_std = all_x[:, i].std()
        if x_std < 0.1:
            issues.append(f"Input {name} has very small std after normalization: {x_std:.4f}")
        if x_std > 5.0:
            issues.append(f"Input {name} has very large std after normalization: {x_std:.4f}")

    for i, name in enumerate(feature_names[:all_y.shape[1]]):
        y_std = all_y[:, i].std()
        if y_std < 0.1:
            issues.append(f"Delta {name} has very small std after normalization: {y_std:.4f}")
        if y_std > 5.0:
            issues.append(f"Delta {name} has very large std after normalization: {y_std:.4f}")

    # Issue 4: Extreme values
    for i, name in enumerate(feature_names[:all_x.shape[1]]):
        x_max = np.abs(all_x[:, i]).max()
        if x_max > 10:
            issues.append(f"Input {name} has extreme values (|max| > 10): {x_max:.2f}")

    for i, name in enumerate(feature_names[:all_y.shape[1]]):
        y_max = np.abs(all_y[:, i]).max()
        if y_max > 10:
            issues.append(f"Delta {name} has extreme values (|max| > 10): {y_max:.2f}")

    # Issue 5: Check node_std vs delta_std ratio
    print("\n--- NODE vs DELTA STD COMPARISON ---")
    for i, name in enumerate(feature_names):
        node_std = dataset.node_std[i]
        delta_std = dataset.delta_std[i]
        ratio = node_std / delta_std if delta_std > 0 else float('inf')
        print(f"  {name}: node_std={node_std:.6e}, delta_std={delta_std:.6e}, ratio={ratio:.2f}")
        if ratio > 100 or ratio < 0.01:
            issues.append(f"{name}: Very large ratio between node_std and delta_std ({ratio:.2f})")

    if issues:
        print("\n*** POTENTIAL ISSUES FOUND ***")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nNo obvious issues detected.")

    return issues


def main():
    config = {
        'input_var': 4,
        'output_var': 4,
        'edge_var': 4,
        'use_node_types': True,
        'use_world_edges': False,
        'use_parallel_stats': True,
    }

    # Use deforming_plate_100.h5
    dataset_path = './dataset/deforming_plate_100.h5'

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        return

    # Run all tests
    print("\n" + "#" * 80)
    print("# DETAILED NORMALIZATION VERIFICATION")
    print("#" * 80)

    # 1. Analyze raw data
    raw_features, raw_deltas = analyze_raw_data(dataset_path, config)

    # 2. Load dataset and verify normalization
    dataset = verify_dataset_normalization(dataset_path, config)

    # 3. Verify __getitem__ behavior
    all_x, all_y = verify_getitem_behavior(dataset, config)

    # 4. Verify denormalization roundtrip
    verify_denormalization_roundtrip(dataset, config)

    # 5. Check for potential issues
    issues = check_potential_issues(dataset, all_x, all_y)

    print_section("FINAL SUMMARY")
    if issues:
        print(f"Found {len(issues)} potential issues that may affect training.")
    else:
        print("Normalization appears to be working correctly.")


if __name__ == '__main__':
    main()
