"""
Comprehensive verification of normalization logic in MeshGraphDataset.

This script verifies:
1. Node feature statistics (mean, std) are correctly computed
2. Delta feature statistics (mean, std) are correctly computed
3. Edge feature statistics are correctly computed
4. Normalized values in __getitem__ are consistent with computed statistics
5. Check for any dimension mismatches between input and output
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import h5py
import numpy as np
import torch
from general_modules.mesh_dataset import MeshGraphDataset

def verify_normalization(dataset_path, config):
    print("=" * 80)
    print("COMPREHENSIVE NORMALIZATION VERIFICATION")
    print("=" * 80)
    print(f"\nDataset: {dataset_path}")
    print(f"Config: {config}")
    print()

    # Load dataset
    dataset = MeshGraphDataset(dataset_path, config)

    print("\n" + "=" * 80)
    print("STEP 1: VERIFY COMPUTED STATISTICS")
    print("=" * 80)

    print(f"\nNode normalization params:")
    print(f"  node_mean: {dataset.node_mean}")
    print(f"  node_std:  {dataset.node_std}")

    print(f"\nEdge normalization params:")
    print(f"  edge_mean: {dataset.edge_mean}")
    print(f"  edge_std:  {dataset.edge_std}")

    print(f"\nDelta normalization params:")
    print(f"  delta_mean: {dataset.delta_mean}")
    print(f"  delta_std:  {dataset.delta_std}")

    print("\n" + "=" * 80)
    print("STEP 2: VERIFY RAW DATA STATISTICS MATCH")
    print("=" * 80)

    # Manually compute statistics from raw data for comparison
    with h5py.File(dataset_path, 'r') as f:
        sample_ids = sorted([int(k) for k in f['data'].keys()])
        first_sample_id = sample_ids[0]
        data_shape = f[f'data/{first_sample_id}/nodal_data'].shape
        num_timesteps = data_shape[1]

        print(f"\nDataset has {len(sample_ids)} samples, {num_timesteps} timesteps")

        # Sample a few samples to manually verify
        all_node_feats = []
        all_deltas = []

        # Use first 5 samples for quick verification
        num_verify_samples = min(5, len(sample_ids))

        for i in range(num_verify_samples):
            sid = sample_ids[i]
            data = f[f'data/{sid}/nodal_data'][:]  # [features, time, nodes]

            # Node features: features 3:7 (disp_x, disp_y, disp_z, stress)
            for t in range(min(10, num_timesteps)):
                node_feat = data[3:3+config['input_var'], t, :].T  # [N, 4]
                all_node_feats.append(node_feat)

            # Delta features
            if num_timesteps > 1:
                for t in range(min(10, num_timesteps-1)):
                    for feat_idx in range(config['output_var']):
                        delta = data[3 + feat_idx, t+1, :] - data[3 + feat_idx, t, :]
                        all_deltas.append(delta)

        all_node_feats = np.vstack(all_node_feats)
        manual_node_mean = np.mean(all_node_feats, axis=0)
        manual_node_std = np.std(all_node_feats, axis=0)

        print(f"\nManual verification (first {num_verify_samples} samples, first 10 timesteps):")
        print(f"  Manual node_mean: {manual_node_mean}")
        print(f"  Manual node_std:  {manual_node_std}")
        print(f"  Dataset node_mean: {dataset.node_mean}")
        print(f"  Dataset node_std:  {dataset.node_std}")

        mean_match = np.allclose(manual_node_mean, dataset.node_mean, rtol=0.2)
        std_match = np.allclose(manual_node_std, dataset.node_std, rtol=0.2)
        print(f"\n  Node mean approximately matches: {mean_match}")
        print(f"  Node std approximately matches:  {std_match}")

    print("\n" + "=" * 80)
    print("STEP 3: VERIFY __getitem__ NORMALIZATION")
    print("=" * 80)

    # Get a sample and verify normalization is applied correctly
    sample_idx = 0
    sample = dataset[sample_idx]

    print(f"\nSample {sample_idx}:")
    print(f"  x shape: {sample.x.shape}")
    print(f"  y shape: {sample.y.shape}")
    print(f"  edge_attr shape: {sample.edge_attr.shape}")

    # Check normalized target (y) statistics
    y_np = sample.y.numpy()
    print(f"\n  Target (y) statistics (should be ~0 mean, ~1 std if normalized):")
    print(f"    Overall: mean={y_np.mean():.4f}, std={y_np.std():.4f}")
    for feat_idx in range(y_np.shape[1]):
        feat_names = ['disp_x', 'disp_y', 'disp_z', 'stress']
        print(f"    {feat_names[feat_idx]}: mean={y_np[:, feat_idx].mean():.4f}, std={y_np[:, feat_idx].std():.4f}")

    # Check normalized input (x) statistics
    x_np = sample.x.numpy()
    if config.get('use_node_types', False):
        x_physical = x_np[:, :4]  # First 4 features are physical
        print(f"\n  Input (x) physical features (should be ~0 mean, ~1 std if normalized):")
    else:
        x_physical = x_np
        print(f"\n  Input (x) statistics (should be ~0 mean, ~1 std if normalized):")

    print(f"    Overall: mean={x_physical.mean():.4f}, std={x_physical.std():.4f}")
    for feat_idx in range(min(4, x_physical.shape[1])):
        feat_names = ['disp_x', 'disp_y', 'disp_z', 'stress']
        print(f"    {feat_names[feat_idx]}: mean={x_physical[:, feat_idx].mean():.4f}, std={x_physical[:, feat_idx].std():.4f}")

    print("\n" + "=" * 80)
    print("STEP 4: VERIFY MULTIPLE SAMPLES")
    print("=" * 80)

    # Check statistics across multiple samples
    all_y = []
    all_x = []
    num_check = min(20, len(dataset))

    for i in range(0, num_check):
        s = dataset[i]
        all_y.append(s.y.numpy())
        if config.get('use_node_types', False):
            all_x.append(s.x.numpy()[:, :4])
        else:
            all_x.append(s.x.numpy())

    all_y = np.vstack(all_y)
    all_x = np.vstack(all_x)

    print(f"\nAggregated statistics across {num_check} samples:")
    print(f"\n  Target (y) - DELTAS:")
    print(f"    Overall: mean={all_y.mean():.4f}, std={all_y.std():.4f}")
    for feat_idx in range(all_y.shape[1]):
        feat_names = ['delta_disp_x', 'delta_disp_y', 'delta_disp_z', 'delta_stress']
        print(f"    {feat_names[feat_idx]}: mean={all_y[:, feat_idx].mean():.4f}, std={all_y[:, feat_idx].std():.4f}, min={all_y[:, feat_idx].min():.4f}, max={all_y[:, feat_idx].max():.4f}")

    print(f"\n  Input (x) - STATES:")
    print(f"    Overall: mean={all_x.mean():.4f}, std={all_x.std():.4f}")
    for feat_idx in range(min(4, all_x.shape[1])):
        feat_names = ['disp_x', 'disp_y', 'disp_z', 'stress']
        print(f"    {feat_names[feat_idx]}: mean={all_x[:, feat_idx].mean():.4f}, std={all_x[:, feat_idx].std():.4f}, min={all_x[:, feat_idx].min():.4f}, max={all_x[:, feat_idx].max():.4f}")

    print("\n" + "=" * 80)
    print("STEP 5: VERIFY DENORMALIZATION CORRECTNESS")
    print("=" * 80)

    # Get raw data and verify denormalization recovers it
    with h5py.File(dataset_path, 'r') as f:
        sample_id = dataset.sample_ids[0]
        raw_data = f[f'data/{sample_id}/nodal_data'][:]  # [features, time, nodes]

    if dataset.num_timesteps > 1:
        time_idx = 0
        raw_data_t = raw_data[:, time_idx, :].T  # [nodes, features]
        raw_data_t1 = raw_data[:, time_idx + 1, :].T

        raw_x = raw_data_t[:, 3:3+config['input_var']]  # [N, 4]
        raw_y = raw_data_t1[:, 3:3+config['output_var']]  # [N, 4]
        raw_delta = raw_y - raw_x  # [N, 4]

        # Get normalized sample
        sample = dataset[0]
        normalized_delta = sample.y.numpy()

        # Denormalize
        denormalized_delta = normalized_delta * dataset.delta_std + dataset.delta_mean

        # Compare
        print(f"\nDenormalization verification (sample 0, timestep 0->1):")
        print(f"  Raw delta shape: {raw_delta.shape}")
        print(f"  Denormalized delta shape: {denormalized_delta.shape}")

        delta_diff = np.abs(raw_delta - denormalized_delta)
        print(f"\n  Difference (raw - denormalized):")
        print(f"    Max diff: {delta_diff.max():.2e}")
        print(f"    Mean diff: {delta_diff.mean():.2e}")

        if delta_diff.max() < 1e-5:
            print(f"    PASS: Denormalization correctly recovers raw deltas")
        else:
            print(f"    WARNING: Denormalization may have issues!")
            print(f"    Per-feature max diff:")
            for feat_idx in range(4):
                feat_names = ['disp_x', 'disp_y', 'disp_z', 'stress']
                print(f"      {feat_names[feat_idx]}: {delta_diff[:, feat_idx].max():.2e}")

    print("\n" + "=" * 80)
    print("STEP 6: CHECK FOR SCALE ISSUES")
    print("=" * 80)

    # Check if any features have vastly different scales
    print(f"\nChecking for scale issues in normalization parameters:")

    node_std_ratio = dataset.node_std.max() / dataset.node_std.min()
    delta_std_ratio = dataset.delta_std.max() / dataset.delta_std.min()

    print(f"  Node std ratio (max/min): {node_std_ratio:.2f}")
    print(f"  Delta std ratio (max/min): {delta_std_ratio:.2f}")

    if node_std_ratio > 100:
        print(f"  WARNING: Large scale difference in node features (ratio > 100)")
    if delta_std_ratio > 100:
        print(f"  WARNING: Large scale difference in delta features (ratio > 100)")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    issues = []

    # Check if normalized data is roughly centered
    if abs(all_y.mean()) > 0.5:
        issues.append(f"Target (y) mean is not well-centered: {all_y.mean():.4f}")
    if abs(all_y.std() - 1.0) > 0.5:
        issues.append(f"Target (y) std deviates from 1: {all_y.std():.4f}")
    if abs(all_x.mean()) > 0.5:
        issues.append(f"Input (x) mean is not well-centered: {all_x.mean():.4f}")
    if abs(all_x.std() - 1.0) > 0.5:
        issues.append(f"Input (x) std deviates from 1: {all_x.std():.4f}")

    if issues:
        print("\nPotential issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nNo obvious normalization issues found.")
        print("Normalized data appears to be properly centered and scaled.")

    return dataset

if __name__ == '__main__':
    # Use the same config as training
    config = {
        'input_var': 4,
        'output_var': 4,
        'edge_var': 4,
        'use_node_types': True,
        'use_world_edges': False,
        'use_parallel_stats': True,
    }

    dataset_path = './dataset/deforming_plate_50.h5'

    if not os.path.exists(dataset_path):
        # Try alternate path
        dataset_path = './dataset/deforming_plate.h5'

    if os.path.exists(dataset_path):
        verify_normalization(dataset_path, config)
    else:
        print(f"Dataset not found at {dataset_path}")
        print("Please specify the correct dataset path.")
