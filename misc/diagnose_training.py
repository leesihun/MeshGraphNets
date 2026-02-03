"""
Diagnose why training loss isn't decreasing.
Checks: normalized data ranges, gradients, model output variance.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import h5py
import numpy as np
import torch


def check_normalized_data_ranges(dataset_path):
    """Check if normalized data has reasonable ranges for learning."""
    print("=" * 60)
    print("CHECK 1: NORMALIZED DATA RANGES")
    print("=" * 60)

    with h5py.File(dataset_path, 'r') as f:
        sample_ids = sorted([int(k) for k in f['data'].keys()])
        sid = sample_ids[0]
        data = f[f'data/{sid}/nodal_data'][:]
        num_timesteps = data.shape[1]

        # Get normalization params
        if 'metadata/normalization_params' not in f:
            print("ERROR: No normalization params found!")
            return

        norm = f['metadata/normalization_params']
        delta_mean = norm['delta_mean'][:]
        delta_std = norm['delta_std'][:]

        feature_names = ['disp_x', 'disp_y', 'disp_z', 'stress']

        # Compute normalized deltas for a few timesteps
        print("\nNormalized delta values at different timesteps:")
        print(f"{'t->t+1':<10} {'norm_dx':<12} {'norm_dy':<12} {'norm_dz':<12} {'norm_stress':<12}")
        print("-" * 60)

        timesteps = [10, 50, 100, 200, 300]
        all_normalized = []

        for t in timesteps:
            if t >= num_timesteps - 1:
                continue

            raw_delta = data[3:7, t+1, :].T - data[3:7, t, :].T  # [N, 4]
            normalized = (raw_delta - delta_mean) / delta_std  # [N, 4]

            means = normalized.mean(axis=0)
            all_normalized.append(normalized)

            print(f"{t}->{t+1:<5} {means[0]:<12.4f} {means[1]:<12.4f} {means[2]:<12.4f} {means[3]:<12.4f}")

        all_normalized = np.vstack(all_normalized)

        print(f"\nOverall normalized target statistics:")
        print(f"  Mean: {all_normalized.mean():.4f} (should be ~0)")
        print(f"  Std:  {all_normalized.std():.4f} (should be ~1)")
        print(f"  Min:  {all_normalized.min():.4f}")
        print(f"  Max:  {all_normalized.max():.4f}")

        # CRITICAL CHECK
        if abs(all_normalized.mean()) > 1.0:
            print("\n  *** WARNING: Mean is far from 0! Targets may be biased. ***")
        if all_normalized.std() < 0.1:
            print("\n  *** WARNING: Std is very small! Targets may lack variance. ***")
        if all_normalized.std() > 5.0:
            print("\n  *** WARNING: Std is very large! Normalization may be off. ***")


def check_input_output_relationship(dataset_path):
    """Check if inputs and outputs have a learnable relationship."""
    print("\n" + "=" * 60)
    print("CHECK 2: INPUT-OUTPUT RELATIONSHIP")
    print("=" * 60)

    with h5py.File(dataset_path, 'r') as f:
        sample_ids = sorted([int(k) for k in f['data'].keys()])
        num_timesteps = f[f'data/{sample_ids[0]}/nodal_data'].shape[1]

        # Check correlation between input state and output delta
        all_inputs = []
        all_outputs = []

        for sid in sample_ids[:5]:  # Sample 5 samples
            data = f[f'data/{sid}/nodal_data'][:]

            for t in range(10, min(50, num_timesteps-1)):  # Sample timesteps
                x = data[3:7, t, :].T  # Input: state at t
                delta = data[3:7, t+1, :].T - x  # Output: delta

                # Sample nodes
                all_inputs.append(x[:100])
                all_outputs.append(delta[:100])

        all_inputs = np.vstack(all_inputs)
        all_outputs = np.vstack(all_outputs)

        print("\nCorrelation between input features and output deltas:")
        feature_names = ['disp_x', 'disp_y', 'disp_z', 'stress']

        for i, name in enumerate(feature_names):
            corr = np.corrcoef(all_inputs[:, i], all_outputs[:, i])[0, 1]
            print(f"  {name}: input-output correlation = {corr:.4f}")

        # Check if outputs are essentially constant (no variance)
        print("\nOutput (delta) variance per feature:")
        for i, name in enumerate(feature_names):
            var = np.var(all_outputs[:, i])
            print(f"  {name}: variance = {var:.6e}")
            if var < 1e-12:
                print(f"    *** WARNING: Near-zero variance! Model has nothing to learn. ***")


def check_trivial_solution():
    """Check if predicting zeros/mean would give low loss."""
    print("\n" + "=" * 60)
    print("CHECK 3: TRIVIAL SOLUTION CHECK")
    print("=" * 60)

    print("""
If the normalized targets have mean ~0, then predicting all zeros
would give MSE = variance of targets.

If variance is ~1 (proper normalization), MSE of trivial solution ≈ 1.0
If your actual training loss is close to 1.0, the model isn't learning!
""")


def check_residual_scaling():
    """Check if residual scaling is too aggressive."""
    print("\n" + "=" * 60)
    print("CHECK 4: RESIDUAL SCALING")
    print("=" * 60)

    # Read config
    config_path = './config.txt'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            content = f.read()

        if 'residual_scale' in content.lower():
            for line in content.split('\n'):
                if 'residual_scale' in line.lower():
                    print(f"Config: {line.strip()}")

        print("""
residual_scale controls how much each GN block's update contributes.
- 0.1 means only 10% of the update is added to the current state
- With 15 message passing blocks, effective contribution = 0.1 * 15 = 1.5

If residual_scale is too small:
  - Gradients vanish through many layers
  - Model output changes very slowly
  - Learning is extremely slow

RECOMMENDATION: Try residual_scale = 0.5 or even 1.0 for faster learning.
""")


def check_learning_rate():
    """Check learning rate appropriateness."""
    print("\n" + "=" * 60)
    print("CHECK 5: LEARNING RATE")
    print("=" * 60)

    config_path = './config.txt'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            content = f.read()

        for line in content.split('\n'):
            if 'learningr' in line.lower():
                print(f"Config: {line.strip()}")

    print("""
With normalized data (mean~0, std~1):
  - LR = 1e-4 is reasonable for AdamW
  - But if residual_scale is small, effective learning is slower

If loss isn't decreasing:
  - Try LR = 1e-3 (10x higher)
  - Or increase residual_scale
""")


def main():
    dataset_path = './dataset/deforming_plate_100.h5'

    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return

    check_normalized_data_ranges(dataset_path)
    check_input_output_relationship(dataset_path)
    check_trivial_solution()
    check_residual_scaling()
    check_learning_rate()

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("""
If loss isn't decreasing, try these in order:

1. INCREASE RESIDUAL_SCALE: Change from 0.1 to 0.5 or 1.0
   This is likely the main issue with 15 message passing blocks.

2. INCREASE LEARNING RATE: Try 1e-3 instead of 1e-4

3. REDUCE MESSAGE PASSING BLOCKS: Try 5 instead of 15
   Deep networks with small residual scale = vanishing updates

4. CHECK GRADIENT FLOW: Set monitor_gradients=True in config
   If gradients are < 1e-6, you have vanishing gradient problem.
""")


if __name__ == '__main__':
    main()
