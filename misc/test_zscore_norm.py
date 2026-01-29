"""Test Z-score normalization for delta targets"""
import sys
sys.path.insert(0, '.')

from general_modules.mesh_dataset import MeshGraphDataset
import numpy as np

# Load config
config = {
    'input_var': 4,
    'output_var': 4,
    'use_node_types': False,
    'use_world_edges': False,
}

# Load dataset
dataset = MeshGraphDataset('dataset/deforming_plate.h5', config)

# Get a sample
sample = dataset[100]  # Arbitrary index

print("=" * 70)
print("Z-SCORE NORMALIZATION VERIFICATION")
print("=" * 70)
print(f"Sample target (y) shape: {sample.y.shape}")
print(f"Target statistics (after z-score normalization):")
print(f"  Mean: {sample.y.mean().item():.6f}")
print(f"  Std: {sample.y.std().item():.6f}")
print(f"  Min: {sample.y.min().item():.6f}")
print(f"  Max: {sample.y.max().item():.6f}")
print()
print("Expected: Mean ~ 0, Std ~ 1 (for z-score normalized data)")
print()

# Check multiple samples to get overall statistics
print("=" * 70)
print("CHECKING MULTIPLE SAMPLES")
print("=" * 70)
all_targets = []
for i in range(0, len(dataset), 100):  # Sample every 100th
    sample = dataset[i]
    all_targets.append(sample.y.numpy())

all_targets = np.concatenate(all_targets, axis=0)
print(f"Combined shape: {all_targets.shape}")
print(f"Overall statistics (after z-score normalization):")
print(f"  Mean: {all_targets.mean():.6f}")
print(f"  Std: {all_targets.std():.6f}")
print(f"  Min: {all_targets.min():.6f}")
print(f"  Max: {all_targets.max():.6f}")
print()
print("Per-feature statistics:")
for i in range(all_targets.shape[1]):
    feat_name = ['disp_x', 'disp_y', 'disp_z', 'stress'][i]
    print(f"  {feat_name}: mean={all_targets[:, i].mean():.4f}, "
          f"std={all_targets[:, i].std():.4f}, "
          f"min={all_targets[:, i].min():.4f}, "
          f"max={all_targets[:, i].max():.4f}")
