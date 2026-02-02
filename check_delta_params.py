"""Check what delta normalization parameters are being used."""
import sys
from general_modules.mesh_dataset import MeshGraphDataset
from general_modules.load_config import load_config
import h5py

config = load_config('config.txt')

print("=" * 80)
print("Checking Delta Normalization Parameters")
print("=" * 80)

# Check what's stored in HDF5 file
print("\n1. Parameters stored in HDF5 file:")
with h5py.File(config['dataset_dir'], 'r') as f:
    delta_mean_h5 = f['metadata/normalization_params/delta_mean'][:]
    delta_std_h5 = f['metadata/normalization_params/delta_std'][:]
    print(f"   delta_mean: {delta_mean_h5}")
    print(f"   delta_std:  {delta_std_h5}")
    print(f"   stress delta_std (index 3): {delta_std_h5[3]:.2f}")

# Load dataset and check computed values
print("\n2. Parameters computed by dataset loader:")
dataset = MeshGraphDataset(config['dataset_dir'], config)
print(f"   delta_mean: {dataset.delta_mean}")
print(f"   delta_std:  {dataset.delta_std}")
print(f"   stress delta_std (index 3): {dataset.delta_std[3]:.2f}")

# Check if they match
print("\n3. Comparison:")
if dataset.delta_std[3] == delta_std_h5[3]:
    print("   [OK] HDF5 and computed parameters MATCH")
else:
    print(f"   [WARNING] Parameters DIFFER!")
    print(f"   HDF5:     {delta_std_h5[3]:.2f}")
    print(f"   Computed: {dataset.delta_std[3]:.2f}")
    print(f"   Ratio:    {delta_std_h5[3]/dataset.delta_std[3]:.2f}x")

# Test sample to show what normalized vs denormalized values look like
print("\n4. Sample normalization check:")
sample = dataset[0]
stress_delta_normalized = sample.y[:, 3]  # Normalized values
stress_delta_denormalized = stress_delta_normalized.numpy() * dataset.delta_std[3] + dataset.delta_mean[3]

print(f"   Normalized stress delta:")
print(f"     range: [{stress_delta_normalized.min():.4f}, {stress_delta_normalized.max():.4f}]")
print(f"     std: {stress_delta_normalized.std():.4f}")

print(f"   Denormalized stress delta (physical units):")
print(f"     range: [{stress_delta_denormalized.min():.2f}, {stress_delta_denormalized.max():.2f}]")
print(f"     std: {stress_delta_denormalized.std():.2f}")

print("\n" + "=" * 80)
