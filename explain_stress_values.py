"""Explain the difference between normalized and denormalized stress delta values."""
from general_modules.mesh_dataset import MeshGraphDataset
from general_modules.load_config import load_config
import numpy as np

config = load_config('config.txt')
dataset = MeshGraphDataset(config['dataset_dir'], config)

print("=" * 80)
print("Understanding Normalized vs Denormalized Stress Delta Values")
print("=" * 80)

# Get samples with actual stress changes (not from early timesteps)
samples_with_stress = []
for idx in range(1000, min(5000, len(dataset))):  # Skip early timesteps
    sample = dataset[idx]
    stress_delta_norm = sample.y[:, 3]
    if stress_delta_norm.std() > 0.01:  # Find samples with variation
        samples_with_stress.append(sample)
        if len(samples_with_stress) >= 5:
            break

if len(samples_with_stress) == 0:
    print("No samples with stress variation found in first 5000 samples")
    print("This is normal if stress only changes later in the simulation")
    exit()

print(f"\nAnalyzing {len(samples_with_stress)} samples with stress variation:")
print("\n" + "=" * 80)

for i, sample in enumerate(samples_with_stress):
    stress_delta_norm = sample.y[:, 3].numpy()
    stress_delta_phys = stress_delta_norm * dataset.delta_std[3] + dataset.delta_mean[3]

    print(f"\nSample {i+1}:")
    print(f"  NORMALIZED stress delta (what the model works with):")
    print(f"    mean: {np.mean(stress_delta_norm):8.4f}")
    print(f"    std:  {np.std(stress_delta_norm):8.4f}")
    print(f"    range: [{np.min(stress_delta_norm):8.4f}, {np.max(stress_delta_norm):8.4f}]")

    print(f"\n  DENORMALIZED stress delta (PHYSICAL units - what you see in plots):")
    print(f"    mean: {np.mean(stress_delta_phys):10.2f} Pa")
    print(f"    std:  {np.std(stress_delta_phys):10.2f} Pa")
    print(f"    range: [{np.min(stress_delta_phys):10.2f}, {np.max(stress_delta_phys):10.2f}] Pa")

print("\n" + "=" * 80)
print("IMPORTANT NOTES:")
print("=" * 80)
print("""
1. The NORMALIZED values (what the model trains on):
   - Should have mean ~0, std ~1
   - Typically range from -3 to +3
   - Model predicts these normalized values

2. The DENORMALIZED values (what you see in visualizations):
   - Are in PHYSICAL UNITS (Pascals for stress)
   - Can be hundreds to thousands of Pa
   - These are the ACTUAL physical stress changes in the material
   - Obtained by: denorm = normalized * delta_std + delta_mean

3. Why values seem "large" now:
   - Before: Incorrect normalization compressed everything to tiny values
   - Now: Correct normalization shows ACTUAL physical stress magnitudes
   - Stress changes of 500-2000 Pa are NORMAL for deforming materials!

4. The displayed values in test visualizations are DENORMALIZED (physical units).
   This is CORRECT - you want to see real physical values, not normalized ones!
""")

print("\nDelta normalization parameters:")
print(f"  delta_mean[stress]: {dataset.delta_mean[3]:.2f} Pa")
print(f"  delta_std[stress]:  {dataset.delta_std[3]:.2f} Pa")
print(f"\nThese parameters ensure proper conversion between normalized and physical values.")
print("=" * 80)
