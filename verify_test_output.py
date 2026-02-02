"""Verify whether test visualizations show normalized or denormalized values."""
import h5py
import numpy as np
import os
import glob

print("=" * 80)
print("Verification: Are Test Visualizations Showing Normalized or Denormalized?")
print("=" * 80)

# Look for test output files
test_output_dirs = glob.glob('outputs/test/*')
print(f"\nSearching for test outputs in: outputs/test/")

if not test_output_dirs:
    print("\nNo test outputs found yet.")
    print("\nBased on code inspection:")
    print("=" * 80)
    print("\nIn training_profiles/training_loop.py lines 203-217:")
    print("```python")
    print("# DENORMALIZE: Convert normalized deltas to actual physical deltas")
    print("if delta_mean is not None and delta_std is not None:")
    print("    predicted_denorm = predicted_np * delta_std + delta_mean")
    print("    target_denorm = target_np * delta_std + delta_mean")
    print("else:")
    print("    # Fallback: use normalized values")
    print("    predicted_denorm = predicted_np")
    print("    target_denorm = target_np")
    print("")
    print("plot_data = save_inference_results_fast(")
    print("    output_path, graph, predicted_denorm, target_denorm,  # <-- DENORMALIZED!")
    print("    ...)")
    print("```")
    print("\n" + "=" * 80)
    print("CONCLUSION: Test visualizations SHOULD show DENORMALIZED (physical) values")
    print("=" * 80)
    print("\nThis means:")
    print("  - The displayed values are in PHYSICAL UNITS (Pascals for stress)")
    print("  - NOT normalized values (which would be ~0-1 range)")
    print("  - Stress delta values of 30-300 Pa are NORMAL and CORRECT")
    print("\nTo verify with actual test outputs, you need to:")
    print("  1. Run training/evaluation first")
    print("  2. Then check outputs/test/<gpu_id>/<epoch>/*.h5 files")
else:
    print(f"\nFound {len(test_output_dirs)} test output directories")

    # Find the most recent test output
    h5_files = []
    for test_dir in test_output_dirs:
        h5_files.extend(glob.glob(os.path.join(test_dir, '**/*.h5'), recursive=True))

    if not h5_files:
        print("No .h5 files found in test outputs")
    else:
        print(f"\nFound {len(h5_files)} test output files")

        # Check the most recent one
        latest_file = max(h5_files, key=os.path.getmtime)
        print(f"\nAnalyzing most recent: {latest_file}")

        with h5py.File(latest_file, 'r') as f:
            predicted = f['nodes/predicted'][:]
            target = f['nodes/target'][:]

            print(f"\nData shape: {predicted.shape}")
            print(f"Number of features: {predicted.shape[1]}")

            # Check stress delta (feature index 3)
            if predicted.shape[1] > 3:
                stress_pred = predicted[:, 3]
                stress_target = target[:, 3]

                print("\n" + "=" * 80)
                print("Feature 3 (Stress Delta) Statistics:")
                print("=" * 80)
                print(f"\nPredicted:")
                print(f"  mean:  {np.mean(stress_pred):10.4f}")
                print(f"  std:   {np.std(stress_pred):10.4f}")
                print(f"  range: [{np.min(stress_pred):10.4f}, {np.max(stress_pred):10.4f}]")

                print(f"\nTarget:")
                print(f"  mean:  {np.mean(stress_target):10.4f}")
                print(f"  std:   {np.std(stress_target):10.4f}")
                print(f"  range: [{np.min(stress_target):10.4f}, {np.max(stress_target):10.4f}]")

                print("\n" + "=" * 80)
                print("INTERPRETATION:")
                print("=" * 80)

                # Check magnitude to determine if normalized or denormalized
                target_magnitude = np.abs(stress_target).max()

                if target_magnitude < 5.0:
                    print("\n✓ Values appear to be NORMALIZED (range -5 to +5)")
                    print("  - Mean ~0, std ~1 is expected for z-score normalization")
                    print("  - THIS IS WHAT THE MODEL WORKS WITH INTERNALLY")
                elif target_magnitude < 100:
                    print("\n✓ Values appear to be DENORMALIZED but SMALL")
                    print(f"  - Physical units (stress in Pascals)")
                    print(f"  - Magnitude: {target_magnitude:.2f} Pa")
                    print(f"  - This is reasonable for small stress changes")
                else:
                    print("\n✓ Values appear to be DENORMALIZED (PHYSICAL UNITS)")
                    print(f"  - Stress in Pascals")
                    print(f"  - Magnitude: {target_magnitude:.2f} Pa")
                    print(f"  - These are REAL physical stress changes in the material")
                    print(f"  - Values of 30-2000 Pa are NORMAL for deforming materials!")

print("\n" + "=" * 80)
