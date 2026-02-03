"""
Quick script to check normalization statistics and visualize denormalized values.
"""
import h5py
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def check_normalization(dataset_path):
    print(f"Checking normalization statistics in: {dataset_path}\n")
    
    with h5py.File(dataset_path, 'r') as f:
        norm_params = f['metadata/normalization_params']
        
        print("=" * 60)
        print("NORMALIZATION STATISTICS")
        print("=" * 60)
        
        # Node feature normalization
        if 'mean' in norm_params and 'std' in norm_params:
            mean = norm_params['mean'][:]
            std = norm_params['std'][:]
            print("\nNode features (z-score):")
            print(f"  Mean: {mean}")
            print(f"  Std:  {std}")
        
        if 'min' in norm_params and 'max' in norm_params:
            min_val = norm_params['min'][:]
            max_val = norm_params['max'][:]
            print(f"\n  Min: {min_val}")
            print(f"  Max: {max_val}")
        
        # Delta normalization (THIS IS KEY!)
        if 'delta_mean' in norm_params and 'delta_std' in norm_params:
            delta_mean = norm_params['delta_mean'][:]
            delta_std = norm_params['delta_std'][:]
            print("\n" + "=" * 60)
            print("DELTA NORMALIZATION (for target values)")
            print("=" * 60)
            print(f"  Delta Mean: {delta_mean}")
            print(f"  Delta Std:  {delta_std}")
            print("\nFeature names: [disp_x, disp_y, disp_z, stress]")
            
            # Show what normalized values correspond to
            print("\n" + "=" * 60)
            print("DENORMALIZATION EXAMPLES")
            print("=" * 60)
            test_normalized_values = [-0.0002, -0.0001, 0, 0.0001, 0.0002]
            
            for feature_idx in range(len(delta_mean)):
                feature_names = ['disp_x', 'disp_y', 'disp_z', 'stress']
                print(f"\nFeature {feature_idx} ({feature_names[feature_idx]}):")
                print(f"  Normalized -> Actual (denormalized)")
                for norm_val in test_normalized_values:
                    actual_val = norm_val * delta_std[feature_idx] + delta_mean[feature_idx]
                    print(f"  {norm_val:8.4f} -> {actual_val:12.6f}")
        else:
            print("\n  WARNING: No delta normalization params found!")

def check_test_results(test_results_path):
    """Check what's actually saved in a test results file."""
    if not os.path.exists(test_results_path):
        print(f"\nTest results file not found: {test_results_path}")
        return
    
    print("\n" + "=" * 60)
    print("TEST RESULTS INSPECTION")
    print("=" * 60)
    print(f"File: {test_results_path}\n")
    
    with h5py.File(test_results_path, 'r') as f:
        # Check node data
        if 'nodes/predicted' in f:
            predicted = f['nodes/predicted'][:]
            target = f['nodes/target'][:]
            
            print(f"Node data shape: {predicted.shape}")
            print(f"  Predicted range: [{predicted.min():.6f}, {predicted.max():.6f}]")
            print(f"  Target range:    [{target.min():.6f}, {target.max():.6f}]")
            
            print("\nPer-feature statistics (predicted):")
            feature_names = ['disp_x', 'disp_y', 'disp_z', 'stress']
            for i in range(predicted.shape[1]):
                fname = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                print(f"  {fname:8s}: min={predicted[:, i].min():10.6f}, max={predicted[:, i].max():10.6f}, mean={predicted[:, i].mean():10.6f}")
        
        # Check face data
        if 'faces/predicted' in f:
            face_pred = f['faces/predicted'][:]
            face_target = f['faces/target'][:]
            
            print(f"\nFace data shape: {face_pred.shape}")
            print(f"  Face predicted range: [{face_pred.min():.6f}, {face_pred.max():.6f}]")
            print(f"  Face target range:    [{face_target.min():.6f}, {face_target.max():.6f}]")

if __name__ == '__main__':
    # Check dataset normalization
    dataset_path = './dataset/deforming_plate.h5'
    check_normalization(dataset_path)
    
    # Check a test result if it exists
    import glob
    test_files = glob.glob('outputs/test/*/0/*.h5')
    if test_files:
        print(f"\n\nFound {len(test_files)} test result files")
        check_test_results(test_files[0])
    else:
        print("\n\nNo test results found yet. Run training first to generate test outputs.")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("After the fix:")
    print("  - Model outputs: normalized deltas (z-scores)")
    print("  - Visualization: DENORMALIZED deltas (actual MPa, mm)")
    print("  - Colorbar labels: Include units (MPa for stress, mm for displacement)")
    print("  - Title: Shows 'Δ Stress' or 'Δ Disp X/Y/Z' with proper MAE in units")
    print("\nExpected stress delta range: ~5-30 MPa")
    print("Expected displacement delta range: ~0.001-0.1 mm")
