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
        print("RAW NORMALIZATION PARAMETERS (stored in HDF5)")
        print("=" * 60)
        
        # Node feature normalization
        node_mean = None
        node_std = None
        if 'mean' in norm_params and 'std' in norm_params:
            node_mean = norm_params['mean'][:]
            node_std = norm_params['std'][:]
            print("\nNode features (z-score parameters):")
            print(f"  Mean: {node_mean}")
            print(f"  Std:  {node_std}")
        
        if 'min' in norm_params and 'max' in norm_params:
            min_val = norm_params['min'][:]
            max_val = norm_params['max'][:]
            print(f"\n  Min: {min_val}")
            print(f"  Max: {max_val}")
        
        # Edge feature normalization
        edge_mean = None
        edge_std = None
        if 'edge_mean' in norm_params and 'edge_std' in norm_params:
            edge_mean = norm_params['edge_mean'][:]
            edge_std = norm_params['edge_std'][:]
            print("\nEdge features (z-score parameters):")
            print(f"  Mean: {edge_mean}")
            print(f"  Std:  {edge_std}")
        
        # Delta normalization (THIS IS KEY!)
        delta_mean = None
        delta_std = None
        if 'delta_mean' in norm_params and 'delta_std' in norm_params:
            delta_mean = norm_params['delta_mean'][:]
            delta_std = norm_params['delta_std'][:]
            print("\nDelta features (z-score parameters for targets):")
            print(f"  Mean: {delta_mean}")
            print(f"  Std:  {delta_std}")
            
            if 'delta_min' in norm_params and 'delta_max' in norm_params:
                delta_min = norm_params['delta_min'][:]
                delta_max = norm_params['delta_max'][:]
                print(f"  Min:  {delta_min}")
                print(f"  Max:  {delta_max}")
            
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
    
    return node_mean, node_std, edge_mean, edge_std, delta_mean, delta_std

def compute_normalized_statistics(dataset_path, node_mean, node_std, edge_mean, edge_std, delta_mean, delta_std, num_samples=100):
    """
    Sample data from the dataset and compute statistics on normalized features.
    This verifies that normalization is working correctly (mean≈0, std≈1).
    
    Args:
        dataset_path: Path to HDF5 dataset
        node_mean, node_std: Normalization parameters for node features
        edge_mean, edge_std: Normalization parameters for edge features
        delta_mean, delta_std: Normalization parameters for delta features
        num_samples: Number of timesteps to sample for statistics
    """
    print("\n" + "=" * 60)
    print("NORMALIZED FEATURE STATISTICS (sampled from data)")
    print("=" * 60)
    print(f"Sampling {num_samples} timesteps to compute normalized statistics...\n")
    
    if node_mean is None or node_std is None:
        print("  ERROR: Node normalization parameters not available!")
        return
    
    all_node_norm = []
    all_edge_norm = []
    all_delta_norm = []
    
    with h5py.File(dataset_path, 'r') as f:
        sample_ids = sorted([int(k) for k in f['data'].keys()])
        
        # Get dimensions from first sample
        first_sample = f[f'data/{sample_ids[0]}/nodal_data'][:]
        num_timesteps = first_sample.shape[1]
        input_dim = 4  # [disp_x, disp_y, disp_z, stress]
        
        # Sample timesteps evenly across the dataset
        samples_collected = 0
        for sample_id in sample_ids:
            if samples_collected >= num_samples:
                break
            
            data = f[f'data/{sample_id}/nodal_data'][:]  # [features, time, nodes]
            mesh_edge = f[f'data/{sample_id}/mesh_edge'][:]  # [2, edges]
            
            # Sample a few timesteps from this sample
            if num_timesteps > 1:
                timesteps_to_sample = min(5, num_timesteps, num_samples - samples_collected)
                timesteps = np.linspace(0, num_timesteps - 1, timesteps_to_sample, dtype=int)
            else:
                timesteps = [0]
            
            for t in timesteps:
                if samples_collected >= num_samples:
                    break
                
                # Node features: [disp_x, disp_y, disp_z, stress]
                node_feat_raw = data[3:3+input_dim, t, :].T  # [N, 4]
                node_feat_norm = (node_feat_raw - node_mean) / node_std
                all_node_norm.append(node_feat_norm)
                
                # Edge features: [dx, dy, dz, distance]
                if edge_mean is not None and edge_std is not None:
                    pos = (data[:3, t, :] + data[3:6, t, :]).T  # [N, 3] - Deformed position
                    edge_idx = np.concatenate([mesh_edge, mesh_edge[[1, 0], :]], axis=1)
                    rel_pos = pos[edge_idx[1]] - pos[edge_idx[0]]
                    dist = np.linalg.norm(rel_pos, axis=1, keepdims=True)
                    edge_feat_raw = np.concatenate([rel_pos, dist], axis=1)  # [2E, 4]
                    edge_feat_norm = (edge_feat_raw - edge_mean) / edge_std
                    all_edge_norm.append(edge_feat_norm)
                
                # Delta features (only if we have next timestep)
                if delta_mean is not None and delta_std is not None and num_timesteps > 1 and t < num_timesteps - 1:
                    feat_t = data[3:3+input_dim, t, :].T  # [N, 4]
                    feat_t1 = data[3:3+input_dim, t + 1, :].T  # [N, 4]
                    delta_raw = feat_t1 - feat_t
                    delta_norm = (delta_raw - delta_mean) / delta_std
                    all_delta_norm.append(delta_norm)
                
                samples_collected += 1
    
    # Compute statistics on normalized features
    feature_names = ['disp_x', 'disp_y', 'disp_z', 'stress']
    
    if all_node_norm:
        all_node_norm = np.vstack(all_node_norm)
        print("Node features (after normalization):")
        print(f"  Mean: {np.mean(all_node_norm, axis=0)}")
        print(f"  Std:  {np.std(all_node_norm, axis=0)}")
        print(f"  Min:  {np.min(all_node_norm, axis=0)}")
        print(f"  Max:  {np.max(all_node_norm, axis=0)}")
        
        # Sanity check
        node_mean_norm = np.mean(all_node_norm, axis=0)
        node_std_norm = np.std(all_node_norm, axis=0)
        if np.allclose(node_mean_norm, 0, atol=0.1) and np.allclose(node_std_norm, 1, atol=0.1):
            print("  ✓ Node normalization looks correct (mean≈0, std≈1)")
        else:
            print("  ⚠️  WARNING: Node normalization may be incorrect!")
            print(f"     Expected mean≈0, got {node_mean_norm}")
            print(f"     Expected std≈1, got {node_std_norm}")
    
    if all_edge_norm:
        all_edge_norm = np.vstack(all_edge_norm)
        print("\nEdge features (after normalization):")
        print(f"  Mean: {np.mean(all_edge_norm, axis=0)}")
        print(f"  Std:  {np.std(all_edge_norm, axis=0)}")
        print(f"  Min:  {np.min(all_edge_norm, axis=0)}")
        print(f"  Max:  {np.max(all_edge_norm, axis=0)}")
        
        # Sanity check
        edge_mean_norm = np.mean(all_edge_norm, axis=0)
        edge_std_norm = np.std(all_edge_norm, axis=0)
        if np.allclose(edge_mean_norm, 0, atol=0.1) and np.allclose(edge_std_norm, 1, atol=0.1):
            print("  ✓ Edge normalization looks correct (mean≈0, std≈1)")
        else:
            print("  ⚠️  WARNING: Edge normalization may be incorrect!")
            print(f"     Expected mean≈0, got {edge_mean_norm}")
            print(f"     Expected std≈1, got {edge_std_norm}")
    
    if all_delta_norm:
        all_delta_norm = np.vstack(all_delta_norm)
        print("\nDelta features (after normalization):")
        print(f"  Mean: {np.mean(all_delta_norm, axis=0)}")
        print(f"  Std:  {np.std(all_delta_norm, axis=0)}")
        print(f"  Min:  {np.min(all_delta_norm, axis=0)}")
        print(f"  Max:  {np.max(all_delta_norm, axis=0)}")
        
        # Sanity check
        delta_mean_norm = np.mean(all_delta_norm, axis=0)
        delta_std_norm = np.std(all_delta_norm, axis=0)
        if np.allclose(delta_mean_norm, 0, atol=0.1) and np.allclose(delta_std_norm, 1, atol=0.1):
            print("  ✓ Delta normalization looks correct (mean≈0, std≈1)")
        else:
            print("  ⚠️  WARNING: Delta normalization may be incorrect!")
            print(f"     Expected mean≈0, got {delta_mean_norm}")
            print(f"     Expected std≈1, got {delta_std_norm}")
        
        # Show per-feature breakdown
        print("\n  Per-feature breakdown (normalized):")
        for i, fname in enumerate(feature_names):
            print(f"    {fname:8s}: mean={delta_mean_norm[i]:7.4f}, std={delta_std_norm[i]:7.4f}, "
                  f"min={np.min(all_delta_norm[:, i]):8.4f}, max={np.max(all_delta_norm[:, i]):8.4f}")


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
    node_mean, node_std, edge_mean, edge_std, delta_mean, delta_std = check_normalization(dataset_path)
    
    # Compute normalized statistics from actual data
    if node_mean is not None and node_std is not None:
        compute_normalized_statistics(
            dataset_path, 
            node_mean, node_std, 
            edge_mean, edge_std, 
            delta_mean, delta_std,
            num_samples=100  # Sample 100 timesteps
        )
    
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
    print("\nNormalized features should have:")
    print("  - Mean ≈ 0 (within ±0.1)")
    print("  - Std ≈ 1.0 (within ±0.1)")
    print("  - Min/Max typically in range [-5, +5] (outliers may extend beyond)")
