"""
Test script for multi-timestep MeshGraphDataset implementation
"""
import sys
sys.path.append('.')

from general_modules.load_config import load_config
from general_modules.mesh_dataset import MeshGraphDataset

def test_dataset():
    # Load configuration
    config = load_config('config.txt')

    # Update dataset path to correct file
    config['dataset_dir'] = './dataset/dataset.h5'

    print("="*60)
    print("Testing MeshGraphDataset with timestep handling")
    print("="*60)

    # Create dataset
    dataset = MeshGraphDataset(
        h5_file=config['dataset_dir'],
        config=config
    )

    print(f"\nDataset length: {len(dataset)}")
    print(f"Number of timesteps: {dataset.num_timesteps}")
    print(f"Input dimension: {dataset.input_dim}")
    print(f"Output dimension: {dataset.output_dim}")

    # Test loading first sample
    print("\n" + "="*60)
    print("Loading first sample...")
    print("="*60)

    sample = dataset[0]

    print(f"\nSample 0 structure:")
    print(f"  sample_id: {sample.sample_id}")
    print(f"  time_idx: {getattr(sample, 'time_idx', 'N/A (single timestep)')}")
    print(f"  x shape: {sample.x.shape} (node features)")
    print(f"  y shape: {sample.y.shape} (targets)")
    print(f"  pos shape: {sample.pos.shape} (positions)")
    print(f"  edge_index shape: {sample.edge_index.shape}")
    print(f"  edge_attr shape: {sample.edge_attr.shape}")

    # Verify single timestep behavior
    if dataset.num_timesteps == 1:
        print("\n" + "="*60)
        print("Single timestep verification:")
        print("="*60)

        # Check that x[:, 0:3] contains positions
        print(f"\n  x[:5, 0:3] (positions):\n{sample.x[:5, 0:3]}")
        print(f"  pos[:5] (should match):\n{sample.pos[:5]}")

        # Check that x[:, 3:] are zeros
        if dataset.input_dim > 3:
            print(f"\n  x[:5, 3:] (should be zeros):\n{sample.x[:5, 3:]}")
            zeros_check = (sample.x[:, 3:] == 0).all()
            print(f"  All non-position features are zero: {zeros_check}")

        # Check y excludes positions
        print(f"\n  y[:5] (should exclude positions):\n{sample.y[:5]}")
        print(f"  y contains {sample.y.shape[1]} features (should be {dataset.output_dim})")

    # Test a few more samples
    print("\n" + "="*60)
    print("Testing multiple samples...")
    print("="*60)

    for i in [1, 10, 100]:
        if i < len(dataset):
            sample = dataset[i]
            print(f"\nSample {i}:")
            print(f"  sample_id: {sample.sample_id}")
            print(f"  num_nodes: {sample.x.shape[0]}")
            print(f"  num_edges: {sample.edge_index.shape[1]}")

    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)

if __name__ == '__main__':
    test_dataset()
