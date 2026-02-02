"""Simple test to verify HDF5 file locking fix."""

import sys
from general_modules.load_config import load_config
from general_modules.mesh_dataset import MeshGraphDataset

try:
    print("="*60)
    print("Testing HDF5 File Locking Fix")
    print("="*60)

    config = load_config('config.txt')
    print("\n[1] Loading dataset with parallel stats enabled...")

    dataset = MeshGraphDataset(config['dataset_dir'], config)

    print("\n[2] Dataset loaded successfully!")
    print(f"    - Samples: {len(dataset.sample_ids)}")
    print(f"    - Timesteps: {dataset.num_timesteps}")
    print(f"    - Node mean: {dataset.node_mean}")
    print(f"    - Node std: {dataset.node_std}")

    print("\n" + "="*60)
    print("SUCCESS: No file locking errors!")
    print("="*60)

except BlockingIOError as e:
    print("\n" + "="*60)
    print(f"FAILED: BlockingIOError still occurring!")
    print(f"Error: {e}")
    print("="*60)
    sys.exit(1)

except Exception as e:
    print("\n" + "="*60)
    print(f"ERROR: {type(e).__name__}: {e}")
    print("="*60)
    import traceback
    traceback.print_exc()
    sys.exit(1)
