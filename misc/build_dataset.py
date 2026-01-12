"""
Dataset Building and Testing Script

Usage:
    python build_dataset.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from general_modules.load_config import load_config
from general_modules.graph_dataset import MeshGraphDataset, collate_graph_batch
from torch_geometric.loader import DataLoader


def main():
    # Load configuration
    config = load_config("config.txt")

    print("=" * 60)
    print("Building MeshGraphNets Dataset")
    print("=" * 60)
    print(f"Dataset path: {config['dataset_dir']}")
    print(f"Input features: {config['input_var']}")
    print(f"Output features: {config['output_var']}")
    print(f"Normalization range: [{config['norm_min']}, {config['norm_max']}]")
    print(f"Batch size: {config['batch_size']}")
    print()

    # Create dataset
    dataset = MeshGraphDataset(
        h5_path=config['dataset_dir'],
        norm_min=config['norm_min'],
        norm_max=config['norm_max']
    )

    print(f"\nTotal samples: {len(dataset)}")

    # Test loading a single sample
    print("\n" + "=" * 60)
    print("Testing Single Sample Load")
    print("=" * 60)
    sample = dataset[0]
    print(f"Sample 0:")
    print(f"  Node features (x): {sample.x.shape}")
    print(f"  Edge index: {sample.edge_index.shape}")
    print(f"  Edge features: {sample.edge_attr.shape}")
    print(f"  Targets (y): {sample.y.shape}")
    print(f"  Positions: {sample.pos.shape}")
    print(f"  x range: [{sample.x.min():.4f}, {sample.x.max():.4f}]")
    print(f"  y range: [{sample.y.min():.4f}, {sample.y.max():.4f}]")
    print(f"  edge_attr range: [{sample.edge_attr.min():.4f}, {sample.edge_attr.max():.4f}]")

    # Test DataLoader with batching
    print("\n" + "=" * 60)
    print("Testing DataLoader with Batching")
    print("=" * 60)

    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_graph_batch,
        num_workers=0  # Use 0 for Windows compatibility
    )

    batch = next(iter(dataloader))
    print(f"Batch with {config['batch_size']} samples:")
    print(f"  Total nodes in batch: {batch.x.shape[0]}")
    print(f"  Total edges in batch: {batch.edge_index.shape[1]}")
    print(f"  Batch attribute shape: {batch.batch.shape}")
    print(f"  Number of graphs: {batch.num_graphs}")

    print("\n" + "=" * 60)
    print("Dataset Ready!")
    print("=" * 60)
    print(f"You can now use this dataset for training MeshGraphNets.")
    print(f"Example usage:")
    print(f"  from general_modules.graph_dataset import MeshGraphDataset")
    print(f"  dataset = MeshGraphDataset('{config['dataset_dir']}')")
    print(f"  sample = dataset[0]")


if __name__ == "__main__":
    main()
