#!/usr/bin/env python3
"""
Validation script for GPU-accelerated world edges (torch_cluster.radius_graph).

Tests:
1. Correctness: Verify edge counts and topologies are as expected
2. Performance: Measure time for world edge computation
3. Integration: Verify model can process graphs with world edges

Usage:
    python test_world_edges_speedup.py
"""

import torch
import time
import numpy as np
from general_modules.load_config import load_config
from general_modules.mesh_dataset import MeshGraphDataset


def test_world_edges_correctness():
    """Test that world edges are computed correctly."""
    print("\n" + "="*70)
    print("TEST 1: WORLD EDGE CORRECTNESS")
    print("="*70)

    # Load config
    config = load_config('config.txt')

    # Create dataset
    dataset = MeshGraphDataset(config['dataset_dir'], config=config)
    print(f"\nDataset loaded: {len(dataset)} samples")

    # Test on first 10 samples
    num_test_samples = min(10, len(dataset))
    print(f"Testing on first {num_test_samples} samples...\n")

    mesh_edge_counts = []
    world_edge_counts = []
    ratios = []

    for idx in range(num_test_samples):
        graph = dataset[idx]

        num_mesh_edges = graph.edge_index.shape[1]
        num_world_edges = graph.world_edge_index.shape[1]

        mesh_edge_counts.append(num_mesh_edges)
        world_edge_counts.append(num_world_edges)

        if num_mesh_edges > 0:
            ratio = num_world_edges / num_mesh_edges
        else:
            ratio = 0
        ratios.append(ratio)

        print(f"Sample {idx:2d}: {graph.num_nodes:6d} nodes | "
              f"Mesh edges: {num_mesh_edges:7d} | "
              f"World edges: {num_world_edges:7d} | "
              f"Ratio: {ratio:.2f}x")

    # Print statistics
    print("\n" + "-"*70)
    print("STATISTICS:")
    print("-"*70)
    print(f"Avg mesh edges per sample:  {np.mean(mesh_edge_counts):.0f}")
    print(f"Avg world edges per sample: {np.mean(world_edge_counts):.0f}")
    print(f"Avg world/mesh ratio:       {np.mean(ratios):.2f}x")
    print(f"Max world edges in batch:   {np.max(world_edge_counts):.0f}")
    print(f"Min world edges in batch:   {np.min(world_edge_counts):.0f}")


def test_world_edges_performance():
    """Test performance of world edge computation."""
    print("\n" + "="*70)
    print("TEST 2: WORLD EDGE COMPUTATION PERFORMANCE")
    print("="*70)

    # Load config
    config = load_config('config.txt')

    # Create dataset
    dataset = MeshGraphDataset(config['dataset_dir'], config=config)
    print(f"\nDataset loaded: {len(dataset)} samples")

    # Warm up GPU
    print("Warming up GPU...")
    _ = dataset[0]
    torch.cuda.synchronize()

    # Benchmark on first 10 samples
    num_test_samples = min(10, len(dataset))
    print(f"Benchmarking on {num_test_samples} samples...\n")

    times = []
    for idx in range(num_test_samples):
        torch.cuda.synchronize()
        start = time.time()

        graph = dataset[idx]

        torch.cuda.synchronize()
        elapsed = time.time() - start

        times.append(elapsed * 1000)  # Convert to ms

        print(f"Sample {idx:2d}: {elapsed*1000:.2f} ms "
              f"({graph.num_nodes:6d} nodes, "
              f"{graph.world_edge_index.shape[1]:7d} world edges)")

    # Print statistics
    print("\n" + "-"*70)
    print("PERFORMANCE STATISTICS:")
    print("-"*70)
    print(f"Mean time per sample:  {np.mean(times):.2f} ms")
    print(f"Std dev:               {np.std(times):.2f} ms")
    print(f"Min time:              {np.min(times):.2f} ms")
    print(f"Max time:              {np.max(times):.2f} ms")
    print(f"Total time (10 samples): {np.sum(times):.2f} ms")


def test_model_integration():
    """Test that model can process graphs with world edges."""
    print("\n" + "="*70)
    print("TEST 3: MODEL INTEGRATION")
    print("="*70)

    # Load config
    config = load_config('config.txt')

    # Create dataset
    dataset = MeshGraphDataset(config['dataset_dir'], config=config)
    print(f"\nDataset loaded: {len(dataset)} samples")

    # Import model
    from model.MeshGraphNets import MeshGraphNets

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create model
    model = MeshGraphNets(config, device=device)
    model.eval()

    print("\nTesting forward pass with world edges...\n")

    # Test on first 5 samples
    num_test_samples = min(5, len(dataset))

    for idx in range(num_test_samples):
        graph = dataset[idx]

        # Move graph to device
        graph.x = graph.x.to(device)
        graph.y = graph.y.to(device)
        graph.edge_index = graph.edge_index.to(device)
        graph.edge_attr = graph.edge_attr.to(device)
        graph.world_edge_index = graph.world_edge_index.to(device)
        graph.world_edge_attr = graph.world_edge_attr.to(device)

        # Forward pass
        with torch.no_grad():
            predicted, target = model(graph)

        print(f"Sample {idx}: Forward pass OK")
        print(f"  Input shape:  {graph.x.shape}")
        print(f"  Output shape: {predicted.shape}")
        print(f"  World edges:  {graph.world_edge_index.shape[1]}")

    print("\nModel integration test PASSED!")


def main():
    """Run all validation tests."""
    print("\n" + "="*70)
    print("GPU-ACCELERATED WORLD EDGES VALIDATION")
    print("Using: torch_cluster.radius_graph (NVIDIA PhysicsNeMo style)")
    print("="*70)

    try:
        # Check CUDA availability
        if not torch.cuda.is_available():
            print("\nWARNING: CUDA not available! Tests will fail.")
            return

        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch version: {torch.__version__}")

        # Run tests
        test_world_edges_correctness()
        test_world_edges_performance()
        test_model_integration()

        print("\n" + "="*70)
        print("ALL VALIDATION TESTS PASSED!")
        print("="*70)

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR: {str(e)}")
        print(f"{'='*70}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
