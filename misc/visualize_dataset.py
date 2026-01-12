"""
Interactive 3D Dataset Visualization

Visualizes mesh structure and features using PyVista.
Shows: displacement magnitude, stress, and individual displacement components.

Requirements:
    pip install pyvista

Usage:
    python visualize_dataset.py [sample_id]

Example:
    python visualize_dataset.py 1
    python visualize_dataset.py 100
"""

import sys
import numpy as np
import h5py
from pathlib import Path

try:
    import pyvista as pv
except ImportError:
    print("ERROR: PyVista not installed.")
    print("Install with: pip install pyvista")
    sys.exit(1)


def load_sample(h5_path, sample_id):
    """
    Load a single mesh sample from H5 file.

    Returns:
        nodes: (N, 3) node coordinates
        edges: (2, E) edge connectivity
        features: (N, 7) node features [x, y, z, dx, dy, dz, stress]
        feature_names: list of feature names
    """
    with h5py.File(h5_path, 'r') as f:
        # Load nodal data
        nodal_data = f[f'data/{sample_id}/nodal_data'][:]  # (7, 1, N)
        nodal_data = nodal_data[:, 0, :].T  # (N, 7)

        # Load edges
        edges = f[f'data/{sample_id}/mesh_edge'][:]  # (2, E)

        # Feature names
        feature_names = [name.decode() for name in f['metadata/feature_names'][:]]

    # Extract coordinates
    nodes = nodal_data[:, :3]  # (N, 3)

    return nodes, edges, nodal_data, feature_names


def create_mesh(nodes, edges):
    """
    Create PyVista mesh from nodes and edges.

    Args:
        nodes: (N, 3) node coordinates
        edges: (2, E) edge connectivity

    Returns:
        pyvista.PolyData mesh
    """
    # Create point cloud
    mesh = pv.PolyData(nodes)

    # Add edges as lines
    lines = []
    for i in range(edges.shape[1]):
        lines.append(2)  # Number of points in line
        lines.append(edges[0, i])
        lines.append(edges[1, i])

    mesh.lines = lines

    return mesh


def visualize_sample(h5_path, sample_id):
    """
    Create interactive visualization of a mesh sample.
    """
    print(f"Loading sample {sample_id}...")
    nodes, edges, features, feature_names = load_sample(h5_path, sample_id)

    num_nodes = nodes.shape[0]
    num_edges = edges.shape[1]

    print(f"  Nodes: {num_nodes:,}")
    print(f"  Edges: {num_edges:,}")
    print(f"  Features: {feature_names}")

    # Create mesh
    mesh = create_mesh(nodes, edges)

    # Compute derived quantities
    dx = features[:, 3]
    dy = features[:, 4]
    dz = features[:, 5]
    stress = features[:, 6]

    displacement_mag = np.sqrt(dx**2 + dy**2 + dz**2)

    # Add scalar fields to mesh
    mesh['dx'] = dx
    mesh['dy'] = dy
    mesh['dz'] = dz
    mesh['stress'] = stress
    mesh['displacement'] = displacement_mag

    # Print statistics
    print(f"\nFeature Statistics:")
    print(f"  Displacement magnitude: [{displacement_mag.min():.6f}, {displacement_mag.max():.6f}]")
    print(f"  Stress: [{stress.min():.2f}, {stress.max():.2f}]")
    print(f"  dx: [{dx.min():.6f}, {dx.max():.6f}]")
    print(f"  dy: [{dy.min():.6f}, {dy.max():.6f}]")
    print(f"  dz: [{dz.min():.6f}, {dz.max():.6f}]")

    # Create plotter
    plotter = pv.Plotter(shape=(2, 2), window_size=[1600, 1200])

    # Subplot 1: Displacement magnitude
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh, scalars='displacement', cmap='viridis',
                     show_edges=True, edge_color='gray', opacity=0.8,
                     scalar_bar_args={'title': 'Displacement Magnitude (mm)'})
    plotter.add_text(f"Sample {sample_id}: Displacement Magnitude", font_size=10)

    # Subplot 2: Stress
    plotter.subplot(0, 1)
    plotter.add_mesh(mesh, scalars='stress', cmap='coolwarm',
                     show_edges=True, edge_color='gray', opacity=0.8,
                     scalar_bar_args={'title': 'Stress (MPa)'})
    plotter.add_text(f"Sample {sample_id}: Stress (MPa)", font_size=10)

    # Subplot 3: X-displacement
    plotter.subplot(1, 0)
    plotter.add_mesh(mesh, scalars='dx', cmap='RdBu_r',
                     show_edges=True, edge_color='gray', opacity=0.8,
                     scalar_bar_args={'title': 'X-Displacement (mm)'})
    plotter.add_text(f"Sample {sample_id}: X-Displacement", font_size=10)

    # Subplot 4: Combined Y and Z displacement
    plotter.subplot(1, 1)
    yz_displacement = np.sqrt(dy**2 + dz**2)
    mesh['yz_displacement'] = yz_displacement
    plotter.add_mesh(mesh, scalars='yz_displacement', cmap='plasma',
                     show_edges=True, edge_color='gray', opacity=0.8,
                     scalar_bar_args={'title': 'YZ-Displacement (mm)'})
    plotter.add_text(f"Sample {sample_id}: YZ-Displacement", font_size=10)

    # Link camera across all subplots
    plotter.link_views()

    print("\n" + "="*60)
    print("Interactive Visualization Controls:")
    print("  - Left click + drag: Rotate")
    print("  - Right click + drag: Pan")
    print("  - Scroll: Zoom")
    print("  - 'q': Quit")
    print("="*60)

    plotter.show()


def main():
    h5_path = "./dataset/dataset.h5"

    if not Path(h5_path).exists():
        print(f"ERROR: Dataset not found at {h5_path}")
        sys.exit(1)

    # Get sample ID from command line or use default
    if len(sys.argv) > 1:
        sample_id = sys.argv[1]
    else:
        sample_id = '1'
        print(f"No sample ID provided, using default: {sample_id}")
        print(f"Usage: python visualize_dataset.py [sample_id]\n")

    # Verify sample exists
    with h5py.File(h5_path, 'r') as f:
        available_samples = sorted(f['data'].keys(), key=int)
        if sample_id not in available_samples:
            print(f"ERROR: Sample {sample_id} not found.")
            print(f"Available samples: {available_samples[:10]} ... {available_samples[-5:]}")
            sys.exit(1)

    visualize_sample(h5_path, sample_id)


if __name__ == "__main__":
    main()
