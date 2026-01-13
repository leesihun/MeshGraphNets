"""
3D Dataset Visualization using Matplotlib

Visualizes mesh as triangulated surface with physical field coloring.

Usage:
    python visualize_dataset.py [sample_id] [--field stress|displacement|dx|dy|dz]

Example:
    python visualize_dataset.py 1
    python visualize_dataset.py 1 --field displacement
"""

import sys
import argparse
import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from collections import defaultdict


def load_sample(h5_path, sample_id):
    """Load a single mesh sample from H5 file."""
    with h5py.File(h5_path, 'r') as f:
        nodal_data = f[f'data/{sample_id}/nodal_data'][:]  # (7, 1, N)
        nodal_data = nodal_data[:, 0, :].T  # (N, 7)
        edges = f[f'data/{sample_id}/mesh_edge'][:]  # (2, E)
        feature_names = [name.decode() for name in f['metadata/feature_names'][:]]

    return nodal_data, edges, feature_names


def find_triangles_from_edges(edges):
    """
    Reconstruct triangular faces from edge connectivity.

    For each edge (u, v), find nodes w that form triangles (u, v, w)
    where edges (u, w) and (v, w) also exist.
    """
    print("  Reconstructing triangular faces from edges...")

    # Build adjacency set for fast lookup
    adj = defaultdict(set)
    edge_set = set()

    for i in range(edges.shape[1]):
        u, v = edges[0, i], edges[1, i]
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))

    # Find triangles
    triangles = set()
    for i in range(edges.shape[1]):
        u, v = edges[0, i], edges[1, i]
        # Find common neighbors
        common = adj[u] & adj[v]
        for w in common:
            # Sort to avoid duplicates
            tri = tuple(sorted([u, v, w]))
            triangles.add(tri)

    triangles = np.array(list(triangles))
    print(f"  Found {len(triangles):,} triangular faces")
    return triangles


def visualize_sample(h5_path, sample_id, field='stress'):
    """
    Visualize mesh as triangulated surface.
    """
    print(f"Loading sample {sample_id}...")
    features, edges, feature_names = load_sample(h5_path, sample_id)

    num_nodes = features.shape[0]
    num_edges = edges.shape[1]

    print(f"  Nodes: {num_nodes:,}")
    print(f"  Edges: {num_edges:,}")

    # Extract coordinates and fields
    coords = features[:, :3]  # (N, 3)
    dx, dy, dz = features[:, 3], features[:, 4], features[:, 5]
    stress = features[:, 6]
    displacement = np.sqrt(dx**2 + dy**2 + dz**2)

    # Select field to visualize
    field_map = {
        'stress': (stress, 'Stress (MPa)', 'coolwarm'),
        'displacement': (displacement, 'Displacement Magnitude (mm)', 'viridis'),
        'dx': (dx, 'X-Displacement (mm)', 'RdBu_r'),
        'dy': (dy, 'Y-Displacement (mm)', 'RdBu_r'),
        'dz': (dz, 'Z-Displacement (mm)', 'RdBu_r'),
    }

    if field not in field_map:
        print(f"Unknown field: {field}. Using 'stress'.")
        field = 'stress'

    values, label, cmap_name = field_map[field]

    # Print field statistics
    print(f"\n{label}:")
    print(f"  Min: {values.min():.4f}")
    print(f"  Max: {values.max():.4f}")
    print(f"  Mean: {values.mean():.4f}")

    # Find triangular faces
    triangles = find_triangles_from_edges(edges)

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Get triangle vertices
    tri_verts = coords[triangles]  # (num_triangles, 3, 3)

    # Compute face colors (average of vertex values)
    face_values = values[triangles].mean(axis=1)  # (num_triangles,)

    # Normalize values for colormap
    cmap = plt.get_cmap(cmap_name)
    vmin, vmax = values.min(), values.max()
    norm_values = (face_values - vmin) / (vmax - vmin + 1e-10)
    face_colors = cmap(norm_values)

    # Create mesh collection
    mesh = Poly3DCollection(
        tri_verts,
        facecolors=face_colors,
        edgecolors='black',
        linewidths=0.1,
        alpha=0.9
    )
    ax.add_collection3d(mesh)

    # Set axis limits
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    max_range = max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min()) / 2
    mid_x = (x.max() + x.min()) / 2
    mid_y = (y.max() + y.min()) / 2
    mid_z = (z.max() + z.min()) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Sample {sample_id}: {label}')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(label)

    plt.tight_layout()
    print("\nControls: drag to rotate, right-drag to zoom")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize mesh dataset')
    parser.add_argument('sample_id', nargs='?', default='1', help='Sample ID to visualize')
    parser.add_argument('--field', default='stress',
                        choices=['stress', 'displacement', 'dx', 'dy', 'dz'],
                        help='Field to visualize')
    args = parser.parse_args()

    # Get project root
    project_root = Path(__file__).parent.parent
    h5_path = project_root / "dataset" / "dataset.h5"

    if not h5_path.exists():
        print(f"ERROR: Dataset not found at {h5_path}")
        sys.exit(1)

    # Verify sample exists
    with h5py.File(h5_path, 'r') as f:
        available_samples = sorted(f['data'].keys(), key=int)
        if args.sample_id not in available_samples:
            print(f"ERROR: Sample {args.sample_id} not found.")
            print(f"Available: {available_samples[:5]} ... {available_samples[-3:]}")
            sys.exit(1)

    visualize_sample(h5_path, args.sample_id, field=args.field)


if __name__ == "__main__":
    main()
