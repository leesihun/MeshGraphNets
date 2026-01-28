"""
Mesh utilities for converting graph edges to mesh faces and saving results.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def edges_to_triangles(edge_index):
    """
    Reconstruct triangular faces from edge connectivity.

    Since edges were extracted from triangular faces, we find all
    triangles by looking for cycles of length 3 in the graph.

    Args:
        edge_index: (2, E) array of edges (can be bidirectional)

    Returns:
        faces: (F, 3) array of triangular face node indices
    """
    # Build adjacency list (use only unique undirected edges)
    edges = edge_index.T  # (E, 2)

    # Create set of undirected edges for O(1) lookup
    edge_set = set()
    adj = defaultdict(set)
    for i in range(edges.shape[0]):
        u, v = int(edges[i, 0]), int(edges[i, 1])
        if u > v:
            u, v = v, u
        if (u, v) not in edge_set:
            edge_set.add((u, v))
            adj[u].add(v)
            adj[v].add(u)

    # Find all triangles: for each edge (u, v), find common neighbors
    triangles = set()
    for u, v in edge_set:
        # Find nodes connected to both u and v
        common = adj[u] & adj[v]
        for w in common:
            # Sort to avoid duplicates
            tri = tuple(sorted([u, v, w]))
            triangles.add(tri)

    if len(triangles) == 0:
        return np.array([], dtype=np.int64).reshape(0, 3)

    return np.array(list(triangles), dtype=np.int64)


def compute_face_values(faces, node_values):
    """
    Compute face-averaged values from node values.

    Args:
        faces: (F, 3) array of face node indices
        node_values: (N, D) array of node feature values

    Returns:
        face_values: (F, D) array of face-averaged values
    """
    if faces.shape[0] == 0:
        return np.array([], dtype=np.float32).reshape(0, node_values.shape[1])

    # Get values at each vertex of each face
    v0 = node_values[faces[:, 0]]  # (F, D)
    v1 = node_values[faces[:, 1]]  # (F, D)
    v2 = node_values[faces[:, 2]]  # (F, D)

    # Average of three vertices
    face_values = (v0 + v1 + v2) / 3.0
    return face_values


def save_inference_results(output_path, graph, predicted, target):
    """
    Save inference results to HDF5 file with mesh reconstruction.

    The file contains:
    - nodes/pos: (N, 3) node positions
    - nodes/predicted: (N, D) predicted nodal features
    - nodes/target: (N, D) ground truth nodal features
    - edges/index: (2, E) edge connectivity
    - edges/attr: (E, 4) edge attributes [dx, dy, dz, dist]
    - faces/index: (F, 3) reconstructed triangular faces
    - faces/predicted: (F, D) face-averaged predicted values
    - faces/target: (F, D) face-averaged ground truth values

    Args:
        output_path: Path to save the HDF5 file
        graph: PyG Data object with pos, edge_index, edge_attr
        predicted: (N, D) numpy array of predicted node features
        target: (N, D) numpy array of target node features
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Convert tensors to numpy if needed
    pos = graph.pos.cpu().numpy() if hasattr(graph.pos, 'cpu') else np.array(graph.pos)
    edge_index = graph.edge_index.cpu().numpy() if hasattr(graph.edge_index, 'cpu') else np.array(graph.edge_index)
    edge_attr = graph.edge_attr.cpu().numpy() if hasattr(graph.edge_attr, 'cpu') else np.array(graph.edge_attr)

    # Reconstruct triangular faces from edges
    faces = edges_to_triangles(edge_index)

    # Compute face-averaged values for element coloring
    pred_face_values = compute_face_values(faces, predicted)
    target_face_values = compute_face_values(faces, target)

    # Save to HDF5
    with h5py.File(output_path, 'w') as f:
        # Node data
        nodes_grp = f.create_group('nodes')
        nodes_grp.create_dataset('pos', data=pos, dtype=np.float32)
        nodes_grp.create_dataset('predicted', data=predicted, dtype=np.float32)
        nodes_grp.create_dataset('target', data=target, dtype=np.float32)

        # Edge data
        edges_grp = f.create_group('edges')
        edges_grp.create_dataset('index', data=edge_index, dtype=np.int64)
        edges_grp.create_dataset('attr', data=edge_attr, dtype=np.float32)

        # Face data (reconstructed mesh)
        faces_grp = f.create_group('faces')
        faces_grp.create_dataset('index', data=faces, dtype=np.int64)
        faces_grp.create_dataset('predicted', data=pred_face_values, dtype=np.float32)
        faces_grp.create_dataset('target', data=target_face_values, dtype=np.float32)

        # Metadata
        f.attrs['num_nodes'] = pos.shape[0]
        f.attrs['num_edges'] = edge_index.shape[1]
        f.attrs['num_faces'] = faces.shape[0]
        f.attrs['num_features'] = predicted.shape[1]

        # Store sample info if available (handle tensor or scalar)
        sample_id = None
        time_idx = None
        if hasattr(graph, 'sample_id') and graph.sample_id is not None:
            sid = graph.sample_id
            if hasattr(sid, 'cpu'):
                sid = sid.cpu()
            if hasattr(sid, 'numpy'):
                sid = sid.numpy()
            sample_id = int(sid) if np.isscalar(sid) or sid.ndim == 0 else int(sid[0])
            f.attrs['sample_id'] = sample_id

        if hasattr(graph, 'time_idx') and graph.time_idx is not None:
            tid = graph.time_idx
            if hasattr(tid, 'cpu'):
                tid = tid.cpu()
            if hasattr(tid, 'numpy'):
                tid = tid.numpy()
            time_idx = int(tid) if np.isscalar(tid) or tid.ndim == 0 else int(tid[0])
            f.attrs['time_idx'] = time_idx

    # Generate side-by-side visualization
    plot_path = output_path.replace('.h5', '.png')
    plot_mesh_comparison(pos, faces, pred_face_values, target_face_values, plot_path,
                         sample_id=sample_id, time_idx=time_idx)


def plot_mesh_comparison(pos, faces, pred_values, target_values, output_path,
                         feature_idx=-1, sample_id=None, time_idx=None):
    """
    Create side-by-side mesh plots comparing predicted vs ground truth.

    Uses 3D isometric view for proper mesh visualization.
    Colors faces by the specified feature (default: last feature, typically stress).

    Args:
        pos: (N, 3) node positions
        faces: (F, 3) triangular face indices
        pred_values: (F, D) predicted face values
        target_values: (F, D) ground truth face values
        output_path: Path to save the PNG
        feature_idx: Which feature to visualize (default -1 = last, i.e. stress)
        sample_id: Sample ID for plot title (optional)
        time_idx: Timestep index for plot title (optional)
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    if faces.shape[0] == 0:
        return

    # Extract the feature to visualize (face-averaged values)
    pred_colors = pred_values[:, feature_idx]
    target_colors = target_values[:, feature_idx]

    # Use same color scale for both plots
    vmin = min(pred_colors.min(), target_colors.min())
    vmax = max(pred_colors.max(), target_colors.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.jet

    # Build face vertex coordinates
    face_verts = pos[faces]  # (F, 3, 3)

    # Create figure with two 3D subplots
    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Create Poly3DCollection for predicted
    pred_facecolors = cmap(norm(pred_colors))
    poly1 = Poly3DCollection(face_verts, facecolors=pred_facecolors, edgecolors='none', linewidths=0)
    ax1.add_collection3d(poly1)

    # Create Poly3DCollection for ground truth
    target_facecolors = cmap(norm(target_colors))
    poly2 = Poly3DCollection(face_verts, facecolors=target_facecolors, edgecolors='none', linewidths=0)
    ax2.add_collection3d(poly2)

    # Set axis limits based on data
    x_min, x_max = pos[:, 0].min(), pos[:, 0].max()
    y_min, y_max = pos[:, 1].min(), pos[:, 1].max()
    z_min, z_max = pos[:, 2].min(), pos[:, 2].max()

    # Compute ranges with epsilon to avoid zero-size dimensions
    eps = 1e-6
    x_range = max(x_max - x_min, eps)
    y_range = max(y_max - y_min, eps)
    z_range = max(z_max - z_min, eps)

    for ax, title in [(ax1, 'Predicted'), (ax2, 'Ground Truth')]:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        # Isometric view: equal angles
        ax.view_init(elev=30, azim=45)
        ax.set_box_aspect([x_range, y_range, z_range])

    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label('Value')

    # Build title with sample/timestep info
    mae = np.abs(pred_colors - target_colors).mean()
    title_parts = []
    if sample_id is not None:
        title_parts.append(f'Sample {sample_id}')
    if time_idx is not None:
        title_parts.append(f'Timestep {time_idx}')

    if title_parts:
        title_str = ', '.join(title_parts) + f' | MAE: {mae:.4f}'
    else:
        title_str = f'Face-Averaged Comparison (MAE: {mae:.4f})'

    fig.suptitle(title_str, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
