"""
BFS Bi-Stride Multi-Scale Coarsening for MeshGraphNets.

Implements the pooling strategy from:
  "Efficient Learning of Mesh-Based Physical Simulation with
   Bi-Stride Multi-Scale Graph Neural Network" (Cao et al., ICML 2023)
  https://arxiv.org/abs/2210.02573

Key ideas:
- BFS assigns each node a depth (hops from seed). Even-depth → coarse; odd-depth → fine-only.
- Each fine-only node is assigned to its BFS parent (always even-depth by construction).
- Coarse edges built via 2nd-order adjacency: for every fine edge (u, v) that bridges two
  different coarse clusters, add a coarse edge between those clusters.
- Multi-source BFS handles disconnected meshes (multi-part FEA) correctly.
- Topology-based (not spatial) → no cross-boundary false edges across part interfaces.

Typical coarsening ratio:
  2D triangular mesh  : M ≈ N/4   (avg degree ~6)
  3D tet mesh         : M ≈ N/2   (avg degree ~15-25)
  3D hex mesh         : M ≈ N/2   (avg degree ~20-30)
Use multiscale_levels=2 for 3D meshes to achieve ~N/4 reduction.

Multi-level support (levels > 2):
  BFS coarsening is applied iteratively: level 0 → level 1 → level 2 → ...
  Each level stores its own fine_to_coarse mapping (NOT composed).
  The model uses a V-cycle / U-Net architecture with per-level GnBlocks.
"""

import re

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, breadth_first_order
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from general_modules.edge_features import compute_edge_attr


# ---------------------------------------------------------------------------
# BFS Bi-Stride Coarsening
# ---------------------------------------------------------------------------

def bfs_bistride_coarsen(edge_index_np: np.ndarray, num_nodes: int):
    """
    Compute BFS bi-stride coarsening of a mesh graph.

    Algorithm:
    1. Multi-source BFS assigns each node a depth (hops from seed).
    2. Even-depth nodes → coarse graph (kept); odd-depth → fine-only.
    3. Each fine-only node is assigned to its BFS parent (always even-depth).
    4. Coarse edges via 2nd-order adjacency: iterate fine edges; if src and dst
       map to different coarse clusters, add a coarse edge.

    Handles disconnected meshes (e.g., multi-part FEA with separate steel plate,
    PCB, chips) by restarting BFS at every unvisited seed.

    Uses scipy.sparse.csgraph for C-level BFS speed (~10-50x faster than Python).

    Args:
        edge_index_np: [2, E] int numpy array of bidirectional mesh edges.
                       Must already be bidirectional (as produced by mesh_dataset).
        num_nodes:     N — total number of fine nodes.

    Returns:
        fine_to_coarse:   [N] int32 numpy array. fine_to_coarse[i] is the coarse
                          cluster index (0 … M-1) of fine node i.
        coarse_edge_index:[2, E_c] int64 numpy array. Bidirectional coarse edges.
        num_coarse:       int M — number of coarse nodes.
    """
    # 1. Build CSR adjacency matrix (vectorized, no Python loop)
    row = edge_index_np[0].astype(np.int32)
    col = edge_index_np[1].astype(np.int32)
    data = np.ones(row.shape[0], dtype=np.int8)
    adj = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

    # 2. Multi-source BFS using scipy C-level routines
    n_comp, comp_labels = connected_components(adj, directed=False)

    depth = np.full(num_nodes, -1, dtype=np.int32)
    bfs_parent = np.arange(num_nodes, dtype=np.int32)  # default: own parent (for coarse nodes)

    for comp_id in range(n_comp):
        # Find seed (first node in this component)
        comp_mask = (comp_labels == comp_id)
        seed = int(np.argmax(comp_mask))

        # Scipy BFS: returns nodes in BFS order + predecessor array (C-level speed)
        order, predecessors = breadth_first_order(adj, i_start=seed, directed=False)

        # Compute depth from BFS order (parents guaranteed processed first)
        depth[seed] = 0
        for i in range(1, len(order)):
            node = order[i]
            depth[node] = depth[predecessors[node]] + 1
            bfs_parent[node] = predecessors[node]

    # 3. Even-depth → coarse; odd-depth → fine-only (vectorized)
    coarse_mask = (depth % 2 == 0)
    coarse_nodes = np.where(coarse_mask)[0]  # original fine node IDs of coarse nodes
    num_coarse = int(len(coarse_nodes))

    # Map original fine node ID → contiguous coarse index [0 … M-1] (vectorized)
    coarse_idx_of = np.full(num_nodes, -1, dtype=np.int32)
    coarse_idx_of[coarse_nodes] = np.arange(num_coarse, dtype=np.int32)

    # 4. Build fine_to_coarse [N] (vectorized):
    #    Even-depth nodes → their own coarse index; odd-depth → their BFS parent's coarse index
    parent_or_self = np.where(coarse_mask, np.arange(num_nodes, dtype=np.int32), bfs_parent)
    fine_to_coarse = coarse_idx_of[parent_or_self]  # [N] int32, values in [0, M-1]

    # 5. Build coarse edges via 2nd-order adjacency (vectorized over all fine edges)
    #    For every fine edge (u, v): if fine_to_coarse[u] != fine_to_coarse[v],
    #    add a coarse edge between the two clusters.
    cu = fine_to_coarse[edge_index_np[0]].astype(np.int64)  # [E]
    cv = fine_to_coarse[edge_index_np[1]].astype(np.int64)  # [E]
    cross_mask = cu != cv
    if cross_mask.any():
        # Canonicalize (min, max) to deduplicate, then make bidirectional
        a = np.minimum(cu[cross_mask], cv[cross_mask])
        b = np.maximum(cu[cross_mask], cv[cross_mask])
        pairs_encoded = a * (num_coarse + 1) + b  # unique int per pair
        unique_encoded = np.unique(pairs_encoded)
        a_uniq = unique_encoded // (num_coarse + 1)
        b_uniq = unique_encoded %  (num_coarse + 1)
        src = np.concatenate([a_uniq, b_uniq])
        dst = np.concatenate([b_uniq, a_uniq])
        coarse_edge_index = np.stack([src, dst], axis=0)  # [2, E_c]
    else:
        coarse_edge_index = np.zeros((2, 0), dtype=np.int64)

    return fine_to_coarse, coarse_edge_index, num_coarse


# ---------------------------------------------------------------------------
# Coarse Edge Features
# ---------------------------------------------------------------------------

def compute_coarse_edge_attr(
    reference_pos: np.ndarray,
    deformed_pos: np.ndarray,
    fine_to_coarse: np.ndarray,
    coarse_edge_index: np.ndarray,
    num_coarse: int,
) -> np.ndarray:
    """
    Compute 8-D coarse edge features between coarse node centroids.

    Coarse reference position = mean reference position of all fine nodes in the cluster.
    Coarse deformed position = mean deformed position of all fine nodes in the cluster.
    Edge features use the same 8-D format as fine mesh edges.

    Args:
        reference_pos:     [N, 3] float array — reference positions of fine nodes.
        deformed_pos:      [N, 3] float array — deformed positions of fine nodes
                           (reference pos + displacement, same as used for fine edges).
        fine_to_coarse:    [N] int32 array — coarse cluster index for each fine node.
        coarse_edge_index: [2, E_c] int64 array — coarse edge indices.
        num_coarse:        int M — number of coarse nodes.

    Returns:
        coarse_edge_attr: [E_c, 8] float32 array.
    """
    if coarse_edge_index.shape[1] == 0:
        return np.zeros((0, 8), dtype=np.float32)

    # Compute coarse centroid positions (mean of fine nodes per cluster)
    # Vectorized: use np.add.at for O(N) scatter-add without a Python loop
    coarse_ref_pos = np.zeros((num_coarse, 3), dtype=np.float64)
    coarse_def_pos = np.zeros((num_coarse, 3), dtype=np.float64)
    np.add.at(coarse_ref_pos, fine_to_coarse, reference_pos.astype(np.float64))
    np.add.at(coarse_def_pos, fine_to_coarse, deformed_pos.astype(np.float64))
    counts = np.bincount(fine_to_coarse, minlength=num_coarse).reshape(-1, 1)
    coarse_ref_pos /= np.maximum(counts, 1)
    coarse_def_pos /= np.maximum(counts, 1)

    return compute_edge_attr(coarse_ref_pos, coarse_def_pos, coarse_edge_index)


def compute_coarse_centroids(
    positions: np.ndarray,
    fine_to_coarse: np.ndarray,
    num_coarse: int,
) -> np.ndarray:
    """
    Compute mean centroid positions for coarse clusters.

    Args:
        positions:      [N, 3] float array — positions of fine/current-level nodes.
        fine_to_coarse: [N] int32 array — coarse cluster index for each node.
        num_coarse:     int M — number of coarse nodes.

    Returns:
        coarse_pos: [M, 3] float64 array — centroid positions.
    """
    coarse_pos = np.zeros((num_coarse, 3), dtype=np.float64)
    np.add.at(coarse_pos, fine_to_coarse, positions.astype(np.float64))
    counts = np.bincount(fine_to_coarse, minlength=num_coarse).reshape(-1, 1)
    coarse_pos /= np.maximum(counts, 1)
    return coarse_pos


# ---------------------------------------------------------------------------
# Pool / Unpool Operators
# ---------------------------------------------------------------------------

def pool_features(
    h_fine: torch.Tensor,
    fine_to_coarse: torch.Tensor,
    num_coarse: int,
) -> torch.Tensor:
    """
    Mean-aggregate fine node features to coarse nodes.

    Args:
        h_fine:         [N, D] fine node feature tensor.
        fine_to_coarse: [N] long tensor, coarse cluster index for each fine node.
        num_coarse:     int M — total number of coarse nodes (handles batching correctly
                        when fine_to_coarse values are already offset by batch).

    Returns:
        h_coarse: [M, D] coarse node features.
    """
    return scatter(h_fine, fine_to_coarse, dim=0, dim_size=num_coarse, reduce='mean')


def unpool_features(
    h_coarse: torch.Tensor,
    fine_to_coarse: torch.Tensor,
) -> torch.Tensor:
    """
    Broadcast coarse node features back to fine nodes via simple gather (no learned weights).

    Args:
        h_coarse:       [M, D] coarse node features.
        fine_to_coarse: [N] long tensor, coarse cluster index for each fine node.

    Returns:
        h_fine: [N, D] — each fine node receives its coarse cluster's features.
    """
    return h_coarse[fine_to_coarse]


# ---------------------------------------------------------------------------
# Custom Data class for proper PyG batching of multiscale attributes
# ---------------------------------------------------------------------------

# Regex to extract level index from attribute names like "fine_to_coarse_0"
_LEVEL_RE = re.compile(r'^(fine_to_coarse|coarse_edge_index|coarse_edge_attr|num_coarse)_(\d+)$')


class MultiscaleData(Data):
    """
    PyTorch Geometric Data subclass that handles correct batching of N-level
    multiscale coarsening attributes.

    Per-level attributes (for level i = 0 .. L-1):
        fine_to_coarse_{i}   [N_i] long  — mapping from level i to level i+1
        coarse_edge_index_{i} [2, E_i]   — edge topology at level i+1
        coarse_edge_attr_{i}  [E_i, 8]   — edge features at level i+1
        num_coarse_{i}        [1] long    — node count at level i+1

    When Batch.from_data_list combines multiple MultiscaleData objects:
    - fine_to_coarse_{i} values are offset by cumulative num_coarse_{i} counts
    - coarse_edge_index_{i} values are offset by cumulative num_coarse_{i} counts
    - coarse_edge_attr_{i} is concatenated along dim 0 (no offset needed)
    - num_coarse_{i} values are concatenated → [B] tensor
    """

    def __inc__(self, key: str, value, *args, **kwargs):
        m = _LEVEL_RE.match(key)
        if m:
            prefix, level = m.group(1), m.group(2)
            if prefix in ('fine_to_coarse', 'coarse_edge_index'):
                return int(self[f'num_coarse_{level}'])
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value, *args, **kwargs):
        m = _LEVEL_RE.match(key)
        if m:
            prefix = m.group(1)
            if prefix == 'coarse_edge_index':
                return 1   # [2, E_c] — concatenate along edge dimension
            if prefix in ('fine_to_coarse', 'coarse_edge_attr'):
                return 0   # [N] and [E_c, 8] — concatenate along node/edge dim
        return super().__cat_dim__(key, value, *args, **kwargs)
