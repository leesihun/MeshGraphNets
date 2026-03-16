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
"""

from collections import deque

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import scatter


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
    # 1. Build adjacency list (edge_index is already bidirectional, so no need to add reverse)
    adj = [[] for _ in range(num_nodes)]
    for i in range(edge_index_np.shape[1]):
        u = int(edge_index_np[0, i])
        v = int(edge_index_np[1, i])
        adj[u].append(v)

    # 2. Multi-source BFS — handles disconnected mesh components
    depth = np.full(num_nodes, -1, dtype=np.int32)
    bfs_parent = np.arange(num_nodes, dtype=np.int32)  # default: own parent (for coarse nodes)

    for seed in range(num_nodes):
        if depth[seed] != -1:
            continue
        depth[seed] = 0
        queue = deque([seed])
        while queue:
            node = queue.popleft()
            for nbr in adj[node]:
                if depth[nbr] == -1:
                    depth[nbr] = depth[node] + 1
                    bfs_parent[nbr] = node
                    queue.append(nbr)

    # 3. Even-depth → coarse; odd-depth → fine-only
    coarse_mask = (depth % 2 == 0)
    coarse_nodes = np.where(coarse_mask)[0]  # original fine node IDs of coarse nodes
    num_coarse = int(len(coarse_nodes))

    # Map original fine node ID → contiguous coarse index [0 … M-1]
    coarse_idx_of = np.full(num_nodes, -1, dtype=np.int32)
    for ci, fn in enumerate(coarse_nodes):
        coarse_idx_of[fn] = ci

    # 4. Build fine_to_coarse [N]: even-depth maps to itself; odd-depth to its BFS parent
    fine_to_coarse = np.empty(num_nodes, dtype=np.int32)
    for i in range(num_nodes):
        if coarse_mask[i]:
            fine_to_coarse[i] = coarse_idx_of[i]
        else:
            # BFS parent of an odd-depth node is always even-depth
            fine_to_coarse[i] = coarse_idx_of[bfs_parent[i]]

    # 5. Build coarse edges via 2nd-order adjacency
    #    For every fine edge (u, v): if fine_to_coarse[u] != fine_to_coarse[v],
    #    add a coarse edge between the two clusters.
    coarse_edge_set = set()
    for i in range(edge_index_np.shape[1]):
        cu = int(fine_to_coarse[edge_index_np[0, i]])
        cv = int(fine_to_coarse[edge_index_np[1, i]])
        if cu != cv:
            # Store as canonical (min, max) to avoid duplicates
            if cu < cv:
                coarse_edge_set.add((cu, cv))
            else:
                coarse_edge_set.add((cv, cu))

    if coarse_edge_set:
        pairs = np.array(sorted(coarse_edge_set), dtype=np.int64)  # [E_c/2, 2]
        # Make bidirectional
        src = np.concatenate([pairs[:, 0], pairs[:, 1]])
        dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
        coarse_edge_index = np.stack([src, dst], axis=0)  # [2, E_c]
    else:
        coarse_edge_index = np.zeros((2, 0), dtype=np.int64)

    return fine_to_coarse, coarse_edge_index, num_coarse


# ---------------------------------------------------------------------------
# Coarse Edge Features
# ---------------------------------------------------------------------------

def compute_coarse_edge_attr(
    deformed_pos: np.ndarray,
    fine_to_coarse: np.ndarray,
    coarse_edge_index: np.ndarray,
    num_coarse: int,
) -> np.ndarray:
    """
    Compute coarse edge features [dx, dy, dz, distance] between coarse node centroids.

    Coarse node position = mean deformed position of all fine nodes in the cluster.
    Edge features use the same format as fine mesh edges so the same normalization
    stats structure applies (separate coarse_edge_mean/std computed during startup).

    Args:
        deformed_pos:      [N, 3] float array — deformed positions of fine nodes
                           (reference pos + displacement, same as used for fine edges).
        fine_to_coarse:    [N] int32 array — coarse cluster index for each fine node.
        coarse_edge_index: [2, E_c] int64 array — coarse edge indices.
        num_coarse:        int M — number of coarse nodes.

    Returns:
        coarse_edge_attr: [E_c, 4] float32 array of [dx, dy, dz, distance].
    """
    if coarse_edge_index.shape[1] == 0:
        return np.zeros((0, 4), dtype=np.float32)

    # Compute coarse centroid positions (mean of fine nodes per cluster)
    coarse_pos = np.zeros((num_coarse, 3), dtype=np.float64)
    counts = np.zeros(num_coarse, dtype=np.int32)
    for i in range(len(fine_to_coarse)):
        ci = int(fine_to_coarse[i])
        coarse_pos[ci] += deformed_pos[i]
        counts[ci] += 1
    coarse_pos /= np.maximum(counts[:, None], 1)  # avoid div-by-zero

    src = coarse_edge_index[0]
    dst = coarse_edge_index[1]
    rel_pos = coarse_pos[dst] - coarse_pos[src]  # [E_c, 3]
    dist = np.linalg.norm(rel_pos, axis=1, keepdims=True)  # [E_c, 1]
    return np.concatenate([rel_pos, dist], axis=1).astype(np.float32)


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

class MultiscaleData(Data):
    """
    PyTorch Geometric Data subclass that handles correct batching of multiscale
    coarsening attributes (fine_to_coarse and coarse_edge_index need index offsets).

    fine_to_coarse[i]   ∈ [0, M-1] — coarse cluster index for fine node i.
    coarse_edge_index   ∈ [0, M-1] — coarse graph edge indices.
    num_coarse          = M (stored as [1] long tensor so batching gives [B] tensor).

    When Batch.from_data_list combines multiple MultiscaleData objects:
    - fine_to_coarse values are offset by cumulative M counts (via __inc__)
    - coarse_edge_index values are offset by cumulative M counts (via __inc__)
    - coarse_edge_attr is concatenated along dim 0 (fine: no offset needed)
    - num_coarse values are concatenated → [B] tensor for total-coarse computation
    """

    def __inc__(self, key: str, value, *args, **kwargs):
        if key == 'fine_to_coarse':
            # fine_to_coarse holds coarse node indices; increment by this sample's M
            return int(self.num_coarse)
        if key == 'coarse_edge_index':
            # coarse_edge_index holds coarse node indices; increment by this sample's M
            return int(self.num_coarse)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value, *args, **kwargs):
        if key == 'coarse_edge_index':
            return 1   # [2, E_c] — concatenate along edge dimension (like edge_index)
        if key in ('fine_to_coarse', 'coarse_edge_attr'):
            return 0   # [N] and [E_c, 4] — concatenate along node/edge dim
        return super().__cat_dim__(key, value, *args, **kwargs)
