import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Set, Tuple
from torch_geometric.data import Data
from scipy.spatial import KDTree

# Try to import torch_cluster for GPU acceleration; fall back to scipy.KDTree if unavailable
try:
    from torch_cluster import radius_graph
    HAS_TORCH_CLUSTER = True
except ImportError:
    HAS_TORCH_CLUSTER = False

class MeshGraphDataset(Dataset):

    def __init__(self, h5_file: str, config: Dict):
        self.h5_file = h5_file
        self.config = config
        # Graph and feature parameters
        self.input_dim = config.get('input_var')  # Physical features only (4)
        self.output_dim = config.get('output_var')  # Physical features only (4)

        # Node type parameters
        self.use_node_types = config.get('use_node_types', False)
        self.num_node_types = None  # Will be computed from dataset if node types exist
        self.node_type_to_idx = None  # Mapping from node type values to contiguous indices

        # World edge parameters
        self.use_world_edges = config.get('use_world_edges', False)
        self.world_radius_multiplier = config.get('world_radius_multiplier', 1.5)
        self.world_max_num_neighbors = config.get('world_max_num_neighbors', 64)
        self.world_edge_radius = None  # Computed from mesh statistics
        self.min_edge_length = None    # Computed from first sample

        # Determine which world edge backend to use
        requested_backend = config.get('world_edge_backend', 'torch_cluster').lower()
        if requested_backend == 'torch_cluster' and HAS_TORCH_CLUSTER:
            self.world_edge_backend = 'torch_cluster'
        elif requested_backend == 'scipy_kdtree' or not HAS_TORCH_CLUSTER:
            self.world_edge_backend = 'scipy_kdtree'
        else:
            # Invalid backend requested, default to available option
            self.world_edge_backend = 'torch_cluster' if HAS_TORCH_CLUSTER else 'scipy_kdtree'

        print(f"Loading MeshGraphDataset: {h5_file}")
        print(f"  input_dim: {self.input_dim}, output_dim: {self.output_dim}")
        print(f"  use_node_types: {self.use_node_types}")
        print(f"  use_world_edges: {self.use_world_edges}")
        if self.use_world_edges:
            print(f"  world_radius_multiplier: {self.world_radius_multiplier}")
            print(f"  world_max_num_neighbors: {self.world_max_num_neighbors}")
            print(f"  world_edge_backend: {self.world_edge_backend}")

        # Load sample IDs, timesteps, and normalization params
        with h5py.File(h5_file, 'r') as f:
            if 'data' not in f:
                raise ValueError(f"HDF5 file missing 'data' group")

            self.sample_ids = sorted([int(k) for k in f['data'].keys()])

            # Check number of timesteps from first sample
            sample_id = self.sample_ids[0]
            data_shape = f[f'data/{sample_id}/nodal_data'].shape
            self.num_timesteps = data_shape[1]  # Shape: (features, time, nodes)

            self.delta_mean = None
            self.delta_std = None

        print(f"Found {len(self.sample_ids)} samples")
        print(f"  num_timesteps: {self.num_timesteps}")

        # Compute z-score normalization statistics for node and edge features
        self._compute_zscore_stats()

        if self.use_node_types:
            self._compute_node_type_info()

        if self.use_world_edges:
            self._compute_world_edge_radius()

    def _compute_zscore_stats(self) -> None:
        """Compute z-score normalization statistics (mean, std) for node, edge, and delta features.

        Also updates the HDF5 file with correct delta normalization parameters computed from
        actual data, fixing any incorrect pre-stored values.
        """
        print('Computing z-score normalization statistics...')

        # Sample a subset for efficiency
        num_samples = len(self.sample_ids) #min(50, len(self.sample_ids))

        all_node_features = []
        all_edge_features = []
        all_delta_features = [[] for _ in range(self.output_dim)]  # For delta normalization

        with h5py.File(self.h5_file, 'r') as f:
            for i in range(num_samples):
                sid = self.sample_ids[i]
                data = f[f'data/{sid}/nodal_data'][:]  # [features, time, nodes]
                mesh_edge = f[f'data/{sid}/mesh_edge'][:]  # [2, edges]

                # Sample timesteps for multi-timestep data
                if self.num_timesteps > 1:
                    timesteps = np.linspace(0, self.num_timesteps - 1, self.num_timesteps, dtype=int)
                else:
                    timesteps = [0]

                for t in timesteps:
                    # Node features: [disp_x, disp_y, disp_z, stress]
                    node_feat = data[3:3+self.input_dim, t, :].T  # [N, 4]
                    all_node_features.append(node_feat)

                    # Edge features: [dx, dy, dz, distance]
                    pos = (data[:3, t, :]+data[3:6, t, :]).T  # [N, 3] - Deformed position (reference + displacement)
                    # Bidirectional edges
                    edge_idx = np.concatenate([mesh_edge, mesh_edge[[1, 0], :]], axis=1)
                    rel_pos = pos[edge_idx[1]] - pos[edge_idx[0]] # Deformed relative position
                    dist = np.linalg.norm(rel_pos, axis=1, keepdims=True)
                    edge_feat = np.concatenate([rel_pos, dist], axis=1)  # [2E, 4] - Deformed edge features (relative position + distance)
                    all_edge_features.append(edge_feat)

                # Compute delta features (differences between consecutive timesteps)
                if self.num_timesteps > 1:
                    # Sample consecutive timestep pairs
                    delta_timesteps = np.linspace(0, self.num_timesteps - 2, self.num_timesteps - 1, dtype=int)
                    for t in delta_timesteps:
                        for feat_idx in range(self.output_dim):
                            feat_t = data[3 + feat_idx, t, :]      # [N]
                            feat_t1 = data[3 + feat_idx, t + 1, :]  # [N] - Next timestep feature
                            delta = feat_t1 - feat_t # Delta nodal feature (including node type)
                            all_delta_features[feat_idx].append(delta)
                else:
                    # Single timestep: delta is the final value itself (from zero initial state)
                    for feat_idx in range(self.output_dim):
                        feat = data[3 + feat_idx, 0, :]  # [N]
                        all_delta_features[feat_idx].append(feat)

        # Compute node and edge statistics
        all_node_features = np.vstack(all_node_features)
        all_edge_features = np.vstack(all_edge_features)

        self.node_mean = np.mean(all_node_features, axis=0).astype(np.float32)
        self.node_std = np.std(all_node_features, axis=0).astype(np.float32)
        self.node_std = np.maximum(self.node_std, 1e-8)  # Prevent division by zero

        self.edge_mean = np.mean(all_edge_features, axis=0).astype(np.float32)
        self.edge_std = np.std(all_edge_features, axis=0).astype(np.float32)
        self.edge_std = np.maximum(self.edge_std, 1e-8)  # Prevent division by zero

        print(f'  Node features - mean: {self.node_mean}, std: {self.node_std}')
        print(f'  Edge features - mean: {self.edge_mean}, std: {self.edge_std}')

        # Compute delta statistics from actual data
        self.delta_mean = np.zeros(self.output_dim, dtype=np.float32)
        self.delta_std = np.zeros(self.output_dim, dtype=np.float32)

        for feat_idx in range(self.output_dim):
            deltas = np.concatenate(all_delta_features[feat_idx])
            self.delta_mean[feat_idx] = np.mean(deltas)
            self.delta_std[feat_idx] = np.std(deltas)
            self.delta_std[feat_idx] = max(self.delta_std[feat_idx], 1e-8)  # Prevent division by zero

        print(f'  Delta features - mean: {self.delta_mean}, std: {self.delta_std}')

        with h5py.File(self.h5_file, 'r+') as f:
            if 'metadata/normalization_params' not in f:
                norm_params = f.create_group('metadata/normalization_params')

            norm_params = f['metadata/normalization_params']

            # Check if stored params differ from computed ones
            if 'delta_mean' in norm_params and 'delta_std' in norm_params:
                
                norm_params['delta_mean'][...] = self.delta_mean
                norm_params['delta_std'][...] = self.delta_std

                # Also update delta_max and delta_min if they exist
                if 'delta_max' in norm_params and 'delta_min' in norm_params:
                    # Compute correct min/max from the same data
                    delta_max = np.zeros(self.output_dim, dtype=np.float32)
                    delta_min = np.zeros(self.output_dim, dtype=np.float32)

                    # Use sampled data from _compute_zscore_stats
                    num_samples = min(50, len(self.sample_ids))
                    with h5py.File(self.h5_file, 'r') as f_read:
                        all_deltas = [[] for _ in range(self.output_dim)]
                        for i in range(num_samples):
                            sid = self.sample_ids[i]
                            data = f_read[f'data/{sid}/nodal_data'][:]
                            if self.num_timesteps > 1:
                                delta_timesteps = np.linspace(0, self.num_timesteps - 2,
                                                                min(10, self.num_timesteps - 1), dtype=int)
                                for t in delta_timesteps:
                                    for feat_idx in range(self.output_dim):
                                        delta = data[3 + feat_idx, t + 1, :] - data[3 + feat_idx, t, :]
                                        all_deltas[feat_idx].append(delta)
                            else:
                                for feat_idx in range(self.output_dim):
                                    all_deltas[feat_idx].append(data[3 + feat_idx, 0, :])

                    for feat_idx in range(self.output_dim):
                        deltas = np.concatenate(all_deltas[feat_idx])
                        delta_max[feat_idx] = np.max(deltas)
                        delta_min[feat_idx] = np.min(deltas)

                    norm_params['delta_max'][...] = delta_max
                    norm_params['delta_min'][...] = delta_min

                print(f'  [OK] HDF5 delta normalization parameters updated successfully')
                    
    def _compute_node_type_info(self) -> None:
        """Compute the number of unique node types from the dataset."""
        print('Computing node type information...')
        with h5py.File(self.h5_file, 'r') as f:
            # Collect unique node types from first few samples
            # Node types are always the last feature
            unique_types = set()
            num_samples = min(10, len(self.sample_ids))
            for i in range(num_samples):
                sid = self.sample_ids[i]
                nodal_data = f[f'data/{sid}/nodal_data'][:]
                node_types = nodal_data[-1, 0, :].astype(np.int32)  # Last feature, first timestep
                unique_types.update(node_types)

            sorted_types = sorted(unique_types)
            self.node_type_to_idx = {t: i for i, t in enumerate(sorted_types)}
            self.num_node_types = len(unique_types)
            print(f'  Found {self.num_node_types} unique node types: {sorted_types}')
            print(f'  Node type mapping: {self.node_type_to_idx}')

    def _compute_world_edge_radius(self) -> None:
        print('Computing world edge radius...')
        num_samples = min(10, len(self.sample_ids))
        min_lengths = []
        with h5py.File(self.h5_file, 'r') as f:
            for i in range(num_samples):
                sid = self.sample_ids[i]
                nd = f[f'data/{sid}/nodal_data'][:]
                me = f[f'data/{sid}/mesh_edge'][:]
                pos = nd[:3, 0, :].T
                lens = np.linalg.norm(pos[me[1]] - pos[me[0]], axis=1)
                min_lengths.append(np.min(lens))
        self.min_edge_length = np.min(min_lengths)
        self.world_edge_radius = self.world_radius_multiplier * self.min_edge_length
        print(f'  min_edge_length: {self.min_edge_length:.6f}')
        print(f'  world_edge_radius: {self.world_edge_radius:.6f}')

    def _compute_world_edges(self, pos, mesh_edges):
        """
        Compute world edges using either torch_cluster (GPU) or scipy.KDTree (CPU).

        Supports two backends:
        - 'torch_cluster': GPU-accelerated (5-10x faster for 68k nodes)
        - 'scipy_kdtree': CPU-based fallback (original implementation)

        Args:
            pos: (N, 3) array of node positions
            mesh_edges: (2, E_mesh) array of existing mesh edge indices

        Returns:
            world_edge_index: (2, E_world) array of world edge indices
            world_edge_attr: (E_world, 4) array with [dx, dy, dz, distance]
        """
        if not self.world_edge_radius:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

        if self.world_edge_backend == 'torch_cluster':
            return self._compute_world_edges_torch_cluster(pos, mesh_edges)
        else:
            return self._compute_world_edges_scipy_kdtree(pos, mesh_edges)

    def _compute_world_edges_torch_cluster(self, pos, mesh_edges):
        """
        Compute world edges using GPU-accelerated torch_cluster.radius_graph().
        Expected 5-10x speedup for 68k-node meshes compared to scipy.KDTree.
        """
        # Convert positions to GPU tensor
        pos_tensor = torch.from_numpy(pos).float().cuda()

        # GPU-accelerated radius query (torch_cluster)
        world_edges = radius_graph(
            x=pos_tensor,
            r=self.world_edge_radius,
            batch=None,                           # Single sample (no batch tensor)
            loop=False,                           # No self-loops
            max_num_neighbors=self.world_max_num_neighbors
        )

        # Convert back to numpy for edge filtering
        world_edges_np = world_edges.cpu().numpy()

        if world_edges_np.shape[1] == 0:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

        # Efficient filtering: remove edges that already exist in mesh topology
        mesh_set = {(int(mesh_edges[0, i]), int(mesh_edges[1, i])) for i in range(mesh_edges.shape[1])}

        valid_mask = np.array([
            (world_edges_np[0, i], world_edges_np[1, i]) not in mesh_set
            for i in range(world_edges_np.shape[1])
        ])

        we = world_edges_np[:, valid_mask]

        if we.shape[1] == 0:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

        # Compute edge features: [relative_position, distance]
        rel = pos[we[1]] - pos[we[0]]
        dist = np.linalg.norm(rel, axis=1, keepdims=True)

        return we, np.concatenate([rel, dist], axis=1).astype(np.float32)

    def _compute_world_edges_scipy_kdtree(self, pos, mesh_edges):
        """
        Compute world edges using scipy.spatial.KDTree (CPU fallback).
        Original implementation, slower but always available.
        """
        tree = KDTree(pos)
        pairs = tree.query_pairs(r=self.world_edge_radius, output_type='ndarray')

        if len(pairs) == 0:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

        # Filter out existing mesh edges
        mesh_set = {(int(mesh_edges[0, i]), int(mesh_edges[1, i])) for i in range(mesh_edges.shape[1])}
        we = []
        for s, r in pairs:
            if (s, r) not in mesh_set:
                we.append([s, r])
            if (r, s) not in mesh_set:
                we.append([r, s])

        if not we:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

        wei = np.array(we, dtype=np.int64).T
        rel = pos[wei[1]] - pos[wei[0]]
        dist = np.linalg.norm(rel, axis=1, keepdims=True)

        return wei, np.concatenate([rel, dist], axis=1).astype(np.float32)

    def __len__(self) -> int:
        """
        Calculate total number of samples.

        For multi-timestep data, each sample can produce (num_timesteps - 1)
        training pairs: (t_0→t_1), (t_1→t_2), ..., (t_n-1→t_n)

        For single timestep data, returns the number of samples.
        """
        if self.num_timesteps > 1:
            # Multiple timesteps: each sample generates (T-1) training pairs
            return len(self.sample_ids) * (self.num_timesteps - 1)
        else:
            # Single timestep: static data
            return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Data:
        """
        Load a single graph sample with optional temporal prediction.

        Dataset structure:
            data/{sample_id}/nodal_data: [7 or 8, time, N]
                Features: [x, y, z, x_disp, y_disp, z_disp, stress, (part_number)]
                Part number (index 7) is optional and used for visualization
            data/{sample_id}/mesh_edge: [2, M]

        Single timestep (T=1):
            x: [N, 4] normalized physical features (all zeros -> zeros)
            pos: [N, 3] positions (unnormalized)
            y: [N, 4] normalized target delta (target - input)

        Multi-timestep (T>1):
            x: [N, 4] normalized physical features at time t
            pos: [N, 3] positions at time t
            y: [N, 4] normalized target delta (state_t+1 - state_t)

        All features are normalized using precomputed global statistics.

        Args:
            idx: Sample index

        Returns:
            Data object with normalized x, y, edge_attr, plus pos, edge_index,
            sample_id, time_idx, and optionally part_ids
        """
        # Calculate sample and timestep indices
        if self.num_timesteps > 1:
            sample_idx = idx // (self.num_timesteps - 1)
            time_idx = idx % (self.num_timesteps - 1)
        else:
            sample_idx = idx
            time_idx = 0

        sample_id = self.sample_ids[sample_idx]

        # Load data from HDF5
        with h5py.File(self.h5_file, 'r') as f:
            data = f[f'data/{sample_id}/nodal_data'][:]  # [7 or 8, time, nodes]
            edge_index = f[f'data/{sample_id}/mesh_edge'][:]  # [2, M]

        if self.config['use_node_types']:
            has_part_info = True
        else:
            has_part_info = False

        if has_part_info:
            part_ids = data[-1, 0, :].astype(np.int32)  # [nodes]
        else:
            part_ids = None

        if self.use_node_types:
            node_types = data[-1, 0, :].astype(np.int32)  # [nodes]
        else:
            node_types = None

            # Essentially, node_types are part_ids

        # Make edges bidirectional (like DeepMind's MeshGraphNets implementation)
        edge_index = np.concatenate([edge_index, edge_index[[1, 0], :]], axis=1)  # [2, 2M]

        # Transpose to [nodes, time, 7]
        data = np.transpose(data, (2, 1, 0))
        # Data shape: [nodes, time, features]

        # Extract data based on timesteps
        if self.num_timesteps == 1: # Static case
            # Single timestep: geometry → physics
            data_t = data[:, 0, :]  # [N, 7]
            pos = data_t[:, :3]  # [N, 3]
            x_raw = np.zeros((data_t.shape[0], self.input_dim), dtype=np.float32)  # [N, 4] zeros
            y_raw = data_t[:, 3:3+self.output_dim]  # [N, 4]
            # Target delta: y - x (for single timestep, x is zeros so delta = y)
            target_delta = y_raw - x_raw  # [N, 4], not including node types
        else:
            # Multi-timestep: state t → state t+1
            data_t = data[:, time_idx, :]  # [N, 7]
            data_t1 = data[:, time_idx + 1, :]  # [N, 7]
            pos = data_t[:, :3]  # [N, 3]
            x_raw = data_t[:, 3:3+self.input_dim]  # [N, 4]
            y_raw = data_t1[:, 3:3+self.output_dim]  # [N, 4]
            # Target delta: difference between next and current state
            target_delta = y_raw - x_raw  # [N, 4]

        displacement = x_raw[:, :3]  # [N, 3] - extract displacement components (x_disp, y_disp, z_disp)
        deformed_pos = pos + displacement  # [N, 3] - actual mesh position at time t

        # Compute edge features (before normalization)
        # Edge features are computed for all edges (including reverse edges)
        # Reverse edges naturally get negated relative_pos since src/dst are swapped
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        relative_pos = deformed_pos[dst_idx] - deformed_pos[src_idx]  # [2M, 3]
        distance = np.linalg.norm(relative_pos, axis=1, keepdims=True)  # [2M, 1]
        edge_attr_raw = np.concatenate([relative_pos, distance], axis=1)  # [2M, 4]

        # Apply z-score normalization to all features
        # Node features: z-score normalization

        x_norm = (x_raw - self.node_mean) / self.node_std

        # Add node types if enabled
        if self.use_node_types and node_types is not None:
            # Map node types to contiguous indices using the precomputed mapping
            # e.g., node type 3 -> index 2 if mapping is {0:0, 1:1, 3:2}
            node_type_indices = np.array([self.node_type_to_idx[t] for t in node_types], dtype=np.int32)
            # One-hot encode node types: [N] -> [N, num_node_types]
            node_type_onehot = np.zeros((len(node_types), self.num_node_types), dtype=np.float32)
            node_type_onehot[np.arange(len(node_types)), node_type_indices] = 1.0
            # Concatenate with physical features: [N, 4] + [N, num_node_types] = [N, 4+num_node_types]
            x_norm = np.concatenate([x_norm, node_type_onehot], axis=1)

        # Target features: z-score normalization using delta-specific parameters
        if self.delta_mean is not None and self.delta_std is not None:
            target_norm = (target_delta - self.delta_mean) / self.delta_std
        else:
            # Fallback: use node stats
            target_norm = (target_delta - self.node_mean) / self.node_std

        # Edge features: z-score normalization
        edge_attr_norm = (edge_attr_raw - self.edge_mean) / self.edge_std

        # Convert to tensors
        pos = torch.from_numpy(pos.astype(np.float32))
        x = torch.from_numpy(x_norm.astype(np.float32)) # nodal state at time t, dx, dy, dz, stress, nodal type, ...
        y = torch.from_numpy(target_norm.astype(np.float32)) # nodal state at time t+1, dx, dy, dz, stress, ...
        edge_index = torch.from_numpy(edge_index).long()
        edge_attr = torch.from_numpy(edge_attr_norm.astype(np.float32))

        # Convert part_ids to tensor if available
        if part_ids is not None:
            part_ids_tensor = torch.from_numpy(part_ids).long()
        else:
            part_ids_tensor = None

        # Create base Data object
        graph_data = Data(
            x=x,
            y=y,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            sample_id=sample_id,
            time_idx=time_idx if self.num_timesteps > 1 else None,
            part_ids=part_ids_tensor
        )

        # Compute world edges if enabled
        # IMPORTANT: Use deformed_pos (reference + displacement) for collision detection
        if self.use_world_edges:
            world_edge_index, world_edge_attr = self._compute_world_edges(
                deformed_pos,  # Use DEFORMED position (pos + displacement), not reference
                edge_index.numpy() if isinstance(edge_index, torch.Tensor) else edge_index
            )
            graph_data.world_edge_index = torch.from_numpy(world_edge_index).long()
            graph_data.world_edge_attr = torch.from_numpy(world_edge_attr)
        else:
            graph_data.world_edge_index = torch.zeros((2, 0), dtype=torch.long)
            graph_data.world_edge_attr = torch.zeros((0, 4), dtype=torch.float32)

        return graph_data

    def split(self, train_ratio: float, val_ratio: float, test_ratio: float, seed: int = 42):
        """
        Split dataset into train, validation, and test sets.

        Args:
            train_ratio: Fraction of data for training (e.g., 0.8)
            val_ratio: Fraction of data for validation (e.g., 0.1)
            test_ratio: Fraction of data for testing (e.g., 0.1)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Validate ratios
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

        # Set random seed for reproducibility
        np.random.seed(seed)

        # Shuffle sample IDs
        shuffled_ids = self.sample_ids.copy()
        np.random.shuffle(shuffled_ids)

        # Calculate split sizes
        n_samples = len(shuffled_ids)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        # Split IDs
        train_ids = shuffled_ids[:n_train]
        val_ids = shuffled_ids[n_train:n_train + n_val]
        test_ids = shuffled_ids[n_train + n_val:]

        # Create subset datasets (copy all attributes including normalization params)
        train_dataset = MeshGraphDataset.__new__(MeshGraphDataset)
        train_dataset.h5_file = self.h5_file
        train_dataset.config = self.config
        train_dataset.input_dim = self.input_dim
        train_dataset.output_dim = self.output_dim
        train_dataset.sample_ids = train_ids
        train_dataset.num_timesteps = self.num_timesteps
        train_dataset.use_node_types = self.use_node_types
        train_dataset.num_node_types = self.num_node_types
        train_dataset.node_type_to_idx = self.node_type_to_idx if self.use_node_types else None
        train_dataset.use_world_edges = self.use_world_edges
        train_dataset.world_radius_multiplier = self.world_radius_multiplier
        train_dataset.world_max_num_neighbors = self.world_max_num_neighbors
        train_dataset.world_edge_backend = self.world_edge_backend
        train_dataset.world_edge_radius = self.world_edge_radius
        train_dataset.min_edge_length = self.min_edge_length
        train_dataset.delta_mean = self.delta_mean
        train_dataset.delta_std = self.delta_std
        train_dataset.node_mean = self.node_mean
        train_dataset.node_std = self.node_std
        train_dataset.edge_mean = self.edge_mean
        train_dataset.edge_std = self.edge_std

        val_dataset = MeshGraphDataset.__new__(MeshGraphDataset)
        val_dataset.h5_file = self.h5_file
        val_dataset.config = self.config
        val_dataset.input_dim = self.input_dim
        val_dataset.output_dim = self.output_dim
        val_dataset.sample_ids = val_ids
        val_dataset.num_timesteps = self.num_timesteps
        val_dataset.use_node_types = self.use_node_types
        val_dataset.num_node_types = self.num_node_types
        val_dataset.node_type_to_idx = self.node_type_to_idx if self.use_node_types else None
        val_dataset.use_world_edges = self.use_world_edges
        val_dataset.world_radius_multiplier = self.world_radius_multiplier
        val_dataset.world_max_num_neighbors = self.world_max_num_neighbors
        val_dataset.world_edge_backend = self.world_edge_backend
        val_dataset.world_edge_radius = self.world_edge_radius
        val_dataset.min_edge_length = self.min_edge_length
        val_dataset.delta_mean = self.delta_mean
        val_dataset.delta_std = self.delta_std
        val_dataset.node_mean = self.node_mean
        val_dataset.node_std = self.node_std
        val_dataset.edge_mean = self.edge_mean
        val_dataset.edge_std = self.edge_std

        test_dataset = MeshGraphDataset.__new__(MeshGraphDataset)
        test_dataset.h5_file = self.h5_file
        test_dataset.config = self.config
        test_dataset.input_dim = self.input_dim
        test_dataset.output_dim = self.output_dim
        test_dataset.sample_ids = test_ids
        test_dataset.num_timesteps = self.num_timesteps
        test_dataset.use_node_types = self.use_node_types
        test_dataset.num_node_types = self.num_node_types
        test_dataset.node_type_to_idx = self.node_type_to_idx if self.use_node_types else None
        test_dataset.use_world_edges = self.use_world_edges
        test_dataset.world_radius_multiplier = self.world_radius_multiplier
        test_dataset.world_max_num_neighbors = self.world_max_num_neighbors
        test_dataset.world_edge_backend = self.world_edge_backend
        test_dataset.world_edge_radius = self.world_edge_radius
        test_dataset.min_edge_length = self.min_edge_length
        test_dataset.delta_mean = self.delta_mean
        test_dataset.delta_std = self.delta_std
        test_dataset.node_mean = self.node_mean
        test_dataset.node_std = self.node_std
        test_dataset.edge_mean = self.edge_mean
        test_dataset.edge_std = self.edge_std

        print(f"Dataset split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")

        return train_dataset, val_dataset, test_dataset
