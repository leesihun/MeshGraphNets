import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict
from torch_geometric.data import Data

class MeshGraphDataset(Dataset):

    def __init__(self, h5_file: str, config: Dict):
        self.h5_file = h5_file
        self.config = config
        # Graph and feature parameters
        self.input_dim = config.get('input_var')  # Physical features only (4)
        self.output_dim = config.get('output_var')  # Physical features only (4)

        print(f"Loading MeshGraphDataset: {h5_file}")
        print(f"  input_dim: {self.input_dim}, output_dim: {self.output_dim}")

        # Load sample IDs, timesteps, and normalization params
        with h5py.File(h5_file, 'r') as f:
            if 'data' not in f:
                raise ValueError(f"HDF5 file missing 'data' group")

            self.sample_ids = sorted([int(k) for k in f['data'].keys()])

            # Check number of timesteps from first sample
            sample_id = self.sample_ids[0]
            data_shape = f[f'data/{sample_id}/nodal_data'].shape
            self.num_timesteps = data_shape[1]  # Shape: (features, time, nodes)

            # Load precomputed normalization parameters
            # Shape: (7,) for [x, y, z, disp_x, disp_y, disp_z, stress]
            # self.norm_mean = f['metadata/normalization_params/mean'][:]
            # self.norm_std = f['metadata/normalization_params/std'][:]
            self.norm_max = f['metadata/normalization_params/max'][:]
            self.norm_min = f['metadata/normalization_params/min'][:]
            # Add small epsilon to avoid division by zero
            # self.norm_std = np.maximum(self.norm_std, 1e-8)

        # Extract stats for node features (indices 3:7 = physical fields)
        # self.node_mean = self.norm_mean[3:3+self.input_dim]
        # self.node_std = self.norm_std[3:3+self.input_dim]
        self.node_max = self.norm_max[3:3+self.input_dim]
        self.node_min = self.norm_min[3:3+self.input_dim]
        
        # Extract stats for edge features (relative positions use coord std)
        # Edge features: [dx, dy, dz, distance]
        # For relative positions, mean is ~0, std is similar to coord std
        # self.edge_mean = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # Use coordinate std for dx, dy, dz; use mean distance for distance normalization
        # coord_std_mean = np.mean(self.norm_std[:3])  # Average std of x, y, z
        # self.edge_std = np.array([
        #     self.norm_std[0],  # std for dx
        #     self.norm_std[1],  # std for dy
        #     self.norm_std[2],  # std for dz
        #     coord_std_mean     # std for distance (approximate)
        # ], dtype=np.float32)

        print(f"Found {len(self.sample_ids)} samples")
        print(f"  num_timesteps: {self.num_timesteps}")
        
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
            data/{sample_id}/nodal_data: [7, time, N]
                Features: [x, y, z, x_disp, y_disp, z_disp, stress]
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
            Data object with normalized x, y, edge_attr, plus pos and edge_index
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
            data = f[f'data/{sample_id}/nodal_data'][:]  # [7, time, nodes]
            edge_index = f[f'data/{sample_id}/mesh_edge'][:]  # [2, M]

        # Make edges bidirectional (like DeepMind's MeshGraphNets implementation)
        # Original: only (src, dst) where src < dst
        # Bidirectional: both (src, dst) and (dst, src)
        edge_index = np.concatenate([edge_index, edge_index[[1, 0], :]], axis=1)  # [2, 2M]

        # Transpose to [nodes, time, 7]
        data = np.transpose(data, (2, 1, 0))

        # Extract data based on timesteps
        if self.num_timesteps == 1: # Static case
            # Single timestep: geometry → physics
            data_t = data[:, 0, :]  # [N, 7]
            pos = data_t[:, :3]  # [N, 3]
            x_raw = np.zeros((data_t.shape[0], self.input_dim), dtype=np.float32)  # [N, 4] zeros
            y_raw = data_t[:, 3:3+self.output_dim]  # [N, 4]
            # Target delta: y - x (for single timestep, x is zeros so delta = y)
            target_delta = y_raw - x_raw  # [N, 4]
        else:
            # Multi-timestep: state t → state t+1
            data_t = data[:, time_idx, :]  # [N, 7]
            data_t1 = data[:, time_idx + 1, :]  # [N, 7]
            pos = data_t[:, :3]  # [N, 3]
            x_raw = data_t[:, 3:3+self.input_dim]  # [N, 4]
            y_raw = data_t1[:, 3:3+self.output_dim]  # [N, 4]
            # Target delta: difference between next and current state
            target_delta = y_raw - x_raw  # [N, 4]

        # Compute edge features (before normalization)
        # Edge features are computed for all edges (including reverse edges)
        # Reverse edges naturally get negated relative_pos since src/dst are swapped
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        relative_pos = pos[dst_idx] - pos[src_idx]  # [2M, 3]
        distance = np.linalg.norm(relative_pos, axis=1, keepdims=True)  # [2M, 1]
        edge_attr_raw = np.concatenate([relative_pos, distance], axis=1)  # [2M, 4]

        # Apply normalization
        # Node features: (x - mean) / std
        # x_norm = (x_raw - self.node_mean) / self.node_std
        x_norm = (((x_raw - self.node_min) / (self.node_max - self.node_min)) * 2 - 1)*self.norm_max
        # Normalized to [norm_min, norm_max]

        # target_norm Need to be normalized to [norm_min, norm_max]
        feature_range = self.node_max - self.node_min
        target_norm = target_delta / feature_range

        # Edge features: no normalization (edge_mean/edge_std not implemented)
        edge_attr_norm = edge_attr_raw

        # Convert to tensors
        pos = torch.from_numpy(pos.astype(np.float32))
        x = torch.from_numpy(x_norm.astype(np.float32))
        y = torch.from_numpy(target_norm.astype(np.float32))
        edge_index = torch.from_numpy(edge_index).long()
        edge_attr = torch.from_numpy(edge_attr_norm.astype(np.float32))

        # Create PyG Data object
        return Data(
            x=x,
            y=y,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            sample_id=sample_id,
            time_idx=time_idx if self.num_timesteps > 1 else None
        )

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
        train_dataset.norm_max = self.norm_max
        train_dataset.norm_min = self.norm_min
        train_dataset.node_max = self.node_max
        train_dataset.node_min = self.node_min

        val_dataset = MeshGraphDataset.__new__(MeshGraphDataset)
        val_dataset.h5_file = self.h5_file
        val_dataset.config = self.config
        val_dataset.input_dim = self.input_dim
        val_dataset.output_dim = self.output_dim
        val_dataset.sample_ids = val_ids
        val_dataset.num_timesteps = self.num_timesteps
        val_dataset.norm_max = self.norm_max
        val_dataset.norm_min = self.norm_min
        val_dataset.node_max = self.node_max
        val_dataset.node_min = self.node_min

        test_dataset = MeshGraphDataset.__new__(MeshGraphDataset)
        test_dataset.h5_file = self.h5_file
        test_dataset.config = self.config
        test_dataset.input_dim = self.input_dim
        test_dataset.output_dim = self.output_dim
        test_dataset.sample_ids = test_ids
        test_dataset.num_timesteps = self.num_timesteps
        test_dataset.norm_max = self.norm_max
        test_dataset.norm_min = self.norm_min
        test_dataset.node_max = self.node_max
        test_dataset.node_min = self.node_min

        print(f"Dataset split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")

        return train_dataset, val_dataset, test_dataset
