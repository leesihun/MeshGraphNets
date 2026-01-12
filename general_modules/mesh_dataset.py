import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict
from torch_geometric.data import Data

class MeshGraphDataset(Dataset):
    """
    Graph-based dataset for MeshGraphNet training

    This dataset produces graph-structured data for GNN processing:
    - Node features (physical variables, etc.)
    - Edge features (dx, dy, dz, distance)

    Data format:
        Input: HDF5 file with structure data/sample_id/data [features, time, nodes], mesh_edge
        Output: torch_geometric.data.Data with node features, edge features, and targets
    """

    def __init__(self, h5_file: str, config: Dict):
        """
        Initialize MeshGraphDataset

        Args:
            h5_file: Path to HDF5 dataset file
            config: Configuration dictionary with:
                - input_var: Number of input features (physical fields only)
                - output_var: Number of output features (physical fields only)
        """
        self.h5_file = h5_file
        self.config = config
        # Graph and feature parameters
        self.input_dim = config.get('input_var')  # Physical features only (4)
        self.output_dim = config.get('output_var')  # Physical features only (4)

        print(f"Loading MeshGraphDataset: {h5_file}")
        print(f"  input_dim: {self.input_dim}, output_dim: {self.output_dim}")

        # Load sample IDs and determine number of timesteps
        with h5py.File(h5_file, 'r') as f:
            if 'data' not in f:
                raise ValueError(f"HDF5 file missing 'data' group")

            self.sample_ids = sorted([int(k) for k in f['data'].keys()])

            # Check number of timesteps from first sample
            sample_id = self.sample_ids[0]
            data_shape = f[f'data/{sample_id}/nodal_data'].shape
            self.num_timesteps = data_shape[1]  # Shape: (features, time, nodes)

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
            x: [N, 4] physical features (all zeros)
            pos: [N, 3] positions
            y: [N, 4] physical features (targets)

        Multi-timestep (T>1):
            x: [N, 4] physical features at time t
            pos: [N, 3] positions at time t
            y: [N, 4] physical features at time t+1

        Args:
            idx: Sample index

        Returns:
            Data object with x, y, pos, edge_index, edge_attr
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

        # Transpose to [nodes, time, 7]
        data = np.transpose(data, (2, 1, 0))

        # Extract data based on timesteps
        if self.num_timesteps == 1:
            # Single timestep: geometry → physics
            data_t = data[:, 0, :]  # [N, 7]
            pos = data_t[:, :3]  # [N, 3]
            x = np.zeros((data_t.shape[0], self.input_dim), dtype=np.float32)  # [N, 4] zeros
            y = data_t[:, 3:3+self.output_dim]  # [N, 4]
        else:
            # Multi-timestep: state t → state t+1
            data_t = data[:, time_idx, :]  # [N, 7]
            data_t1 = data[:, time_idx + 1, :]  # [N, 7]
            pos = data_t[:, :3]  # [N, 3]
            x = data_t[:, 3:3+self.input_dim]  # [N, 4]
            y = data_t1[:, 3:3+self.output_dim]  # [N, 4]

        # Compute edge features
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        relative_pos = pos[dst_idx] - pos[src_idx]  # [M, 3]
        distance = np.linalg.norm(relative_pos, axis=1, keepdims=True)  # [M, 1]
        edge_attr = np.concatenate([relative_pos, distance], axis=1)  # [M, 4]

        # Convert to tensors
        pos = torch.from_numpy(pos).float()
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        edge_index = torch.from_numpy(edge_index).long()
        edge_attr = torch.from_numpy(edge_attr).float()

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

        # Create subset datasets
        train_dataset = MeshGraphDataset.__new__(MeshGraphDataset)
        train_dataset.h5_file = self.h5_file
        train_dataset.config = self.config
        train_dataset.input_dim = self.input_dim
        train_dataset.output_dim = self.output_dim
        train_dataset.sample_ids = train_ids
        train_dataset.num_timesteps = self.num_timesteps

        val_dataset = MeshGraphDataset.__new__(MeshGraphDataset)
        val_dataset.h5_file = self.h5_file
        val_dataset.config = self.config
        val_dataset.input_dim = self.input_dim
        val_dataset.output_dim = self.output_dim
        val_dataset.sample_ids = val_ids
        val_dataset.num_timesteps = self.num_timesteps

        test_dataset = MeshGraphDataset.__new__(MeshGraphDataset)
        test_dataset.h5_file = self.h5_file
        test_dataset.config = self.config
        test_dataset.input_dim = self.input_dim
        test_dataset.output_dim = self.output_dim
        test_dataset.sample_ids = test_ids
        test_dataset.num_timesteps = self.num_timesteps

        print(f"Dataset split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")

        return train_dataset, val_dataset, test_dataset
