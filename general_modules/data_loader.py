#!/usr/bin/env python3

import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from general_modules.mesh_dataset import MeshGraphDataset

class HDF5Dataset(Dataset):
    """PyTorch Dataset for HDF5 files with structure: data/sample_id -> (features, time, nodes)"""

    def __init__(self, h5_file, sample_ids=None, cache_size=50):
        self.h5_file = h5_file
        self.cache_size = cache_size
        self.cache = {}
        self.access_count = {}

        with h5py.File(h5_file, 'r') as f:
            # Get all sample IDs if not provided
            if 'data' in f:
                if sample_ids is None:
                    self.sample_ids = sorted([int(k) for k in f['data'].keys()])
                else:
                    self.sample_ids = sample_ids
            else:
                raise ValueError(f"HDF5 file {h5_file} missing 'data' group")

    def __len__(self):
        return len(self.sample_ids)

    def _load_sample(self, sample_id):
        """Load sample from HDF5 file"""
        with h5py.File(self.h5_file, 'r') as f:
            data = f[f'data/{sample_id}/data'][:]  # Shape: (features, time, nodes)
            return data.astype(np.float32)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        # Check cache first
        if sample_id in self.cache:
            self.access_count[sample_id] += 1
            return torch.from_numpy(self.cache[sample_id]).float(), sample_id

        # Load from file
        data = self._load_sample(sample_id)

        # Add to cache with LRU replacement
        if len(self.cache) < self.cache_size:
            self.cache[sample_id] = data
            self.access_count[sample_id] = 1
        else:
            # Replace least accessed item
            least_accessed = min(self.access_count, key=self.access_count.get)
            del self.cache[least_accessed]
            del self.access_count[least_accessed]
            self.cache[sample_id] = data
            self.access_count[sample_id] = 1

        return torch.from_numpy(data).float(), sample_id

    def get_statistics(self):
        """Get dataset statistics compatible with existing analyze_data_statistics"""
        with h5py.File(self.h5_file, 'r') as f:
            # Get sample shape info from first sample
            first_sample = f[f'data/{self.sample_ids[0]}/data']
            num_features, num_time, nodes_first = first_sample.shape

            # Calculate total nodes across all samples and get min/max nodes
            total_nodes = 0
            min_nodes = float('inf')
            max_nodes = 0

            for sample_id in self.sample_ids:
                sample_shape = f[f'data/{sample_id}/data'].shape
                nodes_count = sample_shape[2]
                total_nodes += nodes_count
                min_nodes = min(min_nodes, nodes_count)
                max_nodes = max(max_nodes, nodes_count)

            # Compute basic statistics from first sample for data range info
            sample_data = first_sample[:]
            data_min = float(np.min(sample_data))
            data_max = float(np.max(sample_data))

            stats = {
                'num_samples': len(self.sample_ids),
                'nodes_per_sample': nodes_first,  # From first sample
                'min_nodes': min_nodes,
                'max_nodes': max_nodes,
                'avg_nodes': total_nodes // len(self.sample_ids),
                'features_per_node': num_features,
                'num_time_steps': num_time,
                'total_nodes': total_nodes,
                'data_dtype': str(first_sample.dtype),
                'data_min': data_min,
                'data_max': data_max,
                'data_range': data_max - data_min
            }

            return stats

    def data_file(self):
        """Property for compatibility with existing code"""
        return self.h5_file

def collate_variable_length(batch):
    """Custom collate function for variable-length mesh sequences"""
    batch_data, batch_ids = zip(*batch)

    # Find max number of nodes in batch
    max_nodes = max(data.shape[2] for data in batch_data)
    batch_size = len(batch_data)
    num_features = batch_data[0].shape[0]
    num_time = batch_data[0].shape[1]

    # Create padded tensor
    padded_data = torch.zeros(batch_size, num_features, num_time, max_nodes)
    lengths = []

    for i, data in enumerate(batch_data):
        num_nodes = data.shape[2]
        padded_data[i, :, :, :num_nodes] = data
        lengths.append(num_nodes)

    return padded_data, torch.tensor(lengths), torch.tensor(batch_ids)

def create_dataloader(dataset, config, is_training=True):
    """Create dataloader for dataset"""

    # Check dataset type
    if hasattr(dataset, 'mesh_template'):
        print(f"Using MeshGraphDataset dataloader")
        return create_mesh_dataloader(dataset, config, is_training)

    # Fallback for other dataset types
    print(f"Using standard dataloader")
    batch_size = config.get('Batch_size', 1)
    num_workers = config.get('num_workers', 0)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


def load_data(config):
    """Create mesh graph dataset"""
    print("Loading mesh graph dataset...")

    data_file = config.get('dataset_dir')

    print(f"Creating MeshGraphDataset from: {data_file}")
    dataset = MeshGraphDataset(data_file, config=config)

    print(f"Dataset loaded: {len(dataset)} samples")
    return dataset


# Develop dataset.split function
