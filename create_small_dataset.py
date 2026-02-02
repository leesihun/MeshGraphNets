"""
Create a smaller dataset by extracting a subset of samples from the full dataset.

Usage:
    python create_small_dataset.py

This script creates a new HDF5 file with only the first 100 samples from the
original deforming_plate.h5 dataset (which has ~1400 samples).
"""

import h5py
import numpy as np
from tqdm import tqdm

# Configuration
INPUT_FILE = './dataset/deforming_plate.h5'
OUTPUT_FILE = './dataset/deforming_plate_100.h5'
NUM_SAMPLES = 100

def create_small_dataset():
    """Extract first NUM_SAMPLES from the input file and save to output file."""

    print(f"Reading from: {INPUT_FILE}")
    print(f"Writing to: {OUTPUT_FILE}")
    print(f"Extracting {NUM_SAMPLES} samples...")

    with h5py.File(INPUT_FILE, 'r') as f_in:
        # Get all sample IDs and sort them
        all_sample_ids = sorted([int(k) for k in f_in['data'].keys()])
        total_samples = len(all_sample_ids)

        print(f"Total samples in input file: {total_samples}")

        # Select first NUM_SAMPLES
        selected_sample_ids = all_sample_ids[:NUM_SAMPLES]

        print(f"Selected sample IDs: {selected_sample_ids[0]} to {selected_sample_ids[-1]}")

        # Create output file
        with h5py.File(OUTPUT_FILE, 'w') as f_out:
            # Create data group
            data_group = f_out.create_group('data')

            # Copy selected samples
            print("\nCopying samples...")
            for sample_id in tqdm(selected_sample_ids):
                # Copy entire sample group (includes nodal_data and mesh_edge)
                f_in.copy(f'data/{sample_id}', data_group, name=str(sample_id))

            # Copy metadata (normalization parameters)
            print("\nCopying metadata...")
            if 'metadata' in f_in:
                f_in.copy('metadata', f_out)
            else:
                print("Warning: No metadata group found in input file")

    print(f"\n[SUCCESS] Created {OUTPUT_FILE} with {NUM_SAMPLES} samples")

    # Verify the output file
    print("\nVerifying output file...")
    with h5py.File(OUTPUT_FILE, 'r') as f:
        output_sample_ids = sorted([int(k) for k in f['data'].keys()])
        print(f"Output file contains {len(output_sample_ids)} samples")
        print(f"Sample IDs: {output_sample_ids[:5]} ... {output_sample_ids[-5:]}")

        # Check first sample structure
        first_sample = output_sample_ids[0]
        print(f"\nSample {first_sample} structure:")
        print(f"  nodal_data shape: {f[f'data/{first_sample}/nodal_data'].shape}")
        print(f"  mesh_edge shape: {f[f'data/{first_sample}/mesh_edge'].shape}")

        # Check metadata
        if 'metadata' in f:
            print(f"\nMetadata groups: {list(f['metadata'].keys())}")
            if 'normalization_params' in f['metadata']:
                print(f"  Normalization params: {list(f['metadata/normalization_params'].keys())}")

if __name__ == '__main__':
    create_small_dataset()
