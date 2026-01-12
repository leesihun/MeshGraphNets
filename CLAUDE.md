# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MeshGraphNets**: PyTorch + PyTorch Geometric implementation of DeepMind's MeshGraphNets (Pfaff et al., ICML 2020) for mesh-based physical simulation using Graph Neural Networks.

- **Paper**: "Learning Mesh-Based Simulation with Graph Networks"
- **Architecture**: Encode-Process-Decode with 15 Graph Network blocks
- **Use Case**: Predicting physical fields (stress, displacement) on FEA meshes
- **Implementation Reference**: Based on NVIDIA PhysicsNeMo

## Running the Code

### Training

**Single GPU/CPU**:
```bash
python MeshGraphNets_main.py
```

**Multi-GPU (Distributed)**:
Edit `config.txt` and set `gpu_ids` to a list (e.g., `[0, 1, 2, 3]`), then run the same command.

### Configuration

All hyperparameters are in [config.txt](config.txt):
- `dataset_dir`: Path to HDF5 dataset (default: `./dataset/dataset.h5`)
- `message_passing_num`: Number of GN blocks (default: 15)
- `Latent_dim`: Hidden feature dimension (default: 256)
- `Batch_size`: Training batch size (default: 16)
- `LearningR`: Learning rate (default: 0.001)
- `Training_epochs`: Number of epochs (default: 2002)
- `gpu_ids`: GPU IDs for training (default: -1 for CPU, or list like `[0, 1]`)

**Note**: The config parser in [general_modules/load_config.py](general_modules/load_config.py) uses a custom format. Lines starting with `%` are comments, and `'` indicates literal strings.

## Architecture

### High-Level Structure

```
Input Mesh → Encoder → Processor (15 GN blocks) → Decoder → Output Fields
```

**Detailed architecture documentation**: See [docs/MESHGRAPHNET_ARCHITECTURE.md](docs/MESHGRAPHNET_ARCHITECTURE.md) for comprehensive diagrams and explanations.

### Key Components

1. **Graph Network Block** ([model/blocks.py](model/blocks.py)):
   - `EdgeBlock`: Updates edge features using [sender_features, receiver_features, edge_features]
   - `NodeBlock`: Updates node features using [node_features, aggregated_edge_messages]
   - Both use residual connections

2. **MeshGraphNets Model** ([model/MeshGraphNets.py](model/MeshGraphNets.py)):
   - `Encoder`: Embeds raw node/edge features to latent space
   - `Processor`: Stack of 15 GraphNetwork blocks
   - `Decoder`: Maps latent features to output predictions
   - Uses LayerNorm in encoder/processor, not in decoder

3. **Training Loop** ([training_profiles/training_loop.py](training_profiles/training_loop.py)):
   - Loss: MSE between predicted and target accelerations
   - Single-GPU: [training_profiles/single_training.py](training_profiles/single_training.py)
   - Multi-GPU: [training_profiles/distributed_training.py](training_profiles/distributed_training.py)

## Dataset Format

**File**: `dataset/dataset.h5` (11 GB, excluded from git)
**Format**: HDF5 with 2,138 samples
**Documentation**: [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md)

### Structure
```
data/{sample_id}/
  ├── nodal_data: (7, 1, num_nodes) - [x, y, z, disp_x, disp_y, disp_z, stress]
  ├── mesh_edge: (2, num_edges) - edge connectivity [source_nodes, target_nodes]
  └── metadata/ - sample statistics and metadata
```

### Important Notes

- **Corner nodes only**: Dataset uses only corner nodes (~36% of FEA nodes), excluding mid-edge nodes
- **Edge extraction**: From triangular faces, not all-pairs connectivity (91% reduction vs naive approach)
- **Node count**: ~68,000 nodes/sample average
- **Edge count**: ~206,000 edges/sample average
- **Normalization**: Global statistics in `metadata/normalization_params/`

### Data Loading

The [general_modules/mesh_dataset.py](general_modules/mesh_dataset.py) defines `MeshDataset` class that:
- Loads samples from HDF5 on-demand
- Computes edge features (relative position + distance) from node positions
- Returns PyG `Data` objects with `x` (node features), `edge_index`, `edge_attr`
- Supports online normalization

## Code Structure

```
.
├── MeshGraphNets_main.py          # Entry point
├── config.txt                      # Hyperparameters
├── model/
│   ├── MeshGraphNets.py           # Main model class
│   └── blocks.py                  # EdgeBlock, NodeBlock
├── training_profiles/
│   ├── single_training.py         # Single GPU training
│   ├── distributed_training.py    # Multi-GPU training
│   └── training_loop.py           # train_epoch, validate_epoch
├── general_modules/
│   ├── load_config.py             # Config parser
│   ├── data_loader.py             # Dataset loading wrapper
│   ├── mesh_dataset.py            # PyG Dataset class
│   └── normalization.py           # Online normalization
├── dataset/
│   ├── dataset.h5                 # Main dataset (excluded from git)
│   └── DATASET_FORMAT.md          # Dataset documentation
├── docs/
│   └── MESHGRAPHNET_ARCHITECTURE.md  # Architecture details
└── misc/
    ├── build_dataset.py           # Convert FEA meshes to HDF5
    ├── visualize_dataset.py       # Dataset visualization
    └── test_timestep_dataset.py   # Dataset validation
```

## Key Implementation Details

### 1. Edge Feature Computation

Edge features are computed from node positions at runtime in [general_modules/mesh_dataset.py](general_modules/mesh_dataset.py):

```python
relative_pos = pos[edge_index[1]] - pos[edge_index[0]]  # [dx, dy, dz]
distance = torch.norm(relative_pos, dim=1, keepdim=True)
edge_attr = torch.cat([relative_pos, distance], dim=1)  # [dx, dy, dz, d]
```

This is geometrically accurate and matches the paper's approach.

### 2. Normalization Strategy

The model uses **online normalization** (accumulated statistics during training) implemented in [general_modules/normalization.py](general_modules/normalization.py). This differs from using pre-computed dataset statistics.

### 3. Output Prediction

The model predicts **node accelerations** (not next-state directly). For autoregressive rollout:
```
position_t+1 = position_t + velocity_t * dt + 0.5 * predicted_acc * dt^2
velocity_t+1 = velocity_t + predicted_acc * dt
```

### 4. Multi-GPU Training

Distributed training uses `torch.distributed` with DDP:
- Set `gpu_ids` in config.txt to list of GPU IDs (e.g., `[0, 1, 2, 3]`)
- Uses `torch.multiprocessing.spawn` to launch workers
- Each worker loads full dataset with `DistributedSampler` for data sharding
- Gradient synchronization handled automatically by DDP

### 5. Memory Considerations

For ~68k nodes × 15 GN blocks:
- **Single sample**: ~5-8 GB VRAM
- **Recommended**: Batch size 1-2 per GPU with ≥16GB VRAM
- Use gradient checkpointing if OOM (not currently implemented)

## Common Workflows

### Modifying Architecture

To change number of message passing layers:
1. Edit `message_passing_num` in [config.txt](config.txt)
2. Model automatically adjusts in [model/MeshGraphNets.py](model/MeshGraphNets.py)

To change latent dimension:
1. Edit `Latent_dim` in [config.txt](config.txt)

### Building New Dataset

If you have new FEA mesh data:
1. Place `.h5` mesh files in a directory (format: VTK-style with `points`, `cells`, `faces` arrays)
2. Run: `python misc/build_dataset.py --source_dir <dir> --output dataset/new_dataset.h5`
3. Update `dataset_dir` in [config.txt](config.txt)

### Resuming Training

Model checkpoints are saved to `outputs/best_model.pth`. To resume:
1. Load checkpoint in training script (not currently implemented)
2. Restore `model.state_dict()` and `optimizer.state_dict()`

### Inference/Evaluation

Currently only training is implemented. For inference:
1. Load checkpoint
2. Set model to eval mode: `model.eval()`
3. Pass graph through model: `predicted_acc, target_acc = model(graph)`
4. Use predicted accelerations for rollout

## Critical Files for Understanding

1. **Architecture**: [docs/MESHGRAPHNET_ARCHITECTURE.md](docs/MESHGRAPHNET_ARCHITECTURE.md) - Start here for architecture deep dive
2. **Dataset**: [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md) - Understanding data format
3. **Model**: [model/MeshGraphNets.py](model/MeshGraphNets.py) - Main model implementation
4. **Blocks**: [model/blocks.py](model/blocks.py) - Core GN block logic
5. **Data Loading**: [general_modules/mesh_dataset.py](general_modules/mesh_dataset.py) - How data is loaded and processed

## Known Limitations

1. **Fixed timesteps**: All samples must have same number of timesteps (currently 1)
2. **No checkpointing**: Training cannot be resumed from checkpoint
3. **No inference mode**: Only training loop is implemented
4. **No visualization**: Model outputs not visualized during training
5. **Bug in validation**: [training_profiles/single_training.py:84](training_profiles/single_training.py#L84) uses `train_loader` instead of `val_loader` for validation

## Dependencies

Key packages (see runtime for versions):
- PyTorch
- PyTorch Geometric
- h5py
- numpy
- tqdm

No requirements.txt file currently exists in the repository.
