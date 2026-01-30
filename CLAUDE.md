# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **MeshGraphNets** implementation - a Graph Neural Network (GNN) for simulating deformable mesh dynamics. The model predicts node displacements and stresses in meshes under load, using graph message passing with encoder-processor-decoder architecture.

## Quick Start Commands

### Training
```bash
# Edit configuration
vim config.txt

# Run training (automatically selects single/multi-GPU based on config)
python MeshGraphNets_main.py

# View logs
cat outputs/<gpu_ids>/<log_file_dir>
```

## Architecture Overview

### Execution Flow
1. **Entry point**: [MeshGraphNets_main.py](MeshGraphNets_main.py)
   - Parses `config.txt` via [general_modules/load_config.py](general_modules/load_config.py)
   - Auto-detects single vs. multi-GPU training based on `gpu_ids` config parameter
   - Routes to either `single_worker()` or spawns multi-GPU workers via `torch.multiprocessing`

### Core Model Architecture: [model/MeshGraphNets.py](model/MeshGraphNets.py)

**Encoder-Processor-Decoder (EPD) Structure:**
- **Encoder**: Projects raw mesh features (node positions/stresses + edge features) into latent space
- **Processor**: Stack of `message_passing_num` Graph Network blocks (configurable: 1-30)
  - Each block contains:
    - [EdgeBlock](model/blocks.py): Updates edge representations via MLP
    - [NodeBlock](model/blocks.py): Updates node representations using aggregated edge messages
  - Uses [checkpointing.py](model/checkpointing.py) for gradient checkpointing (if enabled) to reduce VRAM ~50%
- **Decoder**: Projects latent representations back to output space (displacements/stresses)

**Key features:**
- He initialization with 0.5 scaling for training stability
- Gradient clipping (max_norm=1.0) for deep networks
- Optional noise injection during training (std_noise parameter)
- Optional world edges (radius-based collision detection)
- One-hot node type encoding support

### Data Pipeline: [general_modules/](general_modules/)

- [data_loader.py](general_modules/data_loader.py): PyTorch DataLoader setup with DistributedSampler for multi-GPU
- [mesh_dataset.py](general_modules/mesh_dataset.py): Custom Dataset class loading HDF5 mesh data
- [normalization.py](general_modules/normalization.py): Feature normalization with min/max scaling
- [mesh_utils.py](general_modules/mesh_utils.py) & [mesh_utils_fast.py](general_modules/mesh_utils_fast.py): Mesh geometry utilities (edge construction, world radius queries using scipy KDTree)

### Training Implementations: [training_profiles/](training_profiles/)

- [single_training.py](training_profiles/single_training.py): Single GPU training
- [distributed_training.py](training_profiles/distributed_training.py): Multi-GPU DDP training (auto-called from main via `mp.spawn()`)
- [training_loop.py](training_profiles/training_loop.py): Core `train_epoch()` and `eval_epoch()` functions
  - Uses MSE loss on normalized deltas (y_t+1 - x_t)
  - Detailed CUDA memory logging for debugging OOM issues
  - Test set visualization during training

## Configuration (config.txt)

**Critical rules:**
- File must be named `config.txt` (exact case), located in repo root alongside `MeshGraphNets_main.py`
- Syntax: `key value`, lines starting with `%` are comments, `'` marks section separators
- Keys are converted to lowercase internally
- Comments with `#` are stripped from values

**Key hyperparameters for optimization:**
- `LearningR`: Learning rate (float, 1e-6 to 1e-2)
- `Latent_dim`: MLP hidden dimension (int, 32-512)
- `message_passing_num`: Number of Graph Network blocks (int, 1-30)
- `Batch_size`: Batch size per GPU (int, 1-32)

**Multi-GPU configuration:**
- `gpu_ids 0`: Single GPU (uses single_training.py)
- `gpu_ids 0, 1, 2, 3`: Multi-GPU DDP (uses distributed_training.py, effective batch = Batch_size × num_gpus)
- `gpu_ids -1`: CPU training (not recommended)

**Important parameters:**
- `dataset_dir`: Path to HDF5 dataset file
- `input_var`, `output_var`, `edge_var`: Feature dimensions (typically 4 each)
- `norm_min`, `norm_max`: Normalization range bounds
- `use_checkpointing`: Enable gradient checkpointing for VRAM reduction
- `use_world_edges`: Enable radius-based world edges (collision detection)
- `use_node_types`: Add one-hot node type encoding to features

See [CONFIG_AND_EXECUTION_GUIDE.md](CONFIG_AND_EXECUTION_GUIDE.md) for complete parameter reference and troubleshooting.

## Output Structure

Logs are saved to `outputs/<gpu_ids>/<log_file_dir>` with:
- Full config.txt contents at top
- Per-epoch training/validation loss and learning rate
- Format: `Elapsed time: XXXs Epoch N Train Loss: X.XXe-0X Valid Loss: X.XXe-0X LR: X.XXXXe-0X`
- Best model checkpoint: `outputs/best_model.pth`

## Common Development Tasks

**Debugging a training run:**
- Set `verbose True` in config.txt to enable detailed CUDA memory logging
- Check logs: `cat outputs/<gpu_ids>/<log_file_dir>`
- For OOM: reduce Batch_size, enable use_checkpointing, or reduce Latent_dim

**Hyperparameter optimization:**
1. Modify `config.txt` parameters programmatically
2. Run `python MeshGraphNets_main.py`
3. Parse `outputs/<gpu_ids>/train.log` for final validation loss
4. Track best config; iterate with refined search

**Adding a new feature:**
- Node/edge features: modify [mesh_dataset.py](general_modules/mesh_dataset.py) data loading and update config `input_var`/`edge_var`
- Model architecture: edit [model/blocks.py](model/blocks.py) and [model/MeshGraphNets.py](model/MeshGraphNets.py)
- Visualization: extend [mesh_utils_fast.py](general_modules/mesh_utils_fast.py) with new plotting functions

## Design Notes

- **Graph representation**: Each sample is a separate torch_geometric.data.Data object with node/edge indices constructed from mesh connectivity
- **Batch processing**: Uses torch_geometric DataLoader for dynamic batching of variable-sized graphs
- **Training stability**: Gradient clipping and He initialization help train deep networks (15+ message passing blocks)
- **Memory optimization**: Gradient checkpointing trades compute for memory; useful for large latent dimensions or deep networks
