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

### Inference (Autoregressive Rollout)
```bash
# Set mode: inference in config.txt with model_path, inference_sample_id, num_rollout_steps
python MeshGraphNets_main.py

# Output saved to: infer/<model_name>_sample_<id>.h5
```

### Visualization
```bash
# Generate static loss plot
python misc/plot_loss.py config.txt --output loss_plot.png

# Start real-time dashboard (visit http://localhost:5000)
python misc/plot_loss_realtime.py config.txt
```

### Utility Scripts
```bash
# Generate inference dataset from random samples
python generate_inference_dataset.py

# Auto-commit all changes with timestamp
bash git-auto-push.sh

# Debug model output on specific sample
python misc/debug_model_output.py
```

## Architecture Overview

### Execution Flow
1. **Entry point**: [MeshGraphNets_main.py](MeshGraphNets_main.py)
   - Parses `config.txt` via [general_modules/load_config.py](general_modules/load_config.py)
   - Auto-detects single vs. multi-GPU training based on `gpu_ids` config parameter
   - Routes to `single_worker()`, distributed `train_worker()`, or `run_rollout()` based on `mode` config

2. **Training modes** (selected via `mode` config: `train` or `inference`):
   - Single GPU: [training_profiles/single_training.py](training_profiles/single_training.py)
   - Multi-GPU DDP: [training_profiles/distributed_training.py](training_profiles/distributed_training.py) (spawned via `torch.multiprocessing`)
   - Inference: [inference_profiles/rollout.py](inference_profiles/rollout.py) (autoregressive time-stepping)

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
- Optional noise injection during training (`std_noise` parameter)
- Optional world edges (radius-based collision detection via `use_world_edges`)
- One-hot node type encoding support (`use_node_types`)

### Data Pipeline: [general_modules/](general_modules/)

- [data_loader.py](general_modules/data_loader.py): PyTorch DataLoader setup with DistributedSampler for multi-GPU
- [mesh_dataset.py](general_modules/mesh_dataset.py): Custom Dataset class loading HDF5 mesh data
  - Computes z-score normalization statistics (mean/std) for node, edge, and delta features
  - Handles edge feature construction from deformed mesh positions
  - Optionally adds one-hot encoded node types and world edges
- [mesh_utils_fast.py](general_modules/mesh_utils_fast.py): Mesh geometry utilities
  - Edge construction from mesh connectivity
  - World edge queries using scipy KDTree

### Training Loop: [training_profiles/training_loop.py](training_profiles/training_loop.py)

- Core `train_epoch()` and `eval_epoch()` functions
- Uses MSE loss on normalized deltas (Δy_t = state_t+1 - state_t)
- Detailed CUDA memory logging for debugging OOM issues
- Test set visualization during training
- Normalization statistics saved in checkpoint for inference

## Configuration (config.txt)

**Critical rules:**
- File must be named `config.txt` (exact case), located in repo root alongside `MeshGraphNets_main.py`
- Syntax: `key value`, lines starting with `%` are comments, `'` marks section separators
- Keys are converted to lowercase internally
- Comments with `#` are stripped from values

**Training mode:**
```
mode            train
gpu_ids         0              # Single GPU; use "0, 1, 2, 3" for multi-GPU DDP
dataset_dir     ./dataset/deforming_plate.h5
input_var       4              # Node input dimension (excluding node types)
output_var      4              # Node output dimension
edge_var        4              # Edge feature dimension
```

**Inference mode:**
```
mode                    inference
model_path              outputs/best_model.pth
inference_sample_id     0
rollout_start_step      0
num_rollout_steps       100
inference_output_dir    ./infer/
```

**Hyperparameter tuning targets** (most impactful):
- `LearningR`: Learning rate (float, 1e-6 to 1e-2)
- `Latent_dim`: MLP hidden dimension (int, 32-512)
- `message_passing_num`: Number of Graph Network blocks (int, 1-30)
- `Batch_size`: Batch size per GPU (int, 1-32)

**Memory optimization:**
- `use_checkpointing`: Enable gradient checkpointing (trades compute for ~50% VRAM reduction)
- `use_parallel_stats`: Parallel computation of normalization statistics (recommended for 100+ samples)

**Advanced features:**
- `use_world_edges`: Enable radius-based collision edges (`world_radius_multiplier`, `world_max_num_neighbors`, `world_edge_backend`)
- `use_node_types`: Add one-hot encoded node types to features
- `std_noise`: Additive Gaussian noise during training (small values like 1e-20)
- `verbose`: Enable detailed CUDA memory logging

See [CONFIG_AND_EXECUTION_GUIDE.md](CONFIG_AND_EXECUTION_GUIDE.md) for complete parameter reference and troubleshooting.

## Key Data Concepts

**Normalization:**
- Uses z-score normalization: `x_norm = (x_raw - mean) / std`
- Separate statistics computed for: node features, edge features, delta features
- Statistics are saved in checkpoint (`checkpoint['normalization']`) for use during inference

**Model Prediction:**
- Predicts **normalized deltas** (Δy_norm), not absolute values
- Denormalization: `delta = delta_norm * std + mean`
- State update: `state_t+1 = state_t + delta`
- Edge features computed from **deformed** positions (reference + displacement)

## Output Structure

Logs are saved to `outputs/<gpu_ids>/<log_file_dir>` with:
- Full config.txt contents at top
- Per-epoch training/validation loss and learning rate
- Format: `Elapsed time: XXXs Epoch N Train Loss: X.XXe-0X Valid Loss: X.XXe-0X LR: X.XXXXe-0X`
- Best model checkpoint: `outputs/best_model.pth`

Inference output saved to `<inference_output_dir>/<model_name>_sample_<id>.h5`

## Common Development Tasks

**Debugging a training run:**
- Set `verbose True` in config.txt to enable detailed CUDA memory logging
- Check logs: `cat outputs/<gpu_ids>/<log_file_dir>`
- For OOM errors: reduce `Batch_size`, enable `use_checkpointing`, or reduce `Latent_dim`

**Hyperparameter optimization:**
1. Modify `LearningR`, `Latent_dim`, `message_passing_num`, `Batch_size` in config.txt
2. Run `python MeshGraphNets_main.py`
3. Parse `outputs/<gpu_ids>/<log_file_dir>` for validation loss
4. Track best config; iterate with refined search

**Visualizing training progress:**
- Real-time: `python misc/plot_loss_realtime.py config.txt` → http://localhost:5000
- Static plot: `python misc/plot_loss.py config.txt --output loss.png`

**Running inference on a trained model:**
1. Set `mode inference` in config.txt
2. Set `model_path`, `inference_sample_id`, `num_rollout_steps`
3. Run `python MeshGraphNets_main.py`
4. Check output in `infer/` directory

**Adding new node/edge features:**
- Modify [mesh_dataset.py](general_modules/mesh_dataset.py) data loading (lines ~150-200)
- Update `input_var`/`edge_var` in config.txt
- Update denormalization logic in [inference_profiles/rollout.py](inference_profiles/rollout.py) if needed

**Modifying model architecture:**
- Edit [model/blocks.py](model/blocks.py) for EdgeBlock/NodeBlock changes
- Edit [model/MeshGraphNets.py](model/MeshGraphNets.py) for encoder/decoder/processor changes
- Test with small `message_passing_num` (e.g., 2) for faster iteration

## Design Notes

- **Graph representation**: Each sample is a separate torch_geometric.data.Data object with node/edge indices constructed from mesh connectivity
- **Batch processing**: Uses torch_geometric DataLoader for dynamic batching of variable-sized graphs
- **Training stability**: Gradient clipping and He initialization enable training of deep networks (15+ message passing blocks)
- **Memory optimization**: Gradient checkpointing trades compute for memory; essential for large `Latent_dim` or deep networks
- **Inference**: Autoregressive rollout requires denormalization at each timestep and supports both initial condition and trajectory samples
