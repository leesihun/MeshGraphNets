# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MeshGraphNets** is a Graph Neural Network (GNN) implementation for simulating deformable mesh dynamics. The model predicts node displacements and stresses in meshes under load using graph message passing with an encoder-processor-decoder architecture. Based on NVIDIA PhysicsNeMo and the original DeepMind MeshGraphNets paper.

The model predicts **normalized deltas** (state_{t+1} - state_t), not absolute values. This design allows the same model to generalize across different geometry scales and simulation conditions.

## Quick Start

### Training
```bash
# Edit configuration
vim config.txt  # Set mode=Train

# Run training (auto-detects single/multi-GPU based on gpu_ids)
python MeshGraphNets_main.py

# Monitor in real-time
python misc/plot_loss_realtime.py config.txt  # Visit http://localhost:5000

# View final logs
cat outputs/<gpu_ids>/<log_file_dir>
```

### Inference (Autoregressive Rollout)
```bash
# Configure inference
vim config.txt  # Set mode=Inference, modelpath, infer_dataset, infer_timesteps

# Run inference
python MeshGraphNets_main.py

# Output: infer/<model_name>_sample_<id>.h5
```

### Debugging & Analysis
```bash
# Check model outputs for NaNs/zeros
python misc/debug_model_output.py

# Generate training loss plot (after training)
python misc/plot_loss.py config.txt --output loss_plot.png

# Visualize mesh deformation
python misc/animate_flag_simple.py
```

## Configuration (config.txt)

**Critical requirements:**
- **File name**: Must be `config.txt` (exact case), located in repo root alongside `MeshGraphNets_main.py`
- **Format**: Plain text with custom syntax
  - Lines starting with `%` are comments
  - `'` marks section separators (optional)
  - Keys are case-insensitive (converted to lowercase internally)
  - Comments with `#` are stripped from values

### Essential Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **mode** | str | - | `Train` or `Inference` |
| **gpu_ids** | int/list | 0 | Single GPU (e.g., `0`), multi-GPU DDP (e.g., `0, 1, 2, 3`), or CPU (`-1`) |
| **dataset_dir** | str | - | Path to HDF5 training dataset |
| **infer_dataset** | str | - | Path to HDF5 inference dataset (Inference mode only) |
| **modelpath** | str | - | Path to trained model checkpoint (Inference mode only) |
| **log_file_dir** | str | train.log | Training log filename |
| **input_var** | int | 4 | Node input features (excluding node types) |
| **output_var** | int | 4 | Node output features |
| **edge_var** | int | 4 | Edge features (always: [dx, dy, dz, distance]) |

### Network Hyperparameters (Key Tuning Parameters)

| Parameter | Type | Default | Range | Impact |
|-----------|------|---------|-------|--------|
| **LearningR** | float | 0.0001 | 1e-6 to 1e-2 | **High** - most sensitive parameter |
| **Latent_dim** | int | 128 | 32-512 | **High** - MLP hidden dimension, affects VRAM quadratically |
| **message_passing_num** | int | 15 | 1-30 | **Medium** - GNN depth, affects expressiveness and VRAM linearly |
| **Batch_size** | int | 50 | 1-128 | **High** - per-GPU batch size, affects VRAM linearly |
| **Training_epochs** | int | 500 | - | Total training epochs |
| **num_workers** | int | 10 | 0-16 | DataLoader workers (reduce if CPU bound) |

### Advanced Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **use_checkpointing** | bool | False | Gradient checkpointing: trades 20-30% compute for 60-70% VRAM reduction |
| **use_parallel_stats** | bool | True | Parallel stat computation (3-5× speedup for 100+ samples; requires ~80% CPU cores) |
| **use_node_types** | bool | True | One-hot encode node types from HDF5 metadata |
| **use_world_edges** | bool | True | Radius-based collision detection edges |
| **world_radius_multiplier** | float | 1.5 | Collision radius = multiplier × min_mesh_edge_length |
| **world_max_num_neighbors** | int | 64 | Max neighbors per node in KDTree query (prevents explosion) |
| **std_noise** | float | 0.001 | Gaussian noise augmentation during training |
| **verbose** | bool | False | Detailed CUDA memory and per-feature loss breakdowns |
| **display_testset** | bool | True | Show test set predictions during training |
| **test_batch_idx** | list | 0, 1, 2, 3 | Test batch indices to visualize |
| **plot_feature_idx** | int | -1 | Feature to plot (-1 = last feature, typically stress) |

See [CONFIG_AND_EXECUTION_GUIDE.md](CONFIG_AND_EXECUTION_GUIDE.md) for complete parameter reference.

## Architecture Overview

### Execution Flow
1. **Entry point**: [MeshGraphNets_main.py](MeshGraphNets_main.py)
   - Parses `config.txt` via [general_modules/load_config.py](general_modules/load_config.py)
   - Auto-detects GPU configuration and routes to appropriate training pipeline
   - Routes based on `mode`: `Train` → training, `Inference` → autoregressive rollout

2. **Training pipelines**:
   - Single GPU/CPU: [training_profiles/single_training.py](training_profiles/single_training.py)
   - Multi-GPU DDP: [training_profiles/distributed_training.py](training_profiles/distributed_training.py)
   - Both use [training_profiles/training_loop.py](training_profiles/training_loop.py) for epoch logic
   - Saves best model and normalization stats to `outputs/<gpu_ids>/best_model.pth`

3. **Inference pipeline**:
   - [inference_profiles/rollout.py](inference_profiles/rollout.py)
   - Loads trained model with normalization statistics
   - Iteratively predicts normalized deltas and updates state
   - Saves autoregressive trajectories to HDF5

### Core Model: Encoder-Processor-Decoder (EPD)

**Structure** (in [model/MeshGraphNets.py](model/MeshGraphNets.py)):
```
Input (node + edge features)
         ↓
      Encoder (project to latent_dim)
         ↓
      Processor (message_passing_num blocks)
      ├─ EdgeBlock: concatenates sender/receiver nodes + edge features → MLP
      └─ NodeBlock: aggregates incident edges (sum) → concatenates with node features → MLP
      (repeat message_passing_num times)
         ↓
      Decoder (project to output_dim, no LayerNorm)
         ↓
      Output (normalized deltas)
```

**Components** (in [model/](model/)):
- **Encoder**: Linear projection from input_dim to latent_dim
- **GnBlock** (EdgeBlock + NodeBlock): Message passing iteration
  - EdgeBlock ([model/blocks.py](model/blocks.py)): Updates edges based on node pairs
  - NodeBlock ([model/blocks.py](model/blocks.py)): Aggregates incident edges with sum operation
  - HybridNodeBlock: Extends NodeBlock for mesh + world edges
  - Optional gradient checkpointing ([model/checkpointing.py](model/checkpointing.py)): reduces VRAM ~60-70%
- **Decoder**: Linear projection from latent_dim to output_dim

**Neural Network Details**:
- **MLP structure**: 2 hidden layers with LayerNorm on output (except decoder)
- **Activation**: ReLU (default), also supports gelu, silu, tanh, sigmoid
- **Initialization**: Kaiming/He (uniform, nonlinearity='relu')
- **Aggregation**: Sum (matches NVIDIA PhysicsNeMo)—forces/stresses accumulate at nodes

### Data Pipeline (in [general_modules/](general_modules/))

- **[data_loader.py](general_modules/data_loader.py)**: Creates PyTorch DataLoaders
  - DistributedSampler for multi-GPU DDP (automatic shuffling, no-repeat)
  - Regular sampler for single-GPU

- **[mesh_dataset.py](general_modules/mesh_dataset.py)**: Custom Dataset class
  - Loads HDF5 mesh data (nodal_data, mesh_edge, metadata)
  - Computes Z-score normalization (separate stats for: node features, edge features, delta features)
  - Parallel stat computation (`use_parallel_stats`) for large datasets
  - Optional: one-hot node type encoding (`use_node_types`)
  - Optional: world edge computation (`use_world_edges`)
  - Returns torch_geometric.data.Data graphs

- **[mesh_utils_fast.py](general_modules/mesh_utils_fast.py)**: Mesh utilities
  - Edge feature computation from deformed positions
  - World edge radius query (scipy KDTree backend, optional torch_cluster GPU backend)
  - Visualization helpers

### Training Loop ([training_profiles/training_loop.py](training_profiles/training_loop.py))

**Key functions**:
- `train_epoch()`: Forward pass, MSE loss on normalized deltas, backward, gradient clipping (max_norm=1.0)
- `validate_epoch()`: Evaluation without gradient updates
- `test_model()`: Test set evaluation with optional visualization
- `save_debug_batch()`: Optionally saves model predictions vs. targets

**Optimization**:
- **Optimizer**: Adam
- **Learning rate scheduler**: ExponentialLR (gamma=0.995, decays each epoch)
- **Loss**: MSE on normalized deltas per feature
- **Gradient clipping**: max_norm=1.0 (stabilizes deep networks)

## Data Concepts

### HDF5 Dataset Format
See [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md) for complete spec.

**Per-sample structure**:
- `nodal_data`: `(num_features, num_timesteps, num_nodes)` float32
  - Features: `[x, y, z, x_disp, y_disp, z_disp, stress, ...]`
- `mesh_edge`: `(2, num_edges)` int64 (FEM connectivity, stored bidirectional)
- `metadata`: Sample attributes (num_nodes, num_edges, source_filename, etc.)

**Edge features** (computed dynamically):
- `[dx, dy, dz, distance]` from deformed positions (reference + displacement)
- Always bidirectional: `edge_index = [mesh_edge; mesh_edge[[1,0]]]`

### Normalization
- **Method**: Z-score normalization per feature
  - `x_norm = (x_raw - mean) / std`
- **Statistics**: Computed separately for:
  - Node features (from first timestep, all training samples)
  - Edge features (from deformed positions)
  - Delta features (state transitions—critical for loss)
- **Storage**: Saved in checkpoint under `checkpoint['normalization']` dict
  - Includes: node_mean, node_std, edge_mean, edge_std, delta_mean, delta_std
  - Also: node_type_to_idx (if `use_node_types`), world_edge_radius (if `use_world_edges`)

### Model Prediction & Inference

**Training**:
- Model receives normalized input (node + edge features)
- Predicts normalized deltas: `Δy_norm = model(x_norm, edge_norm)`
- Loss: MSE between predicted and target normalized deltas

**Inference** (autoregressive rollout):
1. Load initial state at t=`rollout_start_step`
2. For each step t → t+1:
   - Normalize current state using checkpoint stats
   - Build graph (edges from deformed positions)
   - Forward pass → predicted normalized delta
   - Denormalize: `delta = delta_norm * std + mean`
   - Update state: `state_{t+1} = state_t + delta`
3. Save all predicted timesteps to HDF5

## Output Structure

### Training Outputs
- **Logs**: `outputs/<gpu_ids>/<log_file_dir>`
  - Format: `Elapsed time: XXXs Epoch N Train Loss: X.XXe-0X Valid Loss: X.XXe-0X LR: X.XXXXe-0X`
  - If `verbose=True`: Per-feature loss breakdowns
- **Checkpoints**: `outputs/<gpu_ids>/`
  - `best_model.pth`: Best epoch (by validation loss)
  - Contains: model_state_dict, optimizer_state_dict, scheduler_state_dict, normalization stats, model config

### Inference Outputs
- **Rollout trajectories**: `<inference_output_dir>/<model_name>_sample_<id>.h5`
  - Predicted nodal_data for all rollout timesteps
  - Mesh connectivity copied from input dataset
  - Ready for visualization or analysis

## Visualization Tools

Located in [misc/](misc/), see [misc/README.md](misc/README.md) for full details:

### Static Loss Plot
```bash
python misc/plot_loss.py config.txt --output loss_plot.png
```
Generates PNG image of training/validation loss on log scale.

### Real-Time Dashboard
```bash
# Install dependencies (FastAPI + Uvicorn)
pip install -r misc/requirements_plotting.txt

# Start server (auto-updates every 2s)
python misc/plot_loss_realtime.py config.txt
# Open browser: http://localhost:5000
```
Live dashboard with statistics, training metadata, and interactive tooltips. Includes FastAPI documentation at `/docs`.

### Mesh Visualization
```bash
python misc/animate_flag_simple.py
```
Generates animated GIFs of mesh deformation from `flag_simple.h5`, colored by stress. Creates 4 views: XZ, XY, YZ, 3D isometric.

### Model Debug
```bash
python misc/debug_model_output.py
```
Checks model for NaN/zero outputs; prints per-layer activations and ranges.

## Common Development Tasks

### Troubleshooting

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| **Out of Memory (OOM)** | `RuntimeError: CUDA out of memory` | 1. Reduce `Batch_size` (most effective)<br>2. Enable `use_checkpointing=True`<br>3. Reduce `Latent_dim` to 64-128<br>4. Use multi-GPU (`gpu_ids 0, 1, 2, 3`) |
| **Loss not decreasing** | Loss constant or increasing | 1. Reduce `LearningR` by 10× (e.g., 1e-4→1e-5)<br>2. Increase `message_passing_num`<br>3. Verify dataset path and format |
| **NaN loss** | Loss becomes `nan` during training | 1. Reduce `LearningR` to 1e-5<br>2. Reduce `std_noise`<br>3. Check normalization stats (may indicate data issues) |
| **Slow training** | Epochs take very long | 1. Adjust `num_workers` (typically 2-8)<br>2. Use multi-GPU<br>3. Reduce dataset size for debugging |

### Hyperparameter Optimization

**Workflow**:
1. Create multiple config files with different hyperparameters (e.g., `config_lr_1e3.txt`, `config_lat_256.txt`)
2. Run training: `python MeshGraphNets_main.py`
3. Compare validation losses in output logs
4. Refine search based on results

**Most impactful parameters** (by sensitivity):
- `LearningR`: [1e-4, 1e-3, 1e-2]
- `Latent_dim`: [64, 128, 256, 512]
- `message_passing_num`: [5, 10, 15, 20, 30]
- `Batch_size`: [1, 5, 10, 20]

### Adding Features

**New node/edge features**:
1. Modify HDF5 dataset generation to include new features in nodal_data
2. Update `input_var` and `output_var` in config.txt
3. Update [mesh_dataset.py](general_modules/mesh_dataset.py) feature loading (~lines 150-200)
4. Normalization stats auto-recompute

**New model architecture**:
- Edit [model/blocks.py](model/blocks.py): EdgeBlock, NodeBlock, HybridNodeBlock
- Edit [model/MeshGraphNets.py](model/MeshGraphNets.py): Encoder, GnBlock, Decoder
- Test with `message_passing_num=2` for fast iteration

**Custom loss functions**:
- Modify [training_profiles/training_loop.py](training_profiles/training_loop.py) `train_epoch()` function
- Currently: MSE on normalized deltas
- Alternatives: L1, weighted MSE per feature, etc.

### Running Inference

**Workflow**:
1. Train model: `python MeshGraphNets_main.py` (get `outputs/<gpu_id>/best_model.pth`)
2. Prepare inference dataset: `python generate_inference_dataset.py`
3. Edit `config.txt`:
   ```
   mode            Inference
   modelpath       outputs/0/best_model.pth
   infer_dataset   ./infer/flag_inference.h5
   infer_timesteps 100
   ```
4. Run: `python MeshGraphNets_main.py`
5. Check output: `infer/<model_name>_sample_<id>.h5`

**Key config parameters**:
- `modelpath`: Full path to checkpoint (lowercase!)
- `infer_dataset`: HDF5 file with initial conditions
- `infer_timesteps`: Number of rollout steps to predict
- `inference_output_dir`: Output directory (auto-created)

## Design Notes

**Key design choices**:
- **Sum aggregation** (not mean): Matches NVIDIA PhysicsNeMo; physical interpretation is forces/stresses add at nodes
- **Z-score normalization**: Stable across feature ranges; enables scale generalization
- **Normalized delta prediction**: Decouples geometry scale from learned dynamics; allows models to work across geometry sizes
- **Autoregressive inference**: Enables multi-step predictions beyond training horizon
- **Bidirectional edges**: Graph is undirected; connectivity stored both directions
- **Kaiming/He initialization**: Suits ReLU; stabilizes gradient flow in deep networks
- **Gradient clipping (max_norm=1.0)**: Stabilizes training in deep networks

**References**:
- Original paper: "Learning Mesh-Based Simulation with Graph Networks" (Pfaff et al., ICML 2020, DeepMind)
- Reference: NVIDIA PhysicsNeMo (deforming_plate example)
- Implementation: PyTorch + PyTorch Geometric

## Additional Documentation

- [CONFIG_AND_EXECUTION_GUIDE.md](CONFIG_AND_EXECUTION_GUIDE.md): Complete parameter reference with all defaults and ranges
- [docs/MESHGRAPHNET_ARCHITECTURE.md](docs/MESHGRAPHNET_ARCHITECTURE.md): Detailed architecture walkthrough
- [docs/WORLD_EDGES_DOCUMENTATION.md](docs/WORLD_EDGES_DOCUMENTATION.md): Collision detection (world edges) implementation
- [docs/VRAM_OPTIMIZATION_PLAN.md](docs/VRAM_OPTIMIZATION_PLAN.md): Memory optimization strategies and profiling
- [docs/VISUALIZATION_DENORMALIZATION.md](docs/VISUALIZATION_DENORMALIZATION.md): Denormalization techniques for visualization
- [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md): HDF5 dataset structure specification
- [misc/README.md](misc/README.md): Training visualization tools (plotting, real-time dashboard)
