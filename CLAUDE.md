# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **MeshGraphNets** implementation - a Graph Neural Network (GNN) for simulating deformable mesh dynamics. The model predicts node displacements and stresses in meshes under load, using graph message passing with encoder-processor-decoder architecture. Based on NVIDIA PhysicsNeMo and the original DeepMind MeshGraphNets paper.

## Quick Start Commands

### Training
```bash
# Edit configuration
vim config.txt

# Run training (automatically selects single/multi-GPU based on config)
python MeshGraphNets_main.py

# View logs
cat outputs/<gpu_ids>/<log_file_dir>

# Monitor training in real-time
python misc/plot_loss_realtime.py config.txt  # Visit http://localhost:5000
```

### Inference (Autoregressive Rollout)
```bash
# Generate random inference samples from existing dataset
python generate_inference_dataset.py

# Configure inference in config.txt and run
python MeshGraphNets_main.py

# Output saved to: infer/<model_name>_sample_<id>.h5
```

### Debugging & Analysis
```bash
# Analyze model output on specific sample
python misc/debug_model_output.py

# Generate training loss plot
python misc/plot_loss.py config.txt --output loss_plot.png

# Create animated GIF of mesh deformation (visualization)
python misc/animate_flag_simple.py
```

### Git Helpers
```bash
# Auto-commit all changes with timestamp
bash git-auto-push.sh

# WARNING: Reset repository to fresh state (deletes history)
bash git-fresh-start.sh
```

## Architecture Overview

### Execution Flow
1. **Entry point**: [MeshGraphNets_main.py](MeshGraphNets_main.py)
   - Parses `config.txt` via [general_modules/load_config.py](general_modules/load_config.py)
   - Auto-detects single vs. multi-GPU training based on `gpu_ids` config parameter
   - Routes based on `mode` config: `train` → training, `inference` → autoregressive rollout

2. **Training pipelines**:
   - Single GPU/CPU: [training_profiles/single_training.py](training_profiles/single_training.py)
   - Multi-GPU DDP: [training_profiles/distributed_training.py](training_profiles/distributed_training.py) (spawned via `torch.multiprocessing`)
   - Both use [training_profiles/training_loop.py](training_profiles/training_loop.py) for epoch logic

3. **Inference pipeline**:
   - Autoregressive rollout: [inference_profiles/rollout.py](inference_profiles/rollout.py)
   - Iteratively predicts normalized deltas and updates state
   - Saves trajectories to HDF5

### Core Model Architecture: [model/MeshGraphNets.py](model/MeshGraphNets.py)

**Encoder-Processor-Decoder (EPD) Structure:**
```
Input Features → Encoder → Latent Space
                    ↓
              Processor (15 blocks)
            - Each: EdgeBlock → NodeBlock
                    ↓
                Latent Space
                    ↓
              Decoder → Output Features
```

**Components:**
- **Encoder** [model/MeshGraphNets.py:Encoder]: Projects node/edge features to latent dimension
- **Processor**: Stack of `message_passing_num` GnBlocks (configurable: 1-30)
  - [EdgeBlock](model/blocks.py): Concatenates sender/receiver node features + edge features → MLP → updated edges
  - [NodeBlock](model/blocks.py): Aggregates (sum) incident edges → concatenates with node features → MLP → updated nodes
  - [HybridNodeBlock](model/blocks.py): Extends NodeBlock to aggregate both mesh edges and world edges (when `use_world_edges=True`)
  - Optional gradient checkpointing via [model/checkpointing.py](model/checkpointing.py): trades compute for ~60-70% less VRAM
- **Decoder** [model/MeshGraphNets.py:Decoder]: Projects latent to output dimension (no LayerNorm on output)

**Neural Network Details:**
- **MLP structure** (`build_mlp`): 2 hidden layers with activation functions
  - Default: ReLU activation + LayerNorm on output (except decoder)
  - Supported activations: relu (default), gelu, silu, tanh, sigmoid
  - Kaiming/He initialization (uniform, nonlinearity='relu')
- **Aggregation**: Sum (matches NVIDIA PhysicsNeMo); forces/stresses from neighbors add up
- **Optional noise injection** (during training): `std_noise` parameter adds Gaussian noise to node features

### Data Pipeline: [general_modules/](general_modules/)

- [data_loader.py](general_modules/data_loader.py): Creates PyTorch DataLoaders
  - DistributedSampler for multi-GPU (automatic shuffling, no-repeat sampling)
  - Regular sampler for single-GPU
- [mesh_dataset.py](general_modules/mesh_dataset.py): Custom Dataset class
  - Loads HDF5 mesh data (nodal_data, mesh_edge, metadata)
  - Computes z-score normalization stats (mean/std) for: node features, edge features, delta features
  - Parallel stat computation (`use_parallel_stats`) for large datasets (100+ samples)
  - Optional: one-hot node type encoding (`use_node_types`)
  - Optional: world edge computation (`use_world_edges`)
  - Returns torch_geometric.data.Data graphs
- [mesh_utils_fast.py](general_modules/mesh_utils_fast.py): Mesh utilities
  - Edge feature computation from deformed positions
  - World edge radius query (scipy KDTree backend)
  - Visualization helpers for mesh geometry

### Training Loop: [training_profiles/training_loop.py](training_profiles/training_loop.py)

**Core functions:**
- `train_epoch()`: Forward pass, loss computation, backward, gradient clipping
- `validate_epoch()`: Evaluation without gradient updates
- `test_model()`: Test set evaluation with optional visualization
- `save_debug_batch()`: Optionally saves model predictions vs. targets for debugging

**Loss & Optimization:**
- **Loss function**: MSE on normalized deltas (Δy = state_{t+1} - state_t)
  - Per-feature loss tracking (mean/max/min/std for diagnostics)
  - Feature-wise contributions visible in logs (useful for identifying problem features)
- **Optimizer**: Adam (learning_rate configurable)
- **Learning rate scheduler**: ExponentialLR (gamma=0.995, decays every epoch)
- **Gradient clipping**: max_norm=1.0 (stabilizes deep networks)

**Memory & Performance:**
- Detailed CUDA memory logging (if `verbose=True`)
- Per-batch memory tracking during training
- Test set visualization (optional, controlled by config)

## Configuration (config.txt)

**Critical rules:**
- File must be named `config.txt` (exact case), located in repo root alongside `MeshGraphNets_main.py`
- Syntax: `key value`, lines starting with `%` are comments, `'` marks section separators
- Keys are converted to lowercase internally; `key_name` in code matches `key_name` in file
- Comments with `#` are stripped from values
- Multi-GPU: parse `gpu_ids` as comma-separated list; maps to GPU device IDs

### Training Configuration
```
mode            train
gpu_ids         0              # Single GPU; use "0, 1, 2, 3" for multi-GPU DDP
dataset_dir     ./dataset/deforming_plate.h5
input_var       4              # Node input dimension (excluding node types)
output_var      4              # Node output dimension
edge_var        4              # Edge feature dimension [dx, dy, dz, distance]
Training_epochs 50
Batch_size      10
LearningR       0.001
Latent_dim      128
message_passing_num 15
num_workers     10
log_file_dir    train.log
```

### Inference Configuration
```
mode                inference
modelpath           outputs/best_model.pth      # (lowercase: modelpath, not model_path)
infer_dataset       ./infer/flag_inference.h5
inference_sample_id 0                            # or omit to use first sample
rollout_start_step  0
infer_timesteps     100                          # Number of rollout steps
inference_output_dir ./infer/
```

### Hyperparameter Tuning (Most Impactful)
- `LearningR`: Learning rate (float, 1e-6 to 1e-2; default 0.001)
- `Latent_dim`: MLP hidden dimension (int, 32-512; default 128)
- `message_passing_num`: Number of Graph Network blocks (int, 1-30; default 15)
- `Batch_size`: Batch size per GPU (int, 1-32; default 10)

### Memory Optimization
- `use_checkpointing`: Enable gradient checkpointing (True/False; trades compute for ~60-70% VRAM reduction)
- `use_parallel_stats`: Parallel normalization stat computation (True/False; recommended for 100+ samples)

### Advanced Features

**World Edges** (radius-based collision detection):
- `use_world_edges`: Enable (True/False)
- `world_radius_multiplier`: Scale factor for world edge radius (default 1.5)
- `world_max_num_neighbors`: Max neighbors per node in KDTree query (default 64; prevents explosion)
- `world_edge_backend`: scipy_kdtree (CPU, always available) or torch_cluster (GPU, faster if installed)

**Node Types** (semantic node classification):
- `use_node_types`: Enable (True/False)
- Automatically detects number of types from HDF5 metadata
- One-hot encoded and concatenated to node features after normalization

**Training Dynamics:**
- `std_noise`: Additive Gaussian noise to node features during training (default 0.0; use 1e-20 for small noise)
- `verbose`: Enable detailed CUDA memory & loss breakdowns per feature (True/False)

**Visualization:**
- `display_testset`: Show test set predictions during training (True/False)
- `test_batch_idx`: Which test batches to visualize (comma-separated, e.g., "0, 1, 2, 3")
- `plot_feature_idx`: Which feature to visualize (-1 = last feature, e.g., stress)

See [CONFIG_AND_EXECUTION_GUIDE.md](CONFIG_AND_EXECUTION_GUIDE.md) for complete parameter reference with defaults and ranges.

## Key Data Concepts

### HDF5 Dataset Format
The project uses HDF5 files to store mesh simulation data. See [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md) for complete spec.

**Structure per sample:**
- `nodal_data`: `(num_features, num_timesteps, num_nodes)` float32
  - Features: `[x, y, z, x_disp, y_disp, z_disp, stress, part_number]`
  - Displacements + stress are the physics of interest
- `mesh_edge`: `(2, num_edges)` int64
  - FEM mesh connectivity (undirected; edges stored as both directions)
- `metadata`: Sample-specific attributes
  - num_nodes, num_edges, source_filename, etc.

**Edge features** are computed dynamically:
- `[dx, dy, dz, distance]` from deformed positions (reference + displacement)
- Computed on-the-fly during data loading

### Normalization
- **Method**: Z-score normalization
  - `x_norm = (x_raw - mean) / std`
- **Statistics**: Computed separately for:
  - Node input features (from first timestep across all training samples)
  - Edge features (from deformed positions)
  - Delta features (state transitions, key for loss)
- **Storage**: Saved in checkpoint as `checkpoint['normalization']` dict
  - Includes: node_mean, node_std, edge_mean, edge_std, delta_mean, delta_std
  - Also stores: node_type_to_idx (if `use_node_types`), world_edge_radius (if `use_world_edges`)

### Model Prediction & Inference

**Training prediction:**
- Model receives **normalized** input features (node + edge)
- Predicts **normalized deltas**: `Δy_norm = model(x_norm, edge_norm)`
- Loss: MSE between predicted and target normalized deltas

**Inference (autoregressive rollout):**
1. Load initial state at t=`rollout_start_step`
2. For each step t → t+1:
   - Normalize current state using stored stats
   - Build graph (edges from deformed positions)
   - Forward pass → predicted normalized delta
   - Denormalize: `delta = delta_norm * std + mean`
   - Update state: `state_{t+1} = state_t + delta`
3. Save all predicted timesteps to HDF5

**State update equation:**
```
state_{t+1} = state_t + delta_pred
```
where `delta_pred = delta_norm * delta_std + delta_mean`

## Output Structure

### Training Outputs
- **Logs**: `outputs/<gpu_ids>/<log_file_dir>`
  - Text log with per-epoch metrics
  - Format: `Elapsed time: XXXs Epoch N Train Loss: X.XXe-0X Valid Loss: X.XXe-0X LR: X.XXXXe-0X`
  - If `verbose=True`: Per-feature loss breakdowns
- **Checkpoints**: `outputs/<gpu_ids>/`
  - `best_model.pth`: Best epoch by validation loss
  - Contains: model_state_dict, optimizer_state_dict, scheduler_state_dict, normalization stats
  - Also stored: epoch number, validation loss, model config

### Inference Outputs
- **Rollout trajectories**: `<inference_output_dir>/<model_name>_sample_<id>.h5`
  - Predicted nodal_data for all rollout timesteps
  - Mesh connectivity copied from input
  - Ready for visualization or further analysis

## Common Development Tasks

### Debugging Training

**Model output analysis:**
```bash
python misc/debug_model_output.py
# Checks: model produces reasonable outputs (not all zeros/NaN)
# Prints: input statistics, per-layer activations, output ranges
```

**Training logs:**
- Check per-epoch loss in `outputs/<gpu_ids>/<log_file_dir>`
- If `verbose=True`: See per-feature loss contributions (e.g., stress vs. displacements)
- Loss should decrease over epochs; if stuck/increasing, try:
  - Reduce `Batch_size` or `Latent_dim` (underfitting)
  - Reduce `LearningR` (divergence)
  - Enable `use_checkpointing` (gradient issues from VRAM limits)

**Out of memory (OOM) errors:**
1. Reduce `Batch_size` (most effective, ~linear scaling)
2. Enable `use_checkpointing=True` (trades 20-30% more compute for 60-70% less memory)
3. Reduce `Latent_dim` (quadratic impact on memory)
4. Reduce `message_passing_num` (linear impact)
5. Check `verbose=True` logs to see memory usage per batch

### Hyperparameter Optimization

**Workflow:**
1. Create multiple configs with different hyperparameters (e.g., `config_lr_1e3.txt`, `config_lat_128.txt`)
2. Run training: `python MeshGraphNets_main.py`
3. Compare validation losses in output logs
4. Refine search based on results

**Key parameters to tune** (by impact):
- `LearningR`: Most sensitive; try [1e-4, 1e-3, 1e-2]
- `Latent_dim`: Memory-intensive; try [64, 128, 256]
- `message_passing_num`: Expressiveness vs. depth; try [5, 10, 15, 20]
- `Batch_size`: Only varies per GPU; try [1, 5, 10]

### Visualizing Training

**Real-time dashboard:**
```bash
python misc/plot_loss_realtime.py config.txt
# Opens http://localhost:5000
# Auto-updates every 2 seconds; shows training/validation loss curves
# Includes training metadata (GPU config, log directory)
```

**Static plot (after training):**
```bash
python misc/plot_loss.py config.txt --output loss.png
# Generates PNG with log-scale loss visualization
# High DPI (150) for publication quality
```

### Running Inference

**Workflow:**
1. Train model, get `outputs/<gpu_id>/best_model.pth`
2. Prepare inference dataset: `python generate_inference_dataset.py`
3. Edit `config.txt`:
   ```
   mode            inference
   modelpath       outputs/best_model.pth
   infer_dataset   ./infer/flag_inference.h5
   infer_timesteps 100
   ```
4. Run: `python MeshGraphNets_main.py`
5. Check output: `infer/<model_name>_sample_<id>.h5`

**Key config parameters:**
- `modelpath`: Full path to checkpoint (lowercase!)
- `infer_dataset`: HDF5 file with initial conditions
- `inference_sample_id`: Which sample to use (or omit for first)
- `rollout_start_step`: Start rollout from this timestep (0 = use t=0)
- `infer_timesteps`: How many steps to predict
- `inference_output_dir`: Where to save results

### Mesh Deformation Visualization

```bash
python misc/animate_flag_simple.py
# Generates animated GIFs from flag_simple.h5 dataset
# Creates 4 views: XZ, XY, YZ, 3D isometric
# Colors nodes by stress (last feature)
# Configurable: DT, FRAME_SKIP, GIF_FPS in script
```

### Adding Features

**New node/edge features:**
1. Modify HDF5 dataset generation to include new feature in nodal_data
2. Update `input_var` in config.txt
3. Edit [mesh_dataset.py](general_modules/mesh_dataset.py):
   - Lines ~150-200: Adjust which features are loaded
   - Renormalization stats will auto-recompute

**New model architecture:**
- Edit [model/blocks.py](model/blocks.py): EdgeBlock, NodeBlock, HybridNodeBlock
- Edit [model/MeshGraphNets.py](model/MeshGraphNets.py): Encoder, GnBlock, Decoder
- Test with `message_passing_num=2` for fast iteration (minutes vs. hours)

**Custom loss functions:**
- Modify [training_profiles/training_loop.py](training_profiles/training_loop.py) `train_epoch()` function
- Currently uses MSE on normalized deltas; alternatives: L1, weighted MSE per feature, etc.

### Checkpoint Management

**Saving & loading:**
- Checkpoints saved to `outputs/<gpu_ids>/best_model.pth`
- Contains: model_state_dict, optimizer_state_dict, scheduler_state_dict
- Also: normalization stats (`checkpoint['normalization']`)
- Also: model config (`checkpoint['model_config']`) for reproducibility

**Resume training** (partial support):
- Currently: best model is saved; no mid-training resume implemented
- To resume: manually load checkpoint into optimizer/scheduler state

## Design Notes & Papers

- **Original paper**: "Learning Mesh-Based Simulation with Graph Networks" (Pfaff et al., ICML 2020, DeepMind)
- **Reference implementation**: NVIDIA PhysicsNeMo (deforming_plate example)
- **Our architecture**:
  - PyTorch + PyTorch Geometric
  - Encoder-Processor-Decoder (EPD) with configurable depth
  - Message passing via EdgeBlock (update edges) + NodeBlock (sum aggregation, update nodes)
  - Supports hybrid edges (mesh + world/collision)
  - Gradient checkpointing for memory efficiency

**Key design choices:**
- **Sum aggregation** (not mean): Physical interpretation—forces/stresses add up at nodes
- **Z-score normalization**: Stable across different feature ranges
- **Normalized delta prediction**: Decouples geometry scale from learning
- **Autoregressive inference**: Allows multi-step predictions beyond training horizon
- **Bidirectional edges**: Graph is undirected; connectivity stored both ways
- **Layer normalization**: Applied before final output (except decoder) per original paper
- **Kaiming/He init**: Suits ReLU activation; stabilizes gradient flow in deep networks

## Additional Documentation

See [docs/](docs/) for detailed architecture and feature documentation:
- [MESHGRAPHNET_ARCHITECTURE.md](docs/MESHGRAPHNET_ARCHITECTURE.md): Detailed architecture walkthrough
- [WORLD_EDGES_DOCUMENTATION.md](docs/WORLD_EDGES_DOCUMENTATION.md): Collision detection implementation
- [VRAM_OPTIMIZATION_PLAN.md](docs/VRAM_OPTIMIZATION_PLAN.md): Memory optimization strategies
- [VISUALIZATION_DENORMALIZATION.md](docs/VISUALIZATION_DENORMALIZATION.md): Denormalization for visualization
- [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md): HDF5 dataset structure specification
