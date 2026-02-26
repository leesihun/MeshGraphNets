# Configuration and Execution Guide

Complete reference for all configuration parameters and execution modes in MeshGraphNets. This guide also serves as a specification for generating valid config files and HDF5 datasets programmatically.

## Table of Contents

- [Config File Format](#config-file-format)
- [Parameter Reference](#parameter-reference)
  - [General Parameters](#general-parameters)
  - [Dataset Parameters](#dataset-parameters)
  - [Network Architecture Parameters](#network-architecture-parameters)
  - [Training Hyperparameters](#training-hyperparameters)
  - [Node Type Parameters](#node-type-parameters)
  - [World Edge Parameters](#world-edge-parameters)
  - [Memory Optimization Parameters](#memory-optimization-parameters)
  - [Performance Optimization Parameters](#performance-optimization-parameters)
  - [Diagnostics and Visualization Parameters](#diagnostics-and-visualization-parameters)
  - [Inference Parameters](#inference-parameters)
- [Execution Modes](#execution-modes)
  - [Training (Single GPU)](#training-single-gpu)
  - [Training (Multi-GPU DDP)](#training-multi-gpu-ddp)
  - [Training (CPU)](#training-cpu)
  - [Inference (Autoregressive Rollout)](#inference-autoregressive-rollout)
- [HDF5 Dataset Structure](#hdf5-dataset-structure)
- [Data Flow: How the Model Uses Data](#data-flow-how-the-model-uses-data)
- [Example Configurations](#example-configurations)
- [Config File Parsing Details](#config-file-parsing-details)
- [Checkpoint Contents](#checkpoint-contents)
- [Output Directory Structure](#output-directory-structure)
- [Troubleshooting](#troubleshooting)

---

## Config File Format

Configuration is stored in a plain-text file (default: `config.txt`) in the repository root. The config file path can be overridden via command-line:

```bash
python MeshGraphNets_main.py --config my_custom_config.txt
```

### Syntax Rules

| Rule | Description | Example |
|------|-------------|---------|
| **Key-value pairs** | Space or tab separated | `LearningR 0.0001` |
| **Comments** | Lines starting with `%` | `% This is a comment` |
| **Inline comments** | Text after `#` is stripped | `Batch_size 50  # per-GPU` |
| **Section separators** | Lines starting with `'` (optional, ignored) | `'` |
| **Case-insensitive keys** | All keys converted to lowercase internally | `LearningR` becomes `learningr` |
| **List values** | Comma-separated | `gpu_ids 0, 1, 2, 3` |
| **Boolean values** | `True` or `False` (case-insensitive) | `verbose False` |
| **Reserved keyword** | Lines with key `reserved` are skipped | `reserved ...` |

### Type Parsing

Values are automatically parsed in this order:
1. **Comma-separated list** -> list of int/float or strings
2. **Space-separated values** -> list of int/float or strings (if multiple tokens)
3. **Boolean** -> `True`/`False`
4. **Numeric** -> int (no decimal point) or float (with decimal point)
5. **String** -> lowercase string (fallback)

**Important implications:**
- `LearningR 0.0001` is stored as `config['learningr'] = 0.0001` (float)
- `gpu_ids 0, 1, 2, 3` is stored as `config['gpu_ids'] = [0, 1, 2, 3]` (list of int)
- `gpu_ids 0` is stored as `config['gpu_ids'] = 0` (int, not list)
- `verbose True` is stored as `config['verbose'] = True` (bool)
- `mode Train` is stored as `config['mode'] = 'train'` (lowercase string)
- String values with spaces may be parsed as lists: `my_param hello world` -> `['hello', 'world']`

---

## Parameter Reference

### General Parameters

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `model` | str | - | Yes | Model architecture name. Currently only `MeshGraphNets` is supported. |
| `mode` | str | - | Yes | Execution mode: `Train` or `Inference`. Case-insensitive (stored as lowercase). |
| `gpu_ids` | int or list | - | Yes | GPU device(s). Single int for single-GPU, comma-separated for multi-GPU DDP, `-1` for CPU. |
| `log_file_dir` | str | - | No | Log filename relative to `outputs/`. E.g., `train0.log` writes to `outputs/train0.log`. |
| `modelpath` | str | - | Yes | Path to save/load model checkpoint (`.pth` file). **Single-GPU only**: this path is used when saving. For multi-GPU DDP, the checkpoint is always saved to `outputs/best_model.pth` regardless of this value (but the key must still be present to avoid errors). |

**GPU routing logic:**
- Single value (e.g., `0`): Routes to `single_training.py`
- Multiple values (e.g., `0, 1, 2, 3`): Routes to `distributed_training.py` with DDP
- `-1`: Routes to `single_training.py` on CPU

---

### Dataset Parameters

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `dataset_dir` | str | - | Train | Path to HDF5 training dataset file. |
| `infer_dataset` | str | - | Inference | Path to HDF5 inference dataset file. |

**Notes:**
- Paths can be relative (to working directory) or absolute
- HDF5 dataset must follow the structure defined in [HDF5 Dataset Structure](#hdf5-dataset-structure) below
- The dataset is automatically split 80/10/10 (train/val/test) during training with seed=42
- **All samples must have the same number of timesteps** (current limitation)

---

### Network Architecture Parameters

These parameters define the model structure. They are saved in the checkpoint and restored during inference.

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `input_var` | int | - | 1+ | Number of physical node input features (excluding node types). Typically 4: `[x_disp, y_disp, z_disp, stress]`. |
| `output_var` | int | - | 1+ | Number of node output features. Typically 4: `[x_disp, y_disp, z_disp, stress]`. |
| `edge_var` | int | - | 1+ | Number of edge features. Always 4: `[dx, dy, dz, distance]`. |
| `Latent_dim` | int | 128 | 32-512 | Hidden dimension of all MLPs. Affects VRAM **quadratically** (MLP weights scale as latent_dim^2). |
| `message_passing_num` | int | 15 | 1-30 | Number of message passing iterations. More = larger receptive field but more VRAM (linear scaling). |

**Architecture details:**
- The actual model input dimension is `input_var + num_node_types` when `use_node_types=True`
- All MLPs use 2 hidden layers with ReLU activation and LayerNorm (except the decoder, which has no LayerNorm)
- Each message passing block contains an EdgeBlock and a NodeBlock (or HybridNodeBlock if world edges are enabled)
- Residual connections are applied to **node features only** (not edges), matching the NVIDIA PhysicsNeMo implementation
- EdgeBlock input: concatenation of sender nodes, receiver nodes, and edge features -> `3 * latent_dim`
- NodeBlock input: concatenation of node features and aggregated edges -> `2 * latent_dim` (or `3 * latent_dim` for HybridNodeBlock with world edges)
- Aggregation: **sum** (not mean) -- forces/stresses accumulate at nodes

---

### Training Hyperparameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `Training_epochs` | int | - | 1+ | Total number of training epochs. |
| `Batch_size` | int | - | 1-128 | Per-GPU batch size. For multi-GPU DDP, effective batch size = `Batch_size * num_GPUs`. |
| `LearningR` | float | 0.0001 | 1e-6 to 1e-2 | Initial learning rate for Adam optimizer. **Most sensitive hyperparameter.** |
| `num_workers` | int | 10 | 0-16 | Number of DataLoader worker processes. Set to 0 for debugging (single-process loading). |
| `std_noise` | float | 0.001 | 0-0.1 | Standard deviation of Gaussian noise added to input features during training. Set to 0 to disable. |
| `residual_scale` | float | 0.1 | 0-1.0 | Defined in config but **not currently used** in model code (residual connections use scale=1.0). Reserved for future use. |

**Optimizer & scheduler details:**
- **Optimizer**: Adam (`torch.optim.Adam`)
- **LR Scheduler (Single GPU)**: `ExponentialLR` with `gamma=0.995` (decays every epoch)
- **LR Scheduler (Multi-GPU DDP)**: `ReduceLROnPlateau` (`factor=0.5`, `patience=2`, `min_lr=1e-8`)
- **Gradient Clipping**: `max_norm=5.0` (applied after gradient computation)
- **Loss Function**: MSE on normalized deltas (averaged across all nodes and features)
- **Weight Initialization**: Kaiming/He uniform (`nonlinearity='relu'`)
- **Data Split**: 80% train, 10% validation, 10% test (fixed `seed=42`)

---

### Node Type Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_node_types` | bool | False | Enable one-hot encoding of node types from HDF5 metadata. Node types are read from the **last feature** (index -1) of `nodal_data`. |

**How node types work:**
1. Unique node types are auto-discovered from the first 10 samples
2. Types are mapped to contiguous indices (e.g., `{0: 0, 1: 1, 3: 2}`)
3. One-hot vectors are concatenated **after** z-score normalization of physical features
4. Model input dimension becomes `input_var + num_node_types`
5. The mapping (`node_type_to_idx`) and count (`num_node_types`) are saved in the checkpoint

---

### World Edge Parameters

World edges provide radius-based collision detection between non-adjacent nodes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_world_edges` | bool | False | Enable world edge computation. |
| `world_radius_multiplier` | float | 1.5 | Collision radius = multiplier * min_mesh_edge_length. Auto-computed from first 10 samples. |
| `world_max_num_neighbors` | int | 64 | Maximum neighbors per node in radius query. Prevents edge explosion in dense regions. |
| `world_edge_backend` | str | `torch_cluster` | Backend: `torch_cluster` (GPU, 5-10x faster) or `scipy_kdtree` (CPU fallback, always available). Falls back to `scipy_kdtree` if `torch_cluster` is not installed. **Must be explicitly specified in inference configs** — `rollout.py` has no fallback default and will crash if this key is absent. |

**How world edges work:**
- Filtered to exclude edges already in the mesh topology
- Use same `[dx, dy, dz, distance]` format and normalization as mesh edges
- When enabled, the processor uses `HybridNodeBlock` which aggregates from both mesh and world edges separately (each through sum aggregation) before combining: `[node_features, mesh_agg, world_agg]`
- The computed `world_edge_radius` is saved in the checkpoint for inference
- See [docs/WORLD_EDGES_DOCUMENTATION.md](docs/WORLD_EDGES_DOCUMENTATION.md) for full details

**Backend default discrepancy:**
- `mesh_dataset.py` default (training): `torch_cluster` (falls back to `scipy_kdtree` if not installed)
- `rollout.py` (inference): **no default** — will crash with `AttributeError` if `world_edge_backend` is absent from the config
- All provided example configs explicitly set `world_edge_backend scipy_kdtree` for portability
- Recommended practice: always explicitly specify this key in every config file

---

### Memory Optimization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_checkpointing` | bool | False | Gradient checkpointing for message passing blocks. Trades 20-30% compute for 60-70% VRAM reduction. Only active during training. |

---

### Performance Optimization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_parallel_stats` | bool | True | Parallel processing for dataset normalization statistics. Uses ~45% of CPU cores (max 64 workers). Auto-disabled for < 10 samples. |

**Notes:**
- Falls back to serial processing if multiprocessing fails
- Each worker opens its own HDF5 file handle (read-only) to avoid file locking
- For datasets with many timesteps, statistics are computed from a subsample of up to 500 timesteps per sample

---

### Diagnostics and Visualization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | bool | False | Detailed diagnostics: per-feature loss breakdowns, CUDA memory tracking, per-layer gradient statistics. |
| `monitor_gradients` | bool | True | Track gradient norms during training. Warns about vanishing (<1e-6) or exploding (>100) gradients. |
| `display_testset` | bool | True | Generate PyVista visualizations during test evaluation (every 10 epochs). |
| `test_batch_idx` | list | `[0]` | Which test batch indices to visualize and save to HDF5. Code fallback default is `[0]`. All standard config files explicitly set this to `0, 1, 2, 3`. |
| `plot_feature_idx` | int | -1 | Feature index to visualize. `-1` = last feature (stress), `-2` = second-to-last (z_disp), or positive index. Standard configs use `-2`. |

**Notes:**
- Test evaluation runs every 10 epochs, saving results to `outputs/test/<gpu_ids>/<epoch>/`
- Test outputs include both normalized and denormalized predictions vs. ground truth
- Debug data (`.npz` files with predictions/targets) is saved starting from epoch 5

---

### Inference Parameters

These parameters are only used when `mode=Inference`.

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `modelpath` | str | - | Yes | Path to trained model checkpoint (`.pth`). Must contain `model_state_dict`, `normalization`, and `model_config` keys. |
| `infer_dataset` | str | - | Yes | Path to HDF5 dataset with initial conditions for rollout. |
| `infer_timesteps` | int | - | Yes* | Number of autoregressive steps. If `None` and dataset has multiple timesteps, defaults to `num_timesteps - 1`. |
| `inference_output_dir` | str | `outputs/rollout` | No | Output directory for rollout HDF5 files. Auto-created. |

**Notes:**
- Architecture parameters (`input_var`, `output_var`, `latent_dim`, etc.) are **overridden** by values in the checkpoint's `model_config`
- Normalization statistics are loaded from the checkpoint's `normalization` dict
- All samples in `infer_dataset` are processed sequentially
- Output files: `rollout_sample{id}_steps{N}.h5`
- Rollout always starts from timestep 0

---

## Execution Modes

### Training (Single GPU)

```bash
# config.txt
mode        Train
gpu_ids     0
dataset_dir ./dataset/my_data.h5
modelpath   ./outputs/my_model.pth
```

```bash
python MeshGraphNets_main.py
```

**Pipeline**: `MeshGraphNets_main.py` -> `single_training.py` -> `training_loop.py`

- Uses `ExponentialLR` scheduler (gamma=0.995)
- Saves best model to path specified by `modelpath`
- Logs to `outputs/<log_file_dir>`
- Test evaluation every 10 epochs

### Training (Multi-GPU DDP)

```bash
# config.txt
mode        Train
gpu_ids     0, 1, 2, 3
dataset_dir ./dataset/my_data.h5
```

```bash
python MeshGraphNets_main.py
```

**Pipeline**: `MeshGraphNets_main.py` -> `mp.spawn()` -> `distributed_training.py` -> `training_loop.py`

- Uses `torch.distributed` with NCCL backend (Gloo on CPU)
- `DistributedSampler` ensures each GPU sees different data
- Effective batch size = `Batch_size * num_GPUs`
- Uses `ReduceLROnPlateau` scheduler (factor=0.5, patience=2)
- Only rank 0 saves checkpoints (to `outputs/best_model.pth`) and writes logs
- Communication: `localhost:12355`

### Training (CPU)

```bash
# config.txt
mode        Train
gpu_ids     -1
dataset_dir ./dataset/my_data.h5
```

Same pipeline as single GPU, but on CPU. Useful for debugging.

### Inference (Autoregressive Rollout)

```bash
# config.txt
mode            Inference
gpu_ids         0
modelpath       ./outputs/my_model.pth
infer_dataset   ./infer/initial_conditions.h5
infer_timesteps 100
```

```bash
python MeshGraphNets_main.py
```

**Pipeline**: `MeshGraphNets_main.py` -> `rollout.py`

**Rollout loop:**
1. Load checkpoint (model weights + normalization stats + model config)
2. Override config with `model_config` from checkpoint
3. For each sample in `infer_dataset`:
   a. Extract initial state at timestep 0
   b. For each step t -> t+1:
      - Normalize current state using checkpoint stats
      - Compute edge features from **deformed** positions
      - Build graph (including world edges if enabled)
      - Forward pass -> predicted normalized delta
      - Denormalize: `delta = delta_norm * delta_std + delta_mean`
      - Update state: `state_{t+1} = state_t + delta`
4. Save predicted trajectory to `<inference_output_dir>/rollout_sample{id}_steps{N}.h5`

---

## HDF5 Dataset Structure

This section defines the exact HDF5 structure that the training code expects. Use this as a specification when generating datasets.

### Complete Hierarchy

```
dataset.h5
|-- [Attributes]                           # File-level metadata
|   |-- num_samples: int                   # Total number of samples
|   |-- num_features: int                  # Number of features per node (typically 8)
|   +-- num_timesteps: int                 # Timesteps per sample (must be equal across all samples)
|
|-- data/                                  # Main data group
|   |-- {sample_id}/                       # Sample (integer ID, not necessarily sequential)
|   |   |-- nodal_data                     # Dataset: (num_features, num_timesteps, num_nodes) float32
|   |   |-- mesh_edge                      # Dataset: (2, num_edges) int64
|   |   +-- metadata/                      # Per-sample metadata group
|   |       |-- [Attributes]
|   |       |   |-- num_nodes: int         # Required
|   |       |   |-- num_edges: int         # Required
|   |       |   |-- source_filename: str   # Optional
|   |       |   +-- ...                    # Other optional attributes
|   |       |-- feature_min                # (num_features,) float32, Optional
|   |       |-- feature_max                # (num_features,) float32, Optional
|   |       |-- feature_mean               # (num_features,) float32, Optional
|   |       +-- feature_std                # (num_features,) float32, Optional
|   |-- {sample_id}/
|   |   +-- ...
|   +-- ...
|
+-- metadata/                              # Global metadata group
    |-- feature_names                      # (num_features,) variable-length byte string
    |-- normalization_params/              # Global normalization statistics
    |   |-- min                            # (num_features,) float32
    |   |-- max                            # (num_features,) float32
    |   |-- mean                           # (num_features,) float32
    |   |-- std                            # (num_features,) float32
    |   |-- delta_mean                     # (output_var,) float32 -- auto-updated by training
    |   +-- delta_std                      # (output_var,) float32 -- auto-updated by training
    +-- splits/                            # Optional pre-defined splits
        |-- train                          # (N_train,) int64
        |-- val                            # (N_val,) int64
        +-- test                           # (N_test,) int64
```

### nodal_data

**Path**: `data/{sample_id}/nodal_data`
**Shape**: `(num_features, num_timesteps, num_nodes)` -- **features-first layout**
**Dtype**: `float32`

**Standard feature order** (8 features):

| Index | Name | Description | Used by model |
|-------|------|-------------|---------------|
| 0 | `x_coord` | X coordinate (reference position) | Yes -- reference geometry |
| 1 | `y_coord` | Y coordinate (reference position) | Yes -- reference geometry |
| 2 | `z_coord` | Z coordinate (reference position) | Yes -- reference geometry |
| 3 | `x_disp` | X displacement (mm) | Yes -- input/output feature |
| 4 | `y_disp` | Y displacement (mm) | Yes -- input/output feature |
| 5 | `z_disp` | Z displacement (mm) | Yes -- input/output feature |
| 6 | `stress` | Stress (MPa) | Yes -- input/output feature |
| 7 | `part_number` | Part number (integer) | Optional -- for node types |

**How the model reads this:**
- **Reference position**: `nodal_data[:3, t, :]` -> `[x, y, z]` coordinates
- **Physical features (input/output)**: `nodal_data[3:3+input_var, t, :]` -> typically `[x_disp, y_disp, z_disp, stress]`
- **Node types**: `nodal_data[-1, 0, :]` -> last feature, first timestep (if `use_node_types=True`)
- **Deformed position**: `reference + displacement` = `nodal_data[:3, t, :] + nodal_data[3:6, t, :]`

**Configuring for different feature counts:**
- `input_var` and `output_var` control which features starting from index 3 are used
- If `input_var=4`: features at indices [3, 4, 5, 6] are used as node input
- If `input_var=3`: only features at indices [3, 4, 5] are used (no stress)
- The first 3 features (indices 0-2) are **always** coordinates and are never directly fed to the model as node features

### mesh_edge

**Path**: `data/{sample_id}/mesh_edge`
**Shape**: `(2, num_edges)` -- stored **unidirectional**
**Dtype**: `int64`

- Row 0: source node indices
- Row 1: target node indices
- Node indices use compact numbering (0 to num_nodes-1)
- No self-loops, no duplicates
- The code automatically creates bidirectional edges: `edge_index = [mesh_edge; mesh_edge[[1,0]]]`

### Timestep Handling

The model supports both single-timestep and multi-timestep datasets:

| Scenario | `num_timesteps` | Training pairs per sample | Target computation |
|----------|-----------------|--------------------------|-------------------|
| **Static** | 1 | 1 | `delta = feature_values - 0` (from zero initial state) |
| **Transient** | T > 1 | T - 1 | `delta = state_{t+1} - state_t` |

For transient data:
- Total training samples = `num_samples * (num_timesteps - 1)`
- Each training pair uses consecutive timesteps: `(t, t+1)`
- **All samples must have the same number of timesteps**

---

## Data Flow: How the Model Uses Data

This section traces exactly how raw HDF5 data is transformed into model inputs and outputs. Use this to verify that your generated data will be processed correctly.

### 1. Loading and Feature Extraction

```
nodal_data shape: [features, timesteps, nodes]

For timestep t:
  reference_pos     = nodal_data[:3, t, :].T            -> [N, 3]
  physical_features = nodal_data[3:3+input_var, t, :].T  -> [N, input_var]

For single timestep (T=1):
  x_raw = zeros(N, input_var)     <-- input is all zeros
  y_raw = nodal_data[3:3+output_var, 0, :].T
  target_delta = y_raw - x_raw    <-- delta equals the feature values themselves

For multi-timestep (T>1):
  x_raw = nodal_data[3:3+input_var, t, :].T       <-- state at time t
  y_raw = nodal_data[3:3+output_var, t+1, :].T    <-- state at time t+1
  target_delta = y_raw - x_raw                     <-- delta between consecutive steps
```

### 2. Normalization

```
Node features:    x_norm      = (x_raw - node_mean) / node_std           -> [N, input_var]
Target deltas:    target_norm = (target_delta - delta_mean) / delta_std   -> [N, output_var]
Edge features:    edge_norm   = (edge_raw - edge_mean) / edge_std        -> [2M, 4]
```

Normalization stats are computed from the **entire training dataset** during initialization:
- **Node stats**: Computed from `nodal_data[3:3+input_var]` across all samples and sampled timesteps
- **Edge stats**: Computed from edge features `[dx, dy, dz, distance]` using deformed positions
- **Delta stats**: Computed from actual `state_{t+1} - state_t` differences (or feature values for T=1)
- Minimum std clamped to `1e-8` to prevent division by zero

### 3. Edge Feature Computation

```
deformed_pos = reference_pos + displacement     <-- displacement = x_raw[:, :3]
relative_pos = deformed_pos[dst] - deformed_pos[src]   -> [2M, 3]
distance     = ||relative_pos||                         -> [2M, 1]
edge_features = [relative_pos, distance]                -> [2M, 4]
```

Edge features are **always computed from deformed (current) positions**, not reference positions. This ensures the graph structure reflects the actual geometry at each timestep.

### 4. Node Type Encoding (if enabled)

```
node_types = nodal_data[-1, 0, :]     <-- last feature, first timestep
one_hot    = to_one_hot(node_types)   -> [N, num_node_types]
x_final    = concat(x_norm, one_hot)  -> [N, input_var + num_node_types]
```

Node types are one-hot encoded and concatenated **after** normalization.

### 5. Model Forward Pass

```
Input:  x_final [N, input_var (+node_types)], edge_attr [2M, 4]
  -> Encoder: project to latent_dim
  -> Processor: message_passing_num x (EdgeBlock + NodeBlock) with node residuals
  -> Decoder: project to output_var (no LayerNorm)
Output: predicted_delta_norm [N, output_var]
```

### 6. Denormalization (Inference)

```
predicted_delta = predicted_delta_norm * delta_std + delta_mean
state_{t+1}     = state_t + predicted_delta
```

---

## Example Configurations

### Minimal Training Config

```
model           MeshGraphNets
mode            Train
gpu_ids         0
dataset_dir     ./dataset/my_data.h5
modelpath       ./outputs/my_model.pth
log_file_dir    train.log
input_var       4
output_var      4
edge_var        4
message_passing_num  15
Training_epochs 500
Batch_size      50
LearningR       0.0001
Latent_dim      128
num_workers     10
```

### Full Training Config (All Options)

```
model           MeshGraphNets
mode            Train
gpu_ids         0
log_file_dir    train.log
modelpath       ./outputs/my_model.pth
%
% Dataset
dataset_dir     ./dataset/my_data.h5
infer_dataset   ./infer/my_inference.h5
infer_timesteps 100
%
% Feature dimensions
input_var       4       # x_disp, y_disp, z_disp, stress
output_var      4       # x_disp, y_disp, z_disp, stress
edge_var        4       # dx, dy, dz, distance
'
% Network architecture
message_passing_num  15
Latent_dim      256
'
% Training hyperparameters
Training_epochs 500
Batch_size      10
LearningR       0.0001
num_workers     10
std_noise       0.01
residual_scale  0.2
verbose         False
monitor_gradients   False
'
% Memory optimization
use_checkpointing   False
'
% Performance optimization
use_parallel_stats  True
'
% Node types
use_node_types  False
'
% World edges
use_world_edges         False
world_radius_multiplier 1.5
world_max_num_neighbors 64
world_edge_backend      scipy_kdtree
'
% Test visualization
display_testset     True
test_batch_idx      0, 1, 2, 3
plot_feature_idx    -1
```

### Multi-GPU Training Config

```
model           MeshGraphNets
mode            Train
gpu_ids         0, 1, 2, 3     # 4-GPU DDP training
dataset_dir     ./dataset/large_data.h5
modelpath       ./outputs/best_model.pth
log_file_dir    ddp_train.log
input_var       4
output_var      4
edge_var        4
message_passing_num  15
Training_epochs 500
Batch_size      10              # Per-GPU (effective = 40)
LearningR       0.0001
Latent_dim      256
num_workers     4               # Fewer workers per GPU to avoid CPU contention
use_checkpointing   True        # Save VRAM for larger models
```

### Inference Config

```
model           MeshGraphNets
mode            Inference
gpu_ids         0
modelpath       ./outputs/my_model.pth
infer_dataset   ./infer/initial_conditions.h5
infer_timesteps 100
%
% These are needed for parsing but overridden by checkpoint during inference:
input_var       4
output_var      4
edge_var        4
message_passing_num  15
Latent_dim      256
%
% World edge backend must match training (if world edges were used):
use_world_edges     False
world_edge_backend  scipy_kdtree
world_max_num_neighbors 64
```

### Hyperparameter Sweep (Parallel Training on Multiple GPUs)

Create separate config files to run simultaneously:

**`_warpage_input/config_train1.txt`** (GPU 0, Latent_dim=256):
```
model           MeshGraphNets
mode            Train
gpu_ids         0
modelpath       ./outputs/warpage1.pth
log_file_dir    train0.log
dataset_dir     ./dataset/warpage.h5
input_var       4
output_var      4
edge_var        4
message_passing_num  15
Training_epochs 500
Batch_size      10
LearningR       0.0001
Latent_dim      256
num_workers     10
std_noise       0.01
```

**`_warpage_input/config_train2.txt`** (GPU 1, Latent_dim=128):
```
model           MeshGraphNets
mode            Train
gpu_ids         1
modelpath       ./outputs/warpage2.pth
log_file_dir    train1.log
dataset_dir     ./dataset/warpage.h5
input_var       4
output_var      4
edge_var        4
message_passing_num  15
Training_epochs 500
Batch_size      10
LearningR       0.0001
Latent_dim      128
num_workers     10
std_noise       0.01
```

Run in parallel terminals:
```bash
python MeshGraphNets_main.py --config _warpage_input/config_train1.txt
python MeshGraphNets_main.py --config _warpage_input/config_train2.txt
```

### Transient (Multi-Timestep) Training Config

For datasets with multiple timesteps (e.g., flag dynamics). Note `Batch_size 1` because each sample already generates `T-1` training pairs:

```
model           MeshGraphNets
mode            Train
gpu_ids         0
log_file_dir    train_flag_dynamic1.log
modelpath       ./outputs/flag_dynamic1.pth
dataset_dir     ./dataset/flag_dynamic.h5
infer_dataset   ./infer/flag_inference.h5
infer_timesteps 1000
input_var       4
output_var      4
edge_var        4
'
message_passing_num 15
Training_epochs	500
Batch_size	1               # Small: each sample generates T-1 pairs
LearningR	0.0001
Latent_dim	256
num_workers 10
std_noise   0.0001          # Smaller noise for transient dynamics
residual_scale  0.1
verbose     False
monitor_gradients  False
'
use_checkpointing   False
use_parallel_stats  True
use_node_types  True
use_world_edges         True
world_radius_multiplier 1.5
world_max_num_neighbors 64
world_edge_backend      scipy_kdtree
display_testset True
test_batch_idx  0, 1, 2, 3
plot_feature_idx    -2
```

---

## Internally-Assigned Config Keys

These keys are **written to the config dict by the code itself** during execution. They are not read from the config file and should not be set by the user:

| Key | Assigned by | Value | Purpose |
|-----|-------------|-------|---------|
| `num_node_types` | `single_training.py`, `distributed_training.py`, `rollout.py` | Computed from dataset | Passes node type count from dataset to model constructor |
| `log_dir` | `single_training.py` | Derived from `log_file_dir` | Directory for debug `.npz` files (passed to `training_loop.py`) |

---

## Required Keys by Mode

### Train (single or multi-GPU)

These keys **must** be present (no fallback default):

```
model, mode, gpu_ids, modelpath, dataset_dir,
input_var, output_var, edge_var,
Latent_dim, message_passing_num,
Training_epochs, Batch_size, LearningR, num_workers
```

### Inference

These keys must be present:

```
model, mode, gpu_ids, modelpath, infer_dataset,
infer_timesteps, world_edge_backend
```

Note: `world_edge_backend` is required even if `use_world_edges=False` because `rollout.py` reads it unconditionally with no default.

---

## Config File Parsing Details

The config parser (`general_modules/load_config.py`) processes each line as follows:

1. Strip whitespace
2. Skip empty lines and lines starting with `%`
3. Strip inline comments (everything after `#`)
4. Split on tab (preferred) or whitespace
5. First token = key (lowercased), remaining tokens = value
6. Value is parsed via `parse_value()`:
   - Comma-separated -> list
   - Space-separated (multi-token) -> list
   - `True`/`False` -> bool
   - Integer (no `.`) -> int
   - Float (has `.`) -> float
   - Otherwise -> lowercase string

---

## Checkpoint Contents

Checkpoints (`.pth` files) saved during training contain:

| Key | Type | Description |
|-----|------|-------------|
| `epoch` | int | Epoch number when saved |
| `model_state_dict` | OrderedDict | Model weights |
| `optimizer_state_dict` | dict | Adam optimizer state |
| `scheduler_state_dict` | dict | LR scheduler state |
| `train_loss` | float | Training loss at this epoch |
| `valid_loss` | float | Best validation loss so far |
| `normalization` | dict | Z-score normalization statistics |
| `model_config` | dict | Model architecture parameters |

### `normalization` dict

| Key | Shape | Description |
|-----|-------|-------------|
| `node_mean` | `[input_var]` | Per-feature mean of node features |
| `node_std` | `[input_var]` | Per-feature std of node features (min 1e-8) |
| `edge_mean` | `[4]` | Per-feature mean of edge features |
| `edge_std` | `[4]` | Per-feature std of edge features (min 1e-8) |
| `delta_mean` | `[output_var]` | Per-feature mean of target deltas |
| `delta_std` | `[output_var]` | Per-feature std of target deltas (min 1e-8) |
| `node_type_to_idx` | dict | *(optional)* Node type -> contiguous index mapping |
| `num_node_types` | int | *(optional)* Number of unique node types |
| `world_edge_radius` | float | *(optional)* Computed world edge radius |

### `model_config` dict

| Key | Type | Description |
|-----|------|-------------|
| `input_var` | int | Node input features |
| `output_var` | int | Node output features |
| `edge_var` | int | Edge features |
| `latent_dim` | int | MLP hidden dimension |
| `message_passing_num` | int | Number of message passing blocks |
| `use_node_types` | bool | Whether node types were used |
| `num_node_types` | int | Number of node types (0 if not used) |
| `use_world_edges` | bool | Whether world edges were used |
| `use_checkpointing` | bool | Whether gradient checkpointing was used |

---

## Output Directory Structure

### Training

```
outputs/
+-- <log_file_dir>              # Training log (e.g., train0.log)
+-- <modelpath>                 # Best checkpoint (single GPU)
+-- best_model.pth              # Best checkpoint (multi-GPU DDP)
+-- debug_epoch*.npz            # Debug data (from epoch 5+)
+-- test/
    +-- <gpu_ids>/
        +-- <epoch>/
            +-- sample{id}_t{time}.h5    # Test predictions (HDF5)
            +-- ...
```

### Inference

```
<inference_output_dir>/         # Default: outputs/rollout/
+-- rollout_sample{id}_steps{N}.h5
+-- rollout_sample{id}_steps{N}.h5
+-- ...
```

### Training Log Format

Header includes timestamp and full config file contents. Each epoch line:
```
Elapsed time: <seconds>s Epoch <N> Train Loss: <X.XXXXe-XX> Valid Loss: <X.XXXXe-XX> LR: <X.XXXXe-XX>
```

---

## Troubleshooting

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `RuntimeError: CUDA out of memory` | Insufficient GPU VRAM | Reduce `Batch_size`, enable `use_checkpointing`, reduce `Latent_dim`, or use multi-GPU |
| `FileNotFoundError: Model checkpoint not found` | Wrong `modelpath` | Check path exists |
| `KeyError: 'normalization'` | Old checkpoint format | Re-train with current code |
| `ValueError: HDF5 file missing 'data' group` | Invalid dataset | Check HDF5 follows structure above |
| Loss is `nan` | LR too high or data issues | Reduce `LearningR` to 1e-5, check data for NaN/Inf |
| Loss not decreasing | LR too high or insufficient capacity | Reduce `LearningR` by 10x, increase `message_passing_num` or `Latent_dim` |
| `Address already in use` (DDP) | Port 12355 occupied | Wait or kill lingering processes |
| Near-zero delta std warning | Constant target features | Check dataset -- features may not vary between timesteps |
| `AttributeError: 'NoneType' has no attribute 'lower'` during inference | `world_edge_backend` missing from config | Add `world_edge_backend scipy_kdtree` to the inference config |
| `torch.save` error with `None` path | `modelpath` missing from config | Add `modelpath ./outputs/my_model.pth` to the config |

### VRAM Usage Estimates

Approximate VRAM for a single sample with ~10k nodes:

| Latent_dim | MP Blocks | Checkpointing | Approx. VRAM |
|------------|-----------|---------------|--------------|
| 128 | 15 | Off | ~2-4 GB |
| 256 | 15 | Off | ~4-8 GB |
| 256 | 15 | On | ~2-3 GB |
| 512 | 15 | Off | ~8-16 GB |
| 512 | 30 | On | ~6-10 GB |

Multiply by `Batch_size` for total VRAM.

### Hyperparameter Tuning Order

Tune in this order (most to least impactful):

1. **LearningR**: Start with 1e-4, try 1e-3 and 1e-5
2. **Latent_dim**: Start with 128, try 256 if underfitting
3. **Batch_size**: Largest that fits in VRAM
4. **message_passing_num**: Start with 15, increase if loss plateaus
5. **std_noise**: Start with 0.01, reduce if loss is noisy
6. **Training_epochs**: Monitor validation loss for convergence

---

*Last updated: 2026-02-26*
