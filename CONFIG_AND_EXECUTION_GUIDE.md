# Configuration and Execution Guide

Complete reference for all configuration parameters and execution modes in MeshGraphNets.

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
- [Example Configurations](#example-configurations)
- [Config File Parsing Details](#config-file-parsing-details)
- [Checkpoint Contents](#checkpoint-contents)
- [Output Directory Structure](#output-directory-structure)
- [Troubleshooting](#troubleshooting)

---

## Config File Format

Configuration is stored in a plain-text file (default: `config.txt`) in the repository root.

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

### Command-Line Override

The config file path can be overridden via command-line:

```bash
python MeshGraphNets_main.py --config my_custom_config.txt
```

Default is `config.txt` in the current working directory.

---

## Parameter Reference

### General Parameters

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `model` | str | - | Yes | Model architecture name. Currently only `MeshGraphNets` is supported. |
| `mode` | str | - | Yes | Execution mode: `Train` or `Inference`. Case-insensitive (stored as lowercase internally). |
| `gpu_ids` | int or list | - | Yes | GPU device(s) to use. Single int for single-GPU, comma-separated list for multi-GPU DDP, `-1` for CPU. |
| `log_file_dir` | str | - | No | Log filename relative to `outputs/`. Training logs are written to `outputs/<log_file_dir>`. The parent directory is auto-created. |

**Notes:**
- `gpu_ids` determines the execution pipeline automatically:
  - Single value (e.g., `0`): Routes to `single_training.py`
  - Multiple values (e.g., `0, 1, 2, 3`): Routes to `distributed_training.py` with DDP
  - `-1`: Routes to `single_training.py` on CPU
- `log_file_dir` is the full relative path from `outputs/`, e.g., `train0.log` writes to `outputs/train0.log`, `gpu0/train.log` writes to `outputs/gpu0/train.log`

---

### Dataset Parameters

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `dataset_dir` | str | - | Train | Path to HDF5 training dataset file. |
| `infer_dataset` | str | - | Inference | Path to HDF5 inference dataset file. |

**Notes:**
- Paths can be relative (to working directory) or absolute
- HDF5 dataset must follow the structure defined in [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md)
- The dataset is automatically split 80/10/10 (train/val/test) during training

---

### Network Architecture Parameters

These parameters define the model structure. They are saved in the checkpoint and restored during inference.

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `input_var` | int | - | 1+ | Number of physical node input features (excluding node types). Typically 4: `[x_disp, y_disp, z_disp, stress]`. |
| `output_var` | int | - | 1+ | Number of node output features. Typically 4: `[x_disp, y_disp, z_disp, stress]`. |
| `edge_var` | int | - | 1+ | Number of edge features. Always 4: `[dx, dy, dz, distance]`. |
| `Latent_dim` | int | 128 | 32-512 | Hidden dimension of all MLPs in the encoder, processor, and decoder. Affects VRAM usage **quadratically** (MLP weights scale as latent_dim^2). |
| `message_passing_num` | int | 15 | 1-30 | Number of message passing iterations in the processor. More iterations = larger receptive field but more VRAM (linear scaling). |

**Notes:**
- The actual model input dimension is `input_var + num_node_types` when `use_node_types=True`
- All MLPs use 2 hidden layers with ReLU activation and LayerNorm (except the decoder, which has no LayerNorm)
- Each message passing block contains an EdgeBlock and a NodeBlock (or HybridNodeBlock if world edges are enabled)
- Residual connections are applied to **node features only** (not edges), matching the NVIDIA PhysicsNeMo implementation

---

### Training Hyperparameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `Training_epochs` | int | - | 1+ | Total number of training epochs. |
| `Batch_size` | int | - | 1-128 | Per-GPU batch size. For multi-GPU DDP, effective batch size = `Batch_size * num_GPUs`. |
| `LearningR` | float | 0.0001 | 1e-6 to 1e-2 | Initial learning rate for Adam optimizer. **Most sensitive hyperparameter.** |
| `num_workers` | int | 10 | 0-16 | Number of DataLoader worker processes. Set to 0 for debugging (single-process loading). |
| `std_noise` | float | 0.001 | 0-0.1 | Standard deviation of Gaussian noise added to input features during training. Set to 0 to disable. |
| `residual_scale` | float | 0.1 | 0-1.0 | Scale factor for residual connections. Currently defined in config but **not used in the model code** (residual connections use scale=1.0). Reserved for future use. |
| `monitor_gradients` | bool | True | - | Enable gradient norm monitoring during training. Logs per-epoch average gradient norms and warnings for vanishing/exploding gradients. |

**Optimizer Details:**
- **Optimizer**: Adam (torch.optim.Adam)
- **LR Scheduler (Single GPU)**: ExponentialLR with gamma=0.995 (decays every epoch)
- **LR Scheduler (Multi-GPU DDP)**: ReduceLROnPlateau (factor=0.5, patience=2, min_lr=1e-8)
- **Gradient Clipping**: max_norm=5.0 (applied after gradient computation)
- **Loss Function**: MSE on normalized deltas per feature
- **Weight Initialization**: Kaiming/He uniform (nonlinearity='relu')
- **Data Split**: 80% train, 10% validation, 10% test (fixed seed=42)

---

### Node Type Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_node_types` | bool | False | Enable one-hot encoding of node types from HDF5 metadata. Node types are read from the **last feature** (index -1) of `nodal_data`. |

**Notes:**
- When enabled, unique node types are auto-discovered from the first 10 samples in the dataset
- Node types are mapped to contiguous indices (e.g., `{0: 0, 1: 1, 3: 2}`)
- One-hot vectors are concatenated **after** z-score normalization of physical features
- The model input dimension becomes `input_var + num_node_types`
- Node type mapping (`node_type_to_idx`) and count (`num_node_types`) are saved in the checkpoint for inference

---

### World Edge Parameters

World edges provide radius-based collision detection between non-adjacent nodes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_world_edges` | bool | False | Enable world edge computation. Adds radius-based proximity edges between nodes that are not connected by mesh edges. |
| `world_radius_multiplier` | float | 1.5 | Collision detection radius = multiplier * min_mesh_edge_length. The min edge length is auto-computed from the first 10 dataset samples. |
| `world_max_num_neighbors` | int | 64 | Maximum number of neighbors per node in the radius query. Prevents edge count explosion in dense regions. |
| `world_edge_backend` | str | `torch_cluster` | Backend for radius queries: `torch_cluster` (GPU-accelerated, 5-10x faster) or `scipy_kdtree` (CPU fallback, always available). Falls back to `scipy_kdtree` if `torch_cluster` is not installed. |

**Notes:**
- World edges are **filtered** to exclude edges that already exist in the mesh topology
- World edge features use the same `[dx, dy, dz, distance]` format as mesh edges
- World edge features are normalized using the **same statistics** as mesh edges
- When world edges are enabled, the processor uses `HybridNodeBlock` which aggregates from both mesh and world edges separately before combining
- The computed `world_edge_radius` value is saved in the checkpoint for inference
- See [docs/WORLD_EDGES_DOCUMENTATION.md](docs/WORLD_EDGES_DOCUMENTATION.md) for full details

---

### Memory Optimization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_checkpointing` | bool | False | Enable gradient checkpointing for message passing blocks. Trades 20-30% extra compute time for 60-70% VRAM reduction. |

**Notes:**
- Gradient checkpointing only activates during training (no effect during inference/eval)
- When enabled, intermediate activations in the processor are recomputed during the backward pass instead of being stored
- Particularly useful for deep networks (high `message_passing_num`) or large latent dimensions
- See [docs/VRAM_OPTIMIZATION_PLAN.md](docs/VRAM_OPTIMIZATION_PLAN.md) for detailed memory optimization strategies

---

### Performance Optimization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_parallel_stats` | bool | True | Enable multiprocessing for dataset normalization statistics computation. Uses ~45% of available CPU cores (max 64 workers). |

**Notes:**
- Automatically disabled for datasets with fewer than 10 samples
- Falls back to serial processing if multiprocessing fails
- Each worker opens its own HDF5 file handle (read-only) to avoid file locking issues
- For datasets with many timesteps, statistics are computed from a subsample of up to 500 timesteps per sample to limit memory usage

---

### Diagnostics and Visualization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | bool | False | Enable detailed diagnostics: per-feature loss breakdowns, CUDA memory tracking per batch, per-layer gradient statistics. |
| `monitor_gradients` | bool | True | Track gradient norms during training. Logs average gradient norm per epoch and warns about vanishing (<1e-6) or exploding (>100) gradients. |
| `display_testset` | bool | True | Generate visualizations when evaluating on the test set (every 10 epochs). Set to `False` to skip rendering for faster training. |
| `test_batch_idx` | list | `0, 1, 2, 3` | Which test batch indices to visualize and save to HDF5 during test evaluation. |
| `plot_feature_idx` | int | -1 | Feature index to visualize in test plots. `-1` = last feature (typically stress), `-2` = second-to-last, or a positive index for a specific feature. |

**Notes:**
- Test evaluation runs every 10 epochs, saving results to `outputs/test/<gpu_ids>/<epoch>/`
- Test outputs include both normalized and denormalized predictions vs. ground truth
- When `verbose=True`, per-feature MSE loss is printed for train, validation, and test sets

---

### Inference Parameters

These parameters are only used when `mode=Inference`.

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `modelpath` | str | - | Yes | Path to the trained model checkpoint (`.pth` file). Must contain `model_state_dict`, `normalization`, and `model_config` keys. |
| `infer_dataset` | str | - | Yes | Path to HDF5 dataset with initial conditions for rollout. |
| `infer_timesteps` | int | - | Yes* | Number of autoregressive rollout steps to predict. If `None` and dataset has multiple timesteps, defaults to `num_timesteps - 1`. |
| `inference_output_dir` | str | `outputs/rollout` | No | Directory for rollout output HDF5 files. Auto-created if it doesn't exist. |

**Notes:**
- During inference, model architecture parameters (`input_var`, `output_var`, `latent_dim`, `message_passing_num`, etc.) are **overridden** by values stored in the checkpoint's `model_config`
- Normalization statistics (means/stds) are loaded from the checkpoint's `normalization` dict
- All samples in `infer_dataset` are processed sequentially
- Output files are named `rollout_sample{id}_steps{N}.h5`
- Rollout always starts from timestep 0 of each sample

---

## Execution Modes

### Training (Single GPU)

```bash
# config.txt
mode        Train
gpu_ids     0
dataset_dir ./dataset/my_data.h5
```

```bash
python MeshGraphNets_main.py
```

**Pipeline**: `MeshGraphNets_main.py` -> `single_training.py` -> `training_loop.py`

**Details:**
- Uses `torch.optim.lr_scheduler.ExponentialLR` (gamma=0.995)
- Saves checkpoint to path specified by `modelpath`
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

**Details:**
- Uses `torch.distributed` with NCCL backend (or Gloo on CPU)
- `DistributedSampler` ensures each GPU sees different data
- Effective batch size = `Batch_size * num_GPUs`
- Uses `torch.optim.lr_scheduler.ReduceLROnPlateau` (factor=0.5, patience=2)
- Only rank 0 saves checkpoints and writes logs
- Checkpoint saved to `outputs/best_model.pth`
- Communication: localhost:12355

### Training (CPU)

```bash
# config.txt
mode        Train
gpu_ids     -1
dataset_dir ./dataset/my_data.h5
```

```bash
python MeshGraphNets_main.py
```

**Pipeline**: Same as single GPU, but on CPU. Useful for debugging.

### Inference (Autoregressive Rollout)

```bash
# config.txt
mode            Inference
gpu_ids         0
modelpath       ./outputs/best_model.pth
infer_dataset   ./infer/my_initial_conditions.h5
infer_timesteps 1000
```

```bash
python MeshGraphNets_main.py
```

**Pipeline**: `MeshGraphNets_main.py` -> `rollout.py`

**Details:**
1. Loads checkpoint (model weights + normalization stats + model config)
2. Overrides config with model config from checkpoint
3. For each sample in `infer_dataset`:
   - Extracts initial state at timestep 0
   - Runs autoregressive loop for `infer_timesteps` steps
   - Each step: normalize -> build graph -> forward pass -> denormalize delta -> update state
4. Saves predicted trajectory to `<inference_output_dir>/rollout_sample{id}_steps{N}.h5`

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
infer_timesteps 1000
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
Batch_size      50
LearningR       0.0001
num_workers     10
std_noise       0.001
residual_scale  0.1
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
use_node_types  True
'
% World edges
use_world_edges         True
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
use_node_types      True
use_world_edges     True
```

### Inference Config

```
model           MeshGraphNets
mode            Inference
gpu_ids         0
modelpath       ./outputs/my_model.pth
infer_dataset   ./infer/initial_conditions.h5
infer_timesteps 1000
%
% These are needed for parsing but overridden by checkpoint during inference:
input_var       4
output_var      4
edge_var        4
message_passing_num  15
Latent_dim      256
use_node_types      True
use_world_edges     True
world_edge_backend  scipy_kdtree
world_max_num_neighbors 64
```

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

**Important implications:**
- `LearningR 0.0001` is stored as `config['learningr'] = 0.0001` (float)
- `gpu_ids 0, 1, 2, 3` is stored as `config['gpu_ids'] = [0, 1, 2, 3]` (list of int)
- `gpu_ids 0` is stored as `config['gpu_ids'] = 0` (int, not list)
- `verbose True` is stored as `config['verbose'] = True` (bool)
- `mode Train` is stored as `config['mode'] = 'train'` (lowercase string)
- String values with spaces may be parsed as lists: `my_param hello world` -> `['hello', 'world']`

---

## Checkpoint Contents

Checkpoints (`.pth` files) saved during training contain the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `epoch` | int | Epoch number when this checkpoint was saved |
| `model_state_dict` | OrderedDict | Model weights (from `model.state_dict()` or `model.module.state_dict()` for DDP) |
| `optimizer_state_dict` | dict | Adam optimizer state |
| `scheduler_state_dict` | dict | LR scheduler state |
| `train_loss` | float | Training loss at this epoch |
| `valid_loss` | float | Validation loss at this epoch (best so far) |
| `normalization` | dict | Z-score normalization statistics (see below) |
| `model_config` | dict | Model architecture parameters (see below) |

### `normalization` dict

| Key | Shape | Description |
|-----|-------|-------------|
| `node_mean` | `[input_var]` | Per-feature mean of node features |
| `node_std` | `[input_var]` | Per-feature std of node features (min 1e-8) |
| `edge_mean` | `[4]` | Per-feature mean of edge features |
| `edge_std` | `[4]` | Per-feature std of edge features (min 1e-8) |
| `delta_mean` | `[output_var]` | Per-feature mean of target deltas |
| `delta_std` | `[output_var]` | Per-feature std of target deltas (min 1e-8) |
| `node_type_to_idx` | dict | *(optional)* Node type value -> contiguous index mapping |
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
├── <log_file_dir>              # Training log (e.g., train0.log)
├── best_model.pth              # Best checkpoint (multi-GPU DDP)
├── <modelpath>                 # Best checkpoint (single GPU, path from config)
└── test/
    └── <gpu_ids>/
        └── <epoch>/
            ├── sample0_t5.h5   # Test predictions (HDF5)
            ├── sample1_t12.h5
            └── ...
```

### Inference

```
<inference_output_dir>/         # Default: outputs/rollout/
├── rollout_sample0_steps1000.h5
├── rollout_sample1_steps1000.h5
└── ...
```

### Training Log Format

Each line in the log file follows this format:
```
Elapsed time: <seconds>s Epoch <N> Train Loss: <X.XXXXe-XX> Valid Loss: <X.XXXXe-XX> LR: <X.XXXXe-XX>
```

The log file header includes the timestamp and the full contents of the config file used.

---

## Troubleshooting

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `KeyError: 'batch_size'` | Config key is case-sensitive in code but `Batch_size` gets lowercased to `batch_size` | Ensure your config has `Batch_size` (the parser lowercases it to `batch_size`) |
| `RuntimeError: CUDA out of memory` | Insufficient GPU VRAM | Reduce `Batch_size`, enable `use_checkpointing`, reduce `Latent_dim`, or use multi-GPU |
| `FileNotFoundError: Model checkpoint not found` | Wrong `modelpath` in inference mode | Check the path exists and is correct |
| `KeyError: 'normalization'` | Old checkpoint missing normalization stats | Re-train with the current code, which saves normalization stats |
| `ValueError: HDF5 file missing 'data' group` | Invalid dataset format | Check your HDF5 file follows [DATASET_FORMAT.md](dataset/DATASET_FORMAT.md) |
| Loss is `nan` | Learning rate too high or data issues | Reduce `LearningR` to 1e-5, check data for NaN/Inf values |
| Loss not decreasing | Learning rate too high, insufficient model capacity, or data mismatch | Reduce `LearningR` by 10x, increase `message_passing_num` or `Latent_dim` |
| `Address already in use` (DDP) | Previous DDP training left port 12355 occupied | Wait for cleanup or kill lingering processes |

### VRAM Usage Estimates

Approximate VRAM usage for a single sample with ~10k nodes:

| Latent_dim | MP Blocks | Checkpointing | Approx. VRAM |
|------------|-----------|---------------|--------------|
| 128 | 15 | Off | ~2-4 GB |
| 256 | 15 | Off | ~4-8 GB |
| 256 | 15 | On | ~2-3 GB |
| 512 | 15 | Off | ~8-16 GB |
| 512 | 30 | On | ~6-10 GB |

Actual usage depends on batch size, number of nodes/edges, and world edges. Multiply by `Batch_size` for total VRAM.

### Hyperparameter Tuning Order

For best results, tune parameters in this order (most to least impactful):

1. **LearningR**: Start with 1e-4, try 1e-3 and 1e-5
2. **Latent_dim**: Start with 128, try 256 if underfitting
3. **Batch_size**: Largest that fits in VRAM
4. **message_passing_num**: Start with 15, increase if loss plateaus
5. **std_noise**: Start with 0.001, reduce if loss is noisy
6. **Training_epochs**: Monitor validation loss for convergence

---

*Last updated: 2026-02-18*
