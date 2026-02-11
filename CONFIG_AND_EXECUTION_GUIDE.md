# MeshGraphNets Configuration & Execution Guide

**Purpose**: Complete reference for running MeshGraphNets training and inference. Designed for automated LLM agent workflows.

---

## Quick Start

### Training
```bash
# 1. Edit config.txt with mode=Train
# 2. Run training
python MeshGraphNets_main.py

# 3. Check results
cat outputs/<gpu_ids>/<log_file_dir>
```

### Inference
```bash
# 1. Edit config.txt with mode=Inference and modelpath pointing to trained model
# 2. Run inference
python MeshGraphNets_main.py

# 3. Check results (output saved to inference_output_dir)
```

---

## Configuration File Requirements

**CRITICAL**: Configuration rules are strict and non-negotiable:

- **File name**: Must be `config.txt` located with MeshGraphNets_main.py (exact name, case-sensitive, and location-sensitive)
- **Location**: Same directory as `MeshGraphNets_main.py`
- **Format**: Plain text with custom syntax
  - Lines starting with `%` are comments
  - `'` marks section separators
  - Keys are case-insensitive (converted to lowercase internally)

**NO OTHER FILE NAMES OR LOCATIONS ARE ALLOWED**

### config.txt Template (Training + Inference)

```
model   MeshGraphNets
mode    Inference  # Train / Inference
gpu_ids 0      # -1 for CPU, GPU ids for multi-GPU training
log_file_dir    train0.log
%   Datasets
dataset_dir ./dataset/flag_simple.h5
infer_dataset   ./infer/flag_inference.h5
infer_timesteps 1000
modelpath   ./output/flag_simple1.pth
%   Common params
input_var   4   # number of input variables: x_disp, y_disp, z_disp, stress (excluding node types)
output_var  4   # number of output variables: x_disp, y_disp, z_disp, stress (excluding node types)
edge_var    4   # dx, dy, dz, disp
'
%   Network parameters
message_passing_num 15
Training_epochs	500
Batch_size	50
LearningR	0.0001
Latent_dim	128	# MeshGraphNets latent dimension
num_workers 10
std_noise   0.001
residual_scale  0.1  # Scale factor for residual connections (0.1 = 10% of update added to current state)
verbose     False
monitor_gradients  False
'
% Memory Optimization
use_checkpointing   False
'
% Performance Optimization
use_parallel_stats  True    # Enable parallel processing for computing dataset statistics (speeds up initialization for large datasets, requires >=100 samples)
'
% Node Type Parameters
use_node_types  True    # Add one-hot encoded node types to node features
'
% World Edge Parameters
use_world_edges         True
world_radius_multiplier 1.5     # r_world = multiplier * min_mesh_edge_length (auto-computed)
world_max_num_neighbors 64      # Max neighbors per node in world edge radius query (prevents edge explosion)
world_edge_backend      scipy_kdtree   # Backend: torch_cluster (GPU, fast) or scipy_kdtree (CPU, fallback)
% Test set control
display_testset True
test_batch_idx  0, 1, 2, 3
plot_feature_idx    -2  # Feature index to visualize in plots (-2 = second to last feature, -1 = last feature)
```

---

## Parameter Reference

### Hyperparameters for Optimization

**These are the ONLY parameters to tune for hyperparameter search:**

| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|
| **LearningR** | float | 0.0001 | 1e-6 to 1e-2 | Learning rate (AdamW optimizer) |
| **Latent_dim** | int | 128 | 32-512 | Hidden dimension for all MLPs |
| **message_passing_num** | int | 15 | 1-30 | Number of Graph Network blocks |
| **Batch_size** | int | 50 | 1-128 | Batch size per GPU |
| **residual_scale** | float | 0.1 | 0.0-1.0 | Scale factor for residual connections (0.1 = 10% of update) |

**DO NOT change other parameters during hyperparameter optimization.**

### Core Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| model | str | Model architecture to use for training or inference |
| mode | str | Execution mode: Train for training, Inference for prediction |
| gpu_ids | int/list | Device configuration: -1 for CPU execution, single integer for single GPU (e.g., 0 for GPU 0), or comma-separated list for multi-GPU DDP training (e.g., [0,1,2,3] for 4 GPUs) |
| log_file_dir | str | Filename for training logs (saved to outputs/<gpu_ids>/ directory) |

### Dataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| dataset_dir | str | ./dataset/dataset.h5 | Path to HDF5 training dataset |
| infer_dataset | str | ./infer/inference.h5 | Path to HDF5 inference dataset (used in Inference mode) |
| infer_timesteps | int | 1000 | Number of rollout timesteps for inference predictions |
| modelpath | str | ./output/model.pth | Path to trained model checkpoint (required for Inference mode) |
| input_var | int | 4 | Node features count |
| output_var | int | 4 | Output features count |
| edge_var | int | 4 | Edge features (fixed: dx, dy, dz, distance) |

### Training Parameters

| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|
| Training_epochs | int | 500 | 1-10000 | Total epochs |
| num_workers | int | 10 | 0-16 | DataLoader workers |
| std_noise | float | 0.001 | 0-0.1 | Input augmentation noise std |
| verbose | bool | False | - | Verbose logging |
| monitor_gradients | bool | False | - | Enable gradient monitoring for debugging |
| norm_min | float | -0.7 | -1.0 to 0 | Normalization range min (optional, commented by default) |
| norm_max | float | 0.7 | 0 to 1.0 | Normalization range max (optional, commented by default) |

### Feature Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| use_checkpointing | bool | False | Gradient checkpointing (reduces VRAM ~50%) |
| use_parallel_stats | bool | True | Parallel processing for dataset statistics computation (3-5x speedup for >=100 samples, uses 80% of CPU cores) |
| use_node_types | bool | True | Add one-hot node types to features |
| use_world_edges | bool | True | Enable radius-based collision edges |
| world_radius_multiplier | float | 1.5 | World edge radius multiplier (r_world = multiplier × min_mesh_edge_length) |
| world_max_num_neighbors | int | 64 | Max neighbors per node in world edge radius query (prevents edge explosion) |
| world_edge_backend | str | scipy_kdtree | Backend for world edge computation: torch_cluster (GPU, fast) or scipy_kdtree (CPU, fallback) |

### Test Visualization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| display_testset | bool | True | Show test predictions during training |
| test_batch_idx | list | 0, 1, 2, 3 | Test batch indices to display |

---

## Execution Modes

### Single GPU
```
gpu_ids 0
```
Uses [training_profiles/single_training.py](training_profiles/single_training.py)

### Multi-GPU (DDP)
```
gpu_ids 0, 1, 2, 3
```
Uses [training_profiles/distributed_training.py](training_profiles/distributed_training.py)
- Effective batch size = `Batch_size × num_gpus`
- NCCL backend for GPU communication
- Each GPU gets full dataset with DistributedSampler

### CPU (Not Recommended)
```
gpu_ids -1
```

---

## Execution Modes (Training vs Inference)

### Training Mode
```
mode    Train
dataset_dir ./dataset/flag_simple.h5
```
- Uses [training_profiles/single_training.py](training_profiles/single_training.py) (single GPU)
- Uses [training_profiles/distributed_training.py](training_profiles/distributed_training.py) (multi-GPU)
- Saves checkpoints and logs to `outputs/<gpu_ids>/` directory
- Best model saved to `outputs/best_model.pth`
- Normalization statistics saved in checkpoint for inference

### Inference Mode
```
mode    Inference
modelpath   ./output/flag_simple1.pth
infer_dataset   ./infer/flag_inference.h5
infer_timesteps 1000
```
- Uses [inference_profiles/rollout.py](inference_profiles/rollout.py)
- Loads pre-trained model from `modelpath`
- Performs autoregressive rollout for `infer_timesteps` timesteps
- Loads inference data from `infer_dataset`
- Loads normalization stats from trained model checkpoint
- Output predictions saved to `inference_output_dir`

**Required for Inference:**
- Pre-trained model checkpoint with saved normalization statistics
- Separate inference dataset in HDF5 format
- `modelpath` parameter pointing to the checkpoint

---

## Hyperparameter Optimization

### Recommended Search Strategy

**Phase 1: Coarse LatinHyperCube Search (10 epochs, ~4, 8 variations)**

**Phase 2: Fine-Tune Best Config (20 epochs, ~4, 8 variations)**

### Programmatic Config Modification required

### Log Parsing

Training logs are saved to `outputs/<gpu_ids>/<log_file_dir>` with format:
```
Training epoch log file
Time: 2026-01-30 10:00:00
Log file absolute path: /path/to/outputs/0/train.log
<config.txt contents>
Elapsed time: 123.45s Epoch 0 Train Loss: 1.23e-02 Valid Loss: 1.45e-02 LR: 1.0000e-04
Elapsed time: 246.78s Epoch 1 Train Loss: 9.87e-03 Valid Loss: 1.23e-02 LR: 1.0000e-04
...

---

## Automated Workflow for LLM Agents

### Training Workflow

For each hyperparameter configuration:

1. **Set mode**: `mode Train` in `config.txt`
2. **Modify** `config.txt` using `write_config()` with new hyperparameters
3. **Execute** `python MeshGraphNets_main.py` and wait for completion
4. **Parse** `outputs/<gpu_ids>/train.log` to extract final validation loss
5. **Track** best configuration based on lowest validation loss
6. **Iterate** with refined hyperparameter search

**Primary optimization metric**: Validation loss (lower is better)

Best model checkpoint auto-saved to: `outputs/best_model.pth`

### Inference Workflow

Once a trained model is available:

1. **Set mode**: `mode Inference` in `config.txt`
2. **Set modelpath**: Point to trained checkpoint (e.g., `./output/flag_simple1.pth`)
3. **Set infer_dataset**: Point to inference dataset (e.g., `./infer/flag_inference.h5`)
4. **Set infer_timesteps**: Number of rollout steps (e.g., 1000)
5. **Execute** `python MeshGraphNets_main.py`
6. **Retrieve output**: Predictions saved to `inference_output_dir` (or stdout if specified)

---

## Troubleshooting

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| **OOM Error** | `RuntimeError: CUDA out of memory` | 1. Set `Batch_size=1`<br>2. Enable `use_checkpointing=True`<br>3. Reduce `Latent_dim` to 64/128<br>4. Use multi-GPU |
| **Loss Not Decreasing** | Loss constant or increasing | 1. Reduce `LearningR` by 10× (e.g., 1e-4 → 1e-5)<br>2. Increase `message_passing_num`<br>3. Verify dataset path |
| **NaN Loss** | Loss becomes `nan` | 1. Reduce `LearningR` to 1e-5<br>2. Reduce `std_noise`<br>3. Check dataset normalization |
| **Slow Training** | Epochs take too long | 1. Adjust `num_workers` (2-8)<br>2. Use multi-GPU<br>3. Reduce dataset size for debugging |

---

