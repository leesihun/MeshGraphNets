# MeshGraphNets Configuration & Execution Guide

**Purpose**: Complete reference for running MeshGraphNets training with hyperparameter optimization. Designed for automated LLM agent workflows.

---

## Quick Start

```bash
# 1. Edit config.txt (see template below)
# 2. Run training
python MeshGraphNets_main.py

# 3. Check results
cat outputs/<gpu_ids>/<log_file_dir>
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

### config.txt Template

```
model   MeshGraphNets
mode    Train  # Train / Inference
gpu_ids 0      # -1 for CPU, GPU ids for multi-GPU training
log_file_dir    train0.log
%   Common params
%   Dim1		1000 # number of parameters
%   Dim2		1   # number of timesteps
%   Dim3		95008 # num nodes, unused in GNN
input_var   4   # number of input variables: x_disp, y_disp, z_disp, stress (excluding node types)
output_var  4   # number of output variables: x_disp, y_disp, z_disp, stress (excluding node types)
edge_var    4   # dx, dy, dz, disp
'
%   Network parameters
dataset_dir ./dataset/deforming_plate.h5
norm_min    -0.7  # Normalization range minimum
norm_max    0.7   # Normalization range maximum
message_passing_num 15
Training_epochs	50
Batch_size	10
LearningR	0.001
Latent_dim	128	# MeshGraphNets latent dimension
num_workers 10
std_noise   0.0000000000000000000001
verbose     False
'
% Memory Optimization
use_checkpointing   False
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
plot_feature_idx    -2  # Feature index to visualize in plots (-1 = last feature, i.e., stress)
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
| **Batch_size** | int | 1 | 1-32 | Batch size per GPU |

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
| dataset_dir | str | ./dataset/dataset.h5 | Path to HDF5 dataset |
| Dim1 | int | 2138 | Number of samples (informational) |
| Dim2 | int | 1 | Timesteps per sample |
| input_var | int | 4 | Node features count |
| output_var | int | 4 | Output features count |
| edge_var | int | 4 | Edge features (fixed: dx, dy, dz, distance) |

### Training Parameters

| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|
| Training_epochs | int | 2002 | 1-10000 | Total epochs |
| num_workers | int | 2 | 0-16 | DataLoader workers |
| std_noise | float | 0.01 | 0-0.1 | Input augmentation noise std |
| verbose | bool | False | - | Verbose logging |
| norm_min | float | -0.7 | -1.0 to 0 | Normalization range min |
| norm_max | float | 0.7 | 0 to 1.0 | Normalization range max |

### Feature Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| use_checkpointing | bool | True | Gradient checkpointing (reduces VRAM ~50%) |
| use_node_types | bool | False | Add one-hot node types to features |
| use_world_edges | bool | False | Enable radius-based collision edges |
| world_radius_multiplier | float | 1.5 | World edge radius multiplier |

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

For each hyperparameter configuration:

1. **Modify** `config.txt` using `write_config()` with new hyperparameters
2. **Execute** `python MeshGraphNets_main.py` and wait for completion
3. **Parse** `outputs/<gpu_ids>/train.log` to extract final validation loss
4. **Track** best configuration based on lowest validation loss
5. **Iterate** with refined hyperparameter search

**Primary optimization metric**: Validation loss (lower is better)

Best model checkpoint auto-saved to: `outputs/best_model.pth`

---

## Troubleshooting

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| **OOM Error** | `RuntimeError: CUDA out of memory` | 1. Set `Batch_size=1`<br>2. Enable `use_checkpointing=True`<br>3. Reduce `Latent_dim` to 64/128<br>4. Use multi-GPU |
| **Loss Not Decreasing** | Loss constant or increasing | 1. Reduce `LearningR` by 10× (e.g., 1e-4 → 1e-5)<br>2. Increase `message_passing_num`<br>3. Verify dataset path |
| **NaN Loss** | Loss becomes `nan` | 1. Reduce `LearningR` to 1e-5<br>2. Reduce `std_noise`<br>3. Check dataset normalization |
| **Slow Training** | Epochs take too long | 1. Adjust `num_workers` (2-8)<br>2. Use multi-GPU<br>3. Reduce dataset size for debugging |

---

