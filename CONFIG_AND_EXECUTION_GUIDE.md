# MeshGraphNets Configuration and Execution Guide

**Purpose**: This document provides complete instructions for running MeshGraphNets training with different hyperparameter configurations. It is designed for automated hyperparameter optimization by an LLM agent.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration File Format](#configuration-file-format)
3. [Complete Parameter Reference](#complete-parameter-reference)
4. [Running the Script](#running-the-script)
5. [Hyperparameter Optimization Guidelines](#hyperparameter-optimization-guidelines)
6. [Output and Logging](#output-and-logging)
7. [Common Issues and Solutions](#common-issues-and-solutions)

---

## Quick Start

### Minimal Steps to Run Training

1. Edit `config.txt` with desired parameters
2. Run: `python MeshGraphNets_main.py`
3. Monitor: Check `outputs/<gpu_id>/train.log` for progress

### Standard Configuration

config.txt
```
model   MeshGraphNets
mode    Train  # Train / Inference
gpu_ids 0      # -1 for CPU, GPU ids for multi-GPU training
log_file_dir    train.log
%   Common params
Dim1		2138 # number of parameters
Dim2		1   # number of timesteps
%   Dim3		95008 # num nodes, unused in GNN
input_var   4   # number of input variables: x_disp, y_disp, z_disp, stress (excluding node types)
output_var  4   # number of output variables: x_disp, y_disp, z_disp, stress (excluding node types)
edge_var    4   # dx, dy, dz, disp
'
%   Network parameters
dataset_dir ./dataset/dataset.h5
norm_min    -0.7  # Normalization range minimum
norm_max    0.7   # Normalization range maximum
message_passing_num 15
Training_epochs	2002
Batch_size	1
LearningR	0.0001
Latent_dim	128	# MeshGraphNets latent dimension
num_workers 2
std_noise   0.01
verbose     False
'
% Memory Optimization
use_checkpointing   True
'
% Node Type Parameters
use_node_types  False    # Add one-hot encoded node types to node features
'
% World Edge Parameters
use_world_edges         False
world_radius_multiplier 1.5     # r_world = multiplier * min_mesh_edge_length (auto-computed)
% Test set control
display_testset True
test_batch_idx  0, 1, 2, 3
```

---

## Configuration File Format

### Syntax Rules

The config parser (`general_modules/load_config.py`) uses a custom format:

| Syntax | Meaning |
|--------|---------|
| `key value` | Parameter assignment (tab or space separated) |
| `% comment` | Line comment (entire line ignored) |
| `# inline comment` | Inline comment (text after `#` ignored) |
| `'` | Section separator (ignored, just for readability) |
| `key value1, value2` | Comma-separated list |
| `key value1 value2` | Space-separated list (parsed as array) |

### Type Inference

The parser automatically converts values:

| Input | Parsed As | Example |
|-------|-----------|---------|
| `123` | `int` | `Batch_size 16` → `{'batch_size': 16}` |
| `0.001` | `float` | `LearningR 0.001` → `{'learningr': 0.001}` |
| `True` / `False` | `bool` | `verbose True` → `{'verbose': True}` |
| `a, b, c` | `list` | `gpu_ids 0, 1` → `{'gpu_ids': [0, 1]}` |
| Other | `str` (lowercase) | `mode Train` → `{'mode': 'train'}` |

---

## Complete Parameter Reference

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `MeshGraphNets` | Model architecture (only `MeshGraphNets` supported) |
| `mode` | str | `Train` | Execution mode: `Train` or `Inference` |
| `gpu_ids` | int or list | `0` | GPU ID(s) for training. `-1` for CPU, single int for single GPU, list for multi-GPU |
| `log_file_dir` | str | `train.log` | Log file name (saved to `outputs/<gpu_ids>/`) |

### Dataset Parameters

| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|
| `dataset_dir` | str | `./dataset/dataset.h5` | - | Path to HDF5 dataset file |
| `Dim1` | int | `2138` | - | Number of samples in dataset (informational) |
| `Dim2` | int | `1` | - | Number of timesteps per sample |
| `input_var` | int | `4` | 1-10 | Number of input node features (x_disp, y_disp, z_disp, stress) |
| `output_var` | int | `4` | 1-10 | Number of output node features |
| `edge_var` | int | `4` | - | Number of edge features (dx, dy, dz, distance) - fixed |

### Network Architecture Parameters

| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|
| `message_passing_num` | int | `15` | 1-30 | Number of Graph Network blocks in processor |
| `Latent_dim` | int | `128` | 32-512 | Hidden dimension for all MLPs |

### Training Parameters

| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|
| `Training_epochs` | int | `2002` | 1-10000 | Total training epochs |
| `Batch_size` | int | `1` | 1-32 | Batch size per GPU |
| `LearningR` | float | `0.0001` | 1e-6 to 1e-2 | Initial learning rate (AdamW) |
| `num_workers` | int | `2` | 0-16 | DataLoader worker processes |
| `std_noise` | float | `0.01` | 0 to 0.1 | Gaussian noise std for input augmentation |
| `verbose` | bool | `False` | - | Enable verbose logging |

### Normalization Parameters

| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|
| `norm_min` | float | `-0.7` | -1.0 to 0 | Min value for normalization range |
| `norm_max` | float | `0.7` | 0 to 1.0 | Max value for normalization range |

### Memory Optimization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_checkpointing` | bool | `True` | Enable gradient checkpointing to reduce VRAM usage |

### Node Type Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_node_types` | bool | `False` | Add one-hot encoded node types to node features |

### World Edge Parameters

| Parameter | Type | Default | Valid Range | Description |
|-----------|------|---------|-------------|-------------|
| `use_world_edges` | bool | `False` | - | Enable world edges (radius-based connections) |
| `world_radius_multiplier` | float | `1.5` | 1.0-5.0 | `r_world = multiplier * min_mesh_edge_length` |

### Test Set Control Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `display_testset` | bool | `True` | Display predictions on test samples during training |
| `test_batch_idx` | list | `0, 1, 2, 3` | Batch indices to display test results for |

---

## Running the Script

### Command

```bash
python MeshGraphNets_main.py
```

The script automatically reads `config.txt` from the current directory.

### Execution Modes

#### Single GPU Training

```
gpu_ids 0
```

Uses `training_profiles/single_training.py`

#### Multi-GPU Distributed Training

```
gpu_ids 0, 1, 2, 3
```

Uses `training_profiles/distributed_training.py` with PyTorch DDP (DistributedDataParallel).

**Notes for multi-GPU**:
- `world_size` is auto-calculated from `len(gpu_ids)`
- Effective batch size = `Batch_size * num_gpus`
- Uses NCCL backend for GPU communication

#### CPU Training

```
gpu_ids -1
```

Falls back to CPU (slow, not recommended).

### Modifying config.txt Programmatically

To modify `config.txt` for hyperparameter optimization:

```python
def write_config(params: dict, config_path: str = "config.txt"):
    """Write a new config.txt file with given parameters."""
    lines = []

    # Core params
    lines.append(f"model   MeshGraphNets")
    lines.append(f"mode    Train")
    lines.append(f"gpu_ids {params.get('gpu_ids', 0)}")
    lines.append(f"log_file_dir    train.log")
    lines.append(f"'")
    lines.append(f"% Common params")
    lines.append(f"Dim1    2138")
    lines.append(f"Dim2    1")
    lines.append(f"input_var   4")
    lines.append(f"output_var  4")
    lines.append(f"edge_var    4")
    lines.append(f"'")
    lines.append(f"% Network parameters")
    lines.append(f"dataset_dir {params.get('dataset_dir', './dataset/dataset.h5')}")
    lines.append(f"norm_min    {params.get('norm_min', -0.7)}")
    lines.append(f"norm_max    {params.get('norm_max', 0.7)}")
    lines.append(f"message_passing_num {params.get('message_passing_num', 15)}")
    lines.append(f"Training_epochs {params.get('training_epochs', 2002)}")
    lines.append(f"Batch_size  {params.get('batch_size', 1)}")
    lines.append(f"LearningR   {params.get('learning_rate', 0.0001)}")
    lines.append(f"Latent_dim  {params.get('latent_dim', 128)}")
    lines.append(f"num_workers {params.get('num_workers', 2)}")
    lines.append(f"std_noise   {params.get('std_noise', 0.01)}")
    lines.append(f"verbose     False")
    lines.append(f"'")
    lines.append(f"% Memory Optimization")
    lines.append(f"use_checkpointing   {params.get('use_checkpointing', True)}")
    lines.append(f"'")
    lines.append(f"% Node Type Parameters")
    lines.append(f"use_node_types  {params.get('use_node_types', False)}")
    lines.append(f"'")
    lines.append(f"% World Edge Parameters")
    lines.append(f"use_world_edges         {params.get('use_world_edges', False)}")
    lines.append(f"world_radius_multiplier {params.get('world_radius_multiplier', 1.5)}")
    lines.append(f"'")
    lines.append(f"% Test set control")
    lines.append(f"display_testset {params.get('display_testset', True)}")
    lines.append(f"test_batch_idx  {params.get('test_batch_idx', '0, 1, 2, 3')}")

    with open(config_path, 'w') as f:
        f.write('\n'.join(lines))
```

---

## Hyperparameter Optimization Guidelines

### Key Hyperparameters for Optimization

These are the most impactful parameters to tune:

| Parameter | Priority | Suggested Search Space | Impact |
|-----------|----------|------------------------|--------|
| `LearningR` | HIGH | [1e-5, 3e-5, 1e-4, 3e-4, 1e-3] | Convergence speed and stability |
| `Latent_dim` | HIGH | [64, 128, 256, 512] | Model capacity |
| `message_passing_num` | HIGH | [5, 10, 15, 20, 25] | Receptive field and depth |
| `Batch_size` | MEDIUM | [1, 2, 4, 8, 16] | Training stability and speed |
| `world_radius_multiplier` | MEDIUM | [1.0, 1.5, 2.0, 3.0] | Long-range interactions |
| `std_noise` | LOW | [0, 0.001, 0.01, 0.1] | Regularization |

### Parameter Dependencies and Constraints

1. **Batch Size vs GPU Memory**:
   - Each sample uses ~5-8 GB VRAM
   - With `use_checkpointing=True`, can use larger batches
   - Keep the batch size < 8

2. **Latent_dim vs Model Size**:
   - Total params ≈ `6 * Latent_dim^2 * message_passing_num`
   - Example: `Latent_dim=128, message_passing_num=15` → ~14M params
   - Example: `Latent_dim=256, message_passing_num=15` → ~57M params

### Recommended Search Strategy

**Phase 1: Coarse Search (fewer epochs)**
```python
coarse_grid = {
    'learning_rate': [2e-5, 1e-4, 1e-3],
    'latent_dim': [64, 128, 256],
    'message_passing_num': [10, 15, 20],
    'training_epochs': 100  # Quick evaluation
}
```

**Phase 2: Fine-tuning (more epochs)**
```python
fine_grid = {
    'learning_rate': [best_lr * 0.5, best_lr, best_lr * 2],
    'latent_dim': [best_dim],
    'message_passing_num': [best_mp - 2, best_mp, best_mp + 2],
    'training_epochs': 500
}
```

---

## Output and Logging

### Output Directory Structure

```
outputs/
├── <gpu_ids>/
│   └── train.log          # Training log with epoch losses
├── best_model.pth         # Best model checkpoint
└── inference_results/     # Inference outputs (every 10 epochs)
    └── epoch_<N>/
        └── sample_<id>.h5
```

### Log File Format

The `train.log` file contains:

```
Training epoch log file
Time: 2026-01-26 10:30:00
Log file absolute path: /path/to/outputs/0/train.log
<config.txt contents>
Elapsed time: 123.45s Epoch 0 Train Loss: 1.2345e-02 Valid Loss: 1.3456e-02 LR: 1.0000e-04
Elapsed time: 246.78s Epoch 1 Train Loss: 9.8765e-03 Valid Loss: 1.1234e-02 LR: 1.0000e-04
...
```

### Parsing Training Results

```python
import re

def parse_log(log_path: str) -> list[dict]:
    """Parse training log file to extract metrics."""
    results = []
    with open(log_path, 'r') as f:
        for line in f:
            match = re.match(
                r'Elapsed time: ([\d.]+)s Epoch (\d+) Train Loss: ([\de.-]+) Valid Loss: ([\de.-]+) LR: ([\de.-]+)',
                line
            )
            if match:
                results.append({
                    'elapsed_time': float(match.group(1)),
                    'epoch': int(match.group(2)),
                    'train_loss': float(match.group(3)),
                    'valid_loss': float(match.group(4)),
                    'learning_rate': float(match.group(5))
                })
    return results
```

### Checkpoint Format

`best_model.pth` contains:

```python
{
    'epoch': int,                    # Best epoch number
    'model_state_dict': dict,        # Model weights
    'optimizer_state_dict': dict,    # Optimizer state
    'scheduler_state_dict': dict,    # LR scheduler state
    'train_loss': float,             # Training loss at best epoch
    'valid_loss': float              # Validation loss at best epoch
}
```

---

## Common Issues and Solutions

### Issue: CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce `Batch_size` to 1
2. Enable `use_checkpointing True`
3. Reduce `Latent_dim` to 64 or 128
4. Use multi-GPU with `gpu_ids 0, 1`

### Issue: Training Loss Not Decreasing

**Symptoms**: Loss stays constant or increases

**Solutions**:
1. Reduce `LearningR` by 10x (e.g., 1e-4 → 1e-5)
2. Increase `message_passing_num` (may need more depth)
3. Check dataset path is correct

### Issue: Slow Training

**Symptoms**: Epochs take too long

**Solutions**:
1. Reduce `num_workers` if CPU-bound
2. Increase `num_workers` if I/O-bound
3. Use faster GPU or multi-GPU
4. Reduce dataset size for debugging

### Issue: NaN Loss

**Symptoms**: Loss becomes `nan`

**Solutions**:
1. Reduce `LearningR` significantly (e.g., to 1e-5)
2. Check `std_noise` is not too large
3. Ensure dataset has valid normalization params

---

## Example Configurations

### Configuration 1: Fast Training (Debugging)

```
model   MeshGraphNets
mode    Train
gpu_ids 0
log_file_dir    debug.log
Dim1    2138
Dim2    1
input_var   4
output_var  4
edge_var    4
'
dataset_dir ./dataset/dataset.h5
norm_min    -0.7
norm_max    0.7
message_passing_num 5
Training_epochs 50
Batch_size  1
LearningR   0.001
Latent_dim  64
num_workers 4
std_noise   0.0
verbose     False
'
use_checkpointing   True
use_node_types  False
use_world_edges False
world_radius_multiplier 1.5
'
display_testset True
test_batch_idx  0, 1
```

### Configuration 2: High Capacity (Best Performance)

```
model   MeshGraphNets
mode    Train
gpu_ids 0, 1
log_file_dir    train.log
Dim1    2138
Dim2    1
input_var   4
output_var  4
edge_var    4
'
dataset_dir ./dataset/dataset.h5
norm_min    -0.7
norm_max    0.7
message_passing_num 20
Training_epochs 2000
Batch_size  1
LearningR   0.0001
Latent_dim  256
num_workers 4
std_noise   0.01
verbose     False
'
use_checkpointing   True
use_node_types  False
use_world_edges False
world_radius_multiplier 2.0
'
display_testset True
test_batch_idx  0, 1, 2, 3
```

### Configuration 3: Memory Efficient (Limited GPU)

```
model   MeshGraphNets
mode    Train
gpu_ids 0
log_file_dir    train.log
Dim1    2138
Dim2    1
input_var   4
output_var  4
edge_var    4
'
dataset_dir ./dataset/dataset.h5
norm_min    -0.7
norm_max    0.7
message_passing_num 10
Training_epochs 1000
Batch_size  1
LearningR   0.0001
Latent_dim  128
num_workers 2
std_noise   0.01
verbose     False
'
use_checkpointing   True
use_node_types  False
use_world_edges False
world_radius_multiplier 1.5
'
display_testset True
test_batch_idx  0, 1, 2, 3
```

---

## Summary for Hyperparameter Optimization Agent

1. **Modify** `config.txt` with new hyperparameters
2. **Run** `python MeshGraphNets_main.py`
3. **Wait** for training to complete
4. **Parse** `outputs/<gpu_ids>/train.log` to extract final validation loss
5. **Compare** results across configurations
6. **Iterate** with refined hyperparameter search

The primary metric to optimize is **validation loss** (lower is better). The best model checkpoint is automatically saved to `outputs/best_model.pth`.
