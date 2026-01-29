# MeshGraphNets Configuration & Execution Guide

**Purpose**: Complete reference for running MeshGraphNets training with hyperparameter optimization. Designed for automated LLM agent workflows.

---

## Quick Start

```bash
# 1. Edit config.txt (see template below)
# 2. Run training
python MeshGraphNets_main.py

# 3. Check results
cat outputs/<gpu_id>/train.log
```

---

## Configuration File Requirements

**CRITICAL**: Configuration rules are strict and non-negotiable:

- **File name**: Must be `config.txt` (exact name, case-sensitive)
- **Location**: Same directory as `MeshGraphNets_main.py`
- **Format**: Plain text with custom syntax
  - Lines starting with `%` are comments
  - `'` marks section separators
  - Keys are case-insensitive (converted to lowercase internally)

**NO OTHER FILE NAMES OR LOCATIONS ARE ALLOWED**

### config.txt Template

```
model   MeshGraphNets
mode    Train
gpu_ids 0      # -1=CPU, 0=GPU0, [0,1,2,3]=multi-GPU
log_file_dir    train.log
'
% Dataset dimensions (informational)
Dim1    2138   # Number of samples
Dim2    1      # Timesteps per sample
input_var   4  # Node features: x_disp, y_disp, z_disp, stress
output_var  4  # Output features (same)
edge_var    4  # Edge features: dx, dy, dz, distance (fixed)
'
% Network and training parameters
dataset_dir ./dataset/dataset.h5
norm_min    -0.7
norm_max    0.7
message_passing_num 15
Training_epochs 2002
Batch_size  1
LearningR   0.0001
Latent_dim  128
num_workers 2
std_noise   0.01
verbose     False
'
% Memory optimization
use_checkpointing   True
'
% Feature toggles
use_node_types  False    # Add one-hot node types to features
'
% World edges (collision detection)
use_world_edges         False
world_radius_multiplier 1.5     # r_world = multiplier × min_mesh_edge
'
% Test set visualization
display_testset True
test_batch_idx  0, 1, 2, 3
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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model | str | MeshGraphNets | Model architecture (fixed) |
| mode | str | Train | Train or Inference |
| gpu_ids | int/list | 0 | -1=CPU, 0=GPU0, [0,1,2,3]=multi-GPU DDP |
| log_file_dir | str | train.log | Log filename (saved to outputs/<gpu_ids>/) |

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

**Phase 1: Coarse Grid Search (100 epochs)**
```python
coarse_grid = {
    'LearningR': [2e-5, 1e-4, 1e-3],
    'Latent_dim': [64, 128, 256],
    'message_passing_num': [10, 15, 20],
    'Batch_size': [1],
    'Training_epochs': 100
}
```

**Phase 2: Fine-Tune Best Config (1000-2000 epochs)**

### Programmatic Config Modification

```python
def write_config(params: dict, path: str = "config.txt"):
    """Write config.txt with given hyperparameters."""
    config = f"""model   MeshGraphNets
mode    Train
gpu_ids {params.get('gpu_ids', 0)}
log_file_dir    train.log
'
% Common params
Dim1    2138
Dim2    1
input_var   4
output_var  4
edge_var    4
'
% Network parameters
dataset_dir {params.get('dataset_dir', './dataset/dataset.h5')}
norm_min    {params.get('norm_min', -0.7)}
norm_max    {params.get('norm_max', 0.7)}
message_passing_num {params['message_passing_num']}
Training_epochs {params['Training_epochs']}
Batch_size  {params['Batch_size']}
LearningR   {params['LearningR']}
Latent_dim  {params['Latent_dim']}
num_workers {params.get('num_workers', 2)}
std_noise   {params.get('std_noise', 0.01)}
verbose     False
'
% Memory Optimization
use_checkpointing   {params.get('use_checkpointing', True)}
'
% Node Type Parameters
use_node_types  {params.get('use_node_types', False)}
'
% World Edge Parameters
use_world_edges         {params.get('use_world_edges', False)}
world_radius_multiplier {params.get('world_radius_multiplier', 1.5)}
'
% Test set control
display_testset {params.get('display_testset', True)}
test_batch_idx  {params.get('test_batch_idx', '0, 1, 2, 3')}
"""
    with open(path, 'w') as f:
        f.write(config)
```

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
```

Parse with regex:
```python
import re

def parse_log(log_path: str) -> list[dict]:
    """Extract training metrics from log file."""
    results = []
    pattern = r'Elapsed time: ([\d.]+)s Epoch (\d+) Train Loss: ([\de.-]+) Valid Loss: ([\de.-]+) LR: ([\de.-]+)'

    with open(log_path, 'r') as f:
        for line in f:
            match = re.match(pattern, line)
            if match:
                results.append({
                    'elapsed_time': float(match.group(1)),
                    'epoch': int(match.group(2)),
                    'train_loss': float(match.group(3)),
                    'valid_loss': float(match.group(4)),
                    'learning_rate': float(match.group(5))
                })
    return results

# Get final validation loss
metrics = parse_log('outputs/0/train.log')
final_val_loss = metrics[-1]['valid_loss']  # Optimize this metric
```

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

## Example Configurations

### Debug (Fast Iteration)
```
message_passing_num 5
Training_epochs 50
Batch_size  1
LearningR   0.001
Latent_dim  64
```
Use case: Quick code testing, architecture debugging

### Standard (Baseline)
```
message_passing_num 15
Training_epochs 2000
Batch_size  1
LearningR   0.0001
Latent_dim  128
```
Use case: Default configuration, balanced performance/cost

### High Capacity (Best Performance)
```
message_passing_num 20
Training_epochs 2000
Batch_size  1
LearningR   0.0001
Latent_dim  256
gpu_ids 0, 1  # Multi-GPU recommended
```
Use case: Maximum model capacity, requires 2+ GPUs

---

## Troubleshooting

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| **OOM Error** | `RuntimeError: CUDA out of memory` | 1. Set `Batch_size=1`<br>2. Enable `use_checkpointing=True`<br>3. Reduce `Latent_dim` to 64/128<br>4. Use multi-GPU |
| **Loss Not Decreasing** | Loss constant or increasing | 1. Reduce `LearningR` by 10× (e.g., 1e-4 → 1e-5)<br>2. Increase `message_passing_num`<br>3. Verify dataset path |
| **NaN Loss** | Loss becomes `nan` | 1. Reduce `LearningR` to 1e-5<br>2. Reduce `std_noise`<br>3. Check dataset normalization |
| **Slow Training** | Epochs take too long | 1. Adjust `num_workers` (2-8)<br>2. Use multi-GPU<br>3. Reduce dataset size for debugging |

---

## Memory Requirements

Approximate VRAM usage per sample (~68k nodes, 206k edges):

| Config | VRAM per Sample |
|--------|-----------------|
| Baseline (no checkpointing) | 5-8 GB |
| With checkpointing (default) | 2-3 GB |

**Recommendations**:
- ≥16 GB VRAM: `Batch_size=1-4`, checkpointing optional
- 8-16 GB VRAM: `Batch_size=1`, checkpointing required
- <8 GB VRAM: Use multi-GPU or reduce `Latent_dim`
