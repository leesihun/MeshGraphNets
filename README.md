# MeshGraphNets

A Graph Neural Network (GNN) for simulating deformable mesh dynamics. Predicts node displacements and stresses using an Encoder-Processor-Decoder architecture with graph message passing. Based on DeepMind's MeshGraphNets paper (Pfaff et al., ICLR 2021) and NVIDIA PhysicsNeMo.

The model predicts **normalized deltas** (state_{t+1} - state_t), not absolute values. This allows the same trained model to generalize across different geometry scales and simulation conditions.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Data Pipeline](#data-pipeline)
- [Training](#training)
- [Inference](#inference)
- [Multi-Scale (Hierarchical GNN)](#multi-scale-hierarchical-gnn)
- [World Edges (Collision Detection)](#world-edges-collision-detection)
- [Normalization](#normalization)
- [Visualization Tools](#visualization-tools)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Design Decisions](#design-decisions)
- [References](#references)
- [Documentation Index](#documentation-index)

---

## Quick Start

### Training

```bash
# Edit configuration (or use example configs from _flag_input/ or _warpage_input/)
vim config.txt  # Set mode=Train, dataset_dir, hyperparameters

# Run training (auto-detects single/multi-GPU based on gpu_ids)
python MeshGraphNets_main.py

# Or specify a config file explicitly
python MeshGraphNets_main.py --config _warpage_input/config_train3.txt

# Monitor training in real-time (separate terminal)
pip install -r misc/requirements_plotting.txt
python misc/plot_loss_realtime.py config.txt  # Visit http://localhost:5000
```

### Inference (Autoregressive Rollout)

```bash
# Set mode=Inference, modelpath, infer_dataset, infer_timesteps in config
python MeshGraphNets_main.py --config _warpage_input/config_infer1.txt
```

### Utilities

```bash
python misc/plot_loss.py config.txt --output loss_plot.png   # Static loss plot
python misc/debug_model_output.py                            # Check for NaN/zero outputs
python generate_inference_dataset.py                         # Create inference dataset from training data
python animate_h5.py                                         # Animate mesh deformation from HDF5
```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.13+ with CUDA support
- PyTorch Geometric
- h5py, numpy, scipy

### Setup

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install h5py numpy scipy scikit-learn matplotlib pyvista tqdm

# Optional: GPU-accelerated world edges
pip install torch-cluster

# Optional: real-time training dashboard
pip install -r misc/requirements_plotting.txt
```

---

## Configuration

Configuration is managed through plain-text config files. By default, `MeshGraphNets_main.py` reads `config.txt` from the repo root. Use `--config` to specify a different file.

Example configs are provided in `_flag_input/` and `_warpage_input/`.

### File Format

- `key value` pairs (one per line)
- `%` starts a comment line
- `'` marks section separators (optional, ignored by parser)
- `#` inline comments are stripped from values
- Keys are **case-insensitive** (converted to lowercase internally by `load_config.py`)
- Comma-separated values become lists (e.g., `gpu_ids 0, 1, 2, 3`)
- Space-separated numeric values also become lists
- Boolean values: `True`/`False` (case-insensitive)

### Essential Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| **mode** | str | `Train` or `Inference` |
| **gpu_ids** | int/list | `-1` for CPU, single GPU (e.g., `0`), multi-GPU DDP (e.g., `0, 1, 2, 3`) |
| **dataset_dir** | str | Path to HDF5 training dataset |
| **modelpath** | str | Checkpoint save path (single-GPU) or load path (inference). DDP always saves to `outputs/best_model.pth` |
| **infer_dataset** | str | Path to HDF5 inference dataset |
| **infer_timesteps** | int | Number of rollout steps to predict (auto-detected if dataset has multiple timesteps) |
| **input_var** | int | Node input features, default 4 (e.g., x_disp, y_disp, z_disp, stress; excluding node types) |
| **output_var** | int | Node output features, default 4 |
| **edge_var** | int | Edge features, always 8: `[deformed_dx, deformed_dy, deformed_dz, deformed_dist, ref_dx, ref_dy, ref_dz, ref_dist]` |

### Network Hyperparameters

| Parameter | Default | Range | Impact | Description |
|-----------|---------|-------|--------|-------------|
| **LearningR** | 0.0001 | 1e-6 to 1e-2 | **High** | Adam learning rate (most sensitive parameter) |
| **Latent_dim** | 128 | 32-512 | **High** | MLP hidden dimension, affects VRAM quadratically |
| **message_passing_num** | 15 | 1-30 | **Medium** | GNN depth (number of processor GnBlocks). Ignored when `use_multiscale=True` |
| **Batch_size** | 50 | 1-128 | **High** | Per-GPU batch size |
| **Training_epochs** | 500 | - | - | Total training epochs |
| **std_noise** | 0.001 | 0-0.1 | **Medium** | Gaussian noise augmentation (training only). Applied to physical features only, with target correction |
| **noise_gamma** | 1.0 | 0-1 | **Low** | Noise target correction factor. 1.0=full correction, 0.0=no correction, 0.1=DeepMind cloth default |
| **residual_scale** | 1.0 | 0-1 | **Low** | Scale factor for node+edge residuals. `0.5` dampens residuals, `1.0`=full (DeepMind default) |

### Performance & Memory Optimization

| Parameter | Default | Description |
|-----------|---------|-------------|
| **use_checkpointing** | False | Gradient checkpointing: trades 20-30% compute for 60-70% VRAM reduction |
| **use_amp** | False | Mixed precision training with bfloat16 (1.5-2x speedup on Ampere+ GPUs). Uses bfloat16 (not float16) due to scatter_add overflow issues in GNNs |
| **use_compile** | False | `torch.compile(dynamic=True)` for kernel fusion (10-30% speedup). First epoch slower due to JIT warmup |
| **grad_accum_steps** | 1 | Gradient accumulation: `1`=per-batch (default), `0`=full epoch (1 optimizer step/epoch), `N`=every N batches |
| **num_workers** | 0 | DataLoader worker processes. Uses `persistent_workers=True` and `prefetch_factor=2` when > 0 |
| **use_parallel_stats** | True | Parallel normalization stat computation (speeds up initialization for large datasets) |

### Loss Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| **feature_loss_weights** | None | Per-feature loss weights (comma-separated). Example: `0.001, 0.001, 1.0` emphasizes z_disp. Auto-normalized to sum to 1.0 |
| **verbose** | False | Per-feature loss breakdowns in training output |

### Test & Visualization

| Parameter | Default | Description |
|-----------|---------|-------------|
| **test_interval** | 10 | Run test/visualization every N epochs |
| **test_max_batches** | 200 | Max test samples per evaluation. Caps test_model runtime to avoid NCCL timeout in DDP |
| **display_testset** | True | Save HDF5 + PNG visualization for test batches |
| **display_trainset** | True | Also save train reconstruction visualizations |
| **test_batch_idx** | 0 | Comma-separated batch indices to visualize (e.g., `0, 1, 2, 3, 15, 16`) |
| **plot_feature_idx** | -1 | Feature index to visualize in plots (`-1` = last/stress, `-2` = z_disp) |
| **monitor_gradients** | True | Gradient norm tracking with vanishing/exploding warnings |

### Logging

| Parameter | Default | Description |
|-----------|---------|-------------|
| **log_file_dir** | None | Log filename under `outputs/` (e.g., `train1.log`). Enables debug npz dumps. Required for loss plotting tools |

### Node Type Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **use_node_types** | False | One-hot encode node types from HDF5 metadata (last feature of nodal_data, typically part number) |

**Note**: `num_node_types` is assigned automatically by code after dataset load; do not set it manually in config.

### World Edge Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **use_world_edges** | False | Enable radius-based collision detection edges |
| **world_radius_multiplier** | 1.5 | `r_world = multiplier * min_mesh_edge_length` (auto-computed from dataset) |
| **world_max_num_neighbors** | 64 | Max neighbors per node in radius query |
| **world_edge_backend** | `torch_cluster` | `torch_cluster` (GPU, fast) or `scipy_kdtree` (CPU). **Must be explicitly set in every config** — rollout.py crashes if absent |

### Multi-Scale Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **use_multiscale** | False | Enable hierarchical V-cycle GNN (BFS Bi-Stride, ICML 2023) |
| **multiscale_levels** | 1 | Number of coarsening levels |
| **mp_per_level** | None | Message passing blocks per V-cycle stage (comma-separated). For L=1: `[pre_0, coarsest, post_0]`. Must have `2*L+1` entries |
| **fine_mp_pre** | 5 | Legacy: pre-coarsening blocks (ignored when `mp_per_level` is set) |
| **coarse_mp_num** | 5 | Legacy: coarsest-level blocks |
| **fine_mp_post** | 5 | Legacy: post-coarsening blocks |

For a complete parameter reference, see [CONFIG_AND_EXECUTION_GUIDE.md](CONFIG_AND_EXECUTION_GUIDE.md).

---

## Architecture

### Execution Flow

```
MeshGraphNets_main.py (entry point, --config flag)
  │
  ├── mode=Train + single GPU/CPU  → single_training.py → training_loop.py
  ├── mode=Train + multi GPU       → distributed_training.py (DDP via mp.spawn) → training_loop.py
  └── mode=Inference               → rollout.py (autoregressive)
```

1. **Entry point**: `MeshGraphNets_main.py` parses the `--config` CLI flag (default: `config.txt`)
2. Loads configuration via `load_config()` from `general_modules/load_config.py`
3. Auto-detects GPU configuration from `gpu_ids` and routes:
   - **Single GPU/CPU** → `training_profiles/single_training.py` calls `single_worker()`
   - **Multi-GPU DDP** → `training_profiles/distributed_training.py` spawns via `mp.spawn(train_worker, ...)`
   - **Inference** → `inference_profiles/rollout.py` calls `run_rollout()`
4. Both training pipelines share `training_profiles/training_loop.py` for epoch logic (train, validate, test)

### Core Model: Encoder-Processor-Decoder

Defined in `model/MeshGraphNets.py` as class `EncoderProcessorDecoder`:

```
Input (node features [N, input_var] + edge features [E, 8])
        │
     Encoder
     ├── nb_encoder: MLP projects node features → latent_dim
     ├── eb_encoder: MLP projects edge features → latent_dim
     └── world_eb_encoder: MLP for world edges (if use_world_edges)
        │
     Processor (message_passing_num × GnBlock)
     │  For each GnBlock:
     │  ├── EdgeBlock: concat [sender, receiver, edge] → MLP → updated edge features
     │  ├── NodeBlock: concat [node, sum_aggregated_edges] → MLP → updated node features
     │  │   (or HybridNodeBlock: aggregates mesh + world edges separately, then concatenates)
     │  └── Residual connections on BOTH nodes AND edges (applied after node update)
     │      edge_out = edge_in + scale * edge_mlp_out
     │      node_out = node_in + scale * node_mlp_out
        │
     Decoder
     └── MLP projects latent_dim → output_var (NO LayerNorm on output)
        │
     Output (predicted normalized delta [N, output_var])
```

**Key model details**:
- **MLPs**: 2 hidden layers, ReLU activation, LayerNorm on output (except decoder)
- **Aggregation**: Sum (forces/stresses accumulate at nodes, matches physics)
- **Initialization**: Kaiming/He uniform for all linear layers
- **Decoder last layer**: Weights scaled by 0.01 at init so initial predictions ≈ 0 (good prior for delta prediction: "predict no change" at start)
- **Residual scale**: Configurable via `residual_scale` (default 1.0 = full residual)
- **Gradient checkpointing**: Optional, controlled by `use_checkpointing`. Trades ~20-30% compute for ~60-70% VRAM reduction

### Noise Injection (Training Only)

During training, the `MeshGraphNets.forward()` method applies Gaussian noise augmentation:

1. **Node noise**: Applied to physical features only (first `output_var` features), not to node type one-hot encodings
2. **Edge noise**: Same std applied to all edge features in normalized space
3. **Target correction** (DeepMind formulation): `target -= gamma * noise * noise_std_ratio`
   - `noise_gamma` controls correction strength (1.0=full, 0.0=none, 0.1=DeepMind cloth default)

---

## Data Pipeline

### HDF5 Dataset Format

**File**: Single HDF5 file (`.h5`) containing all samples with global metadata.

```
dataset.h5
├── [Attributes]
│   ├── num_samples: int
│   ├── num_features: int (typically 8)
│   └── num_timesteps: int
│
├── data/
│   ├── 1/                                  # Sample 1 (sequential ID)
│   │   ├── nodal_data                      # [num_features, num_timesteps, num_nodes] float32
│   │   ├── mesh_edge                       # [2, num_edges] int64
│   │   └── metadata/
│   │       ├── [Attributes]: source_filename, num_nodes, num_edges, ...
│   │       ├── feature_min/max/mean/std     # [num_features] float32
│   ├── 2/
│   └── ...
│
└── metadata/
    ├── feature_names                        # [num_features] string
    ├── normalization_params/                # Global min/max/mean/std
    └── splits/                             # train/val/test sample IDs
```

**Nodal data feature order** (index-based):
| Index | Feature | Unit |
|-------|---------|------|
| 0 | x_coord | mm |
| 1 | y_coord | mm |
| 2 | z_coord | mm |
| 3 | x_disp | mm |
| 4 | y_disp | mm |
| 5 | z_disp | mm |
| 6 | stress | MPa |
| 7 | Part number | - |

**Important**: `nodal_data` shape is `[features, timesteps, nodes]` — features-first, not nodes-first. All samples must have equal timestep counts.

**Edges** are stored as unidirectional in HDF5 (`[2, E]`). The dataset class makes them bidirectional: `edge_index = [mesh_edge; mesh_edge[[1,0]]]`.

Full specification: [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md)

### Dataset Class: MeshGraphDataset

`general_modules/mesh_dataset.py` implements the PyTorch Dataset:

1. **Loads HDF5** with persistent handle (SWMR mode, `HDF5_USE_FILE_LOCKING=FALSE`)
2. **Computes Z-score normalization statistics** across all samples:
   - Node stats: from physical features `nodal_data[3:3+input_var]` across sampled timesteps (up to 500 per sample)
   - Edge stats: from 8-D edge features (deformed + reference relative positions and distances)
   - Delta stats: from actual `state_{t+1} - state_t` transitions
3. **Parallel stat computation**: When `use_parallel_stats=True`, uses multiprocessing to process sample chunks in parallel
4. **Returns** `torch_geometric.data.Data` graphs with normalized features and targets

### Edge Features

Computed by `general_modules/edge_features.py`:

```
[deformed_dx, deformed_dy, deformed_dz, deformed_dist, ref_dx, ref_dy, ref_dz, ref_dist]
```

- **Deformed** relative positions: computed from `reference_pos + displacement` (current geometry)
- **Reference** relative positions: computed from reference positions (initial geometry)
- Both include Euclidean distance as the 4th and 8th feature

### DataLoader

`general_modules/data_loader.py` creates DataLoaders via `torch_geometric.loader.DataLoader`:
- DDP uses `DistributedSampler`
- When `num_workers > 0`: `persistent_workers=True` and `prefetch_factor=2`
- Data split: 80/10/10 train/val/test (seed=42)

---

## Training

### Optimizer & Scheduler

- **Optimizer**: Adam with `fused=True` on CUDA (no weight decay, matches DeepMind original)
- **LR Scheduler**: `CosineAnnealingLR(T_max=training_epochs, eta_min=1e-8)` — single cosine decay over full training for both single-GPU and DDP
- **Gradient clipping**: `max_norm=10.0` (applied after gradient accumulation)

### Loss Function

MSE on normalized deltas, per feature:

```
errors = (predicted - target) ** 2        # [N, output_var]
per_node = weighted_mean(errors, dim=-1)  # [N] — weighted by feature_loss_weights if set
loss = per_node.mean()                    # scalar
```

- Per-feature weights (via `feature_loss_weights`) are auto-normalized to sum to 1.0
- Weights apply consistently to train, validation, and test phases
- Gradient accumulation scales loss by window size to maintain correct gradient magnitude

### Mixed Precision (AMP)

When `use_amp=True`:
- Forward pass + loss wrapped in `torch.amp.autocast('cuda', dtype=torch.bfloat16)`
- Uses **bfloat16** (not float16) due to scatter_add overflow issues in GNN sum aggregation
- 1.5-2x speedup on Ampere+ GPUs (A100, H100, RTX 30xx+)

### Checkpointing

Checkpoints are saved when validation loss improves:

```python
{
    'epoch': int,
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...,
    'train_loss': float,
    'valid_loss': float,
    'normalization': {
        'node_mean', 'node_std',     # [input_var]
        'edge_mean', 'edge_std',     # [8]
        'delta_mean', 'delta_std',   # [output_var]
        'node_type_to_idx': dict,    # if use_node_types
        'num_node_types': int,       # if use_node_types
        'world_edge_radius': float,  # if use_world_edges
        'coarse_edge_means': list,   # if use_multiscale
        'coarse_edge_stds': list,    # if use_multiscale
    },
    'model_config': {
        'input_var', 'output_var', 'edge_var',
        'latent_dim', 'message_passing_num',
        'use_node_types', 'num_node_types',
        'use_world_edges', 'use_checkpointing',
        'use_multiscale', 'multiscale_levels',
        'mp_per_level', ...
    },
}
```

**Save paths**:
- Single GPU: saves to `modelpath` from config
- Multi-GPU DDP: always saves to `outputs/best_model.pth` regardless of `modelpath`

### Test & Visualization During Training

Every `test_interval` epochs (and at the last epoch):
1. Evaluates model on test set (capped at `test_max_batches` samples)
2. For batch indices in `test_batch_idx`:
   - Saves HDF5 files with predicted vs. ground truth (denormalized to physical units)
   - Renders PNG visualizations via PyVista (with GPU-accelerated triangle reconstruction)
3. Output path: `outputs/test/<gpu_ids>/<epoch>/sample<id>_t<timestep>.h5`

### Debug Tools

When `verbose=True`:
- Per-feature MSE breakdown every 10 epochs
- Per-layer gradient statistics
- Memory tracking for validation batches
- Warnings for near-constant predictions or vanishing/exploding gradients

When `log_file_dir` is set:
- Debug NPZ files saved with input/output/prediction tensors
- Training log file written with per-epoch losses and timing

---

## Inference

### Autoregressive Rollout

`inference_profiles/rollout.py` performs time-transient inference:

1. **Load checkpoint**: Extracts model weights, normalization stats, and model config
2. **Model config override**: Checkpoint's `model_config` overrides config file values (ensures architecture matches)
3. **Load initial condition**: Reads timestep 0 from inference HDF5 dataset
4. **Rollout loop** (for each step t → t+1):
   ```
   a. Normalize current state:  x_norm = (x_raw - node_mean) / node_std
   b. Add node type one-hot (if enabled, concatenated after normalization)
   c. Compute deformed positions: deformed = reference + displacement
   d. Compute 8-D edge features from current and reference geometry
   e. Normalize edge features:  edge_norm = (edge_raw - edge_mean) / edge_std
   f. Compute world edges (if enabled)
   g. Build PyG Data graph
   h. Forward pass → predicted normalized delta
   i. Denormalize delta:  delta = delta_norm * delta_std + delta_mean
   j. Update state:  state_{t+1} = state_t + delta
   ```
5. **Save results**: HDF5 file following DATASET_FORMAT.md structure at `<inference_output_dir>/rollout_sample{id}_steps{N}.h5`

### Inference Config Requirements

```
mode            Inference
modelpath       ./outputs/warpage3.pth
infer_dataset   ./dataset/warpage_infer.h5
infer_timesteps 34
world_edge_backend  scipy_kdtree    # REQUIRED — crashes without it
```

---

## Multi-Scale (Hierarchical GNN)

Optional V-cycle / U-Net architecture based on BFS Bi-Stride coarsening (Cao et al., ICML 2023). Implemented in `model/coarsening.py`.

### How It Works

1. **BFS Bi-Stride Coarsening**: Multi-source BFS assigns each node a depth. Even-depth nodes → coarse graph; odd-depth nodes → assigned to their BFS parent.
2. **Topology-based**: No spatial proximity heuristics — works correctly across disconnected parts (multi-part FEA).
3. **V-cycle architecture**: Descending arm (fine → coarse with skip connections), coarsest level processing, ascending arm (coarse → fine with skip merge).

### Typical Coarsening Ratios

| Mesh Type | Ratio per Level |
|-----------|----------------|
| 2D triangular | ~N/4 (avg degree ~6) |
| 3D tet | ~N/2 (avg degree ~15-25) |
| 3D hex | ~N/2 (avg degree ~20-30) |

Use `multiscale_levels=2` for 3D meshes to achieve ~N/4 total reduction.

### V-Cycle Structure

For `multiscale_levels=1` with `mp_per_level 2, 10, 2`:

```
Fine graph:     [2 pre-blocks] → Pool → [10 coarsest blocks] → Unpool → [2 post-blocks]
                     ↓              skip connection              ↑
                     └──────────── concat + linear proj ─────────┘
```

- **Pool**: `pool_features()` — aggregates fine node features to coarse nodes
- **Unpool**: `unpool_features()` — broadcasts coarse features back to fine nodes
- **Skip merge**: `h_merged = LinearProj(concat(h_skip, h_unpooled))`
- Coarse edge features are encoded by separate MLPs per level
- When `use_multiscale=True`, `message_passing_num` is ignored

---

## World Edges (Collision Detection)

Optional radius-based edges for detecting and handling self-contact/collision.

### How It Works

1. **Radius query**: Find all node pairs within `r_world = world_radius_multiplier * min_mesh_edge_length`
2. **Filter**: Exclude pairs that already have mesh edges (avoid duplication)
3. **Edge features**: Same 8-D format as mesh edges (deformed + reference relative positions)
4. **Separate aggregation**: `HybridNodeBlock` aggregates mesh and world edges independently, then concatenates before the node MLP

### Backends

| Backend | Device | Speed | Notes |
|---------|--------|-------|-------|
| `torch_cluster` | GPU | Fast | Uses `radius_graph()`, requires `pip install torch-cluster` |
| `scipy_kdtree` | CPU | Slower | Uses `scipy.spatial.KDTree.query_pairs()`, always available |

Both backends produce identical results. World edges are recomputed at every rollout step during inference (since geometry changes).

Full documentation: [docs/WORLD_EDGES_DOCUMENTATION.md](docs/WORLD_EDGES_DOCUMENTATION.md)

---

## Normalization

Z-score normalization (`x_norm = (x - mean) / std`) applied per feature, computed separately for three domains:

| Domain | Source | Dimensions |
|--------|--------|------------|
| **Node** | Physical features `nodal_data[3:3+input_var]` across all samples and sampled timesteps (up to 500 per sample) | `[input_var]` |
| **Edge** | 8-D edge features (deformed + reference) computed from all mesh edges | `[8]` |
| **Delta** | Actual `state_{t+1} - state_t` transitions across all consecutive timestep pairs | `[output_var]` |

**Important invariants**:
- Delta normalization is **separate** from node normalization (different mean/std)
- For T=1 (static) datasets, the model input `x` is all-zeros and the target delta equals the feature values themselves
- Node types are one-hot encoded **after** normalization and concatenated to normalized features
- Minimum std clamped to `1e-8` to avoid division by zero

**Stats storage**: Saved in checkpoint under `checkpoint['normalization']` dict. During inference, stats are loaded from the checkpoint, not recomputed.

---

## Visualization Tools

### Real-Time Training Dashboard

```bash
pip install -r misc/requirements_plotting.txt
python misc/plot_loss_realtime.py config.txt
# Visit http://localhost:5000
```

- FastAPI-powered with auto-refreshing charts
- Live loss curves, best/latest loss stats
- API docs at `/docs` (Swagger UI)
- Requires `log_file_dir` to be set in config

### Static Loss Plot

```bash
python misc/plot_loss.py config.txt --output loss_plot.png
```

### Mesh Animation

```bash
python animate_h5.py
```

Animates mesh deformation from rollout HDF5 output files using PyVista.

### Debug Model Output

```bash
python misc/debug_model_output.py
```

Checks for NaN/zero outputs and validates model predictions.

Full details: [misc/README.md](misc/README.md)

---

## Project Structure

```
MeshGraphNets/
├── MeshGraphNets_main.py               # Entry point (--config flag)
│
├── model/
│   ├── MeshGraphNets.py                # EncoderProcessorDecoder architecture
│   │                                   # MeshGraphNets wrapper (noise injection, forward)
│   ├── blocks.py                       # EdgeBlock, NodeBlock, HybridNodeBlock
│   ├── coarsening.py                   # BFS Bi-Stride multi-scale coarsening
│   └── checkpointing.py               # Gradient checkpointing utilities
│
├── general_modules/
│   ├── load_config.py                  # Config file parser (key-value, case-insensitive)
│   ├── data_loader.py                  # DataLoader creation (wraps MeshGraphDataset)
│   ├── mesh_dataset.py                 # MeshGraphDataset class, normalization, parallel stats
│   ├── edge_features.py               # 8-D edge feature computation
│   └── mesh_utils_fast.py             # GPU triangle reconstruction, PyVista rendering, HDF5 I/O
│
├── training_profiles/
│   ├── single_training.py              # Single-GPU/CPU training pipeline
│   ├── distributed_training.py         # Multi-GPU DDP training (mp.spawn)
│   └── training_loop.py               # Epoch logic: train_epoch, validate_epoch, test_model
│
├── inference_profiles/
│   └── rollout.py                      # Autoregressive inference with world edge + multiscale support
│
├── misc/
│   ├── plot_loss.py                    # Static loss plot (matplotlib)
│   ├── plot_loss_realtime.py           # Real-time FastAPI dashboard
│   ├── debug_model_output.py           # Model debugging
│   ├── analyze_mesh_topology.py        # Mesh analysis utilities
│   ├── requirements_plotting.txt       # Visualization dependencies (fastapi, uvicorn, matplotlib)
│   └── README.md                       # Visualization tools documentation
│
├── dataset/
│   ├── DATASET_FORMAT.md               # HDF5 dataset specification
│   └── ...                             # Datasets (gitignored)
│
├── _flag_input/                        # Example configs for flag simulation
├── _warpage_input/                     # Example configs for warpage simulation
│
├── build_dataset.py                    # Dataset builder (source .h5 → training HDF5)
├── generate_inference_dataset.py       # Create inference dataset from training data
├── animate_h5.py                       # Mesh deformation animation (PyVista)
│
├── docs/                               # Documentation
├── outputs/                            # Training outputs (auto-created, gitignored)
├── CONFIG_AND_EXECUTION_GUIDE.md       # Complete parameter reference
├── CLAUDE.md                           # Claude Code AI assistant instructions
└── README.md                           # This file
```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| **CUDA OOM** | Batch size too large or model too wide | Reduce `Batch_size` (most effective), enable `use_checkpointing True`, reduce `Latent_dim`, use multi-GPU DDP |
| **Loss not decreasing** | Learning rate too high or too few message passing steps | Reduce `LearningR` by 10x, increase `message_passing_num`, verify `dataset_dir` path |
| **NaN loss** | Numerical instability | Reduce `LearningR` to 1e-5, reduce `std_noise`, check normalization stats for zeros |
| **Slow training** | DataLoader bottleneck | Increase `num_workers`, enable `use_amp True` for mixed precision, use multi-GPU |
| **`AttributeError: 'NoneType' has no attribute 'lower'`** | `world_edge_backend` missing from config | Add `world_edge_backend scipy_kdtree` to config |
| **Crash on first model save** | `modelpath` missing from config | Add `modelpath ./outputs/my_model.pth` |
| **NCCL timeout during DDP test** | Test takes too long on rank 0 while others wait | Reduce `test_max_batches` or increase `test_interval` |
| **First epoch very slow with use_compile** | JIT compilation warmup | Normal behavior — subsequent epochs will be faster |
| **Near-constant predictions** | Model collapsed | Check if decoder last-layer init is working, increase `LearningR`, check data normalization |
| **Import error running misc/ scripts** | Wrong working directory | Run from project root, not from `misc/` |

---

## Design Decisions

- **Sum aggregation** (not mean): Forces and stresses accumulate at nodes — physically correct for FEA mechanics (matches NVIDIA PhysicsNeMo)
- **Normalized delta prediction**: Decouples geometry scale from learned dynamics, enables generalization across geometries
- **Bidirectional edges**: Graph is undirected; edges stored in both directions in `edge_index`
- **Separate delta normalization**: Delta statistics are independent from node statistics — different physical meaning and scale
- **Residual connections on both nodes and edges**: Matches DeepMind original. Residuals applied after the node update step (NodeBlock sees raw edge MLP output)
- **No LayerNorm on decoder**: Allows full output range for delta prediction — LayerNorm would constrain the output distribution
- **Gradient clipping at 10.0**: Stabilizes deep message passing networks (15+ layers)
- **bfloat16 over float16**: GNN scatter_add operations can overflow in float16; bfloat16 has larger dynamic range
- **CosineAnnealingLR**: Smooth decay to near-zero LR by end of training — no learning rate restarts
- **Fused Adam**: Uses `fused=True` on CUDA for ~5-10% optimizer step speedup, no weight decay (matches DeepMind original)

---

## References

- "Learning Mesh-Based Simulation with Graph Networks" (Pfaff et al., ICLR 2021, DeepMind)
- "Efficient Learning of Mesh-Based Physical Simulation with Bi-Stride Multi-Scale Graph Neural Network" (Cao et al., ICML 2023) — for multi-scale coarsening
- NVIDIA PhysicsNeMo (deforming_plate example)
- PyTorch + PyTorch Geometric

---

## Documentation Index

- [CONFIG_AND_EXECUTION_GUIDE.md](CONFIG_AND_EXECUTION_GUIDE.md): Complete parameter reference with execution examples
- [docs/MESHGRAPHNET_ARCHITECTURE.md](docs/MESHGRAPHNET_ARCHITECTURE.md): Architecture walkthrough
- [docs/WORLD_EDGES_DOCUMENTATION.md](docs/WORLD_EDGES_DOCUMENTATION.md): Collision detection implementation
- [docs/VRAM_OPTIMIZATION_PLAN.md](docs/VRAM_OPTIMIZATION_PLAN.md): Memory optimization strategies
- [docs/VISUALIZATION_DENORMALIZATION.md](docs/VISUALIZATION_DENORMALIZATION.md): Denormalization for visualization
- [docs/ADAPTIVE_REMESHING_PLAN.md](docs/ADAPTIVE_REMESHING_PLAN.md): Adaptive remeshing plan
- [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md): HDF5 dataset structure
- [misc/README.md](misc/README.md): Visualization tools

---

Version 1.0.0, 2026-01-06 | Developed by SiHun Lee, Ph. D.
