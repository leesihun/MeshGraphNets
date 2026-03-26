# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MeshGraphNets** is a GNN for simulating deformable mesh dynamics. It predicts node displacements and stresses using an Encoder-Processor-Decoder architecture with graph message passing. Based on DeepMind's MeshGraphNets paper (Pfaff et al., ICLR 2021).

The model predicts **normalized deltas** (state_{t+1} - state_t), not absolute values. This decouples geometry scale from learned dynamics.

## Commands

```bash
# Training (reads config.txt by default, or specify --config)
python MeshGraphNets_main.py
python MeshGraphNets_main.py --config _warpage_input/config_train1.txt

# Inference (set mode=Inference in config)
python MeshGraphNets_main.py --config _warpage_input/config_infer1.txt

# Real-time loss dashboard
pip install -r misc/requirements_plotting.txt
python misc/plot_loss_realtime.py config.txt  # http://localhost:5000

# Static loss plot
python misc/plot_loss.py config.txt --output loss_plot.png

# Debug model outputs
python misc/debug_model_output.py

# Generate inference dataset from training data
python generate_inference_dataset.py

# Animate mesh deformation from HDF5
python animate_h5.py
```

No test suite exists. No linter is configured.

## Configuration (config.txt)

Plain text format: `key value`. Lines starting with `%` are comments. `'` marks section separators. `#` inline comments are stripped. Keys are **case-insensitive** (lowercased by [load_config.py](general_modules/load_config.py)). Comma-separated values become lists.

Example configs in [_flag_input/](\_flag_input/) and [_warpage_input/](\_warpage_input/).

### Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| mode | - | `Train` or `Inference` |
| gpu_ids | 0 | `-1`=CPU, `0`=single GPU, `0,1,2,3`=multi-GPU DDP |
| modelpath | - | Checkpoint save/load path |
| dataset_dir | - | HDF5 training dataset |
| infer_dataset | - | HDF5 inference dataset |
| infer_timesteps | - | Rollout steps for inference |
| input_var | 4 | Node input features (excluding node types) |
| output_var | 4 | Node output features |
| edge_var | 8 | Always `[deformed_dx, deformed_dy, deformed_dz, deformed_dist, ref_dx, ref_dy, ref_dz, ref_dist]` |
| Latent_dim | 128 | MLP hidden dimension |
| message_passing_num | 15 | GNN depth (number of processor blocks) |
| Batch_size | 50 | Per-GPU batch size |
| LearningR | 0.0001 | Adam learning rate |
| Training_epochs | 500 | |
| std_noise | 0.001 | Gaussian noise augmentation (training only). Applied to physical features only, with target correction |
| noise_gamma | 1.0 | Noise target correction factor. 1.0=full correction, 0.0=no correction, 0.1=DeepMind cloth default |
| feature_loss_weights | None | Per-feature loss weights (comma-separated). Example: `1.0, 1.0, 5.0` emphasizes z_disp. Auto-normalized. |
| use_checkpointing | False | Gradient checkpointing (saves ~60-70% VRAM) |
| grad_accum_steps | 1 | Gradient accumulation: `1`=per-batch (default), `0`=full epoch (1 step/epoch), `N`=every N batches |
| use_amp | False | Mixed precision training with bfloat16 (1.5-2x speedup on Ampere+ GPUs). Uses bfloat16 not float16 due to scatter_add overflow issues in GNNs |
| use_compile | False | `torch.compile(dynamic=True)` for kernel fusion (10-30% speedup). First epoch slower due to JIT warmup |
| use_ema | False | EMA shadow model for validation/inference. Smooths SGD noise, ~9 MB extra memory |
| ema_decay | 0.999 | EMA decay factor. `0.999`=~1000-step window, `0.9999`=~10000-step window |
| test_interval | 10 | Run test/visualization every N epochs. Previous default was 1 (every epoch) |
| test_max_batches | 200 | Max test samples per evaluation. Caps test_model runtime to avoid NCCL timeout in DDP |
| use_node_types | False | One-hot encode node types from HDF5 metadata |
| use_world_edges | False | Radius-based collision detection edges |
| use_parallel_stats | True | Parallel normalization stat computation |
| residual_scale | 1.0 | Scale factor for node+edge residuals. `0.5` dampens residuals, `1.0` = full (DeepMind default) |
| verbose | False | Per-feature loss breakdowns |
| monitor_gradients | True | Gradient norm tracking |
| num_workers | 0 | DataLoader worker processes |
| log_file_dir | - | Log filename under `outputs/` (e.g. `train1.log`). Enables debug npz dumps |
| display_testset | True | Save HDF5 + PNG visualization for test batches |
| test_batch_idx | 0 | Comma-separated batch indices to visualize (e.g. `0, 1, 2, 3`) |
| plot_feature_idx | -1 | Feature index to visualize in plots (`-1` = last, i.e. stress) |
| world_edge_backend | `torch_cluster` | `torch_cluster` (GPU) or `scipy_kdtree` (CPU). **Required in every config** — rollout.py has no default and crashes if absent |
| world_radius_multiplier | - | `r_world = multiplier * min_mesh_edge_length` (auto-computed from dataset) |
| world_max_num_neighbors | 64 | Max neighbors per node in world edge radius query |

Full reference: [CONFIG_AND_EXECUTION_GUIDE.md](CONFIG_AND_EXECUTION_GUIDE.md)

## Architecture

### Execution Flow

```
MeshGraphNets_main.py (entry point, --config flag)
  ├── mode=Train + single GPU  → single_training.py → training_loop.py
  ├── mode=Train + multi GPU   → distributed_training.py (DDP) → training_loop.py
  └── mode=Inference            → rollout.py (autoregressive)
```

### Model: EncoderProcessorDecoder ([model/MeshGraphNets.py](model/MeshGraphNets.py))

```
Input → Encoder → [GnBlock × message_passing_num] → Decoder → Output
```

- **Encoder**: Projects node features and edge features to `latent_dim` via MLPs
- **GnBlock** ([model/blocks.py](model/blocks.py)): EdgeBlock updates edges, NodeBlock aggregates edges to nodes
  - **Residual connections on both nodes and edges, applied after the node update** (matches DeepMind: NodeBlock receives raw edge MLP output, then `edge_out = edge_in + edge_mlp`, `node_out = node_in + node_mlp`)
  - With `use_world_edges`: HybridNodeBlock aggregates mesh + world edges separately then concatenates
- **Decoder**: Projects `latent_dim` to `output_var` (no LayerNorm on output)
- **MLP**: 2 hidden layers, ReLU, LayerNorm on output (except decoder)
- **Initialization**: Kaiming/He uniform
- **Aggregation**: Sum (forces/stresses accumulate at nodes)

### Data Pipeline ([general_modules/](general_modules/))

- [mesh_dataset.py](general_modules/mesh_dataset.py): Loads HDF5, computes Z-score normalization stats, returns `torch_geometric.data.Data` graphs
- [data_loader.py](general_modules/data_loader.py): Creates DataLoaders (DistributedSampler for DDP)
- [mesh_utils_fast.py](general_modules/mesh_utils_fast.py): Edge features, world edge KDTree queries, GPU triangle reconstruction, PyVista rendering

### Training

- **Optimizer**: Adam with `fused=True` on CUDA (no weight decay, matches DeepMind original)
- **LR Scheduler**: CosineAnnealingLR(T_max=training_epochs, eta_min=1e-8) for both single-GPU and DDP. Single cosine decay over full training.
- **AMP**: Optional bfloat16 mixed precision via `use_amp` config (bfloat16 preferred over float16 for GNN scatter_add safety)
- **DataLoader**: Uses `persistent_workers=True` and `prefetch_factor=2` to avoid worker respawn overhead
- **Loss**: Sum-over-features then mean-over-nodes on normalized deltas (matches DeepMind), with optional per-feature weighting (via `feature_loss_weights` config)
- **Gradient clipping**: max_norm=5.0
- **EMA**: Optional shadow model via `use_ema` config. Updated after each `optimizer.step()` with `AveragedModel(get_ema_multi_avg_fn)`. EMA model used for validation, test, and inference. Created before `torch.compile`/DDP wrapping
- **Checkpoint path**: single-GPU uses `modelpath` from config; DDP always saves to `outputs/best_model.pth` regardless of `modelpath`
- **Checkpoint contents**: model_state_dict, optimizer, scheduler, normalization stats, model_config, ema_state_dict (if EMA enabled)

### Inference ([inference_profiles/rollout.py](inference_profiles/rollout.py))

Autoregressive rollout: normalize state → build graph → forward pass → denormalize delta → update state → repeat. Outputs HDF5 with predicted trajectories.

## Critical Invariants

These are easy to break if you don't know them:

1. **Edges are always bidirectional**: `edge_index = [mesh_edge; mesh_edge[[1,0]]]`
2. **Node types are one-hot encoded AFTER normalization** and concatenated to normalized features
3. **Edge features are computed from deformed positions** (reference + displacement), not reference positions
4. **Decoder has NO LayerNorm** — this allows full output range for delta prediction
5. **Residual connections on both nodes and edges** (matches DeepMind original)
6. **Delta normalization is separate** from node normalization (delta_mean/delta_std vs node_mean/node_std)
7. **All samples must have equal timestep counts** (current limitation, printed at startup)
8. **HDF5 nodal_data shape**: `[num_features, num_timesteps, num_nodes]` — features-first, not nodes-first
9. **World edges exclude existing mesh edges** to avoid duplication
10. **Config keys are case-insensitive** — `LearningR` becomes `learningr` internally
11. **`world_edge_backend` is required in all configs** — `rollout.py` calls `.get('world_edge_backend').lower()` with no default; omitting it crashes inference with `AttributeError`
12. **For T=1 (static) datasets**, the model input `x` is all-zeros and the target delta equals the feature values themselves
13. **`num_node_types` is assigned to config by code** (not from config file) after dataset load; do not set it manually
14. **Per-feature loss weights are auto-normalized** — specified weights are scaled so they sum to `output_var`, preserving loss magnitude comparability across configs
15. **Loss weights apply to all three phases** (train, validation, test) for consistent optimization and reporting

## HDF5 Dataset Format

Per-sample: `nodal_data` `[features, time, nodes]`, `mesh_edge` `[2, edges]`, `metadata` attributes.
Features: `[x, y, z, x_disp, y_disp, z_disp, stress, (part_number)]`.
Full spec: [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md)

## Normalization

Z-score per feature, computed separately for three domains:
- **Node**: from physical features `nodal_data[3:3+input_var]` across all samples and sampled timesteps (up to 500 per sample)
- **Edge**: from deformed and reference edge positions `[deformed_dx, deformed_dy, deformed_dz, deformed_dist, ref_dx, ref_dy, ref_dz, ref_dist]`
- **Delta**: from actual `state_{t+1} - state_t` transitions (for T=1 datasets, delta = feature value itself)

Stored in checkpoint: `checkpoint['normalization']` dict with `node_mean`, `node_std`, `edge_mean`, `edge_std`, `delta_mean`, `delta_std`, plus `node_type_to_idx` and `world_edge_radius` if applicable.

## Additional Documentation

- [CONFIG_AND_EXECUTION_GUIDE.md](CONFIG_AND_EXECUTION_GUIDE.md): Complete parameter reference
- [docs/MESHGRAPHNET_ARCHITECTURE.md](docs/MESHGRAPHNET_ARCHITECTURE.md): Architecture walkthrough
- [docs/WORLD_EDGES_DOCUMENTATION.md](docs/WORLD_EDGES_DOCUMENTATION.md): Collision detection implementation
- [docs/VRAM_OPTIMIZATION_PLAN.md](docs/VRAM_OPTIMIZATION_PLAN.md): Memory optimization strategies
- [docs/VISUALIZATION_DENORMALIZATION.md](docs/VISUALIZATION_DENORMALIZATION.md): Denormalization for visualization
- [docs/ADAPTIVE_REMESHING_PLAN.md](docs/ADAPTIVE_REMESHING_PLAN.md): Adaptive remeshing plan
- [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md): HDF5 dataset structure
- [misc/README.md](misc/README.md): Visualization tools
