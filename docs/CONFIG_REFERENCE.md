# Configuration Reference

This file describes the live deterministic MeshGraphNets runtime. Legacy branch
config and checkpoint artifacts are rejected by
[general_modules/removed_feature_guard.py](../general_modules/removed_feature_guard.py).

## Parser Rules

[general_modules/load_config.py](../general_modules/load_config.py) reads whitespace
or tab separated `key value` pairs.

- Keys are lowercased.
- Lines starting with `%` are skipped.
- Inline `#` comments are stripped.
- Booleans are parsed case-insensitively.
- Comma-separated values become lists when possible.
- Space-separated multi-values become lists when possible.

## Supported Modes

| Key | Scope | Values |
|-----|-------|--------|
| `mode` | launcher | `train` or `inference` |
| `model` | launcher | `MeshGraphNets` |
| `gpu_ids` | launcher | `-1`, `0`, or comma-separated GPU ids |
| `parallel_mode` | launcher | `ddp` or `model_split` |

Other legacy mode or model names are invalid in this checkout.

## Dataset Keys

| Key | Scope | Meaning |
|-----|-------|---------|
| `dataset_dir` | training | Training HDF5 path. |
| `infer_dataset` | inference | Rollout HDF5 path. |
| `infer_timesteps` | inference | Rollout steps. If omitted and input has multiple timesteps, rollout uses the full available trajectory length. |
| `split_seed` | dataset | Seed for deterministic 80/10/10 sample-id split. Default `42`. |
| `input_var` | dataset/model | Number of physical input channels starting at `nodal_data[3]`. |
| `output_var` | dataset/model | Number of predicted delta channels. |
| `edge_var` | dataset/model | Must be `8`. |
| `positional_features` | dataset/model | Number of geometry/topology positional features appended before normalization. |
| `positional_encoding` | dataset | Positional feature mode, default `rwpe`. |
| `use_node_types` | dataset/model | Append one-hot part IDs from the last nodal channel. |
| `num_node_types` | model | Filled from training data when node types are enabled. |

## Model Keys

| Key | Meaning |
|-----|---------|
| `latent_dim` | Hidden width for encoders, processor blocks, and decoder. Configs may write `Latent_dim`; the parser lowercases it. |
| `message_passing_num` | Number of flat processor blocks. Ignored when `use_multiscale True`. |
| `residual_scale` | Residual multiplier inside `GnBlock`. |
| `use_world_edges` | Enables world-edge construction and hybrid node updates. |
| `world_edge_radius` | Radius saved in training normalization metadata and used by rollout. |
| `world_max_num_neighbors` | Neighbor cap for world-edge construction. |
| `world_edge_backend` | `scipy_kdtree` or `torch_cluster`; rollout falls back to scipy when torch-cluster is unavailable. |
| `use_multiscale` | Enables hierarchical V-cycle processor. |
| `multiscale_levels` | Number of coarsening levels. |
| `mp_per_level` | `2 * multiscale_levels + 1` entries: pre levels, coarsest, post levels. |
| `fine_mp_pre` | Default fine pre-block count when `mp_per_level` is absent. |
| `coarse_mp_num` | Default coarsest block count when `mp_per_level` is absent. |
| `fine_mp_post` | Default fine post-block count when `mp_per_level` is absent. |
| `coarsening_type` | `bfs`, `voronoi_centroid` (alias `voronoi`), `voronoi_inherit`, or `voronoi_seedmean`; scalar or per-level list. |
| `voronoi_clusters` | Coarse node count for Voronoi coarsening, scalar or per-level list. |
| `bipartite_unpool` | Enables learned bipartite unpooling instead of broadcast unpooling. |
| `coarse_world_edges` | Enables world edges on coarse levels when world edges and multiscale are both enabled. |

## Training Keys

| Key | Meaning |
|-----|---------|
| `training_epochs` | Number of epochs. |
| `batch_size` | PyG DataLoader batch size. |
| `learningr` | Adam learning rate. |
| `num_workers` | DataLoader workers. |
| `std_noise` | Training input/edge noise std. |
| `noise_gamma` | Target correction multiplier for input noise. |
| `feature_loss_weights` | Optional per-output feature weights normalized to sum to one. |
| `grad_accum_steps` | `1` means per-batch step, `N` means accumulate N batches, `0` means one optimizer step per epoch. |
| `use_checkpointing` | Gradient checkpointing for processor blocks. |
| `use_amp` | bfloat16 autocast when CUDA is available. |
| `use_ema` | Saves an EMA shadow model. Rollout prefers EMA weights. |
| `ema_decay` | EMA decay. |
| `warmup_epochs` | Linear warmup epochs before cosine restarts. |
| `test_interval` | Test visualization interval. |
| `val_interval` | Validation interval. |
| `test_max_batches` | Cap for test visualization/evaluation batches. |
| `display_testset` | Enables test visualization rendering. |
| `display_trainset` | Enables train reconstruction visualization. |
| `test_batch_idx` | Sample indices to save for visualization. |
| `plot_feature_idx` | Feature index for visualization, `-1` for last feature. |
| `log_file_dir` | Training log path under `outputs/`. |
| `modelpath` | Checkpoint output path for training and input path for inference. |

## Inference Keys

| Key | Meaning |
|-----|---------|
| `modelpath` | Checkpoint to load. Must include `normalization`. |
| `infer_dataset` | HDF5 source for initial conditions. |
| `infer_timesteps` | Number of autoregressive steps. |
| `inference_output_dir` | Rollout output directory. Default `outputs/rollout`. |

Rollout writes one deterministic file per input scene:

```text
rollout_sample{sample_id}_steps{steps}.h5
```

## Checkpoint Contents

Training saves:

- `model_state_dict`
- optional `ema_state_dict`
- optimizer and scheduler state
- `train_loss`, `valid_loss`, `epoch`
- `normalization`
- `model_config`
- model-split metadata when trained with `parallel_mode model_split`

`normalization` includes node, edge, and delta stats plus optional node-type,
world-edge, and coarse-edge stats. `model_config` is applied by rollout before
the model is constructed.

## Legacy Input Rejection

Active configs should contain only the keys documented above. Old branch modes,
model names, config keys, checkpoint artifacts, and state-dict prefixes are
rejected before training or model loading. The deny-list is intentionally kept in
[general_modules/removed_feature_guard.py](../general_modules/removed_feature_guard.py)
instead of repeated in docs.
