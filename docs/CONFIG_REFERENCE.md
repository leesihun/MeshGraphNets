# Config Reference

The config format is line-oriented:

```text
key value
```

Blank lines and `%` comments are ignored. Inline `#` comments are removed before
parsing. Keys are normalized to lowercase by `general_modules/load_config.py`.

## Launch Keys

| Key | Values |
| --- | --- |
| `model` | `MeshGraphNets` |
| `mode` | `train` or `inference` |
| `gpu_ids` | One GPU id for single-process runs, comma-separated ids for DDP/model split, or `-1` for CPU. |
| `parallel_mode` | `ddp` by default, or `model_split` for the experimental pipeline split path. |

The loader rejects old probabilistic branch keys and model names.

## Dataset Keys

| Key | Purpose |
| --- | --- |
| `dataset_dir` | Training HDF5 file. |
| `infer_dataset` | Rollout input HDF5 file. |
| `infer_timesteps` | Rollout step count. If omitted, rollout uses the dataset trajectory length when available. |
| `split_seed` | Seed used for train/validation/test split construction. |

## Feature Keys

| Key | Purpose |
| --- | --- |
| `input_var` | Number of physical input state channels. |
| `output_var` | Number of predicted delta channels. |
| `edge_var` | Must be `8` for the current edge feature implementation. |
| `feature_loss_weights` | Optional per-output-channel Huber loss weights. |
| `positional_features` | Number of static positional/topological features appended to node input. |
| `positional_encoding` | Positional feature mode used by the dataset code. |
| `use_node_types` | Adds node-type one-hot features when the dataset provides part ids. |

## Model Keys

| Key | Purpose |
| --- | --- |
| `message_passing_num` | Flat processor block count. Ignored when multiscale mode is enabled. |
| `latent_dim` | Hidden feature size. |
| `residual_scale` | Residual update scale inside graph blocks. |
| `use_world_edges` | Enables proximity/world-edge message passing. |
| `world_edge_radius` | Radius for world-edge construction when stored/configured. |
| `world_edge_backend` | `scipy_kdtree` or `torch_cluster` when available. |
| `world_max_num_neighbors` | Neighbor cap for world-edge construction. |

## Multiscale Keys

| Key | Purpose |
| --- | --- |
| `use_multiscale` | Enables the V-cycle processor. |
| `multiscale_levels` | Number of hierarchy levels. |
| `mp_per_level` | `2 * multiscale_levels + 1` block counts. |
| `coarsening_type` | `bfs`, `voronoi`, or a per-level list. |
| `voronoi_clusters` | Target coarse node count for Voronoi levels. |
| `bipartite_unpool` | Uses learned coarse-to-fine interpolation when true. |
| `coarse_world_edges` | Carries world edges onto coarse levels when enabled. |

## Training Keys

| Key | Purpose |
| --- | --- |
| `training_epochs` | Number of training epochs. |
| `batch_size` | Batch size. |
| `learningr` | Learning rate. |
| `num_workers` | DataLoader workers. |
| `std_noise` | Existing input/edge noise option for training. |
| `augment_geometry` | Existing dataset augmentation option. |
| `grad_accum_steps` | Gradient accumulation window. `0` means one optimizer step per epoch. |
| `use_checkpointing` | Recompute processor activations in backward to reduce memory. |
| `use_amp` | Uses bfloat16 autocast. |
| `use_ema` | Maintains EMA weights for evaluation and checkpointing. |
| `ema_decay` | EMA decay. |
| `use_compile` | Applies `torch.compile(dynamic=True)` to the model. |
| `val_interval` | Validation interval in epochs. |
| `test_interval` | Test/visualization interval in epochs. |

## Checkpoints And Rollout

Checkpoints store:

- model state
- optional EMA state
- optimizer and scheduler state
- train and validation losses
- normalization statistics
- deterministic architecture config

Rollout writes one file per scene:

```text
rollout_sample{sample_id}_steps{steps}.h5
```
