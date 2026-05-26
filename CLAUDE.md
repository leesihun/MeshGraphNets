# CLAUDE.md

Agent-facing notes for this deterministic MeshGraphNets repository. Keep answers
and edits grounded in the live code, not older prose in historical plan docs.

## Project Objective

This checkout is a deterministic MeshGraphNets simulator. It predicts physical
deltas on FEA-style meshes from normalized graph inputs. Removed latent-sampling
training and inference paths should not be reintroduced unless explicitly
requested as a new feature.

Removed branch artifacts are rejected by
[general_modules/removed_feature_guard.py](general_modules/removed_feature_guard.py).

## Run Commands

```bash
python MeshGraphNets_main.py --config ex1/config_train1.txt
python MeshGraphNets_main.py --config _warpage_input_deterministic/config_train3.txt
python MeshGraphNets_main.py --config ex1/config_infer1.txt
```

`mode` is inside the config. `--config` only picks the file.

## Runtime Modes

| Mode | Code path |
|------|-----------|
| `train` | `single_worker`, DDP `train_worker`, or `parallelism.launcher.launch_model_split`. |
| `inference` | `inference_profiles.rollout.run_rollout`. |

`gpu_ids` length controls single process vs DDP unless `parallel_mode model_split`
is set. `parallel_mode model_split` slices processor blocks across GPUs and saves
a merged checkpoint for standard rollout.

## Key Files

| File | Role |
|------|------|
| [MeshGraphNets_main.py](MeshGraphNets_main.py) | Config load, mode dispatch, DDP/model-split launch. |
| [model/MeshGraphNets.py](model/MeshGraphNets.py) | Top-level deterministic model, flat/multiscale processor. |
| [model/encoder_decoder.py](model/encoder_decoder.py) | Encoder, GnBlock, Decoder. |
| [model/blocks.py](model/blocks.py) | EdgeBlock, NodeBlock, HybridNodeBlock, UnpoolBlock. |
| [model/coarsening.py](model/coarsening.py) | BFS and Voronoi coarsening, pool/unpool, `MultiscaleData`. |
| [general_modules/mesh_dataset.py](general_modules/mesh_dataset.py) | HDF5 loading, split, normalization, positional features, world/multiscale attrs. |
| [general_modules/edge_features.py](general_modules/edge_features.py) | 8-D reference/deformed edge features. |
| [general_modules/world_edges.py](general_modules/world_edges.py) | scipy KDTree or torch-cluster world-edge construction. |
| [general_modules/multiscale_helpers.py](general_modules/multiscale_helpers.py) | Shared hierarchy build/attach used by dataset and rollout. |
| [training_profiles/setup.py](training_profiles/setup.py) | Dataset/model/EMA/optimizer/checkpoint helpers. |
| [training_profiles/training_loop.py](training_profiles/training_loop.py) | Train, validate, test visualization. |
| [inference_profiles/rollout.py](inference_profiles/rollout.py) | Deterministic autoregressive rollout and HDF5 output. |
| [parallelism/](parallelism) | Experimental model-split training across GPUs. |

## Architecture Facts

- Edge features are 8-D. `edge_var` must be `8`.
- Node input size is `input_var + positional_features + optional num_node_types`.
- MLPs use SiLU: `Linear -> SiLU -> Linear -> SiLU -> Linear`.
- LayerNorm is appended once at the MLP output when `layer_norm=True`.
- Decoder output has no LayerNorm. For time-transient runs, decoder final weights
  are scaled by `0.01` at initialization.
- Node aggregation is sum.
- World edges use a separate edge encoder/block and a hybrid node block that
  aggregates mesh and world messages separately.
- Multiscale V-cycle uses per-level skip states merged by
  `Linear(2 * latent_dim, latent_dim)`.

## Data Facts

- HDF5 stores `mesh_edge` as one-directional unique edges; loaders mirror it to
  bidirectional edges.
- `nodal_data[0:3]` are reference coordinates and are not part of `input_var`.
- Physical channels start at `nodal_data[3]`.
- Multi-step targets are `state[t+1] - state[t]`.
- Single-step targets are the stored state from zero input.
- The dataset class ignores HDF5 split datasets and creates a seeded 80/10/10 split.
- Training writes normalization arrays under `metadata/normalization_params`, but
  split IDs are not written by the current training path.

## Removed Feature Boundary

Do not document or use legacy branch modes, model names, keys, checkpoint
artifacts, or state-dict prefixes as active features in this checkout. If an old
config or checkpoint contains them, the correct behavior is a loud error from the
removed-feature guard.

## Documentation Notes

The authoritative docs are:

- [README.md](README.md)
- [QUICKSTART.md](QUICKSTART.md)
- [docs/CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md)
- [docs/MESHGRAPHNET_ARCHITECTURE.md](docs/MESHGRAPHNET_ARCHITECTURE.md)
- [docs/multiscale_coarsening.md](docs/multiscale_coarsening.md)
- [docs/WORLD_EDGES_DOCUMENTATION.md](docs/WORLD_EDGES_DOCUMENTATION.md)
- [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md)

Files named `*_PLAN.md` or `*_RESEARCH.md` can be historical and should not be
treated as current implementation truth without checking code.
