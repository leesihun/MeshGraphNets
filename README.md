# MeshGraphNets

This repository implements a deterministic MeshGraphNets simulator for FEA-style
mesh data. The runtime predicts normalized physical deltas from graph inputs and
supports flat message passing, hierarchical V-cycle message passing, optional
world edges, node-type features, EMA checkpoints, DDP, and an experimental
model-split training path.

Implemented by SiHun Lee, MX, SEC.

The executable entry point is [MeshGraphNets_main.py](MeshGraphNets_main.py).

## Runtime Contract

This checkout is deterministic only. Legacy branch artifacts are rejected at
config and checkpoint load time through
[general_modules/removed_feature_guard.py](general_modules/removed_feature_guard.py).
The exact deny-list lives in that guard so old files fail loudly without keeping
removed features in the user-facing docs.

Supported modes:

| Mode | What runs |
|------|-----------|
| `train` | Simulator training through single-process, DDP, or model-split launch. |
| `inference` | Deterministic autoregressive rollout from an HDF5 initial-condition dataset. |

## Quick Start

Example deterministic training configs:

```bash
python MeshGraphNets_main.py --config ex1/config_train1.txt
python MeshGraphNets_main.py --config _warpage_input_deterministic/config_train3.txt
```

Example deterministic rollout config:

```bash
python MeshGraphNets_main.py --config ex1/config_infer1.txt
```

`mode` is read from the config file. `--config` only selects the file.

## Architecture

The model is implemented in [model/MeshGraphNets.py](model/MeshGraphNets.py).

Node input features are:

```text
physical state channels from nodal_data[3:3+input_var]
+ optional positional features
+ optional one-hot node types
```

Edge features are always 8-D and validated against `edge_var 8`:

```text
[deformed_dx, deformed_dy, deformed_dz, deformed_dist,
 ref_dx,      ref_dy,      ref_dz,      ref_dist]
```

The base model is:

```text
Encoder: node MLP + mesh edge MLP (+ world edge MLP when enabled)
Processor:
  flat: message_passing_num GnBlocks
  multiscale: pre blocks -> pool -> coarsest blocks -> unpool/skip -> post blocks
Decoder: node MLP to normalized delta
```

MLPs are built by [model/mlp.py](model/mlp.py): `Linear -> SiLU -> Linear -> SiLU
-> Linear`, with final `LayerNorm` only when `layer_norm=True`. The decoder omits
final LayerNorm.

## Data

HDF5 samples use:

```text
data/{sample_id}/nodal_data    float32 [features, timesteps, nodes]
data/{sample_id}/mesh_edge     int64   [2, edges]
```

Default feature layout:

| Index | Field | Used as |
|-------|-------|---------|
| `0:3` | reference `x,y,z` | Geometry only, never part of `input_var` |
| `3:6` | displacement `x,y,z` | Default physical input and output |
| `6` | stress | Used when `input_var/output_var` include it |
| `7` | part number | One-hot node type source when `use_node_types True` |

For multi-step data, each training item is `(sample_id, t)` and the target is
`state[t+1] - state[t]`. For single-step data, the input physical state is zero
and the target is the stored state.

More detail: [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md).

## Training

Single-process training is in
[training_profiles/single_training.py](training_profiles/single_training.py).
DDP training is in
[training_profiles/distributed_training.py](training_profiles/distributed_training.py).

`gpu_ids` controls the launcher:

| Value | Behavior |
|-------|----------|
| `-1` | CPU rollout; training uses CPU only when CUDA is unavailable |
| `0` | Single GPU |
| `0,1` | PyTorch DDP via `mp.spawn` |

`parallel_mode model_split` activates [parallelism/launcher.py](parallelism/launcher.py).
It slices processor blocks across GPUs for memory fit and saves a merged checkpoint
that normal rollout can load.

Training uses Huber loss on normalized deltas, optional normalized feature
weights, Adam, linear warmup into cosine restarts, `max_norm=3.0` gradient
clipping, optional bfloat16 AMP, and optional EMA.

## Inference

[inference_profiles/rollout.py](inference_profiles/rollout.py) performs
deterministic autoregressive rollout:

```text
state_t -> normalize graph -> model predicts normalized delta
        -> denormalize delta -> state_{t+1} = state_t + delta
```

For each input scene, rollout writes:

```text
{inference_output_dir}/rollout_sample{sample_id}_steps{N}.h5
```

Rollout loads checkpoint normalization stats and applies `checkpoint['model_config']`
before constructing the model, so the saved architecture is the inference truth.

## Documentation Map

| File | Purpose |
|------|---------|
| [QUICKSTART.md](QUICKSTART.md) | Short run guide and failure checks |
| [CLAUDE.md](CLAUDE.md) | Agent-facing engineering map |
| [docs/CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md) | Current config keys and legacy-input rejection |
| [docs/MESHGRAPHNET_ARCHITECTURE.md](docs/MESHGRAPHNET_ARCHITECTURE.md) | Architecture details grounded in live code |
| [docs/multiscale_coarsening.md](docs/multiscale_coarsening.md) | Hierarchical V-cycle and coarsening |
| [docs/WORLD_EDGES_DOCUMENTATION.md](docs/WORLD_EDGES_DOCUMENTATION.md) | World-edge runtime path |
| [hierarchical_interpolation_mgn_comparison.md](hierarchical_interpolation_mgn_comparison.md) | Paper-style comparison for deterministic hierarchical MGN |

## Installation

Install a PyTorch build matching your CUDA environment, then install project deps:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`torch-cluster` is optional unless `world_edge_backend torch_cluster` is requested.
The code falls back to scipy KDTree world-edge construction when torch-cluster is
not available.
