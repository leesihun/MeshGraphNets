# MeshGraphNets

A Graph Neural Network surrogate model for FEA (Finite Element Analysis) mesh simulations, based on the [DeepMind MeshGraphNets](https://arxiv.org/abs/2010.03409) architecture with extensions for multiscale processing, positional encoding, mixed-precision training, and distributed training.

Implemented by SiHun Lee, MX, SE.

## What it does

Given a mesh at time *t* (node positions + physical state), the model predicts the state at time *t+1* as a **delta**. At inference time it rolls out autoregressively over many timesteps. The primary application is warpage/deformation prediction in manufacturing FEA.

## Setup

Requires PyTorch + PyTorch Geometric. Common extras:

```bash
# HDF5 datasets + rollout I/O
pip install h5py scipy

# World edges (long-range connections)
pip install torch-cluster

# GIF visualization utilities
pip install matplotlib pillow
```

Set `HDF5_USE_FILE_LOCKING=FALSE` in your environment if running on shared storage.

## Running

```bash
# Training example (flag_simple cloth dataset)
python MeshGraphNets_main.py --config _warpage_input/config_train_flag_simple.txt

# Inference / autoregressive rollout example (ConcreteShellFEA)
python MeshGraphNets_main.py --config _warpage_input/config_infer_concreteshellfea.txt
```

The same entry point handles single GPU, multi-GPU DDP, or CPU based on `gpu_ids`. Use `gpu_ids -1` to force CPU. The `--config` flag defaults to `config.txt`, and `mode` is set inside the config file.

For inference, the checkpoint overrides overlapping model config keys. If `positional_features > 0`, rollout recomputes positional features from the reference mesh before normalization, so `positional_features`, `positional_encoding`, and `use_node_types` should stay aligned with the training run.

New example configs:
- `_warpage_input/config_train_flag_simple.txt` - multiscale `flag_simple` training with node types, world edges, AMP, and EMA
- `_warpage_input/config_infer_concreteshellfea.txt` - ConcreteShellFEA rollout with multiscale inference and positional features

See [config_run_docs.md](config_run_docs.md) for a full reference of all configuration keys.

## Dataset Format

HDF5 samples live under `data/<id>/` and store:
- `nodal_data` with shape **`[features, time, nodes]`**
- `mesh_edge` with shape **`[2, edges]`**

`nodal_data` uses this channel layout:

| Index | Feature |
|---|---|
| 0–2 | x, y, z (reference coordinates) |
| 3–5 | x\_disp, y\_disp, z\_disp |
| 6 | stress (von Mises or equivalent) |
| 7 | part\_number (optional) |

## Outputs

Training checkpoints are saved to the path specified by `modelpath`. Each checkpoint contains:
- Model weights (and optionally `ema_state_dict`)
- Optimizer state
- Normalization statistics under `checkpoint['normalization']`
- Per-level coarse edge stats if multiscale is enabled

Inference outputs are written to `inference_output_dir` (default `outputs/rollout`) as `rollout_sample{id}_steps{N}.h5`.

Saved rollout files keep the standard 8-row `nodal_data` layout:
- Rows `0-2` store the reference coordinates for every timestep
- Predicted outputs start at row `3`
- Any unpredicted rows are copied from the input sample at `t=0`
- Part IDs are preserved when present

Loss and validation curves are written to the file specified by `log_file_dir`. Use `misc/plot_loss.py` or `misc/plot_loss_realtime.py` to visualize.

## Utility Scripts

The old root-level `animate_h5.py` now lives at `scripts/animate_h5.py`.

```bash
# Create GIF views from a rollout HDF5
python scripts/animate_h5.py outputs/rollout/rollout_sample0_steps34.h5 --views 3d_iso xz --color-by displacement

# Concatenate same-prefix *_3D_iso.gif clips into one timeline
python scripts/append_prefix_gifs.py --prefix flag_simple sphere_simple
```

`scripts/animate_h5.py` supports `xz`, `xy`, `yz`, `3d_iso`, and `3d_top` views, plus `auto` or displacement-magnitude coloring. `scripts/append_prefix_gifs.py` combines matching GIF clips in the repo root.

## Architecture

**Encode–Process–Decode** GNN operating on FEA mesh graphs:

- **Encoder:** Independent MLPs encode node features and edge features to a shared latent dimension.
- **Processor:** Stack of GnBlocks (EdgeBlock → NodeBlock with residual connections). Supports a multiscale BFS Bi-Stride V-cycle (ICML 2023) for hierarchical processing.
- **Decoder:** MLP from latent to predicted normalized state delta. No LayerNorm on the output layer.

**Edge features** are 8D: `[deformed_dx/dy/dz/dist, ref_dx/dy/dz/dist]`. Edges are always bidirectional.

**Prediction:** Normalized delta (`Δstate`). Denormalized and added to current state: `state_{t+1} = state_t + delta`.

## Key Config Sections

| Section | Purpose |
|---|---|
| Mode / GPU | `mode`, `gpu_ids` |
| Paths | `modelpath`, `dataset_dir`, `infer_dataset`, `log_file_dir`, `inference_output_dir` |
| Model size | `Latent_dim`, `message_passing_num` |
| Training | `LearningR`, `Training_epochs`, `Batch_size`, `use_amp`, `use_ema` |
| Features | `input_var`, `output_var`, `edge_var`, `positional_features`, `positional_encoding`, `use_node_types` |
| Multiscale | `use_multiscale`, `multiscale_levels`, `mp_per_level` |
| Inference | `infer_timesteps` |

Full documentation: [config_run_docs.md](config_run_docs.md)
