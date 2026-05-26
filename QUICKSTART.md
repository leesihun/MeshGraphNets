# QUICKSTART

This is the short operational guide for the deterministic MeshGraphNets checkout.
For details, read [README.md](README.md), [CLAUDE.md](CLAUDE.md), and
[docs/CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md).

## Fast Commands

Train:

```bash
python MeshGraphNets_main.py --config ex1/config_train1.txt
python MeshGraphNets_main.py --config _warpage_input_deterministic/config_train3.txt
```

Roll out:

```bash
python MeshGraphNets_main.py --config ex1/config_infer1.txt
```

`mode` is read from the config file. The CLI only selects the config path.

## Before Running

Training needs:

- `dataset_dir` exists.
- Every GPU in `gpu_ids` is visible.
- `edge_var` is `8`; the model rejects any other edge feature count.
- `modelpath` parent directory already exists; checkpoint save does not create it.

Inference needs:

- `modelpath` exists and contains `normalization`.
- `infer_dataset` exists and has `data/{id}/nodal_data` plus `mesh_edge`.
- Architecture keys in the checkpoint match the model being loaded. Rollout applies
  `checkpoint['model_config']` over the inference config before building the model.

Quick check:

```bash
dir dataset
dir outputs
```

## Modes

| Mode | Use it for |
|------|------------|
| `train` | Deterministic simulator training. |
| `inference` | Deterministic autoregressive rollout. |

Legacy branch modes, model names, keys, and checkpoints fail fast instead of
being treated as active options.

## Config Gotchas

- Keys are lowercased by the parser.
- Full-line comments use `%`; inline comments use `#`.
- Booleans are parsed case-insensitively.
- Comma lists are stripped and parsed as numbers when possible.
- `gpu_ids 0,1` triggers DDP unless `parallel_mode model_split` is set.
- `message_passing_num` is ignored when `use_multiscale True`; the V-cycle uses
  `mp_per_level`.
- `mp_per_level` must have `2 * multiscale_levels + 1` entries.
- `positional_features` increases node input size before normalization.
- Node type one-hot features are appended after node normalization.
- `parallel_mode model_split` is a memory-fit training path that saves a merged
  checkpoint for normal rollout.

## Outputs

| Output | Source |
|--------|--------|
| `outputs/*.pth` | Checkpoints with weights, optimizer/scheduler state, normalization, model_config, optional EMA |
| `outputs/<log_file_dir>` | Epoch logs from training |
| `outputs/test/...` | Per-epoch test reconstruction HDF5/PNG |
| `outputs/train/...` | Optional train-set reconstruction HDF5/PNG |
| `outputs/rollout/...` | Autoregressive rollout HDF5 |

## First Failure Checks

| Symptom | Check |
|---------|-------|
| Legacy-branch error | Use a current deterministic config or checkpoint. |
| `FileNotFoundError` for HDF5 | `dataset_dir` or `infer_dataset` path is wrong or missing. |
| Checkpoint missing normalization | Re-train or re-save with current training code. |
| Size mismatch on load | Checkpoint `model_config` disagrees with the model architecture. |
| DDP hang | Drop to one GPU, then retry with visible `gpu_ids`; the launcher chooses a free port automatically. |
| NaN/Inf or fp16 overflow | Keep `use_amp True`; current AMP path uses bfloat16, not fp16. |
