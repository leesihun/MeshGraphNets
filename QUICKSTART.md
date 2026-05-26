# Quickstart

## Train

```bash
python MeshGraphNets_main.py --config _warpage_input/config_train3.txt
```

Use `mode train` in the config. The trainer builds dataset splits, writes
normalization stats, trains the deterministic simulator, evaluates validation
and test batches, and saves the best checkpoint to `modelpath`.

## Inference

```bash
python MeshGraphNets_main.py --config _warpage_input/config_infer3.txt
```

Use `mode inference` in the config. Rollout loads the checkpoint, applies the
checkpoint architecture config, runs autoregressive prediction, and writes one
HDF5 file per input scene:

```text
outputs/rollout/rollout_sample{sample_id}_steps{steps}.h5
```

## Common Keys

| Key | Purpose |
| --- | --- |
| `model` | Must be `MeshGraphNets`. |
| `mode` | `train` or `inference`. |
| `dataset_dir` | Training dataset HDF5. |
| `infer_dataset` | Rollout input HDF5. |
| `modelpath` | Checkpoint path. |
| `input_var`, `output_var` | Physical feature counts. |
| `edge_var` | Must match the 8-D edge feature implementation. |
| `use_multiscale` | Enables the V-cycle processor. |
| `mp_per_level` | Per-arm processor block counts for multiscale mode. |
| `use_world_edges` | Enables proximity/world-edge message passing. |
| `std_noise`, `augment_geometry` | Existing stochastic training options, preserved as config choices. |

## Notes

The removed probabilistic training and rollout branch is no longer supported.
Configs and checkpoints from that branch fail with a clear error instead of
being silently loaded.
