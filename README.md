# MeshGraphNets

Deterministic MeshGraphNets implementation for structural mesh surrogate
training and autoregressive rollout. The runtime supports flat message passing,
hierarchical V-cycle processing, optional world edges, node-type features,
activation checkpointing, AMP, EMA evaluation weights, DDP, and experimental
model-split training.

## Entry Point

```bash
python MeshGraphNets_main.py --config _warpage_input/config_train3.txt
python MeshGraphNets_main.py --config _warpage_input/config_infer3.txt
```

Supported modes:

| Mode | Behavior |
| --- | --- |
| `train` | Train the simulator and save deterministic checkpoints. |
| `inference` | Run autoregressive rollout and write one HDF5 output per scene. |

Legacy probabilistic modes and keys are intentionally rejected by the config
loader and checkpoint loader. Old checkpoints from that branch must be retrained
with the deterministic simulator before use here.

## Main Runtime Path

- `MeshGraphNets_main.py` loads config and dispatches training or inference.
- `training_profiles/single_training.py` handles single-process training.
- `training_profiles/distributed_training.py` handles DDP training.
- `parallelism/launcher.py` handles `parallel_mode model_split`.
- `inference_profiles/rollout.py` handles autoregressive rollout.
- `model/MeshGraphNets.py` owns the encode-process-decode simulator.

## Model

The simulator predicts normalized deltas from normalized graph inputs:

```text
graph.x -> encoder -> processor blocks / V-cycle -> decoder -> predicted delta
```

The model returns:

```python
predicted, target = model(graph)
```

`target` is `graph.y` during training/evaluation and `None` during rollout.

Edge features are 8-D:

```text
deformed dx, dy, dz, distance
reference dx, dy, dz, distance
```

## Multiscale

When `use_multiscale True`, the processor runs a V-cycle:

```text
fine pre blocks -> pool -> coarse blocks -> unpool -> fine post blocks
```

`mp_per_level` must contain `2 * multiscale_levels + 1` entries:

```text
pre levels..., coarsest, post levels...
```

Coarsening, world edges, and augmentation behavior are controlled by config
exactly as before; this cleanup does not force new determinism settings.

## Outputs

Training checkpoints contain model state, optimizer/scheduler state,
normalization statistics, and deterministic `model_config` architecture keys.

Rollout writes:

```text
{inference_output_dir}/rollout_sample{sample_id}_steps{N}.h5
```

Each output stores predicted physical state over time plus the normalization
parameters used for the run.
