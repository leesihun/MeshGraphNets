# Repository Notes

This repo is now deterministic MeshGraphNets only. The old probabilistic
application has been removed from runtime code, configs, and docs. Keep future
changes on the deterministic simulator path unless the architecture is
intentionally reintroduced as a separate project.

## Launch Modes

| Mode | Path |
| --- | --- |
| `train` | `training_profiles.single_training.single_worker` or DDP worker |
| `inference` | `inference_profiles.rollout.run_rollout` |

No alternate prior-training modes are supported.

## Core Files

| File | Purpose |
| --- | --- |
| `MeshGraphNets_main.py` | Config loading and launch dispatch. |
| `model/MeshGraphNets.py` | Top-level encode-process-decode model. |
| `model/encoder_decoder.py` | Encoder, decoder, and graph network block definitions. |
| `model/coarsening.py` | BFS and Voronoi hierarchy builders. |
| `training_profiles/training_loop.py` | Train, validate, and test helpers. |
| `inference_profiles/rollout.py` | Autoregressive rollout and HDF5 output writer. |
| `parallelism/` | DDP-adjacent model-split training helpers. |

## Model Contract

The model returns two values:

```python
predicted, target = model(graph)
```

`target` is available when `graph.y` exists and is `None` during rollout.
Training loss is Huber loss over normalized deltas.

## Compatibility

The config loader rejects removed probabilistic keys. The checkpoint loader also
rejects old checkpoint artifacts from that branch. Do not silently ignore those
fields; retrain a deterministic checkpoint instead.
