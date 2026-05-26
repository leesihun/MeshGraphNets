# MeshGraphNets Architecture

This document reflects the live deterministic code path.

## Entry Points

| Runtime | Code |
|---------|------|
| Launcher | [MeshGraphNets_main.py](../MeshGraphNets_main.py) |
| Model | [model/MeshGraphNets.py](../model/MeshGraphNets.py) |
| Encoder/processor/decoder blocks | [model/encoder_decoder.py](../model/encoder_decoder.py) |
| Coarsening | [model/coarsening.py](../model/coarsening.py) |
| World edges | [general_modules/world_edges.py](../general_modules/world_edges.py) |
| Rollout | [inference_profiles/rollout.py](../inference_profiles/rollout.py) |

Supported modes are `train` and `inference`.

## Forward Contract

`MeshGraphNets.forward(graph, debug=False, add_noise=None)` returns:

```text
predicted_normalized_delta, target_normalized_delta_or_None
```

Training graphs include `graph.y`; rollout graphs do not. The inner
`EncoderProcessorDecoder` returns only the predicted normalized delta.

## Inputs

Node input size:

```text
input_var + positional_features + optional num_node_types
```

Edge input size is fixed to `8`:

```text
[deformed_dx, deformed_dy, deformed_dz, deformed_dist,
 ref_dx,      ref_dy,      ref_dz,      ref_dist]
```

The model raises if `edge_var` is not `8`.

## Encoder

`Encoder` maps raw graph features into `latent_dim`:

- node encoder: node input size to `latent_dim`
- mesh edge encoder: 8-D edge features to `latent_dim`
- optional world edge encoder: 8-D world-edge features to `latent_dim`

## Processor

### Flat Processor

When `use_multiscale False`, the model runs `message_passing_num` `GnBlock`
modules in sequence.

Each `GnBlock` updates:

1. mesh edge features from sender node, receiver node, and edge features
2. optional world edge features
3. node features from current node features plus aggregated edge messages

Node aggregation uses sum.

### Multiscale V-Cycle

When `use_multiscale True`, the processor is:

```text
fine pre blocks
-> pool to coarse graph
-> coarsest blocks
-> unpool to fine graph
-> skip merge
-> fine post blocks
```

For `L` coarsening levels, `mp_per_level` must contain `2 * L + 1` entries:

```text
[pre_0, pre_1, ..., coarsest, post_{L-1}, ..., post_0]
```

Pooling uses `pool_features`. Broadcast unpool uses `unpool_features`.
When `bipartite_unpool True`, `UnpoolBlock` uses coarse-to-fine bipartite edges
and relative positions. Each skip merge concatenates the saved fine state with the
unpooled coarse state and projects `2 * latent_dim` back to `latent_dim`.

## World Edges

World edges connect spatial neighbors beyond mesh adjacency. The model has a
separate encoder and edge block for world-edge attributes. `HybridNodeBlock`
aggregates mesh and world messages separately before updating nodes.

Rollout reconstructs world edges from the current deformed position and checkpoint
normalization metadata.

## Decoder

`Decoder` maps final latent node states to `output_var` normalized delta channels.
For time-transient delta prediction, the decoder final layer weights are scaled by
`0.01` at initialization so early predictions start near no change.

## Loss

Training uses Huber loss on normalized deltas. Optional `feature_loss_weights`
are normalized to sum to one before reducing per-node feature losses.

## Checkpointing And Parallelism

Gradient checkpointing recomputes processor blocks during backward to reduce
activation memory.

`parallel_mode model_split` partitions processor blocks across GPUs. The split
path sends deterministic activation bundles across ranks and saves a merged
single-model checkpoint so rollout does not need to know how training was split.

## Legacy Branch Boundary

This checkout has no active latent-sampling path. Legacy configs, model names,
checkpoint keys, and state-dict prefixes are rejected by
[general_modules/removed_feature_guard.py](../general_modules/removed_feature_guard.py).
