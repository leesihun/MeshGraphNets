# MeshGraphNets Architecture

The repository now contains a deterministic MeshGraphNets simulator. The model
is an encode-process-decode graph network with optional multiscale processing
and optional world edges.

## Runtime Flow

```text
config -> dataset split -> model -> training loop -> checkpoint
config + checkpoint + input dataset -> rollout -> HDF5 outputs
```

Supported launcher modes are `train` and `inference`.

## Model Contract

`model/MeshGraphNets.py::MeshGraphNets.forward` accepts a PyG graph and returns:

```python
predicted, target = model(graph)
```

`predicted` is a normalized delta with shape `[num_nodes, output_var]`.
`target` is `graph.y` during training/evaluation and `None` during rollout.

## Encoder

The encoder maps raw node and edge features into `latent_dim`.

Node input size is:

```text
input_var + positional_features + optional node type one-hot width
```

Edge input size must match the 8-D edge feature implementation:

```text
deformed dx, dy, dz, distance
reference dx, dy, dz, distance
```

## Processor

Flat mode runs `message_passing_num` `GnBlock` layers.

Multiscale mode replaces the flat stack with a V-cycle:

```text
fine pre blocks
pool to coarse graph
coarse blocks
unpool to fine graph
fine post blocks
```

For `L` levels, `mp_per_level` has `2L + 1` entries:

```text
[pre_0, ..., pre_{L-1}, coarsest, post_{L-1}, ..., post_0]
```

Coarse edge features are recomputed from reference and deformed centroids. Skip
connections merge fine-level features with unpooled coarse features before the
post blocks.

## Decoder

The decoder maps latent node embeddings to normalized output deltas. For
multi-step datasets, the final decoder layer is initialized near zero to start
from a small-delta prior.

## Training Loss

Training and validation use Huber loss on normalized deltas. Optional
per-feature weights are normalized and applied per node before aggregation.

## Checkpoint Compatibility

The checkpoint loader rejects artifacts from the removed probabilistic branch.
Use a checkpoint produced by the deterministic simulator in this repo.
