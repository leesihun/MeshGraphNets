# MeshGraphNets Enhancement Research for Quasi-Static Warpage Prediction

Status as of 2026-05-26: this is a research and experiment-prioritization note,
not a description of the current implemented architecture. The current repo
already implements several items that this note originally proposed: optional
rotation-invariant positional features (`positional_features` with `rwpe`,
`lpe`, or `rwpe+lpe`), optional node-type one-hot features, the multiscale
V-cycle, learned bipartite unpooling, and optional world edges. Treat the paper
comparisons below as motivation for future experiments, then verify live behavior
in `general_modules/mesh_dataset.py`, `model/MeshGraphNets.py`, and
`general_modules/world_edges.py`.

## Core Diagnosis

MeshGraphNets (Pfaff et al., ICLR 2021) was designed for **dynamic** simulations where information propagates locally per timestep. Warpage is a **static equilibrium** problem — the entire displacement field is determined simultaneously by global boundary conditions. Standard message passing cannot propagate information far enough before over-smoothing kills the signal.

This is a **documented failure mode**: Gladstone et al. (Scientific Reports, 2024) states:
> "When a physical system requires long-range interactions, such as in static solid mechanics problems, standard mesh-based GNN approaches typically fail at capturing the physics accurately, because the exchange of information between distant nodes requires a large number of message passing steps."

**Historical evidence that motivated this work:**
- `message_passing_num` 3 vs 15 gave similar results in earlier flat-model runs.
- Validation showed "completely mixed" correlation, meaning the model predicted a
  nearly uniform field.
- A simple Graph U-Net-style experiment outperformed the earlier flat MGN path.

Current mitigation in this repo is no longer just "increase depth": use
`positional_features`, `use_multiscale`, `mp_per_level`, `bipartite_unpool`,
optional `use_world_edges`, and explicit `use_node_types` when the HDF5 data
contains meaningful part or boundary labels.

---

## Key Paper: Gladstone et al. (Scientific Reports, 2024)

**Title:** "Mesh-based GNN surrogates for time-independent PDEs"
**Authors:** Gladstone, Rahmani, Suryakumar, Meidani, D'Elia, Zareei
**URL:** https://www.nature.com/articles/s41598-024-53185-y

### What they propose

Two architectures that outperform baseline MeshGraphNets on static solid mechanics:

#### 1. Edge Augmented GNN (EA-GNN) — best performer
- Adds **random "virtual" edges between distant nodes** (20% augmentation ratio)
- Each virtual edge has a binary flag (1=augmented, 0=mesh edge) as extra edge feature
- Reduces hops required for long-range information propagation
- Same encoder-processor-decoder framework, 6 GnBlocks
- **5x error reduction** vs baseline MeshGraphNets

#### 2. Multi-GNN (M-GNN) — hierarchical
- Multigrid-inspired: graph down-sampling → coarse processing → up-sampling
- Uses GraphSAGE operator (node-only, no edge features since edges change during sampling)
- Computes l-th graph power (l=3) creating links up to 3 hops away
- **2x error reduction** vs baseline MeshGraphNets

### Simulation Coordinates (critical finding)

A coordinate transformation that provides translation/rotation invariance:
1. Center to centroid: `Xc = X - X_mean`
2. Rotate to principal axes: compute eigenvectors of `Xc^T @ Xc`, rotate to that basis

**This transformation ALONE (no architecture change) provides 2-3x improvement!**

| Model | Error (ux) | Error (uy) |
|-------|-----------|-----------|
| Baseline MeshGraphNets | 0.64 | 0.63 |
| Baseline + Simulation Coords | 0.25 | 0.26 |
| EA-GNN + Simulation Coords | **0.05** | **0.05** |
| M-GNN + Simulation Coords | 0.13 | 0.13 |

### Their Node Features (14 total)
- Nodal positions in **simulation coordinates** (x, y)
- Nodal type (interior/boundary)
- Boundary condition type (Dirichlet homogeneous/non-homogenous, Neumann)
- Boundary condition direction and magnitude
- Body force flag and magnitude/direction

### Their Edge Features (4 for EA-GNN)
- Euclidean distance between nodes
- Positional differences (dx, dy)
- Augmentation flag (1 for virtual, 0 for mesh)

### Architecture Details (from paper)

**EA-GNN specifics:**
- Encoder MLPs: node 14→64→128, edge 4→64→128 (single hidden layer)
- 6 GN blocks with SHARED parameters across blocks
- Skip connections between GN blocks (not just residuals within)
- Edge update: 3×128→128→128, Node update: 3 MLPs (φ, γ, β) each 2×128→128→128
- Dropout 0.1 after encoder and between GN blocks
- Decoder: 128→64→2 (displacement) or 128→64→3 (stress)
- Training: Adam, LR 1e-4 to 1.5e-4, weight decay 1e-5, cosine annealing warm restart
- Loss: Scaled MAE (not MSE), scaled by boundary condition magnitudes
- 1500 epochs on Tesla V100

**M-GNN specifics:**
- No edge attributes (GraphSAGE — edges lost during pooling/unpooling)
- Depth d=3 hierarchical levels
- l-th graph power (l=3) connects nodes up to 3 hops away at each level
- Trainable vector p for adaptive node selection during down-sampling
- LR 2e-3 to 3e-3, weight decay 1e-6

**Training data scale:**
- 5,000 random Bezier-curve geometries × 10 boundary conditions each = **50,000 samples**
- ~1,100 nodes per mesh
- 70/10/20 train/val/test split
- Coordinate noise augmentation: N(0, 0.01) ≈ 10% of edge distance

**Critical ablation finding:** EA-GNN with GraphSAGE (no edge attributes) performs similarly
to EA-GNN with edge attributes → the augmented EDGES (connectivity), not edge features, are what matters.

### Key Takeaway for Warpage
They DO use node positions as features — but in a **simulation coordinate frame** (centered + PCA-rotated), not raw world coordinates. This preserves translation invariance (centered) and rotation invariance (PCA-aligned). Their node features also include rich boundary condition encoding (14 features total) — critical for static problems.

**Gap vs. our setup:**
- They have 50,000 diverse training samples; this repo's active datasets may be
  much smaller and should be checked per run.
- They encode boundary conditions explicitly with rich node features. This repo
  only gets explicit boundary/type information when the HDF5 feature at index 7
  is meaningful and `use_node_types True`.
- This repo now has positional node features, but they are not the exact
  PCA-aligned simulation coordinates from Gladstone et al. The implemented
  options are centroid distance, mean edge length, random-walk positional
  encoding, and Laplacian positional encoding.

---

## Tier 1: Try First (highest impact, lowest effort)

### 1. Simulation Coordinates (from Gladstone et al.)

**What:** Transform reference positions to centroid-centered, principal-axis-aligned frame, then use as node features. Provides spatial context with partial invariance.

Current repo note: `positional_features` already supplies invariant geometric and
topological node identity features. PCA-aligned simulation coordinates remain a
separate experiment, not the behavior of the current loader.

| Property | Assessment |
|---|---|
| Spatial discrimination | **Solves it** — each node gets unique position in standardized frame |
| Translation invariance | **Preserved** (centroid-centered) |
| Rotation invariance | **Partially preserved** (PCA-aligned, but sign ambiguity remains) |
| Implementation | ~30 lines — `np.linalg.eigh` on centered coordinates |
| Proven impact | 2-3x improvement from coordinates alone, before architecture changes |

### 2. Edge Augmented GNN (Random Long-Range Edges)

**What:** Add random edges between distant nodes (20% of mesh edge count). Mark with binary flag.

| Property | Assessment |
|---|---|
| Long-range communication | **Solves it** — shortcuts through the graph |
| Invariance | **Preserved** — edges use relative features |
| Implementation | ~50 lines — random pair selection, filter existing edges, add flag |
| Proven impact | 5x error reduction on static mechanics |

### 3. Laplacian Eigenvector Positional Encoding (LapPE)

**What:** Compute k smallest eigenvectors of graph Laplacian `L = D - A`. Each node gets k-dimensional spectral coordinate.

| Property | Assessment |
|---|---|
| Spatial discrimination | **Solves it** — structurally distant nodes get different encodings |
| Translation/rotation invariance | **Fully preserved** — topology-only, no coordinates |
| Implementation | ~100 lines — `scipy.sparse.linalg.eigsh`, concat to node features |
| Sign ambiguity | Use SignNet or random sign flipping during training |

**References:** Dwivedi & Bresson 2020, Lim et al. 2022 (SignNet)

### 4. Virtual Node

**What:** Add one extra node connected to ALL real nodes. Aggregates global state and broadcasts back each GnBlock.

| Property | Assessment |
|---|---|
| Global context | **Direct** — every node sees global state in 1 hop |
| Invariance | **Preserved** — feature space only |
| Implementation | ~50-100 lines |

---

## Tier 2: Try Second

### 5. Boundary Condition Encoding

For quasi-static problems, BCs are the primary driver. Gladstone et al. use 14 node features including BC type, direction, magnitude. Without knowing which nodes are fixed vs. free, the model must infer constraints from displacement — which is circular.

Enable `use_node_types True` if HDF5 encodes fixture/boundary information.

### 6. Enable Existing Multiscale/U-Net Path

Already implemented in `model/MeshGraphNets.py` and active in several config
files through `use_multiscale True`. Coarse-level message passing shortcuts
long-range communication. Ensure coarsest level covers the entire domain and set
`mp_per_level` explicitly for the intended compute distribution.

### 7. Discrete Curvature Features

Gaussian curvature (scalar, fully invariant) encodes local geometry. High-curvature regions typically warp more. Computable from triangle mesh.

---

## Tier 3: If Tier 1-2 Insufficient

### 8. Random Walk PE (supplement to LapPE)
Self-return probabilities — captures local structural properties. No sign ambiguity.

### 9. Data Augmentation
- Rotation/scaling of coordinates
- Mesh perturbation (jiggle node positions)
- Increase `std_noise` to 0.01-0.05
- Coordinate noise ±10% of edge distance (Gladstone et al.)

### 10. Transfer Learning
Apple Research (Feb 2025): Pre-train on diverse simulations, fine-tune on target. 1/16 data outperformed full training from scratch.

---

## Tier 4: Lower Priority

| Approach | Why lower |
|---|---|
| GAT/GATv2 attention | Doesn't fix receptive field problem |
| Graph Transformer | O(N^2) prohibitive for large meshes |
| Self-supervised pre-training | Limited benefit with few geometries |

---

## Recommended Experiment Plan

```
Experiment 1: Validate current positional_features choices on the active dataset
Experiment 2: Enable/use_node_types only when feature index 7 has meaningful labels
Experiment 3: Tune multiscale coarsening and mp_per_level for the target mesh size
Experiment 4: Evaluate coarse_world_edges or random long-range edges for contact/global coupling
Experiment 5: Add PCA-aligned simulation coordinates only if current positional features underperform
Experiment 6: Add virtual node/global context only after the cheaper graph features are exhausted
```

---

## DeepMind MGN Edge Feature Reference

| Simulation | Edge features | Dim |
|---|---|---|
| flag_simple, deformable_plate | `[dx, dy, dz, dist]` — deformed frame | 4 |
| cylinder_flow, airfoil | `[dx, dy, dist]` — deformed frame | 3 |
| flag_dynamic (self-collision) | mesh: 4D + separate world edges: 4D | 4+4 separate |

No edge_var=8 case in DeepMind. Spatial discrimination comes from velocity node features (unavailable in quasi-static problems).

---

## References

- Pfaff et al. "Learning Mesh-Based Simulation with Graph Networks" (ICLR 2021)
- Gladstone et al. "Mesh-based GNN surrogates for time-independent PDEs" (Scientific Reports, 2024) — https://www.nature.com/articles/s41598-024-53185-y
- Fortunato & Pfaff "MultiScale MeshGraphNets" (2022)
- Lim et al. "SignNet/BasisNet" (2022) — sign-invariant spectral features
- Dwivedi et al. "Benchmarking Positional Encodings for GNNs" (2024)
- Apple Research "Transfer Learning in Scalable GNN for Physical Simulation" (Feb 2025)
- Alon & Yahav "On the Bottleneck of Graph Neural Networks" (2021)
