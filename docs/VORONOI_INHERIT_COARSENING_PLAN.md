# Variant-C Coarsening Plan — `coarsening_type = voronoi_inherit`

> **Reusable plan**. The structure below applies to any MeshGraphNets-style
> repo that uses FPS-Voronoi multiscale coarsening with centroid-based pool.
> File paths and line numbers reference this repo; adapt them to the
> equivalent files in another codebase.

---

## Concept

In FPS-Voronoi coarsening, the FPS seed is a real fine-mesh node. Today the
"coarse node" at each level is a virtual point at the cluster's arithmetic
centroid, with features computed by `scatter mean` over cluster members.

This plan adds an opt-in mode where the coarse node **is** the FPS seed:

- Coarse position = seed's position (a real mesh node, never off-mesh).
- Coarse feature = seed's feature (pure gather `x[seeds]`, zero parameters).
- Coarse edges are seed-to-seed (induced subgraph on the seed set).

Pool becomes a degenerate gather. Unpool (broadcast or learned bipartite) is
unchanged. V-cycle structure and `mp_per_level` schema are unchanged. The
existing centroid behavior is preserved as the default.

### Modes (per-level, comma-separated in config)

| Value | Behavior |
|---|---|
| `bfs` | BFS bi-stride, centroid pool (unchanged) |
| `voronoi` | Back-compat alias for `voronoi_centroid` |
| `voronoi_centroid` | FPS-Voronoi, centroid position + mean pool (current default) |
| `voronoi_inherit` | FPS-Voronoi, seed position + gather pool (variant C) |

Old configs and checkpoints continue to work — variant C is opt-in.

---

## Files to Modify

| File | Change |
|---|---|
| [model/coarsening.py](../model/coarsening.py) | Coarseners return seeds; add `coarse_seed_idx` to PyG batching |
| [general_modules/multiscale_helpers.py](../general_modules/multiscale_helpers.py) | Per-level mode-aware position computation and seed-idx write |
| [general_modules/mesh_dataset.py](../general_modules/mesh_dataset.py) | Stats pass branches on per-level mode; accept new mode strings |
| [model/MeshGraphNets.py](../model/MeshGraphNets.py) | Pool branches on presence of `coarse_seed_idx_{i}` attribute |
| [parallelism/model_split.py](../parallelism/model_split.py) | Mirror `MeshGraphNets.py` |
| [docs/multiscale_coarsening.md](multiscale_coarsening.md) | Document new mode |
| [dataset/DATASET_FORMAT.md](../dataset/DATASET_FORMAT.md) | Document new optional attribute |
| [CLAUDE.md](../CLAUDE.md) | Add the new mode to architecture facts |

No existing example configs need changes; the new mode is opt-in via config.

---

## Implementation Steps

### Step 1 — `model/coarsening.py`

**1.1. `fps_voronoi_coarsen`** (function around line 223)

- Return signature: `(ftc, c_ei, n_c)` → `(ftc, c_ei, n_c, seeds)`.
- Expose `seeds` **after** the disconnected-components append loop so any
  extra component seeds are included.
- **After cluster compaction** (the `if num_coarse < k_actual: ... remap` step),
  reindex seeds so `seeds[new_cluster_id]` is the fine-node index for the
  new cluster:
  ```python
  new_seeds = np.empty(num_coarse, dtype=np.int64)
  new_seeds[ftc[old_seeds]] = old_seeds
  ```
- Cast `seeds` to `np.int64`.

**1.2. `bfs_bistride_coarsen`** (function around line 90)

- Return signature: add `seeds` (for API uniformity, even though BFS mode
  does not consume them).
- `seeds = coarse_nodes.astype(np.int64)` — `coarse_nodes` is already the
  array of even-depth fine indices aligned with cluster IDs.

**1.3. `coarsen_graph`** (dispatcher around line 318)

- Accept `method ∈ {'bfs', 'voronoi', 'voronoi_centroid', 'voronoi_inherit'}`.
- All three voronoi spellings dispatch to the same coarsener
  (`fps_voronoi_coarsen`). The mode only affects how the result is *used*
  downstream.
- Propagate the 4-tuple through all branches.
- Raise on unknown method.

**1.4-1.7. Keep `pool_features`, `unpool_features`, `compute_coarse_centroids`,
`build_unpool_edges`** — all still used by centroid mode and/or the simple
unpool path.

**1.8. Add `coarse_seed_idx` to `MultiscaleData` batching**

The attribute is *optional per level* — present only when that level uses
inherit mode. PyG's `__inc__`/`__cat_dim__` are key-driven, so optional
attributes work naturally.

- Extend `_LEVEL_RE`: add `|coarse_seed_idx` to the alternation.
- In `__inc__`, add branch:
  ```python
  if prefix == 'coarse_seed_idx':
      lvl = int(level)
      return self.num_nodes if lvl == 0 else int(self[f'num_coarse_{lvl - 1}'])
  ```
  Same semantics as row-1 of `unpool_edge_index`: seed indices at level `l`
  index into level `l-1`'s node space (or fine-node space at l=0).
- In `__cat_dim__`, add `coarse_seed_idx` to the dim-0 set alongside
  `fine_to_coarse`, `coarse_edge_attr`, `coarse_centroid`.

### Step 2 — `general_modules/multiscale_helpers.py`

**2.1. `build_multiscale_hierarchy`**

- Unpack 4-tuple from `coarsen_graph`.
- Store both `'seeds'` and `'mode': 'inherit' | 'centroid'` in each entry,
  with mode derived from the per-level method string
  (`voronoi_inherit` → `'inherit'`; everything else → `'centroid'`).
- When chaining to the next level, branch:
  ```python
  if entry['mode'] == 'inherit':
      level_ref_pos = level_ref_pos[seeds].astype(np.float32)
  else:
      level_ref_pos = compute_coarse_centroids(
          level_ref_pos, ftc, n_c
      ).astype(np.float32)
  ```

**2.2. `attach_coarse_levels_to_graph`**

- Read `entry['mode']` per level. Branch the position computation:
  ```python
  seeds = entry['seeds']
  if entry['mode'] == 'inherit':
      coarse_ref = cur_ref[seeds]
      coarse_def = cur_def[seeds]
  else:
      coarse_ref = compute_coarse_centroids(cur_ref, ftc, n_c)
      coarse_def = compute_coarse_centroids(cur_def, ftc, n_c)
  ```
- `coarse_centroid_{level}` continues to be written for both modes
  (in inherit mode it holds seed-anchor positions, not centroids; name
  retained for backward-compat with reader code).
- **Only in inherit mode**, additionally write the seed-index attribute:
  ```python
  if entry['mode'] == 'inherit':
      seed_idx_t = torch.from_numpy(seeds.astype(np.int64))
      if device is not None:
          seed_idx_t = seed_idx_t.to(device)
      graph[f'coarse_seed_idx_{level}'] = seed_idx_t
  ```

### Step 3 — `general_modules/mesh_dataset.py`

**3.1. Config parsing**

- Accept `voronoi_centroid` and `voronoi_inherit` in the `coarsening_types`
  parser. Normalize `'voronoi'` → `'voronoi_centroid'` at parse time so
  downstream code sees only canonical values.
- Raise on unknown method with a clear error message listing accepted values.

**3.2. `_compute_coarse_edge_stats`**

- Use the same per-level mode branch as `attach_coarse_levels_to_graph`:
  ```python
  seeds_l = entry['seeds']
  if entry['mode'] == 'inherit':
      coarse_ref = cur_ref[seeds_l]
      coarse_def = cur_def[seeds_l]
  else:
      coarse_ref = compute_coarse_centroids(cur_ref, ftc_l, n_c_l)
      coarse_def = compute_coarse_centroids(cur_def, ftc_l, n_c_l)
  ```

### Step 4 — `model/MeshGraphNets.py`

**4.1. `_extract_level_data`**

- Conditionally read seed indices (only present in inherit mode):
  ```python
  seed_key = f'coarse_seed_idx_{i}'
  if hasattr(graph, seed_key):
      ld['seeds'] = graph[seed_key]
  ```
- Keep the existing `bipartite_unpool` guard for `up_ei`, `coarse_centroid`,
  `fine_pos` unchanged.

**4.2. `forward` V-cycle**

- Branch the pool call:
  ```python
  if 'seeds' in ld:
      h_coarse = current_graph.x[ld['seeds']]              # inherit mode
  else:
      h_coarse = pool_features(current_graph.x, ld['ftc'], ld['n_c'])  # centroid
  ```
- Unpool branches unchanged. In inherit mode `rel_pos = fine_pos - coarse_centroid`
  automatically uses seed positions (because `coarse_centroid_{level}` was
  written with seed positions).

### Step 5 — `parallelism/model_split.py`

Mirror Step 4: same conditional read of `coarse_seed_idx_{level}` in
`_extract_level_data`, same branch in the pool step within
`run_local_blocks_multiscale`.

### Step 6 — Documentation

- `docs/multiscale_coarsening.md`: add "Coarsening modes" section. Note
  per-level mixing is supported.
- `dataset/DATASET_FORMAT.md`: document optional `coarse_seed_idx_{level}`
  attribute and the condition under which it appears.
- `CLAUDE.md`: extend the multiscale "Architecture Facts" bullet with the
  three canonical mode names.

---

## Reused Existing Utilities

| Utility | Purpose | File |
|---|---|---|
| `bfs_bistride_coarsen.coarse_nodes` | BFS "seed" array | `model/coarsening.py` |
| `fps_voronoi_coarsen.seeds` (internal) | FPS seed array | `model/coarsening.py` |
| `pool_features` | Centroid-mode pool | `model/coarsening.py` |
| `compute_coarse_centroids` | Centroid-mode position | `model/coarsening.py` |
| `MultiscaleData.__inc__` row-1 for `unpool_edge_index` | Precedent for seed-idx increment | `model/coarsening.py` |
| `attach_coarse_levels_to_graph` | Single source of truth for per-timestep attrs | `general_modules/multiscale_helpers.py` |
| `UnpoolBlock` | Bipartite GNN-interpolation (unchanged) | `model/blocks.py` |

---

## Out of Scope (Deliberate)

1. **Rename `coarse_centroid_{level}` → `coarse_anchor_{level}`**: would
   touch ~10 reader sites for cosmetic gain. Keep the name.
2. **`bfs_inherit` mode**: scoped to Voronoi only. BFS API is now uniform
   (returns seeds) but the BFS branch never sets `mode='inherit'`.
3. **Composed global seed-index precomputation** (faster rollout): pure
   perf optimization; defer.
4. **Backward-compat shim for inherit-mode checkpoints**: inherit-mode
   `coarse_edge_means/stds` differ from centroid-mode stats. A checkpoint
   trained in one mode cannot be used in the other.

---

## Verification

There is no test suite in this repo. Verify via smoke tests:

1. **Baseline path unchanged**: a config with `coarsening_type voronoi`
   trains identically to before (modulo the now-stored seeds by-product).
2. **Back-compat alias**: `voronoi_centroid` behaves identically to `voronoi`.
3. **New inherit mode end-to-end**: `voronoi_inherit` config completes stats
   recomputation, runs a few epochs with finite loss, produces valid
   `coarse_seed_idx_{level}` tensors.
4. **Mixed per-level modes**: `voronoi_centroid, voronoi_inherit` (L=2)
   correctly applies different modes per level; the appropriate attribute
   set is present in the batched graph.
5. **Batched training in inherit mode**: `Batch_size > 1` — the
   `coarse_seed_idx` batch-offset is the highest-risk new logic.
6. **DDP and model-split** with inherit mode.
7. **Rollout in inherit mode**: train, save checkpoint, run inference,
   inspect output HDF5 numerical ranges.
8. **Optional micro-test**:
   ```python
   assert batch.coarse_seed_idx_0.max() < batch.num_nodes
   assert batch.coarse_seed_idx_1.max() < int(batch.num_coarse_0.sum())
   ```

---

## Estimated Diff

- ~100-130 LOC across 5 code files + 3 doc files.
- No deletions (centroid-mode utilities all retained).
- No example-config edits required.
- New mode is opt-in; old configs and checkpoints continue to work unchanged.
