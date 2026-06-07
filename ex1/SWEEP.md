# ex1 Parameter Sweep Log

Record of train1-train8 sweep experiments on the `ex1` dataset. Goal: reduce
validation loss on stress prediction (`feature_loss_weights 0, 0, 0, 1.0`).

## Common Setup (both sweeps)

- Dataset: `./dataset/ex1.h5`, deterministic 80/10/10 split (seed 42)
- 5000 epochs, EMA `0.999`, bfloat16 AMP, `augment_geometry True`
- `use_node_types True`, `positional_features 4` (centroid_dist + edge_len + 2× RWPE)
- `use_world_edges False`, `use_checkpointing True`
- `feature_loss_weights 0, 0, 0, 1.0` (stress only)
- Stress predicted as normalized delta (Huber loss)

---

## Previous Sweep (completed)

**Baseline (old train1):**
- L=2 multiscale, `voronoi_inherit`, clusters `5000, 1000`
- `mp_per_level 2, 4, 8, 4, 2` (20 blocks)
- `std_noise 0.01`, `Latent_dim 128`, `Batch_size 1`, `LearningR 0.0001`
- `bipartite_unpool True`

### Results

| Config | Knob delta | Val (×1e-3) | Verdict |
|--------|------------|-------------|---------|
| train1 | baseline | 4.8 | — |
| **train2** | L=3, mp `1,2,4,6,4,2,1`, voronoi `5000,1000,200` | **4.2** | **Best — depth wins** |
| train3 | mp `1,2,4,2,1` (10 blocks) | 5.9 | Too small |
| train4 | `std_noise 0.03` | 5.6 | Heavier noise hurt |
| **train5** | voronoi `2500, 500` (aggressive) | **4.5** | Modest win |
| train6 | `positional_features 8`, `rwpe+lpe` | 6.7 | LPE / more feats hurt |
| train7 | mp `3,6,12,6,3` (30 blocks) | 5.0 | Mild overfit |
| train8 | L=1, voronoi `1000`, mp `6,8,6` | 5.8 | Shallow hurt |

### Axes Learned

- **Depth (L=1→2→3):** clean monotonic win — 5.8 → 4.8 → 4.2
- **Capacity (10→20→30 blocks):** sweet spot at 20; concave curve
- **Coarsening aggressiveness:** more compression helps (4.5 vs 4.8)
- **Noise:** 0.01 fine, 0.03 hurts
- **Positional encoding:** RWPE+LPE / 8 features hurt — possibly the extra
  feature count or LPE noise, not cleanly separable from one run

### Key Diagnostic

Train loss bottomed at **5e-5** while val plateaued at **4.2e-3** →
**~80× train/val gap**. Severe memorization. More capacity is not the bottleneck.
Typical well-regularized MGN runs sit at 2–10×.

---

## Current Sweep (running)

**New baseline (new train1) = combo of confirmed wins:**
- L=3 multiscale, `voronoi_inherit`, clusters `2500, 500, 100`
- `mp_per_level 1, 2, 4, 6, 4, 2, 1` (20 blocks)
- `bipartite_unpool False` (broadcast — locked True across all configs)
- `std_noise 0.01`, `Latent_dim 128`, `Batch_size 4`, `LearningR 0.0001`

Each of train2–train8 changes exactly one knob from train1.

### Configs

| Config | GPU | Tier | Knob change | Hypothesis |
|--------|-----|------|-------------|------------|
| train1 | 0 | — | new baseline (combo of prior wins) | New anchor; should beat 4.2 |
| train2 | 1 | S | `Batch_size 2`, `LearningR 0.00007` | SGD noise regularizer vs memorization |
| train3 | 2 | S | `positional_features 2` | Drop RWPE — force node anonymity |
| train4 | 3 | A | L=4, voronoi `2500,800,200,50`, mp `1,2,3,5,6,5,3,2,1` (28 blocks) | Depth saturation push |
| train5 | 0 | A | `std_noise 0.005` | Lighter noise |
| train6 | 1 | A | `LearningR 0.0002` | sqrt(4) LR scaling for bs=4 |
| train7 | 2 | A | `Latent_dim 192` | Wider latent at fixed depth |
| train8 | 3 | B | `augment_geometry False` | Aug ablation |

### Tier Definitions

- **Tier S** — direct attack on the 80× memorization gap (highest-leverage)
- **Tier A** — knobs the user explicitly flagged for testing
- **Tier B** — sanity ablations

### Open Questions

1. **Does smaller batch close the gap?** train2 — if val drops noticeably,
   SGD noise was the missing regularizer.
2. **Is RWPE actively helping, or just feeding memorization?** train3 — if val
   improves with `positional_features 2`, drop RWPE entirely.
3. **Does depth saturate at L=3?** train4 — if val stalls at 4.2, depth is done.
4. **Is aug actually doing useful work?** train8 — if val unchanged, aug isn't
   regularizing and stronger transforms would need to be added.

### Run Plan

```powershell
# Wave 1 — all 4 GPUs parallel
python MeshGraphNets_main.py --config ex1/config_train1.txt   # GPU 0
python MeshGraphNets_main.py --config ex1/config_train2.txt   # GPU 1
python MeshGraphNets_main.py --config ex1/config_train3.txt   # GPU 2
python MeshGraphNets_main.py --config ex1/config_train4.txt   # GPU 3

# Wave 2 — all 4 GPUs parallel
python MeshGraphNets_main.py --config ex1/config_train5.txt   # GPU 0
python MeshGraphNets_main.py --config ex1/config_train6.txt   # GPU 1
python MeshGraphNets_main.py --config ex1/config_train7.txt   # GPU 2
python MeshGraphNets_main.py --config ex1/config_train8.txt   # GPU 3
```

### Results

_Fill in as runs complete._

| Config | Val (×1e-3) | Train (×1e-5) | Gap | Notes |
|--------|-------------|---------------|-----|-------|
| train1 |  |  |  |  |
| train2 |  |  |  |  |
| train3 |  |  |  |  |
| train4 |  |  |  |  |
| train5 |  |  |  |  |
| train6 |  |  |  |  |
| train7 |  |  |  |  |
| train8 |  |  |  |  |

---

## Sweep Design Notes

- **Single-knob isolation:** every non-baseline config differs from train1 by
  exactly one parameter. Combo configs go in a separate "stack the wins" run
  once individual effects are known.
- **GPU assignment:** round-robin 0–3, two waves of 4. No DDP.
- **`message_passing_num 15`** is kept in every config: required by the model
  init dict access at [model/MeshGraphNets.py:76](../model/MeshGraphNets.py#L76)
  even when multiscale is on (the value is ignored at runtime).
- **`noise_gamma`** defaults to `1.0` in
  [model/MeshGraphNets.py:60](../model/MeshGraphNets.py#L60) — full target
  correction matches the original MGN paper. Not set explicitly in configs.
- **Inline comments use `#` not `%`** — `%` only works as a line-start comment
  per [general_modules/load_config.py:17,21](../general_modules/load_config.py#L17-L21);
  inline `%` is parsed as part of the value list.
