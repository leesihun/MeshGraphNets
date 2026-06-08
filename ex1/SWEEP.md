# ex1 Parameter Sweep Log

Record of train1–train8 sweep experiments on the `ex1` dataset. Goal: reduce
validation loss on stress prediction (`feature_loss_weights 0, 0, 0, 1.0`).

## Common Setup (all sweeps unless noted)

- Dataset: `./dataset/ex1.h5`, deterministic 80/10/10 split (seed 42)
- 5000 epochs (sweep 3 introduces 10000 for some configs), EMA `0.999`,
  bfloat16 AMP, `augment_geometry True`
- `use_node_types True`, `positional_features 4` (centroid_dist + edge_len + 2× RWPE)
- `use_world_edges False`, `use_checkpointing True`
- `feature_loss_weights 0, 0, 0, 1.0` (stress only)
- Stress predicted as normalized delta (Huber loss)

---

## Sweep 1 — Initial Landscape (completed)

**Baseline (sweep-1 train1):**
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
| **train5** | voronoi `2500, 500` (aggressive coarsening) | **4.5** | Modest win |
| train6 | `positional_features 8`, `rwpe+lpe` | 6.7 | LPE / more feats hurt |
| train7 | mp `3,6,12,6,3` (30 blocks) | 5.0 | Mild overfit |
| train8 | L=1, voronoi `1000`, mp `6,8,6` | 5.8 | Shallow hurt |

### Axes Learned

- **Depth (L=1→2→3):** clean monotonic win — 5.8 → 4.8 → 4.2
- **Capacity (10→20→30 blocks):** sweet spot at 20; concave curve
- **Coarsening aggressiveness:** more compression helps (4.5 vs 4.8)
- **Noise:** 0.01 fine, 0.03 hurts
- **Positional encoding:** RWPE+LPE / 8 features hurt (couldn't separate cause)

### Key Diagnostic

Train loss bottomed at **5e-5** while val plateaued at **4.2e-3** →
**~80× train/val gap**. Severe memorization. More capacity is not the bottleneck.
Typical well-regularized MGN runs sit at 2–10×.

---

## Sweep 2 — Attack the Memorization Gap (completed)

**Baseline (sweep-2 train1) = combo of sweep-1 confirmed wins:**
- L=3 multiscale, `voronoi_inherit`, clusters `2500, 500, 100`
- `mp_per_level 1, 2, 4, 6, 4, 2, 1` (20 blocks)
- `bipartite_unpool False` (broadcast — later reverted in sweep 3)
- `std_noise 0.01`, `Latent_dim 128`, `Batch_size 4`, `LearningR 0.0001`

### Results

| Config | Knob delta vs baseline | Val (×1e-3) | Verdict |
|--------|------------------------|-------------|---------|
| **train6** | `LearningR 0.0002` | **3.05** | **Best — LR was undertrained** |
| **train2** | `Batch_size 2`, `LearningR 0.00007` | **3.68** | **Big win — SGD noise regularizer** |
| train1 | baseline (combo) | 4.46 | New anchor |
| train5 | `std_noise 0.005` | 4.56 | No change |
| train8 | `augment_geometry False` | 4.81 | Aug helps a little |
| train3 | `positional_features 2` (no encoding) | 4.84 | RWPE is mildly helpful |
| train4 | L=4, mp `1,2,3,5,6,5,3,2,1` (28 blocks), voronoi `2500,800,200,50` | 5.88 | Depth saturated |
| train7 | `Latent_dim 192` | 5.88 | Width hurts |

### Axes Learned

- **LR is THE win.** 0.0001 → 0.0002 gave 32% improvement. Model was undertrained.
- **Smaller batch helps.** bs=4 → bs=2 with sqrt-scaled LR gave 17% improvement.
- **Depth saturates at L=3.** L=4 with 28 blocks badly hurts (confounds depth + capacity).
- **Width is harmful.** Latent_dim 192 hurts as much as L=4.
- **Encoding question:** dropping RWPE costs ~9% — keep it.
- **Aug:** modest help (~8%) — keep it.
- **Sweep-1 combo did not strictly beat sweep-1 train2** (4.46 vs 4.2). The
  aggressive coarsening + broadcast change introduced mixed effects.

### Decision

- **Unpool:** flip back to `bipartite_unpool True` going forward (user-locked).
- **Architecture frozen:** L=3, mp `1,2,4,6,4,2,1` (20 blocks), `Latent_dim 128`,
  voronoi `2500, 500, 100`, RWPE 4 features.
- **Open axis:** LR & batch & training length.

---

## Sweep 3 — LR Push (running)

**Anchor (sweep-3 train1) = sweep-2 train6 setup with `bipartite_unpool True`:**
- L=3, voronoi `2500, 500, 100`, mp `1,2,4,6,4,2,1`, `bipartite_unpool True`
- `LearningR 0.0002`, `Batch_size 4`, `Training_epochs 5000`
- All other defaults preserved

### Configs

| Config | GPU | Change vs anchor | Hypothesis |
|--------|-----|------------------|------------|
| train1 | 0 | — anchor | Reproduce 3.05 with bipartite True |
| **train2** | 1 | `Batch_size 1`, `LearningR 0.0001` (sqrt-scaled) | Stack LR + bs wins |
| **train3** | 2 | `Batch_size 1` (LR stays 0.0002, no rescale) | Stack without sqrt — riskier |
| train4 | 3 | `LearningR 0.0005` | LR push 2.5× |
| train5 | 0 | `LearningR 0.001` | LR ceiling check 5× |
| train6 | 1 | `voronoi_clusters 5000, 1000, 200` | Less aggressive cascade at new LR |
| train7 | 2 | `Training_epochs 10000` | Longer training at anchor |
| **train8** | 3 | bs=1 + LR 0.0001 + epochs 10000 | Best combo guess |

### Watch List

- **train1** sets the bipartite-True calibration vs sweep-2 train6's 3.05
  (which was at `bipartite_unpool False`).
- **train5 (LR 1e-3)** may diverge. NaNs or runaway loss = LR ceiling is below 1e-3.
- **train3 (bs=1, LR 0.0002 unscaled)** has 4× the per-sample gradient magnitude of
  anchor. Could be unstable or could be the win.
- **train7 vs train1** answers whether 5000 epochs was undertrained at the new LR.

### Run Plan

```powershell
# Wave 1
python MeshGraphNets_main.py --config ex1/config_train1.txt   # GPU 0
python MeshGraphNets_main.py --config ex1/config_train2.txt   # GPU 1
python MeshGraphNets_main.py --config ex1/config_train3.txt   # GPU 2
python MeshGraphNets_main.py --config ex1/config_train4.txt   # GPU 3

# Wave 2
python MeshGraphNets_main.py --config ex1/config_train5.txt   # GPU 0
python MeshGraphNets_main.py --config ex1/config_train6.txt   # GPU 1
python MeshGraphNets_main.py --config ex1/config_train7.txt   # GPU 2
python MeshGraphNets_main.py --config ex1/config_train8.txt   # GPU 3
```

### Results

| Config | Val (×1e-3) | Δ vs anchor | Notes |
|--------|-------------|-------------|-------|
| **train7** | **2.22** | **−42%** | Longer training (10000 epochs) — **best of sweep** |
| **train2** | **2.33** | −39% | bs=1 + LR 0.0001 (sqrt) |
| train3 | 2.51 | −34% | bs=1, LR 0.0002 (no rescale) |
| train5 | 2.71 | −29% | LR 0.001 |
| train6 | 2.89 | −24% | voronoi 5000/1000/200 |
| train1 | 3.80 | — | anchor (bipartite True flip) |
| train4 | 3.93 | +3% | LR 0.0005 — worse than both 0.0002 and 0.001 (non-monotonic) |
| train8 | ~3.X | — | combo (bs=1 + LR 0.0001 + 10000 epochs) — wins did not stack |

### Takeaways

- **Longer training wins big.** train7 at 2.22e-3 beat sweep-2 best (3.05) by 27%.
  Previous sweeps were undertrained at the new LR.
- **bs=1 is decisively better than bs=4.** Both train2 (2.33) and train3 (2.51)
  crushed the bs=4 anchor (3.80).
- **`bipartite_unpool True` cost ~25%.** Anchor (sweep-3 train1) regressed from
  3.05 → 3.80 vs sweep-2 train6 (same config except unpool flag). Worth revisiting
  the locked-True decision against this dataset.
- **LR is non-monotonic.** LR 0.001 (2.71) > LR 0.0005 (3.93). The 0.0005 result
  is suspicious; worth re-running. Cosine restart schedule may interact.
- **Combo wins did not stack.** train8 (bs=1 + LR 0.0001 + 10000 epochs) landed at
  ~3.X, equivalent to anchor — under-performs each individual win. Likely cause:
  bs=1 + extended training overshoots cosine schedule sweet spot, or accumulated
  gradient noise over 2× epochs destabilizes. Stacking needs care.

### Cumulative Best So Far

**Sweep-3 train7:** val **2.22e-3** at L=3, mp `1,2,4,6,4,2,1`, voronoi `2500,500,100`,
`bipartite_unpool True`, LR 0.0002, bs=4, **10000 epochs**.

---

---

## Locked Settings (do not sweep)

- `feature_loss_weights 0, 0, 0, 1.0` (stress only — user requirement)
- `ema_decay 0.999` (user — don't change)
- `bipartite_unpool True` (user — locked after sweep 2)
- `use_node_types True`
- `use_amp True` (bfloat16)
- `augment_geometry True`

---

## Sweep Design Notes

- **Single-knob isolation:** every non-anchor config differs from the anchor by
  exactly one parameter. Combo configs go in a dedicated "stack the wins" slot
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
