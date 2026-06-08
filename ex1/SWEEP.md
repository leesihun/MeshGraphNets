# ex1 Parameter Sweep Log

Record of train1–train8 sweep experiments on the `ex1` dataset. Goal: reduce
validation loss on stress prediction (`feature_loss_weights 0, 0, 0, 1.0`).

## Common Setup (all sweeps unless noted)

- Dataset: `./dataset/ex1.h5`, deterministic 80/10/10 split (seed 42)
- 5000 epochs (sweep 3 introduces 10000; sweep 4 uses 10000/20000), bfloat16 AMP, `augment_geometry True`
- `use_node_types True`, `positional_features 4` (centroid_dist + edge_len + 2× RWPE)
- `use_world_edges False`, `use_checkpointing True`
- `feature_loss_weights 0, 0, 0, 1.0` (stress only)
- Stress predicted as normalized delta
- **Loss function:** Huber (δ=0.1) in sweeps 1–3; switched to **MSE on 2026-06-08** for sweep 4 onward (sensitivity to large stress-hot-spot errors)
- **EMA decay:** `0.999` in sweeps 1–3; `0.9995` from sweep 4 (~2000-step window)

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
- `ema_decay 0.9995` (bumped from 0.999 in baseline reset)
- `bipartite_unpool True` (locked — see [feedback memory](../../C:/Users/Lee/.claude/projects/c--Users-Lee-Desktop-Huni-MeshGraphNets/memory/feedback_bipartite_unpool.md))
- `Batch_size 1` (locked — use `grad_accum_steps` to simulate larger effective batch)
- `use_node_types True`
- `use_amp True` (bfloat16)
- `augment_geometry True`
- Loss: **MSE** (switched from Huber on 2026-06-08)

---

## Sweep 5 — Multi-axis around fresh baseline (running)

**Anchor (sweep-5 train1) = user's reset baseline (2026-06-09):**
- L=2 multiscale, `voronoi_inherit`, clusters `5000, 1000`
- `mp_per_level 2, 4, 8, 4, 2` (20 blocks)
- `LearningR 0.0001`, `Batch_size 1`, `grad_accum_steps 1`, `Training_epochs 10000`
- `Latent_dim 128`, `std_noise 0.0`, `positional_features 4` (RWPE)
- `use_checkpointing False`, `ema_decay 0.9995`, MSE loss
- `val_interval 1` (was 10)

### Configs (16 total — 2 per GPU across 8 GPUs)

| # | GPU | Knob change vs anchor | Axis |
|---|-----|----------------------|------|
| train1 | 0 | — anchor | — |
| train2 | 1 | `LearningR 0.00005` | LR ↓ |
| train3 | 2 | `LearningR 0.0002` | LR ↑ |
| train4 | 3 | `LearningR 0.0005` | LR ↑↑ |
| train5 | 4 | `grad_accum_steps 4` | eff bs=4 (anchor LR) |
| train6 | 5 | `Training_epochs 5000` | epochs ↓ |
| train7 | 6 | `std_noise 0.005` | tiny noise |
| train8 | 7 | L=3, voronoi `5000,1000,200`, mp `1,2,4,6,4,2,1` | L=3 less aggressive |
| train11 | 0 | `LearningR 0.001` | LR ↑↑↑ |
| train12 | 1 | `grad_accum_steps 4`, `LearningR 0.0002` | eff bs=4 + sqrt LR |
| train13 | 2 | `Training_epochs 20000` | epochs ↑↑ |
| train14 | 3 | `std_noise 0.01` | moderate noise |
| train15 | 4 | L=3, voronoi `2500,500,100`, mp `1,2,4,6,4,2,1` | L=3 aggressive cascade |
| train16 | 5 | L=3, voronoi `5000,1000,200`, mp `2,4,6,8,6,4,2` (32 blocks) | L=3 + bigger mp |
| train17 | 6 | `mp_per_level 3, 6, 10, 6, 3` (28 blocks at L=2) | bigger mp at L=2 |
| train18 | 7 | `Latent_dim 192` | wider |

### Axis Coverage

- **LR (4):** 0.00005, 0.0002, 0.0005, 0.001 vs anchor 0.0001
- **grad_accum (2):** eff bs=4 at anchor LR; eff bs=4 + sqrt-scaled LR
- **Epochs (2):** 5000, 20000
- **std_noise (2):** 0.005, 0.01 (anchor is 0)
- **Multiscale L=3 (3 variants):** less aggressive, aggressive, bigger mp
- **mp size at L=2 (1):** 28 blocks
- **Latent width (1):** 192

### Watch List

- **train4 (LR 0.0005) and train11 (LR 0.001):** may diverge — kill early if NaN
- **train16 (L=3 + 32 blocks):** highest memory footprint; could OOM at bs=1 + Latent 128 on shared GPU
- **train18 (Latent 192):** also heavy; paired with train8 (L=3) on GPU 7 — watch memory
- **train12 vs train5:** difference is LR scaling at the same effective batch — tells you the SGD-noise-vs-step-size question

### Run Plan

All 16 launch simultaneously via the bundled script:

```powershell
.\ex1\run_sweep5.ps1
```

Each process picks its GPU from the `gpu_ids` field in its config; two processes
per GPU. Logs go to `ex1/trainN.log` (training log) and
`ex1/trainN.stdout.log` / `trainN.stderr.log` (process stdio).

To dry-run or launch a subset:

```powershell
.\ex1\run_sweep5.ps1 -DryRun
.\ex1\run_sweep5.ps1 -Configs 1,11      # just GPU 0
```

### Cross-Sweep Caveat

Sweep 5 numbers are not comparable to sweeps 1–4:
- Loss changed Huber (δ=0.1 then δ=1.0) → MSE on 2026-06-08
- EMA decay 0.999 → 0.9995
- Baseline architecture changed (L=2 + 5000/1000 vs L=3 + 2500/500/100)
- Baseline std_noise changed (0.01 → 0.0)
- `use_checkpointing` changed (True → False)

Treat sweep-5 train1 as the new reference; do not compare to past sweep numbers.

### Results

_Fill in as runs complete._

| Config | Val | Train | Notes |
|--------|-----|-------|-------|
| train1  |  |  | anchor |
| train2  |  |  | LR 0.00005 |
| train3  |  |  | LR 0.0002 |
| train4  |  |  | LR 0.0005 |
| train5  |  |  | grad_accum 4 |
| train6  |  |  | epochs 5000 |
| train7  |  |  | std_noise 0.005 |
| train8  |  |  | L=3 less aggressive |
| train11 |  |  | LR 0.001 |
| train12 |  |  | grad_accum 4 + LR 0.0002 |
| train13 |  |  | epochs 20000 |
| train14 |  |  | std_noise 0.01 |
| train15 |  |  | L=3 aggressive |
| train16 |  |  | L=3 + bigger mp |
| train17 |  |  | mp 28 at L=2 |
| train18 |  |  | Latent 192 |

---

## Sweep 4 — Multi-axis at bs=1 (configured, superseded by sweep 5)

> **Status: Superseded.** Sweep 4 was configured but not run. Baseline was reset
> on 2026-06-09 to a fresh L=2 + std_noise=0 + LR 0.0001 + MSE configuration,
> and the sweep design was rebuilt as sweep 5.



**Anchor (sweep-4 train1):** bs=1, LR 0.0002, 10000 epochs, voronoi `2500,500,100`,
mp `1,2,4,6,4,2,1` (20 blocks), Latent_dim 128, std_noise 0.01, ema_decay 0.9995,
**MSE loss**. This = sweep-3 train7's winning config but with bs=1 + MSE + bumped EMA.

### Configs

| # | GPU | Change vs anchor | Axis |
|---|-----|------------------|------|
| train1 | 0 | — anchor | — |
| train2 | 1 | `Training_epochs 20000` | epochs push 2× |
| train3 | 2 | `LearningR 0.001` | LR push (re-verify train5 win) |
| train4 | 3 | `voronoi_clusters 5000, 1000, 200` | less aggressive 3-step cascade |
| train5 | 4 | `Latent_dim 256` | wider latent (does bs=1 unlock capacity?) |
| train6 | 5 | `std_noise 0` | zero explicit noise (bs=1 SGD noise should suffice) |
| train7 | 6 | `mp_per_level 2, 4, 6, 8, 6, 4, 2` (32 blocks) | larger mp only |
| train8 | 7 | L=2, voronoi `10000, 1000`, mp `2,4,8,4,2` | 2-step large cascade |

### Run Plan

GPUs 0–7 used in parallel — all 8 configs run simultaneously, single wave.

```powershell
python MeshGraphNets_main.py --config ex1/config_train1.txt   # GPU 0
python MeshGraphNets_main.py --config ex1/config_train2.txt   # GPU 1
python MeshGraphNets_main.py --config ex1/config_train3.txt   # GPU 2
python MeshGraphNets_main.py --config ex1/config_train4.txt   # GPU 3
python MeshGraphNets_main.py --config ex1/config_train5.txt   # GPU 4
python MeshGraphNets_main.py --config ex1/config_train6.txt   # GPU 5
python MeshGraphNets_main.py --config ex1/config_train7.txt   # GPU 6
python MeshGraphNets_main.py --config ex1/config_train8.txt   # GPU 7
```

### Caveat on Cross-Sweep Comparison

Sweep 4 numbers are not directly comparable to sweeps 1–3:
- Loss changed Huber (δ=0.1) → MSE
- EMA decay 0.999 → 0.9995

Treat sweep 4's anchor (train1) as the new reference; don't compare to
sweep-3 train7's 2.22e-3 directly.

### Results

_Fill in as runs complete._

| Config | Val | Notes |
|--------|-----|-------|
| train1 |  |  |
| train2 |  |  |
| train3 |  |  |
| train4 |  |  |
| train5 |  |  |
| train6 |  |  |
| train7 |  |  |
| train8 |  |  |

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
