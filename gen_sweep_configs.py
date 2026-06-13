"""Generate ex1/config_train{i}.txt and ex1/config_infer{i}.txt for a parametric
sweep over MP-shape, down/up asymmetry, and level count.

Edit SPECS to add/remove runs, then:  python gen_sweep_configs.py
Only indices listed in SPECS are written; config_train1..4 are left untouched.

mp_per_level layout for L levels (2L+1 entries):
    [pre_0 .. pre_{L-1}, coarsest, post_{L-1} .. post_0]
    first L  = DOWN / coarsening side (run before each pool)
    middle   = bottom (coarsest resolution)
    last L   = UP / upscaling side (run after each unpool + skip-merge)
"""
import os

OUT_DIR = "ex1"
CHAMP = [10000, 2000, 400, 80]      # champion L4 clustering
N0 = 42000                          # nominal finest node count (ex1.h5 mean) for compute proxy


def compute_proxy(clusters, mp):
    """Rough FLOPs proxy = sum over MP blocks of (nodes at that block's level).

    A block at the finest level (~42k nodes) costs far more than one at a coarse
    level (e.g. 80 nodes), so this is what "fixed compute" is matched on.
    Level node counts are [N0, *clusters]; slot s maps to level min(s, 2L-s).
    """
    L = (len(mp) - 1) // 2
    nodes = [N0] + list(clusters)
    return sum(mp[s] * nodes[s if s <= L else 2 * L - s] for s in range(2 * L + 1))

# idx, gpu, coarsening_type, voronoi_clusters, mp_per_level, comment
#
# NEW axes only. Symmetric MP-shapes (even/bell/reverse-bell/shoulder) were
# already swept in BATCH6 and are intentionally NOT repeated here. This batch
# isolates the two open questions: (1) down vs up asymmetry, (2) level count.
SPECS = [
    # --- Group A: DOWN/UP asymmetry at L4 (champion 10k/2k/400/80; bottom=8 and
    #     side-total=24 held fixed, only the down/up split varies -> pure asymmetry signal)
    (5,  4, "voronoi_seedmean", CHAMP, [5, 5, 5, 5, 8, 1, 1, 1, 1], "BATCH7 asym L4: down20/up4 (down-heavy extreme)"),
    (6,  5, "voronoi_seedmean", CHAMP, [4, 4, 4, 4, 8, 2, 2, 2, 2], "BATCH7 asym L4: down16/up8"),
    (7,  6, "voronoi_seedmean", CHAMP, [3, 3, 3, 3, 8, 3, 3, 3, 3], "BATCH7 asym L4: down12/up12 (balanced midpoint)"),
    (8,  7, "voronoi_seedmean", CHAMP, [2, 2, 2, 2, 8, 4, 4, 4, 4], "BATCH7 asym L4: down8/up16 (= champion shape)"),
    (9,  0, "voronoi_seedmean", CHAMP, [1, 1, 1, 1, 8, 5, 5, 5, 5], "BATCH7 asym L4: down4/up20 (up-heavy extreme)"),

    # --- Group B: same asymmetry sweep at L2 (10k/400; bottom=8, side-total=12)
    #     -> does the down/up conclusion hold at a different depth?
    (10, 1, "voronoi_seedmean", [10000, 400], [5, 5, 8, 1, 1], "BATCH7 asym L2: down10/up2 (down-heavy extreme)"),
    (11, 2, "voronoi_seedmean", [10000, 400], [4, 4, 8, 2, 2], "BATCH7 asym L2: down8/up4"),
    (12, 3, "voronoi_seedmean", [10000, 400], [3, 3, 8, 3, 3], "BATCH7 asym L2: down6/up6 (balanced midpoint)"),
    (13, 4, "voronoi_seedmean", [10000, 400], [2, 2, 8, 4, 4], "BATCH7 asym L2: down4/up8"),
    (14, 5, "voronoi_seedmean", [10000, 400], [1, 1, 8, 5, 5], "BATCH7 asym L2: down2/up10 (up-heavy extreme)"),

    # --- Group C: LEVEL ladder, pattern-matched (down2/up4/bottom8 held; endpoints
    #     mesh->...->80 held; only # of intermediate levels varies). L4 baseline = train1.
    (15, 6, "voronoi_seedmean", [10000, 900, 80], [2, 2, 2, 8, 4, 4, 4], "BATCH7 level ladder L3 (pattern-matched, coarsest=80)"),
    (16, 7, "voronoi_seedmean", [10000, 80],       [2, 2, 8, 4, 4],       "BATCH7 level ladder L2 (pattern-matched, coarsest=80)"),
    (17, 0, "voronoi_seedmean", [80],              [2, 8, 4],             "BATCH7 level ladder L1 (single mesh->80 jump)"),

    # --- Group D: LEVEL reduction, budget-matched (total MP held ~32, coarsest=80)
    #     -> separates "fewer levels" from "less compute". L4 baseline = train1 (total 32).
    (18, 1, "voronoi_seedmean", [10000, 900, 80], [3, 3, 3, 8, 5, 5, 5], "BATCH7 level budget-matched L3 (total 32)"),
    (19, 2, "voronoi_seedmean", [10000, 80],       [4, 4, 8, 8, 8],       "BATCH7 level budget-matched L2 (total 32)"),
    (20, 3, "voronoi_seedmean", [80],              [12, 8, 12],           "BATCH7 level budget-matched L1 (total 32)"),

    # --- Group E: DEPTH allocation at fixed TOTAL MP count (=32, champion L4 clustering).
    #     Palindromic (down=up) so this isolates depth from the down/up axis. Peak of the
    #     MP distribution migrates finest(L0) -> bottom(L4). Same block count, but the deep
    #     configs use far less actual compute (see proxy col) -> tests if coarse MP is "free".
    (21, 4, "voronoi_seedmean", CHAMP, [9, 3, 2, 1, 2, 1, 2, 3, 9], "BATCH7 depth@32: shallow-extreme (MP at finest L0)"),
    (22, 5, "voronoi_seedmean", CHAMP, [5, 5, 4, 1, 2, 1, 4, 5, 5], "BATCH7 depth@32: shallow (MP at L0/L1)"),
    (23, 6, "voronoi_seedmean", CHAMP, [1, 2, 4, 6, 6, 6, 4, 2, 1], "BATCH7 depth@32: deep (MP at L3/bottom)"),
    (24, 7, "voronoi_seedmean", CHAMP, [1, 1, 2, 4, 16, 4, 2, 1, 1], "BATCH7 depth@32: deep-extreme (MP at bottom L4)"),
]

TEST_BATCH_TRAIN = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 33, 34"


def build(idx, gpu, ctype, clusters, mp, comment, mode):
    L = (len(mp) - 1) // 2
    assert len(mp) == 2 * L + 1, f"train{idx}: mp len {len(mp)} != 2*{L}+1"
    assert len(clusters) == L, f"train{idx}: clusters len {len(clusters)} != L={L}"

    is_train = mode == "train"
    clusters_s = ", ".join(str(c) for c in clusters)
    mp_s = ", ".join(str(m) for m in mp)
    prefix = "train" if is_train else "infer"
    extra = "" if is_train else f"inference_output_dir   outputs/rollout/ex1/model{idx}\n"

    return f"""model   MeshGraphNets
mode    {mode}  # Train / Inference
gpu_ids {gpu}    # {comment}
log_file_dir    ex1/{prefix}{idx}.log
modelpath   ./outputs/ex1/model{idx}.pth
%   Datasets
dataset_dir ./dataset/ex1.h5
infer_dataset   ./dataset/hex_dataset.h5
{extra}infer_timesteps 1
%   Common params
input_var   4   # number of input variables: x_disp, y_disp, z_disp, stress (excluding node types)
output_var  4   # number of output variables: x_disp, y_disp, z_disp, stress (excluding node types)
feature_loss_weights  0, 0, 0, 1.0  # Per-feature loss weights for [x_disp, y_disp, z_disp, stress]
edge_var    8   # deformed dx/dy/dz/dist + reference dx/dy/dz/dist
positional_features  4   # Rotation-invariant node features: [centroid_dist, mean_edge_len, + encoding]
'
%   Network parameters (BATCH7: MP-shape x asymmetry x level sweep)
message_passing_num 15
Training_epochs	2000
Batch_size	1
LearningR	0.0001
Latent_dim	128	# MeshGraphNets latent dimension
num_workers 2
std_noise   {"0.1" if is_train else "0.0"}
weight_decay    0.0001
residual_scale  1
augment_geometry {"True" if is_train else "False"}
grad_accum_steps    1
'
% Memory Optimization
use_checkpointing   False
'
% Performance Optimization
use_amp             True    # Mixed precision training with bfloat16
use_ema             True    # EMA shadow model for validation/inference
ema_decay           0.99
test_interval       100      # Run test/visualization every N epochs
val_interval        5        # Run val eval every N epochs
'
% Node Type Parameters
use_node_types  True    # Add one-hot encoded node types to node features
'
% World Edge Parameters
use_world_edges         False
'
% Test set control
test_batch_idx  {TEST_BATCH_TRAIN if is_train else "0"}
plot_feature_idx    -1  # Feature index to visualize in plots (-1 = last feature, i.e., stress)
'
% Multi-Scale / Hierarchical Parameters (SWEPT)
use_multiscale      True
coarsening_type     {ctype}
voronoi_clusters    {clusters_s}
multiscale_levels   {L}
mp_per_level        {mp_s}
bipartite_unpool    True    # LOCKED: learned bipartite MP unpool (always)
"""


def main():
    print(f"{'idx':>4} {'gpu':>3} {'L':>2} {'blks':>4} {'compute':>9}  {'clusters':<22} {'mp_per_level':<28} comment")
    for idx, gpu, ctype, clusters, mp, comment in SPECS:
        L = (len(mp) - 1) // 2
        for mode in ("train", "inference"):
            text = build(idx, gpu, ctype, clusters, mp, comment, mode)
            fname = f"config_{'train' if mode == 'train' else 'infer'}{idx}.txt"
            with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as f:
                f.write(text)
        print(f"{idx:>4} {gpu:>3} {L:>2} {sum(mp):>4} {compute_proxy(clusters, mp):>9,}  "
              f"{str(clusters):<22} {str(mp):<28} {comment}")
    print(f"\nWrote {len(SPECS)} train + {len(SPECS)} infer configs to {OUT_DIR}/ (indices {SPECS[0][0]}..{SPECS[-1][0]})")


if __name__ == "__main__":
    main()
