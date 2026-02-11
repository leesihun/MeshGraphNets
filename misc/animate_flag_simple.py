"""
Animated GIFs of transient response for a random flag_simple sample.
Color = nodal_data[-2] (stress, feature index 6).
Produces separate GIF files for XZ, XY, YZ, and 3D isometric views.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image
import random
import sys
import os

# Use absolute path: repo_root/dataset/flag_simple.h5
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H5_PATH = os.path.join(REPO_ROOT, "dataset", "flag_simple.h5")
DT = 0.02
FRAME_SKIP = 4
GIF_FPS = 20
GIF_DURATION_MS = int(1000 / GIF_FPS)

# Feature indices in nodal_data (8, T, N)
IX, IY, IZ = 0, 1, 2
IDX, IDY, IDZ = 3, 4, 5
# nodal_data[-2] => index 6 (stress)
COLOR_FEAT = -2


def load_sample(h5_path, sample_id):
    with h5py.File(h5_path, "r") as f:
        nd = f[f"data/{sample_id}/nodal_data"][:]
        edges = f[f"data/{sample_id}/mesh_edge"][:]
        meta = dict(f[f"data/{sample_id}/metadata"].attrs)
        feat_names = list(f["metadata/feature_names"][:])
    return nd, edges, meta, feat_names


def world_positions(nd, t):
    """Return deformed (wx, wy, wz) at timestep t."""
    wx = nd[IX, 0, :] + nd[IDX, t, :]
    wy = nd[IY, 0, :] + nd[IDY, t, :]
    wz = nd[IZ, 0, :] + nd[IDZ, t, :]  # rest z is 0 for flag
    return wx, wy, wz


# ------------------------------------------------------------------ #
#  2-D view renderer
# ------------------------------------------------------------------ #
def render_frame_2d(nd, edges, t, norm, cmap, fig, ax,
                    axis_a, axis_b, label_a, label_b, lims_a, lims_b):
    ax.clear()
    wx, wy, wz = world_positions(nd, t)
    all_coords = {0: wx, 1: wy, 2: wz}
    ca = all_coords[axis_a]
    cb = all_coords[axis_b]
    coords_2d = np.stack([ca, cb], axis=1)

    color = nd[COLOR_FEAT, t, :]

    segments = np.stack([coords_2d[edges[0]], coords_2d[edges[1]]], axis=1)
    edge_c = (color[edges[0]] + color[edges[1]]) / 2
    lc = LineCollection(segments, colors=cmap(norm(edge_c)),
                        linewidths=0.35, alpha=0.85)
    ax.add_collection(lc)
    ax.scatter(ca, cb, c=color, cmap=cmap, s=0.3, norm=norm)

    ax.set_xlim(lims_a)
    ax.set_ylim(lims_b)
    ax.set_aspect("equal")
    ax.set_xlabel(label_a)
    ax.set_ylabel(label_b)
    ax.set_title(f"t = {t * DT:.2f} s  (step {t})", fontsize=11)
    ax.grid(True, alpha=0.2)

    fig.canvas.draw()
    buf = fig.canvas.get_renderer().buffer_rgba()
    return Image.frombuffer("RGBA", fig.canvas.get_width_height(), buf).convert("RGB")


# ------------------------------------------------------------------ #
#  3-D view renderer
# ------------------------------------------------------------------ #
def render_frame_3d(nd, edges, t, norm, cmap, fig, ax, elev, azim):
    ax.clear()
    wx, wy, wz = world_positions(nd, t)
    color = nd[COLOR_FEAT, t, :]

    # 3D edge segments
    starts = np.stack([wx[edges[0]], wy[edges[0]], wz[edges[0]]], axis=1)
    ends = np.stack([wx[edges[1]], wy[edges[1]], wz[edges[1]]], axis=1)
    segments = np.stack([starts, ends], axis=1)  # (E, 2, 3)
    edge_c = (color[edges[0]] + color[edges[1]]) / 2
    lc = Line3DCollection(segments, colors=cmap(norm(edge_c)),
                          linewidths=0.3, alpha=0.8)
    ax.add_collection3d(lc)
    ax.scatter(wx, wy, wz, c=color, cmap=cmap, s=0.3, norm=norm, depthshade=False)

    ax.set_xlim(-0.5, 4.0)
    ax.set_ylim(-0.5, 2.5)
    ax.set_zlim(-3.0, 3.0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"t = {t * DT:.2f} s  (step {t})", fontsize=11)

    fig.canvas.draw()
    buf = fig.canvas.get_renderer().buffer_rgba()
    return Image.frombuffer("RGBA", fig.canvas.get_width_height(), buf).convert("RGB")


# ------------------------------------------------------------------ #
#  GIF builder
# ------------------------------------------------------------------ #
def build_gif(frames, path):
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=GIF_DURATION_MS, loop=0)


def make_2d_gif(nd, edges, timesteps, norm, cmap, sample_id, num_nodes,
                color_label, axis_a, axis_b, label_a, label_b,
                lims_a, lims_b, view_tag):
    fig, ax = plt.subplots(figsize=(8, 5))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(color_label)
    fig.suptitle(f"Flag Simple  sample {sample_id}  (N={num_nodes})  [{view_tag}]",
                 fontsize=12, fontweight="bold")

    frames = []
    n = len(timesteps)
    for i, t in enumerate(timesteps):
        img = render_frame_2d(nd, edges, t, norm, cmap, fig, ax,
                              axis_a, axis_b, label_a, label_b, lims_a, lims_b)
        frames.append(img)
        sys.stdout.write(f"\r  {view_tag}: frame {i+1}/{n}  ({(i+1)/n*100:.0f}%)")
        sys.stdout.flush()
    plt.close(fig)
    print()

    out = f"flag_simple_s{sample_id}_{view_tag}.gif"
    build_gif(frames, out)
    print(f"  -> {out}  ({n} frames)")
    return out


def make_3d_gif(nd, edges, timesteps, norm, cmap, sample_id, num_nodes,
                color_label, elev, azim, view_tag):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.08, shrink=0.7)
    cbar.set_label(color_label)
    fig.suptitle(f"Flag Simple  sample {sample_id}  (N={num_nodes})  [{view_tag}]",
                 fontsize=12, fontweight="bold")

    frames = []
    n = len(timesteps)
    for i, t in enumerate(timesteps):
        img = render_frame_3d(nd, edges, t, norm, cmap, fig, ax, elev, azim)
        frames.append(img)
        sys.stdout.write(f"\r  {view_tag}: frame {i+1}/{n}  ({(i+1)/n*100:.0f}%)")
        sys.stdout.flush()
    plt.close(fig)
    print()

    out = f"flag_simple_s{sample_id}_{view_tag}.gif"
    build_gif(frames, out)
    print(f"  -> {out}  ({n} frames)")
    return out


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def main():
    with h5py.File(H5_PATH, "r") as f:
        sample_ids = list(f["data"].keys())
    sample_id = random.choice(sample_ids)

    nd, edges, meta, feat_names = load_sample(H5_PATH, sample_id)
    num_features, num_timesteps, num_nodes = nd.shape

    color_name = feat_names[COLOR_FEAT] if isinstance(feat_names[COLOR_FEAT], str) \
        else feat_names[COLOR_FEAT].decode()
    color_data = nd[COLOR_FEAT]  # (T, N)

    print(f"Sample {sample_id}  |  nodes={num_nodes}  timesteps={num_timesteps}")
    print(f"Color feature: nodal_data[{COLOR_FEAT}] = '{color_name}'")
    print(f"  range: [{color_data.min():.6f}, {color_data.max():.6f}]")

    if color_data.max() == color_data.min():
        print(f"  WARNING: '{color_name}' is constant ({color_data.min():.4f}) "
              f"for this dataset. Color will be uniform.")

    # Global axis limits from world positions across all timesteps
    wx_all = nd[IX, 0, :][None, :] + nd[IDX]   # (T, N)
    wy_all = nd[IY, 0, :][None, :] + nd[IDY]
    wz_all = nd[IZ, 0, :][None, :] + nd[IDZ]
    pad = 0.3
    xlims = (wx_all.min() - pad, wx_all.max() + pad)
    ylims = (wy_all.min() - pad, wy_all.max() + pad)
    zlims = (wz_all.min() - pad, wz_all.max() + pad)

    # Color normalization
    vmin, vmax = float(color_data.min()), float(color_data.max())
    if vmin == vmax:
        vmax = vmin + 1.0  # avoid degenerate norm
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.plasma
    color_label = color_name

    timesteps = list(range(0, num_timesteps, FRAME_SKIP))
    print(f"Frames: {len(timesteps)} (every {FRAME_SKIP} steps)\n")

    # --- View 1: X-Z (front) ---
    make_2d_gif(nd, edges, timesteps, norm, cmap, sample_id, num_nodes,
                color_label, axis_a=0, axis_b=2,
                label_a="X (world)", label_b="Z (world)",
                lims_a=xlims, lims_b=zlims, view_tag="XZ_front")

    # --- View 2: X-Y (top-down) ---
    make_2d_gif(nd, edges, timesteps, norm, cmap, sample_id, num_nodes,
                color_label, axis_a=0, axis_b=1,
                label_a="X (world)", label_b="Y (world)",
                lims_a=xlims, lims_b=ylims, view_tag="XY_top")

    # --- View 3: Y-Z (side) ---
    make_2d_gif(nd, edges, timesteps, norm, cmap, sample_id, num_nodes,
                color_label, axis_a=1, axis_b=2,
                label_a="Y (world)", label_b="Z (world)",
                lims_a=ylims, lims_b=zlims, view_tag="YZ_side")

    # --- View 4: 3D isometric ---
    make_3d_gif(nd, edges, timesteps, norm, cmap, sample_id, num_nodes,
                color_label, elev=25, azim=-60, view_tag="3D_iso")

    # --- View 5: 3D top-down ---
    make_3d_gif(nd, edges, timesteps, norm, cmap, sample_id, num_nodes,
                color_label, elev=80, azim=-60, view_tag="3D_top")

    print("\nDone.")


if __name__ == "__main__":
    main()
