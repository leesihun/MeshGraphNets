"""
Mesh animation to GIF from HDF5 files that provide ``data/<id>/nodal_data`` and ``mesh_edge``.

Expected ``nodal_data`` layout: rows [0:3] reference position, [3:6] displacement, optional
extra rows (e.g. stress, part id). Coloring: ``--color-by auto`` (single row, stress or z-disp
heuristic) or ``--color-by displacement`` for L2 norm of rows 3:6.

No figure titles or axis labels in the output frames (colorbar scale only). Optional file dialog
on Windows if no path is passed. ``create_animation`` is an alias for ``generate_animations``.
"""

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from PIL import Image
import random
import sys
import os
import argparse
import ctypes
from pathlib import Path

# Feature indices in nodal_data (F, T, N); cloth/MGN exports use F=8
IX, IY, IZ = 0, 1, 2
IDX, IDY, IDZ = 3, 4, 5
# nodal_data[-2] => index 6 (stress)
COLOR_FEAT = -2
# Smooth, full-spectrum map; easier to read than classic jet/rainbow.
DEFAULT_COLORMAP = "turbo"
# Semi-transparent triangle fill (reconstructed from mesh edges) under wireframe.
TRI_FACE_ALPHA = 0.32


def _edges_to_triangles_numpy(edge_index: np.ndarray) -> np.ndarray:
    """
    Recover triangle node indices from unique undirected edges (same idea as
    ``edges_to_triangles_optimized``). Numpy + sets only — no torch/pyvista.
    """
    u = np.minimum(edge_index[0], edge_index[1])
    v = np.maximum(edge_index[0], edge_index[1])
    pairs = np.unique(np.column_stack([u, v]), axis=0)
    if pairs.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int64)
    num_nodes = int(np.max(pairs)) + 1
    adj = [set() for _ in range(num_nodes)]
    for a, b in pairs:
        ai, bi = int(a), int(b)
        adj[ai].add(bi)
        adj[bi].add(ai)
    tris = []
    for a, b in pairs:
        u_i, v_i = int(a), int(b)
        if u_i >= v_i:
            continue
        for w in adj[u_i] & adj[v_i]:
            if w > v_i:
                tris.append((u_i, v_i, w))
    if not tris:
        return np.zeros((0, 3), dtype=np.int64)
    return np.array(tris, dtype=np.int64)


def _triangle_face_rgba(norm, cmap, node_color: np.ndarray, faces: np.ndarray, alpha: float) -> np.ndarray:
    """Per-triangle RGBA from averaged nodal scalar (same norm/cmap as edges)."""
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fv = (
        node_color[faces[:, 0]]
        + node_color[faces[:, 1]]
        + node_color[faces[:, 2]]
    ) / 3.0
    rgba = np.asarray(sm.to_rgba(fv, alpha=alpha))
    return rgba


def _finalize_colorbar(cbar, color_label: str, *, tick_size: int = 22, label_size: int = 24) -> None:
    """Readable tick/label text; thin bar uses high aspect + small fraction elsewhere."""
    cbar.ax.tick_params(
        labelsize=tick_size,
        width=2.0,
        length=8,
        which="major",
    )
    cbar.set_label(color_label, fontsize=label_size, labelpad=14)
    cbar.outline.set_linewidth(1.5)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))


def load_sample(h5_path, sample_id):
    with h5py.File(h5_path, "r") as f:
        if "data" not in f or sample_id not in f["data"]:
            raise KeyError(f"Missing data/{sample_id} in {h5_path}")
        grp = f[f"data/{sample_id}"]
        nd = grp["nodal_data"][:]
        edges = grp["mesh_edge"][:]
        md = grp.get("metadata")
        meta = dict(md.attrs) if md is not None else {}
        gmeta = f.get("metadata")
        if gmeta is not None and "feature_names" in gmeta:
            feat_names = list(gmeta["feature_names"][:])
        else:
            feat_names = [f"f{i}".encode("ascii", errors="replace") for i in range(nd.shape[0])]
    return nd, edges, meta, feat_names


def world_positions(nd, t):
    """Return deformed (wx, wy, wz) at timestep t."""
    wx = nd[IX, 0, :] + nd[IDX, t, :]
    wy = nd[IY, 0, :] + nd[IDY, t, :]
    wz = nd[IZ, 0, :] + nd[IDZ, t, :]  # rest z is 0 for flag
    return wx, wy, wz


def displacement_magnitude_timeseries(nd: np.ndarray) -> np.ndarray:
    """L2 norm of (dx,dy,dz) per node per timestep; shape (T, N)."""
    if nd.shape[0] < IDZ + 1:
        raise ValueError(
            "displacement coloring needs nodal_data rows 0:6 (reference xyz + dx,dy,dz)"
        )
    d = nd[IDX : IDZ + 1].astype(np.float64, copy=False)
    return np.sqrt(np.sum(d * d, axis=0)).astype(np.float32)


def _world_axis_limits(wx_all, wy_all, wz_all, rel_margin=0.02):
    """
    Min/max world coordinates over all timesteps with a small *relative* margin
    on each axis so the mesh fills the frame (fixed padding makes thin objects
    look tiny in 3D).
    """
    def span_lim(arr):
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        span = hi - lo
        if span <= 0.0 or not np.isfinite(span):
            eps = max(abs(lo), 1.0, 1e-12) * 1e-6
            lo, hi = lo - eps, hi + eps
            span = hi - lo
        m = rel_margin * span
        return lo - m, hi + m

    return span_lim(wx_all), span_lim(wy_all), span_lim(wz_all)


# ------------------------------------------------------------------ #
#  2-D view renderer
# ------------------------------------------------------------------ #
def render_frame_2d(nd, edges, faces, t, norm, cmap, fig, ax,
                    axis_a, axis_b, lims_a, lims_b, node_color):
    ax.clear()
    wx, wy, wz = world_positions(nd, t)
    all_coords = {0: wx, 1: wy, 2: wz}
    ca = all_coords[axis_a]
    cb = all_coords[axis_b]
    coords_2d = np.stack([ca, cb], axis=1)

    color = node_color

    if faces is not None and faces.shape[0] > 0:
        polys = coords_2d[faces]
        fc = _triangle_face_rgba(norm, cmap, color, faces, TRI_FACE_ALPHA)
        pc = PolyCollection(
            polys,
            facecolors=fc,
            edgecolors="none",
            linewidths=0,
            zorder=1,
        )
        ax.add_collection(pc)

    segments = np.stack([coords_2d[edges[0]], coords_2d[edges[1]]], axis=1)
    edge_c = (color[edges[0]] + color[edges[1]]) / 2
    lc = LineCollection(
        segments,
        colors=cmap(norm(edge_c)),
        linewidths=0.55,
        alpha=0.98,
        zorder=2,
    )
    ax.add_collection(lc)
    ax.scatter(
        ca, cb, c=color, cmap=cmap, s=0.55, norm=norm, alpha=1.0, zorder=3,
    )

    ax.set_xlim(lims_a)
    ax.set_ylim(lims_b)
    ax.margins(0)
    ax.set_aspect("equal")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    fig.canvas.draw()
    buf = fig.canvas.get_renderer().buffer_rgba()
    return Image.frombuffer("RGBA", fig.canvas.get_width_height(), buf).convert("RGB")


# ------------------------------------------------------------------ #
#  3-D view renderer
# ------------------------------------------------------------------ #
def render_frame_3d(nd, edges, faces, t, norm, cmap, fig, ax, elev, azim,
                    xlims, ylims, zlims, node_color):
    ax.clear()
    wx, wy, wz = world_positions(nd, t)
    color = node_color

    if faces is not None and faces.shape[0] > 0:
        wxyz = np.column_stack([wx, wy, wz])
        verts = wxyz[faces]
        fc = _triangle_face_rgba(norm, cmap, color, faces, TRI_FACE_ALPHA)
        # mplot3d culls back faces; cloth/sphere windings often face away from iso view.
        verts_both = np.concatenate([verts, verts[:, ::-1, :]], axis=0)
        fc_both = np.concatenate([fc, fc], axis=0)
        poly = Poly3DCollection(
            verts_both,
            facecolors=fc_both,
            edgecolors="none",
            linewidths=0,
            zorder=1,
        )
        ax.add_collection3d(poly)

    # 3D edge segments
    starts = np.stack([wx[edges[0]], wy[edges[0]], wz[edges[0]]], axis=1)
    ends = np.stack([wx[edges[1]], wy[edges[1]], wz[edges[1]]], axis=1)
    segments = np.stack([starts, ends], axis=1)  # (E, 2, 3)
    edge_c = (color[edges[0]] + color[edges[1]]) / 2
    lc = Line3DCollection(
        segments,
        colors=cmap(norm(edge_c)),
        linewidths=0.5,
        alpha=0.97,
        zorder=2,
    )
    ax.add_collection3d(lc)
    ax.scatter(
        wx, wy, wz, c=color, cmap=cmap, s=0.55, norm=norm,
        alpha=1.0, depthshade=False, zorder=3,
    )

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_zlim(zlims)
    dx = xlims[1] - xlims[0]
    dy = ylims[1] - ylims[0]
    dz = zlims[1] - zlims[0]
    ax.set_box_aspect((dx, dy, dz))
    ax.margins(0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    fig.canvas.draw()
    buf = fig.canvas.get_renderer().buffer_rgba()
    return Image.frombuffer("RGBA", fig.canvas.get_width_height(), buf).convert("RGB")


# ------------------------------------------------------------------ #
#  GIF builder
# ------------------------------------------------------------------ #
def build_gif(frames, path, gif_fps):
    gif_duration_ms = int(1000 / gif_fps)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=gif_duration_ms, loop=0)


def make_2d_gif(nd, edges, faces, timesteps, norm, cmap, sample_id, num_nodes,
                color_label, axis_a, axis_b,
                lims_a, lims_b, view_tag, gif_fps,
                output_prefix, color_ts, progress_callback=None):
    fig, ax = plt.subplots(figsize=(15, 9), dpi=110)
    fig.subplots_adjust(left=0.02, right=0.95, top=0.98, bottom=0.02)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.06)
    cbar = fig.colorbar(sm, cax=cax, aspect=28)
    _finalize_colorbar(cbar, color_label)

    frames = []
    n = len(timesteps)
    for i, t in enumerate(timesteps):
        img = render_frame_2d(nd, edges, faces, t, norm, cmap, fig, ax,
                              axis_a, axis_b, lims_a, lims_b, color_ts[t])
        frames.append(img)
        msg = f"{view_tag}: frame {i+1}/{n}  ({(i+1)/n*100:.0f}%)"
        if progress_callback:
            progress_callback(msg)
        else:
            sys.stdout.write(f"\r  {msg}")
            sys.stdout.flush()
    plt.close(fig)
    if not progress_callback:
        print()

    out = f"{output_prefix}_s{sample_id}_{view_tag}.gif"
    build_gif(frames, out, gif_fps)
    msg = f"  -> {out}  ({n} frames)"
    if progress_callback:
        progress_callback(msg)
    else:
        print(msg)
    return out


def make_3d_gif(nd, edges, faces, timesteps, norm, cmap, sample_id, num_nodes,
                color_label, elev, azim, view_tag, gif_fps,
                xlims, ylims, zlims, output_prefix, color_ts,
                progress_callback=None):
    fig = plt.figure(figsize=(18, 11), dpi=110)
    fig.subplots_adjust(left=0.02, right=0.88, top=0.98, bottom=0.02)
    ax = fig.add_subplot(111, projection="3d")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # Narrow strip: small fraction + high aspect (height/width for vertical bar).
    cbar = fig.colorbar(
        sm,
        ax=ax,
        fraction=0.055,
        pad=0.02,
        shrink=0.96,
        aspect=24,
    )
    _finalize_colorbar(cbar, color_label)

    frames = []
    n = len(timesteps)
    for i, t in enumerate(timesteps):
        img = render_frame_3d(nd, edges, faces, t, norm, cmap, fig, ax, elev, azim,
                             xlims, ylims, zlims, color_ts[t])
        frames.append(img)
        msg = f"{view_tag}: frame {i+1}/{n}  ({(i+1)/n*100:.0f}%)"
        if progress_callback:
            progress_callback(msg)
        else:
            sys.stdout.write(f"\r  {msg}")
            sys.stdout.flush()
    plt.close(fig)
    if not progress_callback:
        print()

    out = f"{output_prefix}_s{sample_id}_{view_tag}.gif"
    build_gif(frames, out, gif_fps)
    msg = f"  -> {out}  ({n} frames)"
    if progress_callback:
        progress_callback(msg)
    else:
        print(msg)
    return out


def _decode_feat_name(feat_names, idx: int) -> str:
    raw = feat_names[idx]
    if isinstance(raw, bytes):
        return raw.decode()
    return str(raw)


# ------------------------------------------------------------------ #
#  Main Animation Generator
# ------------------------------------------------------------------ #
def generate_animations(
    h5_path,
    dt=0.02,
    frame_skip=4,
    gif_fps=20,
    progress_callback=None,
    sample_id=None,
    views=None,
    color_feature_index=None,
    color_by="auto",
    colormap=None,
):
    """
    Generate animated GIFs from HDF5 dataset.

    Args:
        h5_path: Path to HDF5 file
        dt: Time step in seconds
        frame_skip: Use every N-th stored timestep
        gif_fps: Frames per second for GIF
        progress_callback: Optional progress callback
        sample_id: HDF5 data key (string or int); random if None
        views: Subset of {'xz','xy','yz','3d_iso','3d_top'}; default all
        color_feature_index: With ``color_by='auto'``: nodal_data row (0--7); if None,
            uses stress (6) or z displacement (5) when stress is constant
        color_by: ``'auto'`` or ``'displacement'`` for L2 norm of (dx,dy,dz) from nodal_data[3:6]
        colormap: Matplotlib colormap name (default: ``DEFAULT_COLORMAP`` / ``turbo``)

    Returns:
        List of generated GIF filenames
    """
    try:
        with h5py.File(h5_path, "r") as f:
            sample_ids = list(f["data"].keys())

        if not sample_ids:
            raise ValueError("No samples found in dataset")

        sid = str(sample_id) if sample_id is not None else random.choice(sample_ids)
        if sid not in sample_ids:
            raise ValueError(f"sample_id {sid!r} not in HDF5 (have {len(sample_ids)} keys)")

        output_prefix = Path(h5_path).stem

        nd, edges, _meta, feat_names = load_sample(h5_path, sid)
        num_features, num_timesteps, num_nodes = nd.shape

        faces_tri = _edges_to_triangles_numpy(edges)
        faces_arg = faces_tri if faces_tri.shape[0] > 0 else None

        mode = (color_by or "auto").strip().lower()
        if mode in ("displacement", "disp", "disp_mag", "disp_magnitude", "total_displacement"):
            color_ts = displacement_magnitude_timeseries(nd)
            color_data = color_ts
            color_label = "|u| (total displacement)"
            msg_color = "Color: ||(dx, dy, dz)|| from nodal_data[3:6]"
        elif mode == "auto":
            if color_feature_index is not None:
                ci = int(color_feature_index)
                if not (0 <= ci < num_features):
                    raise ValueError(f"color_feature_index {ci} out of range for {num_features} features")
            else:
                stress_idx = num_features + COLOR_FEAT if num_features >= 2 else 0
                stress_idx = max(0, min(stress_idx, num_features - 1))
                ci = stress_idx
                if float(nd[ci].max()) == float(nd[ci].min()) and num_features > IDZ:
                    ci = min(IDZ, num_features - 1)
            color_ts = nd[ci]
            color_data = color_ts
            color_name = _decode_feat_name(feat_names, ci)
            color_label = color_name
            msg_color = f"Color feature: nodal_data[{ci}] = '{color_name}'"
        else:
            raise ValueError(f"Unknown color_by {color_by!r}; use 'auto' or 'displacement'")

        msg = f"Sample {sid}  |  nodes={num_nodes}  timesteps={num_timesteps}"
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

        ft_msg = (
            f"  triangle faces: {faces_tri.shape[0]}"
            if faces_arg is not None
            else "  triangle faces: (none — wireframe only)"
        )
        if progress_callback:
            progress_callback(ft_msg)
        else:
            print(ft_msg)

        if progress_callback:
            progress_callback(msg_color)
        else:
            print(msg_color)

        msg = f"  range: [{color_data.min():.6f}, {color_data.max():.6f}]"
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

        # Global axis limits from world positions across all timesteps
        wx_all = nd[IX, 0, :][None, :] + nd[IDX]   # (T, N)
        wy_all = nd[IY, 0, :][None, :] + nd[IDY]
        wz_all = nd[IZ, 0, :][None, :] + nd[IDZ]
        xlims, ylims, zlims = _world_axis_limits(wx_all, wy_all, wz_all)

        vmin, vmax = float(color_data.min()), float(color_data.max())
        if vmin == vmax:
            vmax = vmin + 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap_key = (colormap or DEFAULT_COLORMAP).strip()
        if cmap_key not in plt.colormaps:
            raise ValueError(
                f"Unknown colormap {cmap_key!r}; use a matplotlib registered name "
                f"(default {DEFAULT_COLORMAP!r})."
            )
        cmap = plt.colormaps[cmap_key]

        timesteps = list(range(0, num_timesteps, frame_skip))
        msg = f"Frames: {len(timesteps)} (every {frame_skip} steps)\n"
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

        if views is None:
            view_set = {"xz", "xy", "yz", "3d_iso", "3d_top"}
        else:
            view_set = {v.lower().strip() for v in views}
        valid = {"xz", "xy", "yz", "3d_iso", "3d_top"}
        unknown = view_set - valid
        if unknown:
            raise ValueError(f"Unknown --views entries {unknown}; use {sorted(valid)}")

        gifs = []

        if "xz" in view_set:
            gifs.append(make_2d_gif(
                nd, edges, faces_arg, timesteps, norm, cmap, sid, num_nodes,
                color_label, axis_a=0, axis_b=2,
                lims_a=xlims, lims_b=zlims, view_tag="XZ_front",
                gif_fps=gif_fps,
                output_prefix=output_prefix, color_ts=color_ts,
                progress_callback=progress_callback))

        if "xy" in view_set:
            gifs.append(make_2d_gif(
                nd, edges, faces_arg, timesteps, norm, cmap, sid, num_nodes,
                color_label, axis_a=0, axis_b=1,
                lims_a=xlims, lims_b=ylims, view_tag="XY_top",
                gif_fps=gif_fps,
                output_prefix=output_prefix, color_ts=color_ts,
                progress_callback=progress_callback))

        if "yz" in view_set:
            gifs.append(make_2d_gif(
                nd, edges, faces_arg, timesteps, norm, cmap, sid, num_nodes,
                color_label, axis_a=1, axis_b=2,
                lims_a=ylims, lims_b=zlims, view_tag="YZ_side",
                gif_fps=gif_fps,
                output_prefix=output_prefix, color_ts=color_ts,
                progress_callback=progress_callback))

        if "3d_iso" in view_set:
            gifs.append(make_3d_gif(
                nd, edges, faces_arg, timesteps, norm, cmap, sid, num_nodes,
                color_label, elev=25, azim=-60, view_tag="3D_iso",
                gif_fps=gif_fps,
                xlims=xlims, ylims=ylims, zlims=zlims,
                output_prefix=output_prefix, color_ts=color_ts,
                progress_callback=progress_callback))

        if "3d_top" in view_set:
            gifs.append(make_3d_gif(
                nd, edges, faces_arg, timesteps, norm, cmap, sid, num_nodes,
                color_label, elev=80, azim=-60, view_tag="3D_top",
                gif_fps=gif_fps,
                xlims=xlims, ylims=ylims, zlims=zlims,
                output_prefix=output_prefix, color_ts=color_ts,
                progress_callback=progress_callback))

        msg = "\nDone."
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

        return gifs

    except Exception as e:
        msg = f"ERROR: {str(e)}"
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)
        raise


create_animation = generate_animations


# ================================================================== #
#  Cross-platform File Dialog
# ================================================================== #
def browse_for_file():
    """
    Try to open native file dialog based on platform.
    Falls back to CLI input if unavailable.
    """
    import subprocess
    import platform

    system = platform.system()

    # Try Windows native dialog
    if system == "Windows":
        try:
            import ctypes.wintypes as wintypes
            from ctypes import windll

            # Initialize COM
            windll.ole32.CoInitializeEx(None, 0)

            # Create file open dialog
            file_dialog = windll.comdlg32.GetOpenFileNameA
            file_dialog.argtypes = [wintypes.c_char_p]

            # File filter
            filter_str = b"HDF5 Files (*.h5)\0*.h5\0All Files (*.*)\0*.*\0"

            # Prepare buffer
            file_path = ctypes.create_string_buffer(260)

            # OPENFILENAME structure
            class OPENFILENAME(ctypes.Structure):
                pass

            OPENFILENAME._fields_ = [
                ("lStructSize", wintypes.DWORD),
                ("hwndOwner", wintypes.HWND),
                ("hInstance", wintypes.HANDLE),
                ("lpstrFilter", wintypes.LPCSTR),
                ("lpstrCustomFilter", wintypes.LPCSTR),
                ("nMaxCustFilter", wintypes.DWORD),
                ("nFilterIndex", wintypes.DWORD),
                ("lpstrFile", wintypes.LPSTR),
                ("nMaxFile", wintypes.DWORD),
                ("lpstrFileTitle", wintypes.LPSTR),
                ("nMaxFileTitle", wintypes.DWORD),
                ("lpstrInitialDir", wintypes.LPCSTR),
                ("lpstrTitle", wintypes.LPCSTR),
                ("Flags", wintypes.DWORD),
                ("nFileOffset", wintypes.WORD),
                ("nFileExtension", wintypes.WORD),
                ("lpstrDefExt", wintypes.LPCSTR),
                ("lCustData", wintypes.LPARAM),
                ("lpfnHook", wintypes.c_void_p),
                ("lpTemplateName", wintypes.LPCSTR),
            ]

            ofn = OPENFILENAME()
            ofn.lStructSize = ctypes.sizeof(OPENFILENAME)
            ofn.lpstrFilter = filter_str
            ofn.lpstrFile = file_path
            ofn.nMaxFile = 260
            ofn.lpstrTitle = b"Select HDF5 File"
            ofn.Flags = 0x00000004  # OFN_FILEMUSTEXIST

            if windll.comdlg32.GetOpenFileNameA(ctypes.byref(ofn)):
                return file_path.value.decode("utf-8")
            return None
        except Exception as e:
            print(f"Warning: Could not open Windows file dialog: {e}\n")

    # Try Linux/Unix zenity
    elif system in ["Linux", "Darwin"]:
        try:
            result = subprocess.run(
                ["zenity", "--file-selection", "--file-filter=HDF5 Files (*.h5) | *.h5",
                 "--file-filter=All Files | *"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except FileNotFoundError:
            pass

        # Try kdialog as fallback for KDE
        try:
            result = subprocess.run(
                ["kdialog", "--getopenfilename", str(Path.home()),
                 "*.h5 | HDF5 Files"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except FileNotFoundError:
            pass

    print("Note: Could not open native file dialog.")
    return None


# ================================================================== #
#  Main Entry Point
# ================================================================== #
def main():
    parser = argparse.ArgumentParser(
        description="Mesh motion to GIF from HDF5 (data/<id>/nodal_data + mesh_edge)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/animate_h5.py dataset/sphere_simple.h5 --dt 0.01 --views 3d_iso --seed 0
  python scripts/animate_h5.py dataset/flag_simple.h5 --frame-skip 4 --colormap viridis
        """
    )
    parser.add_argument("h5_file", nargs="?", default=None, help="Path to HDF5 file")
    parser.add_argument("--dt", type=float, default=0.02, help="Seconds per stored timestep (DeepMind cloth dt is often 0.01)")
    parser.add_argument("--frame-skip", type=int, default=4, help="Use every N-th timestep in the GIF")
    parser.add_argument("--gif-fps", type=int, default=20, help="GIF playback FPS")
    parser.add_argument("--sample-id", type=str, default=None, help="HDF5 data/<id> key (default: random)")
    parser.add_argument(
        "--views",
        nargs="+",
        default=None,
        metavar="NAME",
        help="xz xy yz 3d_iso 3d_top (default: all five)",
    )
    parser.add_argument(
        "--color-index",
        type=int,
        default=None,
        help="With --color-by auto: nodal_data row for coloring (default: stress, or z-disp if flat)",
    )
    parser.add_argument(
        "--color-by",
        type=str,
        default="auto",
        choices=("auto", "displacement"),
        help="auto: single nodal row; displacement: ‖(dx,dy,dz)‖ from rows 3–6",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed when using random sample id")
    parser.add_argument(
        "--colormap",
        type=str,
        default=None,
        metavar="NAME",
        help=f"Matplotlib colormap (default: {DEFAULT_COLORMAP})",
    )

    args = parser.parse_args()

    # Get file path
    h5_file = args.h5_file
    if not h5_file:
        print("No H5 file specified. Opening file browser...")
        h5_file = browse_for_file()
        if not h5_file:
            print("No file selected. Exiting.")
            sys.exit(1)

    # Validate file
    if not os.path.exists(h5_file):
        print(f"Error: File not found: {h5_file}")
        sys.exit(1)

    print(f"\nGenerating animations from: {h5_file}")
    print(f"  Time step (dt): {args.dt} s")
    print(f"  Frame skip: {args.frame_skip}")
    print(f"  GIF FPS: {args.gif_fps}")
    print(f"  Colormap: {args.colormap or DEFAULT_COLORMAP}\n")

    if args.seed is not None:
        random.seed(args.seed)

    try:
        generate_animations(
            h5_file,
            dt=args.dt,
            frame_skip=args.frame_skip,
            gif_fps=args.gif_fps,
            sample_id=args.sample_id,
            views=args.views,
            color_feature_index=args.color_index,
            color_by=args.color_by,
            colormap=args.colormap,
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
