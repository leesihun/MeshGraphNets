"""
Animated GIFs of transient response for a random flag_simple sample.
Color = nodal_data[-2] (stress, feature index 6).
Produces separate GIF files for XZ, XY, YZ, and 3D isometric views.

Tkinter GUI for selecting H5_PATH and configuring parameters.
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
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import threading

# Use absolute path: repo_root/dataset/flag_simple.h5
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
                    axis_a, axis_b, label_a, label_b, lims_a, lims_b, dt):
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
    ax.set_title(f"t = {t * dt:.2f} s  (step {t})", fontsize=11)
    ax.grid(True, alpha=0.2)

    fig.canvas.draw()
    buf = fig.canvas.get_renderer().buffer_rgba()
    return Image.frombuffer("RGBA", fig.canvas.get_width_height(), buf).convert("RGB")


# ------------------------------------------------------------------ #
#  3-D view renderer
# ------------------------------------------------------------------ #
def render_frame_3d(nd, edges, t, norm, cmap, fig, ax, elev, azim, dt):
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
    ax.set_title(f"t = {t * dt:.2f} s  (step {t})", fontsize=11)

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


def make_2d_gif(nd, edges, timesteps, norm, cmap, sample_id, num_nodes,
                color_label, axis_a, axis_b, label_a, label_b,
                lims_a, lims_b, view_tag, dt, gif_fps, progress_callback=None):
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
                              axis_a, axis_b, label_a, label_b, lims_a, lims_b, dt)
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

    out = f"flag_simple_s{sample_id}_{view_tag}.gif"
    build_gif(frames, out, gif_fps)
    msg = f"  -> {out}  ({n} frames)"
    if progress_callback:
        progress_callback(msg)
    else:
        print(msg)
    return out


def make_3d_gif(nd, edges, timesteps, norm, cmap, sample_id, num_nodes,
                color_label, elev, azim, view_tag, dt, gif_fps, progress_callback=None):
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
        img = render_frame_3d(nd, edges, t, norm, cmap, fig, ax, elev, azim, dt)
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

    out = f"flag_simple_s{sample_id}_{view_tag}.gif"
    build_gif(frames, out, gif_fps)
    msg = f"  -> {out}  ({n} frames)"
    if progress_callback:
        progress_callback(msg)
    else:
        print(msg)
    return out


# ------------------------------------------------------------------ #
#  Main Animation Generator
# ------------------------------------------------------------------ #
def generate_animations(h5_path, dt=0.02, frame_skip=4, gif_fps=20, progress_callback=None):
    """
    Generate animated GIFs from HDF5 dataset.

    Args:
        h5_path: Path to HDF5 file
        dt: Time step in seconds
        frame_skip: Skip every N frames
        gif_fps: Frames per second for GIF
        progress_callback: Optional callback function for progress updates

    Returns:
        List of generated GIF filenames
    """
    try:
        with h5py.File(h5_path, "r") as f:
            sample_ids = list(f["data"].keys())

        if not sample_ids:
            raise ValueError("No samples found in dataset")

        sample_id = random.choice(sample_ids)

        nd, edges, meta, feat_names = load_sample(h5_path, sample_id)
        num_features, num_timesteps, num_nodes = nd.shape

        color_name = feat_names[COLOR_FEAT] if isinstance(feat_names[COLOR_FEAT], str) \
            else feat_names[COLOR_FEAT].decode()
        color_data = nd[COLOR_FEAT]  # (T, N)

        msg = f"Sample {sample_id}  |  nodes={num_nodes}  timesteps={num_timesteps}"
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

        msg = f"Color feature: nodal_data[{COLOR_FEAT}] = '{color_name}'"
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

        msg = f"  range: [{color_data.min():.6f}, {color_data.max():.6f}]"
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

        if color_data.max() == color_data.min():
            msg = f"  WARNING: '{color_name}' is constant ({color_data.min():.4f}) for this dataset."
            if progress_callback:
                progress_callback(msg)
            else:
                print(msg)

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

        timesteps = list(range(0, num_timesteps, frame_skip))
        msg = f"Frames: {len(timesteps)} (every {frame_skip} steps)\n"
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

        gifs = []

        # --- View 1: X-Z (front) ---
        gif = make_2d_gif(nd, edges, timesteps, norm, cmap, sample_id, num_nodes,
                    color_label, axis_a=0, axis_b=2,
                    label_a="X (world)", label_b="Z (world)",
                    lims_a=xlims, lims_b=zlims, view_tag="XZ_front",
                    dt=dt, gif_fps=gif_fps, progress_callback=progress_callback)
        gifs.append(gif)

        # --- View 2: X-Y (top-down) ---
        gif = make_2d_gif(nd, edges, timesteps, norm, cmap, sample_id, num_nodes,
                    color_label, axis_a=0, axis_b=1,
                    label_a="X (world)", label_b="Y (world)",
                    lims_a=xlims, lims_b=ylims, view_tag="XY_top",
                    dt=dt, gif_fps=gif_fps, progress_callback=progress_callback)
        gifs.append(gif)

        # --- View 3: Y-Z (side) ---
        gif = make_2d_gif(nd, edges, timesteps, norm, cmap, sample_id, num_nodes,
                    color_label, axis_a=1, axis_b=2,
                    label_a="Y (world)", label_b="Z (world)",
                    lims_a=ylims, lims_b=zlims, view_tag="YZ_side",
                    dt=dt, gif_fps=gif_fps, progress_callback=progress_callback)
        gifs.append(gif)

        # --- View 4: 3D isometric ---
        gif = make_3d_gif(nd, edges, timesteps, norm, cmap, sample_id, num_nodes,
                    color_label, elev=25, azim=-60, view_tag="3D_iso",
                    dt=dt, gif_fps=gif_fps, progress_callback=progress_callback)
        gifs.append(gif)

        # --- View 5: 3D top-down ---
        gif = make_3d_gif(nd, edges, timesteps, norm, cmap, sample_id, num_nodes,
                    color_label, elev=80, azim=-60, view_tag="3D_top",
                    dt=dt, gif_fps=gif_fps, progress_callback=progress_callback)
        gifs.append(gif)

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


# ================================================================== #
#  Tkinter GUI
# ================================================================== #
class AnimationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Flag Simple Animation Generator")
        self.root.geometry("600x400")
        self.running = False

        # File selection
        file_frame = tk.Frame(root, pady=10)
        file_frame.pack(fill=tk.X, padx=10)

        tk.Label(file_frame, text="H5 File:", font=("Arial", 10)).pack(side=tk.LEFT)
        self.file_path = tk.StringVar()
        tk.Entry(file_frame, textvariable=self.file_path, width=50).pack(side=tk.LEFT, padx=5)
        tk.Button(file_frame, text="Browse", command=self.browse_file).pack(side=tk.LEFT)

        # Parameters
        params_frame = tk.LabelFrame(root, text="Parameters", padx=10, pady=10)
        params_frame.pack(fill=tk.X, padx=10, pady=5)

        # Time Step
        tk.Label(params_frame, text="Time Step (s):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.dt_var = tk.DoubleVar(value=0.02)
        tk.Spinbox(params_frame, from_=0.001, to=1.0, increment=0.001,
                  textvariable=self.dt_var, width=15).grid(row=0, column=1, sticky=tk.W)

        # Frame Skip
        tk.Label(params_frame, text="Frame Skip:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.frame_skip_var = tk.IntVar(value=4)
        tk.Spinbox(params_frame, from_=1, to=50, increment=1,
                  textvariable=self.frame_skip_var, width=15).grid(row=1, column=1, sticky=tk.W)

        # GIF FPS
        tk.Label(params_frame, text="GIF FPS:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.gif_fps_var = tk.IntVar(value=20)
        tk.Spinbox(params_frame, from_=1, to=60, increment=1,
                  textvariable=self.gif_fps_var, width=15).grid(row=2, column=1, sticky=tk.W)

        # Progress
        progress_frame = tk.LabelFrame(root, text="Progress", padx=10, pady=10)
        progress_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.progress_text = tk.Text(progress_frame, height=10, width=70, state=tk.DISABLED)
        self.progress_text.pack(fill=tk.BOTH, expand=True)

        # Buttons
        button_frame = tk.Frame(root, pady=10)
        button_frame.pack(fill=tk.X, padx=10)

        self.start_btn = tk.Button(button_frame, text="Generate Animations",
                                    command=self.start_animation, bg="#4CAF50", fg="white")
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tk.Button(button_frame, text="Stop", command=self.stop_animation,
                                  bg="#f44336", fg="white", state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

    def browse_file(self):
        file = filedialog.askopenfilename(
            title="Select HDF5 File",
            filetypes=[("HDF5 Files", "*.h5"), ("All Files", "*.*")]
        )
        if file:
            self.file_path.set(file)

    def log_progress(self, message):
        self.progress_text.config(state=tk.NORMAL)
        self.progress_text.insert(tk.END, message + "\n")
        self.progress_text.see(tk.END)
        self.progress_text.config(state=tk.DISABLED)
        self.root.update()

    def start_animation(self):
        h5_path = self.file_path.get()
        if not h5_path:
            messagebox.showerror("Error", "Please select an H5 file")
            return

        if not os.path.exists(h5_path):
            messagebox.showerror("Error", f"File not found: {h5_path}")
            return

        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress_text.config(state=tk.NORMAL)
        self.progress_text.delete(1.0, tk.END)
        self.progress_text.config(state=tk.DISABLED)

        # Run in background thread
        thread = threading.Thread(target=self._run_animation, args=(h5_path,), daemon=True)
        thread.start()

    def _run_animation(self, h5_path):
        try:
            generate_animations(
                h5_path,
                dt=self.dt_var.get(),
                frame_skip=self.frame_skip_var.get(),
                gif_fps=self.gif_fps_var.get(),
                progress_callback=self.log_progress
            )
            if self.running:
                self.root.after(0, lambda: messagebox.showinfo("Success", "Animations generated successfully!"))
        except Exception as e:
            if self.running:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to generate animations:\n{str(e)}"))
        finally:
            self.running = False
            self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))

    def stop_animation(self):
        self.running = False
        self.log_progress("Stopped.")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    gui = AnimationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
