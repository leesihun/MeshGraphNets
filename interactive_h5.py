"""
Interactive elemental viewer for MeshGraphNets HDF5 datasets.

Opens a live matplotlib window with widgets to explore a mesh sample:

  * Time slider   -- scrub through timesteps (the "video")
  * Feature radio -- choose which nodal field colours the elements
  * View radio    -- XY top / XZ / YZ / 3D warped surface
  * X/Y slice     -- range sliders that clip the mesh to a sub-region
  * Sample slider -- step through samples when the file has more than one
  * Toolbar       -- native matplotlib pan / zoom / save

Rendering matches animate_h5.py: filled triangular ELEMENTS (flat shaded),
reconstructed from the stored edge list, coloured per element.

Usage:
  python interactive_h5.py                       # file browser
  python interactive_h5.py dataset/ex1.h5        # open a dataset
  python interactive_h5.py dataset/ex1.h5 --sample 5

Needs an interactive matplotlib backend (TkAgg / QtAgg). Run it from a normal
desktop session, not a headless one.
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.tri import Triangulation
from matplotlib.widgets import Slider, RangeSlider, RadioButtons
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Reuse the exact element reconstruction / loading used by the GIF exporter.
from animate_h5 import build_triangles, load_sample, browse_for_file

# nodal_data feature indices (8, T, N)
IX, IY, IZ = 0, 1, 2
IDX, IDY, IDZ = 3, 4, 5


class MeshViewer:
    """Interactive elemental viewer over one HDF5 file."""

    def __init__(self, h5_path, sample_id=None):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            self.sample_ids = list(f["data"].keys())
        if not self.sample_ids:
            raise ValueError("No samples found in dataset")

        self.sample_idx = 0
        if sample_id is not None:
            sid = str(sample_id)
            if sid not in self.sample_ids:
                raise ValueError(f"Sample '{sid}' not in file "
                                 f"(have {self.sample_ids[:5]}...)")
            self.sample_idx = self.sample_ids.index(sid)

        # View / interaction state.
        self.view = "XY top"
        self.t = 0
        self.feat_key = None            # int index or 'disp_mag'
        self.dataset_name = Path(h5_path).stem

        self._load_sample(self.sample_ids[self.sample_idx])
        self._build_figure()
        self._render()

    # ------------------------------------------------------------------ #
    #  Data loading
    # ------------------------------------------------------------------ #
    def _load_sample(self, sample_id):
        nd, edges, meta, feat_names = load_sample(self.h5_path, sample_id)
        self.nd = nd
        self.num_features, self.num_timesteps, self.num_nodes = nd.shape
        self.feat_names = [n.decode() if isinstance(n, (bytes, bytearray)) else str(n)
                           for n in feat_names]
        self.triangles = build_triangles(edges, self.num_nodes)
        if len(self.triangles) == 0:
            raise ValueError("Could not reconstruct triangular elements from edges")

        # Colourable fields: physical channels 3..(num_features-2), plus |disp|.
        last_phys = self.num_features - 1  # exclude trailing part-number channel
        self.feat_options = []             # list of (label, key)
        for idx in range(3, last_phys):
            self.feat_options.append((self.feat_names[idx], idx))
        self.feat_options.append(("|disp| magnitude", "disp_mag"))

        # Pick a sensible default: the field with the largest variance.
        if self.feat_key is None or self.feat_key not in [k for _, k in self.feat_options]:
            best_key, best_var = self.feat_options[0][1], -1.0
            for _, key in self.feat_options:
                v = float(np.var(self._field_all(key)))
                if v > best_var:
                    best_key, best_var = key, v
            self.feat_key = best_key

        self.t = min(self.t, self.num_timesteps - 1)

        # Reference coords and global world-position extents (for axis limits).
        self.ref = nd[0:3, 0, :]                      # (3, N)
        wx_all = self.ref[IX][None, :] + nd[IDX]       # (T, N)
        wy_all = self.ref[IY][None, :] + nd[IDY]
        wz_all = self.ref[IZ][None, :] + nd[IDZ]
        pad = 0.02 * max(np.ptp(wx_all), np.ptp(wy_all), 1.0)
        self.xlims = (float(wx_all.min() - pad), float(wx_all.max() + pad))
        self.ylims = (float(wy_all.min() - pad), float(wy_all.max() + pad))
        zpad = 0.05 * max(np.ptp(wz_all), 1e-6)
        self.zlims = (float(wz_all.min() - zpad), float(wz_all.max() + zpad))

        # Reference (undeformed) extent drives the slice sliders.
        self.xslice = (float(self.ref[IX].min()), float(self.ref[IX].max()))
        self.yslice = (float(self.ref[IY].min()), float(self.ref[IY].max()))

        self._update_colour_limits()

    def _field_all(self, key):
        """Full (T, N) field for a colour key."""
        if key == "disp_mag":
            return np.sqrt(self.nd[IDX] ** 2 + self.nd[IDY] ** 2 + self.nd[IDZ] ** 2)
        return self.nd[key]

    def _field_t(self, key, t):
        """Field at timestep t, shape (N,)."""
        if key == "disp_mag":
            return np.sqrt(self.nd[IDX, t] ** 2 + self.nd[IDY, t] ** 2
                           + self.nd[IDZ, t] ** 2)
        return self.nd[key, t]

    def _field_label(self, key):
        if key == "disp_mag":
            return "|disp| magnitude"
        return self.feat_names[key]

    def _update_colour_limits(self):
        data = self._field_all(self.feat_key)
        vmin, vmax = float(np.min(data)), float(np.max(data))
        if vmin == vmax:
            vmax = vmin + 1.0
        self.norm = Normalize(vmin=vmin, vmax=vmax)

    def _world_positions(self, t):
        wx = self.ref[IX] + self.nd[IDX, t]
        wy = self.ref[IY] + self.nd[IDY, t]
        wz = self.ref[IZ] + self.nd[IDZ, t]
        return wx, wy, wz

    def _sliced_triangles(self, wx, wy):
        """Triangles whose element centroid lies inside the X/Y slice window."""
        cx = wx[self.triangles].mean(axis=1)
        cy = wy[self.triangles].mean(axis=1)
        mask = ((cx >= self.xslice[0]) & (cx <= self.xslice[1]) &
                (cy >= self.yslice[0]) & (cy <= self.yslice[1]))
        return self.triangles[mask], mask

    # ------------------------------------------------------------------ #
    #  Figure / widgets
    # ------------------------------------------------------------------ #
    def _build_figure(self):
        self.fig = plt.figure(figsize=(12, 7))
        self.fig.canvas.manager.set_window_title(f"MeshViewer - {self.dataset_name}")

        # Main drawing area (recreated when switching 2D <-> 3D).
        self.main_rect = [0.32, 0.22, 0.58, 0.70]
        self.ax = self.fig.add_axes(self.main_rect)

        # Persistent colorbar driven by a standalone ScalarMappable.
        self.sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=self.norm)
        self.sm.set_array([])
        self.cax = self.fig.add_axes([0.92, 0.22, 0.015, 0.70])
        self.cbar = self.fig.colorbar(self.sm, cax=self.cax)

        # View selector.
        ax_view = self.fig.add_axes([0.03, 0.70, 0.20, 0.22])
        ax_view.set_title("View", fontsize=9, loc="left")
        self.radio_view = RadioButtons(ax_view, ["XY top", "XZ", "YZ", "3D"])
        self.radio_view.on_clicked(self._on_view)

        # Feature selector.
        labels = [lbl for lbl, _ in self.feat_options]
        ax_feat = self.fig.add_axes([0.03, 0.40, 0.20, 0.26])
        ax_feat.set_title("Colour field", fontsize=9, loc="left")
        self.radio_feat = RadioButtons(ax_feat, labels)
        cur = self._field_label(self.feat_key)
        if cur in labels:
            self.radio_feat.set_active(labels.index(cur))
        self.radio_feat.on_clicked(self._on_feat)

        # Sample slider (only meaningful with >1 sample).
        ax_samp = self.fig.add_axes([0.42, 0.145, 0.48, 0.025])
        smax = max(len(self.sample_ids) - 1, 1)
        self.slider_sample = Slider(ax_samp, "Sample", 0, smax,
                                    valinit=self.sample_idx, valstep=1, valfmt="%d")
        self.slider_sample.on_changed(self._on_sample)
        if len(self.sample_ids) <= 1:
            ax_samp.set_facecolor("0.9")

        # Spatial slice range sliders.
        ax_xs = self.fig.add_axes([0.42, 0.100, 0.15, 0.025])
        self.slider_x = RangeSlider(ax_xs, "X slice",
                                    self.xslice[0], self.xslice[1],
                                    valinit=self.xslice)
        self.slider_x.on_changed(self._on_xslice)

        ax_ys = self.fig.add_axes([0.75, 0.100, 0.15, 0.025])
        self.slider_y = RangeSlider(ax_ys, "Y slice",
                                    self.yslice[0], self.yslice[1],
                                    valinit=self.yslice)
        self.slider_y.on_changed(self._on_yslice)

        # Time slider (scrub the video).
        ax_time = self.fig.add_axes([0.42, 0.055, 0.48, 0.025])
        tmax = max(self.num_timesteps - 1, 1)
        self.slider_time = Slider(ax_time, "Step", 0, tmax, valinit=self.t,
                                  valstep=1, valfmt="%d")
        self.slider_time.on_changed(self._on_time)
        if self.num_timesteps <= 1:
            ax_time.set_facecolor("0.9")

    def _recreate_axes(self, projection=None):
        self.ax.remove()
        if projection == "3d":
            self.ax = self.fig.add_axes(self.main_rect, projection="3d")
        else:
            self.ax = self.fig.add_axes(self.main_rect)

    # ------------------------------------------------------------------ #
    #  Rendering
    # ------------------------------------------------------------------ #
    def _render(self):
        color = self._field_t(self.feat_key, self.t)
        wx, wy, wz = self._world_positions(self.t)
        tris, mask = self._sliced_triangles(wx, wy)
        elem_c = color[tris].mean(axis=1) if len(tris) else np.zeros(0)

        title = (f"{self.dataset_name}  sample {self.sample_ids[self.sample_idx]}  "
                 f"step {self.t}/{self.num_timesteps - 1}  "
                 f"[{self._field_label(self.feat_key)}]  "
                 f"elems {len(tris)}/{len(self.triangles)}")

        if self.view == "3D":
            self.ax.clear()
            if len(tris):
                verts = np.stack([wx, wy, wz], axis=1)
                faces = verts[tris]
                pc = Poly3DCollection(faces, facecolors=self.sm.cmap(self.norm(elem_c)),
                                      edgecolors="none", linewidths=0.0)
                self.ax.add_collection3d(pc)
            self.ax.set_xlim(self.xlims)
            self.ax.set_ylim(self.ylims)
            self.ax.set_zlim(self.zlims)
            dx = self.xlims[1] - self.xlims[0]
            dy = self.ylims[1] - self.ylims[0]
            dz = self.zlims[1] - self.zlims[0]
            self.ax.set_box_aspect((dx, dy, max(dz, 0.35 * max(dx, dy))))
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z (warp)")
            self.ax.set_title(title, fontsize=10)
        else:
            axis_a, axis_b, la, lb, lima, limb = self._plane()
            coords = {0: wx, 1: wy, 2: wz}
            ca, cb = coords[axis_a], coords[axis_b]
            self.ax.clear()
            if len(tris):
                triang = Triangulation(ca, cb, tris)
                self.ax.tripcolor(triang, facecolors=elem_c, cmap=self.sm.cmap,
                                  norm=self.norm, shading="flat", edgecolors="none")
            self.ax.set_xlim(lima)
            self.ax.set_ylim(limb)
            self.ax.set_aspect("equal")
            self.ax.set_xlabel(la)
            self.ax.set_ylabel(lb)
            self.ax.grid(True, alpha=0.2)
            self.ax.set_title(title, fontsize=10)

        self.sm.set_clim(self.norm.vmin, self.norm.vmax)
        self.cbar.set_label(self._field_label(self.feat_key))
        self.fig.canvas.draw_idle()

    def _plane(self):
        if self.view == "XZ":
            return 0, 2, "X (world)", "Z (world)", self.xlims, self.zlims
        if self.view == "YZ":
            return 1, 2, "Y (world)", "Z (world)", self.ylims, self.zlims
        return 0, 1, "X (world)", "Y (world)", self.xlims, self.ylims  # XY top

    # ------------------------------------------------------------------ #
    #  Widget callbacks
    # ------------------------------------------------------------------ #
    def _on_view(self, label):
        want3d = (label == "3D")
        is3d = self.ax.name == "3d"
        self.view = label
        if want3d != is3d:
            self._recreate_axes(projection="3d" if want3d else None)
        self._render()

    def _on_feat(self, label):
        for lbl, key in self.feat_options:
            if lbl == label:
                self.feat_key = key
                break
        self._update_colour_limits()
        self._render()

    def _on_time(self, val):
        self.t = int(round(val))
        self._render()

    def _on_xslice(self, val):
        self.xslice = (float(val[0]), float(val[1]))
        self._render()

    def _on_yslice(self, val):
        self.yslice = (float(val[0]), float(val[1]))
        self._render()

    def _on_sample(self, val):
        idx = int(round(val))
        if idx == self.sample_idx:
            return
        self.sample_idx = idx
        self._load_sample(self.sample_ids[idx])
        # Reset slice sliders to the new sample's extent.
        self.slider_x.valmin, self.slider_x.valmax = self.xslice
        self.slider_x.ax.set_xlim(self.xslice)
        self.slider_x.set_val(self.xslice)
        self.slider_y.valmin, self.slider_y.valmax = self.yslice
        self.slider_y.ax.set_xlim(self.yslice)
        self.slider_y.set_val(self.yslice)
        self.slider_time.valmax = max(self.num_timesteps - 1, 1)
        self.slider_time.ax.set_xlim(0, max(self.num_timesteps - 1, 1))
        self._render()

    def show(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive elemental viewer for MeshGraphNets HDF5 datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python interactive_h5.py                       # file browser
  python interactive_h5.py dataset/ex1.h5        # open a dataset
  python interactive_h5.py dataset/ex1.h5 --sample 5
        """,
    )
    parser.add_argument("h5_file", nargs="?", default=None, help="Path to HDF5 file")
    parser.add_argument("--sample", default=None,
                        help="Sample id to open first (default: first sample)")
    args = parser.parse_args()

    h5_file = args.h5_file
    if not h5_file:
        print("No H5 file specified. Opening file browser...")
        h5_file = browse_for_file()
        if not h5_file:
            print("No file selected. Exiting.")
            sys.exit(1)

    if not os.path.exists(h5_file):
        print(f"Error: File not found: {h5_file}")
        sys.exit(1)

    print(f"Opening interactive viewer: {h5_file}")
    viewer = MeshViewer(h5_file, sample_id=args.sample)
    viewer.show()


if __name__ == "__main__":
    main()
