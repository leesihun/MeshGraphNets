#!/usr/bin/env python3
"""
Convert ConcreteShellFEA PerfectShell_LinearFEA graph batches (.pt) to MeshGraphNets HDF5
(dataset/DATASET_FORMAT.md): nodal_data [8, T, N], mesh_edge [2, E], metadata.

Graph source layout (after extracting Bath ZIPs):
  <root>/PerfectShell_LinearFEA/graphs/input_and_output/<split>/<batch_size>/batch_*.pt

Official loader (stress): values on disk are in Pa; their Dataset multiplies by 1e-3 for kPa.
This script converts Pa -> MPa for channel [6] (von Mises / nodal stress as stored).

Dependencies: torch, torch_geometric, h5py, numpy

Example:
  python convert_graphs_to_meshgraphnets_h5.py \\
    --graph-root "D:/ConcreteShellFEA/datasets/PerfectShell_LinearFEA/graphs/input_and_output/training/128" \\
    --output ../ConcreteShellFEA_train.h5 \\
    --split-name training \\
    --max-samples 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

FEATURE_NAMES = np.asarray(
    [
        b"x_coord",
        b"y_coord",
        b"z_coord",
        b"x_disp(mm)",
        b"y_disp(mm)",
        b"z_disp(mm)",
        b"stress(MPa)",
        b"Part No.",
    ]
)


def undirected_unique_edges(edge_index: np.ndarray) -> np.ndarray:
    """edge_index (2, E) int64 -> unique undirected edges with u <= v."""
    r0, r1 = edge_index[0], edge_index[1]
    a = np.minimum(r0, r1)
    b = np.maximum(r0, r1)
    pairs = np.stack([a, b], axis=1)
    pairs = np.unique(pairs, axis=0)
    return pairs.T.astype(np.int64)


def stress_pa_to_mpa(stress: torch.Tensor) -> np.ndarray:
    s = stress.detach().float().reshape(-1).cpu().numpy()
    return (s / 1e6).astype(np.float32)


def extract_displacement(data: object, num_nodes: int) -> np.ndarray:
    """Return [N, 3] displacement in mm if present, else zeros."""
    disp = np.zeros((num_nodes, 3), dtype=np.float32)
    for key in ("disp", "displacement", "u", "delta"):
        if hasattr(data, key):
            t = getattr(data, key)
            if isinstance(t, torch.Tensor):
                d = t.detach().float().cpu().numpy()
                if d.shape[0] == num_nodes and d.shape[-1] >= 3:
                    disp[:, :3] = d.reshape(num_nodes, -1)[:, :3].astype(np.float32)
                    return disp
    return disp


def _batch_paths(graph_root: Path) -> list[Path]:
    paths = sorted(
        graph_root.glob("batch_*.pt"),
        key=lambda p: int(p.stem.split("_", 1)[1]),
    )
    if not paths:
        raise FileNotFoundError(f"No batch_*.pt under {graph_root}")
    return paths


def iter_graphs_from_root(graph_root: Path):
    for path in _batch_paths(graph_root):
        batch_obj = torch.load(path, map_location="cpu", weights_only=False)
        for g in batch_obj.to_data_list():
            yield g


def convert_sample(data: object, sample_id: int, debug: bool = False) -> tuple:
    if not hasattr(data, "pos") or data.pos is None:
        raise ValueError(f"Sample {sample_id}: missing pos")
    pos = data.pos.detach().float().cpu().numpy()
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"Sample {sample_id}: pos shape {pos.shape}, expected [N,3]")
    num_nodes = pos.shape[0]

    if debug:
        keys = list(data.keys())
        print(f"  [debug] sample {sample_id} keys: {keys}")
        for k in keys:
            try:
                v = getattr(data, k)
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: {tuple(v.shape)} {v.dtype}")
            except Exception:
                pass

    if not hasattr(data, "stress") or data.stress is None:
        raise ValueError(f"Sample {sample_id}: missing stress")
    stress_mpa = stress_pa_to_mpa(data.stress)
    if stress_mpa.shape[0] != num_nodes:
        raise ValueError(
            f"Sample {sample_id}: stress len {stress_mpa.shape[0]} != N {num_nodes}"
        )

    disp = extract_displacement(data, num_nodes)

    ei = data.edge_index.detach().long().cpu().numpy()
    mesh_edge = undirected_unique_edges(ei)

    timesteps = 1
    nodal = np.zeros((8, timesteps, num_nodes), dtype=np.float32)
    nodal[0, 0, :] = pos[:, 0]
    nodal[1, 0, :] = pos[:, 1]
    nodal[2, 0, :] = pos[:, 2]
    nodal[3, 0, :] = disp[:, 0]
    nodal[4, 0, :] = disp[:, 1]
    nodal[5, 0, :] = disp[:, 2]
    nodal[6, 0, :] = stress_mpa
    nodal[7, 0, :] = 0.0

    buck = None
    if hasattr(data, "buckling") and data.buckling is not None:
        b = data.buckling
        if isinstance(b, torch.Tensor):
            buck = float(b.detach().float().cpu().reshape(-1)[0].item())
    return nodal, mesh_edge, buck


def main() -> None:
    def die(msg: str) -> None:
        print(msg, file=sys.stderr)
        raise SystemExit(1)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--graph-root",
        type=Path,
        required=True,
        help="Folder containing batch_0.pt, batch_1.pt, ... (e.g. .../training/128)",
    )
    ap.add_argument("--output", type=Path, required=True, help="Output .h5 path.")
    ap.add_argument(
        "--split-name",
        type=str,
        default="training",
        help="Label stored in metadata (training|validation|testing).",
    )
    ap.add_argument("--max-samples", type=int, default=None, help="Cap samples (for testing).")
    ap.add_argument("--debug-first", action="store_true", help="Print torch_geometric Data keys for sample 1.")
    args = ap.parse_args()

    root: Path = args.graph_root
    if not root.is_dir():
        die(f"Not a directory: {root}")

    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sample_ids: list[int] = []
    max_n = args.max_samples

    with h5py.File(out_path, "w") as h5:
        data_grp = h5.create_group("data")
        meta_root = h5.create_group("metadata")
        meta_root.create_dataset("feature_names", data=FEATURE_NAMES)
        split_grp = meta_root.create_group("splits")
        written = 0

        for graph in iter_graphs_from_root(root):
            if max_n is not None and written >= max_n:
                break
            sid = written + 1
            debug = args.debug_first and written == 0
            nodal, mesh_edge, buck = convert_sample(graph, sid, debug=debug)

            sg = data_grp.create_group(str(sid))
            sg.create_dataset("nodal_data", data=nodal, compression="gzip", compression_opts=4)
            sg.create_dataset("mesh_edge", data=mesh_edge)
            mg = sg.create_group("metadata")
            mg.attrs["num_nodes"] = nodal.shape[2]
            mg.attrs["num_edges"] = int(mesh_edge.shape[1])
            mg.attrs["num_timesteps"] = int(nodal.shape[1])
            mg.attrs["source"] = "ConcreteShellFEA/PerfectShell_LinearFEA/graphs"
            mg.attrs["split"] = args.split_name
            if buck is not None:
                mg.attrs["buckling_factor"] = buck

            sample_ids.append(sid)
            written += 1
            if written % 100 == 0:
                print(f"  wrote {written} samples...")

        n = len(sample_ids)
        if n == 0:
            die("No samples written.")
        h5.attrs["num_samples"] = n
        h5.attrs["num_features"] = 8
        h5.attrs["num_timesteps"] = 1

        ids = np.array(sample_ids, dtype=np.int64)
        split_grp.create_dataset(args.split_name, data=ids)

    print(f"Wrote {written} samples to {out_path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        raise SystemExit(1) from e
