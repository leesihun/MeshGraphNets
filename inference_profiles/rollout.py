import os
import time

import h5py
import numpy as np
import torch
from torch_geometric.data import Data

from general_modules.edge_features import EDGE_FEATURE_DIM, compute_edge_attr
from general_modules.mesh_dataset import _compute_positional_features
from general_modules.removed_feature_guard import validate_no_removed_checkpoint
from general_modules.world_edges import HAS_TORCH_CLUSTER, compute_world_edges
from model.MeshGraphNets import MeshGraphNets

try:
    from model.coarsening import MultiscaleData
    from general_modules.multiscale_helpers import (
        attach_coarse_levels_to_graph,
        build_multiscale_hierarchy,
    )
    HAS_COARSENING = True
except ImportError:
    HAS_COARSENING = False


def _select_device(config):
    gpu_ids = config.get('gpu_ids')
    if not isinstance(gpu_ids, list):
        gpu_ids = [gpu_ids]

    if torch.cuda.is_available() and gpu_ids[0] >= 0:
        gpu_id = gpu_ids[0]
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU {gpu_id}, device: {device}")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")
    return device


def _load_checkpoint(config, device):
    model_path = config.get('modelpath')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    validate_no_removed_checkpoint(checkpoint, source=f"checkpoint '{model_path}'")
    return model_path, checkpoint


def _load_normalization(checkpoint):
    if 'normalization' not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain normalization statistics. "
            "Re-train or re-save the model with training code that stores them."
        )

    norm = checkpoint['normalization']
    node_mean = norm['node_mean']
    node_std = norm['node_std']
    edge_mean = norm['edge_mean']
    edge_std = norm['edge_std']
    delta_mean = norm['delta_mean']
    delta_std = norm['delta_std']

    if 'coarse_edge_means' in norm:
        coarse_edge_means = norm['coarse_edge_means']
        coarse_edge_stds = norm['coarse_edge_stds']
    elif 'coarse_edge_mean' in norm:
        coarse_edge_means = [norm['coarse_edge_mean']]
        coarse_edge_stds = [norm['coarse_edge_std']]
    else:
        coarse_edge_means = [edge_mean]
        coarse_edge_stds = [edge_std]

    print("  Normalization stats loaded from checkpoint")
    print(f"    node_mean:  {node_mean}")
    print(f"    node_std:   {node_std}")
    print(f"    delta_mean: {delta_mean}")
    print(f"    delta_std:  {delta_std}")

    return {
        'norm': norm,
        'node_mean': node_mean,
        'node_std': node_std,
        'edge_mean': edge_mean,
        'edge_std': edge_std,
        'delta_mean': delta_mean,
        'delta_std': delta_std,
        'coarse_edge_means': coarse_edge_means,
        'coarse_edge_stds': coarse_edge_stds,
    }


def _apply_checkpoint_config(config, checkpoint):
    if 'model_config' not in checkpoint:
        print("\n  WARNING: No model_config in checkpoint, using config file values")
        return

    model_config = checkpoint['model_config']
    print("\n  Model config loaded from checkpoint:")
    for k, v in model_config.items():
        old_val = config.get(k)
        config[k] = v
        if old_val is not None and old_val != v:
            print(f"    {k}: {old_val} -> {v} (overridden by checkpoint)")
        else:
            print(f"    {k}: {v}")


def _load_model(config, checkpoint, device):
    print("\nInitializing model...")
    model = MeshGraphNets(config, str(device)).to(device)

    if 'ema_state_dict' in checkpoint:
        ema_sd = checkpoint['ema_state_dict']
        model_sd = {k[len('module.'):]: v for k, v in ema_sd.items() if k.startswith('module.')}
        model.load_state_dict(model_sd)
        print("  Loaded EMA weights from checkpoint")
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("  Loaded training weights from checkpoint (no EMA available)")
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Checkpoint valid loss: {checkpoint.get('valid_loss', 'unknown')}")
    return model


def _parse_multiscale_config(config):
    multiscale_levels = int(config.get('multiscale_levels', 1))
    raw_ct = config.get('coarsening_type', 'bfs')
    if isinstance(raw_ct, list):
        coarsening_types = [str(t).strip().lower() for t in raw_ct]
    else:
        coarsening_types = [str(raw_ct).strip().lower()] * multiscale_levels
    if len(coarsening_types) == 1 and multiscale_levels > 1:
        coarsening_types = coarsening_types * multiscale_levels

    raw_vc = config.get('voronoi_clusters', None)
    if raw_vc is None:
        voronoi_clusters = [0] * multiscale_levels
    elif isinstance(raw_vc, list):
        voronoi_clusters = [int(v) for v in raw_vc]
    else:
        voronoi_clusters = [int(raw_vc)] * multiscale_levels
    if len(voronoi_clusters) == 1 and multiscale_levels > 1:
        voronoi_clusters = voronoi_clusters * multiscale_levels

    return multiscale_levels, coarsening_types, voronoi_clusters


def _build_static_graph_data(config, ref_pos, mesh_edge, num_nodes):
    edge_index = np.concatenate([mesh_edge, mesh_edge[[1, 0], :]], axis=1)

    num_pos_features = int(config.get('positional_features', 0))
    positional_encoding = str(config.get('positional_encoding', 'rwpe')).lower().strip()
    if num_pos_features > 0:
        pos_features = _compute_positional_features(
            ref_pos, edge_index, num_pos_features, positional_encoding
        )
        print(f"  Positional features: {pos_features.shape} ({positional_encoding})")
    else:
        pos_features = None

    coarse_hierarchy = None
    use_multiscale = config.get('use_multiscale', False)
    multiscale_levels, coarsening_types, voronoi_clusters = _parse_multiscale_config(config)
    if use_multiscale:
        if not HAS_COARSENING:
            raise ImportError("use_multiscale=True but model/coarsening.py could not be imported")
        coarse_hierarchy = build_multiscale_hierarchy(
            edge_index,
            num_nodes,
            ref_pos,
            multiscale_levels,
            coarsening_types,
            voronoi_clusters,
            bipartite_unpool=config.get('bipartite_unpool', False),
        )
        current_n_report = num_nodes
        for level, entry in enumerate(coarse_hierarchy):
            method = coarsening_types[level] if level < len(coarsening_types) else 'bfs'
            n_c = entry['n_c']
            print(
                f"  Coarsening level {level} ({method}): "
                f"{current_n_report} -> {n_c} nodes ({n_c/current_n_report*100:.1f}%)"
            )
            current_n_report = n_c

    return edge_index, pos_features, coarse_hierarchy


def _save_rollout_h5(
    output_path,
    sample_id,
    ref_pos,
    mesh_edge,
    part_ids,
    all_states,
    output_dim,
    steps_this_sample,
    model_path,
    config_filename,
    total_rollout_time,
    norm_data,
):
    output_path_abs = os.path.abspath(output_path)
    print(f"\nSaving results to: {output_path_abs}")

    num_nodes = ref_pos.shape[0]
    with h5py.File(output_path, 'w') as f:
        f.attrs['num_samples'] = 1
        f.attrs['num_features'] = 3 + output_dim + 1
        f.attrs['num_timesteps'] = steps_this_sample + 1

        data_grp = f.create_group('data')
        sample_grp = data_grp.create_group(str(sample_id))

        num_save_features = 3 + output_dim + 1
        nodal_data = np.zeros(
            (num_save_features, steps_this_sample + 1, num_nodes), dtype=np.float32
        )

        nodal_data[0, :, :] = ref_pos[:, 0]
        nodal_data[1, :, :] = ref_pos[:, 1]
        nodal_data[2, :, :] = ref_pos[:, 2]

        for ch in range(output_dim):
            nodal_data[3 + ch, :, :] = all_states[:, :, ch]

        if part_ids is not None:
            nodal_data[3 + output_dim, :, :] = part_ids[np.newaxis, :]
        else:
            nodal_data[3 + output_dim, :, :] = 0

        sample_grp.create_dataset(
            'nodal_data', data=nodal_data,
            compression='gzip', compression_opts=4,
        )
        sample_grp.create_dataset('mesh_edge', data=mesh_edge)

        meta_grp = sample_grp.create_group('metadata')
        meta_grp.attrs['sample_id'] = sample_id
        meta_grp.attrs['num_nodes'] = num_nodes
        meta_grp.attrs['num_edges'] = mesh_edge.shape[1]
        meta_grp.attrs['num_timesteps'] = steps_this_sample + 1
        meta_grp.attrs['model_path'] = model_path
        meta_grp.attrs['config_file'] = config_filename
        meta_grp.attrs['total_rollout_time_s'] = total_rollout_time

        all_feature_names = [
            b'x_coord', b'y_coord', b'z_coord',
            b'x_disp(mm)', b'y_disp(mm)', b'z_disp(mm)',
            b'stress(MPa)', b'Part No.',
        ]
        feature_names = np.array(all_feature_names[:3 + output_dim] + [b'Part No.'])
        feature_min = np.array([nodal_data[i].min() for i in range(num_save_features)], dtype=np.float32)
        feature_max = np.array([nodal_data[i].max() for i in range(num_save_features)], dtype=np.float32)
        feature_mean = np.array([nodal_data[i].mean() for i in range(num_save_features)], dtype=np.float32)
        feature_std = np.array([nodal_data[i].std() for i in range(num_save_features)], dtype=np.float32)

        meta_grp.create_dataset('feature_min', data=feature_min)
        meta_grp.create_dataset('feature_max', data=feature_max)
        meta_grp.create_dataset('feature_mean', data=feature_mean)
        meta_grp.create_dataset('feature_std', data=feature_std)

        global_meta = f.create_group('metadata')
        global_meta.create_dataset('feature_names', data=feature_names)

        norm_grp = global_meta.create_group('normalization_params')
        norm_grp.create_dataset('node_mean', data=norm_data['node_mean'])
        norm_grp.create_dataset('node_std', data=norm_data['node_std'])
        norm_grp.create_dataset('edge_mean', data=norm_data['edge_mean'])
        norm_grp.create_dataset('edge_std', data=norm_data['edge_std'])
        norm_grp.create_dataset('delta_mean', data=norm_data['delta_mean'])
        norm_grp.create_dataset('delta_std', data=norm_data['delta_std'])
        f.flush()

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved ({file_size_mb:.1f} MB)")
    print("  File is now closed and ready to read.")


def run_rollout(config, config_filename='config.txt'):
    """Perform deterministic autoregressive time-transient rollout inference."""
    print("\n" + "=" * 60)
    print("AUTOREGRESSIVE ROLLOUT INFERENCE")
    print("=" * 60)

    device = _select_device(config)
    model_path, checkpoint = _load_checkpoint(config, device)
    norm_data = _load_normalization(checkpoint)
    _apply_checkpoint_config(config, checkpoint)

    norm = norm_data['norm']
    use_node_types = config.get('use_node_types')
    node_type_to_idx = norm.get('node_type_to_idx')
    num_node_types = norm.get('num_node_types')
    if use_node_types and num_node_types is not None and num_node_types > 0:
        config['num_node_types'] = num_node_types
        print(f"  Node types: {num_node_types} types, mapping: {node_type_to_idx}")

    use_world_edges = config.get('use_world_edges')
    world_edge_radius = norm.get('world_edge_radius')
    world_max_num_neighbors = config.get('world_max_num_neighbors', 64)
    requested_backend = config.get('world_edge_backend', 'scipy_kdtree').lower()
    if requested_backend == 'torch_cluster' and HAS_TORCH_CLUSTER:
        world_edge_backend = 'torch_cluster'
    else:
        world_edge_backend = 'scipy_kdtree'
    if use_world_edges:
        print(f"  World edges: radius={world_edge_radius}, backend={world_edge_backend}")

    model = _load_model(config, checkpoint, device)

    dataset_dir = config.get('infer_dataset')
    num_rollout_steps = config.get('infer_timesteps')
    input_dim = config.get('input_var')
    output_dim = config.get('output_var')

    print("\nLoading initial condition...")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Rollout steps: {num_rollout_steps}")

    with h5py.File(dataset_dir, 'r') as f:
        sample_ids = sorted([int(k) for k in f['data'].keys()])

    print(f"  Found {len(sample_ids)} samples: {sample_ids[:10]}{'...' if len(sample_ids) > 10 else ''}")

    for sample_id in sample_ids:
        with h5py.File(dataset_dir, 'r') as f:
            nodal_data = f[f'data/{sample_id}/nodal_data'][:]
            mesh_edge = f[f'data/{sample_id}/mesh_edge'][:]

        num_features, num_timesteps, num_nodes = nodal_data.shape
        print(f"  Data shape: {nodal_data.shape} (features, timesteps, nodes)")
        print(f"  Mesh edges: {mesh_edge.shape[1]} (unidirectional)")

        steps_this_sample = num_rollout_steps
        if steps_this_sample is None:
            if num_timesteps > 1:
                steps_this_sample = num_timesteps - 1
                print(f"  Auto-set rollout steps to {steps_this_sample} (full trajectory)")
            else:
                raise ValueError(
                    f"infer_timesteps not specified and dataset has only {num_timesteps} timestep(s). "
                    "Please set infer_timesteps in config.txt"
                )

        if steps_this_sample > num_timesteps and num_timesteps > 1:
            print(
                f"  INFO: Requested {steps_this_sample} steps, dataset has {num_timesteps} timesteps. "
                f"Will generate {steps_this_sample} new predictions beyond the dataset."
            )

        ref_pos = nodal_data[:3, 0, :].T
        initial_state = nodal_data[3:3 + input_dim, 0, :].T
        if use_node_types and num_features > 7:
            part_ids = nodal_data[-1, 0, :].astype(np.int32)
        else:
            part_ids = None

        edge_index, pos_features, coarse_hierarchy = _build_static_graph_data(
            config, ref_pos, mesh_edge, num_nodes
        )

        print(f"  Reference positions: {ref_pos.shape}")
        print(f"  Initial state: {initial_state.shape}")
        print(f"  Bidirectional edges: {edge_index.shape[1]}")

        print(f"\n{'=' * 60}")
        print(f"Starting rollout: {steps_this_sample} steps")
        print(f"{'=' * 60}")

        all_states = np.zeros((steps_this_sample + 1, num_nodes, output_dim), dtype=np.float32)
        all_states[0] = initial_state[:, :output_dim]

        current_state = initial_state.copy()
        rollout_start_time = time.time()

        with torch.no_grad():
            for step in range(steps_this_sample):
                step_start = time.time()

                if pos_features is not None:
                    x_raw = np.concatenate([current_state, pos_features], axis=1)
                else:
                    x_raw = current_state
                x_norm = (x_raw - norm_data['node_mean']) / norm_data['node_std']

                if use_node_types and part_ids is not None and node_type_to_idx is not None:
                    node_type_indices = np.array(
                        [node_type_to_idx[int(t)] for t in part_ids], dtype=np.int32
                    )
                    node_type_onehot = np.zeros((num_nodes, num_node_types), dtype=np.float32)
                    node_type_onehot[np.arange(num_nodes), node_type_indices] = 1.0
                    x_norm = np.concatenate([x_norm, node_type_onehot], axis=1)

                displacement = current_state[:, :3]
                deformed_pos = ref_pos + displacement
                edge_attr_raw = compute_edge_attr(ref_pos, deformed_pos, edge_index)
                edge_attr_norm = (edge_attr_raw - norm_data['edge_mean']) / norm_data['edge_std']

                DataClass = MultiscaleData if config.get('use_multiscale', False) else Data
                graph = DataClass(
                    x=torch.from_numpy(x_norm.astype(np.float32)).to(device),
                    edge_index=torch.from_numpy(edge_index).long().to(device),
                    edge_attr=torch.from_numpy(edge_attr_norm.astype(np.float32)).to(device),
                    pos=torch.from_numpy(ref_pos.astype(np.float32)).to(device),
                )

                if use_world_edges and world_edge_radius is not None:
                    world_ei, world_ea = compute_world_edges(
                        ref_pos,
                        deformed_pos,
                        edge_index,
                        radius=world_edge_radius,
                        max_num_neighbors=world_max_num_neighbors,
                        backend=world_edge_backend,
                        device=device,
                        edge_mean=norm_data['edge_mean'],
                        edge_std=norm_data['edge_std'],
                    )
                    graph.world_edge_index = torch.from_numpy(world_ei).long().to(device)
                    graph.world_edge_attr = torch.from_numpy(world_ea.astype(np.float32)).to(device)
                else:
                    graph.world_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                    graph.world_edge_attr = torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float32, device=device)

                if config.get('use_multiscale', False) and coarse_hierarchy is not None:
                    use_cwe = bool(config.get('coarse_world_edges', False))
                    world_ei_for_coarse = (
                        graph.world_edge_index.cpu().numpy()
                        if use_world_edges and use_cwe else None
                    )
                    attach_coarse_levels_to_graph(
                        graph,
                        coarse_hierarchy,
                        ref_pos,
                        deformed_pos,
                        norm_data['coarse_edge_means'],
                        norm_data['coarse_edge_stds'],
                        device=device,
                        world_edge_index=world_ei_for_coarse,
                    )

                predicted_delta_norm, _ = model(graph)
                predicted_delta = (
                    predicted_delta_norm.cpu().numpy() * norm_data['delta_std']
                    + norm_data['delta_mean']
                )

                current_state[:, :output_dim] = current_state[:, :output_dim] + predicted_delta
                all_states[step + 1] = current_state[:, :output_dim]

                step_time = time.time() - step_start
                if step % max(1, steps_this_sample // 20) == 0 or step == steps_this_sample - 1:
                    disp_mag = np.linalg.norm(current_state[:, :3], axis=1)
                    print(
                        f"  Step {step+1:>4d}/{steps_this_sample} | "
                        f"time: {step_time:.3f}s | "
                        f"disp range: [{disp_mag.min():.4e}, {disp_mag.max():.4e}]"
                    )

        total_rollout_time = time.time() - rollout_start_time
        if steps_this_sample > 0:
            print(
                f"\nRollout completed in {total_rollout_time:.2f}s "
                f"({total_rollout_time/steps_this_sample:.3f}s/step)"
            )
        else:
            print(f"\nRollout completed in {total_rollout_time:.2f}s (no steps executed)")

        output_dir = config.get('inference_output_dir', 'outputs/rollout')
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"rollout_sample{sample_id}_steps{steps_this_sample}.h5"
        output_path = os.path.join(output_dir, output_filename)

        _save_rollout_h5(
            output_path,
            sample_id,
            ref_pos,
            mesh_edge,
            part_ids,
            all_states,
            output_dim,
            steps_this_sample,
            model_path,
            config_filename,
            total_rollout_time,
            norm_data,
        )

    print(f"\nRollout inference complete. Processed {len(sample_ids)} scene(s).")
