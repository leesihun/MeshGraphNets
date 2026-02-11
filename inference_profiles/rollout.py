import os
import time

import h5py
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial import KDTree

from model.MeshGraphNets import MeshGraphNets

# Try to import torch_cluster for GPU-accelerated world edges
try:
    from torch_cluster import radius_graph
    HAS_TORCH_CLUSTER = True
except ImportError:
    HAS_TORCH_CLUSTER = False


def run_rollout(config, config_filename='config.txt'):
    """
    Perform autoregressive time-transient rollout inference.

    Given an initial condition from an HDF5 dataset and a pretrained model checkpoint,
    this function iteratively predicts the next state from the current state, building
    up a full time-transient trajectory.

    The rollout loop:
        1. Load initial state at t=rollout_start_step
        2. For each step t -> t+1:
            a. Normalize current state -> build graph
            b. Forward pass -> predicted normalized delta
            c. Denormalize delta
            d. Update state: state_{t+1} = state_t + delta
        3. Save all predicted timesteps to HDF5

    Args:
        config: Configuration dictionary (from load_config)
        config_filename: Path to the config file
    """
    print("\n" + "=" * 60)
    print("AUTOREGRESSIVE ROLLOUT INFERENCE")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Setup device
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 2. Load checkpoint (model weights + normalization statistics)
    # -------------------------------------------------------------------------
    model_path = config.get('model_path', 'outputs/best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract normalization statistics from checkpoint
    if 'normalization' not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain normalization statistics. "
            "Re-train or re-save the model with the updated training code that "
            "includes normalization stats in the checkpoint."
        )

    norm = checkpoint['normalization']
    node_mean = norm['node_mean']  # np.ndarray [input_var]
    node_std = norm['node_std']    # np.ndarray [input_var]
    edge_mean = norm['edge_mean']  # np.ndarray [4]
    edge_std = norm['edge_std']    # np.ndarray [4]
    delta_mean = norm['delta_mean']  # np.ndarray [output_var]
    delta_std = norm['delta_std']    # np.ndarray [output_var]

    print(f"  Normalization stats loaded from checkpoint")
    print(f"    node_mean:  {node_mean}")
    print(f"    node_std:   {node_std}")
    print(f"    delta_mean: {delta_mean}")
    print(f"    delta_std:  {delta_std}")

    # Node type info
    use_node_types = config.get('use_node_types', False)
    node_type_to_idx = norm.get('node_type_to_idx', None)
    num_node_types = norm.get('num_node_types', 0)

    if use_node_types and num_node_types > 0:
        config['num_node_types'] = num_node_types
        print(f"  Node types: {num_node_types} types, mapping: {node_type_to_idx}")

    # World edge info
    use_world_edges = config.get('use_world_edges', False)
    world_edge_radius = norm.get('world_edge_radius', None)
    world_max_num_neighbors = config.get('world_max_num_neighbors', 64)

    # Determine world edge backend
    requested_backend = config.get('world_edge_backend', 'scipy_kdtree').lower()
    if requested_backend == 'torch_cluster' and HAS_TORCH_CLUSTER:
        world_edge_backend = 'torch_cluster'
    else:
        world_edge_backend = 'scipy_kdtree'

    if use_world_edges:
        print(f"  World edges: radius={world_edge_radius}, backend={world_edge_backend}")

    # -------------------------------------------------------------------------
    # 3. Initialize model
    # -------------------------------------------------------------------------
    print("\nInitializing model...")
    model = MeshGraphNets(config, str(device)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Checkpoint valid loss: {checkpoint.get('valid_loss', 'unknown')}")

    # -------------------------------------------------------------------------
    # 4. Load initial condition from HDF5
    # -------------------------------------------------------------------------
    dataset_dir = config.get('dataset_dir')
    inference_sample_id = config.get('inference_sample_id', 0)
    rollout_start_step = config.get('rollout_start_step', 0)
    num_rollout_steps = config.get('num_rollout_steps')
    input_dim = config.get('input_var')
    output_dim = config.get('output_var')

    print(f"\nLoading initial condition...")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Sample ID: {inference_sample_id}")
    print(f"  Start step: {rollout_start_step}")
    print(f"  Rollout steps: {num_rollout_steps}")

    with h5py.File(dataset_dir, 'r') as f:
        if f'data/{inference_sample_id}' not in f:
            available = sorted([int(k) for k in f['data'].keys()])
            raise ValueError(
                f"Sample ID {inference_sample_id} not found in dataset. "
                f"Available IDs: {available[:20]}{'...' if len(available) > 20 else ''}"
            )

        nodal_data = f[f'data/{inference_sample_id}/nodal_data'][:]  # [features, time, nodes]
        mesh_edge = f[f'data/{inference_sample_id}/mesh_edge'][:]    # [2, M]

    num_features, num_timesteps, num_nodes = nodal_data.shape
    print(f"  Data shape: {nodal_data.shape} (features, timesteps, nodes)")
    print(f"  Mesh edges: {mesh_edge.shape[1]} (unidirectional)")

    # Validate rollout range
    if num_rollout_steps is None:
        num_rollout_steps = num_timesteps - 1 - rollout_start_step
        print(f"  Auto-set num_rollout_steps to {num_rollout_steps} (full trajectory)")

    if rollout_start_step + num_rollout_steps >= num_timesteps:
        max_steps = num_timesteps - 1 - rollout_start_step
        print(f"  WARNING: Requested {num_rollout_steps} steps but only {max_steps} available. Clamping.")
        num_rollout_steps = max_steps

    # Extract initial state
    # nodal_data layout: [x, y, z, x_disp, y_disp, z_disp, stress, (part_number)]
    ref_pos = nodal_data[:3, rollout_start_step, :].T  # [N, 3] reference position
    initial_state = nodal_data[3:3+input_dim, rollout_start_step, :].T  # [N, input_dim]

    # Extract node types if enabled
    if use_node_types and num_features > 7:
        part_ids = nodal_data[-1, 0, :].astype(np.int32)  # [N]
    else:
        part_ids = None

    # Make edges bidirectional
    edge_index = np.concatenate([mesh_edge, mesh_edge[[1, 0], :]], axis=1)  # [2, 2M]

    print(f"  Reference positions: {ref_pos.shape}")
    print(f"  Initial state: {initial_state.shape}")
    print(f"  Bidirectional edges: {edge_index.shape[1]}")

    # -------------------------------------------------------------------------
    # 5. Autoregressive rollout
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Starting rollout: {num_rollout_steps} steps")
    print(f"{'=' * 60}")

    # Storage for all predicted states: [num_steps+1, N, output_dim]
    all_states = np.zeros((num_rollout_steps + 1, num_nodes, output_dim), dtype=np.float32)
    all_states[0] = initial_state[:, :output_dim]

    current_state = initial_state.copy()  # [N, input_dim], mutable
    rollout_start_time = time.time()

    with torch.no_grad():
        for step in range(num_rollout_steps):
            step_start = time.time()

            # --- a. Build node features ---
            x_raw = current_state  # [N, input_dim]

            # --- b. Normalize node features ---
            x_norm = (x_raw - node_mean) / node_std  # [N, input_dim]

            # --- c. Add node type one-hot if enabled ---
            if use_node_types and part_ids is not None and node_type_to_idx is not None:
                node_type_indices = np.array(
                    [node_type_to_idx[int(t)] for t in part_ids], dtype=np.int32
                )
                node_type_onehot = np.zeros((num_nodes, num_node_types), dtype=np.float32)
                node_type_onehot[np.arange(num_nodes), node_type_indices] = 1.0
                x_norm = np.concatenate([x_norm, node_type_onehot], axis=1)

            # --- d. Compute deformed position ---
            displacement = current_state[:, :3]  # [N, 3]
            deformed_pos = ref_pos + displacement  # [N, 3]

            # --- e. Compute edge features ---
            src_idx = edge_index[0]
            dst_idx = edge_index[1]
            relative_pos = deformed_pos[dst_idx] - deformed_pos[src_idx]  # [2M, 3]
            distance = np.linalg.norm(relative_pos, axis=1, keepdims=True)  # [2M, 1]
            edge_attr_raw = np.concatenate([relative_pos, distance], axis=1)  # [2M, 4]

            # --- f. Normalize edge features ---
            edge_attr_norm = (edge_attr_raw - edge_mean) / edge_std

            # --- g. Build graph Data object ---
            graph = Data(
                x=torch.from_numpy(x_norm.astype(np.float32)).to(device),
                edge_index=torch.from_numpy(edge_index).long().to(device),
                edge_attr=torch.from_numpy(edge_attr_norm.astype(np.float32)).to(device),
                pos=torch.from_numpy(ref_pos.astype(np.float32)).to(device),
            )

            # --- h. Compute world edges if enabled ---
            if use_world_edges and world_edge_radius is not None:
                world_ei, world_ea = _compute_world_edges(
                    deformed_pos, edge_index, world_edge_radius,
                    world_max_num_neighbors, world_edge_backend,
                    edge_mean, edge_std
                )
                graph.world_edge_index = torch.from_numpy(world_ei).long().to(device)
                graph.world_edge_attr = torch.from_numpy(world_ea.astype(np.float32)).to(device)
            else:
                graph.world_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                graph.world_edge_attr = torch.zeros((0, 4), dtype=torch.float32, device=device)

            # --- i. Forward pass ---
            predicted_delta_norm, _ = model(graph)  # [N, output_var]

            # --- j. Denormalize delta ---
            predicted_delta_norm_np = predicted_delta_norm.cpu().numpy()
            predicted_delta = predicted_delta_norm_np * delta_std + delta_mean  # [N, output_var]

            # --- k. Update state ---
            current_state[:, :output_dim] = current_state[:, :output_dim] + predicted_delta

            # --- l. Store result ---
            all_states[step + 1] = current_state[:, :output_dim]

            step_time = time.time() - step_start
            if step % max(1, num_rollout_steps // 20) == 0 or step == num_rollout_steps - 1:
                disp_mag = np.linalg.norm(current_state[:, :3], axis=1)
                print(
                    f"  Step {step+1:>4d}/{num_rollout_steps} | "
                    f"time: {step_time:.3f}s | "
                    f"disp range: [{disp_mag.min():.4e}, {disp_mag.max():.4e}] | "
                    f"stress range: [{current_state[:, 3].min():.4e}, {current_state[:, 3].max():.4e}]"
                )

    total_rollout_time = time.time() - rollout_start_time
    print(f"\nRollout completed in {total_rollout_time:.2f}s ({total_rollout_time/num_rollout_steps:.3f}s/step)")

    # -------------------------------------------------------------------------
    # 6. Save results to HDF5
    # -------------------------------------------------------------------------
    output_dir = config.get('inference_output_dir', 'outputs/rollout')
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"rollout_sample{inference_sample_id}_t{rollout_start_step}_steps{num_rollout_steps}.h5"
    output_path = os.path.join(output_dir, output_filename)

    print(f"\nSaving results to: {output_path}")

    with h5py.File(output_path, 'w') as f:
        # Metadata
        meta = f.create_group('metadata')
        meta.attrs['sample_id'] = inference_sample_id
        meta.attrs['rollout_start_step'] = rollout_start_step
        meta.attrs['num_rollout_steps'] = num_rollout_steps
        meta.attrs['num_nodes'] = num_nodes
        meta.attrs['num_edges'] = mesh_edge.shape[1]
        meta.attrs['model_path'] = model_path
        meta.attrs['config_file'] = config_filename
        meta.attrs['total_rollout_time_s'] = total_rollout_time

        # Normalization stats used
        norm_grp = meta.create_group('normalization')
        norm_grp.create_dataset('node_mean', data=node_mean)
        norm_grp.create_dataset('node_std', data=node_std)
        norm_grp.create_dataset('edge_mean', data=edge_mean)
        norm_grp.create_dataset('edge_std', data=edge_std)
        norm_grp.create_dataset('delta_mean', data=delta_mean)
        norm_grp.create_dataset('delta_std', data=delta_std)

        # Mesh topology (fixed across all steps)
        mesh_grp = f.create_group('mesh')
        mesh_grp.create_dataset('edge_index', data=mesh_edge)  # [2, M] unidirectional
        mesh_grp.create_dataset('ref_pos', data=ref_pos)       # [N, 3]
        if part_ids is not None:
            mesh_grp.create_dataset('part_ids', data=part_ids)

        # Predicted states: [num_steps+1, N, output_dim]
        pred_grp = f.create_group('predictions')
        pred_grp.create_dataset(
            'states', data=all_states,
            compression='gzip', compression_opts=4
        )
        pred_grp.attrs['description'] = (
            'Predicted states at each timestep. '
            'Shape: [num_steps+1, num_nodes, output_dim]. '
            'Index 0 is the initial condition. '
            'Features: [x_disp, y_disp, z_disp, stress]'
        )

        # Also store per-step deformed positions for convenience
        deformed_positions = np.zeros((num_rollout_steps + 1, num_nodes, 3), dtype=np.float32)
        for t in range(num_rollout_steps + 1):
            deformed_positions[t] = ref_pos + all_states[t, :, :3]
        pred_grp.create_dataset(
            'deformed_positions', data=deformed_positions,
            compression='gzip', compression_opts=4
        )
        pred_grp.attrs['deformed_positions_description'] = (
            'Deformed node positions at each timestep. '
            'Shape: [num_steps+1, num_nodes, 3]. '
            'Computed as: ref_pos + displacement.'
        )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved ({file_size_mb:.1f} MB)")
    print(f"\nRollout inference complete.")

    return output_path


def _compute_world_edges(deformed_pos, mesh_edge_index, radius,
                         max_num_neighbors, backend, edge_mean, edge_std):
    """
    Compute world edges (radius-based collision detection) for a single timestep.

    Args:
        deformed_pos: [N, 3] deformed node positions
        mesh_edge_index: [2, 2M] bidirectional mesh edges
        radius: World edge radius
        max_num_neighbors: Maximum neighbors per node
        backend: 'torch_cluster' or 'scipy_kdtree'
        edge_mean: [4] edge normalization mean
        edge_std: [4] edge normalization std

    Returns:
        world_edge_index: [2, E_world] world edge indices
        world_edge_attr_norm: [E_world, 4] normalized world edge features
    """
    if backend == 'torch_cluster' and HAS_TORCH_CLUSTER:
        return _world_edges_torch_cluster(
            deformed_pos, mesh_edge_index, radius,
            max_num_neighbors, edge_mean, edge_std
        )
    else:
        return _world_edges_scipy(
            deformed_pos, mesh_edge_index, radius,
            edge_mean, edge_std
        )


def _world_edges_torch_cluster(pos, mesh_edges, radius,
                                max_num_neighbors, edge_mean, edge_std):
    """GPU-accelerated world edge computation using torch_cluster."""
    pos_tensor = torch.from_numpy(pos).float().cuda()

    world_edges = radius_graph(
        x=pos_tensor, r=radius, batch=None,
        loop=False, max_num_neighbors=max_num_neighbors
    )
    world_edges_np = world_edges.cpu().numpy()

    if world_edges_np.shape[1] == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

    # Filter out existing mesh edges
    mesh_set = {(int(mesh_edges[0, i]), int(mesh_edges[1, i]))
                for i in range(mesh_edges.shape[1])}
    valid_mask = np.array([
        (world_edges_np[0, i], world_edges_np[1, i]) not in mesh_set
        for i in range(world_edges_np.shape[1])
    ])
    we = world_edges_np[:, valid_mask]

    if we.shape[1] == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

    rel = pos[we[1]] - pos[we[0]]
    dist = np.linalg.norm(rel, axis=1, keepdims=True)
    world_attr_raw = np.concatenate([rel, dist], axis=1).astype(np.float32)
    world_attr_norm = (world_attr_raw - edge_mean) / edge_std

    return we, world_attr_norm


def _world_edges_scipy(pos, mesh_edges, radius, edge_mean, edge_std):
    """CPU world edge computation using scipy KDTree."""
    tree = KDTree(pos)
    pairs = tree.query_pairs(r=radius, output_type='ndarray')

    if len(pairs) == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

    mesh_set = {(int(mesh_edges[0, i]), int(mesh_edges[1, i]))
                for i in range(mesh_edges.shape[1])}

    we = []
    for s, r in pairs:
        if (s, r) not in mesh_set:
            we.append([s, r])
        if (r, s) not in mesh_set:
            we.append([r, s])

    if not we:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

    wei = np.array(we, dtype=np.int64).T
    rel = pos[wei[1]] - pos[wei[0]]
    dist = np.linalg.norm(rel, axis=1, keepdims=True)
    world_attr_raw = np.concatenate([rel, dist], axis=1).astype(np.float32)
    world_attr_norm = (world_attr_raw - edge_mean) / edge_std

    return wei, world_attr_norm
