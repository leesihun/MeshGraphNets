import os
import tqdm
import torch
import numpy as np
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from general_modules.mesh_utils_fast import (
    edges_to_triangles_gpu,
    edges_to_triangles_optimized,
    render_plot_data,
    save_inference_results_fast,
)


def build_ema_model(model, config):
    """Create an EMA shadow model if use_ema is enabled."""
    if not config.get('use_ema', False):
        return None
    decay = float(config.get('ema_decay', 0.999))
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay=decay))
    for p in ema_model.parameters():
        p.requires_grad_(False)
    return ema_model


def save_debug_batch(epoch, batch_idx, graph, predicted, target, log_dir):
    """Save actual input/output values to debug file for inspection."""
    try:
        debug_file = os.path.join(log_dir, f'debug_epoch{epoch:03d}_batch{batch_idx:03d}.npz')
        x_np = graph.x.cpu().numpy()
        y_np = target.cpu().numpy()
        pred_np = predicted.cpu().numpy()

        np.savez(
            debug_file,
            x=x_np,
            y=y_np,
            pred=pred_np,
            x_mean=x_np.mean(axis=0),
            x_std=x_np.std(axis=0),
            y_mean=y_np.mean(axis=0),
            y_std=y_np.std(axis=0),
            pred_mean=pred_np.mean(axis=0),
            pred_std=pred_np.std(axis=0),
        )
        tqdm.tqdm.write(f"  Saved debug data to {debug_file}")
    except Exception as e:
        tqdm.tqdm.write(f"  Warning: Could not save debug data: {e}")


def _build_loss_weights(config, device):
    """Build per-feature loss weights normalized to sum to 1."""
    loss_weights = config.get('feature_loss_weights', None)
    if loss_weights is not None:
        if not isinstance(loss_weights, list):
            loss_weights = [loss_weights]
        loss_weights = torch.tensor(loss_weights, dtype=torch.float32, device=device)
        loss_weights = loss_weights / loss_weights.sum()
    return loss_weights


def _per_node_loss(errors, loss_weights):
    """Reduce feature errors to one scalar per node."""
    if loss_weights is not None:
        return torch.sum(errors * loss_weights, dim=-1)
    return torch.mean(errors, dim=-1)


def _loss_from_errors(errors, loss_weights):
    """Return mean loss used for backprop plus exact aggregation stats.

    The batch sum is returned as a detached 0-dim GPU tensor, not a Python
    float: .item() here would force a CPU<->GPU sync on every batch and
    serialize the CUDA pipeline. Callers accumulate on-device and convert
    once per epoch (or every N batches for progress display).
    """
    per_node = _per_node_loss(errors, loss_weights)
    loss_sum = per_node.sum()
    loss_count = per_node.numel()
    return loss_sum / loss_count, loss_sum.detach(), loss_count


def _move_graph_to_device(graph, device, config):
    non_blocking = bool(config.get('_pin_memory', False)) and getattr(device, 'type', None) == 'cuda'
    return graph.to(device, non_blocking=non_blocking)


def _accum_window_size(batch_idx, total_batches, actual_accum):
    """Return the number of batches in the current accumulation window."""
    window_start = (batch_idx // actual_accum) * actual_accum
    window_end = min(window_start + actual_accum, total_batches)
    return window_end - window_start


# Batches skipped before profiling starts (allocator/cudnn warmup): 2 wait + 2 warmup.
_PROFILE_SKIP_BATCHES = 4


def _start_profiler(config, epoch):
    """Start a torch profiler for the first `profile_batches` batches of epoch 0.

    Set `profile_batches N` in the config to enable. Returns None when disabled.
    """
    profile_batches = int(config.get('profile_batches', 0))
    if profile_batches <= 0 or epoch != 0:
        return None

    from torch.profiler import ProfilerActivity, profile, schedule
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    profiler = profile(
        activities=activities,
        schedule=schedule(wait=2, warmup=2, active=profile_batches, repeat=1),
    )
    profiler.start()
    tqdm.tqdm.write(
        f"Profiling batches {_PROFILE_SKIP_BATCHES}..{_PROFILE_SKIP_BATCHES + profile_batches - 1} "
        f"of epoch 0 (profile_batches={profile_batches})"
    )
    return profiler


def _finish_profiler(profiler, config):
    """Stop the profiler, print a kernel-time summary, and export a chrome trace."""
    profiler.stop()
    sort_key = 'self_cuda_time_total' if torch.cuda.is_available() else 'self_cpu_time_total'
    print(profiler.key_averages().table(sort_by=sort_key, row_limit=30))
    trace_path = os.path.join(config.get('log_dir', '.'), 'train_profile_trace.json')
    try:
        profiler.export_chrome_trace(trace_path)
        print(f"Profiler trace written to {trace_path} (open in chrome://tracing or https://ui.perfetto.dev)")
    except Exception as e:
        print(f"Warning: could not export profiler trace: {e}")


def log_training_config(config):
    """Log loss weights and architecture switches to stdout."""
    loss_weights_cfg = config.get('feature_loss_weights', None)
    if loss_weights_cfg is not None:
        if not isinstance(loss_weights_cfg, list):
            loss_weights_cfg = [loss_weights_cfg]
        w = torch.tensor(loss_weights_cfg, dtype=torch.float32)
        w_normalized = (w / w.sum()).tolist()
        print(f"Per-feature loss weights (raw):         {loss_weights_cfg}")
        print(f"Per-feature loss weights (normalized):  {[f'{v:.4f}' for v in w_normalized]}")
    else:
        print("Per-feature loss weights: equal (default)")

    if config.get('use_multiscale', False):
        levels = int(config.get('multiscale_levels', 1))
        mp = config.get('mp_per_level', None)
        if mp is None:
            mp = [
                int(config.get('fine_mp_pre', 5)),
                int(config.get('coarse_mp_num', 5)),
                int(config.get('fine_mp_post', 5)),
            ]
        if not isinstance(mp, list):
            mp = [int(mp)]
        print(f"Multi-Scale: ENABLED (V-cycle, {levels} coarsening levels, {sum(int(x) for x in mp)} total GnBlocks)")
        for i in range(levels):
            print(f"  Level {i} pre:  {mp[i]} blocks")
        print(f"  Coarsest:    {mp[levels]} blocks")
        for i in range(levels - 1, -1, -1):
            print(f"  Level {i} post: {mp[2 * levels - i]} blocks")
        print("  [message_passing_num is IGNORED when use_multiscale=True]")
    else:
        print(f"Multi-Scale: disabled (flat GNN, message_passing_num={config.get('message_passing_num')})")


def train_epoch(model, dataloader, optimizer, device, config, epoch, scheduler=None, ema_model=None, *, _iter=None):
    model.train()
    # On-device accumulator: adding batch sums tensor-to-tensor keeps the loop
    # free of CPU<->GPU syncs; converted to a Python float once at epoch end.
    total_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    total_loss_count = 0
    total_grad_norm = 0.0
    num_steps = 0

    verbose = config.get('verbose', False)
    monitor_gradients = config.get('monitor_gradients', False)
    loss_weights = _build_loss_weights(config, device)
    use_amp = config.get('use_amp', True)
    use_compile = config.get('use_compile', False)
    amp_dtype = torch.bfloat16

    grad_accum_steps = config.get('grad_accum_steps', 1)
    total_batches = len(dataloader)
    actual_accum = total_batches if grad_accum_steps == 0 else grad_accum_steps

    optimizer.zero_grad(set_to_none=True)
    grad_norm = torch.tensor(0.0)

    profiler = _start_profiler(config, epoch)
    profile_end_batch = _PROFILE_SKIP_BATCHES + int(config.get('profile_batches', 0))

    iterable = _iter if _iter is not None else dataloader
    pbar = tqdm.tqdm(iterable, total=total_batches)
    for batch_idx, graph in enumerate(pbar):
        debug_internal = (not use_compile) and (batch_idx == 0 and (epoch < 5 or epoch % 10 == 0))
        graph = _move_graph_to_device(graph, device, config)

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            predicted_acc, target_acc = model(graph, debug=debug_internal)

            if batch_idx == 0 and verbose:
                tqdm.tqdm.write(f"\n=== DEBUG Epoch {epoch} Batch 0 ===")
                tqdm.tqdm.write(f"  Pred:   mean={predicted_acc.mean().item():.6f}, std={predicted_acc.std().item():.6f}, min={predicted_acc.min().item():.4f}, max={predicted_acc.max().item():.4f}")
                tqdm.tqdm.write(f"  Target: mean={target_acc.mean().item():.6f}, std={target_acc.std().item():.6f}, min={target_acc.min().item():.4f}, max={target_acc.max().item():.4f}")
                if predicted_acc.std().item() < 0.01:
                    tqdm.tqdm.write("  WARNING: Pred std < 0.01 - model outputting near-constant values.")
                if epoch >= 5:
                    save_debug_batch(epoch, batch_idx, graph, predicted_acc, target_acc, config.get('log_dir', '.'))

            errors = torch.nn.functional.mse_loss(predicted_acc, target_acc, reduction='none')
            loss, batch_loss_sum, batch_loss_count = _loss_from_errors(errors, loss_weights)
            scaled_loss = loss / _accum_window_size(batch_idx, total_batches, actual_accum)

        scaled_loss.backward()

        if batch_idx == 0 and epoch % 10 == 0 and verbose:
            per_feature_loss_mean = torch.mean(errors, dim=0)
            per_feature_loss_max = torch.max(errors, dim=0)[0]
            per_feature_loss_min = torch.min(errors, dim=0)[0]
            per_feature_loss_std = torch.std(errors, dim=0)
            feature_names = ['x_disp', 'y_disp', 'z_disp', 'stress']
            tqdm.tqdm.write(f"\n=== Per-Feature MSE Loss (Epoch {epoch}, Batch {batch_idx}) ===")
            for feat_idx, feat_name in enumerate(feature_names[:len(per_feature_loss_mean)]):
                tqdm.tqdm.write(
                    f"  {feat_name}: mean={per_feature_loss_mean[feat_idx].item():.2e}, "
                    f"max={per_feature_loss_max[feat_idx].item():.2e}, "
                    f"min={per_feature_loss_min[feat_idx].item():.2e}, "
                    f"std={per_feature_loss_std[feat_idx].item():.2e}"
                )

        total_loss_sum += batch_loss_sum.double()
        total_loss_count += batch_loss_count

        is_last_batch = batch_idx == total_batches - 1
        if (batch_idx + 1) % actual_accum == 0 or is_last_batch:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            if monitor_gradients:
                total_grad_norm += grad_norm.item()
                num_steps += 1
                if verbose:
                    layer_grad_stats = []
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_mean = param.grad.abs().mean().item()
                            grad_max = param.grad.abs().max().item()
                            if grad_mean > 1e-10:
                                layer_grad_stats.append(f"{name}: mean={grad_mean:.2e}, max={grad_max:.2e}")
                    if layer_grad_stats:
                        tqdm.tqdm.write(f"\n=== Gradient Stats (Step after batch {batch_idx}) ===")
                        tqdm.tqdm.write(f"Total grad norm: {grad_norm.item():.2e}")
                        for stat in layer_grad_stats[:5]:
                            tqdm.tqdm.write(f"  {stat}")

            optimizer.step()
            if ema_model is not None:
                ema_model.update_parameters(model)
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if batch_idx % 10 == 0:
            # The only deliberate sync in the loop: one .item() per 10 batches
            # to keep the progress bar live without stalling the pipeline.
            mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            loss_val = batch_loss_sum.item() / batch_loss_count
            postfix = {'loss': f'{loss_val:.2e}', 'mem': f'{mem_gb:.1f}GB'}
            if monitor_gradients:
                postfix['grad'] = f'{grad_norm.item():.2e}'
            pbar.set_postfix(postfix)

        if profiler is not None:
            profiler.step()
            if batch_idx + 1 >= profile_end_batch:
                _finish_profiler(profiler, config)
                profiler = None

    if profiler is not None:  # dataloader shorter than the profiling window
        _finish_profiler(profiler, config)
        profiler = None

    avg_grad_norm = total_grad_norm / num_steps if num_steps > 0 else 0.0
    if monitor_gradients and num_steps > 0:
        tqdm.tqdm.write(f"Epoch {epoch} avg gradient norm: {avg_grad_norm:.2e} ({num_steps} optimizer steps)")
        if avg_grad_norm < 1e-6:
            tqdm.tqdm.write("  WARNING: Very small gradients detected (< 1e-6).")
        elif avg_grad_norm > 1e2:
            tqdm.tqdm.write("  WARNING: Very large gradients detected (> 100).")

    total_loss_sum = total_loss_sum.item()
    mean = total_loss_sum / total_loss_count
    return {'mean': mean, 'total_mean': mean, 'sum': total_loss_sum, 'count': total_loss_count}


def _eval_forward_errors(model, graph, use_amp, amp_dtype):
    with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
        predicted, target = model(graph, add_noise=False)
        errors = torch.nn.functional.mse_loss(predicted, target, reduction='none')
    return errors


def _evaluate_epoch(model, dataloader, device, config, epoch=0, *, progress_name='Validation'):
    model.eval()

    verbose = config.get('verbose', False)
    loss_weights = _build_loss_weights(config, device)
    use_amp = config.get('use_amp', True)
    amp_dtype = torch.bfloat16

    with torch.no_grad():
        total_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        total_loss_count = 0
        accumulated_per_feature_loss = None
        accumulated_per_feature_count = 0

        pbar = tqdm.tqdm(dataloader, desc=progress_name)
        for batch_idx, graph in enumerate(pbar):
            if batch_idx < 3 and verbose:
                mem_before = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                tqdm.tqdm.write(f"\n=== {progress_name} Batch {batch_idx} ===")
                tqdm.tqdm.write(f"Before: {mem_before:.2f}GB")

            graph = _move_graph_to_device(graph, device, config)
            errors = _eval_forward_errors(model, graph, use_amp, amp_dtype)
            loss, batch_loss_sum, batch_loss_count = _loss_from_errors(errors, loss_weights)

            per_feature_loss = torch.sum(errors, dim=0)
            accumulated_per_feature_loss = (
                per_feature_loss if accumulated_per_feature_loss is None
                else accumulated_per_feature_loss + per_feature_loss
            )
            accumulated_per_feature_count += errors.shape[0]

            if batch_idx < 3 and verbose:
                mem_after = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                tqdm.tqdm.write(f"After: {mem_after:.2f}GB (+{mem_after-mem_before:.2f}GB)")
                tqdm.tqdm.write(f"Peak: {peak:.2f}GB\n")

            if batch_idx % 10 == 0:
                mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                loss_val = batch_loss_sum.item() / batch_loss_count
                pbar.set_postfix({'loss': f'{loss_val:.2e}', 'mem': f'{mem_gb:.1f}GB'})

            total_loss_sum += batch_loss_sum.double()
            total_loss_count += batch_loss_count

        if verbose and accumulated_per_feature_loss is not None and accumulated_per_feature_count > 0:
            avg_per_feature_loss = accumulated_per_feature_loss / accumulated_per_feature_count
            feature_names = ['x_disp', 'y_disp', 'z_disp', 'stress']
            tqdm.tqdm.write(f"\n=== Per-Feature {progress_name} Loss (Epoch {epoch}) ===")
            for feat_idx, feat_name in enumerate(feature_names[:len(avg_per_feature_loss)]):
                tqdm.tqdm.write(f"  {feat_name}: {avg_per_feature_loss[feat_idx].item():.2e}")
            tqdm.tqdm.write("")

    total_loss_sum = total_loss_sum.item()
    mean = total_loss_sum / total_loss_count
    return {'mean': mean, 'total_mean': mean, 'sum': total_loss_sum, 'count': total_loss_count}


def validate_epoch(model, dataloader, device, config, epoch=0):
    return _evaluate_epoch(model, dataloader, device, config, epoch, progress_name='Validation')


def test_model(model, dataloader, device, config, epoch, dataset=None, output_prefix='test'):
    model.eval()

    verbose = config.get('verbose', False)
    loss_weights = _build_loss_weights(config, device)
    use_gpu = device.type == 'cuda' if hasattr(device, 'type') else (device != 'cpu')
    mesh_device = device if use_gpu else 'cpu'
    faces_cache = {}

    use_amp = config.get('use_amp', True)
    amp_dtype = torch.bfloat16

    total_test = len(dataloader)
    max_test_batches = int(config.get('test_max_batches', 200))
    effective_total = min(max_test_batches, total_test)
    if effective_total < total_test:
        print(f"  Test: evaluating {effective_total}/{total_test} samples (set test_max_batches in config to change)")

    delta_mean = None
    delta_std = None
    if dataset is not None:
        delta_mean = dataset.delta_mean
        delta_std = dataset.delta_std
        if delta_mean is not None and delta_std is not None:
            print(f"Using denormalization: delta_mean={delta_mean}, delta_std={delta_std}")

    with torch.no_grad():
        total_loss_sum = 0.0
        total_loss_count = 0
        accumulated_per_feature_loss = None
        accumulated_per_feature_count = 0
        plot_data_queue = []

        pbar = tqdm.tqdm(dataloader, total=effective_total)
        for batch_idx, graph in enumerate(pbar):
            if batch_idx >= max_test_batches:
                break

            graph = _move_graph_to_device(graph, device, config)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                predicted, target = model(graph)
                errors = torch.nn.functional.mse_loss(predicted, target, reduction='none')
                loss, batch_loss_sum, batch_loss_count = _loss_from_errors(errors, loss_weights)

            per_feature_loss = torch.sum(errors, dim=0)
            accumulated_per_feature_loss = (
                per_feature_loss if accumulated_per_feature_loss is None
                else accumulated_per_feature_loss + per_feature_loss
            )
            accumulated_per_feature_count += errors.shape[0]

            mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            pbar.set_postfix({'loss': f'{loss.item():.2e}', 'mem': f'{mem_gb:.1f}GB'})

            # Test runs rarely and already syncs for visualization output;
            # a per-batch .item() here is harmless.
            total_loss_sum += batch_loss_sum.item()
            total_loss_count += batch_loss_count

            if batch_idx in config.get('test_batch_idx', [0, 1, 2, 3]):
                gpu_ids = str(config.get('gpu_ids'))
                sample_id = None
                time_idx = None
                if hasattr(graph, 'sample_id') and graph.sample_id is not None:
                    sid = graph.sample_id
                    if hasattr(sid, 'cpu'):
                        sid = sid.cpu()
                    if hasattr(sid, 'item'):
                        sample_id = sid.item()
                    elif hasattr(sid, '__getitem__') and len(sid) > 0:
                        sample_id = int(sid[0])
                    else:
                        sample_id = int(sid)

                if hasattr(graph, 'time_idx') and graph.time_idx is not None:
                    tid = graph.time_idx
                    if hasattr(tid, 'cpu'):
                        tid = tid.cpu()
                    if hasattr(tid, 'item'):
                        time_idx = tid.item()
                    elif hasattr(tid, '__getitem__') and len(tid) > 0:
                        time_idx = int(tid[0])
                    else:
                        time_idx = int(tid)

                if sample_id is not None and time_idx is not None:
                    filename = f'sample{sample_id}_t{time_idx}'
                elif sample_id is not None:
                    filename = f'sample{sample_id}'
                else:
                    filename = f'batch{batch_idx}'

                output_path = f'outputs/{output_prefix}/{gpu_ids}/{str(epoch)}/{filename}.h5'
                predicted_np = predicted.float().cpu().numpy() if hasattr(predicted, 'cpu') else predicted
                target_np = target.float().cpu().numpy() if hasattr(target, 'cpu') else target

                if delta_mean is not None and delta_std is not None:
                    predicted_denorm = predicted_np * delta_std + delta_mean
                    target_denorm = target_np * delta_std + delta_mean
                else:
                    predicted_denorm = predicted_np
                    target_denorm = target_np

                cached_faces = faces_cache.get(sample_id)
                if cached_faces is None and sample_id is not None:
                    if use_gpu and torch.cuda.is_available():
                        edge_index_gpu = graph.edge_index.to(mesh_device)
                        cached_faces = edges_to_triangles_gpu(edge_index_gpu, device=mesh_device)
                    else:
                        ei_np = (
                            graph.edge_index.cpu().numpy()
                            if hasattr(graph.edge_index, 'cpu')
                            else np.array(graph.edge_index)
                        )
                        cached_faces = edges_to_triangles_optimized(ei_np)
                    faces_cache[sample_id] = cached_faces

                plot_data = save_inference_results_fast(
                    output_path, graph,
                    predicted_norm=predicted_np, target_norm=target_np,
                    predicted_denorm=predicted_denorm, target_denorm=target_denorm,
                    skip_visualization=not config.get('display_testset', True),
                    device=mesh_device,
                    feature_idx=config.get('plot_feature_idx', -1),
                    precomputed_faces=cached_faces,
                )
                if plot_data:
                    plot_data_queue.append(plot_data)

        if plot_data_queue:
            print(f"\nRendering {len(plot_data_queue)} visualizations...")
            failed = 0
            for pd in plot_data_queue:
                if not render_plot_data(pd):
                    failed += 1
            if failed:
                print(f"Visualization done with {failed}/{len(plot_data_queue)} failures.")
            else:
                print("All visualizations complete!")

        if verbose and accumulated_per_feature_loss is not None and accumulated_per_feature_count > 0:
            avg_per_feature_loss = accumulated_per_feature_loss / accumulated_per_feature_count
            feature_names = ['x_disp', 'y_disp', 'z_disp', 'stress']
            print(f"\n=== Per-Feature Test Loss (Epoch {epoch}) ===")
            for feat_idx, feat_name in enumerate(feature_names[:len(avg_per_feature_loss)]):
                print(f"  {feat_name}: {avg_per_feature_loss[feat_idx].item():.2e}")
            print("")

    return total_loss_sum / total_loss_count if total_loss_count > 0 else 0.0
