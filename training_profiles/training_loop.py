import tqdm
import torch
from general_modules.mesh_utils_fast import save_inference_results_fast, ParallelVisualizer

def train_epoch(model, dataloader, optimizer, device, config, epoch):
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    num_batches = 0

    verbose = config.get('verbose')
    monitor_gradients = config.get('monitor_gradients', True)

    pbar = tqdm.tqdm(dataloader)
    for batch_idx, graph in enumerate(pbar):

        graph = graph.to(device)

        # DEBUG: Check internal statistics for first batch of epoch 0
        debug_internal = (batch_idx == 0 and epoch == 0)
        predicted_acc, target_acc = model(graph, debug=debug_internal)

        # DEBUG: Check model output statistics (first batch of first 5 epochs)
        if batch_idx == 0 and epoch < 5:
            tqdm.tqdm.write(f"\n=== DEBUG Epoch {epoch} Batch 0 ===")
            tqdm.tqdm.write(f"  Pred:   mean={predicted_acc.mean().item():.6f}, std={predicted_acc.std().item():.6f}, min={predicted_acc.min().item():.4f}, max={predicted_acc.max().item():.4f}")
            tqdm.tqdm.write(f"  Target: mean={target_acc.mean().item():.6f}, std={target_acc.std().item():.6f}, min={target_acc.min().item():.4f}, max={target_acc.max().item():.4f}")
            if predicted_acc.std().item() < 0.01:
                tqdm.tqdm.write(f"  *** WARNING: Pred std < 0.01 - model outputting near-constant values! ***")

        errors = ((predicted_acc - target_acc) ** 2)
        loss = torch.mean(errors) # MSE Loss

        optimizer.zero_grad()
        loss.backward()

        # Per-feature loss breakdown for diagnostics
        if batch_idx == 0 and epoch % 10 == 0 and verbose:
            per_feature_loss_mean = torch.mean(errors, dim=0)  # Average across nodes
            per_feature_loss_max = torch.max(errors, dim=0)[0]  # Max across nodes
            per_feature_loss_min = torch.min(errors, dim=0)[0]  # Min across nodes
            per_feature_loss_std = torch.std(errors, dim=0)  # Std across nodes
            
            feature_names = ['x_disp', 'y_disp', 'z_disp', 'stress']
            tqdm.tqdm.write(f"\n=== Per-Feature MSE Loss (Epoch {epoch}, Batch {batch_idx}) ===")
            for feat_idx, feat_name in enumerate(feature_names[:len(per_feature_loss_mean)]):
                mean_val = per_feature_loss_mean[feat_idx].item()
                max_val = per_feature_loss_max[feat_idx].item()
                min_val = per_feature_loss_min[feat_idx].item()
                std_val = per_feature_loss_std[feat_idx].item()
                tqdm.tqdm.write(f"  {feat_name}: mean={mean_val:.2e}, max={max_val:.2e}, min={min_val:.2e}, std={std_val:.2e}")

        # Compute gradient norm BEFORE clipping for diagnostics
        if monitor_gradients:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            total_grad_norm += grad_norm.item()

            # Detailed gradient stats for first batch and periodically
            if (batch_idx == 0 or batch_idx % 50 == 0) and verbose:
                layer_grad_stats = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_mean = param.grad.abs().mean().item()
                        grad_max = param.grad.abs().max().item()
                        if grad_mean > 1e-10:  # Only show non-zero gradients
                            layer_grad_stats.append(f"{name}: mean={grad_mean:.2e}, max={grad_max:.2e}")

                if layer_grad_stats:
                    tqdm.tqdm.write(f"\n=== Gradient Stats (Batch {batch_idx}) ===")
                    tqdm.tqdm.write(f"Total grad norm: {grad_norm.item():.2e}")
                    tqdm.tqdm.write("\nPer-layer gradients (top 5):")
                    for stat in layer_grad_stats[:5]:
                        tqdm.tqdm.write(f"  {stat}")
                    tqdm.tqdm.write("")

        # Gradient clipping to stabilize training with deep message passing
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        # Update progress bar with current memory and gradient norm
        mem_gb = torch.cuda.memory_allocated() / 1e9
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        postfix = {
            'loss': f'{loss.item():.2e}',
            'mem': f'{mem_gb:.1f}GB',
            'peak': f'{peak_gb:.1f}GB'
        }
        if monitor_gradients:
            postfix['grad'] = f'{grad_norm.item():.2e}'
        pbar.set_postfix(postfix)

        total_loss += loss.item()
        num_batches += 1

    avg_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0.0

    # Print gradient summary for the epoch
    if monitor_gradients and num_batches > 0:
        tqdm.tqdm.write(f"Epoch {epoch} avg gradient norm: {avg_grad_norm:.2e}")
        if avg_grad_norm < 1e-6:
            tqdm.tqdm.write("  ⚠️  WARNING: Very small gradients detected (< 1e-6) - possible vanishing gradient problem!")
        elif avg_grad_norm > 1e2:
            tqdm.tqdm.write("  ⚠️  WARNING: Very large gradients detected (> 100) - possible exploding gradient problem!")

    return total_loss / num_batches

def validate_epoch(model, dataloader, device, config, epoch=0):
    model.eval()

    verbose = config.get('verbose')
    output_var = config.get('output_var', 4)

    with torch.no_grad():
        total_loss = 0.0
        num_batches = 0
        
        # Accumulate per-feature losses across all batches
        accumulated_per_feature_loss = None

        pbar = tqdm.tqdm(dataloader)
        for batch_idx, graph in enumerate(pbar):
            # Memory tracking for first 3 validation batches
            if batch_idx < 3 and verbose:
                mem_before = torch.cuda.memory_allocated() / 1e9
                tqdm.tqdm.write(f"\n=== Validation Batch {batch_idx} ===")
                tqdm.tqdm.write(f"Before: {mem_before:.2f}GB")

            graph = graph.to(device)
            predicted, target = model(graph)
            errors = ((predicted - target) ** 2)
            loss = torch.mean(errors)  # MSE Loss
            
            # Accumulate per-feature losses
            per_feature_loss = torch.mean(errors, dim=0)  # Average across nodes
            if accumulated_per_feature_loss is None:
                accumulated_per_feature_loss = per_feature_loss
            else:
                accumulated_per_feature_loss += per_feature_loss

            if batch_idx < 3 and verbose:
                mem_after = torch.cuda.memory_allocated() / 1e9
                tqdm.tqdm.write(f"After: {mem_after:.2f}GB (+{mem_after-mem_before:.2f}GB)")
                tqdm.tqdm.write(f"Peak: {torch.cuda.max_memory_allocated()/1e9:.2f}GB\n")

            # Update progress bar
            mem_gb = torch.cuda.memory_allocated() / 1e9
            pbar.set_postfix({'loss': f'{loss.item():.2e}', 'mem': f'{mem_gb:.1f}GB'})

            total_loss += loss.item()
            num_batches += 1
        
        # Print per-feature validation loss breakdown
        if verbose and accumulated_per_feature_loss is not None and num_batches > 0:
            avg_per_feature_loss = accumulated_per_feature_loss / num_batches
            feature_names = ['x_disp', 'y_disp', 'z_disp', 'stress']
            tqdm.tqdm.write(f"\n=== Per-Feature Validation MSE Loss (Epoch {epoch}) ===")
            for feat_idx, feat_name in enumerate(feature_names[:len(avg_per_feature_loss)]):
                tqdm.tqdm.write(f"  {feat_name}: {avg_per_feature_loss[feat_idx].item():.2e}")
            tqdm.tqdm.write("")

    return total_loss / num_batches


def test_model(model, dataloader, device, config, epoch, dataset=None):
    model.eval()

    verbose = config.get('verbose')
    output_var = config.get('output_var', 4)

    # Use GPU for triangle reconstruction if available
    use_gpu = device.type == 'cuda' if hasattr(device, 'type') else (device != 'cpu')
    mesh_device = device if use_gpu else 'cpu'

    # Setup parallel visualization (4 workers for matplotlib rendering)
    num_viz_workers = config.get('num_visualization_workers', 4)

    # Get denormalization parameters from dataset
    delta_mean = None
    delta_std = None
    if dataset is not None:
        delta_mean = dataset.delta_mean
        delta_std = dataset.delta_std
        if delta_mean is not None and delta_std is not None:
            print(f"Using denormalization: delta_mean={delta_mean}, delta_std={delta_std}")

    with torch.no_grad():
        total_loss = 0.0
        num_batches = 0
        
        # Accumulate per-feature losses across all test batches
        accumulated_per_feature_loss = None

        # Collect plot data for parallel processing
        plot_data_queue = []

        pbar = tqdm.tqdm(dataloader)
        for batch_idx, graph in enumerate(pbar):

            graph = graph.to(device)
            predicted, target = model(graph)
            errors = ((predicted - target) ** 2)
            loss = torch.mean(errors)  # MSE Loss
            
            # Accumulate per-feature losses
            per_feature_loss = torch.mean(errors, dim=0)  # Average across nodes
            if accumulated_per_feature_loss is None:
                accumulated_per_feature_loss = per_feature_loss
            else:
                accumulated_per_feature_loss += per_feature_loss

            # Update progress bar
            mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            pbar.set_postfix({'loss': f'{loss.item():.2e}', 'mem': f'{mem_gb:.1f}GB'})

            total_loss += loss.item()
            num_batches += 1

            # Save results with GPU-accelerated mesh reconstruction
            if batch_idx in config.get('test_batch_idx',[0]):
                gpu_ids = str(config.get('gpu_ids'))

                # Build filename with sample_id and time_idx for clarity
                # Extract sample_id and time_idx from graph (handle tensor or scalar)
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

                # Build descriptive filename
                if sample_id is not None and time_idx is not None:
                    filename = f'sample{sample_id}_t{time_idx}'
                elif sample_id is not None:
                    filename = f'sample{sample_id}'
                else:
                    filename = f'batch{batch_idx}'

                output_path = f'outputs/test/{gpu_ids}/{str(epoch)}/{filename}.h5'
                
                # Convert to numpy
                predicted_np = predicted.cpu().numpy() if hasattr(predicted, 'cpu') else predicted
                target_np = target.cpu().numpy() if hasattr(target, 'cpu') else target

                # DENORMALIZE: Convert normalized deltas to actual physical deltas
                if delta_mean is not None and delta_std is not None:
                    import numpy as np
                    predicted_denorm = predicted_np * delta_std + delta_mean
                    target_denorm = target_np * delta_std + delta_mean
                else:
                    # Fallback: use normalized values
                    predicted_denorm = predicted_np
                    target_denorm = target_np

                # Use fast GPU-accelerated version, collect plot data
                # Pass both normalized and denormalized values for visualization
                display_testset = config.get('display_testset', True)
                plot_feature_idx = config.get('plot_feature_idx', -1)
                plot_data = save_inference_results_fast(
                    output_path, graph,
                    predicted_norm=predicted_np, target_norm=target_np,
                    predicted_denorm=predicted_denorm, target_denorm=target_denorm,
                    skip_visualization=not display_testset,
                    device=mesh_device,
                    feature_idx=plot_feature_idx
                )

                if plot_data:
                    plot_data_queue.append(plot_data)

        # Now do all visualizations in parallel (after inference loop completes)
        if len(plot_data_queue) > 0:
            print(f"\nGenerating {len(plot_data_queue)} visualizations in parallel with {num_viz_workers} workers...")

            with ParallelVisualizer(num_workers=num_viz_workers) as visualizer:
                for plot_data in plot_data_queue:
                    visualizer.submit(plot_data)

            print("All visualizations complete!")
        
        # Print per-feature test loss breakdown
        if verbose and accumulated_per_feature_loss is not None and num_batches > 0:
            avg_per_feature_loss = accumulated_per_feature_loss / num_batches
            feature_names = ['x_disp', 'y_disp', 'z_disp', 'stress']
            print(f"\n=== Per-Feature Test MSE Loss (Epoch {epoch}) ===")
            for feat_idx, feat_name in enumerate(feature_names[:len(avg_per_feature_loss)]):
                print(f"  {feat_name}: {avg_per_feature_loss[feat_idx].item():.2e}")
            print("")

    return total_loss / num_batches if num_batches > 0 else 0.0