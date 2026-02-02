import tqdm
import torch
from general_modules.mesh_utils_fast import save_inference_results_fast, ParallelVisualizer

def train_epoch(model, dataloader, optimizer, device, config, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0

    verbose = config.get('verbose')

    pbar = tqdm.tqdm(dataloader)
    for batch_idx, graph in enumerate(pbar):

        graph = graph.to(device)

        predicted_acc, target_acc = model(graph)

        errors = ((predicted_acc - target_acc) ** 2)
        loss = torch.mean(errors) # MSE Loss

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to stabilize training with deep message passing
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        # Update progress bar with current memory
        mem_gb = torch.cuda.memory_allocated() / 1e9
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        pbar.set_postfix({
            'loss': f'{loss.item():.2e}',
            'mem': f'{mem_gb:.1f}GB',
            'peak': f'{peak_gb:.1f}GB'
        })

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches

def validate_epoch(model, dataloader, device, config):
    model.eval()

    verbose = config.get('verbose')

    with torch.no_grad():
        total_loss = 0.0
        num_batches = 0

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

            if batch_idx < 3 and verbose:
                mem_after = torch.cuda.memory_allocated() / 1e9
                tqdm.tqdm.write(f"After: {mem_after:.2f}GB (+{mem_after-mem_before:.2f}GB)")
                tqdm.tqdm.write(f"Peak: {torch.cuda.max_memory_allocated()/1e9:.2f}GB\n")

            # Update progress bar
            mem_gb = torch.cuda.memory_allocated() / 1e9
            pbar.set_postfix({'loss': f'{loss.item():.2e}', 'mem': f'{mem_gb:.1f}GB'})

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def test_model(model, dataloader, device, config, epoch, dataset=None):
    model.eval()

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

        # Collect plot data for parallel processing
        plot_data_queue = []

        pbar = tqdm.tqdm(dataloader)
        for batch_idx, graph in enumerate(pbar):

            graph = graph.to(device)
            predicted, target = model(graph)
            errors = ((predicted - target) ** 2)
            loss = torch.mean(errors)  # MSE Loss

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

    return total_loss / num_batches if num_batches > 0 else 0.0