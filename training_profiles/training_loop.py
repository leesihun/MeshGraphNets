import time
import tqdm
import torch

def train_epoch(model, dataloader, optimizer, device, config, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0

    verbose = config.get('verbose')

    pbar = tqdm.tqdm(dataloader)
    for batch_idx, graph in enumerate(pbar):
        # Detailed memory logging for first 5 batches and every 100 batches
        log_detailed = (batch_idx < 5) or (batch_idx % 100 == 0)

        if log_detailed and verbose:
            mem_1 = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            tqdm.tqdm.write(f"\n=== Batch {batch_idx} ===")
            tqdm.tqdm.write(f"1. Start: {mem_1:.2f}GB (reserved: {mem_reserved:.2f}GB)")

        graph = graph.to(device)

        if log_detailed and verbose:
            mem_2 = torch.cuda.memory_allocated() / 1e9
            tqdm.tqdm.write(f"2. After .to(device): {mem_2:.2f}GB (+{mem_2-mem_1:.2f}GB)")

        if epoch==0 and batch_idx==0:
            print(f"Batch {batch_idx} of epoch {epoch} is being processed")
            print(f"Graph: {graph}")
            print(f"Graph.x: {graph.x}")
            print(f"Graph.edge_attr: {graph.edge_attr}")
            print(f"Graph.edge_index: {graph.edge_index}")
            print(f"Graph.y: {graph.y}")

        predicted_acc, target_acc = model(graph)

        import time
        time.sleep(10000)

        if log_detailed and verbose:
            mem_3 = torch.cuda.memory_allocated() / 1e9
            tqdm.tqdm.write(f"3. After forward: {mem_3:.2f}GB (+{mem_3-mem_2:.2f}GB)")

        errors = ((predicted_acc - target_acc) ** 2)
        loss = torch.mean(errors) # MSE Loss

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to stabilize training with deep message passing
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if log_detailed and verbose:
            mem_4 = torch.cuda.memory_allocated() / 1e9
            tqdm.tqdm.write(f"4. After backward: {mem_4:.2f}GB (+{mem_4-mem_3:.2f}GB)")

        optimizer.step()

        if log_detailed and verbose:
            mem_5 = torch.cuda.memory_allocated() / 1e9
            mem_reserved_end = torch.cuda.memory_reserved() / 1e9
            tqdm.tqdm.write(f"5. After optimizer.step: {mem_5:.2f}GB (+{mem_5-mem_4:.2f}GB)")
            tqdm.tqdm.write(f"6. Reserved memory: {mem_reserved_end:.2f}GB (leaked: {mem_reserved_end-mem_reserved:.2f}GB)")
            tqdm.tqdm.write(f"7. Peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

            # Try to force garbage collection
            del graph
            torch.cuda.empty_cache()
            mem_after_gc = torch.cuda.memory_allocated() / 1e9
            mem_reserved_after_gc = torch.cuda.memory_reserved() / 1e9
            tqdm.tqdm.write(f"8. After torch.cuda.empty_cache(): {mem_after_gc:.2f}GB (reserved: {mem_reserved_after_gc:.2f}GB)\n")

            if batch_idx >= 5 and verbose:
                torch.cuda.reset_peak_memory_stats()

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
