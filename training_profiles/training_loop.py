import tqdm
import torch

def train_epoch(model, dataloader, optimizer, device, config):
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm.tqdm(dataloader)
    for batch_idx, graph in enumerate(pbar):
        # Detailed memory logging for first 5 batches
        if batch_idx < 5:
            mem_1 = torch.cuda.memory_allocated() / 1e9
            tqdm.tqdm.write(f"\n=== Batch {batch_idx} ===")
            tqdm.tqdm.write(f"1. Start: {mem_1:.2f}GB")

        graph = graph.to(device)

        if batch_idx < 5:
            mem_2 = torch.cuda.memory_allocated() / 1e9
            tqdm.tqdm.write(f"2. After .to(device): {mem_2:.2f}GB (+{mem_2-mem_1:.2f}GB)")

        predicted_acc, target_acc = model(graph)

        if batch_idx < 5:
            mem_3 = torch.cuda.memory_allocated() / 1e9
            tqdm.tqdm.write(f"3. After forward: {mem_3:.2f}GB (+{mem_3-mem_2:.2f}GB)")

        errors = ((predicted_acc - target_acc) ** 2)
        loss = torch.mean(errors) # MSE Loss

        optimizer.zero_grad()
        loss.backward()

        if batch_idx < 5:
            mem_4 = torch.cuda.memory_allocated() / 1e9
            tqdm.tqdm.write(f"4. After backward: {mem_4:.2f}GB (+{mem_4-mem_3:.2f}GB)")

        optimizer.step()

        if batch_idx < 5:
            mem_5 = torch.cuda.memory_allocated() / 1e9
            tqdm.tqdm.write(f"5. After optimizer.step: {mem_5:.2f}GB (+{mem_5-mem_4:.2f}GB)")
            tqdm.tqdm.write(f"6. Peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f}GB\n")
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

    with torch.no_grad():
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm.tqdm(dataloader)
        for batch_idx, graph in enumerate(pbar):
            # Memory tracking for first 3 validation batches
            if batch_idx < 3:
                mem_before = torch.cuda.memory_allocated() / 1e9
                tqdm.tqdm.write(f"\n=== Validation Batch {batch_idx} ===")
                tqdm.tqdm.write(f"Before: {mem_before:.2f}GB")

            graph = graph.to(device)
            predicted_acc, target_acc = model(graph)
            errors = ((predicted_acc - target_acc) ** 2)
            loss = torch.mean(errors) # MSE Loss

            if batch_idx < 3:
                mem_after = torch.cuda.memory_allocated() / 1e9
                tqdm.tqdm.write(f"After: {mem_after:.2f}GB (+{mem_after-mem_before:.2f}GB)")
                tqdm.tqdm.write(f"Peak: {torch.cuda.max_memory_allocated()/1e9:.2f}GB\n")

            # Update progress bar
            mem_gb = torch.cuda.memory_allocated() / 1e9
            pbar.set_postfix({'loss': f'{loss.item():.2e}', 'mem': f'{mem_gb:.1f}GB'})

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches
