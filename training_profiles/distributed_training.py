import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from general_modules.data_loader import load_data
from torch_geometric.loader import DataLoader
from model.MeshGraphNets import MeshGraphNets
from training_profiles.training_loop import train_epoch, validate_epoch, test_model

def train_worker(rank, world_size, config, gpu_ids):
    """Training worker for distributed training.

    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        config: Configuration dictionary
        gpu_ids: List of GPU IDs to use
    """
    # Get the physical GPU ID for this rank
    gpu_id = gpu_ids[rank]
    setup_distributed(rank, world_size, gpu_id)

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        if rank == 0:
            print(f'[Rank {rank}] Using physical GPU {gpu_id}, device: {device}')
            print(f'Initial GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB')
    else:
        device = torch.device('cpu')
        if rank == 0:
            print(f'Using device: {device}')

    # Generate dataloader from dataset
    if rank == 0:
        print("\nLoading dataset...")
    dataset = load_data(config)
    if torch.cuda.is_available() and rank == 0:
        print(f'After dataset load: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    # Pass num_node_types to config for model to compute input dimension
    if config.get('use_node_types', False) and dataset.num_node_types is not None:
        config['num_node_types'] = dataset.num_node_types
        if rank == 0:
            print(f"  Node types enabled: {dataset.num_node_types} types will be added to input")

    # Divide the dataset into training, validation, and test sets
    if rank == 0:
        print("\nSplitting dataset...")
    train_dataset, val_dataset, test_dataset = dataset.split(0.8, 0.1, 0.1)

    # Create distributed samplers
    if rank == 0:
        print("\nCreating dataloaders with distributed samplers...")
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        sampler=test_sampler,
        pin_memory=True
    )
    if torch.cuda.is_available() and rank == 0:
        print(f'After dataloader creation: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    # Generate MeshGraphNets model
    if rank == 0:
        print("\nInitializing model...")
    model = MeshGraphNets(config, str(device)).to(device)

    # Wrap with DistributedDataParallel
    if torch.cuda.is_available():
        model = DDP(
            model,
            device_ids=[gpu_id],
            broadcast_buffers=True,
            find_unused_parameters=False,
            gradient_as_bucket_view=True
        )
    else:
        model = DDP(
            model,
            broadcast_buffers=True,
            find_unused_parameters=False,
            gradient_as_bucket_view=True
        )

    if torch.cuda.is_available() and rank == 0:
        print(f'After model initialization: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    if rank == 0:
        print('\n'*2)
        print("Model architecture/summary:")
        print(model)
        print('\n'*2)
        print("Model initialized successfully")
        if config.get('use_checkpointing', False):
            print("Gradient checkpointing: ENABLED")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    best_valid_loss = float('inf')
    best_epoch = -1

    # Initialize optimizer
    if rank == 0:
        print("\nInitializing optimizer...")
    learning_rate = config.get('learningr')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        min_lr=1e-8
    )
    if rank == 0:
        print(f"Learning rate scheduler: ReduceLROnPlateau (factor=0.5, patience=10)")

    if torch.cuda.is_available() and rank == 0:
        print(f'After optimizer creation: {torch.cuda.memory_allocated()/1e9:.2f}GB')
        print(f'Peak memory so far: {torch.cuda.max_memory_allocated()/1e9:.2f}GB')

    if rank == 0:
        print("\n" + "="*60)
        print("Starting training loop...")
        print("="*60 + "\n")

    start_time = time.time()

    log_file = None
    log_file_dir = config.get('log_file_dir')
    if log_file_dir and rank == 0:
        log_file = 'outputs/' + log_file_dir
        # if log_file doesn't exist, create it
        if not os.path.exists(log_file):
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'w') as f:
            f.write(f"Training epoch log file\n")
            f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Log file absolute path: {os.path.abspath(log_file)}\n")
            # Write the whole config.txt file here:
            with open('config.txt', 'r') as fc:
                f.write(fc.read())
            fc.close()

    # Synchronize all processes before starting training
    dist.barrier()

    for epoch in range(config.get('training_epochs')):
        # Set epoch for distributed sampler (important for shuffling)
        train_sampler.set_epoch(epoch)

        train_loss = train_epoch(model, train_loader, optimizer, device, config, epoch)
        valid_loss = validate_epoch(model, val_loader, device, config, epoch)

        # Step the learning rate scheduler based on validation loss
        scheduler.step(valid_loss)

        # Per epoch, batch-averaged train, validation losses.
        current_lr = optimizer.param_groups[0]['lr']
        if rank == 0:
            print(f"Epoch {epoch}/{config['training_epochs']} Train Loss: {train_loss:.2e} Valid Loss: {valid_loss:.2e} LR: {current_lr:.2e}")

        # Only rank 0 saves checkpoints
        if valid_loss < best_valid_loss and rank == 0:
            best_valid_loss = valid_loss
            best_epoch = epoch
            checkpoint_path = os.path.join("outputs/", "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),  # Save unwrapped model
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
            }, checkpoint_path)
            print(f"  -> New best model saved at epoch {epoch} with valid loss {valid_loss:.2e}")

        if log_file_dir and rank == 0:
            with open(log_file, 'a') as f:
                f.write(f"Elapsed time: {time.time() - start_time:.2f}s Epoch {epoch} Train Loss: {train_loss:.4e} Valid Loss: {valid_loss:.4e} LR: {current_lr:.4e}\n")

        # For each 10 epochs, test the model on the test set and save the results with the ground truth in a file
        if epoch % 10 == 0 and rank == 0:
            test_loss = test_model(model, test_loader, device, config, epoch, dataset)

    if rank == 0:
        print(f"\nTraining finished. Best model at epoch {best_epoch} with validation loss {best_valid_loss:.2e}")

    # Cleanup
    cleanup_distributed()

def setup_distributed(rank, world_size, gpu_id):
    """Initialize distributed training process group.

    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        gpu_id: Physical GPU ID to use for this rank
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        rank=rank,
        world_size=world_size
    )

    # Set device for this process using the specified GPU ID
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()