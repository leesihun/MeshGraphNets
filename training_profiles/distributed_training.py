import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter
from general_modules.data_loader import load_data
from torch_geometric.loader import DataLoader
from model.MeshGraphNets import MeshGraphNets

import torch
import tqdm
import numpy as np

def train_worker(rank, world_size, config, gpu_ids):
    """Training worker for distributed training.

    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        config: Configuration dictionary
        gpu_ids: List of GPU IDs to use
    """
    if rank == 0:
        os.makedirs('outputs/train', exist_ok=True)

    # Get the physical GPU ID for this rank
    gpu_id = gpu_ids[rank]
    setup_distributed(rank, world_size, gpu_id)

    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')

    print('\n'*2)
    print(f'[Rank {rank}] Using physical GPU {gpu_id}, device: {device}')
    print('\n'*2)

    # Set random seeds for reproducibility across all ranks
    seed = config.get('seed', 42)
    torch.manual_seed(seed + rank)  # Different seed per rank for data augmentation diversity
    np.random.seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)

    # Load dataset (only rank 0 prints)
    if rank == 0:
        print("Loading dataset...")
    dataset = load_data(config)

    # Split dataset
    if rank == 0:
        print("Splitting dataset...")
    train_dataset, val_dataset, test_dataset = dataset.split(0.8, 0.1, 0.1)

    # Create distributed samplers
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
        batch_size=config['batch_size'],
        sampler=test_sampler
    )

    # Create model
    if rank == 0:
        print("Initializing model...")

    # Initialize model on the correct device
    model = MeshGraphNets(config, str(device)).to(device)

    # Wrap with DistributedDataParallel
    # - broadcast_buffers=True: Synchronize buffers (like running stats in BatchNorm) at each forward pass
    # - find_unused_parameters=False: Set to True if you have conditional paths in your model
    # - gradient_as_bucket_view=True: More memory efficient (PyTorch 1.7+)
    model = DDP(
        model,
        broadcast_buffers=True,
        find_unused_parameters=False,
        gradient_as_bucket_view=True
    )

    if rank == 0:
        # Show model architecture/summary
        print('\n'*2)
        print("Model architecture/summary:")
        print(model)
        print('\n'*2)
        print("Model initialized successfully")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    # Initialize optimizer
    learning_rate = config.get('learningr')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir='outputs/train')

    # Initialize learning rate scheduler (optional)
    if config.get('use_scheduler', False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    else:
        scheduler = None

    if rank == 0:
        print(f"Optimizer: Adam (lr={learning_rate})")
        if scheduler:
            print("Scheduler: ReduceLROnPlateau")
        print("Ready for training...")

    # Synchronize all processes before starting training
    dist.barrier()

    # TODO: Training loop

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