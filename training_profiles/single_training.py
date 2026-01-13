import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter
from general_modules.data_loader import load_data
from torch_geometric.loader import DataLoader
from model.MeshGraphNets import MeshGraphNets
from training_profiles.training_loop import train_epoch, validate_epoch

import torch
import tqdm
import numpy as np

def single_worker(config):
    # Single GPU/CPU training
    gpu_ids = config.get('gpu_ids')
    print("Starting single-process training...")

    # Set device using the first (and only) GPU from gpu_ids
    if torch.cuda.is_available():
        gpu_id = 0
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        print(f'Using physical GPU {gpu_id}, device: {device}')
        print(f'Initial GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB')
    else:
        device = torch.device('cpu')
        print(f'Using device: {device}')

    # Generate dataloader from dataset
    print("\nLoading dataset...")
    dataset = load_data(config)
    if torch.cuda.is_available():
        print(f'After dataset load: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    # Divide the dataset into training, validation, and test sets
    print("\nSplitting dataset...")
    train_dataset, val_dataset, test_dataset = dataset.split(0.8, 0.1, 0.1)

    # Create dataloaders (no distributed samplers for single process)
    print("\nCreating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )
    if torch.cuda.is_available():
        print(f'After dataloader creation: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    # Generate MeshGraphNets model
    print("\nInitializing model...")
    model = MeshGraphNets(config, str(device)).to(device)
    if torch.cuda.is_available():
        print(f'After model initialization: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    print('\n'*2)
    print("Model architecture/summary:")
    print(model)
    print('\n'*2)
    print("Model initialized successfully")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    # TODO: Train the model

    best_valid_loss = float('inf')
    best_epoch = -1

    # Initialize optimizer
    print("\nInitializing optimizer...")
    learning_rate = config.get('learningr')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if torch.cuda.is_available():
        print(f'After optimizer creation: {torch.cuda.memory_allocated()/1e9:.2f}GB')
        print(f'Peak memory so far: {torch.cuda.max_memory_allocated()/1e9:.2f}GB')
        print("\n" + "="*60)
        print("Starting training loop...")
        print("="*60 + "\n")

    for epoch in range(config.get('training_epochs')):
    
        train_loss = train_epoch(model, train_loader, optimizer, device, config)
        valid_loss = validate_epoch(model, val_loader, device, config)

        # Per epoch, batch-averaged train, validation losses.
        print(f"Epoch {epoch}/{config['training_epochs']} Train Loss: {train_loss:.2e} Valid Loss: {valid_loss:.2e}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            checkpoint_path = os.path.join("outputs/", "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_loss,
            }, checkpoint_path)
            print(f"  -> New best model saved at epoch {epoch} with valid loss {valid_loss:.2e}")

    print(f"\nTraining finished. Best model at epoch {best_epoch} with validation loss {best_valid_loss:.2e}")