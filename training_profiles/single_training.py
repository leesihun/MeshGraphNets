import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter
from general_modules.data_loader import load_data
from torch_geometric.loader import DataLoader
from model.MeshGraphNets import MeshGraphNets
from training_profiles.training_loop import train_epoch, validate_epoch, infer_model

import torch
import tqdm
import numpy as np
import os
import time

def single_worker(config):
    # Single GPU/CPU training
    gpu_ids = config.get('gpu_ids')
    print("Starting single-process training...")

    # Set device using the first (and only) GPU from gpu_ids
    if torch.cuda.is_available():
        gpu_id = gpu_ids
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

    # Update config with actual input dimension (includes node types if enabled)
    if config.get('use_node_types', False) and dataset.num_node_types is not None:
        actual_input_dim = config['input_var'] + dataset.num_node_types
        print(f"  Node types enabled: input_var {config['input_var']} + {dataset.num_node_types} types = {actual_input_dim} total")
        config['input_var'] = actual_input_dim

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
        batch_size=1,
        shuffle=False,
        pin_memory=True
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
    if config.get('use_checkpointing', False):
        print("Gradient checkpointing: ENABLED")
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,
        patience=20,
        min_lr=1e-8
    )
    print(f"Learning rate scheduler: ReduceLROnPlateau (factor=0.5, patience=10)")

    if torch.cuda.is_available():
        print(f'After optimizer creation: {torch.cuda.memory_allocated()/1e9:.2f}GB')
        print(f'Peak memory so far: {torch.cuda.max_memory_allocated()/1e9:.2f}GB')

    print("\n" + "="*60)
    print("Starting training loop...")
    print("="*60 + "\n")

    start_time = time.time()

    log_file_dir = config.get('log_file_dir')
    if log_file_dir:
        log_file = 'outputs/'+str(config.get('gpu_ids'))+'/'+log_file_dir
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

    for epoch in range(config.get('training_epochs')):
    
        train_loss = train_epoch(model, train_loader, optimizer, device, config, epoch)
        valid_loss = validate_epoch(model, val_loader, device, config)

        # Step the learning rate scheduler based on validation loss
        scheduler.step(valid_loss)

        # Per epoch, batch-averaged train, validation losses.
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{config['training_epochs']} Train Loss: {train_loss:.2e} Valid Loss: {valid_loss:.2e} LR: {current_lr:.2e}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            checkpoint_path = os.path.join("outputs/", "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
            }, checkpoint_path)
            print(f"  -> New best model saved at epoch {epoch} with valid loss {valid_loss:.2e}")

        if log_file_dir:
            with open(log_file, 'a') as f:
                f.write(f"Elapsed time: {time.time() - start_time:.2f}s Epoch {epoch} Train Loss: {train_loss:.4e} Valid Loss: {valid_loss:.4e} LR: {current_lr:.4e}\n")

        # For each 10 epochs, test the model on the test set and save the results with the ground truth in a file
        if epoch % 10 == 0:
            test_loss = infer_model(model, test_loader, device, config, epoch)

    print(f"\nTraining finished. Best model at epoch {best_epoch} with validation loss {best_valid_loss:.2e}")