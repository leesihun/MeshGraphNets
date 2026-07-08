import time

import torch
from torch_geometric.loader import DataLoader

from training_profiles.setup import (
    build_dataset_splits,
    build_model_and_ema,
    build_optimizer_scheduler,
    cleanup_dataloaders,
    init_log_file,
    log_model_summary,
    save_checkpoint,
)
from training_profiles.training_loop import (
    log_training_config,
    run_periodic_test,
    train_epoch,
    validate_epoch,
)


def single_worker(config, config_filename='config.txt'):
    """Single GPU/CPU training entry point."""
    gpu_ids = config.get('gpu_ids')
    print("Starting single-process training...")

    if torch.cuda.is_available():
        gpu_id = gpu_ids
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        print(f'Using physical GPU {gpu_id}, device: {device}')
        print(f'Initial GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB')
    else:
        device = torch.device('cpu')
        print(f'Using device: {device}')

    # ---- Dataset ----
    print("\nLoading dataset...")
    split_seed = int(config.get('split_seed', 42))
    train_dataset, val_dataset, test_dataset = build_dataset_splits(config, split_seed)
    if torch.cuda.is_available():
        print(f'After dataset load: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    print("Writing train-derived normalization stats to HDF5...")
    train_dataset.write_preprocessing_to_hdf5(split_seed)

    if config.get('use_node_types', False) and train_dataset.num_node_types is not None:
        print(f"  Node types enabled: {train_dataset.num_node_types} types will be added to input")

    # ---- DataLoaders ----
    print("\nCreating dataloaders...")
    num_workers = config['num_workers']
    pin_memory = torch.cuda.is_available()
    config['_pin_memory'] = pin_memory
    mp_context = 'spawn' if num_workers > 0 else None
    prefetch_factor = int(config.get('prefetch_factor', 4)) if num_workers > 0 else None
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor,
        multiprocessing_context=mp_context,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor,
        multiprocessing_context=mp_context,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, pin_memory=pin_memory)

    if torch.cuda.is_available():
        print(f'After dataloader creation: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    # ---- Model ----
    print("\nInitializing model...")
    model, ema_model = build_model_and_ema(config, device)
    if torch.cuda.is_available():
        print(f'After model initialization: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    log_model_summary(model, config, ema_model)

    # ---- Optimizer / Scheduler ----
    print("\nInitializing optimizer...")
    total_epochs = config.get('training_epochs')
    optimizer, scheduler, warmup_epochs, cosine_T0 = build_optimizer_scheduler(
        config, model.parameters(), total_epochs
    )
    use_fused = torch.cuda.is_available()
    print(f"Optimizer: AdamW (fused={use_fused}, weight_decay={float(config.get('weight_decay', 1e-4))})")
    print(f"Scheduler: LinearLR warmup ({warmup_epochs} epochs) -> "
          f"CosineAnnealingWarmRestarts (T_0={cosine_T0}, T_mult=2, eta_min=1e-8)")

    if torch.cuda.is_available():
        print(f'After optimizer creation: {torch.cuda.memory_allocated()/1e9:.2f}GB')
        print(f'Peak memory so far: {torch.cuda.max_memory_allocated()/1e9:.2f}GB')

    log_training_config(config)
    print("\n" + "=" * 60)
    print("Starting training loop...")
    print("=" * 60 + "\n")
    start_time = time.time()

    # ---- Logging ----
    log_file = init_log_file(config, config_filename)

    modelname = config.get('modelpath')

    val_interval = int(config.get('val_interval', 1))
    train_loss = float('nan')
    valid_loss = float('nan')

    try:
        for epoch in range(total_epochs):
            train_metrics = train_epoch(
                model, train_loader, optimizer, device, config, epoch, ema_model=ema_model,
            )

            train_loss = train_metrics['mean']
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            do_val = (epoch % val_interval == 0) or (epoch == total_epochs - 1)

            eval_model = ema_model.module if ema_model is not None else model
            if do_val:
                valid_metrics = validate_epoch(eval_model, val_loader, device, config, epoch)
                valid_loss = valid_metrics['mean']
            else:
                valid_metrics = {}

            if do_val:
                print(
                    f"Epoch {epoch}/{total_epochs} "
                    f"TrainOpt: {train_loss:.2e} "
                    f"Valid: {valid_loss:.2e} LR: {current_lr:.2e}"
                )
            else:
                print(
                    f"Epoch {epoch}/{total_epochs} "
                    f"TrainOpt: {train_loss:.2e} LR: {current_lr:.2e}"
                )

            if log_file:
                with open(log_file, 'a') as f:
                    elapsed = time.time() - start_time
                    val_str = f"Valid {valid_loss:.4e}" if do_val else "Valid skipped"
                    f.write(
                        f"Elapsed: {elapsed:.2f}s "
                        f"Epoch {epoch} TrainOpt {train_loss:.4e} "
                        f"{val_str} LR: {current_lr:.4e}\n"
                    )

            test_interval = int(config.get('test_interval', 10))
            last_epoch = epoch == total_epochs - 1
            if epoch % test_interval == 0 or last_epoch:
                run_periodic_test(eval_model, test_loader, device, config, epoch, train_dataset)

        save_checkpoint(
            epoch, model, ema_model, optimizer, scheduler,
            train_loss, valid_loss, config, train_dataset, modelname,
        )
        print(f"\nTraining finished. Final model saved at epoch {epoch} with validation loss {valid_loss:.2e}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. No checkpoint saved.")

    cleanup_dataloaders(train_loader, val_loader, test_loader)
