# MeshGraphNets
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import torch.multiprocessing as mp
from general_modules.load_config import load_config
from general_modules.data_loader import load_data
from model.MeshGraphNets import MeshGraphNets
from training_profiles.distributed_training import train_worker
from training_profiles.single_training import single_worker

def main():
    print('\n'*3)

    # Display ASCII art banner
    print("""
    ██████╗ █████╗ ███████╗    ███╗   ███╗██╗         ███████╗██╗   ██╗██╗████████╗███████╗
   ██╔════╝██╔══██╗██╔════╝    ████╗ ████║██║         ██╔════╝██║   ██║██║╚══██╔══╝██╔════╝
   ██║     ███████║█████╗      ██╔████╔██║██║         ███████╗██║   ██║██║   ██║   █████╗
   ██║     ██╔══██║██╔══╝      ██║╚██╔╝██║██║         ╚════██║██║   ██║██║   ██║   ██╔══╝
   ╚██████╗██║  ██║███████╗    ██║ ╚═╝ ██║███████╗    ███████║╚██████╔╝██║   ██║   ███████╗
    ╚═════╝╚═╝  ╚═╝╚══════╝    ╚═╝     ╚═╝╚══════╝    ╚══════╝ ╚═════╝ ╚═╝   ╚═╝   ╚══════╝
    """)
    print(" " * 64 + "Version 1.0.0, 2026-01-06")
    print(" " * 50 + "Developed by SiHun Lee, Ph. D., MX, SEC")
    print()

    # Load configuration files
    config = load_config("config.txt")

    run_mode = config.get('mode')
    model = config.get('model')

    print('\n'*2)
    print(f'           Selected Model: {model}, Based on Nvidia physicsNeMo implementation')
    print(f'           Running in    : {run_mode} mode')
    print('\n'*2)
    
    # Current limitation: All timesteps must be equal for all samples
    print('\n'*2)
    print('\n'*2)
    print("Current limitation: All timesteps must be equal for all samples")
    print('\n'*2)
    print('\n'*2)

    # Auto-configure distributed training from gpu_ids
    gpu_ids = config.get('gpu_ids')  # Default to GPU 0 if not specified

    # Ensure gpu_ids is a list
    if not isinstance(gpu_ids, list):
        gpu_ids = [gpu_ids]

    # Auto-calculate world_size and use_distributed
    world_size = len(gpu_ids)
    use_distributed = world_size > 1

    print(f"GPU Configuration:")
    print(f"  gpu_ids: {gpu_ids}")
    print(f"  world_size (auto-calculated): {world_size}")
    print(f"  use_distributed (auto-calculated): {use_distributed}")
    print('\n'*2)

    import os
    # Display the current absolute path
    print(f"Current absolute path: {os.path.abspath('.')}")
    
    if use_distributed==False:
        single_worker(config)
        
    else:
        print(f"Starting distributed training with {world_size} processes on GPUs {gpu_ids}...")
        mp.spawn(
            train_worker,
            args=(world_size, config, gpu_ids),
            nprocs=world_size,
            join=True
        )
        # Actual distributed training done in distributed_training.py,
        # with function train_worker being called for each process.

        print("Distributed training completed.")


if __name__ == "__main__":
    main()