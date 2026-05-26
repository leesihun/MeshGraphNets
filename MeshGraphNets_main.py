# MeshGraphNets
import os

# Must be set before h5py is imported transitively by data loading modules.
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
import socket
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import ProcessExitedException

from general_modules.load_config import load_config
from inference_profiles.rollout import run_rollout
from training_profiles.distributed_training import train_worker
from training_profiles.single_training import single_worker


def main():
    parser = argparse.ArgumentParser(description='MeshGraphNets deterministic runtime')
    parser.add_argument('--config', type=str, default='config.txt',
                        help='Path to config file (default: config.txt)')
    args = parser.parse_args()

    print('\n' * 3)
    print("""
    MeshGraphNets
    Deterministic simulator runtime
    """)
    print(" " * 64 + "Version 1.0.0, 2026-01-06")
    print(" " * 50 + "Developed by SiHun Lee, Ph. D., MX, SEC")
    print()

    config = load_config(args.config)
    run_mode = config.get('mode')
    if run_mode not in ('train', 'inference'):
        raise ValueError(f"Unsupported mode '{run_mode}'. This checkout supports only 'train' and 'inference'.")

    model = config.get('model')

    print('\n' * 2)
    print(f'           Config file   : {args.config}')
    print(f'           Selected Model: {model}, Based on Nvidia physicsNeMo implementation')
    print(f'           Running in    : {run_mode} mode')
    print('\n' * 2)

    gpu_ids = config.get('gpu_ids')
    if not isinstance(gpu_ids, list):
        gpu_ids = [gpu_ids]

    world_size = len(gpu_ids)
    use_distributed = world_size > 1

    parallel_mode = str(config.get('parallel_mode', 'ddp')).lower().strip()
    if parallel_mode not in ('ddp', 'model_split'):
        raise ValueError(f"parallel_mode must be 'ddp' or 'model_split', got '{parallel_mode}'")

    print("GPU Configuration:")
    print(f"  gpu_ids: {gpu_ids}")
    print(f"  world_size (auto-calculated): {world_size}")
    print(f"  use_distributed (auto-calculated): {use_distributed}")
    print(f"  parallel_mode: {parallel_mode}")
    print('\n' * 2)
    print(f"Current absolute path: {os.path.abspath('.')}")

    if run_mode == 'inference':
        run_rollout(config, args.config)
    elif parallel_mode == 'model_split':
        from parallelism.launcher import launch_model_split
        launch_model_split(config, args.config)
    elif use_distributed is False:
        single_worker(config, args.config)
    else:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            config['_ddp_port'] = str(s.getsockname()[1])
        print(f"Starting distributed training with {world_size} processes on GPUs {gpu_ids} (port {config['_ddp_port']})...")
        try:
            mp.spawn(
                train_worker,
                args=(world_size, config, gpu_ids, args.config),
                nprocs=world_size,
                join=True
            )
            print("Distributed training completed.")
        except (KeyboardInterrupt, ProcessExitedException):
            print("\nTraining interrupted by user. All worker processes terminated.")
        except Exception as e:
            print(f"\nDistributed training failed: {e}")


if __name__ == "__main__":
    main()
