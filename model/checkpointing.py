"""
Gradient Checkpointing Utilities for MeshGraphNets

Reduces VRAM usage by recomputing activations during backward pass
instead of storing them. Trades ~20-30% more compute for ~60-70% less memory.
Covers the processor GnBlocks (per block) plus the encoder, coarse edge
encoders, and unpool/skip-merge steps of the multiscale V-cycle.

Usage:
    Set `use_checkpointing True` in config.txt to enable.
"""

from torch.utils.checkpoint import checkpoint


def run_checkpointed(fn, *args, enabled: bool = True):
    """Run fn(*args), optionally under non-reentrant gradient checkpointing.

    Used for the non-GnBlock stages (encoder, coarse edge encoders, unpool +
    skip merge) so `use_checkpointing True` covers the whole forward pass, not
    just the processor blocks. Non-reentrant checkpointing tracks module
    parameters and closures, so fn may be an nn.Module or a bound method and
    args may include non-tensors.
    """
    if enabled:
        return checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


def checkpoint_gn_block(block, x, edge_attr, edge_index, world_edge_attr=None, world_edge_index=None):
    """
    Run a GnBlock with gradient checkpointing via its tensor fast path.

    Args:
        block: GnBlock module
        x: Node features [N, latent_dim]
        edge_attr: Edge features [E, latent_dim]
        edge_index: Edge connectivity [2, E]
        world_edge_attr: World edge features [E_world, latent_dim] (optional)
        world_edge_index: World edge connectivity [2, E_world] (optional)

    Returns:
        Tuple of (updated_x, updated_edge_attr, updated_world_edge_attr)
    """
    return checkpoint(
        block.forward_tensors,
        x,
        edge_attr,
        edge_index,
        world_edge_attr,
        world_edge_index,
        use_reentrant=False
    )


def process_with_checkpointing(processor_list, x, edge_attr, edge_index,
                               world_edge_attr=None, world_edge_index=None):
    """
    Run processor blocks with gradient checkpointing on raw tensors.

    Args:
        processor_list: nn.ModuleList of GnBlock modules
        x, edge_attr, edge_index: latent graph tensors (after encoding)
        world_edge_attr, world_edge_index: optional world-edge tensors
    Returns:
        Tuple of (x, edge_attr, world_edge_attr); edge indices are unchanged.
    """
    for block in processor_list:
        x, edge_attr, world_edge_attr = checkpoint_gn_block(
            block, x, edge_attr, edge_index, world_edge_attr, world_edge_index
        )
    return x, edge_attr, world_edge_attr
