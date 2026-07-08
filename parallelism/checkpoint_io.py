"""State-dict merge/slice for pipeline-split MeshGraphNets."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.distributed as dist


_ENCODER_PREFIXES = ('model.encoder.',)
_DECODER_PREFIXES = ('model.decoder.',)


def merge_stage_state_dicts_to_rank0(
    stage_sd: Dict[str, torch.Tensor],
    group=None,
) -> Dict[str, torch.Tensor]:
    """Gather per-stage state dicts to rank 0 and merge into a single dict."""
    if not dist.is_available() or not dist.is_initialized():
        return dict(stage_sd)

    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    cpu_sd = {k: v.detach().cpu() for k, v in stage_sd.items()}
    gathered: List[Dict[str, torch.Tensor]] = [None] * world_size  # type: ignore[list-item]
    dist.all_gather_object(gathered, cpu_sd, group=group)

    if rank != 0:
        return {}

    merged: Dict[str, torch.Tensor] = {}
    for sd in gathered:
        for k, v in sd.items():
            if k in merged:
                print(f"  [merge_state_dict] WARNING: duplicate key '{k}' across stages; last writer wins")
            merged[k] = v
    return merged
