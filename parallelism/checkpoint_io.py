"""State-dict merge/slice for pipeline-split MeshGraphNets."""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.distributed as dist


_ENCODER_PREFIXES = ('model.encoder.',)
_DECODER_PREFIXES = ('model.decoder.',)


def _processor_key_prefix(block_index: int, is_multiscale: bool, multiscale_meta: dict = None) -> List[str]:
    """Return state-dict key prefixes for a processor block."""
    if not is_multiscale:
        return [f"model.processer_list.{block_index}."]

    if multiscale_meta is None:
        raise ValueError("multiscale_meta required when is_multiscale=True")
    L = int(multiscale_meta['L'])
    mp_per_level = [int(x) for x in multiscale_meta['mp_per_level']]

    counts = []
    names = []
    for i in range(L):
        counts.append(mp_per_level[i])
        names.append(('pre', i))
    counts.append(mp_per_level[L])
    names.append(('coarsest', None))
    for i in range(L - 1, -1, -1):
        counts.append(mp_per_level[2 * L - i])
        names.append(('post', i))

    cumulative = 0
    for (kind, level), count in zip(names, counts):
        if block_index < cumulative + count:
            local_idx = block_index - cumulative
            if kind == 'pre':
                return [f"model.pre_blocks.{level}.{local_idx}."]
            if kind == 'post':
                return [f"model.post_blocks.{level}.{local_idx}."]
            return [f"model.coarsest_blocks.{local_idx}."]
        cumulative += count

    raise IndexError(f"block_index {block_index} out of range for multiscale layout")


def slice_state_dict_for_stage(
    full_sd: Dict[str, torch.Tensor],
    stage_idx: int,
    num_stages: int,
    assignment: Sequence[Sequence[int]],
    is_multiscale: bool,
    multiscale_meta: dict = None,
) -> Dict[str, torch.Tensor]:
    """Extract the subset of keys from `full_sd` owned by one stage."""
    stage_blocks = list(assignment[stage_idx])
    allowed_prefixes: List[str] = []

    for block_idx in stage_blocks:
        allowed_prefixes.extend(_processor_key_prefix(block_idx, is_multiscale, multiscale_meta))

    if is_multiscale and multiscale_meta is not None:
        L = int(multiscale_meta['L'])
        mp = [int(x) for x in multiscale_meta['mp_per_level']]

        cumulative_pre = 0
        for level in range(L):
            last_pre_b = cumulative_pre + mp[level] - 1
            if last_pre_b in stage_blocks:
                allowed_prefixes.append(f'model.coarse_eb_encoders.{level}.')
            cumulative_pre += mp[level]

        cumulative_post = sum(mp[:L + 1])
        for level in range(L - 1, -1, -1):
            first_post_b = cumulative_post
            if first_post_b in stage_blocks:
                allowed_prefixes.append(f'model.skip_projs.{level}.')
                allowed_prefixes.append(f'model.unpool_blocks.{level}.')
            cumulative_post += mp[2 * L - level]

    if stage_idx == 0:
        allowed_prefixes.extend(_ENCODER_PREFIXES)
    if stage_idx == num_stages - 1:
        allowed_prefixes.extend(_DECODER_PREFIXES)

    result: Dict[str, torch.Tensor] = {}
    for key, tensor in full_sd.items():
        for prefix in allowed_prefixes:
            if key.startswith(prefix):
                result[key] = tensor
                break
    return result


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
