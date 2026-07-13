"""Pipeline stage implementation for deterministic model-split MeshGraphNets."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch_geometric.data import Data

from general_modules.edge_features import EDGE_FEATURE_DIM
from model.checkpointing import (
    checkpoint_gn_block,
    process_with_checkpointing,
    run_checkpointed,
)
from model.coarsening import pool_features
from model.encoder_decoder import Decoder, Encoder, GnBlock
from model.mlp import build_mlp


_DTYPE_TO_CODE = {
    torch.float32: 0, torch.float64: 1, torch.bfloat16: 2, torch.float16: 3,
    torch.int64: 4, torch.int32: 5, torch.bool: 6,
}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}


# The two pipeline traffic directions use separate process groups. NCCL runs
# each communicator on its own stream, so a queued downstream data-send can
# never serialize ahead of an upstream grad op and deadlock the 1F1B schedule.
_PG_DATA = None
_PG_GRAD = None

# In-flight async sends: (work, buffer) pairs kept alive until the transfer
# completes. Swept opportunistically; fully drained before each optimizer step.
_PENDING_SENDS: List[tuple] = []


def set_pipeline_process_groups(pg_data, pg_grad) -> None:
    global _PG_DATA, _PG_GRAD
    _PG_DATA = pg_data
    _PG_GRAD = pg_grad


def _isend_tracked(t: torch.Tensor, dst: int, group) -> None:
    work = dist.isend(t, dst=dst, group=group)
    _PENDING_SENDS.append((work, t))
    if len(_PENDING_SENDS) > 16:
        _PENDING_SENDS[:] = [(w, b) for w, b in _PENDING_SENDS if not w.is_completed()]


def drain_pending_sends() -> None:
    """Block until every outstanding isend finished; call before optimizer.step()."""
    for work, _ in _PENDING_SENDS:
        work.wait()
    _PENDING_SENDS.clear()


def _dtype_groups(dtypes: Sequence[torch.dtype]) -> List[Tuple[torch.dtype, List[int]]]:
    """Indices grouped by dtype in first-appearance order.

    Both peers derive the grouping from the same (shape, dtype) sequence, so the
    per-dtype flat buffers pair up without any extra negotiation.
    """
    groups: Dict[torch.dtype, List[int]] = {}
    for i, dt in enumerate(dtypes):
        groups.setdefault(dt, []).append(i)
    return list(groups.items())


def _send_bundle(tensors: Sequence[torch.Tensor], dst: int, group) -> None:
    """One header + one flat buffer per dtype instead of 3 messages per tensor."""
    dev = tensors[0].device
    meta: List[int] = [len(tensors)]
    for t in tensors:
        meta.append(t.dim())
        meta.extend(t.shape)
        meta.append(_DTYPE_TO_CODE[t.dtype])
    header = torch.tensor(meta, dtype=torch.long, device=dev)
    header_len = torch.tensor([header.numel()], dtype=torch.long, device=dev)
    _isend_tracked(header_len, dst, group)
    _isend_tracked(header, dst, group)
    for _, idxs in _dtype_groups([t.dtype for t in tensors]):
        parts = [tensors[i].reshape(-1) for i in idxs if tensors[i].numel() > 0]
        if not parts:
            continue
        buf = parts[0].contiguous() if len(parts) == 1 else torch.cat(parts)
        _isend_tracked(buf, dst, group)


def _recv_bundle(src: int, device: torch.device, group) -> List[torch.Tensor]:
    header_len = torch.empty(1, dtype=torch.long, device=device)
    dist.recv(header_len, src=src, group=group)
    header = torch.empty(int(header_len.item()), dtype=torch.long, device=device)
    dist.recv(header, src=src, group=group)
    meta = header.tolist()
    n = meta[0]
    shapes: List[List[int]] = []
    dtypes: List[torch.dtype] = []
    pos = 1
    for _ in range(n):
        ndim = meta[pos]; pos += 1
        shapes.append(meta[pos:pos + ndim]); pos += ndim
        dtypes.append(_CODE_TO_DTYPE[meta[pos]]); pos += 1
    out: List[Optional[torch.Tensor]] = [None] * n
    for dt, idxs in _dtype_groups(dtypes):
        sizes = [math.prod(shapes[i]) for i in idxs]
        nonzero = [(i, s) for i, s in zip(idxs, sizes) if s > 0]
        total = sum(s for _, s in nonzero)
        if total > 0:
            buf = torch.empty(total, dtype=dt, device=device)
            dist.recv(buf, src=src, group=group)
            chunks = torch.split(buf, [s for _, s in nonzero])
            for (i, _), chunk in zip(nonzero, chunks):
                out[i] = chunk.view(shapes[i])
        for i, s in zip(idxs, sizes):
            if s == 0:
                out[i] = torch.empty(shapes[i], dtype=dt, device=device)
    return out


def _send_flat_grads(grads: Sequence[Optional[torch.Tensor]],
                     shapes: Sequence[Tuple[int, ...]],
                     dtypes: Sequence[torch.dtype],
                     dst: int, device: torch.device, group) -> None:
    """Headerless grad send: the peer already knows every shape and dtype."""
    flat: List[torch.Tensor] = []
    for g, shape, dt in zip(grads, shapes, dtypes):
        if g is None:
            g = torch.zeros(shape, dtype=dt, device=device)
        elif g.dtype != dt:
            g = g.to(dt)
        flat.append(g.reshape(-1))
    for _, idxs in _dtype_groups(list(dtypes)):
        parts = [flat[i] for i in idxs if flat[i].numel() > 0]
        if not parts:
            continue
        buf = parts[0].contiguous() if len(parts) == 1 else torch.cat(parts)
        _isend_tracked(buf, dst, group)


def _recv_flat_grads(shapes: Sequence[Tuple[int, ...]],
                     dtypes: Sequence[torch.dtype],
                     src: int, device: torch.device, group) -> List[torch.Tensor]:
    out: List[Optional[torch.Tensor]] = [None] * len(shapes)
    for dt, idxs in _dtype_groups(list(dtypes)):
        sizes = [math.prod(shapes[i]) for i in idxs]
        nonzero = [(i, s) for i, s in zip(idxs, sizes) if s > 0]
        total = sum(s for _, s in nonzero)
        if total > 0:
            buf = torch.empty(total, dtype=dt, device=device)
            dist.recv(buf, src=src, group=group)
            chunks = torch.split(buf, [s for _, s in nonzero])
            for (i, _), chunk in zip(nonzero, chunks):
                out[i] = chunk.view(shapes[i])
        for i, s in zip(idxs, sizes):
            if s == 0:
                out[i] = torch.zeros(shapes[i], dtype=dt, device=device)
    return out


class BundleSend(torch.autograd.Function):
    """Send grad-bearing and non-differentiable tensors downstream."""

    @staticmethod
    def forward(ctx, dst: int, n_grad: int, *all_tensors: torch.Tensor) -> torch.Tensor:
        ctx.dst = dst
        ctx.n_grad = n_grad
        ctx.n_nongd = len(all_tensors) - n_grad
        ctx.device = all_tensors[0].device
        ctx.grad_shapes = [tuple(t.shape) for t in all_tensors[:n_grad]]
        ctx.grad_dtypes = [t.dtype for t in all_tensors[:n_grad]]
        if ctx.n_nongd > 0:
            ctx.mark_non_differentiable(*all_tensors[n_grad:])
        _send_bundle(all_tensors, dst, _PG_DATA)
        return torch.zeros((), device=ctx.device, dtype=all_tensors[0].dtype, requires_grad=True)

    @staticmethod
    def backward(ctx, _grad_sentinel):
        grads = _recv_flat_grads(
            ctx.grad_shapes, ctx.grad_dtypes,
            src=ctx.dst, device=ctx.device, group=_PG_GRAD,
        )
        return (None, None) + tuple(grads) + (None,) * ctx.n_nongd


class BundleRecv(torch.autograd.Function):
    """Receive a tensor bundle from upstream and send gradients back in backward."""

    @staticmethod
    def forward(ctx, src: int, n_grad: int, n_nongd: int,
                device: torch.device, anchor: torch.Tensor):
        ctx.src = src
        ctx.n_grad = n_grad
        ctx.device = device
        tensors = _recv_bundle(src, device, _PG_DATA)
        if len(tensors) != n_grad + n_nongd:
            raise RuntimeError(
                f"bundle size mismatch: received {len(tensors)}, "
                f"expected {n_grad}+{n_nongd}"
            )
        ctx.out_shapes = [tuple(t.shape) for t in tensors[:n_grad]]
        ctx.out_dtypes = [t.dtype for t in tensors[:n_grad]]
        if n_nongd > 0:
            ctx.mark_non_differentiable(*tensors[n_grad:])
        return tuple(tensors)

    @staticmethod
    def backward(ctx, *grads):
        _send_flat_grads(
            grads[:ctx.n_grad], ctx.out_shapes, ctx.out_dtypes,
            dst=ctx.src, device=ctx.device, group=_PG_GRAD,
        )
        return (None, None, None, None, None)


def _bundle_counts(
    use_world_edges: bool,
    n_skips: int,
    is_multiscale: bool,
    use_coarse_world_edges: bool = False,
) -> Tuple[int, int]:
    """Return (n_grad, n_nongd) for a boundary bundle."""
    W = 1 if use_world_edges else 0
    ms = 1 if is_multiscale else 0
    if use_coarse_world_edges:
        n_grad = 2 + W + (2 + W) * n_skips
        n_nongd = 1 + W + (1 + W) * n_skips + ms
    else:
        n_grad = 2 + W + 2 * n_skips + (W if n_skips > 0 else 0)
        n_nongd = 1 + W + n_skips + (W if n_skips > 0 else 0) + ms
    return n_grad, n_nongd


def _pack_bundle(
    x: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_index: torch.Tensor,
    skip_stack: List[dict],
    world_edge_attr: Optional[torch.Tensor],
    world_edge_index: Optional[torch.Tensor],
    current_level_idx: int,
    use_world_edges: bool,
    is_multiscale: bool,
    use_coarse_world_edges: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    dev = x.device
    D = x.shape[-1]

    grad_t: List[torch.Tensor] = [x, edge_attr]
    nongd_t: List[torch.Tensor] = [edge_index]

    if use_world_edges:
        grad_t.append(
            world_edge_attr if world_edge_attr is not None
            else torch.zeros(0, D, device=dev, dtype=x.dtype)
        )
        nongd_t.append(
            world_edge_index if world_edge_index is not None
            else torch.zeros(2, 0, dtype=torch.long, device=dev)
        )

    for ss in skip_stack:
        grad_t.extend([ss['x'], ss['edge_attr']])
        nongd_t.append(ss['edge_index'])

    if use_world_edges and len(skip_stack) > 0:
        if use_coarse_world_edges:
            for ss in skip_stack:
                grad_t.append(ss.get('w_attr') if ss.get('w_attr') is not None
                              else torch.zeros(0, D, device=dev, dtype=x.dtype))
            for ss in skip_stack:
                nongd_t.append(ss.get('w_idx') if ss.get('w_idx') is not None
                               else torch.zeros(2, 0, dtype=torch.long, device=dev))
        else:
            grad_t.append(skip_stack[0].get('w_attr') if skip_stack[0].get('w_attr') is not None
                          else torch.zeros(0, D, device=dev, dtype=x.dtype))
            nongd_t.append(skip_stack[0].get('w_idx') if skip_stack[0].get('w_idx') is not None
                           else torch.zeros(2, 0, dtype=torch.long, device=dev))

    if is_multiscale:
        nongd_t.append(torch.tensor([current_level_idx, len(skip_stack)], dtype=torch.long, device=dev))

    return grad_t, nongd_t


def _unpack_bundle_indexed(
    all_tensors: tuple,
    n_skips: int,
    use_world_edges: bool,
    is_multiscale: bool,
    use_coarse_world_edges: bool = False,
) -> Tuple:
    idx = 0
    x = all_tensors[idx]; idx += 1
    edge_attr = all_tensors[idx]; idx += 1
    world_edge_attr = all_tensors[idx] if use_world_edges else None
    if use_world_edges:
        idx += 1

    skip_x_list, skip_ea_list = [], []
    for _ in range(n_skips):
        skip_x_list.append(all_tensors[idx]); idx += 1
        skip_ea_list.append(all_tensors[idx]); idx += 1

    skip_w_attr_list: List[Optional[torch.Tensor]] = [None] * n_skips
    if use_world_edges and n_skips > 0:
        if use_coarse_world_edges:
            for i in range(n_skips):
                skip_w_attr_list[i] = all_tensors[idx]; idx += 1
        else:
            skip_w_attr_list[0] = all_tensors[idx]; idx += 1

    n_grad, _ = _bundle_counts(use_world_edges, n_skips, is_multiscale, use_coarse_world_edges)
    idx = n_grad

    edge_index = all_tensors[idx]; idx += 1
    world_edge_index = all_tensors[idx] if use_world_edges else None
    if use_world_edges:
        idx += 1

    skip_ei_list = [all_tensors[idx + i] for i in range(n_skips)]
    idx += n_skips

    skip_w_idx_list: List[Optional[torch.Tensor]] = [None] * n_skips
    if use_world_edges and n_skips > 0:
        if use_coarse_world_edges:
            for i in range(n_skips):
                skip_w_idx_list[i] = all_tensors[idx]; idx += 1
        else:
            skip_w_idx_list[0] = all_tensors[idx]; idx += 1

    current_level_idx = 0
    if is_multiscale:
        meta = all_tensors[idx]
        current_level_idx = int(meta[0].item())

    skip_stack = []
    for i in range(n_skips):
        skip_stack.append({
            'x': skip_x_list[i],
            'edge_attr': skip_ea_list[i],
            'edge_index': skip_ei_list[i],
            'w_attr': skip_w_attr_list[i],
            'w_idx': skip_w_idx_list[i],
        })

    return x, edge_attr, edge_index, world_edge_attr, world_edge_index, skip_stack, current_level_idx


def _parse_mp_per_level(config: dict, L: int) -> List[int]:
    mp = config.get('mp_per_level', None)
    if mp is None:
        raise ValueError(
            'use_multiscale=True requires mp_per_level '
            '(2 * multiscale_levels + 1 entries)'
        )
    if not isinstance(mp, list):
        mp = [int(mp)]
    else:
        mp = [int(x) for x in mp]
    return mp


def _block_vcycle_info(b: int, L: int, mp_per_level: List[int]) -> Tuple[str, Optional[int], int]:
    cumulative = 0
    for i in range(L):
        count = mp_per_level[i]
        if b < cumulative + count:
            return 'pre', i, b - cumulative
        cumulative += count
    count = mp_per_level[L]
    if b < cumulative + count:
        return 'coarsest', None, b - cumulative
    cumulative += count
    for i in range(L - 1, -1, -1):
        count = mp_per_level[2 * L - i]
        if b < cumulative + count:
            return 'post', i, b - cumulative
        cumulative += count
    raise IndexError(f"block index {b} out of range for L={L}, mp_per_level={mp_per_level}")


def _build_stage_ops(my_block_indices: List[int], L: int, mp_per_level: List[int]) -> List[tuple]:
    ops = []
    for b in my_block_indices:
        kind, level, local_idx = _block_vcycle_info(b, L, mp_per_level)
        if kind == 'post' and local_idx == 0:
            ops.append(('unpool', level))
        ops.append(('block', kind, level, local_idx))
        if kind == 'pre' and local_idx == mp_per_level[level] - 1:
            ops.append(('save_pool', level))
    return ops


def _compute_out_skip_depth(my_block_indices: List[int], L: int,
                            mp_per_level: List[int], in_skip_depth: int) -> int:
    depth = in_skip_depth
    for b in my_block_indices:
        kind, level, local_idx = _block_vcycle_info(b, L, mp_per_level)
        if kind == 'post' and local_idx == 0:
            depth -= 1
        if kind == 'pre' and local_idx == mp_per_level[level] - 1:
            depth += 1
    return depth


def _compute_in_skip_depth(my_block_indices: List[int], L: int, mp_per_level: List[int]) -> int:
    if not my_block_indices:
        return 0
    first_b = my_block_indices[0]
    depth = 0
    for b in range(first_b):
        kind, level, local_idx = _block_vcycle_info(b, L, mp_per_level)
        if kind == 'post' and local_idx == 0:
            depth -= 1
        if kind == 'pre' and local_idx == mp_per_level[level] - 1:
            depth += 1
    return depth


class _StageInner(nn.Module):
    """Learnable parameters for one pipeline stage."""

    def __init__(
        self,
        *,
        is_first: bool,
        is_last: bool,
        ops_sequence: List[tuple],
        my_blocks: List[int],
        config: dict,
        edge_input_size: int,
        node_input_size: int,
        node_output_size: int,
        latent_dim: int,
        use_world_edges: bool,
        use_multiscale: bool,
        L: int = 0,
        mp_per_level: Optional[List[int]] = None,
    ):
        super().__init__()
        if is_first:
            self.encoder = Encoder(edge_input_size, node_input_size, latent_dim, use_world_edges=use_world_edges)

        if is_last:
            self.decoder = Decoder(latent_dim, node_output_size)

        if not use_multiscale:
            self.processer_list = nn.ModuleDict({
                str(i): GnBlock(latent_dim, use_world_edges=use_world_edges)
                for i in my_blocks
            })
        else:
            assert L > 0 and mp_per_level is not None
            self._build_multiscale_blocks(
                ops_sequence, config, latent_dim,
                edge_input_size, use_world_edges, L,
            )

    def _build_multiscale_blocks(self, ops_sequence, config,
                                 latent_dim, edge_input_size, use_world_edges, L):
        pre_dict: Dict[str, Dict[str, nn.Module]] = {}
        post_dict: Dict[str, Dict[str, nn.Module]] = {}
        coarsest_dict: Dict[str, nn.Module] = {}
        coarse_eb_dict: Dict[str, nn.Module] = {}
        skip_proj_dict: Dict[str, nn.Module] = {}
        unpool_dict: Dict[str, nn.Module] = {}

        use_coarse_we = bool(config.get('coarse_world_edges', False)) and use_world_edges

        for op in ops_sequence:
            if op[0] == 'block':
                _, kind, level, local_idx = op
                if kind == 'pre':
                    lv, li = str(level), str(local_idx)
                    pre_dict.setdefault(lv, {})
                    use_we = use_world_edges if (level == 0 or use_coarse_we) else False
                    pre_dict[lv].setdefault(li, GnBlock(latent_dim, use_world_edges=use_we))
                elif kind == 'coarsest':
                    li = str(local_idx)
                    coarsest_dict.setdefault(
                        li, GnBlock(latent_dim, use_world_edges=use_coarse_we),
                    )
                elif kind == 'post':
                    lv, li = str(level), str(local_idx)
                    post_dict.setdefault(lv, {})
                    use_we = use_world_edges if (level == 0 or use_coarse_we) else False
                    post_dict[lv].setdefault(li, GnBlock(latent_dim, use_world_edges=use_we))
            elif op[0] == 'save_pool':
                lv = str(op[1])
                coarse_eb_dict.setdefault(lv, build_mlp(edge_input_size, latent_dim, latent_dim))
            elif op[0] == 'unpool':
                lv = str(op[1])
                skip_proj_dict.setdefault(lv, nn.Linear(2 * latent_dim, latent_dim))
                from model.blocks import UnpoolBlock
                unpool_dict.setdefault(lv, UnpoolBlock(latent_dim, build_mlp))

        if pre_dict:
            self.pre_blocks = nn.ModuleDict({lv: nn.ModuleDict(blocks) for lv, blocks in pre_dict.items()})
        if coarsest_dict:
            self.coarsest_blocks = nn.ModuleDict(coarsest_dict)
        if post_dict:
            self.post_blocks = nn.ModuleDict({lv: nn.ModuleDict(blocks) for lv, blocks in post_dict.items()})
        if coarse_eb_dict:
            self.coarse_eb_encoders = nn.ModuleDict(coarse_eb_dict)
        if skip_proj_dict:
            self.skip_projs = nn.ModuleDict(skip_proj_dict)
        if unpool_dict:
            self.unpool_blocks = nn.ModuleDict(unpool_dict)


class ModelSplitStage(nn.Module):
    """One deterministic pipeline stage."""

    def __init__(
        self,
        config: dict,
        stage_idx: int,
        num_stages: int,
        assignment: Sequence[Sequence[int]],
        device: torch.device,
    ):
        super().__init__()
        self.config = config
        self.stage_idx = int(stage_idx)
        self.num_stages = int(num_stages)
        self.is_first = self.stage_idx == 0
        self.is_last = self.stage_idx == self.num_stages - 1
        self.device = device

        self.use_multiscale = bool(config.get('use_multiscale', False))
        self.use_world_edges = bool(config.get('use_world_edges', False))
        self.use_coarse_world_edges = (
            bool(config.get('coarse_world_edges', False))
            and self.use_world_edges
            and self.use_multiscale
        )
        self.use_checkpointing = bool(config.get('use_checkpointing', False))

        my_blocks = sorted(assignment[stage_idx])
        self.my_block_indices = my_blocks

        latent_dim = int(config['latent_dim'])
        edge_input_size = int(config['edge_var'])
        if edge_input_size != EDGE_FEATURE_DIM:
            raise ValueError(f"edge_var must be {EDGE_FEATURE_DIM}, got {edge_input_size}")

        node_input_size = int(config['input_var']) + int(config.get('positional_features', 0))
        if config.get('use_node_types', False) and int(config.get('num_node_types', 0)) > 0:
            node_input_size += int(config['num_node_types'])

        L = 0
        mp_per_level: List[int] = []
        ops_sequence: List[tuple] = []
        self._in_skip_depth = 0
        self._out_skip_depth = 0

        if self.use_multiscale:
            L = int(config.get('multiscale_levels', 1))
            mp_per_level = _parse_mp_per_level(config, L)
            ops_sequence = _build_stage_ops(my_blocks, L, mp_per_level)
            self._in_skip_depth = _compute_in_skip_depth(my_blocks, L, mp_per_level)
            self._out_skip_depth = _compute_out_skip_depth(my_blocks, L, mp_per_level, self._in_skip_depth)
        self._ops_sequence = ops_sequence

        self.model = _StageInner(
            is_first=self.is_first,
            is_last=self.is_last,
            ops_sequence=ops_sequence,
            my_blocks=my_blocks,
            config=config,
            edge_input_size=edge_input_size,
            node_input_size=node_input_size,
            node_output_size=int(config['output_var']),
            latent_dim=latent_dim,
            use_world_edges=self.use_world_edges,
            use_multiscale=self.use_multiscale,
            L=L,
            mp_per_level=mp_per_level if self.use_multiscale else None,
        )

        self.to(device)

        num_timesteps = config.get('num_timesteps', None)
        if (num_timesteps is None or num_timesteps > 1) and self.is_last:
            with torch.no_grad():
                self.model.decoder.decode_module[-1].weight.mul_(0.01)

    def send_to_next(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        skip_stack: List[dict],
        world_edge_attr: Optional[torch.Tensor],
        world_edge_index: Optional[torch.Tensor],
        current_level_idx: int,
    ) -> torch.Tensor:
        grad_t, nongd_t = _pack_bundle(
            x, edge_attr, edge_index, skip_stack,
            world_edge_attr, world_edge_index, current_level_idx,
            self.use_world_edges, self.use_multiscale,
            self.use_coarse_world_edges,
        )
        n_grad, _ = _bundle_counts(
            self.use_world_edges, len(skip_stack), self.use_multiscale,
            self.use_coarse_world_edges,
        )
        return BundleSend.apply(self.stage_idx + 1, n_grad, *grad_t, *nongd_t)

    def recv_from_prev(self) -> tuple:
        src = self.stage_idx - 1
        n_skips = self._in_skip_depth
        n_grad, n_nongd = _bundle_counts(
            self.use_world_edges, n_skips, self.use_multiscale,
            self.use_coarse_world_edges,
        )
        anchor = torch.zeros((), device=self.device, requires_grad=True)
        all_tensors = BundleRecv.apply(src, n_grad, n_nongd, self.device, anchor)
        return _unpack_bundle_indexed(
            all_tensors, n_skips,
            self.use_world_edges, self.use_multiscale,
            self.use_coarse_world_edges,
        )

    def apply_input_noise(self, graph) -> None:
        noise_std = self.config.get('std_noise', 0.0)
        if noise_std <= 0:
            return
        output_var = int(self.config['output_var'])
        noise = torch.randn(graph.x.shape[0], output_var,
                            device=graph.x.device, dtype=graph.x.dtype) * noise_std
        noise_padded = torch.zeros_like(graph.x)
        noise_padded[:, :output_var] = noise
        graph.x = graph.x + noise_padded
        noise_gamma = self.config.get('noise_gamma', 0.1)
        noise_std_ratio = self.config.get('noise_std_ratio', None)
        if noise_std_ratio is not None:
            ratio = torch.tensor(noise_std_ratio, device=graph.x.device, dtype=graph.x.dtype)
            graph.y = graph.y - noise_gamma * noise * ratio
        graph.edge_attr = graph.edge_attr + torch.randn_like(graph.edge_attr) * noise_std

    def _ckpt_enabled(self) -> bool:
        """Gradient checkpointing is active only while training with the flag on."""
        return self.use_checkpointing and self.training

    def encode(self, graph) -> Tuple:
        if not self.is_first:
            raise RuntimeError("encode() called on non-first stage")
        encoded = run_checkpointed(
            self.model.encoder, graph, enabled=self._ckpt_enabled(),
        )
        wea = getattr(encoded, 'world_edge_attr', None) if self.use_world_edges else None
        wei = getattr(encoded, 'world_edge_index', None) if self.use_world_edges else None
        return encoded.x, encoded.edge_attr, encoded.edge_index, wea, wei

    def run_local_blocks_flat(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        world_edge_attr: Optional[torch.Tensor] = None,
        world_edge_index: Optional[torch.Tensor] = None,
    ) -> Tuple:
        blocks = [self.model.processer_list[str(i)] for i in self.my_block_indices]
        if self._ckpt_enabled():
            x, edge_attr, world_edge_attr = process_with_checkpointing(
                blocks, x, edge_attr, edge_index, world_edge_attr, world_edge_index,
            )
        else:
            for block in blocks:
                x, edge_attr, world_edge_attr = block.forward_tensors(
                    x, edge_attr, edge_index, world_edge_attr, world_edge_index,
                )

        return x, edge_attr, edge_index, world_edge_attr, world_edge_index

    def _run_block(self, block, g) -> Data:
        """Run one GnBlock on a Data, optionally gradient-checkpointed.

        Mirrors the serial path's tensor fast path so use_checkpointing recomputes
        the block internals in backward instead of holding them resident.
        """
        x, edge_attr = g.x, g.edge_attr
        edge_index = g.edge_index
        wea = getattr(g, 'world_edge_attr', None)
        wei = getattr(g, 'world_edge_index', None)
        if self._ckpt_enabled():
            x, edge_attr, wea = checkpoint_gn_block(block, x, edge_attr, edge_index, wea, wei)
        else:
            x, edge_attr, wea = block.forward_tensors(x, edge_attr, edge_index, wea, wei)
        out = Data(x=x, edge_attr=edge_attr, edge_index=edge_index)
        if wea is not None and wei is not None:
            out.world_edge_attr = wea
            out.world_edge_index = wei
        return out

    def _unpool_merge(self, unpool_level, coarse_x, skip_x, ld):
        """Unpool level coarse features to fine and merge with the skip state.

        Bundled into one method (mirrors MeshGraphNets._unpool_merge_level) so
        use_checkpointing can recompute the whole unpool + skip-projection step,
        whose bipartite edge MLP is one of the largest saved buffers in the V-cycle.
        """
        up_ei = ld['up_ei']
        rel_pos = ld['fine_pos'][up_ei[1]] - ld['coarse_centroid'][up_ei[0]]
        h_up = self.model.unpool_blocks[str(unpool_level)](
            h_coarse=coarse_x,
            h_fine_skip=skip_x,
            unpool_edge_index=up_ei,
            rel_pos=rel_pos,
        )
        return self.model.skip_projs[str(unpool_level)](torch.cat([skip_x, h_up], dim=-1))

    def _extract_level_data(self, graph, level: int) -> dict:
        ld = {
            'ftc': graph[f'fine_to_coarse_{level}'],
            'c_ei': graph[f'coarse_edge_index_{level}'],
            'c_ea': graph[f'coarse_edge_attr_{level}'],
            'n_c': int(graph[f'num_coarse_{level}'].sum()),
            'c_we_idx': getattr(graph, f'coarse_world_edge_index_{level}', None),
            'c_we_attr': getattr(graph, f'coarse_world_edge_attr_{level}', None),
        }
        # Inherit-mode (voronoi_inherit) levels expose seed indices.
        seed_key = f'coarse_seed_idx_{level}'
        if hasattr(graph, seed_key):
            ld['seeds'] = graph[seed_key]
        ld['up_ei'] = graph[f'unpool_edge_index_{level}']
        ld['coarse_centroid'] = getattr(graph, f'coarse_centroid_{level}', None)
        ld['fine_pos'] = graph.pos if level == 0 else getattr(graph, f'coarse_centroid_{level - 1}', None)
        return ld

    def run_local_blocks_multiscale(
        self,
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        skip_stack: List[dict],
        world_edge_attr: Optional[torch.Tensor],
        world_edge_index: Optional[torch.Tensor],
        current_level_idx: int,
        graph,
    ) -> Tuple:
        current_graph = Data(x=x, edge_attr=edge_attr, edge_index=edge_index)
        if self.use_world_edges and world_edge_attr is not None:
            current_graph.world_edge_attr = world_edge_attr
            current_graph.world_edge_index = world_edge_index

        level_idx = current_level_idx

        for op in self._ops_sequence:
            if op[0] == 'block':
                _, kind, level, local_idx = op
                lv, li = str(level), str(local_idx)
                if kind == 'pre':
                    current_graph = self._run_block(self.model.pre_blocks[lv][li], current_graph)
                elif kind == 'coarsest':
                    current_graph = self._run_block(self.model.coarsest_blocks[li], current_graph)
                else:
                    current_graph = self._run_block(self.model.post_blocks[lv][li], current_graph)

            elif op[0] == 'save_pool':
                pool_level = op[1]
                ld = self._extract_level_data(graph, pool_level)
                use_we_here = self.use_world_edges and (pool_level == 0 or self.use_coarse_world_edges)
                skip_stack.append({
                    'x': current_graph.x,
                    'edge_attr': current_graph.edge_attr,
                    'edge_index': current_graph.edge_index,
                    'w_attr': getattr(current_graph, 'world_edge_attr', None) if use_we_here else None,
                    'w_idx': getattr(current_graph, 'world_edge_index', None) if use_we_here else None,
                })
                # Inherit mode: gather seed features. Centroid mode: scatter-mean pool.
                if 'seeds' in ld:
                    h_coarse = current_graph.x[ld['seeds']]
                else:
                    h_coarse = pool_features(current_graph.x, ld['ftc'], ld['n_c'])
                e_coarse = run_checkpointed(
                    self.model.coarse_eb_encoders[str(pool_level)], ld['c_ea'],
                    enabled=self._ckpt_enabled(),
                )
                current_graph = Data(x=h_coarse, edge_attr=e_coarse, edge_index=ld['c_ei'])
                if self.use_coarse_world_edges:
                    c_we_idx = ld.get('c_we_idx')
                    if c_we_idx is not None and c_we_idx.shape[1] > 0:
                        current_graph.world_edge_attr = ld['c_we_attr']
                        current_graph.world_edge_index = c_we_idx
                level_idx += 1

            elif op[0] == 'unpool':
                unpool_level = op[1]
                ld = self._extract_level_data(graph, unpool_level)
                skip = skip_stack[-1]

                h_merged = run_checkpointed(
                    self._unpool_merge, unpool_level, current_graph.x, skip['x'], ld,
                    enabled=self._ckpt_enabled(),
                )
                current_graph = Data(x=h_merged, edge_attr=skip['edge_attr'], edge_index=skip['edge_index'])
                use_we_here = self.use_world_edges and (unpool_level == 0 or self.use_coarse_world_edges)
                if use_we_here and skip.get('w_attr') is not None:
                    current_graph.world_edge_attr = skip['w_attr']
                    current_graph.world_edge_index = skip['w_idx']
                skip_stack.pop()
                level_idx -= 1

        return (
            current_graph.x,
            current_graph.edge_attr,
            current_graph.edge_index,
            skip_stack,
            getattr(current_graph, 'world_edge_attr', None),
            getattr(current_graph, 'world_edge_index', None),
            level_idx,
        )

    def decode(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if not self.is_last:
            raise RuntimeError("decode() called on non-last stage")
        graph = Data(x=x, edge_attr=edge_attr, edge_index=edge_index)
        return self.model.decoder(graph)


def build_stage(
    config: dict,
    stage_idx: int,
    num_stages: int,
    assignment: Sequence[Sequence[int]],
    device: torch.device,
) -> ModelSplitStage:
    return ModelSplitStage(config, stage_idx, num_stages, assignment, device)
