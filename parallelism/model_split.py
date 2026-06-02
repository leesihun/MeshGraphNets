"""Pipeline stage implementation for deterministic model-split MeshGraphNets."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch_geometric.data import Data

from general_modules.edge_features import EDGE_FEATURE_DIM
from model.coarsening import pool_features, unpool_features
from model.encoder_decoder import Decoder, Encoder, GnBlock
from model.mlp import build_mlp


_DTYPE_TO_CODE = {
    torch.float32: 0, torch.float64: 1, torch.bfloat16: 2, torch.float16: 3,
    torch.int64: 4, torch.int32: 5, torch.bool: 6,
}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}


def _send_tensor(t: torch.Tensor, dst: int) -> None:
    dtype_code = _DTYPE_TO_CODE[t.dtype]
    shape = list(t.shape)
    header = torch.tensor([len(shape), *shape, dtype_code], dtype=torch.long, device=t.device)
    header_len = torch.tensor([header.numel()], dtype=torch.long, device=t.device)
    dist.send(header_len, dst=dst)
    dist.send(header, dst=dst)
    dist.send(t.contiguous(), dst=dst)


def _recv_tensor(src: int, device: torch.device) -> torch.Tensor:
    header_len = torch.empty(1, dtype=torch.long, device=device)
    dist.recv(header_len, src=src)
    header = torch.empty(int(header_len.item()), dtype=torch.long, device=device)
    dist.recv(header, src=src)
    ndim = int(header[0].item())
    shape = [int(header[1 + i].item()) for i in range(ndim)]
    dtype = _CODE_TO_DTYPE[int(header[1 + ndim].item())]
    t = torch.empty(*shape, dtype=dtype, device=device)
    dist.recv(t, src=src)
    return t


class BundleSend(torch.autograd.Function):
    """Send grad-bearing and non-differentiable tensors downstream."""

    @staticmethod
    def forward(ctx, dst: int, n_grad: int, *all_tensors: torch.Tensor) -> torch.Tensor:
        ctx.dst = dst
        ctx.n_grad = n_grad
        ctx.n_nongd = len(all_tensors) - n_grad
        ctx.device = all_tensors[0].device
        if ctx.n_nongd > 0:
            ctx.mark_non_differentiable(*all_tensors[n_grad:])
        for t in all_tensors:
            _send_tensor(t.contiguous(), dst=dst)
        return torch.zeros((), device=ctx.device, dtype=all_tensors[0].dtype, requires_grad=True)

    @staticmethod
    def backward(ctx, _grad_sentinel):
        grads = [_recv_tensor(src=ctx.dst, device=ctx.device) for _ in range(ctx.n_grad)]
        return (None, None) + tuple(grads) + (None,) * ctx.n_nongd


class BundleRecv(torch.autograd.Function):
    """Receive a tensor bundle from upstream and send gradients back in backward."""

    @staticmethod
    def forward(ctx, src: int, n_grad: int, n_nongd: int,
                device: torch.device, anchor: torch.Tensor):
        ctx.src = src
        ctx.n_grad = n_grad
        tensors = [_recv_tensor(src=src, device=device) for _ in range(n_grad + n_nongd)]
        if n_nongd > 0:
            ctx.mark_non_differentiable(*tensors[n_grad:])
        return tuple(tensors)

    @staticmethod
    def backward(ctx, *grads):
        for g in grads[:ctx.n_grad]:
            _send_tensor(g.contiguous(), dst=ctx.src)
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
        mp = [
            int(config.get('fine_mp_pre', 5)),
            int(config.get('coarse_mp_num', 5)),
            int(config.get('fine_mp_post', 5)),
        ]
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
        coarse_config = dict(config)
        coarse_config['use_world_edges'] = False

        if is_first:
            self.encoder = Encoder(edge_input_size, node_input_size, latent_dim, use_world_edges=use_world_edges)

        if is_last:
            self.decoder = Decoder(latent_dim, node_output_size)

        if not use_multiscale:
            self.processer_list = nn.ModuleDict({
                str(i): GnBlock(config, latent_dim, use_world_edges=use_world_edges)
                for i in my_blocks
            })
        else:
            assert L > 0 and mp_per_level is not None
            self._build_multiscale_blocks(
                ops_sequence, config, coarse_config, latent_dim,
                edge_input_size, use_world_edges, L,
            )

    def _build_multiscale_blocks(self, ops_sequence, config, coarse_config,
                                 latent_dim, edge_input_size, use_world_edges, L):
        pre_dict: Dict[str, Dict[str, nn.Module]] = {}
        post_dict: Dict[str, Dict[str, nn.Module]] = {}
        coarsest_dict: Dict[str, nn.Module] = {}
        coarse_eb_dict: Dict[str, nn.Module] = {}
        skip_proj_dict: Dict[str, nn.Module] = {}
        unpool_dict: Dict[str, nn.Module] = {}

        bipartite_unpool = bool(config.get('bipartite_unpool', False))
        use_coarse_we = bool(config.get('coarse_world_edges', False)) and use_world_edges

        for op in ops_sequence:
            if op[0] == 'block':
                _, kind, level, local_idx = op
                if kind == 'pre':
                    lv, li = str(level), str(local_idx)
                    pre_dict.setdefault(lv, {})
                    use_we = use_world_edges if (level == 0 or use_coarse_we) else False
                    pre_dict[lv].setdefault(li, GnBlock(config if use_we else coarse_config, latent_dim, use_world_edges=use_we))
                elif kind == 'coarsest':
                    li = str(local_idx)
                    coarsest_dict.setdefault(
                        li,
                        GnBlock(config if use_coarse_we else coarse_config, latent_dim, use_world_edges=use_coarse_we),
                    )
                elif kind == 'post':
                    lv, li = str(level), str(local_idx)
                    post_dict.setdefault(lv, {})
                    use_we = use_world_edges if (level == 0 or use_coarse_we) else False
                    post_dict[lv].setdefault(li, GnBlock(config if use_we else coarse_config, latent_dim, use_world_edges=use_we))
            elif op[0] == 'save_pool':
                lv = str(op[1])
                coarse_eb_dict.setdefault(lv, build_mlp(edge_input_size, latent_dim, latent_dim))
            elif op[0] == 'unpool':
                lv = str(op[1])
                skip_proj_dict.setdefault(lv, nn.Linear(2 * latent_dim, latent_dim))
                if bipartite_unpool:
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

    def encode(self, graph) -> Tuple:
        if not self.is_first:
            raise RuntimeError("encode() called on non-first stage")
        encoded = self.model.encoder(graph)
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
        graph = Data(x=x, edge_attr=edge_attr, edge_index=edge_index)
        if self.use_world_edges and world_edge_attr is not None:
            graph.world_edge_attr = world_edge_attr
            graph.world_edge_index = world_edge_index

        for i in self.my_block_indices:
            graph = self.model.processer_list[str(i)](graph)

        return (
            graph.x,
            graph.edge_attr,
            graph.edge_index,
            getattr(graph, 'world_edge_attr', None),
            getattr(graph, 'world_edge_index', None),
        )

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
        if bool(self.config.get('bipartite_unpool', False)):
            up_ei = getattr(graph, f'unpool_edge_index_{level}', None)
            if up_ei is not None:
                ld['up_ei'] = up_ei
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

        bipartite_unpool = bool(self.config.get('bipartite_unpool', False))
        level_idx = current_level_idx

        for op in self._ops_sequence:
            if op[0] == 'block':
                _, kind, level, local_idx = op
                lv, li = str(level), str(local_idx)
                if kind == 'pre':
                    current_graph = self.model.pre_blocks[lv][li](current_graph)
                elif kind == 'coarsest':
                    current_graph = self.model.coarsest_blocks[li](current_graph)
                else:
                    current_graph = self.model.post_blocks[lv][li](current_graph)

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
                e_coarse = self.model.coarse_eb_encoders[str(pool_level)](ld['c_ea'])
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

                up_ei = ld.get('up_ei')
                if (bipartite_unpool and hasattr(self.model, 'unpool_blocks')
                        and up_ei is not None
                        and ld.get('coarse_centroid') is not None
                        and ld.get('fine_pos') is not None):
                    rel_pos = ld['fine_pos'][up_ei[1]] - ld['coarse_centroid'][up_ei[0]]
                    h_up = self.model.unpool_blocks[str(unpool_level)](
                        h_coarse=current_graph.x,
                        h_fine_skip=skip['x'],
                        unpool_edge_index=up_ei,
                        rel_pos=rel_pos,
                    )
                else:
                    h_up = unpool_features(current_graph.x, ld['ftc'])

                h_merged = self.model.skip_projs[str(unpool_level)](torch.cat([skip['x'], h_up], dim=-1))
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
