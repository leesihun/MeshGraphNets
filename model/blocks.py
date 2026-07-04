import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch_geometric.data import Data


def _split_first_linear(net, parts):
    """Apply net's first Linear without materializing the input concatenation.

    Equivalent to ``net[0](torch.cat([t if idx is None else t[idx] for t, idx
    in parts], dim=-1))`` but projects each part through its column block of
    the weight and gathers afterwards. Autograd then saves only the per-part
    inputs (already alive) and the summed output instead of an [E, sum(dims)]
    concat copy, and node-sized parts are projected on N rows instead of E.
    Assumes net is a build_mlp Sequential whose first layer is nn.Linear.

    Args:
        parts: sequence of (tensor, index_or_None). Column blocks are consumed
               left to right; trailing weight columns may stay unused (their
               input is implicitly zero, e.g. absent world edges).
    """
    first = net[0]
    weight = first.weight
    out = None
    col = 0
    for tensor, index in parts:
        width = tensor.shape[-1]
        # Bias rides on the first part only; per-edge gather preserves it.
        bias = first.bias if out is None else None
        proj = F.linear(tensor, weight[:, col:col + width], bias)
        if index is not None:
            proj = proj[index]
        out = proj if out is None else out + proj
        col += width
    return out


def _run_mlp_tail(net, h):
    """Run the layers after the first Linear (see _split_first_linear)."""
    for i in range(1, len(net)):
        h = net[i](h)
    return h


class EdgeBlock(nn.Module):

    def __init__(self, custom_func:nn.Module):

        super(EdgeBlock, self).__init__()
        self.net = custom_func

    def compute(self, x, edge_attr, edge_index):
        """Tensor fast path: update edge features from sender/receiver nodes.

        The first Linear runs in split form so the [E, 3*latent] concat is
        never built and the two node projections run on N rows instead of E.
        """
        senders_idx, receivers_idx = edge_index
        h = _split_first_linear(self.net, [
            (x, senders_idx),
            (x, receivers_idx),
            (edge_attr, None),
        ])
        return _run_mlp_tail(self.net, h)

    def forward(self, graph):
        edge_attr = self.compute(graph.x, graph.edge_attr, graph.edge_index)
        return Data(x=graph.x, edge_attr=edge_attr, edge_index=graph.edge_index)


class NodeBlock(nn.Module):

    def __init__(self, custom_func:nn.Module):
        super(NodeBlock, self).__init__()
        self.net = custom_func

    def compute(self, x, edge_attr, edge_index, num_nodes):
        """Tensor fast path: update node features from aggregated edges.

        Sum aggregation (matches NVIDIA PhysicsNeMo deforming_plate): forces and
        stresses from neighbors should add up, not average. The first Linear
        runs in split form to skip the [N, 2*latent] concat.
        """
        _, receivers_idx = edge_index
        agg_received_edges = scatter(edge_attr, receivers_idx, dim=0, dim_size=num_nodes, reduce='sum')
        h = _split_first_linear(self.net, [(x, None), (agg_received_edges, None)])
        return _run_mlp_tail(self.net, h)

    def forward(self, graph):
        x = self.compute(graph.x, graph.edge_attr, graph.edge_index, graph.num_nodes)
        return Data(x=x, edge_attr=graph.edge_attr, edge_index=graph.edge_index)

class HybridNodeBlock(nn.Module):
    """Node block that aggregates from both mesh and world edges."""

    def __init__(self, custom_func: nn.Module):
        super(HybridNodeBlock, self).__init__()
        self.net = custom_func

    def compute(self, x, edge_attr, edge_index, world_edge_attr, world_edge_index, num_nodes):
        """Tensor fast path: separate sum aggregation over mesh and world edges.

        With no world edges the world column block of the first Linear is
        simply skipped — identical to projecting an all-zero aggregate.
        """
        _, mesh_receivers = edge_index
        mesh_agg = scatter(edge_attr, mesh_receivers, dim=0, dim_size=num_nodes, reduce='sum')

        parts = [(x, None), (mesh_agg, None)]
        if (world_edge_attr is not None and world_edge_index is not None
                and world_edge_index.shape[1] > 0):
            _, world_receivers = world_edge_index
            world_agg = scatter(world_edge_attr, world_receivers, dim=0, dim_size=num_nodes, reduce='sum')
            parts.append((world_agg, None))

        h = _split_first_linear(self.net, parts)
        return _run_mlp_tail(self.net, h)

    def forward(self, graph):
        world_edge_attr = graph.world_edge_attr if hasattr(graph, 'world_edge_attr') else None
        world_edge_index = graph.world_edge_index if hasattr(graph, 'world_edge_index') else None
        x = self.compute(
            graph.x, graph.edge_attr, graph.edge_index,
            world_edge_attr, world_edge_index, graph.num_nodes,
        )
        return Data(
            x=x,
            edge_attr=graph.edge_attr,
            edge_index=graph.edge_index,
            world_edge_attr=world_edge_attr,
            world_edge_index=world_edge_index
        )


class UnpoolBlock(nn.Module):
    """Bipartite message passing from coarse to fine nodes (learned unpool)."""

    def __init__(self, latent_dim: int, build_mlp_fn):
        super().__init__()
        # EdgeMLP: (h_coarse, h_fine_skip, rel_pos) → message
        self.edge_mlp = build_mlp_fn(2 * latent_dim + 3, latent_dim, latent_dim)
        # NodeMLP: (h_fine_skip, aggregated_messages) → h_up
        self.node_mlp = build_mlp_fn(2 * latent_dim, latent_dim, latent_dim)

    def forward(self, h_coarse, h_fine_skip, unpool_edge_index, rel_pos):
        """
        Args:
            h_coarse:          [M, D] coarse node features
            h_fine_skip:       [N, D] fine node skip features (from descending arm)
            unpool_edge_index: [2, E_up] row0=coarse src, row1=fine dst
            rel_pos:           [E_up, 3] relative position per edge
        Returns:
            h_up: [N, D] unpooled fine node features

        E_up is ~(1 + coarse degree) * N, so the split-form first Linear
        matters most here: it avoids an [E_up, 2D+3] concat (which autocast
        would additionally promote to fp32 because rel_pos stays fp32) and
        runs the coarse/fine projections on M and N rows instead of E_up.
        """
        src_coarse, dst_fine = unpool_edge_index

        h = _split_first_linear(self.edge_mlp, [
            (h_coarse, src_coarse),
            (h_fine_skip, dst_fine),
            (rel_pos, None),
        ])
        messages = _run_mlp_tail(self.edge_mlp, h)

        agg = scatter(messages, dst_fine, dim=0,
                      dim_size=h_fine_skip.shape[0], reduce='sum')

        h_up = _split_first_linear(self.node_mlp, [(h_fine_skip, None), (agg, None)])
        return _run_mlp_tail(self.node_mlp, h_up)
