import torch
import torch.nn as nn
from torch_geometric.utils import scatter
from torch_geometric.data import Data


class EdgeBlock(nn.Module):

    def __init__(self, custom_func:nn.Module):

        super(EdgeBlock, self).__init__()
        self.net = custom_func

    def compute(self, x, edge_attr, edge_index):
        """Tensor fast path: update edge features from sender/receiver nodes."""
        senders_idx, receivers_idx = edge_index
        collected_edges = torch.cat([x[senders_idx], x[receivers_idx], edge_attr], dim=1)
        return self.net(collected_edges)

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
        stresses from neighbors should add up, not average.
        """
        _, receivers_idx = edge_index
        agg_received_edges = scatter(edge_attr, receivers_idx, dim=0, dim_size=num_nodes, reduce='sum')
        collected_nodes = torch.cat([x, agg_received_edges], dim=-1)
        return self.net(collected_nodes)

    def forward(self, graph):
        x = self.compute(graph.x, graph.edge_attr, graph.edge_index, graph.num_nodes)
        return Data(x=x, edge_attr=graph.edge_attr, edge_index=graph.edge_index)

class HybridNodeBlock(nn.Module):
    """Node block that aggregates from both mesh and world edges."""

    def __init__(self, custom_func: nn.Module):
        super(HybridNodeBlock, self).__init__()
        self.net = custom_func

    def compute(self, x, edge_attr, edge_index, world_edge_attr, world_edge_index, num_nodes):
        """Tensor fast path: separate sum aggregation over mesh and world edges."""
        _, mesh_receivers = edge_index
        mesh_agg = scatter(edge_attr, mesh_receivers, dim=0, dim_size=num_nodes, reduce='sum')

        if (world_edge_attr is not None and world_edge_index is not None
                and world_edge_index.shape[1] > 0):
            _, world_receivers = world_edge_index
            world_agg = scatter(world_edge_attr, world_receivers, dim=0, dim_size=num_nodes, reduce='sum')
        else:
            world_agg = torch.zeros_like(mesh_agg)

        collected_nodes = torch.cat([x, mesh_agg, world_agg], dim=-1)
        return self.net(collected_nodes)

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
        """
        src_coarse, dst_fine = unpool_edge_index

        edge_input = torch.cat([
            h_coarse[src_coarse],
            h_fine_skip[dst_fine],
            rel_pos,
        ], dim=-1)
        messages = self.edge_mlp(edge_input)

        agg = scatter(messages, dst_fine, dim=0,
                      dim_size=h_fine_skip.shape[0], reduce='sum')

        h_up = self.node_mlp(torch.cat([h_fine_skip, agg], dim=-1))
        return h_up
