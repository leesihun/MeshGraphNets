import torch
import torch.nn as nn
from torch_geometric.data import Data

from model.blocks import EdgeBlock, HybridNodeBlock, NodeBlock
from model.mlp import build_mlp


class Encoder(nn.Module):

    def __init__(self, edge_input_size, node_input_size, latent_dim, use_world_edges=False):
        super().__init__()
        self.use_world_edges = use_world_edges
        self.eb_encoder = build_mlp(edge_input_size, latent_dim, latent_dim)
        self.nb_encoder = build_mlp(node_input_size, latent_dim, latent_dim)
        if use_world_edges:
            self.world_eb_encoder = build_mlp(edge_input_size, latent_dim, latent_dim)

    def forward(self, graph):
        node_ = self.nb_encoder(graph.x)
        edge_ = self.eb_encoder(graph.edge_attr)
        out = Data(x=node_, edge_attr=edge_, edge_index=graph.edge_index)
        if self.use_world_edges and hasattr(graph, 'world_edge_attr') and graph.world_edge_index.shape[1] > 0:
            out.world_edge_attr = self.world_eb_encoder(graph.world_edge_attr)
            out.world_edge_index = graph.world_edge_index
        elif self.use_world_edges:
            out.world_edge_attr = torch.zeros(0, edge_.shape[1], device=edge_.device)
            out.world_edge_index = torch.zeros(2, 0, dtype=torch.long, device=edge_.device)
        return out


class GnBlock(nn.Module):

    def __init__(self, config, latent_dim, use_world_edges=False):
        super().__init__()
        self.use_world_edges = use_world_edges
        self.residual_scale = config.get('residual_scale', 1.0)
        self.use_pairnorm = config.get('use_pairnorm', False)

        eb_input_dim = 3 * latent_dim
        self.eb_module = EdgeBlock(custom_func=build_mlp(eb_input_dim, latent_dim, latent_dim))

        if use_world_edges:
            self.world_eb_module = EdgeBlock(custom_func=build_mlp(eb_input_dim, latent_dim, latent_dim))
            self.nb_module = HybridNodeBlock(custom_func=build_mlp(3 * latent_dim, latent_dim, latent_dim))
        else:
            self.nb_module = NodeBlock(custom_func=build_mlp(2 * latent_dim, latent_dim, latent_dim))

    def forward_tensors(self, x, edge_attr, edge_index, world_edge_attr=None, world_edge_index=None):
        """Tensor fast path used by the training/inference hot loop.

        Avoids constructing PyG Data objects per block (Python overhead and
        torch.compile graph breaks). Returns (x, edge_attr, world_edge_attr);
        edge_index and world_edge_index are unchanged by the block.
        """
        num_nodes = x.shape[0]
        has_world = (
            self.use_world_edges and world_edge_attr is not None
            and world_edge_index is not None and world_edge_attr.shape[0] > 0
        )

        edge_mlp_out = self.eb_module.compute(x, edge_attr, edge_index)

        world_edge_mlp_out = None
        if has_world:
            world_edge_mlp_out = self.world_eb_module.compute(x, world_edge_attr, world_edge_index)

        if self.use_world_edges:
            x_update = self.nb_module.compute(
                x, edge_mlp_out, edge_index,
                world_edge_mlp_out, world_edge_index if has_world else None,
                num_nodes,
            )
        else:
            x_update = self.nb_module.compute(x, edge_mlp_out, edge_index, num_nodes)

        x_out = x + self.residual_scale * x_update
        if self.use_pairnorm:
            x_centered = x_out - x_out.mean(dim=0, keepdim=True)
            rms = (x_centered.norm(p=2) / (x_out.shape[0] ** 0.5)) + 1e-8
            x_out = x_centered / rms

        edge_attr_out = edge_attr + self.residual_scale * edge_mlp_out
        world_edge_attr_out = (
            (world_edge_attr + self.residual_scale * world_edge_mlp_out)
            if world_edge_mlp_out is not None else world_edge_attr
        )
        return x_out, edge_attr_out, world_edge_attr_out

    def forward(self, graph):
        world_edge_index = graph.world_edge_index if self.use_world_edges and hasattr(graph, 'world_edge_index') else None
        world_edge_attr = graph.world_edge_attr if self.use_world_edges and hasattr(graph, 'world_edge_attr') else None

        x, edge_attr, world_edge_attr = self.forward_tensors(
            graph.x, graph.edge_attr, graph.edge_index, world_edge_attr, world_edge_index
        )

        out = Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)
        if self.use_world_edges:
            out.world_edge_attr = (
                world_edge_attr if world_edge_attr is not None
                else torch.zeros(0, edge_attr.shape[1], device=x.device)
            )
            out.world_edge_index = (
                world_edge_index if world_edge_index is not None
                else torch.zeros(2, 0, dtype=torch.long, device=x.device)
            )
        return out


class Decoder(nn.Module):

    def __init__(self, latent_dim, node_output_size):
        super().__init__()
        self.decode_module = build_mlp(latent_dim, latent_dim, node_output_size, layer_norm=False)

    def forward(self, graph_or_x):
        x = graph_or_x.x if hasattr(graph_or_x, 'x') else graph_or_x
        return self.decode_module(x)
