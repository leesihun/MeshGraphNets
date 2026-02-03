import torch.nn.init as init

import torch.nn as nn
import torch
from torch_geometric.data import Data
from model.blocks import EdgeBlock, NodeBlock, HybridNodeBlock
from model.checkpointing import process_with_checkpointing

def init_weights(m):
    if isinstance(m, nn.Linear):
        # Xavier works better for SiLU/Swish since it doesn't assume ReLU zeroing
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


class MeshGraphNets(nn.Module):
    def __init__(self, config, device: str):
        super(MeshGraphNets, self).__init__()
        self.config = config
        self.device = device

        self.model = EncoderProcessorDecoder(config).to(device)
        self.model.apply(init_weights)
        print('MeshGraphNets model created successfully')

    def set_checkpointing(self, enabled: bool):
        """Enable or disable gradient checkpointing."""
        self.model.set_checkpointing(enabled)

    def forward(self, graph, debug=False):
        """
        Forward pass of the simulator.

        Expects pre-normalized inputs from the dataloader:
            - graph.x: normalized node features [N, input_var]
            - graph.edge_attr: normalized edge features [E, edge_var]
            - graph.y: normalized target delta (y_t+1 - x_t) [N, output_var]

        Returns:
            predicted: predicted normalized delta [N, output_var]
            target: normalized target delta [N, output_var]
        """
        # Optional noise injection during training
        if self.training:
            noise_std = self.config.get('std_noise', 0.0)
            if noise_std > 0:
                noise = torch.randn_like(graph.x) * noise_std
                graph.x = graph.x + noise

        # Forward through encoder-processor-decoder
        predicted = self.model(graph, debug=debug)

        return predicted, graph.y

class EncoderProcessorDecoder(nn.Module):
    def __init__(self, config):
        super(EncoderProcessorDecoder, self).__init__()
        self.config = config

        self.message_passing_num = config['message_passing_num']
        self.edge_input_size = config['edge_var']
        self.latent_dim = config['latent_dim']
        self.use_checkpointing = config.get('use_checkpointing', False)
        self.use_world_edges = config.get('use_world_edges', False)

        # Compute actual node input size (physical features + optional node types)
        base_input_size = config['input_var']
        use_node_types = config.get('use_node_types', False)
        num_node_types = config.get('num_node_types', 0)
        if use_node_types and num_node_types > 0:
            self.node_input_size = base_input_size + num_node_types
            print(f"  Model input: {base_input_size} physical + {num_node_types} node types = {self.node_input_size}")
        else:
            self.node_input_size = base_input_size

        # Output is always physical features only (node types don't change)
        self.node_output_size = config['output_var']

        self.encoder = Encoder(
            self.edge_input_size, self.node_input_size, self.latent_dim,
            use_world_edges=self.use_world_edges
        )

        processer_list = []
        for _ in range(self.message_passing_num):
            processer_list.append(GnBlock(config, self.latent_dim, use_world_edges=self.use_world_edges))
        self.processer_list = nn.ModuleList(processer_list)

        self.decoder = Decoder(self.latent_dim, self.node_output_size)

    def forward(self, graph, debug=False):
        graph = self.encoder(graph)

        if debug:
            print(f"  After Encoder: x std={graph.x.std().item():.4f}, mean={graph.x.mean().item():.4f}")

        if self.use_checkpointing and self.training:
            graph = process_with_checkpointing(self.processer_list, graph)
        else:
            for i, model in enumerate(self.processer_list):
                graph = model(graph)
                if debug and i == len(self.processer_list) - 1:
                    print(f"  After MP block {i}: x std={graph.x.std().item():.4f}, mean={graph.x.mean().item():.4f}")

        output = self.decoder(graph)
        if debug:
            print(f"  After Decoder: out std={output.std().item():.4f}, mean={output.mean().item():.4f}")
        return output

    def set_checkpointing(self, enabled: bool):
        """Enable or disable gradient checkpointing."""
        self.use_checkpointing = enabled

def build_mlp(in_size, hidden_size, out_size, layer_norm=True, activation='silu', decoder=False):
    """
    Build a multi-layer perceptron with Pre-LayerNorm architecture.

    Pre-LN (normalizing inputs) provides better gradient flow for deep networks:
    - Prevents residual path from dominating in deep message passing
    - More stable training with 10-15+ GN blocks
    - Gradients flow more uniformly across layers

    Default activation is SiLU (Swish) which works better than ReLU for GNNs:
    - Smoother gradients prevent dying neurons
    - Better gradient flow in deep networks
    - Self-gated mechanism helps with physics simulations
    """
    if activation == 'relu':
        activation_func = nn.ReLU()
    elif activation == 'gelu':
        activation_func = nn.GELU()
    elif activation == 'silu':
        activation_func = nn.SiLU()
    elif activation == 'tanh':
        activation_func = nn.Tanh()
    elif activation == 'sigmoid':
        activation_func = nn.Sigmoid()
    else:
        raise ValueError(f'Invalid activation function: {activation}')

    if layer_norm:
        # Pre-LayerNorm: normalize BEFORE transformations
        module = nn.Sequential(
            nn.LayerNorm(normalized_shape=in_size),
            nn.Linear(in_size, hidden_size),
            activation_func,
            nn.LayerNorm(normalized_shape=hidden_size),
            nn.Linear(hidden_size, hidden_size),
            activation_func,
            nn.LayerNorm(normalized_shape=hidden_size),
            nn.Linear(hidden_size, out_size),
        )
    elif decoder:
        # Decoder: LayerNorm at input only, no LayerNorm before final output
        # This allows the decoder to output values with full range
        module = nn.Sequential(
            nn.LayerNorm(normalized_shape=in_size),
            nn.Linear(in_size, hidden_size),
            activation_func,
            nn.Linear(hidden_size, hidden_size),
            activation_func,
            nn.Linear(hidden_size, out_size)
        )

    return module

class Encoder(nn.Module):

    def __init__(self,
                edge_input_size,
                node_input_size,
                latent_dim,
                use_world_edges=False):
        super(Encoder, self).__init__()

        self.use_world_edges = use_world_edges
        self.eb_encoder = build_mlp(edge_input_size, latent_dim, latent_dim)
        self.nb_encoder = build_mlp(node_input_size, latent_dim, latent_dim)
        if use_world_edges:
            self.world_eb_encoder = build_mlp(edge_input_size, latent_dim, latent_dim)

    def forward(self, graph):
        node_attr, edge_attr = graph.x, graph.edge_attr
        node_ = self.nb_encoder(node_attr)
        edge_ = self.eb_encoder(edge_attr)

        out = Data(x=node_, edge_attr=edge_, edge_index=graph.edge_index)

        if self.use_world_edges and hasattr(graph, 'world_edge_attr') and graph.world_edge_index.shape[1] > 0:
            world_edge_ = self.world_eb_encoder(graph.world_edge_attr)
            out.world_edge_attr = world_edge_
            out.world_edge_index = graph.world_edge_index
        elif self.use_world_edges:
            out.world_edge_attr = torch.zeros(0, edge_.shape[1], device=edge_.device)
            out.world_edge_index = torch.zeros(2, 0, dtype=torch.long, device=edge_.device)

        return out

class GnBlock(nn.Module):

    def __init__(self, config, latent_dim, use_world_edges=False):
        super(GnBlock, self).__init__()

        self.use_world_edges = use_world_edges
        # Residual scaling factor for deep networks (prevents early layers from dominating)
        # For 10-15 GN blocks, scale down updates to balance old vs new information
        self.residual_scale = config.get('residual_scale', 0.1)

        eb_input_dim = 3 * latent_dim  # Sender, Receiver, edge latent dim
        eb_custom_func = build_mlp(eb_input_dim, latent_dim, latent_dim)
        self.eb_module = EdgeBlock(custom_func=eb_custom_func)

        if use_world_edges:
            # World edge block (separate from mesh edge block)
            world_eb_custom_func = build_mlp(eb_input_dim, latent_dim, latent_dim)
            self.world_eb_module = EdgeBlock(custom_func=world_eb_custom_func)
            # Hybrid node block aggregates from both edge types
            nb_input_dim = 3 * latent_dim  # Node + mesh_agg + world_agg
            nb_custom_func = build_mlp(nb_input_dim, latent_dim, latent_dim)
            self.nb_module = HybridNodeBlock(custom_func=nb_custom_func)
        else:
            nb_input_dim = 2 * latent_dim  # Node + aggregated edges
            nb_custom_func = build_mlp(nb_input_dim, latent_dim, latent_dim)
            self.nb_module = NodeBlock(custom_func=nb_custom_func)

    def forward(self, graph):
        x = graph.x.clone()
        edge_attr = graph.edge_attr.clone()

        # Save world edge info before mesh edge block (which creates new Data)
        world_edge_index = graph.world_edge_index if self.use_world_edges and hasattr(graph, 'world_edge_index') else None
        world_edge_attr = graph.world_edge_attr.clone() if self.use_world_edges and hasattr(graph, 'world_edge_attr') and graph.world_edge_attr is not None and graph.world_edge_attr.shape[0] > 0 else None

        # Update mesh edge features
        mesh_graph = self.eb_module(graph)

        # Update world edge features if present
        if self.use_world_edges and world_edge_attr is not None and world_edge_attr.shape[0] > 0:
            world_graph = Data(
                x=x,
                edge_attr=world_edge_attr,
                edge_index=world_edge_index
            )
            world_graph = self.world_eb_module(world_graph)
            # Apply residual scaling to world edges as well
            updated_world_edge_attr = world_edge_attr + self.residual_scale * world_graph.edge_attr
        else:
            updated_world_edge_attr = world_edge_attr

        # Prepare graph for node block with both edge types
        node_graph = Data(
            x=mesh_graph.x,
            edge_attr=mesh_graph.edge_attr,
            edge_index=mesh_graph.edge_index
        )
        if self.use_world_edges:
            node_graph.world_edge_attr = updated_world_edge_attr if updated_world_edge_attr is not None else torch.zeros(0, mesh_graph.edge_attr.shape[1], device=mesh_graph.x.device)
            node_graph.world_edge_index = world_edge_index if world_edge_index is not None else torch.zeros(2, 0, dtype=torch.long, device=mesh_graph.x.device)

        # Update node features (aggregates from both edge types)
        node_graph = self.nb_module(node_graph)

        # Residual connections with scaling for deep networks
        # Scale down updates to prevent early layers from dominating
        x = x + self.residual_scale * node_graph.x
        edge_attr = edge_attr + self.residual_scale * node_graph.edge_attr

        out = Data(x=x, edge_attr=edge_attr, edge_index=node_graph.edge_index)
        if self.use_world_edges:
            out.world_edge_attr = updated_world_edge_attr if updated_world_edge_attr is not None else torch.zeros(0, edge_attr.shape[1], device=x.device)
            out.world_edge_index = world_edge_index if world_edge_index is not None else torch.zeros(2, 0, dtype=torch.long, device=x.device)

        return out

class Decoder(nn.Module):

    def __init__(self, latent_dim, node_output_size):
        super(Decoder, self).__init__()
        self.decode_module = build_mlp(latent_dim, latent_dim, node_output_size, layer_norm=False, decoder=True)

    def forward(self, graph):
        return self.decode_module(graph.x)
