import torch.nn.init as init

import torch.nn as nn
import torch
from torch_geometric.data import Data
from model.blocks import EdgeBlock, NodeBlock

def init_weights(m):
    if isinstance(m, nn.Linear):
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

    def forward(self, graph):
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
        predicted = self.model(graph)

        return predicted, graph.y

class EncoderProcessorDecoder(nn.Module):
    def __init__(self, config):
        super(EncoderProcessorDecoder, self).__init__()
        self.config = config

        self.message_passing_num = config['message_passing_num']
        self.node_input_size = config['input_var']
        self.node_output_size = config['output_var']
        self.edge_input_size = config['edge_var']
        self.latent_dim = config['latent_dim']

        self.encoder = Encoder(self.edge_input_size, self.node_input_size, self.latent_dim)
        
        processer_list = []
        for _ in range(self.message_passing_num):
            processer_list.append(GnBlock(config, self.latent_dim))
        self.processer_list = nn.ModuleList(processer_list)
        
        self.decoder = Decoder(self.latent_dim, self.node_output_size)

    def forward(self, graph):

        graph= self.encoder(graph)
        for model in self.processer_list:
            graph = model(graph)
        output = self.decoder(graph)

        return output

def build_mlp(in_size, hidden_size, out_size, layer_norm=True, activation='relu'):

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
        module = nn.Sequential(
            nn.Linear(in_size, hidden_size), 
            activation_func,
            nn.LayerNorm(normalized_shape=hidden_size),
            nn.Linear(hidden_size, hidden_size),
            activation_func,
            nn.LayerNorm(normalized_shape=hidden_size),
            nn.Linear(hidden_size, out_size)
        )
    else:
        module = nn.Sequential(
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
                latent_dim):
        super(Encoder, self).__init__()

        self.eb_encoder = build_mlp(edge_input_size, latent_dim, latent_dim)
        self.nb_encoder = build_mlp(node_input_size, latent_dim, latent_dim)
    
    def forward(self, graph):

        node_attr, edge_attr = graph.x, graph.edge_attr
        node_ = self.nb_encoder(node_attr)
        edge_ = self.eb_encoder(edge_attr)
        
        return Data(x=node_, edge_attr=edge_, edge_index=graph.edge_index)
        # Output in graph format

class GnBlock(nn.Module):

    def __init__(self, config, latent_dim):
        super(GnBlock, self).__init__()

        # input_var = config['input_var']
        # edge_var = config['edge_var']

        eb_input_dim = 3 * latent_dim # Sender, Receiver, edge latent dim
        nb_input_dim = 2 * latent_dim # Node, aggregated edges latent dim
        nb_custom_func = build_mlp(nb_input_dim, latent_dim, latent_dim)
        eb_custom_func = build_mlp(eb_input_dim, latent_dim, latent_dim)
        
        self.eb_module = EdgeBlock(custom_func=eb_custom_func)
        self.nb_module = NodeBlock(custom_func=nb_custom_func)

    def forward(self, graph):
    
        x = graph.x # Input nodal features
        edge_attr = graph.edge_attr # Input edge features

        graph = self.eb_module(graph) # Update edge features
        # First go through the edge block to update the edge features
        graph = self.nb_module(graph)
        # Then go through the node block to update the node features based on the connectivity

        x = x + graph.x
        edge_attr = edge_attr + graph.edge_attr
        
        return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)

class Decoder(nn.Module):

    def __init__(self, latent_dim, node_output_size):
        super(Decoder, self).__init__()
        self.decode_module = build_mlp(latent_dim, latent_dim, node_output_size, layer_norm=False)

    def forward(self, graph):
        return self.decode_module(graph.x)
