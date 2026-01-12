import torch.nn.init as init

import torch.nn as nn
import torch
from torch_geometric.data import Data
from general_modules import normalization
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

        self.input_var = config['input_var']
        self.output_var = config['output_var']
        self.edge_var = config['edge_var']

        self._output_normalizer = normalization.Normalizer(
            size=self.output_var, name='output_normalizer', device=device
        )
        self._node_normalizer = normalization.Normalizer(
            size=self.input_var, name='node_normalizer', device=device
        )
        self.edge_normalizer = normalization.Normalizer(
            size=self.edge_var, name='edge_normalizer', device=device
        )

        self.model = EncoderProcessorDecoder(config).to(device)
        self.model.apply(init_weights)
        print('MeshGraphNets model created successfully')

        self.training = config.get('mode') == 'Train'

    def update_node_attr(self, nodal_features: torch.Tensor) -> torch.Tensor:
        normalized_features = self._node_normalizer(nodal_features, self.training)
        return normalized_features

    def forward(self, graph):
        """
        Forward pass of the simulator.

        During training:
            - Inject noise into velocity
            - Predict normalized acceleration
            - Return prediction and normalized target acceleration

        During inference:
            - Use clean velocity
            - Denormalize predicted acceleration to get velocity update
            - Return predicted next-step velocity
        """
        nodal_features = graph.x[:, :]         # [N, input_var-xyz] — current variable of interest

        if self.training:
            noise = torch.randn_like(nodal_features) * self.config.get('std_noise')
            noised_nodal_features = nodal_features + noise
            node_attr = self.update_node_attr(noised_nodal_features)
            graph.x = node_attr

            edge_attr = graph.edge_attr  # [E, 4] (3D)
            edge_attr = self.edge_normalizer(edge_attr, self.training)
            graph.edge_attr = edge_attr

            predicted_norm = self.model(graph)

            target = graph.y
            # Learn the difference between nodal features @ t+1 and t
            target = target-noised_nodal_features 
            target_norm = self._output_normalizer(target, self.training)

            return predicted_norm, target_norm

        else:
            # Inference mode
            node_attr = self.update_node_attr(nodal_features)
            graph.x = node_attr
            
            edge_attr = graph.edge_attr  # [E, 4] (3D)
            edge_attr = self.edge_normalizer(edge_attr, self.training)
            graph.edge_attr = edge_attr
            
            predicted_norm = self.model(graph)  # [N, input_var]
            diff_update = self._output_normalizer.inverse(predicted_norm)  # [N, 2]
            predicted_nodal_features = nodal_features + diff_update
            return predicted_nodal_features

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
            # nn.LayerNorm(normalized_shape=hidden_size),
            activation_func,
            nn.Linear(hidden_size, hidden_size),
            # nn.LayerNorm(normalized_shape=hidden_size),
            activation_func,
            nn.Linear(hidden_size, hidden_size),
            activation_func,
            nn.Linear(hidden_size, out_size),
            nn.LayerNorm(normalized_shape=out_size)
        )
    else:
        module = nn.Sequential(
            nn.Linear(in_size, hidden_size), 
            activation_func,
            nn.Linear(hidden_size, hidden_size),
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
        self.decode_module = build_mlp(latent_dim, latent_dim, node_output_size, layer_norm=True)

    def forward(self, graph):
        return self.decode_module(graph.x)
