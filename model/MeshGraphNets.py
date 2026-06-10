import torch
import torch.nn as nn
from torch_geometric.data import Data

from general_modules.edge_features import EDGE_FEATURE_DIM
from model.checkpointing import process_with_checkpointing
from model.coarsening import pool_features, unpool_features
from model.encoder_decoder import Decoder, Encoder, GnBlock
from model.mlp import build_mlp, init_weights


class MeshGraphNets(nn.Module):
    def __init__(self, config, device: str):
        super().__init__()
        self.config = config
        self.device = device

        self.model = EncoderProcessorDecoder(config).to(device)
        self.model.apply(init_weights)

        # For time-transient delta prediction, start near "no change".
        num_timesteps = config.get('num_timesteps', None)
        if num_timesteps is None or num_timesteps > 1:
            with torch.no_grad():
                last_layer = self.model.decoder.decode_module[-1]
                last_layer.weight.mul_(0.01)

        print('MeshGraphNets model created successfully')

    def set_checkpointing(self, enabled: bool):
        self.model.set_checkpointing(enabled)

    def forward(self, graph, debug=False, add_noise=None):
        """
        Forward pass of the deterministic simulator.

        Expects pre-normalized inputs from the dataloader:
            - graph.x: normalized node features [N, input_var]
            - graph.edge_attr: normalized edge features [E, edge_var]
            - graph.y: normalized target delta [N, output_var]

        Returns:
            predicted: predicted normalized delta [N, output_var]
            target: normalized target delta [N, output_var]
        """
        if add_noise is None:
            add_noise = self.training

        if add_noise:
            noise_std = self.config.get('std_noise', 0.0)
            if noise_std > 0:
                output_var = self.config['output_var']
                noise = torch.randn(
                    graph.x.shape[0], output_var,
                    device=graph.x.device, dtype=graph.x.dtype
                ) * noise_std
                noise_padded = torch.zeros_like(graph.x)
                noise_padded[:, :output_var] = noise
                graph.x = graph.x + noise_padded
                noise_gamma = self.config.get('noise_gamma', 1)
                noise_std_ratio = self.config.get('noise_std_ratio', None)
                if noise_std_ratio is not None:
                    ratio = torch.tensor(noise_std_ratio, device=graph.x.device, dtype=graph.x.dtype)
                    graph.y = graph.y - noise_gamma * noise * ratio
                graph.edge_attr = graph.edge_attr + torch.randn_like(graph.edge_attr) * noise_std

        predicted = self.model(graph, debug=debug)
        return predicted, getattr(graph, 'y', None)


class EncoderProcessorDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.message_passing_num = config['message_passing_num']
        self.edge_input_size = int(config['edge_var'])
        if self.edge_input_size != EDGE_FEATURE_DIM:
            raise ValueError(f"edge_var must be {EDGE_FEATURE_DIM}, got {self.edge_input_size}")
        self.latent_dim = config['latent_dim']
        self.use_checkpointing = config.get('use_checkpointing', False)
        self.use_world_edges = config.get('use_world_edges', False)
        self.use_multiscale = config.get('use_multiscale', False)
        self.use_coarse_world_edges = (
            bool(config.get('coarse_world_edges', False))
            and self.use_world_edges
            and self.use_multiscale
        )

        base_input_size = config['input_var']
        num_pos_features = int(config.get('positional_features', 0))
        base_input_size += num_pos_features
        use_node_types = config.get('use_node_types', False)
        num_node_types = config.get('num_node_types', 0)
        if use_node_types and num_node_types > 0:
            self.node_input_size = base_input_size + num_node_types
            print(
                f"  Model input: {config['input_var']} physical + "
                f"{num_pos_features} positional + {num_node_types} node types = {self.node_input_size}"
            )
        else:
            self.node_input_size = base_input_size
            if num_pos_features > 0:
                print(
                    f"  Model input: {config['input_var']} physical + "
                    f"{num_pos_features} positional = {self.node_input_size}"
                )

        self.node_output_size = config['output_var']
        self.encoder = Encoder(
            self.edge_input_size, self.node_input_size, self.latent_dim,
            use_world_edges=self.use_world_edges
        )

        if not self.use_multiscale:
            self.processer_list = nn.ModuleList([
                GnBlock(config, self.latent_dim, use_world_edges=self.use_world_edges)
                for _ in range(self.message_passing_num)
            ])
        else:
            self._build_multiscale_processor(config)

        self.decoder = Decoder(self.latent_dim, self.node_output_size)

    def _build_multiscale_processor(self, config):
        L = int(config.get('multiscale_levels', 1))
        self.multiscale_levels = L

        mp_per_level = config.get('mp_per_level', None)
        if mp_per_level is None:
            fine_pre = int(config.get('fine_mp_pre', 5))
            coarse_mp = int(config.get('coarse_mp_num', 5))
            fine_post = int(config.get('fine_mp_post', 5))
            mp_per_level = [fine_pre] + [coarse_mp] + [fine_post]
        if not isinstance(mp_per_level, list):
            mp_per_level = [int(mp_per_level)]
        else:
            mp_per_level = [int(x) for x in mp_per_level]
        self.mp_per_level = mp_per_level

        expected_len = 2 * L + 1
        if len(mp_per_level) != expected_len:
            raise ValueError(
                f"mp_per_level must have {expected_len} entries for {L} levels, "
                f"got {len(mp_per_level)}: {mp_per_level}"
            )

        parts = []
        for i in range(L):
            parts.append(f"pre[{i}]={mp_per_level[i]}")
        parts.append(f"coarsest={mp_per_level[L]}")
        for i in range(L - 1, -1, -1):
            parts.append(f"post[{i}]={mp_per_level[2 * L - i]}")
        print(f"  Multiscale V-cycle ({L} levels): {', '.join(parts)}")

        coarse_config = dict(config)
        coarse_config['use_world_edges'] = False

        self.pre_blocks = nn.ModuleList()
        self.post_blocks = nn.ModuleList()
        self.coarse_eb_encoders = nn.ModuleList()
        self.skip_projs = nn.ModuleList()

        for i in range(L):
            pre_count = mp_per_level[i]
            post_count = mp_per_level[2 * L - i]
            use_we = self.use_world_edges if (i == 0 or self.use_coarse_world_edges) else False
            cfg = config if use_we else coarse_config

            self.pre_blocks.append(nn.ModuleList([
                GnBlock(cfg, self.latent_dim, use_world_edges=use_we)
                for _ in range(pre_count)
            ]))
            self.post_blocks.append(nn.ModuleList([
                GnBlock(cfg, self.latent_dim, use_world_edges=use_we)
                for _ in range(post_count)
            ]))
            self.coarse_eb_encoders.append(
                build_mlp(self.edge_input_size, self.latent_dim, self.latent_dim)
            )
            self.skip_projs.append(nn.Linear(2 * self.latent_dim, self.latent_dim))

        self.bipartite_unpool = config.get('bipartite_unpool', False)
        if self.bipartite_unpool:
            from model.blocks import UnpoolBlock
            self.unpool_blocks = nn.ModuleList([
                UnpoolBlock(self.latent_dim, build_mlp) for _ in range(L)
            ])

        coarsest_count = mp_per_level[L]
        coarsest_cfg = config if self.use_coarse_world_edges else coarse_config
        self.coarsest_blocks = nn.ModuleList([
            GnBlock(coarsest_cfg, self.latent_dim, use_world_edges=self.use_coarse_world_edges)
            for _ in range(coarsest_count)
        ])

    def forward(self, graph, debug=False):
        if not self.use_multiscale:
            return self._forward_flat(graph, debug)
        return self._forward_multiscale(graph, debug)

    def _forward_flat(self, graph, debug):
        graph = self.encoder(graph)
        if debug:
            print(f"  After Encoder: x std={graph.x.std().item():.4f}, mean={graph.x.mean().item():.4f}")

        graph = self._run_processor_blocks(self.processer_list, graph)
        if debug:
            print(f"  After {len(self.processer_list)} MP blocks: x std={graph.x.std().item():.4f}, mean={graph.x.mean().item():.4f}")

        output = self.decoder(graph)
        if debug:
            print(f"  After Decoder: out std={output.std().item():.4f}, mean={output.mean().item():.4f}")
        return output

    def _forward_multiscale(self, graph, debug):
        L = self.multiscale_levels
        level_data = self._extract_level_data(graph, L)
        actual_levels = len(level_data)

        graph = self.encoder(graph)
        if debug:
            print(f"  [MS] After Encoder: x std={graph.x.std().item():.4f}")

        skip_states = []
        current_graph = graph

        for i in range(actual_levels):
            current_graph = self._run_processor_blocks(self.pre_blocks[i], current_graph)
            if debug:
                print(f"  [MS] After pre[{i}] ({len(self.pre_blocks[i])} blocks): x std={current_graph.x.std().item():.4f}")

            use_we_here = self.use_world_edges and (i == 0 or self.use_coarse_world_edges)
            skip_states.append({
                'x': current_graph.x,
                'edge_attr': current_graph.edge_attr,
                'edge_index': current_graph.edge_index,
                'w_attr': getattr(current_graph, 'world_edge_attr', None) if use_we_here else None,
                'w_idx': getattr(current_graph, 'world_edge_index', None) if use_we_here else None,
            })

            ld = level_data[i]
            # Inherit mode (voronoi_inherit): coarse feature = seed's feature
            # (pure gather). Centroid mode: coarse feature = mean over cluster.
            if 'seeds' in ld:
                h_coarse = current_graph.x[ld['seeds']]
            else:
                h_coarse = pool_features(current_graph.x, ld['ftc'], ld['n_c'])
            e_coarse = self.coarse_eb_encoders[i](ld['c_ea'])
            current_graph = Data(x=h_coarse, edge_attr=e_coarse, edge_index=ld['c_ei'])
            if self.use_coarse_world_edges and ld['c_we_idx'] is not None and ld['c_we_idx'].shape[1] > 0:
                current_graph.world_edge_attr = ld['c_we_attr']
                current_graph.world_edge_index = ld['c_we_idx']

            if debug:
                print(f"  [MS] After pool[{i}]: {skip_states[-1]['x'].shape[0]} -> {h_coarse.shape[0]} nodes")

        current_graph = self._run_processor_blocks(self.coarsest_blocks, current_graph)
        if debug:
            print(f"  [MS] After coarsest ({len(self.coarsest_blocks)} blocks): x std={current_graph.x.std().item():.4f}")

        for i in range(actual_levels - 1, -1, -1):
            ld = level_data[i]
            if getattr(self, 'bipartite_unpool', False):
                src, dst = ld['up_ei']
                rel_pos = ld['fine_pos'][dst] - ld['coarse_centroid'][src]
                h_up = self.unpool_blocks[i](
                    h_coarse=current_graph.x,
                    h_fine_skip=skip_states[i]['x'],
                    unpool_edge_index=ld['up_ei'],
                    rel_pos=rel_pos,
                )
            else:
                h_up = unpool_features(current_graph.x, ld['ftc'])

            skip = skip_states[i]
            h_merged = self.skip_projs[i](torch.cat([skip['x'], h_up], dim=-1))
            current_graph = Data(x=h_merged, edge_attr=skip['edge_attr'], edge_index=skip['edge_index'])
            use_we_here = self.use_world_edges and (i == 0 or self.use_coarse_world_edges)
            if use_we_here and skip['w_attr'] is not None:
                current_graph.world_edge_attr = skip['w_attr']
                current_graph.world_edge_index = skip['w_idx']

            current_graph = self._run_processor_blocks(self.post_blocks[i], current_graph)
            if debug:
                print(f"  [MS] After post[{i}] ({len(self.post_blocks[i])} blocks): x std={current_graph.x.std().item():.4f}")

        output = self.decoder(current_graph)
        if debug:
            print(f"  [MS] After Decoder: out std={output.std().item():.4f}")
        return output

    def _extract_level_data(self, graph, L):
        """Extract per-level coarsening topology before the encoder drops custom attrs."""
        level_data = {}
        for i in range(L):
            ftc_key = f'fine_to_coarse_{i}'
            if not hasattr(graph, ftc_key):
                break
            ld = {
                'ftc': graph[ftc_key],
                'c_ei': graph[f'coarse_edge_index_{i}'],
                'c_ea': graph[f'coarse_edge_attr_{i}'],
                'n_c': int(graph[f'num_coarse_{i}'].sum()),
                'c_we_idx': getattr(graph, f'coarse_world_edge_index_{i}', None),
                'c_we_attr': getattr(graph, f'coarse_world_edge_attr_{i}', None),
            }
            # Inherit-mode (voronoi_inherit) levels expose seed indices.
            seed_key = f'coarse_seed_idx_{i}'
            if hasattr(graph, seed_key):
                ld['seeds'] = graph[seed_key]
            if self.use_multiscale and getattr(self, 'bipartite_unpool', False):
                ld['up_ei'] = graph[f'unpool_edge_index_{i}']
                ld['coarse_centroid'] = graph[f'coarse_centroid_{i}']
                ld['fine_pos'] = graph.pos if i == 0 else graph[f'coarse_centroid_{i - 1}']
            level_data[i] = ld
        return level_data

    def _run_processor_blocks(self, blocks, graph):
        """Run a stack of GnBlocks on raw tensors (one Data rebuild at the end).

        The per-block Data construction of the old path was measurable Python
        overhead at batch_size=1 and breaks torch.compile graphs.
        """
        x, edge_attr = graph.x, graph.edge_attr
        edge_index = graph.edge_index
        world_edge_attr = getattr(graph, 'world_edge_attr', None)
        world_edge_index = getattr(graph, 'world_edge_index', None)

        if self.use_checkpointing and self.training:
            x, edge_attr, world_edge_attr = process_with_checkpointing(
                blocks, x, edge_attr, edge_index, world_edge_attr, world_edge_index
            )
        else:
            for block in blocks:
                x, edge_attr, world_edge_attr = block.forward_tensors(
                    x, edge_attr, edge_index, world_edge_attr, world_edge_index
                )

        out = Data(x=x, edge_attr=edge_attr, edge_index=edge_index)
        if world_edge_attr is not None and world_edge_index is not None:
            out.world_edge_attr = world_edge_attr
            out.world_edge_index = world_edge_index
        return out

    def set_checkpointing(self, enabled: bool):
        self.use_checkpointing = enabled
