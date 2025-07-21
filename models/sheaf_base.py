# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import math
from torch import nn


class SheafDiffusion(nn.Module):
    """Base class for sheaf diffusion models."""

    def __init__(self, edge_index, args):
        super(SheafDiffusion, self).__init__()

        assert args['d'] > 0
        self.d = args['d']
        self.edge_index = edge_index
        self.add_lp = args['add_lp']
        self.add_hp = args['add_hp']

        self.final_d = self.d
        if self.add_hp:
            self.final_d += 1
        if self.add_lp:
            self.final_d += 1

        self.hidden_dim = args['hidden_channels'] * self.final_d
        self.device = args['device']
        self.graph_size = args['graph_size']
        self.layers = args['layers']
        self.normalised = args['normalised']
        self.deg_normalised = args['deg_normalised']
        self.nonlinear = not args['linear']
        self.input_dropout = args['input_dropout']
        self.dropout = args['dropout']
        self.left_weights = args['left_weights']
        self.right_weights = args['right_weights']
        self.sparse_learner = args['sparse_learner']
        self.use_act = args['use_act']
        self.input_dim = args['input_dim']
        self.hidden_channels = args['hidden_channels']
        self.output_dim = args['output_dim']
        self.layers = args['layers']
        self.sheaf_act = args['sheaf_act']
        self.second_linear = args['second_linear']
        self.orth_trans = args['orth']
        self.use_edge_weights = args['edge_weights']
        self.t = args['max_t']
        self.time_range = torch.tensor([0.0, self.t], device=self.device)
        self.laplacian_builder = None

    def update_edge_index(self, edge_index):
        assert edge_index.max() <= self.graph_size
        self.edge_index = edge_index
        self.laplacian_builder = self.laplacian_builder.create_with_new_edge_index(edge_index)

    def grouped_parameters(self):
        sheaf_learners, others = [], []
        for name, param in self.named_parameters():
            if "sheaf_learner" in name:
                sheaf_learners.append(param)
            else:
                others.append(param)
        assert len(sheaf_learners) > 0
        assert len(sheaf_learners) + len(others) == len(list(self.parameters()))
        return sheaf_learners, others

class TemporalSheafDiffusion(SheafDiffusion):
    """
    Base class for temporal sheaf diffusion models.
    Extends SheafDiffusion to handle temporal graph data with TGB format.
    """
    
    def __init__(self, edge_index, args):
        super(TemporalSheafDiffusion, self).__init__(edge_index, args)
        
        # Temporal-specific parameters
        self.temporal_coeff_fn = args.get('temporal_coeff_fn', 'linear')  # How to compute c(Δt)
        self.base_temporal_coeff = args.get('base_temporal_coeff', 1.0)  # Base coefficient
        self.max_temporal_steps = args.get('max_temporal_steps', 10)  # Max steps between batches
        self.mamba_config = args.get('mamba_config', {})
        
        # Current timestamp tracking
        self.current_time = 0.0
        self.prev_time = 0.0
        
        # Optional 
        # Node embeddings for temporal processing
        self.node_embeddings = nn.Parameter(torch.randn(self.graph_size, self.hidden_channels))
        nn.init.xavier_uniform_(self.node_embeddings)
        
    def compute_temporal_coefficient(self, time_delta: float) -> int:
        """
        Compute number of sheaf convolution steps based on time difference.
        
        Args:
            time_delta: Time difference between consecutive batches
            
        Returns:
            num_steps: Number of sheaf convolution steps to perform
        """
        if self.temporal_coeff_fn == 'linear':
            coeff = self.base_temporal_coeff * time_delta
        elif self.temporal_coeff_fn == 'log':
            coeff = self.base_temporal_coeff * math.log(1 + time_delta)
        elif self.temporal_coeff_fn == 'sqrt':
            coeff = self.base_temporal_coeff * math.sqrt(time_delta)
        else:
            coeff = self.base_temporal_coeff
            
        # Round and clamp to reasonable range
        num_steps = max(1, min(self.max_temporal_steps, round(coeff)))
        return num_steps
    
    def reset_temporal_state(self):
        """Reset temporal state for new sequence (called at start of forward)"""
        self.current_time = 0.0
        self.prev_time = 0.0
        # Clear Mamba caches - to be implemented by subclasses
        pass
