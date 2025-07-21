# temporal_sheaf_diffusion.py
# Minimal imports for temporal sheaf diffusion models

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import laplacian_builders as lb
import torch_sparse

# Import the required components
from utils.edge_indexer import UniqueEdgeIndexer
from models.temporal_sheaf_models import MambaTemporalSheafLearner


class TemporalDiscreteDiagSheafDiffusion(nn.Module):
    """
    Temporal extension of DiscreteDiagSheafDiffusion using MambaTemporalSheafLearner.
    """
    
    def __init__(self, edge_index, args):
        super().__init__()
        
        # Copy necessary attributes from args
        self.d = args['d']
        self.device = args['device']
        self.graph_size = args['graph_size']
        self.layers = args['layers']
        self.hidden_channels = args['hidden_channels']
        self.input_dim = args['input_dim']
        self.output_dim = args['output_dim']
        self.msg_dim = args.get('msg_dim', 128)  # Message dimension
        self.dropout = args.get('dropout', 0.1)
        self.input_dropout = args.get('input_dropout', 0.1)
        self.use_act = args.get('use_act', True)
        self.sheaf_act = args.get('sheaf_act', 'tanh')
        self.second_linear = args.get('second_linear', False)
        self.nonlinear = not args.get('linear', False)
        self.normalised = args.get('normalised', True)
        self.deg_normalised = args.get('deg_normalised', False)
        self.add_hp = args.get('add_hp', False)
        self.add_lp = args.get('add_lp', False)
        
        # Final dimension accounting for high/low pass additions
        self.final_d = self.d
        if self.add_hp:
            self.final_d += 1
        if self.add_lp:
            self.final_d += 1
        
        self.hidden_dim = self.hidden_channels * self.final_d
        
        # Edge indexer
        edge_indexer = UniqueEdgeIndexer(max_edges=args.get('max_edges', 100000))
        
        # Temporal sheaf learners
        self.temporal_sheaf_learners = nn.ModuleList()
        num_learners = min(self.layers, self.layers if self.nonlinear else 1)
        
        for i in range(num_learners):
            learner = MambaTemporalSheafLearner(
                msg_dim=self.msg_dim,
                out_shape=(self.d,),
                edge_indexer=edge_indexer,
                d_model=args.get('mamba_d_model', 128),
                d_state=args.get('mamba_d_state', 16),
                sheaf_act=self.sheaf_act,
                device=self.device
            )
            self.temporal_sheaf_learners.append(learner)
        
        # Laplacian builder placeholder TODO: verify if this is correct
        self.laplacian_builder = lb.DiagLaplacianBuilder(
            self.graph_size, edge_index, d=self.d,
            normalised=self.normalised,
            deg_normalised=self.deg_normalised,
            add_hp=self.add_hp, add_lp=self.add_lp
        )
        
        # Epsilon parameters for residual connections
        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))
        
        # Linear layers
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)
        
        # Left and right weight transformations (optional)
        self.left_weights = args.get('left_weights', False)
        self.right_weights = args.get('right_weights', False)
        
        self.lin_left_weights = nn.ModuleList()
        self.lin_right_weights = nn.ModuleList()
        
        if self.right_weights:
            for i in range(self.layers):
                linear = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
                nn.init.orthogonal_(linear.weight.data)
                self.lin_right_weights.append(linear)
        
        if self.left_weights:
            for i in range(self.layers):
                linear = nn.Linear(self.final_d, self.final_d, bias=False)
                nn.init.eye_(linear.weight.data)
                self.lin_left_weights.append(linear)
    
    def forward(self, x, edge_index, edge_times, edge_attr):
        """
        Forward pass with temporal sheaf learning.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_times: Edge timestamps [num_edges]
            edge_attr: Edge attributes [num_edges, msg_dim]
        """
        # Initial transformation
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        
        # Reshape for sheaf processing
        x = x.view(self.graph_size * self.final_d, -1)
        x0 = x
        
        # Sheaf diffusion layers
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                # Get temporal sheaf maps
                x_for_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                x_nodes = x_for_maps.reshape(self.graph_size, -1)
                
                # Use temporal sheaf learner
                learner_idx = min(layer, len(self.temporal_sheaf_learners) - 1)
                maps = self.temporal_sheaf_learners[learner_idx](
                    x_nodes, edge_index, edge_times, edge_attr
                )
                
                # Build Laplacian
                if self.laplacian_builder is not None:
                    L, trans_maps = self.laplacian_builder(maps)
                    self.temporal_sheaf_learners[learner_idx].set_L(trans_maps)
                else:
                    # Placeholder Laplacian - replace with actual implementation
                    L = (edge_index, torch.ones(edge_index.size(1), device=self.device))
            
            # Apply dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Optional left/right transformations
            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()
            if self.right_weights:
                x = self.lin_right_weights[layer](x)
            
            # Sheaf Laplacian multiplication
            # You'll need to uncomment this when torch_sparse is available:
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)
            
            if self.use_act:
                x = F.elu(x)
            
            # Residual connection with learnable coefficient
            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
            x0 = coeff * x0 - x
            x = x0
        
        # Final transformation
        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)
    
    def reset_temporal_states(self):
        """Reset all temporal states in the learners."""
        for learner in self.temporal_sheaf_learners:
            learner.reset_states()


class TemporalDiscreteBundleSheafDiffusion(nn.Module):
    """
    Temporal extension of DiscreteBundleSheafDiffusion.
    Similar structure but with bundle sheaf maps (matrix outputs).
    """
    
    def __init__(self, edge_index, args):
        super().__init__()
        
        # Similar initialization to TemporalDiscreteDiagSheafDiffusion
        # but with matrix output shape for bundle sheaves
        assert args['d'] > 1
        
        # Copy attributes (same as above)
        self.d = args['d']
        self.device = args['device']
        self.graph_size = args['graph_size']
        self.layers = args['layers']
        self.hidden_channels = args['hidden_channels']
        self.input_dim = args['input_dim']
        self.output_dim = args['output_dim']
        self.msg_dim = args.get('msg_dim', 128)
        self.dropout = args.get('dropout', 0.1)
        self.input_dropout = args.get('input_dropout', 0.1)
        self.use_act = args.get('use_act', True)
        self.sheaf_act = args.get('sheaf_act', 'tanh')
        self.second_linear = args.get('second_linear', False)
        self.nonlinear = not args.get('linear', False)
        self.normalised = args.get('normalised', True)
        self.deg_normalised = args.get('deg_normalised', False)
        self.add_hp = args.get('add_hp', False)
        self.add_lp = args.get('add_lp', False)
        self.orth_trans = args.get('orth', 'matrix_exp')
        
        self.final_d = self.d
        if self.add_hp:
            self.final_d += 1
        if self.add_lp:
            self.final_d += 1
        
        self.hidden_dim = self.hidden_channels * self.final_d
        
        # Edge indexer
        edge_indexer = UniqueEdgeIndexer(max_edges=args.get('max_edges', 100000))
        
        # Temporal sheaf learners with matrix output
        self.temporal_sheaf_learners = nn.ModuleList()
        num_learners = min(self.layers, self.layers if self.nonlinear else 1)
        
        for i in range(num_learners):
            # Bundle sheaves use parameter shape for orthogonal matrices
            if self.orth_trans in ['matrix_exp', 'cayley']:
                param_size = self.d * (self.d + 1) // 2
            else:
                param_size = self.d * (self.d - 1) // 2
                
            learner = MambaTemporalSheafLearner(
                msg_dim=self.msg_dim,
                out_shape=(param_size,),
                edge_indexer=edge_indexer,
                d_model=args.get('mamba_d_model', 128),
                d_state=args.get('mamba_d_state', 16),
                sheaf_act=self.sheaf_act,
                device=self.device
            )
            self.temporal_sheaf_learners.append(learner)
        
        # Rest of initialization similar to TemporalDiscreteDiagSheafDiffusion
        self.laplacian_builder = None  # Replace with NormConnectionLaplacianBuilder
        
        self.epsilons = nn.ParameterList()