import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
import math
from torch_scatter import scatter_add
from torch_geometric.utils import degree
from typing import Optional, Tuple, List

from utils.edge_indexed_mamba import EdgeIndexedMamba, EdgeIndexedInferenceParams
from einops import rearrange, repeat

# Assuming these imports from your existing code
# from . import laplacian_builders as lb
# from .sheaf_models import LocalConcatSheafLearner, EdgeWeightLearner
from .sheaf_base import TemporalSheafDiffusion

"""
######## Note: Edge indexing is not a matrix but edge lists for efficiency
#####  The seq. length in the input shape might be obsolete/redundant due to sparsity
##### Input channel also assumes edge features (so in_channel is not * 2 inside). 
"""


class MambaSheafLearner(nn.Module):
    """
    Sheaf learner using EdgeIndexedMamba for temporal modeling.
    """
    def __init__(
        self,
        in_channels: int,
        out_shape: Tuple[int, ...],
        sheaf_act="tanh",

        ## ssm specification
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 2,
        # dropout: float = 0.1,

        # Other
        device=None,
        dtype=None,
        **mamba_kwargs
    ):
        super().__init__()
        assert len(out_shape) in [1, 2]

        self.in_channels = in_channels 
        self.out_shape = out_shape

        # mamba specific parameters
        self.d_model = d_model
        self.num_layers = num_layers
        # self.dropout = dropout
        
        # Calculate output size
        self.out_size = 1
        for dim in out_shape:
            self.out_size *= dim
            
        # Input projection
        self.input_proj = nn.Linear(in_channels, d_model)
        
        # EdgeIndexedMamba layers
        self.mamba_layers = nn.ModuleList([
            EdgeIndexedMamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                layer_idx=i,
                device=device,
                dtype=dtype,
                **mamba_kwargs
            )
            for i in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, self.out_size)
        
        # # Dropout
        # self.dropout_layer = nn.Dropout(dropout)

        if sheaf_act == 'id':
            self.act = lambda x: x
        elif sheaf_act == 'tanh':
            self.act = torch.tanh
        elif sheaf_act == 'elu':
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")
        
    def forward(self, x, edge_index, inference_params=None):
        """
        Forward pass for MambaSheafLearner.
        
        Args:
            x: Input features [num_edges, seq_len, in_channels]
            edge_index: Edge indices for cache indexing [num_edges] or [2, num_edges]
            inference_params: EdgeIndexedInferenceParams for caching
            
        Returns:
            sheaf_params: Sheaf parameters [num_edges, *out_shape]
        """
        # Input projection
        x = self.input_proj(x)  # [num_edges, seq_len, d_model]
        
        #x = self.dropout_layer(x)
        
        # Pass through EdgeIndexedMamba layers
        for layer in self.mamba_layers:
            x = layer(x, edge_index=edge_index, inference_params=inference_params)
            # x = self.dropout_layer(x)
        
        # Take the last timestep output
        x = x[:, -1, :]  # [num_edges, d_model]
        
        # Output projection
        sheaf_params = self.output_proj(x)  # [num_edges, out_size]
        
        # Reshape to desired output shape
        if len(self.out_shape) > 1:
            sheaf_params = sheaf_params.view(-1, *self.out_shape)
        
        return sheaf_params
    
    def reset_edge_cache(self, edge_idx: int, inference_params: EdgeIndexedInferenceParams):
        """Reset cache for a specific edge."""
        if inference_params is not None:
            inference_params.reset_edge_cache(edge_idx)
    
    def reset_all_caches(self, inference_params: EdgeIndexedInferenceParams):
        """Reset all edge caches."""
        if inference_params is not None:
            inference_params.reset_all_caches()


class TemporalDiscreteBundleSheafDiffusion(TemporalSheafDiffusion):
    """
    Temporal extension of DiscreteBundleSheafDiffusion using Mamba for temporal modeling.
    Handles TGB format: TemporalData(src=[batch], dst=[batch], t=[batch], msg=[batch, feat_dim], y=[batch], n_id=[num_nodes])
    """
    
    def __init__(self, edge_index, args):
        super(TemporalDiscreteBundleSheafDiffusion, self).__init__(edge_index, args)
        
        # Linear transformations
        self.lin_right_weights = nn.ModuleList()                
        self.lin_left_weights = nn.ModuleList()
        
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
                
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        # Mamba sheaf learners
        self.sheaf_learners = nn.ModuleList()
        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        
        # Default Mamba configuration
        mamba_config = self.mamba_config.copy()
        mamba_config.setdefault('in_channels', args.get('msg_dim', 1))  # Message feature dimension
        mamba_config.setdefault('out_shape', (self.get_param_size(),))
        mamba_config.setdefault('d_model', 64)
        mamba_config.setdefault('num_layers', 2)
        
        for i in range(num_sheaf_learners):
            self.sheaf_learners.append(MambaSheafLearner(**mamba_config))
        
        # Partial laplacian builder
        self.laplacian_builder = PartialDiagLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_hp=self.add_hp,
            add_lp=self.add_lp, normalised=self.normalised,
            deg_normalised=self.deg_normalised)

        # Temporal coefficients (epsilons from original model)
        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        # Input/output projections
        self.lin1 = nn.Linear(self.input_dim, self.hidden_channels)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_channels, self.hidden_channels)
        self.lin2 = nn.Linear(self.hidden_channels, self.output_dim)
        
        # Inference parameters for Mamba caching
        self.inference_params = None

    def get_param_size(self):
        """Get the number of parameters for sheaf maps"""
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def left_right_linear(self, x, left, right):
        """Apply left and right linear transformations"""
        if left is not None:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if right is not None:
            x = right(x)

        return x

    def reset_temporal_state(self):
        """Reset temporal state for new sequence"""
        super().reset_temporal_state()
        # Clear Mamba caches
        if hasattr(self, 'inference_params') and self.inference_params is not None:
            self.inference_params.reset_all_caches()

    def _perform_sheaf_convolutions(self, x, maps, target_edge_index, num_steps=1):
        """
        Perform multiple sheaf convolution steps.
        
        Args:
            x: Node features [graph_size * final_d, hidden_channels]
            maps: Sheaf maps [num_nodes, d]
            target_edge_index: Target edges [2, num_edges]
            num_steps: Number of convolution steps
            
        Returns:
            x: Updated node features
        """
        # Build laplacian once for efficiency
        L, trans_maps = self.laplacian_builder(maps, target_edge_index)
        
        for step in range(num_steps):
            x_step = x
            
            for layer in range(self.layers):
                x_step = F.dropout(x_step, p=self.dropout, training=self.training)
                
                # Apply linear transformations
                left_linear = self.lin_left_weights[layer] if layer < len(self.lin_left_weights) else None
                right_linear = self.lin_right_weights[layer] if layer < len(self.lin_right_weights) else None
                x_step = self.left_right_linear(x_step, left_linear, right_linear)
                
                # Sheaf convolution using sparse matrix multiplication
                x_step = torch_sparse.spmm(L[0], L[1], x_step.size(0), x_step.size(0), x_step)
                
                if self.use_act:
                    x_step = F.elu(x_step)
                
                # Residual connection with epsilon
                epsilon = torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)
                x = (1 + epsilon) * x - x_step
        
        return x

    def step(self, temporal_data):
        """
        Process a single temporal batch from TGB dataloader.
        
        Args:
            temporal_data: TemporalData object with fields:
                - src: [batch_size] source node indices
                - dst: [batch_size] destination node indices  
                - t: [batch_size] timestamps
                - msg: [batch_size, msg_dim] message features
                - y: [batch_size] labels (optional)
                - n_id: [num_involved_nodes] involved node indices
                
        Returns:
            node_embeddings: Updated node embeddings for involved nodes
            predictions: Predictions for the batch (if applicable)
        """
        device = self.device
        batch_size = temporal_data.src.size(0)
        
        # Extract temporal information
        src, dst = temporal_data.src, temporal_data.dst
        timestamps = temporal_data.t
        msg_features = temporal_data.msg  # [batch_size, msg_dim]
        involved_nodes = temporal_data.n_id
        
        # Update temporal tracking
        current_batch_time = timestamps.float().mean().item()
        time_delta = current_batch_time - self.prev_time if self.prev_time > 0 else 0.0
        
        # Compute number of sheaf convolution steps
        num_conv_steps = self.compute_temporal_coefficient(time_delta)
        
        # Create edge index for current batch
        batch_edge_index = torch.stack([src, dst], dim=0)  # [2, batch_size]
        
        # Create edge features with temporal history for Mamba
        # For now, use current features; in practice, you'd maintain a temporal buffer
        edge_features = msg_features.unsqueeze(1)  # [batch_size, 1, msg_dim] - single timestep
        
        # Get current node embeddings for involved nodes
        x = self.node_embeddings[involved_nodes]  # [num_involved_nodes, hidden_channels]
        
        # Learn sheaf parameters using Mamba
        for layer in range(min(len(self.sheaf_learners), self.layers)):
            if layer == 0 or self.nonlinear:
                # Create unique edge indices for caching (using src-dst pairs)
                edge_indices = src * self.graph_size + dst  # Unique edge identifier
                
                # Learn sheaf maps
                maps = self.sheaf_learners[layer](
                    edge_features,
                    edge_indices,
                    inference_params=self.inference_params
                )
                
                # Reshape maps to node-level (scatter to involved nodes)
                node_maps = torch.zeros(len(involved_nodes), self.d, device=device)
                # Simple averaging for nodes involved in multiple edges
                src_local = torch.searchsorted(involved_nodes, src)
                dst_local = torch.searchsorted(involved_nodes, dst)
                node_maps.index_add_(0, src_local, maps[:, 0, :])
                node_maps.index_add_(0, dst_local, maps[:, 0, :])
                
                # Normalize by node degree (simplified)
                node_counts = torch.zeros(len(involved_nodes), device=device)
                node_counts.index_add_(0, src_local, torch.ones(batch_size, device=device))
                node_counts.index_add_(0, dst_local, torch.ones(batch_size, device=device))
                node_counts = torch.clamp(node_counts, min=1.0).unsqueeze(1)
                node_maps = node_maps / node_counts
        
        # Prepare for sheaf convolution
        x_conv = x.view(len(involved_nodes) * self.final_d, -1)
        
        # Perform multiple sheaf convolution steps based on time difference
        if num_conv_steps > 0:
            x_conv = self._perform_sheaf_convolutions(
                x_conv, node_maps, batch_edge_index, num_conv_steps
            )
        
        # Update node embeddings
        x_updated = x_conv.reshape(len(involved_nodes), -1)
        self.node_embeddings[involved_nodes] = x_updated
        
        # Update temporal tracking
        self.prev_time = current_batch_time
        
        # Generate predictions if needed (e.g., for link prediction)
        if hasattr(temporal_data, 'y'):
            # Simple prediction using source and destination embeddings
            src_emb = x_updated[src_local]  # [batch_size, hidden_channels]
            dst_emb = x_updated[dst_local]  # [batch_size, hidden_channels]
            edge_predictions = self.lin2(src_emb * dst_emb)  # [batch_size, output_dim]
            return x_updated, edge_predictions
        
        return x_updated, None

    def forward(self, temporal_data_sequence):
        """
        Process a sequence of temporal batches.
        
        Args:
            temporal_data_sequence: List of TemporalData objects
            
        Returns:
            all_predictions: List of predictions for each batch
            final_embeddings: Final node embeddings
        """
        # Reset temporal state at start of sequence
        self.reset_temporal_state()
        
        # Initialize Mamba inference parameters
        from .mamba_sheaf_learner import EdgeIndexedInferenceParams
        max_seq_len = len(temporal_data_sequence)
        self.inference_params = EdgeIndexedInferenceParams(max_seqlen=max_seq_len)
        
        all_predictions = []
        
        # Process each temporal batch
        for i, temporal_data in enumerate(temporal_data_sequence):
            embeddings, predictions = self.step(temporal_data)
            if predictions is not None:
                all_predictions.append(predictions)
        
        return all_predictions, self.node_embeddings




class PartialDiagLaplacianBuilder(nn.Module):
    """
    Modified DiagLaplacianBuilder that supports partial forward computation
    for only specified edges and their 1-hop neighbors.
    """
    
    def __init__(self, size, edge_index, d, normalised=False, deg_normalised=False, 
                 add_hp=False, add_lp=False, augmented=True):
        super(PartialDiagLaplacianBuilder, self).__init__()
        
        self.size = size
        self.edge_index = edge_index
        self.d = d
        self.final_d = d
        self.normalised = normalised
        self.deg_normalised = deg_normalised
        self.add_hp = add_hp
        self.add_lp = add_lp
        self.augmented = augmented
        self.device = edge_index.device
        
        if add_hp:
            self.final_d += 1
        if add_lp:
            self.final_d += 1
            
        # Precompute full graph structures
        self._precompute_indices()
        self.deg = degree(self.edge_index[0], num_nodes=self.size)
        
    def _precompute_indices(self):
        """Precompute indices for full graph"""
        # This would use your existing laplace utilities
        # For now, simplified implementation
        try:
            from lib import laplace as lap
            
            self.full_left_right_idx, _ = lap.compute_left_right_map_index(self.edge_index, full_matrix=True)
            self.left_right_idx, self.vertex_tril_idx = lap.compute_left_right_map_index(self.edge_index)
            
            # Precompute learnable indices
            self.diag_indices, self.tril_indices = lap.compute_learnable_diag_laplacian_indices(
                self.size, self.vertex_tril_idx, self.d, self.final_d)
                
            if self.add_lp or self.add_hp:
                self.fixed_diag_indices, self.fixed_tril_indices = lap.compute_fixed_diag_laplacian_indices(
                    self.size, self.vertex_tril_idx, self.d, self.final_d)
        except ImportError:
            # Fallback implementation
            self._fallback_precompute()
    
    def _fallback_precompute(self):
        """Fallback precomputation when lib.laplace is not available"""
        # Simplified indices for basic functionality
        num_edges = self.edge_index.size(1)
        self.left_right_idx = (torch.arange(num_edges, device=self.device), 
                              torch.arange(num_edges, device=self.device))
        self.vertex_tril_idx = self.edge_index
        
        # Basic diagonal and off-diagonal indices
        self.diag_indices = torch.stack([torch.arange(self.size, device=self.device)] * 2)
        self.tril_indices = self.edge_index
    
    def get_partial_edge_set(self, target_edge_index):
        """
        Get the set of edges that includes target edges and their 1-hop neighbors.
        
        Args:
            target_edge_index: [2, num_target_edges] - edges to focus on
            
        Returns:
            partial_edge_index: [2, num_partial_edges] - extended edge set
            edge_mask: [num_full_edges] - mask indicating which edges are included
            involved_nodes: [num_involved_nodes] - nodes involved in computation
        """
        # Get nodes involved in target edges
        target_nodes = torch.unique(target_edge_index.view(-1))
        
        # Find all edges that involve these nodes (1-hop neighbors)
        row, col = self.edge_index
        node_mask = torch.isin(row, target_nodes) | torch.isin(col, target_nodes)
        
        # Get partial edge index
        partial_edge_index = self.edge_index[:, node_mask]
        involved_nodes = torch.unique(partial_edge_index.view(-1))
        
        return partial_edge_index, node_mask, involved_nodes
    
    def normalise(self, diag, tril, row, col):
        """Normalization for partial computation"""
        if self.normalised:
            d_sqrt_inv = (diag + 1).pow(-0.5) if self.augmented else diag.pow(-0.5)
            d_sqrt_inv.masked_fill_(d_sqrt_inv == float('inf'), 0)
            left_norm, right_norm = d_sqrt_inv[row], d_sqrt_inv[col]
            tril = left_norm * tril * right_norm
            diag = d_sqrt_inv * diag * d_sqrt_inv
        elif self.deg_normalised:
            deg_sqrt_inv = (self.deg + 1).pow(-0.5) if self.augmented else self.deg.pow(-0.5)
            deg_sqrt_inv = deg_sqrt_inv.unsqueeze(-1)
            deg_sqrt_inv.masked_fill_(deg_sqrt_inv == float('inf'), 0)
            left_norm, right_norm = deg_sqrt_inv[row], deg_sqrt_inv[col]
            tril = left_norm * tril * right_norm
            diag = deg_sqrt_inv * diag * deg_sqrt_inv
        return diag, tril
    
    def forward(self, maps, edge_index=None):
        """
        Forward pass with optional partial computation.
        
        Args:
            maps: [num_nodes, d] - sheaf maps for all nodes
            edge_index: [2, num_edges] - if provided, compute only for these edges and neighbors
            
        Returns:
            (edge_index, weights): Sparse laplacian representation
            saved_tril_maps: For potential use in backward pass
        """
        if edge_index is not None:
            return self._partial_forward(maps, edge_index)
        else:
            return self._full_forward(maps)
    
    def _full_forward(self, maps):
        """Full forward pass - simplified version of DiagLaplacianBuilder"""
        assert len(maps.size()) == 2
        assert maps.size(1) == self.d
        
        left_idx, right_idx = self.left_right_idx
        tril_row, tril_col = self.vertex_tril_idx
        row, _ = self.edge_index

        # Compute the un-normalised Laplacian entries
        left_maps = torch.index_select(maps, index=left_idx, dim=0)
        right_maps = torch.index_select(maps, index=right_idx, dim=0)
        tril_maps = -left_maps * right_maps
        saved_tril_maps = tril_maps.detach().clone()
        diag_maps = scatter_add(maps**2, row, dim=0, dim_size=self.size)

        # Normalise the entries
        diag_maps, tril_maps = self.normalise(diag_maps, tril_maps, tril_row, tril_col)
        
        # Create sparse representation
        num_edges = self.edge_index.size(1)
        tril_indices = self.edge_index
        diag_indices = torch.stack([torch.arange(self.size, device=self.device)] * 2)
        
        # Flatten maps
        tril_maps = tril_maps.view(-1)
        diag_maps = diag_maps.view(-1)
        
        # Add upper triangular part
        triu_indices = torch.stack([self.edge_index[1], self.edge_index[0]])
        
        # Combine indices and values
        all_indices = torch.cat([tril_indices, triu_indices, diag_indices], dim=1)
        all_values = torch.cat([tril_maps, tril_maps, diag_maps])
        
        return (all_indices, all_values), saved_tril_maps
    
    def _partial_forward(self, maps, target_edge_index):
        """Partial forward pass for specified edges and neighbors"""
        # Get extended edge set including 1-hop neighbors
        partial_edge_index, edge_mask, involved_nodes = self.get_partial_edge_set(target_edge_index)
        
        # Create node mapping for local indexing
        num_involved = len(involved_nodes)
        node_map = {node.item(): i for i, node in enumerate(involved_nodes)}
        
        # Map to local indices
        row, col = partial_edge_index
        local_row = torch.tensor([node_map[r.item()] for r in row], device=self.device)
        local_col = torch.tensor([node_map[c.item()] for c in col], device=self.device)
        
        # Compute maps for partial graph
        left_maps = maps[row]  # [num_partial_edges, d]
        right_maps = maps[col]  # [num_partial_edges, d]
        tril_maps = -left_maps * right_maps  # [num_partial_edges, d]
        saved_tril_maps = tril_maps.detach().clone()
        
        # Compute diagonal entries for involved nodes only
        diag_maps = torch.zeros(num_involved, self.d, device=self.device)
        for i, node in enumerate(involved_nodes):
            node_edges = (row == node)
            if node_edges.any():
                diag_maps[i] = (maps[node]**2).sum(dim=0, keepdim=True)
        
        # Normalise
        diag_maps, tril_maps = self.normalise(diag_maps, tril_maps, local_row, local_col)
        
        # Build sparse indices
        tril_indices = torch.stack([local_row, local_col])
        diag_indices = torch.stack([torch.arange(num_involved, device=self.device)] * 2)
        triu_indices = torch.stack([local_col, local_row])
        
        # Flatten
        tril_maps = tril_maps.view(-1)
        diag_maps = diag_maps.view(-1)
        
        # Combine all parts
        all_indices = torch.cat([tril_indices, triu_indices, diag_indices], dim=1)
        all_values = torch.cat([tril_maps, tril_maps, diag_maps])
        
        # Map back to original node indices
        all_indices[0] = involved_nodes[all_indices[0]]
        all_indices[1] = involved_nodes[all_indices[1]]
        
        return (all_indices, all_values), saved_tril_maps
    
    def create_with_new_edge_index(self, edge_index):
        """Create new builder with different edge index"""
        new_builder = PartialDiagLaplacianBuilder(
            self.size, edge_index, self.d,
            normalised=self.normalised, deg_normalised=self.deg_normalised,
            add_hp=self.add_hp, add_lp=self.add_lp, augmented=self.augmented)
        new_builder.train(self.training)
        return new_builder