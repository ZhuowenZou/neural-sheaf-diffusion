# temporal_sheaf_learner.py
# Minimal imports for temporal sheaf learning

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple

# Import the required components
from utils.edge_indexer import EdgeIndexer
from models.edge_aware_mamba import EdgeAwareMamba


class TemporalSheafLearner(ABC):
    """Abstract base class for temporal sheaf learning modules."""
    
    def __init__(self):
        self.L = None
    
    @abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                timestamps: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for temporal sheaf learning.
        
        Args:
            x: Node features [num_nodes, feature_dim]
            edge_index: Edge indices [2, num_edges]
            timestamps: Edge timestamps [num_edges]
            edge_attr: Edge attributes/messages [num_edges, msg_dim]
            
        Returns:
            Sheaf maps conforming to out_shape
        """
        pass
    
    @abstractmethod
    def reset_states(self):
        """Reset temporal states."""
        pass
    
    def set_L(self, weights):
        """Set Laplacian weights (compatibility with SheafLearner interface)."""
        self.L = weights.clone().detach()


class MambaTemporalSheafLearner(TemporalSheafLearner, nn.Module):
    """
    Temporal sheaf learner using Mamba with edge-specific states.
    """
    
    def __init__(self, 
                 msg_dim: int,
                 out_shape: Tuple[int, ...],
                 edge_indexer: EdgeIndexer,
                 d_model: int = 128,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 sheaf_act: str = "tanh",
                 device: str = 'cpu',
                 **mamba_kwargs):
        super().__init__()
        self.msg_dim = msg_dim
        self.out_shape = out_shape
        self.edge_indexer = edge_indexer
        self.d_model = d_model
        self.device = device
        
        # Input projection from message dimension to model dimension
        self.input_proj = nn.Linear(msg_dim, d_model, device=device)
        
        # Shared Mamba model
        self.mamba = EdgeAwareMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            device=device,
            **mamba_kwargs
        )
        
        # Output projection to sheaf map parameters
        out_dim = int(torch.prod(torch.tensor(out_shape)))
        self.output_proj = nn.Linear(d_model, out_dim, device=device)
        
        # Edge-specific state storage
        self.edge_states = {}  # Maps edge_id -> (conv_state, ssm_state)
        
        # Activation function
        if sheaf_act == 'id':
            self.act = lambda x: x
        elif sheaf_act == 'tanh':
            self.act = torch.tanh
        elif sheaf_act == 'elu':
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported activation {sheaf_act}")
    
    def _get_or_create_edge_states(self, edge_id: int, batch_size: int = 1):
        """Get or create states for a specific edge."""
        if edge_id not in self.edge_states:
            conv_state, ssm_state = self.mamba.allocate_inference_cache(
                batch_size=batch_size,
                max_seqlen=1,
                dtype=self.mamba.conv1d.weight.dtype
            )
            self.edge_states[edge_id] = (conv_state, ssm_state)
        return self.edge_states[edge_id]
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                timestamps: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Process edge attributes through Mamba with edge-specific states.
        
        Args:
            x: Node features [num_nodes, feature_dim] - not used directly
            edge_index: Edge indices [2, num_edges]
            timestamps: Edge timestamps [num_edges]
            edge_attr: Edge attributes/messages [num_edges, msg_dim]
            
        Returns:
            Sheaf maps [num_edges, *out_shape]
        """
        num_edges = edge_attr.size(0)
        if num_edges == 0:
            # Handle empty batch
            out_dim = int(torch.prod(torch.tensor(self.out_shape)))
            return torch.zeros(0, *self.out_shape, device=self.device)
        
        outputs = []
        
        # Project edge attributes to model dimension
        projected_attr = self.input_proj(edge_attr)  # [num_edges, d_model]
        
        # Process each edge with its own state
        for i in range(num_edges):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            timestamp = timestamps[i].item() if timestamps.numel() > 0 else None
            
            # Get edge ID using the indexer
            edge_id = self.edge_indexer.get_edge_id(src, dst, timestamp)
            
            # Get edge-specific states
            conv_state, ssm_state = self._get_or_create_edge_states(edge_id)
            
            # Process single edge feature through Mamba
            edge_input = projected_attr[i:i+1].unsqueeze(1)  # [1, 1, d_model]
            
            # Use Mamba step function for sequential processing
            mamba_output, conv_state, ssm_state = self.mamba.step(
                edge_input, conv_state, ssm_state
            )
            
            # Update stored states
            self.edge_states[edge_id] = (conv_state, ssm_state)
            
            # Project to output dimension and apply activation
            sheaf_params = self.output_proj(mamba_output.squeeze(1))  # [1, out_dim]
            sheaf_params = self.act(sheaf_params)
            
            outputs.append(sheaf_params)
        
        # Combine outputs and reshape
        output = torch.cat(outputs, dim=0)  # [num_edges, out_dim]
        
        if len(self.out_shape) == 1:
            return output.view(-1, self.out_shape[0])
        elif len(self.out_shape) == 2:
            return output.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return output.view(-1, *self.out_shape)
    
    def reset_states(self):
        """Reset all edge states and indexer."""
        self.edge_states = {}
        self.edge_indexer.reset()