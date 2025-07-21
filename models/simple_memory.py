# simple_memory.py
# Simple static memory module for temporal graph networks

import torch
import torch.nn as nn
from typing import Optional, Tuple


class SimpleTimeEncoder(nn.Module):
    """
    Simple time encoding module for compatibility.
    Note: This is mainly for interface compatibility - temporal dynamics 
    are handled by the sheaf learner.
    """
    
    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        # Simple linear projection - not used for actual temporal modeling
        self.projection = nn.Linear(1, time_dim)
        
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Simple time encoding for interface compatibility.
        
        Args:
            timestamps: [batch_size] tensor of timestamps
            
        Returns:
            Time encodings [batch_size, time_dim]
        """
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(-1)
        return self.projection(timestamps.float())


class StaticMemory(nn.Module):
    """
    Static memory module that maintains node embeddings without temporal dynamics.
    All temporal modeling is delegated to the sheaf learner.
    
    This memory module simply:
    1. Stores static embeddings for each node
    2. Updates embeddings based on messages (without temporal considerations)
    3. Provides interface compatibility with temporal graph frameworks
    """
    
    def __init__(self, 
                 num_nodes: int,
                 msg_dim: int, 
                 mem_dim: int,
                 time_dim: int,
                 device: str = 'cpu'):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.msg_dim = msg_dim
        self.mem_dim = mem_dim
        self.time_dim = time_dim
        self.device = device
        
        # Static node embeddings
        self.memory = nn.Parameter(torch.zeros(num_nodes, mem_dim))
        
        # Last update times (for interface compatibility, not used for dynamics)
        self.register_buffer('last_update_time', torch.zeros(num_nodes))
        
        # Time encoder (for interface compatibility)
        self.time_enc = SimpleTimeEncoder(time_dim)
        
        # Simple message processor (no temporal modeling)
        self.message_processor = nn.Sequential(
            nn.Linear(msg_dim, mem_dim),
            nn.ReLU(),
            nn.Linear(mem_dim, mem_dim)
        )
        
        # Simple memory updater (no GRU or temporal components)
        self.memory_updater = nn.Linear(mem_dim * 2, mem_dim)  # [old_memory, new_message] -> new_memory
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters."""
        nn.init.xavier_uniform_(self.memory.data)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, node_ids: torch.Tensor, current_time: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve static memory for specified nodes.
        
        Args:
            node_ids: [batch_size] tensor of node indices
            current_time: Optional current timestamp (ignored - for interface compatibility)
            
        Returns:
            Tuple of (memory_vectors [batch_size, mem_dim], last_update_times [batch_size])
        """
        # Simply return stored embeddings - no temporal decay
        node_memories = self.memory[node_ids]  # [batch_size, mem_dim]
        last_updates = self.last_update_time[node_ids]  # [batch_size]
        
        return node_memories, last_updates
    
    def update_state(self, 
                    src_nodes: torch.Tensor, 
                    dst_nodes: torch.Tensor, 
                    timestamps: torch.Tensor, 
                    messages: torch.Tensor):
        """
        Update memory state based on new interactions.
        No temporal dynamics - just simple embedding updates.
        
        Args:
            src_nodes: [num_edges] source node indices
            dst_nodes: [num_edges] destination node indices  
            timestamps: [num_edges] interaction timestamps (stored but not used for dynamics)
            messages: [num_edges, msg_dim] interaction messages/features
        """
        if src_nodes.numel() == 0:
            return
            
        # Process messages (no temporal considerations)
        processed_messages = self.message_processor(messages)  # [num_edges, mem_dim]
        
        # Update source nodes
        self._update_node_memories(src_nodes, processed_messages)
        
        # Update destination nodes  
        self._update_node_memories(dst_nodes, processed_messages)
        
        # Update timestamps (for interface compatibility)
        with torch.no_grad():
            self.last_update_time[src_nodes] = timestamps
            self.last_update_time[dst_nodes] = timestamps
    
    def _update_node_memories(self, 
                             node_ids: torch.Tensor, 
                             messages: torch.Tensor):
        """Update memory for specific nodes without temporal dynamics."""
        unique_nodes = torch.unique(node_ids)
        
        for node_id in unique_nodes:
            # Find all messages for this node
            mask = (node_ids == node_id)
            node_messages = messages[mask]  # [num_interactions, mem_dim]
            
            if node_messages.size(0) == 0:
                continue
            
            # Simple aggregation (mean) - no temporal weighting
            if node_messages.size(0) > 1:
                aggregated_message = node_messages.mean(dim=0)
            else:
                aggregated_message = node_messages.squeeze(0)
            
            # Simple memory update: concatenate old memory with new message
            current_memory = self.memory[node_id]
            combined = torch.cat([current_memory, aggregated_message], dim=0)
            new_memory = torch.tanh(self.memory_updater(combined))
            
            # Update stored memory
            with torch.no_grad():
                self.memory[node_id] = new_memory
    
    def reset_state(self):
        """Reset memory state (useful between epochs)."""
        with torch.no_grad():
            nn.init.xavier_uniform_(self.memory.data)
            self.last_update_time.fill_(0.0)
    
    def detach(self):
        """Detach memory from computation graph to prevent gradient accumulation."""
        self.memory.data = self.memory.data.detach()
    
    def get_memory_stats(self) -> dict:
        """Get statistics about the current memory state."""
        with torch.no_grad():
            memory_norm = torch.norm(self.memory, dim=1)
            
            return {
                'memory_mean_norm': memory_norm.mean().item(),
                'memory_max_norm': memory_norm.max().item(),
                'memory_min_norm': memory_norm.min().item(),
                'num_updated_nodes': (self.last_update_time > 0).sum().item(),
                'total_nodes': self.num_nodes,
            }


class MinimalMemory(nn.Module):
    """
    Even simpler memory module that just stores static embeddings
    and updates them with exponential moving average.
    """
    
    def __init__(self, 
                 num_nodes: int,
                 msg_dim: int, 
                 mem_dim: int,
                 time_dim: int,
                 update_rate: float = 0.1,
                 device: str = 'cpu'):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.msg_dim = msg_dim
        self.mem_dim = mem_dim
        self.time_dim = time_dim
        self.update_rate = update_rate
        self.device = device
        
        # Static node embeddings
        self.memory = nn.Parameter(torch.zeros(num_nodes, mem_dim))
        
        # Dummy time encoder for interface compatibility
        self.time_enc = nn.Linear(1, time_dim)
        
        # Simple message projection
        self.msg_projection = nn.Linear(msg_dim, mem_dim)
        
        # Initialize
        nn.init.xavier_uniform_(self.memory.data)
        
    def forward(self, node_ids: torch.Tensor, current_time: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return static embeddings."""
        memories = self.memory[node_ids]
        dummy_times = torch.zeros(len(node_ids), device=self.device)
        return memories, dummy_times
    
    def update_state(self, src_nodes: torch.Tensor, dst_nodes: torch.Tensor, 
                    timestamps: torch.Tensor, messages: torch.Tensor):
        """Simple EMA update of embeddings."""
        if src_nodes.numel() == 0:
            return
            
        # Project messages to memory dimension
        projected_msgs = self.msg_projection(messages)
        
        # Update source and destination nodes with EMA
        all_nodes = torch.cat([src_nodes, dst_nodes])
        all_messages = torch.cat([projected_msgs, projected_msgs])
        
        unique_nodes = torch.unique(all_nodes)
        
        with torch.no_grad():
            for node_id in unique_nodes:
                mask = (all_nodes == node_id)
                node_messages = all_messages[mask].mean(dim=0)
                
                # Exponential moving average update
                self.memory[node_id] = (1 - self.update_rate) * self.memory[node_id] + \
                                     self.update_rate * node_messages
    
    def reset_state(self):
        """Reset embeddings."""
        with torch.no_grad():
            nn.init.xavier_uniform_(self.memory.data)
    
    def detach(self):
        """Detach from computation graph."""
        self.memory.data = self.memory.data.detach()


# Factory function
def create_memory(memory_type: str = 'static', **kwargs):
    """
    Factory function to create memory modules.
    
    Args:
        memory_type: 'static' or 'minimal'
        **kwargs: Arguments for memory constructor
        
    Returns:
        Memory module instance
    """
    if memory_type in ['static', 'simple']:
        return StaticMemory(**kwargs)
    elif memory_type == 'minimal':
        return MinimalMemory(**kwargs)
    else:
        raise ValueError(f"Unknown memory type: {memory_type}. Use 'static' or 'minimal'")


# For backward compatibility
SimpleMemory = StaticMemory
MyMemory = StaticMemory


# Test function
def test_static_memory():
    """Test the static memory module."""
    device = 'cpu'
    num_nodes = 100
    msg_dim = 64
    mem_dim = 128
    time_dim = 32
    
    # Test static memory
    memory = StaticMemory(num_nodes, msg_dim, mem_dim, time_dim, device)
    
    # Test forward pass
    node_ids = torch.randint(0, num_nodes, (10,))
    memories, last_updates = memory(node_ids)
    print(f"Static memory - Memory shape: {memories.shape}")
    
    # Test update
    src_nodes = torch.randint(0, num_nodes, (20,))
    dst_nodes = torch.randint(0, num_nodes, (20,))
    timestamps = torch.rand(20) * 100  # These are stored but don't affect dynamics
    messages = torch.randn(20, msg_dim)
    
    memory.update_state(src_nodes, dst_nodes, timestamps, messages)
    
    # Test retrieval after update
    memories_after, _ = memory(node_ids)
    print(f"Memory after update shape: {memories_after.shape}")
    print(f"Memory changed: {not torch.equal(memories, memories_after)}")
    
    # Test minimal memory
    minimal_memory = MinimalMemory(num_nodes, msg_dim, mem_dim, time_dim, device=device)
    memories_minimal, _ = minimal_memory(node_ids)
    print(f"Minimal memory shape: {memories_minimal.shape}")
    
    print("Static memory test passed!")


if __name__ == "__main__":
    test_static_memory()