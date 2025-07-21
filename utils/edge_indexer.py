# edge_indexer.py
# Minimal imports for edge indexing functionality

from abc import ABC, abstractmethod
from typing import Optional


class EdgeIndexer(ABC):
    """Abstract base class for edge indexing schemes."""
    
    @abstractmethod
    def get_edge_id(self, src: int, dst: int, timestamp: Optional[float] = None) -> int:
        """Get edge ID for given source, destination, and optional timestamp."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset the indexer state."""
        pass


class UniqueEdgeIndexer(EdgeIndexer):
    """Index edges by unique (src, dst) pairs."""
    
    def __init__(self, max_edges: int = 100000):
        self.max_edges = max_edges
        self.edge_id_map = {}  # Maps (src, dst) -> edge_id
        self.next_edge_id = 0
    
    def get_edge_id(self, src: int, dst: int, timestamp: Optional[float] = None) -> int:
        edge_key = (src, dst)
        if edge_key not in self.edge_id_map:
            if self.next_edge_id >= self.max_edges:
                raise RuntimeError(f"Exceeded maximum edges ({self.max_edges})")
            self.edge_id_map[edge_key] = self.next_edge_id
            self.next_edge_id += 1
        return self.edge_id_map[edge_key]
    
    def reset(self):
        self.edge_id_map = {}
        self.next_edge_id = 0


class TemporalEdgeIndexer(EdgeIndexer):
    """Index edges by (src, dst, time_bucket) for temporal granularity."""
    
    def __init__(self, max_edges: int = 100000, time_granularity: float = 1.0):
        self.max_edges = max_edges
        self.time_granularity = time_granularity
        self.edge_id_map = {}  # Maps (src, dst, time_bucket) -> edge_id
        self.next_edge_id = 0
    
    def get_edge_id(self, src: int, dst: int, timestamp: Optional[float] = None) -> int:
        if timestamp is None:
            time_bucket = 0
        else:
            time_bucket = int(timestamp // self.time_granularity)
        
        edge_key = (src, dst, time_bucket)
        if edge_key not in self.edge_id_map:
            if self.next_edge_id >= self.max_edges:
                raise RuntimeError(f"Exceeded maximum edges ({self.max_edges})")
            self.edge_id_map[edge_key] = self.next_edge_id
            self.next_edge_id += 1
        return self.edge_id_map[edge_key]
    
    def reset(self):
        self.edge_id_map = {}
        self.next_edge_id = 0