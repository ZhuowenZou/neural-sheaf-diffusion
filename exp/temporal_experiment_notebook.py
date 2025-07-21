# notebook_experiment.py
# Notebook-friendly version of the temporal sheaf diffusion experiment

import timeit
import torch
import torch.nn as nn
from torch_geometric.loader import TemporalDataLoader
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset
from tgb.nodeproppred.evaluate import Evaluator

# Import utilities with proper path handling
import os, sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../models'))
sys.path.insert(0, os.path.abspath('./models'))
sys.path.insert(0, os.path.abspath('.'))

# Try importing with different possible paths
try:
    from utils.neighbor_loader import LastNeighborLoader
except ImportError:
    try:
        from neighbor_loader import LastNeighborLoader
    except ImportError:
        # Placeholder if not available
        class LastNeighborLoader:
            def __init__(self, num_nodes, size, device):
                self.num_nodes = num_nodes
                self.size = size
                self.device = device
            
            def reset_state(self):
                pass
            
            def insert(self, src, dst):
                pass
            
            def __call__(self, n_id):
                # Placeholder implementation
                return n_id, torch.empty((2, 0), device=self.device), torch.empty(0, device=self.device)


# ============= Simple Memory Implementation =============
class SimpleMemory(nn.Module):
    """
    Simple temporal memory module that maintains node memories and updates them
    based on temporal interactions. Keeps temporal dynamics minimal as requested.
    """
    
    def __init__(self, 
                 num_nodes: int,
                 msg_dim: int, 
                 mem_dim: int,
                 time_dim: int):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.msg_dim = msg_dim
        self.mem_dim = mem_dim
        self.time_dim = time_dim
        
        # Node memory storage - simple learnable embeddings
        self.memory = nn.Parameter(torch.randn(num_nodes, mem_dim) * 0.1)
        
        # Simple time encoder for interface compatibility
        self.time_enc = nn.Linear(1, time_dim)
        
        # Message processor
        self.message_processor = nn.Linear(msg_dim, mem_dim)
        
        # Memory update gate
        self.update_gate = nn.Linear(mem_dim * 2, mem_dim)
        
    def forward(self, node_ids: torch.Tensor):
        """
        Retrieve memory for specified nodes.
        
        Args:
            node_ids: [batch_size] tensor of node indices
            
        Returns:
            Tuple of (memory_vectors [batch_size, mem_dim], dummy_last_update_times [batch_size])
        """
        batch_size = node_ids.size(0)
        
        # Get node memories
        node_memories = self.memory[node_ids]  # [batch_size, mem_dim]
        
        # Return dummy last update times for interface compatibility
        dummy_times = torch.zeros(batch_size, device=node_ids.device)
        
        return node_memories, dummy_times
    
    def update_state(self, 
                    src_nodes: torch.Tensor, 
                    dst_nodes: torch.Tensor, 
                    timestamps: torch.Tensor, 
                    messages: torch.Tensor):
        """
        Update memory state based on new interactions.
        Simple updates without complex temporal dynamics.
        
        Args:
            src_nodes: [num_edges] source node indices
            dst_nodes: [num_edges] destination node indices  
            timestamps: [num_edges] interaction timestamps
            messages: [num_edges, msg_dim] interaction messages/features
        """
        if src_nodes.numel() == 0:
            return
            
        # Process messages
        processed_messages = self.message_processor(messages)  # [num_edges, mem_dim]
        
        # Simple update for both source and destination nodes
        all_nodes = torch.cat([src_nodes, dst_nodes])
        all_messages = torch.cat([processed_messages, processed_messages * 0.5])  # Dest gets scaled message
        
        # Update unique nodes
        unique_nodes = torch.unique(all_nodes)
        
        for node_id in unique_nodes:
            mask = (all_nodes == node_id)
            node_messages = all_messages[mask]
            
            if node_messages.size(0) > 0:
                # Aggregate messages for this node
                aggregated = node_messages.mean(dim=0)
                
                # Simple gated update
                current_mem = self.memory[node_id]
                combined = torch.cat([current_mem, aggregated])
                gate = torch.sigmoid(self.update_gate(combined))
                
                # Update memory
                with torch.no_grad():
                    self.memory[node_id] = gate * aggregated + (1 - gate) * current_mem
    
    def reset_state(self):
        """Reset memory state."""
        with torch.no_grad():
            nn.init.normal_(self.memory.data, mean=0.0, std=0.1)
    
    def detach(self):
        """Detach memory from computation graph."""
        self.memory.data = self.memory.data.detach()


# ============= Configuration =============
class ExperimentConfig:
    """Configuration class for easy parameter management in notebooks."""
    
    def __init__(self):
        # Dataset parameters
        self.data = 'tgbn-trade'
        self.seed = 42
        
        # Training parameters
        self.bs = 200
        self.epochs = 50
        self.lr = 1e-3
        
        # Model architecture
        self.mem_dim = 100
        self.time_dim = 100
        self.emb_dim = 100
        self.d = 4
        self.layers = 3
        self.hidden_channels = 32
        
        # Temporal Mamba parameters
        self.mamba_d_model = 128
        self.mamba_d_state = 16
        self.max_edges = 100000
        
        # Model type
        self.sheaf_type = 'diag'  # 'diag', 'bundle', 'general'
        self.use_simple_gnn = False  # Use simple GNN instead of sheaf diffusion
        
        # Additional options
        self.dropout = 0.1
        self.input_dropout = 0.1
        self.use_act = True
        self.sheaf_act = 'tanh'
        self.second_linear = False
        self.linear = False
        self.normalised = True
        self.deg_normalised = False
        self.add_hp = False
        self.add_lp = False
        self.left_weights = False
        self.right_weights = False
    
    def __repr__(self):
        """Pretty print configuration."""
        config_str = "ExperimentConfig:\n"
        for key, value in self.__dict__.items():
            config_str += f"  {key}: {value}\n"
        return config_str


# ============= Node Predictor =============
class SimpleNodePredictor(nn.Module):
    """Simple node predictor."""
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim // 2, out_dim)
        )
    
    def forward(self, x):
        return self.predictor(x)


# ============= Helper Functions =============
def process_edges(src, dst, t, msg, memory, neighbor_loader):
    """Update memory & neighbor state for each new interaction."""
    if src.numel() == 0:
        return
    memory.update_state(src, dst, t, msg)
    neighbor_loader.insert(src, dst)


def setup_experiment(config):
    """Setup experiment components based on configuration."""
    # Set random seed with better error handling
    try:
        torch.manual_seed(config.seed)
    except RuntimeError as e:
        print(f"Warning: Could not set CUDA seed due to error: {e}")
        print("Falling back to CPU-only execution...")
        # Clear CUDA cache and force CPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Set seed for CPU only
        import random
        import numpy as np
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
    
    # Force CPU if CUDA has issues, otherwise auto-detect
    try:
        if hasattr(config, 'force_cpu') and config.force_cpu:
            device = 'cpu'
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # Test CUDA with a simple operation
            if device == 'cuda':
                test_tensor = torch.randn(2, 2).cuda()
                _ = test_tensor + 1  # Simple operation to test CUDA
                del test_tensor
                torch.cuda.empty_cache()
    except Exception as e:
        print(f"CUDA test failed: {e}")
        print("Falling back to CPU execution...")
        device = 'cpu'
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"Using device: {device}")
    print(f"Loading dataset: {config.data}")
    
    try:
        # Load dataset
        dataset = PyGNodePropPredDataset(name=config.data, root='datasets')
        data = dataset.get_TemporalData()
        
        # Move to device safely
        if device == 'cuda':
            try:
                data = data.to(device)
            except Exception as e:
                print(f"Failed to move data to CUDA: {e}")
                print("Using CPU instead...")
                device = 'cpu'
                data = data.to(device)
        else:
            data = data.to(device)
        
        # Split & loaders
        train = data[dataset.train_mask]
        val = data[dataset.val_mask]
        test = data[dataset.test_mask]
        
        train_loader = TemporalDataLoader(train, batch_size=config.bs, shuffle=True)
        val_loader = TemporalDataLoader(val, batch_size=config.bs)
        test_loader = TemporalDataLoader(test, batch_size=config.bs)
        
        loaders = (train_loader, val_loader, test_loader)
        
        print(f"Dataset info:")
        print(f"  Nodes: {data.num_nodes}")
        print(f"  Edges: {data.num_edges}")
        print(f"  Classes: {dataset.num_classes}")
        print(f"  Train edges: {len(train)}")
        print(f"  Val edges: {len(val)}")
        print(f"  Test edges: {len(test)}")
        
        return dataset, data, loaders, device
        
    except Exception as e:
        print(f"Error during dataset setup: {e}")
        print("This might be due to dataset download issues or CUDA problems.")
        raise


def build_models(config, dataset, data, device):
    """Build model components based on configuration."""
    # Extract dataset info - use actual data first, then config estimates
    graph_size = data.num_nodes
    input_dim = data.x.size(-1) if hasattr(data, 'x') and data.x is not None else config.emb_dim
    output_dim = dataset.num_classes
    msg_dim = data.msg.size(-1) if hasattr(data, 'msg') else 128
    
    # Get dataset statistics for better initialization
    dataset_stats = get_dataset_stats(config.data)
    
    print(f"Model configuration:")
    print(f"  Dataset: {config.data}")
    print(f"  Actual nodes: {graph_size} (expected: {dataset_stats['num_nodes']})")
    print(f"  Actual edges: {data.num_edges} (expected: {dataset_stats['num_edges']})")
    print(f"  Input dim: {input_dim}")
    print(f"  Output dim: {output_dim}")
    print(f"  Message dim: {msg_dim} (expected: {dataset_stats['msg_dim']})")
    print(f"  Memory dim: {config.mem_dim}")
    
    # Verify our estimates vs actual data
    if abs(graph_size - dataset_stats['num_nodes']) > graph_size * 0.1:
        print(f"  ⚠️  Warning: Node count differs significantly from expected")
    if abs(data.num_edges - dataset_stats['num_edges']) > data.num_edges * 0.1:
        print(f"  ⚠️  Warning: Edge count differs significantly from expected")
    
    # Build memory - use actual dataset dimensions
    memory = SimpleMemory(
        num_nodes=graph_size,  # Use actual node count
        msg_dim=msg_dim,       # Use actual message dimension
        mem_dim=config.mem_dim,
        time_dim=config.time_dim,
    ).to(device)
    
    # Adjust max_edges based on actual data if needed
    actual_max_edges = min(config.max_edges, data.num_edges // 2)  # Don't exceed half the edges
    if actual_max_edges != config.max_edges:
        print(f"  Adjusted max_edges: {config.max_edges} → {actual_max_edges}")
        config.max_edges = actual_max_edges
    
    # Build GNN - try to import temporal sheaf models, fallback to simple GNN
    gnn = None
    
    if not config.use_simple_gnn:
        try:
            # Try to import from various possible locations
            temporal_models = None
            for path in ['temporal_sheaf_diffusion', 'models.temporal_sheaf_diffusion', 
                        '../models/temporal_sheaf_diffusion', './models/temporal_sheaf_diffusion']:
                try:
                    temporal_models = __import__(path, fromlist=['TemporalDiscreteDiagSheafDiffusion'])
                    break
                except ImportError:
                    continue
            
            if temporal_models is not None:
                # Create model args with actual dataset dimensions
                model_args = {
                    'd': config.d,
                    'device': device,
                    'graph_size': graph_size,  # Use actual graph size
                    'layers': config.layers,
                    'hidden_channels': config.hidden_channels,
                    'input_dim': input_dim,
                    'output_dim': output_dim,
                    'msg_dim': msg_dim,  # Use actual message dimension
                    'dropout': config.dropout,
                    'input_dropout': config.input_dropout,
                    'use_act': config.use_act,
                    'sheaf_act': config.sheaf_act,
                    'second_linear': config.second_linear,
                    'linear': config.linear,
                    'normalised': config.normalised,
                    'deg_normalised': config.deg_normalised,
                    'add_hp': config.add_hp,
                    'add_lp': config.add_lp,
                    'left_weights': config.left_weights,
                    'right_weights': config.right_weights,
                    'mamba_d_model': config.mamba_d_model,
                    'mamba_d_state': config.mamba_d_state,
                    'max_edges': config.max_edges,
                    'orth': 'matrix_exp',
                }
                
                # Create dummy edge index
                dummy_edge_index = torch.tensor([[0, 1], [1, 0]], device=device)
                
                # Select model type
                if config.sheaf_type == 'diag':
                    gnn = temporal_models.TemporalDiscreteDiagSheafDiffusion(dummy_edge_index, model_args)
                elif config.sheaf_type == 'bundle':
                    gnn = temporal_models.TemporalDiscreteBundleSheafDiffusion(dummy_edge_index, model_args)
                elif config.sheaf_type == 'general':
                    gnn = temporal_models.TemporalDiscreteGeneralSheafDiffusion(dummy_edge_index, model_args)
                
                gnn = gnn.to(device)
                print(f"  Using temporal sheaf diffusion: {config.sheaf_type}")
                
        except Exception as e:
            print(f"  Warning: Could not load temporal sheaf models: {e}")
            gnn = None
    
    # Fallback to simple GNN
    if gnn is None:
        print(f"  Using simple temporal GNN fallback")
        gnn = SimpleTemporalGNN(
            input_dim=config.mem_dim,
            hidden_dim=config.hidden_channels,
            output_dim=config.emb_dim,
            msg_dim=msg_dim,  # Use actual message dimension
            num_layers=config.layers
        ).to(device)
    
    # Node predictor
    pred_input_dim = config.emb_dim if gnn is not None else config.hidden_channels
    node_pred = SimpleNodePredictor(
        in_dim=pred_input_dim,
        out_dim=output_dim
    ).to(device)
    
    # Other components
    neighbor_loader = LastNeighborLoader(graph_size, size=10, device=device)  # Use actual graph size
    evaluator = Evaluator(name=config.data)
    
    # Count parameters and estimate memory
    if gnn is not None:
        total_params = sum(p.numel() for p in gnn.parameters() if p.requires_grad)
        memory_params = sum(p.numel() for p in memory.parameters() if p.requires_grad)
        pred_params = sum(p.numel() for p in node_pred.parameters() if p.requires_grad)
        
        print(f"  Model parameters:")
        print(f"    Memory: {memory_params:,}")
        print(f"    GNN: {total_params:,}")
        print(f"    Predictor: {pred_params:,}")
        print(f"    Total: {total_params + memory_params + pred_params:,}")
        
        # Estimate memory usage
        estimated_mb = estimate_memory_usage(config, dataset_stats)
        print(f"  Estimated memory usage: {estimated_mb:.1f} MB")
    
    return memory, gnn, node_pred, neighbor_loader, evaluator


def train_one_epoch(model_parts, train_loader, dataset, optimizer, criterion, device):
    """Training function for one epoch."""
    memory, gnn, pred, neighbor_loader, evaluator = model_parts
    memory.train(); gnn.train(); pred.train()
    memory.reset_state(); neighbor_loader.reset_state()
    
    # Reset temporal states
    if hasattr(gnn, 'reset_temporal_states'):
        gnn.reset_temporal_states()

    total_loss, total_score, count = 0, 0, 0
    label_t = dataset.get_label_time()

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        if t[-1] > label_t:
            label_ts, label_nodes, labels = dataset.get_node_label(t[-1])
            label_t = dataset.get_label_time()
            label_nodes = label_nodes.to(device)

            mask = t < label_t
            process_edges(src[mask], dst[mask], t[mask], msg[mask], memory, neighbor_loader)
            src, dst, t, msg = src[~mask], dst[~mask], t[~mask], msg[~mask]

            n_id = label_nodes
            n_ids, mem_edge_index, e_id = neighbor_loader(n_id)
            assoc = torch.empty(n_ids.size(0), dtype=torch.long, device=device)
            assoc[n_ids] = torch.arange(n_ids.size(0), device=device)

            z, last_update = memory(n_ids)
            
            # Use GNN
            z = gnn(
                z,                          # Node features
                mem_edge_index,             # Edge connectivity  
                batch.t[e_id].to(device),   # Edge times
                batch.msg[e_id].to(device), # Edge attributes
            )
            z = z[assoc[n_id]]

            out = pred(z)
            loss = criterion(out, labels.to(device))
            loss.backward(); optimizer.step()
            total_loss += loss.item()

            np_pred = out.detach().cpu().numpy()
            np_true = labels.cpu().numpy()
            res = evaluator.eval({
                'y_true': np_true,
                'y_pred': np_pred,
                'eval_metric': [dataset.eval_metric]
            })
            total_score += res[dataset.eval_metric]
            count += 1

        process_edges(src, dst, t, msg, memory, neighbor_loader)
        memory.detach()

    return {
        'loss': total_loss / max(1, count),
        dataset.eval_metric: total_score / max(1, count)
    }


@torch.no_grad()
def evaluate(model_parts, loader, dataset, device):
    """Evaluation function."""
    memory, gnn, pred, neighbor_loader, evaluator = model_parts
    memory.eval(); gnn.eval(); pred.eval()

    total_score, count = 0, 0
    label_t = dataset.get_label_time()

    for batch in loader:
        batch = batch.to(device)
        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        if t[-1] > label_t:
            label_tuple = dataset.get_node_label(t[-1])
            if label_tuple is None:
                break
            label_ts, label_nodes, labels = label_tuple
            label_t = dataset.get_label_time()
            label_nodes = label_nodes.to(device)

            mask = t < label_t
            process_edges(src[mask], dst[mask], t[mask], msg[mask], memory, neighbor_loader)
            src, dst, t, msg = src[~mask], dst[~mask], t[~mask], msg[~mask]

            n_id = label_nodes
            n_ids, mem_edge_index, e_id = neighbor_loader(n_id)
            assoc = torch.empty(n_ids.size(0), dtype=torch.long, device=device)
            assoc[n_ids] = torch.arange(n_ids.size(0), device=device)

            z, last_update = memory(n_ids)
            
            # Use GNN
            z = gnn(
                z,                          # Node features
                mem_edge_index,             # Edge connectivity
                batch.t[e_id].to(device),   # Edge times
                batch.msg[e_id].to(device), # Edge attributes
            )
            z = z[assoc[n_id]]

            out = pred(z)
            np_pred = out.detach().cpu().numpy()
            np_true = labels.cpu().numpy()
            res = evaluator.eval({
                'y_true': np_true,
                'y_pred': np_pred,
                'eval_metric': [dataset.eval_metric]
            })
            total_score += res[dataset.eval_metric]
            count += 1

        process_edges(src, dst, t, msg, memory, neighbor_loader)

    return total_score / max(1, count)


def run_experiment(config, verbose=True):
    """Run the complete experiment with given configuration."""
    if verbose:
        print("="*50)
        print("TEMPORAL GRAPH LEARNING EXPERIMENT")
        print("="*50)
        print(config)
    
    # Setup
    dataset, data, loaders, device = setup_experiment(config)
    train_loader, val_loader, test_loader = loaders
    
    # Build models
    memory, gnn, node_pred, neighbor_loader, evaluator = build_models(
        config, dataset, data, device
    )
    
    # Setup training
    optimizer = torch.optim.Adam(
        list(memory.parameters()) +
        list(gnn.parameters()) +
        list(node_pred.parameters()),
        lr=config.lr
    )
    criterion = torch.nn.CrossEntropyLoss()
    model_parts = (memory, gnn, node_pred, neighbor_loader, evaluator)
    
    if verbose:
        print(f"\nStarting training for {config.epochs} epochs...")
        print("-" * 50)
    
    # Training loop
    best_val, best_test = 0.0, 0.0
    results = {
        'train_metrics': [],
        'val_scores': [],
        'test_scores': [],
        'epoch_times': []
    }
    
    for epoch in range(1, config.epochs + 1):
        start = timeit.default_timer()
        
        train_metrics = train_one_epoch(
            model_parts, train_loader, dataset,
            optimizer, criterion, device
        )
        val_score = evaluate(model_parts, val_loader, dataset, device)
        test_score = evaluate(model_parts, test_loader, dataset, device)
        
        epoch_time = timeit.default_timer() - start
        
        # Store results
        results['train_metrics'].append(train_metrics)
        results['val_scores'].append(val_score)
        results['test_scores'].append(test_score)
        results['epoch_times'].append(epoch_time)
        
        if verbose:
            print(f"[Epoch {epoch:02d}] "
                  f"Train: {train_metrics} | "
                  f"Val: {val_score:.4f} | "
                  f"Test: {test_score:.4f} | "
                  f"Time: {epoch_time:.1f}s")

        if val_score > best_val:
            best_val = val_score
            best_test = test_score
    
    results['best_val'] = best_val
    results['best_test'] = best_test
    
    if verbose:
        print("-" * 50)
        print(f"Best Val: {best_val:.4f}  Corresponding Test: {best_test:.4f}")
        print("="*50)
    
    return results


# ============= Dataset Statistics =============
DATASET_STATS = {
    'tgbn-trade': {
        'num_nodes': 255,
        'num_edges': 468245,
        'num_classes': 2,  # Binary classification
        'msg_dim': 128,    # Typical message dimension
        'avg_degree': 468245 * 2 / 255,  # ~3673 edges per node on average
        'density': 'high',  # High density graph
        'temporal_span': 'long',  # Long temporal sequences
    },
    'tgbn-reddit': {
        'num_nodes': 10984,
        'num_edges': 672447,
        'num_classes': 2,
        'msg_dim': 172,
        'avg_degree': 672447 * 2 / 10984,  # ~122 edges per node
        'density': 'medium',
        'temporal_span': 'medium',
    },
    'tgbn-token': {
        'num_nodes': 13207,
        'num_edges': 403370,
        'num_classes': 2,
        'msg_dim': 172,
        'avg_degree': 403370 * 2 / 13207,  # ~61 edges per node
        'density': 'medium',
        'temporal_span': 'medium',
    },
    # Add more datasets as needed
}

def get_dataset_stats(dataset_name):
    """Get dataset statistics, with fallback for unknown datasets."""
    if dataset_name in DATASET_STATS:
        return DATASET_STATS[dataset_name]
    else:
        print(f"Warning: Unknown dataset {dataset_name}, using default stats")
        return {
            'num_nodes': 1000,
            'num_edges': 50000,
            'num_classes': 2,
            'msg_dim': 128,
            'avg_degree': 100,
            'density': 'medium',
            'temporal_span': 'medium',
        }

def print_dataset_info(dataset_name):
    """Print detailed dataset information."""
    stats = get_dataset_stats(dataset_name)
    print(f"\n=== Dataset: {dataset_name} ===")
    print(f"  Nodes: {stats['num_nodes']:,}")
    print(f"  Edges: {stats['num_edges']:,}")
    print(f"  Classes: {stats['num_classes']}")
    print(f"  Avg degree: {stats['avg_degree']:.1f}")
    print(f"  Density: {stats['density']}")
    print(f"  Temporal span: {stats['temporal_span']}")
    return stats

def get_optimal_model_size(dataset_stats, complexity='medium'):
    """
    Calculate optimal model dimensions based on dataset characteristics.
    
    Args:
        dataset_stats: Dictionary with dataset statistics
        complexity: 'small', 'medium', 'large' - controls model complexity
    
    Returns:
        Dictionary with optimal model dimensions
    """
    num_nodes = dataset_stats['num_nodes']
    num_edges = dataset_stats['num_edges']
    avg_degree = dataset_stats['avg_degree']
    density = dataset_stats['density']
    
    # Base dimensions scaled by dataset size
    if complexity == 'small':
        base_multiplier = 0.5
        max_edges_factor = 0.1
    elif complexity == 'medium':
        base_multiplier = 1.0
        max_edges_factor = 0.2
    else:  # large
        base_multiplier = 1.5
        max_edges_factor = 0.3
    
    # Scale dimensions based on number of nodes
    if num_nodes < 500:
        node_scale = 1.0
    elif num_nodes < 5000:
        node_scale = 1.2
    else:
        node_scale = 1.5
    
    # Scale based on graph density
    if density == 'high':
        density_scale = 1.3
    elif density == 'medium':
        density_scale = 1.0
    else:
        density_scale = 0.8
    
    # Calculate optimal dimensions
    mem_dim = int(64 * base_multiplier * node_scale)
    time_dim = int(32 * base_multiplier)
    emb_dim = int(64 * base_multiplier * node_scale)
    
    # Sheaf dimension based on node connectivity
    if avg_degree > 1000:  # High connectivity like tgbn-trade
        d = max(2, int(6 * base_multiplier))
    elif avg_degree > 100:
        d = max(2, int(4 * base_multiplier))
    else:
        d = max(2, int(2 * base_multiplier))
    
    # Hidden channels scaled by complexity and connectivity
    hidden_channels = int(16 * base_multiplier * density_scale)
    
    # Mamba parameters scaled by temporal complexity
    mamba_d_model = int(64 * base_multiplier * node_scale)
    mamba_d_state = int(8 * base_multiplier)
    
    # Max edges for temporal tracking
    max_edges = int(num_edges * max_edges_factor)
    max_edges = max(1000, min(max_edges, 100000))  # Reasonable bounds
    
    # Layers based on dataset complexity
    if num_nodes > 10000 or avg_degree > 500:
        layers = int(3 * base_multiplier)
    else:
        layers = int(2 * base_multiplier)
    
    layers = max(1, min(layers, 5))  # Reasonable bounds
    
    return {
        'mem_dim': mem_dim,
        'time_dim': time_dim,
        'emb_dim': emb_dim,
        'd': d,
        'hidden_channels': hidden_channels,
        'layers': layers,
        'mamba_d_model': mamba_d_model,
        'mamba_d_state': mamba_d_state,
        'max_edges': max_edges,
    }

def get_optimal_training_params(dataset_stats, complexity='medium'):
    """Get optimal training parameters based on dataset characteristics."""
    num_edges = dataset_stats['num_edges']
    density = dataset_stats['density']
    
    # Batch size based on dataset size and memory constraints
    if num_edges > 400000:  # Large datasets like tgbn-trade
        if complexity == 'small':
            bs = 100
        elif complexity == 'medium':
            bs = 150
        else:
            bs = 200
    elif num_edges > 100000:  # Medium datasets
        if complexity == 'small':
            bs = 150
        elif complexity == 'medium':
            bs = 200
        else:
            bs = 250
    else:  # Small datasets
        bs = 200
    
    # Learning rate based on model complexity and dataset density
    if density == 'high':
        lr = 5e-4  # Slower for dense graphs
    else:
        lr = 1e-3  # Standard rate
    
    # Epochs based on dataset size (larger datasets need fewer epochs typically)
    if complexity == 'small':
        epochs = 5
    elif complexity == 'medium':
        if num_edges > 400000:
            epochs = 15  # Fewer epochs for large datasets
        else:
            epochs = 20
    else:  # large
        if num_edges > 400000:
            epochs = 25
        else:
            epochs = 50
    
    return {
        'bs': bs,
        'lr': lr,
        'epochs': epochs,
    }
# ============= Dataset-Aware Configuration =============
def get_dataset_optimized_config(dataset_name, complexity='medium'):
    """
    Create configuration optimized for specific dataset characteristics.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'tgbn-trade')
        complexity: 'small', 'medium', 'large' - controls model complexity
    
    Returns:
        ExperimentConfig instance optimized for the dataset
    """
    # Get dataset statistics
    dataset_stats = get_dataset_stats(dataset_name)
    print_dataset_info(dataset_name)
    
    # Calculate optimal model dimensions
    model_dims = get_optimal_model_size(dataset_stats, complexity)
    training_params = get_optimal_training_params(dataset_stats, complexity)
    
    # Create configuration
    config = ExperimentConfig()
    
    # Dataset info
    config.data = dataset_name
    config.seed = 42
    config.force_cpu = False
    
    # Apply optimal training parameters
    config.bs = training_params['bs']
    config.lr = training_params['lr']
    config.epochs = training_params['epochs']
    
    # Apply optimal model dimensions
    config.mem_dim = model_dims['mem_dim']
    config.time_dim = model_dims['time_dim']
    config.emb_dim = model_dims['emb_dim']
    config.d = model_dims['d']
    config.layers = model_dims['layers']
    config.hidden_channels = model_dims['hidden_channels']
    
    # Mamba parameters
    config.mamba_d_model = model_dims['mamba_d_model']
    config.mamba_d_state = model_dims['mamba_d_state']
    config.max_edges = model_dims['max_edges']
    
    # Model selection based on dataset characteristics
    if dataset_stats['density'] == 'high' and dataset_stats['num_nodes'] < 1000:
        # For small, dense graphs like tgbn-trade, try sheaf diffusion
        config.use_simple_gnn = False
        config.sheaf_type = 'diag'
        print(f"Selected: Diagonal Sheaf Diffusion (good for dense, small graphs)")
    elif dataset_stats['avg_degree'] > 200:
        # High degree graphs benefit from sheaf diffusion
        config.use_simple_gnn = False
        config.sheaf_type = 'bundle'
        print(f"Selected: Bundle Sheaf Diffusion (good for high-degree graphs)")
    else:
        # Default to simple GNN for reliability
        config.use_simple_gnn = True
        print(f"Selected: Simple Temporal GNN (reliable fallback)")
    
    # Additional options
    config.dropout = 0.1
    config.input_dropout = 0.1
    config.use_act = True
    config.sheaf_act = 'tanh'
    config.second_linear = False
    config.linear = False
    config.normalised = True
    config.deg_normalised = False
    config.add_hp = False
    config.add_lp = False
    config.left_weights = False
    config.right_weights = False
    
    print(f"\n=== Optimized Configuration ({complexity}) ===")
    print(f"  Model: mem_dim={config.mem_dim}, d={config.d}, layers={config.layers}")
    print(f"  Training: bs={config.bs}, lr={config.lr}, epochs={config.epochs}")
    print(f"  Mamba: d_model={config.mamba_d_model}, max_edges={config.max_edges}")
    print(f"  Memory estimate: ~{estimate_memory_usage(config, dataset_stats):.1f} MB")
    
    return config

def estimate_memory_usage(config, dataset_stats):
    """Estimate memory usage for the configuration."""
    num_nodes = dataset_stats['num_nodes']
    
    # Memory components
    node_memory = num_nodes * config.mem_dim * 4 / (1024 * 1024)  # 4 bytes per float32
    mamba_memory = config.max_edges * config.mamba_d_model * config.mamba_d_state * 4 / (1024 * 1024)
    model_params = (config.mem_dim * config.hidden_channels * config.layers) * 4 / (1024 * 1024)
    
    total_mb = node_memory + mamba_memory + model_params
    return total_mb

# ============= Default Configurations =============
    """
    Get a default configuration optimized for different use cases.
    
    Args:
        size: 'small' (fast testing), 'medium' (standard), 'large' (full experiment)
    
    Returns:
        ExperimentConfig instance
    """
    config = ExperimentConfig()
    
    # Add CPU fallback option
    config.force_cpu = False  # Set to True to force CPU execution
    
    if size == 'small':
        # Fast configuration for testing
        config.data = 'tgbn-trade'
        config.seed = 42
        config.bs = 100           # Smaller batch size
        config.epochs = 5         # Few epochs for quick testing
        config.lr = 1e-3
        
        # Smaller model
        config.mem_dim = 64       # Reduced memory dimension
        config.time_dim = 32      # Reduced time dimension
        config.emb_dim = 64       # Reduced embedding dimension
        config.d = 2              # Small sheaf dimension
        config.layers = 2         # Fewer layers
        config.hidden_channels = 16  # Smaller hidden channels
        
        # Reduced Mamba parameters
        config.mamba_d_model = 64
        config.mamba_d_state = 8
        config.max_edges = 10000  # Fewer edges to track
        
        config.use_simple_gnn = True  # Use simple GNN for reliability
        
    elif size == 'medium':
        # Standard configuration for regular experiments
        config.data = 'tgbn-trade'
        config.seed = 42
        config.bs = 200
        config.epochs = 20
        config.lr = 1e-3
        
        # Medium model
        config.mem_dim = 100
        config.time_dim = 100
        config.emb_dim = 100
        config.d = 4
        config.layers = 3
        config.hidden_channels = 32
        
        # Standard Mamba parameters
        config.mamba_d_model = 128
        config.mamba_d_state = 16
        config.max_edges = 50000
        
        config.use_simple_gnn = False  # Try sheaf diffusion
        config.sheaf_type = 'diag'
        
    elif size == 'large':
        # Full configuration for complete experiments
        config.data = 'tgbn-trade'
        config.seed = 42
        config.bs = 200
        config.epochs = 50
        config.lr = 1e-3
        
        # Large model
        config.mem_dim = 128
        config.time_dim = 128
        config.emb_dim = 128
        config.d = 8
        config.layers = 4
        config.hidden_channels = 64
        
        # Full Mamba parameters
        config.mamba_d_model = 256
        config.mamba_d_state = 32
        config.max_edges = 100000
        
        config.use_simple_gnn = False
        config.sheaf_type = 'diag'
    
    else:
        raise ValueError(f"Unknown size: {size}. Use 'small', 'medium', or 'large'")
    
    print(f"Created {size} configuration:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Model size: mem_dim={config.mem_dim}, d={config.d}, layers={config.layers}")
    print(f"  Simple GNN: {config.use_simple_gnn}")
    print(f"  Sheaf type: {config.sheaf_type}")
    print(f"  Force CPU: {config.force_cpu}")
    
    return config


def get_cpu_config(size='small'):
    """Get a configuration that forces CPU execution (for debugging CUDA issues)."""
    config = get_default_config(size)
    config.force_cpu = True
    print(f"Configuration set to force CPU execution")
    return config


def get_test_config():
    """Get a minimal config for quick testing (even smaller than 'small')."""
    config = ExperimentConfig()
    
    # Minimal settings for immediate testing
    config.data = 'tgbn-trade'
    config.seed = 42
    config.bs = 50            # Very small batch
    config.epochs = 3         # Just a few epochs
    config.lr = 1e-3
    
    # Tiny model
    config.mem_dim = 32
    config.time_dim = 16
    config.emb_dim = 32
    config.d = 2
    config.layers = 1         # Single layer
    config.hidden_channels = 8
    
    # Minimal Mamba
    config.mamba_d_model = 32
    config.mamba_d_state = 4
    config.max_edges = 1000
    
    config.use_simple_gnn = True  # Always use simple for testing
    
    print("Created test configuration (minimal for quick testing)")
    return config


def get_sheaf_config(sheaf_type='diag'):
    """Get configuration optimized for specific sheaf diffusion type."""
    config = get_default_config('medium')  # Start with medium config
    
    config.use_simple_gnn = False  # Force use of sheaf diffusion
    config.sheaf_type = sheaf_type
    
    # Adjust parameters based on sheaf type
    if sheaf_type == 'diag':
        # Diagonal sheaf - can use smaller d
        config.d = 4
        config.epochs = 15
        
    elif sheaf_type == 'bundle':
        # Bundle sheaf - needs d > 1
        config.d = 6
        config.epochs = 20
        # Bundle sheaves are more complex
        config.lr = 5e-4  # Slower learning rate
        
    elif sheaf_type == 'general':
        # General sheaf - most complex
        config.d = 4  # Keep moderate to avoid too many parameters
        config.epochs = 25
        config.lr = 5e-4
        config.hidden_channels = 48  # Slightly larger for complexity
    
    print(f"Created {sheaf_type} sheaf configuration")
    return config


# ============= Quick Run Functions =============
def quick_test():
    """Run a quick test to verify everything works."""
    print("Running quick test...")
    config = get_test_config()
    results = run_experiment(config, verbose=True)
    
    print(f"\n✓ Quick test completed!")
    print(f"  Best validation: {results['best_val']:.4f}")
    print(f"  Best test: {results['best_test']:.4f}")
    
    return results


def run_safe_experiment(dataset_name='tgbn-trade'):
    """Run a safe experiment with CPU-only execution and conservative settings."""
    print(f"Running SAFE experiment for {dataset_name} (CPU-only, conservative settings)...")
    
    # Get dataset stats
    dataset_stats = get_dataset_stats(dataset_name)
    print_dataset_info(dataset_name)
    
    # Create very conservative configuration
    config = ExperimentConfig()
    config.data = dataset_name
    config.seed = 42
    config.force_cpu = True  # Force CPU to avoid CUDA issues
    
    # Very conservative training parameters
    config.bs = 50  # Small batch size
    config.epochs = 3  # Few epochs
    config.lr = 1e-3
    
    # Small model dimensions
    config.mem_dim = 32
    config.time_dim = 16
    config.emb_dim = 32
    config.d = 2
    config.layers = 1  # Single layer
    config.hidden_channels = 16
    
    # Minimal Mamba parameters
    config.mamba_d_model = 32
    config.mamba_d_state = 4
    config.max_edges = 1000  # Very limited
    
    # Force simple GNN
    config.use_simple_gnn = True
    
    print(f"Using SAFE configuration:")
    print(f"  CPU-only execution")
    print(f"  Small model: mem_dim={config.mem_dim}, hidden_channels={config.hidden_channels}")
    print(f"  Conservative training: bs={config.bs}, epochs={config.epochs}")
    
    try:
        results = run_experiment(config, verbose=True)
        print(f"\n✓ Safe experiment completed successfully!")
        print(f"  Best validation: {results['best_val']:.4f}")
        print(f"  Best test: {results['best_test']:.4f}")
        return results
    except Exception as e:
        print(f"✗ Even safe experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_optimized_experiment(dataset_name='tgbn-trade', complexity='medium'):
    """Run experiment with dataset-optimized configuration."""
    print(f"Running optimized experiment for {dataset_name}...")
    
    try:
        config = get_dataset_optimized_config(dataset_name, complexity)
        results = run_experiment(config, verbose=True)
        
        print(f"\n✓ Optimized experiment completed!")
        print(f"  Dataset: {dataset_name}")
        print(f"  Complexity: {complexity}")
        print(f"  Best validation: {results['best_val']:.4f}")
        print(f"  Best test: {results['best_test']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"\n✗ Optimized experiment failed: {e}")
        print("Falling back to safe experiment...")
        return run_safe_experiment(dataset_name)

def compare_dataset_configs(dataset_name='tgbn-trade'):
    """Compare different complexity levels for a dataset."""
    print(f"Comparing configurations for {dataset_name}...\n")
    
    results = {}
    
    for complexity in ['small', 'medium', 'large']:
        print(f"\n{'='*20} {complexity.upper()} CONFIG {'='*20}")
        try:
            config = get_dataset_optimized_config(dataset_name, complexity)
            # Reduce epochs for comparison
            config.epochs = min(config.epochs, 10)
            results[complexity] = run_experiment(config, verbose=False)
        except Exception as e:
            print(f"Error with {complexity} config: {e}")
            results[complexity] = None
    
    # Print comparison
    print(f"\n{'='*60}")
    print(f"CONFIGURATION COMPARISON FOR {dataset_name.upper()}:")
    print(f"{'='*60}")
    for complexity, result in results.items():
        if result is not None:
            print(f"{complexity:8s}: Val={result['best_val']:.4f}, Test={result['best_test']:.4f}")
        else:
            print(f"{complexity:8s}: Failed")
    
    return results
    """Run experiment with CPU-only execution (for debugging CUDA issues)."""
    print("Running CPU-only experiment...")
    config = get_cpu_config('small')
    results = run_experiment(config, verbose=True)
    
    print(f"\n✓ CPU experiment completed!")
    print(f"  Best validation: {results['best_val']:.4f}")
    print(f"  Best test: {results['best_test']:.4f}")
    
    return results


def debug_cuda_setup():
    """Debug CUDA setup and provide recommendations."""
    print("=== CUDA Debug Information ===")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name()}")
        
        try:
            # Test basic CUDA operations
            print("\nTesting basic CUDA operations...")
            x = torch.randn(2, 2).cuda()
            y = x + 1
            print("✓ Basic CUDA operations work")
            
            # Check memory
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
            
            # Clean up
            del x, y
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"✗ CUDA operations failed: {e}")
            print("Recommendation: Use CPU-only execution")
            return False
    else:
        print("CUDA not available - will use CPU")
    
    print("\n=== Recommendations ===")
    if torch.cuda.is_available():
        print("1. Try: results = run_cpu_experiment()  # Force CPU execution")
        print("2. Try: results = run_default_experiment()  # Auto CUDA/CPU detection")
        print("3. If CUDA issues persist, restart your kernel and clear GPU memory")
    else:
        print("1. Use: results = run_cpu_experiment()  # CPU-only execution")
    
    return torch.cuda.is_available()


def run_default_experiment():
    """Run experiment with default small configuration."""
    print("Running default experiment...")
    config = get_default_config('small')
    results = run_experiment(config, verbose=True)
    
    print(f"\n✓ Default experiment completed!")
    print(f"  Best validation: {results['best_val']:.4f}")
    print(f"  Best test: {results['best_test']:.4f}")
    
    return results


def run_medium_experiment():
    """Run experiment with medium configuration."""
    print("Running medium experiment...")
    config = get_default_config('medium')
    results = run_experiment(config, verbose=True)
    
    print(f"\n✓ Medium experiment completed!")
    print(f"  Best validation: {results['best_val']:.4f}")
    print(f"  Best test: {results['best_test']:.4f}")
    
    return results


def compare_configurations():
    """Compare different configurations."""
    print("Comparing different configurations...\n")
    
    results = {}
    
    # Test configuration
    print("=" * 30 + " TEST CONFIG " + "=" * 30)
    results['test'] = quick_test()
    
    # Small configuration
    print("\n" + "=" * 30 + " SMALL CONFIG " + "=" * 30)
    config_small = get_default_config('small')
    results['small'] = run_experiment(config_small, verbose=False)
    
    # Medium configuration (if time permits)
    print("\n" + "=" * 30 + " MEDIUM CONFIG " + "=" * 30)
    config_medium = get_default_config('medium')
    config_medium.epochs = 10  # Reduce for comparison
    results['medium'] = run_experiment(config_medium, verbose=False)
    
    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS:")
    print("=" * 70)
    for name, result in results.items():
        print(f"{name:10s}: Val={result['best_val']:.4f}, Test={result['best_test']:.4f}")
    
    return results


# ============= Notebook Helper Functions =============
def quick_experiment(sheaf_type='diag', epochs=10, use_simple_gnn=False, **kwargs):
    """Quick experiment function for notebook testing."""
    config = ExperimentConfig()
    config.sheaf_type = sheaf_type
    config.epochs = epochs
    config.use_simple_gnn = use_simple_gnn
    
    # Update any additional parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown parameter {key}")
    
    return run_experiment(config)


def test_simple_components():
    """Test the simple components."""
    print("Testing simple components...")
    
    # Test memory
    memory = SimpleMemory(100, 64, 128, 32)
    node_ids = torch.randint(0, 100, (10,))
    memories, times = memory(node_ids)
    print(f"✓ Memory test passed: {memories.shape}")
    
    # Test simple GNN
    gnn = SimpleTemporalGNN(128, 64, 32, 64, 2)
    x = torch.randn(100, 128)
    edge_index = torch.randint(0, 100, (2, 50))
    edge_times = torch.rand(50)
    edge_attr = torch.randn(50, 64)
    
    out = gnn(x, edge_index, edge_times, edge_attr)
    print(f"✓ Simple GNN test passed: {out.shape}")
    
    print("✓ All simple components working!")
    return True