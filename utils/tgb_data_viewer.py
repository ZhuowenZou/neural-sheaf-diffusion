#!/usr/bin/env python3
"""
Temporal Graph Benchmark (TGB) Dataset Viewer
This script provides comprehensive functionality to view and explore TGB datasets.

Installation required: pip install py-tgb

TGB includes three main categories:
1. Link Prediction (tgbl-*): Dynamic link prediction datasets
2. Node Prediction (tgbn-*): Dynamic node property prediction datasets  
3. TKG/THG (tkgl-*, thgl-*): Temporal Knowledge Graphs and Temporal Heterogeneous Graphs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class TGBDatasetViewer:
    """
    A comprehensive viewer for TGB (Temporal Graph Benchmark) datasets.
    """
    
    def __init__(self):
        """Initialize the TGB Dataset Viewer."""
    def test_tgb_api(self, dataset_name: str = 'tgbl-wiki') -> None:
        """Test the TGB API to understand the current structure."""
        print(f"🔧 Testing TGB API with dataset: {dataset_name}")
        
        try:
            if dataset_name.startswith('tgbl-'):
                from tgb.linkproppred.dataset import LinkPropPredDataset
                dataset = LinkPropPredDataset(name=dataset_name)
                
            elif dataset_name.startswith('tgbn-'):
                from tgb.nodeproppred.dataset import NodePropPredDataset
                dataset = NodePropPredDataset(name=dataset_name)
            else:
                print("❌ Unknown dataset type")
                return
                
            print("✅ Dataset object created successfully")
            print(f"📋 Available methods: {[method for method in dir(dataset) if not method.startswith('_')]}")
            
            # Test train_val_test_split
            try:
                train_data, val_data, test_data = dataset.train_val_test_split()
                print("✅ train_val_test_split() works")
                print(f"📊 Train data keys: {list(train_data.keys())}")
                print(f"📊 Train edges: {len(train_data['src'])}")
                print(f"📊 Val edges: {len(val_data['src'])}")
                print(f"📊 Test edges: {len(test_data['src'])}")
                
                # Check for additional features
                if 'msg' in train_data:
                    print(f"🔢 Edge features available: {train_data['msg'].shape if train_data['msg'] is not None else 'None'}")
                if 'edge_type' in train_data:
                    print(f"🏷️  Edge types available: {len(np.unique(train_data['edge_type']))}")
                    
            except Exception as e:
                print(f"❌ Error with train_val_test_split(): {e}")
                
        except Exception as e:
            print(f"❌ Error creating dataset: {e}")
            print("💡 Make sure py-tgb is installed: pip install py-tgb")
            # Link Prediction Datasets
            'link_prediction': [
                'tgbl-wiki',        # Wikipedia edits
                'tgbl-review',      # Product reviews  
                'tgbl-coin',        # Cryptocurrency transactions
                'tgbl-comment',     # Reddit comments
                'tgbl-flight',      # Flight connections
                'tgbl-subreddit',   # Subreddit interactions
                'tgbl-lastfm',      # Last.fm music listening
                'tgbl-synthetic'    # Synthetic dataset
            ],
            # Node Prediction Datasets  
            'node_prediction': [
                'tgbn-trade',       # Trade networks
                'tgbn-genre',       # Music genre prediction
                'tgbn-reddit',      # Reddit user activity
                'tgbn-token'        # Token transactions
            ],
            # Temporal Knowledge Graphs (TGB 2.0)
            'temporal_kg': [
                'tkgl-polecat',     # Political events
                'tkgl-icews',       # Integrated Crisis Early Warning System
                'tkgl-smallpedia',  # Small Wikipedia knowledge graph
                'tkgl-yago'         # YAGO knowledge graph
            ],
            # Temporal Heterogeneous Graphs (TGB 2.0)
            'temporal_hg': [
                'thgl-github',      # GitHub interactions
                'thgl-myket',       # Mobile app store
                'thgl-forum',       # Online forum interactions
                'thgl-software'     # Software development
            ]
        }
        
    def test_tgb_api(self, dataset_name: str = 'tgbl-wiki') -> None:
        """Test the TGB API to understand the current structure."""
        print(f"🔧 Testing TGB API with dataset: {dataset_name}")
        
        try:
            if dataset_name.startswith('tgbl-'):
                from tgb.linkproppred.dataset import LinkPropPredDataset
                dataset = LinkPropPredDataset(name=dataset_name)
                
            elif dataset_name.startswith('tgbn-'):
                from tgb.nodeproppred.dataset import NodePropPredDataset
                dataset = NodePropPredDataset(name=dataset_name)
            else:
                print("❌ Unknown dataset type")
                return
                
            print("✅ Dataset object created successfully")
            print(f"📋 Available methods: {[method for method in dir(dataset) if not method.startswith('_')]}")
            
            # Test train_val_test_split
            try:
                train_data, val_data, test_data = dataset.train_val_test_split()
                print("✅ train_val_test_split() works")
                print(f"📊 Train data keys: {list(train_data.keys())}")
                print(f"📊 Train edges: {len(train_data['src'])}")
                print(f"📊 Val edges: {len(val_data['src'])}")
                print(f"📊 Test edges: {len(test_data['src'])}")
                
                # Check for additional features
                if 'msg' in train_data:
                    print(f"🔢 Edge features available: {train_data['msg'].shape if train_data['msg'] is not None else 'None'}")
                if 'edge_type' in train_data:
                    print(f"🏷️  Edge types available: {len(np.unique(train_data['edge_type']))}")
                    
            except Exception as e:
                print(f"❌ Error with train_val_test_split(): {e}")
                
        except Exception as e:
            print(f"❌ Error creating dataset: {e}")
            print("💡 Make sure py-tgb is installed: pip install py-tgb")
        
    def list_all_datasets(self) -> None:
        """Display all available TGB datasets organized by category."""
        print("=" * 60)
        print("TEMPORAL GRAPH BENCHMARK (TGB) DATASETS")
        print("=" * 60)
        
        for category, datasets in self.available_datasets.items():
            print(f"\n🔸 {category.upper().replace('_', ' ')} ({len(datasets)} datasets):")
            for i, dataset in enumerate(datasets, 1):
                print(f"   {i:2d}. {dataset}")
        
        print(f"\n📊 Total datasets available: {sum(len(d) for d in self.available_datasets.values())}")
        print("💡 Use load_dataset(dataset_name) to explore any dataset!")
        
    def load_dataset_simple(self, dataset_name: str, root: str = "./data") -> Dict[str, Any]:
        """
        Simplified dataset loading that works with current TGB API.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'tgbl-wiki', 'tgbn-reddit')
            root: Root directory to store datasets
            
        Returns:
            Dictionary containing dataset information and data splits
        """
        print(f"🔄 Loading dataset: {dataset_name}")
        
        try:
            # Determine dataset type and import appropriate module
            if dataset_name.startswith('tgbl-'):
                from tgb.linkproppred.dataset import LinkPropPredDataset
                dataset = LinkPropPredDataset(name=dataset_name, root=root)
                task_type = "Link Prediction"
                
            elif dataset_name.startswith('tgbn-'):
                from tgb.nodeproppred.dataset import NodePropPredDataset  
                dataset = NodePropPredDataset(name=dataset_name, root=root)
                task_type = "Node Prediction"
                
            elif dataset_name.startswith('tkgl-'):
                from tgb.linkproppred.dataset import LinkPropPredDataset
                dataset = LinkPropPredDataset(name=dataset_name, root=root)
                task_type = "Temporal Knowledge Graph"
                
            elif dataset_name.startswith('thgl-'):
                from tgb.linkproppred.dataset import LinkPropPredDataset
                dataset = LinkPropPredDataset(name=dataset_name, root=root)
                task_type = "Temporal Heterogeneous Graph"
                
            else:
                raise ValueError(f"Unknown dataset type for {dataset_name}")
            
            # Get data splits using the correct API
            train_data, val_data, test_data = dataset.train_val_test_split()
            
            # Reconstruct basic statistics from splits
            full_src = np.concatenate([train_data['src'], val_data['src'], test_data['src']])
            full_dst = np.concatenate([train_data['dst'], val_data['dst'], test_data['dst']]) 
            full_t = np.concatenate([train_data['t'], val_data['t'], test_data['t']])
            
            num_nodes = len(np.unique(np.concatenate([full_src, full_dst])))
            num_edges = len(full_src)
            time_span = (full_t.min(), full_t.max())
            num_timestamps = len(np.unique(full_t))
            
            # Check for edge features
            edge_feat_dim = 0
            if 'msg' in train_data and train_data['msg'] is not None:
                edge_feat_dim = train_data['msg'].shape[1]
                full_msg = np.concatenate([train_data['msg'], val_data['msg'], test_data['msg']])
            else:
                full_msg = None
                
            # Check for edge types
            num_edge_types = 0
            if 'edge_type' in train_data:
                full_edge_type = np.concatenate([train_data['edge_type'], val_data['edge_type'], test_data['edge_type']])
                num_edge_types = len(np.unique(full_edge_type))
            else:
                full_edge_type = None
            
            # Create a simple data object for compatibility
            class SimpleTemporalData:
                def __init__(self, src, dst, t, msg=None, edge_type=None):
                    self.src = src
                    self.dst = dst
                    self.t = t
                    self.msg = msg
                    self.edge_type = edge_type
                    self.num_nodes = len(np.unique(np.concatenate([src, dst])))
                    
            data = SimpleTemporalData(full_src, full_dst, full_t, full_msg, full_edge_type)
                
            dataset_info = {
                'name': dataset_name,
                'task_type': task_type,
                'dataset_obj': dataset,
                'full_data': data,
                'train_data': train_data,
                'val_data': val_data, 
                'test_data': test_data,
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'time_span': time_span,
                'num_timestamps': num_timestamps,
                'edge_feat_dim': edge_feat_dim,
                'node_feat_dim': 0  # Node features typically not available in raw format
            }
            
            if num_edge_types > 0:
                dataset_info['num_edge_types'] = num_edge_types
            
            self._print_dataset_summary(dataset_info)
            return dataset_info
            
        except Exception as e:
            print(f"❌ Error loading dataset {dataset_name}: {str(e)}")
            print("💡 Make sure you have installed: pip install py-tgb")
            return {}
    def load_dataset(self, dataset_name: str, root: str = "./data") -> Dict[str, Any]:
        """
        Load a TGB dataset and return comprehensive information.
        This method tries multiple API approaches for compatibility.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'tgbl-wiki', 'tgbn-reddit')
            root: Root directory to store datasets
            
        Returns:
            Dictionary containing dataset information and data splits
        """
        # First try the simplified approach which works with current TGB API
        return self.load_dataset_simple(dataset_name, root)
    
    def _print_dataset_summary(self, info: Dict[str, Any]) -> None:
        """Print a comprehensive summary of the dataset."""
        print("\n" + "=" * 50)
        print(f"📊 DATASET SUMMARY: {info['name'].upper()}")
        print("=" * 50)
        
        print(f"🎯 Task Type: {info['task_type']}")
        print(f"📈 Nodes: {info['num_nodes']:,}")
        print(f"🔗 Edges: {info['num_edges']:,}")
        print(f"⏰ Timestamps: {info['num_timestamps']:,}")
        print(f"📅 Time Range: {info['time_span'][0]:.0f} - {info['time_span'][1]:.0f}")
        
        if info['edge_feat_dim'] > 0:
            print(f"🔢 Edge Feature Dimension: {info['edge_feat_dim']}")
        if info['node_feat_dim'] > 0:
            print(f"📋 Node Feature Dimension: {info['node_feat_dim']}")
        if 'num_edge_types' in info:
            print(f"🏷️  Edge Types: {info['num_edge_types']}")
            
        # Data split information
        print(f"\n📊 DATA SPLITS:")
        try:
            print(f"   🚂 Train: {len(info['train_data']['src']):,} edges")
            print(f"   ✅ Validation: {len(info['val_data']['src']):,} edges") 
            print(f"   🧪 Test: {len(info['test_data']['src']):,} edges")
        except:
            try:
                print(f"   🚂 Train: {len(info['train_data'].src):,} edges")
                print(f"   ✅ Validation: {len(info['val_data'].src):,} edges") 
                print(f"   🧪 Test: {len(info['test_data'].src):,} edges")
            except:
                print("   ℹ️  Data split information not available")
        
    def explore_dataset(self, dataset_info: Dict[str, Any], plot: bool = True) -> None:
        """
        Explore dataset with detailed analysis and visualizations.
        
        Args:
            dataset_info: Dictionary returned from load_dataset()
            plot: Whether to generate visualizations
        """
        if not dataset_info:
            print("❌ No dataset information provided!")
            return
            
        data = dataset_info['full_data']
        name = dataset_info['name']
        
        print(f"\n🔍 DETAILED EXPLORATION: {name}")
        print("=" * 50)
        
        # Temporal analysis
        self._analyze_temporal_patterns(data, name, plot)
        
        # Node degree analysis  
        self._analyze_node_degrees(data, name, plot)
        
        # Edge features analysis
        if hasattr(data, 'msg') and data.msg is not None:
            self._analyze_edge_features(data, name, plot)
            
        # Node features analysis
        if hasattr(data, 'x') and data.x is not None:
            self._analyze_node_features(data, name, plot)
            
        # Edge type analysis (for multi-relational graphs)
        if hasattr(data, 'edge_type'):
            self._analyze_edge_types(data, name, plot)
    
    def _analyze_temporal_patterns(self, data, name: str, plot: bool = True) -> None:
        """Analyze temporal patterns in the dataset."""
        print("\n⏰ TEMPORAL PATTERNS:")
        
        timestamps = data.t
        unique_times = np.unique(timestamps)
        
        # Time intervals
        if len(unique_times) > 1:
            intervals = np.diff(unique_times)
            print(f"   📊 Min time interval: {intervals.min():.2f}")
            print(f"   📊 Max time interval: {intervals.max():.2f}")  
            print(f"   📊 Avg time interval: {intervals.mean():.2f}")
        
        # Edges per timestamp
        edges_per_time = np.bincount(timestamps.astype(int) - timestamps.min().astype(int))
        print(f"   📈 Avg edges per timestamp: {edges_per_time.mean():.2f}")
        print(f"   📈 Max edges per timestamp: {edges_per_time.max()}")
        
        if plot:
            plt.figure(figsize=(15, 5))
            
            # Plot 1: Edges over time
            plt.subplot(1, 3, 1)
            time_bins = np.linspace(timestamps.min(), timestamps.max(), 50)
            plt.hist(timestamps, bins=time_bins, alpha=0.7, color='skyblue')
            plt.title(f'{name}: Edges Over Time')
            plt.xlabel('Timestamp')
            plt.ylabel('Number of Edges')
            
            # Plot 2: Time intervals
            if len(unique_times) > 1:
                plt.subplot(1, 3, 2)
                plt.hist(intervals, bins=30, alpha=0.7, color='lightgreen')
                plt.title('Time Intervals Distribution')
                plt.xlabel('Time Interval')
                plt.ylabel('Frequency')
            
            # Plot 3: Cumulative edges
            plt.subplot(1, 3, 3)
            sorted_times = np.sort(timestamps)
            cumulative_edges = np.arange(1, len(sorted_times) + 1)
            plt.plot(sorted_times, cumulative_edges, color='orange', linewidth=2)
            plt.title('Cumulative Edges Over Time')
            plt.xlabel('Timestamp')
            plt.ylabel('Cumulative Edge Count')
            
            plt.tight_layout()
            plt.show()
    
    def _analyze_node_degrees(self, data, name: str, plot: bool = True) -> None:
        """Analyze node degree distributions."""
        print("\n📊 NODE DEGREE ANALYSIS:")
        
        # Calculate degrees
        all_nodes = np.concatenate([data.src, data.dst])
        unique_nodes, counts = np.unique(all_nodes, return_counts=True)
        
        # In-degree and out-degree
        in_degrees = np.bincount(data.dst, minlength=data.num_nodes if hasattr(data, 'num_nodes') else all_nodes.max()+1)
        out_degrees = np.bincount(data.src, minlength=data.num_nodes if hasattr(data, 'num_nodes') else all_nodes.max()+1)
        
        print(f"   📈 Avg degree: {counts.mean():.2f}")
        print(f"   📈 Max degree: {counts.max()}")
        print(f"   📈 Avg in-degree: {in_degrees[in_degrees > 0].mean():.2f}")
        print(f"   📈 Avg out-degree: {out_degrees[out_degrees > 0].mean():.2f}")
        
        if plot:
            plt.figure(figsize=(15, 5))
            
            # Plot 1: Degree distribution
            plt.subplot(1, 3, 1)
            plt.hist(counts, bins=50, alpha=0.7, color='skyblue', log=True)
            plt.title(f'{name}: Degree Distribution')
            plt.xlabel('Degree')
            plt.ylabel('Frequency (log scale)')
            plt.yscale('log')
            
            # Plot 2: In-degree vs Out-degree
            plt.subplot(1, 3, 2)
            active_nodes = unique_nodes
            in_deg_active = in_degrees[active_nodes]
            out_deg_active = out_degrees[active_nodes]
            plt.scatter(out_deg_active, in_deg_active, alpha=0.6, s=10)
            plt.title('In-degree vs Out-degree')
            plt.xlabel('Out-degree')
            plt.ylabel('In-degree')
            plt.loglog()
            
            # Plot 3: Degree over time (sample of nodes)
            plt.subplot(1, 3, 3)
            if len(unique_nodes) > 100:
                sample_nodes = np.random.choice(unique_nodes, 100, replace=False)
            else:
                sample_nodes = unique_nodes
                
            for node in sample_nodes[:10]:  # Plot only first 10 to avoid clutter
                node_edges = data.t[(data.src == node) | (data.dst == node)]
                if len(node_edges) > 0:
                    plt.plot(np.sort(node_edges), np.arange(1, len(node_edges) + 1), alpha=0.7)
                    
            plt.title('Node Activity Over Time (Sample)')
            plt.xlabel('Timestamp')
            plt.ylabel('Cumulative Edges')
            
            plt.tight_layout()
            plt.show()
    
    def _analyze_edge_features(self, data, name: str, plot: bool = True) -> None:
        """Analyze edge features if available."""
        print("\n🔢 EDGE FEATURES ANALYSIS:")
        
        edge_features = data.msg
        print(f"   📊 Feature dimension: {edge_features.shape[1]}")
        print(f"   📊 Feature range: [{edge_features.min():.3f}, {edge_features.max():.3f}]")
        print(f"   📊 Feature mean: {edge_features.mean():.3f}")
        print(f"   📊 Feature std: {edge_features.std():.3f}")
        
        if plot and edge_features.shape[1] <= 10:
            plt.figure(figsize=(12, 4))
            
            # Feature distributions
            for i in range(min(edge_features.shape[1], 6)):
                plt.subplot(2, 3, i+1)
                plt.hist(edge_features[:, i], bins=30, alpha=0.7)
                plt.title(f'Feature {i+1}')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                
            plt.suptitle(f'{name}: Edge Feature Distributions')
            plt.tight_layout()
            plt.show()
    
    def _analyze_node_features(self, data, name: str, plot: bool = True) -> None:
        """Analyze node features if available."""
        print("\n📋 NODE FEATURES ANALYSIS:")
        
        node_features = data.x
        print(f"   📊 Feature dimension: {node_features.shape[1]}")
        print(f"   📊 Feature range: [{node_features.min():.3f}, {node_features.max():.3f}]")
        print(f"   📊 Feature mean: {node_features.mean():.3f}")
        print(f"   📊 Feature std: {node_features.std():.3f}")
        
        if plot and node_features.shape[1] <= 10:
            plt.figure(figsize=(12, 4))
            
            for i in range(min(node_features.shape[1], 6)):
                plt.subplot(2, 3, i+1)
                plt.hist(node_features[:, i], bins=30, alpha=0.7)
                plt.title(f'Feature {i+1}')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                
            plt.suptitle(f'{name}: Node Feature Distributions')
            plt.tight_layout()
            plt.show()
    
    def _analyze_edge_types(self, data, name: str, plot: bool = True) -> None:
        """Analyze edge types for multi-relational graphs."""
        print("\n🏷️  EDGE TYPE ANALYSIS:")
        
        edge_types = data.edge_type
        unique_types, counts = np.unique(edge_types, return_counts=True)
        
        print(f"   📊 Number of edge types: {len(unique_types)}")
        for edge_type, count in zip(unique_types, counts):
            print(f"   🔸 Type {edge_type}: {count:,} edges ({count/len(edge_types)*100:.1f}%)")
        
        if plot:
            plt.figure(figsize=(10, 6))
            
            plt.subplot(1, 2, 1)
            plt.bar(unique_types, counts, alpha=0.7, color='lightcoral')
            plt.title(f'{name}: Edge Type Distribution')
            plt.xlabel('Edge Type')
            plt.ylabel('Count')
            
            plt.subplot(1, 2, 2)
            plt.pie(counts, labels=[f'Type {t}' for t in unique_types], autopct='%1.1f%%')
            plt.title('Edge Type Proportions')
            
            plt.tight_layout()
            plt.show()
    
    def compare_datasets(self, dataset_names: List[str]) -> None:
        """Compare multiple datasets side by side."""
        print("🔄 Loading datasets for comparison...")
        datasets_info = []
        
        for name in dataset_names:
            info = self.load_dataset(name)
            if info:
                datasets_info.append(info)
        
        if len(datasets_info) < 2:
            print("❌ Need at least 2 valid datasets for comparison!")
            return
            
        # Create comparison table
        comparison_df = pd.DataFrame([
            {
                'Dataset': info['name'],
                'Task Type': info['task_type'],
                'Nodes': info['num_nodes'],
                'Edges': info['num_edges'],
                'Timestamps': info['num_timestamps'],
                'Edge Features': info['edge_feat_dim'],
                'Node Features': info['node_feat_dim'],
                'Time Span': f"{info['time_span'][1] - info['time_span'][0]:.0f}"
            }
            for info in datasets_info
        ])
        
        print("\n📊 DATASET COMPARISON:")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Nodes comparison
        axes[0, 0].bar(comparison_df['Dataset'], comparison_df['Nodes'], alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Number of Nodes')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Edges comparison  
        axes[0, 1].bar(comparison_df['Dataset'], comparison_df['Edges'], alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Number of Edges')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Timestamps comparison
        axes[1, 0].bar(comparison_df['Dataset'], comparison_df['Timestamps'], alpha=0.7, color='orange')
        axes[1, 0].set_title('Number of Timestamps')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Time span comparison
        axes[1, 1].bar(comparison_df['Dataset'], comparison_df['Time Span'].astype(float), alpha=0.7, color='lightcoral')
        axes[1, 1].set_title('Time Span')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()


# Example usage and demonstration
def main():
    """Demonstrate the TGB Dataset Viewer functionality."""
    viewer = TGBDatasetViewer()
    
    print("🚀 TEMPORAL GRAPH BENCHMARK (TGB) DATASET VIEWER")
    print("=" * 55)
    
    # Show all available datasets
    viewer.list_all_datasets()
    
    # Example: Load and explore a specific dataset
    print("\n" + "=" * 55)
    print("📝 EXAMPLE: Loading and exploring tgbl-wiki dataset")
    print("=" * 55)
    
    # Load Wikipedia dataset (link prediction)
    try:
        dataset_info = viewer.load_dataset('tgbl-wiki')
        
        if dataset_info:
            # Detailed exploration with visualizations
            viewer.explore_dataset(dataset_info, plot=True)
        else:
            print("⚠️  Could not load tgbl-wiki. Trying tgbl-review instead...")
            dataset_info = viewer.load_dataset('tgbl-review')
            if dataset_info:
                viewer.explore_dataset(dataset_info, plot=True)
    except Exception as e:
        print(f"⚠️  Error in example: {e}")
        print("💡 This might be due to network issues or missing dependencies.")
    
    # Example: Compare multiple datasets  
    print("\n" + "=" * 55)
    print("📝 EXAMPLE: Comparing multiple datasets")
    print("=" * 55)
    
    # Compare different types of datasets
    viewer.compare_datasets(['tgbl-wiki', 'tgbl-review', 'tgbn-reddit'])


if __name__ == "__main__":
    # Additional utility functions for specific use cases
    
    def quick_load_and_view(dataset_name: str):
        """Quick function to load and view a single dataset."""
        viewer = TGBDatasetViewer()
        info = viewer.load_dataset(dataset_name)
        if info:
            viewer.explore_dataset(info, plot=True)
        return info
    
    def get_dataset_stats(dataset_name: str) -> Dict[str, Any]:
        """Get basic statistics for a dataset without loading full data."""
        viewer = TGBDatasetViewer()
        return viewer.load_dataset(dataset_name)
    
    def batch_analyze_datasets(dataset_list: List[str]):
        """Analyze multiple datasets in batch."""
        viewer = TGBDatasetViewer()
        results = {}
        
        for dataset in dataset_list:
            print(f"\n🔄 Processing {dataset}...")
            info = viewer.load_dataset(dataset)
            if info:
                results[dataset] = info
                
        return results
    
    # Run the main demonstration
    main()