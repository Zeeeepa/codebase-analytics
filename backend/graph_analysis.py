"""
Graph Analysis Module
Call graph and dependency graph analysis using NetworkX
"""

import networkx as nx
from typing import Dict, List, Any, Tuple
from collections import defaultdict


class CallGraphAnalyzer:
    """Analyzer for function call graphs"""
    
    def __init__(self, codebase):
        self.codebase = codebase
        self.call_graph = nx.DiGraph()
        self.function_map = {}
        
    def analyze(self) -> Dict[str, Any]:
        """Analyze call graph and return metrics"""
        print("🕸️ Building call graph...")
        
        # Build the call graph
        self._build_call_graph()
        
        # Calculate metrics
        metrics = self._calculate_call_graph_metrics()
        
        print(f"✅ Call graph analysis complete: {len(self.call_graph.nodes())} nodes, {len(self.call_graph.edges())} edges")
        return metrics
    
    def _build_call_graph(self):
        """Build the call graph from codebase functions"""
        # First pass: Add all functions as nodes
        for function in self.codebase.functions:
            self.call_graph.add_node(function.name, 
                                   filepath=function.filepath,
                                   function_obj=function)
            self.function_map[function.name] = function
        
        # Second pass: Add edges for function calls
        for function in self.codebase.functions:
            if hasattr(function, 'function_calls'):
                for call in function.function_calls:
                    if hasattr(call, 'name') and call.name in self.function_map:
                        self.call_graph.add_edge(function.name, call.name)
                    elif hasattr(call, 'function_definition'):
                        called_func = call.function_definition
                        if hasattr(called_func, 'name'):
                            self.call_graph.add_edge(function.name, called_func.name)
    
    def _calculate_call_graph_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive call graph metrics"""
        if not self.call_graph.nodes():
            return {}
        
        # Basic metrics
        num_nodes = len(self.call_graph.nodes())
        num_edges = len(self.call_graph.edges())
        
        # Connectivity metrics
        is_connected = nx.is_weakly_connected(self.call_graph)
        num_components = nx.number_weakly_connected_components(self.call_graph)
        
        # Centrality metrics
        in_degree_centrality = nx.in_degree_centrality(self.call_graph)
        out_degree_centrality = nx.out_degree_centrality(self.call_graph)
        betweenness_centrality = nx.betweenness_centrality(self.call_graph)
        
        # Find most important functions
        most_called = max(in_degree_centrality.items(), key=lambda x: x[1]) if in_degree_centrality else ("", 0)
        most_calling = max(out_degree_centrality.items(), key=lambda x: x[1]) if out_degree_centrality else ("", 0)
        most_central = max(betweenness_centrality.items(), key=lambda x: x[1]) if betweenness_centrality else ("", 0)
        
        # Cycle detection
        try:
            cycles = list(nx.simple_cycles(self.call_graph))
            has_cycles = len(cycles) > 0
            cycle_info = self._analyze_cycles(cycles)
        except:
            has_cycles = False
            cycle_info = {}
        
        # Strongly connected components
        scc = list(nx.strongly_connected_components(self.call_graph))
        num_scc = len(scc)
        largest_scc = max(scc, key=len) if scc else set()
        
        # Path metrics
        try:
            avg_path_length = nx.average_shortest_path_length(self.call_graph)
        except:
            avg_path_length = 0
        
        # Degree distribution
        in_degrees = [d for n, d in self.call_graph.in_degree()]
        out_degrees = [d for n, d in self.call_graph.out_degree()]
        
        return {
            'basic_metrics': {
                'total_functions': num_nodes,
                'total_calls': num_edges,
                'average_calls_per_function': num_edges / num_nodes if num_nodes > 0 else 0,
                'is_connected': is_connected,
                'num_components': num_components
            },
            'centrality_analysis': {
                'most_called_function': {
                    'name': most_called[0],
                    'centrality_score': round(most_called[1], 4)
                },
                'most_calling_function': {
                    'name': most_calling[0],
                    'centrality_score': round(most_calling[1], 4)
                },
                'most_central_function': {
                    'name': most_central[0],
                    'betweenness_score': round(most_central[1], 4)
                }
            },
            'cycle_analysis': {
                'has_cycles': has_cycles,
                'num_cycles': len(cycles) if 'cycles' in locals() else 0,
                'cycle_details': cycle_info
            },
            'component_analysis': {
                'num_strongly_connected_components': num_scc,
                'largest_scc_size': len(largest_scc),
                'largest_scc_functions': list(largest_scc)[:10]  # Top 10
            },
            'path_metrics': {
                'average_path_length': round(avg_path_length, 2),
                'is_dag': nx.is_directed_acyclic_graph(self.call_graph)
            },
            'degree_distribution': {
                'in_degree_stats': self._calculate_degree_stats(in_degrees),
                'out_degree_stats': self._calculate_degree_stats(out_degrees)
            },
            'hub_analysis': self._identify_hubs(),
            'leaf_analysis': self._identify_leaves()
        }
    
    def _analyze_cycles(self, cycles: List[List[str]]) -> Dict[str, Any]:
        """Analyze cycles in the call graph"""
        if not cycles:
            return {}
        
        cycle_lengths = [len(cycle) for cycle in cycles]
        
        return {
            'shortest_cycle_length': min(cycle_lengths),
            'longest_cycle_length': max(cycle_lengths),
            'average_cycle_length': sum(cycle_lengths) / len(cycle_lengths),
            'cycles_by_length': {
                'short': len([c for c in cycle_lengths if c <= 3]),
                'medium': len([c for c in cycle_lengths if 3 < c <= 6]),
                'long': len([c for c in cycle_lengths if c > 6])
            },
            'sample_cycles': cycles[:5]  # First 5 cycles as examples
        }
    
    def _calculate_degree_stats(self, degrees: List[int]) -> Dict[str, Any]:
        """Calculate statistics for degree distribution"""
        if not degrees:
            return {}
        
        return {
            'min': min(degrees),
            'max': max(degrees),
            'average': sum(degrees) / len(degrees),
            'median': sorted(degrees)[len(degrees) // 2],
            'distribution': {
                'zero': len([d for d in degrees if d == 0]),
                'low': len([d for d in degrees if 1 <= d <= 3]),
                'medium': len([d for d in degrees if 4 <= d <= 10]),
                'high': len([d for d in degrees if d > 10])
            }
        }
    
    def _identify_hubs(self) -> List[Dict[str, Any]]:
        """Identify hub functions (high in-degree)"""
        hubs = []
        for node in self.call_graph.nodes():
            in_degree = self.call_graph.in_degree(node)
            if in_degree > 5:  # Threshold for hub
                hubs.append({
                    'name': node,
                    'in_degree': in_degree,
                    'out_degree': self.call_graph.out_degree(node),
                    'filepath': self.call_graph.nodes[node].get('filepath', '')
                })
        
        hubs.sort(key=lambda x: x['in_degree'], reverse=True)
        return hubs[:10]  # Top 10 hubs
    
    def _identify_leaves(self) -> List[Dict[str, Any]]:
        """Identify leaf functions (zero out-degree)"""
        leaves = []
        for node in self.call_graph.nodes():
            if self.call_graph.out_degree(node) == 0:
                leaves.append({
                    'name': node,
                    'in_degree': self.call_graph.in_degree(node),
                    'filepath': self.call_graph.nodes[node].get('filepath', '')
                })
        
        return leaves


class DependencyGraphAnalyzer:
    """Analyzer for file dependency graphs"""
    
    def __init__(self, codebase):
        self.codebase = codebase
        self.dependency_graph = nx.DiGraph()
        
    def analyze(self) -> Dict[str, Any]:
        """Analyze dependency graph and return metrics"""
        print("🔗 Building dependency graph...")
        
        # Build the dependency graph
        self._build_dependency_graph()
        
        # Calculate metrics
        metrics = self._calculate_dependency_metrics()
        
        print(f"✅ Dependency analysis complete: {len(self.dependency_graph.nodes())} files, {len(self.dependency_graph.edges())} dependencies")
        return metrics
    
    def _build_dependency_graph(self):
        """Build the dependency graph from file imports"""
        # Add all files as nodes
        for file in self.codebase.files:
            self.dependency_graph.add_node(file.filepath, file_obj=file)
        
        # Add edges for imports
        for file in self.codebase.files:
            if hasattr(file, 'imports'):
                for imp in file.imports:
                    # Try to resolve import to actual file
                    target_file = self._resolve_import_to_file(imp, file.filepath)
                    if target_file and target_file in [f.filepath for f in self.codebase.files]:
                        self.dependency_graph.add_edge(file.filepath, target_file)
    
    def _resolve_import_to_file(self, import_obj, current_file: str) -> str:
        """Resolve an import to an actual file path"""
        # This is a simplified resolution - in practice, this would be more complex
        if hasattr(import_obj, 'module_name'):
            module_name = import_obj.module_name
            
            # Handle relative imports
            if module_name.startswith('.'):
                # Relative import - resolve relative to current file
                current_dir = '/'.join(current_file.split('/')[:-1])
                relative_path = module_name.replace('.', '/') + '.py'
                return f"{current_dir}/{relative_path}"
            
            # Handle absolute imports
            module_path = module_name.replace('.', '/') + '.py'
            
            # Check if this file exists in our codebase
            for file in self.codebase.files:
                if file.filepath.endswith(module_path):
                    return file.filepath
        
        return None
    
    def _calculate_dependency_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive dependency metrics"""
        if not self.dependency_graph.nodes():
            return {}
        
        # Basic metrics
        num_files = len(self.dependency_graph.nodes())
        num_dependencies = len(self.dependency_graph.edges())
        
        # Connectivity
        is_connected = nx.is_weakly_connected(self.dependency_graph)
        num_components = nx.number_weakly_connected_components(self.dependency_graph)
        
        # Cycle detection
        cycles = list(nx.simple_cycles(self.dependency_graph))
        has_circular_deps = len(cycles) > 0
        
        # Centrality metrics
        in_degree_centrality = nx.in_degree_centrality(self.dependency_graph)
        out_degree_centrality = nx.out_degree_centrality(self.dependency_graph)
        
        # Find most dependent files
        most_dependent = max(in_degree_centrality.items(), key=lambda x: x[1]) if in_degree_centrality else ("", 0)
        most_depending = max(out_degree_centrality.items(), key=lambda x: x[1]) if out_degree_centrality else ("", 0)
        
        # Topological analysis
        try:
            is_dag = nx.is_directed_acyclic_graph(self.dependency_graph)
            if is_dag:
                topo_order = list(nx.topological_sort(self.dependency_graph))
            else:
                topo_order = []
        except:
            is_dag = False
            topo_order = []
        
        # Degree distribution
        in_degrees = [d for n, d in self.dependency_graph.in_degree()]
        out_degrees = [d for n, d in self.dependency_graph.out_degree()]
        
        return {
            'basic_metrics': {
                'total_files': num_files,
                'total_dependencies': num_dependencies,
                'average_dependencies_per_file': num_dependencies / num_files if num_files > 0 else 0,
                'is_connected': is_connected,
                'num_components': num_components
            },
            'circular_dependencies': {
                'has_circular_dependencies': has_circular_deps,
                'num_cycles': len(cycles),
                'cycles': cycles[:5] if cycles else []  # First 5 cycles
            },
            'dependency_analysis': {
                'most_dependent_file': {
                    'filepath': most_dependent[0],
                    'dependency_score': round(most_dependent[1], 4)
                },
                'most_depending_file': {
                    'filepath': most_depending[0],
                    'dependency_score': round(most_depending[1], 4)
                }
            },
            'topological_analysis': {
                'is_dag': is_dag,
                'can_be_ordered': is_dag,
                'build_order_available': len(topo_order) > 0
            },
            'degree_distribution': {
                'in_degree_stats': self._calculate_degree_stats(in_degrees),
                'out_degree_stats': self._calculate_degree_stats(out_degrees)
            },
            'dependency_layers': self._calculate_dependency_layers(),
            'isolated_files': self._find_isolated_files()
        }
    
    def find_circular_dependencies(self) -> List[Dict[str, Any]]:
        """Find and analyze circular dependencies"""
        cycles = list(nx.simple_cycles(self.dependency_graph))
        
        circular_deps = []
        for cycle in cycles:
            circular_deps.append({
                'files': cycle,
                'cycle_length': len(cycle),
                'severity': 'high' if len(cycle) <= 3 else 'medium',
                'description': f"Circular dependency involving {len(cycle)} files"
            })
        
        # Sort by severity (shorter cycles are more problematic)
        circular_deps.sort(key=lambda x: x['cycle_length'])
        return circular_deps
    
    def _calculate_degree_stats(self, degrees: List[int]) -> Dict[str, Any]:
        """Calculate statistics for degree distribution"""
        if not degrees:
            return {}
        
        return {
            'min': min(degrees),
            'max': max(degrees),
            'average': round(sum(degrees) / len(degrees), 2),
            'distribution': {
                'independent': len([d for d in degrees if d == 0]),
                'low_coupling': len([d for d in degrees if 1 <= d <= 3]),
                'medium_coupling': len([d for d in degrees if 4 <= d <= 8]),
                'high_coupling': len([d for d in degrees if d > 8])
            }
        }
    
    def _calculate_dependency_layers(self) -> Dict[str, Any]:
        """Calculate dependency layers (files that can be built in parallel)"""
        if not nx.is_directed_acyclic_graph(self.dependency_graph):
            return {'error': 'Cannot calculate layers due to circular dependencies'}
        
        try:
            # Calculate the longest path to each node (dependency depth)
            depths = {}
            for node in nx.topological_sort(self.dependency_graph):
                if self.dependency_graph.in_degree(node) == 0:
                    depths[node] = 0
                else:
                    max_pred_depth = max(depths[pred] for pred in self.dependency_graph.predecessors(node))
                    depths[node] = max_pred_depth + 1
            
            # Group files by depth (layer)
            layers = defaultdict(list)
            for node, depth in depths.items():
                layers[depth].append(node)
            
            return {
                'num_layers': len(layers),
                'max_depth': max(depths.values()) if depths else 0,
                'layers': dict(layers),
                'parallel_build_possible': True
            }
        except:
            return {'error': 'Failed to calculate dependency layers'}
    
    def _find_isolated_files(self) -> List[str]:
        """Find files with no dependencies (in or out)"""
        isolated = []
        for node in self.dependency_graph.nodes():
            if (self.dependency_graph.in_degree(node) == 0 and 
                self.dependency_graph.out_degree(node) == 0):
                isolated.append(node)
        return isolated

