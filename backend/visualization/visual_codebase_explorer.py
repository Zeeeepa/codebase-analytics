#!/usr/bin/env python3
"""
Visual Codebase Explorer

This module provides functionality for visually exploring a codebase,
including dependency graphs, error visualization, and blast radius analysis.
"""

import logging
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from enum import Enum

from codegen.sdk.core.codebase import Codebase
from codegen.sdk.core.file import SourceFile
from codegen.sdk.core.function import Function
from codegen.sdk.core.class_definition import Class
from codegen.sdk.core.symbol import Symbol
from codegen.sdk.enums import EdgeType, SymbolType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ExplorationMode(str, Enum):
    """Modes for visual exploration."""
    
    STRUCTURAL_OVERVIEW = "structural_overview"
    DEPENDENCY_GRAPH = "dependency_graph"
    ERROR_VISUALIZATION = "error_visualization"
    COMPLEXITY_HEATMAP = "complexity_heatmap"
    BLAST_RADIUS = "blast_radius"


def create_visual_exploration(
    codebase: Codebase,
    mode: ExplorationMode = ExplorationMode.STRUCTURAL_OVERVIEW,
    symbol_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a visual exploration of the codebase.
    
    Args:
        codebase: Codebase object
        mode: Exploration mode
        symbol_name: Optional symbol name for blast radius analysis
        
    Returns:
        Dictionary containing the exploration data
    """
    logger.info(f"Creating visual exploration in {mode} mode")
    
    if mode == ExplorationMode.STRUCTURAL_OVERVIEW:
        return create_structural_overview(codebase)
    elif mode == ExplorationMode.DEPENDENCY_GRAPH:
        return create_dependency_graph(codebase)
    elif mode == ExplorationMode.ERROR_VISUALIZATION:
        return create_error_visualization(codebase)
    elif mode == ExplorationMode.COMPLEXITY_HEATMAP:
        return create_complexity_heatmap(codebase)
    elif mode == ExplorationMode.BLAST_RADIUS:
        if not symbol_name:
            raise ValueError("Symbol name is required for blast radius analysis")
        return analyze_error_blast_radius(codebase, symbol_name)
    else:
        raise ValueError(f"Unsupported exploration mode: {mode}")


def create_structural_overview(codebase: Codebase) -> Dict[str, Any]:
    """
    Create a structural overview of the codebase.
    
    Args:
        codebase: Codebase object
        
    Returns:
        Dictionary containing the structural overview
    """
    # Create nodes for files, classes, and functions
    nodes = []
    edges = []
    
    # Add root node
    root_node = {
        "id": "root",
        "label": "Codebase",
        "type": "root",
        "size": 30,
        "color": "#64748b",
    }
    nodes.append(root_node)
    
    # Add file nodes
    for file in codebase.files:
        file_node = {
            "id": file.filepath,
            "label": Path(file.filepath).name,
            "type": "file",
            "size": 20,
            "color": "#3b82f6",  # Blue
            "metadata": {
                "file": file.filepath,
                "functions": len(file.functions),
                "classes": len(file.classes),
            }
        }
        nodes.append(file_node)
        
        # Add edge from root to file
        edges.append({
            "source": "root",
            "target": file.filepath,
            "type": "contains",
        })
        
        # Add class nodes
        for cls in file.classes:
            class_node = {
                "id": f"{file.filepath}:{cls.name}",
                "label": cls.name,
                "type": "class",
                "size": 15,
                "color": "#8b5cf6",  # Purple
                "metadata": {
                    "file": file.filepath,
                    "methods": len(cls.methods) if hasattr(cls, "methods") else 0,
                }
            }
            nodes.append(class_node)
            
            # Add edge from file to class
            edges.append({
                "source": file.filepath,
                "target": f"{file.filepath}:{cls.name}",
                "type": "contains",
            })
            
            # Add method nodes
            if hasattr(cls, "methods"):
                for method in cls.methods:
                    method_node = {
                        "id": f"{file.filepath}:{cls.name}.{method.name}",
                        "label": method.name,
                        "type": "method",
                        "size": 10,
                        "color": "#ec4899",  # Pink
                        "metadata": {
                            "file": file.filepath,
                            "class": cls.name,
                        }
                    }
                    nodes.append(method_node)
                    
                    # Add edge from class to method
                    edges.append({
                        "source": f"{file.filepath}:{cls.name}",
                        "target": f"{file.filepath}:{cls.name}.{method.name}",
                        "type": "contains",
                    })
        
        # Add function nodes (excluding methods)
        for func in file.functions:
            # Skip methods (already added above)
            if hasattr(func, "parent") and func.parent:
                continue
            
            func_node = {
                "id": f"{file.filepath}:{func.name}",
                "label": func.name,
                "type": "function",
                "size": 10,
                "color": "#22c55e",  # Green
                "metadata": {
                    "file": file.filepath,
                }
            }
            nodes.append(func_node)
            
            # Add edge from file to function
            edges.append({
                "source": file.filepath,
                "target": f"{file.filepath}:{func.name}",
                "type": "contains",
            })
    
    # Add function call edges
    for file in codebase.files:
        for func in file.functions:
            if hasattr(func, "calls"):
                for call in func.calls:
                    if hasattr(call, "resolved_symbol") and call.resolved_symbol:
                        # Get source node ID
                        source_id = f"{file.filepath}:{func.name}"
                        if hasattr(func, "parent") and func.parent:
                            source_id = f"{file.filepath}:{func.parent.name}.{func.name}"
                        
                        # Get target node ID
                        target_symbol = call.resolved_symbol
                        target_file = getattr(target_symbol, "file", None)
                        target_name = getattr(target_symbol, "name", None)
                        target_parent = getattr(target_symbol, "parent", None)
                        
                        if target_file and target_name:
                            target_id = f"{target_file.filepath}:{target_name}"
                            if target_parent:
                                target_id = f"{target_file.filepath}:{target_parent.name}.{target_name}"
                            
                            # Add edge from function to called function
                            edges.append({
                                "source": source_id,
                                "target": target_id,
                                "type": "calls",
                            })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "total_files": len([n for n in nodes if n["type"] == "file"]),
            "total_classes": len([n for n in nodes if n["type"] == "class"]),
            "total_functions": len([n for n in nodes if n["type"] in ["function", "method"]]),
            "total_issues": 0,
        }
    }


def create_dependency_graph(codebase: Codebase) -> Dict[str, Any]:
    """
    Create a dependency graph of the codebase.
    
    Args:
        codebase: Codebase object
        
    Returns:
        Dictionary containing the dependency graph
    """
    nodes = []
    edges = []
    
    # Create nodes for each file
    for file in codebase.files:
        nodes.append({
            "id": file.filepath,
            "label": Path(file.filepath).name,
            "type": "file",
            "size": 20,
            "color": "#3b82f6",  # Blue
            "metadata": {
                "file": file.filepath,
                "functions": len(file.functions),
                "classes": len(file.classes),
            }
        })
    
    # Create edges for imports
    for file in codebase.files:
        for imp in getattr(file, "imports", []):
            if hasattr(imp, "resolved_path") and imp.resolved_path:
                edges.append({
                    "source": file.filepath,
                    "target": imp.resolved_path,
                    "type": "import",
                    "color": "#64748b",  # Gray
                })
    
    # Detect circular dependencies
    circular_deps = detect_circular_dependencies(nodes, edges)
    
    # Add circular dependency edges
    for cycle in circular_deps:
        for i in range(len(cycle)):
            source = cycle[i]
            target = cycle[(i + 1) % len(cycle)]
            
            edges.append({
                "source": source,
                "target": target,
                "type": "circular",
                "color": "#ef4444",  # Red
                "weight": 2.0,
                "metadata": {
                    "cycle": cycle,
                }
            })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "circular_dependencies": circular_deps,
        "summary": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "total_circular_dependencies": len(circular_deps),
            "total_issues": len(circular_deps),
        }
    }


def detect_circular_dependencies(nodes, edges) -> List[List[str]]:
    """
    Detect circular dependencies in the dependency graph.
    
    Args:
        nodes: List of nodes
        edges: List of edges
        
    Returns:
        List of cycles, where each cycle is a list of node IDs
    """
    # Build adjacency list
    graph = {}
    for node in nodes:
        graph[node["id"]] = []
    
    for edge in edges:
        if edge["source"] in graph and edge["target"] in graph:
            graph[edge["source"]].append(edge["target"])
    
    # Find cycles using DFS
    def find_cycles(node, visited, path):
        if node in path:
            # Found a cycle
            cycle_start = path.index(node)
            return [path[cycle_start:]]
        
        if node in visited:
            return []
        
        visited.add(node)
        path.append(node)
        
        cycles = []
        for neighbor in graph.get(node, []):
            cycles.extend(find_cycles(neighbor, visited.copy(), path.copy()))
        
        return cycles
    
    all_cycles = []
    for node in graph:
        cycles = find_cycles(node, set(), [])
        for cycle in cycles:
            if cycle not in all_cycles:
                all_cycles.append(cycle)
    
    return all_cycles


def create_error_visualization(codebase: Codebase) -> Dict[str, Any]:
    """
    Create a visualization of errors in the codebase.
    
    Args:
        codebase: Codebase object
        
    Returns:
        Dictionary containing the error visualization
    """
    nodes = []
    edges = []
    errors = []
    
    # Create nodes for each file
    for file in codebase.files:
        nodes.append({
            "id": file.filepath,
            "label": Path(file.filepath).name,
            "type": "file",
            "size": 20,
            "color": "#3b82f6",  # Blue
            "metadata": {
                "file": file.filepath,
                "functions": len(file.functions),
                "classes": len(file.classes),
            }
        })
    
    # Check for import errors
    for file in codebase.files:
        for imp in getattr(file, "imports", []):
            if hasattr(imp, "resolved_path") and imp.resolved_path:
                # Add edge for resolved import
                edges.append({
                    "source": file.filepath,
                    "target": imp.resolved_path,
                    "type": "import",
                    "color": "#64748b",  # Gray
                })
            else:
                # Add error for unresolved import
                error = {
                    "id": f"error_{len(errors)}",
                    "type": "import_error",
                    "severity": "error",
                    "message": f"Unresolved import: {imp.name}",
                    "location": {
                        "file": file.filepath,
                        "line": getattr(imp, "line", None),
                    }
                }
                errors.append(error)
                
                # Add error node
                error_node = {
                    "id": error["id"],
                    "label": f"Import Error: {imp.name}",
                    "type": "error",
                    "size": 15,
                    "color": "#ef4444",  # Red
                    "metadata": {
                        "error": error,
                    }
                }
                nodes.append(error_node)
                
                # Add edge from file to error
                edges.append({
                    "source": file.filepath,
                    "target": error["id"],
                    "type": "has_error",
                    "color": "#ef4444",  # Red
                })
    
    # Check for undefined symbols
    for file in codebase.files:
        for func in file.functions:
            if hasattr(func, "calls"):
                for call in func.calls:
                    if not call.resolved:
                        # Add error for unresolved call
                        error = {
                            "id": f"error_{len(errors)}",
                            "type": "undefined_symbol",
                            "severity": "warning",
                            "message": f"Undefined function call: {call.name}",
                            "location": {
                                "file": file.filepath,
                                "line": getattr(call, "line", None),
                            }
                        }
                        errors.append(error)
                        
                        # Add error node
                        error_node = {
                            "id": error["id"],
                            "label": f"Undefined: {call.name}",
                            "type": "error",
                            "size": 15,
                            "color": "#f97316",  # Orange
                            "metadata": {
                                "error": error,
                            }
                        }
                        nodes.append(error_node)
                        
                        # Get source node ID
                        source_id = f"{file.filepath}:{func.name}"
                        if hasattr(func, "parent") and func.parent:
                            source_id = f"{file.filepath}:{func.parent.name}.{func.name}"
                        
                        # Add function node if it doesn't exist
                        if not any(n["id"] == source_id for n in nodes):
                            func_node = {
                                "id": source_id,
                                "label": func.name,
                                "type": "function",
                                "size": 10,
                                "color": "#22c55e",  # Green
                                "metadata": {
                                    "file": file.filepath,
                                }
                            }
                            nodes.append(func_node)
                            
                            # Add edge from file to function
                            edges.append({
                                "source": file.filepath,
                                "target": source_id,
                                "type": "contains",
                                "color": "#64748b",  # Gray
                            })
                        
                        # Add edge from function to error
                        edges.append({
                            "source": source_id,
                            "target": error["id"],
                            "type": "has_error",
                            "color": "#f97316",  # Orange
                        })
    
    # Check for circular dependencies
    dependency_graph = create_dependency_graph(codebase)
    for cycle in dependency_graph["circular_dependencies"]:
        # Add error for circular dependency
        cycle_str = " -> ".join([Path(file).name for file in cycle])
        error = {
            "id": f"error_{len(errors)}",
            "type": "circular_dependency",
            "severity": "warning",
            "message": f"Circular dependency: {cycle_str}",
            "location": {
                "file": cycle[0],
            }
        }
        errors.append(error)
        
        # Add error node
        error_node = {
            "id": error["id"],
            "label": f"Circular Dependency",
            "type": "error",
            "size": 15,
            "color": "#eab308",  # Yellow
            "metadata": {
                "error": error,
                "cycle": cycle,
            }
        }
        nodes.append(error_node)
        
        # Add edges from files in cycle to error
        for file in cycle:
            edges.append({
                "source": file,
                "target": error["id"],
                "type": "has_error",
                "color": "#eab308",  # Yellow
            })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "errors": errors,
        "summary": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "total_issues": len(errors),
            "issues_by_type": {
                "import_error": len([e for e in errors if e["type"] == "import_error"]),
                "undefined_symbol": len([e for e in errors if e["type"] == "undefined_symbol"]),
                "circular_dependency": len([e for e in errors if e["type"] == "circular_dependency"]),
            }
        }
    }


def create_complexity_heatmap(codebase: Codebase) -> Dict[str, Any]:
    """
    Create a complexity heatmap of the codebase.
    
    Args:
        codebase: Codebase object
        
    Returns:
        Dictionary containing the complexity heatmap
    """
    nodes = []
    
    # Calculate complexity for each file and function
    file_complexity = {}
    function_complexity = []
    
    for file in codebase.files:
        file_metrics = {
            "file": file.filepath,
            "lines": len(file.source.split("\n")) if hasattr(file, "source") else 0,
            "functions": len(file.functions),
            "classes": len(file.classes),
            "complexity": 0,
        }
        
        # Calculate complexity for each function
        for func in file.functions:
            func_complexity = calculate_function_complexity(func)
            file_metrics["complexity"] += func_complexity
            
            function_complexity.append({
                "file": file.filepath,
                "function": func.name,
                "complexity": func_complexity,
                "lines": len(func.source.split("\n")) if hasattr(func, "source") else 0,
            })
        
        file_complexity[file.filepath] = file_metrics
    
    # Create nodes for each file
    for file_path, metrics in file_complexity.items():
        # Determine color based on complexity
        complexity = metrics["complexity"]
        color = "#22c55e"  # Green (low complexity)
        if complexity > 50:
            color = "#ef4444"  # Red (high complexity)
        elif complexity > 20:
            color = "#f97316"  # Orange (medium-high complexity)
        elif complexity > 10:
            color = "#eab308"  # Yellow (medium complexity)
        
        nodes.append({
            "id": file_path,
            "label": Path(file_path).name,
            "type": "file",
            "size": min(30, 10 + metrics["complexity"] / 5),  # Size based on complexity
            "color": color,
            "metadata": {
                "file": file_path,
                "complexity": metrics["complexity"],
                "functions": metrics["functions"],
                "classes": metrics["classes"],
                "lines": metrics["lines"],
            }
        })
    
    # Create nodes for complex functions
    for func in sorted(function_complexity, key=lambda f: f["complexity"], reverse=True)[:20]:
        # Determine color based on complexity
        complexity = func["complexity"]
        color = "#22c55e"  # Green (low complexity)
        if complexity > 15:
            color = "#ef4444"  # Red (high complexity)
        elif complexity > 10:
            color = "#f97316"  # Orange (medium-high complexity)
        elif complexity > 5:
            color = "#eab308"  # Yellow (medium complexity)
        
        nodes.append({
            "id": f"{func['file']}:{func['function']}",
            "label": func["function"],
            "type": "function",
            "size": min(20, 5 + func["complexity"]),  # Size based on complexity
            "color": color,
            "metadata": {
                "file": func["file"],
                "complexity": func["complexity"],
                "lines": func["lines"],
            }
        })
    
    return {
        "nodes": nodes,
        "complexity": {
            "file_complexity": file_complexity,
            "function_complexity": sorted(function_complexity, key=lambda f: f["complexity"], reverse=True)[:50],
            "summary": {
                "total_files": len(file_complexity),
                "total_functions": len(function_complexity),
                "avg_file_complexity": sum(f["complexity"] for f in file_complexity.values()) / len(file_complexity) if file_complexity else 0,
                "avg_function_complexity": sum(f["complexity"] for f in function_complexity) / len(function_complexity) if function_complexity else 0,
                "max_file_complexity": max(f["complexity"] for f in file_complexity.values()) if file_complexity else 0,
                "max_function_complexity": max(f["complexity"] for f in function_complexity) if function_complexity else 0,
            }
        },
        "summary": {
            "total_nodes": len(nodes),
            "total_issues": len([n for n in nodes if n["metadata"].get("complexity", 0) > 10]),
        }
    }


def calculate_function_complexity(func: Function) -> int:
    """
    Calculate cyclomatic complexity for a function.
    
    Args:
        func: Function object
        
    Returns:
        Complexity score
    """
    # This is a simplified calculation
    complexity = 1  # Base complexity
    
    # Count control flow statements
    if hasattr(func, "source"):
        source = func.source
        
        # Count if statements
        complexity += source.count("if ")
        
        # Count else if statements
        complexity += source.count("elif ")
        
        # Count for loops
        complexity += source.count("for ")
        
        # Count while loops
        complexity += source.count("while ")
        
        # Count and/or operators (each adds a path)
        complexity += source.count(" and ")
        complexity += source.count(" or ")
        
        # Count exception handlers
        complexity += source.count("except ")
    
    return complexity


def analyze_error_blast_radius(
    codebase: Codebase,
    symbol_name: str,
) -> Dict[str, Any]:
    """
    Analyze the blast radius of a symbol.
    
    Args:
        codebase: Codebase object
        symbol_name: Name of the symbol to analyze
        
    Returns:
        Dictionary containing the blast radius analysis
    """
    logger.info(f"Analyzing blast radius for symbol: {symbol_name}")
    
    # Find the symbol
    target_symbol = None
    for file in codebase.files:
        # Check functions
        for func in file.functions:
            if func.name == symbol_name:
                target_symbol = func
                break
        
        # Check classes
        if not target_symbol:
            for cls in file.classes:
                if cls.name == symbol_name:
                    target_symbol = cls
                    break
                
                # Check methods
                if hasattr(cls, "methods"):
                    for method in cls.methods:
                        if method.name == symbol_name or f"{cls.name}.{method.name}" == symbol_name:
                            target_symbol = method
                            break
    
    if not target_symbol:
        return {
            "error": f"Symbol '{symbol_name}' not found in the codebase",
        }
    
    # Get symbol info
    symbol_info = {
        "name": target_symbol.name,
        "type": target_symbol.__class__.__name__,
        "file": getattr(target_symbol, "file", None).filepath if hasattr(target_symbol, "file") else None,
    }
    
    # Find affected symbols
    affected_symbols = find_affected_symbols(codebase, target_symbol)
    
    # Create nodes and edges
    nodes = []
    edges = []
    
    # Add target symbol node
    target_node = {
        "id": f"{symbol_info['file']}:{symbol_info['name']}",
        "label": symbol_info["name"],
        "type": symbol_info["type"].lower(),
        "size": 30,
        "color": "#ef4444",  # Red
        "metadata": {
            "file": symbol_info["file"],
            "type": symbol_info["type"],
        }
    }
    nodes.append(target_node)
    
    # Add affected symbol nodes
    for symbol in affected_symbols:
        symbol_type = symbol["type"].lower()
        
        # Determine color based on type
        color = "#3b82f6"  # Blue (default)
        if symbol_type == "function":
            color = "#22c55e"  # Green
        elif symbol_type == "class":
            color = "#8b5cf6"  # Purple
        elif symbol_type == "method":
            color = "#ec4899"  # Pink
        
        node = {
            "id": f"{symbol['file']}:{symbol['name']}",
            "label": symbol["name"],
            "type": symbol_type,
            "size": 20,
            "color": color,
            "metadata": {
                "file": symbol["file"],
                "type": symbol["type"],
                "distance": symbol["distance"],
            }
        }
        nodes.append(node)
        
        # Add edge
        if symbol["caller"]:
            edges.append({
                "source": f"{symbol['caller']['file']}:{symbol['caller']['name']}",
                "target": f"{symbol['file']}:{symbol['name']}",
                "type": "calls",
                "color": "#64748b",  # Gray
            })
        else:
            # Direct dependency on target
            edges.append({
                "source": f"{symbol_info['file']}:{symbol_info['name']}",
                "target": f"{symbol['file']}:{symbol['name']}",
                "type": "affects",
                "color": "#ef4444",  # Red
            })
    
    return {
        "symbol": symbol_info,
        "blast_radius": {
            "affected_symbols": affected_symbols,
            "affected_files": list(set(symbol["file"] for symbol in affected_symbols)),
            "affected_nodes": len(affected_symbols),
        },
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "total_issues": 0,
        }
    }


def find_affected_symbols(codebase: Codebase, target_symbol: Symbol) -> List[Dict[str, Any]]:
    """
    Find symbols affected by a target symbol.
    
    Args:
        codebase: Codebase object
        target_symbol: Target symbol
        
    Returns:
        List of affected symbols
    """
    affected_symbols = []
    visited = set()
    
    def traverse_callers(symbol, distance=1, caller=None):
        # Get symbol ID
        symbol_id = f"{symbol.file.filepath}:{symbol.name}" if hasattr(symbol, "file") else f"{symbol.name}"
        
        # Skip if already visited
        if symbol_id in visited:
            return
        
        visited.add(symbol_id)
        
        # Add to affected symbols
        affected_symbols.append({
            "name": symbol.name,
            "type": symbol.__class__.__name__,
            "file": symbol.file.filepath if hasattr(symbol, "file") else None,
            "distance": distance,
            "caller": caller,
        })
        
        # Find callers
        for file in codebase.files:
            for func in file.functions:
                if hasattr(func, "calls"):
                    for call in func.calls:
                        if hasattr(call, "resolved_symbol") and call.resolved_symbol == symbol:
                            # Found a caller
                            caller_info = {
                                "name": func.name,
                                "type": func.__class__.__name__,
                                "file": file.filepath,
                            }
                            
                            # Recursively traverse callers
                            traverse_callers(func, distance + 1, caller_info)
    
    # Start traversal from target symbol
    traverse_callers(target_symbol)
    
    return affected_symbols
