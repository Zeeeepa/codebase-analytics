#!/usr/bin/env python3
"""
Consolidated Visualization Module

This module contains all the core visualization functions used by the API,
consolidated from the visualization folder to eliminate redundancy and 
provide a clean interface for generating visual representations.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patches as mpatches
    HAS_VIZ_DEPS = True
except ImportError:
    print("Visualization dependencies not found. Please install them with: pip install networkx matplotlib")
    plt = None
    nx = None
    HAS_VIZ_DEPS = False

# Import from Codegen SDK
from codegen.sdk.core.codebase import Codebase
from codegen.sdk.core.class_definition import Class
from codegen.sdk.core.file import SourceFile
from codegen.sdk.core.function import Function
from codegen.sdk.core.symbol import Symbol
from codegen.sdk.enums import EdgeType, SymbolType

# Configuration and Types

class VisualizationType(str, Enum):
    """Types of visualizations that can be generated."""
    CALL_GRAPH = "call_graph"
    DEPENDENCY_GRAPH = "dependency_graph"
    CLASS_HIERARCHY = "class_hierarchy"
    COMPLEXITY_HEATMAP = "complexity_heatmap"
    ISSUES_HEATMAP = "issues_heatmap"
    BLAST_RADIUS = "blast_radius"
    MODULE_DEPENDENCIES = "module_dependencies"
    DEAD_CODE = "dead_code"

class OutputFormat(str, Enum):
    """Output formats for visualizations."""
    PNG = "png"
    SVG = "svg"
    PDF = "pdf"
    HTML = "html"
    JSON = "json"

@dataclass
class VisualizationConfig:
    """Configuration for visualization generation."""
    output_directory: Optional[str] = None
    output_format: OutputFormat = OutputFormat.PNG
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    show_labels: bool = True
    show_legend: bool = True
    color_scheme: str = "default"
    max_nodes: int = 100
    layout_algorithm: str = "spring"

# Color schemes and styling

COLOR_SCHEMES = {
    "default": {
        "function": "#3498db",
        "class": "#e74c3c", 
        "module": "#2ecc71",
        "external": "#95a5a6",
        "critical": "#e74c3c",
        "major": "#f39c12",
        "minor": "#3498db",
        "background": "#ffffff",
        "text": "#2c3e50"
    },
    "dark": {
        "function": "#5dade2",
        "class": "#ec7063",
        "module": "#58d68d", 
        "external": "#aab7b8",
        "critical": "#ec7063",
        "major": "#f7dc6f",
        "minor": "#5dade2",
        "background": "#2c3e50",
        "text": "#ecf0f1"
    }
}

# Core Visualization Functions

def create_call_graph(codebase: Codebase, function_name: str = None, max_depth: int = 3, config: VisualizationConfig = None) -> Dict[str, Any]:
    """Create a call graph visualization."""
    if not HAS_VIZ_DEPS:
        return {
            "error": "Visualization dependencies not available. Please install: pip install networkx matplotlib",
            "type": "call_graph",
            "config": config.__dict__ if config else {}
        }
    
    config = config or VisualizationConfig()
    colors = COLOR_SCHEMES[config.color_scheme]
    
    G = nx.DiGraph()
    
    # If specific function is provided, start from there
    if function_name:
        target_functions = [f for f in codebase.functions if f.name == function_name]
        if not target_functions:
            return {"error": f"Function '{function_name}' not found"}
        start_function = target_functions[0]
    else:
        # Use all functions if no specific function is provided
        start_function = None
    
    # Build the graph
    if start_function:
        _build_call_graph_recursive(G, start_function, codebase, max_depth, 0)
    else:
        # Add all functions and their relationships
        for func in codebase.functions[:config.max_nodes]:
            G.add_node(func.name, type="function", color=colors["function"])
            
            # Add edges for function calls (simplified)
            if hasattr(func, 'called_functions'):
                for called in func.called_functions:
                    G.add_edge(func.name, called.name)
    
    # Generate layout
    if config.layout_algorithm == "hierarchical":
        pos = nx.spring_layout(G, k=1, iterations=50)
    else:
        pos = nx.spring_layout(G)
    
    # Create visualization data
    nodes = []
    edges = []
    
    for node in G.nodes(data=True):
        nodes.append({
            "id": node[0],
            "label": node[0],
            "type": node[1].get("type", "function"),
            "color": node[1].get("color", colors["function"]),
            "x": pos[node[0]][0],
            "y": pos[node[0]][1]
        })
    
    for edge in G.edges():
        edges.append({
            "source": edge[0],
            "target": edge[1]
        })
    
    return {
        "type": "call_graph",
        "nodes": nodes,
        "edges": edges,
        "config": config.__dict__
    }

def _build_call_graph_recursive(graph, function: Function, codebase: Codebase, max_depth: int, current_depth: int):
    """Recursively build call graph."""
    if current_depth >= max_depth:
        return
    
    colors = COLOR_SCHEMES["default"]
    graph.add_node(function.name, type="function", color=colors["function"])
    
    # Add called functions (simplified - would need proper call analysis)
    if hasattr(function, 'called_functions'):
        for called in function.called_functions:
            if called.name not in graph:
                _build_call_graph_recursive(graph, called, codebase, max_depth, current_depth + 1)
            graph.add_edge(function.name, called.name)

def create_dependency_graph(codebase: Codebase, module_path: str = None, config: VisualizationConfig = None) -> Dict[str, Any]:
    """Create a dependency graph visualization."""
    if not HAS_VIZ_DEPS:
        return {
            "error": "Visualization dependencies not available. Please install: pip install networkx matplotlib",
            "type": "dependency_graph",
            "config": config.__dict__ if config else {}
        }
    
    config = config or VisualizationConfig()
    colors = COLOR_SCHEMES[config.color_scheme]
    
    G = nx.DiGraph()
    
    # Build dependency graph from imports
    for file in codebase.files:
        if module_path and not file.filepath.startswith(module_path):
            continue
            
        module_name = file.filepath.replace('/', '.').replace('.py', '')
        G.add_node(module_name, type="module", color=colors["module"])
        
        # Add import dependencies
        for import_stmt in file.imports:
            imported_module = import_stmt.module_name
            if imported_module:
                G.add_node(imported_module, type="external", color=colors["external"])
                G.add_edge(module_name, imported_module)
    
    # Generate layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Create visualization data
    nodes = []
    edges = []
    
    for node in G.nodes(data=True):
        nodes.append({
            "id": node[0],
            "label": node[0],
            "type": node[1].get("type", "module"),
            "color": node[1].get("color", colors["module"]),
            "x": pos[node[0]][0],
            "y": pos[node[0]][1]
        })
    
    for edge in G.edges():
        edges.append({
            "source": edge[0],
            "target": edge[1]
        })
    
    return {
        "type": "dependency_graph",
        "nodes": nodes,
        "edges": edges,
        "config": config.__dict__
    }

def create_class_hierarchy(codebase: Codebase, config: VisualizationConfig = None) -> Dict[str, Any]:
    """Create a class hierarchy visualization."""
    if not HAS_VIZ_DEPS:
        return {
            "error": "Visualization dependencies not available. Please install: pip install networkx matplotlib",
            "type": "class_hierarchy",
            "config": config.__dict__ if config else {}
        }
    
    config = config or VisualizationConfig()
    colors = COLOR_SCHEMES[config.color_scheme]
    
    G = nx.DiGraph()
    
    # Build class hierarchy
    for cls in codebase.classes:
        G.add_node(cls.name, type="class", color=colors["class"])
        
        # Add inheritance relationships
        for superclass in cls.superclasses:
            G.add_node(superclass, type="class", color=colors["class"])
            G.add_edge(superclass, cls.name)  # Parent -> Child
    
    # Generate hierarchical layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Create visualization data
    nodes = []
    edges = []
    
    for node in G.nodes(data=True):
        nodes.append({
            "id": node[0],
            "label": node[0],
            "type": node[1].get("type", "class"),
            "color": node[1].get("color", colors["class"]),
            "x": pos[node[0]][0],
            "y": pos[node[0]][1]
        })
    
    for edge in G.edges():
        edges.append({
            "source": edge[0],
            "target": edge[1]
        })
    
    return {
        "type": "class_hierarchy",
        "nodes": nodes,
        "edges": edges,
        "config": config.__dict__
    }

def create_complexity_heatmap(codebase: Codebase, config: VisualizationConfig = None) -> Dict[str, Any]:
    """Create a complexity heatmap visualization."""
    if not HAS_VIZ_DEPS:
        return {
            "error": "Visualization dependencies not available. Please install: pip install networkx matplotlib",
            "type": "complexity_heatmap",
            "config": config.__dict__ if config else {}
        }
    from .analysis import calculate_cyclomatic_complexity
    
    config = config or VisualizationConfig()
    
    # Calculate complexity for all functions
    complexity_data = []
    
    for file in codebase.files:
        for func in file.functions:
            complexity = calculate_cyclomatic_complexity(func)
            complexity_data.append({
                "file": file.filepath,
                "function": func.name,
                "complexity": complexity,
                "line_start": func.start_point[0] if hasattr(func, 'start_point') else 0,
                "line_end": func.end_point[0] if hasattr(func, 'end_point') else 0
            })
    
    # Sort by complexity
    complexity_data.sort(key=lambda x: x["complexity"], reverse=True)
    
    return {
        "type": "complexity_heatmap",
        "data": complexity_data[:config.max_nodes],
        "config": config.__dict__
    }

def create_issues_heatmap(file_issues: Dict[str, Dict], config: VisualizationConfig = None) -> Dict[str, Any]:
    """Create an issues heatmap visualization."""
    if not HAS_VIZ_DEPS:
        return {
            "error": "Visualization dependencies not available. Please install: pip install networkx matplotlib",
            "type": "issues_heatmap",
            "config": config.__dict__ if config else {}
        }
    config = config or VisualizationConfig()
    colors = COLOR_SCHEMES[config.color_scheme]
    
    # Process issues data
    issues_data = []
    
    for filepath, issues in file_issues.items():
        total_issues = sum(len(issues[severity]) for severity in ['critical', 'major', 'minor'])
        critical_count = len(issues.get('critical', []))
        major_count = len(issues.get('major', []))
        minor_count = len(issues.get('minor', []))
        
        # Determine severity level
        if critical_count > 0:
            severity = "critical"
            color = colors["critical"]
        elif major_count > 0:
            severity = "major"
            color = colors["major"]
        elif minor_count > 0:
            severity = "minor"
            color = colors["minor"]
        else:
            continue
        
        issues_data.append({
            "file": filepath,
            "total_issues": total_issues,
            "critical": critical_count,
            "major": major_count,
            "minor": minor_count,
            "severity": severity,
            "color": color
        })
    
    # Sort by total issues
    issues_data.sort(key=lambda x: x["total_issues"], reverse=True)
    
    return {
        "type": "issues_heatmap",
        "data": issues_data[:config.max_nodes],
        "config": config.__dict__
    }

def create_blast_radius(codebase: Codebase, symbol_name: str, max_depth: int = 2, config: VisualizationConfig = None) -> Dict[str, Any]:
    """Create a blast radius visualization showing impact of changes to a symbol."""
    if not HAS_VIZ_DEPS:
        return {
            "error": "Visualization dependencies not available. Please install: pip install networkx matplotlib",
            "type": "blast_radius",
            "config": config.__dict__ if config else {}
        }
    
    config = config or VisualizationConfig()
    colors = COLOR_SCHEMES[config.color_scheme]
    
    G = nx.DiGraph()
    
    # Find the target symbol
    target_symbol = None
    for symbol in codebase.symbols:
        if symbol.name == symbol_name:
            target_symbol = symbol
            break
    
    if not target_symbol:
        return {"error": f"Symbol '{symbol_name}' not found"}
    
    # Build blast radius graph
    _build_blast_radius_recursive(G, target_symbol, codebase, max_depth, 0, colors)
    
    # Generate layout
    pos = nx.spring_layout(G, k=1.5, iterations=50)
    
    # Create visualization data
    nodes = []
    edges = []
    
    for node in G.nodes(data=True):
        nodes.append({
            "id": node[0],
            "label": node[0],
            "type": node[1].get("type", "symbol"),
            "color": node[1].get("color", colors["function"]),
            "depth": node[1].get("depth", 0),
            "x": pos[node[0]][0],
            "y": pos[node[0]][1]
        })
    
    for edge in G.edges():
        edges.append({
            "source": edge[0],
            "target": edge[1]
        })
    
    return {
        "type": "blast_radius",
        "center_symbol": symbol_name,
        "nodes": nodes,
        "edges": edges,
        "config": config.__dict__
    }

def _build_blast_radius_recursive(graph, symbol: Symbol, codebase: Codebase, max_depth: int, current_depth: int, colors: Dict):
    """Recursively build blast radius graph."""
    if current_depth >= max_depth:
        return
    
    # Determine node color based on depth
    if current_depth == 0:
        color = colors["critical"]  # Center node
    elif current_depth == 1:
        color = colors["major"]     # Direct dependencies
    else:
        color = colors["minor"]     # Indirect dependencies
    
    graph.add_node(symbol.name, type="symbol", color=color, depth=current_depth)
    
    # Add dependencies and usages
    if hasattr(symbol, 'dependencies'):
        for dep in symbol.dependencies:
            if dep.name not in graph:
                _build_blast_radius_recursive(graph, dep, codebase, max_depth, current_depth + 1, colors)
            graph.add_edge(symbol.name, dep.name)
    
    if hasattr(symbol, 'usages'):
        for usage in symbol.usages:
            # Find the symbol that uses this one
            using_symbol = _find_symbol_at_location(codebase, usage.filepath, usage.start_point[0] if hasattr(usage, 'start_point') else 0)
            if using_symbol and using_symbol.name not in graph:
                _build_blast_radius_recursive(graph, using_symbol, codebase, max_depth, current_depth + 1, colors)
                graph.add_edge(using_symbol.name, symbol.name)

def _find_symbol_at_location(codebase: Codebase, filepath: str, line: int) -> Optional[Symbol]:
    """Find a symbol at a specific location."""
    # This is a simplified implementation
    for symbol in codebase.symbols:
        if (symbol.filepath == filepath and 
            hasattr(symbol, 'start_point') and hasattr(symbol, 'end_point') and
            symbol.start_point[0] <= line <= symbol.end_point[0]):
            return symbol
    return None

# Utility Functions

def save_visualization(viz_data: Dict[str, Any], output_path: str, format: OutputFormat = OutputFormat.JSON):
    """Save visualization data to file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if format == OutputFormat.JSON:
        with open(output_path, 'w') as f:
            json.dump(viz_data, f, indent=2)
    elif format == OutputFormat.HTML:
        html_content = generate_html_visualization(viz_data)
        with open(output_path, 'w') as f:
            f.write(html_content)
    else:
        # For image formats, would need matplotlib implementation
        raise NotImplementedError(f"Format {format} not yet implemented")

def generate_html_visualization(viz_data: Dict[str, Any]) -> str:
    """Generate HTML visualization using D3.js or similar."""
    # This is a basic template - would need proper D3.js implementation
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{viz_data['type'].title()} Visualization</title>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .node {{ stroke: #fff; stroke-width: 1.5px; }}
            .link {{ stroke: #999; stroke-opacity: 0.6; }}
        </style>
    </head>
    <body>
        <h1>{viz_data['type'].title()} Visualization</h1>
        <div id="visualization"></div>
        <script>
            const data = {json.dumps(viz_data)};
            // D3.js visualization code would go here
            console.log('Visualization data:', data);
        </script>
    </body>
    </html>
    """
    return html_template

# High-level API Functions

def generate_all_visualizations(codebase: Codebase, file_issues: Dict = None, output_dir: str = "visualizations", config: VisualizationConfig = None) -> Dict[str, Any]:
    """Generate all available visualizations for a codebase."""
    config = config or VisualizationConfig(output_directory=output_dir)
    results = {}
    
    try:
        # Call graph
        results["call_graph"] = create_call_graph(codebase, config=config)
        
        # Dependency graph
        results["dependency_graph"] = create_dependency_graph(codebase, config=config)
        
        # Class hierarchy
        if codebase.classes:
            results["class_hierarchy"] = create_class_hierarchy(codebase, config=config)
        
        # Complexity heatmap
        results["complexity_heatmap"] = create_complexity_heatmap(codebase, config=config)
        
        # Issues heatmap
        if file_issues:
            results["issues_heatmap"] = create_issues_heatmap(file_issues, config=config)
        
    except Exception as e:
        logging.error(f"Error generating visualizations: {e}")
        results["error"] = str(e)
    
    return results

def get_visualization_summary(codebase: Codebase) -> Dict[str, Any]:
    """Get a summary of available visualizations for a codebase."""
    return {
        "total_functions": len(codebase.functions),
        "total_classes": len(codebase.classes),
        "total_files": len(codebase.files),
        "total_symbols": len(codebase.symbols),
        "available_visualizations": [
            "call_graph",
            "dependency_graph", 
            "class_hierarchy" if codebase.classes else None,
            "complexity_heatmap",
            "blast_radius"
        ],
        "recommended_visualizations": _get_recommended_visualizations(codebase)
    }

def _get_recommended_visualizations(codebase: Codebase) -> List[str]:
    """Get recommended visualizations based on codebase characteristics."""
    recommendations = []
    
    if len(codebase.functions) > 10:
        recommendations.append("call_graph")
    
    if len(codebase.files) > 5:
        recommendations.append("dependency_graph")
    
    if len(codebase.classes) > 3:
        recommendations.append("class_hierarchy")
    
    if len(codebase.functions) > 20:
        recommendations.append("complexity_heatmap")
    
    return recommendations
