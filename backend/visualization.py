#!/usr/bin/env python3
"""
Consolidated Visualization Module

This module contains ALL visualization functions consolidated from:
- visualize.py (existing visualization functions)
- enhanced_visualizations.py (comprehensive visualization features)
- All files in backend/visualization/ folder (specialized visualizations)

Organized into logical sections for maintainability.
"""

import json
import math
import os
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
from dataclasses import asdict
from enum import Enum
from pathlib import Path

# Import from Codegen SDK
from codegen.sdk.core.codebase import Codebase
from codegen.sdk.core.class_definition import Class
from codegen.sdk.core.file import SourceFile
from codegen.sdk.core.function import Function
from codegen.sdk.core.symbol import Symbol

# Import analysis modules for visualization data
from analysis import (
    DependencyAnalysis,
    CallGraphAnalysis,
    CodeQualityMetrics,
    ArchitecturalInsights,
    SecurityAnalysis,
    PerformanceAnalysis,
    AnalysisType,
    perform_comprehensive_analysis
)

# ============================================================================
# SECTION 1: VISUALIZATION CONFIGURATION AND ENUMS
# ============================================================================

class VisualizationType(str, Enum):
    """Types of visualizations available."""
    CALL_GRAPH = "call_graph"
    DEPENDENCY_GRAPH = "dependency_graph"
    CLASS_HIERARCHY = "class_hierarchy"
    COMPLEXITY_HEATMAP = "complexity_heatmap"
    ISSUES_HEATMAP = "issues_heatmap"
    BLAST_RADIUS = "blast_radius"
    ENHANCED_DEPENDENCY_GRAPH = "enhanced_dependency_graph"
    CALL_FLOW_DIAGRAM = "call_flow_diagram"
    QUALITY_HEATMAP = "quality_heatmap"
    ARCHITECTURAL_OVERVIEW = "architectural_overview"
    SECURITY_RISK_MAP = "security_risk_map"
    PERFORMANCE_HOTSPOT_MAP = "performance_hotspot_map"
    COMPREHENSIVE_DASHBOARD = "comprehensive_dashboard"

class OutputFormat(str, Enum):
    """Output formats for visualizations."""
    JSON = "json"
    HTML = "html"
    SVG = "svg"
    PNG = "png"
    PDF = "pdf"

class VisualizationConfig:
    """Configuration for visualizations."""
    
    def __init__(
        self,
        output_format: OutputFormat = OutputFormat.JSON,
        max_nodes: int = 100,
        max_edges: int = 200,
        layout: str = "force",
        color_scheme: str = "default",
        show_labels: bool = True,
        show_weights: bool = False,
        node_size_factor: float = 1.0,
        edge_width_factor: float = 1.0,
        include_metadata: bool = True,
        interactive: bool = True,
        width: int = 800,
        height: int = 600
    ):
        self.output_format = output_format
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.layout = layout
        self.color_scheme = color_scheme
        self.show_labels = show_labels
        self.show_weights = show_weights
        self.node_size_factor = node_size_factor
        self.edge_width_factor = edge_width_factor
        self.include_metadata = include_metadata
        self.interactive = interactive
        self.width = width
        self.height = height

# ============================================================================
# SECTION 2: BASIC VISUALIZATION FUNCTIONS (from visualize.py)
# ============================================================================

def create_call_graph(codebase: Codebase, function_name: str = None, max_depth: int = 3, config: VisualizationConfig = None) -> Dict[str, Any]:
    """Create a call graph visualization."""
    if config is None:
        config = VisualizationConfig()
    
    nodes = []
    edges = []
    node_id_map = {}
    
    # Get all functions
    functions = codebase.functions
    if function_name:
        # Filter to specific function and its calls
        target_function = next((f for f in functions if f.name == function_name), None)
        if not target_function:
            return {"nodes": [], "edges": [], "metadata": {"error": f"Function '{function_name}' not found"}}
        functions = [target_function]
    
    # Create nodes
    for i, func in enumerate(functions[:config.max_nodes]):
        node_id = f"func_{i}"
        node_id_map[func.name] = node_id
        
        # Calculate node properties
        complexity = getattr(func, 'complexity', 1)
        node_size = max(10, min(50, complexity * config.node_size_factor))
        
        nodes.append({
            "id": node_id,
            "label": func.name,
            "title": f"Function: {func.name}\\nComplexity: {complexity}",
            "size": node_size,
            "color": get_node_color(complexity, config.color_scheme),
            "group": "function",
            "metadata": {
                "name": func.name,
                "filepath": getattr(func, 'filepath', 'unknown'),
                "complexity": complexity,
                "parameters": len(getattr(func, 'parameters', [])),
                "lines": len(getattr(func, 'code_block', {}).get('source', '').split('\n')) if hasattr(func, 'code_block') else 0
            }
        })
    
    # Create edges for function calls
    edge_id = 0
    for func in functions:
        source_id = node_id_map.get(func.name)
        if not source_id:
            continue
            
        # Get function calls
        calls = []
        if hasattr(func, 'function_calls') and func.function_calls:
            calls = [call.name for call in func.function_calls if hasattr(call, 'name')]
        elif hasattr(func, 'code_block') and func.code_block:
            # Extract calls from code
            calls = extract_function_calls_from_code(func.code_block.source, [f.name for f in functions])
        
        for called_func in calls[:config.max_edges]:
            target_id = node_id_map.get(called_func)
            if target_id and len(edges) < config.max_edges:
                edges.append({
                    "id": f"edge_{edge_id}",
                    "from": source_id,
                    "to": target_id,
                    "arrows": "to",
                    "color": "#666666",
                    "width": config.edge_width_factor,
                    "title": f"{func.name} calls {called_func}"
                })
                edge_id += 1
    
    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "type": "call_graph",
            "total_functions": len(functions),
            "total_calls": len(edges),
            "layout": config.layout,
            "max_depth": max_depth,
            "root_function": function_name
        }
    }

def create_dependency_graph(codebase: Codebase, module_path: str = None, config: VisualizationConfig = None) -> Dict[str, Any]:
    """Create a dependency graph visualization."""
    if config is None:
        config = VisualizationConfig()
    
    nodes = []
    edges = []
    node_id_map = {}
    
    # Get all files
    files = codebase.files
    if module_path:
        files = [f for f in files if module_path in f.filepath]
    
    # Create nodes for files
    for i, file in enumerate(files[:config.max_nodes]):
        node_id = f"file_{i}"
        node_id_map[file.filepath] = node_id
        
        # Calculate node properties
        file_size = len(file.source.split('\n')) if file.source else 0
        node_size = max(15, min(60, file_size / 10 * config.node_size_factor))
        
        # Determine file type for coloring
        file_type = get_file_type(file.filepath)
        
        nodes.append({
            "id": node_id,
            "label": os.path.basename(file.filepath),
            "title": f"File: {file.filepath}\\nLines: {file_size}\\nType: {file_type}",
            "size": node_size,
            "color": get_file_type_color(file_type, config.color_scheme),
            "group": file_type,
            "metadata": {
                "filepath": file.filepath,
                "file_type": file_type,
                "lines_of_code": file_size,
                "functions": len(file.functions),
                "classes": len(file.classes)
            }
        })
    
    # Create edges for imports/dependencies
    edge_id = 0
    for file in files:
        source_id = node_id_map.get(file.filepath)
        if not source_id:
            continue
            
        # Get imports
        imports = []
        if hasattr(file, 'imports') and file.imports:
            imports = [getattr(imp, 'module', getattr(imp, 'name', 'unknown')) for imp in file.imports]
        
        for imported_module in imports:
            # Find corresponding file
            target_file = next((f for f in files if imported_module in f.filepath), None)
            if target_file:
                target_id = node_id_map.get(target_file.filepath)
                if target_id and len(edges) < config.max_edges:
                    edges.append({
                        "id": f"edge_{edge_id}",
                        "from": source_id,
                        "to": target_id,
                        "arrows": "to",
                        "color": "#999999",
                        "width": config.edge_width_factor,
                        "title": f"{file.filepath} imports {imported_module}"
                    })
                    edge_id += 1
    
    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "type": "dependency_graph",
            "total_files": len(files),
            "total_dependencies": len(edges),
            "layout": config.layout,
            "module_filter": module_path
        }
    }

def create_class_hierarchy(codebase: Codebase, config: VisualizationConfig = None) -> Dict[str, Any]:
    """Create a class hierarchy visualization."""
    if config is None:
        config = VisualizationConfig()
    
    nodes = []
    edges = []
    node_id_map = {}
    
    # Get all classes
    classes = codebase.classes[:config.max_nodes]
    
    # Create nodes for classes
    for i, cls in enumerate(classes):
        node_id = f"class_{i}"
        node_id_map[cls.name] = node_id
        
        # Calculate node properties
        method_count = len(getattr(cls, 'methods', []))
        attribute_count = len(getattr(cls, 'attributes', []))
        node_size = max(20, min(80, (method_count + attribute_count) * config.node_size_factor))
        
        # Determine if it's a base class or derived class
        is_base_class = len(getattr(cls, 'superclasses', [])) == 0
        
        nodes.append({
            "id": node_id,
            "label": cls.name,
            "title": f"Class: {cls.name}\\nMethods: {method_count}\\nAttributes: {attribute_count}",
            "size": node_size,
            "color": "#ff6b6b" if is_base_class else "#4ecdc4",
            "shape": "box",
            "group": "base_class" if is_base_class else "derived_class",
            "metadata": {
                "name": cls.name,
                "filepath": getattr(cls, 'filepath', 'unknown'),
                "methods": method_count,
                "attributes": attribute_count,
                "superclasses": getattr(cls, 'superclasses', []),
                "is_base_class": is_base_class
            }
        })
    
    # Create edges for inheritance
    edge_id = 0
    for cls in classes:
        if hasattr(cls, 'superclasses') and cls.superclasses:
            source_id = node_id_map.get(cls.name)
            if not source_id:
                continue
                
            for superclass in cls.superclasses:
                target_id = node_id_map.get(superclass)
                if target_id and len(edges) < config.max_edges:
                    edges.append({
                        "id": f"edge_{edge_id}",
                        "from": source_id,
                        "to": target_id,
                        "arrows": "to",
                        "color": "#333333",
                        "width": 2 * config.edge_width_factor,
                        "title": f"{cls.name} inherits from {superclass}",
                        "dashes": False
                    })
                    edge_id += 1
    
    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "type": "class_hierarchy",
            "total_classes": len(classes),
            "inheritance_relationships": len(edges),
            "layout": "hierarchical"
        }
    }

def create_complexity_heatmap(codebase: Codebase, config: VisualizationConfig = None) -> Dict[str, Any]:
    """Create a complexity heatmap visualization."""
    if config is None:
        config = VisualizationConfig()
    
    heatmap_data = []
    
    # Analyze functions for complexity
    for func in codebase.functions:
        complexity = getattr(func, 'complexity', 1)
        
        # Calculate additional metrics
        lines = len(getattr(func, 'code_block', {}).get('source', '').split('\n')) if hasattr(func, 'code_block') else 0
        parameters = len(getattr(func, 'parameters', []))
        
        # Calculate complexity score
        complexity_score = complexity + (lines / 10) + (parameters * 2)
        
        heatmap_data.append({
            "function": func.name,
            "filepath": getattr(func, 'filepath', 'unknown'),
            "complexity": complexity,
            "lines_of_code": lines,
            "parameters": parameters,
            "complexity_score": complexity_score,
            "color": get_complexity_color(complexity_score),
            "size": max(10, min(50, complexity_score))
        })
    
    # Sort by complexity score
    heatmap_data.sort(key=lambda x: x['complexity_score'], reverse=True)
    
    return {
        "data": heatmap_data[:config.max_nodes],
        "metadata": {
            "type": "complexity_heatmap",
            "total_functions": len(heatmap_data),
            "max_complexity": max(item['complexity_score'] for item in heatmap_data) if heatmap_data else 0,
            "avg_complexity": sum(item['complexity_score'] for item in heatmap_data) / len(heatmap_data) if heatmap_data else 0,
            "color_scheme": config.color_scheme
        }
    }

def create_blast_radius(codebase: Codebase, symbol_name: str, max_depth: int = 2, config: VisualizationConfig = None) -> Dict[str, Any]:
    """Create a blast radius visualization showing impact of changes."""
    if config is None:
        config = VisualizationConfig()
    
    # Find the target symbol
    target_symbol = None
    for symbol in codebase.symbols:
        if symbol.name == symbol_name:
            target_symbol = symbol
            break
    
    if not target_symbol:
        return {"nodes": [], "edges": [], "metadata": {"error": f"Symbol '{symbol_name}' not found"}}
    
    nodes = []
    edges = []
    visited = set()
    
    def add_symbol_and_dependencies(symbol, depth, center_x=400, center_y=300, radius=200):
        if depth > max_depth or symbol.name in visited or len(nodes) >= config.max_nodes:
            return
        
        visited.add(symbol.name)
        
        # Calculate position in circle
        angle = len(nodes) * (2 * math.pi / min(config.max_nodes, 20))
        x = center_x + radius * math.cos(angle) * (1 - depth / max_depth)
        y = center_y + radius * math.sin(angle) * (1 - depth / max_depth)
        
        # Determine node properties based on depth
        node_size = max(15, 40 - depth * 10) * config.node_size_factor
        node_color = get_blast_radius_color(depth, config.color_scheme)
        
        nodes.append({
            "id": symbol.name,
            "label": symbol.name,
            "title": f"Symbol: {symbol.name}\\nType: {getattr(symbol, 'symbol_type', 'unknown')}\\nDepth: {depth}",
            "size": node_size,
            "color": node_color,
            "x": x,
            "y": y,
            "group": f"depth_{depth}",
            "metadata": {
                "name": symbol.name,
                "type": getattr(symbol, 'symbol_type', 'unknown'),
                "filepath": getattr(symbol, 'filepath', 'unknown'),
                "depth": depth,
                "is_target": depth == 0
            }
        })
        
        # Add dependencies
        if hasattr(symbol, 'dependencies'):
            for dep in symbol.dependencies[:5]:  # Limit dependencies
                if len(edges) < config.max_edges:
                    edges.append({
                        "id": f"{symbol.name}_to_{dep.name}",
                        "from": symbol.name,
                        "to": dep.name,
                        "arrows": "to",
                        "color": get_edge_color(depth, config.color_scheme),
                        "width": max(1, 3 - depth) * config.edge_width_factor,
                        "title": f"{symbol.name} depends on {dep.name}"
                    })
                
                add_symbol_and_dependencies(dep, depth + 1, center_x, center_y, radius)
        
        # Add usages (reverse dependencies)
        if hasattr(symbol, 'usages'):
            for usage in symbol.usages[:5]:  # Limit usages
                if len(edges) < config.max_edges:
                    edges.append({
                        "id": f"{usage.name}_uses_{symbol.name}",
                        "from": usage.name,
                        "to": symbol.name,
                        "arrows": "to",
                        "color": get_edge_color(depth, config.color_scheme),
                        "width": max(1, 3 - depth) * config.edge_width_factor,
                        "title": f"{usage.name} uses {symbol.name}",
                        "dashes": True
                    })
                
                add_symbol_and_dependencies(usage, depth + 1, center_x, center_y, radius)
    
    # Start from target symbol
    add_symbol_and_dependencies(target_symbol, 0)
    
    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "type": "blast_radius",
            "target_symbol": symbol_name,
            "max_depth": max_depth,
            "total_affected": len(nodes),
            "layout": "circular"
        }
    }

# ============================================================================
# SECTION 3: ENHANCED VISUALIZATION FUNCTIONS (from enhanced_visualizations.py)
# ============================================================================

def create_enhanced_dependency_graph(codebase: Codebase, analysis: DependencyAnalysis = None) -> Dict[str, Any]:
    """Create an enhanced dependency graph visualization."""
    if analysis is None:
        results = perform_comprehensive_analysis(codebase, [AnalysisType.DEPENDENCY])
        analysis = results.get('dependency_analysis')
    
    if not analysis:
        return {"nodes": [], "edges": [], "metadata": {}}
    
    nodes = []
    edges = []
    node_id_map = {}
    
    # Create nodes for each file/module
    for i, (filepath, deps) in enumerate(analysis.dependency_graph.items()):
        node_id = f"node_{i}"
        node_id_map[filepath] = node_id
        
        # Determine node type and color
        is_external = filepath in analysis.external_dependencies
        is_critical = filepath in analysis.critical_dependencies
        
        node_color = "#ff6b6b" if is_critical else "#4ecdc4" if is_external else "#45b7d1"
        node_size = 20 + (len(deps) * 2)  # Size based on number of dependencies
        
        nodes.append({
            "id": node_id,
            "label": filepath.split('/')[-1],  # Just filename
            "title": filepath,
            "color": node_color,
            "size": node_size,
            "group": "external" if is_external else "internal",
            "dependencies_count": len(deps),
            "is_critical": is_critical
        })
    
    # Create edges for dependencies
    edge_id = 0
    for filepath, deps in analysis.dependency_graph.items():
        source_id = node_id_map.get(filepath)
        if not source_id:
            continue
            
        for dep in deps:
            target_id = node_id_map.get(dep)
            if target_id:
                edges.append({
                    "id": f"edge_{edge_id}",
                    "from": source_id,
                    "to": target_id,
                    "arrows": "to",
                    "color": "#ff6b6b" if dep in analysis.critical_dependencies else "#999999"
                })
                edge_id += 1
    
    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "total_dependencies": analysis.total_dependencies,
            "circular_dependencies_count": len(analysis.circular_dependencies),
            "dependency_depth": analysis.dependency_depth,
            "critical_dependencies": analysis.critical_dependencies,
            "layout": "hierarchical"
        }
    }

def create_comprehensive_dashboard_data(codebase: Codebase) -> Dict[str, Any]:
    """Create comprehensive dashboard data with all visualizations."""
    # Perform all analyses
    comprehensive_results = perform_comprehensive_analysis(codebase)
    
    dashboard_data = {
        "dependency_graph": create_enhanced_dependency_graph(
            codebase, comprehensive_results.get('dependency_analysis')
        ),
        "basic_call_graph": create_call_graph(codebase),
        "basic_dependency_graph": create_dependency_graph(codebase),
        "class_hierarchy": create_class_hierarchy(codebase),
        "complexity_heatmap": create_complexity_heatmap(codebase),
        "metadata": {
            "generated_at": "2024-01-01T00:00:00Z",
            "total_files": len(codebase.files),
            "total_functions": len(codebase.functions),
            "total_classes": len(codebase.classes),
            "analysis_types": [at.value for at in AnalysisType],
            "dashboard_version": "2.0.0"
        }
    }
    
    return dashboard_data

# ============================================================================
# SECTION 4: UTILITY FUNCTIONS
# ============================================================================

def extract_function_calls_from_code(code: str, known_functions: List[str]) -> List[str]:
    """Extract function calls from code using simple pattern matching."""
    calls = []
    pattern = r'(\w+)\s*\('
    matches = re.findall(pattern, code)
    
    for match in matches:
        if match in known_functions:
            calls.append(match)
    
    return calls

def get_file_type(filepath: str) -> str:
    """Get the type of file based on its extension."""
    ext = Path(filepath).suffix.lower()
    if ext in ['.py', '.pyi', '.pyx']:
        return 'python'
    elif ext in ['.js', '.jsx', '.ts', '.tsx']:
        return 'javascript'
    elif ext in ['.java']:
        return 'java'
    elif ext in ['.c', '.cpp', '.h', '.hpp']:
        return 'cpp'
    elif ext in ['.go']:
        return 'go'
    elif ext in ['.rs']:
        return 'rust'
    elif ext in ['.html', '.htm']:
        return 'html'
    elif ext in ['.css', '.scss', '.sass']:
        return 'css'
    elif ext in ['.json']:
        return 'json'
    elif ext in ['.md', '.markdown']:
        return 'markdown'
    else:
        return 'unknown'

def get_node_color(complexity: int, color_scheme: str = "default") -> str:
    """Get node color based on complexity."""
    if color_scheme == "default":
        if complexity <= 5:
            return "#4ecdc4"  # Low complexity - green
        elif complexity <= 10:
            return "#45b7d1"  # Medium complexity - blue
        elif complexity <= 20:
            return "#feca57"  # High complexity - yellow
        else:
            return "#ff6b6b"  # Very high complexity - red
    else:
        return "#45b7d1"  # Default blue

def get_file_type_color(file_type: str, color_scheme: str = "default") -> str:
    """Get color based on file type."""
    colors = {
        'python': '#3776ab',
        'javascript': '#f7df1e',
        'java': '#ed8b00',
        'cpp': '#00599c',
        'go': '#00add8',
        'rust': '#000000',
        'html': '#e34f26',
        'css': '#1572b6',
        'json': '#000000',
        'markdown': '#083fa1',
        'unknown': '#666666'
    }
    return colors.get(file_type, '#666666')

def get_complexity_color(complexity_score: float) -> str:
    """Get color based on complexity score."""
    if complexity_score <= 10:
        return "#4ecdc4"  # Green
    elif complexity_score <= 20:
        return "#45b7d1"  # Blue
    elif complexity_score <= 30:
        return "#feca57"  # Yellow
    else:
        return "#ff6b6b"  # Red

def get_blast_radius_color(depth: int, color_scheme: str = "default") -> str:
    """Get color based on blast radius depth."""
    colors = ["#ff6b6b", "#feca57", "#45b7d1", "#4ecdc4", "#96ceb4"]
    return colors[min(depth, len(colors) - 1)]

def get_edge_color(depth: int, color_scheme: str = "default") -> str:
    """Get edge color based on depth."""
    alpha = max(0.3, 1.0 - (depth * 0.2))
    return f"rgba(102, 102, 102, {alpha})"

def generate_color_palette(count: int) -> List[str]:
    """Generate a color palette for visualizations."""
    base_colors = [
        "#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#feca57",
        "#ff9ff3", "#54a0ff", "#5f27cd", "#00d2d3", "#ff9f43"
    ]
    
    if count <= len(base_colors):
        return base_colors[:count]
    
    # Generate additional colors
    colors = base_colors.copy()
    for i in range(count - len(base_colors)):
        # Generate colors based on HSL
        hue = (i * 137.508) % 360  # Golden angle approximation
        colors.append(f"hsl({hue}, 70%, 60%)")
    
    return colors

def save_visualization(viz_data: Dict[str, Any], output_path: str, format: OutputFormat = OutputFormat.JSON):
    """Save visualization data to file."""
    if format == OutputFormat.JSON:
        with open(output_path, 'w') as f:
            json.dump(viz_data, f, indent=2)
    elif format == OutputFormat.HTML:
        html_content = generate_html_visualization(viz_data)
        with open(output_path, 'w') as f:
            f.write(html_content)
    else:
        raise ValueError(f"Unsupported output format: {format}")

def generate_html_visualization(viz_data: Dict[str, Any]) -> str:
    """Generate HTML visualization from data."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Codebase Visualization</title>
        <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            #visualization {{ width: 100%; height: 600px; border: 1px solid #ccc; }}
            .info {{ margin: 10px; padding: 10px; background: #f5f5f5; }}
        </style>
    </head>
    <body>
        <div class="info">
            <h2>Codebase Visualization</h2>
            <p>Type: {type}</p>
            <p>Nodes: {node_count}</p>
            <p>Edges: {edge_count}</p>
        </div>
        <div id="visualization"></div>
        <script>
            var nodes = new vis.DataSet({nodes});
            var edges = new vis.DataSet({edges});
            var container = document.getElementById('visualization');
            var data = {{ nodes: nodes, edges: edges }};
            var options = {{
                layout: {{ randomSeed: 2 }},
                physics: {{ enabled: true }}
            }};
            var network = new vis.Network(container, data, options);
        </script>
    </body>
    </html>
    """
    
    return html_template.format(
        type=viz_data.get('metadata', {}).get('type', 'unknown'),
        node_count=len(viz_data.get('nodes', [])),
        edge_count=len(viz_data.get('edges', [])),
        nodes=json.dumps(viz_data.get('nodes', [])),
        edges=json.dumps(viz_data.get('edges', []))
    )

def generate_all_visualizations(codebase: Codebase, output_dir: str = "visualizations", config: VisualizationConfig = None) -> Dict[str, Any]:
    """Generate all available visualizations."""
    if config is None:
        config = VisualizationConfig()
    
    os.makedirs(output_dir, exist_ok=True)
    
    visualizations = {
        "call_graph": create_call_graph(codebase, config=config),
        "dependency_graph": create_dependency_graph(codebase, config=config),
        "class_hierarchy": create_class_hierarchy(codebase, config=config),
        "complexity_heatmap": create_complexity_heatmap(codebase, config=config),
        "enhanced_dependency_graph": create_enhanced_dependency_graph(codebase),
        "comprehensive_dashboard": create_comprehensive_dashboard_data(codebase)
    }
    
    # Save each visualization
    for viz_name, viz_data in visualizations.items():
        output_path = os.path.join(output_dir, f"{viz_name}.{config.output_format.value}")
        save_visualization(viz_data, output_path, config.output_format)
    
    return visualizations

def get_visualization_summary(codebase: Codebase) -> Dict[str, Any]:
    """Get a summary of available visualizations."""
    return {
        "available_visualizations": [vt.value for vt in VisualizationType],
        "recommended_visualizations": get_recommended_visualizations(codebase),
        "codebase_stats": {
            "files": len(codebase.files),
            "functions": len(codebase.functions),
            "classes": len(codebase.classes)
        }
    }

def get_recommended_visualizations(codebase: Codebase) -> List[str]:
    """Get recommended visualizations based on codebase characteristics."""
    recommendations = []
    
    if len(codebase.functions) > 10:
        recommendations.append(VisualizationType.CALL_GRAPH.value)
    
    if len(codebase.files) > 5:
        recommendations.append(VisualizationType.DEPENDENCY_GRAPH.value)
        recommendations.append(VisualizationType.ENHANCED_DEPENDENCY_GRAPH.value)
    
    if len(codebase.classes) > 3:
        recommendations.append(VisualizationType.CLASS_HIERARCHY.value)
    
    if len(codebase.functions) > 20:
        recommendations.append(VisualizationType.COMPLEXITY_HEATMAP.value)
    
    # Always recommend comprehensive dashboard for larger codebases
    if len(codebase.files) > 10:
        recommendations.append(VisualizationType.COMPREHENSIVE_DASHBOARD.value)
    
    return recommendations
