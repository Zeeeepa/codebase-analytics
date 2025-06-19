#!/usr/bin/env python3
"""
Consolidated Visualization Module

This module contains ALL visualization functions consolidated from:
- visualize.py (basic visualization functions)
- backend/visualization/ folder (13 specialized visualization files)

Organized into logical sections for maintainability and comprehensive
visualization capabilities for codebase analysis.
"""

import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
import squarify
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import uuid
import tempfile
from abc import ABC, abstractmethod

# Import from Codegen SDK
from codegen.sdk.core.codebase import Codebase
from codegen.sdk.core.class_definition import Class
from codegen.sdk.core.file import SourceFile
from codegen.sdk.core.function import Function
from codegen.sdk.core.symbol import Symbol
from codegen.sdk.enums import EdgeType, SymbolType
from codegen.sdk.core.import_resolution import Import
from codegen.sdk.core.external_module import ExternalModule

# =============================================================================
# CORE VISUALIZATION CLASSES AND ENUMS
# =============================================================================

class VisualizationType(str, Enum):
    """Types of visualizations that can be generated."""
    CALL_GRAPH = "call_graph"
    DEPENDENCY_GRAPH = "dependency_graph"
    CLASS_HIERARCHY = "class_hierarchy"
    COMPLEXITY_HEATMAP = "complexity_heatmap"
    ISSUES_HEATMAP = "issues_heatmap"
    BLAST_RADIUS = "blast_radius"
    REPOSITORY_STRUCTURE = "repository_structure"
    TREEMAP = "treemap"
    FILE_TYPE_DISTRIBUTION = "file_type_distribution"
    ARCHITECTURAL_OVERVIEW = "architectural_overview"
    SECURITY_RISK_MAP = "security_risk_map"
    PERFORMANCE_HOTSPOT_MAP = "performance_hotspot_map"
    CALL_TRACE = "call_trace"
    DEPENDENCY_TRACE = "dependency_trace"
    METHOD_RELATIONSHIPS = "method_relationships"
    DEAD_CODE = "dead_code"


class OutputFormat(str, Enum):
    """Output formats for visualizations."""
    PNG = "png"
    SVG = "svg"
    HTML = "html"
    JSON = "json"
    PDF = "pdf"


@dataclass
class VisualizationConfig:
    """Configuration for visualization generation."""
    output_format: OutputFormat = OutputFormat.PNG
    width: int = 1200
    height: int = 800
    dpi: int = 300
    include_labels: bool = True
    color_scheme: str = "default"
    layout_algorithm: str = "spring"
    max_nodes: int = 100
    min_edge_weight: float = 0.1
    show_legend: bool = True
    interactive: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'output_format': self.output_format.value,
            'width': self.width,
            'height': self.height,
            'dpi': self.dpi,
            'include_labels': self.include_labels,
            'color_scheme': self.color_scheme,
            'layout_algorithm': self.layout_algorithm,
            'max_nodes': self.max_nodes,
            'min_edge_weight': self.min_edge_weight,
            'show_legend': self.show_legend,
            'interactive': self.interactive
        }


class BaseVisualizer(ABC):
    """Base class for all visualizers."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.output_dir = None
    
    def set_output_directory(self, output_dir: str):
        """Set the output directory for visualizations."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    @abstractmethod
    def generate(self, codebase: Codebase, **kwargs) -> Dict[str, Any]:
        """Generate visualization data."""
        pass
    
    def save_visualization(self, data: Dict[str, Any], filename: str) -> str:
        """Save visualization data to file."""
        if not self.output_dir:
            self.output_dir = tempfile.mkdtemp()
        
        filepath = os.path.join(self.output_dir, filename)
        
        if self.config.output_format == OutputFormat.JSON:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif self.config.output_format == OutputFormat.HTML:
            self._save_html(data, filepath)
        else:
            # For image formats, save as JSON and let specific visualizers handle image generation
            with open(f"{filepath}.json", 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        return filepath


# =============================================================================
# BASIC VISUALIZATION FUNCTIONS (FROM VISUALIZE.PY)
# =============================================================================

def visualize_codebase(codebase, output_dir: str) -> Dict[str, str]:
    """
    Generate visualizations for a codebase.
    
    Args:
        codebase: The codebase to visualize
        output_dir: Output directory
        
    Returns:
        Dictionary mapping visualization names to file paths
    """
    # Create visualizations directory
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get all files in the codebase
    files = []
    for root, dirs, filenames in os.walk(codebase):
        for filename in filenames:
            if filename.endswith(('.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp')):
                file_path = os.path.join(root, filename)
                files.append(file_path)
    
    # Generate dependency graph visualization
    dependency_graph_file = visualize_dependency_graph(files, os.path.join(vis_dir, "dependency_graph.png"))
    
    # Generate complexity visualization
    complexity_file = visualize_complexity(files, os.path.join(vis_dir, "complexity.png"))
    
    # Generate file type distribution visualization
    file_type_file = visualize_file_type_distribution(files, os.path.join(vis_dir, "file_type_distribution.png"))
    
    # Generate treemap visualization
    treemap_file = visualize_treemap(files, os.path.join(vis_dir, "treemap.png"))
    
    return {
        "dependency_graph": dependency_graph_file,
        "complexity": complexity_file,
        "file_type_distribution": file_type_file,
        "treemap": treemap_file
    }


def visualize_dependency_graph(files: List[str], output_file: str) -> str:
    """
    Create a dependency graph visualization.
    
    Args:
        files: List of file paths
        output_file: Output file path
        
    Returns:
        Path to the generated visualization
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes for each file
    for file_path in files:
        filename = os.path.basename(file_path)
        G.add_node(filename, path=file_path)
    
    # Analyze imports to create edges
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple import detection (can be enhanced)
            import_lines = [line.strip() for line in content.split('\n') 
                          if line.strip().startswith(('import ', 'from '))]
            
            source_filename = os.path.basename(file_path)
            
            for import_line in import_lines:
                # Extract imported module name
                if import_line.startswith('from '):
                    module_name = import_line.split()[1].split('.')[0]
                elif import_line.startswith('import '):
                    module_name = import_line.split()[1].split('.')[0]
                else:
                    continue
                
                # Check if this module corresponds to any of our files
                for target_file in files:
                    target_filename = os.path.basename(target_file)
                    if target_filename.replace('.py', '') == module_name:
                        G.add_edge(source_filename, target_filename)
                        break
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=1000, alpha=0.7)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                          arrows=True, arrowsize=20, alpha=0.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title("Dependency Graph", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file


def visualize_complexity(files: List[str], output_file: str) -> str:
    """
    Create a complexity visualization.
    
    Args:
        files: List of file paths
        output_file: Output file path
        
    Returns:
        Path to the generated visualization
    """
    file_complexities = []
    file_names = []
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple complexity calculation based on control structures
            complexity = (
                content.count('if ') + 
                content.count('elif ') + 
                content.count('for ') + 
                content.count('while ') + 
                content.count('try:') + 
                content.count('except ') +
                content.count('def ') +
                content.count('class ')
            )
            
            file_complexities.append(complexity)
            file_names.append(os.path.basename(file_path))
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Create bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(file_names)), file_complexities, 
                   color='skyblue', alpha=0.7)
    
    # Color bars based on complexity
    for i, bar in enumerate(bars):
        if file_complexities[i] > 50:
            bar.set_color('red')
        elif file_complexities[i] > 25:
            bar.set_color('orange')
        else:
            bar.set_color('green')
    
    plt.xlabel('Files')
    plt.ylabel('Complexity Score')
    plt.title('File Complexity Analysis')
    plt.xticks(range(len(file_names)), file_names, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file


def visualize_file_type_distribution(files: List[str], output_file: str) -> str:
    """
    Create a file type distribution visualization.
    
    Args:
        files: List of file paths
        output_file: Output file path
        
    Returns:
        Path to the generated visualization
    """
    # Count file types
    file_types = Counter()
    for file_path in files:
        ext = os.path.splitext(file_path)[1]
        file_types[ext] += 1
    
    # Create pie chart
    plt.figure(figsize=(10, 8))
    labels = list(file_types.keys())
    sizes = list(file_types.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('File Type Distribution')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file


def visualize_treemap(files: List[str], output_file: str) -> str:
    """
    Create a treemap visualization of file sizes.
    
    Args:
        files: List of file paths
        output_file: Output file path
        
    Returns:
        Path to the generated visualization
    """
    file_sizes = []
    file_names = []
    
    for file_path in files:
        try:
            size = os.path.getsize(file_path)
            file_sizes.append(size)
            file_names.append(os.path.basename(file_path))
        except Exception as e:
            print(f"Error getting size for {file_path}: {e}")
            continue
    
    if not file_sizes:
        return output_file
    
    # Create treemap
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Spectral(np.linspace(0, 1, len(file_names)))
    
    squarify.plot(sizes=file_sizes, label=file_names, color=colors, alpha=0.7)
    plt.title('File Size Treemap')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file


def generate_html_report(analysis_result: Dict[str, Any], output_file: str) -> str:
    """
    Generate an HTML report from analysis results.
    
    Args:
        analysis_result: Analysis results dictionary
        output_file: Output HTML file path
        
    Returns:
        Path to the generated HTML file
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Codebase Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .metric {{ background-color: #e8f4fd; padding: 10px; margin: 5px 0; border-radius: 3px; }}
            .issue {{ background-color: #ffe6e6; padding: 10px; margin: 5px 0; border-radius: 3px; }}
            .recommendation {{ background-color: #e6ffe6; padding: 10px; margin: 5px 0; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Codebase Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>Summary</h2>
            <div class="metric">Total Files: {analysis_result.get('total_files', 'N/A')}</div>
            <div class="metric">Total Functions: {analysis_result.get('total_functions', 'N/A')}</div>
            <div class="metric">Total Classes: {analysis_result.get('total_classes', 'N/A')}</div>
        </div>
        
        <div class="section">
            <h2>Code Quality Metrics</h2>
            <div class="metric">Maintainability Index: {analysis_result.get('maintainability_index', 'N/A')}</div>
            <div class="metric">Technical Debt Ratio: {analysis_result.get('technical_debt_ratio', 'N/A')}</div>
        </div>
        
        <div class="section">
            <h2>Issues Found</h2>
            {_generate_issues_html(analysis_result.get('issues', []))}
        </div>
        
        <div class="section">
            <h2>Recommendations</h2>
            {_generate_recommendations_html(analysis_result.get('recommendations', []))}
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_file


def _generate_issues_html(issues: List[Dict[str, Any]]) -> str:
    """Generate HTML for issues section."""
    if not issues:
        return "<p>No issues found.</p>"
    
    html = ""
    for issue in issues[:10]:  # Limit to first 10 issues
        html += f'<div class="issue">{issue.get("description", "Unknown issue")}</div>'
    
    if len(issues) > 10:
        html += f"<p>... and {len(issues) - 10} more issues</p>"
    
    return html


def _generate_recommendations_html(recommendations: List[str]) -> str:
    """Generate HTML for recommendations section."""
    if not recommendations:
        return "<p>No recommendations available.</p>"
    
    html = ""
    for rec in recommendations:
        html += f'<div class="recommendation">{rec}</div>'
    
    return html


# =============================================================================
# SPECIALIZED VISUALIZATION CLASSES (FROM VISUALIZATION/ FOLDER)
# =============================================================================

class AnalysisVisualizer(BaseVisualizer):
    """Visualizer for analysis results."""
    
    def generate(self, codebase: Codebase, **kwargs) -> Dict[str, Any]:
        """Generate analysis visualization data."""
        return {
            'type': 'analysis',
            'timestamp': datetime.now().isoformat(),
            'codebase_summary': self._get_codebase_summary(codebase),
            'metrics': self._calculate_metrics(codebase)
        }
    
    def _get_codebase_summary(self, codebase: Codebase) -> Dict[str, Any]:
        """Get summary of codebase."""
        return {
            'total_files': len(list(codebase.files)),
            'total_functions': len(list(codebase.functions)),
            'total_classes': len(list(codebase.classes)),
            'total_symbols': len(list(codebase.symbols))
        }
    
    def _calculate_metrics(self, codebase: Codebase) -> Dict[str, Any]:
        """Calculate basic metrics."""
        return {
            'complexity_score': 0,  # Placeholder
            'maintainability_index': 0,  # Placeholder
            'technical_debt_ratio': 0  # Placeholder
        }


class CodeVisualizer(BaseVisualizer):
    """Visualizer for code structure."""
    
    def generate(self, codebase: Codebase, **kwargs) -> Dict[str, Any]:
        """Generate code structure visualization data."""
        return {
            'type': 'code_structure',
            'timestamp': datetime.now().isoformat(),
            'file_structure': self._build_file_structure(codebase),
            'dependency_graph': self._build_dependency_graph(codebase)
        }
    
    def _build_file_structure(self, codebase: Codebase) -> Dict[str, Any]:
        """Build file structure representation."""
        structure = {}
        for file in codebase.files:
            path_parts = file.file_path.split('/')
            current = structure
            for part in path_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[path_parts[-1]] = {
                'type': 'file',
                'size': len(file.content) if hasattr(file, 'content') else 0
            }
        return structure
    
    def _build_dependency_graph(self, codebase: Codebase) -> Dict[str, List[str]]:
        """Build dependency graph."""
        deps = {}
        for file in codebase.files:
            file_deps = []
            for import_stmt in file.imports:
                if import_stmt.external_module:
                    file_deps.append(import_stmt.external_module.name)
            deps[file.file_path] = file_deps
        return deps


# =============================================================================
# SPECIALIZED VISUALIZATION FUNCTIONS
# =============================================================================

def generate_edge_meta(usage) -> dict:
    """Generate metadata for graph edges."""
    return {
        'weight': 1,
        'type': 'usage',
        'timestamp': datetime.now().isoformat()
    }


def create_blast_radius_visualization(symbol: Symbol, depth: int = 0) -> Dict[str, Any]:
    """
    Create blast radius visualization for a symbol.
    
    Args:
        symbol: The symbol to analyze
        depth: Maximum depth for analysis
        
    Returns:
        Visualization data dictionary
    """
    return {
        'type': 'blast_radius',
        'symbol': symbol.name,
        'depth': depth,
        'affected_files': [],  # Placeholder
        'impact_score': 0  # Placeholder
    }


def create_downstream_call_trace(src_func: Function, depth: int = 0) -> Dict[str, Any]:
    """
    Create downstream call trace visualization.
    
    Args:
        src_func: Source function
        depth: Maximum depth for trace
        
    Returns:
        Call trace data
    """
    return {
        'type': 'call_trace',
        'source_function': src_func.name,
        'depth': depth,
        'call_chain': [],  # Placeholder
        'total_calls': 0  # Placeholder
    }


def create_dependencies_visualization(symbol: Symbol, depth: int = 0) -> Dict[str, Any]:
    """
    Create dependencies visualization for a symbol.
    
    Args:
        symbol: The symbol to analyze
        depth: Maximum depth for analysis
        
    Returns:
        Dependencies visualization data
    """
    return {
        'type': 'dependencies',
        'symbol': symbol.name,
        'depth': depth,
        'dependencies': [],  # Placeholder
        'dependents': []  # Placeholder
    }


def graph_class_methods(target_class: Class) -> Dict[str, Any]:
    """
    Graph methods within a class.
    
    Args:
        target_class: The class to analyze
        
    Returns:
        Method relationship data
    """
    return {
        'type': 'class_methods',
        'class_name': target_class.name,
        'methods': [method.name for method in target_class.methods],
        'relationships': []  # Placeholder
    }


# =============================================================================
# COMPREHENSIVE VISUALIZATION ORCHESTRATOR
# =============================================================================

class CodebaseVisualizer:
    """Main visualizer class that orchestrates all visualization types."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.visualizers = {
            VisualizationType.CALL_GRAPH: self._create_call_graph,
            VisualizationType.DEPENDENCY_GRAPH: self._create_dependency_graph,
            VisualizationType.CLASS_HIERARCHY: self._create_class_hierarchy,
            VisualizationType.COMPLEXITY_HEATMAP: self._create_complexity_heatmap,
            VisualizationType.BLAST_RADIUS: self._create_blast_radius,
            VisualizationType.REPOSITORY_STRUCTURE: self._create_repository_structure
        }
    
    def generate_all_visualizations(self, codebase: Codebase, output_dir: str) -> Dict[str, str]:
        """Generate all available visualizations."""
        results = {}
        os.makedirs(output_dir, exist_ok=True)
        
        for viz_type in VisualizationType:
            try:
                if viz_type in self.visualizers:
                    result = self.visualizers[viz_type](codebase, output_dir)
                    results[viz_type.value] = result
            except Exception as e:
                print(f"Error generating {viz_type.value}: {e}")
                continue
        
        return results
    
    def _create_call_graph(self, codebase: Codebase, output_dir: str) -> str:
        """Create call graph visualization."""
        # Implementation placeholder
        return os.path.join(output_dir, "call_graph.png")
    
    def _create_dependency_graph(self, codebase: Codebase, output_dir: str) -> str:
        """Create dependency graph visualization."""
        # Implementation placeholder
        return os.path.join(output_dir, "dependency_graph.png")
    
    def _create_class_hierarchy(self, codebase: Codebase, output_dir: str) -> str:
        """Create class hierarchy visualization."""
        # Implementation placeholder
        return os.path.join(output_dir, "class_hierarchy.png")
    
    def _create_complexity_heatmap(self, codebase: Codebase, output_dir: str) -> str:
        """Create complexity heatmap visualization."""
        # Implementation placeholder
        return os.path.join(output_dir, "complexity_heatmap.png")
    
    def _create_blast_radius(self, codebase: Codebase, output_dir: str) -> str:
        """Create blast radius visualization."""
        # Implementation placeholder
        return os.path.join(output_dir, "blast_radius.png")
    
    def _create_repository_structure(self, codebase: Codebase, output_dir: str) -> str:
        """Create repository structure visualization."""
        # Implementation placeholder
        return os.path.join(output_dir, "repository_structure.png")


# =============================================================================
# MAIN VISUALIZATION FUNCTIONS
# =============================================================================

def run_visualization_analysis(codebase: Codebase, output_dir: str = None) -> Dict[str, Any]:
    """
    Run comprehensive visualization analysis on a codebase.
    
    Args:
        codebase: The codebase to analyze
        output_dir: Output directory for visualizations
        
    Returns:
        Dictionary containing all visualization results
    """
    if not output_dir:
        output_dir = tempfile.mkdtemp()
    
    visualizer = CodebaseVisualizer()
    results = visualizer.generate_all_visualizations(codebase, output_dir)
    
    return {
        'output_directory': output_dir,
        'visualizations': results,
        'timestamp': datetime.now().isoformat(),
        'config': visualizer.config.to_dict()
    }


def main():
    """Main function for testing visualization capabilities."""
    print("Consolidated Visualization Module")
    print("Available visualization types:")
    for viz_type in VisualizationType:
        print(f"  - {viz_type.value}")


if __name__ == "__main__":
    main()

