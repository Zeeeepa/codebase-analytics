#!/usr/bin/env python3
"""
Visualization Module for Codebase Analysis

This module provides visualization functions for codebase analysis results.
It includes functions for generating various types of visualizations, such as:
- Dependency graphs
- Call graphs
- Issue visualizations
- Code quality visualizations
- Repository structure visualizations
"""

import os
import json
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
import uuid
from datetime import datetime
import tempfile
import base64
from io import BytesIO

# Import from analysis module
from analysis import (
    DependencyAnalysis, CallGraphAnalysis, CodeQualityResult, IssueCollection,
    AnalysisSummary, Issue, IssueSeverity, IssueCategory, IssueStatus,
    CodeLocation, find_issues_in_file
)

# Import from Codegen SDK
from codegen.sdk.core.codebase import Codebase
from codegen.sdk.core.file import SourceFile

# ============================================================================
# SECTION 1: VISUALIZATION UTILITIES
# ============================================================================

def create_figure(width: int = 10, height: int = 8, dpi: int = 100) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a matplotlib figure and axes.
    
    Args:
        width: Figure width in inches
        height: Figure height in inches
        dpi: Dots per inch
        
    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    return fig, ax

def save_figure_to_file(fig: plt.Figure, filename: str) -> str:
    """
    Save a matplotlib figure to a file.
    
    Args:
        fig: Matplotlib figure
        filename: Output filename
        
    Returns:
        Path to the saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save the figure
    fig.savefig(filename, bbox_inches='tight')
    
    return filename

def figure_to_base64(fig: plt.Figure) -> str:
    """
    Convert a matplotlib figure to a base64-encoded string.
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        Base64-encoded string
    """
    # Save figure to a BytesIO object
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    # Encode as base64
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str

def create_color_map(values: List[float], cmap_name: str = 'viridis') -> List[str]:
    """
    Create a color map for a list of values.
    
    Args:
        values: List of values to map to colors
        cmap_name: Name of the colormap to use
        
    Returns:
        List of colors as hex strings
    """
    # Normalize values to [0, 1]
    if len(values) == 0:
        return []
    
    min_val = min(values)
    max_val = max(values)
    
    if min_val == max_val:
        normalized = [0.5] * len(values)
    else:
        normalized = [(val - min_val) / (max_val - min_val) for val in values]
    
    # Create colormap
    cmap = plt.get_cmap(cmap_name)
    
    # Map values to colors
    colors = [cmap(norm) for norm in normalized]
    
    # Convert to hex strings
    hex_colors = [plt.colors.rgb2hex(color) for color in colors]
    
    return hex_colors

# ============================================================================
# SECTION 2: DEPENDENCY GRAPH VISUALIZATION
# ============================================================================

def visualize_dependency_graph(dependency_analysis: DependencyAnalysis, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Visualize the dependency graph.
    
    Args:
        dependency_analysis: DependencyAnalysis object
        output_file: Optional output file path
        
    Returns:
        Dictionary containing visualization data
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    for file, deps in dependency_analysis.dependency_graph.items():
        G.add_node(file)
        for dep in deps:
            G.add_edge(file, dep)
    
    # Create figure
    fig, ax = create_figure(12, 10)
    
    # Calculate node positions using a layout algorithm
    pos = nx.spring_layout(G)
    
    # Calculate node sizes based on degree
    node_sizes = [10 + (G.in_degree(node) + G.out_degree(node)) * 5 for node in G.nodes()]
    
    # Calculate node colors based on betweenness centrality
    betweenness = nx.betweenness_centrality(G)
    node_colors = create_color_map(list(betweenness.values()))
    
    # Draw the graph
    nx.draw_networkx(
        G, pos, ax=ax,
        with_labels=False,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.8,
        arrows=True,
        arrowsize=10,
        width=0.5
    )
    
    # Draw node labels for important nodes
    important_nodes = [node for node in G.nodes() if G.in_degree(node) + G.out_degree(node) > 5]
    important_labels = {node: os.path.basename(node) for node in important_nodes}
    nx.draw_networkx_labels(G, pos, labels=important_labels, font_size=8)
    
    # Highlight circular dependencies
    circular_edges = []
    for cycle in dependency_analysis.circular_dependencies:
        for i in range(len(cycle)):
            source = cycle[i]
            target = cycle[(i + 1) % len(cycle)]
            circular_edges.append((source, target))
    
    nx.draw_networkx_edges(
        G, pos,
        edgelist=circular_edges,
        edge_color='red',
        width=2
    )
    
    # Highlight critical dependencies
    critical_nodes = dependency_analysis.critical_dependencies
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=critical_nodes,
        node_color='red',
        node_size=[node_sizes[list(G.nodes()).index(node)] * 1.5 for node in critical_nodes]
    )
    
    # Set title and labels
    ax.set_title('Dependency Graph')
    ax.set_axis_off()
    
    # Save figure if output file is specified
    if output_file:
        save_figure_to_file(fig, output_file)
    
    # Create visualization data
    visualization = {
        "image": figure_to_base64(fig),
        "nodes": len(G.nodes()),
        "edges": len(G.edges()),
        "circular_dependencies": len(dependency_analysis.circular_dependencies),
        "critical_dependencies": len(dependency_analysis.critical_dependencies)
    }
    
    # Close the figure to free memory
    plt.close(fig)
    
    return visualization



# ============================================================================
# SECTION 3: CALL GRAPH VISUALIZATION
# ============================================================================

def visualize_call_graph(call_graph_analysis: CallGraphAnalysis, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Visualize the call graph.
    
    Args:
        call_graph_analysis: CallGraphAnalysis object
        output_file: Optional output file path
        
    Returns:
        Dictionary containing visualization data
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    for func, called in call_graph_analysis.call_graph.items():
        G.add_node(func)
        for called_func in called:
            G.add_edge(func, called_func)
    
    # Create figure
    fig, ax = create_figure(12, 10)
    
    # Calculate node positions using a layout algorithm
    pos = nx.spring_layout(G)
    
    # Calculate node sizes based on connectivity
    node_sizes = []
    for node in G.nodes():
        connectivity = call_graph_analysis.function_connectivity.get(node, {})
        in_degree = connectivity.get("in_degree", 0)
        out_degree = connectivity.get("out_degree", 0)
        betweenness = connectivity.get("betweenness", 0)
        
        size = 10 + (in_degree + out_degree) * 5 + betweenness * 100
        node_sizes.append(size)
    
    # Calculate node colors based on type
    node_colors = []
    for node in G.nodes():
        if node in call_graph_analysis.entry_points:
            node_colors.append('green')
        elif node in call_graph_analysis.leaf_functions:
            node_colors.append('blue')
        else:
            node_colors.append('gray')
    
    # Draw the graph
    nx.draw_networkx(
        G, pos, ax=ax,
        with_labels=False,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.8,
        arrows=True,
        arrowsize=10,
        width=0.5
    )
    
    # Draw node labels for important nodes
    important_nodes = [node for node in G.nodes() if G.in_degree(node) + G.out_degree(node) > 5]
    important_labels = {node: node for node in important_nodes}
    nx.draw_networkx_labels(G, pos, labels=important_labels, font_size=8)
    
    # Set title and labels
    ax.set_title('Call Graph')
    ax.set_axis_off()
    
    # Add legend
    ax.plot([], [], 'o', color='green', label='Entry Points')
    ax.plot([], [], 'o', color='blue', label='Leaf Functions')
    ax.plot([], [], 'o', color='gray', label='Internal Functions')
    ax.legend()
    
    # Save figure if output file is specified
    if output_file:
        save_figure_to_file(fig, output_file)
    
    # Create visualization data
    visualization = {
        "image": figure_to_base64(fig),
        "nodes": len(G.nodes()),
        "edges": len(G.edges()),
        "entry_points": len(call_graph_analysis.entry_points),
        "leaf_functions": len(call_graph_analysis.leaf_functions),
        "max_call_depth": call_graph_analysis.max_call_depth
    }
    
    # Close the figure to free memory
    plt.close(fig)
    
    return visualization

# ============================================================================
# SECTION 4: ISSUE VISUALIZATION
# ============================================================================

def visualize_issues(issue_collection: IssueCollection, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Visualize the issues.
    
    Args:
        issue_collection: IssueCollection object
        output_file: Optional output file path
        
    Returns:
        Dictionary containing visualization data
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Count issues by severity
    severity_counts = issue_collection.count_by_severity()
    severities = list(severity_counts.keys())
    counts = list(severity_counts.values())
    
    # Create bar chart for severity
    ax1.bar(severities, counts, color=['red', 'orange', 'yellow', 'blue'])
    ax1.set_title('Issues by Severity')
    ax1.set_xlabel('Severity')
    ax1.set_ylabel('Count')
    
    # Count issues by category
    category_counts = issue_collection.count_by_category()
    categories = list(category_counts.keys())
    counts = list(category_counts.values())
    
    # Sort by count
    sorted_indices = np.argsort(counts)[::-1]
    categories = [categories[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]
    
    # Take top 10 categories
    if len(categories) > 10:
        categories = categories[:10]
        counts = counts[:10]
    
    # Create bar chart for category
    ax2.barh(categories, counts, color='skyblue')
    ax2.set_title('Top Issues by Category')
    ax2.set_xlabel('Count')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output file is specified
    if output_file:
        save_figure_to_file(fig, output_file)
    
    # Create visualization data
    visualization = {
        "image": figure_to_base64(fig),
        "total_issues": len(issue_collection.issues),
        "by_severity": issue_collection.count_by_severity(),
        "by_category": issue_collection.count_by_category(),
        "by_status": issue_collection.count_by_status()
    }
    
    # Close the figure to free memory
    plt.close(fig)
    
    return visualization

def visualize_issues_by_file(issue_collection: IssueCollection, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Visualize issues by file.
    
    Args:
        issue_collection: IssueCollection object
        output_file: Optional output file path
        
    Returns:
        Dictionary containing visualization data
    """
    # Count issues by file
    issues_by_file = {}
    for issue in issue_collection.issues:
        file_path = issue.location.file_path
        if file_path not in issues_by_file:
            issues_by_file[file_path] = {
                "total": 0,
                "critical": 0,
                "major": 0,
                "minor": 0,
                "info": 0
            }
        
        issues_by_file[file_path]["total"] += 1
        if issue.severity == IssueSeverity.CRITICAL:
            issues_by_file[file_path]["critical"] += 1
        elif issue.severity == IssueSeverity.MAJOR:
            issues_by_file[file_path]["major"] += 1
        elif issue.severity == IssueSeverity.MINOR:
            issues_by_file[file_path]["minor"] += 1
        elif issue.severity == IssueSeverity.INFO:
            issues_by_file[file_path]["info"] += 1
    
    # Sort files by total issues
    sorted_files = sorted(issues_by_file.items(), key=lambda x: x[1]["total"], reverse=True)
    
    # Take top 20 files
    if len(sorted_files) > 20:
        sorted_files = sorted_files[:20]
    
    # Create figure
    fig, ax = create_figure(12, 10)
    
    # Extract data for plotting
    files = [os.path.basename(file) for file, _ in sorted_files]
    critical = [data["critical"] for _, data in sorted_files]
    major = [data["major"] for _, data in sorted_files]
    minor = [data["minor"] for _, data in sorted_files]
    info = [data["info"] for _, data in sorted_files]
    
    # Create stacked bar chart
    ax.barh(files, critical, color='red', label='Critical')
    ax.barh(files, major, left=critical, color='orange', label='Major')
    ax.barh(files, minor, left=[c + m for c, m in zip(critical, major)], color='yellow', label='Minor')
    ax.barh(files, info, left=[c + m + n for c, m, n in zip(critical, major, minor)], color='blue', label='Info')
    
    # Set title and labels
    ax.set_title('Issues by File')
    ax.set_xlabel('Count')
    ax.set_ylabel('File')
    
    # Add legend
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output file is specified
    if output_file:
        save_figure_to_file(fig, output_file)
    
    # Create visualization data
    visualization = {
        "image": figure_to_base64(fig),
        "issues_by_file": issues_by_file
    }
    
    # Close the figure to free memory
    plt.close(fig)
    
    return visualization



# ============================================================================
# SECTION 5: CODE QUALITY VISUALIZATION
# ============================================================================

def visualize_code_quality(code_quality_result: CodeQualityResult, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Visualize code quality metrics.
    
    Args:
        code_quality_result: CodeQualityResult object
        output_file: Optional output file path
        
    Returns:
        Dictionary containing visualization data
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Create radar chart for metrics
    metrics = [
        'Maintainability',
        'Complexity',
        'Duplication',
        'Technical Debt',
        'Comment Density'
    ]
    
    # Normalize metrics to [0, 1]
    maintainability = code_quality_result.maintainability_index / 100
    complexity = min(1, code_quality_result.cyclomatic_complexity / 20)
    duplication = min(1, code_quality_result.duplication_percentage / 100)
    technical_debt = min(1, code_quality_result.technical_debt_ratio / 100)
    comment_density = code_quality_result.comment_density
    
    # Invert complexity and technical debt (lower is better)
    complexity = 1 - complexity
    duplication = 1 - duplication
    technical_debt = 1 - technical_debt
    
    values = [
        maintainability,
        complexity,
        duplication,
        technical_debt,
        comment_density
    ]
    
    # Add the first value again to close the polygon
    metrics.append(metrics[0])
    values.append(values[0])
    
    # Convert to radians
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    # Create radar chart
    ax1.plot(angles, values, 'o-', linewidth=2)
    ax1.fill(angles, values, alpha=0.25)
    ax1.set_thetagrids(np.degrees(angles[:-1]), metrics[:-1])
    ax1.set_ylim(0, 1)
    ax1.set_title('Code Quality Metrics')
    ax1.grid(True)
    
    # Create bar chart for issues
    issue_counts = code_quality_result.issues.count_by_severity()
    severities = list(issue_counts.keys())
    counts = list(issue_counts.values())
    
    ax2.bar(severities, counts, color=['red', 'orange', 'yellow', 'blue'])
    ax2.set_title('Issues by Severity')
    ax2.set_xlabel('Severity')
    ax2.set_ylabel('Count')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output file is specified
    if output_file:
        save_figure_to_file(fig, output_file)
    
    # Create visualization data
    visualization = {
        "image": figure_to_base64(fig),
        "maintainability_index": code_quality_result.maintainability_index,
        "cyclomatic_complexity": code_quality_result.cyclomatic_complexity,
        "halstead_volume": code_quality_result.halstead_volume,
        "source_lines_of_code": code_quality_result.source_lines_of_code,
        "comment_density": code_quality_result.comment_density,
        "duplication_percentage": code_quality_result.duplication_percentage,
        "technical_debt_ratio": code_quality_result.technical_debt_ratio,
        "issues": len(code_quality_result.issues.issues)
    }
    
    # Close the figure to free memory
    plt.close(fig)
    
    return visualization

# ============================================================================
# SECTION 6: REPOSITORY STRUCTURE VISUALIZATION
# ============================================================================

def visualize_repository_structure(codebase: Codebase, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Visualize the repository structure.
    
    Args:
        codebase: The codebase to analyze
        output_file: Optional output file path
        
    Returns:
        Dictionary containing visualization data
    """
    # Get all files
    files = codebase.get_files()
    
    # Create a tree structure
    tree = {"name": "root", "children": []}
    
    # Add files to the tree
    for file in files:
        # Skip hidden files
        if os.path.basename(file.path).startswith('.'):
            continue
        
        # Split the path into components
        path_components = file.path.split('/')
        
        # Start at the root
        current_node = tree
        
        # Navigate through the path
        for i, component in enumerate(path_components):
            # Check if this is the last component (file)
            is_file = i == len(path_components) - 1
            
            # Find or create the node
            found = False
            for child in current_node.get("children", []):
                if child["name"] == component:
                    current_node = child
                    found = True
                    break
            
            if not found:
                # Create a new node
                new_node = {"name": component}
                if is_file:
                    # This is a file, add file-specific data
                    new_node["type"] = "file"
                    new_node["path"] = file.path
                    new_node["size"] = len(file.content)
                    
                    # Count issues in this file
                    issues = find_issues_in_file(file)
                    new_node["issues"] = {
                        "total": len(issues),
                        "critical": len([i for i in issues if i.severity == IssueSeverity.CRITICAL]),
                        "major": len([i for i in issues if i.severity == IssueSeverity.MAJOR]),
                        "minor": len([i for i in issues if i.severity == IssueSeverity.MINOR]),
                        "info": len([i for i in issues if i.severity == IssueSeverity.INFO])
                    }
                else:
                    # This is a directory, add children
                    new_node["type"] = "directory"
                    new_node["children"] = []
                
                if "children" not in current_node:
                    current_node["children"] = []
                
                current_node["children"].append(new_node)
                current_node = new_node
    
    # Create visualization data
    visualization = {
        "tree": tree,
        "total_files": len(files),
        "total_directories": count_directories(tree),
        "total_size": sum(len(file.content) for file in files)
    }
    
    # Save visualization data to file if output file is specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(visualization, f, indent=2)
    
    return visualization

def count_directories(node: Dict[str, Any]) -> int:
    """
    Count the number of directories in a tree.
    
    Args:
        node: Tree node
        
    Returns:
        Number of directories
    """
    if node.get("type") == "file":
        return 0
    
    count = 1  # Count this directory
    
    for child in node.get("children", []):
        if child.get("type") != "file":
            count += count_directories(child)
    
    return count

def visualize_repository_structure_treemap(codebase: Codebase, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Visualize the repository structure as a treemap.
    
    Args:
        codebase: The codebase to analyze
        output_file: Optional output file path
        
    Returns:
        Dictionary containing visualization data
    """
    # Get all files
    files = codebase.get_files()
    
    # Create a tree structure
    tree = {"name": "root", "children": []}
    
    # Add files to the tree
    for file in files:
        # Skip hidden files
        if os.path.basename(file.path).startswith('.'):
            continue
        
        # Split the path into components
        path_components = file.path.split('/')
        
        # Start at the root
        current_node = tree
        
        # Navigate through the path
        for i, component in enumerate(path_components):
            # Check if this is the last component (file)
            is_file = i == len(path_components) - 1
            
            # Find or create the node
            found = False
            for child in current_node.get("children", []):
                if child["name"] == component:
                    current_node = child
                    found = True
                    break
            
            if not found:
                # Create a new node
                new_node = {"name": component}
                if is_file:
                    # This is a file, add file-specific data
                    new_node["type"] = "file"
                    new_node["path"] = file.path
                    new_node["size"] = len(file.content)
                    
                    # Count issues in this file
                    issues = find_issues_in_file(file)
                    new_node["issues"] = {
                        "total": len(issues),
                        "critical": len([i for i in issues if i.severity == IssueSeverity.CRITICAL]),
                        "major": len([i for i in issues if i.severity == IssueSeverity.MAJOR]),
                        "minor": len([i for i in issues if i.severity == IssueSeverity.MINOR]),
                        "info": len([i for i in issues if i.severity == IssueSeverity.INFO])
                    }
                    
                    # Set value for treemap
                    new_node["value"] = new_node["size"]
                else:
                    # This is a directory, add children
                    new_node["type"] = "directory"
                    new_node["children"] = []
                
                if "children" not in current_node:
                    current_node["children"] = []
                
                current_node["children"].append(new_node)
                current_node = new_node
    
    # Create figure
    fig, ax = create_figure(12, 10)
    
    # Create treemap
    try:
        import squarify
        
        # Flatten the tree for squarify
        flat_tree = []
        
        def flatten_tree(node, path=""):
            if node.get("type") == "file":
                flat_tree.append({
                    "name": path + node["name"],
                    "size": node["size"],
                    "issues": node.get("issues", {}).get("total", 0)
                })
            else:
                for child in node.get("children", []):
                    flatten_tree(child, path + node["name"] + "/")
        
        flatten_tree(tree)
        
        # Sort by size
        flat_tree.sort(key=lambda x: x["size"], reverse=True)
        
        # Take top 50 files
        if len(flat_tree) > 50:
            flat_tree = flat_tree[:50]
        
        # Extract data for squarify
        sizes = [item["size"] for item in flat_tree]
        labels = [item["name"] for item in flat_tree]
        colors = create_color_map([item["issues"] for item in flat_tree], 'YlOrRd')
        
        # Create treemap
        squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.8, ax=ax)
        
        # Set title
        ax.set_title('Repository Structure Treemap (size ~ file size, color ~ issues)')
        ax.set_axis_off()
        
        # Save figure if output file is specified
        if output_file:
            save_figure_to_file(fig, output_file)
        
        # Create visualization data
        visualization = {
            "image": figure_to_base64(fig),
            "total_files": len(files),
            "total_directories": count_directories(tree),
            "total_size": sum(len(file.content) for file in files)
        }
        
        # Close the figure to free memory
        plt.close(fig)
        
        return visualization
    except ImportError:
        # Squarify not available, return empty visualization
        return {
            "error": "Squarify not available for treemap visualization",
            "total_files": len(files),
            "total_directories": count_directories(tree),
            "total_size": sum(len(file.content) for file in files)
        }



# ============================================================================
# SECTION 7: COMPREHENSIVE VISUALIZATION
# ============================================================================

def visualize_analysis_results(analysis_result: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Visualize all analysis results.
    
    Args:
        analysis_result: Analysis result dictionary
        output_dir: Output directory for visualization files
        
    Returns:
        Dictionary containing visualization data
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize dependency graph
    dependency_visualization = visualize_dependency_graph(
        analysis_result["dependency_analysis"],
        os.path.join(output_dir, "dependency_graph.png")
    )
    
    # Visualize call graph
    call_graph_visualization = visualize_call_graph(
        analysis_result["call_graph_analysis"],
        os.path.join(output_dir, "call_graph.png")
    )
    
    # Visualize issues
    issue_visualization = visualize_issues(
        analysis_result["issue_collection"],
        os.path.join(output_dir, "issues.png")
    )
    
    # Visualize issues by file
    issues_by_file_visualization = visualize_issues_by_file(
        analysis_result["issue_collection"],
        os.path.join(output_dir, "issues_by_file.png")
    )
    
    # Visualize code quality
    code_quality_visualization = visualize_code_quality(
        analysis_result["code_quality_result"],
        os.path.join(output_dir, "code_quality.png")
    )
    
    # Create visualization data
    visualization = {
        "dependency_graph": dependency_visualization,
        "call_graph": call_graph_visualization,
        "issues": issue_visualization,
        "issues_by_file": issues_by_file_visualization,
        "code_quality": code_quality_visualization
    }
    
    # Save visualization data to file
    with open(os.path.join(output_dir, "visualization.json"), 'w') as f:
        # Convert to serializable format
        serializable = {
            "dependency_graph": {
                "nodes": dependency_visualization["nodes"],
                "edges": dependency_visualization["edges"],
                "circular_dependencies": dependency_visualization["circular_dependencies"],
                "critical_dependencies": dependency_visualization["critical_dependencies"]
            },
            "call_graph": {
                "nodes": call_graph_visualization["nodes"],
                "edges": call_graph_visualization["edges"],
                "entry_points": call_graph_visualization["entry_points"],
                "leaf_functions": call_graph_visualization["leaf_functions"],
                "max_call_depth": call_graph_visualization["max_call_depth"]
            },
            "issues": {
                "total_issues": issue_visualization["total_issues"],
                "by_severity": issue_visualization["by_severity"],
                "by_category": issue_visualization["by_category"],
                "by_status": issue_visualization["by_status"]
            },
            "code_quality": {
                "maintainability_index": code_quality_visualization["maintainability_index"],
                "cyclomatic_complexity": code_quality_visualization["cyclomatic_complexity"],
                "halstead_volume": code_quality_visualization["halstead_volume"],
                "source_lines_of_code": code_quality_visualization["source_lines_of_code"],
                "comment_density": code_quality_visualization["comment_density"],
                "duplication_percentage": code_quality_visualization["duplication_percentage"],
                "technical_debt_ratio": code_quality_visualization["technical_debt_ratio"],
                "issues": code_quality_visualization["issues"]
            }
        }
        
        json.dump(serializable, f, indent=2)
    
    return visualization

def generate_html_report(analysis_result: Dict[str, Any], output_file: str) -> str:
    """
    Generate an HTML report for the analysis results.
    
    Args:
        analysis_result: Analysis result dictionary
        output_file: Output file path
        
    Returns:
        Path to the generated HTML file
    """
    # Create temporary directory for visualization files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Visualize analysis results
        visualization = visualize_analysis_results(analysis_result, temp_dir)
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Codebase Analysis Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    color: #333;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .section {{
                    margin-bottom: 30px;
                    padding: 20px;
                    background-color: #f9f9f9;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                }}
                .summary-item {{
                    margin-bottom: 10px;
                }}
                .summary-label {{
                    font-weight: bold;
                }}
                .visualization {{
                    margin-top: 20px;
                    text-align: center;
                }}
                .visualization img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                .metrics {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-between;
                }}
                .metric {{
                    flex-basis: 48%;
                    margin-bottom: 15px;
                    padding: 10px;
                    background-color: #fff;
                    border-radius: 5px;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
                }}
                .metric-label {{
                    font-weight: bold;
                    color: #7f8c8d;
                }}
                .metric-value {{
                    font-size: 1.2em;
                    color: #2c3e50;
                }}
                .recommendations {{
                    margin-top: 20px;
                }}
                .recommendation {{
                    margin-bottom: 10px;
                    padding: 10px;
                    background-color: #e8f4f8;
                    border-left: 4px solid #3498db;
                    border-radius: 3px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Codebase Analysis Report</h1>
                
                <div class="section">
                    <h2>Summary</h2>
                    <div class="summary-item">
                        <span class="summary-label">Total Files:</span> {analysis_result["summary"].total_files}
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Total Lines:</span> {analysis_result["summary"].total_lines}
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Total Functions:</span> {analysis_result["summary"].total_functions}
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Total Classes:</span> {analysis_result["summary"].total_classes}
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Total Issues:</span> {analysis_result["summary"].total_issues}
                    </div>
                </div>
                
                <div class="section">
                    <h2>Code Quality</h2>
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-label">Maintainability Index</div>
                            <div class="metric-value">{analysis_result["code_quality_result"].maintainability_index:.2f}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Cyclomatic Complexity</div>
                            <div class="metric-value">{analysis_result["code_quality_result"].cyclomatic_complexity:.2f}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Duplication Percentage</div>
                            <div class="metric-value">{analysis_result["code_quality_result"].duplication_percentage:.2f}%</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Technical Debt Ratio</div>
                            <div class="metric-value">{analysis_result["code_quality_result"].technical_debt_ratio:.2f}%</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Comment Density</div>
                            <div class="metric-value">{analysis_result["code_quality_result"].comment_density:.2f}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Source Lines of Code</div>
                            <div class="metric-value">{analysis_result["code_quality_result"].source_lines_of_code}</div>
                        </div>
                    </div>
                    <div class="visualization">
                        <img src="data:image/png;base64,{visualization["code_quality"]["image"]}" alt="Code Quality Visualization">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Issues</h2>
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-label">Critical Issues</div>
                            <div class="metric-value">{visualization["issues"]["by_severity"].get("critical", 0)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Major Issues</div>
                            <div class="metric-value">{visualization["issues"]["by_severity"].get("major", 0)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Minor Issues</div>
                            <div class="metric-value">{visualization["issues"]["by_severity"].get("minor", 0)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Info Issues</div>
                            <div class="metric-value">{visualization["issues"]["by_severity"].get("info", 0)}</div>
                        </div>
                    </div>
                    <div class="visualization">
                        <img src="data:image/png;base64,{visualization["issues"]["image"]}" alt="Issues Visualization">
                    </div>
                    <div class="visualization">
                        <img src="data:image/png;base64,{visualization["issues_by_file"]["image"]}" alt="Issues by File Visualization">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Dependency Analysis</h2>
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-label">Total Dependencies</div>
                            <div class="metric-value">{analysis_result["dependency_analysis"].total_dependencies}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Circular Dependencies</div>
                            <div class="metric-value">{len(analysis_result["dependency_analysis"].circular_dependencies)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Dependency Depth</div>
                            <div class="metric-value">{analysis_result["dependency_analysis"].dependency_depth}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Critical Dependencies</div>
                            <div class="metric-value">{len(analysis_result["dependency_analysis"].critical_dependencies)}</div>
                        </div>
                    </div>
                    <div class="visualization">
                        <img src="data:image/png;base64,{visualization["dependency_graph"]["image"]}" alt="Dependency Graph Visualization">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Call Graph Analysis</h2>
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-label">Total Functions</div>
                            <div class="metric-value">{analysis_result["call_graph_analysis"].total_functions}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Entry Points</div>
                            <div class="metric-value">{len(analysis_result["call_graph_analysis"].entry_points)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Leaf Functions</div>
                            <div class="metric-value">{len(analysis_result["call_graph_analysis"].leaf_functions)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Max Call Depth</div>
                            <div class="metric-value">{analysis_result["call_graph_analysis"].max_call_depth}</div>
                        </div>
                    </div>
                    <div class="visualization">
                        <img src="data:image/png;base64,{visualization["call_graph"]["image"]}" alt="Call Graph Visualization">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Recommendations</h2>
                    <div class="recommendations">
                        {"".join(f'<div class="recommendation">{recommendation}</div>' for recommendation in analysis_result["recommendations"])}
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write HTML content to file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        return output_file

# ============================================================================
# SECTION 8: MAIN VISUALIZATION FUNCTION
# ============================================================================

def visualize_codebase(codebase: Codebase, output_dir: str) -> Dict[str, Any]:
    """
    Visualize a codebase.
    
    This function analyzes a codebase and generates visualizations for the analysis results.
    
    Args:
        codebase: The codebase to analyze
        output_dir: Output directory for visualization files
        
    Returns:
        Dictionary containing visualization data
    """
    # Import analysis function
    from analysis import analyze_codebase
    
    # Analyze codebase
    analysis_result = analyze_codebase(codebase)
    
    # Visualize analysis results
    visualization = visualize_analysis_results(analysis_result, output_dir)
    
    # Generate HTML report
    html_report = generate_html_report(analysis_result, os.path.join(output_dir, "report.html"))
    
    # Add HTML report to visualization data
    visualization["html_report"] = html_report
    
    return visualization



