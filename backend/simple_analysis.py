#!/usr/bin/env python3
"""
Simplified script to run basic analysis on a GitHub repository.
"""

import os
import json
import tempfile
import subprocess
import argparse
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

# Import from Codegen SDK
from codegen.sdk.core.codebase import Codebase

def analyze_codebase_simple(codebase):
    """
    Perform a simplified analysis of a codebase.
    
    Args:
        codebase: The codebase to analyze
        
    Returns:
        Dictionary containing the analysis results
    """
    # Get all files
    files = []
    for root, dirs, filenames in os.walk(codebase.directory):
        for filename in filenames:
            if filename.endswith(('.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp')):
                file_path = os.path.join(root, filename)
                files.append(file_path)
    
    # Count total files and lines
    total_files = len(files)
    total_lines = 0
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            total_lines += len(f.readlines())
    
    # Count functions and classes (simplified)
    total_functions = 0
    total_classes = 0
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # Very simple heuristic for counting functions and classes
            if file_path.endswith('.py'):
                total_functions += content.count('def ')
                total_classes += content.count('class ')
            elif file_path.endswith(('.js', '.ts', '.jsx', '.tsx')):
                total_functions += content.count('function ')
                total_classes += content.count('class ')
            elif file_path.endswith(('.java', '.c', '.cpp')):
                # This is a very rough approximation
                total_functions += content.count('{')
                total_classes += content.count('class ')
    
    # Build a simple dependency graph
    dependency_graph = {}
    for file_path in files:
        dependency_graph[file_path] = []
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # Look for imports
            if file_path.endswith('.py'):
                # Python imports
                for line in content.split('\n'):
                    if line.startswith('import ') or line.startswith('from '):
                        for other_file in files:
                            module_name = os.path.basename(other_file).split('.')[0]
                            if module_name in line:
                                dependency_graph[file_path].append(other_file)
            elif file_path.endswith(('.js', '.ts', '.jsx', '.tsx')):
                # JavaScript/TypeScript imports
                for line in content.split('\n'):
                    if 'import ' in line or 'require(' in line:
                        for other_file in files:
                            module_name = os.path.basename(other_file).split('.')[0]
                            if module_name in line:
                                dependency_graph[file_path].append(other_file)
    
    # Count total dependencies
    total_dependencies = sum(len(deps) for deps in dependency_graph.values())
    
    # Find circular dependencies
    circular_dependencies = []
    G = nx.DiGraph()
    for file, deps in dependency_graph.items():
        G.add_node(file)
        for dep in deps:
            G.add_edge(file, dep)
    
    try:
        cycles = list(nx.simple_cycles(G))
        circular_dependencies = cycles
    except nx.NetworkXNoCycle:
        # No cycles found
        pass
    
    # Calculate dependency depth
    dependency_depth = 0
    try:
        # Convert to DAG by removing cycles
        DAG = nx.DiGraph(G)
        cycles = list(nx.simple_cycles(G))
        for cycle in cycles:
            if len(cycle) > 1:
                DAG.remove_edge(cycle[0], cycle[1])
        
        # Find the longest path in the DAG
        longest_path_length = 0
        for node in DAG.nodes():
            paths = nx.single_source_shortest_path_length(DAG, node)
            if paths:
                longest_path_length = max(longest_path_length, max(paths.values()))
        
        dependency_depth = longest_path_length
    except (nx.NetworkXError, ValueError):
        # Error calculating longest path
        pass
    
    # Identify critical dependencies
    dependency_counts = Counter()
    for file, deps in dependency_graph.items():
        for dep in deps:
            dependency_counts[dep] += 1
    
    threshold = max(1, len(dependency_graph) // 10)  # At least 10% of files depend on it
    critical_dependencies = [dep for dep, count in dependency_counts.items() if count >= threshold]
    
    # Create analysis result
    analysis_result = {
        "summary": {
            "total_files": total_files,
            "total_lines": total_lines,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "total_issues": 0,  # Simplified version doesn't detect issues
            "issue_counts": {},
            "metrics": {},
            "recommendations": []
        },
        "dependency_analysis": {
            "total_dependencies": total_dependencies,
            "circular_dependencies": circular_dependencies,
            "dependency_depth": dependency_depth,
            "external_dependencies": [],  # Simplified version doesn't detect external dependencies
            "internal_dependencies": [],  # Simplified version doesn't detect internal dependencies
            "critical_dependencies": critical_dependencies,
            "unused_dependencies": []  # Simplified version doesn't detect unused dependencies
        },
        "code_quality_result": {
            "maintainability_index": 0,  # Simplified version doesn't calculate maintainability index
            "cyclomatic_complexity": 0,  # Simplified version doesn't calculate cyclomatic complexity
            "halstead_volume": 0,  # Simplified version doesn't calculate Halstead volume
            "source_lines_of_code": total_lines,
            "comment_density": 0,  # Simplified version doesn't calculate comment density
            "duplication_percentage": 0,  # Simplified version doesn't calculate duplication percentage
            "technical_debt_ratio": 0  # Simplified version doesn't calculate technical debt ratio
        }
    }
    
    return analysis_result

def visualize_dependency_graph(dependency_graph, output_file):
    """
    Visualize the dependency graph.
    
    Args:
        dependency_graph: Dictionary mapping file paths to lists of dependencies
        output_file: Output file path
        
    Returns:
        Path to the visualization file
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    for file, deps in dependency_graph.items():
        G.add_node(file)
        for dep in deps:
            G.add_edge(file, dep)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Calculate node positions using a layout algorithm
    pos = nx.spring_layout(G)
    
    # Calculate node sizes based on degree
    node_sizes = [10 + (G.in_degree(node) + G.out_degree(node)) * 5 for node in G.nodes()]
    
    # Draw the graph
    nx.draw_networkx(
        G, pos,
        with_labels=False,
        node_size=node_sizes,
        node_color='skyblue',
        alpha=0.8,
        arrows=True,
        arrowsize=10,
        width=0.5
    )
    
    # Draw node labels for important nodes
    important_nodes = [node for node in G.nodes() if G.in_degree(node) + G.out_degree(node) > 5]
    important_labels = {node: os.path.basename(node) for node in important_nodes}
    nx.draw_networkx_labels(G, pos, labels=important_labels, font_size=8)
    
    # Set title and labels
    plt.title('Dependency Graph')
    plt.axis('off')
    
    # Save figure
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    return output_file

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run analysis on a GitHub repository")
    parser.add_argument("repo_url", help="URL of the GitHub repository")
    parser.add_argument("--branch", help="Branch to analyze", default=None)
    parser.add_argument("--output-dir", help="Output directory", default="output")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Analyzing repository: {args.repo_url}")
    if args.branch:
        print(f"Branch: {args.branch}")
    
    # Clone the repository
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Cloning repository to {temp_dir}...")
        
        # Clone the repository
        clone_cmd = ["git", "clone", args.repo_url, temp_dir]
        if args.branch:
            clone_cmd.extend(["--branch", args.branch])
        
        subprocess.run(clone_cmd, check=True)
        
        print("Repository cloned successfully")
        
        # Create codebase object
        print("Creating codebase object...")
        codebase = Codebase(temp_dir)
        
        # Analyze codebase
        print("Analyzing codebase...")
        analysis_result = analyze_codebase_simple(codebase)
        
        # Save analysis result
        print("Saving analysis result...")
        with open(os.path.join(args.output_dir, "analysis.json"), "w") as f:
            json.dump(analysis_result, f, indent=2)
        
        # Generate dependency graph visualization
        print("Generating dependency graph visualization...")
        dependency_graph = {}
        for file_path in os.listdir(temp_dir):
            if os.path.isfile(os.path.join(temp_dir, file_path)) and file_path.endswith(('.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp')):
                dependency_graph[file_path] = []
        
        visualization_file = visualize_dependency_graph(dependency_graph, os.path.join(args.output_dir, "dependency_graph.png"))
        
        print(f"Analysis completed successfully. Results saved to {args.output_dir}")
        print(f"Dependency graph visualization: {visualization_file}")

if __name__ == "__main__":
    main()

