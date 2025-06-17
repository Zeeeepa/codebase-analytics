#!/usr/bin/env python3
"""
Visualization module for the Codebase Analytics tool.
"""

import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import squarify
from typing import Dict, List, Any, Optional

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
        "treemap": treemap_file,
    }

def visualize_dependency_graph(files: List[str], output_file: str) -> str:
    """
    Visualize the dependency graph.
    
    Args:
        files: List of file paths
        output_file: Output file path
        
    Returns:
        Path to the visualization file
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for file in files:
        G.add_node(file)
    
    # Add edges (simplified)
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # Look for imports
                if file.endswith('.py'):
                    # Python imports
                    for line in content.split('\n'):
                        if line.startswith('import ') or line.startswith('from '):
                            for other_file in files:
                                module_name = os.path.basename(other_file).split('.')[0]
                                if module_name in line:
                                    G.add_edge(file, other_file)
                elif file.endswith(('.js', '.ts', '.jsx', '.tsx')):
                    # JavaScript/TypeScript imports
                    for line in content.split('\n'):
                        if 'import ' in line or 'require(' in line:
                            for other_file in files:
                                module_name = os.path.basename(other_file).split('.')[0]
                                if module_name in line:
                                    G.add_edge(file, other_file)
        except Exception as e:
            print(f"Error analyzing dependencies in {file}: {str(e)}")
    
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

def visualize_complexity(files: List[str], output_file: str) -> str:
    """
    Visualize code complexity.
    
    Args:
        files: List of file paths
        output_file: Output file path
        
    Returns:
        Path to the visualization file
    """
    # Calculate complexity for each file
    complexity_results = {}
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # Calculate cyclomatic complexity (simplified)
                if_count = content.count('if ')
                for_count = content.count('for ')
                while_count = content.count('while ')
                try_count = content.count('try ')
                
                complexity = 1 + if_count + for_count + while_count + try_count
                
                # Store results
                complexity_results[file_path] = complexity
        except Exception as e:
            print(f"Error analyzing {file_path}: {str(e)}")
    
    # Sort files by complexity
    sorted_files = sorted(complexity_results.items(), key=lambda x: x[1], reverse=True)
    top_files = sorted_files[:20]  # Top 20 most complex files
    
    # Extract data for plotting
    file_names = [os.path.basename(file) for file, _ in top_files]
    complexities = [complexity for _, complexity in top_files]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create bar chart
    plt.barh(file_names, complexities, color='coral')
    plt.xlabel('Cyclomatic Complexity')
    plt.ylabel('File')
    plt.title('Top 20 Most Complex Files')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    return output_file

def visualize_file_type_distribution(files: List[str], output_file: str) -> str:
    """
    Visualize file type distribution.
    
    Args:
        files: List of file paths
        output_file: Output file path
        
    Returns:
        Path to the visualization file
    """
    # Count file types
    file_types = Counter()
    
    for file in files:
        file_ext = os.path.splitext(file)[1]
        file_types[file_ext] += 1
    
    # Extract data for plotting
    labels = list(file_types.keys())
    sizes = list(file_types.values())
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create pie chart
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, shadow=True)
    plt.axis('equal')
    plt.title('File Type Distribution')
    
    # Save figure
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    return output_file

def visualize_treemap(files: List[str], output_file: str) -> str:
    """
    Visualize file sizes as a treemap.
    
    Args:
        files: List of file paths
        output_file: Output file path
        
    Returns:
        Path to the visualization file
    """
    # Calculate file sizes
    file_sizes = {}
    
    for file in files:
        try:
            file_size = os.path.getsize(file)
            file_sizes[os.path.basename(file)] = file_size
        except Exception as e:
            print(f"Error getting size of {file}: {str(e)}")
    
    # Sort files by size
    sorted_files = sorted(file_sizes.items(), key=lambda x: x[1], reverse=True)
    top_files = sorted_files[:50]  # Top 50 largest files
    
    # Extract data for plotting
    file_names = [file for file, _ in top_files]
    sizes = [size for _, size in top_files]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create treemap
    squarify.plot(sizes=sizes, label=file_names, alpha=0.8)
    plt.axis('off')
    plt.title('File Size Distribution (Top 50 Largest Files)')
    
    # Save figure
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    
    return output_file

def generate_html_report(analysis_result: Dict[str, Any], output_file: str) -> str:
    """
    Generate HTML report.
    
    Args:
        analysis_result: Analysis result
        output_file: Output file path
        
    Returns:
        Path to the HTML report
    """
    # Extract data from analysis result
    summary = analysis_result.get("summary", {})
    dependency_analysis = analysis_result.get("dependency_analysis", {})
    code_quality_result = analysis_result.get("code_quality_result", {})
    issue_collection = analysis_result.get("issue_collection", {})
    recommendations = analysis_result.get("recommendations", [])
    
    # Generate HTML content
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
                    <span class="summary-label">Total Files:</span> {summary.get("total_files", 0)}
                </div>
                <div class="summary-item">
                    <span class="summary-label">Total Lines:</span> {summary.get("total_lines", 0)}
                </div>
                <div class="summary-item">
                    <span class="summary-label">Total Functions:</span> {summary.get("total_functions", 0)}
                </div>
                <div class="summary-item">
                    <span class="summary-label">Total Classes:</span> {summary.get("total_classes", 0)}
                </div>
                <div class="summary-item">
                    <span class="summary-label">Total Issues:</span> {summary.get("total_issues", 0)}
                </div>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <div class="visualization">
                    <h3>Dependency Graph</h3>
                    <img src="visualizations/dependency_graph.png" alt="Dependency Graph">
                </div>
                <div class="visualization">
                    <h3>Code Complexity</h3>
                    <img src="visualizations/complexity.png" alt="Code Complexity">
                </div>
                <div class="visualization">
                    <h3>File Type Distribution</h3>
                    <img src="visualizations/file_type_distribution.png" alt="File Type Distribution">
                </div>
                <div class="visualization">
                    <h3>File Size Distribution</h3>
                    <img src="visualizations/treemap.png" alt="File Size Distribution">
                </div>
            </div>
            
            <div class="section">
                <h2>Dependency Analysis</h2>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-label">Total Dependencies</div>
                        <div class="metric-value">{dependency_analysis.get("total_dependencies", 0)}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Circular Dependencies</div>
                        <div class="metric-value">{len(dependency_analysis.get("circular_dependencies", []))}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Dependency Depth</div>
                        <div class="metric-value">{dependency_analysis.get("dependency_depth", 0)}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Critical Dependencies</div>
                        <div class="metric-value">{len(dependency_analysis.get("critical_dependencies", []))}</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Code Quality</h2>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-label">Maintainability Index</div>
                        <div class="metric-value">{code_quality_result.get("maintainability_index", 0)}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Cyclomatic Complexity</div>
                        <div class="metric-value">{code_quality_result.get("cyclomatic_complexity", 0)}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Source Lines of Code</div>
                        <div class="metric-value">{code_quality_result.get("source_lines_of_code", 0)}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Comment Density</div>
                        <div class="metric-value">{code_quality_result.get("comment_density", 0)}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Duplication Percentage</div>
                        <div class="metric-value">{code_quality_result.get("duplication_percentage", 0)}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Technical Debt Ratio</div>
                        <div class="metric-value">{code_quality_result.get("technical_debt_ratio", 0)}</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <div class="recommendations">
                    {"".join(f'<div class="recommendation">{recommendation}</div>' for recommendation in recommendations)}
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML content to file
    with open(output_file, "w") as f:
        f.write(html_content)
    
    return output_file

if __name__ == "__main__":
    # Test visualization functions
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python visualize.py <codebase_dir> <output_dir>")
        sys.exit(1)
    
    codebase_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    visualizations = visualize_codebase(codebase_dir, output_dir)
    
    print(f"Visualizations generated:")
    for name, path in visualizations.items():
        print(f"  {name}: {path}")

