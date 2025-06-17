#!/usr/bin/env python3
"""
Script to run full comprehensive analysis on a GitHub repository.
This script uses the codegen SDK to analyze a codebase and generate a comprehensive report.
"""

import os
import sys
import json
import tempfile
import subprocess
import argparse
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import time

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run full comprehensive analysis on a GitHub repository")
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
        
        # Run comprehensive analysis
        print("Running comprehensive analysis...")
        
        # Start analysis
        print("Starting analysis...")
        start_time = time.time()
        
        # Get all files in the repository
        all_files = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith(('.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp')):
                    file_path = os.path.join(root, file)
                    all_files.append(file_path)
        
        print(f"Found {len(all_files)} files to analyze")
        
        # Analyze code complexity
        print("Analyzing code complexity...")
        complexity_results = {}
        
        for file_path in all_files:
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
                    complexity_results[file_path] = {
                        'cyclomatic_complexity': complexity,
                        'lines_of_code': len(content.split('\n')),
                        'if_count': if_count,
                        'for_count': for_count,
                        'while_count': while_count,
                        'try_count': try_count
                    }
            except Exception as e:
                print(f"Error analyzing {file_path}: {str(e)}")
        
        # Analyze dependencies
        print("Analyzing dependencies...")
        dependency_graph = {}
        
        for file_path in all_files:
            dependency_graph[file_path] = []
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                    # Look for imports
                    if file_path.endswith('.py'):
                        # Python imports
                        for line in content.split('\n'):
                            if line.startswith('import ') or line.startswith('from '):
                                for other_file in all_files:
                                    module_name = os.path.basename(other_file).split('.')[0]
                                    if module_name in line:
                                        dependency_graph[file_path].append(other_file)
                    elif file_path.endswith(('.js', '.ts', '.jsx', '.tsx')):
                        # JavaScript/TypeScript imports
                        for line in content.split('\n'):
                            if 'import ' in line or 'require(' in line:
                                for other_file in all_files:
                                    module_name = os.path.basename(other_file).split('.')[0]
                                    if module_name in line:
                                        dependency_graph[file_path].append(other_file)
            except Exception as e:
                print(f"Error analyzing dependencies in {file_path}: {str(e)}")
        
        # Find circular dependencies
        print("Finding circular dependencies...")
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
        
        # Detect code issues
        print("Detecting code issues...")
        issues = []
        
        for file_path in all_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                    # Check for TODO comments
                    for i, line in enumerate(lines):
                        if 'TODO' in line or 'FIXME' in line:
                            issues.append({
                                'file': file_path,
                                'line': i + 1,
                                'message': f"TODO/FIXME comment: {line.strip()}",
                                'severity': 'info'
                            })
                    
                    # Check for long lines
                    for i, line in enumerate(lines):
                        if len(line) > 100:
                            issues.append({
                                'file': file_path,
                                'line': i + 1,
                                'message': f"Line too long ({len(line)} characters)",
                                'severity': 'minor'
                            })
                    
                    # Check for empty catch blocks
                    if file_path.endswith('.py'):
                        for i, line in enumerate(lines):
                            if 'except' in line and i + 1 < len(lines) and 'pass' in lines[i + 1]:
                                issues.append({
                                    'file': file_path,
                                    'line': i + 1,
                                    'message': "Empty except block",
                                    'severity': 'major'
                                })
                    
                    # Check for unused imports (simplified)
                    if file_path.endswith('.py'):
                        for i, line in enumerate(lines):
                            if line.startswith('import '):
                                module = line.split(' ')[1].strip()
                                if module not in ''.join(lines[i+1:]):
                                    issues.append({
                                        'file': file_path,
                                        'line': i + 1,
                                        'message': f"Potentially unused import: {module}",
                                        'severity': 'minor'
                                    })
            except Exception as e:
                print(f"Error detecting issues in {file_path}: {str(e)}")
        
        # Calculate metrics
        print("Calculating metrics...")
        total_files = len(all_files)
        total_lines = sum(result['lines_of_code'] for result in complexity_results.values())
        avg_complexity = sum(result['cyclomatic_complexity'] for result in complexity_results.values()) / total_files if total_files > 0 else 0
        total_issues = len(issues)
        
        # Generate recommendations
        print("Generating recommendations...")
        recommendations = []
        
        if circular_dependencies:
            recommendations.append(f"Resolve {len(circular_dependencies)} circular dependencies to improve maintainability")
        
        if avg_complexity > 10:
            recommendations.append(f"Reduce average cyclomatic complexity ({avg_complexity:.2f}) by refactoring complex functions")
        
        critical_issues = [issue for issue in issues if issue['severity'] == 'critical']
        if critical_issues:
            recommendations.append(f"Address {len(critical_issues)} critical issues")
        
        # Create visualization directory
        vis_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Generate dependency graph visualization
        print("Generating dependency graph visualization...")
        plt.figure(figsize=(12, 10))
        
        # Calculate node positions
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
        dependency_graph_file = os.path.join(vis_dir, "dependency_graph.png")
        plt.savefig(dependency_graph_file, bbox_inches='tight')
        plt.close()
        
        # Generate complexity visualization
        print("Generating complexity visualization...")
        plt.figure(figsize=(12, 8))
        
        # Sort files by complexity
        sorted_files = sorted(complexity_results.items(), key=lambda x: x[1]['cyclomatic_complexity'], reverse=True)
        top_files = sorted_files[:20]  # Top 20 most complex files
        
        # Extract data for plotting
        file_names = [os.path.basename(file) for file, _ in top_files]
        complexities = [result['cyclomatic_complexity'] for _, result in top_files]
        
        # Create bar chart
        plt.barh(file_names, complexities, color='coral')
        plt.xlabel('Cyclomatic Complexity')
        plt.ylabel('File')
        plt.title('Top 20 Most Complex Files')
        plt.tight_layout()
        
        # Save figure
        complexity_file = os.path.join(vis_dir, "complexity.png")
        plt.savefig(complexity_file, bbox_inches='tight')
        plt.close()
        
        # Generate issues visualization
        print("Generating issues visualization...")
        plt.figure(figsize=(10, 6))
        
        # Count issues by severity
        severity_counts = Counter(issue['severity'] for issue in issues)
        severities = list(severity_counts.keys())
        counts = list(severity_counts.values())
        
        # Create bar chart
        plt.bar(severities, counts, color=['red', 'orange', 'yellow', 'blue'])
        plt.xlabel('Severity')
        plt.ylabel('Count')
        plt.title('Issues by Severity')
        
        # Save figure
        issues_file = os.path.join(vis_dir, "issues.png")
        plt.savefig(issues_file, bbox_inches='tight')
        plt.close()
        
        # Generate HTML report
        print("Generating HTML report...")
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Comprehensive Codebase Analysis Report</title>
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
                .issues-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }}
                .issues-table th, .issues-table td {{
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                .issues-table th {{
                    background-color: #f2f2f2;
                }}
                .severity-critical {{
                    color: #e74c3c;
                    font-weight: bold;
                }}
                .severity-major {{
                    color: #e67e22;
                    font-weight: bold;
                }}
                .severity-minor {{
                    color: #f1c40f;
                }}
                .severity-info {{
                    color: #3498db;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Comprehensive Codebase Analysis Report</h1>
                
                <div class="section">
                    <h2>Summary</h2>
                    <div class="summary-item">
                        <span class="summary-label">Total Files:</span> {total_files}
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Total Lines:</span> {total_lines}
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Average Cyclomatic Complexity:</span> {avg_complexity:.2f}
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Total Issues:</span> {total_issues}
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Circular Dependencies:</span> {len(circular_dependencies)}
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
                        <h3>Issues by Severity</h3>
                        <img src="visualizations/issues.png" alt="Issues by Severity">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Recommendations</h2>
                    <div class="recommendations">
                        {"".join(f'<div class="recommendation">{recommendation}</div>' for recommendation in recommendations)}
                    </div>
                </div>
                
                <div class="section">
                    <h2>Top Issues</h2>
                    <table class="issues-table">
                        <tr>
                            <th>File</th>
                            <th>Line</th>
                            <th>Severity</th>
                            <th>Message</th>
                        </tr>
                        {"".join(f'<tr><td>{os.path.basename(issue["file"])}</td><td>{issue["line"]}</td><td class="severity-{issue["severity"]}">{issue["severity"].upper()}</td><td>{issue["message"]}</td></tr>' for issue in issues[:50])}
                    </table>
                </div>
                
                <div class="section">
                    <h2>Circular Dependencies</h2>
                    <ul>
                        {"".join(f'<li>{" -> ".join(os.path.basename(file) for file in cycle)}</li>' for cycle in circular_dependencies[:10])}
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        html_report_file = os.path.join(args.output_dir, "report.html")
        with open(html_report_file, "w") as f:
            f.write(html_content)
        
        # Save analysis results as JSON
        print("Saving analysis results...")
        analysis_result = {
            "summary": {
                "total_files": total_files,
                "total_lines": total_lines,
                "avg_complexity": avg_complexity,
                "total_issues": total_issues,
                "circular_dependencies": len(circular_dependencies)
            },
            "complexity_results": complexity_results,
            "issues": issues,
            "circular_dependencies": [list(cycle) for cycle in circular_dependencies],
            "recommendations": recommendations
        }
        
        with open(os.path.join(args.output_dir, "analysis.json"), "w") as f:
            json.dump(analysis_result, f, indent=2)
        
        # Calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"Analysis completed successfully in {execution_time:.2f} seconds")
        print(f"Results saved to {args.output_dir}")
        print(f"HTML report: {html_report_file}")

if __name__ == "__main__":
    main()
