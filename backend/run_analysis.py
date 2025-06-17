#!/usr/bin/env python3
"""
Script to run analysis on a GitHub repository.
"""

import os
import json
import tempfile
import subprocess
import argparse
from pathlib import Path

# Import analysis and visualization modules
from analysis import analyze_codebase
from visualize import visualize_codebase, generate_html_report

# Import from Codegen SDK
from codegen.sdk.core.codebase import Codebase

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
        
        # Use the correct method to create a Codebase object
        # Since from_directory is not available, we'll use the constructor directly
        codebase = Codebase(temp_dir)
        
        # Analyze codebase
        print("Analyzing codebase...")
        analysis_result = analyze_codebase(codebase)
        
        # Save analysis result
        print("Saving analysis result...")
        with open(os.path.join(args.output_dir, "analysis.json"), "w") as f:
            # Convert to serializable format
            serializable_result = {
                "summary": {
                    "total_files": analysis_result["summary"].total_files,
                    "total_lines": analysis_result["summary"].total_lines,
                    "total_functions": analysis_result["summary"].total_functions,
                    "total_classes": analysis_result["summary"].total_classes,
                    "total_issues": analysis_result["summary"].total_issues,
                    "issue_counts": analysis_result["summary"].issue_counts,
                    "metrics": analysis_result["summary"].metrics,
                    "recommendations": analysis_result["summary"].recommendations
                },
                "dependency_analysis": {
                    "total_dependencies": analysis_result["dependency_analysis"].total_dependencies,
                    "circular_dependencies": analysis_result["dependency_analysis"].circular_dependencies,
                    "dependency_depth": analysis_result["dependency_analysis"].dependency_depth,
                    "external_dependencies": analysis_result["dependency_analysis"].external_dependencies,
                    "internal_dependencies": analysis_result["dependency_analysis"].internal_dependencies,
                    "critical_dependencies": analysis_result["dependency_analysis"].critical_dependencies,
                    "unused_dependencies": analysis_result["dependency_analysis"].unused_dependencies
                },
                "call_graph_analysis": {
                    "total_functions": analysis_result["call_graph_analysis"].total_functions,
                    "entry_points": analysis_result["call_graph_analysis"].entry_points,
                    "leaf_functions": analysis_result["call_graph_analysis"].leaf_functions,
                    "max_call_depth": analysis_result["call_graph_analysis"].max_call_depth
                },
                "code_quality_result": {
                    "maintainability_index": analysis_result["code_quality_result"].maintainability_index,
                    "cyclomatic_complexity": analysis_result["code_quality_result"].cyclomatic_complexity,
                    "halstead_volume": analysis_result["code_quality_result"].halstead_volume,
                    "source_lines_of_code": analysis_result["code_quality_result"].source_lines_of_code,
                    "comment_density": analysis_result["code_quality_result"].comment_density,
                    "duplication_percentage": analysis_result["code_quality_result"].duplication_percentage,
                    "technical_debt_ratio": analysis_result["code_quality_result"].technical_debt_ratio
                },
                "issue_collection": {
                    "total_issues": len(analysis_result["issue_collection"].issues),
                    "by_severity": analysis_result["issue_collection"].count_by_severity(),
                    "by_category": analysis_result["issue_collection"].count_by_category(),
                    "by_status": analysis_result["issue_collection"].count_by_status()
                },
                "recommendations": analysis_result["recommendations"]
            }
            
            json.dump(serializable_result, f, indent=2)
        
        # Generate visualizations
        print("Generating visualizations...")
        visualization = visualize_codebase(codebase, args.output_dir)
        
        # Generate HTML report
        print("Generating HTML report...")
        html_report = generate_html_report(analysis_result, os.path.join(args.output_dir, "report.html"))
        
        print(f"Analysis completed successfully. Results saved to {args.output_dir}")
        print(f"HTML report: {html_report}")

if __name__ == "__main__":
    main()

