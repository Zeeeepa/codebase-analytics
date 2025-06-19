#!/usr/bin/env python3
"""
Direct analysis script for the codebase-analytics repository.
"""

import os
import sys
import json
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import analyze_codebase
from visualize import visualize_codebase, generate_html_report

def run_direct_analysis():
    """Run analysis directly on the current codebase."""
    
    # Get the repository root (parent of backend directory)
    repo_root = Path(__file__).parent.parent
    output_dir = repo_root / "analysis_output"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    print(f"Analyzing codebase at: {repo_root}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Run analysis
        print("Running codebase analysis...")
        results = analyze_codebase(str(repo_root))
        
        # Save analysis results
        analysis_file = output_dir / "analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Analysis results saved to: {analysis_file}")
        
        # Generate visualizations
        print("Generating visualizations...")
        visualizations = visualize_codebase(str(repo_root), str(output_dir))
        print(f"Visualizations generated: {list(visualizations.keys())}")
        
        # Generate HTML report
        print("Generating HTML report...")
        report_path = generate_html_report(results, visualizations, str(output_dir))
        print(f"HTML report generated: {report_path}")
        
        # Print summary
        print("\n=== ANALYSIS SUMMARY ===")
        if isinstance(results, dict):
            for key, value in results.items():
                if key not in ["detailed_analysis", "files", "dependencies"]:  # Skip large data
                    print(f"{key}: {value}")
        
        print(f"\nFull results available in: {analysis_file}")
        print(f"HTML report available in: {report_path}")
        
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_direct_analysis()
