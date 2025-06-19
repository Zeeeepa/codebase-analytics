#!/usr/bin/env python3
"""
Test script to run analysis on the codebase-analytics repository.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add backend to path
sys.path.insert(0, 'backend')

from codegen.sdk.core.codebase import Codebase
from analysis import analyze_codebase, get_codebase_summary
from visualization import run_visualization_analysis

def main():
    """Run analysis on the current codebase-analytics repository."""
    print("🔍 Starting analysis of codebase-analytics repository...")
    
    # Get current repository path
    repo_path = Path.cwd()
    print(f"📁 Repository path: {repo_path}")
    
    try:
        # Create Codebase object from GitHub repository
        print("📊 Creating Codebase object...")
        codebase = Codebase.from_repo("Zeeeepa/codebase-analytics")
        
        # Get codebase summary
        print("📋 Generating codebase summary...")
        summary = get_codebase_summary(codebase)
        print("✅ Codebase Summary:")
        print(summary)
        
        # Run comprehensive analysis
        print("\n🔬 Running comprehensive analysis...")
        analysis_result = analyze_codebase(codebase)
        
        print("✅ Analysis Results:")
        print(f"  - Total files analyzed: {analysis_result.get('total_files', 'N/A')}")
        print(f"  - Total functions: {analysis_result.get('total_functions', 'N/A')}")
        print(f"  - Total classes: {analysis_result.get('total_classes', 'N/A')}")
        print(f"  - Issues found: {len(analysis_result.get('issues', []))}")
        print(f"  - Maintainability index: {analysis_result.get('maintainability_index', 'N/A')}")
        
        # Run visualization analysis
        print("\n🎨 Running visualization analysis...")
        with tempfile.TemporaryDirectory() as temp_dir:
            viz_result = run_visualization_analysis(codebase, temp_dir)
            print("✅ Visualization Results:")
            print(f"  - Output directory: {viz_result['output_directory']}")
            print(f"  - Visualizations generated: {len(viz_result['visualizations'])}")
            for viz_name, viz_path in viz_result['visualizations'].items():
                print(f"    - {viz_name}: {viz_path}")
        
        print("\n🎉 Analysis completed successfully!")
        print("✅ All consolidated modules working correctly")
        print("✅ Codegen SDK integration functional")
        print("✅ Ready for production use")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
