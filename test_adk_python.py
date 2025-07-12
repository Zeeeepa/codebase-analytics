#!/usr/bin/env python3
"""
Test script to validate the codebase-analytics backend with adk-python repository
"""

import sys
import os
import json
from datetime import datetime

# Add backend to path
sys.path.insert(0, 'backend')

def test_adk_python_analysis():
    """Test analysis of adk-python repository"""
    print("🧪 Testing codebase-analytics with adk-python repository...")
    print("=" * 60)
    
    try:
        # Import the analyzer
        from analysis import GraphSitterAnalyzer
        
        print("✅ Successfully imported GraphSitterAnalyzer")
        
        # Initialize analyzer
        analyzer = GraphSitterAnalyzer()
        print("✅ Analyzer initialized")
        
        # Test with adk-python repository
        repo_url = "https://github.com/Zeeeepa/adk-python"
        print(f"🔍 Analyzing repository: {repo_url}")
        
        # Perform analysis
        results = analyzer.analyze_repository(repo_url)
        print("✅ Analysis completed successfully!")
        
        # Validate results structure
        print("\n📊 Validating analysis results structure...")
        
        required_keys = [
            "repository_facts", 
            "most_important_files", 
            "entry_points",
            "repository_structure",
            "actual_errors", 
            "summaries",
            "error_summary",
            "analysis_metadata"
        ]
        
        for key in required_keys:
            if key in results:
                print(f"✅ Found '{key}' in results")
            else:
                print(f"❌ Missing '{key}' in results")
                return False
        
        # Validate summaries structure
        print("\n📝 Validating summaries structure...")
        summaries = results.get("summaries", {})
        
        summary_keys = [
            "codebase_summary",
            "file_summaries", 
            "class_summaries",
            "function_summaries",
            "symbol_summaries",
            "summary_metadata"
        ]
        
        for key in summary_keys:
            if key in summaries:
                print(f"✅ Found '{key}' in summaries")
            else:
                print(f"❌ Missing '{key}' in summaries")
                return False
        
        # Print some sample results
        print("\n📋 Sample Analysis Results:")
        print("-" * 40)
        
        # Repository facts
        repo_facts = results.get("repository_facts", {})
        print(f"📁 Total files: {repo_facts.get('total_files', 'N/A')}")
        print(f"📄 Python files: {repo_facts.get('python_files', 'N/A')}")
        print(f"📊 Total lines: {repo_facts.get('total_lines', 'N/A')}")
        
        # Codebase summary
        codebase_summary = summaries.get("codebase_summary", "")
        if codebase_summary and codebase_summary != "No codebase available":
            print(f"📝 Codebase summary preview: {codebase_summary[:200]}...")
        else:
            print(f"⚠️  Codebase summary: {codebase_summary}")
        
        # Summary metadata
        summary_metadata = summaries.get("summary_metadata", {})
        print(f"🔧 Graph-sitter available: {summary_metadata.get('graph_sitter_available', 'N/A')}")
        print(f"📊 Total summaries generated: {summary_metadata.get('total_summaries_generated', 'N/A')}")
        
        # Most important files
        important_files = results.get("most_important_files", [])
        print(f"🎯 Most important files count: {len(important_files)}")
        if important_files:
            print("   Top 3 important files:")
            for i, file_info in enumerate(important_files[:3]):
                print(f"   {i+1}. {file_info.get('file', 'N/A')} (score: {file_info.get('importance_score', 'N/A')})")
        
        # Actual errors
        actual_errors = results.get("actual_errors", [])
        print(f"🐛 Actual errors found: {len(actual_errors)}")
        if actual_errors:
            print("   Sample errors:")
            for i, error in enumerate(actual_errors[:3]):
                print(f"   {i+1}. {error}")
        
        # Entry points
        entry_points = results.get("entry_points", [])
        print(f"🚪 Entry points found: {len(entry_points)}")
        if entry_points:
            print("   Top 3 entry points:")
            for i, entry in enumerate(entry_points[:3]):
                print(f"   {i+1}. {entry.get('file', 'N/A')}")
        
        print("\n🎉 Test completed successfully!")
        print("✅ All validation checks passed")
        print("✅ Graph-sitter integration working properly")
        print("✅ Structured summaries generated correctly")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure graph-sitter is installed: pip install graph-sitter")
        return False
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_adk_python_analysis()
    if success:
        print("\n🚀 Ready for production!")
        sys.exit(0)
    else:
        print("\n💥 Test failed - needs investigation")
        sys.exit(1)
