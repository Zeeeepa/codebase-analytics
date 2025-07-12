#!/usr/bin/env python3
"""
Quick test to validate the fixes for missing data
"""

import sys
sys.path.insert(0, 'backend')

from analysis import GraphSitterAnalyzer

def test_fixes():
    """Test the fixes for missing data"""
    print("🔧 Testing fixes for missing data...")
    
    try:
        # Initialize analyzer
        analyzer = GraphSitterAnalyzer()
        
        # Test with adk-python repository
        repo_url = "https://github.com/Zeeeepa/adk-python"
        print(f"🔍 Analyzing repository: {repo_url}")
        
        # Perform analysis
        results = analyzer.analyze_repository(repo_url)
        
        # Check repository facts
        repo_facts = results.get("repository_facts", {})
        print("\n📊 Repository Facts:")
        print(f"  Total files: {repo_facts.get('total_files', 'N/A')}")
        print(f"  Python files: {repo_facts.get('python_files', 'N/A')}")
        print(f"  Total lines: {repo_facts.get('total_lines', 'N/A')}")
        
        # Check most important files
        important_files = results.get("most_important_files", [])
        print(f"\n🎯 Most Important Files:")
        for i, file_info in enumerate(important_files[:3]):
            file_name = file_info.get('file', 'N/A')
            filepath = file_info.get('filepath', 'N/A')
            score = file_info.get('importance_score', 'N/A')
            print(f"  {i+1}. File: {file_name}")
            print(f"     Path: {filepath}")
            print(f"     Score: {score}")
        
        # Check entry points
        entry_points = results.get("entry_points", [])
        print(f"\n🚪 Entry Points (first 3):")
        for i, entry in enumerate(entry_points[:3]):
            if isinstance(entry, dict):
                func_name = entry.get('function_name', 'N/A')
                filepath = entry.get('filepath', 'N/A')
                print(f"  {i+1}. Function: {func_name}")
                print(f"     File: {filepath}")
            else:
                print(f"  {i+1}. {entry}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_fixes()

