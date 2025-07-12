#!/usr/bin/env python3
"""
Debug script to see what keys are actually returned by the analyzer
"""

import sys
sys.path.insert(0, 'backend')

from analysis import GraphSitterAnalyzer

def debug_analysis_results():
    """Debug what keys are returned by the analyzer"""
    print("🔍 Debugging analysis results structure...")
    
    try:
        # Initialize analyzer
        analyzer = GraphSitterAnalyzer()
        
        # Test with adk-python repository
        repo_url = "https://github.com/Zeeeepa/adk-python"
        print(f"🔍 Analyzing repository: {repo_url}")
        
        # Perform analysis
        results = analyzer.analyze_repository(repo_url)
        
        # Print all top-level keys
        print("\n📋 Top-level keys in results:")
        for key in results.keys():
            print(f"  ✅ {key}")
        
        # Check summaries structure
        if "summaries" in results:
            summaries = results["summaries"]
            print("\n📝 Keys in summaries:")
            for key in summaries.keys():
                print(f"  ✅ {key}")
                
            # Print summary metadata
            if "summary_metadata" in summaries:
                metadata = summaries["summary_metadata"]
                print(f"\n📊 Summary metadata:")
                for key, value in metadata.items():
                    print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_analysis_results()

