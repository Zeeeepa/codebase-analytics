#!/usr/bin/env python3
"""
Test script for the enhanced graph-sitter analyzer
"""

import sys
import json
from datetime import datetime

# Add backend to path
sys.path.append('./backend')

try:
    from backend.graph_sitter_analyzer import GraphSitterAnalyzer
    print("✅ Successfully imported GraphSitterAnalyzer")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_analyzer():
    """Test the analyzer with the current repository"""
    print("🔍 Testing GraphSitterAnalyzer...")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = GraphSitterAnalyzer()
    
    # Test with current repository
    print("📊 Analyzing current repository (codebase-analytics)...")
    results = analyzer.analyze_repository(".")
    
    if "error" in results:
        print(f"❌ Analysis failed: {results['error']}")
        return False
    
    # Display results summary
    print("\n✅ Analysis completed successfully!")
    print("=" * 60)
    
    # Repository facts
    facts = results.get("repository_facts", {})
    print(f"📁 Total Files: {facts.get('total_files', 0)}")
    print(f"💻 Code Files: {facts.get('code_files', 0)}")
    print(f"📚 Documentation Files: {facts.get('documentation_files', 0)}")
    print(f"⚙️ Config Files: {facts.get('config_files', 0)}")
    print(f"🔧 Total Functions: {facts.get('total_functions', 0)}")
    print(f"🏗️ Total Classes: {facts.get('total_classes', 0)}")
    print(f"🌐 Languages: {facts.get('languages', {})}")
    
    # Most important files
    important_files = results.get("most_important_files", [])
    print(f"\n🎯 Most Important Files ({len(important_files)}):")
    for i, file_info in enumerate(important_files[:5], 1):
        print(f"  {i}. {file_info.get('filepath', 'unknown')} (Score: {file_info.get('importance_score', 0)})")
    
    # Entry points
    entry_points = results.get("entry_points", [])
    print(f"\n🚀 Entry Points ({len(entry_points)}):")
    for i, ep in enumerate(entry_points[:5], 1):
        print(f"  {i}. {ep.get('function_name', 'unknown')} in {ep.get('filepath', 'unknown')} (Score: {ep.get('importance_score', 0)})")
    
    # Errors
    errors = results.get("actual_errors", [])
    error_summary = results.get("error_summary", {})
    print(f"\n🚨 Actual Runtime Errors: {len(errors)}")
    print(f"   🤖 Auto-fixable: {error_summary.get('auto_fixable', 0)}")
    print(f"   ⚠️ Manual review: {error_summary.get('manual_review_required', 0)}")
    
    if errors:
        print("\n📋 Error Examples:")
        for i, error in enumerate(errors[:3], 1):
            print(f"  {i}. {error.get('error_type', 'unknown')} in {error.get('function_name', 'unknown')}")
            print(f"     {error.get('description', 'No description')}")
    
    # Save summary results (skip full details due to complex objects)
    summary = {
        "repository_facts": facts,
        "important_files_count": len(important_files),
        "entry_points_count": len(entry_points),
        "errors_count": len(errors),
        "error_summary": error_summary,
        "analysis_timestamp": results.get("analysis_metadata", {}).get("timestamp", "unknown")
    }
    
    output_file = f"test_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Analysis summary saved to: {output_file}")
    return True

def test_remote_repository():
    """Test with a remote repository"""
    print("\n🌐 Testing with remote repository...")
    print("=" * 60)
    
    analyzer = GraphSitterAnalyzer()
    
    # Test with a small public repository
    test_repo = "https://github.com/Zeeeepa/codebase-analytics"
    print(f"📊 Analyzing remote repository: {test_repo}")
    
    results = analyzer.analyze_repository(test_repo)
    
    if "error" in results:
        print(f"❌ Remote analysis failed: {results['error']}")
        return False
    
    print("✅ Remote analysis completed successfully!")
    
    # Quick summary
    facts = results.get("repository_facts", {})
    print(f"📁 Files: {facts.get('total_files', 0)}")
    print(f"🔧 Functions: {facts.get('total_functions', 0)}")
    print(f"🚨 Errors: {len(results.get('actual_errors', []))}")
    
    return True

if __name__ == "__main__":
    print("🚀 Graph-sitter Analyzer Test Suite")
    print("=" * 60)
    
    # Test local analysis
    success_local = test_analyzer()
    
    # Test remote analysis (if local works)
    if success_local:
        try:
            success_remote = test_remote_repository()
        except Exception as e:
            print(f"⚠️ Remote test failed: {e}")
            success_remote = False
    else:
        success_remote = False
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")
    print(f"   Local Analysis: {'✅ PASS' if success_local else '❌ FAIL'}")
    print(f"   Remote Analysis: {'✅ PASS' if success_remote else '❌ FAIL'}")
    
    if success_local and success_remote:
        print("\n🎉 All tests passed! Ready for production.")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests failed. Check the output above.")
        sys.exit(1)
