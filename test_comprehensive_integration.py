#!/usr/bin/env python3
"""
Test script to validate comprehensive analysis integration
Tests the enhanced Graph-sitter compliant analysis engine
"""

import sys
import os
sys.path.append('backend')

from analysis import GraphSitterAnalyzer, analyze_codebase, get_symbol_context, get_interactive_symbol_data

def test_basic_analysis():
    """Test basic analysis functionality"""
    print("🔍 Testing basic analysis functionality...")
    
    try:
        # Test with current repository
        analyzer = GraphSitterAnalyzer('.')
        result = analyzer.analyze_codebase()
        
        print(f"✅ Analysis completed successfully!")
        print(f"   📁 Total files: {result.total_files}")
        print(f"   🔧 Total functions: {len(result.all_functions)}")
        print(f"   🚪 Total entry points: {len(result.all_entry_points)}")
        print(f"   ⚠️  Total issues: {len(result.all_issues)}")
        print(f"   🌐 Languages: {', '.join(result.programming_languages)}")
        
        return True
    except Exception as e:
        print(f"❌ Basic analysis failed: {e}")
        return False

def test_comprehensive_features():
    """Test comprehensive analysis features"""
    print("\n🔍 Testing comprehensive analysis features...")
    
    try:
        result = analyze_codebase('.')
        
        # Test interactive symbol data
        symbol_data = get_interactive_symbol_data(result)
        print(f"✅ Interactive symbols: {symbol_data['total_symbols']} symbols")
        print(f"   🔧 Functions: {symbol_data['functions_count']}")
        print(f"   🚪 Entry points: {symbol_data['entry_points_count']}")
        
        # Test symbol context (if we have any functions)
        if result.all_functions:
            first_function = result.all_functions[0]
            context = get_symbol_context(result, first_function.name)
            if context:
                print(f"✅ Symbol context for '{first_function.name}': {len(context['definitions'])} definitions")
            else:
                print(f"⚠️  No context found for '{first_function.name}'")
        
        return True
    except Exception as e:
        print(f"❌ Comprehensive features test failed: {e}")
        return False

def test_issue_detection():
    """Test issue detection capabilities"""
    print("\n🔍 Testing issue detection...")
    
    try:
        result = analyze_codebase('.')
        
        # Group issues by type
        issue_types = {}
        for issue in result.all_issues:
            issue_type = issue.type.value
            if issue_type not in issue_types:
                issue_types[issue_type] = 0
            issue_types[issue_type] += 1
        
        print(f"✅ Issue detection completed!")
        for issue_type, count in issue_types.items():
            print(f"   {issue_type}: {count}")
        
        return True
    except Exception as e:
        print(f"❌ Issue detection test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Comprehensive Analysis Integration Tests")
    print("=" * 60)
    
    tests = [
        test_basic_analysis,
        test_comprehensive_features,
        test_issue_detection
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Integration successful!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

