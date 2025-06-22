#!/usr/bin/env python3
"""
Demo script showing how to use the comprehensive backend analysis system
"""
from api import app, load_codebase
from analysis import analyze_codebase
from visualize import generate_visualization_data
import json
def run_demo():
    """Run a comprehensive demo of the backend system"""
    print("🚀 COMPREHENSIVE BACKEND ANALYSIS DEMO")
    print("=" * 60)
    
    # Load a mock codebase
    print("📁 Loading codebase...")
    codebase = load_codebase("codegen-sh", "graph-sitter")
    
    # Perform analysis
    print("🔍 Performing comprehensive analysis...")
    analysis_results = analyze_codebase(codebase)
    
    # Display key results
    print("\n📊 ANALYSIS SUMMARY:")
    print("-" * 30)
    summary = analysis_results.get('summary', {})
    print(f"📁 Total Files: {summary.get('total_files', 0)}")
    print(f"🔧 Total Functions: {summary.get('total_functions', 0)}")
    print(f"🚨 Total Issues: {summary.get('total_issues', 0)}")
    print(f"⚠️  Critical Issues: {summary.get('critical_issues', 0)}")
    print(f"👉 Major Issues: {summary.get('major_issues', 0)}")
    print(f"🔍 Minor Issues: {summary.get('minor_issues', 0)}")
    print(f"💀 Dead Code Items: {summary.get('dead_code_items', 0)}")
    print(f"🎯 Entry Points: {summary.get('entry_points', 0)}")
    
    # Show most important functions
    print("\n🌟 MOST IMPORTANT FUNCTIONS:")
    print("-" * 35)
    important = analysis_results.get('most_important_functions', {})
    
    most_calls = important.get('most_calls', {})
    print(f"📞 Makes Most Calls: {most_calls.get('name', 'N/A')}")
    print(f"   📊 Call Count: {most_calls.get('call_count', 0)}")
    if most_calls.get('calls'):
        print(f"   🎯 Calls: {', '.join(most_calls['calls'][:3])}...")
    
    most_called = important.get('most_called', {})
    print(f"📈 Most Called: {most_called.get('name', 'N/A')}")
    print(f"   📊 Usage Count: {most_called.get('usage_count', 0)}")
    
    deepest_inheritance = important.get('deepest_inheritance', {})
    if deepest_inheritance.get('name'):
        print(f"🌳 Deepest Inheritance: {deepest_inheritance.get('name')}")
        print(f"   📊 Chain Depth: {deepest_inheritance.get('chain_depth', 0)}")
    
    # Show function contexts
    print("\n🔧 FUNCTION CONTEXTS:")
    print("-" * 25)
    function_contexts = analysis_results.get('function_contexts', {})
    
    for func_name, context in list(function_contexts.items())[:3]:  # Show first 3
        print(f"\n📝 Function: {func_name}")
        print(f"   📁 File: {context.get('filepath', 'N/A')}")
        print(f"   📊 Parameters: {len(context.get('parameters', []))}")
        print(f"   🔗 Dependencies: {len(context.get('dependencies', []))}")
        print(f"   📞 Function Calls: {len(context.get('function_calls', []))}")
        print(f"   📈 Called By: {len(context.get('called_by', []))}")
        print(f"   🚨 Issues: {len(context.get('issues', []))}")
        print(f"   🎯 Entry Point: {context.get('is_entry_point', False)}")
        print(f"   💀 Dead Code: {context.get('is_dead_code', False)}")
        
        if context.get('max_call_chain'):
            chain = context['max_call_chain']
            if len(chain) > 1:
                print(f"   ⛓️  Call Chain: {' → '.join(chain[:3])}...")
    
    # Show issues by severity
    print("\n🚨 ISSUES BY SEVERITY:")
    print("-" * 25)
    issues_by_severity = analysis_results.get('issues_by_severity', {})
    
    for severity, issues in issues_by_severity.items():
        if issues:
            print(f"\n{severity.upper()} ({len(issues)} issues):")
            for issue in issues[:2]:  # Show first 2 issues
                print(f"  • {issue.get('message', 'No message')}")
                print(f"    📁 {issue.get('filepath', 'N/A')}:{issue.get('line_number', 0)}")
            if len(issues) > 2:
                print(f"  ... and {len(issues) - 2} more")
    
    # Show dead code analysis
    print("\n💀 DEAD CODE ANALYSIS:")
    print("-" * 22)
    dead_code = analysis_results.get('dead_code_analysis', {})
    print(f"🔢 Total Dead Functions: {dead_code.get('total_dead_functions', 0)}")
    
    dead_items = dead_code.get('dead_code_items', [])
    if dead_items:
        print("📋 Dead Code Items:")
        for item in dead_items[:3]:  # Show first 3
            print(f"  • {item.get('name', 'N/A')} ({item.get('type', 'unknown')}) - {item.get('reason', 'No reason')}")
            print(f"    📁 {item.get('filepath', 'N/A')}")
            blast_radius = item.get('blast_radius', [])
            if blast_radius:
                print(f"    💥 Blast Radius: {', '.join(blast_radius[:3])}...")
    
    # Show Halstead metrics
    print("\n📊 HALSTEAD METRICS:")
    print("-" * 20)
    halstead = analysis_results.get('halstead_metrics', {})
    print(f"📝 Operators (n1): {halstead.get('n1', 0)}")
    print(f"📝 Operands (n2): {halstead.get('n2', 0)}")
    print(f"📊 Total Operators (N1): {halstead.get('N1', 0)}")
    print(f"📊 Total Operands (N2): {halstead.get('N2', 0)}")
    print(f"📚 Vocabulary: {halstead.get('vocabulary', 0)}")
    print(f"📏 Length: {halstead.get('length', 0)}")
    print(f"📦 Volume: {halstead.get('volume', 0):.2f}")
    print(f"⚡ Difficulty: {halstead.get('difficulty', 0):.2f}")
    print(f"💪 Effort: {halstead.get('effort', 0):.2f}")
    
    # Generate visualization data
    print("\n🎨 GENERATING VISUALIZATION DATA:")
    print("-" * 35)
    viz_data = generate_visualization_data(analysis_results)
    
    print("✅ Repository tree with issue counts")
    print("✅ Issue heatmap and severity distribution")
    print("✅ Dead code blast radius visualization")
    print("✅ Interactive call graph")
    print("✅ Dependency visualization")
    print("✅ Metrics charts and dashboards")
    print("✅ Function context panels")
    
    # Show API endpoints
    print("\n🌐 API ENDPOINTS AVAILABLE:")
    print("-" * 28)
    print("🔗 GET /analyze/username/repo - Complete analysis")
    print("🔗 GET /visualize/username/repo - Interactive visualization")
    
    print("\n🚀 TO START THE API SERVER:")
    print("-" * 27)
    print("python backend/api.py")
    print("\n🌐 Then visit:")
    print("http://localhost:5000/analyze/codegen-sh/graph-sitter")
    print("http://localhost:5000/visualize/codegen-sh/graph-sitter")
    
    print("\n" + "=" * 60)
    print("✨ Demo complete! Backend system ready for integration! ✨")
if __name__ == "__main__":
    run_demo()