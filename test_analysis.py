#!/usr/bin/env python3
"""
Test script to analyze the codebase-analytics repository using our comprehensive analysis.
"""

import sys
import os
sys.path.append('backend')

from codegen_sdk_pink import Codebase
from codegen.configs.models.codebase import CodebaseConfig
from backend.api import ComprehensiveAnalysis
import json

def analyze_codebase_analytics():
    """Analyze the codebase-analytics repository."""
    print("🔍 Starting Comprehensive Analysis of codebase-analytics repository...")
    
    try:
        # Initialize codebase (codegen_sdk_pink doesn't use config parameter)
        # Note: Configuration would be handled differently in codegen_sdk_pink
        
        # Analyze current directory
        codebase = Codebase(".")
        
        # Perform comprehensive analysis
        analyzer = ComprehensiveAnalysis(codebase)
        analysis_results = analyzer.analyze()
        
        # Print results in a structured format
        print_analysis_results(analysis_results)
        
        return analysis_results
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def print_analysis_results(results):
    """Print analysis results in a structured, readable format."""
    
    print("\n📊 COMPREHENSIVE CODEBASE ANALYSIS RESULTS")
    print("=" * 60)
    
    # Summary
    summary = results.get('summary', {})
    print(f"\n📋 SUMMARY:")
    print("-" * 20)
    print(f"📁 Total Files: {summary.get('total_files', 0)}")
    print(f"🔧 Total Functions: {summary.get('total_functions', 0)}")
    print(f"📦 Total Classes: {summary.get('total_classes', 0)}")
    print(f"🔗 Total Imports: {summary.get('total_imports', 0)}")
    print(f"🚨 Total Issues: {summary.get('total_issues', 0)}")
    print(f"⚠️  Critical Issues: {summary.get('critical_issues', 0)}")
    print(f"👉 Major Issues: {summary.get('major_issues', 0)}")
    print(f"🔍 Minor Issues: {summary.get('minor_issues', 0)}")
    print(f"💀 Dead Code Items: {summary.get('dead_code_items', 0)}")
    print(f"🎯 Entry Points: {summary.get('entry_points', 0)}")
    
    # Most Important Files
    important_files = results.get('most_important_files', {})
    print(f"\n🌟 MOST IMPORTANT FILES:")
    print("-" * 30)
    
    entry_points = important_files.get('entry_points', [])
    if entry_points:
        print("🚀 Entry Points:")
        for ep in entry_points[:5]:  # Show top 5
            print(f"   📄 {ep.get('file', 'Unknown')}")
            print(f"      Functions: {ep.get('functions', 0)}, Classes: {ep.get('classes', 0)}")
            print(f"      Importance Score: {ep.get('importance_score', 0)}")
    
    most_called = important_files.get('most_called_functions', [])
    if most_called:
        print("\n📞 Most Called Functions:")
        for func in most_called[:5]:  # Show top 5
            print(f"   🔧 {func.get('function', 'Unknown')} ({func.get('call_count', 0)} calls)")
            print(f"      📄 File: {func.get('file', 'Unknown')}")
            if func.get('error_indicators'):
                print(f"      ⚠️  Issues: {', '.join(func.get('error_indicators', []))}")
    
    highest_deps = important_files.get('highest_dependency_files', [])
    if highest_deps:
        print("\n🔗 Highest Dependency Files:")
        for file in highest_deps[:5]:  # Show top 5
            print(f"   📄 {file.get('file', 'Unknown')}")
            print(f"      Dependency Score: {file.get('dependency_score', 0)}")
            if file.get('potential_issues'):
                print(f"      ⚠️  Issues: {', '.join(file.get('potential_issues', []))}")
    
    # Comprehensive Error Analysis
    error_analysis = results.get('comprehensive_error_analysis', {})
    print(f"\n🚨 COMPREHENSIVE ERROR ANALYSIS:")
    print("-" * 40)
    
    # Runtime Risks
    runtime_risks = error_analysis.get('runtime_risks', [])
    if runtime_risks:
        print(f"⚡ Runtime Risks ({len(runtime_risks)}):")
        for risk in runtime_risks[:3]:  # Show top 3
            print(f"   🔥 {risk.get('type', 'Unknown')}: {risk.get('function', 'Unknown')}")
            print(f"      📄 {risk.get('file', 'Unknown')}")
            print(f"      📝 {risk.get('description', 'No description')}")
    
    # Security Risks
    security_risks = error_analysis.get('security_risks', [])
    if security_risks:
        print(f"\n🛡️  Security Risks ({len(security_risks)}):")
        for risk in security_risks[:3]:  # Show top 3
            print(f"   🚨 {risk.get('type', 'Unknown')}: {risk.get('pattern', 'Unknown')}")
            print(f"      🔧 Function: {risk.get('function', 'Unknown')}")
            print(f"      📄 File: {risk.get('file', 'Unknown')}")
    
    # Circular Dependencies
    circular_deps = error_analysis.get('circular_dependencies', [])
    if circular_deps:
        print(f"\n🔄 Circular Dependencies ({len(circular_deps)}):")
        for dep in circular_deps[:3]:  # Show top 3
            files = dep.get('files', [])
            print(f"   ↻ {' ↔ '.join(files)}")
    
    # Dead Code
    dead_code = error_analysis.get('dead_code', [])
    if dead_code:
        print(f"\n💀 Dead Code ({len(dead_code)}):")
        for code in dead_code[:5]:  # Show top 5
            print(f"   🗑️  {code.get('function', 'Unknown')} in {code.get('file', 'Unknown')}")
    
    # Function Contexts (for functions with issues)
    function_contexts = results.get('function_contexts', {})
    functions_with_issues = {name: ctx for name, ctx in function_contexts.items() 
                           if ctx.get('issues') and len(ctx.get('issues', [])) > 0}
    
    if functions_with_issues:
        print(f"\n🔧 FUNCTIONS WITH ISSUES:")
        print("-" * 30)
        for func_name, context in list(functions_with_issues.items())[:3]:  # Show top 3
            print(f"\n📝 Function: {func_name}")
            print(f"   📄 File: {context.get('filepath', 'Unknown')}")
            print(f"   📊 Parameters: {len(context.get('parameters', []))}")
            print(f"   🔗 Dependencies: {len(context.get('dependencies', []))}")
            print(f"   📞 Function Calls: {len(context.get('function_calls', []))}")
            print(f"   📈 Called By: {len(context.get('called_by', []))}")
            
            # Show issues
            issues = context.get('issues', [])
            if issues:
                print(f"   🚨 Issues ({len(issues)}):")
                for issue in issues[:2]:  # Show top 2 issues
                    print(f"      ⚠️  {issue.get('type', 'Unknown')}: {issue.get('message', 'No message')}")
            
            # Show control flow
            control_flow = context.get('control_flow', {})
            if control_flow:
                print(f"   🔀 Control Flow: {control_flow.get('if_statements', 0)} ifs, "
                      f"{control_flow.get('loops', 0)} loops, {control_flow.get('try_catch_blocks', 0)} try-catch")
            
            # Show performance indicators
            perf = context.get('performance_indicators', {})
            if perf and perf.get('potential_bottlenecks'):
                print(f"   ⚡ Performance Issues: {', '.join(perf.get('potential_bottlenecks', []))}")
    
    # Code Quality Metrics
    quality = results.get('code_quality_metrics', {})
    if quality:
        print(f"\n📈 CODE QUALITY METRICS:")
        print("-" * 30)
        print(f"📚 Documentation Coverage: {quality.get('documentation_coverage', 0):.1%}")
        print(f"🏷️  Type Hint Coverage: {quality.get('type_hint_coverage', 0):.1%}")
        print(f"⭐ Code Quality Score: {quality.get('code_quality_score', 0):.1f}/100")
    
    # Repository Structure
    repo_structure = results.get('repository_structure', {})
    if repo_structure:
        print(f"\n📁 REPOSITORY STRUCTURE:")
        print("-" * 25)
        file_types = repo_structure.get('file_types', {})
        if file_types:
            print("📄 File Types:")
            for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   .{ext}: {count} files")
        
        total_size = repo_structure.get('estimated_total_size', 0)
        avg_size = repo_structure.get('average_file_size', 0)
        print(f"📏 Total Size: {total_size:,} characters")
        print(f"📊 Average File Size: {avg_size:.0f} characters")

if __name__ == "__main__":
    analyze_codebase_analytics()
