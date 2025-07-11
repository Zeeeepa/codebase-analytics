#!/usr/bin/env python3
"""
KAG Repository Analysis Script
Uses the comprehensive analysis system to analyze the OpenSPG/KAG repository
"""

import sys
import os
import json
import time
from datetime import datetime

# Add backend to path
sys.path.append('./backend')

try:
    from graph_sitter.core.codebase import Codebase
    from backend.comprehensive_analyzer import ComprehensiveCodebaseAnalyzer
    from backend.advanced_issues import AdvancedIssueDetector
    from backend.function_context import FunctionContextAnalyzer
    from backend.halstead_metrics import HalsteadMetricsCalculator
    from backend.graph_analysis import GraphAnalyzer
    from backend.dead_code_analysis import DeadCodeAnalyzer
    from backend.health_metrics import HealthMetricsCalculator
    from backend.repository_structure import RepositoryStructureAnalyzer
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the backend branch with all analysis modules")
    sys.exit(1)

def analyze_kag_repository():
    """Analyze the KAG repository using our comprehensive system"""
    
    print("🎯 Starting KAG Repository Analysis...")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Load the KAG repository
        print("📥 Loading KAG repository from GitHub...")
        repo_url = "https://github.com/OpenSPG/KAG"
        
        # Use codegen to load the repository
        codebase = Codebase.from_repo(repo_url)
        print(f"✅ Successfully loaded KAG repository")
        print(f"📊 Repository stats: {len(codebase.files)} files, {len(codebase.functions)} functions")
        
        # Initialize comprehensive analyzer
        print("\n🔍 Initializing Comprehensive Analysis System...")
        analyzer = ComprehensiveCodebaseAnalyzer(codebase)
        
        # Perform comprehensive analysis
        print("🚀 Running comprehensive analysis...")
        results = analyzer.analyze()
        
        # Get structured data
        structured_data = analyzer.get_structured_data()
        health_dashboard = analyzer.get_health_dashboard_data()
        
        processing_time = time.time() - start_time
        
        # Create comprehensive report
        report = {
            "analysis_metadata": {
                "repository": "OpenSPG/KAG",
                "analysis_timestamp": datetime.now().isoformat(),
                "processing_time_seconds": round(processing_time, 2),
                "analyzer_version": "2.0.0"
            },
            "repository_overview": {
                "total_files": len(codebase.files),
                "total_functions": len(codebase.functions),
                "languages_detected": list(set([f.language for f in codebase.files if hasattr(f, 'language')])),
                "repository_size_estimate": f"{len(codebase.files) * 50}+ lines" # Rough estimate
            },
            "analysis_results": structured_data,
            "health_dashboard": health_dashboard,
            "key_insights": generate_key_insights(structured_data, health_dashboard),
            "recommendations": generate_recommendations(structured_data, health_dashboard)
        }
        
        # Save detailed report
        output_file = f"kag_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Display summary
        display_analysis_summary(report)
        
        print(f"\n📄 Detailed report saved to: {output_file}")
        print(f"⏱️  Total analysis time: {processing_time:.2f} seconds")
        
        return report
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def generate_key_insights(structured_data, health_dashboard):
    """Generate key insights from the analysis"""
    insights = []
    
    # Health insights
    health_score = health_dashboard.get('health_score', 0)
    if health_score >= 90:
        insights.append("🟢 Excellent codebase health - well-maintained and high quality")
    elif health_score >= 75:
        insights.append("🟡 Good codebase health with room for improvement")
    elif health_score >= 60:
        insights.append("🟠 Moderate codebase health - attention needed")
    else:
        insights.append("🔴 Poor codebase health - significant issues detected")
    
    # Issue insights
    issues = structured_data.get('issues', {})
    total_issues = issues.get('total_issues', 0)
    if total_issues > 100:
        insights.append(f"⚠️ High issue count detected: {total_issues} issues found")
    
    automated_fixes = issues.get('automated_resolutions_available', 0)
    if automated_fixes > 0:
        insights.append(f"🤖 {automated_fixes} automated fixes available")
    
    # Function insights
    functions = structured_data.get('functions', {})
    entry_points = functions.get('entry_points', [])
    if len(entry_points) > 10:
        insights.append(f"🎯 Complex architecture: {len(entry_points)} entry points detected")
    
    # Dead code insights
    dead_code = structured_data.get('dead_code', {})
    unused_functions = dead_code.get('unused_functions', 0)
    if unused_functions > 20:
        insights.append(f"💀 Significant dead code: {unused_functions} unused functions")
    
    return insights

def generate_recommendations(structured_data, health_dashboard):
    """Generate actionable recommendations"""
    recommendations = []
    
    # Health-based recommendations
    health_score = health_dashboard.get('health_score', 0)
    if health_score < 80:
        recommendations.append({
            "priority": "high",
            "category": "health",
            "title": "Improve Overall Codebase Health",
            "description": f"Current health score: {health_score}/100. Focus on reducing technical debt and fixing critical issues."
        })
    
    # Issue-based recommendations
    issues = structured_data.get('issues', {})
    critical_issues = issues.get('by_severity', {}).get('critical', 0)
    if critical_issues > 0:
        recommendations.append({
            "priority": "critical",
            "category": "issues",
            "title": "Address Critical Issues",
            "description": f"{critical_issues} critical issues require immediate attention."
        })
    
    # Automated fixes
    automated_fixes = issues.get('automated_resolutions_available', 0)
    if automated_fixes > 0:
        recommendations.append({
            "priority": "medium",
            "category": "automation",
            "title": "Apply Automated Fixes",
            "description": f"{automated_fixes} issues can be automatically resolved with high confidence."
        })
    
    # Dead code recommendations
    dead_code = structured_data.get('dead_code', {})
    safe_removals = dead_code.get('blast_radius', {}).get('safe_removals', 0)
    if safe_removals > 0:
        recommendations.append({
            "priority": "low",
            "category": "cleanup",
            "title": "Remove Dead Code",
            "description": f"{safe_removals} functions can be safely removed to reduce codebase size."
        })
    
    return recommendations

def display_analysis_summary(report):
    """Display a formatted summary of the analysis"""
    print("\n" + "="*80)
    print("🎯 KAG REPOSITORY ANALYSIS SUMMARY")
    print("="*80)
    
    # Repository overview
    overview = report['repository_overview']
    print(f"\n📊 REPOSITORY OVERVIEW:")
    print(f"   • Total Files: {overview['total_files']}")
    print(f"   • Total Functions: {overview['total_functions']}")
    print(f"   • Languages: {', '.join(overview['languages_detected'])}")
    print(f"   • Estimated Size: {overview['repository_size_estimate']}")
    
    # Health dashboard
    health = report['health_dashboard']
    print(f"\n🏥 HEALTH ASSESSMENT:")
    print(f"   • Overall Score: {health.get('health_score', 'N/A')}/100")
    print(f"   • Health Grade: {health.get('health_grade', 'N/A')}")
    print(f"   • Risk Level: {health.get('risk_level', 'N/A')}")
    print(f"   • Automated Fixes Available: {health.get('automated_fixes_available', 0)}")
    
    # Issues summary
    issues = report['analysis_results'].get('issues', {})
    print(f"\n🔍 ISSUES DETECTED:")
    print(f"   • Total Issues: {issues.get('total_issues', 0)}")
    severity_breakdown = issues.get('by_severity', {})
    for severity, count in severity_breakdown.items():
        print(f"   • {severity.title()}: {count}")
    
    # Key insights
    insights = report.get('key_insights', [])
    if insights:
        print(f"\n💡 KEY INSIGHTS:")
        for insight in insights:
            print(f"   • {insight}")
    
    # Top recommendations
    recommendations = report.get('recommendations', [])
    if recommendations:
        print(f"\n🎯 TOP RECOMMENDATIONS:")
        for rec in recommendations[:3]:  # Show top 3
            print(f"   • [{rec['priority'].upper()}] {rec['title']}")
            print(f"     {rec['description']}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    print("🚀 KAG Repository Comprehensive Analysis")
    print("Using Advanced Analysis System v2.0")
    print("-" * 50)
    
    result = analyze_kag_repository()
    
    if result:
        print("\n✅ Analysis completed successfully!")
    else:
        print("\n❌ Analysis failed!")
        sys.exit(1)
