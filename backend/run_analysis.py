#!/usr/bin/env python3
"""
Run comprehensive analysis on the codebase-analytics repository.
"""

from analysis import ComprehensiveCodebaseAnalyzer
import json
import os
from collections import Counter

def main():
    # Analyze the current repository (codebase-analytics)
    repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f'🔍 Analyzing repository: {repo_path}')

    analyzer = ComprehensiveCodebaseAnalyzer(repo_path)
    results = analyzer.analyze()

    print('\n' + '='*80)
    print('📊 COMPREHENSIVE CODEBASE ANALYSIS REPORT')
    print('='*80)

    if results.get('success', False):
        print('✅ Analysis completed successfully!')
        
        # Print summary
        summary = results.get('summary', {})
        print(f'\n📈 SUMMARY:')
        print(f'  Total Issues: {summary.get("total_issues", 0)}')
        print(f'  Critical Issues: {summary.get("critical_issues", 0)}')
        print(f'  Error Issues: {summary.get("error_issues", 0)}')
        print(f'  Warning Issues: {summary.get("warning_issues", 0)}')
        print(f'  Info Issues: {summary.get("info_issues", 0)}')
        print(f'  Analysis Duration: {results.get("duration", 0):.2f} seconds')
        
        # Print issues by category
        issues = results.get('issues', [])
        if issues:
            print(f'\n🔍 ISSUES BY CATEGORY:')
            categories = Counter()
            severities = Counter()
            
            for issue in issues:
                if isinstance(issue, dict):
                    categories[issue.get('category', 'unknown')] += 1
                    severities[issue.get('severity', 'unknown')] += 1
                else:
                    # Handle Issue objects
                    categories[getattr(issue, 'category', 'unknown')] += 1
                    severities[getattr(issue, 'severity', 'unknown')] += 1
            
            for category, count in categories.most_common():
                print(f'  {category}: {count}')
            
            print(f'\n⚠️ ISSUES BY SEVERITY:')
            for severity, count in severities.most_common():
                print(f'  {severity}: {count}')
            
            # Show first few issues
            print(f'\n🚨 TOP ISSUES:')
            for i, issue in enumerate(issues[:5]):
                if isinstance(issue, dict):
                    print(f'  {i+1}. [{issue.get("severity", "unknown")}] {issue.get("message", "No message")}')
                    if issue.get('location'):
                        loc = issue['location']
                        print(f'     Location: {loc.get("file_path", "unknown")}:{loc.get("line_start", "?")}')
                else:
                    print(f'  {i+1}. [{getattr(issue, "severity", "unknown")}] {getattr(issue, "message", "No message")}')
                    if hasattr(issue, 'location') and issue.location:
                        loc = issue.location
                        print(f'     Location: {getattr(loc, "file_path", "unknown")}:{getattr(loc, "line_start", "?")}')
        
        # Print enhanced features if available
        if 'enhanced_report' in results:
            enhanced = results['enhanced_report']
            print(f'\n🎯 ENHANCED FEATURES:')
            
            insights = enhanced.get('actionable_insights', [])
            if insights:
                print(f'  Actionable Insights: {len(insights)}')
                for i, insight in enumerate(insights[:3]):
                    print(f'    {i+1}. {insight.get("title", "No title")} (Priority: {insight.get("priority", "unknown")})')
            
            exec_summary = enhanced.get('executive_summary', {})
            if exec_summary:
                print(f'  Executive Summary: {exec_summary.get("overall_health", "unknown")} health')
                print(f'  Key Metrics: {exec_summary.get("key_metrics", {})}')
        
    else:
        print('❌ Analysis failed!')
        print(f'Error: {results.get("error", "Unknown error")}')
        print(f'Issues found: {len(results.get("issues", []))}')
        
        # Still show any issues that were found
        issues = results.get('issues', [])
        if issues:
            print(f'\n🔍 ISSUES FOUND DURING FAILED ANALYSIS:')
            categories = Counter()
            severities = Counter()
            
            for issue in issues:
                if isinstance(issue, dict):
                    categories[issue.get('category', 'unknown')] += 1
                    severities[issue.get('severity', 'unknown')] += 1
                else:
                    # Handle Issue objects
                    categories[getattr(issue, 'category', 'unknown')] += 1
                    severities[getattr(issue, 'severity', 'unknown')] += 1
            
            for category, count in categories.most_common():
                print(f'  {category}: {count}')
            
            print(f'\n⚠️ ISSUES BY SEVERITY:')
            for severity, count in severities.most_common():
                print(f'  {severity}: {count}')
            
            # Show first few issues
            print(f'\n🚨 DETECTED ISSUES:')
            for i, issue in enumerate(issues[:10]):
                if isinstance(issue, dict):
                    print(f'  {i+1}. [{issue.get("severity", "unknown")}] {issue.get("message", "No message")}')
                    if issue.get('location'):
                        loc = issue['location']
                        print(f'     Location: {loc.get("file_path", "unknown")}:{loc.get("line_start", "?")}')
                else:
                    print(f'  {i+1}. [{getattr(issue, "severity", "unknown")}] {getattr(issue, "message", "No message")}')
                    if hasattr(issue, 'location') and issue.location:
                        loc = issue.location
                        print(f'     Location: {getattr(loc, "file_path", "unknown")}:{getattr(loc, "line_start", "?")}')

    print('\n' + '='*80)
    print('📋 ANALYSIS COMPLETE')
    print('='*80)
    
    # Save results to file
    output_file = 'analysis_report.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'📄 Full results saved to: {output_file}')

if __name__ == "__main__":
    main()
