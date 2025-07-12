#!/usr/bin/env python3
"""
Test script to analyze the codebase-analytics repository with enhanced backend
"""

import sys
import os
import json
from pathlib import Path

# Add backend to path
sys.path.insert(0, 'backend')

try:
    from models import CodeIssue, IssueType, IssueSeverity, AnalysisResults
    from analysis import CodebaseAnalyzer
    print("✅ Successfully imported enhanced analysis modules")
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)

def create_mock_codebase():
    """Create a mock codebase object for testing"""
    class MockSourceFile:
        def __init__(self, filepath, source=None):
            self.file_path = filepath
            self.source = source or ""
            self.symbols = []
            self.language = filepath.split('.')[-1] if '.' in filepath else 'unknown'
    
    class MockCodebase:
        def __init__(self):
            self.files = []
            
        def add_file(self, filepath, source=None):
            if source is None and os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    source = f.read()
            self.files.append(MockSourceFile(filepath, source))
    
    return MockCodebase()

def analyze_codebase_analytics():
    """Analyze the codebase-analytics repository"""
    print("🔍 Starting analysis of codebase-analytics repository...")
    
    # Create mock codebase
    codebase = create_mock_codebase()
    
    # Add Python files from the repository
    python_files = []
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories and common ignore patterns
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                python_files.append(filepath)
                codebase.add_file(filepath)
    
    print(f"📁 Found {len(python_files)} Python files to analyze")
    
    # Initialize analyzer
    analyzer = CodebaseAnalyzer()
    
    # Run analysis
    print("🚀 Running enhanced analysis...")
    results = analyzer.analyze(".")
    
    # Display results
    print(f"\n📊 Analysis Results:")
    print(f"   Total Issues: {len(results.issues)}")
    
    # Group issues by type
    issue_counts = {}
    for issue in results.issues:
        issue_type = issue.issue_type.value
        issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
    
    print(f"   Issue Breakdown:")
    for issue_type, count in sorted(issue_counts.items()):
        print(f"     - {issue_type}: {count}")
    
    # Show severity breakdown
    severity_counts = {}
    for issue in results.issues:
        severity = issue.severity.value
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    print(f"   Severity Breakdown:")
    for severity, count in sorted(severity_counts.items()):
        print(f"     - {severity}: {count}")
    
    # Show some example issues
    print(f"\n🔍 Sample Issues (first 5):")
    for i, issue in enumerate(results.issues[:5]):
        print(f"   {i+1}. [{issue.severity.value}] {issue.message}")
        print(f"      File: {issue.filepath}:{issue.line_number}")
        if issue.suggested_fix:
            print(f"      Fix: {issue.suggested_fix}")
        print()
    
    # Test enhanced features
    print(f"📈 Enhanced Features:")
    print(f"   Automated Resolutions: {len(results.automated_resolutions)}")
    print(f"   Health Score: {results.health_score}")
    
    # Save results to file
    results_data = {
        "total_issues": len(results.issues),
        "issue_counts": issue_counts,
        "severity_counts": severity_counts,
        "health_score": results.health_score,
        "automated_resolutions": len(results.automated_resolutions),
        "files_analyzed": len(python_files),
        "sample_issues": [
            {
                "type": issue.issue_type.value,
                "severity": issue.severity.value,
                "message": issue.message,
                "file": issue.filepath,
                "line": issue.line_number,
                "fix": issue.suggested_fix
            }
            for issue in results.issues[:10]
        ]
    }
    
    with open('enhanced_analysis_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"💾 Results saved to enhanced_analysis_results.json")
    
    return results

if __name__ == "__main__":
    try:
        results = analyze_codebase_analytics()
        print("✅ Analysis completed successfully!")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
