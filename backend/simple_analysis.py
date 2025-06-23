#!/usr/bin/env python3
"""
Simple codebase analysis that works without the Codegen SDK.
"""

import os
import ast
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any

def analyze_python_file(file_path: str) -> Dict[str, Any]:
    """Analyze a single Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        analysis = {
            'file_path': file_path,
            'lines_of_code': len(content.splitlines()),
            'functions': [],
            'classes': [],
            'imports': [],
            'issues': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                analysis['functions'].append({
                    'name': node.name,
                    'line': node.lineno,
                    'args': len(node.args.args),
                    'has_docstring': ast.get_docstring(node) is not None
                })
                
                # Check for long functions
                if hasattr(node, 'end_lineno') and node.end_lineno:
                    func_lines = node.end_lineno - node.lineno
                    if func_lines > 50:
                        analysis['issues'].append({
                            'type': 'code_quality',
                            'severity': 'warning',
                            'message': f'Function {node.name} is very long ({func_lines} lines)',
                            'line': node.lineno
                        })
                
                # Check for missing docstring
                if not ast.get_docstring(node) and not node.name.startswith('_'):
                    analysis['issues'].append({
                        'type': 'documentation',
                        'severity': 'info',
                        'message': f'Function {node.name} missing docstring',
                        'line': node.lineno
                    })
            
            elif isinstance(node, ast.ClassDef):
                analysis['classes'].append({
                    'name': node.name,
                    'line': node.lineno,
                    'methods': len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                    'has_docstring': ast.get_docstring(node) is not None
                })
                
                # Check for missing docstring
                if not ast.get_docstring(node):
                    analysis['issues'].append({
                        'type': 'documentation',
                        'severity': 'info',
                        'message': f'Class {node.name} missing docstring',
                        'line': node.lineno
                    })
            
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(alias.name)
                else:
                    module = node.module or ''
                    for alias in node.names:
                        analysis['imports'].append(f"{module}.{alias.name}")
        
        # Check for potential security issues
        if 'eval(' in content:
            analysis['issues'].append({
                'type': 'security',
                'severity': 'error',
                'message': 'Use of eval() detected - potential security risk',
                'line': content[:content.find('eval(')].count('\n') + 1
            })
        
        if 'exec(' in content:
            analysis['issues'].append({
                'type': 'security',
                'severity': 'error',
                'message': 'Use of exec() detected - potential security risk',
                'line': content[:content.find('exec(')].count('\n') + 1
            })
        
        return analysis
        
    except Exception as e:
        return {
            'file_path': file_path,
            'error': str(e),
            'issues': [{
                'type': 'parse_error',
                'severity': 'error',
                'message': f'Failed to parse file: {e}',
                'line': 1
            }]
        }

def analyze_repository(repo_path: str) -> Dict[str, Any]:
    """Analyze the entire repository."""
    repo_path = Path(repo_path)
    
    analysis = {
        'repository_path': str(repo_path),
        'files_analyzed': 0,
        'total_lines': 0,
        'total_functions': 0,
        'total_classes': 0,
        'files': [],
        'issues_summary': Counter(),
        'severity_summary': Counter(),
        'all_issues': []
    }
    
    # Find all Python files
    python_files = list(repo_path.rglob('*.py'))
    
    for file_path in python_files:
        # Skip __pycache__ and .git directories
        if '__pycache__' in str(file_path) or '.git' in str(file_path):
            continue
            
        file_analysis = analyze_python_file(str(file_path))
        analysis['files'].append(file_analysis)
        analysis['files_analyzed'] += 1
        
        if 'lines_of_code' in file_analysis:
            analysis['total_lines'] += file_analysis['lines_of_code']
        if 'functions' in file_analysis:
            analysis['total_functions'] += len(file_analysis['functions'])
        if 'classes' in file_analysis:
            analysis['total_classes'] += len(file_analysis['classes'])
        
        # Collect issues
        for issue in file_analysis.get('issues', []):
            issue['file'] = str(file_path.relative_to(repo_path))
            analysis['all_issues'].append(issue)
            analysis['issues_summary'][issue['type']] += 1
            analysis['severity_summary'][issue['severity']] += 1
    
    return analysis

def print_analysis_report(analysis: Dict[str, Any]):
    """Print a formatted analysis report."""
    print('='*80)
    print('📊 CODEBASE ANALYSIS REPORT')
    print('='*80)
    
    print(f'\n📁 Repository: {analysis["repository_path"]}')
    print(f'📄 Files analyzed: {analysis["files_analyzed"]}')
    print(f'📏 Total lines of code: {analysis["total_lines"]:,}')
    print(f'🔧 Total functions: {analysis["total_functions"]}')
    print(f'🏗️ Total classes: {analysis["total_classes"]}')
    
    # Issues summary
    total_issues = sum(analysis['severity_summary'].values())
    print(f'\n🚨 Total issues found: {total_issues}')
    
    if analysis['severity_summary']:
        print('\n⚠️ Issues by severity:')
        for severity, count in analysis['severity_summary'].most_common():
            print(f'  {severity}: {count}')
    
    if analysis['issues_summary']:
        print('\n🔍 Issues by type:')
        for issue_type, count in analysis['issues_summary'].most_common():
            print(f'  {issue_type}: {count}')
    
    # Top issues
    if analysis['all_issues']:
        print('\n🚨 Top issues:')
        for i, issue in enumerate(analysis['all_issues'][:10]):
            print(f'  {i+1}. [{issue["severity"]}] {issue["message"]}')
            print(f'     File: {issue["file"]}:{issue["line"]}')
    
    # File breakdown
    print('\n📊 File breakdown:')
    for file_info in sorted(analysis['files'], key=lambda x: x.get('lines_of_code', 0), reverse=True)[:10]:
        if 'lines_of_code' in file_info:
            rel_path = Path(file_info['file_path']).name
            print(f'  {rel_path}: {file_info["lines_of_code"]} lines, {len(file_info.get("functions", []))} functions, {len(file_info.get("classes", []))} classes')

def main():
    """Main analysis function."""
    # Get repository path
    repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print(f'🔍 Analyzing repository: {repo_path}')
    
    # Run analysis
    analysis = analyze_repository(repo_path)
    
    # Print report
    print_analysis_report(analysis)
    
    # Save to file
    output_file = 'simple_analysis_report.json'
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f'\n📄 Full analysis saved to: {output_file}')
    print('\n' + '='*80)

if __name__ == "__main__":
    main()

