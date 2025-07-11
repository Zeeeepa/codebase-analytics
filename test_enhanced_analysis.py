#!/usr/bin/env python3
"""
Test Enhanced Analysis System
Simulates comprehensive analysis of the codebase-analytics repository
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path


def analyze_codebase_structure():
    """Analyze the current codebase structure"""
    
    print("🎯 Testing Enhanced Analysis on codebase-analytics repository...")
    print("=" * 60)
    
    # Analyze repository structure
    repo_stats = {
        'total_files': 0,
        'total_functions': 0,
        'total_classes': 0,
        'total_lines_of_code': 0,
        'languages': set(),
        'files_by_type': {},
        'issues_detected': [],
        'entry_points': [],
        'function_contexts': {}
    }
    
    # Scan backend directory
    backend_dir = Path('backend')
    if backend_dir.exists():
        for file_path in backend_dir.glob('*.py'):
            repo_stats['total_files'] += 1
            repo_stats['languages'].add('Python')
            
            # Count file type
            ext = file_path.suffix
            repo_stats['files_by_type'][ext] = repo_stats['files_by_type'].get(ext, 0) + 1
            
            # Analyze file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Count lines
                lines = content.split('\n')
                repo_stats['total_lines_of_code'] += len(lines)
                
                # Count functions and classes
                functions = re.findall(r'def\s+(\w+)', content)
                classes = re.findall(r'class\s+(\w+)', content)
                
                repo_stats['total_functions'] += len(functions)
                repo_stats['total_classes'] += len(classes)
                
                # Detect entry points
                for func in functions:
                    if any(pattern in func.lower() for pattern in ['main', 'run', 'start', 'app']):
                        repo_stats['entry_points'].append(func)
                
                # Detect issues (simplified)
                for i, line in enumerate(lines):
                    # Magic numbers
                    if re.search(r'\b\d{2,}\b', line) and not line.strip().startswith('#'):
                        repo_stats['issues_detected'].append({
                            'type': 'magic_number',
                            'severity': 'minor',
                            'file': str(file_path),
                            'line': i + 1,
                            'message': 'Magic number detected'
                        })
                    
                    # Long lines
                    if len(line) > 120:
                        repo_stats['issues_detected'].append({
                            'type': 'line_length_violation',
                            'severity': 'minor',
                            'file': str(file_path),
                            'line': i + 1,
                            'message': f'Line exceeds 120 characters ({len(line)} chars)'
                        })
                
                # Build function contexts
                for func in functions:
                    repo_stats['function_contexts'][func] = {
                        'name': func,
                        'filepath': str(file_path),
                        'is_entry_point': any(pattern in func.lower() for pattern in ['main', 'run', 'start', 'app']),
                        'complexity_estimate': content.count(func) * 2  # Simple heuristic
                    }
                    
            except Exception as e:
                print(f"Warning: Could not analyze {file_path}: {e}")
    
    # Scan frontend directory
    frontend_dir = Path('frontend')
    if frontend_dir.exists():
        for file_path in frontend_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix in ['.js', '.jsx', '.ts', '.tsx', '.vue', '.html', '.css']:
                repo_stats['total_files'] += 1
                
                # Detect language
                if file_path.suffix in ['.js', '.jsx']:
                    repo_stats['languages'].add('JavaScript')
                elif file_path.suffix in ['.ts', '.tsx']:
                    repo_stats['languages'].add('TypeScript')
                elif file_path.suffix == '.vue':
                    repo_stats['languages'].add('Vue')
                elif file_path.suffix == '.html':
                    repo_stats['languages'].add('HTML')
                elif file_path.suffix == '.css':
                    repo_stats['languages'].add('CSS')
                
                # Count file type
                ext = file_path.suffix
                repo_stats['files_by_type'][ext] = repo_stats['files_by_type'].get(ext, 0) + 1
                
                # Count lines (simplified)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        repo_stats['total_lines_of_code'] += len(content.split('\n'))
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}")
    
    return repo_stats


def calculate_health_metrics(repo_stats):
    """Calculate health metrics based on repository statistics"""
    
    # Calculate health score (0-100)
    health_score = 100
    
    # Deduct points for issues
    issue_count = len(repo_stats['issues_detected'])
    health_score -= min(issue_count * 0.5, 30)  # Max 30 points deduction
    
    # Deduct points for lack of documentation (simplified)
    if repo_stats['total_functions'] > 0:
        # Assume 70% of functions are documented (placeholder)
        doc_coverage = 70
        if doc_coverage < 80:
            health_score -= (80 - doc_coverage) * 0.5
    
    # Ensure score is within bounds
    health_score = max(0, min(100, health_score))
    
    # Determine grade
    if health_score >= 90:
        grade = "A+"
    elif health_score >= 85:
        grade = "A"
    elif health_score >= 80:
        grade = "A-"
    elif health_score >= 75:
        grade = "B+"
    elif health_score >= 70:
        grade = "B"
    elif health_score >= 65:
        grade = "B-"
    elif health_score >= 60:
        grade = "C+"
    else:
        grade = "C"
    
    # Determine risk level
    if health_score >= 80:
        risk_level = "low"
    elif health_score >= 60:
        risk_level = "medium"
    else:
        risk_level = "high"
    
    # Calculate technical debt (simplified)
    technical_debt_hours = issue_count * 0.25  # 15 minutes per issue
    
    return {
        'health_score': round(health_score, 1),
        'health_grade': grade,
        'risk_level': risk_level,
        'technical_debt_hours': round(technical_debt_hours, 1)
    }


def categorize_issues(issues):
    """Categorize issues by severity and type"""
    
    by_severity = {}
    by_type = {}
    
    for issue in issues:
        severity = issue['severity']
        issue_type = issue['type']
        
        by_severity[severity] = by_severity.get(severity, 0) + 1
        by_type[issue_type] = by_type.get(issue_type, 0) + 1
    
    return by_severity, by_type


def generate_automated_resolutions(issues):
    """Generate automated resolution suggestions"""
    
    resolutions = []
    
    for issue in issues:
        if issue['type'] == 'magic_number':
            resolutions.append({
                'issue_type': issue['type'],
                'resolution_type': 'replace_with_constant',
                'description': 'Replace magic number with named constant',
                'confidence': 0.8,
                'file_path': issue['file'],
                'line_number': issue['line'],
                'is_safe': True
            })
        elif issue['type'] == 'line_length_violation':
            resolutions.append({
                'issue_type': issue['type'],
                'resolution_type': 'line_break',
                'description': 'Break long line into multiple lines',
                'confidence': 0.9,
                'file_path': issue['file'],
                'line_number': issue['line'],
                'is_safe': True
            })
    
    return resolutions


def generate_recommendations(health_metrics, repo_stats):
    """Generate actionable recommendations"""
    
    recommendations = []
    
    # Health-based recommendations
    if health_metrics['health_score'] < 80:
        recommendations.append({
            'priority': 'high',
            'category': 'health',
            'title': 'Improve Overall Codebase Health',
            'description': f"Current health score: {health_metrics['health_score']}/100. Focus on reducing issues and improving code quality."
        })
    
    # Issue-based recommendations
    issue_count = len(repo_stats['issues_detected'])
    if issue_count > 10:
        recommendations.append({
            'priority': 'medium',
            'category': 'issues',
            'title': 'Address Code Issues',
            'description': f"{issue_count} issues detected. Review and fix to improve code quality."
        })
    
    # Documentation recommendation
    if repo_stats['total_functions'] > 10:
        recommendations.append({
            'priority': 'medium',
            'category': 'documentation',
            'title': 'Improve Documentation Coverage',
            'description': "Add docstrings to functions lacking documentation."
        })
    
    return recommendations


def main():
    """Main analysis function"""
    
    start_time = datetime.now()
    
    # Analyze codebase structure
    print("📥 Analyzing codebase structure...")
    repo_stats = analyze_codebase_structure()
    
    # Calculate health metrics
    print("🏥 Calculating health metrics...")
    health_metrics = calculate_health_metrics(repo_stats)
    
    # Categorize issues
    issues_by_severity, issues_by_type = categorize_issues(repo_stats['issues_detected'])
    
    # Generate automated resolutions
    automated_resolutions = generate_automated_resolutions(repo_stats['issues_detected'])
    
    # Generate recommendations
    recommendations = generate_recommendations(health_metrics, repo_stats)
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    # Create comprehensive report
    report = {
        'analysis_metadata': {
            'repository': 'Zeeeepa/codebase-analytics',
            'analysis_timestamp': datetime.now().isoformat(),
            'processing_time_seconds': round(processing_time, 2),
            'analyzer_version': '2.0.0-enhanced'
        },
        'repository_overview': {
            'summary': f"Codebase with {repo_stats['total_files']} files, {repo_stats['total_functions']} functions, {repo_stats['total_classes']} classes. Languages: {', '.join(sorted(repo_stats['languages']))}.",
            'total_files': repo_stats['total_files'],
            'total_functions': repo_stats['total_functions'],
            'total_classes': repo_stats['total_classes'],
            'total_lines_of_code': repo_stats['total_lines_of_code'],
            'languages': list(repo_stats['languages']),
            'files_by_type': repo_stats['files_by_type']
        },
        'issues_analysis': {
            'total_issues': len(repo_stats['issues_detected']),
            'issues_by_severity': issues_by_severity,
            'issues_by_type': issues_by_type,
            'critical_issues': [issue for issue in repo_stats['issues_detected'] if issue['severity'] == 'critical'][:10],
            'automated_resolutions': {
                'total_available': len(automated_resolutions),
                'high_confidence': len([r for r in automated_resolutions if r['confidence'] > 0.8]),
                'safe_to_apply': len([r for r in automated_resolutions if r['is_safe']]),
                'resolutions': automated_resolutions[:20]
            }
        },
        'function_analysis': {
            'total_functions': len(repo_stats['function_contexts']),
            'entry_points': repo_stats['entry_points'],
            'dead_functions': [],  # Placeholder - would need call graph analysis
            'most_important_functions': sorted(
                [{'name': name, 'complexity': ctx['complexity_estimate'], 'is_entry_point': ctx['is_entry_point']} 
                 for name, ctx in repo_stats['function_contexts'].items()],
                key=lambda x: x['complexity'], reverse=True
            )[:10],
            'function_contexts': {
                name: ctx for name, ctx in list(repo_stats['function_contexts'].items())[:20]  # Top 20
            }
        },
        'quality_metrics': {
            'halstead_metrics': {
                'program_length': repo_stats['total_lines_of_code'],
                'vocabulary': repo_stats['total_functions'] + repo_stats['total_classes'],
                'estimated_difficulty': 2.5  # Placeholder
            },
            'complexity_metrics': {
                'average_cyclomatic_complexity': 3.2,  # Placeholder
                'max_complexity': 8,  # Placeholder
                'functions_over_threshold': 2  # Placeholder
            },
            'maintainability_metrics': {
                'maintainability_index': 75,  # Placeholder
                'documentation_coverage': 70,  # Placeholder
                'code_duplication': 5  # Placeholder
            }
        },
        'health_assessment': health_metrics,
        'health_dashboard': {
            'health_score': health_metrics['health_score'],
            'health_grade': health_metrics['health_grade'],
            'risk_level': health_metrics['risk_level'],
            'technical_debt_hours': health_metrics['technical_debt_hours'],
            'issues_summary': {
                'total_issues': len(repo_stats['issues_detected']),
                'by_severity': issues_by_severity,
                'by_type': issues_by_type,
                'automated_fixes_available': len(automated_resolutions)
            },
            'function_metrics': {
                'total_functions': len(repo_stats['function_contexts']),
                'entry_points': len(repo_stats['entry_points']),
                'dead_functions': 0,
                'most_important': [
                    {'name': name, 'is_entry_point': ctx['is_entry_point']} 
                    for name, ctx in list(repo_stats['function_contexts'].items())[:5]
                ]
            },
            'quality_indicators': {
                'maintainability_index': 75,
                'documentation_coverage': 70,
                'average_complexity': 3.2,
                'code_duplication': 5
            },
            'recommendations': recommendations
        }
    }
    
    # Save detailed report
    with open('test_analysis_results.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Display summary
    print("\n📊 COMPREHENSIVE ANALYSIS RESULTS:")
    print("=" * 60)
    print(f"Repository: {report['repository_overview']['summary']}")
    print(f"Health Score: {health_metrics['health_score']}/100 (Grade: {health_metrics['health_grade']})")
    print(f"Risk Level: {health_metrics['risk_level'].upper()}")
    print(f"Total Issues: {len(repo_stats['issues_detected'])}")
    print(f"Automated Fixes Available: {len(automated_resolutions)}")
    print(f"Technical Debt: {health_metrics['technical_debt_hours']} hours")
    print(f"Entry Points: {len(repo_stats['entry_points'])}")
    print(f"Processing Time: {processing_time:.2f} seconds")
    
    print("\n🔍 ISSUES BREAKDOWN:")
    for severity, count in issues_by_severity.items():
        print(f"  • {severity.title()}: {count}")
    
    print("\n🎯 TOP ENTRY POINTS:")
    for entry_point in repo_stats['entry_points'][:5]:
        print(f"  • {entry_point}")
    
    print("\n💡 KEY RECOMMENDATIONS:")
    for rec in recommendations[:3]:
        print(f"  • [{rec['priority'].upper()}] {rec['title']}")
        print(f"    {rec['description']}")
    
    print(f"\n✅ Analysis completed successfully!")
    print(f"📄 Detailed report saved to test_analysis_results.json")
    
    return report


if __name__ == "__main__":
    main()

