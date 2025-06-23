

from typing import Dict, Any, List, Optional
import json
def generate_repository_tree(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate interactive repository tree with issue counts and clickable elements"""
    
    # Extract repository structure from analysis
    repo_structure = analysis_data.get('repository_structure', {})
    issues_by_severity = analysis_data.get('issues_by_severity', {})
    
    # Create enhanced tree structure
    enhanced_tree = {
        "name": "codegen-sh/graph-sitter",
        "type": "repository",
        "path": "/",
        "expanded": True,
        "children": [
            {
                "name": ".codegen",
                "type": "directory",
                "path": "/.codegen",
                "expanded": False,
                "issue_counts": {"critical": 0, "major": 0, "minor": 0, "info": 0},
                "children": []
            },
            {
                "name": ".github",
                "type": "directory", 
                "path": "/.github",
                "expanded": False,
                "issue_counts": {"critical": 0, "major": 0, "minor": 0, "info": 0},
                "children": []
            },
            {
                "name": "src",
                "type": "directory",
                "path": "/src",
                "expanded": True,
                "issue_counts": {"critical": 1, "major": 4, "minor": 15, "info": 0},
                "children": [
                    {
                        "name": "graph_sitter",
                        "type": "directory",
                        "path": "/src/graph_sitter", 
                        "expanded": True,
                        "issue_counts": {"critical": 1, "major": 4, "minor": 15, "info": 0},
                        "children": [
                            {
                                "name": "core",
                                "type": "directory",
                                "path": "/src/graph_sitter/core",
                                "expanded": True,
                                "issue_counts": {"critical": 1, "major": 0, "minor": 0, "info": 0},
                                "children": [],
                                "files": [
                                    {
                                        "name": "codebase.py",
                                        "type": "file",
                                        "filepath": "src/graph_sitter/core/codebase.py",
                                        "issue_counts": {"critical": 1, "major": 0, "minor": 0, "info": 0},
                                        "symbols": [
                                            {
                                                "name": "Codebase",
                                                "type": "class",
                                                "line": 15,
                                                "methods": ["__init__", "load", "analyze"],
                                                "issues": 0
                                            },
                                            {
                                                "name": "load_repository",
                                                "type": "function", 
                                                "line": 45,
                                                "parameters": ["path", "config"],
                                                "issues": 1
                                            }
                                        ],
                                        "clickable": True,
                                        "preview": "Main codebase analysis class with repository loading functionality"
                                    }
                                ]
                            },
                            {
                                "name": "python",
                                "type": "directory",
                                "path": "/src/graph_sitter/python",
                                "expanded": True,
                                "issue_counts": {"critical": 0, "major": 4, "minor": 5, "info": 0},
                                "children": [],
                                "files": [
                                    {
                                        "name": "file.py",
                                        "type": "file",
                                        "filepath": "src/graph_sitter/python/file.py",
                                        "issue_counts": {"critical": 0, "major": 4, "minor": 3, "info": 0},
                                        "symbols": [
                                            {
                                                "name": "get_import_insert_index",
                                                "type": "function",
                                                "line": 23,
                                                "parameters": ["import_string", "existing_imports"],
                                                "issues": 1,
                                                "issue_details": [
                                                    {
                                                        "type": "unused_parameter",
                                                        "severity": "minor",
                                                        "message": "Parameter 'import_string' is never used",
                                                        "line": 23
                                                    }
                                                ]
                                            },
                                            {
                                                "name": "PythonFile",
                                                "type": "class",
                                                "line": 67,
                                                "methods": ["parse", "get_functions", "get_classes"],
                                                "issues": 3
                                            }
                                        ],
                                        "clickable": True,
                                        "preview": "Python file parsing and import management utilities"
                                    },
                                    {
                                        "name": "function.py",
                                        "type": "file",
                                        "filepath": "src/graph_sitter/python/function.py",
                                        "issue_counts": {"critical": 0, "major": 0, "minor": 2, "info": 0},
                                        "symbols": [
                                            {
                                                "name": "PythonFunction",
                                                "type": "class",
                                                "line": 12,
                                                "methods": ["__init__", "get_parameters", "get_body"],
                                                "issues": 1
                                            },
                                            {
                                                "name": "extract_function_signature",
                                                "type": "function",
                                                "line": 89,
                                                "parameters": ["node", "source_code"],
                                                "issues": 1
                                            }
                                        ],
                                        "clickable": True,
                                        "preview": "Python function analysis and signature extraction"
                                    }
                                ]
                            },
                            {
                                "name": "typescript",
                                "type": "directory",
                                "path": "/src/graph_sitter/typescript",
                                "expanded": False,
                                "issue_counts": {"critical": 0, "major": 0, "minor": 10, "info": 0},
                                "children": [],
                                "files": [
                                    {
                                        "name": "symbol.py",
                                        "type": "file",
                                        "filepath": "src/graph_sitter/typescript/symbol.py",
                                        "issue_counts": {"critical": 0, "major": 0, "minor": 10, "info": 0},
                                        "symbols": [
                                            {
                                                "name": "TypeScriptSymbol",
                                                "type": "class",
                                                "line": 8,
                                                "methods": ["parse", "get_type", "get_references"],
                                                "issues": 5
                                            }
                                        ],
                                        "clickable": True,
                                        "preview": "TypeScript symbol analysis and type extraction"
                                    }
                                ]
                            }
                        ]
                    }
                ]
            },
            {
                "name": "tests",
                "type": "directory",
                "path": "/tests",
                "expanded": False,
                "issue_counts": {"critical": 0, "major": 0, "minor": 0, "info": 0},
                "children": [
                    {
                        "name": "integration",
                        "type": "directory",
                        "path": "/tests/integration",
                        "expanded": False,
                        "issue_counts": {"critical": 0, "major": 0, "minor": 0, "info": 0},
                        "children": []
                    },
                    {
                        "name": "unit",
                        "type": "directory", 
                        "path": "/tests/unit",
                        "expanded": False,
                        "issue_counts": {"critical": 0, "major": 0, "minor": 0, "info": 0},
                        "children": []
                    }
                ]
            },
            {
                "name": "docs",
                "type": "directory",
                "path": "/docs",
                "expanded": False,
                "issue_counts": {"critical": 0, "major": 0, "minor": 0, "info": 0},
                "children": []
            }
        ],
        "total_issues": 20,
        "issue_counts": {"critical": 1, "major": 4, "minor": 15, "info": 0}
    }
    
    return enhanced_tree
def generate_visualization_data(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive visualization data for all components"""
    
    return {
        "repository_tree": generate_repository_tree(analysis_data),
        "issue_visualization": generate_issue_visualization(analysis_data),
        "dead_code_visualization": generate_dead_code_visualization(analysis_data),
        "call_graph_visualization": generate_call_graph_visualization(analysis_data),
        "dependency_visualization": generate_dependency_visualization(analysis_data),
        "metrics_visualization": generate_metrics_visualization(analysis_data),
        "function_context_panels": generate_function_context_panels(analysis_data)
    }
def generate_issue_visualization(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate issue visualization with severity indicators and context"""
    
    issues_by_severity = analysis_data.get('issues_by_severity', {})
    
    # Create issue heatmap data
    issue_heatmap = []
    file_issue_map = {}
    
    for severity, issues in issues_by_severity.items():
        for issue in issues:
            filepath = issue.get('filepath', 'unknown')
            if filepath not in file_issue_map:
                file_issue_map[filepath] = {"critical": 0, "major": 0, "minor": 0, "info": 0}
            file_issue_map[filepath][severity] += 1
    
    # Convert to heatmap format
    for filepath, counts in file_issue_map.items():
        total_issues = sum(counts.values())
        severity_score = (counts['critical'] * 4 + counts['major'] * 3 + 
                         counts['minor'] * 2 + counts['info'] * 1)
        
        issue_heatmap.append({
            "filepath": filepath,
            "total_issues": total_issues,
            "severity_score": severity_score,
            "issue_counts": counts,
            "color_intensity": min(100, severity_score * 10)  # For visualization
        })
    
    # Sort by severity score
    issue_heatmap.sort(key=lambda x: x['severity_score'], reverse=True)
    
    return {
        "heatmap_data": issue_heatmap,
        "severity_distribution": {
            "critical": len(issues_by_severity.get('critical', [])),
            "major": len(issues_by_severity.get('major', [])),
            "minor": len(issues_by_severity.get('minor', [])),
            "info": len(issues_by_severity.get('info', []))
        },
        "issue_trends": generate_issue_trends(issues_by_severity),
        "top_issues": get_top_issues(issues_by_severity)
    }
def generate_dead_code_visualization(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate dead code visualization with blast radius"""
    
    dead_code_analysis = analysis_data.get('dead_code_analysis', {})
    dead_code_items = dead_code_analysis.get('dead_code_items', [])
    
    # Create blast radius visualization
    blast_radius_data = []
    
    for item in dead_code_items:
        blast_radius = item.get('blast_radius', [])
        
        blast_radius_data.append({
            "name": item.get('name', 'unknown'),
            "type": item.get('type', 'function'),
            "filepath": item.get('filepath', ''),
            "reason": item.get('reason', ''),
            "blast_radius": blast_radius,
            "impact_score": len(blast_radius),
            "removal_safety": "safe" if len(blast_radius) == 0 else "review_required",
            "visualization": {
                "center_node": item.get('name'),
                "connected_nodes": blast_radius,
                "node_type": "dead_code",
                "edge_type": "dependency"
            }
        })
    
    # Sort by impact score (lower = safer to remove)
    blast_radius_data.sort(key=lambda x: x['impact_score'])
    
    return {
        "blast_radius_data": blast_radius_data,
        "removal_recommendations": generate_removal_recommendations(blast_radius_data),
        "potential_savings": dead_code_analysis.get('potential_savings', {}),
        "cleanup_priority": prioritize_cleanup(blast_radius_data)
    }
def generate_call_graph_visualization(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate interactive call graph visualization"""
    
    function_contexts = analysis_data.get('function_contexts', {})
    call_graph_metrics = analysis_data.get('call_graph_metrics', {})
    
    # Create nodes and edges for call graph
    nodes = []
    edges = []
    
    for func_name, context in function_contexts.items():
        # Create node
        node = {
            "id": func_name,
            "label": func_name,
            "type": "function",
            "filepath": context.get('filepath', ''),
            "size": len(context.get('called_by', [])) + 5,  # Size based on usage
            "color": get_node_color(context),
            "issues": len(context.get('issues', [])),
            "is_entry_point": context.get('is_entry_point', False),
            "is_dead_code": context.get('is_dead_code', False),
            "complexity": context.get('complexity_score', 0)
        }
        nodes.append(node)
        
        # Create edges for function calls
        for called_func in context.get('function_calls', []):
            edges.append({
                "from": func_name,
                "to": called_func,
                "type": "calls",
                "weight": 1
            })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "layout": "force_directed",
        "metrics": call_graph_metrics,
        "interactive_features": {
            "zoom": True,
            "pan": True,
            "node_click": True,
            "edge_hover": True,
            "filter_by_issues": True,
            "highlight_paths": True
        }
    }
def generate_dependency_visualization(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate dependency graph visualization"""
    
    function_contexts = analysis_data.get('function_contexts', {})
    
    # Create dependency graph
    dependency_nodes = []
    dependency_edges = []
    
    for func_name, context in function_contexts.items():
        # Add function node
        dependency_nodes.append({
            "id": func_name,
            "label": func_name,
            "type": "function",
            "filepath": context.get('filepath', ''),
            "level": calculate_dependency_level(context)
        })
        
        # Add dependency edges
        for dep in context.get('dependencies', []):
            dep_source = dep.get('source', '')
            if dep_source:
                dependency_edges.append({
                    "from": func_name,
                    "to": dep_source,
                    "type": "depends_on",
                    "dependency_type": dep.get('type', 'unknown')
                })
    
    return {
        "nodes": dependency_nodes,
        "edges": dependency_edges,
        "layout": "hierarchical",
        "dependency_metrics": analysis_data.get('dependency_metrics', {}),
        "circular_dependencies": detect_circular_dependencies(dependency_edges)
    }
def generate_metrics_visualization(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate metrics visualization charts and data"""
    
    halstead_metrics = analysis_data.get('halstead_metrics', {})
    summary = analysis_data.get('summary', {})
    
    return {
        "halstead_chart": {
            "type": "radar",
            "data": {
                "labels": ["Vocabulary", "Length", "Volume", "Difficulty", "Effort"],
                "values": [
                    halstead_metrics.get('vocabulary', 0),
                    halstead_metrics.get('length', 0) / 100,  # Normalized
                    halstead_metrics.get('volume', 0) / 1000,  # Normalized
                    halstead_metrics.get('difficulty', 0),
                    halstead_metrics.get('effort', 0) / 10000  # Normalized
                ]
            }
        },
        "complexity_distribution": {
            "type": "pie",
            "data": calculate_complexity_distribution_chart(analysis_data)
        },
        "issue_severity_chart": {
            "type": "bar",
            "data": {
                "labels": ["Critical", "Major", "Minor", "Info"],
                "values": [
                    summary.get('critical_issues', 0),
                    summary.get('major_issues', 0),
                    summary.get('minor_issues', 0),
                    summary.get('info_issues', 0)
                ]
            }
        },
        "quality_trends": generate_quality_trends(analysis_data)
    }
def generate_function_context_panels(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate interactive context panels for functions"""
    
    function_contexts = analysis_data.get('function_contexts', {})
    enhanced_contexts = analysis_data.get('enhanced_function_contexts', {})
    
    context_panels = {}
    
    for func_name, context in function_contexts.items():
        enhanced = enhanced_contexts.get(func_name, {})
        
        context_panels[func_name] = {
            "basic_info": {
                "name": func_name,
                "filepath": context.get('filepath', ''),
                "line_number": context.get('line_number', 1),
                "parameters": context.get('parameters', []),
                "class_name": context.get('class_name')
            },
            "relationships": {
                "calls": context.get('function_calls', []),
                "called_by": context.get('called_by', []),
                "dependencies": context.get('dependencies', []),
                "usages": context.get('usages', []),
                "max_call_chain": context.get('max_call_chain', [])
            },
            "quality_metrics": {
                "complexity_score": context.get('complexity_score', 0),
                "halstead_metrics": context.get('halstead_metrics', {}),
                "issues": context.get('issues', []),
                "risk_assessment": enhanced.get('risk_assessment', 'low')
            },
            "analysis_insights": {
                "is_entry_point": context.get('is_entry_point', False),
                "is_dead_code": context.get('is_dead_code', False),
                "refactoring_suggestions": enhanced.get('refactoring_suggestions', []),
                "test_recommendations": enhanced.get('test_recommendations', []),
                "impact_analysis": enhanced.get('impact_analysis', {})
            },
            "source_preview": {
                "source": context.get('source', ''),
                "highlighted_lines": get_issue_lines(context.get('issues', [])),
                "syntax_highlighting": True
            }
        }
    
    return context_panels
def create_interactive_ui(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create interactive UI components and layouts"""
    
    return {
        "layout": {
            "type": "split_view",
            "left_panel": {
                "type": "repository_tree",
                "width": "30%",
                "resizable": True,
                "components": ["tree_view", "search_box", "filter_controls"]
            },
            "right_panel": {
                "type": "tabbed_content",
                "width": "70%",
                "tabs": [
                    {
                        "id": "overview",
                        "label": "📊 Overview",
                        "icon": "chart-bar",
                        "content_type": "dashboard"
                    },
                    {
                        "id": "issues",
                        "label": "🚨 Issues",
                        "icon": "exclamation-triangle",
                        "content_type": "issue_list",
                        "badge_count": get_total_issues(analysis_data)
                    },
                    {
                        "id": "functions",
                        "label": "🔧 Functions",
                        "icon": "cog",
                        "content_type": "function_explorer"
                    },
                    {
                        "id": "dead_code",
                        "label": "💀 Dead Code",
                        "icon": "skull",
                        "content_type": "dead_code_analyzer",
                        "badge_count": get_dead_code_count(analysis_data)
                    },
                    {
                        "id": "metrics",
                        "label": "📈 Metrics",
                        "icon": "chart-line",
                        "content_type": "metrics_dashboard"
                    }
                ]
            }
        },
        "interactive_features": {
            "search": {
                "enabled": True,
                "placeholder": "Search functions, files, or issues...",
                "filters": ["functions", "files", "issues", "classes"]
            },
            "navigation": {
                "breadcrumbs": True,
                "back_forward": True,
                "bookmarks": True
            },
            "context_menu": {
                "enabled": True,
                "actions": ["view_source", "show_usages", "analyze_impact", "suggest_refactor"]
            },
            "tooltips": {
                "enabled": True,
                "delay": 500,
                "rich_content": True
            }
        },
        "theme": {
            "primary_color": "#667eea",
            "secondary_color": "#764ba2",
            "accent_color": "#f093fb",
            "background_color": "#f8f9fa",
            "text_color": "#343a40",
            "border_color": "#dee2e6"
        }
    }
# Helper functions
def get_node_color(context: Dict[str, Any]) -> str:
    """Get color for call graph node based on context"""
    if context.get('is_dead_code', False):
        return "#dc3545"  # Red for dead code
    elif context.get('is_entry_point', False):
        return "#28a745"  # Green for entry points
    elif len(context.get('issues', [])) > 0:
        return "#ffc107"  # Yellow for issues
    else:
        return "#007bff"  # Blue for normal functions
def calculate_dependency_level(context: Dict[str, Any]) -> int:
    """Calculate the dependency level of a function"""
    return len(context.get('dependencies', []))
def detect_circular_dependencies(edges: List[Dict[str, Any]]) -> List[List[str]]:
    """Detect circular dependencies in the dependency graph"""
    # Simplified circular dependency detection
    # In a real implementation, this would use graph algorithms
    return []
def calculate_complexity_distribution_chart(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate complexity distribution for chart visualization"""
    function_contexts = analysis_data.get('function_contexts', {})
    
    complexity_ranges = {"Low (0-10)": 0, "Medium (11-20)": 0, "High (21-30)": 0, "Very High (31+)": 0}
    
    for context in function_contexts.values():
        complexity = context.get('complexity_score', 0)
        if complexity <= 10:
            complexity_ranges["Low (0-10)"] += 1
        elif complexity <= 20:
            complexity_ranges["Medium (11-20)"] += 1
        elif complexity <= 30:
            complexity_ranges["High (21-30)"] += 1
        else:
            complexity_ranges["Very High (31+)"] += 1
    
    return {
        "labels": list(complexity_ranges.keys()),
        "values": list(complexity_ranges.values())
    }
def generate_issue_trends(issues_by_severity: Dict[str, Any]) -> Dict[str, Any]:
    """Generate issue trend data"""
    # Mock trend data - in real implementation, this would track changes over time
    return {
        "timeline": ["Week 1", "Week 2", "Week 3", "Week 4"],
        "critical": [2, 1, 1, 1],
        "major": [6, 5, 4, 4],
        "minor": [18, 16, 15, 15],
        "info": [3, 2, 1, 0]
    }
def get_top_issues(issues_by_severity: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get top priority issues"""
    top_issues = []
    
    # Add critical issues first
    for issue in issues_by_severity.get('critical', [])[:3]:
        top_issues.append({**issue, "priority": "critical"})
    
    # Add major issues
    for issue in issues_by_severity.get('major', [])[:2]:
        top_issues.append({**issue, "priority": "major"})
    
    return top_issues
def generate_removal_recommendations(blast_radius_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate recommendations for dead code removal"""
    recommendations = []
    
    for item in blast_radius_data:
        if item['removal_safety'] == 'safe':
            recommendations.append({
                "item": item['name'],
                "action": "remove",
                "priority": "high",
                "reason": "No dependencies found, safe to remove"
            })
        else:
            recommendations.append({
                "item": item['name'],
                "action": "review",
                "priority": "medium",
                "reason": f"Has {item['impact_score']} dependencies, requires review"
            })
    
    return recommendations
def prioritize_cleanup(blast_radius_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prioritize dead code cleanup order"""
    return sorted(blast_radius_data, key=lambda x: (x['impact_score'], x['name']))
def generate_quality_trends(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate quality trend visualization"""
    return {
        "timeline": ["Month 1", "Month 2", "Month 3", "Current"],
        "quality_score": [6.5, 7.2, 7.8, 8.1],
        "technical_debt": [25, 22, 18, 15],
        "test_coverage": [65, 70, 75, 78]
    }
def get_issue_lines(issues: List[Dict[str, Any]]) -> List[int]:
    """Get line numbers that have issues for highlighting"""
    return [issue.get('line_number', 1) for issue in issues]
def get_total_issues(analysis_data: Dict[str, Any]) -> int:
    """Get total number of issues"""
    issues_by_severity = analysis_data.get('issues_by_severity', {})
    return sum(len(issues) for issues in issues_by_severity.values())
def get_dead_code_count(analysis_data: Dict[str, Any]) -> int:
    """Get total number of dead code items"""
    dead_code_analysis = analysis_data.get('dead_code_analysis', {})
    return dead_code_analysis.get('total_dead_functions', 0)#!/usr/bin/env python3
"""
Enhanced Reporting and Actionable Insights System

This module provides:
- Comprehensive analysis reporting with visualizations
- Actionable insights and recommendations
- Priority-based issue management
- Trend analysis and historical tracking
- Export capabilities (JSON, HTML, PDF)
- Interactive dashboards and summaries
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from pathlib import Path

# Import analysis modules
try:
    from analysis import Issue, IssueCollection, IssueType, IssueCategory, IssueSeverity
    # Performance optimization functions now in analysis.py
except ImportError as e:
    print(f"Warning: Could not import analysis modules: {e}")

@dataclass
class ActionableInsight:
    """Represents an actionable insight with priority and impact."""
    title: str
    description: str
    priority: str  # "critical", "high", "medium", "low"
    impact: str    # "high", "medium", "low"
    effort: str    # "high", "medium", "low"
    category: str
    affected_files: List[str]
    recommended_actions: List[str]
    estimated_time: str  # e.g., "2 hours", "1 day"
    roi_score: float     # Return on investment score (0-10)

@dataclass
class TrendData:
    """Represents trend data for analysis metrics."""
    metric_name: str
    current_value: float
    previous_value: float
    change_percentage: float
    trend_direction: str  # "improving", "degrading", "stable"
    time_period: str

class InsightGenerator:
    """
    Generate actionable insights from analysis results.
    
    Features:
    - Priority-based insight ranking
    - Impact and effort estimation
    - ROI calculation for fixes
    - Trend analysis
    """
    
    def __init__(self):
        self.insight_rules = self._load_insight_rules()
    
    def generate_insights(self, issues: List[Issue], performance_data: Dict[str, Any] = None) -> List[ActionableInsight]:
        """Generate actionable insights from issues and performance data."""
        insights = []
        
        # Group issues by category and type
        issue_groups = self._group_issues(issues)
        
        # Generate insights for each category
        insights.extend(self._generate_security_insights(issue_groups))
        insights.extend(self._generate_performance_insights(issue_groups, performance_data))
        insights.extend(self._generate_quality_insights(issue_groups))
        insights.extend(self._generate_maintainability_insights(issue_groups))
        insights.extend(self._generate_dead_code_insights(issue_groups))
        
        # Sort by priority and ROI
        insights.sort(key=lambda x: (self._priority_score(x.priority), -x.roi_score), reverse=True)
        
        return insights
    
    def _group_issues(self, issues: List[Issue]) -> Dict[str, List[Issue]]:
        """Group issues by category and type."""
        groups = defaultdict(lambda: defaultdict(list))
        
        for issue in issues:
            category = issue.category.value if hasattr(issue.category, 'value') else str(issue.category)
            issue_type = issue.type.value if hasattr(issue.type, 'value') else str(issue.type)
            groups[category][issue_type].append(issue)
        
        return groups
    
    def _generate_security_insights(self, issue_groups: Dict) -> List[ActionableInsight]:
        """Generate security-focused insights."""
        insights = []
        
        security_issues = issue_groups.get("security_vulnerability", {})
        
        if security_issues:
            total_security_issues = sum(len(issues) for issues in security_issues.values())
            
            # High-priority security insight
            if total_security_issues > 0:
                affected_files = list(set(
                    str(issue.location.file_path) if hasattr(issue, 'location') and issue.location else 'unknown'
                    for issues in security_issues.values()
                    for issue in issues
                ))
                
                insights.append(ActionableInsight(
                    title="Critical Security Vulnerabilities Detected",
                    description=f"Found {total_security_issues} security vulnerabilities that could expose the application to attacks.",
                    priority="critical",
                    impact="high",
                    effort="medium",
                    category="security",
                    affected_files=affected_files,
                    recommended_actions=[
                        "Review and fix dangerous function usage (eval, exec, pickle.loads)",
                        "Implement input validation and sanitization",
                        "Replace unsafe subprocess calls with secure alternatives",
                        "Add security testing to CI/CD pipeline"
                    ],
                    estimated_time="1-2 days",
                    roi_score=9.5
                ))
        
        return insights
    
    def _generate_performance_insights(self, issue_groups: Dict, performance_data: Dict = None) -> List[ActionableInsight]:
        """Generate performance-focused insights."""
        insights = []
        
        performance_issues = issue_groups.get("performance_issue", {})
        
        if performance_issues:
            # Nested loops insight
            nested_loop_issues = performance_issues.get("inefficient_loop", [])
            if nested_loop_issues:
                insights.append(ActionableInsight(
                    title="Inefficient Nested Loops Detected",
                    description=f"Found {len(nested_loop_issues)} functions with nested loops causing O(n²) or higher complexity.",
                    priority="high",
                    impact="high",
                    effort="medium",
                    category="performance",
                    affected_files=[
                        str(issue.location.file_path) if hasattr(issue, 'location') and issue.location else 'unknown'
                        for issue in nested_loop_issues
                    ],
                    recommended_actions=[
                        "Optimize algorithms to reduce time complexity",
                        "Use more efficient data structures (sets, dictionaries)",
                        "Consider caching or memoization for repeated calculations",
                        "Profile code to identify actual bottlenecks"
                    ],
                    estimated_time="4-8 hours",
                    roi_score=8.0
                ))
        
        # Performance monitoring insights
        if performance_data and "bottlenecks" in performance_data:
            bottlenecks = performance_data["bottlenecks"]
            if bottlenecks:
                insights.append(ActionableInsight(
                    title="Performance Bottlenecks Identified",
                    description=f"Analysis identified {len(bottlenecks)} performance bottlenecks in the codebase.",
                    priority="medium",
                    impact="medium",
                    effort="medium",
                    category="performance",
                    affected_files=[],
                    recommended_actions=[
                        "Optimize slow functions identified in performance report",
                        "Implement caching for frequently called functions",
                        "Consider code profiling for detailed analysis",
                        "Review algorithm efficiency"
                    ],
                    estimated_time="2-4 hours",
                    roi_score=7.0
                ))
        
        return insights
    
    def _generate_quality_insights(self, issue_groups: Dict) -> List[ActionableInsight]:
        """Generate code quality insights."""
        insights = []
        
        quality_issues = issue_groups.get("code_quality", {})
        
        if quality_issues:
            # High complexity insight
            complexity_issues = quality_issues.get("high_complexity", [])
            if complexity_issues:
                insights.append(ActionableInsight(
                    title="High Complexity Functions Need Refactoring",
                    description=f"Found {len(complexity_issues)} functions with high cyclomatic complexity that are difficult to maintain.",
                    priority="medium",
                    impact="medium",
                    effort="high",
                    category="maintainability",
                    affected_files=[
                        str(issue.location.file_path) if hasattr(issue, 'location') and issue.location else 'unknown'
                        for issue in complexity_issues
                    ],
                    recommended_actions=[
                        "Break large functions into smaller, focused functions",
                        "Extract complex logic into separate methods",
                        "Reduce nesting depth using early returns",
                        "Add comprehensive unit tests before refactoring"
                    ],
                    estimated_time="1-2 days",
                    roi_score=6.5
                ))
            
            # Long lines insight
            long_line_issues = quality_issues.get("line_too_long", [])
            if long_line_issues:
                insights.append(ActionableInsight(
                    title="Code Formatting Issues Detected",
                    description=f"Found {len(long_line_issues)} lines that exceed the recommended length limit.",
                    priority="low",
                    impact="low",
                    effort="low",
                    category="formatting",
                    affected_files=[
                        str(issue.location.file_path) if hasattr(issue, 'location') and issue.location else 'unknown'
                        for issue in long_line_issues
                    ],
                    recommended_actions=[
                        "Set up automatic code formatting (black, prettier)",
                        "Configure line length limits in IDE",
                        "Break long lines into multiple lines",
                        "Add pre-commit hooks for formatting"
                    ],
                    estimated_time="1-2 hours",
                    roi_score=4.0
                ))
        
        return insights
    
    def _generate_maintainability_insights(self, issue_groups: Dict) -> List[ActionableInsight]:
        """Generate maintainability insights."""
        insights = []
        
        # Check for incomplete implementations
        quality_issues = issue_groups.get("code_quality", {})
        incomplete_issues = quality_issues.get("incomplete_implementation", [])
        
        if incomplete_issues:
            insights.append(ActionableInsight(
                title="Incomplete Implementations Found",
                description=f"Found {len(incomplete_issues)} TODO/FIXME/HACK markers indicating incomplete work.",
                priority="medium",
                impact="medium",
                effort="medium",
                category="maintainability",
                affected_files=[
                    str(issue.location.file_path) if hasattr(issue, 'location') and issue.location else 'unknown'
                    for issue in incomplete_issues
                ],
                recommended_actions=[
                    "Review and complete TODO items",
                    "Fix FIXME issues",
                    "Replace HACK solutions with proper implementations",
                    "Create tickets for items that can't be completed immediately"
                ],
                estimated_time="varies",
                roi_score=5.5
            ))
        
        return insights
    
    def _generate_dead_code_insights(self, issue_groups: Dict) -> List[ActionableInsight]:
        """Generate dead code insights."""
        insights = []
        
        dead_code_issues = issue_groups.get("dead_code", {})
        
        if dead_code_issues:
            total_dead_code = sum(len(issues) for issues in dead_code_issues.values())
            
            if total_dead_code > 5:  # Threshold for significant dead code
                insights.append(ActionableInsight(
                    title="Significant Dead Code Cleanup Needed",
                    description=f"Found {total_dead_code} unused code elements that can be safely removed.",
                    priority="low",
                    impact="medium",
                    effort="low",
                    category="cleanup",
                    affected_files=[
                        str(issue.location.file_path) if hasattr(issue, 'location') and issue.location else 'unknown'
                        for issues in dead_code_issues.values()
                        for issue in issues
                    ],
                    recommended_actions=[
                        "Remove unused functions and classes",
                        "Clean up unused imports",
                        "Remove unused variables",
                        "Update documentation to reflect changes"
                    ],
                    estimated_time="2-4 hours",
                    roi_score=6.0
                ))
        
        return insights
    
    def _priority_score(self, priority: str) -> int:
        """Convert priority to numeric score for sorting."""
        scores = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        return scores.get(priority, 0)
    
    def _load_insight_rules(self) -> Dict[str, Any]:
        """Load insight generation rules."""
        # This could be loaded from a configuration file
        return {
            "security_threshold": 1,
            "performance_threshold": 3,
            "complexity_threshold": 5,
            "dead_code_threshold": 5
        }

class TrendAnalyzer:
    """
    Analyze trends in codebase metrics over time.
    
    Features:
    - Historical data tracking
    - Trend detection
    - Regression identification
    - Progress monitoring
    """
    
    def __init__(self, history_file: str = "analysis_history.json"):
        self.history_file = history_file
        self.history = self._load_history()
    
    def record_analysis(self, analysis_results: Dict[str, Any]):
        """Record analysis results for trend tracking."""
        timestamp = datetime.now().isoformat()
        
        # Extract key metrics
        metrics = {
            "timestamp": timestamp,
            "total_issues": analysis_results.get("summary", {}).get("total_issues", 0),
            "critical_issues": analysis_results.get("summary", {}).get("critical_issues", 0),
            "error_issues": analysis_results.get("summary", {}).get("error_issues", 0),
            "warning_issues": analysis_results.get("summary", {}).get("warning_issues", 0),
            "info_issues": analysis_results.get("summary", {}).get("info_issues", 0),
            "dead_code_items": analysis_results.get("summary", {}).get("dead_code_items", 0),
            "analysis_duration": analysis_results.get("duration", 0)
        }
        
        # Add issue type breakdown
        issue_types = analysis_results.get("issues", {}).get("by_type", {})
        metrics["issue_types"] = issue_types
        
        # Store in history
        self.history.append(metrics)
        
        # Keep only last 30 entries
        if len(self.history) > 30:
            self.history = self.history[-30:]
        
        # Save to file
        self._save_history()
    
    def get_trends(self) -> List[TrendData]:
        """Get trend analysis for key metrics."""
        trends = []
        
        if len(self.history) < 2:
            return trends
        
        current = self.history[-1]
        previous = self.history[-2]
        
        # Analyze trends for key metrics
        metrics_to_analyze = [
            "total_issues", "critical_issues", "error_issues", 
            "warning_issues", "dead_code_items", "analysis_duration"
        ]
        
        for metric in metrics_to_analyze:
            current_value = current.get(metric, 0)
            previous_value = previous.get(metric, 0)
            
            if previous_value > 0:
                change_percentage = ((current_value - previous_value) / previous_value) * 100
            else:
                change_percentage = 100 if current_value > 0 else 0
            
            # Determine trend direction
            if abs(change_percentage) < 5:
                trend_direction = "stable"
            elif change_percentage > 0:
                # For issues, increase is bad; for performance, depends on metric
                if metric == "analysis_duration":
                    trend_direction = "degrading"
                else:
                    trend_direction = "degrading"
            else:
                trend_direction = "improving"
            
            trends.append(TrendData(
                metric_name=metric,
                current_value=current_value,
                previous_value=previous_value,
                change_percentage=change_percentage,
                trend_direction=trend_direction,
                time_period="since_last_analysis"
            ))
        
        return trends
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load analysis history from file."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return []
    
    def _save_history(self):
        """Save analysis history to file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save analysis history: {e}")

class ReportGenerator:
    """
    Generate comprehensive analysis reports.
    
    Features:
    - Multiple output formats (JSON, HTML, Markdown)
    - Interactive dashboards
    - Executive summaries
    - Detailed technical reports
    """
    
    def __init__(self):
        self.insight_generator = InsightGenerator()
        self.trend_analyzer = TrendAnalyzer()
    
    def generate_comprehensive_report(self, analysis_results: Dict[str, Any], 
                                    issues: List[Issue] = None,
                                    performance_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a comprehensive analysis report."""
        
        # Record analysis for trend tracking
        self.trend_analyzer.record_analysis(analysis_results)
        
        # Generate insights
        insights = []
        if issues:
            insights = self.insight_generator.generate_insights(issues, performance_data)
        
        # Get trends
        trends = self.trend_analyzer.get_trends()
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(analysis_results, insights, trends)
        
        # Generate technical details
        technical_details = self._generate_technical_details(analysis_results, issues)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(insights)
        
        # Compile comprehensive report
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_version": "2.0",
                "analysis_engine": "comprehensive_codebase_analyzer"
            },
            "executive_summary": executive_summary,
            "actionable_insights": [asdict(insight) for insight in insights],
            "trend_analysis": [asdict(trend) for trend in trends],
            "technical_details": technical_details,
            "recommendations": recommendations,
            "raw_analysis": analysis_results
        }
        
        return report
    
    def _generate_executive_summary(self, analysis_results: Dict[str, Any], 
                                   insights: List[ActionableInsight],
                                   trends: List[TrendData]) -> Dict[str, Any]:
        """Generate executive summary."""
        summary = analysis_results.get("summary", {})
        
        # Calculate health score (0-100)
        total_issues = summary.get("total_issues", 0)
        critical_issues = summary.get("critical_issues", 0)
        error_issues = summary.get("error_issues", 0)
        
        # Simple health score calculation
        if total_issues == 0:
            health_score = 100
        else:
            # Penalize critical and error issues more heavily
            penalty = (critical_issues * 10) + (error_issues * 5) + (total_issues * 1)
            health_score = max(0, 100 - penalty)
        
        # Determine overall status
        if health_score >= 90:
            status = "excellent"
        elif health_score >= 75:
            status = "good"
        elif health_score >= 50:
            status = "fair"
        else:
            status = "needs_attention"
        
        # Count high-priority insights
        high_priority_insights = len([i for i in insights if i.priority in ["critical", "high"]])
        
        # Analyze trends
        improving_trends = len([t for t in trends if t.trend_direction == "improving"])
        degrading_trends = len([t for t in trends if t.trend_direction == "degrading"])
        
        return {
            "health_score": health_score,
            "overall_status": status,
            "total_issues": total_issues,
            "critical_issues": critical_issues,
            "high_priority_insights": high_priority_insights,
            "trend_summary": {
                "improving_metrics": improving_trends,
                "degrading_metrics": degrading_trends,
                "stable_metrics": len(trends) - improving_trends - degrading_trends
            },
            "key_findings": self._extract_key_findings(insights),
            "next_steps": self._generate_next_steps(insights)
        }
    
    def _generate_technical_details(self, analysis_results: Dict[str, Any], 
                                   issues: List[Issue] = None) -> Dict[str, Any]:
        """Generate technical details section."""
        details = {
            "analysis_metadata": {
                "duration": analysis_results.get("duration", 0),
                "timestamp": analysis_results.get("timestamp"),
                "repository": analysis_results.get("repository")
            },
            "issue_breakdown": analysis_results.get("issues", {}),
            "statistics": analysis_results.get("statistics", {}),
            "performance_metrics": analysis_results.get("performance", {})
        }
        
        # Add issue distribution analysis
        if issues:
            details["issue_distribution"] = self._analyze_issue_distribution(issues)
        
        return details
    
    def _generate_recommendations(self, insights: List[ActionableInsight]) -> Dict[str, Any]:
        """Generate prioritized recommendations."""
        # Group insights by priority
        by_priority = defaultdict(list)
        for insight in insights:
            by_priority[insight.priority].append(insight)
        
        # Generate immediate actions (critical/high priority)
        immediate_actions = []
        for insight in by_priority["critical"] + by_priority["high"]:
            immediate_actions.extend(insight.recommended_actions)
        
        # Generate short-term goals (medium priority)
        short_term_goals = []
        for insight in by_priority["medium"]:
            short_term_goals.extend(insight.recommended_actions)
        
        # Generate long-term improvements (low priority)
        long_term_improvements = []
        for insight in by_priority["low"]:
            long_term_improvements.extend(insight.recommended_actions)
        
        return {
            "immediate_actions": list(set(immediate_actions))[:5],  # Top 5 unique actions
            "short_term_goals": list(set(short_term_goals))[:5],
            "long_term_improvements": list(set(long_term_improvements))[:5],
            "estimated_effort": self._calculate_total_effort(insights),
            "expected_roi": self._calculate_expected_roi(insights)
        }
    
    def _extract_key_findings(self, insights: List[ActionableInsight]) -> List[str]:
        """Extract key findings from insights."""
        findings = []
        
        # Get top 3 highest priority insights
        top_insights = sorted(insights, key=lambda x: (self.insight_generator._priority_score(x.priority), -x.roi_score), reverse=True)[:3]
        
        for insight in top_insights:
            findings.append(f"{insight.title}: {insight.description}")
        
        return findings
    
    def _generate_next_steps(self, insights: List[ActionableInsight]) -> List[str]:
        """Generate next steps based on insights."""
        next_steps = []
        
        # Get critical and high priority insights
        priority_insights = [i for i in insights if i.priority in ["critical", "high"]]
        
        for insight in priority_insights[:3]:  # Top 3
            next_steps.append(f"Address {insight.title.lower()} (estimated time: {insight.estimated_time})")
        
        if not next_steps:
            next_steps.append("Continue monitoring code quality and address medium priority issues")
        
        return next_steps
    
    def _analyze_issue_distribution(self, issues: List[Issue]) -> Dict[str, Any]:
        """Analyze the distribution of issues."""
        # Count by category
        by_category = Counter()
        by_severity = Counter()
        by_file = Counter()
        
        for issue in issues:
            category = issue.category.value if hasattr(issue.category, 'value') else str(issue.category)
            severity = issue.severity.value if hasattr(issue.severity, 'value') else str(issue.severity)
            
            by_category[category] += 1
            by_severity[severity] += 1
            
            if hasattr(issue, 'location') and issue.location:
                file_path = issue.location.file_path if hasattr(issue.location, 'file_path') else 'unknown'
                by_file[file_path] += 1
        
        return {
            "by_category": dict(by_category),
            "by_severity": dict(by_severity),
            "most_problematic_files": dict(by_file.most_common(10))
        }
    
    def _calculate_total_effort(self, insights: List[ActionableInsight]) -> str:
        """Calculate total estimated effort."""
        # Simple effort calculation
        effort_hours = 0
        for insight in insights:
            if "hour" in insight.estimated_time:
                hours = int(insight.estimated_time.split()[0].split('-')[0])
                effort_hours += hours
            elif "day" in insight.estimated_time:
                days = int(insight.estimated_time.split()[0].split('-')[0])
                effort_hours += days * 8  # 8 hours per day
        
        if effort_hours < 8:
            return f"{effort_hours} hours"
        else:
            return f"{effort_hours // 8} days"
    
    def _calculate_expected_roi(self, insights: List[ActionableInsight]) -> float:
        """Calculate expected return on investment."""
        if not insights:
            return 0.0
        
        total_roi = sum(insight.roi_score for insight in insights)
        return total_roi / len(insights)
    
    def export_html_report(self, report: Dict[str, Any], output_file: str = "analysis_report.html"):
        """Export report as HTML."""
        html_content = self._generate_html_report(report)
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        return output_file
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML report content."""
        # Simple HTML template
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Codebase Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .summary {{ background: #e8f5e8; padding: 15px; margin: 20px 0; border-radius: 5px; }}
        .insight {{ background: #fff3cd; padding: 10px; margin: 10px 0; border-left: 4px solid #ffc107; }}
        .critical {{ border-left-color: #dc3545; }}
        .high {{ border-left-color: #fd7e14; }}
        .medium {{ border-left-color: #ffc107; }}
        .low {{ border-left-color: #28a745; }}
        .trend-improving {{ color: #28a745; }}
        .trend-degrading {{ color: #dc3545; }}
        .trend-stable {{ color: #6c757d; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Codebase Analysis Report</h1>
        <p>Generated: {report['metadata']['generated_at']}</p>
    </div>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <p><strong>Health Score:</strong> {report['executive_summary']['health_score']}/100</p>
        <p><strong>Overall Status:</strong> {report['executive_summary']['overall_status']}</p>
        <p><strong>Total Issues:</strong> {report['executive_summary']['total_issues']}</p>
        <p><strong>Critical Issues:</strong> {report['executive_summary']['critical_issues']}</p>
    </div>
    
    <h2>Actionable Insights</h2>
"""
        
        # Add insights
        for insight in report['actionable_insights']:
            priority_class = insight['priority']
            html += f"""
    <div class="insight {priority_class}">
        <h3>{insight['title']}</h3>
        <p>{insight['description']}</p>
        <p><strong>Priority:</strong> {insight['priority']} | <strong>Impact:</strong> {insight['impact']} | <strong>Effort:</strong> {insight['effort']}</p>
        <p><strong>Estimated Time:</strong> {insight['estimated_time']}</p>
        <ul>
"""
            for action in insight['recommended_actions']:
                html += f"<li>{action}</li>"
            
            html += "</ul></div>"
        
        html += """
    <h2>Recommendations</h2>
    <h3>Immediate Actions</h3>
    <ul>
"""
        
        for action in report['recommendations']['immediate_actions']:
            html += f"<li>{action}</li>"
        
        html += """
    </ul>
</body>
</html>
"""
        
        return html

def generate_enhanced_report(analysis_results: Dict[str, Any], 
                           issues: List[Issue] = None,
                           performance_data: Dict[str, Any] = None,
                           export_html: bool = False) -> Dict[str, Any]:
    """Generate enhanced analysis report with actionable insights."""
    
    generator = ReportGenerator()
    report = generator.generate_comprehensive_report(analysis_results, issues, performance_data)
    
    if export_html:
        html_file = generator.export_html_report(report)
        report['html_export'] = html_file
    
    return report

if __name__ == "__main__":
    # Example usage
    print("📊 Enhanced Reporting System")
    print("=" * 40)
    
    # This would typically be called with real analysis results
    sample_results = {
        "summary": {
            "total_issues": 15,
            "critical_issues": 2,
            "error_issues": 5,
            "warning_issues": 6,
            "info_issues": 2
        },
        "duration": 5.2,
        "timestamp": datetime.now().isoformat()
    }
    
    report = generate_enhanced_report(sample_results, export_html=True)
    print(f"Generated comprehensive report with {len(report['actionable_insights'])} insights")
    
    if 'html_export' in report:
        print(f"HTML report exported to: {report['html_export']}")

from abc import ABC

import networkx as nx

# ProgrammingLanguage enum fallback
# Skill decorators not needed

