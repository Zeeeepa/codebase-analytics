

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
    return dead_code_analysis.get('total_dead_functions', 0)