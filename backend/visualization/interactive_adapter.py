#!/usr/bin/env python3
"""
Interactive Visualization Adapter

This module provides adapters to transform existing static visualization data
into interactive formats suitable for web-based exploration. It bridges the gap
between the existing analysis capabilities and the new interactive frontend.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum

try:
    from codegen.sdk.core.codebase import Codebase
    from codegen.sdk.core.function import Function
    from codegen.sdk.core.class_definition import Class
    from codegen.sdk.core.symbol import Symbol
    from codegen.sdk.core.file import SourceFile
except ImportError:
    print("Codegen SDK not found. Please ensure it's installed.")

logger = logging.getLogger(__name__)


class InteractionType(str, Enum):
    """Types of interactions supported by the adapter."""
    CLICK = "click"
    HOVER = "hover"
    DRILL_DOWN = "drill_down"
    FILTER = "filter"
    ZOOM = "zoom"


class VisualizationFormat(str, Enum):
    """Output formats for interactive visualizations."""
    JSON = "json"
    D3_JSON = "d3_json"
    RECHARTS = "recharts"
    PLOTLY = "plotly"


@dataclass
class InteractiveNode:
    """Represents a node in an interactive visualization."""
    id: str
    label: str
    value: Union[int, float]
    type: str
    metadata: Dict[str, Any]
    interactions: List[InteractionType]
    children: Optional[List['InteractiveNode']] = None
    position: Optional[Tuple[float, float]] = None
    color: Optional[str] = None
    size: Optional[float] = None


@dataclass
class InteractiveEdge:
    """Represents an edge in an interactive visualization."""
    source: str
    target: str
    weight: float
    type: str
    metadata: Dict[str, Any]
    interactions: List[InteractionType]
    color: Optional[str] = None
    style: Optional[str] = None


@dataclass
class InteractiveVisualization:
    """Complete interactive visualization data structure."""
    nodes: List[InteractiveNode]
    edges: List[InteractiveEdge]
    metadata: Dict[str, Any]
    config: Dict[str, Any]
    interactions: Dict[str, Any]


class InteractiveAdapter:
    """Adapter class for converting analysis data to interactive formats."""
    
    def __init__(self, codebase: Optional[Codebase] = None):
        self.codebase = codebase
        self.color_schemes = {
            'severity': {
                'critical': '#ef4444',
                'major': '#f59e0b',
                'minor': '#22c55e',
                'info': '#3b82f6'
            },
            'complexity': {
                'low': '#22c55e',
                'medium': '#f59e0b',
                'high': '#ef4444'
            },
            'type': {
                'function': '#3b82f6',
                'class': '#8b5cf6',
                'module': '#06b6d4',
                'file': '#64748b'
            }
        }
    
    def create_metrics_overview(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create interactive metrics overview for dashboard."""
        metrics = []
        
        # Maintainability Index
        mi_value = analysis_data.get('maintainability_index', {}).get('average', 0)
        metrics.append({
            'id': 'maintainability',
            'name': 'Maintainability Index',
            'value': mi_value,
            'max_value': 100,
            'severity': self._get_mi_severity(mi_value),
            'color': self._get_severity_color(self._get_mi_severity(mi_value)),
            'description': 'Measures how maintainable the code is (0-100)',
            'drill_down': {
                'type': 'file_breakdown',
                'endpoint': '/api/maintainability/files'
            },
            'interactions': ['click', 'hover', 'drill_down']
        })
        
        # Cyclomatic Complexity
        cc_value = analysis_data.get('cyclomatic_complexity', {}).get('average', 0)
        metrics.append({
            'id': 'complexity',
            'name': 'Cyclomatic Complexity',
            'value': cc_value,
            'max_value': 50,
            'severity': self._get_complexity_severity(cc_value),
            'color': self._get_severity_color(self._get_complexity_severity(cc_value)),
            'description': 'Average complexity across all functions',
            'drill_down': {
                'type': 'function_breakdown',
                'endpoint': '/api/complexity/functions'
            },
            'interactions': ['click', 'hover', 'drill_down']
        })
        
        # Comment Density
        cd_value = analysis_data.get('line_metrics', {}).get('total', {}).get('comment_density', 0)
        metrics.append({
            'id': 'documentation',
            'name': 'Comment Density',
            'value': cd_value,
            'max_value': 100,
            'severity': self._get_documentation_severity(cd_value),
            'color': self._get_severity_color(self._get_documentation_severity(cd_value)),
            'description': 'Percentage of lines that are comments',
            'drill_down': {
                'type': 'file_documentation',
                'endpoint': '/api/documentation/files'
            },
            'interactions': ['click', 'hover', 'drill_down']
        })
        
        return {
            'type': 'metrics_overview',
            'data': metrics,
            'config': {
                'interactive': True,
                'drill_down_enabled': True,
                'hover_tooltips': True
            }
        }
    
    def create_call_graph_data(self, functions: List[Function]) -> Dict[str, Any]:
        """Create interactive call graph data."""
        nodes = []
        edges = []
        
        for func in functions:
            # Create function node
            complexity = self._calculate_function_complexity(func)
            node = InteractiveNode(
                id=f"func_{func.name}_{hash(func.filepath)}",
                label=func.name,
                value=complexity,
                type="function",
                metadata={
                    'filepath': func.filepath,
                    'start_line': getattr(func, 'start_point', [0])[0],
                    'end_line': getattr(func, 'end_point', [0])[0],
                    'complexity': complexity,
                    'parameters': len(getattr(func, 'parameters', [])),
                    'calls_count': len(getattr(func, 'calls', []))
                },
                interactions=[InteractionType.CLICK, InteractionType.HOVER, InteractionType.DRILL_DOWN],
                color=self._get_complexity_color(complexity),
                size=max(10, min(50, complexity * 5))
            )
            nodes.append(node)
            
            # Create edges for function calls
            for call in getattr(func, 'calls', []):
                if hasattr(call, 'target') and call.target:
                    edge = InteractiveEdge(
                        source=node.id,
                        target=f"func_{call.target.name}_{hash(call.target.filepath)}",
                        weight=1.0,
                        type="function_call",
                        metadata={
                            'call_type': 'direct',
                            'line_number': getattr(call, 'line_number', 0)
                        },
                        interactions=[InteractionType.HOVER, InteractionType.CLICK],
                        color='#64748b',
                        style='solid'
                    )
                    edges.append(edge)
        
        return {
            'type': 'call_graph',
            'nodes': [asdict(node) for node in nodes],
            'edges': [asdict(edge) for edge in edges],
            'config': {
                'layout': 'force_directed',
                'interactive': True,
                'zoom_enabled': True,
                'filter_enabled': True,
                'search_enabled': True
            }
        }
    
    def create_dependency_tree(self, files: List[SourceFile]) -> Dict[str, Any]:
        """Create interactive dependency tree visualization."""
        nodes = []
        edges = []
        
        for file in files:
            # Create file node
            file_node = InteractiveNode(
                id=f"file_{hash(file.filepath)}",
                label=file.filepath.split('/')[-1],
                value=len(file.imports) if hasattr(file, 'imports') else 0,
                type="file",
                metadata={
                    'full_path': file.filepath,
                    'size': len(file.source) if hasattr(file, 'source') else 0,
                    'functions_count': len(file.functions) if hasattr(file, 'functions') else 0,
                    'classes_count': len(file.classes) if hasattr(file, 'classes') else 0,
                    'imports_count': len(file.imports) if hasattr(file, 'imports') else 0
                },
                interactions=[InteractionType.CLICK, InteractionType.HOVER, InteractionType.DRILL_DOWN],
                color=self.color_schemes['type']['file']
            )
            nodes.append(file_node)
            
            # Create dependency edges
            for import_stmt in getattr(file, 'imports', []):
                if hasattr(import_stmt, 'module_name'):
                    target_id = f"module_{import_stmt.module_name}"
                    edge = InteractiveEdge(
                        source=file_node.id,
                        target=target_id,
                        weight=1.0,
                        type="import",
                        metadata={
                            'import_type': getattr(import_stmt, 'import_type', 'unknown'),
                            'line_number': getattr(import_stmt, 'line_number', 0)
                        },
                        interactions=[InteractionType.HOVER],
                        color='#06b6d4'
                    )
                    edges.append(edge)
        
        return {
            'type': 'dependency_tree',
            'nodes': [asdict(node) for node in nodes],
            'edges': [asdict(edge) for edge in edges],
            'config': {
                'layout': 'hierarchical',
                'interactive': True,
                'collapsible': True,
                'search_enabled': True,
                'filter_by_type': True
            }
        }
    
    def create_issues_heatmap(self, issues_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create interactive issues heatmap."""
        heatmap_data = []
        
        for filepath, issues in issues_data.items():
            for severity, issue_list in issues.items():
                if issue_list:  # Only include files with issues
                    heatmap_data.append({
                        'id': f"issue_{hash(filepath)}_{severity}",
                        'file': filepath.split('/')[-1],
                        'full_path': filepath,
                        'severity': severity,
                        'count': len(issue_list),
                        'issues': issue_list,
                        'color': self._get_severity_color(severity),
                        'interactions': ['click', 'hover', 'drill_down'],
                        'drill_down': {
                            'type': 'issue_details',
                            'data': issue_list
                        }
                    })
        
        return {
            'type': 'issues_heatmap',
            'data': heatmap_data,
            'config': {
                'interactive': True,
                'filterable': True,
                'sortable': True,
                'groupable': True
            }
        }
    
    def create_code_distribution(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create interactive code distribution charts."""
        distribution_data = [
            {
                'name': 'Functions',
                'value': analysis_data.get('num_functions', 0),
                'color': self.color_schemes['type']['function'],
                'type': 'function',
                'interactions': ['click', 'hover'],
                'drill_down': {
                    'type': 'function_list',
                    'endpoint': '/api/functions'
                }
            },
            {
                'name': 'Classes',
                'value': analysis_data.get('num_classes', 0),
                'color': self.color_schemes['type']['class'],
                'type': 'class',
                'interactions': ['click', 'hover'],
                'drill_down': {
                    'type': 'class_list',
                    'endpoint': '/api/classes'
                }
            },
            {
                'name': 'Files',
                'value': analysis_data.get('num_files', 0),
                'color': self.color_schemes['type']['file'],
                'type': 'file',
                'interactions': ['click', 'hover'],
                'drill_down': {
                    'type': 'file_list',
                    'endpoint': '/api/files'
                }
            }
        ]
        
        return {
            'type': 'code_distribution',
            'data': distribution_data,
            'config': {
                'chart_type': 'pie',
                'interactive': True,
                'show_percentages': True,
                'drill_down_enabled': True
            }
        }
    
    def _calculate_function_complexity(self, func: Function) -> float:
        """Calculate complexity score for a function."""
        # Use existing complexity calculation or provide fallback
        try:
            from ..analyzer import calculate_cyclomatic_complexity
            return calculate_cyclomatic_complexity(func)
        except ImportError:
            # Fallback calculation
            return len(getattr(func, 'calls', [])) + len(getattr(func, 'parameters', [])) + 1
    
    def _get_mi_severity(self, value: float) -> str:
        """Get severity level for maintainability index."""
        if value >= 70:
            return 'info'
        elif value >= 50:
            return 'minor'
        elif value >= 30:
            return 'major'
        else:
            return 'critical'
    
    def _get_complexity_severity(self, value: float) -> str:
        """Get severity level for complexity."""
        if value <= 5:
            return 'info'
        elif value <= 10:
            return 'minor'
        elif value <= 20:
            return 'major'
        else:
            return 'critical'
    
    def _get_documentation_severity(self, value: float) -> str:
        """Get severity level for documentation."""
        if value >= 20:
            return 'info'
        elif value >= 10:
            return 'minor'
        elif value >= 5:
            return 'major'
        else:
            return 'critical'
    
    def _get_severity_color(self, severity: str) -> str:
        """Get color for severity level."""
        return self.color_schemes['severity'].get(severity, '#64748b')
    
    def _get_complexity_color(self, complexity: float) -> str:
        """Get color for complexity level."""
        if complexity <= 5:
            return self.color_schemes['complexity']['low']
        elif complexity <= 15:
            return self.color_schemes['complexity']['medium']
        else:
            return self.color_schemes['complexity']['high']
    
    def to_format(self, data: Dict[str, Any], format_type: VisualizationFormat) -> str:
        """Convert visualization data to specified format."""
        if format_type == VisualizationFormat.JSON:
            return json.dumps(data, indent=2)
        elif format_type == VisualizationFormat.D3_JSON:
            return self._to_d3_format(data)
        elif format_type == VisualizationFormat.RECHARTS:
            return self._to_recharts_format(data)
        elif format_type == VisualizationFormat.PLOTLY:
            return self._to_plotly_format(data)
        else:
            return json.dumps(data, indent=2)
    
    def _to_d3_format(self, data: Dict[str, Any]) -> str:
        """Convert to D3.js compatible format."""
        # Transform data structure for D3
        d3_data = {
            'nodes': data.get('nodes', []),
            'links': data.get('edges', []),
            'config': data.get('config', {})
        }
        return json.dumps(d3_data, indent=2)
    
    def _to_recharts_format(self, data: Dict[str, Any]) -> str:
        """Convert to Recharts compatible format."""
        # Transform for Recharts consumption
        recharts_data = data.get('data', [])
        return json.dumps(recharts_data, indent=2)
    
    def _to_plotly_format(self, data: Dict[str, Any]) -> str:
        """Convert to Plotly compatible format."""
        # Transform for Plotly
        plotly_data = {
            'data': data.get('data', []),
            'layout': data.get('config', {}),
            'config': {'responsive': True}
        }
        return json.dumps(plotly_data, indent=2)


def create_interactive_adapter(codebase: Optional[Codebase] = None) -> InteractiveAdapter:
    """Factory function to create an interactive adapter."""
    return InteractiveAdapter(codebase)


# Utility functions for common transformations
def transform_commit_data_for_interaction(commit_data: Dict[str, int]) -> List[Dict[str, Any]]:
    """Transform commit data for interactive charts."""
    return [
        {
            'month': month,
            'commits': count,
            'id': f"commit_{month}",
            'interactions': ['click', 'hover'],
            'drill_down': {
                'type': 'commit_details',
                'month': month,
                'endpoint': f'/api/commits/{month}'
            }
        }
        for month, count in commit_data.items()
    ]


def add_interaction_metadata(data: Dict[str, Any], interactions: List[str]) -> Dict[str, Any]:
    """Add interaction metadata to any data structure."""
    data['interactions'] = interactions
    data['interactive'] = True
    return data

