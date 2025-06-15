#!/usr/bin/env python3
"""
Codebase Visualization Module

This module provides functionality for visualizing codebase analysis results, including:
- Transforming analysis results into visualization-friendly formats
- Generating interactive visualizations for different types of analysis
- Supporting various output formats (JSON, HTML, etc.)

It separates the visualization logic from analysis concerns, making the codebase
more maintainable and easier to extend with new visualization capabilities.
"""

import json
import logging
import math
import re
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
except ImportError:
    nx = None
    plt = None

# Import from analysis module
from analysis import (
    AnalysisResult, 
    AnalysisType,
    CodeQualityAnalysis,
    DependencyAnalysis,
    SecurityAnalysis,
    PerformanceAnalysis,
    TypeAnalysis,
    Issue,
    IssueCollection,
    IssueSeverity,
    IssueCategory,
    IssueStatus,
    CodeLocation
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


#######################################################
# Visualization Enums and Constants
#######################################################

class VisualizationType(str, Enum):
    """Types of visualizations that can be generated."""

    ISSUE_TREEMAP = "issue_treemap"
    DEPENDENCY_GRAPH = "dependency_graph"
    COMPLEXITY_HEATMAP = "complexity_heatmap"
    CALL_GRAPH = "call_graph"
    ISSUE_DISTRIBUTION = "issue_distribution"
    CODE_METRICS = "code_metrics"
    COMPREHENSIVE = "comprehensive"


class OutputFormat(str, Enum):
    """Output formats for visualizations."""

    JSON = "json"
    HTML = "html"
    SVG = "svg"
    PNG = "png"
    INTERACTIVE = "interactive"


#######################################################
# Visualization Data Classes
#######################################################

@dataclass
class VisualizationNode:
    """Node in a visualization graph."""

    id: str
    label: str
    type: str
    size: float = 1.0
    color: str = "#3b82f6"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualizationEdge:
    """Edge in a visualization graph."""

    source: str
    target: str
    type: str
    weight: float = 1.0
    color: str = "#64748b"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualizationGraph:
    """Graph for visualization."""

    nodes: List[VisualizationNode]
    edges: List[VisualizationEdge]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TreeMapItem:
    """Item in a treemap visualization."""

    id: str
    label: str
    value: float
    color: str
    parent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TreeMapVisualization:
    """Treemap visualization."""

    items: List[TreeMapItem]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HeatMapItem:
    """Item in a heatmap visualization."""

    id: str
    x: str
    y: str
    value: float
    color: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HeatMapVisualization:
    """Heatmap visualization."""

    items: List[HeatMapItem]
    x_labels: List[str]
    y_labels: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChartDataPoint:
    """Data point for chart visualizations."""

    label: str
    value: float
    color: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChartVisualization:
    """Chart visualization."""

    data_points: List[ChartDataPoint]
    chart_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualizationResult:
    """Result of a visualization operation."""

    visualization_type: VisualizationType
    output_format: OutputFormat
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


#######################################################
# Visualization Functions
#######################################################

class CodebaseVisualizer:
    """Main visualizer class for codebase analysis results."""

    def __init__(self, analysis_result: AnalysisResult):
        """
        Initialize the visualizer.
        
        Args:
            analysis_result: Analysis result to visualize
        """
        self.analysis_result = analysis_result
        self.color_palette = {
            "severity": {
                IssueSeverity.CRITICAL.value: "#ef4444",  # Red
                IssueSeverity.ERROR.value: "#f97316",     # Orange
                IssueSeverity.WARNING.value: "#eab308",   # Yellow
                IssueSeverity.INFO.value: "#3b82f6",      # Blue
            },
            "category": {
                IssueCategory.DEAD_CODE.value: "#8b5cf6",          # Purple
                IssueCategory.COMPLEXITY.value: "#ec4899",         # Pink
                IssueCategory.STYLE_ISSUE.value: "#14b8a6",        # Teal
                IssueCategory.DOCUMENTATION.value: "#22c55e",      # Green
                IssueCategory.TYPE_ERROR.value: "#f97316",         # Orange
                IssueCategory.PARAMETER_MISMATCH.value: "#f97316", # Orange
                IssueCategory.RETURN_TYPE_ERROR.value: "#f97316",  # Orange
                IssueCategory.IMPLEMENTATION_ERROR.value: "#ef4444", # Red
                IssueCategory.MISSING_IMPLEMENTATION.value: "#ef4444", # Red
                IssueCategory.IMPORT_ERROR.value: "#8b5cf6",       # Purple
                IssueCategory.DEPENDENCY_CYCLE.value: "#8b5cf6",   # Purple
                IssueCategory.MODULE_COUPLING.value: "#8b5cf6",    # Purple
                IssueCategory.API_CHANGE.value: "#0ea5e9",         # Sky
                IssueCategory.API_USAGE_ERROR.value: "#0ea5e9",    # Sky
                IssueCategory.SECURITY_VULNERABILITY.value: "#ef4444", # Red
                IssueCategory.PERFORMANCE_ISSUE.value: "#f97316",  # Orange
            },
            "file_type": {
                ".py": "#3b82f6",    # Blue
                ".js": "#eab308",    # Yellow
                ".ts": "#0ea5e9",    # Sky
                ".tsx": "#0ea5e9",   # Sky
                ".jsx": "#eab308",   # Yellow
                ".html": "#22c55e",  # Green
                ".css": "#ec4899",   # Pink
                ".json": "#8b5cf6",  # Purple
                ".md": "#64748b",    # Gray
                ".txt": "#64748b",   # Gray
                "other": "#64748b",  # Gray
            },
        }
    
    def visualize(
        self,
        visualization_type: VisualizationType,
        output_format: OutputFormat = OutputFormat.JSON,
        output_file: Optional[str] = None,
    ) -> VisualizationResult:
        """
        Generate a visualization of the analysis result.
        
        Args:
            visualization_type: Type of visualization to generate
            output_format: Format for the visualization output
            output_file: Optional path to save the visualization to
            
        Returns:
            VisualizationResult containing the visualization data
        """
        logger.info(f"Generating {visualization_type.value} visualization in {output_format.value} format")
        
        # Generate the visualization
        if visualization_type == VisualizationType.ISSUE_TREEMAP:
            visualization_data = self._generate_issue_treemap()
        elif visualization_type == VisualizationType.DEPENDENCY_GRAPH:
            visualization_data = self._generate_dependency_graph()
        elif visualization_type == VisualizationType.COMPLEXITY_HEATMAP:
            visualization_data = self._generate_complexity_heatmap()
        elif visualization_type == VisualizationType.CALL_GRAPH:
            visualization_data = self._generate_call_graph()
        elif visualization_type == VisualizationType.ISSUE_DISTRIBUTION:
            visualization_data = self._generate_issue_distribution()
        elif visualization_type == VisualizationType.CODE_METRICS:
            visualization_data = self._generate_code_metrics()
        elif visualization_type == VisualizationType.COMPREHENSIVE:
            visualization_data = self._generate_comprehensive_visualization()
        else:
            raise ValueError(f"Unsupported visualization type: {visualization_type}")
        
        # Create the visualization result
        result = VisualizationResult(
            visualization_type=visualization_type,
            output_format=output_format,
            data=visualization_data,
            metadata={
                "generated_at": datetime.now().isoformat(),
                "analysis_time": self.analysis_result.summary.analysis_time,
                "repo_url": self.analysis_result.summary.repo_url,
                "repo_path": self.analysis_result.summary.repo_path,
            },
        )
        
        # Save the visualization if requested
        if output_file:
            self._save_visualization(result, output_file, output_format)
        
        return result
    
    def _generate_issue_treemap(self) -> Dict[str, Any]:
        """Generate a treemap visualization of issues."""
        logger.info("Generating issue treemap")
        
        treemap_items = []
        
        # Add root item
        treemap_items.append(
            TreeMapItem(
                id="root",
                label="All Issues",
                value=self.analysis_result.summary.total_issues,
                color="#64748b",
            )
        )
        
        # Group issues by severity
        if hasattr(self.analysis_result, "issues") and self.analysis_result.issues:
            issues_by_severity = self.analysis_result.issues.group_by_severity()
            
            for severity, issues in issues_by_severity.items():
                if not issues:
                    continue
                
                severity_id = f"severity_{severity.value}"
                severity_color = self.color_palette["severity"].get(severity.value, "#64748b")
                
                # Add severity item
                treemap_items.append(
                    TreeMapItem(
                        id=severity_id,
                        label=f"{severity.value.capitalize()} ({len(issues)})",
                        value=len(issues),
                        color=severity_color,
                        parent="root",
                        metadata={
                            "severity": severity.value,
                            "issue_count": len(issues),
                        },
                    )
                )
                
                # Group issues by category within severity
                issues_by_category = {}
                for issue in issues:
                    if issue.category:
                        category = issue.category.value
                        if category not in issues_by_category:
                            issues_by_category[category] = []
                        issues_by_category[category].append(issue)
                
                for category, category_issues in issues_by_category.items():
                    category_id = f"{severity_id}_{category}"
                    category_color = self.color_palette["category"].get(category, "#64748b")
                    
                    # Add category item
                    treemap_items.append(
                        TreeMapItem(
                            id=category_id,
                            label=f"{category.replace('_', ' ').capitalize()} ({len(category_issues)})",
                            value=len(category_issues),
                            color=category_color,
                            parent=severity_id,
                            metadata={
                                "severity": severity.value,
                                "category": category,
                                "issue_count": len(category_issues),
                            },
                        )
                    )
                    
                    # Group issues by file within category
                    issues_by_file = {}
                    for issue in category_issues:
                        file = issue.location.file
                        if file not in issues_by_file:
                            issues_by_file[file] = []
                        issues_by_file[file].append(issue)
                    
                    for file, file_issues in issues_by_file.items():
                        file_id = f"{category_id}_{hash(file)}"
                        file_ext = Path(file).suffix
                        file_color = self.color_palette["file_type"].get(file_ext, self.color_palette["file_type"]["other"])
                        
                        # Add file item
                        treemap_items.append(
                            TreeMapItem(
                                id=file_id,
                                label=f"{Path(file).name} ({len(file_issues)})",
                                value=len(file_issues),
                                color=file_color,
                                parent=category_id,
                                metadata={
                                    "severity": severity.value,
                                    "category": category,
                                    "file": file,
                                    "issue_count": len(file_issues),
                                },
                            )
                        )
        
        return {
            "treemap": {
                "items": [asdict(item) for item in treemap_items],
                "metadata": {
                    "total_issues": self.analysis_result.summary.total_issues,
                },
            },
        }
    
    def _generate_dependency_graph(self) -> Dict[str, Any]:
        """Generate a graph visualization of dependencies."""
        logger.info("Generating dependency graph")
        
        nodes = []
        edges = []
        
        if (
            self.analysis_result.dependencies
            and hasattr(self.analysis_result.dependencies, "import_graph")
            and self.analysis_result.dependencies.import_graph
        ):
            import_graph = self.analysis_result.dependencies.import_graph
            
            # Add nodes
            for node in import_graph.get("nodes", []):
                file_ext = Path(node).suffix
                file_color = self.color_palette["file_type"].get(file_ext, self.color_palette["file_type"]["other"])
                
                nodes.append(
                    VisualizationNode(
                        id=f"file_{hash(node)}",
                        label=Path(node).name,
                        type="file",
                        color=file_color,
                        metadata={
                            "file": node,
                        },
                    )
                )
            
            # Add edges
            for edge in import_graph.get("edges", []):
                source, target = edge
                
                edges.append(
                    VisualizationEdge(
                        source=f"file_{hash(source)}",
                        target=f"file_{hash(target)}",
                        type="import",
                        metadata={
                            "source_file": source,
                            "target_file": target,
                        },
                    )
                )
            
            # Check for circular dependencies
            if hasattr(self.analysis_result.dependencies, "circular_dependencies"):
                circular_deps = self.analysis_result.dependencies.circular_dependencies
                
                for cycle in circular_deps.get("circular_dependencies", []):
                    for i in range(len(cycle)):
                        source = cycle[i]
                        target = cycle[(i + 1) % len(cycle)]
                        
                        # Add edge for circular dependency
                        edges.append(
                            VisualizationEdge(
                                source=f"file_{hash(source)}",
                                target=f"file_{hash(target)}",
                                type="circular",
                                color="#ef4444",  # Red
                                weight=2.0,
                                metadata={
                                    "source_file": source,
                                    "target_file": target,
                                    "cycle": cycle,
                                },
                            )
                        )
        
        return {
            "graph": {
                "nodes": [asdict(node) for node in nodes],
                "edges": [asdict(edge) for edge in edges],
                "metadata": {
                    "total_nodes": len(nodes),
                    "total_edges": len(edges),
                },
            },
        }
    
    def _generate_complexity_heatmap(self) -> Dict[str, Any]:
        """Generate a heatmap visualization of code complexity."""
        logger.info("Generating complexity heatmap")
        
        heatmap_items = []
        x_labels = []
        y_labels = []
        
        if (
            self.analysis_result.code_quality
            and hasattr(self.analysis_result.code_quality, "complexity")
            and self.analysis_result.code_quality.complexity
        ):
            complexity = self.analysis_result.code_quality.complexity
            
            # Get complexity by file
            complexity_by_file = complexity.get("complexity_by_file", {})
            
            # Sort files by average complexity
            sorted_files = sorted(
                complexity_by_file.items(),
                key=lambda x: x[1].get("average_complexity", 0),
                reverse=True,
            )
            
            # Take top 20 files
            top_files = sorted_files[:20]
            
            # Create heatmap items
            for i, (file, data) in enumerate(top_files):
                file_name = Path(file).name
                avg_complexity = data.get("average_complexity", 0)
                
                # Determine color based on complexity
                if avg_complexity <= 5:
                    color = "#22c55e"  # Green
                elif avg_complexity <= 10:
                    color = "#eab308"  # Yellow
                elif avg_complexity <= 20:
                    color = "#f97316"  # Orange
                else:
                    color = "#ef4444"  # Red
                
                heatmap_items.append(
                    HeatMapItem(
                        id=f"complexity_{hash(file)}",
                        x="Complexity",
                        y=file_name,
                        value=avg_complexity,
                        color=color,
                        metadata={
                            "file": file,
                            "average_complexity": avg_complexity,
                            "total_complexity": data.get("total_complexity", 0),
                            "function_count": data.get("function_count", 0),
                        },
                    )
                )
                
                y_labels.append(file_name)
            
            x_labels = ["Complexity"]
        
        return {
            "heatmap": {
                "items": [asdict(item) for item in heatmap_items],
                "x_labels": x_labels,
                "y_labels": y_labels,
                "metadata": {
                    "total_items": len(heatmap_items),
                },
            },
        }
    
    def _generate_call_graph(self) -> Dict[str, Any]:
        """Generate a graph visualization of function calls."""
        logger.info("Generating call graph")
        
        # This would typically use call graph data from the analysis
        # For this example, we'll just return an empty graph
        
        return {
            "graph": {
                "nodes": [],
                "edges": [],
                "metadata": {
                    "total_nodes": 0,
                    "total_edges": 0,
                },
            },
        }
    
    def _generate_issue_distribution(self) -> Dict[str, Any]:
        """Generate a chart visualization of issue distribution."""
        logger.info("Generating issue distribution")
        
        data_points = []
        
        if hasattr(self.analysis_result, "issues") and self.analysis_result.issues:
            issues_by_severity = self.analysis_result.issues.group_by_severity()
            
            for severity, issues in issues_by_severity.items():
                if not issues:
                    continue
                
                severity_color = self.color_palette["severity"].get(severity.value, "#64748b")
                
                data_points.append(
                    ChartDataPoint(
                        label=severity.value.capitalize(),
                        value=len(issues),
                        color=severity_color,
                        metadata={
                            "severity": severity.value,
                            "issue_count": len(issues),
                        },
                    )
                )
        
        return {
            "chart": {
                "data_points": [asdict(point) for point in data_points],
                "chart_type": "pie",
                "metadata": {
                    "total_issues": self.analysis_result.summary.total_issues,
                },
            },
        }
    
    def _generate_code_metrics(self) -> Dict[str, Any]:
        """Generate a chart visualization of code metrics."""
        logger.info("Generating code metrics")
        
        data_points = []
        
        # Add total files
        data_points.append(
            ChartDataPoint(
                label="Files",
                value=self.analysis_result.summary.total_files,
                color="#3b82f6",  # Blue
                metadata={
                    "metric": "files",
                    "value": self.analysis_result.summary.total_files,
                },
            )
        )
        
        # Add total classes
        data_points.append(
            ChartDataPoint(
                label="Classes",
                value=self.analysis_result.summary.total_classes,
                color="#8b5cf6",  # Purple
                metadata={
                    "metric": "classes",
                    "value": self.analysis_result.summary.total_classes,
                },
            )
        )
        
        # Add total functions
        data_points.append(
            ChartDataPoint(
                label="Functions",
                value=self.analysis_result.summary.total_functions,
                color="#22c55e",  # Green
                metadata={
                    "metric": "functions",
                    "value": self.analysis_result.summary.total_functions,
                },
            )
        )
        
        # Add total issues
        data_points.append(
            ChartDataPoint(
                label="Issues",
                value=self.analysis_result.summary.total_issues,
                color="#ef4444",  # Red
                metadata={
                    "metric": "issues",
                    "value": self.analysis_result.summary.total_issues,
                },
            )
        )
        
        return {
            "chart": {
                "data_points": [asdict(point) for point in data_points],
                "chart_type": "bar",
                "metadata": {
                    "total_metrics": len(data_points),
                },
            },
        }
    
    def _generate_comprehensive_visualization(self) -> Dict[str, Any]:
        """Generate a comprehensive visualization of all analysis results."""
        logger.info("Generating comprehensive visualization")
        
        # Generate all visualizations
        issue_treemap = self._generate_issue_treemap()
        dependency_graph = self._generate_dependency_graph()
        complexity_heatmap = self._generate_complexity_heatmap()
        call_graph = self._generate_call_graph()
        issue_distribution = self._generate_issue_distribution()
        code_metrics = self._generate_code_metrics()
        
        return {
            "comprehensive": {
                "issue_treemap": issue_treemap.get("treemap"),
                "dependency_graph": dependency_graph.get("graph"),
                "complexity_heatmap": complexity_heatmap.get("heatmap"),
                "call_graph": call_graph.get("graph"),
                "issue_distribution": issue_distribution.get("chart"),
                "code_metrics": code_metrics.get("chart"),
                "metadata": {
                    "total_files": self.analysis_result.summary.total_files,
                    "total_classes": self.analysis_result.summary.total_classes,
                    "total_functions": self.analysis_result.summary.total_functions,
                    "total_issues": self.analysis_result.summary.total_issues,
                },
            },
        }
    
    def _save_visualization(
        self,
        visualization: VisualizationResult,
        output_file: str,
        output_format: OutputFormat,
    ) -> None:
        """
        Save a visualization to a file.
        
        Args:
            visualization: Visualization to save
            output_file: Path to save the visualization to
            output_format: Format for the visualization output
        """
        if output_format == OutputFormat.JSON:
            with open(output_file, "w") as f:
                json.dump(asdict(visualization), f, indent=2)
        elif output_format == OutputFormat.HTML:
            self._save_html_visualization(visualization, output_file)
        elif output_format == OutputFormat.SVG or output_format == OutputFormat.PNG:
            self._save_image_visualization(visualization, output_file, output_format)
        else:
            logger.warning(f"Unsupported output format: {output_format}, using JSON")
            with open(output_file, "w") as f:
                json.dump(asdict(visualization), f, indent=2)
        
        logger.info(f"Visualization saved to {output_file}")
    
    def _save_html_visualization(
        self,
        visualization: VisualizationResult,
        output_file: str,
    ) -> None:
        """
        Save a visualization as HTML.
        
        Args:
            visualization: Visualization to save
            output_file: Path to save the visualization to
        """
        # This would typically generate an HTML file with interactive visualizations
        # For this example, we'll just create a simple HTML file
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Codebase Analysis Visualization</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                h1 {{ color: #333; }}
                .visualization {{ margin-bottom: 30px; }}
                pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow: auto; }}
            </style>
        </head>
        <body>
            <h1>Codebase Analysis Visualization</h1>
            <div class="visualization">
                <h2>{visualization.visualization_type.value.replace('_', ' ').title()}</h2>
                <pre>{json.dumps(visualization.data, indent=2)}</pre>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, "w") as f:
            f.write(html)
    
    def _save_image_visualization(
        self,
        visualization: VisualizationResult,
        output_file: str,
        output_format: OutputFormat,
    ) -> None:
        """
        Save a visualization as an image.
        
        Args:
            visualization: Visualization to save
            output_file: Path to save the visualization to
            output_format: Format for the visualization output
        """
        # This would typically generate an image file
        # For this example, we'll just log a message
        
        logger.info(f"Image visualization not implemented, using JSON instead")
        with open(output_file, "w") as f:
            json.dump(asdict(visualization), f, indent=2)


def visualize_analysis(
    analysis_result: AnalysisResult,
    visualization_type: str = "comprehensive",
    output_format: str = "json",
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Visualize analysis results.
    
    Args:
        analysis_result: Analysis result to visualize
        visualization_type: Type of visualization to generate
        output_format: Format for the visualization output
        output_file: Optional path to save the visualization to
        
    Returns:
        Dictionary containing the visualization data
    """
    # Convert string types to enum values
    try:
        viz_type = VisualizationType(visualization_type)
    except ValueError:
        logger.warning(f"Unknown visualization type: {visualization_type}, using comprehensive")
        viz_type = VisualizationType.COMPREHENSIVE
    
    try:
        out_format = OutputFormat(output_format)
    except ValueError:
        logger.warning(f"Unknown output format: {output_format}, using JSON")
        out_format = OutputFormat.JSON
    
    # Initialize visualizer
    visualizer = CodebaseVisualizer(analysis_result)
    
    # Generate visualization
    result = visualizer.visualize(
        visualization_type=viz_type,
        output_format=out_format,
        output_file=output_file,
    )
    
    return result.data
