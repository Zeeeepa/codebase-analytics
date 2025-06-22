#!/usr/bin/env python3
"""
Interactive Codebase Visualization Engine
Provides comprehensive visualization capabilities including:
- Interactive symbol selection and context viewing
- Repository tree with issue counts
- Dependency graphs and relationship mapping
- Issue heatmaps and complexity charts
- Entry point visualization and navigation
"""

import json
import os
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

# Import analysis types
try:
    from analysis import AnalysisResult, FunctionDefinition, EntryPoint, CodeIssue
except ImportError:
    # Fallback for standalone usage
    AnalysisResult = Any
    FunctionDefinition = Any
    EntryPoint = Any
    CodeIssue = Any

logger = logging.getLogger(__name__)

@dataclass
class VisualizationNode:
    """Represents a node in the visualization"""
    id: str
    label: str
    type: str  # file, directory, function, class, etc.
    path: str
    metadata: Dict[str, Any]
    children: List['VisualizationNode'] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.children:
            result['children'] = [child.to_dict() for child in self.children]
        return result

@dataclass
class InteractiveElement:
    """Represents an interactive element in the visualization"""
    element_id: str
    element_type: str
    position: Dict[str, float]
    properties: Dict[str, Any]
    actions: List[str]
    context_data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class CodebaseVisualizer:
    """Main visualization engine for codebase analytics"""
    
    def __init__(self, analysis_result: AnalysisResult):
        self.analysis_result = analysis_result
        self.file_tree = {}
        self.symbol_map = {}
        self.issue_map = {}
        self._build_internal_maps()
    
    def _build_internal_maps(self):
        """Build internal mapping structures for efficient visualization"""
        # Build file tree structure
        self._build_file_tree()
        
        # Build symbol mapping
        self._build_symbol_map()
        
        # Build issue mapping
        self._build_issue_map()
    
    def _build_file_tree(self):
        """Build hierarchical file tree from analysis results"""
        self.file_tree = {
            "name": "Repository",
            "type": "directory",
            "path": "/",
            "expanded": True,
            "issue_counts": {"critical": 0, "major": 0, "minor": 0, "info": 0},
            "children": {}
        }
        
        # Process all functions to build file structure
        for func in self.analysis_result.all_functions:
            self._add_file_to_tree(func.file_path, func)
        
        # Process all issues to add issue counts
        for issue in self.analysis_result.all_issues:
            self._add_issue_to_tree(issue.file_path, issue)
        
        # Convert to final format
        self.file_tree = self._convert_tree_format(self.file_tree)
    
    def _add_file_to_tree(self, file_path: str, func: FunctionDefinition):
        """Add a file and its function to the tree structure"""
        parts = Path(file_path).parts
        current = self.file_tree
        
        # Navigate/create directory structure
        for i, part in enumerate(parts[:-1]):
            if part not in current["children"]:
                current["children"][part] = {
                    "name": part,
                    "type": "directory",
                    "path": "/".join(parts[:i+1]),
                    "expanded": False,
                    "issue_counts": {"critical": 0, "major": 0, "minor": 0, "info": 0},
                    "children": {}
                }
            current = current["children"][part]
        
        # Add the file
        filename = parts[-1]
        if filename not in current["children"]:
            current["children"][filename] = {
                "name": filename,
                "type": "file",
                "path": file_path,
                "expanded": False,
                "issue_counts": {"critical": 0, "major": 0, "minor": 0, "info": 0},
                "functions": [],
                "entry_points": [],
                "children": {}
            }
        
        # Add function to file
        current["children"][filename]["functions"].append({
            "name": func.name,
            "line_start": func.line_start,
            "line_end": func.line_end,
            "is_entry_point": func.is_entry_point,
            "complexity": func.complexity_score,
            "issues": len(func.issues)
        })
    
    def _add_issue_to_tree(self, file_path: str, issue: CodeIssue):
        """Add issue counts to the tree structure"""
        parts = Path(file_path).parts
        current = self.file_tree
        
        # Navigate to file and increment issue counts along the path
        for part in parts:
            if part in current["children"]:
                current["children"][part]["issue_counts"][issue.severity.value] += 1
                current = current["children"][part]
    
    def _convert_tree_format(self, tree_node: Dict[str, Any]) -> Dict[str, Any]:
        """Convert internal tree format to visualization format"""
        result = {
            "name": tree_node["name"],
            "type": tree_node["type"],
            "path": tree_node["path"],
            "expanded": tree_node["expanded"],
            "issue_counts": tree_node["issue_counts"]
        }
        
        if "functions" in tree_node:
            result["functions"] = tree_node["functions"]
        
        if tree_node["children"]:
            result["children"] = [
                self._convert_tree_format(child) 
                for child in tree_node["children"].values()
            ]
        
        return result
    
    def _build_symbol_map(self):
        """Build symbol mapping for quick lookups"""
        self.symbol_map = {}
        
        for func in self.analysis_result.all_functions:
            self.symbol_map[func.name] = {
                "type": "function",
                "definition": func,
                "file_path": func.file_path,
                "line_number": func.line_start,
                "dependencies": func.dependencies,
                "dependents": func.called_by
            }
        
        for ep in self.analysis_result.all_entry_points:
            self.symbol_map[ep.name] = {
                "type": "entry_point",
                "definition": ep,
                "file_path": ep.file_path,
                "line_number": ep.line_number,
                "dependencies": ep.dependencies,
                "dependents": []
            }
    
    def _build_issue_map(self):
        """Build issue mapping by file and severity"""
        self.issue_map = {
            "by_file": {},
            "by_severity": {"critical": [], "major": [], "minor": [], "info": []},
            "by_type": {}
        }
        
        for issue in self.analysis_result.all_issues:
            # Group by file
            if issue.file_path not in self.issue_map["by_file"]:
                self.issue_map["by_file"][issue.file_path] = []
            self.issue_map["by_file"][issue.file_path].append(issue)
            
            # Group by severity
            self.issue_map["by_severity"][issue.severity.value].append(issue)
            
            # Group by type
            issue_type = issue.type.value
            if issue_type not in self.issue_map["by_type"]:
                self.issue_map["by_type"][issue_type] = []
            self.issue_map["by_type"][issue_type].append(issue)

    def generate_interactive_repository_tree(self) -> Dict[str, Any]:
        """Generate interactive repository tree with clickable elements"""
        return {
            "tree_data": self.file_tree,
            "interactive_elements": self._generate_tree_interactive_elements(),
            "navigation_actions": {
                "expand_all": "expand_all_nodes",
                "collapse_all": "collapse_all_nodes",
                "filter_by_issues": "filter_nodes_by_issues",
                "search_symbols": "search_symbol_in_tree"
            }
        }
    
    def _generate_tree_interactive_elements(self) -> List[InteractiveElement]:
        """Generate interactive elements for the repository tree"""
        elements = []
        
        def process_node(node, path=""):
            node_id = f"tree_node_{path}_{node['name']}"
            
            element = InteractiveElement(
                element_id=node_id,
                element_type=node["type"],
                position={"x": 0, "y": 0},  # Will be calculated by frontend
                properties={
                    "name": node["name"],
                    "path": node["path"],
                    "issue_counts": node["issue_counts"],
                    "expandable": "children" in node and len(node.get("children", [])) > 0
                },
                actions=["click", "expand", "collapse", "context_menu"],
                context_data={
                    "file_path": node["path"],
                    "functions": node.get("functions", []),
                    "issues": self.issue_map["by_file"].get(node["path"], [])
                }
            )
            elements.append(element)
            
            # Process children
            for child in node.get("children", []):
                process_node(child, f"{path}_{node['name']}")
        
        process_node(self.file_tree)
        return elements

    def generate_dependency_graph_visualization(self) -> Dict[str, Any]:
        """Generate interactive dependency graph visualization"""
        nodes = []
        edges = []
        clusters = {}
        
        # Create nodes for functions
        for func in self.analysis_result.all_functions:
            node_data = {
                "id": func.name,
                "label": func.name,
                "type": "function",
                "file": func.file_path,
                "complexity": func.complexity_score,
                "is_entry_point": func.is_entry_point,
                "issue_count": len(func.issues),
                "line_count": func.line_end - func.line_start,
                "size": max(10, min(50, func.complexity_score * 5)),  # Visual size
                "color": self._get_node_color(func)
            }
            nodes.append(node_data)
            
            # Group by file for clustering
            file_key = Path(func.file_path).stem
            if file_key not in clusters:
                clusters[file_key] = []
            clusters[file_key].append(func.name)
        
        # Create edges for dependencies
        for func in self.analysis_result.all_functions:
            for dependency in func.dependencies:
                if dependency in [f.name for f in self.analysis_result.all_functions]:
                    edges.append({
                        "source": func.name,
                        "target": dependency,
                        "type": "calls",
                        "weight": 1
                    })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "clusters": clusters,
            "layout_options": {
                "algorithm": "force_directed",
                "cluster_separation": True,
                "edge_bundling": True
            },
            "interactive_features": {
                "node_selection": True,
                "zoom_pan": True,
                "filter_by_complexity": True,
                "highlight_dependencies": True
            }
        }
    
    def _get_node_color(self, func: FunctionDefinition) -> str:
        """Determine node color based on function properties"""
        if func.is_entry_point:
            return "#ff6b6b"  # Red for entry points
        elif len(func.issues) > 0:
            return "#ffa726"  # Orange for functions with issues
        elif func.complexity_score > 10:
            return "#ffeb3b"  # Yellow for complex functions
        else:
            return "#4caf50"  # Green for normal functions

    def generate_issue_heatmap(self) -> Dict[str, Any]:
        """Generate issue heatmap visualization"""
        heatmap_data = []
        
        # Calculate issue density per file
        for file_path, issues in self.issue_map["by_file"].items():
            severity_counts = {"critical": 0, "major": 0, "minor": 0, "info": 0}
            for issue in issues:
                severity_counts[issue.severity.value] += 1
            
            # Calculate heat score (weighted by severity)
            heat_score = (
                severity_counts["critical"] * 10 +
                severity_counts["major"] * 5 +
                severity_counts["minor"] * 2 +
                severity_counts["info"] * 1
            )
            
            heatmap_data.append({
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "total_issues": len(issues),
                "severity_counts": severity_counts,
                "heat_score": heat_score,
                "relative_heat": 0  # Will be calculated after all files processed
            })
        
        # Calculate relative heat scores
        max_heat = max([item["heat_score"] for item in heatmap_data]) if heatmap_data else 1
        for item in heatmap_data:
            item["relative_heat"] = item["heat_score"] / max_heat
        
        return {
            "heatmap_data": heatmap_data,
            "color_scale": {
                "low": "#4caf50",    # Green
                "medium": "#ffeb3b", # Yellow
                "high": "#ff9800",   # Orange
                "critical": "#f44336" # Red
            },
            "interactive_features": {
                "click_to_view_issues": True,
                "filter_by_severity": True,
                "sort_by_heat": True
            }
        }

    def generate_function_complexity_chart(self) -> Dict[str, Any]:
        """Generate function complexity visualization"""
        complexity_data = []
        
        for func in self.analysis_result.all_functions:
            complexity_data.append({
                "name": func.name,
                "file": func.file_path,
                "complexity": func.complexity_score,
                "line_count": func.line_end - func.line_start,
                "issue_count": len(func.issues),
                "is_entry_point": func.is_entry_point,
                "calls_count": len(func.calls),
                "called_by_count": len(func.called_by)
            })
        
        # Sort by complexity
        complexity_data.sort(key=lambda x: x["complexity"], reverse=True)
        
        return {
            "chart_data": complexity_data,
            "chart_types": ["bar", "scatter", "bubble"],
            "metrics": {
                "average_complexity": sum(f["complexity"] for f in complexity_data) / len(complexity_data) if complexity_data else 0,
                "max_complexity": max(f["complexity"] for f in complexity_data) if complexity_data else 0,
                "high_complexity_count": len([f for f in complexity_data if f["complexity"] > 10])
            },
            "interactive_features": {
                "click_to_view_function": True,
                "filter_by_complexity": True,
                "highlight_entry_points": True
            }
        }

    def generate_entry_points_map(self) -> Dict[str, Any]:
        """Generate entry points visualization and navigation"""
        entry_points_by_type = {}
        entry_points_flow = []
        
        for ep in self.analysis_result.all_entry_points:
            # Group by type
            if ep.type not in entry_points_by_type:
                entry_points_by_type[ep.type] = []
            
            ep_data = {
                "name": ep.name,
                "file": ep.file_path,
                "line": ep.line_number,
                "description": ep.description,
                "parameters": ep.parameters,
                "dependencies": ep.dependencies
            }
            entry_points_by_type[ep.type].append(ep_data)
            
            # Create flow data for visualization
            entry_points_flow.append({
                "id": ep.name,
                "type": ep.type,
                "position": {"x": 0, "y": 0},  # Will be calculated by layout algorithm
                "data": ep_data
            })
        
        return {
            "entry_points_by_type": entry_points_by_type,
            "flow_diagram": {
                "nodes": entry_points_flow,
                "connections": self._generate_entry_point_connections()
            },
            "navigation_map": {
                "total_entry_points": len(self.analysis_result.all_entry_points),
                "types": list(entry_points_by_type.keys()),
                "quick_access": [
                    {"name": ep.name, "type": ep.type, "file": ep.file_path}
                    for ep in self.analysis_result.all_entry_points[:10]  # Top 10 for quick access
                ]
            }
        }
    
    def _generate_entry_point_connections(self) -> List[Dict[str, Any]]:
        """Generate connections between entry points based on dependencies"""
        connections = []
        
        for ep in self.analysis_result.all_entry_points:
            for dep in ep.dependencies:
                # Check if dependency is another entry point
                for other_ep in self.analysis_result.all_entry_points:
                    if other_ep.name == dep:
                        connections.append({
                            "source": ep.name,
                            "target": other_ep.name,
                            "type": "dependency"
                        })
        
        return connections

    def generate_symbol_selection_interface(self) -> Dict[str, Any]:
        """Generate interactive symbol selection and context viewing interface"""
        symbol_categories = {
            "functions": [],
            "entry_points": [],
            "classes": [],
            "variables": []
        }
        
        # Categorize symbols
        for symbol_name, symbol_data in self.symbol_map.items():
            if symbol_data["type"] == "function":
                symbol_categories["functions"].append({
                    "name": symbol_name,
                    "file": symbol_data["file_path"],
                    "line": symbol_data["line_number"],
                    "dependencies": len(symbol_data["dependencies"]),
                    "dependents": len(symbol_data["dependents"]),
                    "selectable": True,
                    "context_available": True
                })
            elif symbol_data["type"] == "entry_point":
                symbol_categories["entry_points"].append({
                    "name": symbol_name,
                    "file": symbol_data["file_path"],
                    "line": symbol_data["line_number"],
                    "dependencies": len(symbol_data["dependencies"]),
                    "selectable": True,
                    "context_available": True
                })
        
        return {
            "symbol_categories": symbol_categories,
            "selection_interface": {
                "search_enabled": True,
                "filter_options": ["by_file", "by_type", "by_complexity", "has_issues"],
                "sort_options": ["name", "file", "complexity", "dependencies"],
                "multi_select": True
            },
            "context_viewer": {
                "show_definition": True,
                "show_dependencies": True,
                "show_usage": True,
                "show_issues": True,
                "show_source_code": True,
                "navigation_enabled": True
            }
        }

    def get_symbol_context(self, symbol_name: str) -> Optional[Dict[str, Any]]:
        """Get complete context for a selected symbol"""
        if symbol_name not in self.symbol_map:
            return None
        
        symbol_data = self.symbol_map[symbol_name]
        definition = symbol_data["definition"]
        
        context = {
            "symbol_name": symbol_name,
            "type": symbol_data["type"],
            "definition": definition.to_dict() if hasattr(definition, 'to_dict') else str(definition),
            "location": {
                "file": symbol_data["file_path"],
                "line": symbol_data["line_number"]
            },
            "relationships": {
                "dependencies": symbol_data["dependencies"],
                "dependents": symbol_data["dependents"]
            },
            "related_issues": [
                issue.to_dict() for issue in self.analysis_result.all_issues
                if issue.file_path == symbol_data["file_path"] and
                symbol_data["line_number"] <= issue.line_number <= (
                    definition.line_end if hasattr(definition, 'line_end') else symbol_data["line_number"] + 10
                )
            ],
            "navigation": {
                "go_to_definition": f"file://{symbol_data['file_path']}:{symbol_data['line_number']}",
                "view_dependencies": [
                    self.get_symbol_location(dep) for dep in symbol_data["dependencies"]
                ],
                "view_usages": [
                    self.get_symbol_location(dep) for dep in symbol_data["dependents"]
                ]
            }
        }
        
        return context
    
    def get_symbol_location(self, symbol_name: str) -> Optional[Dict[str, Any]]:
        """Get location information for a symbol"""
        if symbol_name in self.symbol_map:
            symbol_data = self.symbol_map[symbol_name]
            return {
                "name": symbol_name,
                "file": symbol_data["file_path"],
                "line": symbol_data["line_number"],
                "type": symbol_data["type"]
            }
        return None

    def generate_complete_visualization_data(self) -> Dict[str, Any]:
        """Generate all visualization data in one comprehensive structure"""
        return {
            "repository_tree": self.generate_interactive_repository_tree(),
            "dependency_graph": self.generate_dependency_graph_visualization(),
            "issue_heatmap": self.generate_issue_heatmap(),
            "complexity_chart": self.generate_function_complexity_chart(),
            "entry_points_map": self.generate_entry_points_map(),
            "symbol_interface": self.generate_symbol_selection_interface(),
            "metadata": {
                "total_files": len(set(func.file_path for func in self.analysis_result.all_functions)),
                "total_functions": len(self.analysis_result.all_functions),
                "total_entry_points": len(self.analysis_result.all_entry_points),
                "total_issues": len(self.analysis_result.all_issues),
                "visualization_features": [
                    "interactive_tree",
                    "dependency_graph",
                    "issue_heatmap",
                    "complexity_charts",
                    "symbol_selection",
                    "context_viewing",
                    "navigation"
                ]
            }
        }

# Standalone functions for backward compatibility
def generate_repository_tree(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate repository tree from analysis data (backward compatibility)"""
    # Convert analysis_data to AnalysisResult if needed
    if isinstance(analysis_data, dict):
        # Create a mock analysis result for compatibility
        class MockAnalysisResult:
            def __init__(self, data):
                self.all_functions = data.get('functions', [])
                self.all_entry_points = data.get('entry_points', [])
                self.all_issues = data.get('issues', [])
        
        analysis_result = MockAnalysisResult(analysis_data)
    else:
        analysis_result = analysis_data
    
    visualizer = CodebaseVisualizer(analysis_result)
    return visualizer.generate_interactive_repository_tree()

def generate_visualization_data(analysis_result: AnalysisResult) -> Dict[str, Any]:
    """Generate complete visualization data"""
    visualizer = CodebaseVisualizer(analysis_result)
    return visualizer.generate_complete_visualization_data()

def create_interactive_ui(analysis_result: AnalysisResult) -> Dict[str, Any]:
    """Create interactive UI configuration"""
    visualizer = CodebaseVisualizer(analysis_result)
    
    ui_config = {
        "layout": {
            "type": "dashboard",
            "panels": [
                {
                    "id": "repository_tree",
                    "title": "Repository Explorer",
                    "type": "tree_view",
                    "position": {"x": 0, "y": 0, "width": 3, "height": 6},
                    "data_source": "repository_tree"
                },
                {
                    "id": "dependency_graph",
                    "title": "Dependency Graph",
                    "type": "network_graph",
                    "position": {"x": 3, "y": 0, "width": 6, "height": 4},
                    "data_source": "dependency_graph"
                },
                {
                    "id": "issue_heatmap",
                    "title": "Issue Heatmap",
                    "type": "heatmap",
                    "position": {"x": 9, "y": 0, "width": 3, "height": 4},
                    "data_source": "issue_heatmap"
                },
                {
                    "id": "complexity_chart",
                    "title": "Function Complexity",
                    "type": "bar_chart",
                    "position": {"x": 3, "y": 4, "width": 6, "height": 2},
                    "data_source": "complexity_chart"
                },
                {
                    "id": "symbol_context",
                    "title": "Symbol Context",
                    "type": "context_panel",
                    "position": {"x": 9, "y": 4, "width": 3, "height": 2},
                    "data_source": "symbol_interface"
                }
            ]
        },
        "interactions": {
            "symbol_selection": {
                "enabled": True,
                "multi_select": True,
                "show_context": True
            },
            "navigation": {
                "enabled": True,
                "go_to_definition": True,
                "view_dependencies": True
            },
            "filtering": {
                "enabled": True,
                "by_severity": True,
                "by_complexity": True,
                "by_file_type": True
            }
        },
        "data": visualizer.generate_complete_visualization_data()
    }
    
    return ui_config

if __name__ == "__main__":
    # Example usage
    print("Codebase Visualization Engine")
    print("Use this module with analysis results to generate interactive visualizations")
