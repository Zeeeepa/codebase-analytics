#!/usr/bin/env python3
"""
Visual Codebase Explorer

A comprehensive visual exploration system for codebases that focuses on:
- Immediate error detection and visualization
- Blast radius analysis for understanding impact
- Interactive structural navigation
- Real-time context retrieval for issues

Based on graph-sitter principles but focused on visual exploration rather than trends.
"""

import json
import logging
import networkx as nx
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from codegen.sdk.core.codebase import Codebase
    from codegen.sdk.core.function import Function
    from codegen.sdk.core.class_definition import Class
    from codegen.sdk.core.symbol import Symbol
    from codegen.sdk.core.file import SourceFile
    from codegen.sdk.core.import_resolution import Import
    from codegen.sdk.core.external_module import ExternalModule
    from codegen.sdk.core.detached_symbols.function_call import FunctionCall
    from codegen.sdk.core.dataclasses.usage import Usage
    from codegen.sdk.python.function import PyFunction
    from codegen.sdk.python.symbol import PySymbol
    from codegen.sdk.enums import EdgeType, SymbolType
except ImportError:
    print("Codegen SDK not found. Please ensure it's installed.")

logger = logging.getLogger(__name__)


class ExplorationMode(str, Enum):
    """Different modes of visual exploration."""
    STRUCTURAL_OVERVIEW = "structural_overview"
    ERROR_FOCUSED = "error_focused"
    BLAST_RADIUS = "blast_radius"
    CALL_TRACE = "call_trace"
    DEPENDENCY_MAP = "dependency_map"
    CRITICAL_PATHS = "critical_paths"


class IssueImpact(str, Enum):
    """Impact levels for issues based on blast radius."""
    ISOLATED = "isolated"      # Affects only the current function/class
    LOCAL = "local"           # Affects current module/file
    MODULE = "module"         # Affects multiple modules
    SYSTEM = "system"         # Affects core system functionality


@dataclass
class VisualNode:
    """Represents a node in the visual exploration graph."""
    id: str
    name: str
    type: str  # function, class, file, module, error
    path: str
    issues: List[Dict[str, Any]] = field(default_factory=list)
    blast_radius: int = 0
    impact_level: IssueImpact = IssueImpact.ISOLATED
    metadata: Dict[str, Any] = field(default_factory=dict)
    visual_properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VisualEdge:
    """Represents an edge in the visual exploration graph."""
    source: str
    target: str
    relationship: str  # calls, uses, depends_on, affects, inherits
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    visual_properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VisualCodebaseExplorer:
    """
    Visual codebase explorer that provides immediate insights into code structure,
    errors, and impact analysis without focusing on trends.
    """

    def __init__(self, codebase: Codebase):
        self.codebase = codebase
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, VisualNode] = {}
        self.edges: List[VisualEdge] = []
        self.error_hotspots: List[Dict[str, Any]] = []
        self.critical_paths: List[List[str]] = []
        
        # Visual configuration
        self.color_palette = {
            "function": "#3b82f6",      # Blue
            "class": "#8b5cf6",         # Purple  
            "file": "#64748b",          # Gray
            "module": "#06b6d4",        # Cyan
            "error_critical": "#ef4444", # Red
            "error_major": "#f59e0b",    # Orange
            "error_minor": "#eab308",    # Yellow
            "healthy": "#22c55e",        # Green
            "entry_point": "#9cdcfe",    # Light blue
            "high_impact": "#dc2626",    # Dark red
            "medium_impact": "#ea580c",  # Dark orange
            "low_impact": "#65a30d"      # Dark green
        }

    def explore_codebase(self, mode: ExplorationMode = ExplorationMode.STRUCTURAL_OVERVIEW) -> Dict[str, Any]:
        """
        Perform visual exploration of the codebase based on the specified mode.
        """
        logger.info(f"Starting visual exploration in {mode.value} mode")
        
        # Build the base graph structure
        self._build_base_graph()
        
        # Apply mode-specific analysis
        if mode == ExplorationMode.ERROR_FOCUSED:
            self._analyze_error_patterns()
        elif mode == ExplorationMode.BLAST_RADIUS:
            self._analyze_blast_radius()
        elif mode == ExplorationMode.CALL_TRACE:
            self._analyze_call_traces()
        elif mode == ExplorationMode.DEPENDENCY_MAP:
            self._analyze_dependencies()
        elif mode == ExplorationMode.CRITICAL_PATHS:
            self._analyze_critical_paths()
        
        # Calculate visual properties
        self._calculate_visual_properties()
        
        # Generate exploration report
        return self._generate_exploration_report(mode)

    def _build_base_graph(self):
        """Build the base graph structure from the codebase."""
        logger.info("Building base graph structure")
        
        # Add file nodes
        for file in self.codebase.files:
            file_node = VisualNode(
                id=f"file_{hash(file.filepath)}",
                name=Path(file.filepath).name,
                type="file",
                path=file.filepath,
                metadata={
                    "full_path": file.filepath,
                    "size": len(file.source) if hasattr(file, 'source') else 0,
                    "functions_count": len(file.functions) if hasattr(file, 'functions') else 0,
                    "classes_count": len(file.classes) if hasattr(file, 'classes') else 0
                }
            )
            self.nodes[file_node.id] = file_node
            self.graph.add_node(file_node.id, **file_node.to_dict())
            
            # Add function nodes
            if hasattr(file, 'functions'):
                for func in file.functions:
                    func_node = self._create_function_node(func, file)
                    self.nodes[func_node.id] = func_node
                    self.graph.add_node(func_node.id, **func_node.to_dict())
                    
                    # Add edge from file to function
                    edge = VisualEdge(
                        source=file_node.id,
                        target=func_node.id,
                        relationship="contains"
                    )
                    self.edges.append(edge)
                    self.graph.add_edge(edge.source, edge.target, **edge.to_dict())
            
            # Add class nodes
            if hasattr(file, 'classes'):
                for cls in file.classes:
                    class_node = self._create_class_node(cls, file)
                    self.nodes[class_node.id] = class_node
                    self.graph.add_node(class_node.id, **class_node.to_dict())
                    
                    # Add edge from file to class
                    edge = VisualEdge(
                        source=file_node.id,
                        target=class_node.id,
                        relationship="contains"
                    )
                    self.edges.append(edge)
                    self.graph.add_edge(edge.source, edge.target, **edge.to_dict())

    def _create_function_node(self, func: Function, file: SourceFile) -> VisualNode:
        """Create a visual node for a function."""
        issues = self._detect_function_issues(func)
        
        return VisualNode(
            id=f"func_{hash(func.name + file.filepath)}",
            name=func.name,
            type="function",
            path=f"{file.filepath}:{getattr(func, 'start_point', [0])[0]}",
            issues=issues,
            blast_radius=self._calculate_function_blast_radius(func),
            metadata={
                "file": file.filepath,
                "is_method": getattr(func, 'is_method', False),
                "parameters": len(getattr(func, 'parameters', [])),
                "calls_count": len(getattr(func, 'function_calls', [])),
                "usages_count": len(getattr(func, 'usages', [])),
                "complexity": self._calculate_complexity(func)
            }
        )

    def _create_class_node(self, cls: Class, file: SourceFile) -> VisualNode:
        """Create a visual node for a class."""
        issues = self._detect_class_issues(cls)
        
        return VisualNode(
            id=f"class_{hash(cls.name + file.filepath)}",
            name=cls.name,
            type="class",
            path=f"{file.filepath}:{getattr(cls, 'start_point', [0])[0]}",
            issues=issues,
            blast_radius=self._calculate_class_blast_radius(cls),
            metadata={
                "file": file.filepath,
                "methods_count": len(getattr(cls, 'methods', [])),
                "base_classes": [base.name for base in getattr(cls, 'base_classes', [])],
                "usages_count": len(getattr(cls, 'usages', []))
            }
        )

    def _detect_function_issues(self, func: Function) -> List[Dict[str, Any]]:
        """Detect issues in a function."""
        issues = []
        
        # Check for unused function
        if hasattr(func, 'usages') and not func.usages:
            issues.append({
                "type": "unused_function",
                "severity": "minor",
                "message": "Function is defined but never used",
                "suggestion": "Consider removing if truly unused, or check if it should be public API"
            })
        
        # Check for high complexity
        complexity = self._calculate_complexity(func)
        if complexity > 10:
            issues.append({
                "type": "high_complexity",
                "severity": "major" if complexity > 20 else "minor",
                "message": f"Function has high cyclomatic complexity ({complexity})",
                "suggestion": "Consider breaking into smaller functions"
            })
        
        # Check for missing docstring
        if hasattr(func, 'code_block') and not getattr(func, 'docstring', None):
            issues.append({
                "type": "missing_documentation",
                "severity": "minor",
                "message": "Function lacks documentation",
                "suggestion": "Add docstring describing purpose, parameters, and return value"
            })
        
        return issues

    def _detect_class_issues(self, cls: Class) -> List[Dict[str, Any]]:
        """Detect issues in a class."""
        issues = []
        
        # Check for unused class
        if hasattr(cls, 'usages') and not cls.usages:
            issues.append({
                "type": "unused_class",
                "severity": "minor",
                "message": "Class is defined but never used",
                "suggestion": "Consider removing if truly unused"
            })
        
        return issues

    def _calculate_function_blast_radius(self, func: Function) -> int:
        """Calculate the blast radius of a function (how many things it affects)."""
        if not hasattr(func, 'usages'):
            return 0
        return len(func.usages)

    def _calculate_class_blast_radius(self, cls: Class) -> int:
        """Calculate the blast radius of a class."""
        if not hasattr(cls, 'usages'):
            return 0
        return len(cls.usages)

    def _calculate_complexity(self, func: Function) -> int:
        """Calculate cyclomatic complexity of a function."""
        # Simplified complexity calculation
        if not hasattr(func, 'code_block'):
            return 1
        
        code = func.code_block.source if hasattr(func.code_block, 'source') else ""
        complexity = 1  # Base complexity
        
        # Count decision points
        complexity += code.count('if ')
        complexity += code.count('elif ')
        complexity += code.count('while ')
        complexity += code.count('for ')
        complexity += code.count('except ')
        complexity += code.count('and ')
        complexity += code.count('or ')
        
        return complexity

    def _analyze_error_patterns(self):
        """Analyze error patterns and hotspots in the codebase."""
        logger.info("Analyzing error patterns")
        
        error_counts = defaultdict(int)
        
        for node in self.nodes.values():
            if node.issues:
                for issue in node.issues:
                    error_counts[issue['severity']] += 1
                    
                    # Mark high-impact errors
                    if issue['severity'] == 'critical' or node.blast_radius > 10:
                        node.impact_level = IssueImpact.SYSTEM
                    elif issue['severity'] == 'major' or node.blast_radius > 5:
                        node.impact_level = IssueImpact.MODULE
                    elif node.blast_radius > 1:
                        node.impact_level = IssueImpact.LOCAL
                    else:
                        node.impact_level = IssueImpact.ISOLATED
        
        # Identify error hotspots
        self.error_hotspots = [
            {
                "node_id": node.id,
                "name": node.name,
                "type": node.type,
                "issue_count": len(node.issues),
                "blast_radius": node.blast_radius,
                "impact_level": node.impact_level.value
            }
            for node in self.nodes.values()
            if node.issues and len(node.issues) > 1
        ]
        
        # Sort by impact
        self.error_hotspots.sort(key=lambda x: (x['blast_radius'], x['issue_count']), reverse=True)

    def _analyze_blast_radius(self):
        """Analyze blast radius for all symbols in the codebase."""
        logger.info("Analyzing blast radius")
        
        # Build usage relationships
        for file in self.codebase.files:
            if hasattr(file, 'functions'):
                for func in file.functions:
                    if hasattr(func, 'usages'):
                        func_node_id = f"func_{hash(func.name + file.filepath)}"
                        
                        for usage in func.usages:
                            if hasattr(usage, 'usage_symbol'):
                                target_symbol = usage.usage_symbol
                                target_id = self._get_symbol_node_id(target_symbol)
                                
                                if target_id and target_id in self.nodes:
                                    edge = VisualEdge(
                                        source=func_node_id,
                                        target=target_id,
                                        relationship="affects",
                                        metadata={
                                            "usage_type": "function_usage",
                                            "file": getattr(usage.match, 'filepath', '') if hasattr(usage, 'match') else ''
                                        }
                                    )
                                    self.edges.append(edge)
                                    self.graph.add_edge(edge.source, edge.target, **edge.to_dict())

    def _analyze_call_traces(self):
        """Analyze call traces to understand execution flows."""
        logger.info("Analyzing call traces")
        
        # Build call relationships
        for file in self.codebase.files:
            if hasattr(file, 'functions'):
                for func in file.functions:
                    if hasattr(func, 'function_calls'):
                        func_node_id = f"func_{hash(func.name + file.filepath)}"
                        
                        for call in func.function_calls:
                            if hasattr(call, 'function_definition') and call.function_definition:
                                target_func = call.function_definition
                                target_id = self._get_symbol_node_id(target_func)
                                
                                if target_id and target_id in self.nodes:
                                    edge = VisualEdge(
                                        source=func_node_id,
                                        target=target_id,
                                        relationship="calls",
                                        metadata={
                                            "call_name": call.name,
                                            "file": getattr(call, 'filepath', ''),
                                            "line": getattr(call, 'start_point', [0])[0] if hasattr(call, 'start_point') else 0
                                        }
                                    )
                                    self.edges.append(edge)
                                    self.graph.add_edge(edge.source, edge.target, **edge.to_dict())

    def _analyze_dependencies(self):
        """Analyze dependency relationships."""
        logger.info("Analyzing dependencies")
        
        # Build import/dependency relationships
        for file in self.codebase.files:
            file_node_id = f"file_{hash(file.filepath)}"
            
            if hasattr(file, 'imports'):
                for import_stmt in file.imports:
                    if hasattr(import_stmt, 'resolved_symbol') and import_stmt.resolved_symbol:
                        target_symbol = import_stmt.resolved_symbol
                        target_id = self._get_symbol_node_id(target_symbol)
                        
                        if target_id and target_id in self.nodes:
                            edge = VisualEdge(
                                source=file_node_id,
                                target=target_id,
                                relationship="depends_on",
                                metadata={
                                    "import_type": getattr(import_stmt, 'import_type', 'unknown'),
                                    "module_name": getattr(import_stmt, 'module_name', '')
                                }
                            )
                            self.edges.append(edge)
                            self.graph.add_edge(edge.source, edge.target, **edge.to_dict())

    def _analyze_critical_paths(self):
        """Identify critical paths in the codebase."""
        logger.info("Analyzing critical paths")
        
        # Find entry points (functions with high usage but low dependencies)
        entry_points = []
        for node in self.nodes.values():
            if node.type == "function" and node.blast_radius > 5:
                # Check if it's likely an entry point
                incoming_edges = [e for e in self.edges if e.target == node.id and e.relationship == "calls"]
                if len(incoming_edges) < 3:  # Few callers but high impact
                    entry_points.append(node.id)
        
        # Find critical paths from entry points
        for entry_point in entry_points:
            try:
                # Find paths to high-impact nodes
                for target_node in self.nodes.values():
                    if target_node.blast_radius > 3 and target_node.id != entry_point:
                        try:
                            path = nx.shortest_path(self.graph, entry_point, target_node.id)
                            if len(path) > 2:  # Meaningful path
                                self.critical_paths.append(path)
                        except nx.NetworkXNoPath:
                            continue
            except Exception as e:
                logger.warning(f"Error finding paths from {entry_point}: {e}")

    def _calculate_visual_properties(self):
        """Calculate visual properties for nodes and edges."""
        logger.info("Calculating visual properties")
        
        for node in self.nodes.values():
            # Determine node color based on issues and impact
            if node.issues:
                critical_issues = [i for i in node.issues if i['severity'] == 'critical']
                major_issues = [i for i in node.issues if i['severity'] == 'major']
                
                if critical_issues:
                    color = self.color_palette["error_critical"]
                elif major_issues:
                    color = self.color_palette["error_major"]
                else:
                    color = self.color_palette["error_minor"]
            else:
                color = self.color_palette.get(node.type, self.color_palette["healthy"])
            
            # Determine node size based on blast radius
            size = max(10, min(50, 10 + node.blast_radius * 2))
            
            node.visual_properties = {
                "color": color,
                "size": size,
                "border_width": 2 if node.issues else 1,
                "opacity": 0.9 if node.blast_radius > 5 else 0.7
            }
        
        for edge in self.edges:
            # Determine edge properties based on relationship
            if edge.relationship == "affects":
                color = self.color_palette["high_impact"]
                width = 3
            elif edge.relationship == "calls":
                color = self.color_palette["medium_impact"]
                width = 2
            else:
                color = self.color_palette["low_impact"]
                width = 1
            
            edge.visual_properties = {
                "color": color,
                "width": width,
                "style": "solid" if edge.relationship in ["calls", "affects"] else "dashed"
            }

    def _get_symbol_node_id(self, symbol) -> Optional[str]:
        """Get the node ID for a symbol."""
        if isinstance(symbol, Function):
            return f"func_{hash(symbol.name + getattr(symbol, 'filepath', ''))}"
        elif isinstance(symbol, Class):
            return f"class_{hash(symbol.name + getattr(symbol, 'filepath', ''))}"
        elif hasattr(symbol, 'filepath'):
            return f"file_{hash(symbol.filepath)}"
        return None

    def _generate_exploration_report(self, mode: ExplorationMode) -> Dict[str, Any]:
        """Generate a comprehensive exploration report."""
        total_nodes = len(self.nodes)
        total_issues = sum(len(node.issues) for node in self.nodes.values())
        
        # Calculate issue distribution
        issue_distribution = defaultdict(int)
        for node in self.nodes.values():
            for issue in node.issues:
                issue_distribution[issue['severity']] += 1
        
        # Find most critical nodes
        critical_nodes = sorted(
            [node for node in self.nodes.values() if node.issues],
            key=lambda x: (len([i for i in x.issues if i['severity'] == 'critical']), x.blast_radius),
            reverse=True
        )[:10]
        
        return {
            "exploration_mode": mode.value,
            "summary": {
                "total_nodes": total_nodes,
                "total_issues": total_issues,
                "error_hotspots_count": len(self.error_hotspots),
                "critical_paths_count": len(self.critical_paths),
                "issue_distribution": dict(issue_distribution)
            },
            "visual_graph": {
                "nodes": [node.to_dict() for node in self.nodes.values()],
                "edges": [edge.to_dict() for edge in self.edges]
            },
            "error_hotspots": self.error_hotspots[:20],  # Top 20 hotspots
            "critical_paths": self.critical_paths[:10],   # Top 10 critical paths
            "critical_nodes": [
                {
                    "id": node.id,
                    "name": node.name,
                    "type": node.type,
                    "path": node.path,
                    "issues": node.issues,
                    "blast_radius": node.blast_radius,
                    "impact_level": node.impact_level.value
                }
                for node in critical_nodes
            ],
            "visualization_config": {
                "color_palette": self.color_palette,
                "layout_algorithm": "force_directed",
                "interactive_features": [
                    "zoom", "pan", "node_selection", "edge_filtering",
                    "issue_highlighting", "blast_radius_visualization"
                ]
            },
            "exploration_insights": self._generate_insights(mode)
        }

    def _generate_insights(self, mode: ExplorationMode) -> List[Dict[str, Any]]:
        """Generate actionable insights based on the exploration."""
        insights = []
        
        # High-impact error insight
        high_impact_errors = [
            node for node in self.nodes.values()
            if node.issues and node.blast_radius > 5
        ]
        
        if high_impact_errors:
            insights.append({
                "type": "high_impact_errors",
                "priority": "critical",
                "title": f"Found {len(high_impact_errors)} high-impact errors",
                "description": "These errors affect multiple parts of the codebase and should be prioritized",
                "affected_nodes": [node.id for node in high_impact_errors[:5]]
            })
        
        # Unused code insight
        unused_functions = [
            node for node in self.nodes.values()
            if node.type == "function" and any(
                issue['type'] == 'unused_function' for issue in node.issues
            )
        ]
        
        if unused_functions:
            insights.append({
                "type": "unused_code",
                "priority": "medium",
                "title": f"Found {len(unused_functions)} unused functions",
                "description": "Consider removing unused code to improve maintainability",
                "affected_nodes": [node.id for node in unused_functions[:10]]
            })
        
        # Complexity hotspots
        complex_functions = [
            node for node in self.nodes.values()
            if node.type == "function" and node.metadata.get('complexity', 0) > 15
        ]
        
        if complex_functions:
            insights.append({
                "type": "complexity_hotspots",
                "priority": "medium",
                "title": f"Found {len(complex_functions)} highly complex functions",
                "description": "These functions may benefit from refactoring",
                "affected_nodes": [node.id for node in complex_functions[:5]]
            })
        
        return insights


def create_visual_exploration(codebase: Codebase, mode: ExplorationMode = ExplorationMode.STRUCTURAL_OVERVIEW) -> Dict[str, Any]:
    """
    Create a visual exploration of the codebase.
    
    Args:
        codebase: The codebase to explore
        mode: The exploration mode to use
        
    Returns:
        Comprehensive exploration data for visualization
    """
    explorer = VisualCodebaseExplorer(codebase)
    return explorer.explore_codebase(mode)


def analyze_error_blast_radius(codebase: Codebase, symbol_name: str) -> Dict[str, Any]:
    """
    Analyze the blast radius of a specific symbol to understand impact of changes.
    
    Args:
        codebase: The codebase to analyze
        symbol_name: Name of the symbol to analyze
        
    Returns:
        Blast radius analysis data
    """
    explorer = VisualCodebaseExplorer(codebase)
    explorer._build_base_graph()
    
    # Find the target symbol
    target_node = None
    for node in explorer.nodes.values():
        if node.name == symbol_name:
            target_node = node
            break
    
    if not target_node:
        return {"error": f"Symbol '{symbol_name}' not found"}
    
    # Build blast radius graph
    blast_radius_graph = nx.DiGraph()
    visited = set()
    
    def build_blast_radius(node_id: str, depth: int = 0, max_depth: int = 5):
        if depth >= max_depth or node_id in visited:
            return
        
        visited.add(node_id)
        node = explorer.nodes.get(node_id)
        if not node:
            return
        
        blast_radius_graph.add_node(node_id, **node.to_dict())
        
        # Find all nodes that use this symbol
        for edge in explorer.edges:
            if edge.source == node_id and edge.relationship in ["affects", "calls"]:
                blast_radius_graph.add_edge(edge.source, edge.target, **edge.to_dict())
                build_blast_radius(edge.target, depth + 1)
    
    build_blast_radius(target_node.id)
    
    return {
        "target_symbol": {
            "name": target_node.name,
            "type": target_node.type,
            "path": target_node.path,
            "issues": target_node.issues
        },
        "blast_radius": {
            "affected_nodes": len(blast_radius_graph.nodes),
            "affected_edges": len(blast_radius_graph.edges),
            "max_depth": max([
                nx.shortest_path_length(blast_radius_graph, target_node.id, node)
                for node in blast_radius_graph.nodes
                if node != target_node.id and nx.has_path(blast_radius_graph, target_node.id, node)
            ]) if len(blast_radius_graph.nodes) > 1 else 0
        },
        "visual_graph": {
            "nodes": [
                {**explorer.nodes[node_id].to_dict(), "distance": nx.shortest_path_length(blast_radius_graph, target_node.id, node_id) if nx.has_path(blast_radius_graph, target_node.id, node_id) else 0}
                for node_id in blast_radius_graph.nodes
            ],
            "edges": [
                {**edge.to_dict()}
                for edge in explorer.edges
                if edge.source in blast_radius_graph.nodes and edge.target in blast_radius_graph.nodes
            ]
        }
    }

