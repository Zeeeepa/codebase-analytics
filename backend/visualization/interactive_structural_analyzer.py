#!/usr/bin/env python3
"""
Interactive Structural Analyzer

This module provides comprehensive codebase analysis with interactive visualization
capabilities, including detailed error detection, structural tree generation,
and contextual issue reporting similar to the graph-sitter example.

Features:
- Interactive project structural tree with error highlighting
- Comprehensive issue detection and classification
- Contextual error analysis with reasoning
- Entry point identification and usage heat maps
- Inheritance hierarchy analysis
- Real-time issue visualization
"""

import json
import logging
import re
import ast
import inspect
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
    from codegen.sdk.enums import EdgeType, SymbolType
except ImportError:
    print("Codegen SDK not found. Please ensure it's installed.")

from .interactive_adapter import InteractiveAdapter, InteractiveNode, InteractiveEdge

logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    """Error severity levels for structural analysis."""
    CRITICAL = "critical"  # Implementation errors, null references, unsafe operations
    MAJOR = "major"       # Complex methods, incomplete implementations, inefficiencies
    MINOR = "minor"       # Unused parameters, formatting, suboptimal patterns


class ErrorCategory(str, Enum):
    """Categories of errors detected in structural analysis."""
    IMPLEMENTATION_ERROR = "implementation_error"
    TYPE_ERROR = "type_error"
    PARAMETER_ISSUE = "parameter_issue"
    COMPLEXITY_ISSUE = "complexity_issue"
    STYLE_ISSUE = "style_issue"
    DOCUMENTATION_ISSUE = "documentation_issue"
    SECURITY_ISSUE = "security_issue"
    PERFORMANCE_ISSUE = "performance_issue"
    DEPENDENCY_ISSUE = "dependency_issue"


@dataclass
class StructuralError:
    """Represents a structural error in the codebase."""
    file_path: str
    element_type: str  # function, class, method, property, etc.
    element_name: str
    error_type: str
    reasoning: str
    severity: ErrorSeverity
    category: ErrorCategory
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    context: Optional[str] = None
    suggestion: Optional[str] = None
    related_elements: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class StructuralNode:
    """Represents a node in the structural tree."""
    name: str
    type: str  # file, directory, function, class, method
    path: str
    children: List['StructuralNode'] = field(default_factory=list)
    errors: List[StructuralError] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def error_count(self) -> Dict[str, int]:
        """Get count of errors by severity."""
        counts = {"critical": 0, "major": 0, "minor": 0}
        for error in self.errors:
            counts[error.severity.value] += 1
        for child in self.children:
            child_counts = child.error_count
            for severity, count in child_counts.items():
                counts[severity] += count
        return counts

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "type": self.type,
            "path": self.path,
            "children": [child.to_dict() for child in self.children],
            "errors": [error.to_dict() for error in self.errors],
            "error_count": self.error_count,
            "metrics": self.metrics,
            "metadata": self.metadata
        }


class InteractiveStructuralAnalyzer:
    """Comprehensive structural analyzer with interactive visualization."""

    def __init__(self, codebase: Optional[Codebase] = None):
        self.codebase = codebase
        self.errors: List[StructuralError] = []
        self.structural_tree: Optional[StructuralNode] = None
        self.entry_points: List[Dict[str, Any]] = []
        self.usage_heat_map: Dict[str, int] = {}
        self.inheritance_hierarchy: Dict[str, List[str]] = {}
        
        # Error detection patterns
        self.error_patterns = self._initialize_error_patterns()
        
    def _initialize_error_patterns(self) -> Dict[str, Any]:
        """Initialize error detection patterns."""
        return {
            "misspelled_functions": [
                (r"commiter", "committer", "Misspelled function name"),
                (r"recieve", "receive", "Misspelled function name"),
                (r"seperate", "separate", "Misspelled function name"),
            ],
            "unsafe_patterns": [
                (r"assert\s+.*(?:isinstance|type)", "Uses assert for runtime type checking"),
                (r"except\s*:", "Exception handling too broad"),
                (r"\.items\(\)\s*(?!.*if)", "Potential null reference without type checking"),
            ],
            "complexity_indicators": [
                (r"def\s+\w+.*\n(?:.*\n){70,}", "Overly complex method with too many responsibilities"),
                (r"if.*elif.*elif.*elif", "Complex conditional logic"),
                (r"for.*for.*for", "Nested loops indicating complexity"),
            ],
            "style_issues": [
                (r"TODO|FIXME|XXX", "Contains TODOs indicating incomplete implementation"),
                (r"print\(.*\)", "Uses print statements instead of logging"),
                (r"hard.?coded", "Uses hard-coded values"),
            ]
        }

    def analyze_codebase(self, repo_path: str) -> Dict[str, Any]:
        """Perform comprehensive structural analysis of the codebase."""
        logger.info(f"Starting comprehensive structural analysis of {repo_path}")
        
        # Build structural tree
        self.structural_tree = self._build_structural_tree(repo_path)
        
        # Detect errors throughout the codebase
        self._detect_structural_errors()
        
        # Identify entry points and usage patterns
        self._identify_entry_points()
        self._build_usage_heat_map()
        self._build_inheritance_hierarchy()
        
        # Generate comprehensive report
        return self._generate_analysis_report()

    def _build_structural_tree(self, repo_path: str) -> StructuralNode:
        """Build the interactive structural tree."""
        root_path = Path(repo_path)
        root_node = StructuralNode(
            name=root_path.name,
            type="repository",
            path=str(root_path),
            metadata={"is_root": True}
        )
        
        self._build_tree_recursive(root_path, root_node)
        return root_node

    def _build_tree_recursive(self, path: Path, parent_node: StructuralNode):
        """Recursively build the structural tree."""
        try:
            for item in sorted(path.iterdir()):
                if item.name.startswith('.') and item.name not in ['.github', '.vscode']:
                    continue
                    
                if item.is_dir():
                    dir_node = StructuralNode(
                        name=item.name,
                        type="directory",
                        path=str(item)
                    )
                    parent_node.children.append(dir_node)
                    self._build_tree_recursive(item, dir_node)
                    
                elif item.suffix in ['.py', '.js', '.ts', '.tsx', '.jsx']:
                    file_node = self._analyze_file(item)
                    parent_node.children.append(file_node)
                    
        except PermissionError:
            logger.warning(f"Permission denied accessing {path}")

    def _analyze_file(self, file_path: Path) -> StructuralNode:
        """Analyze a single file and create its structural node."""
        file_node = StructuralNode(
            name=file_path.name,
            type="file",
            path=str(file_path)
        )
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            file_node.metrics = self._calculate_file_metrics(content)
            
            # Analyze Python files with AST
            if file_path.suffix == '.py':
                self._analyze_python_file(file_path, content, file_node)
            
            # Detect file-level errors
            file_errors = self._detect_file_errors(file_path, content)
            file_node.errors.extend(file_errors)
            
        except Exception as e:
            logger.warning(f"Error analyzing file {file_path}: {e}")
            file_node.errors.append(StructuralError(
                file_path=str(file_path),
                element_type="file",
                element_name=file_path.name,
                error_type="Analysis error",
                reasoning=f"Could not analyze file: {str(e)}",
                severity=ErrorSeverity.MINOR,
                category=ErrorCategory.IMPLEMENTATION_ERROR
            ))
        
        return file_node

    def _analyze_python_file(self, file_path: Path, content: str, file_node: StructuralNode):
        """Analyze Python file structure using AST."""
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_node = self._analyze_function(node, file_path, content)
                    file_node.children.append(func_node)
                    
                elif isinstance(node, ast.ClassDef):
                    class_node = self._analyze_class(node, file_path, content)
                    file_node.children.append(class_node)
                    
        except SyntaxError as e:
            file_node.errors.append(StructuralError(
                file_path=str(file_path),
                element_type="file",
                element_name=file_path.name,
                error_type="Syntax error",
                reasoning=f"Python syntax error: {str(e)}",
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.IMPLEMENTATION_ERROR,
                line_number=e.lineno
            ))

    def _analyze_function(self, node: ast.FunctionDef, file_path: Path, content: str) -> StructuralNode:
        """Analyze a function and detect issues."""
        func_node = StructuralNode(
            name=node.name,
            type="function",
            path=f"{file_path}:{node.lineno}",
            metadata={
                "line_number": node.lineno,
                "is_method": False,  # Will be updated if inside class
                "parameters": [arg.arg for arg in node.args.args],
                "decorators": [self._get_decorator_name(d) for d in node.decorator_list]
            }
        )
        
        # Calculate function metrics
        func_lines = content.split('\n')[node.lineno-1:node.end_lineno]
        func_content = '\n'.join(func_lines)
        func_node.metrics = self._calculate_function_metrics(func_content, node)
        
        # Detect function-specific errors
        func_errors = self._detect_function_errors(node, file_path, func_content)
        func_node.errors.extend(func_errors)
        
        return func_node

    def _analyze_class(self, node: ast.ClassDef, file_path: Path, content: str) -> StructuralNode:
        """Analyze a class and its methods."""
        class_node = StructuralNode(
            name=node.name,
            type="class",
            path=f"{file_path}:{node.lineno}",
            metadata={
                "line_number": node.lineno,
                "base_classes": [self._get_base_class_name(base) for base in node.bases],
                "decorators": [self._get_decorator_name(d) for d in node.decorator_list]
            }
        )
        
        # Analyze methods within the class
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_node = self._analyze_function(item, file_path, content)
                method_node.type = "method"
                method_node.metadata["is_method"] = True
                method_node.metadata["class_name"] = node.name
                class_node.children.append(method_node)
        
        # Detect class-specific errors
        class_errors = self._detect_class_errors(node, file_path)
        class_node.errors.extend(class_errors)
        
        return class_node

    def _detect_structural_errors(self):
        """Detect structural errors throughout the codebase."""
        if not self.structural_tree:
            return
            
        self._detect_errors_recursive(self.structural_tree)

    def _detect_errors_recursive(self, node: StructuralNode):
        """Recursively detect errors in structural nodes."""
        # Detect node-specific errors based on type
        if node.type == "function":
            self._detect_function_structural_errors(node)
        elif node.type == "class":
            self._detect_class_structural_errors(node)
        elif node.type == "file":
            self._detect_file_structural_errors(node)
            
        # Recurse to children
        for child in node.children:
            self._detect_errors_recursive(child)

    def _detect_function_errors(self, node: ast.FunctionDef, file_path: Path, content: str) -> List[StructuralError]:
        """Detect function-specific errors."""
        errors = []
        
        # Check for unused parameters
        param_names = {arg.arg for arg in node.args.args}
        used_names = set()
        
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                used_names.add(child.id)
        
        unused_params = param_names - used_names - {'self', 'cls'}
        for param in unused_params:
            errors.append(StructuralError(
                file_path=str(file_path),
                element_type="parameter",
                element_name=param,
                error_type="Unused parameter",
                reasoning=f"Parameter '{param}' is defined but never used in the function",
                severity=ErrorSeverity.MINOR,
                category=ErrorCategory.PARAMETER_ISSUE,
                line_number=node.lineno
            ))
        
        # Check for complexity issues
        if node.end_lineno and node.lineno:
            line_count = node.end_lineno - node.lineno
            if line_count > 50:
                errors.append(StructuralError(
                    file_path=str(file_path),
                    element_type="function",
                    element_name=node.name,
                    error_type="Overly complex function",
                    reasoning=f"Function has {line_count} lines, exceeding recommended maximum of 50",
                    severity=ErrorSeverity.MAJOR,
                    category=ErrorCategory.COMPLEXITY_ISSUE,
                    line_number=node.lineno,
                    suggestion="Consider breaking this function into smaller, more focused functions"
                ))
        
        # Check for missing docstring
        if not ast.get_docstring(node):
            errors.append(StructuralError(
                file_path=str(file_path),
                element_type="function",
                element_name=node.name,
                error_type="Missing docstring",
                reasoning="Function lacks documentation",
                severity=ErrorSeverity.MINOR,
                category=ErrorCategory.DOCUMENTATION_ISSUE,
                line_number=node.lineno,
                suggestion="Add a docstring describing the function's purpose, parameters, and return value"
            ))
        
        # Check for implementation errors using patterns
        for pattern, description in self.error_patterns["unsafe_patterns"]:
            if re.search(pattern, content, re.MULTILINE):
                errors.append(StructuralError(
                    file_path=str(file_path),
                    element_type="function",
                    element_name=node.name,
                    error_type="Unsafe pattern",
                    reasoning=description,
                    severity=ErrorSeverity.CRITICAL,
                    category=ErrorCategory.IMPLEMENTATION_ERROR,
                    line_number=node.lineno
                ))
        
        return errors

    def _detect_file_errors(self, file_path: Path, content: str) -> List[StructuralError]:
        """Detect file-level errors."""
        errors = []
        
        # Check for misspelled function names
        for pattern, correct, description in self.error_patterns["misspelled_functions"]:
            if re.search(pattern, content):
                errors.append(StructuralError(
                    file_path=str(file_path),
                    element_type="function",
                    element_name=pattern,
                    error_type="Misspelled function name",
                    reasoning=f"{description} - should be '{correct}' not '{pattern}'",
                    severity=ErrorSeverity.CRITICAL,
                    category=ErrorCategory.IMPLEMENTATION_ERROR,
                    suggestion=f"Rename to '{correct}'"
                ))
        
        # Check for TODOs and incomplete implementations
        todo_matches = re.finditer(r'(TODO|FIXME|XXX).*', content, re.IGNORECASE)
        for match in todo_matches:
            line_num = content[:match.start()].count('\n') + 1
            errors.append(StructuralError(
                file_path=str(file_path),
                element_type="comment",
                element_name="TODO",
                error_type="Incomplete implementation",
                reasoning="Contains TODO indicating incomplete implementation",
                severity=ErrorSeverity.MAJOR,
                category=ErrorCategory.IMPLEMENTATION_ERROR,
                line_number=line_num,
                context=match.group(0).strip()
            ))
        
        return errors

    def _calculate_file_metrics(self, content: str) -> Dict[str, Any]:
        """Calculate metrics for a file."""
        lines = content.split('\n')
        return {
            "total_lines": len(lines),
            "code_lines": len([line for line in lines if line.strip() and not line.strip().startswith('#')]),
            "comment_lines": len([line for line in lines if line.strip().startswith('#')]),
            "blank_lines": len([line for line in lines if not line.strip()]),
            "comment_density": self._calculate_comment_density(content)
        }

    def _calculate_function_metrics(self, content: str, node: ast.FunctionDef) -> Dict[str, Any]:
        """Calculate metrics for a function."""
        complexity = self._calculate_cyclomatic_complexity(node)
        return {
            "cyclomatic_complexity": complexity,
            "parameter_count": len(node.args.args),
            "line_count": node.end_lineno - node.lineno if node.end_lineno else 0,
            "has_docstring": bool(ast.get_docstring(node)),
            "decorator_count": len(node.decorator_list)
        }

    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
                
        return complexity

    def _calculate_comment_density(self, content: str) -> float:
        """Calculate comment density percentage."""
        lines = content.split('\n')
        total_lines = len([line for line in lines if line.strip()])
        comment_lines = len([line for line in lines if line.strip().startswith('#')])
        
        if total_lines == 0:
            return 0.0
        return (comment_lines / total_lines) * 100

    def _identify_entry_points(self):
        """Identify entry points in the codebase."""
        if not self.codebase:
            return
            
        entry_points = []
        
        # Look for main functions, CLI entry points, API endpoints
        for file in self.codebase.files:
            if hasattr(file, 'functions'):
                for func in file.functions:
                    if func.name in ['main', '__main__', 'cli', 'app', 'run']:
                        entry_points.append({
                            "name": func.name,
                            "file": file.path,
                            "type": "main_function",
                            "usage_count": self._count_function_usage(func)
                        })
        
        self.entry_points = entry_points

    def _build_usage_heat_map(self):
        """Build usage heat map for functions and classes."""
        if not self.codebase:
            return
            
        usage_map = defaultdict(int)
        
        # Count function calls and class instantiations
        for file in self.codebase.files:
            if hasattr(file, 'functions'):
                for func in file.functions:
                    if hasattr(func, 'calls'):
                        for call in func.calls:
                            if hasattr(call, 'target') and call.target:
                                usage_map[call.target.name] += 1
        
        self.usage_heat_map = dict(usage_map)

    def _build_inheritance_hierarchy(self):
        """Build inheritance hierarchy map."""
        if not self.codebase:
            return
            
        hierarchy = defaultdict(list)
        
        for file in self.codebase.files:
            if hasattr(file, 'classes'):
                for cls in file.classes:
                    if hasattr(cls, 'base_classes'):
                        for base in cls.base_classes:
                            hierarchy[base.name].append(cls.name)
        
        self.inheritance_hierarchy = dict(hierarchy)

    def _generate_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        total_errors = len(self.errors)
        error_by_severity = Counter(error.severity.value for error in self.errors)
        error_by_category = Counter(error.category.value for error in self.errors)
        
        # Generate detailed error list
        detailed_errors = []
        for i, error in enumerate(self.errors, 1):
            detailed_errors.append({
                "index": i,
                "file_location": error.file_path,
                "error_place": f"{error.element_type}: {error.element_name}",
                "reasoning": error.reasoning,
                "severity": error.severity.value,
                "category": error.category.value,
                "line": error.line_number,
                "suggestion": error.suggestion
            })
        
        return {
            "repository_info": {
                "name": self.structural_tree.name if self.structural_tree else "Unknown",
                "total_files": self._count_nodes_by_type("file"),
                "total_functions": self._count_nodes_by_type("function"),
                "total_classes": self._count_nodes_by_type("class"),
                "total_errors": total_errors
            },
            "error_summary": {
                "by_severity": dict(error_by_severity),
                "by_category": dict(error_by_category),
                "critical_count": error_by_severity.get("critical", 0),
                "major_count": error_by_severity.get("major", 0),
                "minor_count": error_by_severity.get("minor", 0)
            },
            "structural_tree": self.structural_tree.to_dict() if self.structural_tree else None,
            "detailed_errors": detailed_errors,
            "entry_points": self.entry_points,
            "usage_heat_map": self.usage_heat_map,
            "inheritance_hierarchy": self.inheritance_hierarchy,
            "interactive_config": {
                "supports_drill_down": True,
                "supports_filtering": True,
                "supports_search": True,
                "error_highlighting": True
            }
        }

    def _count_nodes_by_type(self, node_type: str) -> int:
        """Count nodes of a specific type in the structural tree."""
        if not self.structural_tree:
            return 0
        return self._count_nodes_recursive(self.structural_tree, node_type)

    def _count_nodes_recursive(self, node: StructuralNode, target_type: str) -> int:
        """Recursively count nodes of a specific type."""
        count = 1 if node.type == target_type else 0
        for child in node.children:
            count += self._count_nodes_recursive(child, target_type)
        return count

    def _get_decorator_name(self, decorator) -> str:
        """Get decorator name from AST node."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        return "unknown"

    def _get_base_class_name(self, base) -> str:
        """Get base class name from AST node."""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return base.attr
        return "unknown"

    def _count_function_usage(self, func: Function) -> int:
        """Count how many times a function is used."""
        return self.usage_heat_map.get(func.name, 0)

    def _detect_function_structural_errors(self, node: StructuralNode):
        """Detect structural errors specific to functions."""
        # Add function-specific structural error detection
        pass

    def _detect_class_structural_errors(self, node: StructuralNode):
        """Detect structural errors specific to classes."""
        # Add class-specific structural error detection
        pass

    def _detect_file_structural_errors(self, node: StructuralNode):
        """Detect structural errors specific to files."""
        # Add file-specific structural error detection
        pass

    def _detect_class_errors(self, node: ast.ClassDef, file_path: Path) -> List[StructuralError]:
        """Detect class-specific errors."""
        errors = []
        
        # Check for missing docstring
        if not ast.get_docstring(node):
            errors.append(StructuralError(
                file_path=str(file_path),
                element_type="class",
                element_name=node.name,
                error_type="Missing docstring",
                reasoning="Class lacks documentation",
                severity=ErrorSeverity.MINOR,
                category=ErrorCategory.DOCUMENTATION_ISSUE,
                line_number=node.lineno
            ))
        
        return errors

    def generate_interactive_visualization(self) -> Dict[str, Any]:
        """Generate data for interactive visualization."""
        if not self.structural_tree:
            return {}
            
        adapter = InteractiveAdapter()
        
        # Create interactive nodes for the structural tree
        nodes = self._create_interactive_nodes(self.structural_tree)
        
        # Create edges for relationships
        edges = self._create_interactive_edges()
        
        return {
            "type": "structural_analysis",
            "nodes": nodes,
            "edges": edges,
            "config": {
                "interactive": True,
                "error_highlighting": True,
                "drill_down_enabled": True,
                "search_enabled": True,
                "filter_enabled": True,
                "layout": "hierarchical"
            },
            "metadata": {
                "total_errors": len(self.errors),
                "analysis_timestamp": datetime.now().isoformat()
            }
        }

    def _create_interactive_nodes(self, node: StructuralNode, parent_id: str = None) -> List[Dict[str, Any]]:
        """Create interactive nodes from structural tree."""
        nodes = []
        
        node_id = f"{node.type}_{hash(node.path)}"
        error_counts = node.error_count
        
        interactive_node = {
            "id": node_id,
            "label": node.name,
            "type": node.type,
            "path": node.path,
            "parent_id": parent_id,
            "error_count": error_counts,
            "severity_color": self._get_severity_color(error_counts),
            "metrics": node.metrics,
            "metadata": node.metadata,
            "interactions": ["click", "hover", "drill_down"],
            "errors": [error.to_dict() for error in node.errors]
        }
        
        nodes.append(interactive_node)
        
        # Add child nodes
        for child in node.children:
            child_nodes = self._create_interactive_nodes(child, node_id)
            nodes.extend(child_nodes)
        
        return nodes

    def _create_interactive_edges(self) -> List[Dict[str, Any]]:
        """Create interactive edges for relationships."""
        edges = []
        
        # Add inheritance relationships
        for parent, children in self.inheritance_hierarchy.items():
            for child in children:
                edges.append({
                    "source": f"class_{hash(parent)}",
                    "target": f"class_{hash(child)}",
                    "type": "inheritance",
                    "weight": 1.0,
                    "color": "#8b5cf6",
                    "interactions": ["hover"]
                })
        
        return edges

    def _get_severity_color(self, error_counts: Dict[str, int]) -> str:
        """Get color based on error severity."""
        if error_counts.get("critical", 0) > 0:
            return "#ef4444"  # Red
        elif error_counts.get("major", 0) > 0:
            return "#f59e0b"  # Orange
        elif error_counts.get("minor", 0) > 0:
            return "#eab308"  # Yellow
        else:
            return "#22c55e"  # Green


def analyze_repository_structure(repo_path: str, codebase: Optional[Codebase] = None) -> Dict[str, Any]:
    """Main function to analyze repository structure with interactive visualization."""
    analyzer = InteractiveStructuralAnalyzer(codebase)
    
    # Perform comprehensive analysis
    analysis_result = analyzer.analyze_codebase(repo_path)
    
    # Generate interactive visualization data
    visualization_data = analyzer.generate_interactive_visualization()
    
    # Combine results
    return {
        **analysis_result,
        "interactive_visualization": visualization_data,
        "analysis_metadata": {
            "analyzer_version": "1.0.0",
            "analysis_type": "comprehensive_structural",
            "timestamp": datetime.now().isoformat()
        }
    }

