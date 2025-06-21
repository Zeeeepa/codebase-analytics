#!/usr/bin/env python3
"""
Comprehensive Codebase Analysis Engine

This module provides a unified interface for advanced codebase analysis, combining
functionality from multiple analyzers and incorporating features from graph-sitter
and codegen research.

Features:
- Interactive repository structure analysis
- Issue detection and classification
- Call graph and dependency analysis
- Code quality metrics
- Performance analysis
- Visualization capabilities
- API integration for headless analysis

Based on research from:
- graph-sitter.com codebase visualization tutorials
- codegen examples and documentation
- Advanced analysis patterns and best practices
"""

import json
import logging
import math
import os
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import networkx as nx
except ImportError:
    nx = None

# Import from Codegen SDK
from codegen.sdk.core.codebase import Codebase
from codegen.sdk.core.class_definition import Class
from codegen.sdk.core.external_module import ExternalModule
from codegen.sdk.core.file import SourceFile
from codegen.sdk.core.function import Function
from codegen.sdk.core.import_resolution import Import
from codegen.sdk.core.symbol import Symbol
from codegen.sdk.enums import EdgeType, SymbolType
from codegen.shared.enums.programming_language import ProgrammingLanguage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


#######################################################
# Enums and Data Classes
#######################################################

class AnalysisType(str, Enum):
    """Types of analysis that can be performed."""
    CODEBASE = "codebase"
    INTERACTIVE_TREE = "interactive_tree"
    ISSUE_DETECTION = "issue_detection"
    CALL_GRAPH = "call_graph"
    DEPENDENCY_ANALYSIS = "dependency_analysis"
    CODE_QUALITY = "code_quality"
    PERFORMANCE = "performance"
    VISUALIZATION = "visualization"
    COMPREHENSIVE = "comprehensive"


class IssueSeverity(str, Enum):
    """Severity levels for issues."""
    CRITICAL = "critical"  # ⚠️ Must be fixed immediately
    MAJOR = "major"       # 👉 Should be fixed soon
    MINOR = "minor"       # 🔍 Could be improved
    INFO = "info"         # ℹ️ Informational


class IssueCategory(str, Enum):
    """Categories of issues that can be detected."""
    DEAD_CODE = "dead_code"
    COMPLEXITY = "complexity"
    STYLE_ISSUE = "style_issue"
    TYPE_ERROR = "type_error"
    PARAMETER_MISMATCH = "parameter_mismatch"
    IMPLEMENTATION_ERROR = "implementation_error"
    SECURITY_ISSUE = "security_issue"
    PERFORMANCE_ISSUE = "performance_issue"
    DEPENDENCY_ISSUE = "dependency_issue"


@dataclass
class Issue:
    """Represents a code issue."""
    id: str
    severity: IssueSeverity
    category: IssueCategory
    message: str
    file_path: str
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class FileAnalysis:
    """Analysis results for a single file."""
    path: str
    issues: List[Issue]
    symbols: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    dependencies: List[str]


@dataclass
class RepositoryStructure:
    """Interactive repository structure with issue counts."""
    name: str
    type: str  # 'file' or 'directory'
    path: str
    children: List['RepositoryStructure'] = field(default_factory=list)
    issue_counts: Dict[IssueSeverity, int] = field(default_factory=dict)
    symbols: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Complete analysis results."""
    repository_structure: RepositoryStructure
    total_issues: Dict[IssueSeverity, int]
    file_analyses: List[FileAnalysis]
    call_graph: Optional[Dict[str, Any]] = None
    dependency_graph: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


#######################################################
# Core Analysis Engine
#######################################################

class ComprehensiveCodebaseAnalyzer:
    """
    Comprehensive codebase analyzer that combines multiple analysis techniques
    and provides interactive visualization capabilities.
    """
    
    def __init__(self, codebase: Codebase):
        self.codebase = codebase
        self.issues: List[Issue] = []
        self.call_graph = nx.DiGraph() if nx else None
        self.dependency_graph = nx.DiGraph() if nx else None
        self.file_analyses: Dict[str, FileAnalysis] = {}
        
    def analyze(self, analysis_types: List[AnalysisType] = None) -> AnalysisResult:
        """
        Perform comprehensive analysis of the codebase.
        
        Args:
            analysis_types: List of analysis types to perform. If None, performs all.
            
        Returns:
            AnalysisResult containing all analysis data
        """
        if analysis_types is None:
            analysis_types = list(AnalysisType)
            
        logger.info(f"Starting comprehensive analysis with types: {analysis_types}")
        
        # Initialize results
        self.issues = []
        self.file_analyses = {}
        
        # Perform different types of analysis
        if AnalysisType.ISSUE_DETECTION in analysis_types:
            self._detect_issues()
            
        if AnalysisType.CALL_GRAPH in analysis_types:
            self._build_call_graph()
            
        if AnalysisType.DEPENDENCY_ANALYSIS in analysis_types:
            self._analyze_dependencies()
            
        if AnalysisType.CODE_QUALITY in analysis_types:
            self._analyze_code_quality()
            
        if AnalysisType.PERFORMANCE in analysis_types:
            self._analyze_performance()
            
        # Build repository structure
        repository_structure = self._build_repository_structure()
        
        # Calculate total issue counts
        total_issues = self._calculate_total_issues()
        
        # Build comprehensive metrics
        metrics = self._build_comprehensive_metrics()
        
        return AnalysisResult(
            repository_structure=repository_structure,
            total_issues=total_issues,
            file_analyses=list(self.file_analyses.values()),
            call_graph=self._serialize_graph(self.call_graph) if self.call_graph else None,
            dependency_graph=self._serialize_graph(self.dependency_graph) if self.dependency_graph else None,
            metrics=metrics
        )
    
    def _detect_issues(self):
        """Detect various types of issues in the codebase."""
        logger.info("Detecting issues...")
        
        for file in self.codebase.files:
            file_issues = []
            
            # Analyze functions in the file
            for function in file.functions:
                # Check for unused parameters
                if hasattr(function, 'parameters'):
                    for param in function.parameters:
                        if not self._is_parameter_used(function, param.name):
                            issue = Issue(
                                id=f"{file.path}:{function.name}:unused_param:{param.name}",
                                severity=IssueSeverity.MINOR,
                                category=IssueCategory.DEAD_CODE,
                                message=f"Unused parameter '{param.name}'",
                                file_path=file.path,
                                function_name=function.name,
                                context={"parameter": param.name}
                            )
                            file_issues.append(issue)
                
                # Check for complexity issues
                complexity = self._calculate_cyclomatic_complexity(function)
                if complexity > 10:
                    issue = Issue(
                        id=f"{file.path}:{function.name}:high_complexity",
                        severity=IssueSeverity.MAJOR if complexity > 15 else IssueSeverity.MINOR,
                        category=IssueCategory.COMPLEXITY,
                        message=f"High cyclomatic complexity: {complexity}",
                        file_path=file.path,
                        function_name=function.name,
                        context={"complexity": complexity}
                    )
                    file_issues.append(issue)
            
            # Analyze classes in the file
            for class_def in file.classes:
                # Check for unused methods
                for method in class_def.methods:
                    if not self._is_method_used(class_def, method):
                        issue = Issue(
                            id=f"{file.path}:{class_def.name}:{method.name}:unused_method",
                            severity=IssueSeverity.MINOR,
                            category=IssueCategory.DEAD_CODE,
                            message=f"Unused method '{method.name}'",
                            file_path=file.path,
                            class_name=class_def.name,
                            function_name=method.name
                        )
                        file_issues.append(issue)
            
            # Store file analysis
            symbols = self._extract_file_symbols(file)
            metrics = self._calculate_file_metrics(file)
            dependencies = self._extract_file_dependencies(file)
            
            self.file_analyses[file.path] = FileAnalysis(
                path=file.path,
                issues=file_issues,
                symbols=symbols,
                metrics=metrics,
                dependencies=dependencies
            )
            
            self.issues.extend(file_issues)
    
    def _build_call_graph(self):
        """Build call graph for the codebase."""
        if not self.call_graph:
            return
            
        logger.info("Building call graph...")
        
        for file in self.codebase.files:
            for function in file.functions:
                function_id = f"{file.path}:{function.name}"
                self.call_graph.add_node(function_id, **{
                    'file': file.path,
                    'function': function.name,
                    'type': 'function'
                })
                
                # Find function calls within this function
                calls = self._find_function_calls(function)
                for called_function in calls:
                    called_id = f"{called_function['file']}:{called_function['name']}"
                    self.call_graph.add_edge(function_id, called_id)
    
    def _analyze_dependencies(self):
        """Analyze dependencies between files and modules."""
        if not self.dependency_graph:
            return
            
        logger.info("Analyzing dependencies...")
        
        for file in self.codebase.files:
            file_id = file.path
            self.dependency_graph.add_node(file_id, **{
                'path': file.path,
                'type': 'file'
            })
            
            # Add import dependencies
            for import_stmt in file.imports:
                if hasattr(import_stmt, 'module_path') and import_stmt.module_path:
                    self.dependency_graph.add_edge(file_id, import_stmt.module_path)
    
    def _analyze_code_quality(self):
        """Analyze code quality metrics."""
        logger.info("Analyzing code quality...")
        
        for file in self.codebase.files:
            # Calculate various quality metrics
            metrics = {
                'lines_of_code': len(file.content.split('\n')) if hasattr(file, 'content') else 0,
                'function_count': len(file.functions),
                'class_count': len(file.classes),
                'import_count': len(file.imports),
                'complexity_score': self._calculate_file_complexity(file)
            }
            
            if file.path in self.file_analyses:
                self.file_analyses[file.path].metrics.update(metrics)
    
    def _analyze_performance(self):
        """Analyze potential performance issues."""
        logger.info("Analyzing performance...")
        
        for file in self.codebase.files:
            performance_issues = []
            
            for function in file.functions:
                # Check for potential performance issues
                if self._has_nested_loops(function):
                    issue = Issue(
                        id=f"{file.path}:{function.name}:nested_loops",
                        severity=IssueSeverity.MINOR,
                        category=IssueCategory.PERFORMANCE_ISSUE,
                        message="Function contains nested loops - consider optimization",
                        file_path=file.path,
                        function_name=function.name
                    )
                    performance_issues.append(issue)
            
            if file.path in self.file_analyses:
                self.file_analyses[file.path].issues.extend(performance_issues)
            
            self.issues.extend(performance_issues)
    
    def _build_repository_structure(self) -> RepositoryStructure:
        """Build interactive repository structure with issue counts."""
        logger.info("Building repository structure...")
        
        # Create root structure
        root = RepositoryStructure(
            name=self.codebase.name or "Repository",
            type="directory",
            path="",
            issue_counts={severity: 0 for severity in IssueSeverity}
        )
        
        # Build file tree
        file_tree = {}
        for file in self.codebase.files:
            parts = file.path.split('/')
            current = file_tree
            
            for i, part in enumerate(parts):
                if part not in current:
                    current[part] = {
                        'type': 'directory' if i < len(parts) - 1 else 'file',
                        'children': {},
                        'issues': [],
                        'symbols': [],
                        'metrics': {}
                    }
                current = current[part]['children']
            
            # Add file analysis data
            if file.path in self.file_analyses:
                analysis = self.file_analyses[file.path]
                file_node = file_tree
                for part in parts:
                    file_node = file_node[part]
                    if part == parts[-1]:  # This is the file
                        file_node['issues'] = analysis.issues
                        file_node['symbols'] = analysis.symbols
                        file_node['metrics'] = analysis.metrics
        
        # Convert tree to RepositoryStructure
        def build_structure(tree_node: dict, name: str, path: str) -> RepositoryStructure:
            structure = RepositoryStructure(
                name=name,
                type=tree_node['type'],
                path=path,
                issue_counts={severity: 0 for severity in IssueSeverity}
            )
            
            if tree_node['type'] == 'file':
                structure.symbols = tree_node.get('symbols', [])
                structure.metrics = tree_node.get('metrics', {})
                
                # Count issues by severity
                for issue in tree_node.get('issues', []):
                    structure.issue_counts[issue.severity] += 1
            else:
                # Build children
                for child_name, child_node in tree_node['children'].items():
                    child_path = f"{path}/{child_name}" if path else child_name
                    child_structure = build_structure(child_node, child_name, child_path)
                    structure.children.append(child_structure)
                    
                    # Aggregate issue counts from children
                    for severity, count in child_structure.issue_counts.items():
                        structure.issue_counts[severity] += count
            
            return structure
        
        # Build the complete structure
        for name, node in file_tree.items():
            child_structure = build_structure(node, name, name)
            root.children.append(child_structure)
            
            # Aggregate to root
            for severity, count in child_structure.issue_counts.items():
                root.issue_counts[severity] += count
        
        return root
    
    def _calculate_total_issues(self) -> Dict[IssueSeverity, int]:
        """Calculate total issue counts by severity."""
        counts = {severity: 0 for severity in IssueSeverity}
        for issue in self.issues:
            counts[issue.severity] += 1
        return counts
    
    def _build_comprehensive_metrics(self) -> Dict[str, Any]:
        """Build comprehensive metrics for the codebase."""
        metrics = {
            'total_files': len(self.codebase.files),
            'total_functions': sum(len(file.functions) for file in self.codebase.files),
            'total_classes': sum(len(file.classes) for file in self.codebase.files),
            'total_lines': sum(len(file.content.split('\n')) if hasattr(file, 'content') else 0 
                             for file in self.codebase.files),
            'average_complexity': self._calculate_average_complexity(),
            'most_called_functions': self._get_most_called_functions(),
            'dead_code_percentage': self._calculate_dead_code_percentage(),
            'dependency_depth': self._calculate_dependency_depth()
        }
        
        return metrics
    
    # Helper methods for analysis
    def _is_parameter_used(self, function: Function, param_name: str) -> bool:
        """Check if a parameter is used within a function."""
        # This is a simplified implementation
        # In a real implementation, you'd analyze the function body
        if hasattr(function, 'body') and function.body:
            return param_name in function.body
        return True  # Assume used if we can't analyze
    
    def _calculate_cyclomatic_complexity(self, function: Function) -> int:
        """Calculate cyclomatic complexity of a function."""
        # Simplified complexity calculation
        # In practice, you'd count decision points (if, while, for, etc.)
        complexity = 1  # Base complexity
        
        if hasattr(function, 'body') and function.body:
            # Count decision points
            decision_keywords = ['if', 'elif', 'while', 'for', 'try', 'except', 'and', 'or']
            for keyword in decision_keywords:
                complexity += function.body.count(keyword)
        
        return complexity
    
    def _is_method_used(self, class_def: Class, method: Function) -> bool:
        """Check if a method is used within the class or externally."""
        # Simplified implementation
        # In practice, you'd check for calls to this method
        return not method.name.startswith('_')  # Assume private methods might be unused
    
    def _extract_file_symbols(self, file: SourceFile) -> List[Dict[str, Any]]:
        """Extract symbols (functions, classes, variables) from a file."""
        symbols = []
        
        # Add functions
        for function in file.functions:
            symbols.append({
                'name': function.name,
                'type': 'function',
                'line_number': getattr(function, 'line_number', None),
                'parameters': [p.name for p in function.parameters] if hasattr(function, 'parameters') else [],
                'return_type': getattr(function, 'return_type', None)
            })
        
        # Add classes
        for class_def in file.classes:
            class_symbol = {
                'name': class_def.name,
                'type': 'class',
                'line_number': getattr(class_def, 'line_number', None),
                'methods': []
            }
            
            # Add methods
            for method in class_def.methods:
                class_symbol['methods'].append({
                    'name': method.name,
                    'type': 'method',
                    'line_number': getattr(method, 'line_number', None),
                    'parameters': [p.name for p in method.parameters] if hasattr(method, 'parameters') else []
                })
            
            symbols.append(class_symbol)
        
        return symbols
    
    def _calculate_file_metrics(self, file: SourceFile) -> Dict[str, Any]:
        """Calculate metrics for a single file."""
        return {
            'lines_of_code': len(file.content.split('\n')) if hasattr(file, 'content') else 0,
            'function_count': len(file.functions),
            'class_count': len(file.classes),
            'import_count': len(file.imports),
            'complexity_score': sum(self._calculate_cyclomatic_complexity(f) for f in file.functions)
        }
    
    def _extract_file_dependencies(self, file: SourceFile) -> List[str]:
        """Extract dependencies for a file."""
        dependencies = []
        for import_stmt in file.imports:
            if hasattr(import_stmt, 'module_path') and import_stmt.module_path:
                dependencies.append(import_stmt.module_path)
        return dependencies
    
    def _find_function_calls(self, function: Function) -> List[Dict[str, str]]:
        """Find function calls within a function."""
        # Simplified implementation
        # In practice, you'd parse the AST to find function calls
        calls = []
        if hasattr(function, 'body') and function.body:
            # This is a very basic pattern matching approach
            # A real implementation would use AST parsing
            import re
            call_pattern = r'(\w+)\s*\('
            matches = re.findall(call_pattern, function.body)
            for match in matches:
                calls.append({
                    'name': match,
                    'file': 'unknown'  # Would need more sophisticated analysis
                })
        return calls
    
    def _calculate_file_complexity(self, file: SourceFile) -> float:
        """Calculate overall complexity score for a file."""
        total_complexity = sum(self._calculate_cyclomatic_complexity(f) for f in file.functions)
        return total_complexity / max(len(file.functions), 1)
    
    def _has_nested_loops(self, function: Function) -> bool:
        """Check if a function has nested loops."""
        if hasattr(function, 'body') and function.body:
            # Simple check for nested loops
            loop_keywords = ['for', 'while']
            lines = function.body.split('\n')
            loop_depth = 0
            max_depth = 0
            
            for line in lines:
                stripped = line.strip()
                if any(stripped.startswith(keyword) for keyword in loop_keywords):
                    loop_depth += 1
                    max_depth = max(max_depth, loop_depth)
                elif stripped in ['', 'pass'] or stripped.startswith('#'):
                    continue
                else:
                    # Simplified: assume any other line ends a loop level
                    loop_depth = max(0, loop_depth - 1)
            
            return max_depth > 1
        return False
    
    def _serialize_graph(self, graph) -> Dict[str, Any]:
        """Serialize NetworkX graph to JSON-serializable format."""
        if not graph:
            return {}
        
        return {
            'nodes': [
                {'id': node, **data} 
                for node, data in graph.nodes(data=True)
            ],
            'edges': [
                {'source': source, 'target': target, **data}
                for source, target, data in graph.edges(data=True)
            ]
        }
    
    def _calculate_average_complexity(self) -> float:
        """Calculate average cyclomatic complexity across all functions."""
        total_complexity = 0
        total_functions = 0
        
        for file in self.codebase.files:
            for function in file.functions:
                total_complexity += self._calculate_cyclomatic_complexity(function)
                total_functions += 1
        
        return total_complexity / max(total_functions, 1)
    
    def _get_most_called_functions(self) -> List[Dict[str, Any]]:
        """Get the most frequently called functions."""
        if not self.call_graph:
            return []
        
        # Calculate in-degree (how many times each function is called)
        call_counts = {}
        for node in self.call_graph.nodes():
            call_counts[node] = self.call_graph.in_degree(node)
        
        # Sort by call count and return top 10
        sorted_functions = sorted(call_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return [
            {
                'function': func,
                'call_count': count,
                'file': func.split(':')[0] if ':' in func else 'unknown'
            }
            for func, count in sorted_functions
        ]
    
    def _calculate_dead_code_percentage(self) -> float:
        """Calculate percentage of potentially dead code."""
        dead_code_issues = [issue for issue in self.issues 
                           if issue.category == IssueCategory.DEAD_CODE]
        total_symbols = sum(len(file.functions) + len(file.classes) for file in self.codebase.files)
        
        return (len(dead_code_issues) / max(total_symbols, 1)) * 100
    
    def _calculate_dependency_depth(self) -> int:
        """Calculate maximum dependency depth."""
        if not self.dependency_graph:
            return 0
        
        try:
            # Find the longest path in the dependency graph
            return max(len(path) for path in nx.all_simple_paths(
                self.dependency_graph, 
                source=list(self.dependency_graph.nodes())[0],
                target=list(self.dependency_graph.nodes())[-1]
            )) if self.dependency_graph.nodes() else 0
        except:
            return 0


#######################################################
# API Integration Functions
#######################################################

def analyze_repository(repo_path: str, analysis_types: List[AnalysisType] = None) -> AnalysisResult:
    """
    Analyze a repository and return comprehensive results.
    
    Args:
        repo_path: Path to the repository
        analysis_types: Types of analysis to perform
        
    Returns:
        AnalysisResult with all analysis data
    """
    try:
        # Load codebase using Codegen SDK
        codebase = Codebase.from_path(repo_path)
        
        # Create analyzer and perform analysis
        analyzer = ComprehensiveCodebaseAnalyzer(codebase)
        result = analyzer.analyze(analysis_types)
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing repository: {e}")
        raise


def analyze_repository_to_text(repo_path: str, output_file: str = None) -> str:
    """
    Analyze repository and return results as formatted text.
    
    Args:
        repo_path: Path to the repository
        output_file: Optional file to save results
        
    Returns:
        Formatted text analysis results
    """
    result = analyze_repository(repo_path)
    
    # Format results as text
    text_output = []
    text_output.append("# Codebase Analysis Report")
    text_output.append(f"Generated: {result.timestamp}")
    text_output.append("")
    
    # Summary
    text_output.append("## Summary")
    text_output.append(f"- Total Files: {result.metrics.get('total_files', 0)}")
    text_output.append(f"- Total Functions: {result.metrics.get('total_functions', 0)}")
    text_output.append(f"- Total Classes: {result.metrics.get('total_classes', 0)}")
    text_output.append(f"- Total Lines: {result.metrics.get('total_lines', 0)}")
    text_output.append("")
    
    # Issues Summary
    text_output.append("## Issues Summary")
    for severity, count in result.total_issues.items():
        emoji = {"critical": "⚠️", "major": "👉", "minor": "🔍", "info": "ℹ️"}.get(severity, "•")
        text_output.append(f"- {emoji} {severity.title()}: {count}")
    text_output.append("")
    
    # Repository Structure
    text_output.append("## Repository Structure")
    
    def format_structure(structure: RepositoryStructure, indent: int = 0) -> List[str]:
        lines = []
        prefix = "  " * indent
        
        if structure.type == "directory":
            icon = "📁"
        else:
            icon = "📄"
        
        # Format issue counts
        issue_summary = []
        for severity, count in structure.issue_counts.items():
            if count > 0:
                emoji = {"critical": "⚠️", "major": "👉", "minor": "🔍", "info": "ℹ️"}.get(severity, "•")
                issue_summary.append(f"{emoji} {severity.title()}: {count}")
        
        issue_text = f" [{', '.join(issue_summary)}]" if issue_summary else ""
        lines.append(f"{prefix}{icon} {structure.name}{issue_text}")
        
        # Add children
        for child in structure.children:
            lines.extend(format_structure(child, indent + 1))
        
        return lines
    
    text_output.extend(format_structure(result.repository_structure))
    text_output.append("")
    
    # Detailed Issues
    if result.file_analyses:
        text_output.append("## Detailed Issues")
        for file_analysis in result.file_analyses:
            if file_analysis.issues:
                text_output.append(f"### {file_analysis.path}")
                for issue in file_analysis.issues:
                    emoji = {"critical": "⚠️", "major": "👉", "minor": "🔍", "info": "ℹ️"}.get(issue.severity.value, "•")
                    location = f":{issue.line_number}" if issue.line_number else ""
                    function_info = f" in {issue.function_name}()" if issue.function_name else ""
                    text_output.append(f"- {emoji} {issue.message}{location}{function_info}")
                text_output.append("")
    
    # Metrics
    text_output.append("## Metrics")
    for key, value in result.metrics.items():
        if isinstance(value, list):
            text_output.append(f"- {key.replace('_', ' ').title()}: {len(value)} items")
        elif isinstance(value, (int, float)):
            text_output.append(f"- {key.replace('_', ' ').title()}: {value}")
    
    formatted_text = "\n".join(text_output)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(formatted_text)
        logger.info(f"Analysis results saved to {output_file}")
    
    return formatted_text


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Codebase Analysis")
    parser.add_argument("--repo", required=True, help="Repository path")
    parser.add_argument("--output", help="Output file for text results")
    parser.add_argument("--format", choices=["json", "text"], default="text", help="Output format")
    
    args = parser.parse_args()
    
    if args.format == "json":
        result = analyze_repository(args.repo)
        output = json.dumps(asdict(result), indent=2, default=str)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
        else:
            print(output)
    else:
        output = analyze_repository_to_text(args.repo, args.output)
        if not args.output:
            print(output)
