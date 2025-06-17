#!/usr/bin/env python3
"""
Consolidated Analysis Module

This module contains ALL analysis functions consolidated from:
- analysis.py (existing analysis functions)
- advanced_analysis.py (comprehensive analysis features)
- analyzer.py (legacy analysis functions)
- comprehensive_analysis.py (additional analysis features)
- API analysis functions from api.py and enhanced_api.py

Organized into logical sections for maintainability.
"""

import math
import re
import os
import ast
import networkx as nx
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from collections import Counter, defaultdict, deque
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import uuid
import inspect
import time
import json

# Import from Codegen SDK
from codegen.sdk.core.codebase import Codebase
from codegen.sdk.core.class_definition import Class
from codegen.sdk.core.file import SourceFile
from codegen.sdk.core.function import Function
from codegen.sdk.core.symbol import Symbol
from codegen.sdk.core.statements.for_loop_statement import ForLoopStatement
from codegen.sdk.core.statements.if_block_statement import IfBlockStatement
from codegen.sdk.core.statements.try_catch_statement import TryCatchStatement
from codegen.sdk.core.statements.while_statement import WhileStatement
from codegen.sdk.core.expressions.binary_expression import BinaryExpression
from codegen.sdk.core.expressions.unary_expression import UnaryExpression
from codegen.sdk.core.expressions.comparison_expression import ComparisonExpression
from codegen.sdk.enums import EdgeType, SymbolType
from codegen.sdk.core.import_resolution import Import
from codegen.sdk.core.external_module import ExternalModule

# ============================================================================
# SECTION 1: DATA CLASSES AND ENUMS
# ============================================================================

@dataclass
class InheritanceAnalysis:
    """Analysis of class inheritance patterns."""
    deepest_class_name: Optional[str] = None
    deepest_class_depth: int = 0
    inheritance_chain: List[str] = None
    
    def __post_init__(self):
        if self.inheritance_chain is None:
            self.inheritance_chain = []

@dataclass
class RecursionAnalysis:
    """Analysis of recursive functions."""
    recursive_functions: List[str] = None
    total_recursive_count: int = 0
    
    def __post_init__(self):
        if self.recursive_functions is None:
            self.recursive_functions = []

@dataclass
class SymbolInfo:
    """Information about a code symbol."""
    id: str
    name: str
    type: str  # 'function', 'class', or 'variable'
    filepath: str
    start_line: int
    end_line: int
    issues: Optional[List[Dict[str, str]]] = None

@dataclass
class DependencyAnalysis:
    """Comprehensive dependency analysis results."""
    total_dependencies: int = 0
    circular_dependencies: List[List[str]] = field(default_factory=list)
    dependency_depth: int = 0
    external_dependencies: List[str] = field(default_factory=list)
    internal_dependencies: List[str] = field(default_factory=list)
    dependency_graph: Dict[str, List[str]] = field(default_factory=dict)
    critical_dependencies: List[str] = field(default_factory=list)
    unused_dependencies: List[str] = field(default_factory=list)

@dataclass
class CallGraphAnalysis:
    """Call graph analysis results."""
    total_call_relationships: int = 0
    call_depth: int = 0
    call_graph: Dict[str, List[str]] = field(default_factory=dict)
    entry_points: List[str] = field(default_factory=list)
    leaf_functions: List[str] = field(default_factory=list)
    most_connected_functions: List[Tuple[str, int]] = field(default_factory=list)
    call_chains: List[List[str]] = field(default_factory=list)

@dataclass
class CodeQualityMetrics:
    """Advanced code quality metrics."""
    technical_debt_ratio: float = 0.0
    code_duplication_percentage: float = 0.0
    test_coverage_estimate: float = 0.0
    documentation_coverage: float = 0.0
    naming_consistency_score: float = 0.0
    architectural_violations: List[str] = field(default_factory=list)
    code_smells: List[Dict[str, Any]] = field(default_factory=list)
    refactoring_opportunities: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ArchitecturalInsights:
    """Architectural analysis insights."""
    architectural_patterns: List[str] = field(default_factory=list)
    layer_violations: List[str] = field(default_factory=list)
    coupling_metrics: Dict[str, float] = field(default_factory=dict)
    cohesion_metrics: Dict[str, float] = field(default_factory=dict)
    modularity_score: float = 0.0
    component_analysis: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SecurityAnalysis:
    """Security-focused code analysis."""
    potential_vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    security_hotspots: List[str] = field(default_factory=list)
    input_validation_issues: List[str] = field(default_factory=list)
    authentication_patterns: List[str] = field(default_factory=list)
    encryption_usage: List[str] = field(default_factory=list)

@dataclass
class PerformanceAnalysis:
    """Performance-related code analysis."""
    performance_hotspots: List[Dict[str, Any]] = field(default_factory=list)
    algorithmic_complexity: Dict[str, str] = field(default_factory=dict)
    memory_usage_patterns: List[str] = field(default_factory=list)
    optimization_opportunities: List[Dict[str, Any]] = field(default_factory=list)

class AnalysisType(Enum):
    """Types of analysis that can be performed."""
    DEPENDENCY = "dependency"
    CALL_GRAPH = "call_graph"
    CODE_QUALITY = "code_quality"
    ARCHITECTURAL = "architectural"
    SECURITY = "security"
    PERFORMANCE = "performance"

# Legacy enums from analyzer.py
class IssueSeverity(str, Enum):
    """Severity levels for issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IssueCategory(str, Enum):
    """Categories of issues."""
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    RELIABILITY = "reliability"
    STYLE = "style"
    DEAD_CODE = "dead_code"
    COMPLEXITY = "complexity"
    STYLE_ISSUE = "style_issue"
    DOCUMENTATION = "documentation"
    TYPE_ERROR = "type_error"
    PARAMETER_MISMATCH = "parameter_mismatch"
    RETURN_TYPE_ERROR = "return_type_error"
    IMPLEMENTATION_ERROR = "implementation_error"
    MISSING_IMPLEMENTATION = "missing_implementation"
    IMPORT_ERROR = "import_error"
    DEPENDENCY_CYCLE = "dependency_cycle"

class IssueStatus(str, Enum):
    """Status of an issue."""
    OPEN = "open"
    FIXED = "fixed"
    WONTFIX = "wontfix"
    INVALID = "invalid"
    DUPLICATE = "duplicate"

class ChangeType(str, Enum):
    """Type of change for a diff."""
    Added = "added"
    Removed = "removed"
    Modified = "modified"
    Renamed = "renamed"

class TransactionPriority(int, Enum):
    """Priority levels for transactions."""
    HIGH = 0
    MEDIUM = 5
    LOW = 10

@dataclass
class CodeLocation:
    """Location of an issue in code."""
    file: str
    line: Optional[int] = None
    column: Optional[int] = None
    end_line: Optional[int] = None
    end_column: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeLocation":
        """Create from dictionary representation."""
        return cls(**{k: v for k, v in data.items() if k in inspect.signature(cls).parameters})

@dataclass
class Issue:
    """Represents an issue found during analysis."""
    # Core fields
    message: str
    severity: IssueSeverity
    location: CodeLocation

    # Classification fields
    category: Optional[IssueCategory] = None
    analysis_type: Optional[AnalysisType] = None
    status: IssueStatus = IssueStatus.OPEN

    # Context fields
    symbol: Optional[str] = None
    code: Optional[str] = None
    suggestion: Optional[str] = None
    related_symbols: List[str] = field(default_factory=list)
    related_locations: List[CodeLocation] = field(default_factory=list)

    # Metadata fields
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: Optional[str] = None
    resolved_at: Optional[str] = None
    resolved_by: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {}
        for k, v in asdict(self).items():
            if k == "location" and v is not None:
                result[k] = v.to_dict()
            elif k == "related_locations" and v:
                result[k] = [loc.to_dict() for loc in v]
            elif v is not None:
                if isinstance(v, Enum):
                    result[k] = v.value
                else:
                    result[k] = v
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Issue":
        """Create from dictionary representation."""
        # Handle nested objects
        if "location" in data and data["location"]:
            data["location"] = CodeLocation.from_dict(data["location"])
        if "related_locations" in data and data["related_locations"]:
            data["related_locations"] = [CodeLocation.from_dict(loc) for loc in data["related_locations"]]
        
        # Handle enums
        if "severity" in data and not isinstance(data["severity"], IssueSeverity):
            data["severity"] = IssueSeverity(data["severity"])
        if "category" in data and data["category"] and not isinstance(data["category"], IssueCategory):
            data["category"] = IssueCategory(data["category"])
        if "status" in data and not isinstance(data["status"], IssueStatus):
            data["status"] = IssueStatus(data["status"])
        if "analysis_type" in data and data["analysis_type"] and not isinstance(data["analysis_type"], AnalysisType):
            data["analysis_type"] = AnalysisType(data["analysis_type"])
        
        return cls(**{k: v for k, v in data.items() if k in inspect.signature(cls).parameters})

class IssueCollection:
    """Collection of issues with filtering and grouping capabilities."""

    def __init__(self, issues: Optional[List[Issue]] = None):
        """
        Initialize the issue collection.

        Args:
            issues: Initial list of issues
        """
        self.issues = issues or []
        self._filters = []

    def add_issue(self, issue: Issue):
        """
        Add an issue to the collection.

        Args:
            issue: Issue to add
        """
        self.issues.append(issue)

    def add_issues(self, issues: List[Issue]):
        """
        Add multiple issues to the collection.

        Args:
            issues: Issues to add
        """
        self.issues.extend(issues)

    def filter(self, **kwargs) -> "IssueCollection":
        """
        Filter issues by attributes.

        Args:
            **kwargs: Attribute-value pairs to filter by

        Returns:
            A new IssueCollection with filtered issues
        """
        filtered_issues = []
        for issue in self.issues:
            match = True
            for key, value in kwargs.items():
                if not hasattr(issue, key) or getattr(issue, key) != value:
                    match = False
                    break
            if match:
                filtered_issues.append(issue)
        return IssueCollection(filtered_issues)

    def group_by(self, attribute: str) -> Dict[Any, "IssueCollection"]:
        """
        Group issues by an attribute.

        Args:
            attribute: Attribute to group by

        Returns:
            Dictionary mapping attribute values to IssueCollections
        """
        groups = {}
        for issue in self.issues:
            if hasattr(issue, attribute):
                key = getattr(issue, attribute)
                if isinstance(key, Enum):
                    key = key.value
                if key not in groups:
                    groups[key] = IssueCollection()
                groups[key].add_issue(issue)
        return groups

    def count(self) -> int:
        """
        Count the number of issues.

        Returns:
            Number of issues
        """
        return len(self.issues)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.

        Returns:
            Dictionary representation
        """
        return {
            "count": self.count(),
            "issues": [issue.to_dict() for issue in self.issues]
        }

    def to_list(self) -> List[Dict[str, Any]]:
        """
        Convert to list representation.

        Returns:
            List of issue dictionaries
        """
        return [issue.to_dict() for issue in self.issues]

@dataclass
class AnalysisSummary:
    """Summary statistics for an analysis."""
    total_files: int = 0
    total_functions: int = 0
    total_classes: int = 0
    total_issues: int = 0
    analysis_time: str = field(default_factory=lambda: datetime.now().isoformat())
    analysis_duration_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {k: v for k, v in asdict(self).items() if v is not None}

@dataclass
class CodeQualityResult:
    """Results of code quality analysis."""
    dead_code: Dict[str, Any] = field(default_factory=dict)
    complexity: Dict[str, Any] = field(default_factory=dict)
    parameter_issues: Dict[str, Any] = field(default_factory=dict)
    style_issues: Dict[str, Any] = field(default_factory=dict)
    implementation_issues: Dict[str, Any] = field(default_factory=dict)
    maintainability: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return dict(asdict(self).items())

@dataclass
class DependencyResult:
    """Results of dependency analysis."""
    import_dependencies: Dict[str, Any] = field(default_factory=dict)
    circular_dependencies: Dict[str, Any] = field(default_factory=dict)
    module_coupling: Dict[str, Any] = field(default_factory=dict)
    external_dependencies: Dict[str, Any] = field(default_factory=dict)
    call_graph: Dict[str, Any] = field(default_factory=dict)
    class_hierarchy: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return dict(asdict(self).items())

@dataclass
class AnalysisResult:
    """Comprehensive analysis result."""
    # Core data
    analysis_types: List[AnalysisType]
    summary: AnalysisSummary = field(default_factory=AnalysisSummary)
    issues: IssueCollection = field(default_factory=IssueCollection)

    # Analysis results
    code_quality: Optional[CodeQualityResult] = None
    dependencies: Optional[DependencyResult] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    repo_name: Optional[str] = None
    repo_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {}
        for k, v in asdict(self).items():
            if v is not None:
                if k == "analysis_types":
                    result[k] = [t.value for t in v]
                elif k == "summary":
                    result[k] = v.to_dict()
                elif k == "issues":
                    result[k] = v.to_dict()
                elif k in ["code_quality", "dependencies"] and v is not None:
                    result[k] = v.to_dict()
                else:
                    result[k] = v
        return result

class CodebaseAnalyzer:
    """
    Comprehensive code analyzer for detecting issues and analyzing codebase structure.
    
    This class provides a unified interface for analyzing codebases, integrating
    functionality from various analyzer modules to detect code issues, analyze
    dependencies, and provide insights into code quality and structure.
    """

    def __init__(
        self,
        repo_url: Optional[str] = None,
        repo_path: Optional[str] = None,
        base_branch: str = "main",
        language: Optional[str] = None,
        file_ignore_list: Optional[List[str]] = None,
    ):
        """
        Initialize the analyzer.
        
        Args:
            repo_url: URL of the repository to analyze
            repo_path: Path to the repository to analyze
            base_branch: Base branch to compare against
            language: Primary language of the codebase
            file_ignore_list: List of file patterns to ignore
        """
        self.repo_url = repo_url
        self.repo_path = repo_path
        self.base_branch = base_branch
        self.language = language
        self.file_ignore_list = file_ignore_list or []
        self.codebase = None
        self.issues = IssueCollection()
        
    def load_codebase(self, codebase: Optional[Codebase] = None) -> Codebase:
        """
        Load a codebase for analysis.
        
        Args:
            codebase: Optional pre-loaded Codebase object
            
        Returns:
            The loaded Codebase object
        """
        if codebase:
            self.codebase = codebase
            return codebase
            
        if not self.codebase:
            # This is a placeholder - in a real implementation, we would use the SDK to load the codebase
            self.codebase = Codebase()
            
        return self.codebase
    
    def analyze(self, analysis_types: Optional[List[AnalysisType]] = None) -> AnalysisResult:
        """
        Analyze the codebase.
        
        Args:
            analysis_types: Types of analysis to perform
            
        Returns:
            Analysis results
        """
        if not self.codebase:
            self.load_codebase()
            
        if not analysis_types:
            analysis_types = list(AnalysisType)
            
        # Create result object
        result = AnalysisResult(
            analysis_types=analysis_types,
            repo_name=self.repo_url.split("/")[-1] if self.repo_url else None,
            repo_path=self.repo_path,
        )
        
        # Perform analysis
        start_time = datetime.now()
        
        # Update summary
        result.summary.total_files = len(list(self.codebase.files)) if hasattr(self.codebase, 'files') else 0
        result.summary.total_functions = len(list(self.codebase.functions)) if hasattr(self.codebase, 'functions') else 0
        result.summary.total_classes = len(list(self.codebase.classes)) if hasattr(self.codebase, 'classes') else 0
        
        # Perform specific analyses
        if AnalysisType.CODE_QUALITY in analysis_types:
            result.code_quality = self._analyze_code_quality()
            
        if AnalysisType.DEPENDENCY in analysis_types:
            result.dependencies = self._analyze_dependencies()
            
        # Calculate duration
        end_time = datetime.now()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)
        result.summary.analysis_duration_ms = duration_ms
        
        # Add issues
        result.issues = self.issues
        result.summary.total_issues = self.issues.count()
        
        return result
    
    def _analyze_code_quality(self) -> CodeQualityResult:
        """
        Analyze code quality.
        
        Returns:
            Code quality analysis results
        """
        # This is a placeholder implementation
        return CodeQualityResult()
    
    def _analyze_dependencies(self) -> DependencyResult:
        """
        Analyze dependencies.
        
        Returns:
            Dependency analysis results
        """
        # This is a placeholder implementation
        return DependencyResult()

    def _analyze_import_dependencies(self) -> Dict[str, Any]:
        """
        Analyze import dependencies between files.

        Returns:
            Dictionary containing import dependency analysis results
        """
        print("Analyzing import dependencies")

        # Generate dependency graph
        dependency_graph = self._get_dependency_graph()

        # Count dependencies per file
        dependency_counts = {
            file_path: len(deps) for file_path, deps in dependency_graph.items()
        }

        # Find files with high number of dependencies
        high_dependency_files = [
            {"file": file_path, "dependency_count": count}
            for file_path, count in dependency_counts.items()
            if count > 10  # Threshold for high dependency count
        ]

        # Sort by dependency count (descending)
        high_dependency_files.sort(key=lambda x: x["dependency_count"], reverse=True)

        # Generate import dependency analysis results
        import_dependencies = {
            "dependency_graph": dependency_graph,
            "high_dependency_files": high_dependency_files[:10],  # Top 10
            "summary": {
                "total_files": len(dependency_graph),
                "avg_dependencies": sum(dependency_counts.values()) / len(dependency_counts) if dependency_counts else 0,
                "max_dependencies": max(dependency_counts.values()) if dependency_counts else 0,
            },
        }

        return import_dependencies

    def _get_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Generate a dependency graph for the codebase.

        Returns:
            Dictionary mapping file paths to lists of imported file paths
        """
        dependency_graph = {}

        # Build dependency graph
        for file in self.codebase.files:
            file_path = file.path
            dependency_graph[file_path] = []

            # Add imported files to dependencies
            if hasattr(file, "imports"):
                for imp in file.imports:
                    # Skip standard library imports
                    if not hasattr(imp, "file") or not imp.file:
                        continue

                    # Add imported file to dependencies
                    imported_file_path = imp.file.path
                    if imported_file_path != file_path:  # Skip self-imports
                        dependency_graph[file_path].append(imported_file_path)

        return dependency_graph

    def _find_circular_dependencies(self) -> Dict[str, Any]:
        """
        Find circular dependencies in the codebase.

        Returns:
            Dictionary containing circular dependency analysis results
        """
        print("Analyzing circular dependencies")

        # Get dependency graph
        dependency_graph = self._get_dependency_graph()
        
        # Find circular dependencies
        circular_deps = []
        visited = set()
        
        def detect_cycles(node, path):
            if node in path:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                
                # Add to circular dependencies if not already found
                cycle_str = " -> ".join(cycle)
                if not any(cycle_str in cd["cycle_str"] for cd in circular_deps):
                    circular_deps.append({
                        "cycle": cycle,
                        "cycle_str": cycle_str,
                        "length": len(cycle),
                    })
                return
            
            if node in visited:
                return
                
            visited.add(node)
            new_path = path + [node]
            
            # Check dependencies
            if node in dependency_graph:
                for dep in dependency_graph[node]:
                    detect_cycles(dep, new_path)
            
            visited.remove(node)
        
        # Check each file for cycles
        for file_path in dependency_graph:
            detect_cycles(file_path, [])
        
        # Sort by cycle length
        circular_deps.sort(key=lambda x: x["length"])
        
        # Generate circular dependency analysis results
        circular_dependencies = {
            "circular_dependencies": circular_deps,
            "summary": {
                "total_cycles": len(circular_deps),
                "max_cycle_length": max([cd["length"] for cd in circular_deps]) if circular_deps else 0,
            },
        }
        
        return circular_dependencies

    def _generate_html_report(self, output_file: Optional[str] = None):
        """
        Generate an HTML report of the analysis results.
        
        Args:
            output_file: Path to save the report to
        """
        if not hasattr(self, "analysis_result") or not self.analysis_result:
            raise ValueError("No analysis results to save")
            
        if not output_file:
            output_file = "codebase_analysis_report.html"
        
        # Simple HTML template
        repo_name = self.repo_url.split("/")[-1] if self.repo_url else (self.repo_path or "Unknown")
        analysis_time = self.analysis_result.summary.analysis_time
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Codebase Analysis Report: {repo_name}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .summary {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{
                    padding: 12px 15px;
                    border-bottom: 1px solid #ddd;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .severity-high {{
                    color: #e74c3c;
                }}
                .severity-medium {{
                    color: #f39c12;
                }}
                .severity-low {{
                    color: #3498db;
                }}
                .severity-info {{
                    color: #2ecc71;
                }}
                .footer {{
                    margin-top: 30px;
                    text-align: center;
                    font-size: 0.8em;
                    color: #7f8c8d;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Codebase Analysis Report: {repo_name}</h1>
                <div class="summary">
                    <h2>Summary</h2>
                    <p><strong>Analysis Time:</strong> {analysis_time}</p>
                    <p><strong>Language:</strong> {self.language or 'Auto-detected'}</p>
                    <p><strong>Total Files:</strong> {self.analysis_result.summary.total_files}</p>
                    <p><strong>Total Classes:</strong> {self.analysis_result.summary.total_classes}</p>
                    <p><strong>Total Functions:</strong> {self.analysis_result.summary.total_functions}</p>
                </div>
        """
        
        # Add code quality section if available
        if hasattr(self.analysis_result, "code_quality"):
            html += self._generate_code_quality_html()
            
        # Add dependency section if available
        if hasattr(self.analysis_result, "dependencies"):
            html += self._generate_dependencies_html()
            
        # Add performance section if available
        if hasattr(self.analysis_result, "performance"):
            html += self._generate_performance_html()
            
        # Add footer
        html += f"""
                <div class="footer">
                    <p>Generated by Codebase Analyzer on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save to file
        with open(output_file, "w") as f:
            f.write(html)
            
        print(f"HTML report saved to: {output_file}")
        
    def _generate_code_quality_html(self) -> str:
        """Generate HTML for code quality section."""
        html = """
                <h2>Code Quality Analysis</h2>
        """
        
        # Add dead code section
        if hasattr(self.analysis_result.code_quality, "dead_code"):
            dead_code = self.analysis_result.code_quality.dead_code
            html += f"""
                <h3>Dead Code</h3>
                <p>Found {dead_code['summary']['unused_functions_count']} unused functions, 
                   {dead_code['summary']['unused_classes_count']} unused classes, and 
                   {dead_code['summary']['unused_imports_count']} unused imports.</p>
            """
            
            if dead_code["unused_functions"]:
                html += """
                <h4>Unused Functions</h4>
                <table>
                    <tr>
                        <th>Function</th>
                        <th>File</th>
                        <th>Line</th>
                    </tr>
                """
                
                for func in dead_code["unused_functions"][:10]:  # Show top 10
                    html += f"""
                    <tr>
                        <td>{func['name']}</td>
                        <td>{func['file']}</td>
                        <td>{func['line'] or 'N/A'}</td>
                    </tr>
                    """
                    
                html += "</table>"
                
                if len(dead_code["unused_functions"]) > 10:
                    html += f"<p>And {len(dead_code['unused_functions']) - 10} more...</p>"
        
        # Add parameter issues section
        if hasattr(self.analysis_result.code_quality, "parameter_issues"):
            param_issues = self.analysis_result.code_quality.parameter_issues
            html += f"""
                <h3>Parameter Issues</h3>
                <p>Found {param_issues['summary']['unused_parameters_count']} unused parameters, 
                   {param_issues['summary']['missing_types_count']} parameters missing type annotations, and 
                   {param_issues['summary']['incorrect_usage_count']} incorrect usages.</p>
            """
        
        # Add implementation issues section
        if hasattr(self.analysis_result.code_quality, "implementation_issues"):
            impl_issues = self.analysis_result.code_quality.implementation_issues
            html += f"""
                <h3>Implementation Issues</h3>
                <p>Found {impl_issues['summary']['empty_functions_count']} empty functions and 
                   {impl_issues['summary']['abstract_methods_without_implementation_count']} unimplemented abstract methods.</p>
            """
            
        return html
        
    def _generate_dependencies_html(self) -> str:
        """Generate HTML for dependencies section."""
        html = """
                <h2>Dependency Analysis</h2>
        """
        
        # Add import dependencies section
        if hasattr(self.analysis_result.dependencies, "import_dependencies"):
            import_deps = self.analysis_result.dependencies.import_dependencies
            html += f"""
                <h3>Import Dependencies</h3>
                <p>Analyzed {import_deps['summary']['total_files']} files with an average of 
                   {import_deps['summary']['avg_dependencies']:.2f} dependencies per file.</p>
            """
            
            if import_deps["high_dependency_files"]:
                html += """
                <h4>Files with High Dependencies</h4>
                <table>
                    <tr>
                        <th>File</th>
                        <th>Dependency Count</th>
                    </tr>
                """
                
                for file in import_deps["high_dependency_files"]:
                    html += f"""
                    <tr>
                        <td>{file['file']}</td>
                        <td>{file['dependency_count']}</td>
                    </tr>
                    """
                    
                html += "</table>"
        
        # Add circular dependencies section
        if hasattr(self.analysis_result.dependencies, "circular_dependencies"):
            circular_deps = self.analysis_result.dependencies.circular_dependencies
            html += f"""
                <h3>Circular Dependencies</h3>
                <p>Found {circular_deps['summary']['total_cycles']} cycles.</p>
            """
            
            if circular_deps["circular_dependencies"]:
                html += """
                <h4>Circular Dependency Cycles</h4>
                <table>
                    <tr>
                        <th>Cycle</th>
                        <th>Length</th>
                    </tr>
                """
                
                for cycle in circular_deps["circular_dependencies"]:
                    html += f"""
                    <tr>
                        <td>{cycle['cycle_str']}</td>
                        <td>{cycle['length']}</td>
                    </tr>
                    """
                    
                html += "</table>"
            
        return html
        
    def _generate_performance_html(self) -> str:
        """Generate HTML for performance section."""
        html = """
                <h2>Performance Analysis</h2>
        """
        
        # Add complexity section
        if hasattr(self.analysis_result.performance, "complexity"):
            complexity = self.analysis_result.performance.complexity
            html += f"""
                <h3>Code Complexity</h3>
                <p>Average cyclomatic complexity: {complexity['summary']['avg_complexity']:.2f}</p>
            """
            
            if complexity.get("high_complexity_functions"):
                html += """
                <h4>High Complexity Functions</h4>
                <table>
                    <tr>
                        <th>Function</th>
                        <th>File</th>
                        <th>Complexity</th>
                    </tr>
                """
                
                for func in complexity["high_complexity_functions"]:
                    html += f"""
                    <tr>
                        <td>{func['name']}</td>
                        <td>{func['file']}</td>
                        <td>{func['complexity']}</td>
                    </tr>
                    """
                    
                html += "</table>"
            
        return html
        
    def _print_console_report(self):
        """Print a summary report to the console."""
        if not hasattr(self, "analysis_result") or not self.analysis_result:
            raise ValueError("No analysis results to print")
            
        repo_name = self.repo_url.split("/")[-1] if self.repo_url else (self.repo_path or "Unknown")
        
        print(f"\n{'=' * 80}")
        print(f"CODEBASE ANALYSIS REPORT: {repo_name}")
        print(f"{'=' * 80}")
        print(f"Analysis Time: {self.analysis_result.summary.analysis_time}")
        print(f"Language: {self.language or 'Auto-detected'}")
        
        # Print summary
        print(f"\n{'-' * 40}")
        print("SUMMARY:")
        print(f"{'-' * 40}")
        print(f"Total Files: {self.analysis_result.summary.total_files}")
        print(f"Total Classes: {self.analysis_result.summary.total_classes}")
        print(f"Total Functions: {self.analysis_result.summary.total_functions}")
        print(f"Total Issues: {self.analysis_result.summary.total_issues}")
        
        # Print code quality summary
        if hasattr(self.analysis_result, "code_quality"):
            print(f"\n{'-' * 40}")
            print("CODE QUALITY:")
            print(f"{'-' * 40}")
            
            # Print dead code summary
            if hasattr(self.analysis_result.code_quality, "dead_code"):
                dead_code = self.analysis_result.code_quality.dead_code
                print(f"Dead Code: {dead_code['summary']['unused_functions_count']} unused functions, "
                      f"{dead_code['summary']['unused_classes_count']} unused classes, "
                      f"{dead_code['summary']['unused_imports_count']} unused imports")
            
            # Print parameter issues summary
            if hasattr(self.analysis_result.code_quality, "parameter_issues"):
                param_issues = self.analysis_result.code_quality.parameter_issues
                print(f"Parameter Issues: {param_issues['summary']['unused_parameters_count']} unused parameters, "
                      f"{param_issues['summary']['missing_types_count']} missing type annotations, "
                      f"{param_issues['summary']['incorrect_usage_count']} incorrect usages")
            
            # Print implementation issues summary
            if hasattr(self.analysis_result.code_quality, "implementation_issues"):
                impl_issues = self.analysis_result.code_quality.implementation_issues
                print(f"Implementation Issues: {impl_issues['summary']['empty_functions_count']} empty functions, "
                      f"{impl_issues['summary']['abstract_methods_without_implementation_count']} unimplemented abstract methods")
        
        # Print dependency summary
        if hasattr(self.analysis_result, "dependencies"):
            print(f"\n{'-' * 40}")
            print("DEPENDENCIES:")
            print(f"{'-' * 40}")
            
            # Print import dependencies summary
            if hasattr(self.analysis_result.dependencies, "import_dependencies"):
                import_deps = self.analysis_result.dependencies.import_dependencies
                print(f"Import Dependencies: {import_deps['summary']['total_files']} files, "
                      f"{import_deps['summary']['avg_dependencies']:.2f} avg dependencies per file")
            
            # Print circular dependencies summary
            if hasattr(self.analysis_result.dependencies, "circular_dependencies"):
                circular_deps = self.analysis_result.dependencies.circular_dependencies
                print(f"Circular Dependencies: {circular_deps['summary']['total_cycles']} cycles found")
        
        # Print performance summary
        if hasattr(self.analysis_result, "performance"):
            print(f"\n{'-' * 40}")
            print("PERFORMANCE:")
            print(f"{'-' * 40}")
            
            # Print complexity summary
            if hasattr(self.analysis_result.performance, "complexity"):
                complexity = self.analysis_result.performance.complexity
                print(f"Code Complexity: {complexity['summary']['avg_complexity']:.2f} avg cyclomatic complexity")
                print(f"High Complexity Functions: {len(complexity.get('high_complexity_functions', []))}")
        
        print(f"\n{'=' * 80}")

# ============================================================================
# SECTION 2: CORE COMPLEXITY AND METRICS FUNCTIONS
# ============================================================================

def calculate_cyclomatic_complexity(function: Function) -> int:
    """Calculate cyclomatic complexity for a function."""
    def analyze_statement(statement):
        complexity = 0

        if isinstance(statement, IfBlockStatement):
            complexity += 1
            if hasattr(statement, "elif_statements"):
                complexity += len(statement.elif_statements)

        elif isinstance(statement, (ForLoopStatement, WhileStatement)):
            complexity += 1

        elif isinstance(statement, TryCatchStatement):
            complexity += len(getattr(statement, "except_blocks", []))

        if hasattr(statement, "condition") and isinstance(statement.condition, str):
            complexity += statement.condition.count(" and ") + statement.condition.count(" or ")

        if hasattr(statement, "nested_code_blocks"):
            for block in statement.nested_code_blocks:
                complexity += analyze_block(block)

        return complexity

    def analyze_block(block):
        if not block or not hasattr(block, "statements"):
            return 0
        return sum(analyze_statement(stmt) for stmt in block.statements)

    return 1 + analyze_block(function.code_block) if hasattr(function, "code_block") else 1

# ============================================================================
# SECTION 6: MISSING FUNCTIONS FROM ANALYZER.PY
# ============================================================================

def create_issue(
    message: str,
    severity: Union[str, IssueSeverity],
    file: str,
    line: Optional[int] = None,
    category: Optional[Union[str, IssueCategory]] = None,
    symbol: Optional[str] = None,
    suggestion: Optional[str] = None,
) -> Issue:
    """
    Create an issue with simplified parameters.

    Args:
        message: Issue message
        severity: Issue severity
        file: File path
        line: Line number
        category: Issue category
        symbol: Symbol name
        suggestion: Suggested fix

    Returns:
        Created issue
    """
    # Convert string severity to enum if needed
    if isinstance(severity, str):
        severity = IssueSeverity(severity)

    # Convert string category to enum if needed
    if isinstance(category, str) and category:
        category = IssueCategory(category)

    # Create location
    location = CodeLocation(file=file, line=line)

    # Create and return issue
    return Issue(
        message=message,
        severity=severity,
        location=location,
        category=category,
        symbol=symbol,
        suggestion=suggestion,
    )

def get_codebase_summary(codebase: Codebase) -> str:
    """
    Generate a comprehensive summary of a codebase.
    
    Args:
        codebase: The Codebase object to summarize
        
    Returns:
        A formatted string containing a summary of the codebase's nodes and edges
    """
    node_summary = f"""Contains {len(codebase.ctx.get_nodes()) if hasattr(codebase, 'ctx') else 0} nodes
- {len(list(codebase.files)) if hasattr(codebase, 'files') else 0} files
- {len(list(codebase.imports)) if hasattr(codebase, 'imports') else 0} imports
- {len(list(codebase.external_modules)) if hasattr(codebase, 'external_modules') else 0} external_modules
- {len(list(codebase.symbols)) if hasattr(codebase, 'symbols') else 0} symbols
\t- {len(list(codebase.classes)) if hasattr(codebase, 'classes') else 0} classes
\t- {len(list(codebase.functions)) if hasattr(codebase, 'functions') else 0} functions
\t- {len([s for s in codebase.symbols if s.symbol_type == SymbolType.VARIABLE]) if hasattr(codebase, 'symbols') else 0} variables"""

    edge_summary = f"""Contains {len(codebase.ctx.get_edges()) if hasattr(codebase, 'ctx') else 0} edges
- {len([e for e in codebase.ctx.get_edges() if e.edge_type == EdgeType.CONTAINS]) if hasattr(codebase, 'ctx') else 0} CONTAINS edges
- {len([e for e in codebase.ctx.get_edges() if e.edge_type == EdgeType.IMPORTS]) if hasattr(codebase, 'ctx') else 0} IMPORTS edges
- {len([e for e in codebase.ctx.get_edges() if e.edge_type == EdgeType.CALLS]) if hasattr(codebase, 'ctx') else 0} CALLS edges
- {len([e for e in codebase.ctx.get_edges() if e.edge_type == EdgeType.INHERITS_FROM]) if hasattr(codebase, 'ctx') else 0} INHERITS_FROM edges
- {len([e for e in codebase.ctx.get_edges() if e.edge_type == EdgeType.REFERENCES]) if hasattr(codebase, 'ctx') else 0} REFERENCES edges"""

    return f"{node_summary}\n\n{edge_summary}"

def get_dependency_graph(codebase: Codebase, file_path: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Generate a dependency graph for a codebase or specific file.
    
    Args:
        codebase: The Codebase object to analyze
        file_path: Optional path to a specific file to analyze
        
    Returns:
        A dictionary mapping file paths to lists of dependencies
    """
    dependency_graph = {}
    
    files_to_analyze = [f for f in codebase.files if not file_path or f.filepath == file_path]
    
    for file in files_to_analyze:
        dependencies = []
        
        # Get imports from the file
        if hasattr(file, 'imports') and file.imports:
            for imp in file.imports:
                if hasattr(imp, 'source_file') and imp.source_file:
                    dependencies.append(imp.source_file.filepath)
                elif hasattr(imp, 'module'):
                    dependencies.append(imp.module)
                elif hasattr(imp, 'name'):
                    dependencies.append(imp.name)
        
        dependency_graph[file.filepath] = dependencies
    
    return dependency_graph

def get_symbol_references(codebase: Codebase, symbol_name: str) -> List[Dict[str, Any]]:
    """
    Find all references to a symbol in the codebase.
    
    Args:
        codebase: The Codebase object to search
        symbol_name: The name of the symbol to find references for
        
    Returns:
        A list of dictionaries containing reference information
    """
    references = []
    
    # Find all symbols with the given name
    target_symbols = [s for s in codebase.symbols if s.name == symbol_name]
    
    if not target_symbols:
        return references
    
    # For each symbol, find references
    for symbol in target_symbols:
        # Get direct references from the graph
        if hasattr(codebase, 'ctx'):
            reference_edges = [
                e for e in codebase.ctx.get_edges() 
                if e.edge_type == EdgeType.REFERENCES and e.target_id == symbol.id
            ]
            
            for edge in reference_edges:
                source_node = codebase.ctx.get_node(edge.source_id)
                if source_node:
                    reference = {
                        "symbol": symbol_name,
                        "referenced_by": source_node.name,
                        "referenced_by_type": source_node.node_type,
                        "file": source_node.filepath if hasattr(source_node, 'filepath') else None,
                        "line": edge.metadata.get("line") if hasattr(edge, 'metadata') else None,
                    }
                    references.append(reference)
    
    # If no references found through graph, try text-based search
    if not references:
        for file in codebase.files:
            if not hasattr(file, 'source') or not file.source:
                continue
                
            lines = file.source.split('\n')
            for i, line in enumerate(lines):
                if symbol_name in line:
                    # Simple check to avoid false positives
                    if re.search(r'\b' + re.escape(symbol_name) + r'\b', line):
                        reference = {
                            "symbol": symbol_name,
                            "referenced_by": file.filepath,
                            "referenced_by_type": "file",
                            "file": file.filepath,
                            "line": i + 1,
                            "context": line.strip(),
                        }
                        references.append(reference)
    
    return references

def analyze_codebase(
    repo_path: Optional[str] = None,
    repo_url: Optional[str] = None,
    output_file: Optional[str] = None,
    analysis_types: Optional[List[str]] = None,
    language: Optional[str] = None,
    output_format: str = "json",
) -> AnalysisResult:
    """
    Analyze a codebase and optionally save results to a file.
    
    Args:
        repo_path: Path to the repository to analyze
        repo_url: URL of the repository to analyze
        output_file: Optional path to save results to
        analysis_types: Optional list of analysis types to perform
        language: Optional language to filter by
        output_format: Output format (json or text)
        
    Returns:
        Analysis results
    """
    # Convert string analysis types to enums
    analysis_type_enums = []
    if analysis_types:
        for analysis_type in analysis_types:
            try:
                analysis_type_enums.append(AnalysisType(analysis_type))
            except ValueError:
                pass
    
    if not analysis_type_enums:
        analysis_type_enums = list(AnalysisType)
    
    # Create analyzer
    analyzer = CodebaseAnalyzer(
        repo_url=repo_url,
        repo_path=repo_path,
        language=language,
    )
    
    # Load codebase
    codebase = None
    
    # This is a placeholder - in a real implementation, we would use the SDK to load the codebase
    # For now, we'll create a dummy codebase if repo_path is provided
    if repo_path:
        from codegen.sdk.codebase.loader import load_codebase
        try:
            codebase = load_codebase(repo_path)
        except Exception as e:
            print(f"Error loading codebase: {e}")
            codebase = Codebase()
    else:
        codebase = Codebase()
    
    # Analyze codebase
    analyzer.load_codebase(codebase)
    result = analyzer.analyze(analysis_type_enums)
    
    # Save results if output file is provided
    if output_file:
        result_dict = result.to_dict()
        
        if output_format == "json":
            import json
            with open(output_file, "w") as f:
                json.dump(result_dict, f, indent=2)
        else:
            with open(output_file, "w") as f:
                f.write(str(result_dict))
    
    return result

# ============================================================================
# SECTION 7: COMPREHENSIVE ANALYSIS FUNCTIONS
# ============================================================================

class IssueType(Enum):
    """Types of issues that can be detected in code analysis."""
    UNUSED_FUNCTION = auto()
    UNUSED_CLASS = auto()
    UNUSED_IMPORT = auto()
    UNUSED_PARAMETER = auto()
    PARAMETER_MISMATCH = auto()
    MISSING_TYPE_ANNOTATION = auto()
    CIRCULAR_DEPENDENCY = auto()
    EMPTY_FUNCTION = auto()
    IMPLEMENTATION_ERROR = auto()

class ComprehensiveAnalyzer:
    """
    Comprehensive analyzer for codebases using the Codegen SDK.
    Implements deep analysis of code issues, dependencies, and metrics.
    """
    
    def __init__(self, repo_path_or_url: str):
        """
        Initialize the analyzer with a repository path or URL.
        
        Args:
            repo_path_or_url: Path to local repo or URL to GitHub repo
        """
        self.repo_path_or_url = repo_path_or_url
        self.issues: List[Issue] = []
        self.start_time = time.time()
        self.codebase = None
        
    def analyze(self) -> Dict[str, Any]:
        """
        Perform a comprehensive analysis of the codebase.
        
        Returns:
            Dictionary with analysis results
        """
        print(f"Starting comprehensive analysis of {self.repo_path_or_url}...")
        
        # Initialize codebase
        try:
            print(f"Initializing codebase from {self.repo_path_or_url}")
            if self.repo_path_or_url.startswith(("http://", "https://")):
                # Extract repo name for GitHub URLs
                parts = self.repo_path_or_url.rstrip('/').split('/')
                repo_name = f"{parts[-2]}/{parts[-1]}"
                try:
                    self.codebase = Codebase.from_repo(repo_full_name=repo_name)
                    print(f"Successfully initialized codebase from GitHub repository: {repo_name}")
                except Exception as e:
                    print(f"Error initializing codebase from GitHub: {e}")
                    self.issues.append(Issue(
                        self.repo_path_or_url,
                        "Initialization Error",
                        f"Failed to initialize codebase from GitHub: {e}",
                        IssueSeverity.ERROR,
                        suggestion="Check your network connection and GitHub access permissions."
                    ))
                    return {
                        "error": f"Failed to initialize codebase: {str(e)}",
                        "success": False
                    }
            else:
                # Local path
                try:
                    self.codebase = Codebase.from_local(self.repo_path_or_url)
                    print(f"Successfully initialized codebase from local path: {self.repo_path_or_url}")
                except Exception as e:
                    print(f"Error initializing codebase from local path: {e}")
                    self.issues.append(Issue(
                        self.repo_path_or_url,
                        "Initialization Error",
                        f"Failed to initialize codebase from local path: {e}",
                        IssueSeverity.ERROR,
                        suggestion="Check if the path exists and is a valid repository."
                    ))
                    return {
                        "error": f"Failed to initialize codebase: {str(e)}",
                        "success": False
                    }
        except Exception as e:
            print(f"Unexpected error initializing codebase: {e}")
            return {
                "error": f"Failed to initialize codebase: {str(e)}",
                "success": False
            }
            
        # Run analysis
        try:
            print("Running dead code analysis...")
            self._analyze_dead_code()
            
            print("Running parameter issues analysis...")
            self._analyze_parameter_issues()
            
            print("Running type annotation analysis...")
            self._analyze_type_annotations()
            
            print("Running circular dependency analysis...")
            self._analyze_circular_dependencies()
            
            print("Running implementation issues analysis...")
            self._analyze_implementation_issues()
            
            # Generate report
            report = self._generate_report()
            
            # Print report
            self._print_report(report)
            
            # Save report
            self._save_report(report)
            
            # Save detailed summaries
            self._save_detailed_summaries("detailed_analysis.txt")
            
            return report
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            return {
                "error": f"Analysis failed: {str(e)}",
                "success": False
            }
    
    def _analyze_dead_code(self):
        """Find and log unused code (functions, classes, imports)."""
        print("Analyzing dead code")

        dead_code = {
            "unused_functions": [],
            "unused_classes": [],
            "unused_variables": [],
            "unused_imports": [],
        }

        # Find unused functions
        for function in self.codebase.functions:
            # Skip decorated functions (as they might be used indirectly)
            if hasattr(function, "decorators") and function.decorators:
                continue

            # Check if function has no call sites or usages
            has_call_sites = (
                hasattr(function, "call_sites") and len(function.call_sites) > 0
            )
            has_usages = hasattr(function, "usages") and len(function.usages) > 0

            if not has_call_sites and not has_usages:
                # Skip magic methods and main functions
                if (hasattr(function, "is_magic") and function.is_magic) or (
                    hasattr(function, "name") and function.name in ["main", "__main__"]
                ):
                    continue

                # Get file path and name safely
                file_path = (
                    function.file.path
                    if hasattr(function, "file") and hasattr(function.file, "path")
                    else "unknown"
                )
                func_name = (
                    function.name if hasattr(function, "name") else str(function)
                )

                # Add to dead code list
                dead_code["unused_functions"].append({
                    "name": func_name,
                    "file": file_path,
                    "line": function.line if hasattr(function, "line") else None,
                })

        # Find unused classes
        for cls in self.codebase.classes:
            # Check if class has no usages
            has_usages = hasattr(cls, "usages") and len(cls.usages) > 0

            if not has_usages:
                # Skip base classes and abstract classes
                if (hasattr(cls, "is_abstract") and cls.is_abstract):
                    continue

                # Get file path and name safely
                file_path = (
                    cls.file.path
                    if hasattr(cls, "file") and hasattr(cls.file, "path")
                    else "unknown"
                )
                cls_name = cls.name if hasattr(cls, "name") else str(cls)

                # Add to dead code list
                dead_code["unused_classes"].append({
                    "name": cls_name,
                    "file": file_path,
                    "line": cls.line if hasattr(cls, "line") else None,
                })

        # Find unused imports
        for file in self.codebase.files:
            if hasattr(file, "imports"):
                for imp in file.imports:
                    # Check if import has no usages
                    has_usages = hasattr(imp, "usages") and len(imp.usages) > 0

                    if not has_usages:
                        # Get import name safely
                        imp_name = imp.name if hasattr(imp, "name") else str(imp)

                        # Add to dead code list
                        dead_code["unused_imports"].append({
                            "name": imp_name,
                            "file": file.path,
                            "line": imp.line if hasattr(imp, "line") else None,
                        })

        # Add summary counts
        dead_code["summary"] = {
            "unused_functions_count": len(dead_code["unused_functions"]),
            "unused_classes_count": len(dead_code["unused_classes"]),
            "unused_variables_count": len(dead_code["unused_variables"]),
            "unused_imports_count": len(dead_code["unused_imports"]),
        }

        return dead_code

    def _analyze_parameter_issues(self):
        """Find and log parameter issues (unused, mismatches)."""
        parameter_issues = {
            "missing_types": [],
            "unused_parameters": [],
            "incorrect_usage": [],
        }

        for func in self.codebase.functions:
            # Skip if no parameters
            if not hasattr(func, "parameters"):
                continue

            file_path = (
                func.file.path
                if hasattr(func, "file") and hasattr(func.file, "path")
                else "unknown"
            )
            func_name = func.name if hasattr(func, "name") else str(func)

            # Check for parameters without type annotations
            if hasattr(func, "parameters"):
                for param in func.parameters:
                    # Skip self parameter in methods
                    if param.name == "self" and hasattr(func, "is_method") and func.is_method:
                        continue

                    # Check for missing type annotations
                    if not hasattr(param, "type") or not param.type:
                        parameter_issues["missing_types"].append({
                            "function": func_name,
                            "parameter": param.name,
                            "file": file_path,
                            "line": func.line if hasattr(func, "line") else None,
                        })

                    # Check for unused parameters
                    param_dependencies = [dep.name for dep in func.dependencies if hasattr(dep, "name")]
                    if param.name not in param_dependencies:
                        # Skip 'self' in methods
                        if param.name == 'self' and func.is_method:
                            continue
                            
                        self.issues.append(Issue(
                            func,
                            IssueType.UNUSED_PARAMETER,
                            f"Function '{func.name}' has unused parameter: {param.name}",
                            IssueSeverity.INFO,
                            suggestion=f"Consider removing the unused parameter '{param.name}' if it's not needed"
                        ))
                        
                        parameter_issues["unused_parameters"].append({
                            "function": func_name,
                            "parameter": param.name,
                            "file": file_path,
                            "line": func.line if hasattr(func, "line") else None,
                        })

            # Check call sites for parameter mismatches
            for call in func.call_sites:
                if hasattr(call, 'args') and hasattr(func, 'parameters'):
                    expected_params = set(p.name for p in func.parameters if not p.is_optional and p.name != 'self')
                    actual_params = set()
                    
                    # Extract parameter names from call arguments
                    if hasattr(call, 'args'):
                        for arg in call.args:
                            if hasattr(arg, 'parameter_name') and arg.parameter_name:
                                actual_params.add(arg.parameter_name)
                    
                    # Check for missing required parameters
                    missing_params = expected_params - actual_params
                    if missing_params:
                        self.issues.append(Issue(
                            call,
                            IssueType.PARAMETER_MISMATCH,
                            f"Call to '{func.name}' is missing required parameters: {', '.join(missing_params)}",
                            IssueSeverity.ERROR,
                            suggestion=f"Add the missing parameters to the function call"
                        ))
                        
                        # Get call location
                        call_file = (
                            call.file.path
                            if hasattr(call, "file") and hasattr(call.file, "path")
                            else "unknown"
                        )
                        call_line = call.line if hasattr(call, "line") else None

                        parameter_issues["incorrect_usage"].append({
                            "function": func_name,
                            "call_file": call_file,
                            "call_line": call_line,
                            "missing_parameters": list(missing_params),
                        })

        # Add summary counts
        parameter_issues["summary"] = {
            "missing_types_count": len(parameter_issues["missing_types"]),
            "unused_parameters_count": len(parameter_issues["unused_parameters"]),
            "incorrect_usage_count": len(parameter_issues["incorrect_usage"]),
        }

        return parameter_issues

    def _analyze_type_annotations(self):
        """Find and log missing type annotations."""
        for func in self.codebase.functions:
            # Skip if function is in a type-annotated file
            file_path = str(func.file.path) if hasattr(func, 'file') and hasattr(func.file, 'path') else ''
            if any(file_ext in file_path for file_ext in ['.pyi']):
                continue
                
            # Check return type
            if not func.return_type and not func.name.startswith('__'):
                self.issues.append(Issue(
                    func,
                    IssueType.MISSING_TYPE_ANNOTATION,
                    f"Function '{func.name}' is missing return type annotation",
                    IssueSeverity.INFO,
                    suggestion="Add a return type annotation to improve type safety"
                ))
            
            # Check parameter types
            params_without_type = [p.name for p in func.parameters 
                                 if not p.type and p.name != 'self' and not p.name.startswith('*')]
            if params_without_type:
                self.issues.append(Issue(
                    func,
                    IssueType.MISSING_TYPE_ANNOTATION,
                    f"Function '{func.name}' has parameters without type annotations: {', '.join(params_without_type)}",
                    IssueSeverity.INFO,
                    suggestion="Add type annotations to all parameters"
                ))
    
    def _analyze_circular_dependencies(self):
        """Find and log circular dependencies."""
        circular_deps = {}
        
        # Basic implementation to detect file-level circular dependencies
        for file in self.codebase.files:
            visited = set()
            path = []
            self._check_circular_deps(file, visited, path, circular_deps)
        
        # Log circular dependencies
        for file_path, cycles in circular_deps.items():
            for cycle in cycles:
                cycle_str = " -> ".join([f.path for f in cycle])
                self.issues.append(Issue(
                    file_path,
                    IssueType.CIRCULAR_DEPENDENCY,
                    f"Circular dependency detected: {cycle_str}",
                    IssueSeverity.ERROR,
                    suggestion="Refactor the code to break the circular dependency"
                ))
    
    def _check_circular_deps(self, file, visited, path, circular_deps):
        """Helper method to check for circular dependencies using DFS."""
        if file in path:
            # Found a cycle
            cycle = path[path.index(file):] + [file]
            if file.path not in circular_deps:
                circular_deps[file.path] = []
            circular_deps[file.path].append(cycle)
            return
        
        if file in visited:
            return
            
        visited.add(file)
        path.append(file)
        
        # Check dependencies
        for dep in file.dependencies:
            if hasattr(dep, 'file') and dep.file:
                self._check_circular_deps(dep.file, visited, path.copy(), circular_deps)
        
        path.pop()
    
    def _analyze_implementation_issues(self):
        """Find and log implementation issues (empty functions, etc.)."""
        implementation_issues = {
            "empty_functions": [],
            "abstract_methods_without_implementation": [],
            "summary": {
                "empty_functions_count": 0,
                "abstract_methods_without_implementation_count": 0,
            },
        }

        # Check for empty functions
        for func in self.codebase.functions:
            # Skip dunder methods and abstract methods
            if func.name.startswith('__') and func.name.endswith('__'):
                continue
            
            # Check for empty function bodies
            if not func.body or not func.body.strip():
                # Skip if it's a method overridden from parent class
                is_override = False
                if hasattr(func, 'parent') and hasattr(func.parent, 'parents'):
                    for parent_class in func.parent.parents:
                        if any(m.name == func.name for m in parent_class.methods):
                            is_override = True
                            break
                
                if not is_override:
                    self.issues.append(Issue(
                        func,
                        IssueType.EMPTY_FUNCTION,
                        f"Function '{func.name}' has an empty body",
                        IssueSeverity.WARNING,
                        suggestion="Implement the function or remove it if it's not needed"
                    ))
                    
                    # Get file path
                    file_path = (
                        func.file.path
                        if hasattr(func, "file") and hasattr(func.file, "path")
                        else "unknown"
                    )
                    
                    # Add to implementation issues
                    implementation_issues["empty_functions"].append({
                        "name": func.name,
                        "file": file_path,
                        "line": func.line if hasattr(func, "line") else None,
                    })
                    implementation_issues["summary"]["empty_functions_count"] += 1

        # Check for abstract methods without implementation
        for cls in self.codebase.classes:
            if not hasattr(cls, "methods") or not hasattr(cls, "parents"):
                continue
                
            # Get abstract methods from parent classes
            abstract_methods = set()
            for parent in cls.parents:
                if hasattr(parent, "methods"):
                    for method in parent.methods:
                        if hasattr(method, "is_abstract") and method.is_abstract:
                            abstract_methods.add(method.name)
            
            # Check if abstract methods are implemented
            implemented_methods = {method.name for method in cls.methods}
            missing_implementations = abstract_methods - implemented_methods
            
            if missing_implementations:
                # Get file path
                file_path = (
                    cls.file.path
                    if hasattr(cls, "file") and hasattr(cls.file, "path")
                    else "unknown"
                )
                
                for method_name in missing_implementations:
                    implementation_issues["abstract_methods_without_implementation"].append({
                        "class": cls.name,
                        "method": method_name,
                        "file": file_path,
                        "line": cls.line if hasattr(cls, "line") else None,
                    })
                    implementation_issues["summary"]["abstract_methods_without_implementation_count"] += 1

        return implementation_issues

    def _generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report of the analysis results.
        
        Returns:
            Dictionary with analysis results
        """
        # Calculate analysis time
        analysis_time = time.time() - self.start_time
        
        # Count issues by type and severity
        issue_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for issue in self.issues:
            issue_type = issue.issue_type if hasattr(issue, 'issue_type') else "Unknown"
            severity = issue.severity if hasattr(issue, 'severity') else "Unknown"
            
            issue_counts[str(issue_type)] += 1
            severity_counts[str(severity)] += 1
        
        # Generate summary statistics
        total_files = len(self.codebase.files) if self.codebase else 0
        total_functions = len(self.codebase.functions) if self.codebase else 0
        total_classes = len(self.codebase.classes) if self.codebase else 0
        total_issues = len(self.issues)
        
        # Calculate issue density
        issue_density = total_issues / total_files if total_files > 0 else 0
        
        # Generate report
        report = {
            "summary": {
                "repo": self.repo_path_or_url,
                "analysis_time_seconds": analysis_time,
                "total_files": total_files,
                "total_functions": total_functions,
                "total_classes": total_classes,
                "total_issues": total_issues,
                "issue_density": issue_density
            },
            "issues_by_type": dict(issue_counts),
            "issues_by_severity": dict(severity_counts),
            "success": True
        }
        
        return report
    
    def _print_report(self, report: Dict[str, Any]):
        """Print a summary of the analysis report to the console."""
        summary = report["summary"]
        
        print("\n" + "="*80)
        print(f"ANALYSIS REPORT FOR: {summary['repo']}")
        print("="*80)
        
        print(f"\nAnalysis completed in {summary['analysis_time_seconds']:.2f} seconds")
        print(f"Files analyzed: {summary['total_files']}")
        print(f"Functions analyzed: {summary['total_functions']}")
        print(f"Classes analyzed: {summary['total_classes']}")
        print(f"Total issues found: {summary['total_issues']}")
        print(f"Issue density: {summary['issue_density']:.2f} issues per file")
        
        print("\nIssues by type:")
        for issue_type, count in report["issues_by_type"].items():
            print(f"  - {issue_type}: {count}")
        
        print("\nIssues by severity:")
        for severity, count in report["issues_by_severity"].items():
            print(f"  - {severity}: {count}")
        
        print("\nTop issues:")
        for i, issue in enumerate(self.issues[:10]):
            print(f"  {i+1}. {issue}")
        
        if len(self.issues) > 10:
            print(f"  ... and {len(self.issues) - 10} more issues")
        
        print("\n" + "="*80)
    
    def _save_report(self, report: Dict[str, Any]):
        """Save the analysis report to a JSON file."""
        try:
            # Create output directory if it doesn't exist
            output_dir = Path("analysis_reports")
            output_dir.mkdir(exist_ok=True)
            
            # Generate filename based on repo name
            repo_name = self.repo_path_or_url.split('/')[-1].replace('.', '_')
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = output_dir / f"analysis_{repo_name}_{timestamp}.json"
            
            # Save report
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
                
            print(f"\nReport saved to: {filename}")
            
        except Exception as e:
            print(f"Error saving report: {e}")
    
    def _save_detailed_summaries(self, filename: str):
        """Save detailed summaries of the codebase to a text file."""
        try:
            # Create output directory if it doesn't exist
            output_dir = Path("analysis_reports")
            output_dir.mkdir(exist_ok=True)
            
            # Generate full path
            filepath = output_dir / filename
            
            with open(filepath, 'w') as f:
                f.write("="*80 + "\n")
                f.write(f"DETAILED ANALYSIS SUMMARY FOR: {self.repo_path_or_url}\n")
                f.write("="*80 + "\n\n")
                
                # Write file summaries
                f.write("FILE SUMMARIES\n")
                f.write("-"*80 + "\n")
                
                for file in self.codebase.files:
                    try:
                        f.write(f"File: {file.path}\n")
                        f.write(f"  Size: {file.size} bytes\n")
                        f.write(f"  Lines: {len(file.content.splitlines())}\n")
                        f.write(f"  Functions: {len(file.functions)}\n")
                        f.write(f"  Classes: {len(file.classes)}\n")
                        f.write(f"  Imports: {len(file.imports)}\n")
                        f.write("\n")
                    except Exception as e:
                        f.write(f"Error generating file summary: {str(e)}\n\n")
                
                # Write function summaries
                f.write("\nFUNCTION SUMMARIES\n")
                f.write("-"*80 + "\n")
                
                for func in self.codebase.functions:
                    try:
                        f.write(f"Function: {func.name}\n")
                        f.write(f"  File: {func.file.path if hasattr(func, 'file') else 'Unknown'}\n")
                        f.write(f"  Parameters: {', '.join(p.name for p in func.parameters)}\n")
                        f.write(f"  Return type: {func.return_type if func.return_type else 'None'}\n")
                        f.write(f"  Usages: {len(func.usages)}\n")
                        f.write(f"  Dependencies: {len(func.dependencies)}\n")
                        f.write("\n")
                    except Exception as e:
                        f.write(f"Error generating function summary: {str(e)}\n\n")
                
                # Write class summaries
                f.write("\nCLASS SUMMARIES\n")
                f.write("-"*80 + "\n")
                
                for cls in self.codebase.classes:
                    try:
                        f.write(f"Class: {cls.name}\n")
                        f.write(f"  File: {cls.file.path if hasattr(cls, 'file') else 'Unknown'}\n")
                        f.write(f"  Methods: {len(cls.methods)}\n")
                        f.write(f"  Attributes: {len(cls.attributes)}\n")
                        f.write(f"  Parents: {', '.join(p.name for p in cls.parents) if hasattr(cls, 'parents') else 'None'}\n")
                        f.write(f"  Usages: {len(cls.usages)}\n")
                        f.write("\n")
                    except Exception as e:
                        f.write(f"Error generating class summary: {str(e)}\n\n")
                
                # Write issue summaries
                f.write("\nISSUE SUMMARIES\n")
                f.write("-"*80 + "\n")
                
                for issue in self.issues:
                    f.write(f"{issue}\n\n")
                
            print(f"\nDetailed summaries saved to: {filepath}")
            
        except Exception as e:
            print(f"Error saving detailed summaries: {e}")

def analyze_comprehensive(repo_path_or_url: str) -> Dict[str, Any]:
    """
    Perform a comprehensive analysis of a codebase.
    
    Args:
        repo_path_or_url: Path to local repo or URL to GitHub repo
        
    Returns:
        Dictionary with analysis results
    """
    analyzer = ComprehensiveAnalyzer(repo_path_or_url)
    return analyzer.analyze()
