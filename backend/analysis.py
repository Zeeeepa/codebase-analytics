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
from enum import Enum
from datetime import datetime
import uuid
import inspect

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
