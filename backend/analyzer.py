#!/usr/bin/env python3
"""
Comprehensive Codebase Analyzer

This module provides a unified interface for codebase analysis, integrating functionality
from various analyzer modules to detect code issues, analyze dependencies, and provide
insights into code quality and structure.

It combines the following analyzer capabilities:
- Code quality analysis (dead code, complexity, style issues)
- Issue tracking and management
- Dependency analysis
- Import and usage analysis
- Transaction management for code modifications
- Code metrics and visualization
- Performance analysis
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
except ImportError:
    nx = None

# Import from Codegen SDK - Explicitly require the SDK
from codegen.configs.models.codebase import CodebaseConfig
from codegen.configs.models.secrets import SecretsConfig
from codegen.git.repo_operator.repo_operator import RepoOperator
from codegen.git.schemas.repo_config import RepoConfig
from codegen.sdk.codebase.config import ProjectConfig
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
# Enums and Constants
#######################################################

class AnalysisType(str, Enum):
    """Types of analysis that can be performed."""

    CODEBASE = "codebase"
    PR = "pr"
    COMPARISON = "comparison"
    CODE_QUALITY = "code_quality"
    DEPENDENCY = "dependency"
    SECURITY = "security"
    PERFORMANCE = "performance"
    TYPE_CHECKING = "type_checking"
    COMPREHENSIVE = "comprehensive"  # New comprehensive analysis type


class IssueSeverity(str, Enum):
    """Severity levels for issues."""

    CRITICAL = "critical"  # Must be fixed immediately, blocks functionality
    ERROR = "error"  # Must be fixed, causes errors or undefined behavior
    WARNING = "warning"  # Should be fixed, may cause problems in future
    INFO = "info"  # Informational, could be improved but not critical


class IssueCategory(str, Enum):
    """Categories of issues that can be detected."""

    # Code Quality Issues
    DEAD_CODE = "dead_code"  # Unused variables, functions, etc.
    COMPLEXITY = "complexity"  # Code too complex, needs refactoring
    STYLE_ISSUE = "style_issue"  # Code style issues (line length, etc.)
    DOCUMENTATION = "documentation"  # Missing or incomplete documentation

    # Type and Parameter Issues
    TYPE_ERROR = "type_error"  # Type errors or inconsistencies
    PARAMETER_MISMATCH = "parameter_mismatch"  # Parameter type or count mismatch
    RETURN_TYPE_ERROR = "return_type_error"  # Return type error or mismatch

    # Implementation Issues
    IMPLEMENTATION_ERROR = "implementation_error"  # Incorrect implementation
    MISSING_IMPLEMENTATION = "missing_implementation"  # Missing implementation

    # Dependency Issues
    IMPORT_ERROR = "import_error"  # Import errors or issues
    DEPENDENCY_CYCLE = "dependency_cycle"  # Circular dependency
    MODULE_COUPLING = "module_coupling"  # High coupling between modules

    # API Issues
    API_CHANGE = "api_change"  # API has changed in a breaking way
    API_USAGE_ERROR = "api_usage_error"  # Incorrect API usage

    # Security Issues
    SECURITY_VULNERABILITY = "security_vulnerability"  # Security vulnerability

    # Performance Issues
    PERFORMANCE_ISSUE = "performance_issue"  # Performance issue


class IssueStatus(str, Enum):
    """Status of an issue."""

    OPEN = "open"  # Issue is open and needs to be fixed
    FIXED = "fixed"  # Issue has been fixed
    WONTFIX = "wontfix"  # Issue will not be fixed
    INVALID = "invalid"  # Issue is invalid or not applicable
    DUPLICATE = "duplicate"  # Issue is a duplicate of another


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


#######################################################
# Issue Management
#######################################################

@dataclass
class CodeLocation:
    """Location of an issue in code."""

    file: str
    line: int | None = None
    column: int | None = None
    end_line: int | None = None
    end_column: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeLocation":
        """Create from dictionary representation."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

    def __str__(self) -> str:
        """Convert to string representation."""
        if self.line is not None:
            if self.column is not None:
                return f"{self.file}:{self.line}:{self.column}"
            return f"{self.file}:{self.line}"
        return self.file


@dataclass
class Issue:
    """Represents an issue found during analysis."""

    # Core fields
    message: str
    severity: IssueSeverity
    location: CodeLocation

    # Classification fields
    category: IssueCategory | None = None
    analysis_type: AnalysisType | None = None
    status: IssueStatus = IssueStatus.OPEN

    # Context fields
    symbol: str | None = None
    code: str | None = None
    suggestion: str | None = None
    related_symbols: list[str] = field(default_factory=list)
    related_locations: list[CodeLocation] = field(default_factory=list)

    # Metadata fields
    id: str | None = None
    hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize derived fields."""
        # Generate an ID if not provided
        if self.id is None:
            import hashlib

            # Create a hash based on location and message
            hash_input = f"{self.location.file}:{self.location.line}:{self.message}"
            self.id = hashlib.md5(hash_input.encode()).hexdigest()[:12]

    @property
    def file(self) -> str:
        """Get the file path."""
        return self.location.file

    @property
    def line(self) -> int | None:
        """Get the line number."""
        return self.location.line

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "id": self.id,
            "message": self.message,
            "severity": self.severity.value,
            "location": self.location.to_dict(),
            "status": self.status.value,
        }

        # Add optional fields if present
        if self.category:
            result["category"] = self.category.value

        if self.analysis_type:
            result["analysis_type"] = self.analysis_type.value

        if self.symbol:
            result["symbol"] = self.symbol

        if self.code:
            result["code"] = self.code

        if self.suggestion:
            result["suggestion"] = self.suggestion

        if self.related_symbols:
            result["related_symbols"] = self.related_symbols

        if self.related_locations:
            result["related_locations"] = [
                loc.to_dict() for loc in self.related_locations
            ]

        if self.metadata:
            result["metadata"] = self.metadata

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Issue":
        """Create from dictionary representation."""
        # Convert string enums to actual enum values
        if "severity" in data and isinstance(data["severity"], str):
            data["severity"] = IssueSeverity(data["severity"])

        if "category" in data and isinstance(data["category"], str):
            data["category"] = IssueCategory(data["category"])

        if "analysis_type" in data and isinstance(data["analysis_type"], str):
            data["analysis_type"] = AnalysisType(data["analysis_type"])

        if "status" in data and isinstance(data["status"], str):
            data["status"] = IssueStatus(data["status"])

        # Convert location dict to CodeLocation
        if "location" in data and isinstance(data["location"], dict):
            data["location"] = CodeLocation.from_dict(data["location"])

        # Convert related_locations dicts to CodeLocation objects
        if "related_locations" in data and isinstance(data["related_locations"], list):
            data["related_locations"] = [
                CodeLocation.from_dict(loc) if isinstance(loc, dict) else loc
                for loc in data["related_locations"]
            ]

        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


class IssueCollection:
    """Collection of issues with filtering and grouping capabilities."""

    def __init__(self, issues: list[Issue] | None = None):
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

    def add_issues(self, issues: list[Issue]):
        """
        Add multiple issues to the collection.

        Args:
            issues: Issues to add
        """
        self.issues.extend(issues)

    def add_filter(self, filter_func, description: str = ""):
        """
        Add a filter function.

        Args:
            filter_func: Function that returns True if issue should be included
            description: Description of the filter
        """
        self._filters.append((filter_func, description))

    def get_issues(
        self,
        severity: IssueSeverity | None = None,
        category: IssueCategory | None = None,
        status: IssueStatus | None = None,
        file_path: str | None = None,
        symbol: str | None = None,
    ) -> list[Issue]:
        """
        Get issues matching the specified criteria.

        Args:
            severity: Severity to filter by
            category: Category to filter by
            status: Status to filter by
            file_path: File path to filter by
            symbol: Symbol name to filter by

        Returns:
            List of matching issues
        """
        filtered_issues = self.issues

        # Apply custom filters
        for filter_func, _ in self._filters:
            filtered_issues = [i for i in filtered_issues if filter_func(i)]

        # Apply standard filters
        if severity:
            filtered_issues = [i for i in filtered_issues if i.severity == severity]

        if category:
            filtered_issues = [i for i in filtered_issues if i.category == category]

        if status:
            filtered_issues = [i for i in filtered_issues if i.status == status]

        if file_path:
            filtered_issues = [
                i for i in filtered_issues if i.location.file == file_path
            ]

        if symbol:
            filtered_issues = [
                i
                for i in filtered_issues
                if (
                    i.symbol == symbol
                    or (i.related_symbols and symbol in i.related_symbols)
                )
            ]

        return filtered_issues

    def group_by_severity(self) -> dict[IssueSeverity, list[Issue]]:
        """
        Group issues by severity.

        Returns:
            Dictionary mapping severities to lists of issues
        """
        result = {severity: [] for severity in IssueSeverity}

        for issue in self.issues:
            result[issue.severity].append(issue)

        return result

    def group_by_category(self) -> dict[IssueCategory, list[Issue]]:
        """
        Group issues by category.

        Returns:
            Dictionary mapping categories to lists of issues
        """
        result = {category: [] for category in IssueCategory}

        for issue in self.issues:
            if issue.category:
                result[issue.category].append(issue)

        return result

    def group_by_file(self) -> dict[str, list[Issue]]:
        """
        Group issues by file.

        Returns:
            Dictionary mapping file paths to lists of issues
        """
        result = {}

        for issue in self.issues:
            if issue.location.file not in result:
                result[issue.location.file] = []

            result[issue.location.file].append(issue)

        return result

    def statistics(self) -> dict[str, Any]:
        """
        Get statistics about the issues.

        Returns:
            Dictionary with issue statistics
        """
        by_severity = self.group_by_severity()
        by_category = self.group_by_category()
        by_status = {status: [] for status in IssueStatus}
        for issue in self.issues:
            by_status[issue.status].append(issue)

        return {
            "total": len(self.issues),
            "by_severity": {
                severity.value: len(issues) for severity, issues in by_severity.items()
            },
            "by_category": {
                category.value: len(issues)
                for category, issues in by_category.items()
                if len(issues) > 0  # Only include non-empty categories
            },
            "by_status": {
                status.value: len(issues) for status, issues in by_status.items()
            },
            "file_count": len(self.group_by_file()),
        }

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary representation.

        Returns:
            Dictionary representation of the issue collection
        """
        return {
            "issues": [issue.to_dict() for issue in self.issues],
            "statistics": self.statistics(),
            "filters": [desc for _, desc in self._filters if desc],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IssueCollection":
        """
        Create from dictionary representation.

        Args:
            data: Dictionary representation

        Returns:
            Issue collection
        """
        collection = cls()

        if "issues" in data and isinstance(data["issues"], list):
            collection.add_issues([
                Issue.from_dict(issue) if isinstance(issue, dict) else issue
                for issue in data["issues"]
            ])

        return collection

    def save_to_file(self, file_path: str, format: str = "json"):
        """
        Save to file.

        Args:
            file_path: Path to save to
            format: Format to save in
        """
        if format == "json":
            with open(file_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

    @classmethod
    def load_from_file(cls, file_path: str) -> "IssueCollection":
        """
        Load from file.

        Args:
            file_path: Path to load from

        Returns:
            Issue collection
        """
        with open(file_path) as f:
            data = json.load(f)

        return cls.from_dict(data)


def create_issue(
    message: str,
    severity: str | IssueSeverity,
    file: str,
    line: int | None = None,
    category: str | IssueCategory | None = None,
    symbol: str | None = None,
    suggestion: str | None = None,
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
        Issue object
    """
    # Convert string severity to enum
    if isinstance(severity, str):
        severity = IssueSeverity(severity)

    # Convert string category to enum
    if isinstance(category, str) and category:
        category = IssueCategory(category)

    # Create location
    location = CodeLocation(file=file, line=line)

    # Create issue
    return Issue(
        message=message,
        severity=severity,
        location=location,
        category=category,
        symbol=symbol,
        suggestion=suggestion,
    )


#######################################################
# Analysis Results
#######################################################

@dataclass
class AnalysisSummary:
    """Summary statistics for an analysis."""

    total_files: int = 0
    total_functions: int = 0
    total_classes: int = 0
    total_issues: int = 0
    analysis_time: str = field(default_factory=lambda: datetime.now().isoformat())
    analysis_duration_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class CodeQualityResult:
    """Results of code quality analysis."""

    dead_code: dict[str, Any] = field(default_factory=dict)
    complexity: dict[str, Any] = field(default_factory=dict)
    parameter_issues: dict[str, Any] = field(default_factory=dict)
    style_issues: dict[str, Any] = field(default_factory=dict)
    implementation_issues: dict[str, Any] = field(default_factory=dict)
    maintainability: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return dict(asdict(self).items())


@dataclass
class DependencyResult:
    """Results of dependency analysis."""

    import_dependencies: dict[str, Any] = field(default_factory=dict)
    circular_dependencies: dict[str, Any] = field(default_factory=dict)
    module_coupling: dict[str, Any] = field(default_factory=dict)
    external_dependencies: dict[str, Any] = field(default_factory=dict)
    call_graph: dict[str, Any] = field(default_factory=dict)
    class_hierarchy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return dict(asdict(self).items())


@dataclass
class AnalysisResult:
    """Comprehensive analysis result."""

    # Core data
    analysis_types: list[AnalysisType]
    summary: AnalysisSummary = field(default_factory=AnalysisSummary)
    issues: IssueCollection = field(default_factory=IssueCollection)

    # Analysis results
    code_quality: CodeQualityResult | None = None
    dependencies: DependencyResult | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    repo_name: str | None = None
    repo_path: str | None = None
    language: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "analysis_types": [at.value for at in self.analysis_types],
            "summary": self.summary.to_dict(),
            "issues": self.issues.to_dict(),
            "metadata": self.metadata,
        }

        # Add optional sections if present
        if self.repo_name:
            result["repo_name"] = self.repo_name

        if self.repo_path:
            result["repo_path"] = self.repo_path

        if self.language:
            result["language"] = self.language

        # Add analysis results if present
        if self.code_quality:
            result["code_quality"] = self.code_quality.to_dict()

        if self.dependencies:
            result["dependencies"] = self.dependencies.to_dict()

        return result

    def save_to_file(self, file_path: str, indent: int = 2):
        """
        Save analysis result to a file.

        Args:
            file_path: Path to save to
            indent: JSON indentation level
        """
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisResult":
        """
        Create analysis result from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Analysis result object
        """
        # Convert analysis types
        analysis_types = [
            AnalysisType(at) if isinstance(at, str) else at
            for at in data.get("analysis_types", [])
        ]

        # Create summary
        summary = (
            AnalysisSummary(**data.get("summary", {}))
            if "summary" in data
            else AnalysisSummary()
        )

        # Create issues collection
        issues = (
            IssueCollection.from_dict(data.get("issues", {}))
            if "issues" in data
            else IssueCollection()
        )

        # Create result object
        result = cls(
            analysis_types=analysis_types,
            summary=summary,
            issues=issues,
            repo_name=data.get("repo_name"),
            repo_path=data.get("repo_path"),
            language=data.get("language"),
            metadata=data.get("metadata", {}),
        )

        # Add analysis results if present
        if "code_quality" in data:
            result.code_quality = CodeQualityResult(**data["code_quality"])

        if "dependencies" in data:
            result.dependencies = DependencyResult(**data["dependencies"])

        return result

    @classmethod
    def load_from_file(cls, file_path: str) -> "AnalysisResult":
        """
        Load analysis result from file.

        Args:
            file_path: Path to load from

        Returns:
            Analysis result object
        """
        with open(file_path) as f:
            data = json.load(f)

        return cls.from_dict(data)

#######################################################
# Codebase Analysis Utilities
#######################################################

def get_codebase_summary(codebase: Codebase) -> str:
    """
    Generate a comprehensive summary of a codebase.
    
    Args:
        codebase: The Codebase object to summarize
        
    Returns:
        A formatted string containing a summary of the codebase's nodes and edges
    """
    node_summary = f"""Contains {len(codebase.ctx.get_nodes())} nodes
- {len(list(codebase.files))} files
- {len(list(codebase.imports))} imports
- {len(list(codebase.external_modules))} external_modules
- {len(list(codebase.symbols))} symbols
\t- {len(list(codebase.classes))} classes
\t- {len(list(codebase.functions))} functions
\t- {len(list(codebase.global_vars))} global_vars
\t- {len(list(codebase.interfaces))} interfaces
"""
    edge_summary = f"""Contains {len(codebase.ctx.edges)} edges
- {len([x for x in codebase.ctx.edges if x[2].type == EdgeType.SYMBOL_USAGE])} symbol -> used symbol
- {len([x for x in codebase.ctx.edges if x[2].type == EdgeType.IMPORT_SYMBOL_RESOLUTION])} import -> used symbol
- {len([x for x in codebase.ctx.edges if x[2].type == EdgeType.EXPORT])} export -> exported symbol
    """

    return f"{node_summary}\n{edge_summary}"


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
    
    files_to_analyze = [f for f in codebase.files if not file_path or f.file_path == file_path]
    
    for file in files_to_analyze:
        dependencies = []
        
        # Add direct imports
        for imp in file.imports:
            if hasattr(imp, "imported_symbol") and hasattr(imp.imported_symbol, "file"):
                if hasattr(imp.imported_symbol.file, "file_path"):
                    dependencies.append(imp.imported_symbol.file.file_path)
        
        # Add symbol dependencies
        for symbol in file.symbols:
            for dep in symbol.dependencies:
                if hasattr(dep, "file") and hasattr(dep.file, "file_path"):
                    dependencies.append(dep.file.file_path)
        
        # Remove duplicates and self-references
        unique_deps = list(set([d for d in dependencies if d != file.file_path]))
        dependency_graph[file.file_path] = unique_deps
    
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
    
    for symbol in target_symbols:
        # Find all edges that reference this symbol
        for edge in codebase.ctx.edges:
            if edge[1] == symbol.id:  # If the edge points to our symbol
                source_node = codebase.ctx.get_node(edge[0])
                if source_node:
                    # Get file and line information if available
                    file_path = None
                    line_number = None
                    
                    if hasattr(source_node, "file") and hasattr(source_node.file, "file_path"):
                        file_path = source_node.file.file_path
                    
                    if hasattr(source_node, "line"):
                        line_number = source_node.line
                    
                    references.append(
                        {
                            "file_path": file_path,
                            "line": line_number,
                            "source_type": type(source_node).__name__,
                            "source_name": getattr(source_node, "name", str(source_node)),
                            "edge_type": edge[2].type.name
                            if hasattr(edge[2], "type")
                            else "Unknown",
                        }
                    )
    
    return references


def calculate_cyclomatic_complexity(func: Function) -> int:
    """
    Calculate the cyclomatic complexity of a function.
    
    Args:
        func: The Function object to analyze
        
    Returns:
        An integer representing the cyclomatic complexity
    """
    complexity = 1  # Base complexity
    
    if not hasattr(func, "source") or not func.source:
        return complexity
    
    # Simple heuristic: count control flow statements
    source = func.source.lower()
    
    # Count if statements
    complexity += source.count(" if ") + source.count("\nif ")
    
    # Count else if / elif statements
    complexity += source.count("elif ") + source.count("else if ")
    
    # Count loops
    complexity += source.count(" for ") + source.count("\nfor ")
    complexity += source.count(" while ") + source.count("\nwhile ")
    
    # Count exception handlers
    complexity += source.count("except ") + source.count("catch ")
    
    # Count logical operators (each one creates a new path)
    complexity += source.count(" and ") + source.count(" && ")
    complexity += source.count(" or ") + source.count(" || ")
    
    return complexity


#######################################################
# Main Analyzer Class
#######################################################

class CodebaseAnalyzer:
    """
    Comprehensive code analyzer for detecting issues and analyzing codebase structure.
    
    This class provides a unified interface for analyzing codebases, integrating
    functionality from various analyzer modules to detect code issues, analyze
    dependencies, and provide insights into code quality and structure.
    """

    def __init__(
        self,
        repo_url: str | None = None,
        repo_path: str | None = None,
        base_branch: str = "main",
        language: str | None = None,
        file_ignore_list: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the codebase analyzer.

        Args:
            repo_url: URL of the repository to analyze
            repo_path: Local path to the repository to analyze
            base_branch: Base branch for comparison
            language: Programming language of the codebase
            file_ignore_list: List of file patterns to ignore
            config: Additional configuration options
        """
        self.repo_url = repo_url
        self.repo_path = repo_path
        self.base_branch = base_branch
        self.language = language

        # Use custom ignore list or default global list
        self.file_ignore_list = file_ignore_list or []

        # Configuration options
        self.config = config or {}

        # Codebase object
        self.codebase = None

        # Analysis results
        self.issues = IssueCollection()
        self.analysis_result = None

        # Initialize codebase based on provided parameters
        if repo_url:
            self._init_from_url(repo_url, language)
        elif repo_path:
            self._init_from_path(repo_path, language)

    def _init_from_url(self, repo_url: str, language: str | None = None):
        """
        Initialize codebase from a repository URL.

        Args:
            repo_url: URL of the repository
            language: Programming language of the codebase
        """
            
        try:
            # Extract repository information
            if repo_url.endswith(".git"):
                repo_url = repo_url[:-4]

            parts = repo_url.rstrip("/").split("/")
            repo_name = parts[-1]
            owner = parts[-2]
            repo_full_name = f"{owner}/{repo_name}"

            # Create temporary directory for cloning
            tmp_dir = tempfile.mkdtemp(prefix="analyzer_")

            # Set up configuration
            config = CodebaseConfig(
                debug=False,
                allow_external=True,
                py_resolve_syspath=True,
            )

            secrets = SecretsConfig()

            # Determine programming language
            prog_lang = None
            if language:
                prog_lang = ProgrammingLanguage(language.upper())

            # Initialize the codebase
            logger.info(f"Initializing codebase from {repo_url}")

            # Use from_repo method with correct parameter name
            # Specify only Python language to avoid TypeScript engine issues
            self.codebase = Codebase.from_repo(
                repo_full_name=repo_full_name,
                tmp_dir=tmp_dir,
                language="python",  # Force Python language to avoid TypeScript engine issues
                config=config,
                secrets=secrets,
            )

            logger.info(f"Successfully initialized codebase from {repo_url}")

        except Exception as e:
            logger.exception(f"Error initializing codebase from URL: {e}")
            raise

    def _init_from_path(self, repo_path: str, language: str | None = None):
        """
        Initialize codebase from a local repository path.

        Args:
            repo_path: Path to the repository
            language: Programming language of the codebase
        """
            
        try:
            # Set up configuration
            config = CodebaseConfig(
                debug=False,
                allow_external=True,
                py_resolve_syspath=True,
            )

            secrets = SecretsConfig()

            # Initialize the codebase
            logger.info(f"Initializing codebase from {repo_path}")

            # Determine programming language
            prog_lang = None
            if language:
                prog_lang = ProgrammingLanguage(language.upper())

            # Set up repository configuration
            repo_config = RepoConfig.from_repo_path(repo_path)
            repo_config.respect_gitignore = False
            repo_operator = RepoOperator(repo_config=repo_config, bot_commit=False)

            # Create project configuration
            project_config = ProjectConfig(
                repo_operator=repo_operator,
                programming_language=prog_lang if prog_lang else None,
            )

            # Initialize codebase
            self.codebase = Codebase(
                projects=[project_config], config=config, secrets=secrets
            )

            logger.info(f"Successfully initialized codebase from {repo_path}")

        except Exception as e:
            logger.exception(f"Error initializing codebase from path: {e}")
            raise

    def analyze(self, analysis_types: list[AnalysisType] | None = None, output_format: str = "json", output_file: str | None = None) -> AnalysisResult:
        """
        Perform comprehensive analysis on the codebase.

        Args:
            analysis_types: Types of analysis to perform. Defaults to code quality analysis.
            output_format: Format of the output (json, html, console)
            output_file: Path to save results to

        Returns:
            AnalysisResult containing the findings
        """
        if not self.codebase:
            raise ValueError("Codebase not initialized")

        if not analysis_types:
            analysis_types = [AnalysisType.CODE_QUALITY]

        # Start measuring analysis time
        start_time = datetime.now()

        # Create result object
        self.analysis_result = AnalysisResult(
            analysis_types=analysis_types,
            repo_name=self.repo_url.split("/")[-1] if self.repo_url else None,
            repo_path=self.repo_path,
            language=self.language,
        )

        # Perform analyses based on requested types
        logger.info(f"Starting analysis with types: {[at.value for at in analysis_types]}")

        # Comprehensive analysis (includes all other types)
        if AnalysisType.COMPREHENSIVE in analysis_types:
            logger.info("Running comprehensive analysis (includes all analysis types)")
            self._analyze_code_quality()
            self._analyze_dependencies()
            self._analyze_performance()
            self._analyze_parameter_issues()
            self._analyze_type_annotations()
            self._analyze_circular_dependencies()
            # Add more comprehensive checks here
        else:
            # Individual analysis types
            # Code quality analysis
            if AnalysisType.CODE_QUALITY in analysis_types:
                self._analyze_code_quality()

            # Dependency analysis
            if AnalysisType.DEPENDENCY in analysis_types:
                self._analyze_dependencies()

            # Performance analysis
            if AnalysisType.PERFORMANCE in analysis_types:
                self._analyze_performance()

        # Calculate analysis duration
        analysis_duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        self.analysis_result.summary.analysis_duration_ms = analysis_duration_ms

        # Update summary
        self.analysis_result.summary.total_files = len(list(self.codebase.files))
        self.analysis_result.summary.total_functions = len(list(self.codebase.functions))
        self.analysis_result.summary.total_classes = len(list(self.codebase.classes))
        self.analysis_result.summary.total_issues = len(self.issues.issues)
        
        # Set issues collection
        self.analysis_result.issues = self.issues

        # Save or display results if specified
        if output_file and output_format == "json":
            self.analysis_result.save_to_file(output_file)
            logger.info(f"Results saved to {output_file}")
        elif output_format == "html":
            self._generate_html_report(output_file)
        elif output_format == "console":
            self._print_console_report()

        return self.analysis_result

    def _analyze_code_quality(self):
        """Analyze code quality issues in the codebase."""
        logger.info("Starting code quality analysis")
        
        # Initialize result
        result = CodeQualityResult()
        
        # Find dead code
        result.dead_code = self._find_dead_code()
        
        # Analyze function parameters
        result.parameter_issues = self._check_function_parameters()
        
        # Analyze implementation issues
        result.implementation_issues = self._check_implementations()
        
        # Set results
        self.analysis_result.code_quality = result
        
        logger.info(f"Code quality analysis complete. Found {len(self.issues.issues)} issues.")

    def _analyze_dependencies(self):
        """Analyze dependencies between components in the codebase."""
        logger.info("Starting dependency analysis")
        
        # Initialize result
        result = DependencyResult()
        
        # Generate dependency graph
        result.import_dependencies = self._analyze_import_dependencies()
        
        # Find circular dependencies
        result.circular_dependencies = self._find_circular_dependencies()
        
        # Set results
        self.analysis_result.dependencies = result
        
        logger.info("Dependency analysis complete")
        
    def _analyze_performance(self):
        """Analyze performance characteristics of the codebase."""
        logger.info("Starting performance analysis")
        
        # This could include:
        # - Analyzing function complexity and identifying bottlenecks
        # - Calculating cyclomatic complexity metrics
        # - Identifying functions that might benefit from optimization
        # - Finding recursive functions that might cause performance issues
        
        # For now, we'll add performance issues based on function complexity
        functions = list(self.codebase.functions)
        
        for func in functions:
            # Calculate cyclomatic complexity
            complexity = calculate_cyclomatic_complexity(func)
            
            # Functions with high complexity are likely performance bottlenecks
            if complexity > 15:
                file_path = (
                    func.file.file_path
                    if hasattr(func, "file") and hasattr(func.file, "file_path")
                    else "unknown"
                )
                func_name = func.name if hasattr(func, "name") else str(func)
                
                self.issues.add_issue(
                    create_issue(
                        message=f"High complexity function may be a performance bottleneck: {func_name} (complexity: {complexity})",
                        severity=IssueSeverity.WARNING,
                        file=file_path,
                        line=func.line if hasattr(func, "line") else None,
                        category=IssueCategory.PERFORMANCE_ISSUE,
                        symbol=func_name,
                        suggestion="Consider refactoring this function to reduce complexity and improve performance",
                    )
                )
                
            # Check for recursive functions
            if hasattr(func, "function_calls"):
                for call in func.function_calls:
                    if hasattr(call, "name") and call.name == func.name:
                        file_path = (
                            func.file.file_path
                            if hasattr(func, "file") and hasattr(func.file, "file_path")
                            else "unknown"
                        )
                        func_name = func.name if hasattr(func, "name") else str(func)
                        
                        self.issues.add_issue(
                            create_issue(
                                message=f"Recursive function may cause performance issues: {func_name}",
                                severity=IssueSeverity.INFO,
                                file=file_path,
                                line=func.line if hasattr(func, "line") else None,
                                category=IssueCategory.PERFORMANCE_ISSUE,
                                symbol=func_name,
                                suggestion="Ensure recursive function has a proper base case and won't cause stack overflow",
                            )
                        )
                        
        logger.info("Performance analysis complete")

    def _find_dead_code(self) -> dict[str, Any]:
        """
        Find unused code (dead code) in the codebase.

        Returns:
            Dictionary containing dead code analysis results
        """
        logger.info("Analyzing dead code")

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
                    function.file.file_path
                    if hasattr(function, "file") and hasattr(function.file, "file_path")
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

                # Add issue
                self.issues.add_issue(
                    create_issue(
                        message=f"Unused function: {func_name}",
                        severity=IssueSeverity.WARNING,
                        file=file_path,
                        line=function.line if hasattr(function, "line") else None,
                        category=IssueCategory.DEAD_CODE,
                        symbol=func_name,
                        suggestion="Consider removing this unused function or documenting why it's needed",
                    )
                )

        # Find unused classes
        for cls in self.codebase.classes:
            # Check if class has no usages
            has_usages = hasattr(cls, "usages") and len(cls.usages) > 0

            if not has_usages:
                # Get file path and name safely
                file_path = (
                    cls.file.file_path
                    if hasattr(cls, "file") and hasattr(cls.file, "file_path")
                    else "unknown"
                )
                cls_name = cls.name if hasattr(cls, "name") else str(cls)

                # Add to dead code list
                dead_code["unused_classes"].append({
                    "name": cls_name,
                    "file": file_path,
                    "line": cls.line if hasattr(cls, "line") else None,
                })

                # Add issue
                self.issues.add_issue(
                    create_issue(
                        message=f"Unused class: {cls_name}",
                        severity=IssueSeverity.WARNING,
                        file=file_path,
                        line=cls.line if hasattr(cls, "line") else None,
                        category=IssueCategory.DEAD_CODE,
                        symbol=cls_name,
                        suggestion="Consider removing this unused class or documenting why it's needed",
                    )
                )

        # Find unused imports
        for file in self.codebase.files:
            if hasattr(file, "is_binary") and file.is_binary:
                continue

            if not hasattr(file, "imports"):
                continue

            file_path = file.file_path if hasattr(file, "file_path") else str(file)

            for imp in file.imports:
                if not hasattr(imp, "usages"):
                    continue

                if len(imp.usages) == 0:
                    # Get import source safely
                    import_source = imp.source if hasattr(imp, "source") else str(imp)

                    # Add to dead code list
                    dead_code["unused_imports"].append({
                        "import": import_source,
                        "file": file_path,
                        "line": imp.line if hasattr(imp, "line") else None,
                    })

                    # Add issue
                    # Create Issue directly without using create_issue
                    location = CodeLocation(
                        file=file_path,
                        line=imp.line if hasattr(imp, "line") else None
                    )
                    self.issues.add_issue(
                        Issue(
                            message=f"Unused import: {import_source}",
                            severity=IssueSeverity.INFO,
                            location=location,
                            category=IssueCategory.DEAD_CODE,
                            suggestion="Remove this unused import",
                        )
                    )

        # Add summary statistics
        dead_code["summary"] = {
            "unused_functions_count": len(dead_code["unused_functions"]),
            "unused_classes_count": len(dead_code["unused_classes"]),
            "unused_variables_count": len(dead_code["unused_variables"]),
            "unused_imports_count": len(dead_code["unused_imports"]),
            "total_dead_code_count": (
                len(dead_code["unused_functions"])
                + len(dead_code["unused_classes"])
                + len(dead_code["unused_variables"])
                + len(dead_code["unused_imports"])
            ),
        }

        return dead_code

    def _check_function_parameters(self) -> dict[str, Any]:
        """
        Check for function parameter issues.

        Returns:
            Dictionary containing parameter analysis results
        """
        logger.info("Analyzing function parameters")

        parameter_issues = {
            "missing_types": [],
            "unused_parameters": [],
            "incorrect_usage": [],
        }

        for function in self.codebase.functions:
            # Skip if no parameters
            if not hasattr(function, "parameters"):
                continue

            file_path = (
                function.file.file_path
                if hasattr(function, "file") and hasattr(function.file, "file_path")
                else "unknown"
            )
            func_name = function.name if hasattr(function, "name") else str(function)

            # Check for missing type annotations
            missing_types = []
            for param in function.parameters:
                if not hasattr(param, "name"):
                    continue

                if not hasattr(param, "type") or not param.type:
                    missing_types.append(param.name)

            if missing_types:
                parameter_issues["missing_types"].append({
                    "function": func_name,
                    "file": file_path,
                    "line": function.line if hasattr(function, "line") else None,
                    "parameters": missing_types,
                })

                self.issues.add_issue(
                    create_issue(
                        message=f"Function '{func_name}' has parameters without type annotations: {', '.join(missing_types)}",
                        severity=IssueSeverity.WARNING,
                        file=file_path,
                        line=function.line if hasattr(function, "line") else None,
                        category=IssueCategory.TYPE_ERROR,
                        symbol=func_name,
                        suggestion="Add type annotations to all parameters",
                    )
                )

            # Check for incorrect parameter usage at call sites
            if hasattr(function, "call_sites"):
                for call_site in function.call_sites:
                    # Skip if call site has no arguments
                    if not hasattr(call_site, "args"):
                        continue

                    # Get required parameter count (excluding those with defaults)
                    required_count = 0
                    if hasattr(function, "parameters"):
                        required_count = sum(
                            1
                            for p in function.parameters
                            if not hasattr(p, "has_default") or not p.has_default
                        )

                    # Get call site file info
                    call_file = (
                        call_site.file.file_path
                        if hasattr(call_site, "file")
                        and hasattr(call_site.file, "file_path")
                        else "unknown"
                    )
                    call_line = call_site.line if hasattr(call_site, "line") else None

                    # Check parameter count
                    arg_count = len(call_site.args)
                    if arg_count < required_count:
                        parameter_issues["incorrect_usage"].append({
                            "function": func_name,
                            "caller_file": call_file,
                            "caller_line": call_line,
                            "required_count": required_count,
                            "provided_count": arg_count,
                        })

                        self.issues.add_issue(
                            create_issue(
                                message=f"Call to '{func_name}' has too few arguments ({arg_count} provided, {required_count} required)",
                                severity=IssueSeverity.ERROR,
                                file=call_file,
                                line=call_line,
                                category=IssueCategory.PARAMETER_MISMATCH,
                                symbol=func_name,
                                suggestion=f"Provide all required arguments to '{func_name}'",
                            )
                        )

        # Add summary statistics
        parameter_issues["summary"] = {
            "missing_types_count": len(parameter_issues["missing_types"]),
            "unused_parameters_count": len(parameter_issues["unused_parameters"]),
            "incorrect_usage_count": len(parameter_issues["incorrect_usage"]),
            "total_issues": (
                len(parameter_issues["missing_types"])
                + len(parameter_issues["unused_parameters"])
                + len(parameter_issues["incorrect_usage"])
            ),
        }

        return parameter_issues

    def _check_implementations(self) -> dict[str, Any]:
        """
        Check for implementation issues.

        Returns:
            Dictionary containing implementation analysis results
        """
        logger.info("Analyzing implementations")

        implementation_issues = {
            "empty_functions": [],
            "abstract_methods_without_implementation": [],
            "summary": {
                "empty_functions_count": 0,
                "abstract_methods_without_implementation_count": 0,
            },
        }

        # Check for empty functions
        for function in self.codebase.functions:
            # Get function source
            if hasattr(function, "source"):
                source = function.source

                # Check if function is empty or just has 'pass'
                is_empty = False

                if not source or source.strip() == "":
                    is_empty = True
                else:
                    # Extract function body (skip the first line with the def)
                    body_lines = source.split("\n")[1:] if "\n" in source else []

                    # Check if body is empty or just has whitespace, docstring, or pass
                    non_empty_lines = [
                        line
                        for line in body_lines
                        if line.strip()
                        and not line.strip().startswith("#")
                        and not (
                            line.strip().startswith('"""')
                            or line.strip().startswith("'''")
                        )
                        and line.strip() != "pass"
                    ]

                    if not non_empty_lines:
                        is_empty = True

                if is_empty:
                    # Get file path and name safely
                    file_path = (
                        function.file.file_path
                        if hasattr(function, "file")
                        and hasattr(function.file, "file_path")
                        else "unknown"
                    )
                    func_name = (
                        function.name if hasattr(function, "name") else str(function)
                    )

                    # Skip interface/abstract methods that are supposed to be empty
                    is_abstract = (
                        hasattr(function, "is_abstract") and function.is_abstract
                    ) or (
                        hasattr(function, "parent")
                        and hasattr(function.parent, "is_interface")
                        and function.parent.is_interface
                    )

                    if not is_abstract:
                        # Add to empty functions list
                        implementation_issues["empty_functions"].append({
                            "name": func_name,
                            "file": file_path,
                            "line": function.line
                            if hasattr(function, "line")
                            else None,
                        })

                        # Add issue
                        self.issues.add_issue(
                            create_issue(
                                message=f"Function '{func_name}' is empty",
                                severity=IssueSeverity.WARNING,
                                file=file_path,
                                line=function.line
                                if hasattr(function, "line")
                                else None,
                                category=IssueCategory.MISSING_IMPLEMENTATION,
                                symbol=func_name,
                                suggestion="Implement this function or remove it if not needed",
                            )
                        )

        # Add summary statistics
        implementation_issues["summary"]["empty_functions_count"] = len(
            implementation_issues["empty_functions"]
        )
        implementation_issues["summary"][
            "abstract_methods_without_implementation_count"
        ] = len(implementation_issues["abstract_methods_without_implementation"])

        return implementation_issues

    def _analyze_import_dependencies(self) -> dict[str, Any]:
        """
        Analyze import dependencies between files.

        Returns:
            Dictionary containing import dependency analysis results
        """
        logger.info("Analyzing import dependencies")

        # Generate dependency graph
        dependency_graph = get_dependency_graph(self.codebase)

        # Count dependencies per file
        dependency_counts = {
            file_path: len(deps) for file_path, deps in dependency_graph.items()
        }

        # Find files with high number of dependencies
        high_dependency_files = [
            {"file": file_path, "dependency_count": count}
            for file_path, count in dependency_counts.items()
            if count > 10  # Threshold for "high dependencies"
        ]

        # Sort by dependency count
        high_dependency_files.sort(key=lambda x: x["dependency_count"], reverse=True)

        # Report high dependency files as issues
        for item in high_dependency_files[:10]:  # Limit to top 10
            self.issues.add_issue(
                create_issue(
                    message=f"File has high number of dependencies: {item['dependency_count']}",
                    severity=IssueSeverity.WARNING,
                    file=item["file"],
                    category=IssueCategory.MODULE_COUPLING,
                    suggestion="Consider refactoring to reduce dependencies",
                )
            )

        return {
            "dependency_graph": dependency_graph,
            "dependency_counts": dependency_counts,
            "high_dependency_files": high_dependency_files,
            "summary": {
                "total_files": len(dependency_graph),
                "files_with_high_dependencies": len(high_dependency_files),
                "max_dependencies": max(dependency_counts.values()) if dependency_counts else 0,
                "avg_dependencies": (
                    sum(dependency_counts.values()) / len(dependency_counts)
                    if dependency_counts
                    else 0
                ),
            },
        }

    def _find_circular_dependencies(self) -> dict[str, Any]:
        """
        Find circular dependencies in the codebase.

        Returns:
            Dictionary containing circular dependency analysis results
        """
        logger.info("Analyzing circular dependencies")

        # Get dependency graph
        dependency_graph = get_dependency_graph(self.codebase)
        
        # Find circular dependencies
        circular_deps = []
        visited = set()
        
        def detect_cycles(node, path):
            if node in path:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycle_key = "->".join(sorted(cycle))
                
                if cycle_key not in visited:
                    visited.add(cycle_key)
                    circular_deps.append(cycle)
                return
                
            # Continue DFS
            new_path = path + [node]
            for dep in dependency_graph.get(node, []):
                detect_cycles(dep, new_path)
        
        # Start DFS from each node
        for node in dependency_graph:
            detect_cycles(node, [])
            
        # Report circular dependencies as issues
        for cycle in circular_deps:
            cycle_str = " -> ".join(cycle)
            # Report on the first file in the cycle
            self.issues.add_issue(
                create_issue(
                    message=f"Circular dependency detected: {cycle_str}",
                    severity=IssueSeverity.ERROR,
                    file=cycle[0],
                    category=IssueCategory.DEPENDENCY_CYCLE,
                    suggestion="Break the circular dependency by refactoring one of the modules",
                )
            )
            
        return {
            "circular_dependencies": circular_deps,
            "summary": {
                "circular_dependency_count": len(circular_deps),
            },
        }

    def save_results(self, output_file: str):
        """
        Save analysis results to a file.

        Args:
            output_file: Path to the output file
        """
        if not self.analysis_result:
            raise ValueError("No analysis results to save")
            
        self.analysis_result.save_to_file(output_file)
        logger.info(f"Results saved to {output_file}")
        
    def _generate_html_report(self, output_file: str | None = None):
        """
        Generate an HTML report of the analysis results.
        
        Args:
            output_file: Path to save the report to
        """
        if not self.analysis_result:
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
            <title>Codebase Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .section {{ margin-bottom: 30px; }}
                .metric {{ margin-bottom: 20px; }}
                .metric-title {{ font-weight: bold; }}
                pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>Codebase Analysis Report</h1>
            <div class="section">
                <h2>Metadata</h2>
                <p><strong>Repository:</strong> {repo_name}</p>
                <p><strong>Analysis Time:</strong> {analysis_time}</p>
                <p><strong>Language:</strong> {self.language or "Auto-detected"}</p>
            </div>
        """
        
        # Add summary section
        html += f"""
        <div class="section">
            <h2>Summary</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Files</td>
                    <td>{self.analysis_result.summary.total_files}</td>
                </tr>
                <tr>
                    <td>Total Classes</td>
                    <td>{self.analysis_result.summary.total_classes}</td>
                </tr>
                <tr>
                    <td>Total Functions</td>
                    <td>{self.analysis_result.summary.total_functions}</td>
                </tr>
                <tr>
                    <td>Total Issues</td>
                    <td>{self.analysis_result.summary.total_issues}</td>
                </tr>
                <tr>
                    <td>Analysis Duration</td>
                    <td>{self.analysis_result.summary.analysis_duration_ms / 1000:.2f} seconds</td>
                </tr>
            </table>
        </div>
        """
        
        # Add issues section if there are any
        if self.issues and self.issues.issues:
            issues_by_severity = self.issues.group_by_severity()
            
            html += """
            <div class="section">
                <h2>Issues</h2>
                <h3>By Severity</h3>
                <table>
                    <tr>
                        <th>Severity</th>
                        <th>Count</th>
                    </tr>
            """
            
            for severity in IssueSeverity:
                count = len(issues_by_severity[severity])
                if count > 0:
                    html += f"""
                    <tr>
                        <td>{severity.value}</td>
                        <td>{count}</td>
                    </tr>
                    """
                    
            html += """
                </table>
                
                <h3>Top Issues</h3>
                <table>
                    <tr>
                        <th>Severity</th>
                        <th>Category</th>
                        <th>Message</th>
                        <th>Location</th>
                    </tr>
            """
            
            # Sort issues by severity
            sorted_issues = sorted(self.issues.issues, key=lambda x: {
                IssueSeverity.CRITICAL: 0,
                IssueSeverity.ERROR: 1,
                IssueSeverity.WARNING: 2,
                IssueSeverity.INFO: 3,
            }.get(x.severity, 4))
            
            # Add top 20 issues
            for issue in sorted_issues[:20]:
                location = f"{issue.location.file}"
                if issue.location.line:
                    location += f":{issue.location.line}"
                    
                category = issue.category.value if issue.category else ""
                
                html += f"""
                <tr>
                    <td>{issue.severity.value}</td>
                    <td>{category}</td>
                    <td>{issue.message}</td>
                    <td>{location}</td>
                </tr>
                """
                
            html += """
                </table>
            </div>
            """
        
        # Add analysis results sections
        if self.analysis_result.code_quality:
            html += """
            <div class="section">
                <h2>Code Quality</h2>
            """
            
            # Add dead code section
            if "dead_code" in self.analysis_result.code_quality.__dict__:
                dead_code = self.analysis_result.code_quality.dead_code
                if "summary" in dead_code:
                    html += """
                    <h3>Dead Code</h3>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Count</th>
                        </tr>
                    """
                    
                    for key, value in dead_code["summary"].items():
                        html += f"""
                        <tr>
                            <td>{key.replace("_", " ").title()}</td>
                            <td>{value}</td>
                        </tr>
                        """
                        
                    html += """
                    </table>
                    """
                    
            # Add other code quality sections as needed...
            
            html += """
            </div>
            """
        
        if self.analysis_result.dependencies:
            html += """
            <div class="section">
                <h2>Dependencies</h2>
            """
            
            # Add circular dependencies section
            if "circular_dependencies" in self.analysis_result.dependencies.__dict__:
                circular_deps = self.analysis_result.dependencies.circular_dependencies
                if "circular_dependencies" in circular_deps:
                    html += """
                    <h3>Circular Dependencies</h3>
                    <table>
                        <tr>
                            <th>#</th>
                            <th>Dependency Cycle</th>
                        </tr>
                    """
                    
                    for i, cycle in enumerate(circular_deps["circular_dependencies"][:10], 1):
                        html += f"""
                        <tr>
                            <td>{i}</td>
                            <td>{" -> ".join(cycle)}</td>
                        </tr>
                        """
                        
                    html += """
                    </table>
                    """
            
            html += """
            </div>
            """
        
        # Close the HTML
        html += """
        </body>
        </html>
        """
        
        with open(output_file, "w") as f:
            f.write(html)
            
        logger.info(f"HTML report saved to {output_file}")
        
    def _print_console_report(self):
        """Print a summary report to the console."""
        if not self.analysis_result:
            raise ValueError("No analysis results to print")
            
        repo_name = self.repo_url.split("/")[-1] if self.repo_url else (self.repo_path or "Unknown")
        
        print(f"\n{'=' * 80}")
        print(f"CODEBASE ANALYSIS REPORT: {repo_name}")
        print(f"{'=' * 80}")
        print(f"Analysis Time: {self.analysis_result.summary.analysis_time}")
        print(f"Language: {self.language or 'Auto-detected'}")
        print(f"Analysis Duration: {self.analysis_result.summary.analysis_duration_ms / 1000:.2f} seconds")
        
        # Print summary
        print(f"\n{'-' * 40}")
        print("SUMMARY:")
        print(f"{'-' * 40}")
        print(f"Total Files: {self.analysis_result.summary.total_files}")
        print(f"Total Classes: {self.analysis_result.summary.total_classes}")
        print(f"Total Functions: {self.analysis_result.summary.total_functions}")
        print(f"Total Issues: {self.analysis_result.summary.total_issues}")
        
        # Print issues by severity if there are any
        if self.issues and self.issues.issues:
            issues_by_severity = self.issues.group_by_severity()
            
            print(f"\n{'-' * 40}")
            print("ISSUES BY SEVERITY:")
            print(f"{'-' * 40}")
            for severity in IssueSeverity:
                count = len(issues_by_severity[severity])
                if count > 0:
                    print(f"{severity.value}: {count}")
            
            # Print top issues
            print(f"\n{'-' * 40}")
            print("TOP ISSUES:")
            print(f"{'-' * 40}")
            
            # Sort issues by severity
            sorted_issues = sorted(self.issues.issues, key=lambda x: {
                IssueSeverity.CRITICAL: 0,
                IssueSeverity.ERROR: 1,
                IssueSeverity.WARNING: 2,
                IssueSeverity.INFO: 3,
            }.get(x.severity, 4))
            
            # Print top 10 issues
            for i, issue in enumerate(sorted_issues[:10], 1):
                severity_icon = {
                    IssueSeverity.CRITICAL: "❌",
                    IssueSeverity.ERROR: "⛔",
                    IssueSeverity.WARNING: "⚠️",
                    IssueSeverity.INFO: "ℹ️",
                }.get(issue.severity, "")
                
                location = f"{issue.location.file}"
                if issue.location.line:
                    location += f":{issue.location.line}"
                
                category = f"[{issue.category.value}]" if issue.category else ""
                
                print(f"{i}. {severity_icon} {category} {issue.message}")
                print(f"   Location: {location}")
                if issue.suggestion:
                    print(f"   Suggestion: {issue.suggestion}")
                print()
        
        # Print analysis type summaries
        if self.analysis_result.code_quality:
            print(f"\n{'-' * 40}")
            print("CODE QUALITY SUMMARY:")
            print(f"{'-' * 40}")
            
            # Print dead code summary
            if "dead_code" in self.analysis_result.code_quality.__dict__:
                dead_code = self.analysis_result.code_quality.dead_code
                if "summary" in dead_code:
                    for key, value in dead_code["summary"].items():
                        print(f"{key.replace('_', ' ').title()}: {value}")
                        
        # Print dependencies summary
        if self.analysis_result.dependencies:
            print(f"\n{'-' * 40}")
            print("DEPENDENCIES SUMMARY:")
            print(f"{'-' * 40}")
            
            # Print circular dependencies
            if "circular_dependencies" in self.analysis_result.dependencies.__dict__:
                circular_deps = self.analysis_result.dependencies.circular_dependencies
                if "circular_dependencies" in circular_deps:
                    cycles = circular_deps["circular_dependencies"]
                    print(f"Circular Dependencies: {len(cycles)}")
                    if cycles:
                        print("\nExample circular dependencies:")
                        for i, cycle in enumerate(cycles[:3], 1):
                            print(f"{i}. {' -> '.join(cycle)}")


#######################################################
# Helper Functions
#######################################################

def analyze_codebase(
    repo_path: str | None = None,
    repo_url: str | None = None,
    output_file: str | None = None,
    analysis_types: list[str] | None = None,
    language: str | None = None,
    output_format: str = "json",
) -> AnalysisResult:
    """
    Analyze a codebase and optionally save results to a file.
    
    Args:
        repo_path: Path to the repository to analyze
        repo_url: URL of the repository to analyze
        output_file: Optional path to save results to
        analysis_types: Optional list of analysis types to perform
        language: Optional programming language of the codebase
        output_format: Format for output (json, html, console)
        
    Returns:
        AnalysisResult containing the findings
    """
    # Convert string analysis types to enum values
    analysis_type_enums = []
    if analysis_types:
        for at in analysis_types:
            try:
                analysis_type_enums.append(AnalysisType(at))
            except ValueError:
                logger.warning(f"Unknown analysis type: {at}")
    
    # Initialize analyzer
    if repo_path:
        analyzer = CodebaseAnalyzer(repo_path=repo_path, language=language)
    elif repo_url:
        analyzer = CodebaseAnalyzer(repo_url=repo_url, language=language)
    else:
        raise ValueError("Either repo_path or repo_url must be provided")
    
    # Perform analysis
    result = analyzer.analyze(
        analysis_type_enums,
        output_format=output_format,
        output_file=output_file
    )
    
    return result