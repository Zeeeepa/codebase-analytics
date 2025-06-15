#!/usr/bin/env python3
"""
Codebase Analysis Module

This module provides functionality for analyzing codebases, including:
- Code quality analysis (complexity, dead code detection, etc.)
- Issue detection and classification
- Dependency analysis
- Metrics calculation

It separates the analysis logic from visualization concerns, making the codebase
more maintainable and easier to extend with new analysis capabilities.
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

    def add(self, issue: Issue) -> None:
        """
        Add an issue to the collection.

        Args:
            issue: The issue to add
        """
        self.issues.append(issue)

    def add_all(self, issues: list[Issue]) -> None:
        """
        Add multiple issues to the collection.

        Args:
            issues: The issues to add
        """
        self.issues.extend(issues)

    def filter(self, predicate: callable) -> "IssueCollection":
        """
        Filter issues based on a predicate.

        Args:
            predicate: Function that takes an issue and returns a boolean

        Returns:
            A new IssueCollection with filtered issues
        """
        return IssueCollection([issue for issue in self.issues if predicate(issue)])

    def filter_by_severity(self, severity: IssueSeverity) -> "IssueCollection":
        """
        Filter issues by severity.

        Args:
            severity: The severity to filter by

        Returns:
            A new IssueCollection with filtered issues
        """
        return self.filter(lambda issue: issue.severity == severity)

    def filter_by_category(self, category: IssueCategory) -> "IssueCollection":
        """
        Filter issues by category.

        Args:
            category: The category to filter by

        Returns:
            A new IssueCollection with filtered issues
        """
        return self.filter(lambda issue: issue.category == category)

    def filter_by_file(self, file_path: str) -> "IssueCollection":
        """
        Filter issues by file path.

        Args:
            file_path: The file path to filter by

        Returns:
            A new IssueCollection with filtered issues
        """
        return self.filter(lambda issue: issue.location.file == file_path)

    def group_by_severity(self) -> dict[IssueSeverity, list[Issue]]:
        """
        Group issues by severity.

        Returns:
            Dictionary mapping severity to list of issues
        """
        result = {severity: [] for severity in IssueSeverity}
        for issue in self.issues:
            result[issue.severity].append(issue)
        return result

    def group_by_category(self) -> dict[IssueCategory, list[Issue]]:
        """
        Group issues by category.

        Returns:
            Dictionary mapping category to list of issues
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
            Dictionary mapping file path to list of issues
        """
        result: dict[str, list[Issue]] = {}
        for issue in self.issues:
            file_path = issue.location.file
            if file_path not in result:
                result[file_path] = []
            result[file_path].append(issue)
        return result

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary representation.

        Returns:
            Dictionary representation of the issue collection
        """
        return {
            "issues": [issue.to_dict() for issue in self.issues],
            "count": len(self.issues),
            "severity_counts": {
                severity.value: len(issues)
                for severity, issues in self.group_by_severity().items()
                if issues
            },
            "category_counts": {
                category.value: len(issues)
                for category, issues in self.group_by_category().items()
                if issues
            },
            "file_counts": {
                file_path: len(issues)
                for file_path, issues in self.group_by_file().items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IssueCollection":
        """
        Create from dictionary representation.

        Args:
            data: Dictionary representation of the issue collection

        Returns:
            New IssueCollection instance
        """
        issues = [Issue.from_dict(issue_data) for issue_data in data.get("issues", [])]
        return cls(issues)

    def __len__(self) -> int:
        """Get the number of issues in the collection."""
        return len(self.issues)

    def __iter__(self):
        """Iterate over the issues in the collection."""
        return iter(self.issues)

    def __getitem__(self, index: int) -> Issue:
        """Get an issue by index."""
        return self.issues[index]


#######################################################
# Analysis Result Classes
#######################################################

@dataclass
class AnalysisSummary:
    """Summary of analysis results."""

    analysis_time: str
    analysis_duration_ms: float
    total_files: int
    total_classes: int
    total_functions: int
    total_issues: int
    language: str | None = None
    repo_url: str | None = None
    repo_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeQualityAnalysis:
    """Results of code quality analysis."""

    dead_code: dict[str, Any] = field(default_factory=dict)
    complexity: dict[str, Any] = field(default_factory=dict)
    style_issues: dict[str, Any] = field(default_factory=dict)
    documentation: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyAnalysis:
    """Results of dependency analysis."""

    import_graph: dict[str, Any] = field(default_factory=dict)
    circular_dependencies: dict[str, Any] = field(default_factory=dict)
    module_coupling: dict[str, Any] = field(default_factory=dict)
    external_dependencies: dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityAnalysis:
    """Results of security analysis."""

    vulnerabilities: dict[str, Any] = field(default_factory=dict)
    secrets: dict[str, Any] = field(default_factory=dict)
    permissions: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAnalysis:
    """Results of performance analysis."""

    hotspots: dict[str, Any] = field(default_factory=dict)
    complexity_metrics: dict[str, Any] = field(default_factory=dict)
    memory_usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class TypeAnalysis:
    """Results of type analysis."""

    type_errors: dict[str, Any] = field(default_factory=dict)
    type_coverage: dict[str, Any] = field(default_factory=dict)
    parameter_issues: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Complete results of codebase analysis."""

    summary: AnalysisSummary
    code_quality: CodeQualityAnalysis | None = None
    dependencies: DependencyAnalysis | None = None
    security: SecurityAnalysis | None = None
    performance: PerformanceAnalysis | None = None
    type_analysis: TypeAnalysis | None = None
    raw_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "summary": asdict(self.summary),
            "code_quality": asdict(self.code_quality) if self.code_quality else None,
            "dependencies": asdict(self.dependencies) if self.dependencies else None,
            "security": asdict(self.security) if self.security else None,
            "performance": asdict(self.performance) if self.performance else None,
            "type_analysis": asdict(self.type_analysis) if self.type_analysis else None,
            "raw_data": self.raw_data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisResult":
        """Create from dictionary representation."""
        summary = AnalysisSummary(**data.get("summary", {}))
        
        code_quality = None
        if data.get("code_quality"):
            code_quality = CodeQualityAnalysis(**data["code_quality"])
            
        dependencies = None
        if data.get("dependencies"):
            dependencies = DependencyAnalysis(**data["dependencies"])
            
        security = None
        if data.get("security"):
            security = SecurityAnalysis(**data["security"])
            
        performance = None
        if data.get("performance"):
            performance = PerformanceAnalysis(**data["performance"])
            
        type_analysis = None
        if data.get("type_analysis"):
            type_analysis = TypeAnalysis(**data["type_analysis"])
            
        return cls(
            summary=summary,
            code_quality=code_quality,
            dependencies=dependencies,
            security=security,
            performance=performance,
            type_analysis=type_analysis,
            raw_data=data.get("raw_data", {})
        )


#######################################################
# Core Analysis Functions
#######################################################

class CodebaseAnalyzer:
    """Main analyzer class for codebase analysis."""

    def __init__(
        self,
        repo_path: str | None = None,
        repo_url: str | None = None,
        language: str | None = None,
    ):
        """
        Initialize the analyzer.
        
        Args:
            repo_path: Path to the repository to analyze
            repo_url: URL of the repository to analyze
            language: Programming language of the codebase
        """
        self.repo_path = repo_path
        self.repo_url = repo_url
        self.language = language
        self.codebase = None
        self.issues = IssueCollection()
        self.analysis_result = None
        
        # Initialize the codebase
        if repo_path:
            self._init_from_path(repo_path)
        elif repo_url:
            self._init_from_url(repo_url)
        else:
            raise ValueError("Either repo_path or repo_url must be provided")
    
    def _init_from_path(self, repo_path: str) -> None:
        """Initialize from a local repository path."""
        logger.info(f"Initializing analyzer from path: {repo_path}")
        try:
            self.codebase = Codebase.from_path(repo_path)
            logger.info(f"Loaded codebase with {len(self.codebase.files)} files")
        except Exception as e:
            logger.error(f"Error loading codebase from path: {str(e)}")
            raise
    
    def _init_from_url(self, repo_url: str) -> None:
        """Initialize from a repository URL."""
        logger.info(f"Initializing analyzer from URL: {repo_url}")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.info(f"Cloning repository to {temp_dir}")
                repo_config = RepoConfig(url=repo_url)
                repo_operator = RepoOperator(repo_config)
                repo_operator.clone(temp_dir)
                
                self.repo_path = temp_dir
                self._init_from_path(temp_dir)
        except Exception as e:
            logger.error(f"Error loading codebase from URL: {str(e)}")
            raise
    
    def analyze(
        self,
        analysis_types: list[AnalysisType] | None = None,
        output_format: str = "json",
        output_file: str | None = None,
    ) -> AnalysisResult:
        """
        Perform analysis on the codebase.
        
        Args:
            analysis_types: Types of analysis to perform
            output_format: Format for output (json, html, console)
            output_file: Optional path to save results to
            
        Returns:
            AnalysisResult containing the findings
        """
        start_time = datetime.now()
        
        # Default to comprehensive analysis if no types specified
        if not analysis_types:
            analysis_types = [AnalysisType.COMPREHENSIVE]
        
        logger.info(f"Starting analysis with types: {[at.value for at in analysis_types]}")
        
        # Perform the requested analyses
        if AnalysisType.CODE_QUALITY in analysis_types or AnalysisType.COMPREHENSIVE in analysis_types:
            self._analyze_code_quality()
        
        if AnalysisType.DEPENDENCY in analysis_types or AnalysisType.COMPREHENSIVE in analysis_types:
            self._analyze_dependencies()
        
        if AnalysisType.SECURITY in analysis_types or AnalysisType.COMPREHENSIVE in analysis_types:
            self._analyze_security()
        
        if AnalysisType.PERFORMANCE in analysis_types or AnalysisType.COMPREHENSIVE in analysis_types:
            self._analyze_performance()
        
        if AnalysisType.TYPE_CHECKING in analysis_types or AnalysisType.COMPREHENSIVE in analysis_types:
            self._analyze_types()
        
        # Calculate analysis duration
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Create analysis summary
        summary = AnalysisSummary(
            analysis_time=start_time.isoformat(),
            analysis_duration_ms=duration_ms,
            total_files=len(self.codebase.files) if self.codebase else 0,
            total_classes=sum(len(file.classes) for file in self.codebase.files) if self.codebase else 0,
            total_functions=sum(len(file.functions) for file in self.codebase.files) if self.codebase else 0,
            total_issues=len(self.issues),
            language=self.language,
            repo_url=self.repo_url,
            repo_path=self.repo_path,
        )
        
        # Create analysis result
        self.analysis_result = AnalysisResult(
            summary=summary,
            code_quality=self.code_quality if hasattr(self, "code_quality") else None,
            dependencies=self.dependencies if hasattr(self, "dependencies") else None,
            security=self.security if hasattr(self, "security") else None,
            performance=self.performance if hasattr(self, "performance") else None,
            type_analysis=self.type_analysis if hasattr(self, "type_analysis") else None,
        )
        
        # Output results if requested
        if output_file:
            self._output_results(output_file, output_format)
        elif output_format == "console":
            self._print_console_report()
        
        logger.info(f"Analysis completed in {duration_ms/1000:.2f} seconds")
        
        return self.analysis_result
    
    def _analyze_code_quality(self) -> None:
        """Analyze code quality (complexity, dead code, style issues, etc.)."""
        logger.info("Analyzing code quality")
        
        # Initialize code quality analysis
        self.code_quality = CodeQualityAnalysis()
        
        # Analyze dead code
        self._analyze_dead_code()
        
        # Analyze complexity
        self._analyze_complexity()
        
        # Analyze style issues
        self._analyze_style_issues()
        
        # Analyze documentation
        self._analyze_documentation()
    
    def _analyze_dead_code(self) -> None:
        """Analyze dead code (unused variables, functions, etc.)."""
        logger.info("Analyzing dead code")
        
        unused_functions = []
        unused_classes = []
        unused_variables = []
        
        # Find unused functions
        for file in self.codebase.files:
            for func in file.functions:
                if not hasattr(func, "usages") or not func.usages:
                    # Skip if it's a test function or special method
                    if (
                        func.name.startswith("test_")
                        or func.name == "__init__"
                        or func.name.startswith("__")
                        and func.name.endswith("__")
                    ):
                        continue
                    
                    unused_functions.append({
                        "name": func.name,
                        "file": file.filepath,
                        "line": getattr(func, "start_point", [0])[0] if hasattr(func, "start_point") else 0,
                    })
                    
                    # Add issue
                    self.issues.add(
                        Issue(
                            message=f"Unused function '{func.name}'",
                            severity=IssueSeverity.WARNING,
                            category=IssueCategory.DEAD_CODE,
                            location=CodeLocation(
                                file=file.filepath,
                                line=getattr(func, "start_point", [0])[0] if hasattr(func, "start_point") else None,
                            ),
                            suggestion="Consider removing this function if it's not needed, or documenting why it exists if it's intended for future use.",
                        )
                    )
        
        # Find unused classes
        for file in self.codebase.files:
            for cls in file.classes:
                if not hasattr(cls, "usages") or not cls.usages:
                    # Skip if it's a test class
                    if cls.name.startswith("Test") or cls.name.endswith("Test"):
                        continue
                    
                    unused_classes.append({
                        "name": cls.name,
                        "file": file.filepath,
                        "line": getattr(cls, "start_point", [0])[0] if hasattr(cls, "start_point") else 0,
                    })
                    
                    # Add issue
                    self.issues.add(
                        Issue(
                            message=f"Unused class '{cls.name}'",
                            severity=IssueSeverity.WARNING,
                            category=IssueCategory.DEAD_CODE,
                            location=CodeLocation(
                                file=file.filepath,
                                line=getattr(cls, "start_point", [0])[0] if hasattr(cls, "start_point") else None,
                            ),
                            suggestion="Consider removing this class if it's not needed, or documenting why it exists if it's intended for future use.",
                        )
                    )
        
        # Store results
        self.code_quality.dead_code = {
            "unused_functions": unused_functions,
            "unused_classes": unused_classes,
            "unused_variables": unused_variables,
            "summary": {
                "unused_functions_count": len(unused_functions),
                "unused_classes_count": len(unused_classes),
                "unused_variables_count": len(unused_variables),
                "total_dead_code_count": len(unused_functions) + len(unused_classes) + len(unused_variables),
            },
        }
    
    def _analyze_complexity(self) -> None:
        """Analyze code complexity."""
        logger.info("Analyzing code complexity")
        
        complex_functions = []
        complexity_by_file = {}
        total_complexity = 0
        function_count = 0
        
        # Calculate complexity for each function
        for file in self.codebase.files:
            file_complexity = 0
            file_function_count = 0
            
            for func in file.functions:
                # Calculate cyclomatic complexity
                complexity = self._calculate_cyclomatic_complexity(func)
                
                file_complexity += complexity
                file_function_count += 1
                total_complexity += complexity
                function_count += 1
                
                # Check if function is too complex
                if complexity > 10:
                    complex_functions.append({
                        "name": func.name,
                        "file": file.filepath,
                        "line": getattr(func, "start_point", [0])[0] if hasattr(func, "start_point") else 0,
                        "complexity": complexity,
                    })
                    
                    # Add issue
                    severity = IssueSeverity.INFO if complexity <= 15 else (
                        IssueSeverity.WARNING if complexity <= 25 else IssueSeverity.ERROR
                    )
                    
                    self.issues.add(
                        Issue(
                            message=f"Function '{func.name}' has high cyclomatic complexity ({complexity})",
                            severity=severity,
                            category=IssueCategory.COMPLEXITY,
                            location=CodeLocation(
                                file=file.filepath,
                                line=getattr(func, "start_point", [0])[0] if hasattr(func, "start_point") else None,
                            ),
                            suggestion="Consider refactoring this function into smaller, more manageable pieces.",
                        )
                    )
            
            # Store file complexity
            if file_function_count > 0:
                complexity_by_file[file.filepath] = {
                    "total_complexity": file_complexity,
                    "function_count": file_function_count,
                    "average_complexity": file_complexity / file_function_count,
                }
        
        # Store results
        self.code_quality.complexity = {
            "complex_functions": complex_functions,
            "complexity_by_file": complexity_by_file,
            "summary": {
                "total_complexity": total_complexity,
                "function_count": function_count,
                "average_complexity": total_complexity / function_count if function_count > 0 else 0,
                "complex_functions_count": len(complex_functions),
            },
        }
    
    def _calculate_cyclomatic_complexity(self, func: Function) -> int:
        """Calculate cyclomatic complexity for a function."""
        # This is a simplified calculation
        # In a real implementation, we would analyze the AST
        complexity = 1  # Base complexity
        
        # Count control flow statements
        if hasattr(func, "source"):
            source = func.source
            
            # Count if statements
            complexity += source.count("if ")
            
            # Count else if statements
            complexity += source.count("elif ")
            
            # Count for loops
            complexity += source.count("for ")
            
            # Count while loops
            complexity += source.count("while ")
            
            # Count and/or operators (each adds a path)
            complexity += source.count(" and ")
            complexity += source.count(" or ")
            
            # Count exception handlers
            complexity += source.count("except ")
        
        return complexity
    
    def _analyze_style_issues(self) -> None:
        """Analyze code style issues."""
        logger.info("Analyzing style issues")
        
        # This would typically use a linter like flake8 or pylint
        # For this example, we'll just do some basic checks
        
        style_issues = []
        
        for file in self.codebase.files:
            if hasattr(file, "source"):
                lines = file.source.split("\n")
                
                for i, line in enumerate(lines):
                    # Check line length
                    if len(line) > 100:
                        style_issues.append({
                            "file": file.filepath,
                            "line": i + 1,
                            "message": f"Line too long ({len(line)} > 100 characters)",
                            "type": "line_length",
                        })
                        
                        self.issues.add(
                            Issue(
                                message=f"Line too long ({len(line)} > 100 characters)",
                                severity=IssueSeverity.INFO,
                                category=IssueCategory.STYLE_ISSUE,
                                location=CodeLocation(
                                    file=file.filepath,
                                    line=i + 1,
                                ),
                            )
                        )
        
        # Store results
        self.code_quality.style_issues = {
            "issues": style_issues,
            "summary": {
                "total_issues": len(style_issues),
            },
        }
    
    def _analyze_documentation(self) -> None:
        """Analyze code documentation."""
        logger.info("Analyzing documentation")
        
        undocumented_functions = []
        undocumented_classes = []
        documentation_by_file = {}
        
        for file in self.codebase.files:
            file_doc_count = 0
            file_total_count = 0
            
            # Check function documentation
            for func in file.functions:
                file_total_count += 1
                
                if hasattr(func, "docstring") and func.docstring:
                    file_doc_count += 1
                else:
                    # Skip simple or private functions
                    if func.name.startswith("_") or len(getattr(func, "source", "")) < 5:
                        continue
                    
                    undocumented_functions.append({
                        "name": func.name,
                        "file": file.filepath,
                        "line": getattr(func, "start_point", [0])[0] if hasattr(func, "start_point") else 0,
                    })
                    
                    self.issues.add(
                        Issue(
                            message=f"Function '{func.name}' lacks documentation",
                            severity=IssueSeverity.INFO,
                            category=IssueCategory.DOCUMENTATION,
                            location=CodeLocation(
                                file=file.filepath,
                                line=getattr(func, "start_point", [0])[0] if hasattr(func, "start_point") else None,
                            ),
                            suggestion="Add a docstring to describe what this function does, its parameters, and return value.",
                        )
                    )
            
            # Check class documentation
            for cls in file.classes:
                file_total_count += 1
                
                if hasattr(cls, "docstring") and cls.docstring:
                    file_doc_count += 1
                else:
                    undocumented_classes.append({
                        "name": cls.name,
                        "file": file.filepath,
                        "line": getattr(cls, "start_point", [0])[0] if hasattr(cls, "start_point") else 0,
                    })
                    
                    self.issues.add(
                        Issue(
                            message=f"Class '{cls.name}' lacks documentation",
                            severity=IssueSeverity.INFO,
                            category=IssueCategory.DOCUMENTATION,
                            location=CodeLocation(
                                file=file.filepath,
                                line=getattr(cls, "start_point", [0])[0] if hasattr(cls, "start_point") else None,
                            ),
                            suggestion="Add a docstring to describe what this class represents and its purpose.",
                        )
                    )
            
            # Store file documentation stats
            if file_total_count > 0:
                documentation_by_file[file.filepath] = {
                    "documented_count": file_doc_count,
                    "total_count": file_total_count,
                    "documentation_percentage": (file_doc_count / file_total_count) * 100,
                }
        
        # Store results
        self.code_quality.documentation = {
            "undocumented_functions": undocumented_functions,
            "undocumented_classes": undocumented_classes,
            "documentation_by_file": documentation_by_file,
            "summary": {
                "undocumented_functions_count": len(undocumented_functions),
                "undocumented_classes_count": len(undocumented_classes),
                "total_undocumented_count": len(undocumented_functions) + len(undocumented_classes),
            },
        }
    
    def _analyze_dependencies(self) -> None:
        """Analyze dependencies between modules."""
        logger.info("Analyzing dependencies")
        
        # Initialize dependency analysis
        self.dependencies = DependencyAnalysis()
        
        # Build dependency graph
        if nx is not None:
            dependency_graph = nx.DiGraph()
            
            # Add nodes for each file
            for file in self.codebase.files:
                dependency_graph.add_node(file.filepath)
            
            # Add edges for imports
            for file in self.codebase.files:
                for imp in getattr(file, "imports", []):
                    if hasattr(imp, "resolved_path") and imp.resolved_path:
                        dependency_graph.add_edge(file.filepath, imp.resolved_path)
            
            # Find circular dependencies
            try:
                cycles = list(nx.simple_cycles(dependency_graph))
                
                for cycle in cycles:
                    # Add issue for each file in the cycle
                    for file_path in cycle:
                        self.issues.add(
                            Issue(
                                message=f"File is part of a circular dependency: {' -> '.join(cycle)}",
                                severity=IssueSeverity.WARNING,
                                category=IssueCategory.DEPENDENCY_CYCLE,
                                location=CodeLocation(
                                    file=file_path,
                                ),
                                suggestion="Refactor the code to break the circular dependency, possibly by extracting common functionality to a separate module.",
                            )
                        )
                
                # Store results
                self.dependencies.circular_dependencies = {
                    "circular_dependencies": cycles,
                    "summary": {
                        "total_cycles": len(cycles),
                    },
                }
            except Exception as e:
                logger.warning(f"Error finding circular dependencies: {str(e)}")
                self.dependencies.circular_dependencies = {
                    "circular_dependencies": [],
                    "summary": {
                        "total_cycles": 0,
                    },
                }
            
            # Store import graph
            self.dependencies.import_graph = {
                "nodes": list(dependency_graph.nodes()),
                "edges": list(dependency_graph.edges()),
                "summary": {
                    "total_nodes": len(dependency_graph.nodes()),
                    "total_edges": len(dependency_graph.edges()),
                },
            }
        else:
            logger.warning("NetworkX not available, skipping dependency graph analysis")
            self.dependencies.import_graph = {
                "nodes": [],
                "edges": [],
                "summary": {
                    "total_nodes": 0,
                    "total_edges": 0,
                },
            }
            self.dependencies.circular_dependencies = {
                "circular_dependencies": [],
                "summary": {
                    "total_cycles": 0,
                },
            }
    
    def _analyze_security(self) -> None:
        """Analyze security vulnerabilities."""
        logger.info("Analyzing security")
        
        # Initialize security analysis
        self.security = SecurityAnalysis()
        
        # This would typically use a security scanner
        # For this example, we'll just do some basic checks
        
        vulnerabilities = []
        
        for file in self.codebase.files:
            if hasattr(file, "source"):
                # Check for hardcoded secrets
                if re.search(r"(password|secret|key|token)\s*=\s*['\"][^'\"]+['\"]", file.source, re.IGNORECASE):
                    vulnerabilities.append({
                        "file": file.filepath,
                        "type": "hardcoded_secret",
                        "message": "Possible hardcoded secret found",
                    })
                    
                    self.issues.add(
                        Issue(
                            message="Possible hardcoded secret found",
                            severity=IssueSeverity.CRITICAL,
                            category=IssueCategory.SECURITY_VULNERABILITY,
                            location=CodeLocation(
                                file=file.filepath,
                            ),
                            suggestion="Store secrets in environment variables or a secure secret management system, not in code.",
                        )
                    )
                
                # Check for SQL injection vulnerabilities
                if re.search(r"execute\([^)]*\+", file.source):
                    vulnerabilities.append({
                        "file": file.filepath,
                        "type": "sql_injection",
                        "message": "Possible SQL injection vulnerability",
                    })
                    
                    self.issues.add(
                        Issue(
                            message="Possible SQL injection vulnerability",
                            severity=IssueSeverity.CRITICAL,
                            category=IssueCategory.SECURITY_VULNERABILITY,
                            location=CodeLocation(
                                file=file.filepath,
                            ),
                            suggestion="Use parameterized queries or an ORM instead of string concatenation for SQL queries.",
                        )
                    )
        
        # Store results
        self.security.vulnerabilities = {
            "vulnerabilities": vulnerabilities,
            "summary": {
                "total_vulnerabilities": len(vulnerabilities),
            },
        }
    
    def _analyze_performance(self) -> None:
        """Analyze performance issues."""
        logger.info("Analyzing performance")
        
        # Initialize performance analysis
        self.performance = PerformanceAnalysis()
        
        # This would typically use a profiler
        # For this example, we'll just do some basic checks
        
        hotspots = []
        
        for file in self.codebase.files:
            for func in file.functions:
                # Check for nested loops
                if hasattr(func, "source"):
                    source = func.source
                    
                    # Count nested loops
                    for_count = source.count("for ")
                    while_count = source.count("while ")
                    
                    if for_count + while_count >= 2 and (
                        "for " in source and "for " in source[source.find("for ") + 4:]
                        or "while " in source and "while " in source[source.find("while ") + 6:]
                    ):
                        hotspots.append({
                            "name": func.name,
                            "file": file.filepath,
                            "line": getattr(func, "start_point", [0])[0] if hasattr(func, "start_point") else 0,
                            "type": "nested_loops",
                            "message": f"Function contains nested loops",
                        })
                        
                        self.issues.add(
                            Issue(
                                message=f"Function '{func.name}' contains nested loops",
                                severity=IssueSeverity.WARNING,
                                category=IssueCategory.PERFORMANCE_ISSUE,
                                location=CodeLocation(
                                    file=file.filepath,
                                    line=getattr(func, "start_point", [0])[0] if hasattr(func, "start_point") else None,
                                ),
                                suggestion="Consider refactoring to avoid nested loops or use more efficient algorithms.",
                            )
                        )
        
        # Store results
        self.performance.hotspots = {
            "hotspots": hotspots,
            "summary": {
                "total_hotspots": len(hotspots),
            },
        }
    
    def _analyze_types(self) -> None:
        """Analyze type issues."""
        logger.info("Analyzing types")
        
        # Initialize type analysis
        self.type_analysis = TypeAnalysis()
        
        # This would typically use a type checker like mypy
        # For this example, we'll just do some basic checks
        
        type_errors = []
        
        for file in self.codebase.files:
            for func in file.functions:
                # Check for missing return type annotations
                if (
                    hasattr(func, "return_type")
                    and not func.return_type
                    and not func.name.startswith("__")
                ):
                    type_errors.append({
                        "name": func.name,
                        "file": file.filepath,
                        "line": getattr(func, "start_point", [0])[0] if hasattr(func, "start_point") else 0,
                        "type": "missing_return_type",
                        "message": f"Function '{func.name}' is missing return type annotation",
                    })
                    
                    self.issues.add(
                        Issue(
                            message=f"Function '{func.name}' is missing return type annotation",
                            severity=IssueSeverity.INFO,
                            category=IssueCategory.TYPE_ERROR,
                            location=CodeLocation(
                                file=file.filepath,
                                line=getattr(func, "start_point", [0])[0] if hasattr(func, "start_point") else None,
                            ),
                            suggestion="Add a return type annotation to improve type safety and documentation.",
                        )
                    )
                
                # Check for missing parameter type annotations
                if hasattr(func, "parameters"):
                    for param in func.parameters:
                        if hasattr(param, "type") and not param.type:
                            type_errors.append({
                                "name": func.name,
                                "file": file.filepath,
                                "line": getattr(func, "start_point", [0])[0] if hasattr(func, "start_point") else 0,
                                "type": "missing_parameter_type",
                                "message": f"Parameter '{param.name}' in function '{func.name}' is missing type annotation",
                            })
                            
                            self.issues.add(
                                Issue(
                                    message=f"Parameter '{param.name}' in function '{func.name}' is missing type annotation",
                                    severity=IssueSeverity.INFO,
                                    category=IssueCategory.TYPE_ERROR,
                                    location=CodeLocation(
                                        file=file.filepath,
                                        line=getattr(func, "start_point", [0])[0] if hasattr(func, "start_point") else None,
                                    ),
                                    suggestion="Add a type annotation to improve type safety and documentation.",
                                )
                            )
        
        # Store results
        self.type_analysis.type_errors = {
            "type_errors": type_errors,
            "summary": {
                "total_type_errors": len(type_errors),
            },
        }
    
    def _output_results(self, output_file: str, output_format: str) -> None:
        """
        Output analysis results to a file.
        
        Args:
            output_file: Path to save results to
            output_format: Format for output (json, html, console)
        """
        if not self.analysis_result:
            raise ValueError("No analysis results to output")
        
        if output_format == "json":
            self._output_json(output_file)
        elif output_format == "html":
            self._output_html(output_file)
        else:
            logger.warning(f"Unsupported output format: {output_format}, using JSON")
            self._output_json(output_file)
    
    def _output_json(self, output_file: str) -> None:
        """Output analysis results as JSON."""
        with open(output_file, "w") as f:
            json.dump(self.analysis_result.to_dict(), f, indent=2)
            
        logger.info(f"JSON report saved to {output_file}")


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
