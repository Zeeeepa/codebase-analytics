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
import time
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
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
from codegen.sdk.core.statements.for_loop_statement import ForLoopStatement
from codegen.sdk.core.statements.if_block_statement import IfBlockStatement
from codegen.sdk.core.statements.try_catch_statement import TryCatchStatement
from codegen.sdk.core.statements.while_statement import WhileStatement
from codegen.sdk.core.expressions.binary_expression import BinaryExpression
from codegen.sdk.core.expressions.unary_expression import UnaryExpression
from codegen.sdk.core.expressions.comparison_expression import ComparisonExpression

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
    COMPREHENSIVE = "comprehensive"  # Comprehensive analysis type


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
    UNUSED_PARAMETER = "unused_parameter"  # Unused parameter in function
    MISSING_TYPE_ANNOTATION = "missing_type_annotation"  # Missing type annotation

    # Implementation Issues
    IMPLEMENTATION_ERROR = "implementation_error"  # Incorrect implementation
    MISSING_IMPLEMENTATION = "missing_implementation"  # Missing implementation
    EMPTY_FUNCTION = "empty_function"  # Empty function implementation
    UNREACHABLE_CODE = "unreachable_code"  # Unreachable code

    # Dependency Issues
    IMPORT_ERROR = "import_error"  # Import errors or issues
    DEPENDENCY_CYCLE = "dependency_cycle"  # Circular dependency
    MODULE_COUPLING = "module_coupling"  # High coupling between modules
    UNUSED_IMPORT = "unused_import"  # Unused import
    UNUSED_CLASS = "unused_class"  # Unused class
    UNUSED_FUNCTION = "unused_function"  # Unused function

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

    # ... rest of the IssueCollection class ...
    
    # Keep the existing IssueCollection class implementation from analyzer.py
    # and add any missing methods from comprehensive_analysis.py
    
    # ... existing code from analyzer.py ...

# ... existing code from analyzer.py ...

class CodebaseAnalyzer:
    """
    Comprehensive analyzer for codebases using the Codegen SDK.
    Implements deep analysis of code issues, dependencies, and metrics.
    """
    
    # ... existing code from analyzer.py ...
    
    def analyze(self, analysis_types: list[AnalysisType] | None = None, output_format: str = "json", output_file: str | None = None) -> "AnalysisResult":
        """
        Perform a comprehensive analysis of the codebase.
        
        Args:
            analysis_types: List of analysis types to perform
            output_format: Format for output (json, html, console)
            output_file: Optional path to save results to
            
        Returns:
            AnalysisResult containing the findings
        """
        # ... existing code from analyzer.py ...
        
        # Add support for comprehensive analysis
        if analysis_types and AnalysisType.COMPREHENSIVE in analysis_types:
            logger.info("Performing comprehensive analysis...")
            self._analyze_dead_code()
            self._analyze_parameter_issues()
            self._analyze_type_annotations()
            self._analyze_circular_dependencies()
            self._analyze_implementation_issues()
        
        # ... rest of the existing analyze method ...
    
    # Add methods from comprehensive_analysis.py
    
    def _analyze_dead_code(self):
        """Find and log unused code (functions, classes, imports)."""
        # Find unused functions
        for func in self.codebase.functions:
            if not func.usages:
                self.issues.add_issue(Issue(
                    message=f"Unused function: {func.name}",
                    severity=IssueSeverity.WARNING,
                    location=CodeLocation(file=getattr(func, 'filepath', 'Unknown')),
                    category=IssueCategory.UNUSED_FUNCTION,
                    suggestion="Consider removing this unused function or documenting why it's needed.",
                    symbol=func.name
                ))
        
        # Find unused classes
        for cls in self.codebase.classes:
            if not cls.usages:
                self.issues.add_issue(Issue(
                    message=f"Unused class: {cls.name}",
                    severity=IssueSeverity.WARNING,
                    location=CodeLocation(file=getattr(cls, 'filepath', 'Unknown')),
                    category=IssueCategory.UNUSED_CLASS,
                    suggestion="Consider removing this unused class or documenting why it's needed.",
                    symbol=cls.name
                ))
        
        # Find unused imports
        for file in self.codebase.files:
            if hasattr(file, 'imports'):
                for imp in file.imports:
                    if not imp.usages:
                        self.issues.add_issue(Issue(
                            message=f"Unused import: {imp.name}",
                            severity=IssueSeverity.INFO,
                            location=CodeLocation(file=getattr(file, 'filepath', 'Unknown')),
                            category=IssueCategory.UNUSED_IMPORT,
                            suggestion="Consider removing this unused import.",
                            symbol=imp.name
                        ))
    
    def _analyze_parameter_issues(self):
        """Find and log parameter issues (unused, mismatches)."""
        for func in self.codebase.functions:
            if not hasattr(func, 'parameters'):
                continue
                
            # Check for unused parameters
            for param in func.parameters:
                if hasattr(func, 'code_block') and func.code_block:
                    param_name = getattr(param, 'name', '')
                    if param_name and param_name not in func.code_block.source:
                        self.issues.add_issue(Issue(
                            message=f"Unused parameter '{param_name}' in function '{func.name}'",
                            severity=IssueSeverity.WARNING,
                            location=CodeLocation(file=getattr(func, 'filepath', 'Unknown')),
                            category=IssueCategory.UNUSED_PARAMETER,
                            suggestion=f"Consider removing the unused parameter '{param_name}' or using it in the function body.",
                            symbol=func.name
                        ))
            
            # Check for parameter mismatches in function calls
            if hasattr(func, 'usages'):
                for usage in func.usages:
                    if hasattr(usage, 'args') and len(usage.args) != len(func.parameters):
                        self.issues.add_issue(Issue(
                            message=f"Parameter count mismatch in call to '{func.name}': expected {len(func.parameters)}, got {len(usage.args)}",
                            severity=IssueSeverity.ERROR,
                            location=CodeLocation(file=getattr(usage, 'filepath', 'Unknown')),
                            category=IssueCategory.PARAMETER_MISMATCH,
                            suggestion="Ensure the function is called with the correct number of arguments.",
                            symbol=func.name
                        ))
    
    def _analyze_type_annotations(self):
        """Find and log type annotation issues."""
        for func in self.codebase.functions:
            # Check for missing return type annotations
            if hasattr(func, 'return_type') and not func.return_type:
                self.issues.add_issue(Issue(
                    message=f"Missing return type annotation in function '{func.name}'",
                    severity=IssueSeverity.INFO,
                    location=CodeLocation(file=getattr(func, 'filepath', 'Unknown')),
                    category=IssueCategory.MISSING_TYPE_ANNOTATION,
                    suggestion="Add a return type annotation to improve code clarity and enable better type checking.",
                    symbol=func.name
                ))
            
            # Check for missing parameter type annotations
            if hasattr(func, 'parameters'):
                for param in func.parameters:
                    if hasattr(param, 'type') and not param.type:
                        self.issues.add_issue(Issue(
                            message=f"Missing type annotation for parameter '{param.name}' in function '{func.name}'",
                            severity=IssueSeverity.INFO,
                            location=CodeLocation(file=getattr(func, 'filepath', 'Unknown')),
                            category=IssueCategory.MISSING_TYPE_ANNOTATION,
                            suggestion=f"Add a type annotation for parameter '{param.name}' to improve code clarity and enable better type checking.",
                            symbol=func.name
                        ))
    
    def _analyze_circular_dependencies(self):
        """Find and log circular dependencies."""
        if not nx:
            logger.warning("NetworkX not available, skipping circular dependency analysis")
            return
            
        # Build a directed graph of file dependencies
        G = nx.DiGraph()
        
        # Add nodes for all files
        for file in self.codebase.files:
            file_path = getattr(file, 'filepath', str(file))
            G.add_node(file_path)
        
        # Add edges for imports
        for file in self.codebase.files:
            file_path = getattr(file, 'filepath', str(file))
            if hasattr(file, 'imports'):
                for imp in file.imports:
                    if hasattr(imp, 'source_file') and imp.source_file:
                        import_path = getattr(imp.source_file, 'filepath', str(imp.source_file))
                        G.add_edge(file_path, import_path)
        
        # Find cycles
        try:
            cycles = list(nx.simple_cycles(G))
            for cycle in cycles:
                cycle_str = " -> ".join(cycle)
                self.issues.add_issue(Issue(
                    message=f"Circular dependency detected: {cycle_str}",
                    severity=IssueSeverity.ERROR,
                    location=CodeLocation(file=cycle[0]),
                    category=IssueCategory.DEPENDENCY_CYCLE,
                    suggestion="Refactor the code to break the circular dependency. Consider using dependency injection or moving shared code to a common module.",
                ))
        except nx.NetworkXNoCycle:
            # No cycles found
            pass
    
    def _analyze_implementation_issues(self):
        """Find and log implementation issues."""
        for func in self.codebase.functions:
            # Check for empty functions
            if hasattr(func, 'code_block') and func.code_block:
                code = func.code_block.source.strip()
                if not code or code == "pass":
                    self.issues.add_issue(Issue(
                        message=f"Empty function: {func.name}",
                        severity=IssueSeverity.WARNING,
                        location=CodeLocation(file=getattr(func, 'filepath', 'Unknown')),
                        category=IssueCategory.EMPTY_FUNCTION,
                        suggestion="Implement the function or document why it's intentionally empty.",
                        symbol=func.name
                    ))
                
                # Check for TODOs and FIXMEs
                if "TODO" in code or "FIXME" in code:
                    self.issues.add_issue(Issue(
                        message=f"Implementation note found in function '{func.name}': TODO/FIXME",
                        severity=IssueSeverity.INFO,
                        location=CodeLocation(file=getattr(func, 'filepath', 'Unknown')),
                        category=IssueCategory.IMPLEMENTATION_ERROR,
                        suggestion="Address the TODO or FIXME comment in the code.",
                        symbol=func.name
                    ))
                
                # Check for unreachable code (simple case after return)
                lines = code.split("\n")
                found_return = False
                for i, line in enumerate(lines):
                    if "return" in line and not line.strip().startswith("#"):
                        found_return = True
                    elif found_return and line.strip() and not line.strip().startswith("#"):
                        self.issues.add_issue(Issue(
                            message=f"Unreachable code detected after return in function '{func.name}'",
                            severity=IssueSeverity.ERROR,
                            location=CodeLocation(
                                file=getattr(func, 'filepath', 'Unknown'),
                                line=i + 1
                            ),
                            category=IssueCategory.UNREACHABLE_CODE,
                            suggestion="Remove or fix the unreachable code after the return statement.",
                            symbol=func.name
                        ))
                        break

    def calculate_cyclomatic_complexity(self, function):
        """Calculate the cyclomatic complexity of a function."""
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
                complexity += statement.condition.count(
                    " and "
                ) + statement.condition.count(" or ")

            if hasattr(statement, "nested_code_blocks"):
                for block in statement.nested_code_blocks:
                    complexity += analyze_block(block)

            return complexity

        def analyze_block(block):
            if not block or not hasattr(block, "statements"):
                return 0
            return sum(analyze_statement(stmt) for stmt in block.statements)

        return (
            1 + analyze_block(function.code_block) if hasattr(function, "code_block") else 1
        )

    def get_operators_and_operands(self, function):
        """Get operators and operands from a function."""
        operators = []
        operands = []

        if not hasattr(function, "code_block") or not function.code_block:
            return operators, operands

        for statement in function.code_block.statements:
            if hasattr(statement, "function_calls"):
                for call in statement.function_calls:
                    operators.append(call.name)
                    if hasattr(call, "args"):
                        for arg in call.args:
                            if hasattr(arg, "source"):
                                operands.append(arg.source)

            if hasattr(statement, "expressions"):
                for expr in statement.expressions:
                    if isinstance(expr, BinaryExpression):
                        operators.extend([op.source for op in expr.operators])
                        operands.extend([elem.source for elem in expr.elements])
                    elif isinstance(expr, UnaryExpression):
                        operators.append(expr.ts_node.type)
                        operands.append(expr.argument.source)
                    elif isinstance(expr, ComparisonExpression):
                        operators.extend([op.source for op in expr.operators])
                        operands.extend([elem.source for elem in expr.elements])

            if hasattr(statement, "expression"):
                expr = statement.expression
                if isinstance(expr, BinaryExpression):
                    operators.extend([op.source for op in expr.operators])
                    operands.extend([elem.source for elem in expr.elements])
                elif isinstance(expr, UnaryExpression):
                    operators.append(expr.ts_node.type)
                    operands.append(expr.argument.source)
                elif isinstance(expr, ComparisonExpression):
                    operators.extend([op.source for op in expr.operators])
                    operands.extend([elem.source for elem in expr.elements])

        return operators, operands

    def calculate_halstead_volume(self, operators, operands):
        """Calculate Halstead volume metrics."""
        n1 = len(set(operators))  # Number of unique operators
        n2 = len(set(operands))   # Number of unique operands
        N1 = len(operators)       # Total number of operators
        N2 = len(operands)        # Total number of operands
        
        # Calculate program vocabulary and length
        n = n1 + n2               # Program vocabulary
        N = N1 + N2               # Program length
        
        # Calculate volume
        if n > 0:
            volume = N * math.log2(n)
        else:
            volume = 0
            
        return volume, N1, N2, n1, n2

    def calculate_maintainability_index(self, volume, complexity, loc):
        """Calculate maintainability index."""
        if volume <= 0 or complexity <= 0 or loc <= 0:
            return 100  # Default to perfect score for invalid inputs
            
        # Original formula: 171 - 5.2 * ln(volume) - 0.23 * complexity - 16.2 * ln(loc)
        # Adjusted to 0-100 scale
        mi = 171 - 5.2 * math.log(volume) - 0.23 * complexity - 16.2 * math.log(loc)
        
        # Normalize to 0-100 scale
        mi_normalized = max(0, min(100, mi * 100 / 171))
        
        return mi_normalized

    def calculate_doi(self, cls):
        """Calculate the depth of inheritance for a given class."""
        return len(cls.superclasses) if hasattr(cls, 'superclasses') else 0

    def count_lines(self, source_code):
        """Count different types of lines in source code."""
        if not source_code:
            return 0, 0, 0, 0
            
        lines = source_code.split('\n')
        loc = len(lines)  # Lines of code
        
        # Count comment lines
        comment_lines = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('/*') or stripped.endswith('*/'):
                comment_lines += 1
                
        # Count blank lines
        blank_lines = sum(1 for line in lines if not line.strip())
        
        # Calculate SLOC and LLOC
        sloc = loc - blank_lines  # Source lines of code (non-blank)
        lloc = sloc - comment_lines  # Logical lines of code (non-blank, non-comment)
        
        return loc, lloc, sloc, comment_lines

    # ... rest of the CodebaseAnalyzer class ...

# ... rest of the analyzer.py file ...

def analyze_codebase(
    repo_path: str | None = None,
    repo_url: str | None = None,
    output_file: str | None = None,
    analysis_types: list[str] | None = None,
    language: str | None = None,
    output_format: str = "json",
) -> "AnalysisResult":
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

