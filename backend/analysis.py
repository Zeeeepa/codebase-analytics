#!/usr/bin/env python3
"""
Enhanced Consolidated Analysis Module

This module contains ALL analysis functions consolidated from:
- analysis.py (existing analysis functions)
- analyzer.py (legacy analysis functions)
- comprehensive_analysis.py (additional analysis features)

Organized into logical sections for maintainability and enhanced with
additional features for more comprehensive analysis.
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
import sys
import logging
import tempfile

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
from codegen.configs.models.codebase import CodebaseConfig
from codegen.configs.models.secrets import SecretsConfig
from codegen.git.repo_operator.repo_operator import RepoOperator
from codegen.git.schemas.repo_config import RepoConfig
from codegen.sdk.codebase.config import ProjectConfig
from codegen.shared.enums.programming_language import ProgrammingLanguage

# ============================================================================
# SECTION 1: DATA CLASSES AND ENUMS
# ============================================================================

class InheritanceAnalysis:
    """Analysis of inheritance patterns in a codebase."""
    def __init__(self):
        self.inheritance_chains = []
        self.max_inheritance_depth = 0
        self.classes_with_multiple_inheritance = []
        self.interface_implementations = {}
        self.abstract_class_implementations = {}

class RecursionAnalysis:
    """Analysis of recursion patterns in a codebase."""
    def __init__(self):
        self.recursive_functions = []
        self.max_recursion_depth = 0
        self.mutual_recursion_groups = []

class SymbolInfo:
    """Information about a symbol in the codebase."""
    def __init__(self, name, symbol_type, file_path, line_number):
        self.name = name
        self.symbol_type = symbol_type
        self.file_path = file_path
        self.line_number = line_number
        self.references = []
        self.context = {}

class DependencyAnalysis:
    """Comprehensive dependency analysis results."""
    def __init__(self):
        self.total_dependencies = 0
        self.circular_dependencies = []
        self.dependency_depth = 0
        self.external_dependencies = []
        self.internal_dependencies = []
        self.dependency_graph = {}
        self.critical_dependencies = []
        self.unused_dependencies = []

class CallGraphAnalysis:
    """Call graph analysis results."""
    def __init__(self):
        self.total_functions = 0
        self.entry_points = []
        self.leaf_functions = []
        self.call_chains = []
        self.max_call_depth = 0
        self.call_graph = {}
        self.function_connectivity = {}

class CodeQualityMetrics:
    """Code quality metrics."""
    def __init__(self):
        self.average_complexity = 0.0
        self.max_complexity = 0
        self.complex_functions = []
        self.duplication_percentage = 0.0
        self.duplicated_blocks = []
        self.maintainability_index = 0.0
        self.technical_debt_ratio = 0.0

class ArchitecturalInsights:
    """Architectural insights."""
    def __init__(self):
        self.component_coupling = {}
        self.component_cohesion = {}
        self.architectural_patterns = []
        self.modularity_score = 0.0
        self.layering_violations = []

class SecurityAnalysis:
    """Security analysis results."""
    def __init__(self):
        self.security_hotspots = []
        self.vulnerability_count = 0
        self.input_validation_issues = []
        self.security_risk_level = "low"

class PerformanceAnalysis:
    """Performance analysis results."""
    def __init__(self):
        self.performance_hotspots = []
        self.algorithmic_complexity_issues = []
        self.memory_usage_patterns = []

@dataclass
class AnalysisType(Enum):
    """Types of analysis available."""
    DEPENDENCY = auto()
    CALL_GRAPH = auto()
    CODE_QUALITY = auto()
    ARCHITECTURAL = auto()
    SECURITY = auto()
    PERFORMANCE = auto()
    INHERITANCE = auto()
    RECURSION = auto()

class IssueSeverity(str, Enum):
    """Severity levels for issues."""
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"

class IssueCategory(str, Enum):
    """Categories for issues."""
    IMPLEMENTATION_ERROR = "implementation_error"
    MISSPELLED_FUNCTION = "misspelled_function"
    NULL_REFERENCE = "null_reference"
    UNSAFE_ASSERTION = "unsafe_assertion"
    IMPROPER_EXCEPTION = "improper_exception"
    INCOMPLETE_IMPLEMENTATION = "incomplete_implementation"
    INEFFICIENT_PATTERN = "inefficient_pattern"
    CODE_DUPLICATION = "code_duplication"
    UNUSED_PARAMETER = "unused_parameter"
    REDUNDANT_CODE = "redundant_code"
    FORMATTING_ISSUE = "formatting_issue"
    SUBOPTIMAL_DEFAULT = "suboptimal_default"
    WRONG_PARAMETER = "wrong_parameter"
    RUNTIME_ERROR = "runtime_error"
    DEAD_CODE = "dead_code"
    SECURITY_VULNERABILITY = "security_vulnerability"
    PERFORMANCE_ISSUE = "performance_issue"

class IssueStatus(str, Enum):
    """Status of an issue."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    WONT_FIX = "wont_fix"
    DUPLICATE = "duplicate"

class ChangeType(str, Enum):
    """Types of changes that can be made to code."""
    ADD = "add"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"
    MOVE = "move"

class TransactionPriority(int, Enum):
    """Priority levels for transactions."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3



@dataclass
class CodeLocation:
    """Location of code in a file."""
    file_path: str
    line_start: int
    line_end: Optional[int] = None
    column_start: Optional[int] = None
    column_end: Optional[int] = None
    
    def __str__(self):
        if self.line_end and self.line_end != self.line_start:
            return f"{self.file_path}:{self.line_start}-{self.line_end}"
        return f"{self.file_path}:{self.line_start}"

@dataclass
class Issue:
    """Representation of a code issue."""
    id: str
    location: CodeLocation
    message: str
    severity: IssueSeverity
    category: IssueCategory
    status: IssueStatus = IssueStatus.OPEN
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    assigned_to: Optional[str] = None
    related_issues: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def update_status(self, status: IssueStatus):
        """Update the status of the issue."""
        self.status = status
        self.updated_at = datetime.now()
    
    def assign(self, assignee: str):
        """Assign the issue to someone."""
        self.assigned_to = assignee
        self.updated_at = datetime.now()
    
    def add_related_issue(self, issue_id: str):
        """Add a related issue."""
        if issue_id not in self.related_issues:
            self.related_issues.append(issue_id)
            self.updated_at = datetime.now()
    
    def add_tag(self, tag: str):
        """Add a tag to the issue."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now()
    
    def add_context(self, key: str, value: Any):
        """Add context information to the issue."""
        self.context[key] = value
        self.updated_at = datetime.now()
    
    def to_dict(self):
        """Convert the issue to a dictionary."""
        return {
            "id": self.id,
            "location": str(self.location),
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "assigned_to": self.assigned_to,
            "related_issues": self.related_issues,
            "tags": self.tags,
            "context": self.context
        }

@dataclass
class IssueCollection:
    """Collection of issues."""
    issues: List[Issue] = field(default_factory=list)
    
    def add_issue(self, issue: Issue):
        """Add an issue to the collection."""
        self.issues.append(issue)
    
    def get_issue_by_id(self, issue_id: str) -> Optional[Issue]:
        """Get an issue by its ID."""
        for issue in self.issues:
            if issue.id == issue_id:
                return issue
        return None
    
    def get_issues_by_severity(self, severity: IssueSeverity) -> List[Issue]:
        """Get issues by severity."""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def get_issues_by_category(self, category: IssueCategory) -> List[Issue]:
        """Get issues by category."""
        return [issue for issue in self.issues if issue.category == category]
    
    def get_issues_by_status(self, status: IssueStatus) -> List[Issue]:
        """Get issues by status."""
        return [issue for issue in self.issues if issue.status == status]
    
    def get_issues_by_file(self, file_path: str) -> List[Issue]:
        """Get issues by file path."""
        return [issue for issue in self.issues if issue.location.file_path == file_path]
    
    def get_issues_by_tag(self, tag: str) -> List[Issue]:
        """Get issues by tag."""
        return [issue for issue in self.issues if tag in issue.tags]
    
    def get_issues_by_assignee(self, assignee: str) -> List[Issue]:
        """Get issues by assignee."""
        return [issue for issue in self.issues if issue.assigned_to == assignee]
    
    def get_open_issues(self) -> List[Issue]:
        """Get all open issues."""
        return self.get_issues_by_status(IssueStatus.OPEN)
    
    def get_resolved_issues(self) -> List[Issue]:
        """Get all resolved issues."""
        return self.get_issues_by_status(IssueStatus.RESOLVED)
    
    def get_critical_issues(self) -> List[Issue]:
        """Get all critical issues."""
        return self.get_issues_by_severity(IssueSeverity.CRITICAL)
    
    def get_major_issues(self) -> List[Issue]:
        """Get all major issues."""
        return self.get_issues_by_severity(IssueSeverity.MAJOR)
    
    def get_minor_issues(self) -> List[Issue]:
        """Get all minor issues."""
        return self.get_issues_by_severity(IssueSeverity.MINOR)
    
    def get_info_issues(self) -> List[Issue]:
        """Get all info issues."""
        return self.get_issues_by_severity(IssueSeverity.INFO)
    
    def count_by_severity(self) -> Dict[str, int]:
        """Count issues by severity."""
        counts = {severity.value: 0 for severity in IssueSeverity}
        for issue in self.issues:
            counts[issue.severity.value] += 1
        return counts
    
    def count_by_category(self) -> Dict[str, int]:
        """Count issues by category."""
        counts = {category.value: 0 for category in IssueCategory}
        for issue in self.issues:
            counts[issue.category.value] += 1
        return counts
    
    def count_by_status(self) -> Dict[str, int]:
        """Count issues by status."""
        counts = {status.value: 0 for status in IssueStatus}
        for issue in self.issues:
            counts[issue.status.value] += 1
        return counts
    
    def to_dict(self):
        """Convert the collection to a dictionary."""
        return {
            "issues": [issue.to_dict() for issue in self.issues],
            "counts": {
                "by_severity": self.count_by_severity(),
                "by_category": self.count_by_category(),
                "by_status": self.count_by_status()
            }
        }

@dataclass
class AnalysisSummary:
    """Summary of analysis results."""
    total_files: int = 0
    total_lines: int = 0
    total_functions: int = 0
    total_classes: int = 0
    total_issues: int = 0
    issue_counts: Dict[str, int] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class CodeQualityResult:
    """Result of code quality analysis."""
    maintainability_index: float = 0.0
    cyclomatic_complexity: float = 0.0
    halstead_volume: float = 0.0
    source_lines_of_code: int = 0
    comment_density: float = 0.0
    duplication_percentage: float = 0.0
    technical_debt_ratio: float = 0.0
    issues: IssueCollection = field(default_factory=IssueCollection)



# ============================================================================
# SECTION 2: CORE COMPLEXITY AND METRICS FUNCTIONS
# ============================================================================

def calculate_cyclomatic_complexity(func: Function) -> int:
    """
    Calculate the cyclomatic complexity of a function.
    
    Cyclomatic complexity is a measure of the number of linearly independent paths
    through a program's source code. It is calculated as:
    
    M = E - N + 2P
    
    Where:
    - E is the number of edges in the control flow graph
    - N is the number of nodes in the control flow graph
    - P is the number of connected components (usually 1 for a single function)
    
    For simplicity, we count:
    - 1 for the function itself
    - +1 for each if, elif, for, while, except, with, assert
    - +1 for each boolean operator (and, or) in conditions
    
    Args:
        func: The function to analyze
        
    Returns:
        The cyclomatic complexity score
    """
    complexity = 1  # Base complexity for the function
    
    # Count control flow statements
    if_statements = func.find_statements(IfBlockStatement)
    complexity += len(if_statements)
    
    for_loops = func.find_statements(ForLoopStatement)
    complexity += len(for_loops)
    
    while_loops = func.find_statements(WhileStatement)
    complexity += len(while_loops)
    
    try_catch = func.find_statements(TryCatchStatement)
    complexity += len(try_catch)
    
    # Count boolean operators in conditions
    binary_expressions = func.find_expressions(BinaryExpression)
    for expr in binary_expressions:
        if expr.operator in ["and", "or"]:
            complexity += 1
    
    return complexity

def calculate_halstead_metrics(func: Function) -> Dict[str, float]:
    """
    Calculate Halstead complexity metrics for a function.
    
    Halstead metrics are based on the number of operators and operands in the code:
    - n1 = number of distinct operators
    - n2 = number of distinct operands
    - N1 = total number of operators
    - N2 = total number of operands
    
    From these, we calculate:
    - Program vocabulary: n = n1 + n2
    - Program length: N = N1 + N2
    - Calculated program length: N_hat = n1 * log2(n1) + n2 * log2(n2)
    - Volume: V = N * log2(n)
    - Difficulty: D = (n1/2) * (N2/n2)
    - Effort: E = D * V
    - Time to implement: T = E / 18 (in seconds)
    - Bugs delivered: B = V / 3000
    
    Args:
        func: The function to analyze
        
    Returns:
        Dictionary containing the Halstead metrics
    """
    # Get all expressions
    binary_expressions = func.find_expressions(BinaryExpression)
    unary_expressions = func.find_expressions(UnaryExpression)
    comparison_expressions = func.find_expressions(ComparisonExpression)
    
    # Count operators and operands
    operators = set()
    operands = set()
    total_operators = 0
    total_operands = 0
    
    # Process binary expressions
    for expr in binary_expressions:
        operators.add(expr.operator)
        total_operators += 1
        
        # Left and right are operands
        if hasattr(expr, 'left') and expr.left:
            operands.add(str(expr.left))
            total_operands += 1
        
        if hasattr(expr, 'right') and expr.right:
            operands.add(str(expr.right))
            total_operands += 1
    
    # Process unary expressions
    for expr in unary_expressions:
        operators.add(expr.operator)
        total_operators += 1
        
        if hasattr(expr, 'expression') and expr.expression:
            operands.add(str(expr.expression))
            total_operands += 1
    
    # Process comparison expressions
    for expr in comparison_expressions:
        operators.add(expr.operator)
        total_operators += 1
        
        if hasattr(expr, 'left') and expr.left:
            operands.add(str(expr.left))
            total_operands += 1
        
        if hasattr(expr, 'right') and expr.right:
            operands.add(str(expr.right))
            total_operands += 1
    
    # Calculate metrics
    n1 = len(operators)
    n2 = len(operands)
    N1 = total_operators
    N2 = total_operands
    
    # Handle edge cases
    if n1 == 0 or n2 == 0:
        return {
            "vocabulary": 0,
            "length": 0,
            "calculated_length": 0,
            "volume": 0,
            "difficulty": 0,
            "effort": 0,
            "time": 0,
            "bugs": 0
        }
    
    n = n1 + n2
    N = N1 + N2
    
    # Calculate log2 values safely
    log2_n1 = math.log2(n1) if n1 > 0 else 0
    log2_n2 = math.log2(n2) if n2 > 0 else 0
    log2_n = math.log2(n) if n > 0 else 0
    
    N_hat = n1 * log2_n1 + n2 * log2_n2
    V = N * log2_n
    D = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
    E = D * V
    T = E / 18
    B = V / 3000
    
    return {
        "vocabulary": n,
        "length": N,
        "calculated_length": N_hat,
        "volume": V,
        "difficulty": D,
        "effort": E,
        "time": T,
        "bugs": B
    }

def calculate_maintainability_index(func: Function) -> float:
    """
    Calculate the maintainability index for a function.
    
    The maintainability index is a composite metric that incorporates several other metrics:
    - Halstead Volume (V)
    - Cyclomatic Complexity (CC)
    - Lines of Code (LOC)
    - Comment Percentage (CP)
    
    The formula is:
    MI = 171 - 5.2 * ln(V) - 0.23 * CC - 16.2 * ln(LOC) + 50 * sin(sqrt(2.4 * CP))
    
    For simplicity, we use a modified version:
    MI = 171 - 5.2 * ln(V) - 0.23 * CC - 16.2 * ln(LOC)
    
    Args:
        func: The function to analyze
        
    Returns:
        The maintainability index (0-100, higher is better)
    """
    # Calculate Halstead Volume
    halstead_metrics = calculate_halstead_metrics(func)
    V = halstead_metrics["volume"]
    
    # Calculate Cyclomatic Complexity
    CC = calculate_cyclomatic_complexity(func)
    
    # Calculate Lines of Code
    LOC = func.line_range.stop - 1 - func.line_range.start + 1
    
    # Calculate Maintainability Index
    if V <= 0 or LOC <= 0:
        return 100  # Perfect score for very simple functions
    
    MI = 171 - 5.2 * math.log(V) - 0.23 * CC - 16.2 * math.log(LOC)
    
    # Normalize to 0-100 scale
    MI = max(0, min(100, MI))
    
    return MI

def calculate_code_duplication(codebase: Codebase, min_lines: int = 6) -> Dict[str, Any]:
    """
    Calculate code duplication metrics for a codebase.
    
    This function identifies duplicated code blocks across the codebase.
    A code block is considered duplicated if it appears in multiple places
    and has at least min_lines lines.
    
    Args:
        codebase: The codebase to analyze
        min_lines: Minimum number of lines for a block to be considered duplicated
        
    Returns:
        Dictionary containing duplication metrics and duplicated blocks
    """
    files = codebase.files
    
    # Extract content from all files
    file_contents = {}
    for file in files:
        if str(file.path).endswith(('.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.hpp')):
            file_contents[str(file.path)] = file.content.split('\n')
    
    # Find duplicated blocks
    duplicated_blocks = []
    processed_blocks = set()
    
    for file_path, lines in file_contents.items():
        for i in range(len(lines) - min_lines + 1):
            # Create a block of min_lines lines
            block = '\n'.join(lines[i:i+min_lines])
            block_hash = hash(block)
            
            # Skip if we've already processed this block
            if block_hash in processed_blocks:
                continue
            
            processed_blocks.add(block_hash)
            
            # Find all occurrences of this block
            occurrences = []
            for other_path, other_lines in file_contents.items():
                for j in range(len(other_lines) - min_lines + 1):
                    other_block = '\n'.join(other_lines[j:j+min_lines])
                    if block == other_block:
                        occurrences.append({
                            "file_path": other_path,
                            "start_line": j + 1,
                            "end_line": j + min_lines
                        })
            
            # If the block appears more than once, it's duplicated
            if len(occurrences) > 1:
                duplicated_blocks.append({
                    "block": block,
                    "occurrences": occurrences
                })
    
    # Calculate duplication percentage
    total_lines = sum(len(lines) for lines in file_contents.values())
    duplicated_lines = sum(len(block["occurrences"]) * min_lines for block in duplicated_blocks)
    duplication_percentage = (duplicated_lines / total_lines) * 100 if total_lines > 0 else 0
    
    return {
        "duplication_percentage": duplication_percentage,
        "duplicated_blocks": duplicated_blocks,
        "total_lines": total_lines,
        "duplicated_lines": duplicated_lines
    }

def calculate_technical_debt_ratio(codebase: Codebase) -> float:
    """
    Calculate the technical debt ratio for a codebase.
    
    Technical debt ratio is calculated as:
    TDR = (Remediation Cost / Development Cost) * 100
    
    Where:
    - Remediation Cost is the estimated effort to fix all issues
    - Development Cost is the estimated effort to develop the codebase
    
    For simplicity, we use a heuristic approach:
    - Each issue has a remediation cost based on its severity
    - Development cost is estimated based on codebase size and complexity
    
    Args:
        codebase: The codebase to analyze
        
    Returns:
        The technical debt ratio as a percentage
    """
    # Analyze the codebase to find issues
    issues = find_issues(codebase)
    
    # Assign remediation costs based on severity
    remediation_costs = {
        IssueSeverity.CRITICAL: 8,  # hours
        IssueSeverity.MAJOR: 4,     # hours
        IssueSeverity.MINOR: 1,     # hours
        IssueSeverity.INFO: 0.5     # hours
    }
    
    # Calculate total remediation cost
    total_remediation_cost = 0
    for issue in issues:
        total_remediation_cost += remediation_costs[issue.severity]
    
    # Estimate development cost based on codebase size and complexity
    files = codebase.files
    total_lines = sum(len(file.content.split('\n')) for file in files)
    
    # Rough estimate: 10 lines of code per hour
    development_cost = total_lines / 10
    
    # Calculate technical debt ratio
    if development_cost > 0:
        technical_debt_ratio = (total_remediation_cost / development_cost) * 100
    else:
        technical_debt_ratio = 0
    
    return technical_debt_ratio

def find_issues(codebase: Codebase) -> List[Issue]:
    """
    Find issues in a codebase.
    
    This function analyzes the codebase and identifies various issues such as:
    - Implementation errors
    - Misspelled function names
    - Null references
    - Unsafe assertions
    - Improper exception handling
    - Incomplete implementations
    - Inefficient patterns
    - Code duplication
    - Unused parameters
    - Redundant code
    - Formatting issues
    - Suboptimal defaults
    - Wrong parameters
    - Runtime errors
    - Dead code
    - Security vulnerabilities
    - Performance issues
    
    Args:
        codebase: The codebase to analyze
        
    Returns:
        List of issues found in the codebase
    """
    issues = []
    
    # Get all files
    files = codebase.files
    
    # Analyze each file
    for file in files:
        # Skip non-code files
        if not str(file.path).endswith(('.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.hpp')):
            continue
        
        # Find issues in the file
        file_issues = find_issues_in_file(file, codebase)
        issues.extend(file_issues)
    
    return issues

def find_issues_in_file(file: SourceFile, codebase: Codebase = None) -> List[Issue]:
    """
    Find issues in a file.
    
    Args:
        file: The file to analyze
        codebase: The codebase containing the file (optional)
        
    Returns:
        List of issues found in the file
    """
    issues = []
    
    # Get all functions in the file
    if codebase:
        functions = [f for f in codebase.functions if f.filepath == str(file.path)]
    else:
        functions = []
    
    # Analyze each function
    for func in functions:
        # Check for unused parameters
        unused_params = find_unused_parameters(func)
        for param in unused_params:
            issues.append(Issue(
                id=str(uuid.uuid4()),
                location=CodeLocation(
                    file_path=str(file.path),
                    line_start=func.line_range.start,
                    line_end=func.line_range.stop - 1
                ),
                message=f"Unused parameter '{param}' in function '{func.name}'",
                severity=IssueSeverity.MINOR,
                category=IssueCategory.UNUSED_PARAMETER
            ))
        
        # Check for complex functions
        complexity = calculate_cyclomatic_complexity(func)
        if complexity > 15:
            issues.append(Issue(
                id=str(uuid.uuid4()),
                location=CodeLocation(
                    file_path=str(file.path),
                    line_start=func.line_range.start,
                    line_end=func.line_range.stop - 1
                ),
                message=f"Function '{func.name}' has high cyclomatic complexity ({complexity})",
                severity=IssueSeverity.MAJOR if complexity > 25 else IssueSeverity.MINOR,
                category=IssueCategory.PERFORMANCE_ISSUE
            ))
        
        # Check for improper exception handling
        improper_exceptions = find_improper_exception_handling(func)
        for exception in improper_exceptions:
            issues.append(Issue(
                id=str(uuid.uuid4()),
                location=CodeLocation(
                    file_path=str(file.path),
                    line_start=exception["line"],
                    line_end=exception["line"]
                ),
                message=f"Improper exception handling: {exception['message']}",
                severity=IssueSeverity.MAJOR,
                category=IssueCategory.IMPROPER_EXCEPTION
            ))
    
    # Check for code duplication within the file
    duplications = find_duplicated_code_in_file(file)
    for duplication in duplications:
        issues.append(Issue(
            id=str(uuid.uuid4()),
            location=CodeLocation(
                file_path=str(file.path),
                line_start=duplication["start_line"],
                line_end=duplication["end_line"]
            ),
            message=f"Duplicated code block ({duplication['lines']} lines)",
            severity=IssueSeverity.MINOR,
            category=IssueCategory.CODE_DUPLICATION
        ))
    
    return issues

def find_unused_parameters(func: Function) -> List[str]:
    """
    Find unused parameters in a function.
    
    Args:
        func: The function to analyze
        
    Returns:
        List of unused parameter names
    """
    unused_params = []
    
    # Get all parameters
    params = func.parameters
    
    # Get function body as text
    body_text = func.body
    
    # Check each parameter
    for param in params:
        param_name = param.name
        
        # Skip self and cls parameters in methods
        if param_name in ["self", "cls"]:
            continue
        
        # Check if the parameter is used in the function body
        if param_name not in body_text:
            unused_params.append(param_name)
    
    return unused_params

def find_improper_exception_handling(func: Function) -> List[Dict[str, Any]]:
    """
    Find improper exception handling in a function.
    
    Args:
        func: The function to analyze
        
    Returns:
        List of improper exception handling instances
    """
    improper_exceptions = []
    
    # Get all try-catch statements
    try_catch_statements = func.find_statements(TryCatchStatement)
    
    for stmt in try_catch_statements:
        # Check for bare except
        for handler in stmt.handlers:
            if handler.type is None or handler.type == "Exception":
                improper_exceptions.append({
                    "line": int(handler.start_line),
                    "message": "Bare except or catching Exception is too broad"
                })
            
            # Check for pass in except block
            if "pass" in handler.body:
                improper_exceptions.append({
                    "line": int(handler.start_line),
                    "message": "Empty except block (pass) silently ignores exceptions"
                })
    
    return improper_exceptions

def find_duplicated_code_in_file(file: SourceFile, min_lines: int = 6) -> List[Dict[str, Any]]:
    """
    Find duplicated code blocks within a file.
    
    Args:
        file: The file to analyze
        min_lines: Minimum number of lines for a block to be considered duplicated
        
    Returns:
        List of duplicated code blocks
    """
    duplicated_blocks = []
    
    # Get file content as lines
    lines = file.content.split('\n')
    
    # Find duplicated blocks
    processed_blocks = set()
    
    for i in range(len(lines) - min_lines + 1):
        # Create a block of min_lines lines
        block = '\n'.join(lines[i:i+min_lines])
        block_hash = hash(block)
        
        # Skip if we've already processed this block
        if block_hash in processed_blocks:
            continue
        
        processed_blocks.add(block_hash)
        
        # Find all occurrences of this block
        occurrences = []
        for j in range(len(lines) - min_lines + 1):
            other_block = '\n'.join(lines[j:j+min_lines])
            if block == other_block and i != j:
                occurrences.append({
                    "start_line": j + 1,
                    "end_line": j + min_lines
                })
        
        # If the block appears more than once, it's duplicated
        if len(occurrences) > 0:
            duplicated_blocks.append({
                "start_line": i + 1,
                "end_line": i + min_lines,
                "lines": min_lines,
                "occurrences": occurrences
            })
    
    return duplicated_blocks



# ============================================================================
# SECTION 3: DEPENDENCY ANALYSIS FUNCTIONS
# ============================================================================

def analyze_dependencies(codebase: Codebase) -> DependencyAnalysis:
    """
    Analyze dependencies in a codebase.
    
    This function analyzes the dependencies between files and modules in a codebase,
    identifying circular dependencies, dependency depth, and critical dependencies.
    
    Args:
        codebase: The codebase to analyze
        
    Returns:
        DependencyAnalysis object containing the results
    """
    dependency_analysis = DependencyAnalysis()
    
    # Build dependency graph
    dependency_graph = build_dependency_graph(codebase)
    dependency_analysis.dependency_graph = dependency_graph
    
    # Count total dependencies
    dependency_analysis.total_dependencies = sum(len(deps) for deps in dependency_graph.values())
    
    # Find circular dependencies
    circular_deps = find_circular_dependencies(dependency_graph)
    dependency_analysis.circular_dependencies = circular_deps
    
    # Calculate dependency depth
    dependency_analysis.dependency_depth = calculate_dependency_depth(dependency_graph)
    
    # Identify external dependencies
    dependency_analysis.external_dependencies = find_external_dependencies(codebase)
    
    # Identify internal dependencies
    dependency_analysis.internal_dependencies = find_internal_dependencies(codebase)
    
    # Identify critical dependencies
    dependency_analysis.critical_dependencies = find_critical_dependencies(dependency_graph)
    
    # Identify unused dependencies
    dependency_analysis.unused_dependencies = find_unused_dependencies(codebase)
    
    return dependency_analysis

def build_dependency_graph(codebase: Codebase) -> Dict[str, List[str]]:
    """
    Build a dependency graph for a codebase.
    
    Args:
        codebase: The codebase to analyze
        
    Returns:
        Dictionary mapping file paths to lists of dependencies
    """
    dependency_graph = {}
    
    # Get all files
    files = codebase.files
    
    # Build the graph
    for file in files:
        # Skip non-code files
        if not str(file.path).endswith(('.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.hpp')):
            continue
        
        # Get imports
        imports = [i for i in codebase.imports if str(i.file.path) == str(file.path)]
        
        # Map imports to file paths
        dependencies = []
        for imp in imports:
            # Skip external imports
            if isinstance(imp, ExternalModule):
                continue
            
            # Add the dependency
            if hasattr(imp, 'source_file') and imp.source_file:
                dependencies.append(str(imp.source_file.path))
        
        # Add to graph
        dependency_graph[str(file.path)] = dependencies
    
    return dependency_graph

def find_circular_dependencies(dependency_graph: Dict[str, List[str]]) -> List[List[str]]:
    """
    Find circular dependencies in a dependency graph.
    
    Args:
        dependency_graph: Dictionary mapping file paths to lists of dependencies
        
    Returns:
        List of circular dependency chains
    """
    circular_deps = []
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    for file, deps in dependency_graph.items():
        G.add_node(file)
        for dep in deps:
            G.add_edge(file, dep)
    
    # Find cycles
    try:
        cycles = list(nx.simple_cycles(G))
        circular_deps = cycles
    except nx.NetworkXNoCycle:
        # No cycles found
        pass
    
    return circular_deps

def calculate_dependency_depth(dependency_graph: Dict[str, List[str]]) -> int:
    """
    Calculate the maximum dependency depth in a dependency graph.
    
    Args:
        dependency_graph: Dictionary mapping file paths to lists of dependencies
        
    Returns:
        Maximum dependency depth
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    for file, deps in dependency_graph.items():
        G.add_node(file)
        for dep in deps:
            G.add_edge(file, dep)
    
    # Find the longest path
    try:
        # Convert to DAG by removing cycles
        DAG = nx.DiGraph(G)
        cycles = list(nx.simple_cycles(G))
        for cycle in cycles:
            if len(cycle) > 1:
                DAG.remove_edge(cycle[0], cycle[1])
        
        # Find the longest path in the DAG
        longest_path_length = 0
        for node in DAG.nodes():
            paths = nx.single_source_shortest_path_length(DAG, node)
            if paths:
                longest_path_length = max(longest_path_length, max(paths.values()))
        
        return longest_path_length
    except (nx.NetworkXError, ValueError):
        # Error calculating longest path
        return 0

def find_external_dependencies(codebase: Codebase) -> List[str]:
    """
    Find external dependencies in a codebase.
    
    Args:
        codebase: The codebase to analyze
        
    Returns:
        List of external dependencies
    """
    external_deps = set()
    
    # Get all files
    files = codebase.files
    
    # Find external imports
    for file in files:
        # Skip non-code files
        if not str(file.path).endswith(('.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.hpp')):
            continue
        
        # Get imports
        imports = [i for i in codebase.imports if str(i.file.path) == str(file.path)]
        
        # Find external imports
        for imp in imports:
            if isinstance(imp, ExternalModule):
                external_deps.add(imp.name)
    
    return list(external_deps)

def find_internal_dependencies(codebase: Codebase) -> List[Dict[str, str]]:
    """
    Find internal dependencies in a codebase.
    
    Args:
        codebase: The codebase to analyze
        
    Returns:
        List of internal dependencies
    """
    internal_deps = []
    
    # Get all files
    files = codebase.files
    
    # Find internal imports
    for file in files:
        # Skip non-code files
        if not str(file.path).endswith(('.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.hpp')):
            continue
        
        # Get imports
        imports = [i for i in codebase.imports if str(i.file.path) == str(file.path)]
        
        # Find internal imports
        for imp in imports:
            if not isinstance(imp, ExternalModule) and hasattr(imp, 'source_file') and imp.source_file:
                internal_deps.append({
                    "from": str(file.path),
                    "to": str(imp.source_file.path)
                })
    
    return internal_deps

def find_critical_dependencies(dependency_graph: Dict[str, List[str]]) -> List[str]:
    """
    Find critical dependencies in a dependency graph.
    
    Critical dependencies are files that many other files depend on.
    
    Args:
        dependency_graph: Dictionary mapping file paths to lists of dependencies
        
    Returns:
        List of critical dependencies
    """
    # Count how many files depend on each file
    dependency_counts = Counter()
    
    for file, deps in dependency_graph.items():
        for dep in deps:
            dependency_counts[dep] += 1
    
    # Find files with high dependency counts
    threshold = max(1, len(dependency_graph) // 10)  # At least 10% of files depend on it
    critical_deps = [dep for dep, count in dependency_counts.items() if count >= threshold]
    
    return critical_deps

def find_unused_dependencies(codebase: Codebase) -> List[Dict[str, Any]]:
    """
    Find unused dependencies in a codebase.
    
    Unused dependencies are imports that are not used in the file.
    
    Args:
        codebase: The codebase to analyze
        
    Returns:
        List of unused dependencies
    """
    unused_deps = []
    
    # Get all files
    files = codebase.files
    
    # Find unused imports
    for file in files:
        # Skip non-code files
        if not str(file.path).endswith(('.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.hpp')):
            continue
        
        # Get imports
        imports = [i for i in codebase.imports if str(i.file.path) == str(file.path)]
        
        # Check each import
        for imp in imports:
            # Skip external imports
            if isinstance(imp, ExternalModule):
                continue
            
            # Check if the import is used
            if hasattr(imp, 'symbol') and imp.symbol:
                symbol = imp.symbol
                
                # Get file content
                content = file.content
                
                # Check if the symbol is used
                if symbol.name not in content or symbol.name + "." not in content:
                    unused_deps.append({
                        "file": str(file.path),
                        "import": symbol.name,
                        "line": imp.line if hasattr(imp, 'line') else None
                    })
    
    return unused_deps

# ============================================================================
# SECTION 4: CALL GRAPH ANALYSIS FUNCTIONS
# ============================================================================

def analyze_call_graph(codebase: Codebase) -> CallGraphAnalysis:
    """
    Analyze the call graph of a codebase.
    
    This function analyzes the call graph of a codebase, identifying entry points,
    leaf functions, call chains, and function connectivity.
    
    Args:
        codebase: The codebase to analyze
        
    Returns:
        CallGraphAnalysis object containing the results
    """
    call_graph_analysis = CallGraphAnalysis()
    
    # Build call graph
    call_graph = build_call_graph(codebase)
    call_graph_analysis.call_graph = call_graph
    
    # Count total functions
    call_graph_analysis.total_functions = len(call_graph)
    
    # Find entry points
    call_graph_analysis.entry_points = find_entry_points(call_graph)
    
    # Find leaf functions
    call_graph_analysis.leaf_functions = find_leaf_functions(call_graph)
    
    # Find call chains
    call_graph_analysis.call_chains = find_call_chains(call_graph)
    
    # Calculate max call depth
    call_graph_analysis.max_call_depth = calculate_max_call_depth(call_graph)
    
    # Calculate function connectivity
    call_graph_analysis.function_connectivity = calculate_function_connectivity(call_graph)
    
    return call_graph_analysis

def build_call_graph(codebase: Codebase) -> Dict[str, List[str]]:
    """
    Build a call graph for a codebase.
    
    Args:
        codebase: The codebase to analyze
        
    Returns:
        Dictionary mapping function names to lists of called functions
    """
    call_graph = {}
    
    # Get all functions
    functions = []
    for file in codebase.files:
        functions.extend([f for f in codebase.functions if f.filepath == str(file.path)])
    
    # Build the graph
    for func in functions:
        # Get function name
        func_name = func.name
        if hasattr(func, 'parent') and func.parent:
            func_name = f"{func.parent.name}.{func_name}"
        
        # Get called functions
        called_functions = []
        
        # Parse function body to find function calls
        # This is a simplified approach and may not catch all calls
        body = func.body
        for other_func in functions:
            other_name = other_func.name
            if hasattr(other_func, 'parent') and other_func.parent:
                other_name = f"{other_func.parent.name}.{other_name}"
            
            # Check if the function is called
            if f"{other_name}(" in body:
                called_functions.append(other_name)
        
        # Add to graph
        call_graph[func_name] = called_functions
    
    return call_graph

def find_entry_points(call_graph: Dict[str, List[str]]) -> List[str]:
    """
    Find entry points in a call graph.
    
    Entry points are functions that are not called by any other function.
    
    Args:
        call_graph: Dictionary mapping function names to lists of called functions
        
    Returns:
        List of entry point function names
    """
    # Find all functions that are called
    called_functions = set()
    for func, called in call_graph.items():
        called_functions.update(called)
    
    # Find functions that are not called
    entry_points = [func for func in call_graph.keys() if func not in called_functions]
    
    return entry_points

def find_leaf_functions(call_graph: Dict[str, List[str]]) -> List[str]:
    """
    Find leaf functions in a call graph.
    
    Leaf functions are functions that do not call any other function.
    
    Args:
        call_graph: Dictionary mapping function names to lists of called functions
        
    Returns:
        List of leaf function names
    """
    # Find functions that do not call any other function
    leaf_functions = [func for func, called in call_graph.items() if not called]
    
    return leaf_functions

def find_call_chains(call_graph: Dict[str, List[str]]) -> List[List[str]]:
    """
    Find call chains in a call graph.
    
    A call chain is a sequence of function calls from an entry point to a leaf function.
    
    Args:
        call_graph: Dictionary mapping function names to lists of called functions
        
    Returns:
        List of call chains
    """
    call_chains = []
    
    # Find entry points
    entry_points = find_entry_points(call_graph)
    
    # Find call chains for each entry point
    for entry_point in entry_points:
        # Use DFS to find all paths
        def dfs(func, path):
            # Add the current function to the path
            path.append(func)
            
            # If this is a leaf function, add the path to call chains
            if not call_graph.get(func, []):
                call_chains.append(path.copy())
            
            # Explore called functions
            for called_func in call_graph.get(func, []):
                # Avoid cycles
                if called_func not in path:
                    dfs(called_func, path.copy())
        
        # Start DFS from the entry point
        dfs(entry_point, [])
    
    return call_chains

def calculate_max_call_depth(call_graph: Dict[str, List[str]]) -> int:
    """
    Calculate the maximum call depth in a call graph.
    
    Args:
        call_graph: Dictionary mapping function names to lists of called functions
        
    Returns:
        Maximum call depth
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    for func, called in call_graph.items():
        G.add_node(func)
        for called_func in called:
            G.add_edge(func, called_func)
    
    # Find the longest path
    try:
        # Convert to DAG by removing cycles
        DAG = nx.DiGraph(G)
        cycles = list(nx.simple_cycles(G))
        for cycle in cycles:
            if len(cycle) > 1:
                DAG.remove_edge(cycle[0], cycle[1])
        
        # Find the longest path in the DAG
        longest_path_length = 0
        for node in DAG.nodes():
            paths = nx.single_source_shortest_path_length(DAG, node)
            if paths:
                longest_path_length = max(longest_path_length, max(paths.values()))
        
        return longest_path_length
    except (nx.NetworkXError, ValueError):
        # Error calculating longest path
        return 0

def calculate_function_connectivity(call_graph: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
    """
    Calculate function connectivity metrics.
    
    Args:
        call_graph: Dictionary mapping function names to lists of called functions
        
    Returns:
        Dictionary mapping function names to connectivity metrics
    """
    connectivity = {}
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    for func, called in call_graph.items():
        G.add_node(func)
        for called_func in called:
            G.add_edge(func, called_func)
    
    # Calculate connectivity metrics for each function
    for func in call_graph.keys():
        # Calculate in-degree (number of functions that call this function)
        in_degree = G.in_degree(func) if func in G else 0
        
        # Calculate out-degree (number of functions this function calls)
        out_degree = G.out_degree(func) if func in G else 0
        
        # Calculate betweenness centrality (how important this function is for connecting other functions)
        try:
            betweenness = nx.betweenness_centrality(G).get(func, 0)
        except:
            betweenness = 0
        
        # Add to connectivity metrics
        connectivity[func] = {
            "in_degree": in_degree,
            "out_degree": out_degree,
            "betweenness": betweenness
        }
    
    return connectivity



# ============================================================================
# SECTION 5: ISSUE DETECTION FUNCTIONS
# ============================================================================

def detect_issues(codebase: Codebase) -> IssueCollection:
    """
    Detect issues in a codebase.
    
    This function analyzes the codebase and detects various issues such as:
    - Implementation errors
    - Misspelled function names
    - Null references
    - Unsafe assertions
    - Improper exception handling
    - Incomplete implementations
    - Inefficient patterns
    - Code duplication
    - Unused parameters
    - Redundant code
    - Formatting issues
    - Suboptimal defaults
    - Wrong parameters
    - Runtime errors
    - Dead code
    - Security vulnerabilities
    - Performance issues
    
    Args:
        codebase: The codebase to analyze
        
    Returns:
        IssueCollection containing the detected issues
    """
    issue_collection = IssueCollection()
    
    # Simple issue detection that works with TypeScript/JavaScript files
    import re
    
    for file in codebase.files:
        file_path = str(file.path)
        content = file.content
        lines = content.split('\n')
        
        # 1. TODO/FIXME comments
        for line_num, line in enumerate(lines, 1):
            if 'TODO' in line or 'FIXME' in line or 'HACK' in line:
                issue = Issue(
                    id="",
                    location=CodeLocation(file_path=file_path, line_start=line_num),
                    message=f"TODO/FIXME comment found: {line.strip()[:60]}...",
                    severity=IssueSeverity.INFO,
                    category=IssueCategory.FORMATTING_ISSUE
                )
                issue_collection.add_issue(issue)
        
        # 2. Console statements
        for line_num, line in enumerate(lines, 1):
            if 'console.log' in line or 'console.warn' in line or 'console.error' in line:
                issue = Issue(
                    id="",
                    location=CodeLocation(file_path=file_path, line_start=line_num),
                    message=f"Console statement found: {line.strip()[:60]}...",
                    severity=IssueSeverity.MINOR,
                    category=IssueCategory.FORMATTING_ISSUE
                )
                issue_collection.add_issue(issue)
        
        # 3. Long lines (>120 characters)
        for line_num, line in enumerate(lines, 1):
            if len(line) > 120:
                issue = Issue(
                    id="",
                    location=CodeLocation(file_path=file_path, line_start=line_num),
                    message=f"Line too long ({len(line)} chars)",
                    severity=IssueSeverity.MINOR,
                    category=IssueCategory.FORMATTING_ISSUE
                )
                issue_collection.add_issue(issue)
        
        # 4. Empty catch blocks
        for line_num, line in enumerate(lines, 1):
            if 'catch' in line and '{}' in line:
                issue = Issue(
                    id="",
                    location=CodeLocation(file_path=file_path, line_start=line_num),
                    message="Empty catch block - should handle errors properly",
                    severity=IssueSeverity.MAJOR,
                    category=IssueCategory.IMPLEMENTATION_ERROR
                )
                issue_collection.add_issue(issue)
        
        # 5. Potential security issues
        for line_num, line in enumerate(lines, 1):
            if 'eval(' in line:
                issue = Issue(
                    id="",
                    location=CodeLocation(file_path=file_path, line_start=line_num),
                    message="Use of eval() function - potential security risk",
                    severity=IssueSeverity.CRITICAL,
                    category=IssueCategory.SECURITY_VULNERABILITY
                )
                issue_collection.add_issue(issue)
        
        # 6. Hardcoded URLs
        for line_num, line in enumerate(lines, 1):
            if re.search(r'https?://[^\s\'"]+', line):
                issue = Issue(
                    id="",
                    location=CodeLocation(file_path=file_path, line_start=line_num),
                    message=f"Hardcoded URL found: {line.strip()[:60]}...",
                    severity=IssueSeverity.MINOR,
                    category=IssueCategory.FORMATTING_ISSUE
                )
                issue_collection.add_issue(issue)
        
        # 7. Missing error handling
        for line_num, line in enumerate(lines, 1):
            if 'fetch(' in line and 'catch' not in content[content.find(line):content.find(line)+200]:
                issue = Issue(
                    id="",
                    location=CodeLocation(file_path=file_path, line_start=line_num),
                    message="fetch() call without proper error handling",
                    severity=IssueSeverity.MAJOR,
                    category=IssueCategory.IMPLEMENTATION_ERROR
                )
                issue_collection.add_issue(issue)
    
    # Run the existing detection functions that are implemented
    try:
        implementation_errors = detect_implementation_errors(codebase)
        for issue in implementation_errors:
            issue_collection.add_issue(issue)
    except Exception:
        pass
    
    try:
        misspelled_functions = detect_misspelled_functions(codebase)
        for issue in misspelled_functions:
            issue_collection.add_issue(issue)
    except Exception:
        pass
    
    try:
        null_references = detect_null_references(codebase)
        for issue in null_references:
            issue_collection.add_issue(issue)
    except Exception:
        pass
    
    return issue_collection

def detect_implementation_errors(codebase: Codebase) -> List[Issue]:
    """
    Detect implementation errors in a codebase.
    
    Implementation errors are logical errors in the code that may cause incorrect behavior.
    
    Args:
        codebase: The codebase to analyze
        
    Returns:
        List of issues representing implementation errors
    """
    issues = []
    
    # Get all files
    files = codebase.files
    
    # Analyze each file
    for file in files:
        # Skip non-code files
        if not str(file.path).endswith(('.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.hpp')):
            continue
        
        # Get all functions from codebase ([f for f in codebase.functions if f.filepath == str(file.path)] not available)
        file_functions = [f for f in codebase.functions if f.filepath == str(file.path)]
        
        # Analyze each function
        for func in file_functions:
            # Check for common implementation errors
            
            # Check for unreachable code
            unreachable_code = find_unreachable_code(func)
            for code in unreachable_code:
                issues.append(Issue(
                    id=str(uuid.uuid4()),
                    location=CodeLocation(
                        file_path=str(file.path),
                        line_start=code["line"],
                        line_end=code["line"]
                    ),
                    message=f"Unreachable code: {code['message']}",
                    severity=IssueSeverity.MAJOR,
                    category=IssueCategory.IMPLEMENTATION_ERROR
                ))
            
            # Check for infinite loops
            infinite_loops = find_infinite_loops(func)
            for loop in infinite_loops:
                issues.append(Issue(
                    id=str(uuid.uuid4()),
                    location=CodeLocation(
                        file_path=str(file.path),
                        line_start=loop["line"],
                        line_end=loop["line"]
                    ),
                    message=f"Potential infinite loop: {loop['message']}",
                    severity=IssueSeverity.CRITICAL,
                    category=IssueCategory.IMPLEMENTATION_ERROR
                ))
            
            # Check for off-by-one errors
            off_by_one_errors = find_off_by_one_errors(func)
            for error in off_by_one_errors:
                issues.append(Issue(
                    id=str(uuid.uuid4()),
                    location=CodeLocation(
                        file_path=str(file.path),
                        line_start=error["line"],
                        line_end=error["line"]
                    ),
                    message=f"Potential off-by-one error: {error['message']}",
                    severity=IssueSeverity.MAJOR,
                    category=IssueCategory.IMPLEMENTATION_ERROR
                ))
    
    return issues

def find_unreachable_code(func: Function) -> List[Dict[str, Any]]:
    """
    Find unreachable code in a function.
    
    Args:
        func: The function to analyze
        
    Returns:
        List of unreachable code instances
    """
    unreachable_code = []
    
    # Get function body as text
    body_text = func.body
    
    # Find return statements
    return_statements = re.finditer(r'^\s*return', body_text, re.MULTILINE)
    
    # Check for code after return statements
    for match in return_statements:
        # Get the line number
        line_number = body_text[:match.start()].count('\n') + func.line_range.start
        
        # Check if there's code after this return statement
        lines_after_return = body_text[match.end():].strip()
        if lines_after_return and not lines_after_return.startswith('#'):
            # Find the next non-empty line
            next_line = lines_after_return.split('\n')[0].strip()
            if next_line and not next_line.startswith('#'):
                unreachable_code.append({
                    "line": line_number + 1,
                    "message": f"Code after return statement: '{next_line}'"
                })
    
    return unreachable_code

def find_infinite_loops(func: Function) -> List[Dict[str, Any]]:
    """
    Find potential infinite loops in a function.
    
    Args:
        func: The function to analyze
        
    Returns:
        List of potential infinite loop instances
    """
    infinite_loops = []
    
    # Get while statements
    while_statements = func.find_statements(WhileStatement)
    
    # Check each while statement
    for stmt in while_statements:
        # Check if the condition is always true
        if stmt.condition == "True" or stmt.condition == "true" or stmt.condition == "1":
            # Check if there's a break statement in the loop
            if "break" not in stmt.body:
                infinite_loops.append({
                    "line": int(stmt.start_line),
                    "message": "While loop with condition that is always true and no break statement"
                })
    
    # Get for statements
    for_statements = func.find_statements(ForLoopStatement)
    
    # Check each for statement
    for stmt in for_statements:
        # Check if the loop variable is not modified in the loop body
        loop_var = stmt.target if hasattr(stmt, 'target') else None
        if loop_var and loop_var not in stmt.body:
            infinite_loops.append({
                "line": int(stmt.start_line),
                "message": f"For loop where loop variable '{loop_var}' is not modified in the loop body"
            })
    
    return infinite_loops

def find_off_by_one_errors(func: Function) -> List[Dict[str, Any]]:
    """
    Find potential off-by-one errors in a function.
    
    Args:
        func: The function to analyze
        
    Returns:
        List of potential off-by-one error instances
    """
    off_by_one_errors = []
    
    # Get function body as text
    body_text = func.body
    
    # Find array/list access with hardcoded indices
    array_accesses = re.finditer(r'(\w+)\[(\d+)\]', body_text)
    
    # Check each array access
    for match in array_accesses:
        # Get the array name and index
        array_name = match.group(1)
        index = int(match.group(2))
        
        # Check if the index is suspicious (very high or exactly at common boundaries)
        if index > 1000 or index == 0:
            # Get the line number
            line_number = body_text[:match.start()].count('\n') + func.line_range.start
            
            off_by_one_errors.append({
                "line": line_number,
                "message": f"Suspicious array index: {array_name}[{index}]"
            })
    
    # Find range/len expressions
    range_expressions = re.finditer(r'range\((\w+)\)', body_text)
    
    # Check each range expression
    for match in range_expressions:
        # Get the range argument
        range_arg = match.group(1)
        
        # Check if the range argument is a length
        if range_arg.startswith('len('):
            # This is fine, range(len(x)) is a common pattern
            continue
        
        # Get the line number
        line_number = body_text[:match.start()].count('\n') + func.line_range.start
        
        # Check if there's a comparison with the range variable
        # This is a heuristic and may produce false positives
        if re.search(r'for \w+ in range\(' + range_arg + r'\):\s+if \w+ ==', body_text):
            off_by_one_errors.append({
                "line": line_number,
                "message": f"Potential off-by-one error: comparison with range({range_arg}) variable"
            })
    
    return off_by_one_errors

def detect_misspelled_functions(codebase: Codebase) -> List[Issue]:
    """
    Detect misspelled function names in a codebase.
    
    Args:
        codebase: The codebase to analyze
        
    Returns:
        List of issues representing misspelled function names
    """
    issues = []
    
    # Get all functions
    all_functions = []
    for file in codebase.files:
        all_functions.extend([f for f in codebase.functions if f.filepath == str(file.path)])
    
    # Create a dictionary of function names
    function_names = {func.name: func for func in all_functions}
    
    # Get all files
    files = codebase.files
    
    # Analyze each file
    for file in files:
        # Skip non-code files
        if not str(file.path).endswith(('.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.hpp')):
            continue
        
        # Get file content
        content = file.content
        
        # Find function calls
        function_calls = re.finditer(r'(\w+)\(', content)
        
        # Check each function call
        for match in function_calls:
            # Get the function name
            func_name = match.group(1)
            
            # Skip if this is a known function
            if func_name in function_names:
                continue
            
            # Skip common built-in functions and keywords
            if func_name in ['print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple', 'super', 'isinstance', 'issubclass', 'hasattr', 'getattr', 'setattr', 'delattr', 'property', 'staticmethod', 'classmethod', 'enumerate', 'zip', 'map', 'filter', 'reduce', 'sorted', 'reversed', 'sum', 'min', 'max', 'any', 'all', 'abs', 'round', 'pow', 'divmod', 'complex', 'hash', 'id', 'type', 'repr', 'ascii', 'format', 'bin', 'oct', 'hex', 'chr', 'ord', 'input', 'open', 'exec', 'eval', 'compile', 'globals', 'locals', 'vars', 'dir', 'help', 'next', 'iter', 'slice', 'sorted', 'reversed', 'enumerate', 'zip', 'filter', 'map', 'sum', 'all', 'any', 'callable', 'delattr', 'getattr', 'hasattr', 'setattr', 'isinstance', 'issubclass', 'breakpoint', 'bytes', 'bytearray', 'memoryview', 'frozenset', 'object', 'property', 'classmethod', 'staticmethod', 'super', 'type', 'bool', 'complex', 'dict', 'float', 'int', 'list', 'set', 'str', 'tuple', 'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'finally', 'with', 'def', 'class', 'return', 'yield', 'lambda', 'pass', 'break', 'continue', 'assert', 'raise', 'import', 'from', 'as', 'global', 'nonlocal', 'and', 'or', 'not', 'is', 'in']:
                continue
            
            # Check if this might be a misspelled function
            for known_func in function_names:
                # Calculate Levenshtein distance
                distance = levenshtein_distance(func_name, known_func)
                
                # If the distance is small and the names are similar length, it might be misspelled
                if distance <= 2 and abs(len(func_name) - len(known_func)) <= 2:
                    # Get the line number
                    line_number = content[:match.start()].count('\n') + 1
                    
                    issues.append(Issue(
                        id=str(uuid.uuid4()),
                        location=CodeLocation(
                            file_path=str(file.path),
                            line_start=line_number,
                            line_end=line_number
                        ),
                        message=f"Potential misspelled function: '{func_name}' (did you mean '{known_func}'?)",
                        severity=IssueSeverity.MAJOR,
                        category=IssueCategory.MISSPELLED_FUNCTION
                    ))
                    
                    # Only report one potential correction per function call
                    break
    
    return issues

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings.
    
    The Levenshtein distance is the minimum number of single-character edits
    (insertions, deletions, or substitutions) required to change one string into the other.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Levenshtein distance between the two strings
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def detect_null_references(codebase: Codebase) -> List[Issue]:
    """
    Detect potential null references in a codebase.
    
    Args:
        codebase: The codebase to analyze
        
    Returns:
        List of issues representing potential null references
    """
    issues = []
    
    # Get all files
    files = codebase.files
    
    # Analyze each file
    for file in files:
        # Skip non-code files
        if not str(file.path).endswith(('.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.hpp')):
            continue
        
        # Get file content
        content = file.content
        
        # Find potential null references
        null_refs = []
        
        # Python: None
        none_refs = re.finditer(r'(\w+)\.(\w+)\s*\(', content)
        for match in none_refs:
            # Get the object and method
            obj = match.group(1)
            method = match.group(2)
            
            # Check if there's a null check before this call
            before_call = content[:match.start()]
            if f"if {obj} is None" not in before_call and f"if {obj} is not None" not in before_call:
                # Get the line number
                line_number = content[:match.start()].count('\n') + 1
                
                null_refs.append({
                    "line": line_number,
                    "message": f"Potential null reference: '{obj}.{method}()' without null check"
                })
        
        # JavaScript/TypeScript: null/undefined
        null_refs_js = re.finditer(r'(\w+)\.(\w+)\s*\(', content)
        for match in null_refs_js:
            # Get the object and method
            obj = match.group(1)
            method = match.group(2)
            
            # Check if there's a null check before this call
            before_call = content[:match.start()]
            if f"if ({obj} === null" not in before_call and f"if ({obj} !== null" not in before_call and f"if ({obj} === undefined" not in before_call and f"if ({obj} !== undefined" not in before_call:
                # Get the line number
                line_number = content[:match.start()].count('\n') + 1
                
                null_refs.append({
                    "line": line_number,
                    "message": f"Potential null reference: '{obj}.{method}()' without null check"
                })
        
        # Create issues for null references
        for ref in null_refs:
            issues.append(Issue(
                id=str(uuid.uuid4()),
                location=CodeLocation(
                    file_path=str(file.path),
                    line_start=ref["line"],
                    line_end=ref["line"]
                ),
                message=ref["message"],
                severity=IssueSeverity.MAJOR,
                category=IssueCategory.NULL_REFERENCE
            ))
    
    return issues



# ============================================================================
# SECTION 6: CODE QUALITY METRICS FUNCTIONS
# ============================================================================

def analyze_code_quality(codebase: Codebase) -> CodeQualityResult:
    """
    Analyze code quality metrics for a codebase.
    
    This function calculates various code quality metrics for a codebase, including:
    - Maintainability index
    - Cyclomatic complexity
    - Halstead volume
    - Source lines of code
    - Comment density
    - Duplication percentage
    - Technical debt ratio
    
    Args:
        codebase: The codebase to analyze
        
    Returns:
        CodeQualityResult object containing the metrics
    """
    result = CodeQualityResult()
    
    # Get all files
    files = codebase.files
    
    # Calculate metrics for each file
    total_maintainability = 0
    total_complexity = 0
    total_halstead = 0
    total_sloc = 0
    total_comments = 0
    file_count = 0
    
    for file in files:
        # Skip non-code files
        if not str(file.path).endswith(('.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.hpp')):
            continue
        
        file_count += 1
        
        # Calculate maintainability index
        maintainability = calculate_file_maintainability_index(file, codebase)
        total_maintainability += maintainability
        
        # Calculate cyclomatic complexity
        complexity = calculate_file_cyclomatic_complexity(file, codebase)
        total_complexity += complexity
        
        # Calculate Halstead volume
        halstead = calculate_file_halstead_volume(file, codebase)
        total_halstead += halstead
        
        # Calculate source lines of code
        sloc = calculate_source_lines_of_code(file)
        total_sloc += sloc
        
        # Calculate comment density
        comments = calculate_comment_density(file)
        total_comments += comments
    
    # Calculate averages
    if file_count > 0:
        result.maintainability_index = total_maintainability / file_count
        result.cyclomatic_complexity = total_complexity / file_count
        result.halstead_volume = total_halstead / file_count
        result.source_lines_of_code = total_sloc
        result.comment_density = total_comments / file_count
    
    # Calculate duplication percentage
    duplication = calculate_code_duplication(codebase)
    result.duplication_percentage = duplication["duplication_percentage"]
    
    # Calculate technical debt ratio
    result.technical_debt_ratio = calculate_technical_debt_ratio(codebase)
    
    # Find issues
    issues = detect_issues(codebase)
    result.issues = issues
    
    return result

def calculate_file_maintainability_index(file: SourceFile, codebase: Codebase) -> float:
    """
    Calculate the maintainability index for a file.
    
    Args:
        file: The file to analyze
        codebase: The codebase containing the file
        
    Returns:
        Maintainability index (0-100, higher is better)
    """
    # Get all functions in the file
    functions = [f for f in codebase.functions if f.filepath == str(file.path)]
    
    # Calculate maintainability index for each function
    total_maintainability = 0
    for func in functions:
        maintainability = calculate_maintainability_index(func)
        total_maintainability += maintainability
    
    # Calculate average maintainability index
    if functions:
        return total_maintainability / len(functions)
    else:
        return 100  # Perfect score for files with no functions
    
def calculate_file_cyclomatic_complexity(file: SourceFile, codebase: Codebase = None) -> float:
    """
    Calculate the average cyclomatic complexity for a file.
    
    Args:
        file: The file to analyze
        codebase: The codebase containing the file (optional)
        
    Returns:
        Average cyclomatic complexity
    """
    # Get all functions in the file
    if codebase:
        functions = [f for f in codebase.functions if f.filepath == str(file.path)]
    else:
        functions = []
    
    # Calculate cyclomatic complexity for each function
    total_complexity = 0
    for func in functions:
        complexity = calculate_cyclomatic_complexity(func)
        total_complexity += complexity
    
    # Calculate average cyclomatic complexity
    if functions:
        return total_complexity / len(functions)
    else:
        return 1  # Minimum complexity for files with no functions

def calculate_file_halstead_volume(file: SourceFile, codebase: Codebase = None) -> float:
    """
    Calculate the average Halstead volume for a file.
    
    Args:
        file: The file to analyze
        codebase: The codebase containing the file (optional)
        
    Returns:
        Average Halstead volume
    """
    # Get all functions in the file
    if codebase:
        functions = [f for f in codebase.functions if f.filepath == str(file.path)]
    else:
        functions = []
    
    # Calculate Halstead volume for each function
    total_volume = 0
    for func in functions:
        halstead = calculate_halstead_metrics(func)
        total_volume += halstead["volume"]
    
    # Calculate average Halstead volume
    if functions:
        return total_volume / len(functions)
    else:
        return 0  # Minimum volume for files with no functions

def calculate_source_lines_of_code(file: SourceFile) -> int:
    """
    Calculate the source lines of code for a file.
    
    Args:
        file: The file to analyze
        
    Returns:
        Number of source lines of code
    """
    # Get file content
    content = file.content
    
    # Split into lines
    lines = content.split('\n')
    
    # Count non-empty, non-comment lines
    sloc = 0
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('//') and not line.startswith('/*') and not line.startswith('*') and not line.startswith('*/'):
            sloc += 1
    
    return sloc

def calculate_comment_density(file: SourceFile) -> float:
    """
    Calculate the comment density for a file.
    
    Comment density is the ratio of comment lines to total lines.
    
    Args:
        file: The file to analyze
        
    Returns:
        Comment density (0-1, higher means more comments)
    """
    # Get file content
    content = file.content
    
    # Split into lines
    lines = content.split('\n')
    
    # Count comment lines and total lines
    comment_lines = 0
    total_lines = len(lines)
    
    for line in lines:
        line = line.strip()
        if line.startswith('#') or line.startswith('//') or line.startswith('/*') or line.startswith('*') or line.startswith('*/'):
            comment_lines += 1
    
    # Calculate comment density
    if total_lines > 0:
        return comment_lines / total_lines
    else:
        return 0

# ============================================================================
# SECTION 7: VISUALIZATION FUNCTIONS
# ============================================================================

def generate_dependency_graph_visualization(dependency_analysis: DependencyAnalysis) -> Dict[str, Any]:
    """
    Generate a visualization of the dependency graph.
    
    Args:
        dependency_analysis: DependencyAnalysis object containing the dependency graph
        
    Returns:
        Dictionary containing visualization data
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    for file, deps in dependency_analysis.dependency_graph.items():
        G.add_node(file)
        for dep in deps:
            G.add_edge(file, dep)
    
    # Calculate node positions using a layout algorithm
    pos = nx.spring_layout(G)
    
    # Convert positions to a format suitable for visualization
    nodes = []
    for node, position in pos.items():
        nodes.append({
            "id": node,
            "label": os.path.basename(node),
            "x": float(position[0]),
            "y": float(position[1]),
            "size": 1 + len(G.in_edges(node)) + len(G.out_edges(node))
        })
    
    # Create edges
    edges = []
    for source, target in G.edges():
        edges.append({
            "source": source,
            "target": target
        })
    
    # Create visualization data
    visualization = {
        "nodes": nodes,
        "edges": edges,
        "circular_dependencies": dependency_analysis.circular_dependencies,
        "critical_dependencies": dependency_analysis.critical_dependencies
    }
    
    return visualization

def generate_call_graph_visualization(call_graph_analysis: CallGraphAnalysis) -> Dict[str, Any]:
    """
    Generate a visualization of the call graph.
    
    Args:
        call_graph_analysis: CallGraphAnalysis object containing the call graph
        
    Returns:
        Dictionary containing visualization data
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    for func, called in call_graph_analysis.call_graph.items():
        G.add_node(func)
        for called_func in called:
            G.add_edge(func, called_func)
    
    # Calculate node positions using a layout algorithm
    pos = nx.spring_layout(G)
    
    # Convert positions to a format suitable for visualization
    nodes = []
    for node, position in pos.items():
        # Get node connectivity
        connectivity = call_graph_analysis.function_connectivity.get(node, {})
        in_degree = connectivity.get("in_degree", 0)
        out_degree = connectivity.get("out_degree", 0)
        betweenness = connectivity.get("betweenness", 0)
        
        nodes.append({
            "id": node,
            "label": node,
            "x": float(position[0]),
            "y": float(position[1]),
            "size": 1 + in_degree + out_degree,
            "color": "rgb(255, 0, 0)" if node in call_graph_analysis.entry_points else "rgb(0, 0, 255)" if node in call_graph_analysis.leaf_functions else "rgb(0, 255, 0)",
            "in_degree": in_degree,
            "out_degree": out_degree,
            "betweenness": betweenness
        })
    
    # Create edges
    edges = []
    for source, target in G.edges():
        edges.append({
            "source": source,
            "target": target
        })
    
    # Create visualization data
    visualization = {
        "nodes": nodes,
        "edges": edges,
        "entry_points": call_graph_analysis.entry_points,
        "leaf_functions": call_graph_analysis.leaf_functions,
        "max_call_depth": call_graph_analysis.max_call_depth
    }
    
    return visualization

def generate_issue_visualization(issue_collection: IssueCollection) -> Dict[str, Any]:
    """
    Generate a visualization of the issues.
    
    Args:
        issue_collection: IssueCollection object containing the issues
        
    Returns:
        Dictionary containing visualization data
    """
    # Count issues by file
    issues_by_file = {}
    for issue in issue_collection.issues:
        file_path = issue.location.file_path
        if file_path not in issues_by_file:
            issues_by_file[file_path] = {
                "total": 0,
                "critical": 0,
                "major": 0,
                "minor": 0,
                "info": 0
            }
        
        issues_by_file[file_path]["total"] += 1
        if issue.severity == IssueSeverity.CRITICAL:
            issues_by_file[file_path]["critical"] += 1
        elif issue.severity == IssueSeverity.MAJOR:
            issues_by_file[file_path]["major"] += 1
        elif issue.severity == IssueSeverity.MINOR:
            issues_by_file[file_path]["minor"] += 1
        elif issue.severity == IssueSeverity.INFO:
            issues_by_file[file_path]["info"] += 1
    
    # Count issues by category
    issues_by_category = issue_collection.count_by_category()
    
    # Count issues by severity
    issues_by_severity = issue_collection.count_by_severity()
    
    # Create visualization data
    visualization = {
        "issues_by_file": issues_by_file,
        "issues_by_category": issues_by_category,
        "issues_by_severity": issues_by_severity,
        "total_issues": len(issue_collection.issues)
    }
    
    return visualization

def generate_code_quality_visualization(code_quality_result: CodeQualityResult) -> Dict[str, Any]:
    """
    Generate a visualization of the code quality metrics.
    
    Args:
        code_quality_result: CodeQualityResult object containing the metrics
        
    Returns:
        Dictionary containing visualization data
    """
    # Create visualization data
    visualization = {
        "maintainability_index": code_quality_result.maintainability_index,
        "cyclomatic_complexity": code_quality_result.cyclomatic_complexity,
        "halstead_volume": code_quality_result.halstead_volume,
        "source_lines_of_code": code_quality_result.source_lines_of_code,
        "comment_density": code_quality_result.comment_density,
        "duplication_percentage": code_quality_result.duplication_percentage,
        "technical_debt_ratio": code_quality_result.technical_debt_ratio,
        "issues": generate_issue_visualization(code_quality_result.issues)
    }
    
    return visualization

def generate_repository_structure_visualization(codebase: Codebase) -> Dict[str, Any]:
    """
    Generate a visualization of the repository structure.
    
    Args:
        codebase: The codebase to analyze
        
    Returns:
        Dictionary containing visualization data
    """
    # Get all files
    files = codebase.files
    
    # Create a tree structure
    tree = {"name": "root", "children": []}
    
    # Add files to the tree
    for file in files:
        # Skip hidden files
        if os.path.basename(file.path).startswith('.'):
            continue
        
        # Split the path into components
        path_components = str(file.path).split('/')
        
        # Start at the root
        current_node = tree
        
        # Navigate through the path
        for i, component in enumerate(path_components):
            # Check if this is the last component (file)
            is_file = i == len(path_components) - 1
            
            # Find or create the node
            found = False
            for child in current_node["children"]:
                if child["name"] == component:
                    current_node = child
                    found = True
                    break
            
            if not found:
                # Create a new node
                new_node = {"name": component}
                if is_file:
                    # This is a file, add file-specific data
                    new_node["type"] = "file"
                    new_node["path"] = file.path
                    new_node["size"] = len(file.content)
                    
                    # Count issues in this file
                    issues = find_issues_in_file(file)
                    new_node["issues"] = {
                        "total": len(issues),
                        "critical": len([i for i in issues if i.severity == IssueSeverity.CRITICAL]),
                        "major": len([i for i in issues if i.severity == IssueSeverity.MAJOR]),
                        "minor": len([i for i in issues if i.severity == IssueSeverity.MINOR]),
                        "info": len([i for i in issues if i.severity == IssueSeverity.INFO])
                    }
                else:
                    # This is a directory, add children
                    new_node["type"] = "directory"
                    new_node["children"] = []
                
                current_node["children"].append(new_node)
                current_node = new_node
    
    # Create visualization data
    visualization = {
        "tree": tree,
        "total_files": len(files),
        "total_directories": count_directories(tree),
        "total_size": sum(len(file.content) if file.content else 0 for file in files)
    }
    
    return visualization

def count_directories(node: Dict[str, Any]) -> int:
    """
    Count the number of directories in a tree.
    
    Args:
        node: Tree node
        
    Returns:
        Number of directories
    """
    if node.get("type") == "file":
        return 0
    
    count = 1  # Count this directory
    
    for child in node.get("children", []):
        if child.get("type") == "directory":
            count += count_directories(child)
    
    return count

# ============================================================================
# SECTION 8: MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_codebase(codebase: Codebase) -> Dict[str, Any]:
    """
    Perform a comprehensive analysis of a codebase.
    
    This function analyzes a codebase and returns a comprehensive analysis result,
    including dependency analysis, call graph analysis, code quality metrics,
    issue detection, and visualizations.
    
    Args:
        codebase: The codebase to analyze
        
    Returns:
        Dictionary containing the analysis results
    """
    # Create analysis summary
    summary = AnalysisSummary()
    
    # Count files and lines
    files = codebase.files
    summary.total_files = len(files)
    summary.total_lines = sum(len(file.content.split('\n')) for file in files)
    
    # Count functions and classes
    functions = codebase.functions
    classes = codebase.classes
    
    summary.total_functions = len(functions)
    summary.total_classes = len(classes)
    
    # Analyze dependencies
    dependency_analysis = analyze_dependencies(codebase)
    
    # Analyze call graph
    call_graph_analysis = analyze_call_graph(codebase)
    
    # Analyze code quality
    code_quality_result = analyze_code_quality(codebase)
    
    # Detect issues
    issue_collection = detect_issues(codebase)
    summary.total_issues = len(issue_collection.issues)
    summary.issue_counts = issue_collection.count_by_severity()
    
    # Generate visualizations
    dependency_visualization = generate_dependency_graph_visualization(dependency_analysis)
    call_graph_visualization = generate_call_graph_visualization(call_graph_analysis)
    issue_visualization = generate_issue_visualization(issue_collection)
    code_quality_visualization = generate_code_quality_visualization(code_quality_result)
    repository_visualization = generate_repository_structure_visualization(codebase)
    
    # Generate recommendations
    recommendations = generate_recommendations(dependency_analysis, call_graph_analysis, code_quality_result, issue_collection)
    summary.recommendations = recommendations
    
    # Create analysis result
    analysis_result = {
        "summary": summary,
        "dependency_analysis": dependency_analysis,
        "call_graph_analysis": call_graph_analysis,
        "code_quality_result": code_quality_result,
        "issue_collection": issue_collection,
        "visualizations": {
            "dependency_graph": dependency_visualization,
            "call_graph": call_graph_visualization,
            "issues": issue_visualization,
            "code_quality": code_quality_visualization,
            "repository_structure": repository_visualization
        },
        "recommendations": recommendations
    }
    
    return analysis_result

def generate_recommendations(dependency_analysis: DependencyAnalysis, call_graph_analysis: CallGraphAnalysis, code_quality_result: CodeQualityResult, issue_collection: IssueCollection) -> List[str]:
    """
    Generate recommendations based on the analysis results.
    
    Args:
        dependency_analysis: DependencyAnalysis object
        call_graph_analysis: CallGraphAnalysis object
        code_quality_result: CodeQualityResult object
        issue_collection: IssueCollection object
        
    Returns:
        List of recommendations
    """
    recommendations = []
    
    # Recommendations based on dependency analysis
    if dependency_analysis.circular_dependencies:
        recommendations.append("Resolve circular dependencies to improve maintainability")
    
    if dependency_analysis.unused_dependencies:
        recommendations.append("Remove unused dependencies to reduce code bloat")
    
    # Recommendations based on call graph analysis
    if call_graph_analysis.max_call_depth > 10:
        recommendations.append("Reduce call depth to improve performance and maintainability")
    
    # Recommendations based on code quality
    if code_quality_result.maintainability_index < 50:
        recommendations.append("Improve code maintainability by reducing complexity and improving documentation")
    
    if code_quality_result.cyclomatic_complexity > 10:
        recommendations.append("Reduce cyclomatic complexity by breaking down complex functions")
    
    if code_quality_result.duplication_percentage > 10:
        recommendations.append("Reduce code duplication by extracting common functionality into reusable components")
    
    if code_quality_result.technical_debt_ratio > 20:
        recommendations.append("Reduce technical debt by addressing issues and improving code quality")
    
    # Recommendations based on issues
    critical_issues = issue_collection.get_critical_issues()
    if critical_issues:
        recommendations.append(f"Address {len(critical_issues)} critical issues to prevent potential failures")
    
    major_issues = issue_collection.get_major_issues()
    if major_issues:
        recommendations.append(f"Address {len(major_issues)} major issues to improve code quality")
    
    # Return recommendations
    return recommendations


# =============================================================================
# HALSTEAD METRICS ENGINE
# =============================================================================

def get_operators_and_operands(func: Function) -> Tuple[List[str], List[str]]:
    """
    Extract operators and operands from a function for Halstead metrics.
    
    Args:
        func: Function to analyze
        
    Returns:
        Tuple of (operators, operands) lists
    """
    operators = []
    operands = []
    
    try:
        # Parse the function source code
        if hasattr(func, 'source') and func.source:
            tree = ast.parse(func.source)
            
            # Walk through AST nodes
            for node in ast.walk(tree):
                # Operators
                if isinstance(node, ast.BinOp):
                    operators.append(type(node.op).__name__)
                elif isinstance(node, ast.UnaryOp):
                    operators.append(type(node.op).__name__)
                elif isinstance(node, ast.Compare):
                    operators.extend([type(op).__name__ for op in node.ops])
                elif isinstance(node, ast.BoolOp):
                    operators.append(type(node.op).__name__)
                elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                    operators.append(type(node).__name__)
                elif isinstance(node, ast.FunctionDef):
                    operators.append('def')
                elif isinstance(node, ast.ClassDef):
                    operators.append('class')
                elif isinstance(node, ast.Return):
                    operators.append('return')
                elif isinstance(node, ast.Assign):
                    operators.append('=')
                elif isinstance(node, ast.AugAssign):
                    operators.append(type(node.op).__name__ + '=')
                
                # Operands
                elif isinstance(node, ast.Name):
                    operands.append(node.id)
                elif isinstance(node, ast.Constant):
                    operands.append(str(node.value))
                elif isinstance(node, ast.Attribute):
                    operands.append(node.attr)
                elif isinstance(node, ast.Call) and hasattr(node.func, 'id'):
                    operands.append(node.func.id)
                    
    except Exception as e:
        print(f"Error parsing function {func.name}: {e}")
    
    return operators, operands

def calculate_halstead_volume(operators: List[str], operands: List[str]) -> Tuple[float, int, int, int, int]:
    """
    Calculate Halstead complexity metrics.
    
    Args:
        operators: List of operators
        operands: List of operands
        
    Returns:
        Tuple of (volume, n1, n2, N1, N2) where:
        - volume: Halstead volume
        - n1: Number of distinct operators
        - n2: Number of distinct operands  
        - N1: Total number of operators
        - N2: Total number of operands
    """
    n1 = len(set(operators))  # Distinct operators
    n2 = len(set(operands))   # Distinct operands
    N1 = len(operators)       # Total operators
    N2 = len(operands)        # Total operands
    
    # Calculate vocabulary and length
    vocabulary = n1 + n2
    length = N1 + N2
    
    # Calculate volume (V = N * log2(n))
    if vocabulary > 0:
        volume = length * math.log2(vocabulary)
    else:
        volume = 0
    
    return volume, n1, n2, N1, N2

def calculate_halstead_metrics(func: Function) -> Dict[str, Any]:
    """
    Calculate comprehensive Halstead metrics for a function.
    
    Args:
        func: Function to analyze
        
    Returns:
        Dictionary containing all Halstead metrics
    """
    operators, operands = get_operators_and_operands(func)
    volume, n1, n2, N1, N2 = calculate_halstead_volume(operators, operands)
    
    # Calculate additional metrics
    vocabulary = n1 + n2
    length = N1 + N2
    
    # Difficulty (D = (n1/2) * (N2/n2))
    difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
    
    # Effort (E = D * V)
    effort = difficulty * volume
    
    # Time to program (T = E / 18 seconds)
    time_to_program = effort / 18 if effort > 0 else 0
    
    # Bugs delivered (B = V / 3000)
    bugs_delivered = volume / 3000
    
    return {
        'function_name': func.name,
        'operators': {
            'distinct': n1,
            'total': N1,
            'list': list(set(operators))
        },
        'operands': {
            'distinct': n2,
            'total': N2,
            'list': list(set(operands))
        },
        'vocabulary': vocabulary,
        'length': length,
        'volume': volume,
        'difficulty': difficulty,
        'effort': effort,
        'time_to_program': time_to_program,
        'bugs_delivered': bugs_delivered
    }

# =============================================================================
# ADVANCED INHERITANCE ANALYSIS
# =============================================================================

@dataclass
class InheritanceNode:
    """Represents a class in the inheritance hierarchy."""
    name: str
    file_path: str
    superclasses: List[str] = field(default_factory=list)
    subclasses: List[str] = field(default_factory=list)
    depth: int = 0
    complexity_score: float = 0.0
    is_top_level: bool = False
    methods_count: int = 0
    attributes_count: int = 0

def analyze_inheritance_hierarchy(codebase: Codebase) -> Dict[str, Any]:
    """
    Analyze inheritance patterns and create hierarchy mappings.
    
    Args:
        codebase: The codebase to analyze
        
    Returns:
        Dictionary containing inheritance analysis results
    """
    inheritance_map = {}
    class_nodes = {}
    
    # Build inheritance nodes
    for cls in codebase.classes:
        node = InheritanceNode(
            name=cls.name,
            file_path=str(cls.filepath),
            superclasses=[sc.name for sc in cls.superclasses] if hasattr(cls, 'superclasses') else [],
            methods_count=len(cls.methods) if hasattr(cls, 'methods') else 0,
            attributes_count=len(cls.attributes) if hasattr(cls, 'attributes') else 0
        )
        class_nodes[cls.name] = node
        inheritance_map[cls.name] = node
    
    # Calculate inheritance relationships
    for class_name, node in class_nodes.items():
        # Find subclasses
        for other_name, other_node in class_nodes.items():
            if class_name in other_node.superclasses:
                node.subclasses.append(other_name)
    
    # Calculate inheritance depth and complexity
    for node in class_nodes.values():
        node.depth = len(node.superclasses)
        node.is_top_level = len(node.subclasses) == 0 and len(node.superclasses) > 0
        
        # Calculate complexity score based on multiple factors
        complexity_factors = [
            node.depth * 2,  # Inheritance depth
            len(node.subclasses),  # Number of subclasses
            len(node.superclasses),  # Multiple inheritance
            node.methods_count * 0.1,  # Method complexity
            node.attributes_count * 0.05  # Attribute complexity
        ]
        node.complexity_score = sum(complexity_factors)
    
    # Find top-level inheritance parents
    top_level_parents = [node for node in class_nodes.values() 
                        if len(node.subclasses) > 0 and len(node.superclasses) == 0]
    
    # Sort by complexity score
    top_level_parents.sort(key=lambda x: x.complexity_score, reverse=True)
    
    return {
        'inheritance_map': inheritance_map,
        'top_level_parents': top_level_parents[:10],  # Top 10
        'total_classes': len(class_nodes),
        'classes_with_inheritance': len([n for n in class_nodes.values() if n.superclasses]),
        'max_inheritance_depth': max([n.depth for n in class_nodes.values()], default=0),
        'multiple_inheritance_classes': [n for n in class_nodes.values() if len(n.superclasses) > 1]
    }

# =============================================================================
# ENTRY POINTS DETECTION AND HEAT MAPPING
# =============================================================================

@dataclass
class EntryPoint:
    """Represents a critical entry point in the codebase."""
    name: str
    file_path: str
    function_type: str  # 'main', 'api_endpoint', 'public_method', 'high_usage'
    usage_count: int = 0
    call_depth: int = 0
    complexity_score: float = 0.0
    heat_score: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)

def detect_entry_points(codebase: Codebase) -> Dict[str, Any]:
    """
    Detect critical entry points and calculate usage heat maps.
    
    Args:
        codebase: The codebase to analyze
        
    Returns:
        Dictionary containing entry point analysis
    """
    entry_points = []
    function_usage = {}
    
    # Analyze all functions
    for func in codebase.functions:
        usage_count = len(func.call_sites) if hasattr(func, 'call_sites') else 0
        calls_made = len(func.function_calls) if hasattr(func, 'function_calls') else 0
        
        function_usage[func.name] = {
            'usage_count': usage_count,
            'calls_made': calls_made,
            'file_path': str(func.filepath)
        }
        
        # Determine if this is an entry point
        is_entry_point = False
        entry_type = 'regular'
        
        # Check for main functions
        if func.name in ['main', '__main__', 'run', 'start', 'execute']:
            is_entry_point = True
            entry_type = 'main'
        
        # Check for high usage functions
        elif usage_count > 5:  # Threshold for high usage
            is_entry_point = True
            entry_type = 'high_usage'
        
        # Check for API endpoints (functions with decorators like @app.route)
        elif hasattr(func, 'decorators') and any('route' in str(d) for d in func.decorators):
            is_entry_point = True
            entry_type = 'api_endpoint'
        
        # Check for public methods (not starting with _)
        elif not func.name.startswith('_') and usage_count > 0:
            is_entry_point = True
            entry_type = 'public_method'
        
        if is_entry_point:
            # Calculate heat score based on multiple factors
            heat_score = (
                usage_count * 2 +  # Usage frequency
                calls_made * 0.5 +  # Complexity (how much it calls)
                (10 if entry_type == 'main' else 0) +  # Main function bonus
                (5 if entry_type == 'api_endpoint' else 0)  # API endpoint bonus
            )
            
            entry_point = EntryPoint(
                name=func.name,
                file_path=str(func.filepath),
                function_type=entry_type,
                usage_count=usage_count,
                call_depth=calls_made,
                heat_score=heat_score,
                dependencies=[call.name for call in func.function_calls] if hasattr(func, 'function_calls') else [],
                dependents=[site.parent_function.name for site in func.call_sites] if hasattr(func, 'call_sites') else []
            )
            entry_points.append(entry_point)
    
    # Sort by heat score
    entry_points.sort(key=lambda x: x.heat_score, reverse=True)
    
    # Create usage heat map
    heat_map = {}
    for func_name, data in function_usage.items():
        heat_map[func_name] = {
            'usage_intensity': min(data['usage_count'] / 10, 1.0),  # Normalize to 0-1
            'file_path': data['file_path'],
            'category': 'hot' if data['usage_count'] > 10 else 'warm' if data['usage_count'] > 3 else 'cold'
        }
    
    return {
        'entry_points': entry_points[:20],  # Top 20 entry points
        'heat_map': heat_map,
        'total_functions': len(function_usage),
        'high_usage_functions': len([f for f in function_usage.values() if f['usage_count'] > 5]),
        'unused_functions': len([f for f in function_usage.values() if f['usage_count'] == 0])
    }

# =============================================================================
# COMPREHENSIVE ISSUE DETECTION SYSTEM
# =============================================================================

def detect_comprehensive_issues(codebase: Codebase) -> Dict[str, Any]:
    """
    Detect comprehensive issues with full context.
    
    Args:
        codebase: The codebase to analyze
        
    Returns:
        Dictionary containing all detected issues with context
    """
    all_issues = []
    issue_categories = {
        'implementation_error': [],
        'misspelled_function': [],
        'null_reference': [],
        'unsafe_assertion': [],
        'improper_exception': [],
        'incomplete_implementation': [],
        'inefficient_pattern': [],
        'code_duplication': [],
        'unused_parameter': [],
        'redundant_code': [],
        'formatting_issue': [],
        'suboptimal_default': [],
        'wrong_parameter': [],
        'runtime_error': [],
        'dead_code': [],
        'security_vulnerability': [],
        'performance_issue': []
    }
    
    # Analyze each file
    for file in codebase.files:
        file_issues = analyze_file_issues(file)
        all_issues.extend(file_issues)
        
        # Categorize issues
        for issue in file_issues:
            if issue['category'] in issue_categories:
                issue_categories[issue['category']].append(issue)
    
    # Analyze functions for specific issues
    for func in codebase.functions:
        func_issues = analyze_function_issues(func)
        all_issues.extend(func_issues)
        
        for issue in func_issues:
            if issue['category'] in issue_categories:
                issue_categories[issue['category']].append(issue)
    
    # Create detailed issue report
    detailed_issues = []
    for i, issue in enumerate(all_issues, 1):
        detailed_issue = {
            'id': i,
            'file_path': issue['file_path'],
            'function_name': issue.get('function_name', 'N/A'),
            'parameter': issue.get('parameter', 'N/A'),
            'line_number': issue.get('line_number', 0),
            'category': issue['category'],
            'severity': issue['severity'],
            'message': issue['message'],
            'context': issue.get('context', {}),
            'suggestion': issue.get('suggestion', 'No suggestion available')
        }
        detailed_issues.append(detailed_issue)
    
    return {
        'total_issues': len(all_issues),
        'issues_by_category': {k: len(v) for k, v in issue_categories.items()},
        'detailed_issues': detailed_issues,
        'critical_issues': [i for i in detailed_issues if i['severity'] == 'critical'],
        'major_issues': [i for i in detailed_issues if i['severity'] == 'major'],
        'minor_issues': [i for i in detailed_issues if i['severity'] == 'minor']
    }

def analyze_file_issues(file: SourceFile) -> List[Dict[str, Any]]:
    """Analyze issues in a specific file."""
    issues = []
    
    try:
        if hasattr(file, 'content') and file.content:
            lines = file.content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                # Check for formatting issues
                if len(line) > 120:
                    issues.append({
                        'file_path': str(file.file_path),
                        'line_number': line_num,
                        'category': 'formatting_issue',
                        'severity': 'minor',
                        'message': f'Line too long ({len(line)} chars)',
                        'context': {'line_content': line[:100] + '...' if len(line) > 100 else line}
                    })
                
                # Check for potential null references
                if 'None.' in line or '.None' in line:
                    issues.append({
                        'file_path': str(file.file_path),
                        'line_number': line_num,
                        'category': 'null_reference',
                        'severity': 'major',
                        'message': 'Potential null reference detected',
                        'context': {'line_content': line.strip()}
                    })
                
                # Check for TODO/FIXME comments (incomplete implementation)
                if 'TODO' in line or 'FIXME' in line or 'HACK' in line:
                    issues.append({
                        'file_path': str(file.file_path),
                        'line_number': line_num,
                        'category': 'incomplete_implementation',
                        'severity': 'minor',
                        'message': 'Incomplete implementation detected',
                        'context': {'line_content': line.strip()}
                    })
                
                # Check for security vulnerabilities
                if any(vuln in line.lower() for vuln in ['eval(', 'exec(', 'input(', '__import__']):
                    issues.append({
                        'file_path': str(file.file_path),
                        'line_number': line_num,
                        'category': 'security_vulnerability',
                        'severity': 'critical',
                        'message': 'Potential security vulnerability detected',
                        'context': {'line_content': line.strip()}
                    })
    
    except Exception as e:
        print(f"Error analyzing file {file.file_path}: {e}")
    
    return issues

def analyze_function_issues(func: Function) -> List[Dict[str, Any]]:
    """Analyze issues in a specific function."""
    issues = []
    
    try:
        # Check for unused parameters
        if hasattr(func, 'parameters') and hasattr(func, 'source'):
            for param in func.parameters:
                if param.name not in func.source:
                    issues.append({
                        'file_path': str(func.filepath),
                        'function_name': func.name,
                        'parameter': param.name,
                        'line_number': func.line_range.start if hasattr(func, 'line_range') else 0,
                        'category': 'unused_parameter',
                        'severity': 'minor',
                        'message': f'Unused parameter "{param.name}"',
                        'context': {'function_signature': f"def {func.name}(...)"}
                    })
        
        # Check for dead code (functions with no callers)
        if hasattr(func, 'call_sites') and len(func.call_sites) == 0 and not func.name.startswith('_'):
            issues.append({
                'file_path': str(func.filepath),
                'function_name': func.name,
                'line_number': func.line_range.start if hasattr(func, 'line_range') else 0,
                'category': 'dead_code',
                'severity': 'minor',
                'message': f'Function "{func.name}" appears to be unused',
                'context': {'function_name': func.name}
            })
        
        # Check for high complexity
        if hasattr(func, 'source') and func.source:
            complexity = calculate_cyclomatic_complexity(func.source)
            if complexity > 10:
                issues.append({
                    'file_path': str(func.filepath),
                    'function_name': func.name,
                    'line_number': func.line_range.start if hasattr(func, 'line_range') else 0,
                    'category': 'performance_issue',
                    'severity': 'major',
                    'message': f'High cyclomatic complexity ({complexity})',
                    'context': {'complexity_score': complexity}
                })
    
    except Exception as e:
        print(f"Error analyzing function {func.name}: {e}")
    
    return issues

def calculate_cyclomatic_complexity(source_code: str) -> int:
    """Calculate cyclomatic complexity of source code."""
    try:
        tree = ast.parse(source_code)
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.Try):
                complexity += len(node.handlers)
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    except:
        return 1

# =============================================================================
# FUNCTION CONTEXT ANALYSIS ENGINE
# =============================================================================

def get_function_context(func: Function, codebase: Codebase) -> Dict[str, Any]:
    """
    Get comprehensive context for a function including implementation, dependencies, and usages.
    
    Args:
        func: Function to analyze
        codebase: The codebase containing the function
        
    Returns:
        Dictionary containing function context
    """
    context = {
        "implementation": {
            "source": func.source if hasattr(func, 'source') else '',
            "filepath": str(func.filepath),
            "line_start": func.line_range.start if hasattr(func, 'line_range') else 0,
            "line_end": func.line_range.stop if hasattr(func, 'line_range') else 0,
            "parameters": [param.name for param in func.parameters] if hasattr(func, 'parameters') else [],
            "return_type": func.return_type if hasattr(func, 'return_type') else None
        },
        "dependencies": [],
        "usages": [],
        "metrics": {},
        "issues": [],
        "call_chain": [],
        "impact_analysis": {}
    }
    
    # Add dependencies (what this function calls)
    if hasattr(func, 'function_calls'):
        for call in func.function_calls:
            dep_context = {
                "name": call.name,
                "type": "function_call",
                "file_path": str(call.filepath) if hasattr(call, 'filepath') else 'unknown'
            }
            context["dependencies"].append(dep_context)
    
    # Add usages (what calls this function)
    if hasattr(func, 'call_sites'):
        for usage in func.call_sites:
            usage_context = {
                "caller": usage.parent_function.name if hasattr(usage, 'parent_function') else 'unknown',
                "file_path": str(usage.filepath) if hasattr(usage, 'filepath') else 'unknown',
                "line_number": usage.start_point[0] if hasattr(usage, 'start_point') else 0
            }
            context["usages"].append(usage_context)
    
    # Calculate Halstead metrics
    try:
        halstead_metrics = calculate_halstead_metrics(func)
        context["metrics"] = halstead_metrics
    except Exception as e:
        context["metrics"] = {"error": str(e)}
    
    # Find issues related to this function
    func_issues = analyze_function_issues(func)
    context["issues"] = func_issues
    
    # Build call chain
    context["call_chain"] = build_call_chain(func, codebase)
    
    # Impact analysis
    context["impact_analysis"] = {
        "direct_callers": len(context["usages"]),
        "direct_calls": len(context["dependencies"]),
        "complexity_score": context["metrics"].get("difficulty", 0),
        "risk_level": determine_risk_level(func, context)
    }
    
    return context

def build_call_chain(func: Function, codebase: Codebase, max_depth: int = 5) -> List[Dict[str, Any]]:
    """Build call chain for a function."""
    call_chain = []
    visited = set()
    
    def traverse_calls(current_func, depth):
        if depth >= max_depth or current_func.name in visited:
            return
        
        visited.add(current_func.name)
        
        if hasattr(current_func, 'function_calls'):
            for call in current_func.function_calls:
                call_info = {
                    "function": call.name,
                    "depth": depth,
                    "file_path": str(call.filepath) if hasattr(call, 'filepath') else 'unknown'
                }
                call_chain.append(call_info)
                
                # Find the actual function object and recurse
                called_func = next((f for f in codebase.functions if f.name == call.name), None)
                if called_func:
                    traverse_calls(called_func, depth + 1)
    
    traverse_calls(func, 0)
    return call_chain

def determine_risk_level(func: Function, context: Dict[str, Any]) -> str:
    """Determine risk level of a function based on various factors."""
    risk_score = 0
    
    # High usage increases risk
    risk_score += len(context["usages"]) * 2
    
    # High complexity increases risk
    complexity = context["metrics"].get("difficulty", 0)
    if complexity > 10:
        risk_score += 10
    elif complexity > 5:
        risk_score += 5
    
    # Issues increase risk
    critical_issues = len([i for i in context["issues"] if i.get("severity") == "critical"])
    major_issues = len([i for i in context["issues"] if i.get("severity") == "major"])
    risk_score += critical_issues * 10 + major_issues * 5
    
    # Determine risk level
    if risk_score > 20:
        return "high"
    elif risk_score > 10:
        return "medium"
    else:
        return "low"

# =============================================================================
# ADDITIONAL FUNCTIONS FROM ANALYZER.PY
# =============================================================================

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
        
        # Get imports for this file
        for import_stmt in file.imports:
            if import_stmt.resolved_symbol:
                # If the import resolves to a symbol in our codebase
                if hasattr(import_stmt.resolved_symbol, 'file') and import_stmt.resolved_symbol.file:
                    dep_file_path = import_stmt.resolved_symbol.file.file_path
                    if dep_file_path != file.file_path:
                        dependencies.append(dep_file_path)
            elif import_stmt.external_module:
                # External dependency
                dependencies.append(f"external:{import_stmt.external_module.name}")
        
        dependency_graph[file.file_path] = list(set(dependencies))
    
    return dependency_graph


def get_symbol_references(codebase: Codebase, symbol_name: str) -> List[Dict[str, Any]]:
    """
    Find all references to a specific symbol in the codebase.
    
    Args:
        codebase: The Codebase object to search
        symbol_name: Name of the symbol to find references for
        
    Returns:
        List of dictionaries containing reference information
    """
    references = []
    
    # Find the symbol first
    target_symbols = [s for s in codebase.symbols if s.name == symbol_name]
    
    for target_symbol in target_symbols:
        # Find all edges where this symbol is used
        usage_edges = [
            edge for edge in codebase.ctx.edges 
            if edge[2].type == EdgeType.SYMBOL_USAGE and edge[1] == target_symbol.id
        ]
        
        for edge in usage_edges:
            source_node = codebase.ctx.get_node(edge[0])
            if source_node:
                reference_info = {
                    'symbol_name': symbol_name,
                    'used_in': source_node.name if hasattr(source_node, 'name') else str(source_node),
                    'file_path': source_node.file.file_path if hasattr(source_node, 'file') and source_node.file else 'unknown',
                    'line_number': int(source_node.start_line) if hasattr(source_node, 'start_line') and source_node.start_line else None,
                    'usage_type': 'symbol_usage'
                }
                references.append(reference_info)
    
    return references

# =============================================================================
# ADVANCED STATISTICS AND INSIGHTS
# =============================================================================

def get_advanced_codebase_statistics(codebase: Codebase) -> Dict[str, Any]:
    """
    Get comprehensive statistics and insights about the codebase.
    
    Args:
        codebase: The codebase to analyze
        
    Returns:
        Dictionary containing advanced statistics
    """
    stats = {
        'overview': {},
        'test_analysis': {},
        'recursive_functions': [],
        'most_called_functions': [],
        'unused_functions': [],
        'inheritance_analysis': {},
        'complexity_analysis': {},
        'architectural_insights': {}
    }
    
    # Basic overview
    stats['overview'] = {
        'total_classes': len(list(codebase.classes)),
        'total_functions': len(list(codebase.functions)),
        'total_imports': len(list(codebase.imports)),
        'total_files': len(list(codebase.files)),
        'total_symbols': len(list(codebase.symbols))
    }
    
    # Test analysis
    stats['test_analysis'] = analyze_test_files(codebase)
    
    # Find recursive functions
    stats['recursive_functions'] = find_recursive_functions(codebase)
    
    # Find most called functions
    stats['most_called_functions'] = find_most_called_functions(codebase)
    
    # Find unused functions (dead code)
    stats['unused_functions'] = find_unused_functions(codebase)
    
    # Inheritance analysis
    stats['inheritance_analysis'] = analyze_inheritance_patterns(codebase)
    
    # Complexity analysis
    stats['complexity_analysis'] = analyze_complexity_patterns(codebase)
    
    # Architectural insights
    stats['architectural_insights'] = generate_architectural_insights(codebase)
    
    return stats

def analyze_test_files(codebase: Codebase) -> Dict[str, Any]:
    """Analyze test files and test coverage patterns."""
    test_files = []
    test_classes = []
    test_functions = []
    
    # Identify test files
    for file in codebase.files:
        file_path = str(file.file_path).lower()
        if any(pattern in file_path for pattern in ['test_', '_test', 'tests/', '/test']):
            test_files.append({
                'file_path': str(file.file_path),
                'functions': len(file.functions) if hasattr(file, 'functions') else 0,
                'classes': len(file.classes) if hasattr(file, 'classes') else 0,
                'size': len(file.content) if hasattr(file, 'content') else 0
            })
    
    # Identify test classes
    for cls in codebase.classes:
        if 'test' in cls.name.lower() or any('test' in method.name.lower() for method in cls.methods if hasattr(cls, 'methods')):
            test_classes.append({
                'name': cls.name,
                'file_path': str(cls.filepath),
                'methods': len(cls.methods) if hasattr(cls, 'methods') else 0
            })
    
    # Identify test functions
    for func in codebase.functions:
        if func.name.startswith('test_') or 'test' in func.name.lower():
            test_functions.append({
                'name': func.name,
                'file_path': str(func.filepath),
                'complexity': calculate_cyclomatic_complexity(func.source) if hasattr(func, 'source') else 1
            })
    
    # Calculate test coverage insights
    file_test_counts = Counter([tf['file_path'] for tf in test_files])
    top_test_files = sorted(file_test_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        'total_test_files': len(test_files),
        'total_test_classes': len(test_classes),
        'total_test_functions': len(test_functions),
        'top_test_files': [{'file_path': path, 'test_count': count} for path, count in top_test_files],
        'test_coverage_ratio': len(test_functions) / max(len(list(codebase.functions)), 1),
        'average_tests_per_file': len(test_functions) / max(len(test_files), 1) if test_files else 0
    }

def find_recursive_functions(codebase: Codebase) -> List[Dict[str, Any]]:
    """Find functions that call themselves (recursive functions)."""
    recursive_functions = []
    
    for func in codebase.functions:
        if hasattr(func, 'function_calls'):
            # Check if function calls itself
            if any(call.name == func.name for call in func.function_calls):
                recursive_functions.append({
                    'name': func.name,
                    'file_path': str(func.filepath),
                    'recursion_type': 'direct',
                    'complexity': calculate_cyclomatic_complexity(func.source) if hasattr(func, 'source') else 1
                })
    
    return recursive_functions[:10]  # Top 10

def find_most_called_functions(codebase: Codebase) -> List[Dict[str, Any]]:
    """Find the most frequently called functions."""
    function_call_counts = {}
    
    for func in codebase.functions:
        call_count = len(func.call_sites) if hasattr(func, 'call_sites') else 0
        function_call_counts[func.name] = {
            'call_count': call_count,
            'file_path': str(func.filepath),
            'callers': [site.parent_function.name for site in func.call_sites] if hasattr(func, 'call_sites') else []
        }
    
    # Sort by call count
    sorted_functions = sorted(function_call_counts.items(), key=lambda x: x[1]['call_count'], reverse=True)
    
    return [
        {
            'name': name,
            'call_count': data['call_count'],
            'file_path': data['file_path'],
            'unique_callers': len(set(data['callers'])),
            'callers': data['callers'][:5]  # Top 5 callers
        }
        for name, data in sorted_functions[:10]  # Top 10 most called
    ]

def find_unused_functions(codebase: Codebase) -> List[Dict[str, Any]]:
    """Find functions that are never called (dead code)."""
    unused_functions = []
    
    for func in codebase.functions:
        call_count = len(func.call_sites) if hasattr(func, 'call_sites') else 0
        
        # Consider a function unused if it has no callers and doesn't look like an entry point
        if (call_count == 0 and 
            not func.name.startswith('_') and  # Not private
            func.name not in ['main', '__main__', 'run', 'start', 'execute'] and  # Not main functions
            not func.name.startswith('test_')):  # Not test functions
            
            unused_functions.append({
                'name': func.name,
                'file_path': str(func.filepath),
                'line_start': func.line_range.start if hasattr(func, 'line_range') else 0,
                'complexity': calculate_cyclomatic_complexity(func.source) if hasattr(func, 'source') else 1,
                'parameters': len(func.parameters) if hasattr(func, 'parameters') else 0
            })
    
    return unused_functions

def analyze_inheritance_patterns(codebase: Codebase) -> Dict[str, Any]:
    """Analyze inheritance patterns in the codebase."""
    if not list(codebase.classes):
        return {
            'deepest_inheritance_chain': None,
            'classes_with_multiple_inheritance': [],
            'inheritance_depth_distribution': {},
            'total_inheritance_relationships': 0
        }
    
    # Find class with deepest inheritance
    deepest_class = None
    max_depth = 0
    
    for cls in codebase.classes:
        depth = len(cls.superclasses) if hasattr(cls, 'superclasses') else 0
        if depth > max_depth:
            max_depth = depth
            deepest_class = cls
    
    # Find classes with multiple inheritance
    multiple_inheritance = []
    for cls in codebase.classes:
        if hasattr(cls, 'superclasses') and len(cls.superclasses) > 1:
            multiple_inheritance.append({
                'name': cls.name,
                'file_path': str(cls.filepath),
                'superclasses': [sc.name for sc in cls.superclasses],
                'inheritance_count': len(cls.superclasses)
            })
    
    # Calculate inheritance depth distribution
    depth_distribution = {}
    for cls in codebase.classes:
        depth = len(cls.superclasses) if hasattr(cls, 'superclasses') else 0
        depth_distribution[depth] = depth_distribution.get(depth, 0) + 1
    
    return {
        'deepest_inheritance_chain': {
            'class_name': deepest_class.name,
            'file_path': str(deepest_class.filepath),
            'depth': max_depth,
            'chain': [sc.name for sc in deepest_class.superclasses] if hasattr(deepest_class, 'superclasses') else []
        } if deepest_class else None,
        'classes_with_multiple_inheritance': multiple_inheritance,
        'inheritance_depth_distribution': depth_distribution,
        'total_inheritance_relationships': sum(len(cls.superclasses) if hasattr(cls, 'superclasses') else 0 for cls in codebase.classes)
    }

def analyze_complexity_patterns(codebase: Codebase) -> Dict[str, Any]:
    """Analyze complexity patterns across the codebase."""
    complexity_data = []
    
    for func in codebase.functions:
        if hasattr(func, 'source') and func.source:
            complexity = calculate_cyclomatic_complexity(func.source)
            complexity_data.append({
                'function_name': func.name,
                'file_path': str(func.filepath),
                'complexity': complexity,
                'lines_of_code': len(func.source.split('\n'))
            })
    
    if not complexity_data:
        return {
            'average_complexity': 0,
            'max_complexity': 0,
            'most_complex_functions': [],
            'complexity_distribution': {}
        }
    
    # Calculate statistics
    complexities = [item['complexity'] for item in complexity_data]
    average_complexity = sum(complexities) / len(complexities)
    max_complexity = max(complexities)
    
    # Find most complex functions
    most_complex = sorted(complexity_data, key=lambda x: x['complexity'], reverse=True)[:5]
    
    # Calculate complexity distribution
    complexity_distribution = {}
    for complexity in complexities:
        if complexity <= 5:
            category = 'low'
        elif complexity <= 10:
            category = 'medium'
        elif complexity <= 20:
            category = 'high'
        else:
            category = 'very_high'
        
        complexity_distribution[category] = complexity_distribution.get(category, 0) + 1
    
    return {
        'average_complexity': average_complexity,
        'max_complexity': max_complexity,
        'most_complex_functions': most_complex,
        'complexity_distribution': complexity_distribution,
        'functions_needing_refactoring': len([c for c in complexities if c > 10])
    }

def generate_architectural_insights(codebase: Codebase) -> Dict[str, Any]:
    """Generate high-level architectural insights."""
    insights = {
        'modularity_score': 0,
        'coupling_analysis': {},
        'cohesion_analysis': {},
        'architectural_patterns': [],
        'recommendations': []
    }
    
    # Calculate modularity score based on file organization
    files = list(codebase.files)
    total_files = len(files)
    
    if total_files > 0:
        # Simple modularity metric based on directory structure
        directories = set()
        for file in files:
            path_parts = str(file.file_path).split('/')
            if len(path_parts) > 1:
                directories.add('/'.join(path_parts[:-1]))
        
        insights['modularity_score'] = min(len(directories) / total_files, 1.0)
    
    # Analyze coupling (how interconnected modules are)
    import_graph = {}
    for file in files:
        file_imports = []
        if hasattr(file, 'imports'):
            for imp in file.imports:
                if hasattr(imp, 'external_module') and imp.external_module:
                    file_imports.append(imp.external_module.name)
        import_graph[str(file.file_path)] = file_imports
    
    # Calculate average imports per file (coupling metric)
    if import_graph:
        avg_imports = sum(len(imports) for imports in import_graph.values()) / len(import_graph)
        insights['coupling_analysis'] = {
            'average_imports_per_file': avg_imports,
            'highly_coupled_files': [
                {'file': file, 'import_count': len(imports)} 
                for file, imports in import_graph.items() 
                if len(imports) > avg_imports * 1.5
            ][:5]
        }
    
    # Generate recommendations
    recommendations = []
    
    if insights['modularity_score'] < 0.3:
        recommendations.append("Consider organizing code into more modular directory structure")
    
    if insights['coupling_analysis'].get('average_imports_per_file', 0) > 10:
        recommendations.append("High coupling detected - consider reducing dependencies between modules")
    
    complexity_analysis = analyze_complexity_patterns(codebase)
    if complexity_analysis['functions_needing_refactoring'] > 5:
        recommendations.append(f"{complexity_analysis['functions_needing_refactoring']} functions have high complexity and may need refactoring")
    
    insights['recommendations'] = recommendations
    
    return insights

# =============================================================================
# INTERACTIVE REPOSITORY STRUCTURE BUILDER
# =============================================================================

def build_interactive_repository_structure(codebase: Codebase) -> Dict[str, Any]:
    """
    Build an interactive repository structure with issue counts and clickable elements.
    
    Args:
        codebase: The codebase to analyze
        
    Returns:
        Dictionary containing interactive repository structure
    """
    # Get comprehensive issues first
    issues_analysis = detect_comprehensive_issues(codebase)
    detailed_issues = issues_analysis['detailed_issues']
    
    # Group issues by file path
    issues_by_file = {}
    for issue in detailed_issues:
        file_path = issue['file_path']
        if file_path not in issues_by_file:
            issues_by_file[file_path] = {
                'critical': 0,
                'major': 0,
                'minor': 0,
                'info': 0,
                'issues': []
            }
        issues_by_file[file_path][issue['severity']] += 1
        issues_by_file[file_path]['issues'].append(issue)
    
    # Build directory structure
    structure = {}
    
    for file in codebase.files:
        file_path = str(file.file_path)
        path_parts = file_path.split('/')
        
        # Navigate through directory structure
        current = structure
        for i, part in enumerate(path_parts[:-1]):
            if part not in current:
                current[part] = {
                    'type': 'directory',
                    'children': {},
                    'issues': {'critical': 0, 'major': 0, 'minor': 0, 'info': 0, 'total': 0}
                }
            current = current[part]['children']
        
        # Add file
        filename = path_parts[-1]
        file_issues = issues_by_file.get(file_path, {'critical': 0, 'major': 0, 'minor': 0, 'info': 0, 'issues': []})
        
        # Get symbols for this file
        file_symbols = {
            'classes': [],
            'functions': []
        }
        
        # Add classes
        for cls in codebase.classes:
            if str(cls.filepath) == file_path:
                class_info = {
                    'name': cls.name,
                    'type': 'class',
                    'methods': [],
                    'attributes': [],
                    'issues': []
                }
                
                # Add methods
                if hasattr(cls, 'methods'):
                    for method in cls.methods:
                        method_issues = [issue for issue in detailed_issues if issue['function_name'] == method.name]
                        class_info['methods'].append({
                            'name': method.name,
                            'parameters': [param.name for param in method.parameters] if hasattr(method, 'parameters') else [],
                            'issues': method_issues,
                            'context': get_function_context(method, codebase) if hasattr(method, 'source') else {}
                        })
                
                # Add attributes
                if hasattr(cls, 'attributes'):
                    class_info['attributes'] = [attr.name for attr in cls.attributes]
                
                # Add class-level issues
                class_info['issues'] = [issue for issue in detailed_issues if cls.name in issue.get('context', {}).get('line_content', '')]
                
                file_symbols['classes'].append(class_info)
        
        # Add functions
        for func in codebase.functions:
            if str(func.filepath) == file_path:
                func_issues = [issue for issue in detailed_issues if issue['function_name'] == func.name]
                function_info = {
                    'name': func.name,
                    'type': 'function',
                    'parameters': [param.name for param in func.parameters] if hasattr(func, 'parameters') else [],
                    'issues': func_issues,
                    'context': get_function_context(func, codebase),
                    'halstead_metrics': calculate_halstead_metrics(func) if hasattr(func, 'source') else {}
                }
                file_symbols['functions'].append(function_info)
        
        current[filename] = {
            'type': 'file',
            'path': file_path,
            'size': len(file.content) if hasattr(file, 'content') else 0,
            'issues': {
                'critical': file_issues['critical'],
                'major': file_issues['major'],
                'minor': file_issues['minor'],
                'info': file_issues['info'],
                'total': sum([file_issues['critical'], file_issues['major'], file_issues['minor'], file_issues['info']])
            },
            'symbols': file_symbols,
            'detailed_issues': file_issues.get('issues', [])
        }
    
    # Propagate issue counts up the directory tree
    def propagate_issues(node):
        if node['type'] == 'directory':
            total_issues = {'critical': 0, 'major': 0, 'minor': 0, 'info': 0, 'total': 0}
            for child in node['children'].values():
                child_issues = propagate_issues(child)
                for severity in total_issues:
                    total_issues[severity] += child_issues[severity]
            node['issues'] = total_issues
            return total_issues
        else:
            return node['issues']
    
    # Apply issue propagation to root directories
    for root_item in structure.values():
        if root_item['type'] == 'directory':
            propagate_issues(root_item)
    
    return {
        'structure': structure,
        'total_issues': len(detailed_issues),
        'issues_by_severity': {
            'critical': len([i for i in detailed_issues if i['severity'] == 'critical']),
            'major': len([i for i in detailed_issues if i['severity'] == 'major']),
            'minor': len([i for i in detailed_issues if i['severity'] == 'minor']),
            'info': len([i for i in detailed_issues if i['severity'] == 'info'])
        },
        'summary': {
            'total_files': len(list(codebase.files)),
            'total_directories': len([item for item in structure.values() if item['type'] == 'directory']),
            'total_functions': len(list(codebase.functions)),
            'total_classes': len(list(codebase.classes))
        }
    }
