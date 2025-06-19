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
    LOC = func.end_line - func.start_line + 1
    
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
            file_contents[file.path] = file.content.split('\n')
    
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
                    file_path=file.path,
                    line_start=func.start_line,
                    line_end=func.end_line
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
                    file_path=file.path,
                    line_start=func.start_line,
                    line_end=func.end_line
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
                    file_path=file.path,
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
                file_path=file.path,
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
                    "line": handler.start_line,
                    "message": "Bare except or catching Exception is too broad"
                })
            
            # Check for pass in except block
            if "pass" in handler.body:
                improper_exceptions.append({
                    "line": handler.start_line,
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
                dependencies.append(imp.source_file.path)
        
        # Add to graph
        dependency_graph[file.path] = dependencies
    
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
                    "from": file.path,
                    "to": imp.source_file.path
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
                        "file": file.path,
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
                        file_path=file.path,
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
                        file_path=file.path,
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
                        file_path=file.path,
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
        line_number = body_text[:match.start()].count('\n') + func.start_line
        
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
                    "line": stmt.start_line,
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
                "line": stmt.start_line,
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
            line_number = body_text[:match.start()].count('\n') + func.start_line
            
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
        line_number = body_text[:match.start()].count('\n') + func.start_line
        
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
                            file_path=file.path,
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
                    file_path=file.path,
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
        "total_size": sum(file.content for file in files)
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
                    'line_number': source_node.start_line if hasattr(source_node, 'start_line') else None,
                    'usage_type': 'symbol_usage'
                }
                references.append(reference_info)
    
    return references
