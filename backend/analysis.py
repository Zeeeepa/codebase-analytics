#!/usr/bin/env python3
"""
Comprehensive Codebase Analysis Engine
Consolidated implementation using Codegen SDK (graph-sitter) for deep codebase analysis.

This module provides comprehensive analysis including:
- Dead code detection (unused functions, classes, imports)
- Parameter issues (unused, mismatches)
- Type annotation analysis
- Circular dependency detection
- Implementation error detection
- Function context analysis
- Call graph metrics
- Dependency analysis
- AI-powered prompt generation

Uses the Codegen SDK's pre-computed graph structure for efficient analysis.
"""

import os
import sys
import time
import ast
import re
import networkx as nx
from datetime import datetime
from dataclasses import dataclass, asdict, field
import uuid
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from enum import Enum
from collections import Counter, defaultdict
from pathlib import Path

# Codegen SDK imports - core functionality
try:
    from codegen.sdk.core.codebase import Codebase
    from codegen.sdk.core.file import SourceFile, File
    from codegen.sdk.core.function import Function
    from codegen.sdk.core.class_definition import Class
    from codegen.sdk.core.symbol import Symbol
    from codegen.sdk.core.import_resolution import Import
    from codegen.sdk.core.external_module import ExternalModule
    from codegen.sdk.enums import EdgeType, SymbolType, NodeType
    from codegen.sdk.core.interfaces.editable import Editable
    
    # Import functions from Codegen SDK modules
    from codegen.sdk.codebase.codebase_analysis import (
        get_codebase_summary,
        get_file_summary,
        get_class_summary,
        get_function_summary,
        get_symbol_summary
    )
    from codegen.sdk.codebase.codebase_context import (
        CodebaseContext,
        get_function_context as sdk_get_function_context
    )
    from codegen.sdk.codebase.codebase_ai import (
        generate_system_prompt,
        generate_flag_system_prompt,
        generate_context,
        generate_tools,
        generate_flag_tools
    )
    
    CODEGEN_SDK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Codegen SDK not available: {e}")
    print("Falling back to basic analysis without SDK features")
    CODEGEN_SDK_AVAILABLE = False
    
    # Define minimal classes for fallback
    class Codebase: pass
    class SourceFile: pass
    class Function: pass
    class Class: pass
    class Symbol: pass
    class Import: pass
    class ExternalModule: pass
    class Editable: pass
    class CodebaseContext: pass
    
    # Define fallback functions
    def get_codebase_summary(codebase): return "Codebase summary not available (SDK not loaded)"
    def get_file_summary(file): return "File summary not available (SDK not loaded)"
    def get_class_summary(cls): return "Class summary not available (SDK not loaded)"
    def get_function_summary(func): return "Function summary not available (SDK not loaded)"
    def get_symbol_summary(symbol): return "Symbol summary not available (SDK not loaded)"
    def sdk_get_function_context(function): return {"error": "Codegen SDK not available"}
    def generate_system_prompt(target=None, context=None): return "System prompt generation not available (SDK not loaded)"
    def generate_flag_system_prompt(target, context=None): return "Flag system prompt generation not available (SDK not loaded)"
    def generate_context(context=None): return "Context generation not available (SDK not loaded)"
    def generate_tools(): return []
    def generate_flag_tools(): return []

# ====================================================================
# ENHANCED ISSUE DETECTION ARCHITECTURE
# ====================================================================

class IssueSeverity(Enum):
    """Severity levels for detected issues."""
    CRITICAL = "critical"  # Critical issues requiring immediate attention
    ERROR = "error"        # Errors that may cause runtime failures
    WARNING = "warning"    # Issues that should be addressed
    INFO = "info"         # Informational suggestions

class IssueCategory(Enum):
    """High-level categories for issue classification."""
    IMPLEMENTATION_ERROR = "implementation_error"
    SECURITY_VULNERABILITY = "security_vulnerability"
    PERFORMANCE_ISSUE = "performance_issue"
    CODE_QUALITY = "code_quality"
    DEAD_CODE = "dead_code"
    FORMATTING_ISSUE = "formatting_issue"
    TYPE_SAFETY = "type_safety"
    DEPENDENCY_ISSUE = "dependency_issue"

class IssueType(Enum):
    """Specific types of issues that can be detected."""
    
    # Dead Code Issues
    UNUSED_FUNCTION = "unused_function"
    UNUSED_CLASS = "unused_class"
    UNUSED_IMPORT = "unused_import"
    UNUSED_PARAMETER = "unused_parameter"
    EMPTY_FUNCTION = "empty_function"
    
    # Implementation Errors
    NULL_REFERENCE = "null_reference"
    TYPE_MISMATCH = "type_mismatch"
    UNDEFINED_VARIABLE = "undefined_variable"
    MISSING_RETURN = "missing_return"
    UNREACHABLE_CODE = "unreachable_code"
    INFINITE_LOOP = "infinite_loop"
    OFF_BY_ONE_ERROR = "off_by_one_error"
    
    # Function Issues
    MISSPELLED_FUNCTION = "misspelled_function"
    WRONG_PARAMETER_COUNT = "wrong_parameter_count"
    PARAMETER_TYPE_MISMATCH = "parameter_type_mismatch"
    PARAMETER_MISMATCH = "parameter_mismatch"
    
    # Type Issues
    MISSING_TYPE_ANNOTATION = "missing_type_annotation"
    
    # Dependency Issues
    CIRCULAR_DEPENDENCY = "circular_dependency"
    
    # Exception Handling
    IMPROPER_EXCEPTION_HANDLING = "improper_exception_handling"
    MISSING_ERROR_HANDLING = "missing_error_handling"
    UNSAFE_ASSERTION = "unsafe_assertion"
    RESOURCE_LEAK = "resource_leak"
    
    # Code Quality
    CODE_DUPLICATION = "code_duplication"
    INEFFICIENT_PATTERN = "inefficient_pattern"
    MAGIC_NUMBER = "magic_number"
    LONG_FUNCTION = "long_function"
    DEEP_NESTING = "deep_nesting"
    HIGH_COMPLEXITY = "high_complexity"
    
    # Formatting & Style
    INCONSISTENT_NAMING = "inconsistent_naming"
    MISSING_DOCUMENTATION = "missing_documentation"
    INCONSISTENT_INDENTATION = "inconsistent_indentation"
    IMPORT_ORGANIZATION = "import_organization"
    LINE_TOO_LONG = "line_too_long"
    
    # Runtime Risks
    DIVISION_BY_ZERO = "division_by_zero"
    ARRAY_BOUNDS = "array_bounds"
    STACK_OVERFLOW = "stack_overflow"
    CONCURRENCY_ISSUE = "concurrency_issue"
    
    # Security Vulnerabilities
    DANGEROUS_FUNCTION_USAGE = "dangerous_function_usage"
    POTENTIAL_INJECTION = "potential_injection"
    UNSAFE_EVAL = "unsafe_eval"
    
    # Performance Issues
    INEFFICIENT_LOOP = "inefficient_loop"
    MEMORY_LEAK = "memory_leak"
    UNNECESSARY_COMPUTATION = "unnecessary_computation"
    
    # General
    IMPLEMENTATION_ERROR = "implementation_error"
    INCOMPLETE_IMPLEMENTATION = "incomplete_implementation"

class IssueStatus(Enum):
    """Status tracking for issues."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    WONT_FIX = "wont_fix"
    DUPLICATE = "duplicate"

@dataclass
class CodeLocation:
    """Precise location of code in a file."""
    file_path: str
    line_start: int
    line_end: Optional[int] = None
    column_start: Optional[int] = None
    column_end: Optional[int] = None
    
    def __str__(self):
        if self.line_end and self.line_end != self.line_start:
            return f"{self.file_path}:{self.line_start}-{self.line_end}"
        return f"{self.file_path}:{self.line_start}"
    
    def __post_init__(self):
        if self.line_end is None:
            self.line_end = self.line_start

@dataclass
class Issue:
    """Enhanced representation of a code issue with comprehensive context."""
    # Core identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: Union[IssueType, str] = IssueType.IMPLEMENTATION_ERROR
    category: Union[IssueCategory, str] = IssueCategory.CODE_QUALITY
    severity: Union[IssueSeverity, str] = IssueSeverity.WARNING
    
    # Issue description
    message: str = ""
    suggestion: Optional[str] = None
    
    # Location information
    location: Optional[Union[CodeLocation, str]] = None
    item: Optional[Any] = None  # Original item for backward compatibility
    
    # Status and lifecycle
    status: IssueStatus = IssueStatus.OPEN
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Relationships and organization
    related_issues: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    assigned_to: Optional[str] = None
    
    # Rich context information
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize the issue with proper defaults and location detection."""
        # Ensure enums are properly set
        if isinstance(self.type, str):
            try:
                self.type = IssueType(self.type)
            except ValueError:
                pass  # Keep as string if not in enum
        
        if isinstance(self.category, str):
            try:
                self.category = IssueCategory(self.category)
            except ValueError:
                pass  # Keep as string if not in enum
                
        if isinstance(self.severity, str):
            try:
                self.severity = IssueSeverity(self.severity)
            except ValueError:
                pass  # Keep as string if not in enum
        
        # Auto-detect location if not provided
        if self.location is None and self.item is not None:
            self.location = self._detect_location()
        
        # Ensure location is CodeLocation object
        if isinstance(self.location, str):
            self.location = self._parse_location_string(self.location)
    
    def _detect_location(self) -> Optional[CodeLocation]:
        """Detect location from the item."""
        if not self.item:
            return None
            
        try:
            # Try to extract file path
            file_path = None
            if hasattr(self.item, 'file') and hasattr(self.item.file, 'path'):
                file_path = str(self.item.file.path)
            elif hasattr(self.item, 'filepath'):
                file_path = str(self.item.filepath)
            elif hasattr(self.item, 'path'):
                file_path = str(self.item.path)
            
            if not file_path:
                return None
            
            # Try to extract line information
            line_start = None
            line_end = None
            
            if hasattr(self.item, 'line_range'):
                line_start = self.item.line_range.start
                line_end = self.item.line_range.end
            elif hasattr(self.item, 'line'):
                line_start = self.item.line
            elif hasattr(self.item, 'start_line'):
                line_start = self.item.start_line
                if hasattr(self.item, 'end_line'):
                    line_end = self.item.end_line
            
            if line_start is not None:
                return CodeLocation(
                    file_path=file_path,
                    line_start=int(line_start),
                    line_end=int(line_end) if line_end else None
                )
            else:
                return CodeLocation(file_path=file_path, line_start=1)
                
        except Exception:
            return None
    
    def _parse_location_string(self, location_str: str) -> Optional[CodeLocation]:
        """Parse a location string into a CodeLocation object."""
        try:
            if ':' in location_str:
                parts = location_str.split(':')
                file_path = parts[0]
                if len(parts) > 1:
                    line_part = parts[1]
                    if '-' in line_part:
                        line_start, line_end = map(int, line_part.split('-'))
                        return CodeLocation(file_path=file_path, line_start=line_start, line_end=line_end)
                    else:
                        line_start = int(line_part)
                        return CodeLocation(file_path=file_path, line_start=line_start)
                else:
                    return CodeLocation(file_path=file_path, line_start=1)
            else:
                return CodeLocation(file_path=location_str, line_start=1)
        except Exception:
            return None
    
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
    
    def get_category_from_type(self) -> IssueCategory:
        """Automatically determine category from issue type."""
        type_to_category = {
            # Dead Code
            IssueType.UNUSED_FUNCTION: IssueCategory.DEAD_CODE,
            IssueType.UNUSED_CLASS: IssueCategory.DEAD_CODE,
            IssueType.UNUSED_IMPORT: IssueCategory.DEAD_CODE,
            IssueType.UNUSED_PARAMETER: IssueCategory.DEAD_CODE,
            IssueType.EMPTY_FUNCTION: IssueCategory.DEAD_CODE,
            
            # Implementation Errors
            IssueType.NULL_REFERENCE: IssueCategory.IMPLEMENTATION_ERROR,
            IssueType.UNREACHABLE_CODE: IssueCategory.IMPLEMENTATION_ERROR,
            IssueType.INFINITE_LOOP: IssueCategory.IMPLEMENTATION_ERROR,
            IssueType.OFF_BY_ONE_ERROR: IssueCategory.IMPLEMENTATION_ERROR,
            IssueType.MISSING_RETURN: IssueCategory.IMPLEMENTATION_ERROR,
            
            # Security
            IssueType.DANGEROUS_FUNCTION_USAGE: IssueCategory.SECURITY_VULNERABILITY,
            IssueType.POTENTIAL_INJECTION: IssueCategory.SECURITY_VULNERABILITY,
            IssueType.UNSAFE_EVAL: IssueCategory.SECURITY_VULNERABILITY,
            
            # Performance
            IssueType.HIGH_COMPLEXITY: IssueCategory.PERFORMANCE_ISSUE,
            IssueType.INEFFICIENT_LOOP: IssueCategory.PERFORMANCE_ISSUE,
            IssueType.MEMORY_LEAK: IssueCategory.PERFORMANCE_ISSUE,
            
            # Type Safety
            IssueType.MISSING_TYPE_ANNOTATION: IssueCategory.TYPE_SAFETY,
            IssueType.TYPE_MISMATCH: IssueCategory.TYPE_SAFETY,
            
            # Dependencies
            IssueType.CIRCULAR_DEPENDENCY: IssueCategory.DEPENDENCY_ISSUE,
            
            # Formatting
            IssueType.LINE_TOO_LONG: IssueCategory.FORMATTING_ISSUE,
            IssueType.INCONSISTENT_INDENTATION: IssueCategory.FORMATTING_ISSUE,
        }
        
        return type_to_category.get(self.type, IssueCategory.CODE_QUALITY)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the issue to a dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value if isinstance(self.type, IssueType) else self.type,
            "category": self.category.value if isinstance(self.category, IssueCategory) else self.category,
            "severity": self.severity.value if isinstance(self.severity, IssueSeverity) else self.severity,
            "message": self.message,
            "suggestion": self.suggestion,
            "location": str(self.location) if self.location else None,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "related_issues": self.related_issues,
            "tags": self.tags,
            "assigned_to": self.assigned_to,
            "context": self.context
        }
    
    def __str__(self) -> str:
        """Return a string representation of the issue."""
        severity_str = self.severity.value if isinstance(self.severity, IssueSeverity) else self.severity
        type_str = self.type.value if isinstance(self.type, IssueType) else self.type
        location_str = str(self.location) if self.location else "Unknown location"
        
        base = f"[{severity_str.upper()}] {type_str}: {self.message} ({location_str})"
        if self.suggestion:
            base += f" - Suggestion: {self.suggestion}"
        return base

@dataclass
class IssueCollection:
    """Collection of issues with management and analysis capabilities."""
    issues: List[Issue] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_issue(self, issue: Issue):
        """Add an issue to the collection."""
        self.issues.append(issue)
    
    def get_by_severity(self, severity: IssueSeverity) -> List[Issue]:
        """Get all issues with a specific severity."""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def get_by_category(self, category: IssueCategory) -> List[Issue]:
        """Get all issues in a specific category."""
        return [issue for issue in self.issues if issue.category == category]
    
    def get_by_type(self, issue_type: IssueType) -> List[Issue]:
        """Get all issues of a specific type."""
        return [issue for issue in self.issues if issue.type == issue_type]
    
    def get_by_status(self, status: IssueStatus) -> List[Issue]:
        """Get all issues with a specific status."""
        return [issue for issue in self.issues if issue.status == status]
    
    def get_by_file(self, file_path: str) -> List[Issue]:
        """Get all issues in a specific file."""
        return [issue for issue in self.issues 
                if issue.location and isinstance(issue.location, CodeLocation) 
                and issue.location.file_path == file_path]
    
    def get_critical_issues(self) -> List[Issue]:
        """Get all critical issues."""
        return self.get_by_severity(IssueSeverity.CRITICAL)
    
    def get_security_issues(self) -> List[Issue]:
        """Get all security-related issues."""
        return self.get_by_category(IssueCategory.SECURITY_VULNERABILITY)
    
    def get_performance_issues(self) -> List[Issue]:
        """Get all performance-related issues."""
        return self.get_by_category(IssueCategory.PERFORMANCE_ISSUE)
    
    def get_dead_code_issues(self) -> List[Issue]:
        """Get all dead code issues."""
        return self.get_by_category(IssueCategory.DEAD_CODE)
    
    def count_by_severity(self) -> Dict[str, int]:
        """Count issues by severity."""
        counts = {}
        for severity in IssueSeverity:
            counts[severity.value] = len(self.get_by_severity(severity))
        return counts
    
    def count_by_category(self) -> Dict[str, int]:
        """Count issues by category."""
        counts = {}
        for category in IssueCategory:
            counts[category.value] = len(self.get_by_category(category))
        return counts
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the issue collection."""
        return {
            "total_issues": len(self.issues),
            "by_severity": self.count_by_severity(),
            "by_category": self.count_by_category(),
            "critical_count": len(self.get_critical_issues()),
            "security_count": len(self.get_security_issues()),
            "performance_count": len(self.get_performance_issues()),
            "dead_code_count": len(self.get_dead_code_issues()),
            "created_at": self.created_at.isoformat()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the collection to a dictionary."""
        return {
            "issues": [issue.to_dict() for issue in self.issues],
            "summary": self.get_summary(),
            "created_at": self.created_at.isoformat()
        }
    
    def __len__(self) -> int:
        """Return the number of issues in the collection."""
        return len(self.issues)
    
    def __iter__(self):
        """Iterate over the issues."""
        return iter(self.issues)

@dataclass
class FunctionContext:
    """Comprehensive context information for a function."""
    name: str
    filepath: str
    source: str
    parameters: List[Dict[str, Any]]
    dependencies: List[Dict[str, Any]]
    usages: List[Dict[str, Any]]
    function_calls: List[str]
    called_by: List[str]
    max_call_chain: List[str]
    issues: List[Issue]
    is_entry_point: bool
    is_dead_code: bool
    class_name: Optional[str] = None
    complexity_score: int = 0
    halstead_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.halstead_metrics is None:
            self.halstead_metrics = {}

# ====================================================================
# CODEGEN SDK FUNCTIONS - Now imported directly from SDK modules
# ====================================================================
# Functions are imported from:
# - codegen.sdk.codebase.codebase_analysis: get_codebase_summary, get_file_summary, etc.
# - codegen.sdk.codebase.codebase_context: CodebaseContext, sdk_get_function_context
# - codegen.sdk.codebase.codebase_ai: generate_system_prompt, generate_tools, etc.

# ====================================================================
# CORE ERROR DETECTION PATTERNS
# ====================================================================

def detect_implementation_errors(codebase) -> List[Issue]:
    """
    Detect implementation errors in a codebase.
    
    Implementation errors are logical errors in the code that may cause incorrect behavior.
    
    Args:
        codebase: The codebase to analyze
        
    Returns:
        List of issues representing implementation errors
    """
    issues = []
    
    if not CODEGEN_SDK_AVAILABLE or not hasattr(codebase, 'functions'):
        return issues
    
    # Analyze each function for implementation errors
    for func in codebase.functions:
        try:
            # Check for unreachable code
            unreachable_issues = find_unreachable_code(func)
            issues.extend(unreachable_issues)
            
            # Check for infinite loops
            infinite_loop_issues = find_infinite_loops(func)
            issues.extend(infinite_loop_issues)
            
            # Check for off-by-one errors
            off_by_one_issues = find_off_by_one_errors(func)
            issues.extend(off_by_one_issues)
            
        except Exception as e:
            # Create an issue for analysis failure
            issues.append(Issue(
                type=IssueType.IMPLEMENTATION_ERROR,
                category=IssueCategory.IMPLEMENTATION_ERROR,
                severity=IssueSeverity.WARNING,
                message=f"Failed to analyze function '{func.name}': {str(e)}",
                item=func,
                suggestion="Check function syntax and structure"
            ))
    
    return issues

def find_unreachable_code(func) -> List[Issue]:
    """
    Find unreachable code in a function.
    
    Args:
        func: The function to analyze
        
    Returns:
        List of unreachable code issues
    """
    issues = []
    
    if not hasattr(func, 'source') or not func.source:
        return issues
    
    try:
        # Get function body as text
        body_text = func.source
        
        # Find return statements
        return_statements = re.finditer(r'^\s*return', body_text, re.MULTILINE)
        
        # Check for code after return statements
        for match in return_statements:
            # Get the line number
            lines_before = body_text[:match.start()].count('\n')
            
            # Check if there's code after this return statement
            remaining_text = body_text[match.end():].strip()
            if remaining_text:
                # Find the next non-empty, non-comment line
                lines = remaining_text.split('\n')
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('"""') and not line.startswith("'''"):
                        # Found unreachable code
                        line_number = lines_before + i + 2  # +2 for return line and next line
                        
                        location = CodeLocation(
                            file_path=str(func.filepath) if hasattr(func, 'filepath') else 'unknown',
                            line_start=line_number
                        )
                        
                        issues.append(Issue(
                            type=IssueType.UNREACHABLE_CODE,
                            category=IssueCategory.IMPLEMENTATION_ERROR,
                            severity=IssueSeverity.WARNING,
                            message=f"Unreachable code after return statement: '{line[:50]}...'",
                            location=location,
                            item=func,
                            suggestion="Remove unreachable code or restructure the function logic"
                        ))
                        break
    
    except Exception as e:
        # Return an issue for analysis failure
        issues.append(Issue(
            type=IssueType.IMPLEMENTATION_ERROR,
            category=IssueCategory.IMPLEMENTATION_ERROR,
            severity=IssueSeverity.INFO,
            message=f"Could not analyze unreachable code in function '{func.name}': {str(e)}",
            item=func
        ))
    
    return issues

def find_infinite_loops(func) -> List[Issue]:
    """
    Find potential infinite loops in a function.
    
    Args:
        func: The function to analyze
        
    Returns:
        List of potential infinite loop issues
    """
    issues = []
    
    if not hasattr(func, 'source') or not func.source:
        return issues
    
    try:
        # Get function body as text
        body_text = func.source
        
        # Check for while True loops without break
        while_true_pattern = r'while\s+True\s*:'
        while_matches = re.finditer(while_true_pattern, body_text, re.IGNORECASE)
        
        for match in while_matches:
            # Get the line number
            line_number = body_text[:match.start()].count('\n') + 1
            
            # Check if there's a break statement in the loop
            # This is a simplified check - could be enhanced with AST parsing
            remaining_text = body_text[match.end():]
            
            # Find the end of this while loop (simplified)
            lines = remaining_text.split('\n')
            loop_content = []
            indent_level = None
            
            for line in lines:
                if line.strip():
                    current_indent = len(line) - len(line.lstrip())
                    if indent_level is None:
                        indent_level = current_indent
                    elif current_indent <= indent_level and line.strip():
                        break  # End of loop
                    loop_content.append(line)
            
            loop_text = '\n'.join(loop_content)
            if 'break' not in loop_text and 'return' not in loop_text:
                location = CodeLocation(
                    file_path=str(func.filepath) if hasattr(func, 'filepath') else 'unknown',
                    line_start=line_number
                )
                
                issues.append(Issue(
                    type=IssueType.INFINITE_LOOP,
                    category=IssueCategory.IMPLEMENTATION_ERROR,
                    severity=IssueSeverity.ERROR,
                    message="Potential infinite loop: 'while True' without break or return statement",
                    location=location,
                    item=func,
                    suggestion="Add a break statement or return condition to prevent infinite loop"
                ))
    
    except Exception as e:
        # Return an issue for analysis failure
        issues.append(Issue(
            type=IssueType.IMPLEMENTATION_ERROR,
            category=IssueCategory.IMPLEMENTATION_ERROR,
            severity=IssueSeverity.INFO,
            message=f"Could not analyze infinite loops in function '{func.name}': {str(e)}",
            item=func
        ))
    
    return issues

def find_off_by_one_errors(func) -> List[Issue]:
    """
    Find potential off-by-one errors in a function.
    
    Args:
        func: The function to analyze
        
    Returns:
        List of potential off-by-one error issues
    """
    issues = []
    
    if not hasattr(func, 'source') or not func.source:
        return issues
    
    try:
        # Get function body as text
        body_text = func.source
        
        # Find array/list access with suspicious patterns
        # Pattern 1: array[len(array)] - should be array[len(array)-1]
        len_access_pattern = r'(\w+)\[len\(\1\)\]'
        len_matches = re.finditer(len_access_pattern, body_text)
        
        for match in len_matches:
            line_number = body_text[:match.start()].count('\n') + 1
            array_name = match.group(1)
            
            location = CodeLocation(
                file_path=str(func.filepath) if hasattr(func, 'filepath') else 'unknown',
                line_start=line_number
            )
            
            issues.append(Issue(
                type=IssueType.OFF_BY_ONE_ERROR,
                category=IssueCategory.IMPLEMENTATION_ERROR,
                severity=IssueSeverity.ERROR,
                message=f"Potential off-by-one error: '{array_name}[len({array_name})]' will cause IndexError",
                location=location,
                item=func,
                suggestion=f"Use '{array_name}[len({array_name})-1]' or '{array_name}[-1]' to access the last element"
            ))
        
        # Pattern 2: range(len(array)) in for loops where array is accessed
        range_len_pattern = r'for\s+(\w+)\s+in\s+range\(len\((\w+)\)\):'
        range_matches = re.finditer(range_len_pattern, body_text)
        
        for match in range_matches:
            line_number = body_text[:match.start()].count('\n') + 1
            index_var = match.group(1)
            array_name = match.group(2)
            
            # Check if the array is accessed with index+1 in the loop
            loop_start = match.end()
            remaining_text = body_text[loop_start:]
            
            # Simple check for array[index+1] pattern
            if f'{array_name}[{index_var}+1]' in remaining_text:
                location = CodeLocation(
                    file_path=str(func.filepath) if hasattr(func, 'filepath') else 'unknown',
                    line_start=line_number
                )
                
                issues.append(Issue(
                    type=IssueType.OFF_BY_ONE_ERROR,
                    category=IssueCategory.IMPLEMENTATION_ERROR,
                    severity=IssueSeverity.WARNING,
                    message=f"Potential off-by-one error: accessing '{array_name}[{index_var}+1]' in range(len({array_name})) loop",
                    location=location,
                    item=func,
                    suggestion=f"Consider using 'range(len({array_name})-1)' or check array bounds"
                ))
    
    except Exception as e:
        # Return an issue for analysis failure
        issues.append(Issue(
            type=IssueType.IMPLEMENTATION_ERROR,
            category=IssueCategory.IMPLEMENTATION_ERROR,
            severity=IssueSeverity.INFO,
            message=f"Could not analyze off-by-one errors in function '{func.name}': {str(e)}",
            item=func
        ))
    
    return issues

def detect_security_vulnerabilities(codebase) -> List[Issue]:
    """
    Detect security vulnerabilities in a codebase.
    
    Args:
        codebase: The codebase to analyze
        
    Returns:
        List of security vulnerability issues
    """
    issues = []
    
    if not CODEGEN_SDK_AVAILABLE or not hasattr(codebase, 'files'):
        return issues
    
    # Dangerous functions to check for
    dangerous_functions = {
        'eval(': 'Use of eval() can execute arbitrary code',
        'exec(': 'Use of exec() can execute arbitrary code',
        'input(': 'Use of input() in Python 2 can execute arbitrary code',
        '__import__(': 'Dynamic imports can be dangerous',
        'subprocess.call(': 'Subprocess calls can be dangerous without proper sanitization',
        'os.system(': 'OS system calls can be dangerous',
        'pickle.loads(': 'Pickle deserialization can execute arbitrary code',
    }
    
    # Analyze each file
    for file in codebase.files:
        if not hasattr(file, 'source') or not file.source:
            continue
            
        try:
            lines = file.source.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                line_lower = line.lower()
                
                # Check for dangerous function usage
                for dangerous_func, message in dangerous_functions.items():
                    if dangerous_func.lower() in line_lower:
                        location = CodeLocation(
                            file_path=str(file.path) if hasattr(file, 'path') else 'unknown',
                            line_start=line_num
                        )
                        
                        issues.append(Issue(
                            type=IssueType.DANGEROUS_FUNCTION_USAGE,
                            category=IssueCategory.SECURITY_VULNERABILITY,
                            severity=IssueSeverity.CRITICAL,
                            message=f"Security vulnerability: {message}",
                            location=location,
                            item=file,
                            suggestion="Review the usage and consider safer alternatives"
                        ))
                
                # Check for potential null references
                if 'None.' in line or '.None' in line:
                    location = CodeLocation(
                        file_path=str(file.path) if hasattr(file, 'path') else 'unknown',
                        line_start=line_num
                    )
                    
                    issues.append(Issue(
                        type=IssueType.NULL_REFERENCE,
                        category=IssueCategory.IMPLEMENTATION_ERROR,
                        severity=IssueSeverity.ERROR,
                        message="Potential null reference detected",
                        location=location,
                        item=file,
                        suggestion="Add null checks before accessing object methods"
                    ))
        
        except Exception as e:
            # Create an issue for analysis failure
            issues.append(Issue(
                type=IssueType.IMPLEMENTATION_ERROR,
                category=IssueCategory.IMPLEMENTATION_ERROR,
                severity=IssueSeverity.WARNING,
                message=f"Failed to analyze file '{file.path}': {str(e)}",
                item=file
            ))
    
    return issues

# ====================================================================
# UTILITY FUNCTIONS
# ====================================================================

def get_function_context(function) -> dict:
    """Get the implementation, dependencies, and usages of a function using Codegen SDK."""
    # Use the imported SDK function
    return sdk_get_function_context(function)

def get_max_call_chain(function):
    """Calculate maximum call chain for a function using NetworkX."""
    G = nx.DiGraph()
    
    def build_graph(func, depth=0):
        if depth > 10:  # Prevent infinite recursion
            return
        try:
            for call in func.function_calls:
                if hasattr(call, 'function_definition') and call.function_definition:
                    called_func = call.function_definition
                    G.add_edge(func.name, called_func.name)
                    build_graph(called_func, depth + 1)
        except:
            pass
    
    build_graph(function)
    try:
        return nx.dag_longest_path(G)
    except:
        return [function.name]

def analyze_codebase(codebase):
    """Main analysis function that returns comprehensive codebase analysis."""
    analyzer = ComprehensiveCodebaseAnalyzer(codebase)
    return analyzer.analyze()
# ====================================================================
# COMPREHENSIVE ANALYZER CLASS
# ====================================================================

class ComprehensiveCodebaseAnalyzer:
    """
    Comprehensive analyzer for codebases using the Codegen SDK.
    Consolidates functionality from comprehensive_analysis.py and SDK functions.
    """
    
    def __init__(self, codebase_or_path):
        """
        Initialize the analyzer with a codebase or path.
        
        Args:
            codebase_or_path: Either a Codebase object or path/URL to repository
        """
        self.issues = []
        self.function_contexts = {}
        self.dead_code_items = []
        self.call_graph = nx.DiGraph()
        self.operators = set()
        self.operands = set()
        self.start_time = time.time()
        
        # Initialize codebase
        if isinstance(codebase_or_path, str):
            self.repo_path_or_url = codebase_or_path
            self.codebase = self._initialize_codebase(codebase_or_path)
        else:
            self.codebase = codebase_or_path
            self.repo_path_or_url = getattr(codebase_or_path, 'repo_path', 'Unknown')
    
    def _initialize_codebase(self, repo_path_or_url):
        """Initialize codebase from path or URL."""
        if not CODEGEN_SDK_AVAILABLE:
            self.issues.append(Issue(
                repo_path_or_url,
                IssueType.IMPLEMENTATION_ERROR.value,
                "Codegen SDK not available - falling back to basic analysis",
                IssueSeverity.ERROR.value,
                suggestion="Install the Codegen SDK for full functionality"
            ))
            return None
        
        try:
            print(f"Initializing codebase from {repo_path_or_url}")
            if repo_path_or_url.startswith(("http://", "https://")):
                # Extract repo name for GitHub URLs
                parts = repo_path_or_url.rstrip('/').split('/')
                repo_name = f"{parts[-2]}/{parts[-1]}"
                try:
                    codebase = Codebase.from_repo(repo_full_name=repo_name)
                    print(f"Successfully initialized codebase from GitHub repository: {repo_name}")
                    return codebase
                except Exception as e:
                    print(f"Error initializing codebase from GitHub: {e}")
                    self.issues.append(Issue(
                        repo_path_or_url,
                        IssueType.IMPLEMENTATION_ERROR.value,
                        f"Failed to initialize codebase from GitHub: {e}",
                        IssueSeverity.ERROR.value,
                        suggestion="Check your network connection and GitHub access permissions."
                    ))
                    return None
            else:
                # Local path
                try:
                    codebase = Codebase(repo_path_or_url)
                    print(f"Successfully initialized codebase from local path: {repo_path_or_url}")
                    return codebase
                except Exception as e:
                    print(f"Error initializing codebase from local path: {e}")
                    self.issues.append(Issue(
                        repo_path_or_url,
                        IssueType.IMPLEMENTATION_ERROR.value,
                        f"Failed to initialize codebase from local path: {e}",
                        IssueSeverity.ERROR.value,
                        suggestion="Ensure the path exists and contains valid source code."
                    ))
                    return None
        except Exception as e:
            self.issues.append(Issue(
                repo_path_or_url,
                IssueType.IMPLEMENTATION_ERROR.value,
                f"Unexpected error during codebase initialization: {e}",
                IssueSeverity.ERROR.value
            ))
            return None
    
    def analyze(self):
        """
        Perform comprehensive analysis of the codebase.
        
        Returns:
            Dictionary with analysis results
        """
        print(f"🔍 Starting comprehensive analysis of {self.repo_path_or_url}...")
        
        if self.codebase is None:
            return {
                "error": "Codebase initialization failed",
                "success": False,
                "issues": [str(issue) for issue in self.issues]
            }
        
        # Check if codebase was initialized correctly
        if not hasattr(self.codebase, 'files') or not self.codebase.files:
            self.issues.append(Issue(
                self.repo_path_or_url,
                IssueType.IMPLEMENTATION_ERROR.value,
                "Codebase was initialized but contains no files",
                IssueSeverity.ERROR.value,
                suggestion="Check if the repository contains supported language files."
            ))
            print("Warning: Codebase contains no files")
        
        # Run all analyses with error handling
        analysis_steps = [
            ("Building call graph", self._build_call_graph),
            ("Analyzing functions", self._analyze_functions),
            ("Detecting dead code", self._analyze_dead_code),
            ("Analyzing parameter issues", self._analyze_parameter_issues),
            ("Analyzing type annotations", self._analyze_type_annotations),
            ("Detecting circular dependencies", self._analyze_circular_dependencies),
            ("Analyzing implementation issues", self._analyze_implementation_issues),
            # New comprehensive error detection patterns
            ("Detecting implementation errors", self._detect_implementation_errors),
            ("Detecting security vulnerabilities", self._detect_security_vulnerabilities),
            ("Analyzing code quality", self._analyze_code_quality),
        ]
        
        for step_name, step_func in analysis_steps:
            try:
                print(f"  📊 {step_name}...")
                step_func()
            except Exception as e:
                print(f"  ❌ Error in {step_name.lower()}: {e}")
                self.issues.append(Issue(
                    self.repo_path_or_url,
                    IssueType.IMPLEMENTATION_ERROR.value,
                    f"{step_name} failed: {e}",
                    IssueSeverity.ERROR.value
                ))
        
        # Generate final report
        return self._generate_comprehensive_report()
    
    def _build_call_graph(self):
        """Build the function call graph using SDK's pre-computed relationships."""
        if not CODEGEN_SDK_AVAILABLE or not self.codebase:
            return
        
        try:
            for function in self.codebase.functions:
                self.call_graph.add_node(function.name)
                for call in function.function_calls:
                    if hasattr(call, 'function_definition') and call.function_definition:
                        self.call_graph.add_edge(function.name, call.function_definition.name)
        except Exception as e:
            print(f"Error building call graph: {e}")
    
    def _analyze_functions(self):
        """Analyze all functions and build contexts using SDK."""
        if not CODEGEN_SDK_AVAILABLE or not self.codebase:
            return
        
        try:
            for function in self.codebase.functions:
                context = self._get_function_context(function)
                self.function_contexts[function.name] = context
        except Exception as e:
            print(f"Error analyzing functions: {e}")
    
    def _get_function_context(self, function):
        """Get comprehensive context for a function using SDK."""
        try:
            # Get dependencies using SDK's pre-computed relationships
            dependencies = []
            for dep in function.dependencies:
                if hasattr(dep, 'source') and hasattr(dep, 'filepath'):
                    dependencies.append({"source": dep.source, "filepath": dep.filepath})
            
            # Get usages using SDK's pre-computed relationships
            usages = []
            for usage in function.usages:
                if hasattr(usage, 'usage_symbol'):
                    usages.append({
                        "source": usage.usage_symbol.source,
                        "filepath": usage.usage_symbol.filepath,
                    })
            
            # Get function calls
            function_calls = [call.name for call in function.function_calls if hasattr(call, 'name')]
            
            # Get called by relationships
            called_by = [call.parent_function.name for call in function.call_sites 
                        if hasattr(call, 'parent_function') and call.parent_function]
            
            # Calculate max call chain
            max_call_chain = get_max_call_chain(function)
            
            # Check if it's an entry point (no callers)
            is_entry_point = len(called_by) == 0
            
            # Check if it's dead code (no usages)
            is_dead_code = len(usages) == 0 and not is_entry_point
            
            return FunctionContext(
                name=function.name,
                filepath=function.filepath,
                source=function.source,
                parameters=[{"name": param.name, "type": getattr(param, 'type', None)} 
                           for param in function.parameters],
                dependencies=dependencies,
                usages=usages,
                function_calls=function_calls,
                called_by=called_by,
                max_call_chain=max_call_chain,
                issues=[],
                is_entry_point=is_entry_point,
                is_dead_code=is_dead_code,
                class_name=getattr(function, 'class_name', None),
                complexity_score=self._calculate_complexity(function),
                halstead_metrics=self._calculate_halstead_metrics_for_function(function)
            )
        except Exception as e:
            print(f"Error getting function context for {function.name}: {e}")
            return FunctionContext(
                name=function.name,
                filepath=getattr(function, 'filepath', 'Unknown'),
                source=getattr(function, 'source', ''),
                parameters=[],
                dependencies=[],
                usages=[],
                function_calls=[],
                called_by=[],
                max_call_chain=[function.name],
                issues=[],
                is_entry_point=False,
                is_dead_code=False
            )
    
    def _analyze_dead_code(self):
        """Analyze dead code using SDK's usage information with enhanced context."""
        if not CODEGEN_SDK_AVAILABLE or not self.codebase:
            return
        
        try:
            # Analyze unused functions
            for function in self.codebase.functions:
                if len(function.call_sites) == 0 and not self._is_entry_point(function):
                    issue = Issue(
                        type=IssueType.UNUSED_FUNCTION,
                        category=IssueCategory.DEAD_CODE,
                        severity=IssueSeverity.WARNING,
                        message=f"Function '{function.name}' is never called",
                        item=function,
                        suggestion="Consider removing this function if it's truly unused"
                    )
                    
                    # Add rich context
                    issue.add_context("function_name", function.name)
                    issue.add_context("call_sites_count", len(function.call_sites))
                    if hasattr(function, 'parameters'):
                        issue.add_context("parameter_count", len(function.parameters))
                    if hasattr(function, 'source'):
                        issue.add_context("line_count", len(function.source.split('\n')))
                        issue.add_context("complexity_estimate", self._estimate_complexity(function.source))
                    
                    self.issues.append(issue)
                    self.dead_code_items.append(function)
            
            # Analyze unused classes
            for cls in self.codebase.classes:
                if len(cls.usages) == 0:
                    issue = Issue(
                        type=IssueType.UNUSED_CLASS,
                        category=IssueCategory.DEAD_CODE,
                        severity=IssueSeverity.WARNING,
                        message=f"Class '{cls.name}' is never used",
                        item=cls,
                        suggestion="Consider removing this class if it's truly unused"
                    )
                    
                    # Add rich context
                    issue.add_context("class_name", cls.name)
                    issue.add_context("usage_count", len(cls.usages))
                    if hasattr(cls, 'methods'):
                        issue.add_context("method_count", len(cls.methods))
                    if hasattr(cls, 'attributes'):
                        issue.add_context("attribute_count", len(cls.attributes))
                    
                    self.issues.append(issue)
                    self.dead_code_items.append(cls)
            
            # Analyze unused imports
            for import_stmt in self.codebase.imports:
                if hasattr(import_stmt, 'imported_symbol') and import_stmt.imported_symbol:
                    if len(import_stmt.imported_symbol.usages) == 0:
                        issue = Issue(
                            type=IssueType.UNUSED_IMPORT,
                            category=IssueCategory.DEAD_CODE,
                            severity=IssueSeverity.INFO,
                            message=f"Import '{import_stmt.name}' is never used",
                            item=import_stmt,
                            suggestion="Remove this unused import"
                        )
                        
                        # Add rich context
                        issue.add_context("import_name", import_stmt.name)
                        issue.add_context("usage_count", len(import_stmt.imported_symbol.usages))
                        
                        self.issues.append(issue)
                        self.dead_code_items.append(import_stmt)
        
        except Exception as e:
            print(f"Error in dead code analysis: {e}")
    
    def _estimate_complexity(self, source_code: str) -> str:
        """Estimate the complexity of source code."""
        if not source_code:
            return "low"
        
        lines = len(source_code.split('\n'))
        if lines > 50:
            return "high"
        elif lines > 20:
            return "medium"
        else:
            return "low"
    
    def _analyze_parameter_issues(self):
        """Analyze parameter-related issues using SDK."""
        if not CODEGEN_SDK_AVAILABLE or not self.codebase:
            return
        
        try:
            for function in self.codebase.functions:
                # Check for unused parameters
                for param in function.parameters:
                    if hasattr(param, 'usages') and len(param.usages) == 0:
                        self.issues.append(Issue(
                            param,
                            IssueType.UNUSED_PARAMETER.value,
                            f"Parameter '{param.name}' in function '{function.name}' is never used",
                            IssueSeverity.WARNING.value,
                            suggestion="Remove unused parameter or use it in the function body"
                        ))
                
                # Check for parameter count mismatches in function calls
                for call in function.function_calls:
                    if hasattr(call, 'function_definition') and call.function_definition:
                        expected_params = len(call.function_definition.parameters)
                        actual_args = len(getattr(call, 'arguments', []))
                        if expected_params != actual_args:
                            self.issues.append(Issue(
                                call,
                                IssueType.PARAMETER_MISMATCH.value,
                                f"Function call to '{call.name}' has {actual_args} arguments but expects {expected_params}",
                                IssueSeverity.ERROR.value,
                                suggestion="Check the function signature and provide the correct number of arguments"
                            ))
        
        except Exception as e:
            print(f"Error in parameter analysis: {e}")
    
    def _analyze_type_annotations(self):
        """Analyze type annotation issues."""
        if not CODEGEN_SDK_AVAILABLE or not self.codebase:
            return
        
        try:
            for function in self.codebase.functions:
                # Check for missing return type annotations
                if not hasattr(function, 'return_type') or not function.return_type:
                    self.issues.append(Issue(
                        function,
                        IssueType.MISSING_TYPE_ANNOTATION.value,
                        f"Function '{function.name}' is missing return type annotation",
                        IssueSeverity.INFO.value,
                        suggestion="Add return type annotation for better code documentation"
                    ))
                
                # Check for missing parameter type annotations
                for param in function.parameters:
                    if not hasattr(param, 'type') or not param.type:
                        self.issues.append(Issue(
                            param,
                            IssueType.MISSING_TYPE_ANNOTATION.value,
                            f"Parameter '{param.name}' in function '{function.name}' is missing type annotation",
                            IssueSeverity.INFO.value,
                            suggestion="Add type annotation for better code documentation"
                        ))
        
        except Exception as e:
            print(f"Error in type annotation analysis: {e}")
    
    def _analyze_circular_dependencies(self):
        """Analyze circular dependencies using the call graph."""
        try:
            # Find strongly connected components (cycles) in the call graph
            cycles = list(nx.strongly_connected_components(self.call_graph))
            
            for cycle in cycles:
                if len(cycle) > 1:  # Only report actual cycles, not self-loops
                    cycle_list = list(cycle)
                    self.issues.append(Issue(
                        cycle_list[0],  # Use first function as the item
                        IssueType.CIRCULAR_DEPENDENCY.value,
                        f"Circular dependency detected between functions: {', '.join(cycle_list)}",
                        IssueSeverity.WARNING.value,
                        suggestion="Refactor to break the circular dependency"
                    ))
        
        except Exception as e:
            print(f"Error in circular dependency analysis: {e}")
    
    def _analyze_implementation_issues(self):
        """Analyze various implementation issues."""
        if not CODEGEN_SDK_AVAILABLE or not self.codebase:
            return
        
        try:
            for function in self.codebase.functions:
                # Check for empty functions
                if not function.source.strip() or function.source.strip() == "pass":
                    self.issues.append(Issue(
                        function,
                        IssueType.EMPTY_FUNCTION.value,
                        f"Function '{function.name}' is empty or only contains 'pass'",
                        IssueSeverity.INFO.value,
                        suggestion="Implement the function or remove if not needed"
                    ))
                
                # Check for functions with no return statements but should return something
                if len(function.return_statements) == 0 and function.name not in ['__init__', 'setUp', 'tearDown']:
                    # Simple heuristic: if function name suggests it should return something
                    if any(keyword in function.name.lower() for keyword in ['get', 'find', 'calculate', 'compute', 'generate']):
                        self.issues.append(Issue(
                            function,
                            IssueType.MISSING_RETURN.value,
                            f"Function '{function.name}' appears to be missing a return statement",
                            IssueSeverity.WARNING.value,
                            suggestion="Add appropriate return statement or rename function to indicate it doesn't return a value"
                        ))
        
        except Exception as e:
            print(f"Error in implementation analysis: {e}")
    
    def _is_entry_point(self, function):
        """Check if a function is an entry point (main, test, etc.)."""
        entry_point_patterns = ['main', 'test_', '__main__', 'setUp', 'tearDown']
        return any(pattern in function.name for pattern in entry_point_patterns)
    
    def _calculate_complexity(self, function):
        """Calculate cyclomatic complexity (simplified)."""
        try:
            # Simple complexity calculation based on control flow keywords
            complexity_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally', 'with']
            complexity = 1  # Base complexity
            
            for keyword in complexity_keywords:
                complexity += function.source.count(keyword)
            
            return complexity
        except:
            return 1
    
    def _calculate_halstead_metrics_for_function(self, function):
        """Calculate Halstead metrics for a function."""
        try:
            # Simple implementation - count operators and operands
            operators = set()
            operands = set()
            
            # This is a simplified version - in practice, you'd use AST parsing
            operator_chars = ['+', '-', '*', '/', '=', '==', '!=', '<', '>', '<=', '>=', 'and', 'or', 'not']
            
            for op in operator_chars:
                if op in function.source:
                    operators.add(op)
            
            # Count unique identifiers as operands (simplified)
            words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', function.source)
            operands.update(words)
            
            n1 = len(operators)  # Number of distinct operators
            n2 = len(operands)   # Number of distinct operands
            N1 = sum(function.source.count(op) for op in operators)  # Total operators
            N2 = len(words)      # Total operands
            
            vocabulary = n1 + n2
            length = N1 + N2
            volume = length * (vocabulary.bit_length() if vocabulary > 0 else 0)
            
            return {
                "vocabulary": vocabulary,
                "length": length,
                "volume": volume,
                "distinct_operators": n1,
                "distinct_operands": n2,
                "total_operators": N1,
                "total_operands": N2
            }
        except:
            return {}
    
    def _detect_implementation_errors(self):
        """Wrapper method for implementation error detection."""
        try:
            implementation_issues = detect_implementation_errors(self.codebase)
            self.issues.extend(implementation_issues)
            print(f"Found {len(implementation_issues)} implementation errors")
        except Exception as e:
            print(f"Error in implementation error detection: {e}")
    
    def _detect_security_vulnerabilities(self):
        """Wrapper method for security vulnerability detection."""
        try:
            security_issues = detect_security_vulnerabilities(self.codebase)
            self.issues.extend(security_issues)
            print(f"Found {len(security_issues)} security vulnerabilities")
        except Exception as e:
            print(f"Error in security vulnerability detection: {e}")
    
    def _analyze_code_quality(self):
        """Analyze code quality issues."""
        try:
            quality_issues = []
            
            # Check for long lines
            for file in self.codebase.files:
                if not hasattr(file, 'source') or not file.source:
                    continue
                    
                lines = file.source.split('\n')
                for line_num, line in enumerate(lines, 1):
                    if len(line) > 120:  # Standard line length limit
                        location = CodeLocation(
                            file_path=str(file.path) if hasattr(file, 'path') else 'unknown',
                            line_start=line_num
                        )
                        
                        issue = Issue(
                            type=IssueType.LINE_TOO_LONG,
                            category=IssueCategory.FORMATTING_ISSUE,
                            severity=IssueSeverity.INFO,
                            message=f"Line too long ({len(line)} characters, limit is 120)",
                            location=location,
                            item=file,
                            suggestion="Break long lines into multiple lines for better readability"
                        )
                        
                        issue.add_context("line_length", len(line))
                        issue.add_context("line_content", line[:100] + "..." if len(line) > 100 else line)
                        quality_issues.append(issue)
            
            # Check for TODO/FIXME/HACK comments
            for file in self.codebase.files:
                if not hasattr(file, 'source') or not file.source:
                    continue
                    
                lines = file.source.split('\n')
                for line_num, line in enumerate(lines, 1):
                    line_lower = line.lower()
                    if any(keyword in line_lower for keyword in ['todo', 'fixme', 'hack']):
                        location = CodeLocation(
                            file_path=str(file.path) if hasattr(file, 'path') else 'unknown',
                            line_start=line_num
                        )
                        
                        issue = Issue(
                            type=IssueType.INCOMPLETE_IMPLEMENTATION,
                            category=IssueCategory.CODE_QUALITY,
                            severity=IssueSeverity.INFO,
                            message=f"Incomplete implementation marker found: {line.strip()}",
                            location=location,
                            item=file,
                            suggestion="Complete the implementation or remove the marker"
                        )
                        
                        issue.add_context("marker_type", "TODO/FIXME/HACK")
                        issue.add_context("line_content", line.strip())
                        quality_issues.append(issue)
            
            self.issues.extend(quality_issues)
            print(f"Found {len(quality_issues)} code quality issues")
            
        except Exception as e:
            print(f"Error in code quality analysis: {e}")
    
    def _generate_comprehensive_report(self):
        """Generate the final comprehensive analysis report."""
        duration = time.time() - self.start_time
        
        # Group issues by severity
        issues_by_severity = defaultdict(list)
        for issue in self.issues:
            issues_by_severity[issue.severity].append(issue)
        
        # Count issues by type
        issue_counts = Counter(issue.type for issue in self.issues)
        
        # Find most important functions
        important_functions = self._find_most_important_functions()
        
        # Generate statistics
        stats = self._generate_statistics()
        
        # Generate summaries using SDK functions
        summaries = self._generate_summaries()
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
            "repository": self.repo_path_or_url,
            "summary": {
                "total_issues": len(self.issues),
                "critical_issues": len(issues_by_severity[IssueSeverity.CRITICAL.value]),
                "error_issues": len(issues_by_severity[IssueSeverity.ERROR.value]),
                "warning_issues": len(issues_by_severity[IssueSeverity.WARNING.value]),
                "info_issues": len(issues_by_severity[IssueSeverity.INFO.value]),
                "dead_code_items": len(self.dead_code_items)
            },
            "statistics": stats,
            "issues": {
                "by_severity": {k: [str(issue) for issue in v] for k, v in issues_by_severity.items()},
                "by_type": dict(issue_counts),
                "all": [str(issue) for issue in self.issues]
            },
            "function_contexts": {name: asdict(ctx) for name, ctx in self.function_contexts.items()},
            "dead_code_analysis": {
                "items": [str(item) for item in self.dead_code_items],
                "count": len(self.dead_code_items)
            },
            "most_important_functions": important_functions,
            "call_graph_metrics": self._calculate_call_graph_metrics(),
            "summaries": summaries
        }
    
    def _find_most_important_functions(self):
        """Find most important functions using SDK data."""
        if not CODEGEN_SDK_AVAILABLE or not self.codebase or not list(self.codebase.functions):
            return {}
        
        try:
            functions = list(self.codebase.functions)
            
            # Find function that makes the most calls
            most_calls = max(functions, key=lambda f: len(f.function_calls))
            
            # Find the most called function
            most_called = max(functions, key=lambda f: len(f.call_sites))
            
            # Find class with most inheritance
            deepest_class = None
            if self.codebase.classes:
                classes = list(self.codebase.classes)
                deepest_class = max(classes, key=lambda x: len(getattr(x, 'superclasses', [])))
            
            return {
                "most_calls": {
                    "name": most_calls.name,
                    "call_count": len(most_calls.function_calls),
                    "calls": [call.name for call in most_calls.function_calls if hasattr(call, 'name')]
                },
                "most_called": {
                    "name": most_called.name,
                    "usage_count": len(most_called.call_sites),
                    "called_from": [call.parent_function.name for call in most_called.call_sites 
                                  if hasattr(call, 'parent_function') and call.parent_function]
                },
                "deepest_inheritance": {
                    "name": deepest_class.name if deepest_class else None,
                    "chain_depth": len(getattr(deepest_class, 'superclasses', [])) if deepest_class else 0,
                    "chain": [s.name for s in getattr(deepest_class, 'superclasses', [])] if deepest_class else []
                }
            }
        except Exception as e:
            print(f"Error finding important functions: {e}")
            return {}
    
    def _generate_statistics(self):
        """Generate codebase statistics."""
        if not CODEGEN_SDK_AVAILABLE or not self.codebase:
            return {}
        
        try:
            return {
                "total_files": len(list(self.codebase.files)),
                "total_functions": len(list(self.codebase.functions)),
                "total_classes": len(list(self.codebase.classes)),
                "total_imports": len(list(self.codebase.imports)),
                "total_symbols": len(list(self.codebase.symbols)),
                "total_issues": len(self.issues)
            }
        except Exception as e:
            print(f"Error generating statistics: {e}")
            return {}
    
    def _generate_summaries(self):
        """Generate summaries using SDK functions."""
        if not CODEGEN_SDK_AVAILABLE or not self.codebase:
            return {}
        
        try:
            return {
                "codebase_summary": get_codebase_summary(self.codebase),
                "file_count": len(list(self.codebase.files)),
                "function_count": len(list(self.codebase.functions)),
                "class_count": len(list(self.codebase.classes))
            }
        except Exception as e:
            print(f"Error generating summaries: {e}")
            return {}
    
    def _calculate_call_graph_metrics(self):
        """Calculate call graph metrics."""
        try:
            if not self.call_graph.nodes():
                return {}
            
            return {
                "total_nodes": len(self.call_graph.nodes()),
                "total_edges": len(self.call_graph.edges()),
                "strongly_connected_components": len(list(nx.strongly_connected_components(self.call_graph))),
                "is_dag": nx.is_directed_acyclic_graph(self.call_graph),
                "density": nx.density(self.call_graph)
            }
        except Exception as e:
            print(f"Error calculating call graph metrics: {e}")
            return {}
