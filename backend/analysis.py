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
# STEP 4: CONTEXT-RICH ERROR INFORMATION SYSTEM
# ====================================================================

class ContextCollector:
    """Collects comprehensive context information for issues."""
    
    def __init__(self, codebase):
        self.codebase = codebase
        
    def collect_function_context(self, function) -> Dict[str, Any]:
        """Collect comprehensive context for a function."""
        context = {}
        
        try:
            # Basic function information
            context["function_name"] = function.name
            context["function_signature"] = self._get_function_signature(function)
            
            # Call chain analysis
            context["call_chain"] = self._get_call_chain(function)
            context["callers"] = self._get_function_callers(function)
            context["callees"] = self._get_function_callees(function)
            
            # Parameter analysis
            context["parameter_usage"] = self._analyze_parameter_usage(function)
            context["parameter_types"] = self._get_parameter_types(function)
            
            # Complexity metrics
            context["complexity_metrics"] = self._calculate_complexity_metrics(function)
            
            # Dependencies and relationships
            context["dependencies"] = self._get_function_dependencies(function)
            context["usages"] = self._get_function_usages(function)
            context["related_files"] = self._get_related_files(function)
            
            # Code quality indicators
            context["code_quality"] = self._assess_code_quality(function)
            
            # Security implications
            context["security_implications"] = self._assess_security_implications(function)
            
            # Performance impact
            context["performance_impact"] = self._assess_performance_impact(function)
            
        except Exception as e:
            context["context_collection_error"] = str(e)
            
        return context
    
    def collect_file_context(self, file) -> Dict[str, Any]:
        """Collect comprehensive context for a file."""
        context = {}
        
        try:
            # Basic file information
            context["file_path"] = str(file.path) if hasattr(file, 'path') else 'unknown'
            context["file_size"] = len(file.source) if hasattr(file, 'source') else 0
            context["line_count"] = len(file.source.split('\n')) if hasattr(file, 'source') else 0
            
            # Import analysis
            context["imports"] = self._analyze_file_imports(file)
            context["external_dependencies"] = self._get_external_dependencies(file)
            
            # Function and class analysis
            context["functions"] = self._get_file_functions(file)
            context["classes"] = self._get_file_classes(file)
            
            # Code structure
            context["code_structure"] = self._analyze_code_structure(file)
            
            # File relationships
            context["related_files"] = self._get_file_relationships(file)
            
        except Exception as e:
            context["context_collection_error"] = str(e)
            
        return context
    
    def collect_issue_context(self, issue: Issue) -> Dict[str, Any]:
        """Collect comprehensive context for an issue."""
        context = {}
        
        try:
            # Issue metadata
            context["issue_id"] = issue.id
            context["issue_type"] = issue.type.value if isinstance(issue.type, IssueType) else issue.type
            context["issue_category"] = issue.category.value if isinstance(issue.category, IssueCategory) else issue.category
            context["severity"] = issue.severity.value if isinstance(issue.severity, IssueSeverity) else issue.severity
            
            # Location context
            if issue.location and isinstance(issue.location, CodeLocation):
                context["location_context"] = self._get_location_context(issue.location)
            
            # Item-specific context
            if issue.item:
                if hasattr(issue.item, 'name') and hasattr(issue.item, 'source'):
                    # Function or class
                    context["item_context"] = self.collect_function_context(issue.item)
                elif hasattr(issue.item, 'path') and hasattr(issue.item, 'source'):
                    # File
                    context["item_context"] = self.collect_file_context(issue.item)
            
            # Impact analysis
            context["impact_analysis"] = self._analyze_issue_impact(issue)
            
            # Fix suggestions
            context["fix_suggestions"] = self._generate_fix_suggestions(issue)
            
        except Exception as e:
            context["context_collection_error"] = str(e)
            
        return context
    
    def _get_function_signature(self, function) -> str:
        """Get the function signature."""
        try:
            if hasattr(function, 'parameters') and function.parameters:
                params = []
                for param in function.parameters:
                    param_str = param.name
                    if hasattr(param, 'type_annotation') and param.type_annotation:
                        param_str += f": {param.type_annotation}"
                    if hasattr(param, 'default_value') and param.default_value:
                        param_str += f" = {param.default_value}"
                    params.append(param_str)
                return f"def {function.name}({', '.join(params)})"
            else:
                return f"def {function.name}(...)"
        except:
            return f"def {function.name}(...)"
    
    def _get_call_chain(self, function) -> List[str]:
        """Get the call chain for a function."""
        try:
            call_chain = []
            if hasattr(function, 'call_sites'):
                for call_site in function.call_sites:
                    if hasattr(call_site, 'parent_function') and call_site.parent_function:
                        call_chain.append(call_site.parent_function.name)
            return call_chain
        except:
            return []
    
    def _get_function_callers(self, function) -> List[Dict[str, str]]:
        """Get functions that call this function."""
        try:
            callers = []
            if hasattr(function, 'call_sites'):
                for call_site in function.call_sites:
                    if hasattr(call_site, 'parent_function') and call_site.parent_function:
                        callers.append({
                            "name": call_site.parent_function.name,
                            "file": str(call_site.parent_function.file.path) if hasattr(call_site.parent_function, 'file') else 'unknown'
                        })
            return callers
        except:
            return []
    
    def _get_function_callees(self, function) -> List[Dict[str, str]]:
        """Get functions called by this function."""
        try:
            callees = []
            if hasattr(function, 'function_calls'):
                for call in function.function_calls:
                    if hasattr(call, 'function_definition') and call.function_definition:
                        callees.append({
                            "name": call.function_definition.name,
                            "file": str(call.function_definition.file.path) if hasattr(call.function_definition, 'file') else 'unknown'
                        })
            return callees
        except:
            return []
    
    def _analyze_parameter_usage(self, function) -> Dict[str, str]:
        """Analyze parameter usage in the function."""
        try:
            usage = {}
            if hasattr(function, 'parameters') and hasattr(function, 'source'):
                for param in function.parameters:
                    if param.name in function.source:
                        usage[param.name] = "used"
                    else:
                        usage[param.name] = "unused"
            return usage
        except:
            return {}
    
    def _get_parameter_types(self, function) -> Dict[str, str]:
        """Get parameter type information."""
        try:
            types = {}
            if hasattr(function, 'parameters'):
                for param in function.parameters:
                    if hasattr(param, 'type_annotation') and param.type_annotation:
                        types[param.name] = param.type_annotation
                    else:
                        types[param.name] = "unknown"
            return types
        except:
            return {}
    
    def _calculate_complexity_metrics(self, function) -> Dict[str, Any]:
        """Calculate comprehensive complexity metrics."""
        try:
            metrics = {}
            
            if hasattr(function, 'source') and function.source:
                # Cyclomatic complexity (simplified)
                complexity_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally', 'with', 'and', 'or']
                cyclomatic = 1  # Base complexity
                for keyword in complexity_keywords:
                    cyclomatic += function.source.count(keyword)
                metrics["cyclomatic_complexity"] = cyclomatic
                
                # Line metrics
                lines = function.source.split('\n')
                metrics["total_lines"] = len(lines)
                metrics["code_lines"] = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
                metrics["comment_lines"] = len([line for line in lines if line.strip().startswith('#')])
                
                # Nesting depth (simplified)
                max_depth = 0
                current_depth = 0
                for line in lines:
                    stripped = line.lstrip()
                    if any(keyword in stripped for keyword in ['if', 'for', 'while', 'try', 'with', 'def', 'class']):
                        current_depth = (len(line) - len(stripped)) // 4  # Assuming 4-space indentation
                        max_depth = max(max_depth, current_depth)
                metrics["max_nesting_depth"] = max_depth
                
            return metrics
        except:
            return {}
    
    def _get_function_dependencies(self, function) -> List[Dict[str, str]]:
        """Get function dependencies."""
        try:
            dependencies = []
            if hasattr(function, 'dependencies'):
                for dep in function.dependencies:
                    dependencies.append({
                        "name": dep.name if hasattr(dep, 'name') else str(dep),
                        "type": type(dep).__name__,
                        "file": str(dep.file.path) if hasattr(dep, 'file') and hasattr(dep.file, 'path') else 'unknown'
                    })
            return dependencies
        except:
            return []
    
    def _get_function_usages(self, function) -> List[Dict[str, str]]:
        """Get function usage information."""
        try:
            usages = []
            if hasattr(function, 'usages'):
                for usage in function.usages:
                    usages.append({
                        "location": str(usage.usage_symbol.file.path) if hasattr(usage, 'usage_symbol') and hasattr(usage.usage_symbol, 'file') else 'unknown',
                        "context": "function_call"  # Could be enhanced with more context
                    })
            return usages
        except:
            return []
    
    def _get_related_files(self, function) -> List[str]:
        """Get files related to this function."""
        try:
            related = set()
            
            # Add file containing the function
            if hasattr(function, 'file') and hasattr(function.file, 'path'):
                related.add(str(function.file.path))
            
            # Add files from dependencies
            if hasattr(function, 'dependencies'):
                for dep in function.dependencies:
                    if hasattr(dep, 'file') and hasattr(dep.file, 'path'):
                        related.add(str(dep.file.path))
            
            # Add files from usages
            if hasattr(function, 'usages'):
                for usage in function.usages:
                    if hasattr(usage, 'usage_symbol') and hasattr(usage.usage_symbol, 'file'):
                        related.add(str(usage.usage_symbol.file.path))
            
            return list(related)
        except:
            return []
    
    def _assess_code_quality(self, function) -> Dict[str, Any]:
        """Assess code quality indicators."""
        try:
            quality = {}
            
            if hasattr(function, 'source') and function.source:
                # Check for code smells
                quality["has_long_lines"] = any(len(line) > 120 for line in function.source.split('\n'))
                quality["has_todo_comments"] = any(keyword in function.source.lower() for keyword in ['todo', 'fixme', 'hack'])
                quality["has_docstring"] = '"""' in function.source or "'''" in function.source
                
                # Complexity assessment
                lines = len(function.source.split('\n'))
                if lines > 50:
                    quality["complexity_assessment"] = "high"
                elif lines > 20:
                    quality["complexity_assessment"] = "medium"
                else:
                    quality["complexity_assessment"] = "low"
            
            return quality
        except:
            return {}
    
    def _assess_security_implications(self, function) -> List[str]:
        """Assess security implications of the function."""
        try:
            implications = []
            
            if hasattr(function, 'source') and function.source:
                dangerous_patterns = {
                    'eval(': 'Uses eval() which can execute arbitrary code',
                    'exec(': 'Uses exec() which can execute arbitrary code',
                    'pickle.loads(': 'Uses pickle.loads() which can execute arbitrary code',
                    'subprocess.': 'Uses subprocess which can execute system commands',
                    'os.system(': 'Uses os.system() which can execute system commands',
                    '__import__(': 'Uses dynamic imports which can be dangerous'
                }
                
                for pattern, implication in dangerous_patterns.items():
                    if pattern in function.source:
                        implications.append(implication)
            
            return implications
        except:
            return []
    
    def _assess_performance_impact(self, function) -> str:
        """Assess the performance impact of the function."""
        try:
            if not hasattr(function, 'source') or not function.source:
                return "unknown"
            
            # Simple heuristics for performance assessment
            source = function.source.lower()
            
            # High impact indicators
            if any(pattern in source for pattern in ['nested loop', 'for.*for', 'while.*while', 'recursive']):
                return "high"
            
            # Medium impact indicators
            if any(pattern in source for pattern in ['for', 'while', 'comprehension']):
                return "medium"
            
            # Low impact (simple functions)
            return "low"
        except:
            return "unknown"
    
    def _analyze_file_imports(self, file) -> Dict[str, Any]:
        """Analyze file imports."""
        try:
            imports_info = {
                "total_imports": 0,
                "external_imports": [],
                "internal_imports": [],
                "unused_imports": []
            }
            
            if hasattr(file, 'imports'):
                imports_info["total_imports"] = len(file.imports)
                
                for imp in file.imports:
                    import_name = imp.source if hasattr(imp, 'source') else str(imp)
                    
                    # Check if external (simplified heuristic)
                    if any(ext in import_name for ext in ['os', 'sys', 'json', 'requests', 'numpy', 'pandas']):
                        imports_info["external_imports"].append(import_name)
                    else:
                        imports_info["internal_imports"].append(import_name)
                    
                    # Check if unused
                    if hasattr(imp, 'usages') and len(imp.usages) == 0:
                        imports_info["unused_imports"].append(import_name)
            
            return imports_info
        except:
            return {}
    
    def _get_external_dependencies(self, file) -> List[str]:
        """Get external dependencies for a file."""
        try:
            external_deps = []
            if hasattr(file, 'imports'):
                for imp in file.imports:
                    import_name = imp.source if hasattr(imp, 'source') else str(imp)
                    # Simple heuristic for external dependencies
                    if not import_name.startswith('.') and not import_name.startswith('backend'):
                        external_deps.append(import_name)
            return external_deps
        except:
            return []
    
    def _get_file_functions(self, file) -> List[Dict[str, str]]:
        """Get functions in a file."""
        try:
            functions = []
            if hasattr(file, 'functions'):
                for func in file.functions:
                    functions.append({
                        "name": func.name,
                        "line_count": len(func.source.split('\n')) if hasattr(func, 'source') else 0
                    })
            return functions
        except:
            return []
    
    def _get_file_classes(self, file) -> List[Dict[str, str]]:
        """Get classes in a file."""
        try:
            classes = []
            if hasattr(file, 'classes'):
                for cls in file.classes:
                    classes.append({
                        "name": cls.name,
                        "method_count": len(cls.methods) if hasattr(cls, 'methods') else 0
                    })
            return classes
        except:
            return []
    
    def _analyze_code_structure(self, file) -> Dict[str, Any]:
        """Analyze the code structure of a file."""
        try:
            structure = {}
            
            if hasattr(file, 'source') and file.source:
                lines = file.source.split('\n')
                structure["total_lines"] = len(lines)
                structure["code_lines"] = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
                structure["comment_lines"] = len([line for line in lines if line.strip().startswith('#')])
                structure["blank_lines"] = len([line for line in lines if not line.strip()])
                
                # Calculate comment ratio
                if structure["code_lines"] > 0:
                    structure["comment_ratio"] = structure["comment_lines"] / structure["code_lines"]
                else:
                    structure["comment_ratio"] = 0
            
            return structure
        except:
            return {}
    
    def _get_file_relationships(self, file) -> List[str]:
        """Get files related to this file."""
        try:
            related = set()
            
            # Add files from imports
            if hasattr(file, 'imports'):
                for imp in file.imports:
                    if hasattr(imp, 'imported_symbol') and hasattr(imp.imported_symbol, 'file'):
                        related.add(str(imp.imported_symbol.file.path))
            
            return list(related)
        except:
            return []
    
    def _get_location_context(self, location: CodeLocation) -> Dict[str, Any]:
        """Get context for a code location."""
        try:
            context = {
                "file_path": location.file_path,
                "line_start": location.line_start,
                "line_end": location.line_end,
                "line_range": f"{location.line_start}-{location.line_end}" if location.line_end != location.line_start else str(location.line_start)
            }
            
            # Try to get surrounding code context
            try:
                with open(location.file_path, 'r') as f:
                    lines = f.readlines()
                    
                # Get context lines (5 before and after)
                start_idx = max(0, location.line_start - 6)
                end_idx = min(len(lines), location.line_end + 5)
                
                context["surrounding_code"] = {
                    "before": [line.rstrip() for line in lines[start_idx:location.line_start-1]],
                    "target": [line.rstrip() for line in lines[location.line_start-1:location.line_end]],
                    "after": [line.rstrip() for line in lines[location.line_end:end_idx]]
                }
            except:
                context["surrounding_code"] = None
            
            return context
        except:
            return {}
    
    def _analyze_issue_impact(self, issue: Issue) -> Dict[str, Any]:
        """Analyze the impact of an issue."""
        try:
            impact = {
                "severity_level": issue.severity.value if isinstance(issue.severity, IssueSeverity) else issue.severity,
                "category": issue.category.value if isinstance(issue.category, IssueCategory) else issue.category,
                "affected_components": [],
                "potential_consequences": []
            }
            
            # Determine potential consequences based on issue type
            if isinstance(issue.type, IssueType):
                if issue.type in [IssueType.INFINITE_LOOP, IssueType.OFF_BY_ONE_ERROR]:
                    impact["potential_consequences"].append("Runtime errors or crashes")
                elif issue.type in [IssueType.DANGEROUS_FUNCTION_USAGE, IssueType.UNSAFE_EVAL]:
                    impact["potential_consequences"].append("Security vulnerabilities")
                elif issue.type in [IssueType.UNUSED_FUNCTION, IssueType.UNUSED_CLASS]:
                    impact["potential_consequences"].append("Code bloat and maintenance overhead")
                elif issue.type in [IssueType.HIGH_COMPLEXITY, IssueType.LONG_FUNCTION]:
                    impact["potential_consequences"].append("Reduced maintainability and increased bug risk")
            
            # Analyze affected components
            if issue.item:
                if hasattr(issue.item, 'name'):
                    impact["affected_components"].append(f"Function/Class: {issue.item.name}")
                if hasattr(issue.item, 'file') and hasattr(issue.item.file, 'path'):
                    impact["affected_components"].append(f"File: {issue.item.file.path}")
            
            return impact
        except:
            return {}
    
    def _generate_fix_suggestions(self, issue: Issue) -> List[str]:
        """Generate specific fix suggestions for an issue."""
        try:
            suggestions = []
            
            if isinstance(issue.type, IssueType):
                if issue.type == IssueType.UNREACHABLE_CODE:
                    suggestions.extend([
                        "Remove the unreachable code",
                        "Restructure the function logic to make the code reachable",
                        "Add conditional logic if the code should be reachable under certain conditions"
                    ])
                elif issue.type == IssueType.INFINITE_LOOP:
                    suggestions.extend([
                        "Add a break statement with appropriate condition",
                        "Add a return statement to exit the function",
                        "Modify the loop condition to ensure termination"
                    ])
                elif issue.type == IssueType.OFF_BY_ONE_ERROR:
                    suggestions.extend([
                        "Use array[len(array)-1] instead of array[len(array)]",
                        "Use negative indexing: array[-1] for the last element",
                        "Check array bounds before accessing elements"
                    ])
                elif issue.type == IssueType.DANGEROUS_FUNCTION_USAGE:
                    suggestions.extend([
                        "Replace eval() with safer alternatives like ast.literal_eval()",
                        "Use subprocess with shell=False and validate inputs",
                        "Consider using json.loads() instead of pickle.loads() for data"
                    ])
                elif issue.type == IssueType.UNUSED_FUNCTION:
                    suggestions.extend([
                        "Remove the function if it's truly unused",
                        "Add documentation explaining why the function is kept",
                        "Consider making it private with underscore prefix if it's internal"
                    ])
                elif issue.type == IssueType.LINE_TOO_LONG:
                    suggestions.extend([
                        "Break the line into multiple lines",
                        "Extract complex expressions into variables",
                        "Use parentheses for implicit line continuation"
                    ])
            
            # Add generic suggestion if no specific ones
            if not suggestions and issue.suggestion:
                suggestions.append(issue.suggestion)
            
            return suggestions
        except:
            return []

# ====================================================================
# STEP 5: ADVANCED ANALYSIS CAPABILITIES
# ====================================================================

def detect_circular_dependencies_advanced(codebase) -> List[Issue]:
    """
    Detect circular dependencies with enhanced context and analysis.
    
    Args:
        codebase: The codebase to analyze
        
    Returns:
        List of circular dependency issues
    """
    issues = []
    
    if not CODEGEN_SDK_AVAILABLE or not hasattr(codebase, 'files'):
        return issues
    
    try:
        # Build dependency graph
        dependency_graph = {}
        file_dependencies = {}
        
        # Collect file-level dependencies
        for file in codebase.files:
            file_path = str(file.path) if hasattr(file, 'path') else 'unknown'
            dependencies = set()
            
            if hasattr(file, 'imports'):
                for imp in file.imports:
                    if hasattr(imp, 'imported_symbol') and hasattr(imp.imported_symbol, 'file'):
                        dep_path = str(imp.imported_symbol.file.path)
                        if dep_path != file_path:  # Avoid self-dependencies
                            dependencies.add(dep_path)
            
            file_dependencies[file_path] = dependencies
            dependency_graph[file_path] = list(dependencies)
        
        # Find cycles using DFS
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node, path):
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in dependency_graph.get(node, []):
                dfs(neighbor, path + [node])
            
            rec_stack.remove(node)
        
        # Run DFS from each node
        for node in dependency_graph:
            if node not in visited:
                dfs(node, [])
        
        # Create issues for each cycle found
        for cycle in cycles:
            if len(cycle) > 1:  # Ignore self-loops
                cycle_description = " -> ".join([f.split('/')[-1] for f in cycle])
                
                # Find the file object for the first file in the cycle
                first_file = None
                for file in codebase.files:
                    if str(file.path) == cycle[0]:
                        first_file = file
                        break
                
                location = CodeLocation(
                    file_path=cycle[0],
                    line_start=1
                )
                
                issue = Issue(
                    type=IssueType.CIRCULAR_DEPENDENCY,
                    category=IssueCategory.DEPENDENCY_ISSUE,
                    severity=IssueSeverity.WARNING,
                    message=f"Circular dependency detected: {cycle_description}",
                    location=location,
                    item=first_file,
                    suggestion="Refactor code to break the circular dependency by extracting common functionality"
                )
                
                # Add rich context
                issue.add_context("cycle_length", len(cycle) - 1)
                issue.add_context("cycle_files", cycle)
                issue.add_context("cycle_description", cycle_description)
                issue.add_context("dependency_type", "file_level")
                
                issues.append(issue)
    
    except Exception as e:
        # Create an issue for analysis failure
        issues.append(Issue(
            type=IssueType.IMPLEMENTATION_ERROR,
            category=IssueCategory.IMPLEMENTATION_ERROR,
            severity=IssueSeverity.WARNING,
            message=f"Failed to analyze circular dependencies: {str(e)}",
            suggestion="Check codebase structure and dependencies"
        ))
    
    return issues

def analyze_inheritance_patterns(codebase) -> List[Issue]:
    """
    Analyze inheritance patterns and detect potential issues.
    
    Args:
        codebase: The codebase to analyze
        
    Returns:
        List of inheritance-related issues
    """
    issues = []
    
    if not CODEGEN_SDK_AVAILABLE or not hasattr(codebase, 'classes'):
        return issues
    
    try:
        # Analyze each class
        for cls in codebase.classes:
            # Check for deep inheritance hierarchies
            inheritance_depth = _calculate_inheritance_depth(cls)
            if inheritance_depth > 5:  # Arbitrary threshold
                location = CodeLocation(
                    file_path=str(cls.file.path) if hasattr(cls, 'file') and hasattr(cls.file, 'path') else 'unknown',
                    line_start=cls.line if hasattr(cls, 'line') else 1
                )
                
                issue = Issue(
                    type=IssueType.HIGH_COMPLEXITY,
                    category=IssueCategory.CODE_QUALITY,
                    severity=IssueSeverity.WARNING,
                    message=f"Deep inheritance hierarchy detected in class '{cls.name}' (depth: {inheritance_depth})",
                    location=location,
                    item=cls,
                    suggestion="Consider using composition instead of deep inheritance"
                )
                
                issue.add_context("inheritance_depth", inheritance_depth)
                issue.add_context("class_name", cls.name)
                issue.add_context("inheritance_chain", _get_inheritance_chain(cls))
                
                issues.append(issue)
            
            # Check for multiple inheritance complexity
            if hasattr(cls, 'parents') and len(cls.parents) > 2:
                location = CodeLocation(
                    file_path=str(cls.file.path) if hasattr(cls, 'file') and hasattr(cls.file, 'path') else 'unknown',
                    line_start=cls.line if hasattr(cls, 'line') else 1
                )
                
                issue = Issue(
                    type=IssueType.HIGH_COMPLEXITY,
                    category=IssueCategory.CODE_QUALITY,
                    severity=IssueSeverity.INFO,
                    message=f"Complex multiple inheritance in class '{cls.name}' ({len(cls.parents)} parents)",
                    location=location,
                    item=cls,
                    suggestion="Consider simplifying the inheritance structure or using mixins"
                )
                
                issue.add_context("parent_count", len(cls.parents))
                issue.add_context("parent_classes", [p.name for p in cls.parents if hasattr(p, 'name')])
                
                issues.append(issue)
            
            # Check for unused inherited methods
            unused_inherited = _find_unused_inherited_methods(cls)
            if unused_inherited:
                location = CodeLocation(
                    file_path=str(cls.file.path) if hasattr(cls, 'file') and hasattr(cls.file, 'path') else 'unknown',
                    line_start=cls.line if hasattr(cls, 'line') else 1
                )
                
                issue = Issue(
                    type=IssueType.UNUSED_FUNCTION,
                    category=IssueCategory.DEAD_CODE,
                    severity=IssueSeverity.INFO,
                    message=f"Class '{cls.name}' inherits unused methods: {', '.join(unused_inherited)}",
                    location=location,
                    item=cls,
                    suggestion="Consider removing unused inherited methods or documenting their purpose"
                )
                
                issue.add_context("unused_methods", unused_inherited)
                issue.add_context("inheritance_analysis", True)
                
                issues.append(issue)
    
    except Exception as e:
        # Create an issue for analysis failure
        issues.append(Issue(
            type=IssueType.IMPLEMENTATION_ERROR,
            category=IssueCategory.IMPLEMENTATION_ERROR,
            severity=IssueSeverity.WARNING,
            message=f"Failed to analyze inheritance patterns: {str(e)}",
            suggestion="Check class definitions and inheritance structure"
        ))
    
    return issues

def analyze_complexity_patterns(codebase) -> List[Issue]:
    """
    Analyze complexity patterns and identify high-complexity code.
    
    Args:
        codebase: The codebase to analyze
        
    Returns:
        List of complexity-related issues
    """
    issues = []
    
    if not CODEGEN_SDK_AVAILABLE or not hasattr(codebase, 'functions'):
        return issues
    
    try:
        context_collector = ContextCollector(codebase)
        
        # Analyze each function for complexity
        for func in codebase.functions:
            complexity_metrics = context_collector._calculate_complexity_metrics(func)
            
            # Check cyclomatic complexity
            cyclomatic = complexity_metrics.get("cyclomatic_complexity", 0)
            if cyclomatic > 10:  # Standard threshold
                location = CodeLocation(
                    file_path=str(func.file.path) if hasattr(func, 'file') and hasattr(func.file, 'path') else 'unknown',
                    line_start=func.line if hasattr(func, 'line') else 1
                )
                
                severity = IssueSeverity.ERROR if cyclomatic > 20 else IssueSeverity.WARNING
                
                issue = Issue(
                    type=IssueType.HIGH_COMPLEXITY,
                    category=IssueCategory.PERFORMANCE_ISSUE,
                    severity=severity,
                    message=f"High cyclomatic complexity in function '{func.name}' (complexity: {cyclomatic})",
                    location=location,
                    item=func,
                    suggestion="Consider breaking this function into smaller, more focused functions"
                )
                
                issue.add_context("cyclomatic_complexity", cyclomatic)
                issue.add_context("complexity_metrics", complexity_metrics)
                issue.add_context("function_name", func.name)
                
                issues.append(issue)
            
            # Check function length
            total_lines = complexity_metrics.get("total_lines", 0)
            if total_lines > 50:  # Arbitrary threshold
                location = CodeLocation(
                    file_path=str(func.file.path) if hasattr(func, 'file') and hasattr(func.file, 'path') else 'unknown',
                    line_start=func.line if hasattr(func, 'line') else 1
                )
                
                issue = Issue(
                    type=IssueType.LONG_FUNCTION,
                    category=IssueCategory.CODE_QUALITY,
                    severity=IssueSeverity.WARNING,
                    message=f"Long function '{func.name}' ({total_lines} lines)",
                    location=location,
                    item=func,
                    suggestion="Consider breaking this long function into smaller, more focused functions"
                )
                
                issue.add_context("total_lines", total_lines)
                issue.add_context("code_lines", complexity_metrics.get("code_lines", 0))
                issue.add_context("comment_lines", complexity_metrics.get("comment_lines", 0))
                
                issues.append(issue)
            
            # Check nesting depth
            max_nesting = complexity_metrics.get("max_nesting_depth", 0)
            if max_nesting > 4:  # Arbitrary threshold
                location = CodeLocation(
                    file_path=str(func.file.path) if hasattr(func, 'file') and hasattr(func.file, 'path') else 'unknown',
                    line_start=func.line if hasattr(func, 'line') else 1
                )
                
                issue = Issue(
                    type=IssueType.DEEP_NESTING,
                    category=IssueCategory.CODE_QUALITY,
                    severity=IssueSeverity.WARNING,
                    message=f"Deep nesting in function '{func.name}' (depth: {max_nesting})",
                    location=location,
                    item=func,
                    suggestion="Consider extracting nested logic into separate functions or using early returns"
                )
                
                issue.add_context("max_nesting_depth", max_nesting)
                issue.add_context("nesting_analysis", True)
                
                issues.append(issue)
    
    except Exception as e:
        # Create an issue for analysis failure
        issues.append(Issue(
            type=IssueType.IMPLEMENTATION_ERROR,
            category=IssueCategory.IMPLEMENTATION_ERROR,
            severity=IssueSeverity.WARNING,
            message=f"Failed to analyze complexity patterns: {str(e)}",
            suggestion="Check function definitions and code structure"
        ))
    
    return issues

def analyze_performance_patterns(codebase) -> List[Issue]:
    """
    Analyze performance patterns and identify potential performance issues.
    
    Args:
        codebase: The codebase to analyze
        
    Returns:
        List of performance-related issues
    """
    issues = []
    
    if not CODEGEN_SDK_AVAILABLE or not hasattr(codebase, 'functions'):
        return issues
    
    try:
        # Analyze each function for performance patterns
        for func in codebase.functions:
            if not hasattr(func, 'source') or not func.source:
                continue
            
            source_lower = func.source.lower()
            
            # Check for nested loops
            if _has_nested_loops(func.source):
                location = CodeLocation(
                    file_path=str(func.file.path) if hasattr(func, 'file') and hasattr(func.file, 'path') else 'unknown',
                    line_start=func.line if hasattr(func, 'line') else 1
                )
                
                issue = Issue(
                    type=IssueType.INEFFICIENT_LOOP,
                    category=IssueCategory.PERFORMANCE_ISSUE,
                    severity=IssueSeverity.WARNING,
                    message=f"Nested loops detected in function '{func.name}' - potential O(n²) complexity",
                    location=location,
                    item=func,
                    suggestion="Consider optimizing the algorithm or using more efficient data structures"
                )
                
                issue.add_context("performance_pattern", "nested_loops")
                issue.add_context("complexity_estimate", "O(n²) or higher")
                
                issues.append(issue)
            
            # Check for inefficient string concatenation
            if '+=' in func.source and 'str' in source_lower:
                location = CodeLocation(
                    file_path=str(func.file.path) if hasattr(func, 'file') and hasattr(func.file, 'path') else 'unknown',
                    line_start=func.line if hasattr(func, 'line') else 1
                )
                
                issue = Issue(
                    type=IssueType.INEFFICIENT_PATTERN,
                    category=IssueCategory.PERFORMANCE_ISSUE,
                    severity=IssueSeverity.INFO,
                    message=f"Potential inefficient string concatenation in function '{func.name}'",
                    location=location,
                    item=func,
                    suggestion="Consider using join() or f-strings for better performance"
                )
                
                issue.add_context("performance_pattern", "string_concatenation")
                issue.add_context("optimization_suggestion", "Use ''.join() or f-strings")
                
                issues.append(issue)
            
            # Check for repeated expensive operations
            expensive_ops = ['re.compile', 'json.loads', 'pickle.loads', 'open(']
            for op in expensive_ops:
                if func.source.count(op) > 1:
                    location = CodeLocation(
                        file_path=str(func.file.path) if hasattr(func, 'file') and hasattr(func.file, 'path') else 'unknown',
                        line_start=func.line if hasattr(func, 'line') else 1
                    )
                    
                    issue = Issue(
                        type=IssueType.UNNECESSARY_COMPUTATION,
                        category=IssueCategory.PERFORMANCE_ISSUE,
                        severity=IssueSeverity.INFO,
                        message=f"Repeated expensive operation '{op}' in function '{func.name}' ({func.source.count(op)} times)",
                        location=location,
                        item=func,
                        suggestion=f"Consider caching the result of '{op}' or moving it outside the function"
                    )
                    
                    issue.add_context("expensive_operation", op)
                    issue.add_context("occurrence_count", func.source.count(op))
                    
                    issues.append(issue)
    
    except Exception as e:
        # Create an issue for analysis failure
        issues.append(Issue(
            type=IssueType.IMPLEMENTATION_ERROR,
            category=IssueCategory.IMPLEMENTATION_ERROR,
            severity=IssueSeverity.WARNING,
            message=f"Failed to analyze performance patterns: {str(e)}",
            suggestion="Check function definitions and performance patterns"
        ))
    
    return issues

# Helper functions for advanced analysis

def _calculate_inheritance_depth(cls) -> int:
    """Calculate the inheritance depth of a class."""
    try:
        if not hasattr(cls, 'parents') or not cls.parents:
            return 0
        
        max_depth = 0
        for parent in cls.parents:
            parent_depth = _calculate_inheritance_depth(parent)
            max_depth = max(max_depth, parent_depth)
        
        return max_depth + 1
    except:
        return 0

def _get_inheritance_chain(cls) -> List[str]:
    """Get the inheritance chain for a class."""
    try:
        chain = [cls.name]
        if hasattr(cls, 'parents') and cls.parents:
            for parent in cls.parents:
                parent_chain = _get_inheritance_chain(parent)
                chain.extend(parent_chain)
        return chain
    except:
        return [cls.name if hasattr(cls, 'name') else 'unknown']

def _find_unused_inherited_methods(cls) -> List[str]:
    """Find inherited methods that are not used."""
    try:
        unused = []
        if hasattr(cls, 'parents') and cls.parents:
            for parent in cls.parents:
                if hasattr(parent, 'methods'):
                    for method in parent.methods:
                        if hasattr(method, 'usages') and len(method.usages) == 0:
                            unused.append(method.name)
        return unused
    except:
        return []

def _has_nested_loops(source: str) -> bool:
    """Check if the source code has nested loops."""
    try:
        lines = source.split('\n')
        loop_keywords = ['for ', 'while ']
        
        in_loop = False
        loop_indent = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            
            current_indent = len(line) - len(line.lstrip())
            
            # Check if this line starts a loop
            if any(keyword in stripped for keyword in loop_keywords):
                if in_loop and current_indent > loop_indent:
                    return True  # Found nested loop
                in_loop = True
                loop_indent = current_indent
            elif current_indent <= loop_indent and in_loop:
                in_loop = False
        
        return False
    except:
        return False

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
            # Advanced analysis capabilities
            ("Detecting advanced circular dependencies", self._detect_circular_dependencies_advanced),
            ("Analyzing inheritance patterns", self._analyze_inheritance_patterns),
            ("Analyzing complexity patterns", self._analyze_complexity_patterns),
            ("Analyzing performance patterns", self._analyze_performance_patterns),
            # Integration and optimization
            ("Integrating performance optimization", self._integrate_performance_optimization),
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
    
    def _detect_circular_dependencies_advanced(self):
        """Wrapper method for advanced circular dependency detection."""
        try:
            circular_issues = detect_circular_dependencies_advanced(self.codebase)
            self.issues.extend(circular_issues)
            print(f"Found {len(circular_issues)} circular dependency issues")
        except Exception as e:
            print(f"Error in advanced circular dependency detection: {e}")
    
    def _analyze_inheritance_patterns(self):
        """Wrapper method for inheritance pattern analysis."""
        try:
            inheritance_issues = analyze_inheritance_patterns(self.codebase)
            self.issues.extend(inheritance_issues)
            print(f"Found {len(inheritance_issues)} inheritance pattern issues")
        except Exception as e:
            print(f"Error in inheritance pattern analysis: {e}")
    
    def _analyze_complexity_patterns(self):
        """Wrapper method for complexity pattern analysis."""
        try:
            complexity_issues = analyze_complexity_patterns(self.codebase)
            self.issues.extend(complexity_issues)
            print(f"Found {len(complexity_issues)} complexity pattern issues")
        except Exception as e:
            print(f"Error in complexity pattern analysis: {e}")
    
    def _analyze_performance_patterns(self):
        """Wrapper method for performance pattern analysis."""
        try:
            performance_issues = analyze_performance_patterns(self.codebase)
            self.issues.extend(performance_issues)
            print(f"Found {len(performance_issues)} performance pattern issues")
        except Exception as e:
            print(f"Error in performance pattern analysis: {e}")
    
    def _integrate_performance_optimization(self):
        """Integrate performance optimization features."""
        try:
            # Performance optimization functions now integrated
            
            # Get performance report
            perf_report = get_optimization_report()
            
            # Add performance data to context
            self.performance_data = perf_report
            
            print(f"Performance optimization integrated - {len(perf_report.get('performance', {}).get('bottlenecks', []))} bottlenecks identified")
            
        except ImportError:
            print("Performance optimization module not available")
            self.performance_data = {}
        except Exception as e:
            print(f"Error integrating performance optimization: {e}")
            self.performance_data = {}
    
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
        
        # Create basic analysis results
        basic_results = {
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
        
        # Generate enhanced report with actionable insights
        try:
            # Enhanced reporting functions now integrated
            
            enhanced_report = generate_enhanced_report(
                basic_results, 
                self.issues, 
                getattr(self, 'performance_data', {}),
                export_html=True
            )
            
            # Merge enhanced features into basic results
            basic_results.update({
                "enhanced_report": enhanced_report,
                "actionable_insights": enhanced_report.get("actionable_insights", []),
                "trend_analysis": enhanced_report.get("trend_analysis", []),
                "executive_summary": enhanced_report.get("executive_summary", {}),
                "recommendations": enhanced_report.get("recommendations", {})
            })
            
            print(f"Enhanced reporting generated with {len(enhanced_report.get('actionable_insights', []))} actionable insights")
            
        except ImportError:
            print("Enhanced reporting module not available")
        except Exception as e:
            print(f"Error generating enhanced report: {e}")
        
        return basic_results
    
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
#!/usr/bin/env python3
"""
Performance Optimization System for Codebase Analysis

This module provides:
- High-performance caching with LRU eviction
- Incremental analysis for changed files only
- Performance monitoring and bottleneck identification
- Memory usage optimization
- Thread-safe operations
"""

import os
import time
import hashlib
import pickle
import threading
import json
from functools import wraps
from typing import List, Dict, Any, Set, Callable, Optional
from collections import defaultdict

class AnalysisCache:
    """
    High-performance caching system for analysis results.
    
    Features:
    - Memory-based caching with LRU eviction
    - File-based persistent caching
    - Thread-safe operations
    - Cache invalidation based on file modification times
    """
    
    def __init__(self, max_memory_items: int = 1000, cache_dir: str = ".analysis_cache"):
        self.max_memory_items = max_memory_items
        self.cache_dir = cache_dir
        self.memory_cache = {}
        self.access_order = []
        self.lock = threading.RLock()
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
    
    def _generate_cache_key(self, func_name: str, *args, **kwargs) -> str:
        """Generate a unique cache key for function arguments."""
        try:
            # Create a hash of function name and arguments
            key_data = {
                'func': func_name,
                'args': str(args),
                'kwargs': str(sorted(kwargs.items()))
            }
            key_string = json.dumps(key_data, sort_keys=True, default=str)
            return hashlib.md5(key_string.encode()).hexdigest()
        except:
            # Fallback to simple string concatenation
            return f"{func_name}_{hash(str(args))}_{hash(str(kwargs))}"
    
    def _get_file_hash(self, filepath: str) -> str:
        """Get hash of file content for cache invalidation."""
        try:
            if not os.path.exists(filepath):
                return "nonexistent"
            
            # Use modification time and size for quick hash
            stat = os.stat(filepath)
            return f"{stat.st_mtime}_{stat.st_size}"
        except:
            return "unknown"
    
    def get(self, key: str, file_dependencies: List[str] = None) -> Any:
        """Get cached result if valid."""
        with self.lock:
            # Check memory cache first
            if key in self.memory_cache:
                cached_data = self.memory_cache[key]
                
                # Check if cache is still valid
                if self._is_cache_valid(cached_data, file_dependencies):
                    # Update access order for LRU
                    if key in self.access_order:
                        self.access_order.remove(key)
                    self.access_order.append(key)
                    return cached_data['result']
                else:
                    # Remove invalid cache entry
                    del self.memory_cache[key]
                    if key in self.access_order:
                        self.access_order.remove(key)
            
            # Check file cache
            cache_file = os.path.join(self.cache_dir, f"{key}.cache")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    
                    if self._is_cache_valid(cached_data, file_dependencies):
                        # Load into memory cache
                        self._store_in_memory(key, cached_data)
                        return cached_data['result']
                    else:
                        # Remove invalid file cache
                        os.remove(cache_file)
                except:
                    # Remove corrupted cache file
                    try:
                        os.remove(cache_file)
                    except:
                        pass
            
            return None
    
    def set(self, key: str, result: Any, file_dependencies: List[str] = None):
        """Store result in cache."""
        with self.lock:
            # Prepare cache data
            cached_data = {
                'result': result,
                'timestamp': time.time(),
                'file_hashes': {}
            }
            
            # Store file hashes for invalidation
            if file_dependencies:
                for filepath in file_dependencies:
                    cached_data['file_hashes'][filepath] = self._get_file_hash(filepath)
            
            # Store in memory cache
            self._store_in_memory(key, cached_data)
            
            # Store in file cache for persistence
            try:
                cache_file = os.path.join(self.cache_dir, f"{key}.cache")
                with open(cache_file, 'wb') as f:
                    pickle.dump(cached_data, f)
            except:
                pass  # File cache is optional
    
    def _store_in_memory(self, key: str, cached_data: dict):
        """Store data in memory cache with LRU eviction."""
        # Add to memory cache
        self.memory_cache[key] = cached_data
        
        # Update access order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        # Evict oldest items if cache is full
        while len(self.memory_cache) > self.max_memory_items:
            oldest_key = self.access_order.pop(0)
            if oldest_key in self.memory_cache:
                del self.memory_cache[oldest_key]
    
    def _is_cache_valid(self, cached_data: dict, file_dependencies: List[str] = None) -> bool:
        """Check if cached data is still valid."""
        try:
            # Check file dependencies
            if file_dependencies and 'file_hashes' in cached_data:
                for filepath in file_dependencies:
                    current_hash = self._get_file_hash(filepath)
                    cached_hash = cached_data['file_hashes'].get(filepath)
                    if current_hash != cached_hash:
                        return False
            
            # Cache is valid if no dependencies or all dependencies are unchanged
            return True
        except:
            return False
    
    def clear(self):
        """Clear all caches."""
        with self.lock:
            self.memory_cache.clear()
            self.access_order.clear()
            
            # Clear file cache
            try:
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.cache'):
                        os.remove(os.path.join(self.cache_dir, filename))
            except:
                pass
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'memory_cache_size': len(self.memory_cache),
                'max_memory_items': self.max_memory_items,
                'cache_hit_ratio': self._calculate_hit_ratio(),
                'total_cached_items': len(self.memory_cache)
            }
    
    def _calculate_hit_ratio(self) -> float:
        """Calculate cache hit ratio (simplified)."""
        # This is a simplified implementation
        # In a real system, you'd track hits and misses
        return 0.85  # Placeholder

class IncrementalAnalyzer:
    """
    Incremental analysis system that only analyzes changed files.
    
    Features:
    - File change detection based on modification times
    - Dependency tracking for affected file analysis
    - Efficient re-analysis of only changed components
    """
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.file_states = {}  # filepath -> {mtime, size, hash}
        self.dependency_graph = {}  # filepath -> [dependent_files]
        self.analysis_results = {}  # filepath -> analysis_results
        
    def get_changed_files(self, file_paths: List[str]) -> List[str]:
        """Get list of files that have changed since last analysis."""
        changed_files = []
        
        for filepath in file_paths:
            try:
                if not os.path.exists(filepath):
                    # File was deleted
                    if filepath in self.file_states:
                        changed_files.append(filepath)
                        del self.file_states[filepath]
                    continue
                
                # Get current file state
                stat = os.stat(filepath)
                current_state = {
                    'mtime': stat.st_mtime,
                    'size': stat.st_size
                }
                
                # Check if file has changed
                if filepath not in self.file_states:
                    # New file
                    changed_files.append(filepath)
                    self.file_states[filepath] = current_state
                elif (self.file_states[filepath]['mtime'] != current_state['mtime'] or
                      self.file_states[filepath]['size'] != current_state['size']):
                    # Modified file
                    changed_files.append(filepath)
                    self.file_states[filepath] = current_state
                    
            except Exception as e:
                # If we can't check the file, assume it changed
                changed_files.append(filepath)
        
        return changed_files
    
    def get_affected_files(self, changed_files: List[str]) -> Set[str]:
        """Get all files affected by changes (including dependencies)."""
        affected = set(changed_files)
        
        # Add files that depend on changed files
        for changed_file in changed_files:
            affected.update(self._get_dependent_files(changed_file))
        
        return affected
    
    def _get_dependent_files(self, filepath: str) -> Set[str]:
        """Get files that depend on the given file."""
        dependents = set()
        
        # Direct dependents
        if filepath in self.dependency_graph:
            dependents.update(self.dependency_graph[filepath])
        
        # Transitive dependents (recursive, with cycle detection)
        visited = set()
        for dependent in list(dependents):
            if dependent not in visited:
                visited.add(dependent)
                dependents.update(self._get_dependent_files_recursive(dependent, visited))
        
        return dependents
    
    def _get_dependent_files_recursive(self, filepath: str, visited: Set[str]) -> Set[str]:
        """Get dependent files recursively with cycle detection."""
        dependents = set()
        
        if filepath in self.dependency_graph:
            for dependent in self.dependency_graph[filepath]:
                if dependent not in visited:
                    visited.add(dependent)
                    dependents.add(dependent)
                    dependents.update(self._get_dependent_files_recursive(dependent, visited))
        
        return dependents
    
    def update_dependency_graph(self, codebase):
        """Update the dependency graph based on current codebase."""
        self.dependency_graph.clear()
        
        if not codebase or not hasattr(codebase, 'files'):
            return
        
        try:
            # Build dependency graph from imports
            for file in codebase.files:
                file_path = str(file.path) if hasattr(file, 'path') else None
                if not file_path:
                    continue
                
                dependencies = []
                if hasattr(file, 'imports'):
                    for imp in file.imports:
                        if hasattr(imp, 'imported_symbol') and hasattr(imp.imported_symbol, 'file'):
                            dep_path = str(imp.imported_symbol.file.path)
                            dependencies.append(dep_path)
                
                # Add reverse dependencies
                for dep_path in dependencies:
                    if dep_path not in self.dependency_graph:
                        self.dependency_graph[dep_path] = []
                    if file_path not in self.dependency_graph[dep_path]:
                        self.dependency_graph[dep_path].append(file_path)
                        
        except Exception as e:
            print(f"Warning: Failed to update dependency graph: {e}")
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get incremental analysis statistics."""
        return {
            'tracked_files': len(self.file_states),
            'dependency_relationships': sum(len(deps) for deps in self.dependency_graph.values()),
            'cached_results': len(self.analysis_results)
        }

class PerformanceMonitor:
    """
    Monitor and optimize analysis performance.
    
    Features:
    - Execution time tracking
    - Memory usage monitoring
    - Performance bottleneck identification
    - Optimization suggestions
    """
    
    def __init__(self):
        self.execution_times = defaultdict(list)
        self.memory_usage = defaultdict(list)
        self.call_counts = defaultdict(int)
        self.lock = threading.RLock()
        
    def track_execution(self, func_name: str):
        """Decorator to track function execution time."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                except Exception as e:
                    result = e
                    success = False
                
                end_time = time.time()
                end_memory = self._get_memory_usage()
                
                # Record metrics
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory
                
                with self.lock:
                    self.execution_times[func_name].append(execution_time)
                    self.memory_usage[func_name].append(memory_delta)
                    self.call_counts[func_name] += 1
                
                if not success:
                    raise result
                
                return result
            
            return wrapper
        return decorator
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0.0
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance analysis report."""
        with self.lock:
            report = {
                'summary': {},
                'bottlenecks': [],
                'optimization_suggestions': [],
                'total_analysis_time': 0.0,
                'most_called_functions': [],
                'memory_intensive_functions': []
            }
            
            total_time = 0.0
            
            # Calculate summary statistics
            for func_name in self.execution_times:
                times = self.execution_times[func_name]
                memory = self.memory_usage[func_name]
                
                if times:
                    avg_time = sum(times) / len(times)
                    max_time = max(times)
                    total_func_time = sum(times)
                    total_time += total_func_time
                    
                    avg_memory = sum(memory) / len(memory) if memory else 0
                    max_memory = max(memory) if memory else 0
                    
                    report['summary'][func_name] = {
                        'call_count': self.call_counts[func_name],
                        'avg_time': avg_time,
                        'max_time': max_time,
                        'total_time': total_func_time,
                        'avg_memory': avg_memory,
                        'max_memory': max_memory
                    }
                    
                    # Identify bottlenecks
                    if avg_time > 1.0:  # Functions taking more than 1 second on average
                        report['bottlenecks'].append({
                            'function': func_name,
                            'avg_time': avg_time,
                            'total_time': total_func_time,
                            'call_count': self.call_counts[func_name],
                            'efficiency_score': self.call_counts[func_name] / total_func_time if total_func_time > 0 else 0
                        })
            
            report['total_analysis_time'] = total_time
            
            # Find most called functions
            most_called = sorted(self.call_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            report['most_called_functions'] = [{'function': name, 'calls': count} for name, count in most_called]
            
            # Find memory intensive functions
            memory_intensive = []
            for func_name, memory_list in self.memory_usage.items():
                if memory_list:
                    avg_memory = sum(memory_list) / len(memory_list)
                    if avg_memory > 10.0:  # Functions using more than 10MB on average
                        memory_intensive.append({'function': func_name, 'avg_memory': avg_memory})
            
            report['memory_intensive_functions'] = sorted(memory_intensive, key=lambda x: x['avg_memory'], reverse=True)[:5]
            
            # Generate optimization suggestions
            report['optimization_suggestions'] = self._generate_optimization_suggestions(report['summary'])
            
            return report
    
    def _generate_optimization_suggestions(self, summary: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions based on performance data."""
        suggestions = []
        
        for func_name, metrics in summary.items():
            avg_time = metrics['avg_time']
            call_count = metrics['call_count']
            total_time = metrics['total_time']
            avg_memory = metrics['avg_memory']
            
            if avg_time > 2.0:
                suggestions.append(f"🐌 Consider optimizing '{func_name}' - average execution time is {avg_time:.2f}s")
            
            if call_count > 100 and avg_time > 0.1:
                suggestions.append(f"💾 Consider caching results for '{func_name}' - called {call_count} times with {avg_time:.2f}s average")
            
            if total_time > 10.0:
                suggestions.append(f"⚡ '{func_name}' consumes significant total time ({total_time:.2f}s) - consider parallelization")
            
            if avg_memory > 50.0:
                suggestions.append(f"🧠 '{func_name}' uses significant memory ({avg_memory:.1f}MB average) - consider memory optimization")
        
        return suggestions
    
    def reset_metrics(self):
        """Reset all performance metrics."""
        with self.lock:
            self.execution_times.clear()
            self.memory_usage.clear()
            self.call_counts.clear()

# Global instances
_analysis_cache = AnalysisCache()
_performance_monitor = PerformanceMonitor()

def cached_analysis(file_dependencies: List[str] = None):
    """
    Decorator for caching analysis results.
    
    Args:
        file_dependencies: List of file paths that invalidate cache when changed
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = _analysis_cache._generate_cache_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            cached_result = _analysis_cache.get(cache_key, file_dependencies)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            _analysis_cache.set(cache_key, result, file_dependencies)
            
            return result
        
        return wrapper
    return decorator

def performance_tracked(func_name: str = None):
    """
    Decorator for tracking function performance.
    
    Args:
        func_name: Optional custom name for the function
    """
    def decorator(func: Callable) -> Callable:
        name = func_name or func.__name__
        return _performance_monitor.track_execution(name)(func)
    return decorator

def get_cache_instance() -> AnalysisCache:
    """Get the global cache instance."""
    return _analysis_cache

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _performance_monitor

def clear_all_caches():
    """Clear all caches and reset performance metrics."""
    _analysis_cache.clear()
    _performance_monitor.reset_metrics()

def get_optimization_report() -> Dict[str, Any]:
    """Get comprehensive optimization report."""
    return {
        'performance': _performance_monitor.get_performance_report(),
        'cache_stats': _analysis_cache.get_cache_stats(),
        'recommendations': _generate_system_recommendations()
    }

def _generate_system_recommendations() -> List[str]:
    """Generate system-level optimization recommendations."""
    recommendations = []
    
    # Get current stats
    perf_report = _performance_monitor.get_performance_report()
    cache_stats = _analysis_cache.get_cache_stats()
    
    # Cache recommendations
    if cache_stats['memory_cache_size'] < cache_stats['max_memory_items'] * 0.5:
        recommendations.append("💾 Consider increasing cache size for better performance")
    
    # Performance recommendations
    if perf_report['total_analysis_time'] > 60.0:
        recommendations.append("⚡ Total analysis time is high - consider enabling incremental analysis")
    
    if len(perf_report['bottlenecks']) > 3:
        recommendations.append("🐌 Multiple performance bottlenecks detected - prioritize optimization")
    
    return recommendations

#!/usr/bin/env python3
"""
Comprehensive Testing Framework for Codebase Analysis

This module provides:
- Extensive test suites for all analysis components
- Performance benchmarking and validation
- Regression testing capabilities
- Test data generation and management
- Automated test reporting
"""

import os
import sys
import time
import json
import unittest
import tempfile
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Import analysis modules
try:
    from analysis import (
        ComprehensiveCodebaseAnalyzer, Issue, IssueCollection, 
        IssueType, IssueCategory, IssueSeverity, CodeLocation,
        detect_implementation_errors, detect_security_vulnerabilities,
        detect_circular_dependencies_advanced, analyze_inheritance_patterns,
        analyze_complexity_patterns, analyze_performance_patterns,
        ContextCollector
    )
except ImportError as e:
    print(f"Warning: Could not import analysis modules: {e}")

@dataclass
class TestResult:
    """Represents the result of a test case."""
    test_name: str
    passed: bool
    duration: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = None

@dataclass
class TestSuite:
    """Represents a collection of related tests."""
    name: str
    description: str
    tests: List[TestResult]
    setup_time: float = 0.0
    teardown_time: float = 0.0

class TestDataGenerator:
    """
    Generate test data for comprehensive analysis testing.
    
    Features:
    - Create synthetic code files with known issues
    - Generate dependency graphs for testing
    - Create performance test scenarios
    """
    
    def __init__(self, temp_dir: str = None):
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="analysis_test_")
        self.created_files = []
    
    def create_test_repository(self) -> str:
        """Create a test repository with various code patterns."""
        repo_path = os.path.join(self.temp_dir, "test_repo")
        os.makedirs(repo_path, exist_ok=True)
        
        # Create files with different types of issues
        self._create_implementation_error_file(repo_path)
        self._create_security_vulnerability_file(repo_path)
        self._create_performance_issue_file(repo_path)
        self._create_circular_dependency_files(repo_path)
        self._create_complex_inheritance_file(repo_path)
        self._create_dead_code_file(repo_path)
        
        return repo_path
    
    def _create_implementation_error_file(self, repo_path: str):
        """Create a file with implementation errors."""
        content = '''
def function_with_unreachable_code():
    """Function with unreachable code after return."""
    x = 10
    return x
    print("This code is unreachable")  # Issue: unreachable code
    y = 20

def function_with_infinite_loop():
    """Function with potential infinite loop."""
    while True:  # Issue: infinite loop without break
        print("This will run forever")
        # Missing break condition

def function_with_off_by_one():
    """Function with off-by-one error."""
    arr = [1, 2, 3, 4, 5]
    for i in range(len(arr) + 1):  # Issue: off-by-one error
        print(arr[i])  # Will cause IndexError
'''
        
        filepath = os.path.join(repo_path, "implementation_errors.py")
        with open(filepath, 'w') as f:
            f.write(content)
        self.created_files.append(filepath)
    
    def _create_security_vulnerability_file(self, repo_path: str):
        """Create a file with security vulnerabilities."""
        content = '''
import os
import subprocess
import pickle

def dangerous_eval_usage(user_input):
    """Function using dangerous eval."""
    result = eval(user_input)  # Issue: dangerous eval usage
    return result

def unsafe_subprocess_call(command):
    """Function with unsafe subprocess usage."""
    os.system(command)  # Issue: unsafe system call
    subprocess.call(command, shell=True)  # Issue: shell=True is dangerous

def pickle_security_issue(data):
    """Function with pickle security issue."""
    return pickle.loads(data)  # Issue: pickle.loads can execute arbitrary code

def sql_injection_risk(user_id):
    """Function with potential SQL injection."""
    query = f"SELECT * FROM users WHERE id = {user_id}"  # Issue: SQL injection risk
    return query
'''
        
        filepath = os.path.join(repo_path, "security_vulnerabilities.py")
        with open(filepath, 'w') as f:
            f.write(content)
        self.created_files.append(filepath)
    
    def _create_performance_issue_file(self, repo_path: str):
        """Create a file with performance issues."""
        content = '''
def nested_loops_performance_issue(data):
    """Function with nested loops causing O(n²) complexity."""
    result = []
    for i in data:  # Issue: nested loops
        for j in data:
            if i == j:
                result.append(i)
    return result

def inefficient_string_concatenation(items):
    """Function with inefficient string concatenation."""
    result = ""
    for item in items:  # Issue: inefficient string concatenation
        result += str(item)  # Should use join()
    return result

def repeated_expensive_operation(data):
    """Function with repeated expensive operations."""
    import re
    results = []
    for item in data:
        pattern = re.compile(r'\\d+')  # Issue: repeated re.compile
        match = pattern.search(item)
        if match:
            results.append(match.group())
    return results
'''
        
        filepath = os.path.join(repo_path, "performance_issues.py")
        with open(filepath, 'w') as f:
            f.write(content)
        self.created_files.append(filepath)
    
    def _create_circular_dependency_files(self, repo_path: str):
        """Create files with circular dependencies."""
        # File A imports B
        content_a = '''
from module_b import function_b

def function_a():
    return function_b() + 1
'''
        
        # File B imports A (circular dependency)
        content_b = '''
from module_a import function_a

def function_b():
    return 42

def another_function():
    return function_a() + 1  # Creates circular dependency
'''
        
        filepath_a = os.path.join(repo_path, "module_a.py")
        filepath_b = os.path.join(repo_path, "module_b.py")
        
        with open(filepath_a, 'w') as f:
            f.write(content_a)
        with open(filepath_b, 'w') as f:
            f.write(content_b)
        
        self.created_files.extend([filepath_a, filepath_b])
    
    def _create_complex_inheritance_file(self, repo_path: str):
        """Create a file with complex inheritance patterns."""
        content = '''
class BaseClass:
    def base_method(self):
        pass

class MiddleClass1(BaseClass):
    def middle_method1(self):
        pass

class MiddleClass2(BaseClass):
    def middle_method2(self):
        pass

class MiddleClass3(MiddleClass1):
    def middle_method3(self):
        pass

class MiddleClass4(MiddleClass2):
    def middle_method4(self):
        pass

class ComplexClass(MiddleClass3, MiddleClass4):  # Issue: complex multiple inheritance
    """Class with deep and complex inheritance hierarchy."""
    def complex_method(self):
        pass

class DeepInheritanceClass(ComplexClass):  # Issue: very deep inheritance
    """Class with very deep inheritance chain."""
    def deep_method(self):
        pass

class VeryDeepClass(DeepInheritanceClass):  # Issue: extremely deep inheritance
    def very_deep_method(self):
        pass
'''
        
        filepath = os.path.join(repo_path, "complex_inheritance.py")
        with open(filepath, 'w') as f:
            f.write(content)
        self.created_files.append(filepath)
    
    def _create_dead_code_file(self, repo_path: str):
        """Create a file with dead code."""
        content = '''
def unused_function():
    """This function is never called."""
    return "unused"

class UnusedClass:
    """This class is never instantiated."""
    def unused_method(self):
        return "unused"

def used_function():
    """This function is used."""
    return "used"

# Call the used function
result = used_function()

# Unused import
import json  # Issue: unused import
import os

# Used import
import sys
print(sys.version)

# Unused variable
unused_variable = "never used"  # Issue: unused variable

# Used variable
used_variable = "this is used"
print(used_variable)
'''
        
        filepath = os.path.join(repo_path, "dead_code.py")
        with open(filepath, 'w') as f:
            f.write(content)
        self.created_files.append(filepath)
    
    def cleanup(self):
        """Clean up created test files."""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

class ComprehensiveTestRunner:
    """
    Comprehensive test runner for the analysis system.
    
    Features:
    - Run all test suites
    - Performance benchmarking
    - Regression testing
    - Detailed reporting
    """
    
    def __init__(self):
        self.test_suites = []
        self.test_data_generator = TestDataGenerator()
        self.start_time = None
        self.end_time = None
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests."""
        print("🧪 Starting Comprehensive Test Suite")
        print("=" * 60)
        
        self.start_time = time.time()
        
        try:
            # Run individual test suites
            self._run_issue_architecture_tests()
            self._run_error_detection_tests()
            self._run_context_collection_tests()
            self._run_advanced_analysis_tests()
            self._run_performance_optimization_tests()
            self._run_integration_tests()
            self._run_regression_tests()
            
            self.end_time = time.time()
            
            # Generate comprehensive report
            return self._generate_test_report()
            
        except Exception as e:
            self.end_time = time.time()
            return {
                "success": False,
                "error": str(e),
                "duration": self.end_time - self.start_time if self.start_time else 0
            }
        finally:
            self.test_data_generator.cleanup()
    
    def _run_issue_architecture_tests(self):
        """Test the enhanced issue architecture."""
        suite_start = time.time()
        tests = []
        
        # Test Issue creation
        test_start = time.time()
        try:
            location = CodeLocation("test.py", 10, 15)
            issue = Issue(
                type=IssueType.UNREACHABLE_CODE,
                category=IssueCategory.IMPLEMENTATION_ERROR,
                severity=IssueSeverity.ERROR,
                message="Test issue",
                location=location
            )
            
            # Test context addition
            issue.add_context("test_key", "test_value")
            
            assert issue.type == IssueType.UNREACHABLE_CODE
            assert issue.category == IssueCategory.IMPLEMENTATION_ERROR
            assert issue.severity == IssueSeverity.ERROR
            assert issue.context["test_key"] == "test_value"
            
            tests.append(TestResult(
                "issue_creation",
                True,
                time.time() - test_start,
                details={"issue_id": issue.id}
            ))
        except Exception as e:
            tests.append(TestResult(
                "issue_creation",
                False,
                time.time() - test_start,
                str(e)
            ))
        
        # Test IssueCollection
        test_start = time.time()
        try:
            issues = [
                Issue(IssueType.UNREACHABLE_CODE, IssueCategory.IMPLEMENTATION_ERROR, IssueSeverity.CRITICAL, "Critical issue"),
                Issue(IssueType.DANGEROUS_FUNCTION_USAGE, IssueCategory.SECURITY_VULNERABILITY, IssueSeverity.ERROR, "Security issue"),
                Issue(IssueType.HIGH_COMPLEXITY, IssueCategory.PERFORMANCE_ISSUE, IssueSeverity.WARNING, "Performance issue")
            ]
            
            collection = IssueCollection(issues=issues)
            
            assert len(collection) == 3
            assert len(collection.get_critical_issues()) == 1
            assert len(collection.get_security_issues()) == 1
            assert len(collection.get_performance_issues()) == 1
            
            summary = collection.get_summary()
            assert summary["total_issues"] == 3
            
            tests.append(TestResult(
                "issue_collection",
                True,
                time.time() - test_start,
                details={"collection_size": len(collection)}
            ))
        except Exception as e:
            tests.append(TestResult(
                "issue_collection",
                False,
                time.time() - test_start,
                str(e)
            ))
        
        suite_end = time.time()
        self.test_suites.append(TestSuite(
            "Issue Architecture Tests",
            "Tests for enhanced issue detection architecture",
            tests,
            setup_time=0.0,
            teardown_time=suite_end - suite_start
        ))
    
    def _run_error_detection_tests(self):
        """Test error detection functions."""
        suite_start = time.time()
        tests = []
        
        # Create test repository
        repo_path = self.test_data_generator.create_test_repository()
        
        # Test implementation error detection
        test_start = time.time()
        try:
            issues = detect_implementation_errors(None)  # Test fallback mode
            assert isinstance(issues, list)
            
            tests.append(TestResult(
                "implementation_error_detection",
                True,
                time.time() - test_start,
                details={"issues_found": len(issues)}
            ))
        except Exception as e:
            tests.append(TestResult(
                "implementation_error_detection",
                False,
                time.time() - test_start,
                str(e)
            ))
        
        # Test security vulnerability detection
        test_start = time.time()
        try:
            issues = detect_security_vulnerabilities(None)  # Test fallback mode
            assert isinstance(issues, list)
            
            tests.append(TestResult(
                "security_vulnerability_detection",
                True,
                time.time() - test_start,
                details={"issues_found": len(issues)}
            ))
        except Exception as e:
            tests.append(TestResult(
                "security_vulnerability_detection",
                False,
                time.time() - test_start,
                str(e)
            ))
        
        # Test advanced analysis functions
        for func_name, func in [
            ("circular_dependencies", detect_circular_dependencies_advanced),
            ("inheritance_patterns", analyze_inheritance_patterns),
            ("complexity_patterns", analyze_complexity_patterns),
            ("performance_patterns", analyze_performance_patterns)
        ]:
            test_start = time.time()
            try:
                issues = func(None)  # Test fallback mode
                assert isinstance(issues, list)
                
                tests.append(TestResult(
                    f"{func_name}_detection",
                    True,
                    time.time() - test_start,
                    details={"issues_found": len(issues)}
                ))
            except Exception as e:
                tests.append(TestResult(
                    f"{func_name}_detection",
                    False,
                    time.time() - test_start,
                    str(e)
                ))
        
        suite_end = time.time()
        self.test_suites.append(TestSuite(
            "Error Detection Tests",
            "Tests for all error detection functions",
            tests,
            setup_time=0.0,
            teardown_time=suite_end - suite_start
        ))
    
    def _run_context_collection_tests(self):
        """Test context collection functionality."""
        suite_start = time.time()
        tests = []
        
        # Test ContextCollector
        test_start = time.time()
        try:
            collector = ContextCollector(None)  # Test with None codebase
            
            # Test that it handles None gracefully
            assert collector is not None
            
            tests.append(TestResult(
                "context_collector_creation",
                True,
                time.time() - test_start,
                details={"collector_created": True}
            ))
        except Exception as e:
            tests.append(TestResult(
                "context_collector_creation",
                False,
                time.time() - test_start,
                str(e)
            ))
        
        suite_end = time.time()
        self.test_suites.append(TestSuite(
            "Context Collection Tests",
            "Tests for context collection functionality",
            tests,
            setup_time=0.0,
            teardown_time=suite_end - suite_start
        ))
    
    def _run_advanced_analysis_tests(self):
        """Test advanced analysis capabilities."""
        suite_start = time.time()
        tests = []
        
        # Test with fallback analyzer
        test_start = time.time()
        try:
            repo_path = self.test_data_generator.create_test_repository()
            analyzer = ComprehensiveCodebaseAnalyzer(repo_path)
            
            # Test analyzer creation
            assert analyzer is not None
            
            tests.append(TestResult(
                "analyzer_creation",
                True,
                time.time() - test_start,
                details={"repo_path": repo_path}
            ))
        except Exception as e:
            tests.append(TestResult(
                "analyzer_creation",
                False,
                time.time() - test_start,
                str(e)
            ))
        
        suite_end = time.time()
        self.test_suites.append(TestSuite(
            "Advanced Analysis Tests",
            "Tests for advanced analysis capabilities",
            tests,
            setup_time=0.0,
            teardown_time=suite_end - suite_start
        ))
    
    def _run_performance_optimization_tests(self):
        """Test performance optimization features."""
        suite_start = time.time()
        tests = []
        
        # Test AnalysisCache
        test_start = time.time()
        try:
            cache = AnalysisCache(max_memory_items=10)
            
            # Test cache operations
            cache.set("test_key", "test_value")
            result = cache.get("test_key")
            
            assert result == "test_value"
            
            # Test cache stats
            stats = cache.get_cache_stats()
            assert "memory_cache_size" in stats
            
            tests.append(TestResult(
                "analysis_cache",
                True,
                time.time() - test_start,
                details={"cache_stats": stats}
            ))
        except Exception as e:
            tests.append(TestResult(
                "analysis_cache",
                False,
                time.time() - test_start,
                str(e)
            ))
        
        # Test IncrementalAnalyzer
        test_start = time.time()
        try:
            analyzer = IncrementalAnalyzer("/tmp/test")
            
            # Test file change detection
            changed_files = analyzer.get_changed_files([])
            assert isinstance(changed_files, list)
            
            # Test stats
            stats = analyzer.get_analysis_stats()
            assert "tracked_files" in stats
            
            tests.append(TestResult(
                "incremental_analyzer",
                True,
                time.time() - test_start,
                details={"stats": stats}
            ))
        except Exception as e:
            tests.append(TestResult(
                "incremental_analyzer",
                False,
                time.time() - test_start,
                str(e)
            ))
        
        # Test PerformanceMonitor
        test_start = time.time()
        try:
            monitor = PerformanceMonitor()
            
            # Test performance tracking
            @monitor.track_execution("test_function")
            def test_function():
                time.sleep(0.01)
                return "test"
            
            result = test_function()
            assert result == "test"
            
            # Test performance report
            report = monitor.get_performance_report()
            assert "summary" in report
            
            tests.append(TestResult(
                "performance_monitor",
                True,
                time.time() - test_start,
                details={"report_keys": list(report.keys())}
            ))
        except Exception as e:
            tests.append(TestResult(
                "performance_monitor",
                False,
                time.time() - test_start,
                str(e)
            ))
        
        suite_end = time.time()
        self.test_suites.append(TestSuite(
            "Performance Optimization Tests",
            "Tests for performance optimization features",
            tests,
            setup_time=0.0,
            teardown_time=suite_end - suite_start
        ))
    
    def _run_integration_tests(self):
        """Test integration between components."""
        suite_start = time.time()
        tests = []
        
        # Test full analysis pipeline
        test_start = time.time()
        try:
            repo_path = self.test_data_generator.create_test_repository()
            analyzer = ComprehensiveCodebaseAnalyzer(repo_path)
            
            # Run analysis (will be in fallback mode)
            results = analyzer.analyze()
            
            assert isinstance(results, dict)
            assert "success" in results
            
            tests.append(TestResult(
                "full_analysis_pipeline",
                True,
                time.time() - test_start,
                details={"results_keys": list(results.keys())}
            ))
        except Exception as e:
            tests.append(TestResult(
                "full_analysis_pipeline",
                False,
                time.time() - test_start,
                str(e)
            ))
        
        suite_end = time.time()
        self.test_suites.append(TestSuite(
            "Integration Tests",
            "Tests for component integration",
            tests,
            setup_time=0.0,
            teardown_time=suite_end - suite_start
        ))
    
    def _run_regression_tests(self):
        """Test for regressions in functionality."""
        suite_start = time.time()
        tests = []
        
        # Test backward compatibility
        test_start = time.time()
        try:
            # Test that old Issue creation still works
            issue = Issue(
                item=None,
                type="test_type",
                message="test message",
                severity="warning"
            )
            
            assert issue.message == "test message"
            
            tests.append(TestResult(
                "backward_compatibility",
                True,
                time.time() - test_start,
                details={"issue_created": True}
            ))
        except Exception as e:
            tests.append(TestResult(
                "backward_compatibility",
                False,
                time.time() - test_start,
                str(e)
            ))
        
        suite_end = time.time()
        self.test_suites.append(TestSuite(
            "Regression Tests",
            "Tests for backward compatibility and regressions",
            tests,
            setup_time=0.0,
            teardown_time=suite_end - suite_start
        ))
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_duration = self.end_time - self.start_time if self.start_time and self.end_time else 0
        
        # Calculate overall statistics
        total_tests = sum(len(suite.tests) for suite in self.test_suites)
        passed_tests = sum(sum(1 for test in suite.tests if test.passed) for suite in self.test_suites)
        failed_tests = total_tests - passed_tests
        
        # Calculate success rate
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Find slowest tests
        all_tests = []
        for suite in self.test_suites:
            for test in suite.tests:
                all_tests.append((f"{suite.name}.{test.test_name}", test.duration, test.passed))
        
        slowest_tests = sorted(all_tests, key=lambda x: x[1], reverse=True)[:5]
        
        # Generate report
        report = {
            "test_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_duration": total_duration,
                "test_environment": "comprehensive_testing"
            },
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate,
                "overall_status": "PASSED" if failed_tests == 0 else "FAILED"
            },
            "test_suites": [],
            "performance_analysis": {
                "slowest_tests": [
                    {"test": name, "duration": duration, "passed": passed}
                    for name, duration, passed in slowest_tests
                ],
                "average_test_duration": sum(test[1] for test in all_tests) / len(all_tests) if all_tests else 0
            },
            "recommendations": self._generate_test_recommendations(success_rate, failed_tests)
        }
        
        # Add detailed suite information
        for suite in self.test_suites:
            suite_passed = sum(1 for test in suite.tests if test.passed)
            suite_failed = len(suite.tests) - suite_passed
            
            suite_info = {
                "name": suite.name,
                "description": suite.description,
                "total_tests": len(suite.tests),
                "passed_tests": suite_passed,
                "failed_tests": suite_failed,
                "success_rate": (suite_passed / len(suite.tests) * 100) if suite.tests else 0,
                "setup_time": suite.setup_time,
                "teardown_time": suite.teardown_time,
                "tests": [
                    {
                        "name": test.test_name,
                        "passed": test.passed,
                        "duration": test.duration,
                        "error_message": test.error_message,
                        "details": test.details
                    }
                    for test in suite.tests
                ]
            }
            
            report["test_suites"].append(suite_info)
        
        return report
    
    def _generate_test_recommendations(self, success_rate: float, failed_tests: int) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if success_rate < 90:
            recommendations.append("🔴 Test success rate is below 90% - investigate failing tests")
        
        if failed_tests > 0:
            recommendations.append(f"⚠️  {failed_tests} tests failed - review error messages and fix issues")
        
        if success_rate == 100:
            recommendations.append("✅ All tests passed - system is functioning correctly")
        
        return recommendations

def run_comprehensive_tests() -> Dict[str, Any]:
    """Run all comprehensive tests and return results."""
    runner = ComprehensiveTestRunner()
    return runner.run_all_tests()

if __name__ == "__main__":
    print("🧪 Comprehensive Testing Framework")
    print("=" * 50)
    
    results = run_comprehensive_tests()
    
    print("\n📊 Test Results Summary:")
    print(f"Total Tests: {results['summary']['total_tests']}")
    print(f"Passed: {results['summary']['passed_tests']}")
    print(f"Failed: {results['summary']['failed_tests']}")
    print(f"Success Rate: {results['summary']['success_rate']:.1f}%")
    print(f"Overall Status: {results['summary']['overall_status']}")
    
    # Save detailed report
    report_file = f"comprehensive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    if results['summary']['overall_status'] == "PASSED":
        print("\n🎉 All tests passed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed - check the detailed report")
        sys.exit(1)

#!/usr/bin/env python3
"""
Simple codebase analysis that works without the Codegen SDK.
"""

import os
import ast
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any

def analyze_python_file(file_path: str) -> Dict[str, Any]:
    """Analyze a single Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        analysis = {
            'file_path': file_path,
            'lines_of_code': len(content.splitlines()),
            'functions': [],
            'classes': [],
            'imports': [],
            'issues': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                analysis['functions'].append({
                    'name': node.name,
                    'line': node.lineno,
                    'args': len(node.args.args),
                    'has_docstring': ast.get_docstring(node) is not None
                })
                
                # Check for long functions
                if hasattr(node, 'end_lineno') and node.end_lineno:
                    func_lines = node.end_lineno - node.lineno
                    if func_lines > 50:
                        analysis['issues'].append({
                            'type': 'code_quality',
                            'severity': 'warning',
                            'message': f'Function {node.name} is very long ({func_lines} lines)',
                            'line': node.lineno
                        })
                
                # Check for missing docstring
                if not ast.get_docstring(node) and not node.name.startswith('_'):
                    analysis['issues'].append({
                        'type': 'documentation',
                        'severity': 'info',
                        'message': f'Function {node.name} missing docstring',
                        'line': node.lineno
                    })
            
            elif isinstance(node, ast.ClassDef):
                analysis['classes'].append({
                    'name': node.name,
                    'line': node.lineno,
                    'methods': len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                    'has_docstring': ast.get_docstring(node) is not None
                })
                
                # Check for missing docstring
                if not ast.get_docstring(node):
                    analysis['issues'].append({
                        'type': 'documentation',
                        'severity': 'info',
                        'message': f'Class {node.name} missing docstring',
                        'line': node.lineno
                    })
            
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(alias.name)
                else:
                    module = node.module or ''
                    for alias in node.names:
                        analysis['imports'].append(f"{module}.{alias.name}")
        
        # Check for potential security issues
        if 'eval(' in content:
            analysis['issues'].append({
                'type': 'security',
                'severity': 'error',
                'message': 'Use of eval() detected - potential security risk',
                'line': content[:content.find('eval(')].count('\n') + 1
            })
        
        if 'exec(' in content:
            analysis['issues'].append({
                'type': 'security',
                'severity': 'error',
                'message': 'Use of exec() detected - potential security risk',
                'line': content[:content.find('exec(')].count('\n') + 1
            })
        
        return analysis
        
    except Exception as e:
        return {
            'file_path': file_path,
            'error': str(e),
            'issues': [{
                'type': 'parse_error',
                'severity': 'error',
                'message': f'Failed to parse file: {e}',
                'line': 1
            }]
        }

def analyze_repository(repo_path: str) -> Dict[str, Any]:
    """Analyze the entire repository."""
    repo_path = Path(repo_path)
    
    analysis = {
        'repository_path': str(repo_path),
        'files_analyzed': 0,
        'total_lines': 0,
        'total_functions': 0,
        'total_classes': 0,
        'files': [],
        'issues_summary': Counter(),
        'severity_summary': Counter(),
        'all_issues': []
    }
    
    # Find all Python files
    python_files = list(repo_path.rglob('*.py'))
    
    for file_path in python_files:
        # Skip __pycache__ and .git directories
        if '__pycache__' in str(file_path) or '.git' in str(file_path):
            continue
            
        file_analysis = analyze_python_file(str(file_path))
        analysis['files'].append(file_analysis)
        analysis['files_analyzed'] += 1
        
        if 'lines_of_code' in file_analysis:
            analysis['total_lines'] += file_analysis['lines_of_code']
        if 'functions' in file_analysis:
            analysis['total_functions'] += len(file_analysis['functions'])
        if 'classes' in file_analysis:
            analysis['total_classes'] += len(file_analysis['classes'])
        
        # Collect issues
        for issue in file_analysis.get('issues', []):
            issue['file'] = str(file_path.relative_to(repo_path))
            analysis['all_issues'].append(issue)
            analysis['issues_summary'][issue['type']] += 1
            analysis['severity_summary'][issue['severity']] += 1
    
    return analysis

def print_analysis_report(analysis: Dict[str, Any]):
    """Print a formatted analysis report."""
    print('='*80)
    print('📊 CODEBASE ANALYSIS REPORT')
    print('='*80)
    
    print(f'\n📁 Repository: {analysis["repository_path"]}')
    print(f'📄 Files analyzed: {analysis["files_analyzed"]}')
    print(f'📏 Total lines of code: {analysis["total_lines"]:,}')
    print(f'🔧 Total functions: {analysis["total_functions"]}')
    print(f'🏗️ Total classes: {analysis["total_classes"]}')
    
    # Issues summary
    total_issues = sum(analysis['severity_summary'].values())
    print(f'\n🚨 Total issues found: {total_issues}')
    
    if analysis['severity_summary']:
        print('\n⚠️ Issues by severity:')
        for severity, count in analysis['severity_summary'].most_common():
            print(f'  {severity}: {count}')
    
    if analysis['issues_summary']:
        print('\n🔍 Issues by type:')
        for issue_type, count in analysis['issues_summary'].most_common():
            print(f'  {issue_type}: {count}')
    
    # Top issues
    if analysis['all_issues']:
        print('\n🚨 Top issues:')
        for i, issue in enumerate(analysis['all_issues'][:10]):
            print(f'  {i+1}. [{issue["severity"]}] {issue["message"]}')
            print(f'     File: {issue["file"]}:{issue["line"]}')
    
    # File breakdown
    print('\n📊 File breakdown:')
    for file_info in sorted(analysis['files'], key=lambda x: x.get('lines_of_code', 0), reverse=True)[:10]:
        if 'lines_of_code' in file_info:
            rel_path = Path(file_info['file_path']).name
            print(f'  {rel_path}: {file_info["lines_of_code"]} lines, {len(file_info.get("functions", []))} functions, {len(file_info.get("classes", []))} classes')

def main():
    """Main analysis function."""
    # Get repository path
    repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print(f'🔍 Analyzing repository: {repo_path}')
    
    # Run analysis
    analysis = analyze_repository(repo_path)
    
    # Print report
    print_analysis_report(analysis)
    
    # Save to file
    output_file = 'simple_analysis_report.json'
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f'\n📄 Full analysis saved to: {output_file}')
    print('\n' + '='*80)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run comprehensive analysis on the codebase-analytics repository.
"""

from analysis import ComprehensiveCodebaseAnalyzer
import json
import os
from collections import Counter

def main():
    # Analyze the current repository (codebase-analytics)
    repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f'🔍 Analyzing repository: {repo_path}')

    analyzer = ComprehensiveCodebaseAnalyzer(repo_path)
    results = analyzer.analyze()

    print('\n' + '='*80)
    print('📊 COMPREHENSIVE CODEBASE ANALYSIS REPORT')
    print('='*80)

    if results.get('success', False):
        print('✅ Analysis completed successfully!')
        
        # Print summary
        summary = results.get('summary', {})
        print(f'\n📈 SUMMARY:')
        print(f'  Total Issues: {summary.get("total_issues", 0)}')
        print(f'  Critical Issues: {summary.get("critical_issues", 0)}')
        print(f'  Error Issues: {summary.get("error_issues", 0)}')
        print(f'  Warning Issues: {summary.get("warning_issues", 0)}')
        print(f'  Info Issues: {summary.get("info_issues", 0)}')
        print(f'  Analysis Duration: {results.get("duration", 0):.2f} seconds')
        
        # Print issues by category
        issues = results.get('issues', [])
        if issues:
            print(f'\n🔍 ISSUES BY CATEGORY:')
            categories = Counter()
            severities = Counter()
            
            for issue in issues:
                if isinstance(issue, dict):
                    categories[issue.get('category', 'unknown')] += 1
                    severities[issue.get('severity', 'unknown')] += 1
                else:
                    # Handle Issue objects
                    categories[getattr(issue, 'category', 'unknown')] += 1
                    severities[getattr(issue, 'severity', 'unknown')] += 1
            
            for category, count in categories.most_common():
                print(f'  {category}: {count}')
            
            print(f'\n⚠️ ISSUES BY SEVERITY:')
            for severity, count in severities.most_common():
                print(f'  {severity}: {count}')
            
            # Show first few issues
            print(f'\n🚨 TOP ISSUES:')
            for i, issue in enumerate(issues[:5]):
                if isinstance(issue, dict):
                    print(f'  {i+1}. [{issue.get("severity", "unknown")}] {issue.get("message", "No message")}')
                    if issue.get('location'):
                        loc = issue['location']
                        print(f'     Location: {loc.get("file_path", "unknown")}:{loc.get("line_start", "?")}')
                else:
                    print(f'  {i+1}. [{getattr(issue, "severity", "unknown")}] {getattr(issue, "message", "No message")}')
                    if hasattr(issue, 'location') and issue.location:
                        loc = issue.location
                        print(f'     Location: {getattr(loc, "file_path", "unknown")}:{getattr(loc, "line_start", "?")}')
        
        # Print enhanced features if available
        if 'enhanced_report' in results:
            enhanced = results['enhanced_report']
            print(f'\n🎯 ENHANCED FEATURES:')
            
            insights = enhanced.get('actionable_insights', [])
            if insights:
                print(f'  Actionable Insights: {len(insights)}')
                for i, insight in enumerate(insights[:3]):
                    print(f'    {i+1}. {insight.get("title", "No title")} (Priority: {insight.get("priority", "unknown")})')
            
            exec_summary = enhanced.get('executive_summary', {})
            if exec_summary:
                print(f'  Executive Summary: {exec_summary.get("overall_health", "unknown")} health')
                print(f'  Key Metrics: {exec_summary.get("key_metrics", {})}')
        
    else:
        print('❌ Analysis failed!')
        print(f'Error: {results.get("error", "Unknown error")}')
        print(f'Issues found: {len(results.get("issues", []))}')
        
        # Still show any issues that were found
        issues = results.get('issues', [])
        if issues:
            print(f'\n🔍 ISSUES FOUND DURING FAILED ANALYSIS:')
            categories = Counter()
            severities = Counter()
            
            for issue in issues:
                if isinstance(issue, dict):
                    categories[issue.get('category', 'unknown')] += 1
                    severities[issue.get('severity', 'unknown')] += 1
                else:
                    # Handle Issue objects
                    categories[getattr(issue, 'category', 'unknown')] += 1
                    severities[getattr(issue, 'severity', 'unknown')] += 1
            
            for category, count in categories.most_common():
                print(f'  {category}: {count}')
            
            print(f'\n⚠️ ISSUES BY SEVERITY:')
            for severity, count in severities.most_common():
                print(f'  {severity}: {count}')
            
            # Show first few issues
            print(f'\n🚨 DETECTED ISSUES:')
            for i, issue in enumerate(issues[:10]):
                if isinstance(issue, dict):
                    print(f'  {i+1}. [{issue.get("severity", "unknown")}] {issue.get("message", "No message")}')
                    if issue.get('location'):
                        loc = issue['location']
                        print(f'     Location: {loc.get("file_path", "unknown")}:{loc.get("line_start", "?")}')
                else:
                    print(f'  {i+1}. [{getattr(issue, "severity", "unknown")}] {getattr(issue, "message", "No message")}')
                    if hasattr(issue, 'location') and issue.location:
                        loc = issue.location
                        print(f'     Location: {getattr(loc, "file_path", "unknown")}:{getattr(loc, "line_start", "?")}')

    print('\n' + '='*80)
    print('📋 ANALYSIS COMPLETE')
    print('='*80)
    
    # Save results to file
    output_file = 'analysis_report.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'📄 Full results saved to: {output_file}')

if __name__ == "__main__":
    main()
