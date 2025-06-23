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
from dataclasses import dataclass, asdict
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

# Issue and severity definitions
class IssueSeverity(Enum):
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class IssueType(Enum):
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
    
    # Formatting & Style
    INCONSISTENT_NAMING = "inconsistent_naming"
    MISSING_DOCUMENTATION = "missing_documentation"
    INCONSISTENT_INDENTATION = "inconsistent_indentation"
    IMPORT_ORGANIZATION = "import_organization"
    
    # Runtime Risks
    DIVISION_BY_ZERO = "division_by_zero"
    ARRAY_BOUNDS = "array_bounds"
    INFINITE_LOOP = "infinite_loop"
    STACK_OVERFLOW = "stack_overflow"
    CONCURRENCY_ISSUE = "concurrency_issue"
    
    # General
    IMPLEMENTATION_ERROR = "implementation_error"

@dataclass
class Issue:
    """Represents an issue found during codebase analysis."""
    item: Any
    type: str
    message: str
    severity: str = IssueSeverity.WARNING.value
    location: Optional[str] = None
    suggestion: Optional[str] = None
    filepath: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.location is None:
            self.location = self._get_location()
    
    def _get_location(self) -> str:
        """Get a string representation of the item's location."""
        if hasattr(self.item, 'file') and hasattr(self.item.file, 'path'):
            file_path = self.item.file.path
            if hasattr(self.item, 'line'):
                return f"{file_path}:{self.item.line}"
            return file_path
        elif hasattr(self.item, 'filepath'):
            return self.item.filepath
        elif hasattr(self.item, 'path'):
            return self.item.path
        else:
            return "Unknown location"
    
    def __str__(self) -> str:
        """Return a string representation of the issue."""
        base = f"[{self.severity.upper()}] {self.type}: {self.message} ({self.location})"
        if self.suggestion:
            base += f" - Suggestion: {self.suggestion}"
        return base

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
        """Analyze dead code using SDK's usage information."""
        if not CODEGEN_SDK_AVAILABLE or not self.codebase:
            return
        
        try:
            # Analyze unused functions
            for function in self.codebase.functions:
                if len(function.call_sites) == 0 and not self._is_entry_point(function):
                    self.issues.append(Issue(
                        function,
                        IssueType.UNUSED_FUNCTION.value,
                        f"Function '{function.name}' is never called",
                        IssueSeverity.WARNING.value,
                        suggestion="Consider removing this function if it's truly unused"
                    ))
                    self.dead_code_items.append(function)
            
            # Analyze unused classes
            for cls in self.codebase.classes:
                if len(cls.usages) == 0:
                    self.issues.append(Issue(
                        cls,
                        IssueType.UNUSED_CLASS.value,
                        f"Class '{cls.name}' is never used",
                        IssueSeverity.WARNING.value,
                        suggestion="Consider removing this class if it's truly unused"
                    ))
                    self.dead_code_items.append(cls)
            
            # Analyze unused imports
            for import_stmt in self.codebase.imports:
                if hasattr(import_stmt, 'imported_symbol') and import_stmt.imported_symbol:
                    if len(import_stmt.imported_symbol.usages) == 0:
                        self.issues.append(Issue(
                            import_stmt,
                            IssueType.UNUSED_IMPORT.value,
                            f"Import '{import_stmt.name}' is never used",
                            IssueSeverity.INFO.value,
                            suggestion="Remove this unused import"
                        ))
                        self.dead_code_items.append(import_stmt)
        
        except Exception as e:
            print(f"Error in dead code analysis: {e}")
    
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
