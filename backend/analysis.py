"""
Comprehensive Codebase Analysis Engine
Provides deep analysis including error detection, function contexts, dead code analysis, and metrics
"""
import networkx as nx
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
import re
import ast
import os
class IssueSeverity(Enum):
    CRITICAL = "critical"
    MAJOR = "major" 
    MINOR = "minor"
    INFO = "info"
class IssueType(Enum):
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
    UNUSED_PARAMETER = "unused_parameter"
    
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
@dataclass
class Issue:
    type: IssueType
    severity: IssueSeverity
    message: str
    filepath: str
    line_number: int
    column_number: int = 0
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
@dataclass
class FunctionContext:
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
def get_function_context(function) -> dict:
    """Get the implementation, dependencies, and usages of a function."""
    context = {
        "implementation": {"source": function.source, "filepath": function.filepath},
        "dependencies": [],
        "usages": [],
    }
    # Add dependencies
    for dep in function.dependencies:
        if hasattr(dep, 'source') and hasattr(dep, 'filepath'):
            context["dependencies"].append({"source": dep.source, "filepath": dep.filepath})
    # Add usages
    for usage in function.usages:
        if hasattr(usage, 'usage_symbol'):
            context["usages"].append({
                "source": usage.usage_symbol.source,
                "filepath": usage.usage_symbol.filepath,
            })
    return context
def get_max_call_chain(function):
    """Calculate maximum call chain for a function"""
    G = nx.DiGraph()
    def build_graph(func, depth=0):
        if depth > 10:  # Prevent infinite recursion
            return
        for call in func.function_calls:
            if hasattr(call, 'function_definition') and call.function_definition:
                called_func = call.function_definition
                G.add_edge(func.name, called_func.name)
                build_graph(called_func, depth + 1)
    build_graph(function)
    try:
        return nx.dag_longest_path(G)
    except:
        return [function.name]
def analyze_codebase(codebase):
    """Main analysis function that returns comprehensive codebase analysis"""
    analyzer = ComprehensiveCodebaseAnalyzer(codebase)
    return analyzer.analyze()
class ComprehensiveCodebaseAnalyzer:
    def __init__(self, codebase):
        self.codebase = codebase
        self.issues = []
        self.function_contexts = {}
        self.dead_code_items = []
        self.call_graph = nx.DiGraph()
        self.operators = set()
        self.operands = set()
        
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive codebase analysis"""
        print("🔍 Starting comprehensive analysis...")
        
        # Build call graph
        self._build_call_graph()
        
        # Analyze functions
        self._analyze_functions()
        
        # Detect issues
        self._detect_all_issues()
        
        # Analyze dead code
        self._analyze_dead_code()
        
        # Find most important functions
        important_functions = self._find_most_important_functions()
        
        # Build repository structure
        repo_structure = self._build_repository_structure()
        
        # Generate summary
        summary = self._generate_summary()
        
        return {
            "summary": summary,
            "issues_by_severity": self._group_issues_by_severity(),
            "function_contexts": {name: self._serialize_function_context(ctx) 
                               for name, ctx in self.function_contexts.items()},
            "dead_code_analysis": self._analyze_dead_code_detailed(),
            "most_important_functions": important_functions,
            "repository_structure": self._serialize_directory_node(repo_structure),
            "call_graph_metrics": self._calculate_call_graph_metrics(),
            "dependency_metrics": self._calculate_dependency_metrics(),
            "entry_points": self._identify_entry_points(),
            "halstead_metrics": self._calculate_halstead_metrics(),
            "operators_analysis": self._analyze_operators()
        }
    
    def _find_most_important_functions(self):
        """Find most important functions using the exact algorithms requested"""
        if not self.codebase.functions:
            return {}
        
        # Find function that makes the most calls
        most_calls = max(self.codebase.functions, key=lambda f: len(f.function_calls))
        
        # Find the most called function
        most_called = max(self.codebase.functions, key=lambda f: len(f.call_sites))
        
        # Find class with most inheritance
        deepest_class = None
        if self.codebase.classes:
            deepest_class = max(self.codebase.classes, key=lambda x: len(x.superclasses))
        
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
                "chain_depth": len(deepest_class.superclasses) if deepest_class else 0,
                "chain": [s.name for s in deepest_class.superclasses] if deepest_class else []
            }
        }
    
    # Additional methods would be added here...
    def _build_call_graph(self):
        """Build the function call graph"""
        for function in self.codebase.functions:
            self.call_graph.add_node(function.name)
            for call in function.function_calls:
                if hasattr(call, 'function_definition') and call.function_definition:
                    self.call_graph.add_edge(function.name, call.function_definition.name)
    
    def _analyze_functions(self):
        """Analyze all functions and build contexts"""
        for function in self.codebase.functions:
            context = self._get_function_context(function)
            self.function_contexts[function.name] = context
    
    def _get_function_context(self, function) -> FunctionContext:
        """Get comprehensive context for a function"""
        # Implementation details...
        return FunctionContext(
            name=function.name,
            filepath=function.filepath,
            source=function.source,
            parameters=[],
            dependencies=[],
            usages=[],
            function_calls=[],
            called_by=[],
            max_call_chain=[],
            issues=[],
            is_entry_point=False,
            is_dead_code=False
        )
    
    # Mock implementations for other methods
    def _detect_all_issues(self): pass
    def _analyze_dead_code(self): pass
    def _build_repository_structure(self): return {}
    def _generate_summary(self): return {}
    def _group_issues_by_severity(self): return {}
    def _serialize_function_context(self, ctx): return ctx
    def _serialize_directory_node(self, node): return node
    def _calculate_call_graph_metrics(self): return {}
    def _calculate_dependency_metrics(self): return {}
    def _identify_entry_points(self): return []
    def _calculate_halstead_metrics(self): return {}
    def _analyze_operators(self): return {}