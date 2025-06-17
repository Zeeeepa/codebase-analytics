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

def calculate_doi(cls: Class) -> int:
    """Calculate the depth of inheritance for a given class."""
    return len(cls.superclasses)

def get_operators_and_operands(function: Function) -> Tuple[List[str], List[str]]:
    """Extract operators and operands from function code for Halstead metrics."""
    if not hasattr(function, "code_block") or not function.code_block:
        return [], []
    
    code = function.code_block.source
    
    # Common operators
    operators = [
        "+", "-", "*", "/", "//", "%", "**",
        "=", "+=", "-=", "*=", "/=", "//=", "%=", "**=",
        "==", "!=", "<", ">", "<=", ">=",
        "and", "or", "not", "in", "is",
        "&", "|", "^", "~", "<<", ">>",
        "(", ")", "[", "]", "{", "}",
        ",", ":", ";", ".", "->",
        "if", "else", "elif", "for", "while", "try", "except", "finally",
        "def", "class", "return", "yield", "import", "from", "as",
        "lambda", "with", "assert", "del", "global", "nonlocal",
        "pass", "break", "continue", "raise"
    ]
    
    found_operators = []
    found_operands = []
    
    # Simple tokenization - this could be improved with proper AST parsing
    tokens = re.findall(r'\b\w+\b|[^\w\s]', code)
    
    for token in tokens:
        if token in operators:
            found_operators.append(token)
        elif token.isidentifier() and not token.isdigit():
            found_operands.append(token)
        elif token.isdigit() or (token.startswith('"') and token.endswith('"')) or (token.startswith("'") and token.endswith("'")):
            found_operands.append(token)
    
    return found_operators, found_operands

def calculate_halstead_volume(operators: List[str], operands: List[str]) -> Tuple[float, int, int, int, int]:
    """Calculate Halstead volume and related metrics."""
    if not operators and not operands:
        return 0.0, 0, 0, 0, 0
    
    # Count unique and total operators/operands
    n1 = len(set(operators))  # Unique operators
    n2 = len(set(operands))   # Unique operands
    N1 = len(operators)       # Total operators
    N2 = len(operands)        # Total operands
    
    # Calculate volume
    N = N1 + N2  # Program length
    n = n1 + n2  # Vocabulary size
    
    if n <= 0:
        return 0.0, N1, N2, n1, n2
    
    volume = N * math.log2(n) if n > 0 else 0
    return volume, N1, N2, n1, n2

def count_lines(source: str) -> Tuple[int, int, int, int]:
    """Count different types of lines in source code."""
    if not source:
        return 0, 0, 0, 0
    
    lines = source.split('\n')
    total_lines = len(lines)
    
    logical_lines = 0
    source_lines = 0
    comment_lines = 0
    
    in_multiline_comment = False
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            continue
            
        # Handle multiline comments (Python docstrings)
        if '"""' in stripped or "'''" in stripped:
            if in_multiline_comment:
                in_multiline_comment = False
                comment_lines += 1
                continue
            else:
                in_multiline_comment = True
                comment_lines += 1
                continue
        
        if in_multiline_comment:
            comment_lines += 1
            continue
            
        # Single line comments
        if stripped.startswith('#'):
            comment_lines += 1
            continue
            
        # Source lines (non-empty, non-comment)
        source_lines += 1
        
        # Logical lines (statements)
        # Simple heuristic: count semicolons and certain keywords
        logical_lines += max(1, stripped.count(';') + 1) if stripped.endswith(':') else 1
    
    return total_lines, logical_lines, source_lines, comment_lines

def calculate_maintainability_index(volume: float, complexity: int, loc: int) -> float:
    """Calculate maintainability index."""
    if loc == 0:
        return 100.0
    
    # Maintainability Index formula
    # MI = 171 - 5.2 * ln(Halstead Volume) - 0.23 * (Cyclomatic Complexity) - 16.2 * ln(Lines of Code)
    try:
        mi = 171 - 5.2 * math.log(volume) - 0.23 * complexity - 16.2 * math.log(loc)
        return max(0, min(100, mi))  # Clamp between 0 and 100
    except (ValueError, ZeroDivisionError):
        return 50.0  # Default middle value

def get_maintainability_rank(mi_score: float) -> str:
    """Get maintainability rank based on MI score."""
    if mi_score >= 85:
        return "Excellent"
    elif mi_score >= 70:
        return "Good"
    elif mi_score >= 50:
        return "Moderate"
    elif mi_score >= 25:
        return "Poor"
    else:
        return "Critical"

# ============================================================================
# SECTION 3: DEPENDENCY ANALYSIS FUNCTIONS
# ============================================================================

def analyze_dependencies_comprehensive(codebase: Codebase) -> DependencyAnalysis:
    """Perform comprehensive dependency analysis."""
    analysis = DependencyAnalysis()
    
    # Build dependency graph
    dependency_graph = {}
    external_deps = set()
    internal_deps = set()
    
    for file in codebase.files:
        file_deps = []
        if hasattr(file, 'imports') and file.imports:
            for imp in file.imports:
                module_name = getattr(imp, 'module', None) or getattr(imp, 'name', 'unknown')
                file_deps.append(module_name)
                
                # Classify as external or internal
                if any(module_name.startswith(prefix) for prefix in ['os', 'sys', 'json', 'datetime', 'typing', 'collections']):
                    external_deps.add(module_name)
                else:
                    internal_deps.add(module_name)
        
        dependency_graph[file.filepath] = file_deps
    
    analysis.dependency_graph = dependency_graph
    analysis.external_dependencies = list(external_deps)
    analysis.internal_dependencies = list(internal_deps)
    analysis.total_dependencies = len(external_deps) + len(internal_deps)
    
    # Detect circular dependencies
    analysis.circular_dependencies = detect_circular_dependencies(dependency_graph)
    
    # Calculate dependency depth
    analysis.dependency_depth = calculate_dependency_depth(dependency_graph)
    
    # Identify critical dependencies (most used)
    dep_usage = Counter()
    for deps in dependency_graph.values():
        dep_usage.update(deps)
    analysis.critical_dependencies = [dep for dep, count in dep_usage.most_common(10)]
    
    return analysis

def detect_circular_dependencies(dependency_graph: Dict[str, List[str]]) -> List[List[str]]:
    """Detect circular dependencies in the dependency graph."""
    def dfs(node, visited, rec_stack, path):
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        for neighbor in dependency_graph.get(node, []):
            if neighbor in rec_stack:
                # Found a cycle
                cycle_start = path.index(neighbor)
                return path[cycle_start:] + [neighbor]
            elif neighbor not in visited:
                cycle = dfs(neighbor, visited, rec_stack, path)
                if cycle:
                    return cycle
        
        rec_stack.remove(node)
        path.pop()
        return None
    
    visited = set()
    cycles = []
    
    for node in dependency_graph:
        if node not in visited:
            cycle = dfs(node, visited, set(), [])
            if cycle:
                cycles.append(cycle)
    
    return cycles

def calculate_dependency_depth(dependency_graph: Dict[str, List[str]]) -> int:
    """Calculate the maximum dependency depth."""
    def get_depth(node, visited):
        if node in visited:
            return 0  # Avoid infinite recursion
        
        visited.add(node)
        max_depth = 0
        
        for dep in dependency_graph.get(node, []):
            depth = get_depth(dep, visited.copy())
            max_depth = max(max_depth, depth)
        
        return max_depth + 1
    
    max_depth = 0
    for node in dependency_graph:
        depth = get_depth(node, set())
        max_depth = max(max_depth, depth)
    
    return max_depth

# ============================================================================
# SECTION 4: CALL GRAPH ANALYSIS FUNCTIONS
# ============================================================================

def analyze_call_graph(codebase: Codebase) -> CallGraphAnalysis:
    """Analyze function call relationships."""
    analysis = CallGraphAnalysis()
    
    # Build call graph
    call_graph = {}
    all_functions = set()
    
    for func in codebase.functions:
        all_functions.add(func.name)
        calls = []
        
        if hasattr(func, 'function_calls') and func.function_calls:
            calls = [call.name for call in func.function_calls if hasattr(call, 'name')]
        elif hasattr(func, 'code_block') and func.code_block:
            # Extract function calls from code
            calls = extract_function_calls_from_code(func.code_block.source, all_functions)
        
        call_graph[func.name] = calls
    
    analysis.call_graph = call_graph
    analysis.total_call_relationships = sum(len(calls) for calls in call_graph.values())
    
    # Find entry points (functions not called by others)
    called_functions = set()
    for calls in call_graph.values():
        called_functions.update(calls)
    
    analysis.entry_points = [func for func in all_functions if func not in called_functions]
    
    # Find leaf functions (functions that don't call others)
    analysis.leaf_functions = [func for func, calls in call_graph.items() if not calls]
    
    # Find most connected functions
    connection_counts = {}
    for func in all_functions:
        incoming = sum(1 for calls in call_graph.values() if func in calls)
        outgoing = len(call_graph.get(func, []))
        connection_counts[func] = incoming + outgoing
    
    analysis.most_connected_functions = sorted(
        connection_counts.items(), key=lambda x: x[1], reverse=True
    )[:10]
    
    # Calculate call depth
    analysis.call_depth = calculate_call_depth(call_graph)
    
    # Find interesting call chains
    analysis.call_chains = find_call_chains(call_graph, max_chains=5, max_depth=5)
    
    return analysis

def extract_function_calls_from_code(code: str, known_functions: Set[str]) -> List[str]:
    """Extract function calls from code using simple pattern matching."""
    calls = []
    
    # Simple regex to find function calls
    pattern = r'(\w+)\s*\('
    matches = re.findall(pattern, code)
    
    for match in matches:
        if match in known_functions:
            calls.append(match)
    
    return calls

def calculate_call_depth(call_graph: Dict[str, List[str]]) -> int:
    """Calculate the maximum call depth."""
    def get_depth(func, visited):
        if func in visited:
            return 0  # Avoid infinite recursion
        
        visited.add(func)
        max_depth = 0
        
        for called_func in call_graph.get(func, []):
            depth = get_depth(called_func, visited.copy())
            max_depth = max(max_depth, depth)
        
        return max_depth + 1
    
    max_depth = 0
    for func in call_graph:
        depth = get_depth(func, set())
        max_depth = max(max_depth, depth)
    
    return max_depth

def find_call_chains(call_graph: Dict[str, List[str]], max_chains: int = 5, max_depth: int = 5) -> List[List[str]]:
    """Find interesting call chains in the call graph."""
    chains = []
    
    def dfs_chains(current_func, path, depth):
        if depth >= max_depth or len(chains) >= max_chains:
            return
        
        path.append(current_func)
        
        called_functions = call_graph.get(current_func, [])
        if not called_functions:
            # End of chain
            if len(path) > 2:  # Only include chains with at least 3 functions
                chains.append(path.copy())
        else:
            for called_func in called_functions:
                if called_func not in path:  # Avoid cycles
                    dfs_chains(called_func, path, depth + 1)
        
        path.pop()
    
    # Start from entry points
    entry_points = [func for func in call_graph if not any(func in calls for calls in call_graph.values())]
    
    for entry_point in entry_points[:3]:  # Limit starting points
        dfs_chains(entry_point, [], 0)
    
    return chains

# ============================================================================
# SECTION 5: CODE QUALITY ANALYSIS FUNCTIONS
# ============================================================================

def analyze_code_quality(codebase: Codebase) -> CodeQualityMetrics:
    """Perform comprehensive code quality analysis."""
    metrics = CodeQualityMetrics()
    
    total_lines = 0
    documented_functions = 0
    total_functions = len(codebase.functions)
    code_blocks = []
    
    # Analyze each file
    for file in codebase.files:
        if file.source:
            lines = file.source.split('\n')
            total_lines += len(lines)
            code_blocks.append(file.source)
    
    # Calculate documentation coverage
    for func in codebase.functions:
        if hasattr(func, 'code_block') and func.code_block:
            code = func.code_block.source
            if '"""' in code or "'''" in code or code.strip().startswith('#'):
                documented_functions += 1
    
    metrics.documentation_coverage = (documented_functions / total_functions * 100) if total_functions > 0 else 0
    
    # Estimate code duplication
    metrics.code_duplication_percentage = estimate_code_duplication(code_blocks)
    
    # Analyze naming consistency
    metrics.naming_consistency_score = analyze_naming_consistency(codebase)
    
    # Detect code smells
    metrics.code_smells = detect_code_smells(codebase)
    
    # Identify refactoring opportunities
    metrics.refactoring_opportunities = identify_refactoring_opportunities(codebase)
    
    # Estimate technical debt
    metrics.technical_debt_ratio = estimate_technical_debt(codebase)
    
    return metrics

def estimate_code_duplication(code_blocks: List[str]) -> float:
    """Estimate code duplication percentage."""
    if len(code_blocks) < 2:
        return 0.0
    
    # Simple approach: look for similar lines across files
    all_lines = []
    for code in code_blocks:
        lines = [line.strip() for line in code.split('\n') if line.strip() and not line.strip().startswith('#')]
        all_lines.extend(lines)
    
    if not all_lines:
        return 0.0
    
    line_counts = Counter(all_lines)
    duplicated_lines = sum(count - 1 for count in line_counts.values() if count > 1)
    
    return (duplicated_lines / len(all_lines) * 100) if all_lines else 0.0

def analyze_naming_consistency(codebase: Codebase) -> float:
    """Analyze naming consistency across the codebase."""
    function_names = [func.name for func in codebase.functions]
    class_names = [cls.name for cls in codebase.classes]
    
    # Check naming conventions
    snake_case_functions = sum(1 for name in function_names if re.match(r'^[a-z_][a-z0-9_]*$', name))
    pascal_case_classes = sum(1 for name in class_names if re.match(r'^[A-Z][a-zA-Z0-9]*$', name))
    
    total_symbols = len(function_names) + len(class_names)
    consistent_names = snake_case_functions + pascal_case_classes
    
    return (consistent_names / total_symbols * 100) if total_symbols > 0 else 100.0

def detect_code_smells(codebase: Codebase) -> List[Dict[str, Any]]:
    """Detect common code smells."""
    smells = []
    
    for func in codebase.functions:
        if hasattr(func, 'code_block') and func.code_block:
            code = func.code_block.source
            lines = code.split('\n')
            
            # Long method
            if len(lines) > 50:
                smells.append({
                    'type': 'long_method',
                    'function': func.name,
                    'severity': 'medium',
                    'description': f'Function {func.name} has {len(lines)} lines (>50)',
                    'suggestion': 'Consider breaking this function into smaller functions'
                })
            
            # Too many parameters
            if hasattr(func, 'parameters') and len(func.parameters) > 5:
                smells.append({
                    'type': 'too_many_parameters',
                    'function': func.name,
                    'severity': 'medium',
                    'description': f'Function {func.name} has {len(func.parameters)} parameters (>5)',
                    'suggestion': 'Consider using a configuration object or breaking the function'
                })
            
            # Nested complexity
            nesting_level = max(line.count('    ') for line in lines if line.strip())
            if nesting_level > 4:
                smells.append({
                    'type': 'deep_nesting',
                    'function': func.name,
                    'severity': 'high',
                    'description': f'Function {func.name} has deep nesting (level {nesting_level})',
                    'suggestion': 'Consider extracting nested logic into separate functions'
                })
    
    return smells

def identify_refactoring_opportunities(codebase: Codebase) -> List[Dict[str, Any]]:
    """Identify refactoring opportunities."""
    opportunities = []
    
    # Find similar functions that could be consolidated
    function_signatures = {}
    for func in codebase.functions:
        if hasattr(func, 'parameters'):
            param_types = tuple(getattr(p, 'type', 'unknown') for p in func.parameters)
            signature = (len(func.parameters), param_types)
            
            if signature not in function_signatures:
                function_signatures[signature] = []
            function_signatures[signature].append(func.name)
    
    for signature, functions in function_signatures.items():
        if len(functions) > 1:
            opportunities.append({
                'type': 'similar_functions',
                'functions': functions,
                'description': f'Functions with similar signatures: {", ".join(functions)}',
                'suggestion': 'Consider consolidating these functions or creating a common base'
            })
    
    return opportunities

def estimate_technical_debt(codebase: Codebase) -> float:
    """Estimate technical debt ratio."""
    debt_indicators = 0
    total_indicators = 0
    
    for file in codebase.files:
        if file.source:
            code = file.source.lower()
            total_indicators += 1
            
            # Look for debt indicators
            if any(indicator in code for indicator in ['todo', 'fixme', 'hack', 'workaround']):
                debt_indicators += 1
    
    return (debt_indicators / total_indicators * 100) if total_indicators > 0 else 0.0

# ============================================================================
# SECTION 6: ARCHITECTURAL ANALYSIS FUNCTIONS
# ============================================================================

def analyze_architecture(codebase: Codebase) -> ArchitecturalInsights:
    """Analyze architectural patterns and insights."""
    insights = ArchitecturalInsights()
    
    # Detect architectural patterns
    insights.architectural_patterns = detect_architectural_patterns(codebase)
    
    # Calculate coupling metrics
    insights.coupling_metrics = calculate_coupling_metrics(codebase)
    
    # Calculate cohesion metrics
    insights.cohesion_metrics = calculate_cohesion_metrics(codebase)
    
    # Calculate modularity score
    insights.modularity_score = calculate_modularity_score(codebase)
    
    # Analyze components
    insights.component_analysis = analyze_components(codebase)
    
    return insights

def detect_architectural_patterns(codebase: Codebase) -> List[str]:
    """Detect common architectural patterns."""
    patterns = []
    
    class_names = [cls.name.lower() for cls in codebase.classes]
    file_paths = [file.filepath.lower() for file in codebase.files]
    
    # MVC Pattern
    if any('controller' in name for name in class_names) and \
       any('model' in name for name in class_names) and \
       any('view' in name for name in class_names):
        patterns.append('MVC')
    
    # Repository Pattern
    if any('repository' in name for name in class_names):
        patterns.append('Repository')
    
    # Factory Pattern
    if any('factory' in name for name in class_names):
        patterns.append('Factory')
    
    # Observer Pattern
    if any('observer' in name or 'listener' in name for name in class_names):
        patterns.append('Observer')
    
    # Layered Architecture
    if any('service' in path for path in file_paths) and \
       any('controller' in path for path in file_paths) and \
       any('model' in path or 'entity' in path for path in file_paths):
        patterns.append('Layered Architecture')
    
    return patterns

def calculate_coupling_metrics(codebase: Codebase) -> Dict[str, float]:
    """Calculate coupling metrics between modules."""
    metrics = {}
    
    # Afferent coupling (Ca) - incoming dependencies
    # Efferent coupling (Ce) - outgoing dependencies
    
    file_dependencies = {}
    for file in codebase.files:
        deps = []
        if hasattr(file, 'imports') and file.imports:
            deps = [getattr(imp, 'module', 'unknown') for imp in file.imports]
        file_dependencies[file.filepath] = deps
    
    for filepath, deps in file_dependencies.items():
        # Efferent coupling
        ce = len(deps)
        
        # Afferent coupling
        ca = sum(1 for other_deps in file_dependencies.values() if filepath in other_deps)
        
        # Instability (I = Ce / (Ca + Ce))
        instability = ce / (ca + ce) if (ca + ce) > 0 else 0
        
        metrics[filepath] = {
            'afferent_coupling': ca,
            'efferent_coupling': ce,
            'instability': instability
        }
    
    return metrics

def calculate_cohesion_metrics(codebase: Codebase) -> Dict[str, float]:
    """Calculate cohesion metrics for classes."""
    metrics = {}
    
    for cls in codebase.classes:
        if hasattr(cls, 'methods') and cls.methods:
            # LCOM (Lack of Cohesion of Methods)
            method_attribute_usage = {}
            
            for method in cls.methods:
                if hasattr(method, 'code_block') and method.code_block:
                    # Simple heuristic: look for attribute usage
                    code = method.code_block.source
                    used_attributes = []
                    
                    if hasattr(cls, 'attributes'):
                        for attr in cls.attributes:
                            if attr.name in code:
                                used_attributes.append(attr.name)
                    
                    method_attribute_usage[method.name] = used_attributes
            
            # Calculate LCOM
            if method_attribute_usage:
                total_pairs = 0
                cohesive_pairs = 0
                
                methods = list(method_attribute_usage.keys())
                for i in range(len(methods)):
                    for j in range(i + 1, len(methods)):
                        total_pairs += 1
                        attrs1 = set(method_attribute_usage[methods[i]])
                        attrs2 = set(method_attribute_usage[methods[j]])
                        if attrs1.intersection(attrs2):
                            cohesive_pairs += 1
                
                lcom = 1 - (cohesive_pairs / total_pairs) if total_pairs > 0 else 0
                metrics[cls.name] = {'lcom': lcom}
    
    return metrics

def calculate_modularity_score(codebase: Codebase) -> float:
    """Calculate overall modularity score."""
    if not codebase.files:
        return 0.0
    
    # Simple modularity metric based on file organization
    directories = set()
    for file in codebase.files:
        dir_path = os.path.dirname(file.filepath)
        if dir_path:
            directories.add(dir_path)
    
    # More directories relative to files suggests better modularity
    modularity = len(directories) / len(codebase.files) if codebase.files else 0
    return min(modularity * 100, 100)  # Cap at 100

def analyze_components(codebase: Codebase) -> Dict[str, Any]:
    """Analyze components and their relationships."""
    components = {}
    
    # Group files by directory (component)
    for file in codebase.files:
        component = os.path.dirname(file.filepath) or 'root'
        if component not in components:
            components[component] = {
                'files': [],
                'functions': 0,
                'classes': 0,
                'lines_of_code': 0
            }
        
        components[component]['files'].append(file.filepath)
        components[component]['functions'] += len(file.functions)
        components[component]['classes'] += len(file.classes)
        
        if file.source:
            components[component]['lines_of_code'] += len(file.source.split('\n'))
    
    return components

# ============================================================================
# SECTION 7: SECURITY AND PERFORMANCE ANALYSIS FUNCTIONS
# ============================================================================

def analyze_security(codebase: Codebase) -> SecurityAnalysis:
    """Perform security-focused analysis."""
    analysis = SecurityAnalysis()
    
    for file in codebase.files:
        if file.source:
            code = file.source.lower()
            
            # Look for potential security issues
            if 'password' in code and ('=' in code or 'input' in code):
                analysis.potential_vulnerabilities.append({
                    'type': 'hardcoded_password',
                    'file': file.filepath,
                    'description': 'Potential hardcoded password or password handling'
                })
            
            if 'sql' in code and any(keyword in code for keyword in ['select', 'insert', 'update', 'delete']):
                analysis.potential_vulnerabilities.append({
                    'type': 'sql_injection_risk',
                    'file': file.filepath,
                    'description': 'Potential SQL injection vulnerability'
                })
            
            if any(pattern in code for pattern in ['eval(', 'exec(', 'subprocess.call']):
                analysis.potential_vulnerabilities.append({
                    'type': 'code_injection_risk',
                    'file': file.filepath,
                    'description': 'Potential code injection vulnerability'
                })
    
    return analysis

def analyze_performance(codebase: Codebase) -> PerformanceAnalysis:
    """Perform performance-focused analysis."""
    analysis = PerformanceAnalysis()
    
    for func in codebase.functions:
        if hasattr(func, 'code_block') and func.code_block:
            code = func.code_block.source.lower()
            
            # Look for performance issues
            if 'for' in code and 'for' in code:  # Nested loops
                nested_loops = code.count('for') > 1
                if nested_loops:
                    analysis.performance_hotspots.append({
                        'type': 'nested_loops',
                        'function': func.name,
                        'description': 'Potential O(n²) or higher complexity'
                    })
            
            # Database queries in loops
            if 'for' in code and any(db_keyword in code for db_keyword in ['select', 'query', 'find']):
                analysis.performance_hotspots.append({
                    'type': 'n_plus_one_query',
                    'function': func.name,
                    'description': 'Potential N+1 query problem'
                })
    
    return analysis

# ============================================================================
# SECTION 8: INHERITANCE AND RECURSION ANALYSIS
# ============================================================================

def analyze_inheritance_patterns(codebase: Codebase) -> InheritanceAnalysis:
    """Analyze inheritance patterns in the codebase."""
    if not codebase.classes:
        return InheritanceAnalysis()
    
    deepest_class = None
    max_depth = 0
    inheritance_chain = []
    
    for cls in codebase.classes:
        depth = calculate_doi(cls)
        if depth > max_depth:
            max_depth = depth
            deepest_class = cls
            # Build inheritance chain
            chain = [cls.name]
            current = cls
            while current.superclasses:
                # Take the first superclass for simplicity
                parent_obj = current.superclasses[0] if current.superclasses else None
                if parent_obj:
                    # Convert to string to avoid serialization issues
                    parent_name = str(parent_obj) if hasattr(parent_obj, '__str__') else getattr(parent_obj, 'name', str(parent_obj))
                    chain.append(parent_name)
                    # Find parent class in codebase
                    parent_cls = next((c for c in codebase.classes if c.name == parent_name), None)
                    if parent_cls:
                        current = parent_cls
                    else:
                        break
                else:
                    break
            inheritance_chain = list(reversed(chain))
    
    return InheritanceAnalysis(
        deepest_class_name=deepest_class.name if deepest_class else None,
        deepest_class_depth=max_depth,
        inheritance_chain=inheritance_chain
    )

def analyze_recursive_functions(codebase: Codebase) -> RecursionAnalysis:
    """Analyze recursive functions in the codebase."""
    recursive_functions = []
    
    for function in codebase.functions:
        if is_recursive_function(function):
            recursive_functions.append(function.name)
    
    return RecursionAnalysis(
        recursive_functions=recursive_functions,
        total_recursive_count=len(recursive_functions)
    )

def is_recursive_function(function: Function) -> bool:
    """Check if a function is recursive."""
    if not hasattr(function, 'code_block') or not function.code_block:
        return False
    
    code = function.code_block.source
    function_name = function.name
    
    # Simple check: look for function name in the code body
    # This is a basic heuristic and could be improved with proper AST analysis
    return function_name in code and code.count(function_name) > 1

# ============================================================================
# SECTION 9: MAIN COMPREHENSIVE ANALYSIS FUNCTION
# ============================================================================

def perform_comprehensive_analysis(codebase: Codebase, analysis_types: List[AnalysisType] = None) -> Dict[str, Any]:
    """Perform comprehensive analysis based on specified types."""
    if analysis_types is None:
        analysis_types = list(AnalysisType)
    
    results = {}
    
    if AnalysisType.DEPENDENCY in analysis_types:
        results['dependency_analysis'] = analyze_dependencies_comprehensive(codebase)
    
    if AnalysisType.CALL_GRAPH in analysis_types:
        results['call_graph_analysis'] = analyze_call_graph(codebase)
    
    if AnalysisType.CODE_QUALITY in analysis_types:
        results['code_quality_metrics'] = analyze_code_quality(codebase)
    
    if AnalysisType.ARCHITECTURAL in analysis_types:
        results['architectural_insights'] = analyze_architecture(codebase)
    
    if AnalysisType.SECURITY in analysis_types:
        results['security_analysis'] = analyze_security(codebase)
    
    if AnalysisType.PERFORMANCE in analysis_types:
        results['performance_analysis'] = analyze_performance(codebase)
    
    return results
