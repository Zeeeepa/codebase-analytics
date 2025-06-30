#!/usr/bin/env python3
"""
Consolidated Analysis Module

This module contains all the core analysis functions used by the API,
consolidated from analyzer.py, comprehensive_analysis.py, and api.py
to eliminate redundancy and provide a clean interface.
"""

import math
import re
import os
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import Counter, defaultdict
from pathlib import Path

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

# Data classes for analysis results
from dataclasses import dataclass
from typing import Optional

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

# Core Complexity Analysis Functions

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

# Inheritance and Recursion Analysis

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
                parent_name = current.superclasses[0] if current.superclasses else None
                if parent_name:
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

# File and Issue Analysis

def analyze_file_issues(file: SourceFile) -> Dict[str, List[Dict[str, str]]]:
    """Analyze issues in a source file."""
    issues = {
        'critical': [],
        'major': [],
        'minor': []
    }
    
    if not file.source:
        return issues
    
    # Analyze functions for issues
    for function in file.functions:
        # Check for unused functions
        if not any(function.name in str(usage) for usage in function.usages):
            issues['minor'].append({
                'type': 'unused_function',
                'message': f'Function "{function.name}" appears to be unused',
                'line': function.start_point[0] if hasattr(function, 'start_point') else 0
            })
        
        # Check for potential issues in code
        if hasattr(function, 'code_block') and function.code_block:
            code = function.code_block.source
            
            # Check for potential null references
            if 'None' in code and not any(s in code for s in ['is None', '== None', '!= None']):
                issues['critical'].append({
                    'type': 'null_reference',
                    'message': f'Potential unsafe null reference in function "{function.name}"',
                    'line': function.start_point[0] if hasattr(function, 'start_point') else 0
                })
            
            # Check for TODO/FIXME comments
            if 'TODO' in code or 'FIXME' in code:
                issues['major'].append({
                    'type': 'incomplete_implementation',
                    'message': f'Incomplete implementation in function "{function.name}"',
                    'line': function.start_point[0] if hasattr(function, 'start_point') else 0
                })
            
            # Check for code duplication (simple heuristic)
            lines = code.split('\n')
            seen_blocks = {}
            for i, line in enumerate(lines):
                stripped = line.strip()
                if len(stripped) > 20:  # Only check substantial lines
                    if stripped in seen_blocks:
                        issues['minor'].append({
                            'type': 'code_duplication',
                            'message': f'Potential code duplication in function "{function.name}"',
                            'line': i + (function.start_point[0] if hasattr(function, 'start_point') else 0)
                        })
                    else:
                        seen_blocks[stripped] = function.name
    
    return issues

# Repository Structure Building

def build_repo_structure(files: List[SourceFile], file_issues: Dict[str, Dict], file_symbols: Dict[str, List]) -> Dict:
    """Build a hierarchical repository structure with issue counts and symbols."""
    root = {
        'name': 'root',
        'type': 'directory',
        'path': '',
        'children': {},
        'issues': {'critical': 0, 'major': 0, 'minor': 0},
        'stats': {
            'files': 0,
            'directories': 0,
            'symbols': 0,
            'issues': 0
        }
    }
    
    # First pass: Create all directories
    all_dirs = set()
    for file in files:
        dir_path = os.path.dirname(file.filepath)
        if dir_path:
            parts = dir_path.split('/')
            current_path = ''
            for part in parts:
                current_path = os.path.join(current_path, part) if current_path else part
                all_dirs.add(current_path)
    
    # Create directory nodes
    for dir_path in sorted(all_dirs):
        parts = dir_path.split('/')
        current = root
        current_path = ''
        
        for part in parts:
            current_path = os.path.join(current_path, part) if current_path else part
            if part not in current['children']:
                current['children'][part] = {
                    'name': part,
                    'type': 'directory',
                    'path': current_path,
                    'children': {},
                    'issues': {'critical': 0, 'major': 0, 'minor': 0},
                    'stats': {
                        'files': 0,
                        'directories': 0,
                        'symbols': 0,
                        'issues': 0
                    }
                }
                current['stats']['directories'] += 1
            current = current['children'][part]
    
    # Add files
    for file in sorted(files, key=lambda f: f.filepath):
        dir_path = os.path.dirname(file.filepath)
        filename = os.path.basename(file.filepath)
        
        # Navigate to the correct directory
        current = root
        if dir_path:
            for part in dir_path.split('/'):
                current = current['children'][part]
        
        # Create file node
        file_node = {
            'name': filename,
            'type': 'file',
            'file_type': get_file_type(filename),
            'path': file.filepath,
            'issues': {'critical': 0, 'major': 0, 'minor': 0},
            'stats': {
                'symbols': 0,
                'issues': 0
            }
        }
        
        # Add issue counts
        if file.filepath in file_issues:
            issues = file_issues[file.filepath]
            file_node['issues'] = {
                'critical': len(issues['critical']),
                'major': len(issues['major']),
                'minor': len(issues['minor'])
            }
            file_node['stats']['issues'] = sum(file_node['issues'].values())
            
            # Propagate issue counts up the tree
            temp_path = dir_path
            temp = current
            while temp is not None:
                for severity in ['critical', 'major', 'minor']:
                    temp['issues'][severity] += file_node['issues'][severity]
                temp['stats']['issues'] += file_node['stats']['issues']
                if temp_path:
                    parent_path = os.path.dirname(temp_path)
                    temp = root
                    if parent_path:
                        for part in parent_path.split('/'):
                            temp = temp['children'][part]
                    temp_path = parent_path
                else:
                    temp = None
        
        # Add symbols
        if file.filepath in file_symbols:
            file_node['symbols'] = file_symbols[file.filepath]
            file_node['stats']['symbols'] = len(file_symbols[file.filepath])
            
            # Propagate symbol counts up the tree
            temp_path = dir_path
            temp = current
            while temp is not None:
                temp['stats']['symbols'] += file_node['stats']['symbols']
                if temp_path:
                    parent_path = os.path.dirname(temp_path)
                    temp = root
                    if parent_path:
                        for part in parent_path.split('/'):
                            temp = temp['children'][part]
                    temp_path = parent_path
                else:
                    temp = None
        
        current['children'][filename] = file_node
        current['stats']['files'] += 1
        root['stats']['files'] += 1
    
    return root

def get_file_type(filename: str) -> str:
    """Get the type of file based on its extension."""
    ext = Path(filename).suffix.lower()
    if ext in ['.py', '.pyi', '.pyx']:
        return 'python'
    elif ext in ['.js', '.jsx', '.ts', '.tsx']:
        return 'javascript'
    elif ext in ['.java']:
        return 'java'
    elif ext in ['.c', '.cpp', '.h', '.hpp']:
        return 'cpp'
    elif ext in ['.go']:
        return 'go'
    elif ext in ['.rs']:
        return 'rust'
    elif ext in ['.rb']:
        return 'ruby'
    elif ext in ['.php']:
        return 'php'
    elif ext in ['.cs']:
        return 'csharp'
    elif ext in ['.swift']:
        return 'swift'
    elif ext in ['.kt']:
        return 'kotlin'
    elif ext in ['.scala']:
        return 'scala'
    elif ext in ['.html', '.htm']:
        return 'html'
    elif ext in ['.css', '.scss', '.sass', '.less']:
        return 'css'
    elif ext in ['.json']:
        return 'json'
    elif ext in ['.xml']:
        return 'xml'
    elif ext in ['.md', '.markdown']:
        return 'markdown'
    elif ext in ['.yml', '.yaml']:
        return 'yaml'
    elif ext in ['.sh', '.bash']:
        return 'shell'
    elif ext in ['.sql']:
        return 'sql'
    elif ext in ['.dockerfile', '.containerfile']:
        return 'docker'
    elif ext in ['.gitignore', '.dockerignore']:
        return 'config'
    elif ext in ['.txt']:
        return 'text'
    else:
        return 'unknown'

# Symbol Context and Call Chain Analysis

def get_detailed_symbol_context(symbol: Symbol) -> Dict[str, Any]:
    """Get detailed context for any symbol type."""
    base_info = {
        'id': str(hash(symbol.name + symbol.filepath)),
        'name': symbol.name,
        'type': symbol.symbol_type.value if hasattr(symbol, 'symbol_type') else 'unknown',
        'filepath': symbol.filepath,
        'start_line': symbol.start_point[0] if hasattr(symbol, 'start_point') else 0,
        'end_line': symbol.end_point[0] if hasattr(symbol, 'end_point') else 0,
        'dependencies': [],
        'usages': [],
        'issues': [],
        'metrics': {}
    }
    
    # Add dependencies
    if hasattr(symbol, 'dependencies'):
        base_info['dependencies'] = [
            {
                'name': dep.name,
                'type': dep.symbol_type.value if hasattr(dep, 'symbol_type') else 'unknown',
                'filepath': dep.filepath
            }
            for dep in symbol.dependencies
        ]
    
    # Add usages
    if hasattr(symbol, 'usages'):
        base_info['usages'] = [
            {
                'filepath': usage.filepath,
                'line': usage.start_point[0] if hasattr(usage, 'start_point') else 0
            }
            for usage in symbol.usages
        ]
    
    # Add metrics for functions
    if isinstance(symbol, Function):
        complexity = calculate_cyclomatic_complexity(symbol)
        operators, operands = get_operators_and_operands(symbol)
        volume, N1, N2, n1, n2 = calculate_halstead_volume(operators, operands)
        loc = len(symbol.code_block.source.splitlines()) if hasattr(symbol, 'code_block') and symbol.code_block else 0
        mi_score = calculate_maintainability_index(volume, complexity, loc)
        
        base_info['metrics'] = {
            'cyclomatic_complexity': complexity,
            'halstead_volume': volume,
            'lines_of_code': loc,
            'maintainability_index': mi_score,
            'operators': N1,
            'operands': N2,
            'unique_operators': n1,
            'unique_operands': n2,
            'rank': get_maintainability_rank(mi_score)
        }
    
    return base_info

def get_max_call_chain(function: Function) -> List[Function]:
    """Get the maximum call chain for a function using DFS."""
    visited = set()
    max_chain = []
    
    def build_graph(func, current_chain):
        nonlocal max_chain
        
        if func.name in visited:
            return
        
        visited.add(func.name)
        current_chain.append(func)
        
        if len(current_chain) > len(max_chain):
            max_chain = current_chain.copy()
        
        # Get called functions (this is a simplified approach)
        if hasattr(func, 'called_functions'):
            for called_func in func.called_functions:
                build_graph(called_func, current_chain)
        
        current_chain.pop()
        visited.remove(func.name)
    
    build_graph(function, [])
    return max_chain

