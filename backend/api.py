from typing import Dict, List, Tuple, Any, Optional
from codegen import Codebase
import math
import re
import requests
from datetime import datetime, timedelta
import subprocess
import os
import tempfile
from collections import Counter
import networkx as nx
from pathlib import Path

def get_monthly_commits(repo_url: str) -> Dict[str, int]:
    """Get commit counts by month for the last year."""
    monthly_counts = {}
    
    try:
        # Clone repo to temp dir
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(["git", "clone", repo_url, temp_dir], check=True, capture_output=True)
            
            # Get commits for last year
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            # Format dates for git log
            date_format = "%Y-%m-%d"
            start_str = start_date.strftime(date_format)
            end_str = end_date.strftime(date_format)
            
            # Get commit dates
            result = subprocess.run(
                ["git", "log", f"--since={start_str}", f"--until={end_str}", "--format=%aI"],
                cwd=temp_dir,
                check=True,
                capture_output=True,
                text=True
            )
            
            # Parse commit dates and count by month
            for line in result.stdout.splitlines():
                try:
                    commit_date = datetime.fromisoformat(line.strip())
                    month_key = commit_date.strftime("%Y-%m")
                    monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1
                except ValueError:
                    continue
                    
    except subprocess.CalledProcessError:
        # Return empty dict if git operations fail
        pass
        
    return monthly_counts

def analyze_file_issues(file_path: Path) -> Dict[str, int]:
    """Analyze a file for issues and return counts by severity."""
    issues = {"critical": 0, "major": 0, "minor": 0}
    
    # Read file content
    try:
        content = file_path.read_text()
    except Exception:
        return issues
    
    # Check for critical issues
    critical_patterns = [
        r"TODO\s*:",  # Incomplete code
        r"FIXME\s*:",  # Known bugs
        r"raise\s+NotImplementedError",  # Unimplemented code
        r"print\s*\(",  # Debug prints
        r"debugger;",  # JavaScript debugger statements
    ]
    
    # Check for major issues
    major_patterns = [
        r"catch\s*\(\s*Exception\s*\)",  # Bare except
        r"except\s*:",  # Bare except in Python
        r"console\.(log|debug|info)",  # Console logging
        r"alert\s*\(",  # JavaScript alerts
    ]
    
    # Check for minor issues
    minor_patterns = [
        r"#\s*type:\s*ignore",  # Type ignore comments
        r"//\s*eslint-disable",  # ESLint disable comments
        r"#\s*noqa",  # Flake8 disable comments
    ]
    
    for pattern in critical_patterns:
        issues["critical"] += len(re.findall(pattern, content))
    
    for pattern in major_patterns:
        issues["major"] += len(re.findall(pattern, content))
        
    for pattern in minor_patterns:
        issues["minor"] += len(re.findall(pattern, content))
    
    return issues

def calculate_line_metrics(codebase: Codebase) -> Dict[str, Dict[str, float]]:
    """Calculate various line-based metrics for the codebase."""
    metrics = {}
    
    for file in codebase.files:
        file_metrics = {
            "total_lines": len(file.source.split('\n')),
            "code_lines": len([l for l in file.source.split('\n') if l.strip() and not l.strip().startswith('#')]),
            "comment_lines": len([l for l in file.source.split('\n') if l.strip().startswith('#')]),
            "blank_lines": len([l for l in file.source.split('\n') if not l.strip()]),
            "function_lines": sum(f.end_point[0] - f.start_point[0] for f in file.functions),
            "class_lines": sum(c.end_point[0] - c.start_point[0] for c in file.classes),
        }
        metrics[file.filepath] = file_metrics
    
    return metrics

def calculate_complexity_metrics(codebase: Codebase) -> Dict[str, Dict[str, float]]:
    """Calculate various complexity metrics for the codebase."""
    metrics = {
        "cyclomatic": {},
        "inheritance": {},
        "halstead": {},
        "maintainability": {}
    }
    
    for file in codebase.files:
        # Calculate cyclomatic complexity
        cyclomatic = 0
        for func in file.functions:
            # Count decision points
            cyclomatic += 1  # Base complexity
            cyclomatic += len([s for s in func.statements if isinstance(s, (IfBlockStatement, ForLoopStatement, WhileStatement, TryCatchStatement))])
            cyclomatic += len([e for e in func.expressions if isinstance(e, (BinaryExpression, UnaryExpression, ComparisonExpression))])
        metrics["cyclomatic"][file.filepath] = cyclomatic
        
        # Calculate inheritance depth
        max_inheritance = 0
        for cls in file.classes:
            inheritance_depth = len(cls.superclasses)
            max_inheritance = max(max_inheritance, inheritance_depth)
        metrics["inheritance"][file.filepath] = max_inheritance
        
        # Calculate Halstead metrics
        operators = set()
        operands = set()
        for func in file.functions:
            for expr in func.expressions:
                if isinstance(expr, (BinaryExpression, UnaryExpression)):
                    operators.add(expr.operator)
                    operands.update(expr.operands)
        
        n1 = len(operators)  # Unique operators
        n2 = len(operands)   # Unique operands
        N1 = sum(1 for f in file.functions for e in f.expressions if isinstance(e, (BinaryExpression, UnaryExpression)))  # Total operators
        N2 = sum(len(e.operands) for f in file.functions for e in f.expressions if isinstance(e, (BinaryExpression, UnaryExpression)))  # Total operands
        
        if n1 > 0 and n2 > 0:
            program_length = N1 + N2
            vocabulary = n1 + n2
            volume = program_length * math.log2(vocabulary) if vocabulary > 0 else 0
            difficulty = (n1 * N2) / (2 * n2) if n2 > 0 else 0
            effort = volume * difficulty
            metrics["halstead"][file.filepath] = int(effort)
        else:
            metrics["halstead"][file.filepath] = 0
        
        # Calculate maintainability index
        if metrics["halstead"][file.filepath] > 0:
            loc = len(file.source.split('\n'))
            comments = len([l for l in file.source.split('\n') if l.strip().startswith('#')])
            comment_ratio = comments / loc if loc > 0 else 0
            
            maintainability = (
                171 - 
                5.2 * math.log(metrics["halstead"][file.filepath]) -
                0.23 * metrics["cyclomatic"][file.filepath] -
                16.2 * math.log(loc)
            ) * (0.75 + 0.25 * comment_ratio)
            
            metrics["maintainability"][file.filepath] = int(maintainability)
        else:
            metrics["maintainability"][file.filepath] = 100  # Perfect score for empty/simple files
    
    return metrics

def find_dead_code(codebase: Codebase) -> List[Dict[str, str]]:
    """Find functions that are never called."""
    dead_functions = []
    for function in codebase.functions:
        if not any(function.function_calls):
            dead_functions.append({
                "name": function.name,
                "filepath": function.filepath
            })
    return dead_functions

def get_max_call_chain(function) -> List:
    """Get the longest call chain starting from a function."""
    G = nx.DiGraph()

    def build_graph(func, depth=0):
        if depth > 10:  # Prevent infinite recursion
            return
        for call in func.function_calls:
            called_func = call.function_definition
            G.add_edge(func, called_func)
            build_graph(called_func, depth + 1)

    build_graph(function)
    return nx.dag_longest_path(G)

def analyze_codebase(repo_url: str) -> Dict[str, Any]:
    """Analyze a repository and return detailed metrics."""
    codebase = Codebase(repo_url)
    
    # Get test analysis
    test_functions = [x for x in codebase.functions if x.name.startswith('test_')]
    test_classes = [x for x in codebase.classes if x.name.startswith('Test')]
    tests_per_file = len(test_functions) / len(codebase.files) if codebase.files else 0
    
    # Get top test files
    file_test_counts = Counter([x.file for x in test_classes])
    top_test_files = []
    for file, num_tests in file_test_counts.most_common()[:5]:
        top_test_files.append({
            "filepath": file.filepath,
            "test_count": num_tests,
            "file_length": len(file.source.split('\n')),
            "function_count": len(file.functions)
        })
    
    # Get function analysis
    most_called = max(codebase.functions, key=lambda f: len(f.call_sites))
    most_called_info = {
        "name": most_called.name,
        "call_count": len(most_called.call_sites),
        "callers": [{"name": call.parent_function.name, "line": call.start_point[0]} 
                   for call in most_called.call_sites]
    }
    
    most_calls = max(codebase.functions, key=lambda f: len(f.function_calls))
    most_calls_info = {
        "name": most_calls.name,
        "calls_count": len(most_calls.function_calls),
        "called_functions": [call.name for call in most_calls.function_calls]
    }
    
    # Find unused functions (potential dead code)
    unused = [f for f in codebase.functions if len(f.call_sites) == 0]
    unused_info = [{"name": f.name, "filepath": f.filepath} for f in unused]
    
    # Find recursive functions
    recursive = [f for f in codebase.functions 
                if any(call.name == f.name for call in f.function_calls)]
    recursive_names = [f.name for f in recursive]
    
    # Get class analysis
    deepest_inheritance = None
    if codebase.classes:
        deepest_class = max(codebase.classes, key=lambda x: len(x.superclasses))
        deepest_inheritance = {
            "name": deepest_class.name,
            "depth": len(deepest_class.superclasses),
            "chain": [s.name for s in deepest_class.superclasses]
        }
    
    # Calculate line metrics
    line_metrics = calculate_line_metrics(codebase)
    
    # Calculate complexity metrics
    complexity_metrics = calculate_complexity_metrics(codebase)
    
    # Get git metrics
    monthly_commits = get_monthly_commits(repo_url)
    
    return {
        "test_analysis": {
            "total_test_functions": len(test_functions),
            "total_test_classes": len(test_classes),
            "tests_per_file": tests_per_file,
            "top_test_files": top_test_files
        },
        "function_analysis": {
            "total_functions": len(codebase.functions),
            "most_called_function": most_called_info,
            "function_with_most_calls": most_calls_info,
            "recursive_functions": recursive_names,
            "unused_functions": unused_info,
            "dead_code": find_dead_code(codebase)
        },
        "class_analysis": {
            "total_classes": len(codebase.classes),
            "deepest_inheritance": deepest_inheritance,
            "total_imports": len(codebase.imports)
        },
        "metrics": {
            "line_metrics": line_metrics,
            "cyclomatic_complexity": complexity_metrics["cyclomatic"],
            "depth_of_inheritance": complexity_metrics["inheritance"],
            "halstead_metrics": complexity_metrics["halstead"],
            "maintainability_index": complexity_metrics["maintainability"],
            "monthly_commits": monthly_commits
        }
    }

