from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Tuple, Any, Optional
from codegen import Codebase
from codegen.sdk.core.statements.for_loop_statement import ForLoopStatement
from codegen.sdk.core.statements.if_block_statement import IfBlockStatement
from codegen.sdk.core.statements.try_catch_statement import TryCatchStatement
from codegen.sdk.core.statements.while_statement import WhileStatement
from codegen.sdk.core.expressions.binary_expression import BinaryExpression
from codegen.sdk.core.expressions.unary_expression import UnaryExpression
from codegen.sdk.core.expressions.comparison_expression import ComparisonExpression
import math
import re
import requests
from datetime import datetime, timedelta
import subprocess
import os
import tempfile
from fastapi.middleware.cors import CORSMiddleware
import modal
from collections import Counter
import networkx as nx

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "codegen", "fastapi", "uvicorn", "gitpython", "requests", "pydantic", "datetime",
        "networkx"  # Added for call chain analysis
    )
)

app = modal.App(name="analytics-app", image=image)
fastapi_app = FastAPI()

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Existing helper functions
def get_monthly_commits(repo_path: str) -> Dict[str, int]:
    """Get the number of commits per month for the last 12 months."""
    # ... [Keep existing implementation] ...

def calculate_cyclomatic_complexity(function):
    """Calculate cyclomatic complexity of a function."""
    # ... [Keep existing implementation] ...

def cc_rank(complexity):
    """Convert complexity score to letter grade."""
    # ... [Keep existing implementation] ...

def calculate_doi(cls):
    """Calculate the depth of inheritance for a given class."""
    # ... [Keep existing implementation] ...

def get_operators_and_operands(function):
    """Extract operators and operands from a function."""
    # ... [Keep existing implementation] ...

def calculate_halstead_volume(operators, operands):
    """Calculate Halstead volume metrics."""
    # ... [Keep existing implementation] ...

def count_lines(source: str):
    """Count different types of lines in source code."""
    # ... [Keep existing implementation] ...

def calculate_maintainability_index(halstead_volume: float, cyclomatic_complexity: float, loc: int) -> int:
    """Calculate the normalized maintainability index."""
    # ... [Keep existing implementation] ...

def get_maintainability_rank(mi_score: float) -> str:
    """Convert maintainability index score to letter grade."""
    # ... [Keep existing implementation] ...

def get_github_repo_description(repo_url):
    """Get repository description from GitHub API."""
    # ... [Keep existing implementation] ...

# New analysis helper functions
def find_dead_code(codebase) -> List:
    """Find functions that are never called."""
    dead_functions = []
    for function in codebase.functions:
        if not any(function.function_calls):
            dead_functions.append(function)
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

def analyze_file_issues(file) -> Dict[str, List[Dict[str, str]]]:
    """Analyze a file for various types of issues."""
    issues = {
        'critical': [],
        'major': [],
        'minor': []
    }
    
    # Check for implementation errors
    for function in file.functions:
        # Check for unused parameters
        for param in function.parameters:
            if not any(param.name in str(usage) for usage in function.usages):
                issues['minor'].append({
                    'type': 'unused_parameter',
                    'message': f'Unused parameter "{param.name}" in function "{function.name}"'
                })

        # Check for null references
        if hasattr(function, 'code_block'):
            code = function.code_block.source
            if 'None' in code and not any(s in code for s in ['is None', '== None', '!= None']):
                issues['critical'].append({
                    'type': 'unsafe_null_check',
                    'message': f'Potential unsafe null reference in function "{function.name}"'
                })

        # Check for incomplete implementations
        if 'TODO' in function.source or 'FIXME' in function.source:
            issues['major'].append({
                'type': 'incomplete_implementation',
                'message': f'Incomplete implementation in function "{function.name}"'
            })

    # Check for code duplication
    seen_blocks = {}
    for function in file.functions:
        if hasattr(function, 'code_block'):
            code = function.code_block.source.strip()
            if len(code) > 50:  # Only check substantial blocks
                if code in seen_blocks:
                    issues['major'].append({
                        'type': 'code_duplication',
                        'message': f'Code duplication between functions "{function.name}" and "{seen_blocks[code]}"'
                    })
                else:
                    seen_blocks[code] = function.name

    return issues

def build_repo_structure(files, file_issues) -> Dict:
    """Build a hierarchical repository structure with issue counts."""
    root = {'name': 'root', 'children': {}}
    
    for file in files:
        path_parts = file.filepath.split('/')
        current = root
        
        # Build the tree structure
        for i, part in enumerate(path_parts[:-1]):
            if part not in current['children']:
                current['children'][part] = {
                    'name': part,
                    'type': 'directory',
                    'children': {},
                    'issues': {'critical': 0, 'major': 0, 'minor': 0}
                }
            current = current['children'][part]
        
        # Add the file
        filename = path_parts[-1]
        file_node = {
            'name': filename,
            'type': 'file',
            'issues': {'critical': 0, 'major': 0, 'minor': 0}
        }
        
        # Add issue counts if present
        if file.filepath in file_issues:
            issues = file_issues[file.filepath]
            file_node['issues'] = {
                'critical': len(issues.critical),
                'major': len(issues.major),
                'minor': len(issues.minor)
            }
            
            # Propagate counts up the tree
            temp = root
            for part in path_parts[:-1]:
                temp = temp['children'][part]
                for severity in ['critical', 'major', 'minor']:
                    temp['issues'][severity] += file_node['issues'][severity]
        
        current['children'][filename] = file_node
    
    return root

# Request models
class RepoRequest(BaseModel):
    repo_url: str

class TestAnalysis(BaseModel):
    total_test_functions: int
    total_test_classes: int
    tests_per_file: float
    top_test_files: List[Dict[str, any]]

class FunctionAnalysis(BaseModel):
    total_functions: int
    most_called_function: Dict[str, any]
    function_with_most_calls: Dict[str, any]
    recursive_functions: List[str]
    unused_functions: List[Dict[str, str]]
    dead_code: List[Dict[str, str]]

class ClassAnalysis(BaseModel):
    total_classes: int
    deepest_inheritance: Optional[Dict[str, any]]
    total_imports: int

class FileIssue(BaseModel):
    critical: List[Dict[str, str]]
    major: List[Dict[str, str]]
    minor: List[Dict[str, str]]

class ExtendedAnalysis(BaseModel):
    test_analysis: TestAnalysis
    function_analysis: FunctionAnalysis
    class_analysis: ClassAnalysis
    file_issues: Dict[str, FileIssue]
    repo_structure: Dict[str, any]

# Endpoints
@fastapi_app.post("/analyze_repo")
async def analyze_repo(request: RepoRequest) -> Dict[str, Any]:
    """Analyze a repository and return comprehensive metrics."""
    repo_url = request.repo_url
    codebase = Codebase.from_repo(repo_url)

    # Original analysis
    num_files = len(codebase.files(extensions="*"))
    num_functions = len(codebase.functions)
    num_classes = len(codebase.classes)

    total_loc = total_lloc = total_sloc = total_comments = 0
    total_complexity = 0
    total_volume = 0
    total_mi = 0
    total_doi = 0

    monthly_commits = get_monthly_commits(repo_url)

    for file in codebase.files:
        loc, lloc, sloc, comments = count_lines(file.source)
        total_loc += loc
        total_lloc += lloc
        total_sloc += sloc
        total_comments += comments

    callables = codebase.functions + [m for c in codebase.classes for m in c.methods]

    num_callables = 0
    for func in callables:
        if not hasattr(func, "code_block"):
            continue

        complexity = calculate_cyclomatic_complexity(func)
        operators, operands = get_operators_and_operands(func)
        volume, _, _, _, _ = calculate_halstead_volume(operators, operands)
        loc = len(func.code_block.source.splitlines())
        mi_score = calculate_maintainability_index(volume, complexity, loc)

        total_complexity += complexity
        total_volume += volume
        total_mi += mi_score
        num_callables += 1

    for cls in codebase.classes:
        doi = calculate_doi(cls)
        total_doi += doi

    desc = get_github_repo_description(repo_url)

    # New extended analysis
    test_functions = [x for x in codebase.functions if x.name.startswith('test_')]
    test_classes = [x for x in codebase.classes if x.name.startswith('Test')]
    tests_per_file = len(test_functions) / len(codebase.files) if codebase.files else 0
    
    # Get top test files
    file_test_counts = Counter([x.file for x in test_classes])
    top_test_files = [
        {
            'filepath': file.filepath,
            'test_count': num_tests,
            'file_length': len(file.source),
            'function_count': len(file.functions)
        }
        for file, num_tests in file_test_counts.most_common(5)
    ]

    # Function analysis
    recursive = [
        f.name for f in codebase.functions 
        if any(call.name == f.name for call in f.function_calls)
    ][:5]

    most_called = max(codebase.functions, key=lambda f: len(f.call_sites))
    most_called_info = {
        'name': most_called.name,
        'call_count': len(most_called.call_sites),
        'callers': [
            {
                'function': call.parent_function.name,
                'line': call.start_point[0]
            }
            for call in most_called.call_sites
        ]
    }

    most_calls = max(codebase.functions, key=lambda f: len(f.function_calls))
    most_calls_info = {
        'name': most_calls.name,
        'calls_count': len(most_calls.function_calls),
        'called_functions': [call.name for call in most_calls.function_calls]
    }

    unused = [
        {'name': f.name, 'filepath': f.filepath}
        for f in codebase.functions if len(f.call_sites) == 0
    ]

    dead_code = [
        {'name': f.name, 'filepath': f.filepath}
        for f in find_dead_code(codebase)
    ]

    # Class analysis
    deepest_class = None
    if codebase.classes:
        deepest = max(codebase.classes, key=lambda x: len(x.superclasses))
        deepest_class = {
            'name': deepest.name,
            'depth': len(deepest.superclasses),
            'chain': [s.name for s in deepest.superclasses]
        }

    # File issues analysis
    file_issues = {}
    for file in codebase.files:
        issues = analyze_file_issues(file)
        if any(len(v) > 0 for v in issues.values()):
            file_issues[file.filepath] = FileIssue(**issues)

    # Repository structure
    repo_structure = build_repo_structure(codebase.files, file_issues)

    # Combine original and new analysis
    results = {
        "repo_url": repo_url,
        "line_metrics": {
            "total": {
                "loc": total_loc,
                "lloc": total_lloc,
                "sloc": total_sloc,
                "comments": total_comments,
                "comment_density": (total_comments / total_loc * 100)
                if total_loc > 0
                else 0,
            },
        },
        "cyclomatic_complexity": {
            "average": total_complexity / num_callables if num_callables > 0 else 0,
        },
        "depth_of_inheritance": {
            "average": total_doi / len(codebase.classes) if codebase.classes else 0,
        },
        "halstead_metrics": {
            "total_volume": int(total_volume),
            "average_volume": int(total_volume / num_callables)
            if num_callables > 0
            else 0,
        },
        "maintainability_index": {
            "average": int(total_mi / num_callables) if num_callables > 0 else 0,
        },
        "description": desc,
        "num_files": num_files,
        "num_functions": num_functions,
        "num_classes": num_classes,
        "monthly_commits": monthly_commits,
        "extended_analysis": ExtendedAnalysis(
            test_analysis=TestAnalysis(
                total_test_functions=len(test_functions),
                total_test_classes=len(test_classes),
                tests_per_file=tests_per_file,
                top_test_files=top_test_files
            ),
            function_analysis=FunctionAnalysis(
                total_functions=len(codebase.functions),
                most_called_function=most_called_info,
                function_with_most_calls=most_calls_info,
                recursive_functions=recursive,
                unused_functions=unused,
                dead_code=dead_code
            ),
            class_analysis=ClassAnalysis(
                total_classes=len(codebase.classes),
                deepest_inheritance=deepest_class,
                total_imports=len(codebase.imports)
            ),
            file_issues=file_issues,
            repo_structure=repo_structure
        )
    }

    return results

@app.function(image=image)
@modal.asgi_app()
def fastapi_modal_app():
    return fastapi_app

if __name__ == "__main__":
    app.deploy("analytics-app")

