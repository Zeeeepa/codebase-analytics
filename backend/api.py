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

def get_monthly_commits(repo_path: str) -> Dict[str, int]:
    """
    Get the number of commits per month for the last 12 months.

    Args:
        repo_path: Path to the git repository

    Returns:
        Dictionary with month-year as key and number of commits as value
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    date_format = "%Y-%m-%d"
    since_date = start_date.strftime(date_format)
    until_date = end_date.strftime(date_format)
    repo_path = "https://github.com/" + repo_path

    try:
        original_dir = os.getcwd()

        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(["git", "clone", repo_path, temp_dir], check=True)
            os.chdir(temp_dir)

            cmd = [
                "git",
                "log",
                f"--since={since_date}",
                f"--until={until_date}",
                "--format=%aI",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            commit_dates = result.stdout.strip().split("\n")

            monthly_counts = {}
            current_date = start_date
            while current_date <= end_date:
                month_key = current_date.strftime("%Y-%m")
                monthly_counts[month_key] = 0
                current_date = (
                    current_date.replace(day=1) + timedelta(days=32)
                ).replace(day=1)

            for date_str in commit_dates:
                if date_str:  # Skip empty lines
                    commit_date = datetime.fromisoformat(date_str.strip())
                    month_key = commit_date.strftime("%Y-%m")
                    if month_key in monthly_counts:
                        monthly_counts[month_key] += 1

            os.chdir(original_dir)
            return dict(sorted(monthly_counts.items()))

    except subprocess.CalledProcessError as e:
        print(f"Error executing git command: {e}")
        return {}
    except Exception as e:
        print(f"Error processing git commits: {e}")
        return {}
    finally:
        try:
            os.chdir(original_dir)
        except:
            pass

def calculate_cyclomatic_complexity(function):
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
            complexity += statement.condition.count(
                " and "
            ) + statement.condition.count(" or ")

        if hasattr(statement, "nested_code_blocks"):
            for block in statement.nested_code_blocks:
                complexity += analyze_block(block)

        return complexity

    def analyze_block(block):
        if not block or not hasattr(block, "statements"):
            return 0
        return sum(analyze_statement(stmt) for stmt in block.statements)

    return (
        1 + analyze_block(function.code_block) if hasattr(function, "code_block") else 1
    )

def cc_rank(complexity):
    if complexity < 0:
        raise ValueError("Complexity must be a non-negative value")

    ranks = [
        (1, 5, "A"),
        (6, 10, "B"),
        (11, 20, "C"),
        (21, 30, "D"),
        (31, 40, "E"),
        (41, float("inf"), "F"),
    ]
    for low, high, rank in ranks:
        if low <= complexity <= high:
            return rank
    return "F"

def calculate_doi(cls):
    """Calculate the depth of inheritance for a given class."""
    return len(cls.superclasses)

def get_operators_and_operands(function):
    operators = []
    operands = []

    for statement in function.code_block.statements:
        for call in statement.function_calls:
            operators.append(call.name)
            for arg in call.args:
                operands.append(arg.source)

        if hasattr(statement, "expressions"):
            for expr in statement.expressions:
                if isinstance(expr, BinaryExpression):
                    operators.extend([op.source for op in expr.operators])
                    operands.extend([elem.source for elem in expr.elements])
                elif isinstance(expr, UnaryExpression):
                    operators.append(expr.ts_node.type)
                    operands.append(expr.argument.source)
                elif isinstance(expr, ComparisonExpression):
                    operators.extend([op.source for op in expr.operators])
                    operands.extend([elem.source for elem in expr.elements])

        if hasattr(statement, "expression"):
            expr = statement.expression
            if isinstance(expr, BinaryExpression):
                operators.extend([op.source for op in expr.operators])
                operands.extend([elem.source for elem in expr.elements])
            elif isinstance(expr, UnaryExpression):
                operators.append(expr.ts_node.type)
                operands.append(expr.argument.source)
            elif isinstance(expr, ComparisonExpression):
                operators.extend([op.source for op in expr.operators])
                operands.extend([elem.source for elem in expr.elements])

    return operators, operands

def calculate_halstead_volume(operators, operands):
    n1 = len(set(operators))
    n2 = len(set(operands))

    N1 = len(operators)
    N2 = len(operands)

    N = N1 + N2
    n = n1 + n2

    if n > 0:
        volume = N * math.log2(n)
        return volume, N1, N2, n1, n2
    return 0, N1, N2, n1, n2

def count_lines(source: str):
    """Count different types of lines in source code."""
    if not source.strip():
        return 0, 0, 0, 0

    lines = [line.strip() for line in source.splitlines()]
    loc = len(lines)
    sloc = len([line for line in lines if line])

    in_multiline = False
    comments = 0
    code_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]
        code_part = line
        if not in_multiline and "#" in line:
            comment_start = line.find("#")
            if not re.search(r'["\'].*#.*["\']', line[:comment_start]):
                code_part = line[:comment_start].strip()
                if line[comment_start:].strip():
                    comments += 1

        if ('"""' in line or "'''" in line) and not (
            line.count('"""') % 2 == 0 or line.count("'''") % 2 == 0
        ):
            if in_multiline:
                in_multiline = False
                comments += 1
            else:
                in_multiline = True
                comments += 1
                if line.strip().startswith('"""') or line.strip().startswith("'''"):
                    code_part = ""
        elif in_multiline:
            comments += 1
            code_part = ""
        elif line.strip().startswith("#"):
            comments += 1
            code_part = ""

        if code_part.strip():
            code_lines.append(code_part)

        i += 1

    lloc = 0
    continued_line = False
    for line in code_lines:
        if continued_line:
            if not any(line.rstrip().endswith(c) for c in ("\\", ",", "{", "[", "(")):
                continued_line = False
            continue

        lloc += len([stmt for stmt in line.split(";") if stmt.strip()])

        if any(line.rstrip().endswith(c) for c in ("\\", ",", "{", "[", "(")):
            continued_line = True

    return loc, lloc, sloc, comments

def calculate_maintainability_index(
    halstead_volume: float, cyclomatic_complexity: float, loc: int
) -> int:
    """Calculate the normalized maintainability index for a given function."""
    if loc <= 0:
        return 100

    try:
        raw_mi = (
            171
            - 5.2 * math.log(max(1, halstead_volume))
            - 0.23 * cyclomatic_complexity
            - 16.2 * math.log(max(1, loc))
        )
        normalized_mi = max(0, min(100, raw_mi * 100 / 171))
        return int(normalized_mi)
    except (ValueError, TypeError):
        return 0

def get_maintainability_rank(mi_score: float) -> str:
    """Convert maintainability index score to a letter grade."""
    if mi_score >= 85:
        return "A"
    elif mi_score >= 65:
        return "B"
    elif mi_score >= 45:
        return "C"
    elif mi_score >= 25:
        return "D"
    else:
        return "F"

def get_github_repo_description(repo_url):
    api_url = f"https://api.github.com/repos/{repo_url}"

    response = requests.get(api_url)

    if response.status_code == 200:
        repo_data = response.json()
        return repo_data.get("description", "No description available")
    else:
        return ""

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

