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
from pathlib import Path
from codegen.sdk.core.class_definition import Class
from codegen.sdk.core.codebase import Codebase
from codegen.sdk.core.external_module import ExternalModule
from codegen.sdk.core.file import SourceFile
from codegen.sdk.core.function import Function
from codegen.sdk.core.import_resolution import Import
from codegen.sdk.core.symbol import Symbol
from codegen.sdk.enums import EdgeType, SymbolType

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

# Base models for codebase analysis
class CodebaseStats(BaseModel):
    test_functions_count: int
    test_classes_count: int
    tests_per_file: float
    total_classes: int
    total_functions: int
    total_imports: int
    deepest_inheritance_class: Optional[Dict]
    recursive_functions: List[str]
    most_called_function: Dict
    function_with_most_calls: Dict
    unused_functions: List[Dict]
    dead_code: List[Dict]

class FileTestStats(BaseModel):
    filepath: str
    test_class_count: int
    file_length: int
    function_count: int

class FunctionContext(BaseModel):
    implementation: Dict
    dependencies: List[Dict]
    usages: List[Dict]

# Models for extended analysis
class TestAnalysis(BaseModel):
    total_test_functions: int
    total_test_classes: int
    tests_per_file: float
    top_test_files: List[Dict[str, Any]]  # Changed from 'any' to 'Any'

class FunctionAnalysis(BaseModel):
    total_functions: int
    most_called_function: Dict[str, Any]
    function_with_most_calls: Dict[str, Any]
    recursive_functions: List[str]
    unused_functions: List[Dict[str, str]]
    dead_code: List[Dict[str, str]]

class ClassAnalysis(BaseModel):
    total_classes: int
    deepest_inheritance: Optional[Dict[str, Any]]
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
    repo_structure: Dict[str, Any]

class RepoRequest(BaseModel):
    repo_url: str

class Symbol(BaseModel):
    id: str
    name: str
    type: str  # 'function', 'class', or 'variable'
    filepath: str
    start_line: int
    end_line: int
    issues: Optional[List[Dict[str, str]]] = None

class FileNode(BaseModel):
    name: str
    type: str  # 'file' or 'directory'
    path: str
    issues: Optional[Dict[str, int]] = None
    symbols: Optional[List[Symbol]] = None
    children: Optional[Dict[str, 'FileNode']] = None

class AnalysisResponse(BaseModel):
    # Basic stats
    repo_url: str
    description: str
    num_files: int
    num_functions: int
    num_classes: int
    
    # Line metrics
    line_metrics: Dict[str, Dict[str, float]]
    
    # Complexity metrics
    cyclomatic_complexity: Dict[str, float]
    depth_of_inheritance: Dict[str, float]
    halstead_metrics: Dict[str, int]
    maintainability_index: Dict[str, int]
    
    # Git metrics
    monthly_commits: Dict[str, int]
    
    # Repository structure with symbols
    repo_structure: FileNode

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

def build_repo_structure(files, file_issues, file_symbols) -> Dict:
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

def get_detailed_symbol_context(symbol: Symbol) -> Dict[str, Any]:
    """Get detailed context for any symbol type."""
    base_info = {
        'id': str(hash(symbol.name + symbol.filepath)),
        'name': symbol.name,
        'type': symbol.__class__.__name__,
        'filepath': symbol.filepath,
        'start_line': symbol.start_point[0] if hasattr(symbol, 'start_point') else 0,
        'end_line': symbol.end_point[0] if hasattr(symbol, 'end_point') else 0,
        'source': symbol.source if hasattr(symbol, 'source') else None,
    }

    # Get usage statistics
    usages = symbol.symbol_usages
    imported_symbols = [x.imported_symbol for x in usages if isinstance(x, Import)]
    
    usage_stats = {
        'total_usages': len(usages),
        'usage_breakdown': {
            'functions': len([x for x in usages if isinstance(x, Symbol) and x.symbol_type == SymbolType.Function]),
            'classes': len([x for x in usages if isinstance(x, Symbol) and x.symbol_type == SymbolType.Class]),
            'global_vars': len([x for x in usages if isinstance(x, Symbol) and x.symbol_type == SymbolType.GlobalVar]),
            'interfaces': len([x for x in usages if isinstance(x, Symbol) and x.symbol_type == SymbolType.Interface])
        },
        'imports': {
            'total': len(imported_symbols),
            'breakdown': {
                'functions': len([x for x in imported_symbols if isinstance(x, Symbol) and x.symbol_type == SymbolType.Function]),
                'classes': len([x for x in imported_symbols if isinstance(x, Symbol) and x.symbol_type == SymbolType.Class]),
                'global_vars': len([x for x in imported_symbols if isinstance(x, Symbol) and x.symbol_type == SymbolType.GlobalVar]),
                'interfaces': len([x for x in imported_symbols if isinstance(x, Symbol) and x.symbol_type == SymbolType.Interface]),
                'external_modules': len([x for x in imported_symbols if isinstance(x, ExternalModule)]),
                'files': len([x for x in imported_symbols if isinstance(x, SourceFile)])
            }
        }
    }

    # Add type-specific information
    if isinstance(symbol, Function):
        base_info.update({
            'function_info': {
                'return_statements': len(symbol.return_statements),
                'parameters': [
                    {
                        'name': p.name,
                        'type': p.type if hasattr(p, 'type') else None,
                        'default_value': p.default_value if hasattr(p, 'default_value') else None
                    }
                    for p in symbol.parameters
                ],
                'function_calls': [
                    {
                        'name': call.name,
                        'args': [arg.source for arg in call.args] if hasattr(call, 'args') else [],
                        'line': call.start_point[0] if hasattr(call, 'start_point') else 0
                    }
                    for call in symbol.function_calls
                ],
                'call_sites': [
                    {
                        'caller': site.parent_function.name if hasattr(site, 'parent_function') else None,
                        'line': site.start_point[0] if hasattr(site, 'start_point') else 0,
                        'file': site.filepath if hasattr(site, 'filepath') else None
                    }
                    for site in symbol.call_sites
                ],
                'decorators': [d.source for d in symbol.decorators] if hasattr(symbol, 'decorators') else [],
                'dependencies': [
                    {
                        'name': dep.name,
                        'type': dep.__class__.__name__,
                        'filepath': dep.filepath if hasattr(dep, 'filepath') else None
                    }
                    for dep in symbol.dependencies
                ] if hasattr(symbol, 'dependencies') else []
            }
        })

        # Add complexity metrics
        if hasattr(symbol, 'code_block'):
            complexity = calculate_cyclomatic_complexity(symbol)
            operators, operands = get_operators_and_operands(symbol)
            volume, N1, N2, n1, n2 = calculate_halstead_volume(operators, operands)
            loc = len(symbol.code_block.source.splitlines())
            mi_score = calculate_maintainability_index(volume, complexity, loc)

            base_info['metrics'] = {
                'cyclomatic_complexity': {
                    'value': complexity,
                    'rank': cc_rank(complexity)
                },
                'halstead_metrics': {
                    'volume': volume,
                    'unique_operators': n1,
                    'unique_operands': n2,
                    'total_operators': N1,
                    'total_operands': N2
                },
                'maintainability_index': {
                    'value': mi_score,
                    'rank': get_maintainability_rank(mi_score)
                },
                'lines_of_code': {
                    'total': loc,
                    'code': len([l for l in symbol.code_block.source.splitlines() if l.strip()]),
                    'comments': len([l for l in symbol.code_block.source.splitlines() if l.strip().startswith('#')])
                }
            }

    elif isinstance(symbol, Class):
        base_info.update({
            'class_info': {
                'parent_classes': symbol.parent_class_names,
                'methods': [
                    {
                        'name': m.name,
                        'parameters': len(m.parameters) if hasattr(m, 'parameters') else 0,
                        'line': m.start_point[0] if hasattr(m, 'start_point') else 0
                    }
                    for m in symbol.methods
                ],
                'attributes': [
                    {
                        'name': a.name,
                        'type': a.type if hasattr(a, 'type') else None,
                        'line': a.start_point[0] if hasattr(a, 'start_point') else 0
                    }
                    for a in symbol.attributes
                ],
                'decorators': [d.source for d in symbol.decorators] if hasattr(symbol, 'decorators') else [],
                'dependencies': [
                    {
                        'name': dep.name,
                        'type': dep.__class__.__name__,
                        'filepath': dep.filepath if hasattr(dep, 'filepath') else None
                    }
                    for dep in symbol.dependencies
                ] if hasattr(symbol, 'dependencies') else [],
                'inheritance_depth': len(symbol.superclasses) if hasattr(symbol, 'superclasses') else 0,
                'inheritance_chain': [
                    {
                        'name': s.name,
                        'filepath': s.filepath if hasattr(s, 'filepath') else None
                    }
                    for s in symbol.superclasses
                ] if hasattr(symbol, 'superclasses') else []
            }
        })

    base_info['usage_stats'] = usage_stats
    return base_info

@fastapi_app.get("/api/codebase/stats")
async def get_codebase_stats(codebase_id: str) -> CodebaseStats:
    """Get comprehensive statistics about the codebase."""
    try:
        # Filter test functions and classes
        test_functions = [x for x in codebase.functions if x.name.startswith('test_')]
        test_classes = [x for x in codebase.classes if x.name.startswith('Test')]
        
        # Calculate tests per file
        tests_per_file = len(test_functions) / len(codebase.files) if codebase.files else 0
        
        # Find class with deepest inheritance
        deepest_class = None
        if codebase.classes:
            deepest = max(codebase.classes, key=lambda x: len(x.superclasses))
            deepest_class = {
                'name': deepest.name,
                'depth': len(deepest.superclasses),
                'chain': [s.name for s in deepest.superclasses]
            }
        
        # Find recursive functions
        recursive = [f.name for f in codebase.functions 
                    if any(call.name == f.name for call in f.function_calls)][:5]
        
        # Find most called function
        most_called = max(codebase.functions, key=lambda f: len(f.call_sites))
        most_called_info = {
            'name': most_called.name,
            'call_count': len(most_called.call_sites),
            'callers': [{'function': call.parent_function.name, 
                        'line': call.start_point[0]} 
                       for call in most_called.call_sites]
        }
        
        # Find function with most calls
        most_calls = max(codebase.functions, key=lambda f: len(f.function_calls))
        most_calls_info = {
            'name': most_calls.name,
            'calls_count': len(most_calls.function_calls),
            'called_functions': [call.name for call in most_calls.function_calls]
        }
        
        # Find unused functions
        unused = [{'name': f.name, 'filepath': f.filepath} 
                 for f in codebase.functions if len(f.call_sites) == 0]
        
        # Find dead code
        dead_code = find_dead_code(codebase)
        dead_code_info = [{'name': f.name, 'filepath': f.filepath} for f in dead_code]
        
        return CodebaseStats(
            test_functions_count=len(test_functions),
            test_classes_count=len(test_classes),
            tests_per_file=tests_per_file,
            total_classes=len(codebase.classes),
            total_functions=len(codebase.functions),
            total_imports=len(codebase.imports),
            deepest_inheritance_class=deepest_class,
            recursive_functions=recursive,
            most_called_function=most_called_info,
            function_with_most_calls=most_calls_info,
            unused_functions=unused,
            dead_code=dead_code_info
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@fastapi_app.get("/api/codebase/test-files")
async def get_test_file_stats(codebase_id: str) -> List[FileTestStats]:
    """Get statistics about test files in the codebase."""
    try:
        test_classes = [x for x in codebase.classes if x.name.startswith('Test')]
        file_test_counts = Counter([x.file for x in test_classes])
        
        stats = []
        for file, num_tests in file_test_counts.most_common()[:5]:
            stats.append(FileTestStats(
                filepath=file.filepath,
                test_class_count=num_tests,
                file_length=len(file.source),
                function_count=len(file.functions)
            ))
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@fastapi_app.get("/api/function/{function_id}/context")
async def get_function_context(function_id: str) -> FunctionContext:
    """Get detailed context for a specific function."""
    try:
        function = get_function_by_id(function_id)  # You'll need to implement this
        
        context = {
            "implementation": {
                "source": function.source,
                "filepath": function.filepath
            },
            "dependencies": [],
            "usages": []
        }
        
        # Add dependencies
        for dep in function.dependencies:
            if isinstance(dep, Import):
                dep = hop_through_imports(dep)  # You'll need to implement this
            context["dependencies"].append({
                "source": dep.source,
                "filepath": dep.filepath
            })
        
        # Add usages
        for usage in function.usages:
            context["usages"].append({
                "source": usage.usage_symbol.source,
                "filepath": usage.usage_symbol.filepath
            })
        
        return FunctionContext(**context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@fastapi_app.get("/api/function/{function_id}/call-chain")
async def get_function_call_chain(function_id: str) -> List[str]:
    """Get the maximum call chain for a function."""
    try:
        function = get_function_by_id(function_id)
        chain = get_max_call_chain(function)
        return [f.name for f in chain]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@fastapi_app.post("/analyze_repo")
async def analyze_repo(request: RepoRequest) -> AnalysisResponse:
    """Single entry point for repository analysis."""
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

    # Analyze files and collect symbols
    file_issues = {}
    file_symbols = {}
    
    for file in codebase.files:
        # Line metrics
        loc, lloc, sloc, comments = count_lines(file.source)
        total_loc += loc
        total_lloc += lloc
        total_sloc += sloc
        total_comments += comments

        # Analyze issues
        issues = analyze_file_issues(file)
        if any(len(v) > 0 for v in issues.values()):
            file_issues[file.filepath] = issues

        # Collect symbols
        symbols = []
        
        # Add functions as symbols
        for func in file.functions:
            issues = []
            
            # Check for issues
            if not any(func.name in str(usage) for usage in func.usages):
                issues.append({
                    'type': 'minor',
                    'message': f'Unused function'
                })
            
            if hasattr(func, 'code_block'):
                code = func.code_block.source
                if 'None' in code and not any(s in code for s in ['is None', '== None', '!= None']):
                    issues.append({
                        'type': 'critical',
                        'message': f'Potential unsafe null reference'
                    })
                
                if 'TODO' in code or 'FIXME' in code:
                    issues.append({
                        'type': 'major',
                        'message': f'Incomplete implementation'
                    })

            symbols.append(Symbol(
                id=str(hash(func.name + file.filepath)),
                name=func.name,
                type='function',
                filepath=file.filepath,
                start_line=func.start_point[0] if hasattr(func, 'start_point') else 0,
                end_line=func.end_point[0] if hasattr(func, 'end_point') else 0,
                issues=issues if issues else None
            ))
        
        # Add classes as symbols
        for cls in file.classes:
            symbols.append(Symbol(
                id=str(hash(cls.name + file.filepath)),
                name=cls.name,
                type='class',
                filepath=file.filepath,
                start_line=cls.start_point[0] if hasattr(cls, 'start_point') else 0,
                end_line=cls.end_point[0] if hasattr(cls, 'end_point') else 0
            ))
        
        if symbols:
            file_symbols[file.filepath] = symbols

    # Build repository structure with symbols
    repo_structure = build_repo_structure(codebase.files, file_issues, file_symbols)

    # Calculate metrics
    callables = codebase.functions + [m for c in codebase.classes for m in c.methods]
    num_callables = 0
    
    for func in callables:
        if not hasattr(func, "code_block"):
            continue

        complexity = calculate_cyclomatic_complexity(func)
        operators, operands = get_operators_and_operands(func)
        volume, N1, N2, n1, n2 = calculate_halstead_volume(operators, operands)
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

    return AnalysisResponse(
        repo_url=repo_url,
        description=desc,
        num_files=num_files,
        num_functions=num_functions,
        num_classes=num_classes,
        line_metrics={
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
        cyclomatic_complexity={
            "average": total_complexity / num_callables if num_callables > 0 else 0,
        },
        depth_of_inheritance={
            "average": total_doi / len(codebase.classes) if codebase.classes else 0,
        },
        halstead_metrics={
            "total_volume": int(total_volume),
            "average_volume": int(total_volume / num_callables)
            if num_callables > 0
            else 0,
        },
        maintainability_index={
            "average": int(total_mi / num_callables) if num_callables > 0 else 0,
        },
        monthly_commits=monthly_commits,
        repo_structure=repo_structure
    )

@fastapi_app.get("/function/{function_id}/call-chain")
async def get_function_call_chain(function_id: str) -> List[str]:
    """Get the maximum call chain for a function."""
    try:
        function = get_function_by_id(function_id)
        chain = get_max_call_chain(function)
        return [f.name for f in chain]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@fastapi_app.get("/function/{function_id}/context")
async def get_function_context(function_id: str) -> FunctionContext:
    """Get detailed context for a specific function."""
    try:
        function = get_function_by_id(function_id)
        
        context = {
            "implementation": {
                "source": function.source,
                "filepath": function.filepath
            },
            "dependencies": [],
            "usages": []
        }
        
        # Add dependencies
        for dep in function.dependencies:
            if isinstance(dep, Import):
                dep = hop_through_imports(dep)
            context["dependencies"].append({
                "source": dep.source,
                "filepath": dep.filepath
            })
        
        # Add usages
        for usage in function.usages:
            context["usages"].append({
                "source": usage.usage_symbol.source,
                "filepath": usage.usage_symbol.filepath
            })
        
        return FunctionContext(**context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@fastapi_app.get("/symbol/{symbol_id}/context")
async def get_symbol_context(symbol_id: str) -> Dict[str, Any]:
    """Get detailed context for any symbol."""
    try:
        symbol = get_symbol_by_id(symbol_id)  # You'll need to implement this
        return get_detailed_symbol_context(symbol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Helper function to get a function by ID (you'll need to implement this)
def get_function_by_id(function_id: str):
    # Implementation depends on how you store/retrieve functions
    pass

# Helper function to resolve imports (you'll need to implement this)
def hop_through_imports(import_symbol):
    # Implementation depends on how you handle imports
    pass

@app.function(image=image)
@modal.asgi_app()
def fastapi_modal_app():
    return fastapi_app

if __name__ == "__main__":
    import uvicorn
    import socket
    
    def find_available_port(start_port=8000, max_port=8100):
        """Find an available port starting from start_port"""
        for port in range(start_port, max_port):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('0.0.0.0', port))
                    return port
            except OSError:
                continue
        raise RuntimeError(f"No available ports found between {start_port} and {max_port}")
    
    # Find an available port
    port = find_available_port()
    print(f"🚀 Starting FastAPI server on http://localhost:{port}")
    print(f"📚 API documentation available at http://localhost:{port}/docs")
    
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)
