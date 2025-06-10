from fastapi import FastAPI
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
from dataclasses import dataclass
from collections import defaultdict, Counter
import ast
import networkx as nx

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "codegen", "fastapi", "uvicorn", "gitpython", "requests", "pydantic", "datetime", "networkx"
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


def detect_code_issues(codebase) -> List[CodeIssue]:
    """Detect various code quality issues and errors"""
    issues = []
    
    for file in codebase.files:
        if not file.source.strip():
            continue
            
        # Analyze file-level issues
        file_issues = []
        
        # 1. Complexity Issues
        for func in file.functions:
            if hasattr(func, 'code_block'):
                complexity = calculate_cyclomatic_complexity(func)
                if complexity > 15:
                    file_issues.append(CodeIssue(
                        type="High Complexity",
                        severity="high" if complexity > 25 else "medium",
                        message=f"Function '{func.name}' has high cyclomatic complexity ({complexity})",
                        file_path=file.path,
                        line_number=getattr(func, 'line_number', None),
                        function_name=func.name,
                        suggestion="Consider breaking this function into smaller, more focused functions",
                        impact="High complexity makes code harder to understand, test, and maintain",
                        category="complexity"
                    ))
                
                # Long function detection
                if hasattr(func.code_block, 'source'):
                    lines = len(func.code_block.source.splitlines())
                    if lines > 50:
                        file_issues.append(CodeIssue(
                            type="Long Function",
                            severity="medium" if lines > 100 else "low",
                            message=f"Function '{func.name}' is too long ({lines} lines)",
                            file_path=file.path,
                            line_number=getattr(func, 'line_number', None),
                            function_name=func.name,
                            suggestion="Break this function into smaller, single-purpose functions",
                            impact="Long functions are harder to understand and maintain",
                            category="maintainability"
                        ))
        
        # 2. Security Issues
        dangerous_patterns = [
            (r'eval\s*\(', "Use of eval() function", "critical", "Avoid eval() as it can execute arbitrary code"),
            (r'exec\s*\(', "Use of exec() function", "critical", "Avoid exec() as it can execute arbitrary code"),
            (r'subprocess\.call\s*\([^)]*shell\s*=\s*True', "Shell injection risk", "high", "Avoid shell=True in subprocess calls"),
            (r'pickle\.loads?\s*\(', "Unsafe deserialization", "high", "Use safer serialization methods like json"),
            (r'input\s*\([^)]*\)', "Use of input() function", "medium", "Validate and sanitize user input"),
        ]
        
        for pattern, issue_type, severity, suggestion in dangerous_patterns:
            matches = re.finditer(pattern, file.source, re.IGNORECASE)
            for match in matches:
                line_num = file.source[:match.start()].count('\n') + 1
                file_issues.append(CodeIssue(
                    type=issue_type,
                    severity=severity,
                    message=f"Potential security issue: {issue_type.lower()}",
                    file_path=file.path,
                    line_number=line_num,
                    function_name=None,
                    suggestion=suggestion,
                    impact="Security vulnerabilities can lead to code injection attacks",
                    category="security"
                ))
        
        # 3. Code Style Issues
        style_patterns = [
            (r'^\s*#.*TODO', "TODO comment", "low", "Address TODO items or create proper issues"),
            (r'^\s*#.*FIXME', "FIXME comment", "medium", "Address FIXME items promptly"),
            (r'^\s*#.*HACK', "HACK comment", "medium", "Replace hack with proper solution"),
            (r'print\s*\(', "Debug print statement", "low", "Remove debug prints or use proper logging"),
        ]
        
        for pattern, issue_type, severity, suggestion in style_patterns:
            matches = re.finditer(pattern, file.source, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                line_num = file.source[:match.start()].count('\n') + 1
                file_issues.append(CodeIssue(
                    type=issue_type,
                    severity=severity,
                    message=f"Code style issue: {issue_type.lower()}",
                    file_path=file.path,
                    line_number=line_num,
                    function_name=None,
                    suggestion=suggestion,
                    impact="Code style issues affect readability and maintainability",
                    category="style"
                ))
        
        # 4. Import Issues
        try:
            tree = ast.parse(file.source)
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            # Check for unused imports (simplified)
            for imp in imports:
                if imp not in file.source.replace(f"import {imp}", ""):
                    file_issues.append(CodeIssue(
                        type="Unused Import",
                        severity="low",
                        message=f"Import '{imp}' appears to be unused",
                        file_path=file.path,
                        line_number=None,
                        function_name=None,
                        suggestion="Remove unused imports to keep code clean",
                        impact="Unused imports increase file size and confusion",
                        category="style"
                    ))
        except:
            pass  # Skip AST analysis if parsing fails
        
        issues.extend(file_issues)
    
    return issues

def generate_dependency_graph(codebase) -> Dict[str, Any]:
    """Generate dependency graph data for visualization"""
    graph = nx.DiGraph()
    nodes = []
    edges = []
    
    # Add file nodes
    for file in codebase.files:
        if file.source.strip():
            complexity = sum(calculate_cyclomatic_complexity(func) 
                           for func in file.functions if hasattr(func, 'code_block'))
            
            nodes.append({
                "id": file.path,
                "label": file.path.split('/')[-1],
                "type": "file",
                "size": len(file.source.splitlines()),
                "complexity": complexity,
                "group": file.path.split('/')[0] if '/' in file.path else "root"
            })
            
            # Add function nodes
            for func in file.functions:
                func_id = f"{file.path}::{func.name}"
                nodes.append({
                    "id": func_id,
                    "label": func.name,
                    "type": "function",
                    "size": len(func.code_block.source.splitlines()) if hasattr(func, 'code_block') else 1,
                    "complexity": calculate_cyclomatic_complexity(func) if hasattr(func, 'code_block') else 1,
                    "group": file.path
                })
                
                # Edge from file to function
                edges.append({
                    "from": file.path,
                    "to": func_id,
                    "type": "contains"
                })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "file_count": len([n for n in nodes if n["type"] == "file"]),
            "function_count": len([n for n in nodes if n["type"] == "function"])
        }
    }

def generate_complexity_heatmap(codebase) -> Dict[str, Any]:
    """Generate complexity heatmap data"""
    heatmap_data = []
    
    for file in codebase.files:
        if not file.source.strip():
            continue
            
        file_complexity = 0
        function_complexities = []
        
        for func in file.functions:
            if hasattr(func, 'code_block'):
                complexity = calculate_cyclomatic_complexity(func)
                file_complexity += complexity
                function_complexities.append({
                    "name": func.name,
                    "complexity": complexity,
                    "lines": len(func.code_block.source.splitlines())
                })
        
        heatmap_data.append({
            "file": file.path,
            "total_complexity": file_complexity,
            "avg_complexity": file_complexity / len(file.functions) if file.functions else 0,
            "lines": len(file.source.splitlines()),
            "functions": function_complexities,
            "risk_level": "high" if file_complexity > 50 else "medium" if file_complexity > 20 else "low"
        })
    
    return {
        "files": heatmap_data,
        "summary": {
            "total_files": len(heatmap_data),
            "high_risk_files": len([f for f in heatmap_data if f["risk_level"] == "high"]),
            "medium_risk_files": len([f for f in heatmap_data if f["risk_level"] == "medium"]),
            "low_risk_files": len([f for f in heatmap_data if f["risk_level"] == "low"])
        }
    }

def generate_file_metrics(codebase) -> List[Dict[str, Any]]:
    """Generate detailed metrics for each file"""
    file_metrics = []
    
    for file in codebase.files:
        if not file.source.strip():
            continue
            
        loc, lloc, sloc, comments = count_lines(file.source)
        
        # Calculate file-level complexity
        total_complexity = sum(calculate_cyclomatic_complexity(func) 
                             for func in file.functions if hasattr(func, 'code_block'))
        
        # Calculate maintainability (simplified)
        maintainability = max(0, 100 - (total_complexity * 2) - (loc / 10))
        
        file_metrics.append({
            "path": file.path,
            "name": file.path.split('/')[-1],
            "loc": loc,
            "sloc": sloc,
            "lloc": lloc,
            "comments": comments,
            "comment_ratio": (comments / loc * 100) if loc > 0 else 0,
            "complexity": total_complexity,
            "maintainability": maintainability,
            "functions": len(file.functions),
            "classes": len(file.classes),
            "function_names": [func.name for func in file.functions],
            "class_names": [cls.name for cls in file.classes],
            "risk_score": min(100, total_complexity + (loc / 100))
        })
    
    return file_metrics


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


class RepoRequest(BaseModel):
    repo_url: str


@dataclass
class CodeIssue:
    """Represents a code quality issue or error"""
    type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    file_path: str
    line_number: Optional[int]
    function_name: Optional[str]
    suggestion: str
    impact: str
    category: str  # 'complexity', 'maintainability', 'security', 'performance', 'style'

@dataclass
class FileMetrics:
    """Detailed metrics for a single file"""
    path: str
    loc: int
    complexity: float
    maintainability: float
    issues: List[CodeIssue]
    functions: List[str]
    classes: List[str]
    imports: List[str]
    dependencies: List[str]

@dataclass
class DependencyNode:
    """Node in dependency graph"""
    name: str
    type: str  # 'file', 'function', 'class'
    size: int  # lines of code or complexity
    dependencies: List[str]
    dependents: List[str]

@fastapi_app.post("/analyze_repo")
async def analyze_repo(request: RepoRequest) -> Dict[str, Any]:
    """Analyze a repository and return comprehensive metrics with rich visualizations."""
    repo_url = request.repo_url
    codebase = Codebase.from_repo(repo_url)
    
    num_files = len(codebase.files(extensions="*"))
    num_functions = len(codebase.functions)
    num_classes = len(codebase.classes)
    
    total_loc = total_lloc = total_sloc = total_comments = 0
    total_complexity = 0
    total_volume = 0
    total_mi = 0
    total_doi = 0
    
    monthly_commits = get_monthly_commits(repo_url)
    
    # Generate comprehensive analysis data
    code_issues = detect_code_issues(codebase)
    dependency_graph = generate_dependency_graph(codebase)
    complexity_heatmap = generate_complexity_heatmap(codebase)
    file_metrics = generate_file_metrics(codebase)
    
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
    
    # Calculate issue statistics
    issue_stats = {
        "total": len(code_issues),
        "critical": len([i for i in code_issues if i.severity == "critical"]),
        "high": len([i for i in code_issues if i.severity == "high"]),
        "medium": len([i for i in code_issues if i.severity == "medium"]),
        "low": len([i for i in code_issues if i.severity == "low"]),
        "by_category": {
            "security": len([i for i in code_issues if i.category == "security"]),
            "complexity": len([i for i in code_issues if i.category == "complexity"]),
            "maintainability": len([i for i in code_issues if i.category == "maintainability"]),
            "style": len([i for i in code_issues if i.category == "style"]),
            "performance": len([i for i in code_issues if i.category == "performance"])
        }
    }
    
    # Calculate overall quality score
    avg_maintainability = int(total_mi / num_callables) if num_callables > 0 else 0
    quality_score = max(0, min(100, avg_maintainability - (issue_stats["critical"] * 10) - (issue_stats["high"] * 5)))
    
    results = {
        "repo_url": repo_url,
        "description": desc,
        "quality_score": quality_score,
        "quality_grade": get_maintainability_rank(quality_score),
        
        # Basic metrics (existing)
        "line_metrics": {
            "total": {
                "loc": total_loc,
                "lloc": total_lloc,
                "sloc": total_sloc,
                "comments": total_comments,
                "comment_density": (total_comments / total_loc * 100) if total_loc > 0 else 0,
            },
        },
        "cyclomatic_complexity": {
            "average": total_complexity / num_callables if num_callables > 0 else 0,
            "total": total_complexity,
        },
        "depth_of_inheritance": {
            "average": total_doi / len(codebase.classes) if codebase.classes else 0,
        },
        "halstead_metrics": {
            "total_volume": int(total_volume),
            "average_volume": int(total_volume / num_callables) if num_callables > 0 else 0,
        },
        "maintainability_index": {
            "average": avg_maintainability,
        },
        "num_files": num_files,
        "num_functions": num_functions,
        "num_classes": num_classes,
        "monthly_commits": monthly_commits,
        
        # Enhanced analysis data
        "issues": {
            "statistics": issue_stats,
            "details": [
                {
                    "type": issue.type,
                    "severity": issue.severity,
                    "message": issue.message,
                    "file_path": issue.file_path,
                    "line_number": issue.line_number,
                    "function_name": issue.function_name,
                    "suggestion": issue.suggestion,
                    "impact": issue.impact,
                    "category": issue.category
                }
                for issue in code_issues[:50]  # Limit to first 50 issues for performance
            ]
        },
        
        # Rich visualization data
        "visualizations": {
            "dependency_graph": dependency_graph,
            "complexity_heatmap": complexity_heatmap,
            "file_metrics": file_metrics,
            "risk_distribution": {
                "high_risk_files": len([f for f in file_metrics if f["risk_score"] > 70]),
                "medium_risk_files": len([f for f in file_metrics if 30 <= f["risk_score"] <= 70]),
                "low_risk_files": len([f for f in file_metrics if f["risk_score"] < 30])
            }
        }
    }
    
    return results


@app.function(image=image)
@modal.asgi_app()
def fastapi_modal_app():
    return fastapi_app


if __name__ == "__main__":
    app.deploy("analytics-app")
