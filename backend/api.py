from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List, Tuple, Any, Optional
from graph_sitter.core import Codebase
from graph_sitter.codebase.codebase_analysis import (
    get_codebase_summary,
    get_file_summary,
    get_class_summary,
    get_function_summary,
    get_symbol_summary
)
from graph_sitter.codebase.codebase_ai import generate_context
import math
import re
import requests
from datetime import datetime, timedelta
import subprocess
import os
import tempfile
from fastapi.middleware.cors import CORSMiddleware
import modal

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "graph-sitter", "fastapi", "uvicorn", "gitpython", "requests", "pydantic", "datetime"
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
    if not hasattr(cls, 'parent_classes') or not cls.parent_classes:
        return 0
    
    max_depth = 0
    for parent in cls.parent_classes:
        if hasattr(parent, 'parent_classes'):
            depth = 1 + calculate_doi(parent)
            max_depth = max(max_depth, depth)
        else:
            max_depth = max(max_depth, 1)
    
    return max_depth


def calculate_maintainability_index(
    halstead_volume: float, cyclomatic_complexity: float, lines_of_code: float
) -> float:
    """Calculate the Maintainability Index for a function."""
    if lines_of_code == 0:
        return 0
    
    # Maintainability Index formula
    mi = (
        171
        - 5.2 * math.log(halstead_volume)
        - 0.23 * cyclomatic_complexity
        - 16.2 * math.log(lines_of_code)
    )
    
    return max(0, mi)  # Ensure non-negative


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


def get_github_repo_description(repo_url):
    """Get repository description from GitHub API."""
    try:
        # Extract owner and repo name from URL
        parts = repo_url.replace("https://github.com/", "").split("/")
        if len(parts) >= 2:
            owner, repo = parts[0], parts[1]
            api_url = f"https://api.github.com/repos/{owner}/{repo}"
            response = requests.get(api_url)
            if response.status_code == 200:
                repo_data = response.json()
                return repo_data.get("description", "No description available")
        return repo_data.get("description", "No description available")
    else:
        return ""


def analyze_codebase_summary(codebase: Codebase) -> str:
    """Get high-level statistical overview of the entire codebase."""
    try:
        return get_codebase_summary(codebase)
    except Exception as e:
        return f"Error analyzing codebase summary: {str(e)}"


def analyze_file_summaries(codebase: Codebase) -> List[Dict[str, Any]]:
    """Analyze all source files and return their summaries."""
    file_summaries = []
    try:
        for file in codebase.source_files:
            summary = get_file_summary(file)
            file_summaries.append({
                "file_path": file.path if hasattr(file, 'path') else str(file),
                "summary": summary
            })
    except Exception as e:
        file_summaries.append({
            "error": f"Error analyzing file summaries: {str(e)}"
        })
    return file_summaries


def analyze_class_summaries(codebase: Codebase) -> List[Dict[str, Any]]:
    """Analyze all classes and return their summaries."""
    class_summaries = []
    try:
        for file in codebase.source_files:
            if hasattr(file, 'classes'):
                for cls in file.classes:
                    summary = get_class_summary(cls)
                    class_summaries.append({
                        "class_name": cls.name if hasattr(cls, 'name') else str(cls),
                        "file_path": file.path if hasattr(file, 'path') else str(file),
                        "summary": summary
                    })
    except Exception as e:
        class_summaries.append({
            "error": f"Error analyzing class summaries: {str(e)}"
        })
    return class_summaries


def analyze_function_summaries(codebase: Codebase) -> List[Dict[str, Any]]:
    """Analyze all functions and return their summaries."""
    function_summaries = []
    try:
        for file in codebase.source_files:
            if hasattr(file, 'functions'):
                for func in file.functions:
                    summary = get_function_summary(func)
                    function_summaries.append({
                        "function_name": func.name if hasattr(func, 'name') else str(func),
                        "file_path": file.path if hasattr(file, 'path') else str(file),
                        "summary": summary
                    })
    except Exception as e:
        function_summaries.append({
            "error": f"Error analyzing function summaries: {str(e)}"
        })
    return function_summaries


def analyze_symbol_summaries(codebase: Codebase) -> List[Dict[str, Any]]:
    """Analyze all symbols and return their summaries."""
    symbol_summaries = []
    try:
        # Get all symbols from the codebase
        all_symbols = []
        for file in codebase.source_files:
            if hasattr(file, 'symbols'):
                all_symbols.extend(file.symbols)
        
        for symbol in all_symbols:
            summary = get_symbol_summary(symbol)
            symbol_summaries.append({
                "symbol_name": symbol.name if hasattr(symbol, 'name') else str(symbol),
                "symbol_type": type(symbol).__name__,
                "summary": summary
            })
    except Exception as e:
        symbol_summaries.append({
            "error": f"Error analyzing symbol summaries: {str(e)}"
        })
    return symbol_summaries


def generate_ai_context(codebase: Codebase, elements: List[Any] = None) -> str:
    """Generate AI-formatted context strings for code elements."""
    try:
        if elements is None:
            # Generate context for the entire codebase
            return generate_context(codebase)
        else:
            # Generate context for specific elements
            return generate_context(elements)
    except Exception as e:
        return f"Error generating AI context: {str(e)}"


class RepoRequest(BaseModel):
    repo_url: str


@fastapi_app.post("/analyze_repo")
async def analyze_repo(request: RepoRequest) -> Dict[str, Any]:
    """Analyze a repository and return comprehensive metrics using graph-sitter."""
    repo_url = request.repo_url
    
    try:
        # Initialize graph-sitter codebase
        codebase = Codebase(repo_path=repo_url)
        
        # Get basic repository information
        desc = get_github_repo_description(repo_url)
        monthly_commits = get_monthly_commits(repo_url)
        
        # Perform graph-sitter analysis
        codebase_summary = analyze_codebase_summary(codebase)
        file_summaries = analyze_file_summaries(codebase)
        class_summaries = analyze_class_summaries(codebase)
        function_summaries = analyze_function_summaries(codebase)
        symbol_summaries = analyze_symbol_summaries(codebase)
        ai_context = generate_ai_context(codebase)
        
        # Extract basic metrics for backward compatibility
        num_files = len(codebase.source_files) if hasattr(codebase, 'source_files') else 0
        num_functions = len(function_summaries)
        num_classes = len(class_summaries)
        
        # Calculate basic line metrics from codebase summary
        total_loc = total_lloc = total_sloc = total_comments = 0
        try:
            for file in codebase.source_files:
                if hasattr(file, 'source'):
                    loc, lloc, sloc, comments = count_lines(file.source)
                    total_loc += loc
                    total_lloc += lloc
                    total_sloc += sloc
                    total_comments += comments
        except Exception as e:
            print(f"Error calculating line metrics: {e}")
        
        # Calculate DOI (Depth of Inheritance) for classes
        total_doi = 0
        doi_count = 0
        try:
            for file in codebase.source_files:
                if hasattr(file, 'classes'):
                    for cls in file.classes:
                        doi = calculate_doi(cls)
                        total_doi += doi
                        doi_count += 1
        except Exception as e:
            print(f"Error calculating DOI: {e}")
        
        average_doi = total_doi / doi_count if doi_count > 0 else 0
        
        # Build comprehensive response with both legacy and new analysis
        results = {
            "repo_url": repo_url,
            "description": desc,
            "monthly_commits": monthly_commits,
            
            # Legacy metrics for backward compatibility
            "line_metrics": {
                "total": {
                    "loc": total_loc,
                    "lloc": total_lloc,
                    "sloc": total_sloc,
                    "comments": total_comments,
                    "comment_density": (total_comments / total_loc * 100) if total_loc > 0 else 0,
                },
            },
            "num_files": num_files,
            "num_functions": num_functions,
            "num_classes": num_classes,
            
            # New graph-sitter analysis results
            "graph_sitter_analysis": {
                "codebase_summary": codebase_summary,
                "file_summaries": file_summaries,
                "class_summaries": class_summaries,
                "function_summaries": function_summaries,
                "symbol_summaries": symbol_summaries,
                "ai_context": ai_context
            },
            
            # Placeholder values for metrics that require more complex analysis
            "cyclomatic_complexity": {"average": 0},
            "depth_of_inheritance": {"average": average_doi},  # Now using actual DOI calculation
            "halstead_metrics": {"total_volume": 0, "average_volume": 0},
            "maintainability_index": {"average": 0},
        }
        
        return results
        
    except Exception as e:
        # Return error response with basic structure
        return {
            "repo_url": repo_url,
            "error": f"Analysis failed: {str(e)}",
            "description": get_github_repo_description(repo_url),
            "monthly_commits": {},
            "line_metrics": {"total": {"loc": 0, "lloc": 0, "sloc": 0, "comments": 0, "comment_density": 0}},
            "num_files": 0,
            "num_functions": 0,
            "num_classes": 0,
            "graph_sitter_analysis": {
                "codebase_summary": f"Error: {str(e)}",
                "file_summaries": [],
                "class_summaries": [],
                "function_summaries": [],
                "symbol_summaries": [],
                "ai_context": ""
            },
            "cyclomatic_complexity": {"average": 0},
            "depth_of_inheritance": {"average": 0},
            "halstead_metrics": {"total_volume": 0, "average_volume": 0},
            "maintainability_index": {"average": 0},
        }


@app.function(image=image)
@modal.asgi_app()
def fastapi_modal_app():
    return fastapi_app


if __name__ == "__main__":
    app.deploy("analytics-app")
