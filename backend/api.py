from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Tuple, Any, Optional, Union
import graph_sitter
from graph_sitter.core.class_definition import Class
from graph_sitter.core.codebase import Codebase
from graph_sitter.core.external_module import ExternalModule
from graph_sitter.core.file import SourceFile
from graph_sitter.core.function import Function
from graph_sitter.core.import_resolution import Import
from graph_sitter.core.symbol import Symbol
from graph_sitter.enums import EdgeType, SymbolType
from graph_sitter.statements.for_loop_statement import ForLoopStatement
from graph_sitter.core.statements.if_block_statement import IfBlockStatement
from graph_sitter.core.statements.try_catch_statement import TryCatchStatement
from graph_sitter.core.statements.while_statement import WhileStatement
from graph_sitter.core.expressions.binary_expression import BinaryExpression
from graph_sitter.core.expressions.unary_expression import UnaryExpression
from graph_sitter.core.expressions.comparison_expression import ComparisonExpression
import math
import re
import requests
from datetime import datetime, timedelta
import subprocess
import os
import tempfile
from fastapi.middleware.cors import CORSMiddleware
import modal
import ast
import json
from collections import defaultdict, Counter
from enum import Enum
import hashlib

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "codegen", "fastapi", "uvicorn", "gitpython", "requests", "pydantic", "datetime", "networkx", "pygments"
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

def get_codebase_summary(codebase: Codebase) -> str:
    node_summary = f"""Contains {len(codebase.ctx.get_nodes())} nodes
- {len(list(codebase.files))} files
- {len(list(codebase.imports))} imports
- {len(list(codebase.external_modules))} external_modules
- {len(list(codebase.symbols))} symbols
\t- {len(list(codebase.classes))} classes
\t- {len(list(codebase.functions))} functions
\t- {len(list(codebase.global_vars))} global_vars
\t- {len(list(codebase.interfaces))} interfaces
"""
    edge_summary = f"""Contains {len(codebase.ctx.edges)} edges
- {len([x for x in codebase.ctx.edges if x[2].type == EdgeType.SYMBOL_USAGE])} symbol -> used symbol
- {len([x for x in codebase.ctx.edges if x[2].type == EdgeType.IMPORT_SYMBOL_RESOLUTION])} import -> used symbol
- {len([x for x in codebase.ctx.edges if x[2].type == EdgeType.EXPORT])} export -> exported symbol
    """

    return f"{node_summary}\n{edge_summary}"


def get_file_summary(file: SourceFile) -> str:
    return f"""==== [ `{file.name}` (SourceFile) Dependency Summary ] ====
- {len(file.imports)} imports
- {len(file.symbols)} symbol references
\t- {len(file.classes)} classes
\t- {len(file.functions)} functions
\t- {len(file.global_vars)} global variables
\t- {len(file.interfaces)} interfaces

==== [ `{file.name}` Usage Summary ] ====
- {len(file.imports)} importers
"""


def get_class_summary(cls: Class) -> str:
    return f"""==== [ `{cls.name}` (Class) Dependency Summary ] ====
- parent classes: {cls.parent_class_names}
- {len(cls.methods)} methods
- {len(cls.attributes)} attributes
- {len(cls.decorators)} decorators
- {len(cls.dependencies)} dependencies

{get_symbol_summary(cls)}
    """


def get_function_summary(func: Function) -> str:
    return f"""==== [ `{func.name}` (Function) Dependency Summary ] ====
- {len(func.return_statements)} return statements
- {len(func.parameters)} parameters
- {len(func.function_calls)} function calls
- {len(func.call_sites)} call sites
- {len(func.decorators)} decorators
- {len(func.dependencies)} dependencies

{get_symbol_summary(func)}
        """


def get_symbol_summary(symbol: Symbol) -> str:
    usages = symbol.symbol_usages
    imported_symbols = [x.imported_symbol for x in usages if isinstance(x, Import)]

    return f"""==== [ `{symbol.name}` ({type(symbol).__name__}) Usage Summary ] ====
- {len(usages)} usages
\t- {len([x for x in usages if isinstance(x, Symbol) and x.symbol_type == SymbolType.Function])} functions
\t- {len([x for x in usages if isinstance(x, Symbol) and x.symbol_type == SymbolType.Class])} classes
\t- {len([x for x in usages if isinstance(x, Symbol) and x.symbol_type == SymbolType.GlobalVar])} global variables
\t- {len([x for x in usages if isinstance(x, Symbol) and x.symbol_type == SymbolType.Interface])} interfaces
\t- {len(imported_symbols)} imports
\t\t- {len([x for x in imported_symbols if isinstance(x, Symbol) and x.symbol_type == SymbolType.Function])} functions
\t\t- {len([x for x in imported_symbols if isinstance(x, Symbol) and x.symbol_type == SymbolType.Class])} classes
\t\t- {len([x for x in imported_symbols if isinstance(x, Symbol) and x.symbol_type == SymbolType.GlobalVar])} global variables
\t\t- {len([x for x in imported_symbols if isinstance(x, Symbol) and x.symbol_type == SymbolType.Interface])} interfaces
\t\t- {len([x for x in imported_symbols if isinstance(x, ExternalModule)])} external modules
\t\t- {len([x for x in imported_symbols if isinstance(x, SourceFile)])} files
    """

def get_function_context(function) -> dict:
    """Get the implementation, dependencies, and usages of a function."""
    context = {
        "implementation": {"source": function.source, "filepath": function.filepath},
        "dependencies": [],
        "usages": [],
    }

    # Add dependencies
    for dep in function.dependencies:
        # Hop through imports to find the root symbol source
        if isinstance(dep, Import):
            dep = hop_through_imports(dep)

        context["dependencies"].append({"source": dep.source, "filepath": dep.filepath})

    # Add usages
    for usage in function.usages:
        context["usages"].append({
            "source": usage.usage_symbol.source,
            "filepath": usage.usage_symbol.filepath,
        })

    return context

def hop_through_imports(imp: Import) -> Symbol | ExternalModule:
    """Finds the root symbol for an import."""
    if isinstance(imp.imported_symbol, Import):
        return hop_through_imports(imp.imported_symbol)
    return imp.imported_symbol
    
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


# ============================================================================
# COMPREHENSIVE CODEBASE ANALYSIS FUNCTIONS
# ============================================================================

def detect_entrypoints(codebase: Codebase) -> List[EntryPoint]:
    """Detect various types of entry points in the codebase."""
    entrypoints = []
    
    for file in codebase.files:
        # Skip non-Python files for now (can be extended)
        if not file.name.endswith(('.py', '.js', '.ts', '.java', '.cpp', '.c')):
            continue
            
        try:
            # Detect main functions
            for func in file.functions:
                if func.name in ['main', '__main__', 'run', 'start', 'execute']:
                    entrypoints.append(EntryPoint(
                        type="main",
                        file_path=file.filepath,
                        function_name=func.name,
                        line_number=getattr(func, 'line_number', None),
                        description=f"Main function '{func.name}' in {file.name}",
                        confidence=0.9,
                        dependencies=[dep.name for dep in func.dependencies[:5]]
                    ))
                
                # Detect CLI entry points
                if any(keyword in func.name.lower() for keyword in ['cli', 'command', 'parse_args', 'argparse']):
                    entrypoints.append(EntryPoint(
                        type="cli",
                        file_path=file.filepath,
                        function_name=func.name,
                        line_number=getattr(func, 'line_number', None),
                        description=f"CLI function '{func.name}' in {file.name}",
                        confidence=0.7,
                        dependencies=[dep.name for dep in func.dependencies[:5]]
                    ))
                
                # Detect web endpoints
                if any(decorator in str(func.decorators) for decorator in ['@app.route', '@router.', '@fastapi_app.', '@get', '@post', '@put', '@delete']):
                    entrypoints.append(EntryPoint(
                        type="web_endpoint",
                        file_path=file.filepath,
                        function_name=func.name,
                        line_number=getattr(func, 'line_number', None),
                        description=f"Web endpoint '{func.name}' in {file.name}",
                        confidence=0.95,
                        dependencies=[dep.name for dep in func.dependencies[:5]]
                    ))
                
                # Detect test functions
                if func.name.startswith('test_') or any(keyword in func.name.lower() for keyword in ['test', 'spec']):
                    entrypoints.append(EntryPoint(
                        type="test",
                        file_path=file.filepath,
                        function_name=func.name,
                        line_number=getattr(func, 'line_number', None),
                        description=f"Test function '{func.name}' in {file.name}",
                        confidence=0.8,
                        dependencies=[dep.name for dep in func.dependencies[:3]]
                    ))
            
            # Detect script entry points (if __name__ == "__main__")
            if '__name__' in file.source and '__main__' in file.source:
                entrypoints.append(EntryPoint(
                    type="script",
                    file_path=file.filepath,
                    description=f"Script entry point in {file.name}",
                    confidence=0.85,
                    dependencies=[]
                ))
                
        except Exception as e:
            print(f"Error analyzing file {file.name}: {e}")
            continue
    
    return entrypoints


def analyze_code_issues(codebase: Codebase, max_issues: int = 100) -> List[CodeIssue]:
    """Comprehensive code issue analysis using graph-sitter AST."""
    issues = []
    issue_counter = 0
    
    for file in codebase.files:
        if issue_counter >= max_issues:
            break
            
        try:
            # Syntax and parsing issues
            if not file.source.strip():
                issues.append(CodeIssue(
                    id=f"empty_file_{hashlib.md5(file.filepath.encode()).hexdigest()[:8]}",
                    type=IssueType.CODE_SMELL,
                    severity=IssueSeverity.LOW,
                    file_path=file.filepath,
                    message="Empty file",
                    description="File contains no code",
                    context={"file_size": len(file.source)}
                ))
                issue_counter += 1
                continue
            
            # Analyze functions for complexity and maintainability issues
            for func in file.functions:
                if issue_counter >= max_issues:
                    break
                    
                try:
                    # High cyclomatic complexity
                    complexity = calculate_cyclomatic_complexity(func)
                    if complexity > 15:
                        severity = IssueSeverity.CRITICAL if complexity > 30 else IssueSeverity.HIGH
                        issues.append(CodeIssue(
                            id=f"complexity_{hashlib.md5(f'{file.filepath}_{func.name}'.encode()).hexdigest()[:8]}",
                            type=IssueType.COMPLEXITY_ISSUE,
                            severity=severity,
                            file_path=file.filepath,
                            function_name=func.name,
                            line_number=getattr(func, 'line_number', None),
                            message=f"High cyclomatic complexity: {complexity}",
                            description=f"Function '{func.name}' has cyclomatic complexity of {complexity}, which exceeds recommended threshold of 15",
                            context={
                                "complexity": complexity,
                                "complexity_rank": cc_rank(complexity),
                                "parameters_count": len(func.parameters),
                                "return_statements": len(func.return_statements)
                            },
                            related_symbols=[func.name],
                            fix_suggestions=[
                                "Break down the function into smaller, more focused functions",
                                "Extract complex conditional logic into separate methods",
                                "Consider using strategy pattern for complex branching logic"
                            ]
                        ))
                        issue_counter += 1
                    
                    # Long functions (high LOC)
                    func_lines = len(func.source.splitlines()) if hasattr(func, 'source') else 0
                    if func_lines > 50:
                        severity = IssueSeverity.HIGH if func_lines > 100 else IssueSeverity.MEDIUM
                        issues.append(CodeIssue(
                            id=f"long_func_{hashlib.md5(f'{file.filepath}_{func.name}'.encode()).hexdigest()[:8]}",
                            type=IssueType.MAINTAINABILITY_ISSUE,
                            severity=severity,
                            file_path=file.filepath,
                            function_name=func.name,
                            line_number=getattr(func, 'line_number', None),
                            message=f"Long function: {func_lines} lines",
                            description=f"Function '{func.name}' is {func_lines} lines long, exceeding recommended maximum of 50 lines",
                            context={
                                "lines_of_code": func_lines,
                                "parameters_count": len(func.parameters),
                                "complexity": complexity
                            },
                            related_symbols=[func.name],
                            fix_suggestions=[
                                "Split function into smaller, single-responsibility functions",
                                "Extract reusable logic into utility functions",
                                "Consider using composition over large monolithic functions"
                            ]
                        ))
                        issue_counter += 1
                    
                    # Too many parameters
                    if len(func.parameters) > 7:
                        issues.append(CodeIssue(
                            id=f"many_params_{hashlib.md5(f'{file.filepath}_{func.name}'.encode()).hexdigest()[:8]}",
                            type=IssueType.CODE_SMELL,
                            severity=IssueSeverity.MEDIUM,
                            file_path=file.filepath,
                            function_name=func.name,
                            line_number=getattr(func, 'line_number', None),
                            message=f"Too many parameters: {len(func.parameters)}",
                            description=f"Function '{func.name}' has {len(func.parameters)} parameters, exceeding recommended maximum of 7",
                            context={
                                "parameters_count": len(func.parameters),
                                "parameter_names": [p.name for p in func.parameters]
                            },
                            related_symbols=[func.name],
                            fix_suggestions=[
                                "Group related parameters into a configuration object",
                                "Use builder pattern for complex parameter sets",
                                "Consider if some parameters can have default values"
                            ]
                        ))
                        issue_counter += 1
                        
                except Exception as e:
                    print(f"Error analyzing function {func.name}: {e}")
                    continue
            
            # Analyze classes for design issues
            for cls in file.classes:
                if issue_counter >= max_issues:
                    break
                    
                try:
                    # Too many methods (God class)
                    if len(cls.methods) > 20:
                        severity = IssueSeverity.HIGH if len(cls.methods) > 30 else IssueSeverity.MEDIUM
                        issues.append(CodeIssue(
                            id=f"god_class_{hashlib.md5(f'{file.filepath}_{cls.name}'.encode()).hexdigest()[:8]}",
                            type=IssueType.CODE_SMELL,
                            severity=severity,
                            file_path=file.filepath,
                            class_name=cls.name,
                            line_number=getattr(cls, 'line_number', None),
                            message=f"God class: {len(cls.methods)} methods",
                            description=f"Class '{cls.name}' has {len(cls.methods)} methods, indicating it may have too many responsibilities",
                            context={
                                "methods_count": len(cls.methods),
                                "attributes_count": len(cls.attributes),
                                "method_names": [m.name for m in cls.methods[:10]]
                            },
                            related_symbols=[cls.name],
                            affected_functions=[m.name for m in cls.methods],
                            fix_suggestions=[
                                "Apply Single Responsibility Principle - split class into smaller classes",
                                "Extract related methods into separate classes or modules",
                                "Consider using composition instead of inheritance"
                            ]
                        ))
                        issue_counter += 1
                    
                    # Deep inheritance hierarchy
                    doi = calculate_doi(cls)
                    if doi > 5:
                        issues.append(CodeIssue(
                            id=f"deep_inheritance_{hashlib.md5(f'{file.filepath}_{cls.name}'.encode()).hexdigest()[:8]}",
                            type=IssueType.CODE_SMELL,
                            severity=IssueSeverity.MEDIUM,
                            file_path=file.filepath,
                            class_name=cls.name,
                            line_number=getattr(cls, 'line_number', None),
                            message=f"Deep inheritance: {doi} levels",
                            description=f"Class '{cls.name}' has inheritance depth of {doi}, which may indicate over-engineering",
                            context={
                                "depth_of_inheritance": doi,
                                "parent_classes": cls.parent_class_names
                            },
                            related_symbols=[cls.name] + cls.parent_class_names,
                            fix_suggestions=[
                                "Consider using composition over inheritance",
                                "Flatten the inheritance hierarchy",
                                "Use interfaces/protocols instead of deep inheritance"
                            ]
                        ))
                        issue_counter += 1
                        
                except Exception as e:
                    print(f"Error analyzing class {cls.name}: {e}")
                    continue
            
            # Security vulnerability patterns
            security_patterns = [
                (r'eval\s*\(', "Use of eval() function", IssueSeverity.CRITICAL),
                (r'exec\s*\(', "Use of exec() function", IssueSeverity.CRITICAL),
                (r'subprocess\.call\s*\(.*shell\s*=\s*True', "Shell injection vulnerability", IssueSeverity.HIGH),
                (r'pickle\.loads?\s*\(', "Unsafe pickle deserialization", IssueSeverity.HIGH),
                (r'input\s*\(.*\)', "Use of input() function", IssueSeverity.MEDIUM),
                (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password", IssueSeverity.HIGH),
                (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key", IssueSeverity.HIGH),
            ]
            
            for pattern, message, severity in security_patterns:
                if issue_counter >= max_issues:
                    break
                    
                matches = re.finditer(pattern, file.source, re.IGNORECASE)
                for match in matches:
                    line_num = file.source[:match.start()].count('\n') + 1
                    issues.append(CodeIssue(
                        id=f"security_{hashlib.md5(f'{file.filepath}_{line_num}_{message}'.encode()).hexdigest()[:8]}",
                        type=IssueType.SECURITY_VULNERABILITY,
                        severity=severity,
                        file_path=file.filepath,
                        line_number=line_num,
                        message=message,
                        description=f"Potential security vulnerability: {message}",
                        context={
                            "matched_text": match.group(),
                            "pattern": pattern
                        },
                        fix_suggestions=[
                            "Review and validate the security implications",
                            "Consider safer alternatives",
                            "Add proper input validation and sanitization"
                        ]
                    ))
                    issue_counter += 1
                    if issue_counter >= max_issues:
                        break
                        
        except Exception as e:
            print(f"Error analyzing file {file.name}: {e}")
            continue
    
    return issues


def identify_critical_files(codebase: Codebase) -> List[CriticalFile]:
    """Identify the most critical/important files in the codebase."""
    file_metrics = {}
    
    for file in codebase.files:
        try:
            # Calculate various importance metrics
            dependencies_count = len(file.imports)
            dependents_count = len([f for f in codebase.files if any(imp.imported_symbol == file for imp in f.imports)])
            
            # Count symbols defined in file
            symbols_count = len(file.functions) + len(file.classes) + len(file.global_vars)
            
            # Calculate complexity score
            total_complexity = 0
            for func in file.functions:
                try:
                    total_complexity += calculate_cyclomatic_complexity(func)
                except:
                    pass
            
            # Calculate lines of code
            loc, _, _, _ = count_lines(file.source)
            
            # Calculate importance score (0-100)
            importance_score = min(100, (
                dependents_count * 10 +  # Files that depend on this file
                dependencies_count * 2 +  # External dependencies
                symbols_count * 3 +      # Number of symbols defined
                (total_complexity / max(1, len(file.functions))) * 2 +  # Average complexity
                (loc / 100) * 1          # Size factor
            ))
            
            reasons = []
            if dependents_count > 5:
                reasons.append(f"High dependency usage ({dependents_count} files depend on it)")
            if dependencies_count > 10:
                reasons.append(f"Many external dependencies ({dependencies_count})")
            if symbols_count > 10:
                reasons.append(f"Defines many symbols ({symbols_count})")
            if total_complexity > 50:
                reasons.append(f"High total complexity ({total_complexity})")
            if loc > 500:
                reasons.append(f"Large file ({loc} lines)")
            if file.name in ['main.py', 'app.py', 'server.py', 'index.js', 'main.js']:
                reasons.append("Common entry point filename")
                importance_score += 20
            
            if not reasons:
                reasons.append("Standard file")
            
            file_metrics[file.filepath] = CriticalFile(
                file_path=file.filepath,
                importance_score=importance_score,
                reasons=reasons,
                metrics={
                    "functions_count": len(file.functions),
                    "classes_count": len(file.classes),
                    "global_vars_count": len(file.global_vars),
                    "total_complexity": total_complexity,
                    "average_complexity": total_complexity / max(1, len(file.functions))
                },
                dependencies_count=dependencies_count,
                dependents_count=dependents_count,
                complexity_score=total_complexity,
                lines_of_code=loc
            )
            
        except Exception as e:
            print(f"Error analyzing file {file.name}: {e}")
            continue
    
    # Sort by importance score and return top files
    critical_files = sorted(file_metrics.values(), key=lambda x: x.importance_score, reverse=True)
    return critical_files[:20]  # Return top 20 critical files


def build_dependency_graph(codebase: Codebase) -> Dict[str, DependencyNode]:
    """Build a comprehensive dependency graph of the codebase."""
    nodes = {}
    
    # Create nodes for files
    for file in codebase.files:
        node_id = f"file:{file.filepath}"
        nodes[node_id] = DependencyNode(
            name=file.name,
            type="file",
            file_path=file.filepath,
            dependencies=[],
            dependents=[]
        )
    
    # Create nodes for functions
    for file in codebase.files:
        for func in file.functions:
            node_id = f"function:{file.filepath}:{func.name}"
            nodes[node_id] = DependencyNode(
                name=func.name,
                type="function",
                file_path=file.filepath,
                dependencies=[],
                dependents=[]
            )
    
    # Create nodes for classes
    for file in codebase.files:
        for cls in file.classes:
            node_id = f"class:{file.filepath}:{cls.name}"
            nodes[node_id] = DependencyNode(
                name=cls.name,
                type="class",
                file_path=file.filepath,
                dependencies=[],
                dependents=[]
            )
    
    # Build dependency relationships
    for file in codebase.files:
        file_node_id = f"file:{file.filepath}"
        
        # File-level dependencies
        for imp in file.imports:
            if hasattr(imp, 'imported_symbol') and hasattr(imp.imported_symbol, 'filepath'):
                dep_file_id = f"file:{imp.imported_symbol.filepath}"
                if dep_file_id in nodes:
                    nodes[file_node_id].dependencies.append(dep_file_id)
                    nodes[dep_file_id].dependents.append(file_node_id)
        
        # Function-level dependencies
        for func in file.functions:
            func_node_id = f"function:{file.filepath}:{func.name}"
            for dep in func.dependencies:
                if hasattr(dep, 'filepath') and hasattr(dep, 'name'):
                    dep_id = f"function:{dep.filepath}:{dep.name}"
                    if dep_id in nodes:
                        nodes[func_node_id].dependencies.append(dep_id)
                        nodes[dep_id].dependents.append(func_node_id)
    
    # Calculate centrality scores (simplified betweenness centrality)
    for node_id, node in nodes.items():
        # Simple centrality based on number of connections
        centrality = (len(node.dependencies) + len(node.dependents)) / max(1, len(nodes))
        node.centrality_score = min(1.0, centrality * 10)  # Normalize to 0-1
    
    return nodes


def get_file_detailed_analysis(codebase: Codebase, file_path: str) -> Dict[str, Any]:
    """Get detailed analysis for a specific file."""
    target_file = None
    for file in codebase.files:
        if file.filepath == file_path or file.name == file_path:
            target_file = file
            break
    
    if not target_file:
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    # Analyze the file comprehensively
    loc, lloc, sloc, comments = count_lines(target_file.source)
    
    functions_analysis = []
    for func in target_file.functions:
        try:
            complexity = calculate_cyclomatic_complexity(func)
            operators, operands = get_operators_and_operands(func)
            volume, n1, n2, N1, N2 = calculate_halstead_volume(operators, operands)
            mi_score = calculate_maintainability_index(volume, complexity, len(func.source.splitlines()) if hasattr(func, 'source') else 0)
            
            functions_analysis.append({
                "name": func.name,
                "line_number": getattr(func, 'line_number', None),
                "parameters_count": len(func.parameters),
                "return_statements_count": len(func.return_statements),
                "cyclomatic_complexity": complexity,
                "complexity_rank": cc_rank(complexity),
                "halstead_volume": volume,
                "maintainability_index": mi_score,
                "maintainability_rank": get_maintainability_rank(mi_score),
                "dependencies": [dep.name for dep in func.dependencies[:10]],
                "function_calls": [call.name for call in func.function_calls[:10]]
            })
        except Exception as e:
            functions_analysis.append({
                "name": func.name,
                "error": str(e)
            })
    
    classes_analysis = []
    for cls in target_file.classes:
        try:
            classes_analysis.append({
                "name": cls.name,
                "line_number": getattr(cls, 'line_number', None),
                "methods_count": len(cls.methods),
                "attributes_count": len(cls.attributes),
                "parent_classes": cls.parent_class_names,
                "depth_of_inheritance": calculate_doi(cls),
                "dependencies": [dep.name for dep in cls.dependencies[:10]]
            })
        except Exception as e:
            classes_analysis.append({
                "name": cls.name,
                "error": str(e)
            })
    
    return {
        "file_path": target_file.filepath,
        "file_name": target_file.name,
        "line_metrics": {
            "loc": loc,
            "lloc": lloc,
            "sloc": sloc,
            "comments": comments,
            "comment_density": (comments / loc * 100) if loc > 0 else 0
        },
        "symbols_count": {
            "functions": len(target_file.functions),
            "classes": len(target_file.classes),
            "global_vars": len(target_file.global_vars),
            "imports": len(target_file.imports)
        },
        "functions": functions_analysis,
        "classes": classes_analysis,
        "imports": [{"name": imp.name, "source": getattr(imp, 'source', 'unknown')} for imp in target_file.imports[:20]],
        "issues": analyze_code_issues(Codebase.from_files([target_file]), max_issues=50)
    }


# Enums for issue severity and types
class IssueSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class IssueType(str, Enum):
    SYNTAX_ERROR = "syntax_error"
    TYPE_ERROR = "type_error"
    SECURITY_VULNERABILITY = "security_vulnerability"
    CODE_SMELL = "code_smell"
    PERFORMANCE_ISSUE = "performance_issue"
    MAINTAINABILITY_ISSUE = "maintainability_issue"
    COMPLEXITY_ISSUE = "complexity_issue"
    DEPENDENCY_ISSUE = "dependency_issue"

# Data models
class CodeIssue(BaseModel):
    id: str = Field(..., description="Unique identifier for the issue")
    type: IssueType
    severity: IssueSeverity
    file_path: str
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    message: str
    description: str
    context: Dict[str, Any] = Field(default_factory=dict)
    related_symbols: List[str] = Field(default_factory=list)
    affected_functions: List[str] = Field(default_factory=list)
    affected_classes: List[str] = Field(default_factory=list)
    fix_suggestions: List[str] = Field(default_factory=list)

class EntryPoint(BaseModel):
    type: str  # "main", "cli", "web_endpoint", "test", "script"
    file_path: str
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    line_number: Optional[int] = None
    description: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    dependencies: List[str] = Field(default_factory=list)

class CriticalFile(BaseModel):
    file_path: str
    importance_score: float = Field(..., ge=0.0, le=100.0)
    reasons: List[str]
    metrics: Dict[str, Any] = Field(default_factory=dict)
    dependencies_count: int = 0
    dependents_count: int = 0
    complexity_score: float = 0.0
    lines_of_code: int = 0

class DependencyNode(BaseModel):
    name: str
    type: str  # "file", "function", "class", "module"
    file_path: str
    dependencies: List[str] = Field(default_factory=list)
    dependents: List[str] = Field(default_factory=list)
    centrality_score: float = 0.0

class RepoRequest(BaseModel):
    repo_url: str

class DetailedAnalysisRequest(BaseModel):
    repo_url: str
    include_issues: bool = Field(default=True, description="Include code issue analysis")
    include_entrypoints: bool = Field(default=True, description="Include entrypoint detection")
    include_critical_files: bool = Field(default=True, description="Include critical file analysis")
    include_dependency_graph: bool = Field(default=True, description="Include dependency graph analysis")
    max_issues: int = Field(default=100, description="Maximum number of issues to return")

class FileAnalysisRequest(BaseModel):
    repo_url: str
    file_path: str


@fastapi_app.post("/analyze_repo")
async def analyze_repo(request: RepoRequest) -> Dict[str, Any]:
    """Analyze a repository and return comprehensive metrics."""
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
    print(monthly_commits)

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
            "average": total_complexity if num_callables > 0 else 0,
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
    }

    return results


@fastapi_app.post("/comprehensive_analysis")
async def comprehensive_analysis(request: DetailedAnalysisRequest) -> Dict[str, Any]:
    """
    Perform comprehensive codebase analysis including:
    - Entry point detection
    - Critical file identification  
    - Code issue analysis
    - Dependency graph analysis
    """
    try:
        codebase = Codebase.from_repo(request.repo_url)
        
        result = {
            "repo_url": request.repo_url,
            "analysis_timestamp": datetime.now().isoformat(),
            "summary": get_codebase_summary(codebase)
        }
        
        if request.include_entrypoints:
            print("Detecting entry points...")
            entrypoints = detect_entrypoints(codebase)
            result["entrypoints"] = {
                "total_count": len(entrypoints),
                "by_type": {
                    entry_type: len([ep for ep in entrypoints if ep.type == entry_type])
                    for entry_type in ["main", "cli", "web_endpoint", "test", "script"]
                },
                "details": [ep.dict() for ep in entrypoints]
            }
        
        if request.include_critical_files:
            print("Identifying critical files...")
            critical_files = identify_critical_files(codebase)
            result["critical_files"] = {
                "total_count": len(critical_files),
                "top_10": [cf.dict() for cf in critical_files[:10]],
                "summary": {
                    "avg_importance_score": sum(cf.importance_score for cf in critical_files) / len(critical_files) if critical_files else 0,
                    "high_importance_count": len([cf for cf in critical_files if cf.importance_score > 70]),
                    "medium_importance_count": len([cf for cf in critical_files if 40 <= cf.importance_score <= 70]),
                    "low_importance_count": len([cf for cf in critical_files if cf.importance_score < 40])
                }
            }
        
        if request.include_issues:
            print("Analyzing code issues...")
            issues = analyze_code_issues(codebase, request.max_issues)
            result["issues"] = {
                "total_count": len(issues),
                "by_severity": {
                    severity.value: len([issue for issue in issues if issue.severity == severity])
                    for severity in IssueSeverity
                },
                "by_type": {
                    issue_type.value: len([issue for issue in issues if issue.type == issue_type])
                    for issue_type in IssueType
                },
                "critical_issues": [issue.dict() for issue in issues if issue.severity == IssueSeverity.CRITICAL],
                "high_priority_issues": [issue.dict() for issue in issues if issue.severity == IssueSeverity.HIGH][:20],
                "all_issues": [issue.dict() for issue in issues] if len(issues) <= 50 else [issue.dict() for issue in issues[:50]]
            }
        
        if request.include_dependency_graph:
            print("Building dependency graph...")
            dependency_graph = build_dependency_graph(codebase)
            
            # Get top nodes by centrality
            top_central_nodes = sorted(dependency_graph.values(), key=lambda x: x.centrality_score, reverse=True)[:20]
            
            result["dependency_graph"] = {
                "total_nodes": len(dependency_graph),
                "node_types": {
                    "files": len([n for n in dependency_graph.values() if n.type == "file"]),
                    "functions": len([n for n in dependency_graph.values() if n.type == "function"]),
                    "classes": len([n for n in dependency_graph.values() if n.type == "class"])
                },
                "most_central_nodes": [node.dict() for node in top_central_nodes],
                "graph_metrics": {
                    "avg_centrality": sum(n.centrality_score for n in dependency_graph.values()) / len(dependency_graph),
                    "max_centrality": max(n.centrality_score for n in dependency_graph.values()) if dependency_graph else 0,
                    "highly_connected_nodes": len([n for n in dependency_graph.values() if n.centrality_score > 0.1])
                }
            }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@fastapi_app.post("/analyze_file")
async def analyze_file(request: FileAnalysisRequest) -> Dict[str, Any]:
    """Get detailed analysis for a specific file in the repository."""
    try:
        codebase = Codebase.from_repo(request.repo_url)
        return get_file_detailed_analysis(codebase, request.file_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File analysis failed: {str(e)}")


@fastapi_app.post("/detect_entrypoints")
async def detect_entrypoints_endpoint(request: RepoRequest) -> Dict[str, Any]:
    """Detect all entry points in the codebase."""
    try:
        codebase = Codebase.from_repo(request.repo_url)
        entrypoints = detect_entrypoints(codebase)
        
        return {
            "repo_url": request.repo_url,
            "total_entrypoints": len(entrypoints),
            "entrypoints_by_type": {
                entry_type: [ep.dict() for ep in entrypoints if ep.type == entry_type]
                for entry_type in ["main", "cli", "web_endpoint", "test", "script"]
            },
            "all_entrypoints": [ep.dict() for ep in entrypoints]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Entry point detection failed: {str(e)}")


@fastapi_app.post("/identify_critical_files")
async def identify_critical_files_endpoint(request: RepoRequest) -> Dict[str, Any]:
    """Identify the most critical files in the codebase."""
    try:
        codebase = Codebase.from_repo(request.repo_url)
        critical_files = identify_critical_files(codebase)
        
        return {
            "repo_url": request.repo_url,
            "total_files_analyzed": len(list(codebase.files)),
            "critical_files_count": len(critical_files),
            "critical_files": [cf.dict() for cf in critical_files],
            "summary": {
                "avg_importance_score": sum(cf.importance_score for cf in critical_files) / len(critical_files) if critical_files else 0,
                "highest_score": max(cf.importance_score for cf in critical_files) if critical_files else 0,
                "files_above_80": len([cf for cf in critical_files if cf.importance_score > 80]),
                "files_above_60": len([cf for cf in critical_files if cf.importance_score > 60])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Critical file identification failed: {str(e)}")


@fastapi_app.post("/analyze_issues")
async def analyze_issues_endpoint(request: DetailedAnalysisRequest) -> Dict[str, Any]:
    """Comprehensive code issue analysis."""
    try:
        codebase = Codebase.from_repo(request.repo_url)
        issues = analyze_code_issues(codebase, request.max_issues)
        
        # Group issues by file for better organization
        issues_by_file = defaultdict(list)
        for issue in issues:
            issues_by_file[issue.file_path].append(issue)
        
        # Get severity statistics
        severity_stats = {severity.value: 0 for severity in IssueSeverity}
        type_stats = {issue_type.value: 0 for issue_type in IssueType}
        
        for issue in issues:
            severity_stats[issue.severity.value] += 1
            type_stats[issue.type.value] += 1
        
        return {
            "repo_url": request.repo_url,
            "analysis_summary": {
                "total_issues": len(issues),
                "files_with_issues": len(issues_by_file),
                "critical_issues": severity_stats[IssueSeverity.CRITICAL.value],
                "high_priority_issues": severity_stats[IssueSeverity.HIGH.value],
                "medium_priority_issues": severity_stats[IssueSeverity.MEDIUM.value],
                "low_priority_issues": severity_stats[IssueSeverity.LOW.value]
            },
            "issues_by_severity": severity_stats,
            "issues_by_type": type_stats,
            "issues_by_file": {
                file_path: {
                    "issue_count": len(file_issues),
                    "critical_count": len([i for i in file_issues if i.severity == IssueSeverity.CRITICAL]),
                    "high_count": len([i for i in file_issues if i.severity == IssueSeverity.HIGH]),
                    "issues": [issue.dict() for issue in file_issues]
                }
                for file_path, file_issues in issues_by_file.items()
            },
            "most_problematic_files": [
                {
                    "file_path": file_path,
                    "total_issues": len(file_issues),
                    "critical_issues": len([i for i in file_issues if i.severity == IssueSeverity.CRITICAL]),
                    "high_issues": len([i for i in file_issues if i.severity == IssueSeverity.HIGH])
                }
                for file_path, file_issues in sorted(issues_by_file.items(), key=lambda x: len(x[1]), reverse=True)[:10]
            ],
            "all_issues": [issue.dict() for issue in issues]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Issue analysis failed: {str(e)}")


@fastapi_app.post("/dependency_graph")
async def dependency_graph_endpoint(request: RepoRequest) -> Dict[str, Any]:
    """Build and analyze the dependency graph of the codebase."""
    try:
        codebase = Codebase.from_repo(request.repo_url)
        dependency_graph = build_dependency_graph(codebase)
        
        # Calculate graph statistics
        total_edges = sum(len(node.dependencies) for node in dependency_graph.values())
        
        # Find most connected nodes
        most_connected = sorted(dependency_graph.values(), 
                              key=lambda x: len(x.dependencies) + len(x.dependents), 
                              reverse=True)[:20]
        
        # Find nodes with highest centrality
        most_central = sorted(dependency_graph.values(), 
                            key=lambda x: x.centrality_score, 
                            reverse=True)[:20]
        
        return {
            "repo_url": request.repo_url,
            "graph_statistics": {
                "total_nodes": len(dependency_graph),
                "total_edges": total_edges,
                "average_connections": total_edges / len(dependency_graph) if dependency_graph else 0,
                "node_types": {
                    "files": len([n for n in dependency_graph.values() if n.type == "file"]),
                    "functions": len([n for n in dependency_graph.values() if n.type == "function"]),
                    "classes": len([n for n in dependency_graph.values() if n.type == "class"])
                }
            },
            "most_connected_nodes": [
                {
                    "name": node.name,
                    "type": node.type,
                    "file_path": node.file_path,
                    "total_connections": len(node.dependencies) + len(node.dependents),
                    "dependencies_count": len(node.dependencies),
                    "dependents_count": len(node.dependents)
                }
                for node in most_connected
            ],
            "most_central_nodes": [node.dict() for node in most_central],
            "dependency_graph": {node_id: node.dict() for node_id, node in dependency_graph.items()}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dependency graph analysis failed: {str(e)}")


@fastapi_app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.function(image=image)
@modal.asgi_app()
def fastapi_modal_app():
    return fastapi_app


if __name__ == "__main__":
    app.deploy("analytics-app")
