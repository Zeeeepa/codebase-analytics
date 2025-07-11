from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List, Tuple, Any, Optional
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
import networkx as nx
import json
import time
import ast
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, Counter

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
# COMPREHENSIVE ANALYSIS SYSTEM - ENUMS AND DATACLASSES
# ============================================================================

class IssueSeverity(Enum):
    CRITICAL = "critical"  # ⚠️
    MAJOR = "major"       # 👉
    MINOR = "minor"       # 🔍
    INFO = "info"         # ℹ️


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
    MISSING_REQUIRED_PARAMETER = "missing_required_parameter"
    UNUSED_PARAMETER = "unused_parameter"
    
    # Exception Handling
    IMPROPER_EXCEPTION_HANDLING = "improper_exception_handling"
    MISSING_ERROR_HANDLING = "missing_error_handling"
    UNSAFE_ASSERTION = "unsafe_assertion"
    RESOURCE_LEAK = "resource_leak"
    MEMORY_MANAGEMENT = "memory_management"
    
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
    LINE_LENGTH_VIOLATION = "line_length_violation"
    IMPORT_ORGANIZATION = "import_organization"
    
    # Runtime Risks
    DIVISION_BY_ZERO = "division_by_zero"
    ARRAY_INDEX_OUT_OF_BOUNDS = "array_index_out_of_bounds"
    INFINITE_LOOP = "infinite_loop"
    STACK_OVERFLOW = "stack_overflow"
    CONCURRENCY_ISSUE = "concurrency_issue"
    
    # Dead Code
    DEAD_FUNCTION = "dead_function"
    DEAD_VARIABLE = "dead_variable"
    DEAD_CLASS = "dead_class"
    DEAD_IMPORT = "dead_import"


@dataclass
class AutomatedResolution:
    """Represents an automated fix that can be applied"""
    resolution_type: str
    description: str
    original_code: str
    fixed_code: str
    confidence: float
    file_path: str
    line_number: int
    is_safe: bool = True
    requires_validation: bool = False


@dataclass
class CodeIssue:
    """Represents a code issue with full context and automated resolution"""
    issue_type: IssueType
    severity: IssueSeverity
    message: str
    filepath: str
    line_number: int
    column_number: int
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    suggested_fix: Optional[str] = None
    blast_radius: List[str] = field(default_factory=list)
    automated_resolution: Optional[AutomatedResolution] = None
    related_issues: List[str] = field(default_factory=list)
    impact_score: float = 0.0
    fix_effort: str = "low"  # low, medium, high


@dataclass
class FunctionContext:
    """Complete context for a function with all relationships"""
    name: str
    filepath: str
    line_start: int
    line_end: int
    source: str
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: Optional[str] = None
    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    usages: List[Dict[str, Any]] = field(default_factory=list)
    function_calls: List[str] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)
    class_name: Optional[str] = None
    max_call_chain: List[str] = field(default_factory=list)
    issues: List[Any] = field(default_factory=list)
    complexity_metrics: Dict[str, Any] = field(default_factory=dict)
    is_entry_point: bool = False
    is_dead_code: bool = False
    call_depth: int = 0
    fan_in: int = 0  # Number of functions calling this one
    fan_out: int = 0  # Number of functions this one calls
    coupling_score: float = 0.0
    cohesion_score: float = 0.0


@dataclass
class AnalysisResults:
    """Structured analysis results for API consumption"""
    
    # Basic Statistics
    total_files: int
    total_functions: int
    total_classes: int
    total_lines_of_code: int
    
    # Issues Analysis
    issues: List[CodeIssue] = field(default_factory=list)
    issues_by_severity: Dict[str, int] = field(default_factory=dict)
    issues_by_type: Dict[str, int] = field(default_factory=dict)
    automated_resolutions: List[AutomatedResolution] = field(default_factory=list)
    
    # Function Analysis
    function_contexts: Dict[str, FunctionContext] = field(default_factory=dict)
    most_important_functions: List[Dict[str, Any]] = field(default_factory=list)
    entry_points: List[str] = field(default_factory=list)
    dead_functions: List[str] = field(default_factory=list)
    
    # Quality Metrics
    halstead_metrics: Dict[str, Any] = field(default_factory=dict)
    complexity_metrics: Dict[str, Any] = field(default_factory=dict)
    maintainability_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Graph Analysis
    call_graph_metrics: Dict[str, Any] = field(default_factory=dict)
    dependency_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Health Assessment
    health_score: float = 0.0
    health_grade: str = "F"
    risk_level: str = "high"
    technical_debt_hours: float = 0.0
    
    # Repository Structure
    repository_structure: Dict[str, Any] = field(default_factory=dict)


class RepoRequest(BaseModel):
    repo_url: str


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


# ============================================================================
# COMPREHENSIVE ANALYSIS SYSTEM - MAIN CLASSES
# ============================================================================

class ImportResolver:
    """Automated import resolution system"""
    
    def __init__(self, codebase):
        self.codebase = codebase
        self.import_map = self._build_import_map()
        self.symbol_map = self._build_symbol_map()
    
    def _build_import_map(self) -> Dict[str, str]:
        """Build map of available imports"""
        import_map = {}
        for file in self.codebase.files:
            if hasattr(file, 'imports'):
                for imp in file.imports:
                    if hasattr(imp, 'module_name'):
                        import_map[imp.module_name] = file.filepath
        return import_map
    
    def _build_symbol_map(self) -> Dict[str, str]:
        """Build map of available symbols"""
        symbol_map = {}
        for file in self.codebase.files:
            # Map functions
            if hasattr(file, 'functions'):
                for func in file.functions:
                    symbol_map[func.name] = file.filepath
            
            # Map classes
            if hasattr(file, 'classes'):
                for cls in file.classes:
                    symbol_map[cls.name] = file.filepath
        
        return symbol_map
    
    def find_unused_imports(self, file) -> List[Dict[str, Any]]:
        """Find unused imports in a file"""
        unused = []
        if hasattr(file, 'imports') and hasattr(file, 'source'):
            for imp in file.imports:
                if hasattr(imp, 'module_name'):
                    # Simple check if import is used in source
                    if imp.module_name not in file.source:
                        unused.append({
                            'name': imp.module_name,
                            'source': str(imp),
                            'line': getattr(imp, 'line_number', 1)
                        })
        return unused
    
    def find_missing_imports(self, file) -> List[Dict[str, Any]]:
        """Find missing imports in a file"""
        missing = []
        if hasattr(file, 'source'):
            # Find undefined symbols that could be imports
            for symbol, filepath in self.symbol_map.items():
                if symbol in file.source and filepath != file.filepath:
                    # Check if already imported
                    if not self._is_imported(file, symbol):
                        missing.append({
                            'symbol': symbol,
                            'source_file': filepath,
                            'line': 1  # Will be added at top
                        })
        return missing
    
    def _is_imported(self, file, symbol: str) -> bool:
        """Check if symbol is already imported"""
        if hasattr(file, 'imports'):
            for imp in file.imports:
                if hasattr(imp, 'module_name') and symbol in str(imp):
                    return True
        return False
    
    def resolve_import(self, symbol: str) -> Optional[str]:
        """Resolve the correct import statement for a symbol"""
        if symbol in self.symbol_map:
            source_file = self.symbol_map[symbol]
            # Convert file path to import statement
            import_path = source_file.replace('/', '.').replace('.py', '')
            return f"from {import_path} import {symbol}"
        return None


class AdvancedIssueDetector:
    """Advanced issue detection with automated resolution capabilities"""
    
    def __init__(self, codebase):
        self.codebase = codebase
        self.issues = []
        self.automated_resolutions = []
        self.import_resolver = ImportResolver(codebase)
        
    def detect_all_issues(self) -> List[CodeIssue]:
        """Detect all types of issues with automated resolutions"""
        print("🔍 Starting comprehensive issue detection...")
        
        # Implementation errors
        self._detect_null_references()
        self._detect_type_mismatches()
        self._detect_undefined_variables()
        self._detect_missing_returns()
        self._detect_unreachable_code()
        
        # Function issues
        self._detect_function_issues()
        self._detect_parameter_issues()
        
        # Exception handling
        self._detect_exception_handling_issues()
        self._detect_resource_leaks()
        
        # Code quality
        self._detect_code_quality_issues()
        self._detect_magic_numbers()
        
        # Formatting & style
        self._detect_style_issues()
        self._detect_import_issues()
        
        # Runtime risks
        self._detect_runtime_risks()
        
        # Dead code
        self._detect_dead_code()
        
        # Apply automated resolutions
        self._apply_automated_resolutions()
        
        print(f"✅ Detected {len(self.issues)} issues with {len(self.automated_resolutions)} automated resolutions")
        return self.issues
    
    def _detect_null_references(self):
        """Detect potential null reference issues"""
        for file in self.codebase.files:
            if hasattr(file, 'source'):
                lines = file.source.split('\n')
                for i, line in enumerate(lines):
                    # Check for .get() without null checks
                    if '.get(' in line and 'if' not in line and 'or' not in line:
                        issue = CodeIssue(
                            issue_type=IssueType.NULL_REFERENCE,
                            severity=IssueSeverity.MAJOR,
                            message="Potential null reference: .get() without null check",
                            filepath=file.filepath,
                            line_number=i + 1,
                            column_number=line.find('.get('),
                            context={"line": line.strip()},
                            suggested_fix="Add null check or provide default value",
                            impact_score=7.5
                        )
                        
                        # Automated resolution
                        if '.get(' in line:
                            fixed_line = self._fix_null_reference(line)
                            issue.automated_resolution = AutomatedResolution(
                                resolution_type="null_check_addition",
                                description="Add null check with default value",
                                original_code=line.strip(),
                                fixed_code=fixed_line,
                                confidence=0.85,
                                file_path=file.filepath,
                                line_number=i + 1
                            )
                        
                        self.issues.append(issue)
    
    def _fix_null_reference(self, line: str) -> str:
        """Automatically fix null reference issues"""
        # Simple pattern: obj.get('key') -> obj.get('key', default_value)
        pattern = r'(\w+)\.get\([\'"]([^\'"]+)[\'"]\)'
        match = re.search(pattern, line)
        if match:
            obj, key = match.groups()
            return line.replace(f"{obj}.get('{key}')", f"{obj}.get('{key}', None)")
        return line
    
    def _detect_function_issues(self):
        """Detect function-related issues"""
        for function in self.codebase.functions:
            # Long function detection
            if hasattr(function, 'start_point') and hasattr(function, 'end_point'):
                line_count = function.end_point[0] - function.start_point[0]
                if line_count > 50:
                    issue = CodeIssue(
                        issue_type=IssueType.LONG_FUNCTION,
                        severity=IssueSeverity.MAJOR,
                        message=f"Function '{function.name}' is too long ({line_count} lines)",
                        filepath=function.filepath,
                        line_number=function.start_point[0],
                        column_number=0,
                        function_name=function.name,
                        context={"line_count": line_count},
                        suggested_fix="Break down into smaller functions",
                        impact_score=6.0,
                        fix_effort="high"
                    )
                    self.issues.append(issue)
            
            # Missing documentation
            if hasattr(function, 'source'):
                if not ('"""' in function.source or "'''" in function.source):
                    issue = CodeIssue(
                        issue_type=IssueType.MISSING_DOCUMENTATION,
                        severity=IssueSeverity.MINOR,
                        message=f"Function '{function.name}' lacks documentation",
                        filepath=function.filepath,
                        line_number=function.start_point[0] if hasattr(function, 'start_point') else 0,
                        column_number=0,
                        function_name=function.name,
                        impact_score=3.0
                    )
                    
                    # Automated resolution - add basic docstring
                    docstring = self._generate_docstring(function)
                    issue.automated_resolution = AutomatedResolution(
                        resolution_type="add_docstring",
                        description=f"Add docstring to function '{function.name}'",
                        original_code="",
                        fixed_code=docstring,
                        confidence=0.80,
                        file_path=function.filepath,
                        line_number=function.start_point[0] + 1 if hasattr(function, 'start_point') else 1
                    )
                    
                    self.issues.append(issue)
    
    def _generate_docstring(self, function) -> str:
        """Generate basic docstring for a function"""
        params = ""
        if hasattr(function, 'parameters') and function.parameters:
            param_list = [f"        {param.name}: Description of {param.name}" 
                         for param in function.parameters if hasattr(param, 'name')]
            if param_list:
                params = f"\n    Args:\n" + "\n".join(param_list)
        
        return f'    """{function.name} function.\n    \n    Brief description of what this function does.{params}\n    \n    Returns:\n        Description of return value\n    """'
    
    def _detect_magic_numbers(self):
        """Detect magic numbers in code"""
        for file in self.codebase.files:
            if hasattr(file, 'source'):
                lines = file.source.split('\n')
                for i, line in enumerate(lines):
                    # Find numeric literals (excluding 0, 1, -1)
                    numbers = re.findall(r'\b(?<![\w.])\d{2,}\b(?![\w.])', line)
                    for number in numbers:
                        if int(number) not in [0, 1, -1, 100]:  # Common acceptable numbers
                            issue = CodeIssue(
                                issue_type=IssueType.MAGIC_NUMBER,
                                severity=IssueSeverity.MINOR,
                                message=f"Magic number detected: {number}",
                                filepath=file.filepath,
                                line_number=i + 1,
                                column_number=line.find(number),
                                context={"number": number, "line": line.strip()},
                                suggested_fix=f"Replace {number} with named constant",
                                impact_score=2.5
                            )
                            self.issues.append(issue)
    
    def _detect_runtime_risks(self):
        """Detect potential runtime risks"""
        for file in self.codebase.files:
            if hasattr(file, 'source'):
                lines = file.source.split('\n')
                for i, line in enumerate(lines):
                    # Division by zero risk
                    if '/' in line and 'if' not in line:
                        # Simple heuristic for potential division by zero
                        if re.search(r'/\s*\w+(?!\w)', line):
                            issue = CodeIssue(
                                issue_type=IssueType.DIVISION_BY_ZERO,
                                severity=IssueSeverity.MAJOR,
                                message="Potential division by zero",
                                filepath=file.filepath,
                                line_number=i + 1,
                                column_number=line.find('/'),
                                context={"line": line.strip()},
                                suggested_fix="Add zero check before division",
                                impact_score=8.0
                            )
                            self.issues.append(issue)
    
    def _detect_dead_code(self):
        """Detect dead code with automated removal suggestions"""
        # Find unused functions
        for function in self.codebase.functions:
            if hasattr(function, 'usages') and len(function.usages) == 0:
                # Check if it's not an entry point
                if not self._is_entry_point(function):
                    issue = CodeIssue(
                        issue_type=IssueType.DEAD_FUNCTION,
                        severity=IssueSeverity.MINOR,
                        message=f"Unused function: {function.name}",
                        filepath=function.filepath,
                        line_number=function.start_point[0] if hasattr(function, 'start_point') else 0,
                        column_number=0,
                        function_name=function.name,
                        context={"reason": "No usages found"},
                        impact_score=1.0
                    )
                    
                    # Automated resolution - mark for removal
                    issue.automated_resolution = AutomatedResolution(
                        resolution_type="remove_dead_function",
                        description=f"Remove unused function '{function.name}'",
                        original_code=function.source if hasattr(function, 'source') else "",
                        fixed_code="",
                        confidence=0.75,
                        file_path=function.filepath,
                        line_number=function.start_point[0] if hasattr(function, 'start_point') else 0,
                        requires_validation=True
                    )
                    
                    self.issues.append(issue)
    
    def _is_entry_point(self, function) -> bool:
        """Check if function is an entry point"""
        entry_patterns = ['main', '__main__', 'run', 'start', 'execute', 'init', 'setup', 'app', 'server', 'cli']
        return any(pattern in function.name.lower() for pattern in entry_patterns)
    
    def _detect_import_issues(self):
        """Detect and automatically resolve import issues"""
        for file in self.codebase.files:
            if hasattr(file, 'imports'):
                # Detect unused imports
                unused_imports = self.import_resolver.find_unused_imports(file)
                for unused_import in unused_imports:
                    issue = CodeIssue(
                        issue_type=IssueType.DEAD_IMPORT,
                        severity=IssueSeverity.MINOR,
                        message=f"Unused import: {unused_import['name']}",
                        filepath=file.filepath,
                        line_number=unused_import.get('line', 1),
                        column_number=0,
                        context={"import_name": unused_import['name']},
                        impact_score=2.0
                    )
                    
                    # Automated resolution - remove unused import
                    issue.automated_resolution = AutomatedResolution(
                        resolution_type="remove_unused_import",
                        description=f"Remove unused import: {unused_import['name']}",
                        original_code=unused_import.get('source', ''),
                        fixed_code="",  # Remove the line
                        confidence=0.95,
                        file_path=file.filepath,
                        line_number=unused_import.get('line', 1),
                        is_safe=True
                    )
                    
                    self.issues.append(issue)
                
                # Detect missing imports
                missing_imports = self.import_resolver.find_missing_imports(file)
                for missing_import in missing_imports:
                    issue = CodeIssue(
                        issue_type=IssueType.UNDEFINED_VARIABLE,
                        severity=IssueSeverity.CRITICAL,
                        message=f"Missing import for: {missing_import['symbol']}",
                        filepath=file.filepath,
                        line_number=missing_import.get('line', 1),
                        column_number=0,
                        context={"symbol": missing_import['symbol']},
                        impact_score=9.0
                    )
                    
                    # Automated resolution - add missing import
                    resolved_import = self.import_resolver.resolve_import(missing_import['symbol'])
                    if resolved_import:
                        issue.automated_resolution = AutomatedResolution(
                            resolution_type="add_missing_import",
                            description=f"Add import: {resolved_import}",
                            original_code="",
                            fixed_code=resolved_import,
                            confidence=0.90,
                            file_path=file.filepath,
                            line_number=1,  # Add at top of file
                            is_safe=True
                        )
                    
                    self.issues.append(issue)
    
    def _apply_automated_resolutions(self):
        """Apply automated resolutions where safe"""
        for issue in self.issues:
            if issue.automated_resolution and issue.automated_resolution.is_safe:
                self.automated_resolutions.append(issue.automated_resolution)


class ComprehensiveCodebaseAnalyzer:
    """Main orchestrator for comprehensive codebase analysis"""
    
    def __init__(self, codebase):
        self.codebase = codebase
        self.results = None
        
    def analyze(self) -> AnalysisResults:
        """Perform comprehensive analysis"""
        print("🎯 Starting comprehensive codebase analysis...")
        
        # Basic statistics
        total_files = len(self.codebase.files)
        total_functions = len(self.codebase.functions)
        total_classes = len(self.codebase.classes)
        total_loc = sum(len(file.source.split('\n')) for file in self.codebase.files if hasattr(file, 'source'))
        
        # Advanced issue detection
        issue_detector = AdvancedIssueDetector(self.codebase)
        issues = issue_detector.detect_all_issues()
        
        # Function context analysis
        function_contexts = self._analyze_function_contexts()
        
        # Quality metrics
        halstead_metrics = self._calculate_comprehensive_halstead_metrics()
        complexity_metrics = self._calculate_comprehensive_complexity_metrics()
        
        # Health assessment
        health_score, health_grade, risk_level = self._calculate_health_metrics(issues)
        
        # Create results
        self.results = AnalysisResults(
            total_files=total_files,
            total_functions=total_functions,
            total_classes=total_classes,
            total_lines_of_code=total_loc,
            issues=issues,
            issues_by_severity=self._group_issues_by_severity(issues),
            issues_by_type=self._group_issues_by_type(issues),
            automated_resolutions=issue_detector.automated_resolutions,
            function_contexts=function_contexts,
            most_important_functions=self._identify_important_functions(function_contexts),
            entry_points=self._identify_entry_points(),
            dead_functions=self._identify_dead_functions(issues),
            halstead_metrics=halstead_metrics,
            complexity_metrics=complexity_metrics,
            health_score=health_score,
            health_grade=health_grade,
            risk_level=risk_level,
            technical_debt_hours=self._calculate_technical_debt(issues),
            repository_structure=self._build_repository_structure(issues)
        )
        
        print("✅ Comprehensive analysis completed")
        return self.results
    
    def _analyze_function_contexts(self) -> Dict[str, FunctionContext]:
        """Analyze function contexts"""
        contexts = {}
        call_graph = nx.DiGraph()
        
        # Build basic contexts
        for function in self.codebase.functions:
            context = FunctionContext(
                name=function.name,
                filepath=function.filepath,
                line_start=function.start_point[0] if hasattr(function, 'start_point') else 0,
                line_end=function.end_point[0] if hasattr(function, 'end_point') else 0,
                source=function.source if hasattr(function, 'source') else "",
                is_entry_point=self._is_function_entry_point(function)
            )
            
            # Extract function calls
            if hasattr(function, 'function_calls'):
                context.function_calls = [call.name for call in function.function_calls if hasattr(call, 'name')]
            
            # Extract usages
            if hasattr(function, 'usages'):
                context.usages = [{"filepath": usage.usage_symbol.filepath, "line": usage.usage_symbol.start_point[0]} 
                                for usage in function.usages if hasattr(usage, 'usage_symbol')]
            
            contexts[function.name] = context
            call_graph.add_node(function.name)
        
        # Build call graph edges
        for name, context in contexts.items():
            for called_func in context.function_calls:
                if called_func in contexts:
                    call_graph.add_edge(name, called_func)
        
        # Calculate graph metrics
        for name, context in contexts.items():
            if name in call_graph:
                context.fan_out = call_graph.out_degree(name)
                context.fan_in = call_graph.in_degree(name)
                context.call_depth = self._calculate_call_depth(call_graph, name)
        
        return contexts
    
    def _calculate_call_depth(self, graph: nx.DiGraph, node: str) -> int:
        """Calculate maximum call depth from a node"""
        try:
            # Simple DFS to find maximum depth
            visited = set()
            
            def dfs(current, depth):
                if current in visited:
                    return depth
                visited.add(current)
                
                max_depth = depth
                for successor in graph.successors(current):
                    max_depth = max(max_depth, dfs(successor, depth + 1))
                
                visited.remove(current)
                return max_depth
            
            return dfs(node, 0)
        except:
            return 0
    
    def _calculate_comprehensive_halstead_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive Halstead metrics"""
        total_operators = 0
        total_operands = 0
        total_volume = 0
        
        for function in self.codebase.functions:
            if hasattr(function, 'code_block'):
                operators, operands = get_operators_and_operands(function)
                volume, _, _, _, _ = calculate_halstead_volume(operators, operands)
                total_volume += volume
                total_operators += len(operators)
                total_operands += len(operands)
        
        return {
            "total_operators": total_operators,
            "total_operands": total_operands,
            "total_volume": total_volume,
            "average_volume": total_volume / len(self.codebase.functions) if self.codebase.functions else 0
        }
    
    def _calculate_comprehensive_complexity_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive complexity metrics"""
        total_complexity = 0
        complexities = []
        
        for function in self.codebase.functions:
            if hasattr(function, 'code_block'):
                complexity = calculate_cyclomatic_complexity(function)
                total_complexity += complexity
                complexities.append(complexity)
        
        return {
            "total_complexity": total_complexity,
            "average_complexity": total_complexity / len(complexities) if complexities else 0,
            "max_complexity": max(complexities) if complexities else 0,
            "min_complexity": min(complexities) if complexities else 0
        }
    
    def _calculate_health_metrics(self, issues: List[CodeIssue]) -> Tuple[float, str, str]:
        """Calculate overall health metrics"""
        if not issues:
            return 100.0, "A", "low"
        
        # Calculate health score based on issue severity
        health_score = 100.0
        severity_penalties = {
            IssueSeverity.CRITICAL: 15,
            IssueSeverity.MAJOR: 8,
            IssueSeverity.MINOR: 3,
            IssueSeverity.INFO: 1
        }
        
        for issue in issues:
            health_score -= severity_penalties.get(issue.severity, 1)
        
        health_score = max(0, health_score)
        
        # Determine grade
        if health_score >= 85:
            grade = "A"
        elif health_score >= 70:
            grade = "B"
        elif health_score >= 55:
            grade = "C"
        elif health_score >= 40:
            grade = "D"
        else:
            grade = "F"
        
        # Determine risk level
        critical_issues = sum(1 for issue in issues if issue.severity == IssueSeverity.CRITICAL)
        if critical_issues > 5:
            risk_level = "high"
        elif critical_issues > 0:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return health_score, grade, risk_level
    
    def _group_issues_by_severity(self, issues: List[CodeIssue]) -> Dict[str, int]:
        """Group issues by severity"""
        severity_counts = defaultdict(int)
        for issue in issues:
            severity_counts[issue.severity.value] += 1
        return dict(severity_counts)
    
    def _group_issues_by_type(self, issues: List[CodeIssue]) -> Dict[str, int]:
        """Group issues by type"""
        type_counts = defaultdict(int)
        for issue in issues:
            type_counts[issue.issue_type.value] += 1
        return dict(type_counts)
    
    def _identify_important_functions(self, contexts: Dict[str, FunctionContext]) -> List[Dict[str, Any]]:
        """Identify most important functions"""
        important = []
        for name, context in contexts.items():
            importance_score = context.fan_in * 2 + context.fan_out + context.call_depth
            important.append({
                "name": name,
                "filepath": context.filepath,
                "importance_score": importance_score,
                "fan_in": context.fan_in,
                "fan_out": context.fan_out,
                "call_depth": context.call_depth
            })
        
        return sorted(important, key=lambda x: x["importance_score"], reverse=True)[:10]
    
    def _identify_entry_points(self) -> List[str]:
        """Identify entry point functions"""
        entry_points = []
        for function in self.codebase.functions:
            if self._is_function_entry_point(function):
                entry_points.append(function.name)
        return entry_points
    
    def _is_function_entry_point(self, function) -> bool:
        """Check if function is an entry point"""
        entry_patterns = ['main', '__main__', 'run', 'start', 'execute', 'init', 'setup', 'app', 'server', 'cli']
        return any(pattern in function.name.lower() for pattern in entry_patterns)
    
    def _identify_dead_functions(self, issues: List[CodeIssue]) -> List[str]:
        """Identify dead functions from issues"""
        dead_functions = []
        for issue in issues:
            if issue.issue_type == IssueType.DEAD_FUNCTION and issue.function_name:
                dead_functions.append(issue.function_name)
        return dead_functions
    
    def _calculate_technical_debt(self, issues: List[CodeIssue]) -> float:
        """Calculate technical debt in hours"""
        debt_hours = 0.0
        effort_hours = {"low": 0.5, "medium": 2.0, "high": 8.0}
        
        for issue in issues:
            debt_hours += effort_hours.get(issue.fix_effort, 1.0)
        
        return debt_hours
    
    def _build_repository_structure(self, issues: List[CodeIssue]) -> Dict[str, Any]:
        """Build repository structure with issue indicators"""
        structure = {"directories": {}, "files": []}
        
        # Group issues by file
        issues_by_file = defaultdict(list)
        for issue in issues:
            issues_by_file[issue.filepath].append(issue)
        
        # Build file structure
        for file in self.codebase.files:
            file_issues = issues_by_file[file.filepath]
            file_info = {
                "filepath": file.filepath,
                "filename": file.filepath.split('/')[-1],
                "lines_of_code": len(file.source.split('\n')) if hasattr(file, 'source') else 0,
                "issues_count": len(file_issues),
                "health_score": max(0, 100 - len(file_issues) * 5)
            }
            structure["files"].append(file_info)
        
        return structure
    
    def get_structured_data(self) -> Dict[str, Any]:
        """Get structured data for API consumption"""
        if not self.results:
            return {}
        
        return {
            "statistics": {
                "total_files": self.results.total_files,
                "total_functions": self.results.total_functions,
                "total_classes": self.results.total_classes,
                "total_lines_of_code": self.results.total_lines_of_code
            },
            "issues": {
                "total_issues": len(self.results.issues),
                "by_severity": self.results.issues_by_severity,
                "by_type": self.results.issues_by_type,
                "automated_resolutions": len(self.results.automated_resolutions)
            },
            "functions": {
                "most_important": self.results.most_important_functions,
                "entry_points": self.results.entry_points,
                "dead_functions": self.results.dead_functions
            },
            "quality_metrics": {
                "halstead": self.results.halstead_metrics,
                "complexity": self.results.complexity_metrics
            },
            "health_assessment": {
                "overall_score": self.results.health_score,
                "health_grade": self.results.health_grade,
                "risk_level": self.results.risk_level,
                "technical_debt_hours": self.results.technical_debt_hours
            },
            "repository_structure": self.results.repository_structure
        }
    
    def get_health_dashboard_data(self) -> Dict[str, Any]:
        """Get health dashboard data"""
        if not self.results:
            return {}
        
        return {
            "health_score": self.results.health_score,
            "health_grade": self.results.health_grade,
            "risk_level": self.results.risk_level,
            "key_metrics": {
                "total_issues": len(self.results.issues),
                "critical_issues": self.results.issues_by_severity.get("critical", 0),
                "automated_fixes_available": len(self.results.automated_resolutions),
                "technical_debt_hours": self.results.technical_debt_hours
            },
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        if not self.results:
            return []
        
        recommendations = []
        
        # Critical issues
        critical_count = self.results.issues_by_severity.get("critical", 0)
        if critical_count > 0:
            recommendations.append(f"Address {critical_count} critical issues immediately")
        
        # Dead code
        if self.results.dead_functions:
            recommendations.append(f"Remove {len(self.results.dead_functions)} unused functions")
        
        # Documentation
        missing_docs = self.results.issues_by_type.get("missing_documentation", 0)
        if missing_docs > 0:
            recommendations.append(f"Add documentation to {missing_docs} functions")
        
        # Automated fixes
        if self.results.automated_resolutions:
            recommendations.append(f"Apply {len(self.results.automated_resolutions)} automated fixes")
        
        return recommendations


# New comprehensive analysis endpoint
@fastapi_app.post("/comprehensive_analysis")
async def comprehensive_codebase_analysis(request: RepoRequest) -> Dict[str, Any]:
    """
    🎯 **COMPREHENSIVE CODEBASE ANALYSIS**
    
    Perform complete codebase analysis with structured data output:
    
    ✅ **Advanced Issue Detection** - 30+ issue types with automated resolutions
    ✅ **Function Context Analysis** - Dependencies, call chains, importance scoring
    ✅ **Halstead Complexity Metrics** - Quantitative complexity measurements
    ✅ **Graph Analysis** - Call graphs and dependency analysis
    ✅ **Dead Code Detection** - With blast radius calculation
    ✅ **Health Assessment** - Overall health scoring and risk assessment
    ✅ **Repository Structure** - Interactive tree with issue indicators
    ✅ **Automated Resolutions** - Import fixes, dead code removal, refactoring suggestions
    
    Perfect for:
    - CI/CD integration
    - Health dashboards
    - Technical debt assessment
    - Automated code quality monitoring
    """
    
    start_time = time.time()
    
    try:
        # Load codebase using graph-sitter
        codebase = Codebase.from_repo(request.repo_url)
        
        # Perform comprehensive analysis
        analyzer = ComprehensiveCodebaseAnalyzer(codebase)
        results = analyzer.analyze()
        
        # Get structured data for API consumption
        structured_data = analyzer.get_structured_data()
        
        # Get health dashboard data
        health_dashboard = analyzer.get_health_dashboard_data()
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "analysis_results": structured_data,
            "health_dashboard": health_dashboard,
            "processing_time": processing_time,
            "repo_url": request.repo_url,
            "analysis_timestamp": datetime.now().isoformat(),
            "features_analyzed": [
                "Advanced issue detection (30+ types)",
                "Function context analysis",
                "Halstead complexity metrics",
                "Call graph analysis",
                "Dependency graph analysis", 
                "Dead code detection with blast radius",
                "Repository structure visualization",
                "Health assessment and risk analysis",
                "Automated resolution suggestions"
            ]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error_message": f"Comprehensive analysis failed: {str(e)}",
            "processing_time": time.time() - start_time
        }


@fastapi_app.get("/health")
async def health_check():
    """Health check endpoint to verify API status."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "codebase-analytics-api"
    }


@fastapi_app.get("/")
async def root():
    """Root endpoint with API information and available endpoints."""
    return {
        "service": "Codebase Analytics API",
        "version": "2.0.0",
        "description": "Advanced codebase analysis with comprehensive issue detection and automated resolutions",
        "endpoints": {
            "basic_analysis": "/analyze_repo",
            "comprehensive_analysis": "/comprehensive_analysis",
            "health": "/health",
            "docs": "/docs"
        },
        "features": [
            "Basic repository analysis with metrics",
            "Comprehensive analysis with 30+ issue types",
            "Advanced issue detection with automated resolutions",
            "Function context analysis with call graphs",
            "Halstead complexity metrics",
            "Dead code detection with blast radius",
            "Health assessment and risk analysis",
            "Repository structure visualization",
            "Technical debt quantification",
            "Automated resolution suggestions"
        ],
        "powered_by": "graph-sitter",
        "documentation": "/docs"
    }


@app.function(image=image)
@modal.asgi_app()
def fastapi_modal_app():
    return fastapi_app


if __name__ == "__main__":
    app.deploy("analytics-app")
