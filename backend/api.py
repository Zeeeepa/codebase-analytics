from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List, Tuple, Any
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

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "codegen", "fastapi", "uvicorn", "gitpython", "requests", "pydantic", "datetime"
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


class ComprehensiveCodebaseAnalyzer:
    """
    Advanced codebase analyzer using comprehensive graph-sitter capabilities.
    
    Provides error detection, dependency analysis, call graph analysis,
    dead code detection, and comprehensive codebase state reporting.
    """
    
    def __init__(self, codebase: Codebase):
        self.codebase = codebase
        self.issues = []
        self.function_contexts = {}
        self.entry_points = []
        self.dead_code_items = []
        
    def analyze_comprehensive(self) -> Dict[str, Any]:
        """Perform comprehensive analysis of the entire codebase."""
        print("🔍 Starting comprehensive codebase analysis...")
        
        # Initialize with advanced configuration
        self._initialize_advanced_analysis()
        
        # Perform all analysis types
        repository_structure = self._analyze_repository_structure()
        error_analysis = self._detect_errors_and_issues()
        dependency_analysis = self._analyze_dependencies()
        call_graph_analysis = self._analyze_call_graph()
        dead_code_analysis = self._analyze_dead_code()
        entry_points = self._find_entry_points()
        function_contexts = self._build_function_contexts()
        symbol_analysis = self._analyze_symbols()
        import_analysis = self._analyze_imports()
        inheritance_analysis = self._analyze_inheritance()
        
        return {
            "repository_structure": repository_structure,
            "summary": self._generate_summary(),
            "most_important_functions": self._find_most_important_functions(),
            "dead_code_analysis": dead_code_analysis,
            "entry_points": entry_points,
            "issues_by_severity": self._group_issues_by_severity(),
            "function_contexts": function_contexts,
            "call_graph_metrics": call_graph_analysis,
            "dependency_metrics": dependency_analysis,
            "symbol_analysis": symbol_analysis,
            "import_analysis": import_analysis,
            "inheritance_analysis": inheritance_analysis,
            "error_detection": error_analysis,
            "codebase_health_score": self._calculate_health_score()
        }
    
    def _initialize_advanced_analysis(self):
        """Initialize advanced analysis with optimal configuration."""
        # Use advanced graph-sitter features
        try:
            # Enable comprehensive analysis features
            if hasattr(self.codebase, 'ctx') and hasattr(self.codebase.ctx, 'config'):
                config = self.codebase.ctx.config
                config.method_usages = True
                config.generics = True
                config.full_range_index = True
                config.sync_enabled = True
        except Exception as e:
            print(f"⚠️ Could not enable advanced features: {e}")
    
    def _analyze_repository_structure(self) -> Dict[str, Any]:
        """Analyze repository structure and organization."""
        files = list(self.codebase.files)
        directories = list(self.codebase.directories) if hasattr(self.codebase, 'directories') else []
        
        # Language distribution
        language_stats = {}
        file_sizes = []
        
        for file in files:
            # Get file extension
            ext = file.filepath.split('.')[-1] if '.' in file.filepath else 'no_extension'
            language_stats[ext] = language_stats.get(ext, 0) + 1
            
            # Calculate file size
            if hasattr(file, 'source'):
                lines = len(file.source.splitlines())
                file_sizes.append(lines)
        
        return {
            "total_files": len(files),
            "total_directories": len(directories),
            "language_distribution": language_stats,
            "file_size_stats": {
                "average": sum(file_sizes) / len(file_sizes) if file_sizes else 0,
                "max": max(file_sizes) if file_sizes else 0,
                "min": min(file_sizes) if file_sizes else 0
            },
            "largest_files": sorted([
                {"path": f.filepath, "lines": len(f.source.splitlines())}
                for f in files if hasattr(f, 'source')
            ], key=lambda x: x["lines"], reverse=True)[:10]
        }
    
    def _detect_errors_and_issues(self) -> Dict[str, Any]:
        """Detect various types of errors and issues in the codebase."""
        critical_issues = []
        major_issues = []
        minor_issues = []
        
        # Analyze functions for issues
        for func in self.codebase.functions:
            issues = self._analyze_function_issues(func)
            for issue in issues:
                if issue["severity"] == "critical":
                    critical_issues.append(issue)
                elif issue["severity"] == "major":
                    major_issues.append(issue)
                else:
                    minor_issues.append(issue)
        
        # Analyze classes for issues
        for cls in self.codebase.classes:
            issues = self._analyze_class_issues(cls)
            for issue in issues:
                if issue["severity"] == "critical":
                    critical_issues.append(issue)
                elif issue["severity"] == "major":
                    major_issues.append(issue)
                else:
                    minor_issues.append(issue)
        
        return {
            "critical_issues": critical_issues,
            "major_issues": major_issues,
            "minor_issues": minor_issues,
            "total_issues": len(critical_issues) + len(major_issues) + len(minor_issues)
        }
    
    def _analyze_function_issues(self, func) -> List[Dict[str, Any]]:
        """Analyze a function for various issues."""
        issues = []
        
        try:
            # Check for missing docstring
            if not hasattr(func, 'docstring') or not func.docstring:
                issues.append({
                    "type": "missing_documentation",
                    "severity": "minor",
                    "message": f"Function '{func.name}' lacks documentation",
                    "function": func.name,
                    "file": getattr(func, 'file', {}).get('filepath', 'unknown')
                })
            
            # Check for high complexity
            if hasattr(func, 'code_block'):
                complexity = calculate_cyclomatic_complexity(func)
                if complexity > 15:
                    issues.append({
                        "type": "high_complexity",
                        "severity": "major",
                        "message": f"Function '{func.name}' has high complexity ({complexity})",
                        "function": func.name,
                        "complexity": complexity,
                        "file": getattr(func, 'file', {}).get('filepath', 'unknown')
                    })
                elif complexity > 25:
                    issues.append({
                        "type": "extreme_complexity",
                        "severity": "critical",
                        "message": f"Function '{func.name}' has extreme complexity ({complexity})",
                        "function": func.name,
                        "complexity": complexity,
                        "file": getattr(func, 'file', {}).get('filepath', 'unknown')
                    })
            
            # Check for unused parameters
            if hasattr(func, 'parameters'):
                for param in func.parameters:
                    # This is a simplified check - in real implementation would need AST analysis
                    if hasattr(func, 'code_block') and hasattr(func.code_block, 'source'):
                        if param.name not in func.code_block.source:
                            issues.append({
                                "type": "unused_parameter",
                                "severity": "minor",
                                "message": f"Parameter '{param.name}' appears unused in function '{func.name}'",
                                "function": func.name,
                                "parameter": param.name,
                                "file": getattr(func, 'file', {}).get('filepath', 'unknown')
                            })
            
            # Check for no usages (potential dead code)
            if hasattr(func, 'usages') and len(func.usages) == 0:
                # Check if it's an entry point
                if not self._is_entry_point(func):
                    issues.append({
                        "type": "dead_code",
                        "severity": "major",
                        "message": f"Function '{func.name}' appears to be unused",
                        "function": func.name,
                        "file": getattr(func, 'file', {}).get('filepath', 'unknown')
                    })
            
        except Exception as e:
            issues.append({
                "type": "analysis_error",
                "severity": "minor",
                "message": f"Could not fully analyze function '{func.name}': {str(e)}",
                "function": func.name,
                "file": getattr(func, 'file', {}).get('filepath', 'unknown')
            })
        
        return issues
    
    def _analyze_class_issues(self, cls) -> List[Dict[str, Any]]:
        """Analyze a class for various issues."""
        issues = []
        
        try:
            # Check for missing docstring
            if not hasattr(cls, 'docstring') or not cls.docstring:
                issues.append({
                    "type": "missing_documentation",
                    "severity": "minor",
                    "message": f"Class '{cls.name}' lacks documentation",
                    "class": cls.name,
                    "file": getattr(cls, 'file', {}).get('filepath', 'unknown')
                })
            
            # Check for too many methods (God class)
            if hasattr(cls, 'methods'):
                method_count = len(list(cls.methods))
                if method_count > 20:
                    issues.append({
                        "type": "god_class",
                        "severity": "major",
                        "message": f"Class '{cls.name}' has too many methods ({method_count})",
                        "class": cls.name,
                        "method_count": method_count,
                        "file": getattr(cls, 'file', {}).get('filepath', 'unknown')
                    })
            
            # Check for deep inheritance
            if hasattr(cls, 'superclasses'):
                inheritance_depth = len(list(cls.superclasses))
                if inheritance_depth > 5:
                    issues.append({
                        "type": "deep_inheritance",
                        "severity": "major",
                        "message": f"Class '{cls.name}' has deep inheritance chain ({inheritance_depth})",
                        "class": cls.name,
                        "inheritance_depth": inheritance_depth,
                        "file": getattr(cls, 'file', {}).get('filepath', 'unknown')
                    })
            
        except Exception as e:
            issues.append({
                "type": "analysis_error",
                "severity": "minor",
                "message": f"Could not fully analyze class '{cls.name}': {str(e)}",
                "class": cls.name,
                "file": getattr(cls, 'file', {}).get('filepath', 'unknown')
            })
        
        return issues
    
    def _analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze dependency relationships in the codebase."""
        dependency_graph = {}
        circular_deps = []
        
        for file in self.codebase.files:
            file_deps = []
            if hasattr(file, 'imports'):
                for imp in file.imports:
                    if hasattr(imp, 'resolved_symbol') and hasattr(imp.resolved_symbol, 'file'):
                        dep_file = imp.resolved_symbol.file.filepath
                        if dep_file != file.filepath:
                            file_deps.append(dep_file)
            
            dependency_graph[file.filepath] = file_deps
        
        # Simple circular dependency detection
        for file_path, deps in dependency_graph.items():
            for dep in deps:
                if dep in dependency_graph and file_path in dependency_graph[dep]:
                    circular_deps.append((file_path, dep))
        
        return {
            "dependency_graph": dependency_graph,
            "circular_dependencies": circular_deps,
            "total_dependencies": sum(len(deps) for deps in dependency_graph.values())
        }
    
    def _analyze_call_graph(self) -> Dict[str, Any]:
        """Analyze function call relationships."""
        call_graph = {}
        most_called = {}
        most_calls = {}
        
        for func in self.codebase.functions:
            # Functions this function calls
            calls = []
            if hasattr(func, 'function_calls'):
                calls = [call.name for call in func.function_calls]
            call_graph[func.name] = calls
            
            # Track usage counts
            if hasattr(func, 'usages'):
                most_called[func.name] = len(func.usages)
            
            # Track call counts
            most_calls[func.name] = len(calls)
        
        return {
            "call_graph": call_graph,
            "most_called_function": max(most_called.items(), key=lambda x: x[1]) if most_called else None,
            "function_with_most_calls": max(most_calls.items(), key=lambda x: x[1]) if most_calls else None,
            "total_function_calls": sum(most_calls.values())
        }
    
    def _analyze_dead_code(self) -> Dict[str, Any]:
        """Analyze dead code in the codebase."""
        dead_functions = []
        dead_classes = []
        
        for func in self.codebase.functions:
            if hasattr(func, 'usages') and len(func.usages) == 0:
                if not self._is_entry_point(func):
                    dead_functions.append({
                        "name": func.name,
                        "file": getattr(func, 'file', {}).get('filepath', 'unknown'),
                        "type": "function"
                    })
        
        for cls in self.codebase.classes:
            if hasattr(cls, 'usages') and len(cls.usages) == 0:
                dead_classes.append({
                    "name": cls.name,
                    "file": getattr(cls, 'file', {}).get('filepath', 'unknown'),
                    "type": "class"
                })
        
        return {
            "dead_functions": dead_functions,
            "dead_classes": dead_classes,
            "total_dead_items": len(dead_functions) + len(dead_classes)
        }
    
    def _find_entry_points(self) -> List[Dict[str, Any]]:
        """Find entry points in the codebase."""
        entry_points = []
        
        for func in self.codebase.functions:
            if self._is_entry_point(func):
                entry_points.append({
                    "name": func.name,
                    "file": getattr(func, 'file', {}).get('filepath', 'unknown'),
                    "type": "function"
                })
        
        return entry_points
    
    def _is_entry_point(self, func) -> bool:
        """Check if a function is an entry point."""
        # Common entry point patterns
        entry_patterns = ['main', '__main__', 'run', 'start', 'execute', 'handler']
        
        # Check function name
        if any(pattern in func.name.lower() for pattern in entry_patterns):
            return True
        
        # Check if it's a special method
        if func.name.startswith('__') and func.name.endswith('__'):
            return True
        
        # Check if it has decorators that indicate entry points
        if hasattr(func, 'decorators'):
            for decorator in func.decorators:
                if hasattr(decorator, 'name'):
                    if any(pattern in decorator.name.lower() for pattern in ['app.', 'route', 'endpoint', 'api']):
                        return True
        
        return False
    
    def _build_function_contexts(self) -> Dict[str, Any]:
        """Build detailed context for each function."""
        contexts = {}
        
        for func in self.codebase.functions:
            context = {
                "name": func.name,
                "filepath": getattr(func, 'file', {}).get('filepath', 'unknown'),
                "parameters": [p.name for p in func.parameters] if hasattr(func, 'parameters') else [],
                "dependencies": [],
                "function_calls": [],
                "called_by": [],
                "issues": [],
                "is_entry_point": self._is_entry_point(func),
                "is_dead_code": False,
                "complexity": 0,
                "max_call_chain": []
            }
            
            # Get dependencies
            if hasattr(func, 'dependencies'):
                context["dependencies"] = [dep.name for dep in func.dependencies]
            
            # Get function calls
            if hasattr(func, 'function_calls'):
                context["function_calls"] = [call.name for call in func.function_calls]
            
            # Get called by (usages)
            if hasattr(func, 'usages'):
                context["called_by"] = [usage.name for usage in func.usages if hasattr(usage, 'name')]
                context["is_dead_code"] = len(func.usages) == 0 and not context["is_entry_point"]
            
            # Get complexity
            if hasattr(func, 'code_block'):
                context["complexity"] = calculate_cyclomatic_complexity(func)
            
            # Build call chain
            context["max_call_chain"] = self._get_call_chain(func)
            
            contexts[func.name] = context
        
        return contexts
    
    def _get_call_chain(self, func, visited=None, max_depth=10) -> List[str]:
        """Get the call chain for a function."""
        if visited is None:
            visited = set()
        
        if func.name in visited or max_depth <= 0:
            return [func.name]
        
        visited.add(func.name)
        chain = [func.name]
        
        if hasattr(func, 'function_calls') and func.function_calls:
            # Get the first function call to continue the chain
            first_call = func.function_calls[0]
            if hasattr(first_call, 'name'):
                # Find the actual function object
                called_func = None
                for f in self.codebase.functions:
                    if f.name == first_call.name:
                        called_func = f
                        break
                
                if called_func:
                    sub_chain = self._get_call_chain(called_func, visited.copy(), max_depth - 1)
                    chain.extend(sub_chain[1:])  # Avoid duplicating the function name
        
        return chain
    
    def _analyze_symbols(self) -> Dict[str, Any]:
        """Analyze symbols in the codebase."""
        symbol_stats = {
            "total_symbols": len(list(self.codebase.symbols)),
            "functions": len(list(self.codebase.functions)),
            "classes": len(list(self.codebase.classes)),
            "global_vars": len(list(self.codebase.global_vars)) if hasattr(self.codebase, 'global_vars') else 0,
            "interfaces": len(list(self.codebase.interfaces)) if hasattr(self.codebase, 'interfaces') else 0
        }
        
        # Find most used symbols
        most_used = []
        for symbol in self.codebase.symbols:
            if hasattr(symbol, 'usages'):
                usage_count = len(symbol.usages)
                if usage_count > 0:
                    most_used.append({
                        "name": symbol.name,
                        "usage_count": usage_count,
                        "type": type(symbol).__name__
                    })
        
        most_used.sort(key=lambda x: x["usage_count"], reverse=True)
        
        return {
            "symbol_statistics": symbol_stats,
            "most_used_symbols": most_used[:10]
        }
    
    def _analyze_imports(self) -> Dict[str, Any]:
        """Analyze import patterns."""
        import_stats = {}
        external_imports = []
        
        for imp in self.codebase.imports:
            if hasattr(imp, 'source'):
                import_stats[imp.source] = import_stats.get(imp.source, 0) + 1
            
            # Check if external
            if hasattr(imp, 'is_external') and imp.is_external:
                external_imports.append(imp.source)
        
        return {
            "total_imports": len(list(self.codebase.imports)),
            "external_imports": list(set(external_imports)),
            "most_imported": sorted(import_stats.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def _analyze_inheritance(self) -> Dict[str, Any]:
        """Analyze inheritance patterns."""
        inheritance_chains = []
        deepest_inheritance = None
        max_depth = 0
        
        for cls in self.codebase.classes:
            if hasattr(cls, 'superclasses'):
                chain_depth = len(list(cls.superclasses))
                if chain_depth > max_depth:
                    max_depth = chain_depth
                    deepest_inheritance = {
                        "class": cls.name,
                        "depth": chain_depth,
                        "chain": [sc.name for sc in cls.superclasses]
                    }
                
                if chain_depth > 0:
                    inheritance_chains.append({
                        "class": cls.name,
                        "depth": chain_depth,
                        "parents": [sc.name for sc in cls.superclasses]
                    })
        
        return {
            "total_inheritance_chains": len(inheritance_chains),
            "deepest_inheritance": deepest_inheritance,
            "average_inheritance_depth": sum(chain["depth"] for chain in inheritance_chains) / len(inheritance_chains) if inheritance_chains else 0
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive summary."""
        return {
            "total_files": len(list(self.codebase.files)),
            "total_functions": len(list(self.codebase.functions)),
            "total_classes": len(list(self.codebase.classes)),
            "total_symbols": len(list(self.codebase.symbols)),
            "total_imports": len(list(self.codebase.imports)),
            "total_issues": len(self.issues),
            "critical_issues": len([i for i in self.issues if i.get("severity") == "critical"]),
            "major_issues": len([i for i in self.issues if i.get("severity") == "major"]),
            "minor_issues": len([i for i in self.issues if i.get("severity") == "minor"]),
            "dead_code_items": len(self.dead_code_items),
            "entry_points": len(self.entry_points)
        }
    
    def _find_most_important_functions(self) -> Dict[str, Any]:
        """Find the most important functions in the codebase."""
        most_called = None
        most_calls = None
        deepest_inheritance = None
        
        max_usage = 0
        max_calls = 0
        max_inheritance = 0
        
        # Find most called function
        for func in self.codebase.functions:
            if hasattr(func, 'usages'):
                usage_count = len(func.usages)
                if usage_count > max_usage:
                    max_usage = usage_count
                    most_called = {
                        "name": func.name,
                        "usage_count": usage_count,
                        "file": getattr(func, 'file', {}).get('filepath', 'unknown')
                    }
            
            if hasattr(func, 'function_calls'):
                call_count = len(func.function_calls)
                if call_count > max_calls:
                    max_calls = call_count
                    most_calls = {
                        "name": func.name,
                        "call_count": call_count,
                        "calls": [call.name for call in func.function_calls][:5],  # First 5 calls
                        "file": getattr(func, 'file', {}).get('filepath', 'unknown')
                    }
        
        # Find class with deepest inheritance
        for cls in self.codebase.classes:
            if hasattr(cls, 'superclasses'):
                inheritance_depth = len(list(cls.superclasses))
                if inheritance_depth > max_inheritance:
                    max_inheritance = inheritance_depth
                    deepest_inheritance = {
                        "name": cls.name,
                        "chain_depth": inheritance_depth,
                        "chain": [sc.name for sc in cls.superclasses],
                        "file": getattr(cls, 'file', {}).get('filepath', 'unknown')
                    }
        
        return {
            "most_called": most_called,
            "most_calls": most_calls,
            "deepest_inheritance": deepest_inheritance
        }
    
    def _group_issues_by_severity(self) -> Dict[str, List[Dict[str, Any]]]:
        """Group issues by severity level."""
        grouped = {
            "critical": [],
            "major": [],
            "minor": []
        }
        
        for issue in self.issues:
            severity = issue.get("severity", "minor")
            if severity in grouped:
                grouped[severity].append(issue)
        
        return grouped
    
    def _calculate_health_score(self) -> Dict[str, Any]:
        """Calculate overall codebase health score."""
        total_functions = len(list(self.codebase.functions))
        total_classes = len(list(self.codebase.classes))
        total_issues = len(self.issues)
        
        # Simple health scoring algorithm
        base_score = 100
        
        # Deduct points for issues
        critical_penalty = len([i for i in self.issues if i.get("severity") == "critical"]) * 10
        major_penalty = len([i for i in self.issues if i.get("severity") == "major"]) * 5
        minor_penalty = len([i for i in self.issues if i.get("severity") == "minor"]) * 1
        
        health_score = max(0, base_score - critical_penalty - major_penalty - minor_penalty)
        
        # Calculate grade
        if health_score >= 90:
            grade = "A"
        elif health_score >= 80:
            grade = "B"
        elif health_score >= 70:
            grade = "C"
        elif health_score >= 60:
            grade = "D"
        else:
            grade = "F"
        
        return {
            "score": health_score,
            "grade": grade,
            "total_issues": total_issues,
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for improving codebase health."""
        recommendations = []
        
        critical_count = len([i for i in self.issues if i.get("severity") == "critical"])
        major_count = len([i for i in self.issues if i.get("severity") == "major"])
        minor_count = len([i for i in self.issues if i.get("severity") == "minor"])
        
        if critical_count > 0:
            recommendations.append(f"🚨 Address {critical_count} critical issues immediately")
        
        if major_count > 5:
            recommendations.append(f"⚠️ Reduce {major_count} major issues to improve maintainability")
        
        if minor_count > 20:
            recommendations.append(f"📝 Consider addressing {minor_count} minor issues for code quality")
        
        if len(self.dead_code_items) > 0:
            recommendations.append(f"🗑️ Remove {len(self.dead_code_items)} dead code items")
        
        return recommendations


class RepoRequest(BaseModel):
    repo_url: str


@fastapi_app.post("/analyze_repo")
async def analyze_repo(request: RepoRequest) -> Dict[str, Any]:
    """
    🔍 COMPREHENSIVE CODEBASE ANALYSIS
    
    Performs advanced graph-sitter analysis including:
    - Error detection (critical, major, minor issues)
    - Dead code analysis
    - Call graph analysis
    - Dependency analysis
    - Symbol usage analysis
    - Entry point detection
    - Function context mapping
    - Inheritance analysis
    - Codebase health scoring
    """
    try:
        repo_url = request.repo_url
        print(f"🚀 Starting comprehensive analysis of {repo_url}")
        
        # Initialize codebase with advanced configuration
        from graph_sitter.configs.models.codebase import CodebaseConfig
        
        config = CodebaseConfig(
            # Performance optimizations
            method_usages=True,          # Enable method usage resolution
            generics=True,               # Enable generic type resolution
            sync_enabled=True,           # Enable graph sync during commits
            
            # Advanced analysis
            full_range_index=True,       # Full range-to-node mapping
            # py_resolve_syspath=True,     # Resolve sys.path imports
            
            # Experimental features
            # exp_lazy_graph=False,        # Lazy graph construction
        )
        
        codebase = Codebase.from_repo(repo_url, config=config)
        
        # Initialize comprehensive analyzer
        analyzer = ComprehensiveCodebaseAnalyzer(codebase)
        
        # Perform comprehensive analysis
        comprehensive_results = analyzer.analyze_comprehensive()
        
        # Get monthly commits (legacy feature)
        monthly_commits = get_monthly_commits(repo_url)
        
        # Calculate traditional metrics for backward compatibility
        num_files = len(list(codebase.files))
        num_functions = len(list(codebase.functions))
        num_classes = len(list(codebase.classes))
        
        total_loc = total_lloc = total_sloc = total_comments = 0
        total_complexity = 0
        total_volume = 0
        total_mi = 0
        total_doi = 0
        
        for file in codebase.files:
            if hasattr(file, 'source'):
                loc, lloc, sloc, comments = count_lines(file.source)
                total_loc += loc
                total_lloc += lloc
                total_sloc += sloc
                total_comments += comments

        callables = list(codebase.functions) + [m for c in codebase.classes for m in c.methods if hasattr(c, 'methods')]

        num_callables = 0
        for func in callables:
            if not hasattr(func, "code_block"):
                continue

            complexity = calculate_cyclomatic_complexity(func)
            operators, operands = get_operators_and_operands(func)
            volume, _, _, _, _ = calculate_halstead_volume(operators, operands)
            loc = len(func.code_block.source.splitlines()) if hasattr(func.code_block, 'source') else 0
            mi_score = calculate_maintainability_index(volume, complexity, loc)

            total_complexity += complexity
            total_volume += volume
            total_mi += mi_score
            num_callables += 1

        for cls in codebase.classes:
            doi = calculate_doi(cls)
            total_doi += doi

        desc = get_github_repo_description(repo_url)

        # Combine comprehensive analysis with legacy metrics
        results = {
            "repo_url": repo_url,
            "description": desc,
            "analysis_timestamp": datetime.now().isoformat(),
            "monthly_commits": monthly_commits,
            
            # 🔍 COMPREHENSIVE ANALYSIS RESULTS
            **comprehensive_results,
            
            # 📊 LEGACY METRICS (for backward compatibility)
            "legacy_metrics": {
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
                },
                "depth_of_inheritance": {
                    "average": total_doi / len(list(codebase.classes)) if list(codebase.classes) else 0,
                },
                "halstead_metrics": {
                    "total_volume": int(total_volume),
                    "average_volume": int(total_volume / num_callables) if num_callables > 0 else 0,
                },
                "maintainability_index": {
                    "average": int(total_mi / num_callables) if num_callables > 0 else 0,
                },
                "num_files": num_files,
                "num_functions": num_functions,
                "num_classes": num_classes,
            }
        }
        
        # 📈 PRINT COMPREHENSIVE SUMMARY
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE CODEBASE ANALYSIS COMPLETE")
        print("="*60)
        
        summary = comprehensive_results.get('summary', {})
        print(f"📁 Total Files: {summary.get('total_files', 0)}")
        print(f"🔧 Total Functions: {summary.get('total_functions', 0)}")
        print(f"🏗️ Total Classes: {summary.get('total_classes', 0)}")
        print(f"🔗 Total Symbols: {summary.get('total_symbols', 0)}")
        print(f"📦 Total Imports: {summary.get('total_imports', 0)}")
        print(f"🚨 Total Issues: {summary.get('total_issues', 0)}")
        print(f"⚠️ Critical Issues: {summary.get('critical_issues', 0)}")
        print(f"👉 Major Issues: {summary.get('major_issues', 0)}")
        print(f"🔍 Minor Issues: {summary.get('minor_issues', 0)}")
        print(f"💀 Dead Code Items: {summary.get('dead_code_items', 0)}")
        print(f"🎯 Entry Points: {summary.get('entry_points', 0)}")
        
        # Show health score
        health = comprehensive_results.get('codebase_health_score', {})
        print(f"\n🏥 CODEBASE HEALTH: {health.get('score', 0)}/100 (Grade: {health.get('grade', 'F')})")
        
        # Show most important functions
        important = comprehensive_results.get('most_important_functions', {})
        if important.get('most_called'):
            most_called = important['most_called']
            print(f"\n📞 Most Called Function: {most_called.get('name', 'N/A')} ({most_called.get('usage_count', 0)} usages)")
        
        if important.get('most_calls'):
            most_calls = important['most_calls']
            print(f"📈 Function with Most Calls: {most_calls.get('name', 'N/A')} ({most_calls.get('call_count', 0)} calls)")
        
        print("="*60)
        
        return results
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comprehensive analysis failed: {str(e)}")


@app.function(image=image)
@modal.asgi_app()
def fastapi_modal_app():
    return fastapi_app


if __name__ == "__main__":
    app.deploy("analytics-app")
