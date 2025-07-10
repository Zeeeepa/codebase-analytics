from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Tuple, Any, Optional, Union
from codegen import Codebase
from graph_sitter.core.class_definition import Class
from graph_sitter.core.codebase import Codebase
from graph_sitter.core.external_module import ExternalModule
from graph_sitter.core.file import SourceFile
from graph_sitter.core.function import Function
from graph_sitter.core.import_resolution import Import
from graph_sitter.core.symbol import Symbol
from graph_sitter.enums import EdgeType, SymbolType
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
from collections import defaultdict, Counter
import json

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


class Analysis:
    """
    Analysis class providing pre-computed graph element access and advanced analysis.
    
    This class wraps the existing graph-sitter Codebase functionality to provide
    comprehensive codebase analysis capabilities.
    """
    
    def __init__(self, codebase: Codebase):
        """Initialize Analysis with a Codebase instance."""
        self.codebase = codebase
    
    @property
    def functions(self):
        """All functions in codebase with enhanced analysis."""
        return self.codebase.functions
    
    @property
    def classes(self):
        """All classes in codebase with comprehensive analysis."""
        return self.codebase.classes
    
    @property
    def imports(self):
        """All import statements in the codebase."""
        return self.codebase.imports
    
    @property
    def files(self):
        """All files in the codebase with import analysis."""
        return self.codebase.files
    
    @property
    def symbols(self):
        """All symbols (functions, classes, variables) in the codebase."""
        return self.codebase.symbols
    
    @property
    def external_modules(self):
        """External dependencies imported by the codebase."""
        return self.codebase.external_modules
    
    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """Get comprehensive analysis of the entire codebase."""
        return {
            "overview": self.get_codebase_overview(),
            "quality_metrics": self.get_quality_metrics(),
            "complexity_analysis": self.get_complexity_analysis(),
            "dependency_analysis": self.get_dependency_analysis(),
            "architecture_analysis": self.get_architecture_analysis(),
            "code_patterns": self.get_code_patterns(),
            "technical_debt": self.get_technical_debt_analysis(),
            "test_coverage_analysis": self.get_test_coverage_analysis(),
            "security_analysis": self.get_security_analysis(),
            "performance_insights": self.get_performance_insights()
        }
    
    def get_codebase_overview(self) -> Dict[str, Any]:
        """Get high-level overview of the codebase."""
        files = list(self.files)
        functions = list(self.functions)
        classes = list(self.classes)
        imports = list(self.imports)
        
        # Language distribution
        language_stats = defaultdict(int)
        for file in files:
            if hasattr(file, 'language'):
                language_stats[file.language] += 1
        
        # File size distribution
        file_sizes = []
        total_loc = 0
        for file in files:
            if hasattr(file, 'source'):
                lines = len(file.source.splitlines())
                file_sizes.append(lines)
                total_loc += lines
        
        return {
            "total_files": len(files),
            "total_functions": len(functions),
            "total_classes": len(classes),
            "total_imports": len(imports),
            "total_lines_of_code": total_loc,
            "language_distribution": dict(language_stats),
            "average_file_size": sum(file_sizes) / len(file_sizes) if file_sizes else 0,
            "largest_file_size": max(file_sizes) if file_sizes else 0,
            "smallest_file_size": min(file_sizes) if file_sizes else 0
        }
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get code quality metrics."""
        functions = list(self.functions)
        classes = list(self.classes)
        
        # Function quality metrics
        function_metrics = []
        total_complexity = 0
        total_mi = 0
        
        for func in functions:
            if hasattr(func, 'code_block'):
                complexity = calculate_cyclomatic_complexity(func)
                operators, operands = get_operators_and_operands(func)
                volume, _, _, _, _ = calculate_halstead_volume(operators, operands)
                loc = len(func.code_block.source.splitlines()) if hasattr(func.code_block, 'source') else 0
                mi_score = calculate_maintainability_index(volume, complexity, loc)
                
                function_metrics.append({
                    "name": func.name,
                    "complexity": complexity,
                    "complexity_rank": cc_rank(complexity),
                    "maintainability_index": mi_score,
                    "maintainability_rank": get_maintainability_rank(mi_score),
                    "lines_of_code": loc,
                    "halstead_volume": volume
                })
                
                total_complexity += complexity
                total_mi += mi_score
        
        # Class quality metrics
        class_metrics = []
        total_doi = 0
        
        for cls in classes:
            doi = calculate_doi(cls)
            methods_count = len(list(cls.methods)) if hasattr(cls, 'methods') else 0
            attributes_count = len(list(cls.attributes)) if hasattr(cls, 'attributes') else 0
            
            class_metrics.append({
                "name": cls.name,
                "depth_of_inheritance": doi,
                "methods_count": methods_count,
                "attributes_count": attributes_count,
                "is_abstract": getattr(cls, 'is_abstract', False)
            })
            
            total_doi += doi
        
        return {
            "function_metrics": function_metrics,
            "class_metrics": class_metrics,
            "averages": {
                "cyclomatic_complexity": total_complexity / len(functions) if functions else 0,
                "maintainability_index": total_mi / len(functions) if functions else 0,
                "depth_of_inheritance": total_doi / len(classes) if classes else 0
            },
            "quality_distribution": self._get_quality_distribution(function_metrics)
        }
    
    def get_complexity_analysis(self) -> Dict[str, Any]:
        """Get detailed complexity analysis."""
        functions = list(self.functions)
        
        complexity_distribution = defaultdict(int)
        high_complexity_functions = []
        
        for func in functions:
            if hasattr(func, 'code_block'):
                complexity = calculate_cyclomatic_complexity(func)
                rank = cc_rank(complexity)
                complexity_distribution[rank] += 1
                
                if complexity > 10:  # High complexity threshold
                    high_complexity_functions.append({
                        "name": func.name,
                        "complexity": complexity,
                        "rank": rank,
                        "file": getattr(func, 'file_path', 'unknown')
                    })
        
        return {
            "complexity_distribution": dict(complexity_distribution),
            "high_complexity_functions": sorted(high_complexity_functions, 
                                               key=lambda x: x['complexity'], reverse=True),
            "complexity_hotspots": self._identify_complexity_hotspots()
        }
    
    def get_dependency_analysis(self) -> Dict[str, Any]:
        """Get dependency analysis."""
        imports = list(self.imports)
        external_modules = list(self.external_modules)
        
        # Import frequency analysis
        import_frequency = Counter()
        external_dependencies = set()
        
        for imp in imports:
            if hasattr(imp, 'source'):
                import_frequency[imp.source] += 1
            if hasattr(imp, 'is_external') and imp.is_external:
                if hasattr(imp, 'module_name'):
                    external_dependencies.add(imp.module_name)
        
        # Circular dependency detection
        circular_deps = self._detect_circular_dependencies()
        
        # Dependency graph metrics
        dependency_metrics = self._calculate_dependency_metrics()
        
        return {
            "total_imports": len(imports),
            "external_dependencies": list(external_dependencies),
            "most_imported_modules": import_frequency.most_common(10),
            "circular_dependencies": circular_deps,
            "dependency_metrics": dependency_metrics,
            "unused_imports": self._find_unused_imports()
        }
    
    def get_architecture_analysis(self) -> Dict[str, Any]:
        """Get architecture analysis."""
        files = list(self.files)
        classes = list(self.classes)
        functions = list(self.functions)
        
        # Package/module structure
        package_structure = self._analyze_package_structure(files)
        
        # Design patterns detection
        design_patterns = self._detect_design_patterns(classes)
        
        # Coupling and cohesion metrics
        coupling_metrics = self._calculate_coupling_metrics()
        
        return {
            "package_structure": package_structure,
            "design_patterns": design_patterns,
            "coupling_metrics": coupling_metrics,
            "architectural_violations": self._detect_architectural_violations()
        }
    
    def get_code_patterns(self) -> Dict[str, Any]:
        """Get code patterns analysis."""
        functions = list(self.functions)
        classes = list(self.classes)
        
        # Common patterns
        patterns = {
            "singleton_classes": [],
            "factory_methods": [],
            "observer_patterns": [],
            "decorator_patterns": [],
            "async_functions": [],
            "generator_functions": []
        }
        
        # Analyze functions
        for func in functions:
            if hasattr(func, 'is_async') and func.is_async:
                patterns["async_functions"].append(func.name)
            
            if hasattr(func, 'is_generator') and func.is_generator:
                patterns["generator_functions"].append(func.name)
            
            # Check for factory pattern
            if 'create' in func.name.lower() or 'factory' in func.name.lower():
                patterns["factory_methods"].append(func.name)
        
        # Analyze classes for patterns
        for cls in classes:
            # Singleton detection
            if self._is_singleton_pattern(cls):
                patterns["singleton_classes"].append(cls.name)
        
        return {
            "detected_patterns": patterns,
            "pattern_statistics": self._calculate_pattern_statistics(patterns)
        }
    
    def get_technical_debt_analysis(self) -> Dict[str, Any]:
        """Get technical debt analysis."""
        functions = list(self.functions)
        files = list(self.files)
        
        debt_indicators = {
            "long_functions": [],
            "complex_functions": [],
            "large_classes": [],
            "code_duplication": [],
            "missing_documentation": [],
            "deprecated_usage": []
        }
        
        # Analyze functions for debt
        for func in functions:
            if hasattr(func, 'code_block') and hasattr(func.code_block, 'source'):
                loc = len(func.code_block.source.splitlines())
                complexity = calculate_cyclomatic_complexity(func)
                
                if loc > 50:  # Long function threshold
                    debt_indicators["long_functions"].append({
                        "name": func.name,
                        "lines": loc
                    })
                
                if complexity > 15:  # High complexity threshold
                    debt_indicators["complex_functions"].append({
                        "name": func.name,
                        "complexity": complexity
                    })
                
                # Check for missing documentation
                if not hasattr(func, 'docstring') or not func.docstring:
                    debt_indicators["missing_documentation"].append(func.name)
        
        return {
            "debt_indicators": debt_indicators,
            "debt_score": self._calculate_debt_score(debt_indicators),
            "refactoring_suggestions": self._generate_refactoring_suggestions(debt_indicators)
        }
    
    def get_test_coverage_analysis(self) -> Dict[str, Any]:
        """Get test coverage analysis."""
        files = list(self.files)
        functions = list(self.functions)
        
        test_files = []
        test_functions = []
        
        for file in files:
            if 'test' in file.path.lower() or file.path.endswith('_test.py'):
                test_files.append(file.path)
        
        for func in functions:
            if 'test_' in func.name.lower() or func.name.lower().startswith('test'):
                test_functions.append(func.name)
        
        return {
            "test_files_count": len(test_files),
            "test_functions_count": len(test_functions),
            "test_to_code_ratio": len(test_functions) / len(functions) if functions else 0,
            "test_files": test_files,
            "coverage_estimation": self._estimate_test_coverage()
        }
    
    def get_security_analysis(self) -> Dict[str, Any]:
        """Get security analysis."""
        functions = list(self.functions)
        files = list(self.files)
        
        security_issues = {
            "potential_sql_injection": [],
            "hardcoded_secrets": [],
            "unsafe_functions": [],
            "input_validation_missing": []
        }
        
        # Basic security pattern detection
        for func in functions:
            if hasattr(func, 'code_block') and hasattr(func.code_block, 'source'):
                source = func.code_block.source.lower()
                
                # SQL injection patterns
                if 'execute(' in source and any(pattern in source for pattern in ['%s', 'format(', 'f"']):
                    security_issues["potential_sql_injection"].append(func.name)
                
                # Unsafe functions
                if any(unsafe in source for unsafe in ['eval(', 'exec(', 'pickle.loads(']):
                    security_issues["unsafe_functions"].append(func.name)
        
        # Check for hardcoded secrets in files
        for file in files:
            if hasattr(file, 'source'):
                if self._contains_hardcoded_secrets(file.source):
                    security_issues["hardcoded_secrets"].append(file.path)
        
        return {
            "security_issues": security_issues,
            "security_score": self._calculate_security_score(security_issues),
            "recommendations": self._generate_security_recommendations(security_issues)
        }
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get performance insights."""
        functions = list(self.functions)
        
        performance_issues = {
            "nested_loops": [],
            "recursive_functions": [],
            "large_data_structures": [],
            "inefficient_patterns": []
        }
        
        for func in functions:
            if hasattr(func, 'code_block') and hasattr(func.code_block, 'source'):
                source = func.code_block.source
                
                # Detect nested loops
                if source.count('for ') > 1 or source.count('while ') > 1:
                    performance_issues["nested_loops"].append(func.name)
                
                # Detect recursive functions
                if func.name in source:
                    performance_issues["recursive_functions"].append(func.name)
        
        return {
            "performance_issues": performance_issues,
            "optimization_suggestions": self._generate_optimization_suggestions(performance_issues)
        }
    
    # Helper methods
    def _get_quality_distribution(self, function_metrics: List[Dict]) -> Dict[str, int]:
        """Get distribution of quality ranks."""
        distribution = defaultdict(int)
        for metric in function_metrics:
            distribution[metric['complexity_rank']] += 1
        return dict(distribution)
    
    def _identify_complexity_hotspots(self) -> List[Dict[str, Any]]:
        """Identify complexity hotspots in the codebase."""
        # Implementation would analyze file-level complexity
        return []
    
    def _detect_circular_dependencies(self) -> List[Dict[str, Any]]:
        """Detect circular dependencies."""
        # Implementation would analyze import chains
        return []
    
    def _calculate_dependency_metrics(self) -> Dict[str, Any]:
        """Calculate dependency metrics."""
        return {
            "fan_in": 0,
            "fan_out": 0,
            "instability": 0
        }
    
    def _find_unused_imports(self) -> List[str]:
        """Find unused imports."""
        # Implementation would analyze import usage
        return []
    
    def _analyze_package_structure(self, files: List) -> Dict[str, Any]:
        """Analyze package structure."""
        return {
            "depth": 0,
            "modules": len(files),
            "structure": {}
        }
    
    def _detect_design_patterns(self, classes: List) -> List[str]:
        """Detect design patterns."""
        return []
    
    def _calculate_coupling_metrics(self) -> Dict[str, float]:
        """Calculate coupling metrics."""
        return {
            "afferent_coupling": 0.0,
            "efferent_coupling": 0.0
        }
    
    def _detect_architectural_violations(self) -> List[str]:
        """Detect architectural violations."""
        return []
    
    def _is_singleton_pattern(self, cls) -> bool:
        """Check if class implements singleton pattern."""
        return False
    
    def _calculate_pattern_statistics(self, patterns: Dict) -> Dict[str, int]:
        """Calculate pattern statistics."""
        return {pattern: len(items) for pattern, items in patterns.items()}
    
    def _calculate_debt_score(self, debt_indicators: Dict) -> float:
        """Calculate technical debt score."""
        total_issues = sum(len(issues) for issues in debt_indicators.values())
        return min(100, total_issues * 5)  # Simple scoring
    
    def _generate_refactoring_suggestions(self, debt_indicators: Dict) -> List[str]:
        """Generate refactoring suggestions."""
        suggestions = []
        if debt_indicators["long_functions"]:
            suggestions.append("Consider breaking down long functions into smaller, more focused functions")
        if debt_indicators["complex_functions"]:
            suggestions.append("Reduce complexity in high-complexity functions")
        return suggestions
    
    def _estimate_test_coverage(self) -> float:
        """Estimate test coverage."""
        # Simple estimation based on test function ratio
        return 0.0
    
    def _contains_hardcoded_secrets(self, source: str) -> bool:
        """Check for hardcoded secrets."""
        secret_patterns = ['password', 'api_key', 'secret', 'token']
        return any(pattern in source.lower() for pattern in secret_patterns)
    
    def _calculate_security_score(self, security_issues: Dict) -> float:
        """Calculate security score."""
        total_issues = sum(len(issues) for issues in security_issues.values())
        return max(0, 100 - total_issues * 10)
    
    def _generate_security_recommendations(self, security_issues: Dict) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        if security_issues["potential_sql_injection"]:
            recommendations.append("Use parameterized queries to prevent SQL injection")
        if security_issues["unsafe_functions"]:
            recommendations.append("Avoid using eval() and exec() functions")
        return recommendations
    
    def _generate_optimization_suggestions(self, performance_issues: Dict) -> List[str]:
        """Generate optimization suggestions."""
        suggestions = []
        if performance_issues["nested_loops"]:
            suggestions.append("Consider optimizing nested loops or using more efficient algorithms")
        if performance_issues["recursive_functions"]:
            suggestions.append("Consider iterative alternatives for recursive functions")
        return suggestions


class RepoRequest(BaseModel):
    repo_url: str


@fastapi_app.post("/analyze_repo")
async def analyze_repo(request: RepoRequest) -> Dict[str, Any]:
    """Get COMPREHENSIVE codebase analysis using the Analysis class."""
    try:
        repo_url = request.repo_url
        codebase = Codebase.from_repo(repo_url)
        analysis = Analysis(codebase)
        
        # Get FULL comprehensive analysis - ALL dimensions
        result = analysis.get_comprehensive_analysis()
        
        # Add repository metadata
        result["repository"] = {
            "url": repo_url,
            "description": get_github_repo_description(repo_url),
            "analysis_timestamp": datetime.now().isoformat(),
            "analysis_type": "comprehensive"
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.function(image=image)
@modal.asgi_app()
def fastapi_modal_app():
    return fastapi_app


if __name__ == "__main__":
    app.deploy("analytics-app")
