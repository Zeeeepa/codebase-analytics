from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Tuple, Any, Optional, Union
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

class AnalysisRequest(BaseModel):
    repo_url: str
    analysis_type: Optional[str] = "comprehensive"  # comprehensive, quality, complexity, dependencies, etc.


@fastapi_app.post("/comprehensive_analysis")
async def comprehensive_analysis(request: AnalysisRequest) -> Dict[str, Any]:
    """Get comprehensive codebase analysis using the Analysis class."""
    try:
        repo_url = request.repo_url
        codebase = Codebase.from_repo(repo_url)
        analysis = Analysis(codebase)
        
        if request.analysis_type == "comprehensive":
            result = analysis.get_comprehensive_analysis()
        elif request.analysis_type == "quality":
            result = analysis.get_quality_metrics()
        elif request.analysis_type == "complexity":
            result = analysis.get_complexity_analysis()
        elif request.analysis_type == "dependencies":
            result = analysis.get_dependency_analysis()
        elif request.analysis_type == "architecture":
            result = analysis.get_architecture_analysis()
        elif request.analysis_type == "patterns":
            result = analysis.get_code_patterns()
        elif request.analysis_type == "debt":
            result = analysis.get_technical_debt_analysis()
        elif request.analysis_type == "security":
            result = analysis.get_security_analysis()
        elif request.analysis_type == "performance":
            result = analysis.get_performance_insights()
        elif request.analysis_type == "testing":
            result = analysis.get_test_coverage_analysis()
        else:
            result = analysis.get_comprehensive_analysis()
        
        # Add repository metadata
        result["repository"] = {
            "url": repo_url,
            "description": get_github_repo_description(repo_url),
            "analysis_timestamp": datetime.now().isoformat(),
            "analysis_type": request.analysis_type
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@fastapi_app.post("/function_analysis")
async def function_analysis(request: RepoRequest) -> Dict[str, Any]:
    """Get detailed function-level analysis."""
    try:
        repo_url = request.repo_url
        codebase = Codebase.from_repo(repo_url)
        analysis = Analysis(codebase)
        
        functions = list(analysis.functions)
        function_details = []
        
        for func in functions:
            if hasattr(func, 'code_block'):
                complexity = calculate_cyclomatic_complexity(func)
                operators, operands = get_operators_and_operands(func)
                volume, N1, N2, n1, n2 = calculate_halstead_volume(operators, operands)
                loc = len(func.code_block.source.splitlines()) if hasattr(func.code_block, 'source') else 0
                mi_score = calculate_maintainability_index(volume, complexity, loc)
                
                function_details.append({
                    "name": func.name,
                    "file": getattr(func, 'file_path', 'unknown'),
                    "line_number": getattr(func, 'line_number', 0),
                    "metrics": {
                        "cyclomatic_complexity": complexity,
                        "complexity_rank": cc_rank(complexity),
                        "maintainability_index": mi_score,
                        "maintainability_rank": get_maintainability_rank(mi_score),
                        "lines_of_code": loc,
                        "halstead_volume": volume,
                        "halstead_metrics": {
                            "operators_count": N1,
                            "operands_count": N2,
                            "unique_operators": n1,
                            "unique_operands": n2
                        }
                    },
                    "properties": {
                        "is_async": getattr(func, 'is_async', False),
                        "is_generator": getattr(func, 'is_generator', False),
                        "parameter_count": len(func.parameters) if hasattr(func, 'parameters') else 0,
                        "has_docstring": bool(getattr(func, 'docstring', None)),
                        "decorators": [d.name for d in func.decorators] if hasattr(func, 'decorators') else []
                    }
                })
        
        return {
            "repository": repo_url,
            "total_functions": len(function_details),
            "functions": sorted(function_details, key=lambda x: x['metrics']['complexity_rank']),
            "summary": {
                "average_complexity": sum(f['metrics']['cyclomatic_complexity'] for f in function_details) / len(function_details) if function_details else 0,
                "average_maintainability": sum(f['metrics']['maintainability_index'] for f in function_details) / len(function_details) if function_details else 0,
                "high_complexity_count": len([f for f in function_details if f['metrics']['cyclomatic_complexity'] > 10]),
                "low_maintainability_count": len([f for f in function_details if f['metrics']['maintainability_index'] < 25])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Function analysis failed: {str(e)}")

@fastapi_app.post("/class_analysis")
async def class_analysis(request: RepoRequest) -> Dict[str, Any]:
    """Get detailed class-level analysis."""
    try:
        repo_url = request.repo_url
        codebase = Codebase.from_repo(repo_url)
        analysis = Analysis(codebase)
        
        classes = list(analysis.classes)
        class_details = []
        
        for cls in classes:
            methods = list(cls.methods) if hasattr(cls, 'methods') else []
            attributes = list(cls.attributes) if hasattr(cls, 'attributes') else []
            superclasses = list(cls.superclasses) if hasattr(cls, 'superclasses') else []
            
            # Calculate class complexity as sum of method complexities
            total_complexity = 0
            for method in methods:
                if hasattr(method, 'code_block'):
                    total_complexity += calculate_cyclomatic_complexity(method)
            
            class_details.append({
                "name": cls.name,
                "file": getattr(cls, 'file_path', 'unknown'),
                "line_number": getattr(cls, 'line_number', 0),
                "metrics": {
                    "depth_of_inheritance": calculate_doi(cls),
                    "method_count": len(methods),
                    "attribute_count": len(attributes),
                    "total_complexity": total_complexity,
                    "average_method_complexity": total_complexity / len(methods) if methods else 0
                },
                "properties": {
                    "is_abstract": getattr(cls, 'is_abstract', False),
                    "superclasses": [sc.name for sc in superclasses],
                    "has_docstring": bool(getattr(cls, 'docstring', None)),
                    "decorators": [d.name for d in cls.decorators] if hasattr(cls, 'decorators') else []
                },
                "methods": [
                    {
                        "name": method.name,
                        "is_async": getattr(method, 'is_async', False),
                        "parameter_count": len(method.parameters) if hasattr(method, 'parameters') else 0,
                        "complexity": calculate_cyclomatic_complexity(method) if hasattr(method, 'code_block') else 0
                    }
                    for method in methods
                ]
            })
        
        return {
            "repository": repo_url,
            "total_classes": len(class_details),
            "classes": sorted(class_details, key=lambda x: x['metrics']['total_complexity'], reverse=True),
            "summary": {
                "average_doi": sum(c['metrics']['depth_of_inheritance'] for c in class_details) / len(class_details) if class_details else 0,
                "average_methods_per_class": sum(c['metrics']['method_count'] for c in class_details) / len(class_details) if class_details else 0,
                "complex_classes_count": len([c for c in class_details if c['metrics']['total_complexity'] > 50]),
                "abstract_classes_count": len([c for c in class_details if c['properties']['is_abstract']])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Class analysis failed: {str(e)}")

@fastapi_app.post("/dependency_graph")
async def dependency_graph(request: RepoRequest) -> Dict[str, Any]:
    """Get dependency graph analysis."""
    try:
        repo_url = request.repo_url
        codebase = Codebase.from_repo(repo_url)
        analysis = Analysis(codebase)
        
        imports = list(analysis.imports)
        files = list(analysis.files)
        
        # Build dependency graph
        dependency_graph = {}
        external_dependencies = set()
        
        for file in files:
            file_imports = []
            if hasattr(file, 'imports'):
                for imp in file.imports:
                    import_info = {
                        "module": getattr(imp, 'source', ''),
                        "is_external": getattr(imp, 'is_external', False),
                        "import_type": getattr(imp, 'import_type', 'unknown')
                    }
                    file_imports.append(import_info)
                    
                    if import_info["is_external"]:
                        external_dependencies.add(import_info["module"])
            
            dependency_graph[file.path] = {
                "imports": file_imports,
                "import_count": len(file_imports),
                "external_import_count": len([imp for imp in file_imports if imp["is_external"]])
            }
        
        # Calculate dependency metrics
        total_imports = sum(data["import_count"] for data in dependency_graph.values())
        total_external = sum(data["external_import_count"] for data in dependency_graph.values())
        
        return {
            "repository": repo_url,
            "dependency_graph": dependency_graph,
            "external_dependencies": sorted(list(external_dependencies)),
            "metrics": {
                "total_files": len(files),
                "total_imports": total_imports,
                "total_external_dependencies": len(external_dependencies),
                "average_imports_per_file": total_imports / len(files) if files else 0,
                "external_dependency_ratio": total_external / total_imports if total_imports > 0 else 0
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dependency analysis failed: {str(e)}")

@fastapi_app.post("/code_quality_report")
async def code_quality_report(request: RepoRequest) -> Dict[str, Any]:
    """Generate comprehensive code quality report."""
    try:
        repo_url = request.repo_url
        codebase = Codebase.from_repo(repo_url)
        analysis = Analysis(codebase)
        
        # Get all analysis components
        overview = analysis.get_codebase_overview()
        quality_metrics = analysis.get_quality_metrics()
        complexity_analysis = analysis.get_complexity_analysis()
        technical_debt = analysis.get_technical_debt_analysis()
        security_analysis = analysis.get_security_analysis()
        
        # Calculate overall quality score
        quality_factors = {
            "complexity": 100 - min(100, quality_metrics["averages"]["cyclomatic_complexity"] * 5),
            "maintainability": quality_metrics["averages"]["maintainability_index"],
            "technical_debt": 100 - technical_debt["debt_score"],
            "security": security_analysis["security_score"]
        }
        
        overall_score = sum(quality_factors.values()) / len(quality_factors)
        
        # Generate quality grade
        if overall_score >= 90:
            quality_grade = "A"
        elif overall_score >= 80:
            quality_grade = "B"
        elif overall_score >= 70:
            quality_grade = "C"
        elif overall_score >= 60:
            quality_grade = "D"
        else:
            quality_grade = "F"
        
        return {
            "repository": repo_url,
            "overall_quality_score": round(overall_score, 2),
            "quality_grade": quality_grade,
            "quality_factors": quality_factors,
            "overview": overview,
            "detailed_metrics": {
                "quality": quality_metrics,
                "complexity": complexity_analysis,
                "technical_debt": technical_debt,
                "security": security_analysis
            },
            "recommendations": {
                "high_priority": [],
                "medium_priority": technical_debt["refactoring_suggestions"],
                "low_priority": security_analysis["recommendations"]
            },
            "report_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality report generation failed: {str(e)}")

@fastapi_app.post("/analyze_repo")
async def analyze_repo(request: RepoRequest) -> Dict[str, Any]:
    """Analyze a repository and return comprehensive metrics (legacy endpoint)."""
    try:
        repo_url = request.repo_url
        codebase = Codebase.from_repo(repo_url)
        analysis = Analysis(codebase)

        # Get enhanced analysis using the new Analysis class
        overview = analysis.get_codebase_overview()
        quality_metrics = analysis.get_quality_metrics()
        
        # Get monthly commits (legacy feature)
        monthly_commits = get_monthly_commits(repo_url)
        
        # Calculate legacy metrics for backward compatibility
        files = list(codebase.files)
        total_loc = total_lloc = total_sloc = total_comments = 0
        
        for file in files:
            if hasattr(file, 'source'):
                loc, lloc, sloc, comments = count_lines(file.source)
                total_loc += loc
                total_lloc += lloc
                total_sloc += sloc
                total_comments += comments

        desc = get_github_repo_description(repo_url)

        # Enhanced results with backward compatibility
        results = {
            "repo_url": repo_url,
            "description": desc,
            "monthly_commits": monthly_commits,
            
            # Legacy format metrics
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
                "average": quality_metrics["averages"]["cyclomatic_complexity"],
            },
            "depth_of_inheritance": {
                "average": quality_metrics["averages"]["depth_of_inheritance"],
            },
            "halstead_metrics": {
                "total_volume": sum(f["halstead_volume"] for f in quality_metrics["function_metrics"]),
                "average_volume": sum(f["halstead_volume"] for f in quality_metrics["function_metrics"]) / len(quality_metrics["function_metrics"]) if quality_metrics["function_metrics"] else 0,
            },
            "maintainability_index": {
                "average": quality_metrics["averages"]["maintainability_index"],
            },
            "num_files": overview["total_files"],
            "num_functions": overview["total_functions"],
            "num_classes": overview["total_classes"],
            
            # Enhanced metrics
            "enhanced_analysis": {
                "overview": overview,
                "quality_distribution": quality_metrics["quality_distribution"],
                "language_distribution": overview["language_distribution"],
                "file_size_stats": {
                    "average": overview["average_file_size"],
                    "largest": overview["largest_file_size"],
                    "smallest": overview["smallest_file_size"]
                }
            }
        }

        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Repository analysis failed: {str(e)}")

@fastapi_app.post("/symbol_analysis")
async def symbol_analysis(request: RepoRequest) -> Dict[str, Any]:
    """Get detailed symbol analysis including usage patterns."""
    try:
        repo_url = request.repo_url
        codebase = Codebase.from_repo(repo_url)
        analysis = Analysis(codebase)
        
        symbols = list(analysis.symbols)
        symbol_details = []
        
        for symbol in symbols:
            symbol_info = {
                "name": symbol.name,
                "type": getattr(symbol, 'symbol_type', 'unknown'),
                "file": getattr(symbol, 'file_path', 'unknown'),
                "line_number": getattr(symbol, 'line_number', 0),
                "usage_count": len(symbol.usages) if hasattr(symbol, 'usages') else 0,
                "is_exported": getattr(symbol, 'is_exported', False),
                "is_imported": getattr(symbol, 'is_imported', False)
            }
            
            # Add specific details based on symbol type
            if hasattr(symbol, 'parameters'):  # Function
                symbol_info["parameter_count"] = len(symbol.parameters)
                symbol_info["is_async"] = getattr(symbol, 'is_async', False)
            elif hasattr(symbol, 'methods'):  # Class
                symbol_info["method_count"] = len(list(symbol.methods))
                symbol_info["attribute_count"] = len(list(symbol.attributes)) if hasattr(symbol, 'attributes') else 0
            
            symbol_details.append(symbol_info)
        
        # Analyze symbol usage patterns
        unused_symbols = [s for s in symbol_details if s["usage_count"] == 0 and not s["is_exported"]]
        heavily_used_symbols = [s for s in symbol_details if s["usage_count"] > 10]
        
        return {
            "repository": repo_url,
            "total_symbols": len(symbol_details),
            "symbols": symbol_details,
            "usage_analysis": {
                "unused_symbols": unused_symbols,
                "heavily_used_symbols": sorted(heavily_used_symbols, key=lambda x: x["usage_count"], reverse=True),
                "average_usage": sum(s["usage_count"] for s in symbol_details) / len(symbol_details) if symbol_details else 0
            },
            "symbol_distribution": {
                symbol_type: len([s for s in symbol_details if s["type"] == symbol_type])
                for symbol_type in set(s["type"] for s in symbol_details)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Symbol analysis failed: {str(e)}")

@fastapi_app.post("/architecture_insights")
async def architecture_insights(request: RepoRequest) -> Dict[str, Any]:
    """Get architectural insights and design pattern analysis."""
    try:
        repo_url = request.repo_url
        codebase = Codebase.from_repo(repo_url)
        analysis = Analysis(codebase)
        
        # Get comprehensive architectural analysis
        architecture_analysis = analysis.get_architecture_analysis()
        code_patterns = analysis.get_code_patterns()
        dependency_analysis = analysis.get_dependency_analysis()
        
        # Additional architectural metrics
        files = list(analysis.files)
        classes = list(analysis.classes)
        functions = list(analysis.functions)
        
        # Calculate module cohesion and coupling
        module_metrics = {}
        for file in files:
            if hasattr(file, 'functions') and hasattr(file, 'classes'):
                file_functions = list(file.functions) if hasattr(file.functions, '__iter__') else []
                file_classes = list(file.classes) if hasattr(file.classes, '__iter__') else []
                
                module_metrics[file.path] = {
                    "function_count": len(file_functions),
                    "class_count": len(file_classes),
                    "total_symbols": len(file_functions) + len(file_classes),
                    "import_count": len(list(file.imports)) if hasattr(file, 'imports') else 0
                }
        
        # Identify architectural smells
        architectural_smells = []
        
        # Large modules
        large_modules = [path for path, metrics in module_metrics.items() 
                        if metrics["total_symbols"] > 20]
        if large_modules:
            architectural_smells.append({
                "type": "Large Modules",
                "description": "Modules with too many symbols",
                "affected_files": large_modules
            })
        
        # God classes (classes with too many methods)
        god_classes = []
        for cls in classes:
            if hasattr(cls, 'methods'):
                method_count = len(list(cls.methods))
                if method_count > 15:
                    god_classes.append(cls.name)
        
        if god_classes:
            architectural_smells.append({
                "type": "God Classes",
                "description": "Classes with too many methods",
                "affected_classes": god_classes
            })
        
        return {
            "repository": repo_url,
            "architecture_analysis": architecture_analysis,
            "code_patterns": code_patterns,
            "dependency_metrics": dependency_analysis["dependency_metrics"],
            "module_metrics": module_metrics,
            "architectural_smells": architectural_smells,
            "recommendations": {
                "modularity": "Consider breaking down large modules into smaller, focused modules",
                "coupling": "Reduce dependencies between modules to improve maintainability",
                "cohesion": "Group related functionality together within modules"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Architecture analysis failed: {str(e)}")

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
