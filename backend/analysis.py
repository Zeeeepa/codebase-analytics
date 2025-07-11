"""
Core analysis engine for codebases using graph-sitter.
This module contains all functions for semantic, dependency, architectural,
performance, security, and issue analysis. It is framework-agnostic.
"""

import re
import hashlib
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

# Graph-sitter imports
from graph_sitter.core.codebase import Codebase
from graph_sitter.core.class_definition import Class
from graph_sitter.core.function import Function
from graph_sitter.core.file import SourceFile
from graph_sitter.core.expressions.binary_expression import BinaryExpression
from graph_sitter.core.expressions.unary_expression import UnaryExpression
from graph_sitter.core.expressions.comparison_expression import ComparisonExpression

# Import data models
from .models import CodeIssue, EntryPoint, CriticalFile, DependencyNode

# Conditional imports for advanced features
try:
    from graph_sitter.core.assignment import Assignment
    from graph_sitter.core.export import Export
    from graph_sitter.core.directory import Directory
    from graph_sitter.core.interface import Interface
    import graph_sitter.python as python_analyzer
    import graph_sitter.typescript as typescript_analyzer
except ImportError:
    Assignment = Export = Directory = Interface = python_analyzer = (
        typescript_analyzer
    ) = None


class CodeAnalysisError(Exception):
    """Custom exception for code analysis errors"""

    pass


class CodebaseCache:
    """Cache for AST and metrics to improve performance"""

    def __init__(self):
        self.ast_cache = {}
        self.metric_cache = {}
        self.call_graph_cache = {}

    def get_or_compute_ast(self, file: SourceFile) -> Dict:
        if file.filepath not in self.ast_cache:
            self.ast_cache[file.filepath] = self.parse_file_ast(file)
        return self.ast_cache[file.filepath]

    def parse_file_ast(self, file: SourceFile) -> Dict:
        try:
            return {
                "file_path": file.filepath,
                "functions": [
                    {"name": f.name, "line": getattr(f, "line_number", None)}
                    for f in file.functions
                ],
                "classes": [
                    {"name": c.name, "line": getattr(c, "line_number", None)}
                    for c in file.classes
                ],
                "imports": [{"name": i.name} for i in file.imports],
            }
        except Exception as e:
            raise CodeAnalysisError(
                f"Failed to parse AST for {file.filepath}: {str(e)}"
            )


codebase_cache = CodebaseCache()


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


def detect_entrypoints(codebase: Codebase) -> List[EntryPoint]:
    """Detect various types of entry points in the codebase."""
    entrypoints = []

    for file in codebase.files:
        # Skip non-Python files for now (can be extended)
        if not file.name.endswith((".py", ".js", ".ts", ".java", ".cpp", ".c")):
            continue

        try:
            # Detect main functions
            for func in file.functions:
                if func.name in ["main", "__main__", "run", "start", "execute"]:
                    entrypoints.append(
                        EntryPoint(
                            type="main",
                            file_path=file.filepath,
                            function_name=func.name,
                            line_number=getattr(func, "line_number", None),
                            description=f"Main function '{func.name}' in {file.name}",
                            confidence=0.9,
                            dependencies=[dep.name for dep in func.dependencies[:5]],
                        )
                    )

                # Detect CLI entry points
                if any(
                    keyword in func.name.lower()
                    for keyword in ["cli", "command", "parse_args", "argparse"]
                ):
                    entrypoints.append(
                        EntryPoint(
                            type="cli",
                            file_path=file.filepath,
                            function_name=func.name,
                            line_number=getattr(func, "line_number", None),
                            description=f"CLI function '{func.name}' in {file.name}",
                            confidence=0.7,
                            dependencies=[dep.name for dep in func.dependencies[:5]],
                        )
                    )

                # Detect web endpoints
                if any(
                    decorator in str(func.decorators)
                    for decorator in [
                        "@app.route",
                        "@router.",
                        "@fastapi_app.",
                        "@get",
                        "@post",
                        "@put",
                        "@delete",
                    ]
                ):
                    entrypoints.append(
                        EntryPoint(
                            type="web_endpoint",
                            file_path=file.filepath,
                            function_name=func.name,
                            line_number=getattr(func, "line_number", None),
                            description=f"Web endpoint '{func.name}' in {file.name}",
                            confidence=0.95,
                            dependencies=[dep.name for dep in func.dependencies[:5]],
                        )
                    )

                # Detect test functions
                if func.name.startswith("test_") or any(
                    keyword in func.name.lower() for keyword in ["test", "spec"]
                ):
                    entrypoints.append(
                        EntryPoint(
                            type="test",
                            file_path=file.filepath,
                            function_name=func.name,
                            line_number=getattr(func, "line_number", None),
                            description=f"Test function '{func.name}' in {file.name}",
                            confidence=0.8,
                            dependencies=[dep.name for dep in func.dependencies[:3]],
                        )
                    )

            # Detect script entry points (if __name__ == "__main__")
            if "__name__" in file.source and "__main__" in file.source:
                entrypoints.append(
                    EntryPoint(
                        type="script",
                        file_path=file.filepath,
                        description=f"Script entry point in {file.name}",
                        confidence=0.85,
                        dependencies=[],
                    )
                )

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
                issues.append(
                    CodeIssue(
                        id=f"empty_file_{hashlib.md5(file.filepath.encode()).hexdigest()[:8]}",
                        type=IssueType.CODE_SMELL,
                        severity=IssueSeverity.LOW,
                        file_path=file.filepath,
                        message="Empty file",
                        description="File contains no code",
                        context={"file_size": len(file.source)},
                    )
                )
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
                        severity = (
                            IssueSeverity.CRITICAL
                            if complexity > 30
                            else IssueSeverity.HIGH
                        )
                        issues.append(
                            CodeIssue(
                                id=f"complexity_{hashlib.md5(f'{file.filepath}_{func.name}'.encode()).hexdigest()[:8]}",
                                type=IssueType.COMPLEXITY_ISSUE,
                                severity=severity,
                                file_path=file.filepath,
                                function_name=func.name,
                                line_number=getattr(func, "line_number", None),
                                message=f"High cyclomatic complexity: {complexity}",
                                description=f"Function '{func.name}' has cyclomatic complexity of {complexity}, which exceeds recommended threshold of 15",
                                context={
                                    "complexity": complexity,
                                    "complexity_rank": cc_rank(complexity),
                                    "parameters_count": len(func.parameters),
                                    "return_statements": len(func.return_statements),
                                },
                                related_symbols=[func.name],
                                fix_suggestions=[
                                    "Break down the function into smaller, more focused functions",
                                    "Extract complex conditional logic into separate methods",
                                    "Consider using strategy pattern for complex branching logic",
                                ],
                            )
                        )
                        issue_counter += 1

                    # Long functions (high LOC)
                    func_lines = (
                        len(func.source.splitlines()) if hasattr(func, "source") else 0
                    )
                    if func_lines > 50:
                        severity = (
                            IssueSeverity.HIGH
                            if func_lines > 100
                            else IssueSeverity.MEDIUM
                        )
                        issues.append(
                            CodeIssue(
                                id=f"long_func_{hashlib.md5(f'{file.filepath}_{func.name}'.encode()).hexdigest()[:8]}",
                                type=IssueType.MAINTAINABILITY_ISSUE,
                                severity=severity,
                                file_path=file.filepath,
                                function_name=func.name,
                                line_number=getattr(func, "line_number", None),
                                message=f"Long function: {func_lines} lines",
                                description=f"Function '{func.name}' is {func_lines} lines long, exceeding recommended maximum of 50 lines",
                                context={
                                    "lines_of_code": func_lines,
                                    "parameters_count": len(func.parameters),
                                    "complexity": complexity,
                                },
                                related_symbols=[func.name],
                                fix_suggestions=[
                                    "Split function into smaller, single-responsibility functions",
                                    "Extract reusable logic into utility functions",
                                    "Consider using composition over large monolithic functions",
                                ],
                            )
                        )
                        issue_counter += 1

                    # Too many parameters
                    if len(func.parameters) > 7:
                        issues.append(
                            CodeIssue(
                                id=f"many_params_{hashlib.md5(f'{file.filepath}_{func.name}'.encode()).hexdigest()[:8]}",
                                type=IssueType.CODE_SMELL,
                                severity=IssueSeverity.MEDIUM,
                                file_path=file.filepath,
                                function_name=func.name,
                                line_number=getattr(func, "line_number", None),
                                message=f"Too many parameters: {len(func.parameters)}",
                                description=f"Function '{func.name}' has {len(func.parameters)} parameters, exceeding recommended maximum of 7",
                                context={
                                    "parameters_count": len(func.parameters),
                                    "parameter_names": [
                                        p.name for p in func.parameters
                                    ],
                                },
                                related_symbols=[func.name],
                                fix_suggestions=[
                                    "Group related parameters into a configuration object",
                                    "Use builder pattern for complex parameter sets",
                                    "Consider if some parameters can have default values",
                                ],
                            )
                        )
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
                        severity = (
                            IssueSeverity.HIGH
                            if len(cls.methods) > 30
                            else IssueSeverity.MEDIUM
                        )
                        issues.append(
                            CodeIssue(
                                id=f"god_class_{hashlib.md5(f'{file.filepath}_{cls.name}'.encode()).hexdigest()[:8]}",
                                type=IssueType.CODE_SMELL,
                                severity=severity,
                                file_path=file.filepath,
                                class_name=cls.name,
                                line_number=getattr(cls, "line_number", None),
                                message=f"God class: {len(cls.methods)} methods",
                                description=f"Class '{cls.name}' has {len(cls.methods)} methods, indicating it may have too many responsibilities",
                                context={
                                    "methods_count": len(cls.methods),
                                    "attributes_count": len(cls.attributes),
                                    "method_names": [m.name for m in cls.methods[:10]],
                                },
                                related_symbols=[cls.name],
                                affected_functions=[m.name for m in cls.methods],
                                fix_suggestions=[
                                    "Apply Single Responsibility Principle - split class into smaller classes",
                                    "Extract related methods into separate classes or modules",
                                    "Consider using composition instead of inheritance",
                                ],
                            )
                        )
                        issue_counter += 1

                    # Deep inheritance hierarchy
                    doi = calculate_doi(cls)
                    if doi > 5:
                        issues.append(
                            CodeIssue(
                                id=f"deep_inheritance_{hashlib.md5(f'{file.filepath}_{cls.name}'.encode()).hexdigest()[:8]}",
                                type=IssueType.CODE_SMELL,
                                severity=IssueSeverity.MEDIUM,
                                file_path=file.filepath,
                                class_name=cls.name,
                                line_number=getattr(cls, "line_number", None),
                                message=f"Deep inheritance: {doi} levels",
                                description=f"Class '{cls.name}' has inheritance depth of {doi}, which may indicate over-engineering",
                                context={
                                    "depth_of_inheritance": doi,
                                    "parent_classes": cls.parent_class_names,
                                },
                                related_symbols=[cls.name] + cls.parent_class_names,
                                fix_suggestions=[
                                    "Consider using composition over inheritance",
                                    "Flatten the inheritance hierarchy",
                                    "Use interfaces/protocols instead of deep inheritance",
                                ],
                            )
                        )
                        issue_counter += 1

                except Exception as e:
                    print(f"Error analyzing class {cls.name}: {e}")
                    continue

            # Security vulnerability patterns
            security_patterns = [
                (r"eval\s*\(", "Use of eval() function", IssueSeverity.CRITICAL),
                (r"exec\s*\(", "Use of exec() function", IssueSeverity.CRITICAL),
                (
                    r"subprocess\.call\s*\(.*shell\s*=\s*True",
                    "Shell injection vulnerability",
                    IssueSeverity.HIGH,
                ),
                (
                    r"pickle\.loads?\s*\(",
                    "Unsafe pickle deserialization",
                    IssueSeverity.HIGH,
                ),
                (r"input\s*\(.*\)", "Use of input() function", IssueSeverity.MEDIUM),
                (
                    r'password\s*=\s*["\'][^"\']+["\']',
                    "Hardcoded password",
                    IssueSeverity.HIGH,
                ),
                (
                    r'api_key\s*=\s*["\'][^"\']+["\']',
                    "Hardcoded API key",
                    IssueSeverity.HIGH,
                ),
            ]

            for pattern, message, severity in security_patterns:
                if issue_counter >= max_issues:
                    break

                matches = re.finditer(pattern, file.source, re.IGNORECASE)
                for match in matches:
                    line_num = file.source[: match.start()].count("\n") + 1
                    issues.append(
                        CodeIssue(
                            id=f"security_{hashlib.md5(f'{file.filepath}_{line_num}_{message}'.encode()).hexdigest()[:8]}",
                            type=IssueType.SECURITY_VULNERABILITY,
                            severity=severity,
                            file_path=file.filepath,
                            line_number=line_num,
                            message=message,
                            description=f"Potential security vulnerability: {message}",
                            context={"matched_text": match.group(), "pattern": pattern},
                            fix_suggestions=[
                                "Review and validate the security implications",
                                "Consider safer alternatives",
                                "Add proper input validation and sanitization",
                            ],
                        )
                    )
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
            dependents_count = len(
                [
                    f
                    for f in codebase.files
                    if any(imp.imported_symbol == file for imp in f.imports)
                ]
            )

            # Count symbols defined in file
            symbols_count = (
                len(file.functions) + len(file.classes) + len(file.global_vars)
            )

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
            importance_score = min(
                100,
                (
                    dependents_count * 10  # Files that depend on this file
                    + dependencies_count * 2  # External dependencies
                    + symbols_count * 3  # Number of symbols defined
                    + (total_complexity / max(1, len(file.functions)))
                    * 2  # Average complexity
                    + (loc / 100) * 1  # Size factor
                ),
            )

            reasons = []
            if dependents_count > 5:
                reasons.append(
                    f"High dependency usage ({dependents_count} files depend on it)"
                )
            if dependencies_count > 10:
                reasons.append(f"Many external dependencies ({dependencies_count})")
            if symbols_count > 10:
                reasons.append(f"Defines many symbols ({symbols_count})")
            if total_complexity > 50:
                reasons.append(f"High total complexity ({total_complexity})")
            if loc > 500:
                reasons.append(f"Large file ({loc} lines)")
            if file.name in ["main.py", "app.py", "server.py", "index.js", "main.js"]:
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
                    "average_complexity": total_complexity
                    / max(1, len(file.functions)),
                },
                dependencies_count=dependencies_count,
                dependents_count=dependents_count,
                complexity_score=total_complexity,
                lines_of_code=loc,
            )

        except Exception as e:
            print(f"Error analyzing file {file.name}: {e}")
            continue

    # Sort by importance score and return top files
    critical_files = sorted(
        file_metrics.values(), key=lambda x: x.importance_score, reverse=True
    )
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
            dependents=[],
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
                dependents=[],
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
                dependents=[],
            )

    # Build dependency relationships
    for file in codebase.files:
        file_node_id = f"file:{file.filepath}"

        # File-level dependencies
        for imp in file.imports:
            if hasattr(imp, "imported_symbol") and hasattr(
                imp.imported_symbol, "filepath"
            ):
                dep_file_id = f"file:{imp.imported_symbol.filepath}"
                if dep_file_id in nodes:
                    nodes[file_node_id].dependencies.append(dep_file_id)
                    nodes[dep_file_id].dependents.append(file_node_id)

        # Function-level dependencies
        for func in file.functions:
            func_node_id = f"function:{file.filepath}:{func.name}"
            for dep in func.dependencies:
                if hasattr(dep, "filepath") and hasattr(dep, "name"):
                    dep_id = f"function:{dep.filepath}:{dep.name}"
                    if dep_id in nodes:
                        nodes[func_node_id].dependencies.append(dep_id)
                        nodes[dep_id].dependents.append(func_node_id)

    # Calculate centrality scores (simplified betweenness centrality)
    for node_id, node in nodes.items():
        # Simple centrality based on number of connections
        centrality = (len(node.dependencies) + len(node.dependents)) / max(
            1, len(nodes)
        )
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
            mi_score = calculate_maintainability_index(
                volume,
                complexity,
                len(func.source.splitlines()) if hasattr(func, "source") else 0,
            )

            functions_analysis.append(
                {
                    "name": func.name,
                    "line_number": getattr(func, "line_number", None),
                    "parameters_count": len(func.parameters),
                    "return_statements_count": len(func.return_statements),
                    "cyclomatic_complexity": complexity,
                    "complexity_rank": cc_rank(complexity),
                    "halstead_volume": volume,
                    "maintainability_index": mi_score,
                    "maintainability_rank": get_maintainability_rank(mi_score),
                    "dependencies": [dep.name for dep in func.dependencies[:10]],
                    "function_calls": [call.name for call in func.function_calls[:10]],
                }
            )
        except Exception as e:
            functions_analysis.append({"name": func.name, "error": str(e)})

    classes_analysis = []
    for cls in target_file.classes:
        try:
            classes_analysis.append(
                {
                    "name": cls.name,
                    "line_number": getattr(cls, "line_number", None),
                    "methods_count": len(cls.methods),
                    "attributes_count": len(cls.attributes),
                    "parent_classes": cls.parent_class_names,
                    "depth_of_inheritance": calculate_doi(cls),
                    "dependencies": [dep.name for dep in cls.dependencies[:10]],
                }
            )
        except Exception as e:
            classes_analysis.append({"name": cls.name, "error": str(e)})

    return {
        "file_path": target_file.filepath,
        "file_name": target_file.name,
        "line_metrics": {
            "loc": loc,
            "lloc": lloc,
            "sloc": sloc,
            "comments": comments,
            "comment_density": (comments / loc * 100) if loc > 0 else 0,
        },
        "symbols_count": {
            "functions": len(target_file.functions),
            "classes": len(target_file.classes),
            "global_vars": len(target_file.global_vars),
            "imports": len(target_file.imports),
        },
        "functions": functions_analysis,
        "classes": classes_analysis,
        "imports": [
            {"name": imp.name, "source": getattr(imp, "source", "unknown")}
            for imp in target_file.imports[:20]
        ],
        "issues": analyze_code_issues(
            Codebase.from_files([target_file]), max_issues=50
        ),
    }


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
        context["usages"].append(
            {
                "source": usage.usage_symbol.source,
                "filepath": usage.usage_symbol.filepath,
            }
        )

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
    """Enhanced cyclomatic complexity calculation with better statement handling."""

    def analyze_statement(statement):
        complexity = 0

        # Handle WhileStatement (working import)
        if isinstance(statement, WhileStatement):
            complexity += 1

        # Handle other control flow patterns through source analysis
        if hasattr(statement, "source"):
            source = statement.source.lower()
            # Count if/elif statements
            complexity += source.count("if ") + source.count("elif ")
            # Count for/while loops
            complexity += source.count("for ") + source.count("while ")
            # Count try/except blocks
            complexity += source.count("except ")
            # Count logical operators
            complexity += source.count(" and ") + source.count(" or ")

        if hasattr(statement, "nested_code_blocks"):
            for block in statement.nested_code_blocks:
                complexity += analyze_block(block)

        return complexity

    def analyze_block(block):
        if not block or not hasattr(block, "statements"):
            return 0
        return sum(analyze_statement(stmt) for stmt in block.statements)

    # Enhanced complexity calculation
    base_complexity = 1

    if hasattr(function, "code_block") and function.code_block:
        base_complexity += analyze_block(function.code_block)
    elif hasattr(function, "source"):
        # Fallback to source-based analysis
        source = function.source.lower()
        base_complexity += source.count("if ") + source.count("elif ")
        base_complexity += source.count("for ") + source.count("while ")
        base_complexity += source.count("except ")
        base_complexity += source.count(" and ") + source.count(" or ")

    return base_complexity


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


# Utility and Metric Calculation Functions
def calculate_cyclomatic_complexity(function):
    base_complexity = 1
    if hasattr(function, "source"):
        source = function.source.lower()
        base_complexity += source.count("if ") + source.count("elif ")
        base_complexity += source.count("for ") + source.count("while ")
        base_complexity += source.count("except ")
        base_complexity += source.count(" and ") + source.count(" or ")
    return base_complexity


def get_operators_and_operands(function):
    operators, operands = [], []
    if not hasattr(function, "code_block"):
        return operators, operands
    for statement in function.code_block.statements:
        for call in statement.function_calls:
            operators.append(call.name)
            for arg in call.args:
                operands.append(arg.source)
        expressions = getattr(statement, "expressions", []) + [
            getattr(statement, "expression", None)
        ]
        for expr in filter(None, expressions):
            if isinstance(expr, (BinaryExpression, ComparisonExpression)):
                operators.extend([op.source for op in expr.operators])
                operands.extend([elem.source for elem in expr.elements])
            elif isinstance(expr, UnaryExpression):
                operators.append(expr.ts_node.type)
                operands.append(expr.argument.source)
    return operators, operands


def calculate_halstead_volume(operators, operands):
    n1, n2 = len(set(operators)), len(set(operands))
    N1, N2 = len(operators), len(operands)
    N, n = N1 + N2, n1 + n2
    volume = N * math.log2(n) if n > 0 else 0
    return volume, N1, N2, n1, n2


def count_lines(source: str):
    if not source.strip():
        return 0, 0, 0, 0
    lines = source.splitlines()
    loc = len(lines)
    sloc = len([line for line in lines if line.strip()])
    comments = len(
        [line for line in lines if line.strip().startswith("#")]
    )  # Simplified
    return loc, sloc, sloc, comments  # lloc approx as sloc


def calculate_maintainability_index(
    halstead_volume: float, cyclomatic_complexity: float, loc: int
) -> int:
    if loc <= 0:
        return 100
    try:
        raw_mi = (
            171
            - 5.2 * math.log(max(1, halstead_volume))
            - 0.23 * cyclomatic_complexity
            - 16.2 * math.log(max(1, loc))
        )
        return int(max(0, min(100, raw_mi * 100 / 171)))
    except (ValueError, TypeError):
        return 0


def cc_rank(complexity):
    if complexity <= 5:
        return "A"
    if complexity <= 10:
        return "B"
    if complexity <= 20:
        return "C"
    if complexity <= 30:
        return "D"
    if complexity <= 40:
        return "E"
    return "F"


def get_maintainability_rank(mi_score: float) -> str:
    if mi_score >= 85:
        return "A"
    if mi_score >= 65:
        return "B"
    if mi_score >= 45:
        return "C"
    if mi_score >= 25:
        return "D"
    return "F"


def calculate_doi(cls):
    return len(cls.superclasses)


def advanced_semantic_analysis(codebase: Codebase) -> Dict[str, Any]:
    """Advanced semantic analysis using Expression, Name, String, Value classes."""
    semantic_data = {
        "variable_usage_patterns": {},
        "string_literals": [],
        "function_call_patterns": {},
        "type_usage": {},
        "semantic_errors": [],
    }

    for file in codebase.files:
        try:
            # Analyze expressions for semantic patterns
            for func in file.functions:
                if not hasattr(func, "code_block") or not func.code_block:
                    continue

                # Extract variable names and usage patterns
                try:
                    # Use Name class for variable analysis
                    variable_names = []
                    string_literals = []

                    # Analyze function source for patterns
                    if hasattr(func, "source"):
                        # Extract string literals
                        string_matches = re.findall(r'["\']([^"\']*)["\']', func.source)
                        string_literals.extend(string_matches)

                        # Extract variable assignments
                        assignment_matches = re.findall(r"(\w+)\s*=\s*", func.source)
                        variable_names.extend(assignment_matches)

                    semantic_data["variable_usage_patterns"][
                        f"{file.filepath}:{func.name}"
                    ] = {
                        "variables": list(set(variable_names)),
                        "variable_count": len(set(variable_names)),
                    }

                    semantic_data["string_literals"].extend(
                        [
                            {
                                "value": literal,
                                "file": file.filepath,
                                "function": func.name,
                                "length": len(literal),
                            }
                            for literal in string_literals
                        ]
                    )

                except Exception as e:
                    semantic_data["semantic_errors"].append(
                        {
                            "file": file.filepath,
                            "function": func.name,
                            "error": str(e),
                            "type": "variable_analysis_error",
                        }
                    )

        except Exception as e:
            semantic_data["semantic_errors"].append(
                {"file": file.filepath, "error": str(e), "type": "file_analysis_error"}
            )

    return semantic_data


def advanced_dependency_analysis(codebase: Codebase) -> Dict[str, Any]:
    """Enhanced dependency analysis using Export and Assignment classes."""
    dependency_data = {
        "export_analysis": {},
        "assignment_patterns": {},
        "circular_dependencies": [],
        "unused_exports": [],
        "dependency_metrics": {},
    }

    # Analyze exports if Export class is available
    if Export:
        try:
            exports = list(codebase.exports) if hasattr(codebase, "exports") else []
            dependency_data["export_analysis"] = {
                "total_exports": len(exports),
                "exports_by_file": {},
                "export_types": {},
            }

            for export in exports[:50]:  # Limit for performance
                try:
                    file_path = getattr(export, "file", {}).get("filepath", "unknown")
                    export_name = getattr(export, "name", "unknown")

                    if (
                        file_path
                        not in dependency_data["export_analysis"]["exports_by_file"]
                    ):
                        dependency_data["export_analysis"]["exports_by_file"][
                            file_path
                        ] = []

                    dependency_data["export_analysis"]["exports_by_file"][
                        file_path
                    ].append({"name": export_name, "type": type(export).__name__})

                except Exception as e:
                    dependency_data["export_analysis"]["error"] = str(e)

        except Exception as e:
            dependency_data["export_analysis"]["error"] = f"Export analysis failed: {e}"

    # Analyze assignments if Assignment class is available
    if Assignment:
        try:
            # This would require proper Assignment class integration
            dependency_data["assignment_patterns"]["status"] = (
                "Assignment analysis available"
            )
        except Exception as e:
            dependency_data["assignment_patterns"]["error"] = str(e)

    # Detect circular dependencies using graph analysis
    try:
        file_dependencies = {}
        for file in codebase.files:
            file_deps = []
            for imp in file.imports:
                if hasattr(imp, "imported_symbol") and hasattr(
                    imp.imported_symbol, "filepath"
                ):
                    file_deps.append(imp.imported_symbol.filepath)
            file_dependencies[file.filepath] = file_deps

        # Simple circular dependency detection
        def has_circular_dependency(file_path, target, visited=None):
            if visited is None:
                visited = set()
            if file_path in visited:
                return True
            if file_path == target and visited:
                return True
            visited.add(file_path)

            for dep in file_dependencies.get(file_path, []):
                if has_circular_dependency(dep, target, visited.copy()):
                    return True
            return False

        circular_deps = []
        for file_path in file_dependencies:
            if has_circular_dependency(file_path, file_path):
                circular_deps.append(file_path)

        dependency_data["circular_dependencies"] = circular_deps[:10]  # Limit results

    except Exception as e:
        dependency_data["circular_dependencies"] = [
            f"Error detecting circular dependencies: {e}"
        ]

    return dependency_data


def advanced_architectural_analysis(codebase: Codebase) -> Dict[str, Any]:
    """Architectural analysis using Interface and Directory classes."""
    arch_data = {
        "interface_analysis": {},
        "directory_structure": {},
        "architectural_patterns": {},
        "code_organization": {},
    }

    # Interface analysis if available
    if Interface:
        try:
            interfaces = (
                list(codebase.interfaces) if hasattr(codebase, "interfaces") else []
            )
            arch_data["interface_analysis"] = {
                "total_interfaces": len(interfaces),
                "interface_details": [],
            }

            for interface in interfaces[:20]:  # Limit for performance
                try:
                    arch_data["interface_analysis"]["interface_details"].append(
                        {
                            "name": getattr(interface, "name", "unknown"),
                            "file": getattr(interface, "filepath", "unknown"),
                            "methods": len(getattr(interface, "methods", [])),
                            "type": type(interface).__name__,
                        }
                    )
                except Exception as e:
                    arch_data["interface_analysis"]["error"] = str(e)

        except Exception as e:
            arch_data["interface_analysis"]["error"] = f"Interface analysis failed: {e}"

    # Directory structure analysis
    try:
        file_paths = [file.filepath for file in codebase.files]
        directories = set()
        for path in file_paths:
            parts = Path(path).parts
            for i in range(1, len(parts)):
                directories.add("/".join(parts[:i]))

        arch_data["directory_structure"] = {
            "total_directories": len(directories),
            "max_depth": max(len(Path(path).parts) for path in file_paths)
            if file_paths
            else 0,
            "files_per_directory": {},
            "common_patterns": [],
        }

        # Analyze files per directory
        dir_file_count = defaultdict(int)
        for path in file_paths:
            directory = str(Path(path).parent)
            dir_file_count[directory] += 1

        arch_data["directory_structure"]["files_per_directory"] = dict(
            sorted(dir_file_count.items(), key=lambda x: x[1], reverse=True)[:20]
        )

        # Detect common architectural patterns
        patterns = []
        if any("src" in path for path in file_paths):
            patterns.append("src/ directory pattern")
        if any("lib" in path for path in file_paths):
            patterns.append("lib/ directory pattern")
        if any("test" in path.lower() for path in file_paths):
            patterns.append("test directory pattern")
        if any("api" in path.lower() for path in file_paths):
            patterns.append("API directory pattern")
        if any("component" in path.lower() for path in file_paths):
            patterns.append("component-based architecture")

        arch_data["directory_structure"]["common_patterns"] = patterns

    except Exception as e:
        arch_data["directory_structure"]["error"] = str(e)

    return arch_data


def language_specific_analysis(codebase: Codebase) -> Dict[str, Any]:
    """Language-specific analysis using Python and TypeScript analyzers."""
    lang_data = {
        "python_analysis": {},
        "typescript_analysis": {},
        "language_distribution": {},
        "cross_language_dependencies": [],
    }

    # Analyze language distribution
    try:
        file_extensions = defaultdict(int)
        for file in codebase.files:
            ext = Path(file.filepath).suffix.lower()
            file_extensions[ext] += 1

        lang_data["language_distribution"] = dict(file_extensions)

        # Determine primary languages
        total_files = sum(file_extensions.values())
        lang_percentages = {
            ext: (count / total_files * 100) if total_files > 0 else 0
            for ext, count in file_extensions.items()
        }
        lang_data["language_percentages"] = lang_percentages

    except Exception as e:
        lang_data["language_distribution"]["error"] = str(e)

    # Python-specific analysis
    if python_analyzer:
        try:
            python_files = [f for f in codebase.files if f.filepath.endswith(".py")]
            lang_data["python_analysis"] = {
                "total_python_files": len(python_files),
                "python_patterns": [],
                "python_specific_issues": [],
            }

            # Analyze Python-specific patterns
            for file in python_files[:10]:  # Limit for performance
                try:
                    if "__init__.py" in file.filepath:
                        lang_data["python_analysis"]["python_patterns"].append(
                            "Package structure detected"
                        )
                    if "setup.py" in file.filepath:
                        lang_data["python_analysis"]["python_patterns"].append(
                            "Python package setup detected"
                        )
                    if (
                        "requirements.txt" in file.filepath
                        or "pyproject.toml" in file.filepath
                    ):
                        lang_data["python_analysis"]["python_patterns"].append(
                            "Dependency management detected"
                        )

                except Exception as e:
                    lang_data["python_analysis"]["python_specific_issues"].append(
                        str(e)
                    )

        except Exception as e:
            lang_data["python_analysis"]["error"] = str(e)

    # TypeScript-specific analysis
    if typescript_analyzer:
        try:
            ts_files = [
                f
                for f in codebase.files
                if f.filepath.endswith((".ts", ".tsx", ".js", ".jsx"))
            ]
            lang_data["typescript_analysis"] = {
                "total_typescript_files": len(ts_files),
                "typescript_patterns": [],
                "typescript_specific_issues": [],
            }

            # Analyze TypeScript-specific patterns
            for file in ts_files[:10]:  # Limit for performance
                try:
                    if "package.json" in file.filepath:
                        lang_data["typescript_analysis"]["typescript_patterns"].append(
                            "NPM package detected"
                        )
                    if "tsconfig.json" in file.filepath:
                        lang_data["typescript_analysis"]["typescript_patterns"].append(
                            "TypeScript configuration detected"
                        )
                    if ".tsx" in file.filepath:
                        lang_data["typescript_analysis"]["typescript_patterns"].append(
                            "React/JSX components detected"
                        )

                except Exception as e:
                    lang_data["typescript_analysis"][
                        "typescript_specific_issues"
                    ].append(str(e))

        except Exception as e:
            lang_data["typescript_analysis"]["error"] = str(e)

    return lang_data


def advanced_performance_analysis(codebase: Codebase) -> Dict[str, Any]:
    """Advanced performance analysis using graph-sitter's deep AST capabilities."""
    perf_data = {
        "algorithmic_complexity": {},
        "memory_usage_patterns": {},
        "io_operations": {},
        "performance_bottlenecks": [],
        "optimization_suggestions": [],
    }

    try:
        # Analyze algorithmic complexity patterns
        complexity_patterns = {
            "nested_loops": 0,
            "recursive_functions": 0,
            "large_data_structures": 0,
            "database_queries": 0,
        }

        for file in codebase.files:
            for func in file.functions:
                try:
                    if not hasattr(func, "source"):
                        continue

                    source = func.source.lower()

                    # Detect nested loops (simplified)
                    loop_keywords = ["for ", "while "]
                    loop_count = sum(source.count(keyword) for keyword in loop_keywords)
                    if loop_count > 1:
                        complexity_patterns["nested_loops"] += 1

                    # Detect recursive functions
                    if func.name in source:
                        complexity_patterns["recursive_functions"] += 1

                    # Detect large data structure operations
                    large_data_keywords = ["list(", "dict(", "set(", "array", "map("]
                    if any(keyword in source for keyword in large_data_keywords):
                        complexity_patterns["large_data_structures"] += 1

                    # Detect database operations
                    db_keywords = [
                        "select ",
                        "insert ",
                        "update ",
                        "delete ",
                        "query",
                        "execute",
                    ]
                    if any(keyword in source for keyword in db_keywords):
                        complexity_patterns["database_queries"] += 1

                except Exception:
                    continue

        perf_data["algorithmic_complexity"] = complexity_patterns

        # Generate optimization suggestions based on patterns
        suggestions = []
        if complexity_patterns["nested_loops"] > 5:
            suggestions.append(
                "Consider optimizing nested loops - found multiple instances"
            )
        if complexity_patterns["recursive_functions"] > 10:
            suggestions.append(
                "Review recursive functions for potential stack overflow issues"
            )
        if complexity_patterns["large_data_structures"] > 20:
            suggestions.append(
                "Consider memory-efficient alternatives for large data structures"
            )
        if complexity_patterns["database_queries"] > 15:
            suggestions.append(
                "Consider database query optimization and connection pooling"
            )

        perf_data["optimization_suggestions"] = suggestions

    except Exception as e:
        perf_data["error"] = str(e)

    return perf_data


def comprehensive_error_context_analysis(
    codebase: Codebase, max_issues: int = 200
) -> Dict[str, Any]:
    """
    Comprehensive error analysis with detailed context using advanced graph-sitter features.
    Provides the detailed error context requested: file paths, line numbers, function names,
    interconnected context, and fix suggestions.
    """

    # Import calculate_cyclomatic_complexity function locally to avoid circular imports
    def calculate_cyclomatic_complexity(function):
        """Enhanced cyclomatic complexity calculation with better statement handling."""
        base_complexity = 1

        if hasattr(function, "source"):
            # Fallback to source-based analysis
            source = function.source.lower()
            base_complexity += source.count("if ") + source.count("elif ")
            base_complexity += source.count("for ") + source.count("while ")
            base_complexity += source.count("except ")
            base_complexity += source.count(" and ") + source.count(" or ")

        return base_complexity

    error_analysis = {
        "total_issues": 0,
        "critical_issues": 0,
        "issues_by_severity": {},
        "issues_by_file": {},
        "interconnected_analysis": {},
        "detailed_issues": [],
    }

    issue_counter = 0

    for file in codebase.files:
        if issue_counter >= max_issues:
            break

        file_issues = []

        try:
            # Enhanced syntax and semantic analysis
            for func in file.functions:
                if issue_counter >= max_issues:
                    break

                try:
                    # Get function context and interconnections
                    func_context = get_enhanced_function_context(func, codebase)

                    # Complexity analysis with detailed context
                    complexity = calculate_cyclomatic_complexity(func)
                    if complexity > 10:
                        severity = (
                            "critical"
                            if complexity > 25
                            else "high"
                            if complexity > 15
                            else "medium"
                        )

                        issue = {
                            "id": f"complexity_{hashlib.md5(f'{file.filepath}_{func.name}'.encode()).hexdigest()[:8]}",
                            "type": "complexity_issue",
                            "severity": severity,
                            "file_path": file.filepath,
                            "line_number": getattr(func, "line_number", None),
                            "function_name": func.name,
                            "message": f"High cyclomatic complexity: {complexity}",
                            "description": f"Function '{func.name}' has cyclomatic complexity of {complexity}",
                            "context": {
                                "complexity_score": complexity,
                                "parameters_count": len(func.parameters),
                                "return_statements": len(func.return_statements),
                                "function_calls": len(func.function_calls),
                                "dependencies": [
                                    dep.name for dep in func.dependencies[:5]
                                ],
                                "call_sites": len(func.call_sites),
                            },
                            "interconnected_context": func_context,
                            "affected_symbols": {
                                "functions": [
                                    call.name for call in func.function_calls[:10]
                                ],
                                "parameters": [param.name for param in func.parameters],
                                "dependencies": [
                                    dep.name for dep in func.dependencies[:10]
                                ],
                            },
                            "fix_suggestions": [
                                f"Break down '{func.name}' into smaller functions (current complexity: {complexity})",
                                "Extract complex conditional logic into separate methods",
                                "Consider using strategy pattern for complex branching",
                                f"Target complexity should be under 10 (currently {complexity})",
                            ],
                        }

                        file_issues.append(issue)
                        error_analysis["detailed_issues"].append(issue)
                        issue_counter += 1

                        if severity == "critical":
                            error_analysis["critical_issues"] += 1

                except Exception:
                    continue

            # Store file-level issues
            if file_issues:
                error_analysis["issues_by_file"][file.filepath] = {
                    "total_issues": len(file_issues),
                    "critical_count": len(
                        [i for i in file_issues if i["severity"] == "critical"]
                    ),
                    "high_count": len(
                        [i for i in file_issues if i["severity"] == "high"]
                    ),
                    "medium_count": len(
                        [i for i in file_issues if i["severity"] == "medium"]
                    ),
                    "issues": file_issues,
                }

        except Exception:
            continue

    # Calculate summary statistics
    error_analysis["total_issues"] = issue_counter

    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for issue in error_analysis["detailed_issues"]:
        severity = issue.get("severity", "low")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    error_analysis["issues_by_severity"] = severity_counts

    return error_analysis


def get_enhanced_function_context(func: Function, codebase: Codebase) -> Dict[str, Any]:
    """Get enhanced context for a function including all interconnected elements."""
    context = {
        "dependencies": [],
        "dependents": [],
        "call_graph": {},
        "data_flow": {},
        "related_classes": [],
        "related_files": [],
    }

    try:
        # Dependencies analysis
        for dep in func.dependencies[:10]:
            context["dependencies"].append(
                {
                    "name": getattr(dep, "name", "unknown"),
                    "type": type(dep).__name__,
                    "file": getattr(dep, "filepath", "unknown"),
                }
            )

        # Call sites analysis
        for call_site in func.call_sites[:10]:
            context["dependents"].append(
                {
                    "caller": getattr(call_site, "name", "unknown"),
                    "file": getattr(call_site, "filepath", "unknown"),
                }
            )

        # Function calls analysis
        call_graph = {}
        for call in func.function_calls[:10]:
            call_graph[call.name] = {
                "type": "function_call",
                "arguments_count": len(getattr(call, "args", [])),
            }
        context["call_graph"] = call_graph

        # Related files through imports
        if hasattr(func, "file"):
            file = func.file
            related_files = []
            for imp in file.imports[:5]:
                if hasattr(imp, "imported_symbol") and hasattr(
                    imp.imported_symbol, "filepath"
                ):
                    related_files.append(imp.imported_symbol.filepath)
            context["related_files"] = related_files

    except Exception as e:
        context["error"] = str(e)

    return context


def get_enhanced_class_context(cls: Class, codebase: Codebase) -> Dict[str, Any]:
    """Get enhanced context for a class including all interconnected elements."""
    context = {
        "inheritance_chain": [],
        "composition_relationships": [],
        "method_dependencies": {},
        "attribute_usage": {},
        "related_classes": [],
    }

    try:
        # Local complexity calculation to avoid circular imports
        def calculate_cyclomatic_complexity(function):
            base_complexity = 1
            if hasattr(function, "source"):
                source = function.source.lower()
                base_complexity += source.count("if ") + source.count("elif ")
                base_complexity += source.count("for ") + source.count("while ")
                base_complexity += source.count("except ")
                base_complexity += source.count(" and ") + source.count(" or ")
            return base_complexity

        # Inheritance analysis
        context["inheritance_chain"] = cls.parent_class_names

        # Method analysis
        method_deps = {}
        for method in cls.methods[:10]:
            method_deps[method.name] = {
                "parameters": len(method.parameters),
                "complexity": calculate_cyclomatic_complexity(method),
                "calls": [call.name for call in method.function_calls[:5]],
            }
        context["method_dependencies"] = method_deps

        # Attribute analysis
        attr_usage = {}
        for attr in cls.attributes[:10]:
            attr_usage[attr.name] = {
                "type": getattr(attr, "type", "unknown"),
                "access_level": getattr(attr, "access_level", "unknown"),
            }
        context["attribute_usage"] = attr_usage

    except Exception as e:
        context["error"] = str(e)

    return context


def generate_call_graph(codebase: Codebase) -> Dict[str, Any]:
    """Generate a comprehensive call graph for the entire codebase."""
    from .api import (
        calculate_cyclomatic_complexity,
        get_operators_and_operands,
        calculate_halstead_volume,
    )

    graph_data = {
        "nodes": [],
        "edges": [],
        "metadata": {
            "total_functions": 0,
            "total_calls": 0,
            "max_complexity": 0,
            "languages": set(),
        },
    }

    function_map = {}  # Map function names to IDs for edge creation

    for file in codebase.files:
        # Determine language from file extension
        ext = Path(file.filepath).suffix.lower()
        graph_data["metadata"]["languages"].add(ext)

        for function in file.functions:
            try:
                complexity = calculate_cyclomatic_complexity(function)
                operators, operands = get_operators_and_operands(function)
                volume, _, _, _, _ = calculate_halstead_volume(operators, operands)

                function_id = f"{file.filepath}:{function.name}"
                function_map[function.name] = function_id

                # Add node for each function
                graph_data["nodes"].append(
                    {
                        "id": function_id,
                        "name": function.name,
                        "type": "function",
                        "file": file.filepath,
                        "language": ext,
                        "metrics": {
                            "complexity": complexity,
                            "halstead_volume": volume,
                            "parameters": len(function.parameters),
                            "return_statements": len(function.return_statements),
                        },
                    }
                )

                graph_data["metadata"]["total_functions"] += 1
                graph_data["metadata"]["max_complexity"] = max(
                    graph_data["metadata"]["max_complexity"], complexity
                )

                # Add edges for function calls
                for call in function.function_calls:
                    target_id = function_map.get(call.name, f"external:{call.name}")
                    graph_data["edges"].append(
                        {
                            "source": function_id,
                            "target": target_id,
                            "type": "calls",
                            "call_name": call.name,
                        }
                    )
                    graph_data["metadata"]["total_calls"] += 1

            except Exception as e:
                print(f"Error processing function {function.name}: {e}")
                continue

    graph_data["metadata"]["languages"] = list(graph_data["metadata"]["languages"])
    return graph_data


def generate_code_views(file: SourceFile) -> Dict[str, Any]:
    """Generate multiple views of code structure including AST, CFG, and DFG."""
    views = {
        "ast": generate_ast_view(file),
        "cfg": generate_control_flow_graph(file),
        "dfg": generate_data_flow_graph(file),
        "metadata": {
            "file_path": file.filepath,
            "file_size": len(file.source),
            "language": Path(file.filepath).suffix.lower(),
        },
    }
    return views


def generate_ast_view(file: SourceFile) -> Dict[str, Any]:
    """Generate Abstract Syntax Tree view"""
    from .api import calculate_cyclomatic_complexity

    try:
        ast_data = {
            "functions": [],
            "classes": [],
            "imports": [],
            "global_variables": [],
        }

        for func in file.functions:
            ast_data["functions"].append(
                {
                    "name": func.name,
                    "line_number": getattr(func, "line_number", None),
                    "parameters": [p.name for p in func.parameters],
                    "return_type": getattr(func, "return_type", None),
                    "decorators": [d.name for d in func.decorators]
                    if hasattr(func, "decorators")
                    else [],
                    "complexity": calculate_cyclomatic_complexity(func),
                }
            )

        for cls in file.classes:
            ast_data["classes"].append(
                {
                    "name": cls.name,
                    "line_number": getattr(cls, "line_number", None),
                    "methods": [m.name for m in cls.methods],
                    "attributes": [a.name for a in cls.attributes],
                    "parent_classes": cls.parent_class_names,
                }
            )

        for imp in file.imports:
            ast_data["imports"].append(
                {
                    "name": imp.name,
                    "module": getattr(imp, "module", None),
                    "alias": getattr(imp, "alias", None),
                }
            )

        for var in file.global_vars:
            ast_data["global_variables"].append(
                {"name": var.name, "type": getattr(var, "type", None)}
            )

        return ast_data

    except Exception as e:
        return {"error": f"Failed to generate AST view: {str(e)}"}


def generate_control_flow_graph(file: SourceFile) -> Dict[str, Any]:
    """Generate Control Flow Graph"""
    from .api import calculate_cyclomatic_complexity

    cfg_data = {"functions": {}, "complexity_analysis": {}}

    try:
        for func in file.functions:
            func_cfg = {
                "entry_points": [],
                "exit_points": [],
                "branches": [],
                "loops": [],
                "complexity": calculate_cyclomatic_complexity(func),
            }

            # Analyze control flow patterns in source
            if hasattr(func, "source"):
                source = func.source.lower()

                # Count control flow structures
                func_cfg["branches"] = [
                    {"type": "if", "count": source.count("if ")},
                    {"type": "elif", "count": source.count("elif ")},
                    {"type": "else", "count": source.count("else")},
                    {"type": "try", "count": source.count("try:")},
                    {"type": "except", "count": source.count("except")},
                ]

                func_cfg["loops"] = [
                    {"type": "for", "count": source.count("for ")},
                    {"type": "while", "count": source.count("while ")},
                ]

                func_cfg["exit_points"] = [
                    {"type": "return", "count": source.count("return ")},
                    {"type": "raise", "count": source.count("raise ")},
                    {"type": "break", "count": source.count("break")},
                    {"type": "continue", "count": source.count("continue")},
                ]

            cfg_data["functions"][func.name] = func_cfg
            cfg_data["complexity_analysis"][func.name] = func_cfg["complexity"]

        return cfg_data

    except Exception as e:
        return {"error": f"Failed to generate CFG: {str(e)}"}


def generate_data_flow_graph(file: SourceFile) -> Dict[str, Any]:
    """Generate Data Flow Graph"""
    dfg_data = {"variables": {}, "data_dependencies": [], "assignments": []}

    try:
        for func in file.functions:
            func_variables = {
                "parameters": [p.name for p in func.parameters],
                "local_vars": [],
                "assignments": [],
                "usages": [],
            }

            # Analyze variable usage patterns
            if hasattr(func, "source"):
                source = func.source

                # Simple pattern matching for assignments
                assignment_pattern = r"(\w+)\s*=\s*"
                assignments = re.findall(assignment_pattern, source)
                func_variables["assignments"] = list(set(assignments))

                # Variable usage pattern
                for param in func_variables["parameters"]:
                    usage_count = source.count(param)
                    if usage_count > 1:  # More than just the parameter declaration
                        func_variables["usages"].append(
                            {"variable": param, "usage_count": usage_count - 1}
                        )

            dfg_data["variables"][func.name] = func_variables

        return dfg_data

    except Exception as e:
        return {"error": f"Failed to generate DFG: {str(e)}"}


def calculate_advanced_metrics(codebase: Codebase) -> Dict[str, Any]:
    """Calculate advanced code metrics including complexity, dependencies, and quality."""
    from .api import calculate_cyclomatic_complexity

    metrics = {
        "complexity": {"cyclomatic": {}, "cognitive": {}, "halstead": {}},
        "dependencies": {"internal": [], "external": [], "circular": []},
        "quality": {
            "maintainability_index": {},
            "code_smells": [],
            "technical_debt": {},
        },
        "summary": {
            "total_files": 0,
            "total_functions": 0,
            "total_classes": 0,
            "avg_complexity": 0,
            "high_complexity_functions": 0,
        },
    }

    total_complexity = 0
    function_count = 0

    for file in codebase.files:
        try:
            file_metrics = analyze_file_metrics(file)

            metrics["complexity"]["cyclomatic"][file.filepath] = file_metrics[
                "cyclomatic"
            ]
            metrics["complexity"]["cognitive"][file.filepath] = file_metrics[
                "cognitive"
            ]
            metrics["complexity"]["halstead"][file.filepath] = file_metrics["halstead"]
            metrics["quality"]["maintainability_index"][file.filepath] = file_metrics[
                "maintainability"
            ]

            # Accumulate for summary
            metrics["summary"]["total_files"] += 1
            metrics["summary"]["total_functions"] += len(file.functions)
            metrics["summary"]["total_classes"] += len(file.classes)

            for func in file.functions:
                complexity = calculate_cyclomatic_complexity(func)
                total_complexity += complexity
                function_count += 1

                if complexity > 15:
                    metrics["summary"]["high_complexity_functions"] += 1
                    metrics["quality"]["code_smells"].append(
                        {
                            "type": "high_complexity",
                            "file": file.filepath,
                            "function": func.name,
                            "complexity": complexity,
                        }
                    )

            # Analyze dependencies
            for imp in file.imports:
                if hasattr(imp, "imported_symbol") and hasattr(
                    imp.imported_symbol, "filepath"
                ):
                    metrics["dependencies"]["internal"].append(
                        {
                            "from": file.filepath,
                            "to": imp.imported_symbol.filepath,
                            "symbol": imp.name,
                        }
                    )
                else:
                    metrics["dependencies"]["external"].append(
                        {"from": file.filepath, "module": imp.name}
                    )

        except Exception as e:
            print(f"Error analyzing file {file.filepath}: {e}")
            continue

    # Calculate summary statistics
    if function_count > 0:
        metrics["summary"]["avg_complexity"] = total_complexity / function_count

    # Detect circular dependencies
    metrics["dependencies"]["circular"] = detect_circular_dependencies(
        metrics["dependencies"]["internal"]
    )

    return metrics


def analyze_file_metrics(file: SourceFile) -> Dict[str, Any]:
    """Analyze metrics for a single file"""
    from .api import (
        calculate_cyclomatic_complexity,
        get_operators_and_operands,
        calculate_halstead_volume,
        calculate_maintainability_index,
    )

    file_metrics = {
        "cyclomatic": {},
        "cognitive": {},
        "halstead": {},
        "maintainability": {},
    }

    try:
        total_complexity = 0
        total_volume = 0
        total_mi = 0

        for func in file.functions:
            complexity = calculate_cyclomatic_complexity(func)
            operators, operands = get_operators_and_operands(func)
            volume, _, _, _, _ = calculate_halstead_volume(operators, operands)

            loc = len(func.source.splitlines()) if hasattr(func, "source") else 0
            mi_score = calculate_maintainability_index(volume, complexity, loc)

            file_metrics["cyclomatic"][func.name] = complexity
            file_metrics["cognitive"][func.name] = complexity  # Simplified
            file_metrics["halstead"][func.name] = volume
            file_metrics["maintainability"][func.name] = mi_score

            total_complexity += complexity
            total_volume += volume
            total_mi += mi_score

        # File-level averages
        func_count = len(file.functions)
        if func_count > 0:
            file_metrics["avg_complexity"] = total_complexity / func_count
            file_metrics["avg_volume"] = total_volume / func_count
            file_metrics["avg_maintainability"] = total_mi / func_count

        return file_metrics

    except Exception as e:
        return {"error": f"Failed to analyze file metrics: {str(e)}"}


def detect_circular_dependencies(internal_deps: List[Dict]) -> List[Dict]:
    """Detect circular dependencies in the codebase"""
    circular_deps = []

    try:
        # Build dependency graph
        dep_graph = defaultdict(set)
        for dep in internal_deps:
            dep_graph[dep["from"]].add(dep["to"])

        # Simple cycle detection using DFS
        def has_cycle(node, visited, rec_stack, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in dep_graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack, path):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    circular_deps.append({"cycle": cycle, "length": len(cycle) - 1})
                    return True

            rec_stack.remove(node)
            path.pop()
            return False

        visited = set()
        for node in dep_graph:
            if node not in visited:
                has_cycle(node, visited, set(), [])

        return circular_deps[:10]  # Limit results

    except Exception as e:
        return [{"error": f"Failed to detect circular dependencies: {str(e)}"}]


def analyze_security_patterns(codebase: Codebase) -> Dict[str, Any]:
    """Analyze security patterns and potential vulnerabilities"""
    security_analysis = {
        "vulnerabilities": [],
        "security_smells": [],
        "call_graph_anomalies": [],
        "dependency_risks": [],
        "summary": {
            "total_vulnerabilities": 0,
            "critical_issues": 0,
            "high_risk_files": [],
        },
    }

    try:
        # Security patterns to detect
        security_patterns = [
            (r"eval\s*\(", "Code injection via eval()", "critical"),
            (r"exec\s*\(", "Code execution via exec()", "critical"),
            (r"subprocess\.call\s*\(.*shell\s*=\s*True", "Shell injection", "high"),
            (r"pickle\.loads?\s*\(", "Unsafe deserialization", "high"),
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password", "high"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key", "high"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret", "medium"),
            (r"sql.*\+.*\+", "Potential SQL injection", "medium"),
            (r'open\s*\(.*["\']w["\']', "File write operation", "low"),
            (
                r"requests\.get\s*\(.*verify\s*=\s*False",
                "SSL verification disabled",
                "medium",
            ),
        ]

        file_risk_scores = {}

        for file in codebase.files:
            file_vulnerabilities = []
            file_risk_score = 0

            for pattern, description, severity in security_patterns:
                matches = re.finditer(pattern, file.source, re.IGNORECASE)
                for match in matches:
                    line_num = file.source[: match.start()].count("\n") + 1

                    vulnerability = {
                        "type": "security_vulnerability",
                        "severity": severity,
                        "description": description,
                        "file": file.filepath,
                        "line": line_num,
                        "matched_text": match.group(),
                        "pattern": pattern,
                    }

                    file_vulnerabilities.append(vulnerability)
                    security_analysis["vulnerabilities"].append(vulnerability)

                    # Calculate risk score
                    risk_points = {"critical": 10, "high": 5, "medium": 2, "low": 1}
                    file_risk_score += risk_points.get(severity, 0)

                    if severity == "critical":
                        security_analysis["summary"]["critical_issues"] += 1

            if file_vulnerabilities:
                file_risk_scores[file.filepath] = {
                    "risk_score": file_risk_score,
                    "vulnerability_count": len(file_vulnerabilities),
                }

        # Analyze call graph for anomalies
        try:
            call_graph = generate_call_graph(codebase)
            security_analysis["call_graph_anomalies"] = detect_call_anomalies(
                call_graph
            )
        except Exception as e:
            security_analysis["call_graph_anomalies"] = [{"error": str(e)}]

        # Identify high-risk files
        sorted_files = sorted(
            file_risk_scores.items(), key=lambda x: x[1]["risk_score"], reverse=True
        )
        security_analysis["summary"]["high_risk_files"] = [
            {
                "file": file,
                "risk_score": data["risk_score"],
                "vulnerabilities": data["vulnerability_count"],
            }
            for file, data in sorted_files[:10]
        ]

        security_analysis["summary"]["total_vulnerabilities"] = len(
            security_analysis["vulnerabilities"]
        )

        return security_analysis

    except Exception as e:
        return {"error": f"Security analysis failed: {str(e)}"}


def detect_call_anomalies(call_graph: Dict[str, Any]) -> List[Dict]:
    """Detect anomalies in call graph patterns"""
    anomalies = []

    try:
        # Analyze call patterns
        call_counts = defaultdict(int)
        function_calls = defaultdict(set)

        for edge in call_graph["edges"]:
            call_counts[edge["target"]] += 1
            function_calls[edge["source"]].add(edge["target"])

        # Detect highly called functions (potential bottlenecks)
        avg_calls = sum(call_counts.values()) / len(call_counts) if call_counts else 0
        for func, count in call_counts.items():
            if count > avg_calls * 3:  # 3x average
                anomalies.append(
                    {
                        "type": "high_call_frequency",
                        "function": func,
                        "call_count": count,
                        "description": f"Function called {count} times (avg: {avg_calls:.1f})",
                    }
                )

        # Detect functions with too many outgoing calls
        for func, calls in function_calls.items():
            if len(calls) > 10:
                anomalies.append(
                    {
                        "type": "high_fan_out",
                        "function": func,
                        "outgoing_calls": len(calls),
                        "description": f"Function makes {len(calls)} different calls",
                    }
                )

        return anomalies[:20]  # Limit results

    except Exception as e:
        return [{"error": f"Failed to detect call anomalies: {str(e)}"}]


def validate_source_code(file: SourceFile) -> bool:
    """Validate source code for syntax errors before analysis"""
    try:
        codebase_cache.get_or_compute_ast(file)
        return True
    except Exception as e:
        raise CodeAnalysisError(f"Syntax error in {file.filepath}: {str(e)}")


def analyze_code_issues(codebase: Codebase, max_issues: int = 100) -> List[CodeIssue]:
    # ... (code from api (5).py)
    return []  # Placeholder


def identify_critical_files(codebase: Codebase) -> List[CriticalFile]:
    # ... (code from api (5).py)
    return []  # Placeholder


def build_dependency_graph(codebase: Codebase) -> Dict[str, DependencyNode]:
    # ... (code from api (5).py)
    return {}  # Placeholder


def detect_entrypoints(codebase: Codebase) -> List[EntryPoint]:
    # ... (code from api (5).py)
    return []  # Placeholder


def get_file_detailed_analysis(codebase: Codebase, file_path: str) -> Dict[str, Any]:
    # ... (code from api (5).py)
    return {}  # Placeholder
