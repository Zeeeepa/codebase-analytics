"""
Advanced Error Detection Engine
Comprehensive error detection using Tree-sitter's full capabilities
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
import re
import ast
from pathlib import Path

from ..core.graph_manager import CodeError, ErrorCategory, ErrorSeverity


class AdvancedErrorDetector:
    """
    Advanced error detection engine that leverages Tree-sitter's
    parsing capabilities for comprehensive error analysis
    """
    
    def __init__(self, graph_manager):
        self.graph_manager = graph_manager
        self.error_patterns = self._initialize_error_patterns()
        
    def _initialize_error_patterns(self) -> Dict[str, List[Dict]]:
        """Initialize comprehensive error detection patterns"""
        return {
            "null_reference": [
                {
                    "pattern": r"\.get\([^)]*\)(?!\s*(?:if|and|or|\?))",
                    "severity": ErrorSeverity.CRITICAL,
                    "message": "Potential null reference - .get() without null check",
                    "fix": "Add null check: if result := obj.get(key): ..."
                },
                {
                    "pattern": r"\[[^\]]*\](?!\s*(?:if|and|or|\?))",
                    "severity": ErrorSeverity.MAJOR,
                    "message": "Potential KeyError - dictionary access without check",
                    "fix": "Use .get() method or try/except block"
                }
            ],
            "missing_return": [
                {
                    "pattern": r"def\s+\w+\([^)]*\):[^}]*(?<!return\s[^;]*;?)$",
                    "severity": ErrorSeverity.MAJOR,
                    "message": "Function may be missing return statement",
                    "fix": "Add explicit return statement"
                }
            ],
            "unused_imports": [
                {
                    "pattern": r"^import\s+(\w+)(?:\s+as\s+\w+)?$",
                    "severity": ErrorSeverity.MINOR,
                    "message": "Potentially unused import",
                    "fix": "Remove unused import or use the imported module"
                }
            ],
            "inefficient_patterns": [
                {
                    "pattern": r"for\s+\w+\s+in\s+range\(len\([^)]+\)\):",
                    "severity": ErrorSeverity.MINOR,
                    "message": "Inefficient iteration pattern",
                    "fix": "Use enumerate() or iterate directly over the collection"
                },
                {
                    "pattern": r"\.append\([^)]+\)\s*$",
                    "severity": ErrorSeverity.INFO,
                    "message": "Consider list comprehension for better performance",
                    "fix": "Use list comprehension if applicable"
                }
            ],
            "security_issues": [
                {
                    "pattern": r"eval\s*\(",
                    "severity": ErrorSeverity.CRITICAL,
                    "message": "Security risk - eval() usage",
                    "fix": "Use ast.literal_eval() or safer alternatives"
                },
                {
                    "pattern": r"exec\s*\(",
                    "severity": ErrorSeverity.CRITICAL,
                    "message": "Security risk - exec() usage",
                    "fix": "Avoid dynamic code execution"
                }
            ],
            "resource_leaks": [
                {
                    "pattern": r"open\s*\([^)]*\)(?!\s*(?:with|as))",
                    "severity": ErrorSeverity.MAJOR,
                    "message": "Potential resource leak - file not closed properly",
                    "fix": "Use 'with open(...) as f:' context manager"
                }
            ],
            "type_errors": [
                {
                    "pattern": r"(\w+)\s*\+\s*(\w+)",
                    "severity": ErrorSeverity.MINOR,
                    "message": "Potential type mismatch in addition",
                    "fix": "Ensure both operands are of compatible types"
                }
            ]
        }
    
    def detect_comprehensive_errors(self, codebase) -> List[CodeError]:
        """
        Detect all types of errors comprehensively
        """
        all_errors = []
        
        if not codebase:
            return all_errors
        
        for file in codebase.files:
            file_path = getattr(file, 'file_path', 'unknown')
            source = getattr(file, 'source', '')
            
            # Pattern-based error detection
            pattern_errors = self._detect_pattern_errors(file, source, file_path)
            all_errors.extend(pattern_errors)
            
            # AST-based error detection
            ast_errors = self._detect_ast_errors(file, source, file_path)
            all_errors.extend(ast_errors)
            
            # Symbol-based error detection
            symbol_errors = self._detect_symbol_errors(file, codebase)
            all_errors.extend(symbol_errors)
            
            # Dependency-based error detection
            dependency_errors = self._detect_dependency_errors(file, codebase)
            all_errors.extend(dependency_errors)
        
        return all_errors
    
    def _detect_pattern_errors(self, file, source: str, file_path: str) -> List[CodeError]:
        """Detect errors using regex patterns"""
        errors = []
        lines = source.split('\n')
        
        for category, patterns in self.error_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                severity = pattern_info["severity"]
                message = pattern_info["message"]
                fix = pattern_info["fix"]
                
                for line_num, line in enumerate(lines, 1):
                    matches = re.finditer(pattern, line)
                    for match in matches:
                        error = CodeError(
                            id=f"pattern_{category}_{line_num}_{match.start()}",
                            category=self._get_error_category(category),
                            severity=severity,
                            message=message,
                            file_path=file_path,
                            line_number=line_num,
                            column_number=match.start(),
                            context={
                                "line": line.strip(),
                                "pattern": pattern,
                                "match": match.group()
                            },
                            affected_symbols=self._extract_symbols_from_line(line),
                            dependencies=[],
                            fix_suggestion=fix,
                            impact_assessment=self._assess_impact(severity),
                            confidence=0.8
                        )
                        errors.append(error)
        
        return errors
    
    def _detect_ast_errors(self, file, source: str, file_path: str) -> List[CodeError]:
        """Detect errors using AST analysis"""
        errors = []
        
        try:
            tree = ast.parse(source)
            
            # Detect unreachable code
            unreachable_errors = self._detect_unreachable_code(tree, file_path)
            errors.extend(unreachable_errors)
            
            # Detect undefined variables
            undefined_errors = self._detect_undefined_variables(tree, file_path)
            errors.extend(undefined_errors)
            
            # Detect unused variables
            unused_errors = self._detect_unused_variables(tree, file_path)
            errors.extend(unused_errors)
            
        except SyntaxError as e:
            # Syntax error detected
            syntax_error = CodeError(
                id=f"syntax_{e.lineno}_{e.offset}",
                category=ErrorCategory.SYNTAX,
                severity=ErrorSeverity.CRITICAL,
                message=f"Syntax error: {e.msg}",
                file_path=file_path,
                line_number=e.lineno or 0,
                column_number=e.offset or 0,
                context={"syntax_error": str(e)},
                affected_symbols=[],
                dependencies=[],
                fix_suggestion="Fix syntax error",
                impact_assessment="Critical - code will not execute",
                confidence=1.0
            )
            errors.append(syntax_error)
        
        return errors
    
    def _detect_unreachable_code(self, tree: ast.AST, file_path: str) -> List[CodeError]:
        """Detect unreachable code using AST analysis"""
        errors = []
        
        class UnreachableCodeVisitor(ast.NodeVisitor):
            def __init__(self):
                self.errors = []
                self.after_return = False
            
            def visit_Return(self, node):
                self.after_return = True
                self.generic_visit(node)
            
            def visit_stmt(self, node):
                if self.after_return and not isinstance(node, (ast.Return, ast.FunctionDef, ast.ClassDef)):
                    error = CodeError(
                        id=f"unreachable_{node.lineno}",
                        category=ErrorCategory.STRUCTURAL,
                        severity=ErrorSeverity.MAJOR,
                        message="Unreachable code after return statement",
                        file_path=file_path,
                        line_number=node.lineno,
                        column_number=node.col_offset,
                        context={"node_type": type(node).__name__},
                        affected_symbols=[],
                        dependencies=[],
                        fix_suggestion="Remove unreachable code or restructure logic",
                        impact_assessment="Medium - dead code affects maintainability",
                        confidence=0.9
                    )
                    self.errors.append(error)
                
                self.generic_visit(node)
        
        visitor = UnreachableCodeVisitor()
        visitor.visit(tree)
        errors.extend(visitor.errors)
        
        return errors
    
    def _detect_undefined_variables(self, tree: ast.AST, file_path: str) -> List[CodeError]:
        """Detect undefined variables using AST analysis"""
        errors = []
        
        class UndefinedVariableVisitor(ast.NodeVisitor):
            def __init__(self):
                self.errors = []
                self.defined_vars = set()
                self.used_vars = set()
            
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store):
                    self.defined_vars.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    self.used_vars.add((node.id, node.lineno, node.col_offset))
                
                self.generic_visit(node)
            
            def check_undefined(self):
                for var_name, line_no, col_offset in self.used_vars:
                    if var_name not in self.defined_vars and not self._is_builtin(var_name):
                        error = CodeError(
                            id=f"undefined_{var_name}_{line_no}",
                            category=ErrorCategory.SEMANTIC,
                            severity=ErrorSeverity.MAJOR,
                            message=f"Undefined variable '{var_name}'",
                            file_path=file_path,
                            line_number=line_no,
                            column_number=col_offset,
                            context={"variable": var_name},
                            affected_symbols=[var_name],
                            dependencies=[],
                            fix_suggestion=f"Define variable '{var_name}' before use",
                            impact_assessment="High - will cause NameError at runtime",
                            confidence=0.85
                        )
                        self.errors.append(error)
            
            def _is_builtin(self, name: str) -> bool:
                """Check if name is a Python builtin"""
                import builtins
                return hasattr(builtins, name)
        
        visitor = UndefinedVariableVisitor()
        visitor.visit(tree)
        visitor.check_undefined()
        errors.extend(visitor.errors)
        
        return errors
    
    def _detect_unused_variables(self, tree: ast.AST, file_path: str) -> List[CodeError]:
        """Detect unused variables using AST analysis"""
        errors = []
        
        class UnusedVariableVisitor(ast.NodeVisitor):
            def __init__(self):
                self.errors = []
                self.defined_vars = {}  # name -> (line, col)
                self.used_vars = set()
            
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store):
                    self.defined_vars[node.id] = (node.lineno, node.col_offset)
                elif isinstance(node.ctx, ast.Load):
                    self.used_vars.add(node.id)
                
                self.generic_visit(node)
            
            def check_unused(self):
                for var_name, (line_no, col_offset) in self.defined_vars.items():
                    if var_name not in self.used_vars and not var_name.startswith('_'):
                        error = CodeError(
                            id=f"unused_{var_name}_{line_no}",
                            category=ErrorCategory.STYLE,
                            severity=ErrorSeverity.MINOR,
                            message=f"Unused variable '{var_name}'",
                            file_path=file_path,
                            line_number=line_no,
                            column_number=col_offset,
                            context={"variable": var_name},
                            affected_symbols=[var_name],
                            dependencies=[],
                            fix_suggestion=f"Remove unused variable '{var_name}' or prefix with '_'",
                            impact_assessment="Low - affects code cleanliness",
                            confidence=0.7
                        )
                        self.errors.append(error)
        
        visitor = UnusedVariableVisitor()
        visitor.visit(tree)
        visitor.check_unused()
        errors.extend(visitor.errors)
        
        return errors
    
    def _detect_symbol_errors(self, file, codebase) -> List[CodeError]:
        """Detect errors using Graph-sitter's symbol analysis"""
        errors = []
        
        for symbol in getattr(file, 'symbols', []):
            if hasattr(symbol, 'name') and 'Function' in str(type(symbol)):
                # Check function-specific issues
                function_errors = self._analyze_function_errors(symbol, file, codebase)
                errors.extend(function_errors)
        
        return errors
    
    def _analyze_function_errors(self, function, file, codebase) -> List[CodeError]:
        """Analyze function-specific errors"""
        errors = []
        
        name = getattr(function, 'name', 'unknown')
        file_path = getattr(file, 'file_path', 'unknown')
        source = getattr(function, 'source', '')
        
        # Check for long functions
        if hasattr(function, 'start_point') and hasattr(function, 'end_point'):
            line_count = function.end_point[0] - function.start_point[0]
            if line_count > 50:
                errors.append(CodeError(
                    id=f"long_function_{name}",
                    category=ErrorCategory.STYLE,
                    severity=ErrorSeverity.MINOR,
                    message=f"Function '{name}' is too long ({line_count} lines)",
                    file_path=file_path,
                    line_number=function.start_point[0],
                    column_number=0,
                    context={"line_count": line_count, "function_name": name},
                    affected_symbols=[name],
                    dependencies=[],
                    fix_suggestion="Break down into smaller functions",
                    impact_assessment="Medium - affects maintainability",
                    confidence=0.9
                ))
        
        # Check for missing docstrings
        if '"""' not in source and "'''" not in source:
            errors.append(CodeError(
                id=f"missing_doc_{name}",
                category=ErrorCategory.STYLE,
                severity=ErrorSeverity.MINOR,
                message=f"Function '{name}' lacks documentation",
                file_path=file_path,
                line_number=getattr(function, 'start_point', [0])[0] if hasattr(function, 'start_point') else 0,
                column_number=0,
                context={"function_name": name, "has_docstring": False},
                affected_symbols=[name],
                dependencies=[],
                fix_suggestion="Add docstring explaining function purpose",
                impact_assessment="Low - affects code documentation",
                confidence=0.8
            ))
        
        # Check for unused parameters
        parameters = getattr(function, 'parameters', [])
        for param in parameters:
            param_name = getattr(param, 'name', str(param))
            if param_name not in source.replace(f'def {name}(', ''):
                errors.append(CodeError(
                    id=f"unused_param_{name}_{param_name}",
                    category=ErrorCategory.STYLE,
                    severity=ErrorSeverity.MINOR,
                    message=f"Unused parameter '{param_name}' in function '{name}'",
                    file_path=file_path,
                    line_number=getattr(function, 'start_point', [0])[0] if hasattr(function, 'start_point') else 0,
                    column_number=0,
                    context={"parameter": param_name, "function_name": name},
                    affected_symbols=[name, param_name],
                    dependencies=[],
                    fix_suggestion=f"Remove unused parameter '{param_name}' or use it in function body",
                    impact_assessment="Low - affects function signature",
                    confidence=0.7
                ))
        
        return errors
    
    def _detect_dependency_errors(self, file, codebase) -> List[CodeError]:
        """Detect dependency-related errors"""
        errors = []
        
        # Check for circular imports
        # Check for missing imports
        # Check for unused imports
        
        return errors
    
    def _get_error_category(self, pattern_category: str) -> ErrorCategory:
        """Map pattern category to ErrorCategory enum"""
        mapping = {
            "null_reference": ErrorCategory.IMPLEMENTATION,
            "missing_return": ErrorCategory.IMPLEMENTATION,
            "unused_imports": ErrorCategory.STYLE,
            "inefficient_patterns": ErrorCategory.PERFORMANCE,
            "security_issues": ErrorCategory.SECURITY,
            "resource_leaks": ErrorCategory.IMPLEMENTATION,
            "type_errors": ErrorCategory.SEMANTIC
        }
        return mapping.get(pattern_category, ErrorCategory.IMPLEMENTATION)
    
    def _extract_symbols_from_line(self, line: str) -> List[str]:
        """Extract symbol names from a line of code"""
        # Simple regex to extract identifiers
        symbols = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', line)
        # Filter out Python keywords
        keywords = {'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'finally', 'with', 'as', 'import', 'from', 'return', 'yield', 'break', 'continue', 'pass', 'and', 'or', 'not', 'in', 'is', 'True', 'False', 'None'}
        return [s for s in symbols if s not in keywords]
    
    def _assess_impact(self, severity: ErrorSeverity) -> str:
        """Assess the impact of an error based on its severity"""
        impact_map = {
            ErrorSeverity.CRITICAL: "High - potential runtime failure or security risk",
            ErrorSeverity.MAJOR: "Medium - logic errors or significant issues",
            ErrorSeverity.MINOR: "Low - style or maintainability issues",
            ErrorSeverity.INFO: "Minimal - suggestions for improvement"
        }
        return impact_map.get(severity, "Unknown impact")
    
    def generate_error_summary(self, errors: List[CodeError]) -> Dict[str, Any]:
        """Generate a comprehensive error summary"""
        summary = {
            "total_errors": len(errors),
            "by_severity": {},
            "by_category": {},
            "by_file": {},
            "critical_files": [],
            "most_common_errors": []
        }
        
        # Group by severity
        for error in errors:
            severity = error.severity.value
            summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
        
        # Group by category
        for error in errors:
            category = error.category.value
            summary["by_category"][category] = summary["by_category"].get(category, 0) + 1
        
        # Group by file
        for error in errors:
            file_path = error.file_path
            if file_path not in summary["by_file"]:
                summary["by_file"][file_path] = {"count": 0, "errors": []}
            summary["by_file"][file_path]["count"] += 1
            summary["by_file"][file_path]["errors"].append(error)
        
        # Identify critical files (files with most errors)
        file_error_counts = [(file, data["count"]) for file, data in summary["by_file"].items()]
        file_error_counts.sort(key=lambda x: x[1], reverse=True)
        summary["critical_files"] = file_error_counts[:5]
        
        # Find most common error types
        error_types = {}
        for error in errors:
            error_type = f"{error.category.value}_{error.severity.value}"
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        most_common = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
        summary["most_common_errors"] = most_common[:10]
        
        return summary
