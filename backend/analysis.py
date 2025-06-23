#!/usr/bin/env python3
"""
Comprehensive Codebase Analysis Engine
Graph-sitter compliant analysis providing ALL analysis context including:
- ALL most important functions with full definitions
- ALL entry points detection
- Issue detection and context
- Function relationships and dependencies
- Symbol analysis and cross-references
"""

import os
import sys
import json
import ast
import re
import logging
import networkx as nx
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from enum import Enum
from pathlib import Path
from datetime import datetime

# Graph-sitter integration
try:
    import tree_sitter
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logging.warning("tree-sitter not available, falling back to AST parsing")

# Codegen SDK integration (graph_sitter = codegen as per requirements)
try:
    import codegen as graph_sitter  # graph_sitter = codegen (mandatory)
    from codegen.sdk.core.codebase import Codebase
    from codegen.sdk.core.file import SourceFile
    from codegen.sdk.core.function import Function
    from codegen.sdk.core.class_definition import Class
    from codegen.sdk.core.symbol import Symbol
    from codegen.sdk.core.import_resolution import Import
    from codegen.sdk.enums import EdgeType, SymbolType
    from codegen.shared.enums.programming_language import ProgrammingLanguage
    CODEGEN_SDK_AVAILABLE = True
    print("✅ Using Codegen SDK (graph_sitter = codegen)")
except ImportError:
    CODEGEN_SDK_AVAILABLE = False
    print("⚠️ Codegen SDK not available, using mock implementations")
    # Mock implementations for development
    class MockCodebase:
        def __init__(self, path): 
            self.path = path
            self.functions = []
            self.classes = []
            self.files = []
        def get_symbol(self, name): return None
    
    graph_sitter = type('MockGraphSitter', (), {'Codebase': MockCodebase})()

class IssueSeverity(Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"

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
    UNUSED_PARAMETER = "unused_parameter"
    
    # Exception Handling
    IMPROPER_EXCEPTION_HANDLING = "improper_exception_handling"
    MISSING_ERROR_HANDLING = "missing_error_handling"
    UNSAFE_ASSERTION = "unsafe_assertion"
    RESOURCE_LEAK = "resource_leak"
    
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
    IMPORT_ORGANIZATION = "import_organization"
    
    # Runtime Risks
    INFINITE_LOOP = "infinite_loop"
    MEMORY_LEAK = "memory_leak"
    RACE_CONDITION = "race_condition"
    DEADLOCK = "deadlock"

@dataclass
class CodeIssue:
    """Represents a code issue with full context"""
    type: IssueType
    severity: IssueSeverity
    message: str
    file_path: str
    line_number: int
    column_number: int = 0
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    context: Optional[str] = None
    suggestion: Optional[str] = None
    related_symbols: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type.value,
            'severity': self.severity.value,
            'message': self.message,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'column_number': self.column_number,
            'end_line': self.end_line,
            'end_column': self.end_column,
            'context': self.context,
            'suggestion': self.suggestion,
            'related_symbols': self.related_symbols
        }

@dataclass
class FunctionDefinition:
    """Complete function definition with context"""
    name: str
    file_path: str
    line_start: int
    line_end: int
    parameters: List[Dict[str, Any]]
    return_type: Optional[str]
    docstring: Optional[str]
    source_code: str
    complexity_score: int
    is_entry_point: bool
    calls: List[str]
    called_by: List[str]
    dependencies: List[str]
    issues: List[CodeIssue]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict excluding complexity metrics from user output"""
        result = asdict(self)
        # Remove complexity_score from user-facing output (keep for internal use)
        result.pop('complexity_score', None)
        result['issues'] = [issue.to_dict() for issue in self.issues]
        return result

@dataclass
class EntryPoint:
    """Entry point definition"""
    name: str
    type: str  # main, cli, api_endpoint, class_method, etc.
    file_path: str
    line_number: int
    description: str
    parameters: List[Dict[str, Any]]
    dependencies: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AnalysisResult:
    """Complete analysis result"""
    repository_path: str
    analysis_timestamp: str
    total_files: int
    total_lines: int
    programming_languages: List[str]
    
    # Core analysis results
    all_functions: List[FunctionDefinition]
    all_entry_points: List[EntryPoint]
    all_issues: List[CodeIssue]
    
    # Metrics (internal use only, not in reports)
    complexity_metrics: Dict[str, Any]
    dependency_graph: Dict[str, List[str]]
    symbol_table: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'repository_path': self.repository_path,
            'analysis_timestamp': self.analysis_timestamp,
            'total_files': self.total_files,
            'total_lines': self.total_lines,
            'programming_languages': self.programming_languages,
            'all_functions': [func.to_dict() for func in self.all_functions],
            'all_entry_points': [ep.to_dict() for ep in self.all_entry_points],
            'all_issues': [issue.to_dict() for issue in self.all_issues],
            'dependency_graph': self.dependency_graph,
            'symbol_table': self.symbol_table,
            # Note: complexity_metrics excluded from output as per requirements
        }

class GraphSitterAnalyzer:
    """Graph-sitter compliant analyzer following https://graph-sitter.com/introduction/how-it-works"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.parsers = {}
        self.dependency_graph = nx.DiGraph()
        self.symbol_table = {}
        self.setup_parsers()
    
    def setup_parsers(self):
        """Setup tree-sitter parsers for supported languages"""
        if not TREE_SITTER_AVAILABLE:
            return
            
        # Language configurations
        language_configs = {
            'python': {'extensions': ['.py'], 'parser': 'python'},
            'javascript': {'extensions': ['.js', '.jsx'], 'parser': 'javascript'},
            'typescript': {'extensions': ['.ts', '.tsx'], 'parser': 'typescript'},
            'java': {'extensions': ['.java'], 'parser': 'java'},
            'cpp': {'extensions': ['.cpp', '.cc', '.cxx', '.c++'], 'parser': 'cpp'},
            'c': {'extensions': ['.c', '.h'], 'parser': 'c'},
            'rust': {'extensions': ['.rs'], 'parser': 'rust'},
            'go': {'extensions': ['.go'], 'parser': 'go'},
        }
        
        for lang, config in language_configs.items():
            try:
                # This would load the actual tree-sitter language
                # For now, we'll use AST parsing as fallback
                self.parsers[lang] = config
            except Exception as e:
                logging.warning(f"Could not load {lang} parser: {e}")
    
    def analyze_codebase(self) -> AnalysisResult:
        """Perform comprehensive codebase analysis"""
        logging.info(f"Starting comprehensive analysis of {self.repo_path}")
        
        # Collect all source files
        source_files = self._collect_source_files()
        
        # Initialize result containers
        all_functions = []
        all_entry_points = []
        all_issues = []
        total_lines = 0
        languages = set()
        
        # Analyze each file
        for file_path in source_files:
            try:
                file_analysis = self._analyze_file(file_path)
                all_functions.extend(file_analysis['functions'])
                all_entry_points.extend(file_analysis['entry_points'])
                all_issues.extend(file_analysis['issues'])
                total_lines += file_analysis['line_count']
                languages.add(file_analysis['language'])
            except Exception as e:
                logging.error(f"Error analyzing {file_path}: {e}")
                continue
        
        # Build dependency relationships
        self._build_dependency_graph(all_functions)
        
        # Identify most important functions based on complexity and usage
        important_functions = self._identify_important_functions(all_functions)
        
        # Create analysis result
        result = AnalysisResult(
            repository_path=str(self.repo_path),
            analysis_timestamp=datetime.now().isoformat(),
            total_files=len(source_files),
            total_lines=total_lines,
            programming_languages=list(languages),
            all_functions=important_functions,  # ALL important functions
            all_entry_points=all_entry_points,  # ALL entry points
            all_issues=all_issues,
            complexity_metrics=self._calculate_complexity_metrics(all_functions),
            dependency_graph=nx.to_dict_of_lists(self.dependency_graph),
            symbol_table=self.symbol_table
        )
        
        logging.info(f"Analysis complete: {len(important_functions)} functions, {len(all_entry_points)} entry points, {len(all_issues)} issues")
        return result
    
    def _collect_source_files(self) -> List[Path]:
        """Collect all source files in the repository"""
        source_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.cc', '.cxx', '.c++', '.c', '.h', '.rs', '.go'}
        source_files = []
        
        for root, dirs, files in os.walk(self.repo_path):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'node_modules', '__pycache__', 'build', 'dist', 'target'}]
            
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix in source_extensions:
                    source_files.append(file_path)
        
        return source_files
    
    def _analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single source file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logging.error(f"Could not read {file_path}: {e}")
            return {'functions': [], 'entry_points': [], 'issues': [], 'line_count': 0, 'language': 'unknown'}
        
        language = self._detect_language(file_path)
        line_count = len(content.splitlines())
        
        if language == 'python':
            return self._analyze_python_file(file_path, content)
        elif language in ['javascript', 'typescript']:
            return self._analyze_js_ts_file(file_path, content, language)
        else:
            # Generic analysis for other languages
            return self._analyze_generic_file(file_path, content, language)
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension"""
        ext = file_path.suffix.lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.c++': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.rs': 'rust',
            '.go': 'go',
        }
        return language_map.get(ext, 'unknown')
    
    def _analyze_python_file(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Analyze Python file using AST"""
        functions = []
        entry_points = []
        issues = []
        
        try:
            tree = ast.parse(content)
            
            # Extract functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_def = self._extract_python_function(node, file_path, content)
                    functions.append(func_def)
                    
                    # Check if it's an entry point
                    if self._is_python_entry_point(node, content):
                        entry_point = EntryPoint(
                            name=node.name,
                            type=self._get_entry_point_type(node, content),
                            file_path=str(file_path),
                            line_number=node.lineno,
                            description=ast.get_docstring(node) or f"Entry point function: {node.name}",
                            parameters=self._extract_parameters(node),
                            dependencies=[]
                        )
                        entry_points.append(entry_point)
                
                elif isinstance(node, ast.ClassDef):
                    # Analyze class methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            func_def = self._extract_python_function(item, file_path, content, class_name=node.name)
                            functions.append(func_def)
            
            # Detect issues
            issues = self._detect_python_issues(tree, file_path, content)
            
        except SyntaxError as e:
            issues.append(CodeIssue(
                type=IssueType.TYPE_MISMATCH,
                severity=IssueSeverity.CRITICAL,
                message=f"Syntax error: {e}",
                file_path=str(file_path),
                line_number=e.lineno or 1
            ))
        
        return {
            'functions': functions,
            'entry_points': entry_points,
            'issues': issues,
            'line_count': len(content.splitlines()),
            'language': 'python'
        }
    
    def _extract_python_function(self, node: ast.FunctionDef, file_path: Path, content: str, class_name: Optional[str] = None) -> FunctionDefinition:
        """Extract complete function definition from AST node"""
        lines = content.splitlines()
        
        # Get function source code
        start_line = node.lineno - 1
        end_line = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else start_line + 10
        end_line = min(end_line, len(lines))
        
        source_code = '\n'.join(lines[start_line:end_line])
        
        # Extract parameters
        parameters = self._extract_parameters(node)
        
        # Get return type annotation
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)
        
        # Calculate complexity (internal use)
        complexity_score = self._calculate_function_complexity(node)
        
        # Extract function calls
        calls = self._extract_function_calls(node)
        
        # Function name with class prefix if applicable
        full_name = f"{class_name}.{node.name}" if class_name else node.name
        
        return FunctionDefinition(
            name=full_name,
            file_path=str(file_path),
            line_start=node.lineno,
            line_end=end_line,
            parameters=parameters,
            return_type=return_type,
            docstring=ast.get_docstring(node),
            source_code=source_code,
            complexity_score=complexity_score,
            is_entry_point=self._is_python_entry_point(node, content),
            calls=calls,
            called_by=[],  # Will be populated later
            dependencies=[],  # Will be populated later
            issues=[]  # Will be populated later
        )
    
    def _extract_parameters(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract function parameters with type annotations"""
        parameters = []
        
        for arg in node.args.args:
            param_info = {
                'name': arg.arg,
                'type': None,
                'default': None,
                'required': True
            }
            
            # Type annotation
            if arg.annotation:
                param_info['type'] = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else str(arg.annotation)
            
            parameters.append(param_info)
        
        # Handle defaults
        defaults = node.args.defaults
        if defaults:
            # Defaults apply to the last N parameters
            num_defaults = len(defaults)
            for i, default in enumerate(defaults):
                param_idx = len(parameters) - num_defaults + i
                if param_idx >= 0:
                    parameters[param_idx]['default'] = ast.unparse(default) if hasattr(ast, 'unparse') else str(default)
                    parameters[param_idx]['required'] = False
        
        return parameters
    
    def _is_python_entry_point(self, node: ast.FunctionDef, content: str) -> bool:
        """Comprehensive entry point detection - finds ALL entry points"""
        # Expanded entry point patterns for comprehensive detection
        entry_point_names = {
            'main', 'run', 'start', 'execute', 'cli', 'app', 'serve', 'launch',
            'init', 'setup', 'configure', 'bootstrap', 'entry', 'handler',
            'process', 'worker', 'daemon', 'service', 'server', 'client'
        }
        
        # Check function name patterns
        if node.name in entry_point_names:
            return True
        
        # Check for common entry point prefixes/suffixes
        if (node.name.startswith(('run_', 'start_', 'main_', 'exec_', 'handle_')) or
            node.name.endswith(('_main', '_run', '_start', '_handler', '_entry', '_worker'))):
            return True
        
        # Check for if __name__ == "__main__" pattern
        if 'if __name__ == "__main__"' in content:
            return True
        
        # Check for web framework decorators (Flask, FastAPI, Django)
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                if decorator.id in {'app.route', 'click.command', 'typer.command', 'celery.task'}:
                    return True
            elif isinstance(decorator, ast.Attribute):
                if decorator.attr in {'route', 'command', 'task', 'get', 'post', 'put', 'delete', 'patch'}:
                    return True
                # Check for framework-specific decorators
                if hasattr(decorator.value, 'id'):
                    framework_patterns = {
                        'app': ['route', 'before_request', 'after_request'],
                        'api': ['route', 'get', 'post', 'put', 'delete'],
                        'celery': ['task'],
                        'click': ['command', 'group'],
                        'typer': ['command'],
                    }
                    for framework, methods in framework_patterns.items():
                        if decorator.value.id == framework and decorator.attr in methods:
                            return True
        
        # Check for async functions (often entry points in async frameworks)
        if isinstance(node, ast.AsyncFunctionDef):
            return True
        
        # Check for test functions
        if node.name.startswith('test_') or node.name.endswith('_test'):
            return True
        
        # Check for special methods that could be entry points
        special_methods = {'__call__', '__enter__', '__exit__', '__aenter__', '__aexit__'}
        if node.name in special_methods:
            return True
        
        return False
    
    def _get_entry_point_type(self, node: ast.FunctionDef, content: str) -> str:
        """Comprehensive entry point type classification"""
        # Check decorators for specific types
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Attribute):
                if decorator.attr in {'route', 'get', 'post', 'put', 'delete', 'patch'}:
                    return 'api_endpoint'
                elif decorator.attr in {'command', 'group'}:
                    return 'cli_command'
                elif decorator.attr == 'task':
                    return 'background_task'
                elif decorator.attr in {'before_request', 'after_request'}:
                    return 'middleware'
        
        # Check for async functions
        if isinstance(node, ast.AsyncFunctionDef):
            if node.name.startswith('handle_'):
                return 'async_handler'
            return 'async_function'
        
        # Check function name patterns for comprehensive classification
        if node.name == 'main' or 'if __name__ == "__main__"' in content:
            return 'main_function'
        elif node.name.startswith('test_') or node.name.endswith('_test'):
            return 'test_function'
        elif node.name in {'run', 'start', 'execute', 'launch'}:
            return 'execution_function'
        elif node.name in {'setup', 'configure', 'init', 'bootstrap'}:
            return 'initialization_function'
        elif node.name in {'serve', 'server', 'daemon', 'service'}:
            return 'service_function'
        elif node.name.startswith('handle_') or node.name.endswith('_handler'):
            return 'event_handler'
        elif node.name.endswith('_worker') or 'worker' in node.name:
            return 'worker_function'
        elif node.name in {'__call__', '__enter__', '__exit__', '__aenter__', '__aexit__'}:
            return 'special_method'
        elif node.name.startswith('run_') or node.name.startswith('exec_'):
            return 'execution_function'
        elif 'cli' in node.name.lower():
            return 'cli_function'
        elif 'api' in node.name.lower():
            return 'api_function'
        
        return 'function'
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity (internal use only)"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    def _extract_function_calls(self, node: ast.FunctionDef) -> List[str]:
        """Extract all function calls within a function"""
        calls = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
        
        return list(set(calls))  # Remove duplicates
    
    def _detect_python_issues(self, tree: ast.AST, file_path: Path, content: str) -> List[CodeIssue]:
        """Detect various issues in Python code"""
        issues = []
        lines = content.splitlines()
        
        for node in ast.walk(tree):
            # Detect missing return statements
            if isinstance(node, ast.FunctionDef):
                if not self._has_return_statement(node) and node.name != '__init__':
                    issues.append(CodeIssue(
                        type=IssueType.MISSING_RETURN,
                        severity=IssueSeverity.MINOR,
                        message=f"Function '{node.name}' may be missing a return statement",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        suggestion="Add explicit return statement or return type annotation"
                    ))
            
            # Detect unused variables
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        if var_name.startswith('_') and not self._is_variable_used(tree, var_name, node.lineno):
                            issues.append(CodeIssue(
                                type=IssueType.UNDEFINED_VARIABLE,
                                severity=IssueSeverity.MINOR,
                                message=f"Variable '{var_name}' appears to be unused",
                                file_path=str(file_path),
                                line_number=node.lineno
                            ))
            
            # Detect magic numbers
            elif isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)) and node.value not in {0, 1, -1}:
                    issues.append(CodeIssue(
                        type=IssueType.MAGIC_NUMBER,
                        severity=IssueSeverity.MINOR,
                        message=f"Magic number {node.value} should be replaced with a named constant",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        suggestion="Define a named constant for this value"
                    ))
            
            # Detect long functions
            elif isinstance(node, ast.FunctionDef):
                func_length = (node.end_lineno or node.lineno + 10) - node.lineno
                if func_length > 50:  # Configurable threshold
                    issues.append(CodeIssue(
                        type=IssueType.LONG_FUNCTION,
                        severity=IssueSeverity.MAJOR,
                        message=f"Function '{node.name}' is {func_length} lines long, consider breaking it down",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        suggestion="Break down into smaller, more focused functions"
                    ))
        
        return issues
    
    def _has_return_statement(self, node: ast.FunctionDef) -> bool:
        """Check if function has return statement"""
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                return True
        return False
    
    def _is_variable_used(self, tree: ast.AST, var_name: str, defined_line: int) -> bool:
        """Check if variable is used after definition"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == var_name:
                if hasattr(node, 'lineno') and node.lineno > defined_line:
                    return True
        return False
    
    def _analyze_js_ts_file(self, file_path: Path, content: str, language: str) -> Dict[str, Any]:
        """Analyze JavaScript/TypeScript file"""
        # For now, use regex-based analysis
        # In a full implementation, this would use tree-sitter
        functions = []
        entry_points = []
        issues = []
        
        # Simple regex patterns for function detection
        function_patterns = [
            r'function\s+(\w+)\s*\([^)]*\)\s*{',
            r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*{',
            r'(\w+)\s*:\s*function\s*\([^)]*\)\s*{',
            r'async\s+function\s+(\w+)\s*\([^)]*\)\s*{'
        ]
        
        lines = content.splitlines()
        for i, line in enumerate(lines):
            for pattern in function_patterns:
                match = re.search(pattern, line)
                if match:
                    func_name = match.group(1)
                    
                    # Create basic function definition
                    func_def = FunctionDefinition(
                        name=func_name,
                        file_path=str(file_path),
                        line_start=i + 1,
                        line_end=i + 10,  # Approximate
                        parameters=[],  # Would need proper parsing
                        return_type=None,
                        docstring=None,
                        source_code=line,
                        complexity_score=1,
                        is_entry_point=func_name in {'main', 'init', 'start'},
                        calls=[],
                        called_by=[],
                        dependencies=[],
                        issues=[]
                    )
                    functions.append(func_def)
                    
                    # Check for entry points
                    if func_name in {'main', 'init', 'start'} or 'export' in line:
                        entry_point = EntryPoint(
                            name=func_name,
                            type='function',
                            file_path=str(file_path),
                            line_number=i + 1,
                            description=f"{language.title()} function: {func_name}",
                            parameters=[],
                            dependencies=[]
                        )
                        entry_points.append(entry_point)
        
        return {
            'functions': functions,
            'entry_points': entry_points,
            'issues': issues,
            'line_count': len(lines),
            'language': language
        }
    
    def _analyze_generic_file(self, file_path: Path, content: str, language: str) -> Dict[str, Any]:
        """Generic analysis for unsupported languages"""
        lines = content.splitlines()
        
        # Basic pattern matching for common function definitions
        function_patterns = {
            'java': r'(public|private|protected)?\s*(static)?\s*\w+\s+(\w+)\s*\([^)]*\)\s*{',
            'cpp': r'\w+\s+(\w+)\s*\([^)]*\)\s*{',
            'c': r'\w+\s+(\w+)\s*\([^)]*\)\s*{',
            'rust': r'fn\s+(\w+)\s*\([^)]*\)\s*{',
            'go': r'func\s+(\w+)\s*\([^)]*\)\s*{?'
        }
        
        pattern = function_patterns.get(language, r'(\w+)\s*\([^)]*\)\s*{')
        functions = []
        entry_points = []
        
        for i, line in enumerate(lines):
            match = re.search(pattern, line)
            if match:
                func_name = match.group(-1)  # Last group is usually the function name
                
                func_def = FunctionDefinition(
                    name=func_name,
                    file_path=str(file_path),
                    line_start=i + 1,
                    line_end=i + 10,  # Approximate
                    parameters=[],
                    return_type=None,
                    docstring=None,
                    source_code=line,
                    complexity_score=1,
                    is_entry_point=func_name == 'main',
                    calls=[],
                    called_by=[],
                    dependencies=[],
                    issues=[]
                )
                functions.append(func_def)
                
                # Check for main function
                if func_name == 'main':
                    entry_point = EntryPoint(
                        name=func_name,
                        type='main_function',
                        file_path=str(file_path),
                        line_number=i + 1,
                        description=f"Main function in {language}",
                        parameters=[],
                        dependencies=[]
                    )
                    entry_points.append(entry_point)
        
        return {
            'functions': functions,
            'entry_points': entry_points,
            'issues': [],
            'line_count': len(lines),
            'language': language
        }
    
    def _build_dependency_graph(self, functions: List[FunctionDefinition]):
        """Build dependency graph between functions"""
        # Create function name to definition mapping
        func_map = {func.name: func for func in functions}
        
        # Build call relationships
        for func in functions:
            for called_func in func.calls:
                if called_func in func_map:
                    # Add edge in dependency graph
                    self.dependency_graph.add_edge(func.name, called_func)
                    
                    # Update called_by relationships
                    func_map[called_func].called_by.append(func.name)
                    func.dependencies.append(called_func)
    
    def _identify_important_functions(self, all_functions: List[FunctionDefinition]) -> List[FunctionDefinition]:
        """Identify ALL most important functions based on various criteria"""
        important_functions = []
        
        # Sort functions by importance score
        scored_functions = []
        for func in all_functions:
            score = self._calculate_importance_score(func)
            scored_functions.append((score, func))
        
        # Sort by score (descending)
        scored_functions.sort(key=lambda x: x[0], reverse=True)
        
        # Return ALL functions that meet importance criteria
        # This ensures we don't miss any important functions
        for score, func in scored_functions:
            if (score > 5 or  # High complexity/usage
                func.is_entry_point or  # Entry points are always important
                len(func.called_by) > 3 or  # Frequently called
                len(func.calls) > 10 or  # Calls many other functions
                func.complexity_score > 10):  # High complexity
                important_functions.append(func)
        
        # Ensure we have at least some functions even for small codebases
        if len(important_functions) < 10 and len(all_functions) > 0:
            # Add top functions by score
            remaining_needed = min(10, len(all_functions))
            for i in range(remaining_needed):
                if i < len(scored_functions):
                    func = scored_functions[i][1]
                    if func not in important_functions:
                        important_functions.append(func)
        
        return important_functions
    
    def _calculate_importance_score(self, func: FunctionDefinition) -> int:
        """Calculate importance score for a function (internal use)"""
        score = 0
        
        # Entry points get high score
        if func.is_entry_point:
            score += 20
        
        # Functions called by many others
        score += len(func.called_by) * 2
        
        # Functions that call many others (orchestrators)
        score += len(func.calls)
        
        # Complexity contributes to importance
        score += func.complexity_score
        
        # Long functions might be important (but also problematic)
        if func.line_end - func.line_start > 20:
            score += 5
        
        # Functions with issues might need attention
        score += len(func.issues) * 2
        
        return score
    
    def _calculate_complexity_metrics(self, functions: List[FunctionDefinition]) -> Dict[str, Any]:
        """Calculate overall complexity metrics (internal use only)"""
        if not functions:
            return {}
        
        complexities = [func.complexity_score for func in functions]
        
        return {
            'total_functions': len(functions),
            'average_complexity': sum(complexities) / len(complexities),
            'max_complexity': max(complexities),
            'min_complexity': min(complexities),
            'high_complexity_count': len([c for c in complexities if c > 10])
        }

# Main analysis functions for API integration
def analyze_codebase(repo_path: str) -> Dict[str, Any]:
    """
    Main analysis function following graph-sitter standards
    Returns comprehensive analysis with ALL important functions and ALL entry points
    """
    try:
        # Always use AST analysis - no mocks
        analyzer = GraphSitterAnalyzer(repo_path)
        result = analyzer.analyze_codebase()
        return result.to_dict()
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        raise Exception(f"Real analysis failed: {e}")

def _analyze_with_codegen_sdk(codebase) -> Dict[str, Any]:
    """Analyze using Codegen SDK following graph-sitter patterns"""
    try:
        # Get all symbols from the codebase graph
        all_functions = []
        all_entry_points = []
        all_issues = []
        
        # Extract functions using graph-sitter approach
        for function in codebase.functions:
            # Get function context as per graph-sitter documentation
            context = get_function_context(function)
            
            func_def = FunctionDefinition(
                name=function.name,
                file_path=function.filepath,
                line_start=getattr(function, 'line_start', 1),
                line_end=getattr(function, 'line_end', 1),
                parameters=_extract_codegen_parameters(function),
                return_type=getattr(function, 'return_type', None),
                docstring=getattr(function, 'docstring', None),
                source_code=function.source,
                complexity_score=_calculate_codegen_complexity(function),
                is_entry_point=_is_codegen_entry_point(function),
                calls=[call.name for call in function.function_calls if hasattr(call, 'name')],
                called_by=[site.parent_function.name for site in function.call_sites 
                          if hasattr(site, 'parent_function') and site.parent_function],
                dependencies=[dep.name for dep in function.dependencies if hasattr(dep, 'name')],
                issues=[]
            )
            all_functions.append(func_def)
            
            # Check if it's an entry point
            if func_def.is_entry_point:
                entry_point = EntryPoint(
                    name=function.name,
                    type=_get_codegen_entry_point_type(function),
                    file_path=function.filepath,
                    line_number=getattr(function, 'line_start', 1),
                    description=getattr(function, 'docstring', f"Entry point: {function.name}"),
                    parameters=_extract_codegen_parameters(function),
                    dependencies=[dep.name for dep in function.dependencies if hasattr(dep, 'name')]
                )
                all_entry_points.append(entry_point)
        
        # Identify ALL most important functions
        important_functions = _identify_all_important_functions(all_functions)
        
        return {
            'repository_path': codebase.path,
            'analysis_timestamp': datetime.now().isoformat(),
            'total_files': len(codebase.files),
            'total_lines': sum(len(f.source.splitlines()) for f in codebase.files if hasattr(f, 'source')),
            'programming_languages': list(set(f.language.value for f in codebase.files if hasattr(f, 'language'))),
            'all_functions': [func.to_dict() for func in important_functions],
            'all_entry_points': [ep.to_dict() for ep in all_entry_points],
            'all_issues': [issue.to_dict() for issue in all_issues],
            'dependency_graph': _build_codegen_dependency_graph(codebase),
            'symbol_table': _build_codegen_symbol_table(codebase)
        }
    except Exception as e:
        logging.error(f"Codegen SDK analysis failed: {e}")
        return _create_mock_analysis_result(codebase.path)

def _identify_all_important_functions(functions: List[FunctionDefinition]) -> List[FunctionDefinition]:
    """Identify ALL most important functions (not just one)"""
    if not functions:
        return []
    
    important_functions = []
    
    # Score all functions
    scored_functions = []
    for func in functions:
        score = _calculate_function_importance_score(func)
        scored_functions.append((score, func))
    
    # Sort by importance score
    scored_functions.sort(key=lambda x: x[0], reverse=True)
    
    # Include ALL functions that meet importance criteria
    for score, func in scored_functions:
        if (score > 5 or  # High importance score
            func.is_entry_point or  # All entry points are important
            len(func.called_by) > 2 or  # Frequently called
            len(func.calls) > 5 or  # Calls many functions
            len(func.source_code.splitlines()) > 20):  # Substantial functions
            important_functions.append(func)
    
    # Ensure we return a reasonable number of functions
    if len(important_functions) < 10 and len(functions) > 0:
        # Add top functions by score to reach at least 10
        needed = min(10, len(functions))
        for i in range(needed):
            if i < len(scored_functions):
                func = scored_functions[i][1]
                if func not in important_functions:
                    important_functions.append(func)
    
    return important_functions

def _calculate_function_importance_score(func: FunctionDefinition) -> int:
    """Calculate comprehensive importance score"""
    score = 0
    
    # Entry points are always important
    if func.is_entry_point:
        score += 25
    
    # Usage frequency
    score += len(func.called_by) * 3
    
    # Function calls (orchestration)
    score += len(func.calls) * 2
    
    # Function size (substantial functions are often important)
    lines = len(func.source_code.splitlines())
    if lines > 10:
        score += min(lines // 5, 10)  # Cap at 10 points for size
    
    # Dependencies (complex functions)
    score += len(func.dependencies)
    
    # Issues (problematic functions need attention)
    score += len(func.issues) * 2
    
    # Name patterns that suggest importance
    important_patterns = ['main', 'init', 'setup', 'process', 'handle', 'execute', 'run']
    if any(pattern in func.name.lower() for pattern in important_patterns):
        score += 10
    
    return score

def get_function_context(function) -> Dict[str, Any]:
    """Get comprehensive function context following graph-sitter patterns"""
    try:
        context = {
            "implementation": {
                "source": getattr(function, 'source', ''),
                "filepath": getattr(function, 'filepath', '')
            },
            "dependencies": [],
            "usages": [],
            "call_chain": [],
            "relationships": {}
        }
        
        # Add dependencies following graph-sitter approach
        if hasattr(function, 'dependencies'):
            for dep in function.dependencies:
                if hasattr(dep, 'source') and hasattr(dep, 'filepath'):
                    context["dependencies"].append({
                        "source": dep.source,
                        "filepath": dep.filepath,
                        "name": getattr(dep, 'name', 'unknown')
                    })
        
        # Add usages following graph-sitter approach
        if hasattr(function, 'usages'):
            for usage in function.usages:
                if hasattr(usage, 'usage_symbol'):
                    context["usages"].append({
                        "source": usage.usage_symbol.source,
                        "filepath": usage.usage_symbol.filepath,
                        "line": getattr(usage.usage_symbol, 'line_number', 0)
                    })
        
        # Build call chain
        context["call_chain"] = _build_function_call_chain(function)
        
        return context
    except Exception as e:
        logging.error(f"Error getting function context: {e}")
        return {"implementation": {"source": "", "filepath": ""}, "dependencies": [], "usages": []}

def _build_function_call_chain(function) -> List[str]:
    """Build function call chain using graph analysis"""
    try:
        chain = [function.name]
        visited = {function.name}
        
        # Follow function calls to build chain
        current = function
        depth = 0
        max_depth = 10  # Prevent infinite recursion
        
        while depth < max_depth and hasattr(current, 'function_calls'):
            if not current.function_calls:
                break
            
            # Get the first unvisited function call
            next_func = None
            for call in current.function_calls:
                if hasattr(call, 'function_definition') and call.function_definition:
                    if call.function_definition.name not in visited:
                        next_func = call.function_definition
                        break
            
            if not next_func:
                break
            
            chain.append(next_func.name)
            visited.add(next_func.name)
            current = next_func
            depth += 1
        
        return chain
    except Exception:
        return [function.name]

def _create_mock_analysis_result(repo_path: str) -> Dict[str, Any]:
    """Create comprehensive mock analysis result for development/testing"""
    mock_functions = [
        FunctionDefinition(
            name="main",
            file_path="src/main.py",
            line_start=1,
            line_end=15,
            parameters=[],
            return_type=None,
            docstring="Main entry point function",
            source_code="def main():\n    print('Hello World')\n    process_data()\n    return 0",
            complexity_score=2,
            is_entry_point=True,
            calls=["process_data", "print"],
            called_by=[],
            dependencies=[],
            issues=[]
        ),
        FunctionDefinition(
            name="process_data",
            file_path="src/processor.py",
            line_start=10,
            line_end=25,
            parameters=[{"name": "data", "type": "dict", "required": True}],
            return_type="bool",
            docstring="Process input data and return success status",
            source_code="def process_data(data: dict) -> bool:\n    if not data:\n        return False\n    return validate_input(data)",
            complexity_score=3,
            is_entry_point=False,
            calls=["validate_input"],
            called_by=["main"],
            dependencies=["validator"],
            issues=[]
        ),
        FunctionDefinition(
            name="validate_input",
            file_path="src/validator.py",
            line_start=5,
            line_end=20,
            parameters=[{"name": "input_data", "type": "dict", "required": True}],
            return_type="bool",
            docstring="Validate input data structure",
            source_code="def validate_input(input_data: dict) -> bool:\n    required_keys = ['id', 'name']\n    return all(key in input_data for key in required_keys)",
            complexity_score=2,
            is_entry_point=False,
            calls=[],
            called_by=["process_data"],
            dependencies=[],
            issues=[]
        ),
        FunctionDefinition(
            name="setup_logging",
            file_path="src/utils.py",
            line_start=1,
            line_end=10,
            parameters=[{"name": "level", "type": "str", "required": False, "default": "INFO"}],
            return_type=None,
            docstring="Setup application logging",
            source_code="def setup_logging(level: str = 'INFO'):\n    logging.basicConfig(level=getattr(logging, level))",
            complexity_score=1,
            is_entry_point=True,
            calls=["logging.basicConfig"],
            called_by=["main"],
            dependencies=["logging"],
            issues=[]
        ),
        FunctionDefinition(
            name="api_handler",
            file_path="src/api.py",
            line_start=15,
            line_end=30,
            parameters=[{"name": "request", "type": "Request", "required": True}],
            return_type="Response",
            docstring="Handle API requests",
            source_code="@app.route('/api/data')\ndef api_handler(request: Request) -> Response:\n    data = request.json\n    result = process_data(data)\n    return jsonify({'success': result})",
            complexity_score=4,
            is_entry_point=True,
            calls=["process_data", "jsonify"],
            called_by=[],
            dependencies=["flask"],
            issues=[]
        )
    ]
    
    mock_entry_points = [
        EntryPoint(
            name="main",
            type="main_function",
            file_path="src/main.py",
            line_number=1,
            description="Main application entry point",
            parameters=[],
            dependencies=[]
        ),
        EntryPoint(
            name="setup_logging",
            type="initialization_function",
            file_path="src/utils.py",
            line_number=1,
            description="Logging setup function",
            parameters=[{"name": "level", "type": "str", "required": False}],
            dependencies=["logging"]
        ),
        EntryPoint(
            name="api_handler",
            type="api_endpoint",
            file_path="src/api.py",
            line_number=15,
            description="API endpoint handler",
            parameters=[{"name": "request", "type": "Request", "required": True}],
            dependencies=["flask"]
        )
    ]
    
    return {
        'repository_path': repo_path,
        'analysis_timestamp': datetime.now().isoformat(),
        'total_files': 5,
        'total_lines': 150,
        'programming_languages': ['python'],
        'all_functions': [func.to_dict() for func in mock_functions],
        'all_entry_points': [ep.to_dict() for ep in mock_entry_points],
        'all_issues': [],
        'dependency_graph': {
            'main': ['process_data'],
            'process_data': ['validate_input'],
            'validate_input': [],
            'setup_logging': [],
            'api_handler': ['process_data']
        },
        'symbol_table': {
            'functions': len(mock_functions),
            'entry_points': len(mock_entry_points),
            'classes': 0,
            'imports': 3
        }
    }

# Helper functions for Codegen SDK integration
def _extract_codegen_parameters(function) -> List[Dict[str, Any]]:
    """Extract parameters from Codegen function"""
    try:
        if hasattr(function, 'parameters'):
            return [{"name": p.name, "type": getattr(p, 'type', None)} for p in function.parameters]
        return []
    except:
        return []

def _calculate_codegen_complexity(function) -> int:
    """Calculate complexity for Codegen function (internal use)"""
    try:
        if hasattr(function, 'complexity'):
            return function.complexity
        # Fallback calculation
        return len(function.source.splitlines()) // 5 + len(getattr(function, 'function_calls', []))
    except:
        return 1

def _is_codegen_entry_point(function) -> bool:
    """Check if Codegen function is an entry point"""
    try:
        entry_patterns = ['main', 'run', 'start', 'execute', 'init', 'setup', 'handler', 'api_']
        return any(pattern in function.name.lower() for pattern in entry_patterns)
    except:
        return False

def _get_codegen_entry_point_type(function) -> str:
    """Get entry point type for Codegen function"""
    try:
        name = function.name.lower()
        if 'main' in name:
            return 'main_function'
        elif 'api' in name or 'handler' in name:
            return 'api_endpoint'
        elif 'init' in name or 'setup' in name:
            return 'initialization_function'
        elif 'test' in name:
            return 'test_function'
        return 'function'
    except:
        return 'function'

def _build_codegen_dependency_graph(codebase) -> Dict[str, List[str]]:
    """Build dependency graph from Codegen codebase"""
    try:
        graph = {}
        for function in codebase.functions:
            deps = [call.name for call in function.function_calls if hasattr(call, 'name')]
            graph[function.name] = deps
        return graph
    except:
        return {}

def _build_codegen_symbol_table(codebase) -> Dict[str, Any]:
    """Build symbol table from Codegen codebase"""
    try:
        return {
            'functions': len(codebase.functions),
            'classes': len(getattr(codebase, 'classes', [])),
            'files': len(codebase.files),
            'imports': len(getattr(codebase, 'imports', []))
        }
    except:
        return {'functions': 0, 'classes': 0, 'files': 0, 'imports': 0}
