"""
Comprehensive Codebase Analysis Engine
Consolidated from all analysis systems with enhanced functionality
Includes: Backend Branch + Merge-Comprehensive-Analysis + Enhanced Analysis + Metrics
"""

import re
import hashlib
import math
import json
import time
import networkx as nx
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime

# Graph-sitter imports
from graph_sitter.core.codebase import Codebase
from graph_sitter.core.class_definition import Class
from graph_sitter.core.function import Function
from graph_sitter.core.file import SourceFile
from graph_sitter.core.expressions.binary_expression import BinaryExpression
from graph_sitter.core.expressions.unary_expression import UnaryExpression
from graph_sitter.core.expressions.comparison_expression import ComparisonExpression
from graph_sitter.enums import EdgeType, SymbolType

# Import enhanced data models
from .models import (
    CodeIssue, EntryPoint, CriticalFile, DependencyNode, IssueSeverity, IssueType,
    AutomatedResolution, HealthMetrics, FunctionContext, HalsteadMetrics,
    GraphMetrics, DeadCodeAnalysis, RepositoryStructure, AnalysisResults
)

# Conditional imports for advanced features
try:
    from graph_sitter.core.assignment import Assignment
    from graph_sitter.core.export import Export
    from graph_sitter.core.directory import Directory
    from graph_sitter.core.interface import Interface
    from graph_sitter.statements.for_loop_statement import ForLoopStatement
    from graph_sitter.core.statements.if_block_statement import IfBlockStatement
    from graph_sitter.core.statements.try_catch_statement import TryCatchStatement
    from graph_sitter.core.statements.while_statement import WhileStatement
    import graph_sitter.python as python_analyzer
    import graph_sitter.typescript as typescript_analyzer
except ImportError:
    Assignment = Export = Directory = Interface = python_analyzer = (
        typescript_analyzer
    ) = ForLoopStatement = IfBlockStatement = TryCatchStatement = WhileStatement = None


class CodeAnalysisError(Exception):
    """Custom exception for code analysis errors"""
    pass


<<<<<<< HEAD
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
                            id=f"null_ref_{file.path}_{i}",
                            type=IssueType.NULL_REFERENCE,
                            severity=IssueSeverity.HIGH,
                            file_path=file.path,
                            line_number=i + 1,
                            message="Potential null reference: .get() without null check",
                            description="Dictionary .get() call without null check or default value",
                            context={"line": line.strip()},
                            fix_suggestions=["Add null check or provide default value"]
                        )
                        
                        # Automated resolution
                        if '.get(' in line:
                            fixed_line = self._fix_null_reference(line)
                            issue.automated_resolution = AutomatedResolution(
                                issue_id=issue.id,
                                resolution_type="null_check_addition",
                                description="Add null check with default value",
                                fix_code=fixed_line,
                                confidence=0.85,
                                blast_radius={"affected_lines": [i + 1]},
                                validation_status="pending"
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
                        id=f"long_func_{function.filepath}_{function.name}",
                        type=IssueType.LONG_FUNCTION,
                        severity=IssueSeverity.HIGH,
                        file_path=function.filepath,
                        function_name=function.name,
                        line_number=function.start_point[0],
                        message=f"Function '{function.name}' is too long ({line_count} lines)",
                        description=f"Function exceeds recommended maximum of 50 lines",
                        context={"line_count": line_count},
                        fix_suggestions=["Break down into smaller functions"]
                    )
                    self.issues.append(issue)
            
            # Missing documentation
            if hasattr(function, 'source'):
                if not ('"""' in function.source or "'''" in function.source):
                    issue = CodeIssue(
                        id=f"no_doc_{function.filepath}_{function.name}",
                        type=IssueType.MISSING_DOCUMENTATION,
                        severity=IssueSeverity.LOW,
                        file_path=function.filepath,
                        function_name=function.name,
                        line_number=function.start_point[0] if hasattr(function, 'start_point') else 0,
                        message=f"Function '{function.name}' lacks documentation",
                        description="Function missing docstring",
                        fix_suggestions=["Add docstring explaining function purpose"]
                    )
                    
                    # Automated resolution - add basic docstring
                    docstring = self._generate_docstring(function)
                    issue.automated_resolution = AutomatedResolution(
                        issue_id=issue.id,
                        resolution_type="add_docstring",
                        description=f"Add docstring to function '{function.name}'",
                        fix_code=docstring,
                        confidence=0.80,
                        blast_radius={"affected_lines": [function.start_point[0] + 1] if hasattr(function, 'start_point') else [1]},
                        validation_status="pending"
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
                    numbers = re.findall(r'\b(?<![.\w])\d{2,}\b(?![.\w])', line)
                    for number in numbers:
                        if int(number) not in [0, 1, -1, 100]:  # Common acceptable numbers
                            issue = CodeIssue(
                                id=f"magic_{file.path}_{i}_{number}",
                                type=IssueType.MAGIC_NUMBER,
                                severity=IssueSeverity.LOW,
                                file_path=file.path,
                                line_number=i + 1,
                                column_number=line.find(number),
                                message=f"Magic number detected: {number}",
                                description=f"Numeric literal {number} should be replaced with named constant",
                                context={"number": number, "line": line.strip()},
                                fix_suggestions=[f"Replace {number} with named constant"]
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
                                id=f"div_zero_{file.path}_{i}",
                                type=IssueType.DIVISION_BY_ZERO,
                                severity=IssueSeverity.HIGH,
                                file_path=file.path,
                                line_number=i + 1,
                                column_number=line.find('/'),
                                message="Potential division by zero",
                                description="Division operation without zero check",
                                context={"line": line.strip()},
                                fix_suggestions=["Add zero check before division"]
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
                        id=f"dead_func_{function.filepath}_{function.name}",
                        type=IssueType.DEAD_FUNCTION,
                        severity=IssueSeverity.LOW,
                        file_path=function.filepath,
                        function_name=function.name,
                        line_number=function.start_point[0] if hasattr(function, 'start_point') else 0,
                        message=f"Unused function: {function.name}",
                        description="Function is defined but never called",
                        context={"reason": "No usages found"},
                        fix_suggestions=["Remove unused function", "Verify if function is needed"]
                    )
                    
                    # Automated resolution - mark for removal
                    issue.automated_resolution = AutomatedResolution(
                        issue_id=issue.id,
                        resolution_type="remove_dead_function",
                        description=f"Remove unused function '{function.name}'",
                        fix_code="",  # Empty means remove
                        confidence=0.75,
                        blast_radius={"affected_lines": list(range(
                            function.start_point[0] if hasattr(function, 'start_point') else 0,
                            function.end_point[0] if hasattr(function, 'end_point') else 0
                        ))},
                        validation_status="pending"
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
                        id=f"unused_import_{file.path}_{unused_import['name']}",
                        type=IssueType.DEAD_IMPORT,
                        severity=IssueSeverity.LOW,
                        file_path=file.path,
                        line_number=unused_import.get('line', 1),
                        message=f"Unused import: {unused_import['name']}",
                        description="Import statement is not used in the file",
                        context={"import_name": unused_import['name']},
                        fix_suggestions=["Remove unused import"]
                    )
                    
                    # Automated resolution - remove unused import
                    issue.automated_resolution = AutomatedResolution(
                        issue_id=issue.id,
                        resolution_type="remove_unused_import",
                        description=f"Remove unused import: {unused_import['name']}",
                        fix_code="",  # Remove the line
                        confidence=0.95,
                        blast_radius={"affected_lines": [unused_import.get('line', 1)]},
                        validation_status="pending"
                    )
                    
                    self.issues.append(issue)
                
                # Detect missing imports
                missing_imports = self.import_resolver.find_missing_imports(file)
                for missing_import in missing_imports:
                    issue = CodeIssue(
                        id=f"missing_import_{file.path}_{missing_import['symbol']}",
                        type=IssueType.UNDEFINED_VARIABLE,
                        severity=IssueSeverity.CRITICAL,
                        file_path=file.path,
                        line_number=missing_import.get('line', 1),
                        message=f"Missing import for: {missing_import['symbol']}",
                        description="Symbol used but not imported",
                        context={"symbol": missing_import['symbol']},
                        fix_suggestions=["Add missing import statement"]
                    )
                    
                    # Automated resolution - add missing import
                    resolved_import = self.import_resolver.resolve_import(missing_import['symbol'])
                    if resolved_import:
                        issue.automated_resolution = AutomatedResolution(
                            issue_id=issue.id,
                            resolution_type="add_missing_import",
                            description=f"Add import: {resolved_import}",
                            fix_code=resolved_import,
                            confidence=0.90,
                            blast_radius={"affected_lines": [1]},  # Add at top of file
                            validation_status="pending"
                        )
                    
                    self.issues.append(issue)
    
    # Placeholder methods for other detection types
    def _detect_type_mismatches(self): pass
    def _detect_undefined_variables(self): pass
    def _detect_missing_returns(self): pass
    def _detect_unreachable_code(self): pass
    def _detect_parameter_issues(self): pass
    def _detect_exception_handling_issues(self): pass
    def _detect_resource_leaks(self): pass
    def _detect_code_quality_issues(self): pass
    def _detect_style_issues(self): pass
    
    def _apply_automated_resolutions(self):
        """Apply automated resolutions where safe"""
        for issue in self.issues:
            if hasattr(issue, 'automated_resolution') and issue.automated_resolution:
                self.automated_resolutions.append(issue.automated_resolution)
=======
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


# Global cache instance
codebase_cache = CodebaseCache()
>>>>>>> e76cfb7 (Consolidate comprehensive functionality from all backend branches)


class ComprehensiveCodebaseAnalyzer:
    """
    Main analyzer class that orchestrates all analysis components
    Consolidated from all 4 systems with enhanced functionality
    """
    
    def __init__(self, codebase: Codebase):
        self.codebase = codebase
        self.cache = {}
        self.analysis_start_time = None
        
    def analyze_comprehensive(self, request_config: Dict[str, Any]) -> AnalysisResults:
        """
        Perform comprehensive codebase analysis
        """
        self.analysis_start_time = time.time()
        print("🚀 Starting comprehensive codebase analysis...")
        
        # Initialize results structure
        results = self._initialize_results()
        
        try:
            # Basic statistics
            if request_config.get('include_basic_stats', True):
                results = self._analyze_basic_statistics(results)
            
            # Issue detection
            if request_config.get('include_issues', True):
                results = self._analyze_issues(results, request_config)
            
            # Entry points detection
            if request_config.get('include_entry_points', True):
                results = self._analyze_entry_points(results)
            
            # Function context analysis
            if request_config.get('include_function_context', True):
                results = self._analyze_function_contexts(results)
            
            # Halstead metrics
            if request_config.get('include_halstead_metrics', True):
                results = self._analyze_halstead_metrics(results)
            
            # Graph analysis
            if request_config.get('include_graph_analysis', True):
                results = self._analyze_graphs(results)
            
            # Dead code analysis
            if request_config.get('include_dead_code_analysis', True):
                results = self._analyze_dead_code(results)
            
            # Health metrics
            if request_config.get('include_health_metrics', True):
                results = self._analyze_health_metrics(results)
            
            # Repository structure
            if request_config.get('include_repository_structure', True):
                results = self._analyze_repository_structure(results)
            
            # Automated resolutions
            if request_config.get('include_automated_resolutions', True):
                results = self._generate_automated_resolutions(results)
            
            # Finalize results
            results.analysis_duration = time.time() - self.analysis_start_time
            results.analysis_timestamp = datetime.now()
            
            print(f"✅ Analysis completed in {results.analysis_duration:.2f}s")
            return results
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            results.errors.append(f"Analysis failed: {str(e)}")
            results.analysis_duration = time.time() - self.analysis_start_time
            return results
    
    def _initialize_results(self) -> AnalysisResults:
        """Initialize empty results structure"""
        return AnalysisResults(
            total_files=0,
            total_functions=0,
            total_classes=0,
            total_lines_of_code=0,
            effective_lines_of_code=0,
            total_issues=0,
            issues_by_severity={},
            issues_by_type={},
            critical_issues=[],
            automated_resolutions=[],
            most_important_functions=[],
            entry_points=[],
            function_contexts={},
            halstead_metrics={},
            complexity_metrics={},
            maintainability_score=0.0,
            technical_debt_score=0.0,
            call_graph={},
            dependency_graph={},
            graph_metrics={},
            dead_code_analysis=DeadCodeAnalysis(),
            health_metrics=HealthMetrics(
                overall_score=0.0,
                maintainability_score=0.0,
                technical_debt_score=0.0,
                complexity_score=0.0
            ),
            repository_structure=RepositoryStructure(),
            analysis_timestamp=datetime.now(),
            analysis_duration=0.0,
            errors=[]
        )
    
    def _analyze_basic_statistics(self, results: AnalysisResults) -> AnalysisResults:
        """Analyze basic codebase statistics"""
        print("📊 Analyzing basic statistics...")
        
        total_files = len(self.codebase.files)
        total_functions = sum(len(file.functions) for file in self.codebase.files)
        total_classes = sum(len(file.classes) for file in self.codebase.files)
        total_lines = sum(len(file.content.split('\n')) for file in self.codebase.files)
        
        results.total_files = total_files
        results.total_functions = total_functions
        results.total_classes = total_classes
        results.total_lines_of_code = total_lines
        results.effective_lines_of_code = int(total_lines * 0.7)  # Estimate
        
        return results
    
    def _analyze_issues(self, results: AnalysisResults, config: Dict[str, Any]) -> AnalysisResults:
        """Comprehensive issue detection using AdvancedIssueDetector"""
        print("🔍 Detecting issues...")
        
        # Use the advanced issue detector
        advanced_detector = AdvancedIssueDetector(self.codebase)
        issues = advanced_detector.detect_all_issues()
        
        # Also run basic issue detection for additional coverage
        basic_issues = []
        for file in self.codebase.files:
            file_issues = self._detect_file_issues(file)
            basic_issues.extend(file_issues)
        
        # Combine issues and deduplicate
        all_issues = issues + basic_issues
        unique_issues = []
        seen_ids = set()
        
        for issue in all_issues:
            if issue.id not in seen_ids:
                unique_issues.append(issue)
                seen_ids.add(issue.id)
        
        issues = unique_issues
        
        # Categorize issues
        issues_by_severity = defaultdict(int)
        issues_by_type = defaultdict(int)
        
        for issue in issues:
            issues_by_severity[issue.severity.value] += 1
            issues_by_type[issue.type.value] += 1
        
        # Filter by severity if specified
        if config.get('severity_filter'):
            issues = [i for i in issues if i.severity in config['severity_filter']]
        
        # Limit number of issues
        max_issues = config.get('max_issues', 200)
        if len(issues) > max_issues:
            issues = sorted(issues, key=lambda x: (x.severity.value, x.type.value))[:max_issues]
        
        results.total_issues = len(issues)
        results.issues_by_severity = dict(issues_by_severity)
        results.issues_by_type = dict(issues_by_type)
        results.critical_issues = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
        
        # Store automated resolutions
        results.automated_resolutions = advanced_detector.automated_resolutions
        
        return results
    
    def _detect_file_issues(self, file: SourceFile) -> List[CodeIssue]:
        """Detect issues in a single file"""
        issues = []
        
        # Syntax and structural issues
        issues.extend(self._detect_syntax_issues(file))
        issues.extend(self._detect_complexity_issues(file))
        issues.extend(self._detect_maintainability_issues(file))
        
        return issues
    
    def _detect_syntax_issues(self, file: SourceFile) -> List[CodeIssue]:
        """Detect syntax-related issues"""
        issues = []
        
        # Check for common syntax patterns
        content = file.content
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Long lines
            if len(line) > 120:
                issues.append(CodeIssue(
                    id=f"long_line_{file.path}_{i}",
                    type=IssueType.CODE_SMELL,
                    severity=IssueSeverity.LOW,
                    file_path=file.path,
                    line_number=i,
                    message="Line too long",
                    description=f"Line {i} has {len(line)} characters (>120)",
                    fix_suggestions=["Break line into multiple lines", "Refactor complex expression"]
                ))
            
            # TODO comments
            if 'TODO' in line.upper() or 'FIXME' in line.upper():
                issues.append(CodeIssue(
                    id=f"todo_{file.path}_{i}",
                    type=IssueType.MAINTAINABILITY_ISSUE,
                    severity=IssueSeverity.INFO,
                    file_path=file.path,
                    line_number=i,
                    message="TODO/FIXME comment found",
                    description="Unresolved TODO or FIXME comment",
                    fix_suggestions=["Resolve the TODO item", "Create a proper issue tracker item"]
                ))
        
        return issues
    
    def _detect_complexity_issues(self, file: SourceFile) -> List[CodeIssue]:
        """Detect complexity-related issues"""
        issues = []
        
        for func in file.functions:
            complexity = self._calculate_cyclomatic_complexity(func)
            
            if complexity > 10:
                severity = IssueSeverity.HIGH if complexity > 20 else IssueSeverity.MEDIUM
                issues.append(CodeIssue(
                    id=f"complexity_{file.path}_{func.name}",
                    type=IssueType.COMPLEXITY_ISSUE,
                    severity=severity,
                    file_path=file.path,
                    function_name=func.name,
                    line_number=getattr(func, 'line_number', None),
                    message=f"High cyclomatic complexity: {complexity}",
                    description=f"Function '{func.name}' has cyclomatic complexity of {complexity}",
                    fix_suggestions=[
                        "Break function into smaller functions",
                        "Reduce conditional complexity",
                        "Extract complex logic into separate methods"
                    ]
                ))
        
        return issues
    
    def _detect_maintainability_issues(self, file: SourceFile) -> List[CodeIssue]:
        """Detect maintainability-related issues"""
        issues = []
        
        # Large files
        lines_count = len(file.content.split('\n'))
        if lines_count > 500:
            issues.append(CodeIssue(
                id=f"large_file_{file.path}",
                type=IssueType.MAINTAINABILITY_ISSUE,
                severity=IssueSeverity.MEDIUM,
                file_path=file.path,
                message=f"Large file: {lines_count} lines",
                description=f"File has {lines_count} lines, consider splitting",
                fix_suggestions=[
                    "Split file into smaller modules",
                    "Extract related functionality into separate files"
                ]
            ))
        
        # Too many functions in a file
        if len(file.functions) > 20:
            issues.append(CodeIssue(
                id=f"many_functions_{file.path}",
                type=IssueType.MAINTAINABILITY_ISSUE,
                severity=IssueSeverity.LOW,
                file_path=file.path,
                message=f"Many functions: {len(file.functions)}",
                description=f"File has {len(file.functions)} functions, consider refactoring",
                fix_suggestions=[
                    "Group related functions into classes",
                    "Split functionality across multiple files"
                ]
            ))
        
        return issues
    
    def _calculate_cyclomatic_complexity(self, func: Function) -> int:
        """Calculate cyclomatic complexity for a function"""
        # Simplified complexity calculation
        complexity = 1  # Base complexity
        
        # Count decision points in function content
        if hasattr(func, 'content'):
            content = func.content.lower()
            
            # Count control flow statements
            complexity += content.count('if ')
            complexity += content.count('elif ')
            complexity += content.count('while ')
            complexity += content.count('for ')
            complexity += content.count('except ')
            complexity += content.count('case ')
            complexity += content.count('&&')
            complexity += content.count('||')
            complexity += content.count('and ')
            complexity += content.count('or ')
        
        return complexity
    
    def _analyze_entry_points(self, results: AnalysisResults) -> AnalysisResults:
        """Analyze entry points"""
        print("🚪 Detecting entry points...")
        results.entry_points = [ep.__dict__ for ep in detect_entry_points(self.codebase)]
        return results
    
    def _analyze_function_contexts(self, results: AnalysisResults) -> AnalysisResults:
        """Analyze function contexts"""
        print("🔗 Analyzing function contexts...")
        # Simplified function context analysis
        for file in self.codebase.files:
            for func in file.functions:
                context = FunctionContext(
                    function_name=func.name,
                    file_path=file.path,
                    signature=f"def {func.name}(...)",
                    line_number=getattr(func, 'line_number', None)
                )
                results.function_contexts[f"{file.path}::{func.name}"] = context
        return results
    
    def _analyze_halstead_metrics(self, results: AnalysisResults) -> AnalysisResults:
        """Analyze comprehensive Halstead metrics"""
        print("📏 Calculating Halstead metrics...")
        
        # Comprehensive operator and operand patterns
        operator_patterns = [
            r'\+', r'-', r'\*', r'/', r'%', r'\*\*',  # Arithmetic
            r'==', r'!=', r'<', r'>', r'<=', r'>=',   # Comparison
            r'and', r'or', r'not',                    # Logical
            r'&', r'\|', r'\^', r'~', r'<<', r'>>',   # Bitwise
            r'=', r'\+=', r'-=', r'\*=', r'/=',       # Assignment
            r'if', r'else', r'elif', r'while', r'for', r'try', r'except', r'finally',  # Control
            r'def', r'class', r'return', r'yield', r'import', r'from', r'as',  # Keywords
            r'\.', r'\[', r'\]', r'\(', r'\)', r'\{', r'\}', r',', r';', r':'  # Delimiters
        ]
        
        operand_patterns = [
            r'\b[a-zA-Z_][a-zA-Z0-9_]*\b',  # Identifiers
            r'\b\d+\.?\d*\b',               # Numbers
            r'["\'][^"\']*["\']',           # Strings
        ]
        
        total_operators = {}
        total_operands = {}
        file_metrics = {}
        
        for file in self.codebase.files:
            if not hasattr(file, 'content') or not file.content:
                continue
                
            content = file.content
            file_operators = {}
            file_operands = {}
            
            # Count operators
            for pattern in operator_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    file_operators[match] = file_operators.get(match, 0) + 1
                    total_operators[match] = total_operators.get(match, 0) + 1
            
            # Count operands
            for pattern in operand_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    # Filter out keywords and operators
                    if match not in ['if', 'else', 'elif', 'while', 'for', 'def', 'class', 'return', 'import', 'from', 'as', 'try', 'except', 'finally', 'and', 'or', 'not']:
                        file_operands[match] = file_operands.get(match, 0) + 1
                        total_operands[match] = total_operands.get(match, 0) + 1
            
            # Calculate file-level metrics
            n1 = len(file_operators)  # Unique operators
            n2 = len(file_operands)   # Unique operands
            N1 = sum(file_operators.values())  # Total operators
            N2 = sum(file_operands.values())   # Total operands
            
            if n1 > 0 and n2 > 0:
                vocabulary = n1 + n2
                length = N1 + N2
                volume = length * math.log2(vocabulary) if vocabulary > 0 else 0
                difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
                effort = difficulty * volume
                time_to_program = effort / 18  # Stroud number
                bugs_delivered = volume / 3000  # Empirical constant
                
                file_metrics[file.path] = {
                    'unique_operators': n1,
                    'unique_operands': n2,
                    'total_operators': N1,
                    'total_operands': N2,
                    'vocabulary': vocabulary,
                    'length': length,
                    'volume': volume,
                    'difficulty': difficulty,
                    'effort': effort,
                    'time_to_program': time_to_program,
                    'bugs_delivered': bugs_delivered
                }
        
        # Calculate overall metrics
        n1_total = len(total_operators)
        n2_total = len(total_operands)
        N1_total = sum(total_operators.values())
        N2_total = sum(total_operands.values())
        
        if n1_total > 0 and n2_total > 0:
            vocabulary_total = n1_total + n2_total
            length_total = N1_total + N2_total
            volume_total = length_total * math.log2(vocabulary_total) if vocabulary_total > 0 else 0
            difficulty_total = (n1_total / 2) * (N2_total / n2_total) if n2_total > 0 else 0
            effort_total = difficulty_total * volume_total
            time_total = effort_total / 18
            bugs_total = volume_total / 3000
            
            results.halstead_metrics = {
                'operators': dict(total_operators),
                'operands': dict(total_operands),
                'unique_operators': n1_total,
                'unique_operands': n2_total,
                'total_operators': N1_total,
                'total_operands': N2_total,
                'vocabulary': vocabulary_total,
                'length': length_total,
                'volume': volume_total,
                'difficulty': difficulty_total,
                'effort': effort_total,
                'time_to_program': time_total,
                'bugs_delivered': bugs_total,
                'file_metrics': file_metrics
            }
        else:
            results.halstead_metrics = {
                'operators': {},
                'operands': {},
                'unique_operators': 0,
                'unique_operands': 0,
                'total_operators': 0,
                'total_operands': 0,
                'vocabulary': 0,
                'length': 0,
                'volume': 0,
                'difficulty': 0,
                'effort': 0,
                'time_to_program': 0,
                'bugs_delivered': 0,
                'file_metrics': {}
            }
        
        return results
    
    def _analyze_graphs(self, results: AnalysisResults) -> AnalysisResults:
        """Analyze call and dependency graphs with comprehensive NetworkX metrics"""
        print("🕸️ Building graphs...")
        
        # Build comprehensive call graph
        call_graph = nx.DiGraph()
        dependency_graph = nx.DiGraph()
        
        # Build call graph with function relationships
        function_map = {}
        for file in self.codebase.files:
            for func in file.functions:
                node_id = f"{file.path}::{func.name}"
                function_map[func.name] = node_id
                call_graph.add_node(node_id, 
                    file=file.path, 
                    function=func.name,
                    complexity=self._calculate_cyclomatic_complexity(func),
                    lines=len(func.source.split('\n')) if hasattr(func, 'source') and func.source else 0
                )
        
        # Add edges based on function calls
        for file in self.codebase.files:
            for func in file.functions:
                caller_id = f"{file.path}::{func.name}"
                if hasattr(func, 'function_calls'):
                    for call in func.function_calls:
                        if call.name in function_map:
                            callee_id = function_map[call.name]
                            call_graph.add_edge(caller_id, callee_id, weight=1)
        
        # Build dependency graph with file relationships
        for file in self.codebase.files:
            dependency_graph.add_node(file.path, 
                lines=len(file.content.split('\n')) if hasattr(file, 'content') and file.content else 0,
                functions=len(file.functions),
                classes=len(file.classes)
            )
        
        # Add dependency edges based on imports
        for file in self.codebase.files:
            if hasattr(file, 'imports'):
                for imp in file.imports:
                    if hasattr(imp, 'imported_symbol') and hasattr(imp.imported_symbol, 'filepath'):
                        target_file = imp.imported_symbol.filepath
                        if target_file != file.path:
                            dependency_graph.add_edge(file.path, target_file, type='import')
        
        # Calculate comprehensive graph metrics
        graph_metrics = {}
        
        # Call graph metrics
        if call_graph.number_of_nodes() > 0:
            try:
                # Centrality measures
                betweenness = nx.betweenness_centrality(call_graph)
                closeness = nx.closeness_centrality(call_graph)
                degree_centrality = nx.degree_centrality(call_graph)
                
                # PageRank
                pagerank = nx.pagerank(call_graph)
                
                # Clustering
                clustering = nx.clustering(call_graph.to_undirected())
                
                # Find most important functions
                important_functions = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
                
                graph_metrics['call_graph'] = {
                    'nodes': call_graph.number_of_nodes(),
                    'edges': call_graph.number_of_edges(),
                    'density': nx.density(call_graph),
                    'strongly_connected_components': len(list(nx.strongly_connected_components(call_graph))),
                    'weakly_connected_components': len(list(nx.weakly_connected_components(call_graph))),
                    'average_clustering': sum(clustering.values()) / len(clustering) if clustering else 0,
                    'most_important_functions': [{'function': func, 'score': score} for func, score in important_functions],
                    'centrality_metrics': {
                        'highest_betweenness': max(betweenness.items(), key=lambda x: x[1]) if betweenness else None,
                        'highest_closeness': max(closeness.items(), key=lambda x: x[1]) if closeness else None,
                        'highest_degree': max(degree_centrality.items(), key=lambda x: x[1]) if degree_centrality else None
                    }
                }
                
                # Store detailed metrics for each function
                function_graph_metrics = {}
                for node in call_graph.nodes():
                    function_graph_metrics[node] = {
                        'betweenness_centrality': betweenness.get(node, 0),
                        'closeness_centrality': closeness.get(node, 0),
                        'degree_centrality': degree_centrality.get(node, 0),
                        'pagerank': pagerank.get(node, 0),
                        'clustering_coefficient': clustering.get(node, 0),
                        'in_degree': call_graph.in_degree(node),
                        'out_degree': call_graph.out_degree(node)
                    }
                
                results.graph_metrics = function_graph_metrics
                
            except Exception as e:
                print(f"Warning: Could not calculate some graph metrics: {e}")
                graph_metrics['call_graph'] = {
                    'nodes': call_graph.number_of_nodes(),
                    'edges': call_graph.number_of_edges(),
                    'density': nx.density(call_graph),
                    'error': str(e)
                }
        else:
            graph_metrics['call_graph'] = {
                'nodes': 0,
                'edges': 0,
                'density': 0,
                'message': 'No functions found for call graph analysis'
            }
        
        # Dependency graph metrics
        if dependency_graph.number_of_nodes() > 0:
            try:
                dep_pagerank = nx.pagerank(dependency_graph)
                dep_clustering = nx.clustering(dependency_graph.to_undirected())
                
                # Find most critical files
                critical_files = sorted(dep_pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
                
                graph_metrics['dependency_graph'] = {
                    'nodes': dependency_graph.number_of_nodes(),
                    'edges': dependency_graph.number_of_edges(),
                    'density': nx.density(dependency_graph),
                    'strongly_connected_components': len(list(nx.strongly_connected_components(dependency_graph))),
                    'average_clustering': sum(dep_clustering.values()) / len(dep_clustering) if dep_clustering else 0,
                    'most_critical_files': [{'file': file, 'score': score} for file, score in critical_files]
                }
            except Exception as e:
                print(f"Warning: Could not calculate dependency graph metrics: {e}")
                graph_metrics['dependency_graph'] = {
                    'nodes': dependency_graph.number_of_nodes(),
                    'edges': dependency_graph.number_of_edges(),
                    'error': str(e)
                }
        else:
            graph_metrics['dependency_graph'] = {
                'nodes': 0,
                'edges': 0,
                'message': 'No files found for dependency graph analysis'
            }
        
        # Store graph data
        results.call_graph = {
            'nodes': [{'id': node, **call_graph.nodes[node]} for node in call_graph.nodes()],
            'edges': [{'source': edge[0], 'target': edge[1], **call_graph.edges[edge]} for edge in call_graph.edges()],
            'metrics': graph_metrics['call_graph']
        }
        
        results.dependency_graph = {
            'nodes': [{'id': node, **dependency_graph.nodes[node]} for node in dependency_graph.nodes()],
            'edges': [{'source': edge[0], 'target': edge[1], **dependency_graph.edges[edge]} for edge in dependency_graph.edges()],
            'metrics': graph_metrics['dependency_graph']
        }
        
        return results
    
    def _analyze_dead_code(self, results: AnalysisResults) -> AnalysisResults:
        """Analyze dead code"""
        print("💀 Detecting dead code...")
        
        # Simplified dead code detection
        all_functions = []
        called_functions = set()
        
        for file in self.codebase.files:
            for func in file.functions:
                all_functions.append(func.name)
                
                # Look for function calls in content
                if hasattr(func, 'content'):
                    for other_func in all_functions:
                        if other_func in func.content and other_func != func.name:
                            called_functions.add(other_func)
        
        unused_functions = [f for f in all_functions if f not in called_functions and f != 'main']
        
        results.dead_code_analysis = DeadCodeAnalysis(
            unused_functions=unused_functions,
            dead_code_percentage=len(unused_functions) / max(1, len(all_functions)) * 100
        )
        
        return results
    
    def _analyze_health_metrics(self, results: AnalysisResults) -> AnalysisResults:
        """Analyze comprehensive health metrics"""
        print("🏥 Calculating health metrics...")
        
        # Calculate weighted issue scores
        issue_weights = {
            IssueSeverity.CRITICAL: 20,
            IssueSeverity.HIGH: 10,
            IssueSeverity.MEDIUM: 5,
            IssueSeverity.LOW: 2,
            IssueSeverity.INFO: 1
        }
        
        weighted_issue_score = 0
        for severity, count in results.issues_by_severity.items():
            severity_enum = IssueSeverity(severity)
            weighted_issue_score += count * issue_weights.get(severity_enum, 1)
        
        # Calculate complexity score
        avg_complexity = 0
        if results.function_contexts:
            complexities = [
                ctx.complexity_metrics.get('cyclomatic_complexity', 0) 
                for ctx in results.function_contexts.values()
            ]
            avg_complexity = sum(complexities) / len(complexities) if complexities else 0
        
        complexity_score = max(0, 100 - (avg_complexity - 5) * 10)  # Penalty for complexity > 5
        
        # Calculate maintainability score
        maintainability_factors = {
            'documentation_coverage': self._calculate_documentation_coverage(results),
            'dead_code_ratio': len(results.dead_code_analysis.unused_functions) / max(1, results.total_functions) * 100,
            'average_function_length': self._calculate_average_function_length(results),
            'test_coverage': 0  # Placeholder - would need test detection
        }
        
        # Documentation score (0-100)
        doc_score = maintainability_factors['documentation_coverage']
        
        # Dead code penalty
        dead_code_penalty = maintainability_factors['dead_code_ratio'] * 2
        
        # Function length penalty
        avg_func_length = maintainability_factors['average_function_length']
        length_penalty = max(0, (avg_func_length - 20) * 2)  # Penalty for functions > 20 lines
        
        maintainability_score = max(0, 100 - dead_code_penalty - length_penalty + (doc_score * 0.3))
        
        # Calculate technical debt score
        technical_debt_hours = 0
        for severity, count in results.issues_by_severity.items():
            if severity == IssueSeverity.CRITICAL.value:
                technical_debt_hours += count * 8  # 8 hours per critical issue
            elif severity == IssueSeverity.HIGH.value:
                technical_debt_hours += count * 4  # 4 hours per high issue
            elif severity == IssueSeverity.MEDIUM.value:
                technical_debt_hours += count * 2  # 2 hours per medium issue
            elif severity == IssueSeverity.LOW.value:
                technical_debt_hours += count * 1  # 1 hour per low issue
            else:
                technical_debt_hours += count * 0.5  # 0.5 hours per info issue
        
        technical_debt_score = min(100, technical_debt_hours / max(1, results.total_functions) * 10)
        
        # Calculate overall health score
        overall_score = (
            complexity_score * 0.3 +
            maintainability_score * 0.3 +
            (100 - technical_debt_score) * 0.4
        )
        
        # Determine health grade
        if overall_score >= 90:
            grade = "A"
            risk_level = "low"
        elif overall_score >= 80:
            grade = "B"
            risk_level = "low"
        elif overall_score >= 70:
            grade = "C"
            risk_level = "medium"
        elif overall_score >= 60:
            grade = "D"
            risk_level = "medium"
        else:
            grade = "F"
            risk_level = "high"
        
        # Determine trend direction (simplified)
        trend_direction = "neutral"  # Would need historical data for real trend analysis
        if overall_score >= 85:
            trend_direction = "improving"
        elif overall_score <= 50:
            trend_direction = "declining"
        
        # Calculate test coverage (placeholder)
        test_coverage_score = 0  # Would need actual test detection
        
        results.health_metrics = HealthMetrics(
            overall_score=overall_score,
            maintainability_score=maintainability_score,
            technical_debt_score=technical_debt_score,
            complexity_score=complexity_score,
            test_coverage_score=test_coverage_score,
            documentation_score=doc_score,
            trend_direction=trend_direction,
            risk_level=risk_level
        )
        
        # Store technical debt hours in results
        results.technical_debt_score = technical_debt_hours
        
        return results
    
    def _calculate_documentation_coverage(self, results: AnalysisResults) -> float:
        """Calculate documentation coverage percentage"""
        if not results.function_contexts:
            return 0.0
        
        documented_functions = 0
        for context in results.function_contexts.values():
            if hasattr(context, 'source') and ('"""' in context.source or "'''" in context.source):
                documented_functions += 1
        
        return (documented_functions / len(results.function_contexts)) * 100
    
    def _calculate_average_function_length(self, results: AnalysisResults) -> float:
        """Calculate average function length in lines"""
        if not results.function_contexts:
            return 0.0
        
        total_lines = 0
        for context in results.function_contexts.values():
            if hasattr(context, 'line_end') and hasattr(context, 'line_start'):
                total_lines += context.line_end - context.line_start
        
        return total_lines / len(results.function_contexts)
    
    def _analyze_repository_structure(self, results: AnalysisResults) -> AnalysisResults:
        """Analyze repository structure"""
        print("🏗️ Analyzing repository structure...")
        
        files_by_type = defaultdict(int)
        largest_files = []
        
        for file in self.codebase.files:
            # Get file extension
            ext = Path(file.path).suffix or 'no_extension'
            files_by_type[ext] += 1
            
            # Track largest files
            lines_count = len(file.content.split('\n'))
            largest_files.append({
                'path': file.path,
                'lines': lines_count,
                'functions': len(file.functions),
                'classes': len(file.classes)
            })
        
        # Sort and limit largest files
        largest_files = sorted(largest_files, key=lambda x: x['lines'], reverse=True)[:10]
        
        results.repository_structure = RepositoryStructure(
            total_files=len(self.codebase.files),
            files_by_type=dict(files_by_type),
            largest_files=largest_files,
            architectural_patterns=['MVC', 'Layered']  # Placeholder
        )
        
        return results
    
    def _generate_automated_resolutions(self, results: AnalysisResults) -> AnalysisResults:
        """Generate automated resolutions for issues"""
        print("🔧 Generating automated resolutions...")
        
        resolutions = []
        
        for issue in results.critical_issues[:10]:  # Limit to top 10 critical issues
            if issue.type == IssueType.CODE_SMELL and 'long' in issue.message.lower():
                resolutions.append(AutomatedResolution(
                    issue_id=issue.id,
                    resolution_type="refactor",
                    confidence=0.7,
                    description="Break long line into multiple lines",
                    fix_code="# Suggested line break at appropriate points"
                ))
            elif issue.type == IssueType.COMPLEXITY_ISSUE:
                resolutions.append(AutomatedResolution(
                    issue_id=issue.id,
                    resolution_type="refactor",
                    confidence=0.6,
                    description="Extract complex logic into separate functions",
                    fix_code="# Extract method refactoring suggested"
                ))
        
        results.automated_resolutions = resolutions
        return results


def analyze_codebase_comprehensive(codebase: Codebase, config: Dict[str, Any]) -> AnalysisResults:
    """
    Main entry point for comprehensive codebase analysis
    """
    analyzer = ComprehensiveCodebaseAnalyzer(codebase)
    return analyzer.analyze_comprehensive(config)


# Additional utility functions for specific analysis types
def detect_entry_points(codebase: Codebase) -> List[EntryPoint]:
    """Detect entry points in the codebase"""
    entry_points = []
    
    for file in codebase.files:
        # Main function detection
        for func in file.functions:
            if func.name == 'main':
                entry_points.append(EntryPoint(
                    type="main",
                    file_path=file.path,
                    function_name=func.name,
                    line_number=getattr(func, 'line_number', None),
                    description="Main function entry point",
                    confidence=0.9
                ))
        
        # Script detection (if __name__ == "__main__")
        if '__name__' in file.content and '__main__' in file.content:
            entry_points.append(EntryPoint(
                type="script",
                file_path=file.path,
                description="Python script entry point",
                confidence=0.8
            ))
    
    return entry_points


def identify_critical_files(codebase: Codebase) -> List[CriticalFile]:
    """Identify critical files in the codebase"""
    critical_files = []
    
    for file in codebase.files:
        importance_score = 0.0
        reasons = []
        
        # High function count
        if len(file.functions) > 10:
            importance_score += 20
            reasons.append(f"High function count: {len(file.functions)}")
        
        # High class count
        if len(file.classes) > 5:
            importance_score += 15
            reasons.append(f"High class count: {len(file.classes)}")
        
        # Large file size
        lines_count = len(file.content.split('\n'))
        if lines_count > 200:
            importance_score += 10
            reasons.append(f"Large file: {lines_count} lines")
        
        # Central imports (simplified)
        import_count = file.content.count('import ')
        if import_count > 10:
            importance_score += 10
            reasons.append(f"Many imports: {import_count}")
        
        if importance_score > 30:
            critical_files.append(CriticalFile(
                file_path=file.path,
                importance_score=min(importance_score, 100.0),
                reasons=reasons,
                lines_of_code=lines_count,
                metrics={
                    'functions_count': len(file.functions),
                    'classes_count': len(file.classes),
                    'imports_count': import_count
                }
            ))
    
    return sorted(critical_files, key=lambda x: x.importance_score, reverse=True)


# ============================================================================
# COMPREHENSIVE METRICS AND ANALYSIS FUNCTIONS
# ============================================================================

def get_codebase_summary(codebase: Codebase) -> str:
    """Generate comprehensive codebase summary"""
    try:
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
    except Exception as e:
        return f"Error generating codebase summary: {str(e)}"


def calculate_cyclomatic_complexity(func: Function) -> int:
    """Calculate cyclomatic complexity for a function"""
    try:
        complexity = 1  # Base complexity
        
        if hasattr(func, 'content') and func.content:
            content = func.content.lower()
            
            # Count decision points
            complexity += content.count('if ')
            complexity += content.count('elif ')
            complexity += content.count('while ')
            complexity += content.count('for ')
            complexity += content.count('except ')
            complexity += content.count('case ')
            complexity += content.count('&&')
            complexity += content.count('||')
            complexity += content.count('and ')
            complexity += content.count('or ')
        
        return complexity
    except Exception:
        return 1


def calculate_halstead_metrics(func: Function) -> Dict[str, float]:
    """Calculate Halstead complexity metrics"""
    try:
        if not hasattr(func, 'content') or not func.content:
            return {'volume': 0, 'difficulty': 0, 'effort': 0}
        
        source = func.content
        operators = re.findall(r'[+\-*/=<>!&|^%]', source)
        operands = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', source)
        
        n1 = len(set(operators))  # Unique operators
        n2 = len(set(operands))   # Unique operands
        N1 = len(operators)       # Total operators
        N2 = len(operands)        # Total operands
        
        if n1 == 0 or n2 == 0:
            return {'volume': 0, 'difficulty': 0, 'effort': 0}
        
        vocabulary = n1 + n2
        length = N1 + N2
        volume = length * math.log2(vocabulary) if vocabulary > 0 else 0
        difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
        effort = difficulty * volume
        
        return {
            'volume': volume,
            'difficulty': difficulty,
            'effort': effort,
            'vocabulary': vocabulary,
            'length': length
        }
    except Exception:
        return {'volume': 0, 'difficulty': 0, 'effort': 0}


def calculate_doi(cls: Class) -> int:
    """Calculate Depth of Inheritance for a class"""
    try:
        if not hasattr(cls, 'parent_class_names') or not cls.parent_class_names:
            return 0
        return len(cls.parent_class_names)
    except Exception:
        return 0


def calculate_cbo(cls: Class) -> int:
    """Calculate Coupling Between Objects for a class"""
    try:
        coupling = 0
        if hasattr(cls, 'dependencies'):
            coupling += len(cls.dependencies)
        if hasattr(cls, 'methods'):
            for method in cls.methods:
                if hasattr(method, 'dependencies'):
                    coupling += len(method.dependencies)
        return coupling
    except Exception:
        return 0


def calculate_lcom(cls: Class) -> float:
    """Calculate Lack of Cohesion of Methods for a class"""
    try:
        if not hasattr(cls, 'methods') or len(cls.methods) <= 1:
            return 0.0
        
        methods = cls.methods
        attributes = getattr(cls, 'attributes', [])
        
        if not attributes:
            return 0.0
        
        # Simplified LCOM calculation
        method_attribute_usage = {}
        for method in methods:
            used_attributes = []
            if hasattr(method, 'content'):
                for attr in attributes:
                    if attr.name in method.content:
                        used_attributes.append(attr.name)
            method_attribute_usage[method.name] = used_attributes
        
        # Calculate cohesion
        total_pairs = len(methods) * (len(methods) - 1) / 2
        cohesive_pairs = 0
        
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                attrs1 = set(method_attribute_usage.get(method1.name, []))
                attrs2 = set(method_attribute_usage.get(method2.name, []))
                if attrs1.intersection(attrs2):
                    cohesive_pairs += 1
        
        return 1 - (cohesive_pairs / total_pairs) if total_pairs > 0 else 0.0
    except Exception:
        return 0.0


def generate_repository_analysis_report(codebase: Codebase, repo_url: str = "") -> str:
    """Generate comprehensive repository analysis report"""
    try:
        repo_name = repo_url.split('/')[-1] if repo_url else "Unknown Repository"
        
        # Calculate basic statistics
        files = list(codebase.files)
        functions = list(codebase.functions)
        classes = list(codebase.classes)
        
        total_files = len(files)
        total_functions = len(functions)
        total_classes = len(classes)
        
        # Analyze issues (simplified)
        issues = []
        for file in files:
            for func in file.functions:
                complexity = calculate_cyclomatic_complexity(func)
                if complexity > 10:
                    issues.append({
                        'type': 'high_complexity',
                        'severity': 'medium',
                        'file_path': file.filepath,
                        'function': func.name,
                        'complexity': complexity
                    })
        
        critical_issues = [issue for issue in issues if issue.get('severity') == 'critical']
        
        # Build the report
        report = f"""📊 Repository Analysis Report 📊
==================================================
📁 Repository: {repo_name}
📝 Description: Advanced codebase analysis with graph-sitter integration

📁 Files: {total_files}
🔄 Functions: {total_functions}
📏 Classes: {total_classes}

🔍 **ISSUES DETECTED** ({len(issues)} issues found)
- Critical: {len(critical_issues)}
- High Complexity Functions: {len([i for i in issues if i['type'] == 'high_complexity'])}

📈 **GRAPH-SITTER INTEGRATION INSIGHTS**
- Pre-computed dependency graph with {len(list(codebase.ctx.edges))} edges
- Symbol usage analysis across {total_files} files
- Multi-language support: Python, TypeScript, React & JSX
- Advanced static analysis for code manipulation operations

🔧 **RECOMMENDED ACTIONS**
- Review high complexity functions for refactoring opportunities
- Consider breaking down large functions into smaller, more manageable pieces
- Implement comprehensive testing for critical components
"""
        
        return report
        
    except Exception as e:
        return f"Error generating repository analysis report: {str(e)}"


def calculate_comprehensive_metrics(codebase: Codebase) -> Dict[str, Any]:
    """Calculate comprehensive metrics for functions, classes, and files"""
    function_metrics = []
    class_metrics = []
    file_metrics = []
    
    try:
        for file in codebase.files:
            # File metrics
            file_loc = len(file.content.split('\n')) if hasattr(file, 'content') and file.content else 0
            file_metrics.append({
                'file_path': file.filepath,
                'lines_of_code': file_loc,
                'functions_count': len(file.functions),
                'classes_count': len(file.classes),
                'imports_count': len(file.imports),
                'complexity_score': sum(calculate_cyclomatic_complexity(f) for f in file.functions) / max(len(file.functions), 1),
                'maintainability_index': calculate_maintainability_index(file),
                'importance_score': calculate_file_importance(file, codebase)
            })
            
            # Function metrics
            for func in file.functions:
                try:
                    complexity = calculate_cyclomatic_complexity(func)
                    halstead_metrics = calculate_halstead_metrics(func)
                    
                    function_metrics.append({
                        'function_name': func.name,
                        'file_path': file.filepath,
                        'line_number': getattr(func, 'line_number', None),
                        'cyclomatic_complexity': complexity,
                        'maintainability_index': calculate_function_maintainability(func),
                        'lines_of_code': len(func.content.split('\n')) if hasattr(func, 'content') and func.content else 0,
                        'halstead_volume': halstead_metrics.get('volume', 0.0),
                        'halstead_difficulty': halstead_metrics.get('difficulty', 0.0),
                        'halstead_effort': halstead_metrics.get('effort', 0.0),
                        'parameters_count': len(func.parameters) if hasattr(func, 'parameters') else 0,
                        'importance_score': calculate_function_importance(func, codebase)
                    })
                except Exception as e:
                    print(f"Error calculating metrics for function {func.name}: {e}")
                    continue
            
            # Class metrics
            for cls in file.classes:
                try:
                    class_metrics.append({
                        'class_name': cls.name,
                        'file_path': file.filepath,
                        'line_number': getattr(cls, 'line_number', None),
                        'methods_count': len(cls.methods) if hasattr(cls, 'methods') else 0,
                        'attributes_count': len(cls.attributes) if hasattr(cls, 'attributes') else 0,
                        'depth_of_inheritance': calculate_doi(cls),
                        'coupling_between_objects': calculate_cbo(cls),
                        'lack_of_cohesion': calculate_lcom(cls),
                        'lines_of_code': len(cls.content.split('\n')) if hasattr(cls, 'content') and cls.content else 0,
                        'importance_score': calculate_class_importance(cls, codebase)
                    })
                except Exception as e:
                    print(f"Error calculating metrics for class {cls.name}: {e}")
                    continue
                    
    except Exception as e:
        print(f"Error in comprehensive metrics calculation: {e}")
    
    return {
        'function_metrics': function_metrics,
        'class_metrics': class_metrics,
        'file_metrics': file_metrics
    }


def calculate_maintainability_index(file: SourceFile) -> float:
    """Calculate maintainability index for a file"""
    try:
        if not hasattr(file, 'content') or not file.content:
            return 0.0
        
        lines = file.content.split('\n')
        loc = len(lines)
        complexity = sum(calculate_cyclomatic_complexity(f) for f in file.functions)
        
        # Simplified maintainability index
        if loc == 0:
            return 100.0
        
        mi = 171 - 5.2 * math.log(max(1, complexity)) - 0.23 * loc
        return max(0, min(100, mi))
    except Exception:
        return 0.0


def calculate_file_importance(file: SourceFile, codebase: Codebase) -> float:
    """Calculate importance score for a file"""
    try:
        score = 0.0
        
        # Number of functions and classes
        score += len(file.functions) * 2
        score += len(file.classes) * 3
        
        # Number of imports (indicates dependencies)
        score += len(file.imports) * 0.5
        
        # File size factor
        if hasattr(file, 'content') and file.content:
            score += len(file.content.split('\n')) * 0.01
        
        return score
    except Exception:
        return 0.0


def calculate_function_maintainability(func: Function) -> float:
    """Calculate maintainability index for a function"""
    try:
        if not hasattr(func, 'content') or not func.content:
            return 100.0
        
        lines = func.content.split('\n')
        loc = len(lines)
        complexity = calculate_cyclomatic_complexity(func)
        
        # Simplified maintainability index
        if loc == 0:
            return 100.0
        
        mi = 171 - 5.2 * math.log(max(1, complexity)) - 0.23 * loc
        return max(0, min(100, mi))
    except Exception:
        return 100.0


def calculate_function_importance(func: Function, codebase: Codebase) -> float:
    """Calculate importance score for a function"""
    try:
        score = 0.0
        
        # Complexity factor
        complexity = calculate_cyclomatic_complexity(func)
        score += complexity * 2
        
        # Parameter count
        if hasattr(func, 'parameters'):
            score += len(func.parameters) * 0.5
        
        # Entry point bonus
        if is_entry_point_function(func):
            score += 10
        
        # Function size
        if hasattr(func, 'content') and func.content:
            score += len(func.content.split('\n')) * 0.1
        
        return score
    except Exception:
        return 0.0


def calculate_class_importance(cls: Class, codebase: Codebase) -> float:
    """Calculate importance score for a class"""
    try:
        score = 0.0
        
        # Methods and attributes count
        if hasattr(cls, 'methods'):
            score += len(cls.methods) * 2
        if hasattr(cls, 'attributes'):
            score += len(cls.attributes) * 1
        
        # Inheritance depth
        score += calculate_doi(cls) * 3
        
        # Coupling
        score += calculate_cbo(cls) * 1.5
        
        return score
    except Exception:
        return 0.0


def is_entry_point_function(func: Function) -> bool:
    """Check if a function is likely an entry point"""
    entry_point_patterns = [
        'main', 'run', 'start', 'init', 'setup', 'launch', 'execute',
        'app', 'server', 'cli', 'command', 'handler', 'endpoint'
    ]
    
    return any(pattern in func.name.lower() for pattern in entry_point_patterns)


def create_health_dashboard(results: AnalysisResults) -> Dict[str, Any]:
    """Create a comprehensive health dashboard"""
    try:
        return {
            "overview": {
                "health_score": getattr(results, 'health_score', 0),
                "health_grade": getattr(results, 'health_grade', 'N/A'),
                "risk_level": getattr(results, 'risk_level', 'unknown'),
                "technical_debt_hours": getattr(results, 'technical_debt_hours', 0)
            },
            "metrics": {
                "total_issues": len(getattr(results, 'issues', [])),
                "critical_issues": results.issues_by_severity.get("critical", 0) if hasattr(results, 'issues_by_severity') else 0,
                "major_issues": results.issues_by_severity.get("major", 0) if hasattr(results, 'issues_by_severity') else 0,
                "minor_issues": results.issues_by_severity.get("minor", 0) if hasattr(results, 'issues_by_severity') else 0
            },
            "quality": {
                "average_complexity": results.complexity_metrics.get("average_cyclomatic_complexity", 0) if hasattr(results, 'complexity_metrics') else 0,
                "documentation_coverage": results.maintainability_metrics.get("documentation_coverage", 0) if hasattr(results, 'maintainability_metrics') else 0,
                "dead_functions": len(getattr(results, 'dead_functions', []))
            },
            "recommendations": generate_dashboard_recommendations(results)
        }
    except Exception as e:
        return {"error": f"Failed to create health dashboard: {str(e)}"}


def generate_dashboard_recommendations(results: AnalysisResults) -> List[str]:
    """Generate actionable recommendations for the dashboard"""
    recommendations = []
    
    try:
        if hasattr(results, 'issues_by_severity') and results.issues_by_severity.get("critical", 0) > 0:
            recommendations.append("🚨 Address critical issues immediately")
        
        if hasattr(results, 'dead_functions') and len(results.dead_functions) > 5:
            recommendations.append("🧹 Remove dead code to improve maintainability")
        
        if hasattr(results, 'complexity_metrics') and results.complexity_metrics.get("average_cyclomatic_complexity", 0) > 10:
            recommendations.append("🔄 Refactor complex functions to improve readability")
        
        if hasattr(results, 'maintainability_metrics') and results.maintainability_metrics.get("documentation_coverage", 0) < 50:
            recommendations.append("📝 Improve documentation coverage")
        
        if hasattr(results, 'technical_debt_hours') and results.technical_debt_hours > 40:
            recommendations.append("⏰ Significant technical debt detected - plan refactoring sprint")
    except Exception:
        recommendations.append("📊 Run comprehensive analysis for detailed recommendations")
    
    return recommendations


# Main analysis function for API compatibility
def analyze_codebase_comprehensive(codebase: Codebase, config: Dict[str, Any] = None) -> AnalysisResults:
    """Main entry point for comprehensive codebase analysis"""
    if config is None:
        config = {}
    
    analyzer = ComprehensiveCodebaseAnalyzer(codebase)
    return analyzer.analyze_comprehensive(config)
