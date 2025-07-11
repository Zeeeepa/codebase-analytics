"""
Comprehensive Codebase Analysis Engine
Consolidated from all analysis systems with enhanced functionality
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
    import graph_sitter.python as python_analyzer
    import graph_sitter.typescript as typescript_analyzer
except ImportError:
    Assignment = Export = Directory = Interface = python_analyzer = (
        typescript_analyzer
    ) = None


class CodeAnalysisError(Exception):
    """Custom exception for code analysis errors"""
    pass


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
        """Comprehensive issue detection"""
        print("🔍 Detecting issues...")
        
        issues = []
        issues_by_severity = defaultdict(int)
        issues_by_type = defaultdict(int)
        
        for file in self.codebase.files:
            file_issues = self._detect_file_issues(file)
            issues.extend(file_issues)
            
            for issue in file_issues:
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
        """Analyze Halstead metrics"""
        print("📏 Calculating Halstead metrics...")
        # Simplified Halstead metrics
        total_operators = 0
        total_operands = 0
        
        for file in self.codebase.files:
            # Count operators and operands (simplified)
            content = file.content
            operators = content.count('+') + content.count('-') + content.count('*') + content.count('/')
            operands = len(re.findall(r'\b\w+\b', content))
            
            total_operators += operators
            total_operands += operands
        
        results.halstead_metrics = {
            'total_operators': total_operators,
            'total_operands': total_operands,
            'volume': total_operators + total_operands,
            'difficulty': max(1, total_operators / max(1, total_operands)),
            'effort': (total_operators + total_operands) * max(1, total_operators / max(1, total_operands))
        }
        return results
    
    def _analyze_graphs(self, results: AnalysisResults) -> AnalysisResults:
        """Analyze call and dependency graphs"""
        print("🕸️ Building graphs...")
        
        # Build simple call graph
        call_graph = nx.DiGraph()
        
        for file in self.codebase.files:
            for func in file.functions:
                node_id = f"{file.path}::{func.name}"
                call_graph.add_node(node_id, file=file.path, function=func.name)
        
        results.call_graph = {
            'nodes': list(call_graph.nodes()),
            'edges': list(call_graph.edges()),
            'node_count': call_graph.number_of_nodes(),
            'edge_count': call_graph.number_of_edges()
        }
        
        results.dependency_graph = {
            'files': [file.path for file in self.codebase.files],
            'dependencies': []  # Simplified
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
        """Analyze health metrics"""
        print("🏥 Calculating health metrics...")
        
        # Calculate health scores based on issues and metrics
        issue_penalty = min(50, results.total_issues * 2)  # Max 50 point penalty
        complexity_penalty = 0  # Would be calculated from complexity metrics
        
        overall_score = max(0, 100 - issue_penalty - complexity_penalty)
        
        results.health_metrics = HealthMetrics(
            overall_score=overall_score,
            maintainability_score=max(0, 100 - issue_penalty),
            technical_debt_score=min(100, results.total_issues * 5),
            complexity_score=50.0  # Placeholder
        )
        
        return results
    
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
