"""
Codebase Analytics - Core Analysis Engine
Comprehensive analysis engine for code quality, issue detection, and metrics calculation
"""

import math
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
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

from .models import (
    CodeIssue, IssueType, IssueSeverity, FunctionContext, 
    AnalysisResults, AutomatedResolution, AnalysisConfig
)


# ============================================================================
# CORE ANALYSIS ENGINE
# ============================================================================

class CodebaseAnalyzer:
    """Main analysis engine for comprehensive codebase analysis"""
    
    def __init__(self):
        self.config = AnalysisConfig()
        self.issues: List[CodeIssue] = []
        self.function_contexts: Dict[str, FunctionContext] = {}
        self.call_graph: Dict[str, List[str]] = {}
        self.reverse_call_graph: Dict[str, List[str]] = {}
        
    def analyze_codebase(self, codebase: Codebase) -> AnalysisResults:
        """Perform comprehensive codebase analysis"""
        
        # Initialize analysis results
        results = AnalysisResults(
            total_files=len(codebase.source_files),
            total_functions=0,
            total_classes=0,
            total_lines_of_code=0
        )
        
        # Phase 1: Basic metrics collection
        self._collect_basic_metrics(codebase, results)
        
        # Phase 2: Function analysis and context building
        self._analyze_functions(codebase, results)
        
        # Phase 3: Issue detection
        self._detect_issues(codebase, results)
        
        # Phase 4: Call graph analysis
        self._analyze_call_graph(codebase, results)
        
        # Phase 5: Quality metrics calculation
        self._calculate_quality_metrics(codebase, results)
        
        # Phase 6: Health assessment
        self._assess_health(results)
        
        # Phase 7: Generate automated resolutions
        self._generate_automated_resolutions(results)
        
        return results
    
    def _collect_basic_metrics(self, codebase: Codebase, results: AnalysisResults):
        """Collect basic codebase metrics"""
        total_lines = 0
        total_functions = 0
        total_classes = 0
        
        for source_file in codebase.source_files:
            if source_file.source:
                total_lines += len(source_file.source.split('\n'))
            
            # Count functions and classes
            for symbol in source_file.symbols:
                if isinstance(symbol, Function):
                    total_functions += 1
                elif isinstance(symbol, Class):
                    total_classes += 1
        
        results.total_lines_of_code = total_lines
        results.total_functions = total_functions
        results.total_classes = total_classes
    
    def _analyze_functions(self, codebase: Codebase, results: AnalysisResults):
        """Analyze all functions and build contexts"""
        
        for source_file in codebase.source_files:
            for symbol in source_file.symbols:
                if isinstance(symbol, Function):
                    context = self._build_function_context(symbol, source_file, codebase)
                    self.function_contexts[context.name] = context
                    
                    # Build call graph
                    self.call_graph[context.name] = context.function_calls
                    
                    # Build reverse call graph
                    for called_func in context.function_calls:
                        if called_func not in self.reverse_call_graph:
                            self.reverse_call_graph[called_func] = []
                        self.reverse_call_graph[called_func].append(context.name)
        
        # Update function contexts with call relationships
        for func_name, context in self.function_contexts.items():
            context.called_by = self.reverse_call_graph.get(func_name, [])
            context.fan_in = len(context.called_by)
            context.fan_out = len(context.function_calls)
        
        results.function_contexts = self.function_contexts
    
    def _build_function_context(self, function: Function, source_file: SourceFile, codebase: Codebase) -> FunctionContext:
        """Build comprehensive context for a function"""
        
        context = FunctionContext(
            name=function.name,
            filepath=source_file.file_path,
            line_start=function.line_start,
            line_end=function.line_end,
            source=function.source or ""
        )
        
        # Extract parameters
        if hasattr(function, 'parameters'):
            context.parameters = [
                {"name": param.name, "type": getattr(param, 'type', None)}
                for param in function.parameters
            ]
        
        # Extract function calls
        context.function_calls = self._extract_function_calls(function)
        
        # Calculate complexity metrics
        context.complexity_metrics = self._calculate_function_complexity(function)
        
        # Detect if it's an entry point
        context.is_entry_point = self._is_entry_point(function.name)
        
        return context
    
    def _extract_function_calls(self, function: Function) -> List[str]:
        """Extract all function calls from a function"""
        calls = []
        
        # This would need to be implemented based on graph_sitter's API
        # For now, return empty list as placeholder
        return calls
    
    def _calculate_function_complexity(self, function: Function) -> Dict[str, Any]:
        """Calculate various complexity metrics for a function"""
        
        source = function.source or ""
        lines = source.split('\n')
        
        metrics = {
            'lines_of_code': len(lines),
            'cyclomatic_complexity': self._calculate_cyclomatic_complexity(function),
            'halstead_metrics': self._calculate_halstead_metrics(source),
            'nesting_depth': self._calculate_nesting_depth(source)
        }
        
        return metrics
    
    def _calculate_cyclomatic_complexity(self, function: Function) -> int:
        """Calculate cyclomatic complexity"""
        # Simplified implementation - count decision points
        source = function.source or ""
        
        decision_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'and', 'or']
        complexity = 1  # Base complexity
        
        for keyword in decision_keywords:
            complexity += source.count(keyword)
        
        return complexity
    
    def _calculate_halstead_metrics(self, source: str) -> Dict[str, float]:
        """Calculate Halstead complexity metrics"""
        
        # Simplified implementation
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
    
    def _calculate_nesting_depth(self, source: str) -> int:
        """Calculate maximum nesting depth"""
        lines = source.split('\n')
        max_depth = 0
        current_depth = 0
        
        for line in lines:
            stripped = line.strip()
            if any(keyword in stripped for keyword in ['if', 'for', 'while', 'try', 'with', 'def', 'class']):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif stripped in ['else:', 'elif', 'except:', 'finally:']:
                continue
            elif stripped == '' or stripped.startswith('#'):
                continue
            else:
                # Simplified - should properly track indentation
                pass
        
        return max_depth
    
    def _is_entry_point(self, function_name: str) -> bool:
        """Check if function is likely an entry point"""
        return any(pattern in function_name.lower() for pattern in self.config.ENTRY_POINT_PATTERNS)
    
    def _detect_issues(self, codebase: Codebase, results: AnalysisResults):
        """Detect various code issues"""
        
        for source_file in codebase.source_files:
            self._detect_file_issues(source_file, results)
            
            for symbol in source_file.symbols:
                if isinstance(symbol, Function):
                    self._detect_function_issues(symbol, source_file, results)
        
        # Categorize issues
        results.issues = self.issues
        results.issues_by_severity = self._categorize_issues_by_severity()
        results.issues_by_type = self._categorize_issues_by_type()
    
    def _detect_file_issues(self, source_file: SourceFile, results: AnalysisResults):
        """Detect file-level issues"""
        
        if not source_file.source:
            return
        
        lines = source_file.source.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 120:
                self._add_issue(
                    IssueType.LINE_LENGTH_VIOLATION,
                    IssueSeverity.MINOR,
                    f"Line exceeds 120 characters ({len(line)} chars)",
                    source_file.file_path,
                    line_num,
                    0
                )
            
            # Check for magic numbers
            magic_numbers = re.findall(r'\b\d+\b', line)
            for num_str in magic_numbers:
                num = int(num_str)
                if num not in self.config.MAGIC_NUMBER_EXCLUSIONS:
                    self._add_issue(
                        IssueType.MAGIC_NUMBER,
                        IssueSeverity.MINOR,
                        f"Magic number found: {num}",
                        source_file.file_path,
                        line_num,
                        line.find(num_str)
                    )
    
    def _detect_function_issues(self, function: Function, source_file: SourceFile, results: AnalysisResults):
        """Detect function-level issues"""
        
        # Check function length
        if function.line_end - function.line_start > self.config.LONG_FUNCTION_THRESHOLD:
            self._add_issue(
                IssueType.LONG_FUNCTION,
                IssueSeverity.MAJOR,
                f"Function '{function.name}' is too long ({function.line_end - function.line_start} lines)",
                source_file.file_path,
                function.line_start,
                0,
                function_name=function.name
            )
        
        # Check for missing documentation
        if not function.docstring:
            self._add_issue(
                IssueType.MISSING_DOCUMENTATION,
                IssueSeverity.MINOR,
                f"Function '{function.name}' lacks documentation",
                source_file.file_path,
                function.line_start,
                0,
                function_name=function.name
            )
        
        # Check complexity
        context = self.function_contexts.get(function.name)
        if context and context.complexity_metrics.get('cyclomatic_complexity', 0) > 10:
            self._add_issue(
                IssueType.INEFFICIENT_PATTERN,
                IssueSeverity.MAJOR,
                f"Function '{function.name}' has high cyclomatic complexity",
                source_file.file_path,
                function.line_start,
                0,
                function_name=function.name
            )
    
    def _add_issue(self, issue_type: IssueType, severity: IssueSeverity, message: str, 
                   filepath: str, line_number: int, column_number: int, 
                   function_name: Optional[str] = None, class_name: Optional[str] = None):
        """Add an issue to the issues list"""
        
        issue = CodeIssue(
            issue_type=issue_type,
            severity=severity,
            message=message,
            filepath=filepath,
            line_number=line_number,
            column_number=column_number,
            function_name=function_name,
            class_name=class_name
        )
        
        self.issues.append(issue)
    
    def _categorize_issues_by_severity(self) -> Dict[str, int]:
        """Categorize issues by severity"""
        categories = {}
        for issue in self.issues:
            severity = issue.severity.value
            categories[severity] = categories.get(severity, 0) + 1
        return categories
    
    def _categorize_issues_by_type(self) -> Dict[str, int]:
        """Categorize issues by type"""
        categories = {}
        for issue in self.issues:
            issue_type = issue.issue_type.value
            categories[issue_type] = categories.get(issue_type, 0) + 1
        return categories
    
    def _analyze_call_graph(self, codebase: Codebase, results: AnalysisResults):
        """Analyze call graph and identify important functions"""
        
        # Find entry points
        entry_points = [
            name for name, context in self.function_contexts.items()
            if context.is_entry_point
        ]
        results.entry_points = entry_points
        
        # Find dead functions (not called by anyone)
        dead_functions = [
            name for name, context in self.function_contexts.items()
            if not context.called_by and not context.is_entry_point
        ]
        results.dead_functions = dead_functions
        
        # Calculate importance scores
        important_functions = []
        for name, context in self.function_contexts.items():
            importance_score = context.fan_in * 2 + context.fan_out
            if context.is_entry_point:
                importance_score += 10
            
            important_functions.append({
                'name': name,
                'importance_score': importance_score,
                'fan_in': context.fan_in,
                'fan_out': context.fan_out,
                'is_entry_point': context.is_entry_point
            })
        
        # Sort by importance
        important_functions.sort(key=lambda x: x['importance_score'], reverse=True)
        results.most_important_functions = important_functions[:10]
        
        # Call graph metrics
        results.call_graph_metrics = {
            'total_functions': len(self.function_contexts),
            'entry_points': len(entry_points),
            'dead_functions': len(dead_functions),
            'average_fan_in': sum(c.fan_in for c in self.function_contexts.values()) / len(self.function_contexts) if self.function_contexts else 0,
            'average_fan_out': sum(c.fan_out for c in self.function_contexts.values()) / len(self.function_contexts) if self.function_contexts else 0
        }
    
    def _calculate_quality_metrics(self, codebase: Codebase, results: AnalysisResults):
        """Calculate overall quality metrics"""
        
        # Halstead metrics aggregation
        total_volume = sum(
            context.complexity_metrics.get('halstead_metrics', {}).get('volume', 0)
            for context in self.function_contexts.values()
        )
        
        total_effort = sum(
            context.complexity_metrics.get('halstead_metrics', {}).get('effort', 0)
            for context in self.function_contexts.values()
        )
        
        results.halstead_metrics = {
            'total_volume': total_volume,
            'total_effort': total_effort,
            'average_volume': total_volume / len(self.function_contexts) if self.function_contexts else 0
        }
        
        # Complexity metrics
        avg_complexity = sum(
            context.complexity_metrics.get('cyclomatic_complexity', 0)
            for context in self.function_contexts.values()
        ) / len(self.function_contexts) if self.function_contexts else 0
        
        results.complexity_metrics = {
            'average_cyclomatic_complexity': avg_complexity,
            'max_nesting_depth': max(
                (context.complexity_metrics.get('nesting_depth', 0) for context in self.function_contexts.values()),
                default=0
            )
        }
        
        # Maintainability metrics
        results.maintainability_metrics = {
            'code_duplication_ratio': 0.0,  # Placeholder
            'test_coverage': 0.0,  # Placeholder
            'documentation_coverage': self._calculate_documentation_coverage()
        }
    
    def _calculate_documentation_coverage(self) -> float:
        """Calculate documentation coverage percentage"""
        if not self.function_contexts:
            return 0.0
        
        documented_functions = sum(
            1 for context in self.function_contexts.values()
            if 'docstring' in context.source or '"""' in context.source or "'''" in context.source
        )
        
        return (documented_functions / len(self.function_contexts)) * 100
    
    def _assess_health(self, results: AnalysisResults):
        """Assess overall codebase health"""
        
        # Calculate weighted issue score
        issue_score = 0
        for issue in self.issues:
            if issue.severity == IssueSeverity.CRITICAL:
                issue_score += self.config.CRITICAL_ISSUE_WEIGHT
            elif issue.severity == IssueSeverity.MAJOR:
                issue_score += self.config.MAJOR_ISSUE_WEIGHT
            elif issue.severity == IssueSeverity.MINOR:
                issue_score += self.config.MINOR_ISSUE_WEIGHT
            else:
                issue_score += self.config.INFO_ISSUE_WEIGHT
        
        # Calculate health score (0-100)
        max_possible_score = len(self.function_contexts) * 10  # Arbitrary baseline
        health_score = max(0, 100 - (issue_score / max_possible_score * 100)) if max_possible_score > 0 else 0
        
        results.health_score = health_score
        
        # Assign health grade
        for threshold, grade in sorted(self.config.HEALTH_GRADES.items(), reverse=True):
            if health_score >= threshold:
                results.health_grade = grade
                break
        
        # Determine risk level
        if health_score >= 80:
            results.risk_level = "low"
        elif health_score >= 60:
            results.risk_level = "medium"
        else:
            results.risk_level = "high"
        
        # Calculate technical debt
        technical_debt = 0
        for issue in self.issues:
            if issue.severity == IssueSeverity.CRITICAL:
                technical_debt += self.config.CRITICAL_ISSUE_HOURS
            elif issue.severity == IssueSeverity.MAJOR:
                technical_debt += self.config.MAJOR_ISSUE_HOURS
            elif issue.severity == IssueSeverity.MINOR:
                technical_debt += self.config.MINOR_ISSUE_HOURS
            else:
                technical_debt += self.config.INFO_ISSUE_HOURS
        
        results.technical_debt_hours = technical_debt
    
    def _generate_automated_resolutions(self, results: AnalysisResults):
        """Generate automated resolutions for detected issues"""
        
        resolutions = []
        
        for issue in self.issues:
            resolution = self._create_automated_resolution(issue)
            if resolution:
                resolutions.append(resolution)
                issue.automated_resolution = resolution
        
        results.automated_resolutions = resolutions
    
    def _create_automated_resolution(self, issue: CodeIssue) -> Optional[AutomatedResolution]:
        """Create an automated resolution for a specific issue"""
        
        if issue.issue_type == IssueType.MAGIC_NUMBER:
            return AutomatedResolution(
                resolution_type="extract_constant",
                description="Extract magic number to a named constant",
                original_code="",  # Would need actual code context
                fixed_code="",     # Would need to generate fix
                confidence=0.8,
                file_path=issue.filepath,
                line_number=issue.line_number,
                is_safe=True,
                requires_validation=False
            )
        
        elif issue.issue_type == IssueType.MISSING_DOCUMENTATION:
            return AutomatedResolution(
                resolution_type="add_docstring",
                description="Add basic docstring to function",
                original_code="",
                fixed_code="",
                confidence=0.9,
                file_path=issue.filepath,
                line_number=issue.line_number,
                is_safe=True,
                requires_validation=False
            )
        
        # Add more resolution types as needed
        return None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_codebase_summary(codebase: Codebase) -> str:
    """Generate a human-readable summary of the codebase"""
    
    total_files = len(codebase.source_files)
    total_functions = sum(
        len([s for s in sf.symbols if isinstance(s, Function)])
        for sf in codebase.source_files
    )
    total_classes = sum(
        len([s for s in sf.symbols if isinstance(s, Class)])
        for sf in codebase.source_files
    )
    
    return f"""
    📊 Codebase Summary:
    • Files: {total_files}
    • Functions: {total_functions}
    • Classes: {total_classes}
    """


def create_health_dashboard(results: AnalysisResults) -> Dict[str, Any]:
    """Create a health dashboard from analysis results"""
    
    return {
        "overview": {
            "health_score": results.health_score,
            "health_grade": results.health_grade,
            "risk_level": results.risk_level,
            "technical_debt_hours": results.technical_debt_hours
        },
        "metrics": {
            "total_issues": len(results.issues),
            "critical_issues": results.issues_by_severity.get("critical", 0),
            "major_issues": results.issues_by_severity.get("major", 0),
            "minor_issues": results.issues_by_severity.get("minor", 0)
        },
        "quality": {
            "average_complexity": results.complexity_metrics.get("average_cyclomatic_complexity", 0),
            "documentation_coverage": results.maintainability_metrics.get("documentation_coverage", 0),
            "dead_functions": len(results.dead_functions)
        },
        "recommendations": _generate_recommendations(results)
    }


def _generate_recommendations(results: AnalysisResults) -> List[str]:
    """Generate actionable recommendations based on analysis"""
    
    recommendations = []
    
    if results.issues_by_severity.get("critical", 0) > 0:
        recommendations.append("🚨 Address critical issues immediately")
    
    if len(results.dead_functions) > 5:
        recommendations.append("🧹 Remove dead code to improve maintainability")
    
    if results.complexity_metrics.get("average_cyclomatic_complexity", 0) > 10:
        recommendations.append("🔄 Refactor complex functions to improve readability")
    
    if results.maintainability_metrics.get("documentation_coverage", 0) < 50:
        recommendations.append("📝 Improve documentation coverage")
    
    if results.technical_debt_hours > 40:
        recommendations.append("⏰ Significant technical debt detected - plan refactoring sprint")
    
    return recommendations
