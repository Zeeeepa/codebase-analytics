"""
Comprehensive Codebase Analyzer
Main orchestrator for advanced codebase analysis with structured data output
"""

import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import json
import time
from datetime import datetime

from .advanced_issues import AdvancedIssueDetector, CodeIssue, IssueSeverity, IssueType
from .function_context import FunctionContextAnalyzer, FunctionContext
from .halstead_metrics import HalsteadCalculator
from .graph_analysis import CallGraphAnalyzer, DependencyGraphAnalyzer
from .dead_code_analysis import DeadCodeAnalyzer
from .repository_structure import RepositoryStructureBuilder
from .health_metrics import CodebaseHealthCalculator


@dataclass
class AnalysisResults:
    """Structured analysis results for API consumption"""
    
    # Basic Statistics
    total_files: int
    total_functions: int
    total_classes: int
    total_lines_of_code: int
    effective_lines_of_code: int
    
    # Issue Analysis
    total_issues: int
    issues_by_severity: Dict[str, int]
    issues_by_type: Dict[str, int]
    critical_issues: List[CodeIssue]
    automated_resolutions: List[Dict[str, Any]]
    
    # Function Analysis
    most_important_functions: List[Dict[str, Any]]
    entry_points: List[Dict[str, Any]]
    function_contexts: Dict[str, FunctionContext]
    
    # Code Quality Metrics
    halstead_metrics: Dict[str, Any]
    complexity_metrics: Dict[str, Any]
    maintainability_score: float
    technical_debt_score: float
    
    # Graph Analysis
    call_graph_metrics: Dict[str, Any]
    dependency_metrics: Dict[str, Any]
    circular_dependencies: List[Dict[str, Any]]
    
    # Dead Code Analysis
    dead_code_analysis: Dict[str, Any]
    blast_radius_analysis: Dict[str, List[str]]
    
    # Repository Structure
    repository_structure: Dict[str, Any]
    file_health_map: Dict[str, Dict[str, Any]]
    directory_health_scores: Dict[str, float]
    
    # Health Indicators
    overall_health_score: float
    health_trends: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    
    # Metadata
    analysis_timestamp: str
    analysis_duration: float
    analysis_version: str = "2.0.0"


class ComprehensiveCodebaseAnalyzer:
    """Main analyzer class for comprehensive codebase analysis"""
    
    def __init__(self, codebase):
        self.codebase = codebase
        self.results = None
        
        # Initialize analyzers
        self.issue_detector = AdvancedIssueDetector(codebase)
        self.function_analyzer = FunctionContextAnalyzer(codebase)
        self.halstead_calculator = HalsteadCalculator(codebase)
        self.call_graph_analyzer = CallGraphAnalyzer(codebase)
        self.dependency_analyzer = DependencyGraphAnalyzer(codebase)
        self.dead_code_analyzer = DeadCodeAnalyzer(codebase)
        self.structure_builder = RepositoryStructureBuilder(codebase)
        self.health_calculator = CodebaseHealthCalculator(codebase)
        
    def analyze(self) -> AnalysisResults:
        """Perform comprehensive analysis and return structured results"""
        start_time = time.time()
        print("🔍 Starting comprehensive codebase analysis...")
        
        # 1. Basic Statistics
        basic_stats = self._calculate_basic_statistics()
        
        # 2. Issue Detection with Automated Resolutions
        print("🔍 Detecting issues and generating automated resolutions...")
        issues = self.issue_detector.detect_all_issues()
        automated_resolutions = self._extract_automated_resolutions(issues)
        
        # 3. Function Analysis
        print("📊 Analyzing function contexts and relationships...")
        function_contexts = self.function_analyzer.analyze_all_functions()
        most_important = self._find_most_important_functions(function_contexts)
        entry_points = self._identify_entry_points(function_contexts)
        
        # 4. Code Quality Metrics
        print("📈 Calculating quality metrics...")
        halstead_metrics = self.halstead_calculator.calculate_all_metrics()
        complexity_metrics = self._calculate_complexity_metrics(function_contexts)
        maintainability_score = self._calculate_maintainability_score(halstead_metrics, complexity_metrics)
        technical_debt_score = self._calculate_technical_debt_score(issues)
        
        # 5. Graph Analysis
        print("🕸️ Building and analyzing graphs...")
        call_graph_metrics = self.call_graph_analyzer.analyze()
        dependency_metrics = self.dependency_analyzer.analyze()
        circular_deps = self.dependency_analyzer.find_circular_dependencies()
        
        # 6. Dead Code Analysis
        print("💀 Analyzing dead code and blast radius...")
        dead_code_analysis = self.dead_code_analyzer.analyze()
        blast_radius = self.dead_code_analyzer.calculate_blast_radius()
        
        # 7. Repository Structure
        print("🏗️ Building repository structure...")
        repo_structure = self.structure_builder.build_structure(issues)
        file_health_map = self._calculate_file_health_map(issues, halstead_metrics)
        directory_health = self._calculate_directory_health_scores(repo_structure)
        
        # 8. Health Assessment
        print("🏥 Calculating health indicators...")
        overall_health = self.health_calculator.calculate_overall_health(
            issues, complexity_metrics, dead_code_analysis
        )
        health_trends = self.health_calculator.analyze_trends()
        risk_assessment = self.health_calculator.assess_risks(issues, circular_deps)
        
        # Compile results
        analysis_duration = time.time() - start_time
        
        self.results = AnalysisResults(
            # Basic Statistics
            total_files=basic_stats['files'],
            total_functions=basic_stats['functions'],
            total_classes=basic_stats['classes'],
            total_lines_of_code=basic_stats['total_loc'],
            effective_lines_of_code=basic_stats['effective_loc'],
            
            # Issue Analysis
            total_issues=len(issues),
            issues_by_severity=self._group_issues_by_severity(issues),
            issues_by_type=self._group_issues_by_type(issues),
            critical_issues=[i for i in issues if i.severity == IssueSeverity.CRITICAL],
            automated_resolutions=automated_resolutions,
            
            # Function Analysis
            most_important_functions=most_important,
            entry_points=entry_points,
            function_contexts=function_contexts,
            
            # Code Quality Metrics
            halstead_metrics=halstead_metrics,
            complexity_metrics=complexity_metrics,
            maintainability_score=maintainability_score,
            technical_debt_score=technical_debt_score,
            
            # Graph Analysis
            call_graph_metrics=call_graph_metrics,
            dependency_metrics=dependency_metrics,
            circular_dependencies=circular_deps,
            
            # Dead Code Analysis
            dead_code_analysis=dead_code_analysis,
            blast_radius_analysis=blast_radius,
            
            # Repository Structure
            repository_structure=repo_structure,
            file_health_map=file_health_map,
            directory_health_scores=directory_health,
            
            # Health Indicators
            overall_health_score=overall_health,
            health_trends=health_trends,
            risk_assessment=risk_assessment,
            
            # Metadata
            analysis_timestamp=datetime.now().isoformat(),
            analysis_duration=analysis_duration
        )
        
        print(f"✅ Analysis complete! Duration: {analysis_duration:.2f}s")
        return self.results
    
    def get_structured_data(self) -> Dict[str, Any]:
        """Get analysis results as structured data for API consumption"""
        if not self.results:
            raise ValueError("Analysis not performed yet. Call analyze() first.")
        
        return {
            "metadata": {
                "analysis_timestamp": self.results.analysis_timestamp,
                "analysis_duration": self.results.analysis_duration,
                "analysis_version": self.results.analysis_version
            },
            "statistics": {
                "total_files": self.results.total_files,
                "total_functions": self.results.total_functions,
                "total_classes": self.results.total_classes,
                "total_lines_of_code": self.results.total_lines_of_code,
                "effective_lines_of_code": self.results.effective_lines_of_code
            },
            "issues": {
                "total_issues": self.results.total_issues,
                "by_severity": self.results.issues_by_severity,
                "by_type": self.results.issues_by_type,
                "critical_issues": [self._serialize_issue(issue) for issue in self.results.critical_issues],
                "automated_resolutions": self.results.automated_resolutions
            },
            "functions": {
                "most_important": self.results.most_important_functions,
                "entry_points": self.results.entry_points,
                "total_analyzed": len(self.results.function_contexts)
            },
            "quality_metrics": {
                "halstead": self.results.halstead_metrics,
                "complexity": self.results.complexity_metrics,
                "maintainability_score": self.results.maintainability_score,
                "technical_debt_score": self.results.technical_debt_score
            },
            "graph_analysis": {
                "call_graph": self.results.call_graph_metrics,
                "dependencies": self.results.dependency_metrics,
                "circular_dependencies": self.results.circular_dependencies
            },
            "dead_code": {
                "analysis": self.results.dead_code_analysis,
                "blast_radius": self.results.blast_radius_analysis
            },
            "repository_structure": self.results.repository_structure,
            "health_assessment": {
                "overall_score": self.results.overall_health_score,
                "file_health_map": self.results.file_health_map,
                "directory_health_scores": self.results.directory_health_scores,
                "trends": self.results.health_trends,
                "risk_assessment": self.results.risk_assessment
            }
        }
    
    def get_health_dashboard_data(self) -> Dict[str, Any]:
        """Get data optimized for health dashboard display"""
        if not self.results:
            raise ValueError("Analysis not performed yet. Call analyze() first.")
        
        return {
            "health_score": self.results.overall_health_score,
            "health_grade": self._calculate_health_grade(self.results.overall_health_score),
            "key_metrics": {
                "total_issues": self.results.total_issues,
                "critical_issues": len(self.results.critical_issues),
                "technical_debt_score": self.results.technical_debt_score,
                "maintainability_score": self.results.maintainability_score,
                "dead_code_percentage": self._calculate_dead_code_percentage()
            },
            "trends": self.results.health_trends,
            "top_issues": self._get_top_issues_summary(),
            "automated_fixes_available": len(self.results.automated_resolutions),
            "risk_level": self._calculate_risk_level(),
            "recommendations": self._generate_recommendations()
        }
    
    def _calculate_basic_statistics(self) -> Dict[str, int]:
        """Calculate basic codebase statistics"""
        total_loc = 0
        effective_loc = 0
        
        for file in self.codebase.files:
            if hasattr(file, 'source'):
                lines = file.source.split('\n')
                total_loc += len(lines)
                effective_loc += len([line for line in lines if line.strip()])
        
        return {
            'files': len(self.codebase.files),
            'functions': len(self.codebase.functions),
            'classes': len(self.codebase.classes),
            'total_loc': total_loc,
            'effective_loc': effective_loc
        }
    
    def _extract_automated_resolutions(self, issues: List[CodeIssue]) -> List[Dict[str, Any]]:
        """Extract automated resolutions from issues"""
        resolutions = []
        for issue in issues:
            if issue.automated_resolution:
                resolutions.append({
                    "issue_id": f"{issue.filepath}:{issue.line_number}",
                    "issue_type": issue.issue_type.value,
                    "resolution_type": issue.automated_resolution.resolution_type,
                    "description": issue.automated_resolution.description,
                    "original_code": issue.automated_resolution.original_code,
                    "fixed_code": issue.automated_resolution.fixed_code,
                    "confidence": issue.automated_resolution.confidence,
                    "file_path": issue.automated_resolution.file_path,
                    "line_number": issue.automated_resolution.line_number,
                    "is_safe": issue.automated_resolution.is_safe,
                    "requires_validation": issue.automated_resolution.requires_validation
                })
        return resolutions
    
    def _find_most_important_functions(self, function_contexts: Dict[str, FunctionContext]) -> List[Dict[str, Any]]:
        """Find most important functions based on various metrics"""
        functions_with_scores = []
        
        for name, context in function_contexts.items():
            importance_score = (
                len(context.usages) * 3 +  # Usage frequency
                len(context.dependencies) * 2 +  # Dependency count
                len(context.function_calls) * 1 +  # Calls made
                (10 if context.is_entry_point else 0) +  # Entry point bonus
                context.complexity_metrics.get('cyclomatic_complexity', 0) * 0.5  # Complexity factor
            )
            
            functions_with_scores.append({
                "name": name,
                "filepath": context.filepath,
                "importance_score": importance_score,
                "usage_count": len(context.usages),
                "dependency_count": len(context.dependencies),
                "complexity": context.complexity_metrics.get('cyclomatic_complexity', 0),
                "is_entry_point": context.is_entry_point,
                "line_start": context.line_start,
                "line_end": context.line_end
            })
        
        # Sort by importance score and return top 20
        functions_with_scores.sort(key=lambda x: x['importance_score'], reverse=True)
        return functions_with_scores[:20]
    
    def _identify_entry_points(self, function_contexts: Dict[str, FunctionContext]) -> List[Dict[str, Any]]:
        """Identify entry point functions"""
        entry_points = []
        for name, context in function_contexts.items():
            if context.is_entry_point:
                entry_points.append({
                    "name": name,
                    "filepath": context.filepath,
                    "type": "entry_point",
                    "calls_made": len(context.function_calls),
                    "complexity": context.complexity_metrics.get('cyclomatic_complexity', 0),
                    "line_start": context.line_start
                })
        return entry_points
    
    def _calculate_complexity_metrics(self, function_contexts: Dict[str, FunctionContext]) -> Dict[str, Any]:
        """Calculate overall complexity metrics"""
        complexities = [ctx.complexity_metrics.get('cyclomatic_complexity', 0) 
                      for ctx in function_contexts.values()]
        
        if not complexities:
            return {}
        
        return {
            "average_complexity": sum(complexities) / len(complexities),
            "max_complexity": max(complexities),
            "min_complexity": min(complexities),
            "high_complexity_functions": len([c for c in complexities if c > 15]),
            "complexity_distribution": {
                "low": len([c for c in complexities if c <= 5]),
                "medium": len([c for c in complexities if 5 < c <= 15]),
                "high": len([c for c in complexities if c > 15])
            }
        }
    
    def _calculate_maintainability_score(self, halstead_metrics: Dict[str, Any], 
                                       complexity_metrics: Dict[str, Any]) -> float:
        """Calculate overall maintainability score (0-100)"""
        base_score = 100.0
        
        # Penalize high complexity
        avg_complexity = complexity_metrics.get('average_complexity', 0)
        complexity_penalty = min(avg_complexity * 2, 30)
        
        # Penalize high Halstead difficulty
        avg_difficulty = halstead_metrics.get('average_difficulty', 0)
        difficulty_penalty = min(avg_difficulty, 20)
        
        # Penalize high volume
        avg_volume = halstead_metrics.get('average_volume', 0)
        volume_penalty = min(avg_volume / 100, 15)
        
        maintainability = base_score - complexity_penalty - difficulty_penalty - volume_penalty
        return max(0, min(100, maintainability))
    
    def _calculate_technical_debt_score(self, issues: List[CodeIssue]) -> float:
        """Calculate technical debt score based on issues"""
        if not issues:
            return 0.0
        
        debt_score = 0.0
        severity_weights = {
            IssueSeverity.CRITICAL: 10,
            IssueSeverity.MAJOR: 5,
            IssueSeverity.MINOR: 2,
            IssueSeverity.INFO: 1
        }
        
        for issue in issues:
            debt_score += severity_weights.get(issue.severity, 1) * issue.impact_score
        
        # Normalize to 0-100 scale
        return min(100, debt_score / len(issues))
    
    def _group_issues_by_severity(self, issues: List[CodeIssue]) -> Dict[str, int]:
        """Group issues by severity"""
        grouped = {severity.value: 0 for severity in IssueSeverity}
        for issue in issues:
            grouped[issue.severity.value] += 1
        return grouped
    
    def _group_issues_by_type(self, issues: List[CodeIssue]) -> Dict[str, int]:
        """Group issues by type"""
        grouped = defaultdict(int)
        for issue in issues:
            grouped[issue.issue_type.value] += 1
        return dict(grouped)
    
    def _calculate_file_health_map(self, issues: List[CodeIssue], 
                                 halstead_metrics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Calculate health metrics for each file"""
        file_health = {}
        
        # Group issues by file
        issues_by_file = defaultdict(list)
        for issue in issues:
            issues_by_file[issue.filepath].append(issue)
        
        for file in self.codebase.files:
            file_issues = issues_by_file[file.filepath]
            
            # Calculate health score
            health_score = 100.0
            for issue in file_issues:
                severity_penalty = {
                    IssueSeverity.CRITICAL: 15,
                    IssueSeverity.MAJOR: 8,
                    IssueSeverity.MINOR: 3,
                    IssueSeverity.INFO: 1
                }
                health_score -= severity_penalty.get(issue.severity, 1)
            
            health_score = max(0, health_score)
            
            file_health[file.filepath] = {
                "health_score": health_score,
                "issue_count": len(file_issues),
                "critical_issues": len([i for i in file_issues if i.severity == IssueSeverity.CRITICAL]),
                "lines_of_code": len(file.source.split('\n')) if hasattr(file, 'source') else 0,
                "health_grade": self._calculate_health_grade(health_score)
            }
        
        return file_health
    
    def _calculate_directory_health_scores(self, repo_structure: Dict[str, Any]) -> Dict[str, float]:
        """Calculate health scores for directories"""
        # Placeholder implementation
        return {}
    
    def _calculate_health_grade(self, score: float) -> str:
        """Convert health score to letter grade"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _calculate_dead_code_percentage(self) -> float:
        """Calculate percentage of dead code"""
        if not self.results:
            return 0.0
        
        dead_functions = self.results.dead_code_analysis.get('total_dead_functions', 0)
        total_functions = self.results.total_functions
        
        if total_functions == 0:
            return 0.0
        
        return (dead_functions / total_functions) * 100
    
    def _get_top_issues_summary(self) -> List[Dict[str, Any]]:
        """Get summary of top issues"""
        if not self.results:
            return []
        
        # Return top 5 critical issues
        top_issues = []
        for issue in self.results.critical_issues[:5]:
            top_issues.append({
                "type": issue.issue_type.value,
                "message": issue.message,
                "file": issue.filepath,
                "line": issue.line_number,
                "impact_score": issue.impact_score
            })
        
        return top_issues
    
    def _calculate_risk_level(self) -> str:
        """Calculate overall risk level"""
        if not self.results:
            return "unknown"
        
        critical_count = len(self.results.critical_issues)
        health_score = self.results.overall_health_score
        
        if critical_count > 10 or health_score < 50:
            return "high"
        elif critical_count > 5 or health_score < 70:
            return "medium"
        else:
            return "low"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        if not self.results:
            return []
        
        recommendations = []
        
        # Critical issues
        if len(self.results.critical_issues) > 0:
            recommendations.append(f"Address {len(self.results.critical_issues)} critical issues immediately")
        
        # Automated fixes
        if len(self.results.automated_resolutions) > 0:
            recommendations.append(f"Apply {len(self.results.automated_resolutions)} automated fixes available")
        
        # Dead code
        dead_percentage = self._calculate_dead_code_percentage()
        if dead_percentage > 10:
            recommendations.append(f"Remove {dead_percentage:.1f}% dead code to improve maintainability")
        
        # Complexity
        avg_complexity = self.results.complexity_metrics.get('average_complexity', 0)
        if avg_complexity > 10:
            recommendations.append("Refactor high-complexity functions to improve readability")
        
        return recommendations
    
    def _serialize_issue(self, issue: CodeIssue) -> Dict[str, Any]:
        """Serialize issue for JSON output"""
        return {
            "type": issue.issue_type.value,
            "severity": issue.severity.value,
            "message": issue.message,
            "filepath": issue.filepath,
            "line_number": issue.line_number,
            "column_number": issue.column_number,
            "function_name": issue.function_name,
            "class_name": issue.class_name,
            "context": issue.context,
            "suggested_fix": issue.suggested_fix,
            "impact_score": issue.impact_score,
            "fix_effort": issue.fix_effort,
            "has_automated_resolution": issue.automated_resolution is not None
        }


def analyze_codebase(codebase) -> AnalysisResults:
    """Main entry point for comprehensive codebase analysis"""
    analyzer = ComprehensiveCodebaseAnalyzer(codebase)
    return analyzer.analyze()

