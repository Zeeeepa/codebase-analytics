"""
Enhanced Analysis Methods
Additional analysis capabilities for advanced function context analysis
"""

from typing import Dict, List, Any, Set
from .models import AnalysisResults, FunctionContext


def calculate_coupling_cohesion_metrics(analyzer):
    """Calculate coupling and cohesion metrics for functions"""
    
    for func_name, context in analyzer.function_contexts.items():
        # Coupling score: based on number of external dependencies
        external_calls = len([call for call in context.function_calls 
                            if call not in analyzer.function_contexts])
        internal_calls = len([call for call in context.function_calls 
                            if call in analyzer.function_contexts])
        
        total_calls = external_calls + internal_calls
        context.coupling_score = external_calls / max(total_calls, 1)
        
        # Cohesion score: simplified metric based on function focus
        lines_of_code = context.line_end - context.line_start
        complexity = context.complexity_metrics.get('cyclomatic_complexity', 1)
        
        # Simple heuristic: cohesion inversely related to complexity per line
        context.cohesion_score = max(0, 1 - (complexity / max(lines_of_code, 1)))


def detect_function_importance(analyzer, results: AnalysisResults):
    """Detect and rank function importance"""
    
    importance_scores = {}
    
    for func_name, context in analyzer.function_contexts.items():
        score = 0
        
        # Entry points get high importance
        if context.is_entry_point:
            score += 10
        
        # Functions called by many others are important
        score += context.fan_in * 2
        
        # Functions that call many others might be coordinators
        score += context.fan_out * 0.5
        
        # Long call chains indicate important orchestration functions
        score += len(context.max_call_chain) * 0.3
        
        # Lower coupling is better (more maintainable)
        score += (1 - context.coupling_score) * 2
        
        # Higher cohesion is better
        score += context.cohesion_score * 2
        
        importance_scores[func_name] = score
    
    # Sort by importance and store top functions
    sorted_functions = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    results.most_important_functions = [
        {
            "name": func_name,
            "importance_score": score,
            "filepath": analyzer.function_contexts[func_name].filepath,
            "fan_in": analyzer.function_contexts[func_name].fan_in,
            "fan_out": analyzer.function_contexts[func_name].fan_out,
            "is_entry_point": analyzer.function_contexts[func_name].is_entry_point,
            "coupling_score": analyzer.function_contexts[func_name].coupling_score,
            "cohesion_score": analyzer.function_contexts[func_name].cohesion_score
        }
        for func_name, score in sorted_functions[:20]  # Top 20
    ]
    
    # Detect entry points
    results.entry_points = [
        func_name for func_name, context in analyzer.function_contexts.items()
        if context.is_entry_point
    ]


def build_call_chains(analyzer):
    """Build call chains for all functions"""
    
    for func_name, context in analyzer.function_contexts.items():
        context.max_call_chain = build_call_chain(analyzer, func_name, set())
        context.call_depth = len(context.max_call_chain) - 1


def build_call_chain(analyzer, function_name: str, visited: Set[str]) -> List[str]:
    """Build the maximum call chain from a function"""
    if function_name in visited or function_name not in analyzer.function_contexts:
        return [function_name]
    
    visited.add(function_name)
    context = analyzer.function_contexts[function_name]
    
    max_chain = [function_name]
    for called_func in context.function_calls:
        if called_func not in visited:
            chain = build_call_chain(analyzer, called_func, visited.copy())
            if len(chain) > len(max_chain) - 1:
                max_chain = [function_name] + chain
    
    return max_chain


def create_health_dashboard(results: AnalysisResults) -> Dict[str, Any]:
    """Create comprehensive health dashboard data"""
    
    dashboard = {
        "health_score": results.health_score,
        "health_grade": results.health_grade,
        "risk_level": results.risk_level,
        "technical_debt_hours": results.technical_debt_hours,
        
        # Issue summary
        "issues_summary": {
            "total_issues": len(results.issues),
            "by_severity": results.issues_by_severity,
            "by_type": results.issues_by_type,
            "automated_fixes_available": len(results.automated_resolutions)
        },
        
        # Function metrics
        "function_metrics": {
            "total_functions": len(results.function_contexts),
            "entry_points": len(results.entry_points),
            "dead_functions": len(results.dead_functions),
            "most_important": results.most_important_functions[:5]  # Top 5
        },
        
        # Quality indicators
        "quality_indicators": {
            "maintainability_index": results.maintainability_metrics.get("maintainability_index", 0),
            "documentation_coverage": results.maintainability_metrics.get("documentation_coverage", 0),
            "average_complexity": results.complexity_metrics.get("average_cyclomatic_complexity", 0),
            "code_duplication": results.complexity_metrics.get("duplication_percentage", 0)
        },
        
        # Recommendations
        "recommendations": generate_health_recommendations(results),
        
        # Trends (placeholder for future implementation)
        "trends": {
            "health_trend": "stable",
            "issue_trend": "improving",
            "debt_trend": "stable"
        }
    }
    
    return dashboard


def generate_health_recommendations(results: AnalysisResults) -> List[Dict[str, Any]]:
    """Generate actionable health recommendations"""
    
    recommendations = []
    
    # Critical issues
    critical_count = results.issues_by_severity.get("critical", 0)
    if critical_count > 0:
        recommendations.append({
            "priority": "critical",
            "category": "issues",
            "title": "Address Critical Issues",
            "description": f"{critical_count} critical issues require immediate attention",
            "action": "Review and fix critical issues in issue list",
            "impact": "high"
        })
    
    # Technical debt
    if results.technical_debt_hours > 40:
        recommendations.append({
            "priority": "high",
            "category": "debt",
            "title": "Reduce Technical Debt",
            "description": f"{results.technical_debt_hours:.1f} hours of technical debt detected",
            "action": "Plan refactoring sprint to address major issues",
            "impact": "medium"
        })
    
    # Dead code
    dead_count = len(results.dead_functions)
    if dead_count > 5:
        recommendations.append({
            "priority": "medium",
            "category": "cleanup",
            "title": "Remove Dead Code",
            "description": f"{dead_count} unused functions found",
            "action": "Review and remove unused functions",
            "impact": "low"
        })
    
    # Documentation
    doc_coverage = results.maintainability_metrics.get("documentation_coverage", 0)
    if doc_coverage < 50:
        recommendations.append({
            "priority": "medium",
            "category": "documentation",
            "title": "Improve Documentation",
            "description": f"Only {doc_coverage:.1f}% of functions are documented",
            "action": "Add docstrings to undocumented functions",
            "impact": "medium"
        })
    
    # Automated fixes
    auto_fixes = len(results.automated_resolutions)
    if auto_fixes > 0:
        recommendations.append({
            "priority": "low",
            "category": "automation",
            "title": "Apply Automated Fixes",
            "description": f"{auto_fixes} issues can be automatically resolved",
            "action": "Review and apply high-confidence automated fixes",
            "impact": "low"
        })
    
    return recommendations


def get_codebase_summary(codebase) -> str:
    """Generate a brief summary of the codebase"""
    
    file_count = len(codebase.source_files)
    
    # Count functions and classes
    function_count = 0
    class_count = 0
    
    for source_file in codebase.source_files:
        for symbol in source_file.symbols:
            if hasattr(symbol, '__class__'):
                if 'Function' in str(symbol.__class__):
                    function_count += 1
                elif 'Class' in str(symbol.__class__):
                    class_count += 1
    
    # Detect primary languages
    languages = set()
    for source_file in codebase.source_files:
        if hasattr(source_file, 'language') and source_file.language:
            languages.add(source_file.language)
    
    language_str = ", ".join(sorted(languages)) if languages else "Multiple languages"
    
    return f"Codebase with {file_count} files, {function_count} functions, {class_count} classes. Primary languages: {language_str}."

