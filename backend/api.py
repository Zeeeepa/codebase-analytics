
from flask import Flask, jsonify, request
from flask_cors import CORS
import json
from typing import Dict, Any, Optional
from analysis import analyze_codebase, ComprehensiveCodebaseAnalyzer
from visualize import generate_repository_tree, generate_visualization_data, create_interactive_ui
import os
app = Flask(__name__)
CORS(app)
# In-memory cache for analysis results
analysis_cache = {}
@app.route('/analyze/<username>/<repo_name>', methods=['GET'])
def analyze_repository(username: str, repo_name: str):
    """
    MAIN ANALYSIS ENDPOINT - Complete comprehensive analysis
    Returns: All function contexts, issues, dead code, metrics, entry points, etc.
    """
    try:
        # Check cache first
        cache_key = f"{username}/{repo_name}"
        if cache_key in analysis_cache:
            return jsonify(analysis_cache[cache_key])
        
        # Load codebase (this would integrate with graph-sitter)
        codebase = load_codebase(username, repo_name)
        
        if not codebase:
            return jsonify({"error": "Repository not found"}), 404
        
        # Perform comprehensive analysis
        analysis_results = analyze_codebase(codebase)
        
        # Enhance with additional context
        enhanced_results = enhance_analysis_results(analysis_results, codebase)
        
        # Cache results
        analysis_cache[cache_key] = enhanced_results
        
        return jsonify(enhanced_results)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/visualize/<username>/<repo_name>', methods=['GET'])
def visualize_repository(username: str, repo_name: str):
    """
    MAIN VISUALIZATION ENDPOINT - Interactive visualization data
    Returns: Repository tree, issue counts, symbol trees, interactive UI data
    """
    try:
        # Check cache first
        cache_key = f"{username}/{repo_name}"
        if cache_key not in analysis_cache:
            # Trigger analysis if not cached
            analyze_repository(username, repo_name)
        
        analysis = analysis_cache.get(cache_key, {})
        
        # Generate visualization data
        visualization_data = generate_visualization_data(analysis)
        
        # Add interactive UI components
        ui_data = create_interactive_ui(analysis)
        
        # Combine all visualization elements
        complete_visualization = {
            **visualization_data,
            "ui_components": ui_data,
            "interactive_features": {
                "clickable_tree": True,
                "issue_badges": True,
                "symbol_navigation": True,
                "context_panels": True,
                "dead_code_visualization": True,
                "call_graph_interactive": True
            }
        }
        
        return jsonify(complete_visualization)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# Helper functions
def load_codebase(username: str, repo_name: str):
    """Load codebase using graph-sitter integration"""
    # This would integrate with the actual graph-sitter library
    # For now, return a mock codebase with comprehensive data
    class MockFunction:
        def __init__(self, name, filepath, source="", parameters=None, usages=None, function_calls=None, call_sites=None, dependencies=None):
            self.name = name
            self.filepath = filepath
            self.source = source
            self.parameters = parameters or []
            self.usages = usages or []
            self.function_calls = function_calls or []
            self.call_sites = call_sites or []
            self.dependencies = dependencies or []
            self.start_point = (1, 0)
            self.end_point = (10, 0)
    
    class MockClass:
        def __init__(self, name, filepath, superclasses=None):
            self.name = name
            self.filepath = filepath
            self.superclasses = superclasses or []
            self.start_point = (1, 0)
    
    class MockFile:
        def __init__(self, filepath, source=""):
            self.filepath = filepath
            self.source = source
            self.imports = []
    
    class MockParam:
        def __init__(self, name, param_type="unknown"):
            self.name = name
            self.type = param_type
    
    class MockUsage:
        def __init__(self, source, filepath, line=1):
            self.usage_symbol = type('MockSymbol', (), {
                'source': source,
                'filepath': filepath,
                'start_point': [line, 0]
            })
    
    class MockCall:
        def __init__(self, name, parent_function=None):
            self.name = name
            self.parent_function = parent_function
            self.start_point = (1, 0)
    
    class MockCodebase:
        def __init__(self):
            # Create comprehensive mock functions with various scenarios
            self.functions = [
                MockFunction(
                    name="process_data",
                    filepath="src/main.py",
                    source='''def process_data(data, unused_param):
    """Process input data and return results"""
    if not data:
        return None
    
    result = validate_input(data)
    if result:
        total = calculate_total(data.get('items', []))
        return {"total": total, "processed": True}
    return {"error": "Invalid data"}''',
                    parameters=[
                        MockParam("data", "dict"),
                        MockParam("unused_param", "str")
                    ],
                    usages=[
                        MockUsage("process_data(user_data)", "src/app.py", 25)
                    ],
                    function_calls=[
                        MockCall("validate_input"),
                        MockCall("calculate_total")
                    ],
                    call_sites=[
                        MockCall("process_data", type('MockFunc', (), {'name': 'main'}))
                    ]
                ),
                MockFunction(
                    name="validate_input",
                    filepath="src/validation.py",
                    source='''def validate_input(data):
    """Validate input data structure"""
    if not isinstance(data, dict):
        return False
    
    required_fields = ['id', 'name', 'value']
    for field in required_fields:
        if field not in data:
            return False
    
    return True''',
                    parameters=[MockParam("data", "dict")],
                    usages=[
                        MockUsage("validate_input(data)", "src/main.py", 8),
                        MockUsage("validate_input(user_input)", "src/app.py", 15)
                    ],
                    function_calls=[],
                    call_sites=[
                        MockCall("validate_input", type('MockFunc', (), {'name': 'process_data'})),
                        MockCall("validate_input", type('MockFunc', (), {'name': 'handle_request'}))
                    ]
                ),
                MockFunction(
                    name="calculate_total",
                    filepath="src/utils.py",
                    source='''def calculate_total(items):
    """Calculate total value of items"""
    total = 0
    for item in items:
        if hasattr(item, 'value') and item.value > 0:
            total += item.value
    return total''',
                    parameters=[MockParam("items", "list")],
                    usages=[
                        MockUsage("calculate_total(data['items'])", "src/main.py", 12)
                    ],
                    function_calls=[],
                    call_sites=[
                        MockCall("calculate_total", type('MockFunc', (), {'name': 'process_data'}))
                    ]
                ),
                MockFunction(
                    name="unused_helper",
                    filepath="src/helpers.py",
                    source='''def unused_helper(param):
    """This function is never used - dead code"""
    return param * 2''',
                    parameters=[MockParam("param", "int")],
                    usages=[],  # No usages - dead code
                    function_calls=[],
                    call_sites=[]
                ),
                MockFunction(
                    name="main",
                    filepath="src/app.py",
                    source='''def main():
    """Main entry point"""
    user_data = get_user_input()
    result = process_data(user_data)
    
    if result:
        print(f"Processing complete: {result}")
    else:
        print("Processing failed")''',
                    parameters=[],
                    usages=[],
                    function_calls=[
                        MockCall("get_user_input"),
                        MockCall("process_data")
                    ],
                    call_sites=[]
                )
            ]
            
            self.classes = [
                MockClass(
                    name="DataProcessor",
                    filepath="src/processor.py",
                    superclasses=[
                        type('MockClass', (), {'name': 'BaseProcessor'}),
                        type('MockClass', (), {'name': 'Validator'})
                    ]
                )
            ]
            
            self.files = [
                MockFile("src/main.py", "# Main application file"),
                MockFile("src/validation.py", "# Input validation utilities"),
                MockFile("src/utils.py", "# Utility functions"),
                MockFile("src/helpers.py", "# Helper functions"),
                MockFile("src/app.py", "# Application entry point"),
                MockFile("src/processor.py", "# Data processing classes"),
                MockFile("tests/test_main.py", "# Test file")
            ]
    
    return MockCodebase()
def enhance_analysis_results(analysis_results: Dict[str, Any], codebase) -> Dict[str, Any]:
    """Enhance analysis results with additional context and metadata"""
    
    # Add comprehensive metadata
    enhanced = {
        **analysis_results,
        "metadata": {
            "analysis_timestamp": "2024-06-21T17:00:00Z",
            "repository": f"{codebase.__class__.__name__}",
            "analysis_version": "1.0.0",
            "features_analyzed": [
                "function_contexts",
                "error_detection", 
                "dead_code_analysis",
                "call_graph_metrics",
                "halstead_metrics",
                "dependency_analysis",
                "entry_point_detection"
            ]
        },
        
        # Enhanced function contexts with additional details
        "enhanced_function_contexts": enhance_function_contexts(analysis_results.get('function_contexts', {})),
        
        # Detailed issue analysis
        "issue_analysis": {
            "by_category": categorize_issues_by_type(analysis_results.get('issues_by_severity', {})),
            "severity_distribution": calculate_severity_distribution(analysis_results.get('issues_by_severity', {})),
            "file_issue_density": calculate_file_issue_density(analysis_results.get('issues_by_severity', {})),
            "most_problematic_files": identify_most_problematic_files(analysis_results.get('issues_by_severity', {}))
        },
        
        # Enhanced dead code analysis
        "enhanced_dead_code": enhance_dead_code_analysis(analysis_results.get('dead_code_analysis', {})),
        
        # Code quality metrics
        "quality_metrics": {
            "overall_score": calculate_quality_score(analysis_results),
            "maintainability_index": calculate_maintainability_index(analysis_results),
            "technical_debt_ratio": calculate_technical_debt_ratio(analysis_results)
        }
    }
    
    return enhanced
def enhance_function_contexts(function_contexts: Dict[str, Any]) -> Dict[str, Any]:
    """Add additional context to function analysis"""
    enhanced = {}
    
    for func_name, context in function_contexts.items():
        enhanced[func_name] = {
            **context,
            "risk_assessment": assess_function_risk(context),
            "refactoring_suggestions": generate_refactoring_suggestions(context),
            "impact_analysis": analyze_function_impact(context),
            "test_recommendations": suggest_test_coverage(context)
        }
    
    return enhanced
def categorize_issues_by_type(issues_by_severity: Dict[str, Any]) -> Dict[str, Any]:
    """Categorize issues by their type for better analysis"""
    categories = {
        "implementation_errors": [],
        "function_issues": [],
        "exception_handling": [],
        "code_quality": [],
        "formatting_style": [],
        "runtime_risks": []
    }
    
    type_to_category = {
        "null_reference": "implementation_errors",
        "type_mismatch": "implementation_errors",
        "undefined_variable": "implementation_errors",
        "missing_return": "implementation_errors",
        "unreachable_code": "implementation_errors",
        
        "misspelled_function": "function_issues",
        "wrong_parameter_count": "function_issues",
        "parameter_type_mismatch": "function_issues",
        "unused_parameter": "function_issues",
        
        "improper_exception_handling": "exception_handling",
        "missing_error_handling": "exception_handling",
        "unsafe_assertion": "exception_handling",
        "resource_leak": "exception_handling",
        
        "code_duplication": "code_quality",
        "inefficient_pattern": "code_quality",
        "magic_number": "code_quality",
        "long_function": "code_quality",
        "deep_nesting": "code_quality",
        
        "inconsistent_naming": "formatting_style",
        "missing_documentation": "formatting_style",
        "inconsistent_indentation": "formatting_style",
        "import_organization": "formatting_style",
        
        "division_by_zero": "runtime_risks",
        "array_bounds": "runtime_risks",
        "infinite_loop": "runtime_risks",
        "stack_overflow": "runtime_risks",
        "concurrency_issue": "runtime_risks"
    }
    
    for severity, issues in issues_by_severity.items():
        for issue in issues:
            category = type_to_category.get(issue.get('type', ''), 'other')
            if category in categories:
                categories[category].append({**issue, "severity": severity})
    
    return categories
def calculate_severity_distribution(issues_by_severity: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate distribution of issue severities"""
    total_issues = sum(len(issues) for issues in issues_by_severity.values())
    
    if total_issues == 0:
        return {"critical": 0, "major": 0, "minor": 0, "info": 0}
    
    return {
        severity: (len(issues) / total_issues) * 100
        for severity, issues in issues_by_severity.items()
    }
def calculate_file_issue_density(issues_by_severity: Dict[str, Any]) -> Dict[str, float]:
    """Calculate issue density per file"""
    file_issues = {}
    
    for severity, issues in issues_by_severity.items():
        for issue in issues:
            filepath = issue.get('filepath', 'unknown')
            if filepath not in file_issues:
                file_issues[filepath] = 0
            file_issues[filepath] += 1
    
    return file_issues
def identify_most_problematic_files(issues_by_severity: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify files with the most issues"""
    file_issues = calculate_file_issue_density(issues_by_severity)
    
    sorted_files = sorted(file_issues.items(), key=lambda x: x[1], reverse=True)
    
    return [
        {"filepath": filepath, "issue_count": count}
        for filepath, count in sorted_files[:10]  # Top 10 most problematic files
    ]
def enhance_dead_code_analysis(dead_code_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance dead code analysis with additional insights"""
    return {
        **dead_code_analysis,
        "removal_priority": prioritize_dead_code_removal(dead_code_analysis),
        "dependency_impact": analyze_dead_code_dependencies(dead_code_analysis),
        "cleanup_recommendations": generate_cleanup_recommendations(dead_code_analysis)
    }
def assess_function_risk(context: Dict[str, Any]) -> str:
    """Assess the risk level of a function"""
    risk_factors = 0
    
    # High complexity
    if context.get('complexity_score', 0) > 20:
        risk_factors += 2
    
    # Many issues
    if len(context.get('issues', [])) > 3:
        risk_factors += 2
    
    # Dead code
    if context.get('is_dead_code', False):
        risk_factors += 1
    
    # Long call chain
    if len(context.get('max_call_chain', [])) > 5:
        risk_factors += 1
    
    if risk_factors >= 4:
        return "high"
    elif risk_factors >= 2:
        return "medium"
    else:
        return "low"
def generate_refactoring_suggestions(context: Dict[str, Any]) -> List[str]:
    """Generate refactoring suggestions for a function"""
    suggestions = []
    
    if context.get('complexity_score', 0) > 20:
        suggestions.append("Consider breaking this function into smaller, more focused functions")
    
    if len(context.get('issues', [])) > 0:
        suggestions.append("Address the identified code issues to improve quality")
    
    if context.get('is_dead_code', False):
        suggestions.append("This function appears to be unused and could be removed")
    
    if len(context.get('parameters', [])) > 5:
        suggestions.append("Consider using a configuration object instead of many parameters")
    
    return suggestions
def analyze_function_impact(context: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the impact of modifying a function"""
    return {
        "direct_dependents": len(context.get('called_by', [])),
        "indirect_impact": len(context.get('max_call_chain', [])),
        "modification_risk": assess_function_risk(context),
        "testing_priority": "high" if len(context.get('called_by', [])) > 3 else "medium"
    }
def suggest_test_coverage(context: Dict[str, Any]) -> List[str]:
    """Suggest test coverage improvements"""
    suggestions = []
    
    if context.get('is_entry_point', False):
        suggestions.append("Add integration tests for this entry point function")
    
    if len(context.get('called_by', [])) > 2:
        suggestions.append("Add unit tests due to high usage")
    
    if context.get('complexity_score', 0) > 15:
        suggestions.append("Add comprehensive tests due to high complexity")
    
    return suggestions
def calculate_quality_score(analysis_results: Dict[str, Any]) -> float:
    """Calculate overall code quality score (0-10)"""
    base_score = 10.0
    
    # Deduct for issues
    issues = analysis_results.get('issues_by_severity', {})
    critical_issues = len(issues.get('critical', []))
    major_issues = len(issues.get('major', []))
    minor_issues = len(issues.get('minor', []))
    
    base_score -= (critical_issues * 2.0)
    base_score -= (major_issues * 1.0)
    base_score -= (minor_issues * 0.5)
    
    # Deduct for dead code
    dead_code_count = analysis_results.get('dead_code_analysis', {}).get('total_dead_functions', 0)
    base_score -= (dead_code_count * 0.5)
    
    return max(0.0, min(10.0, base_score))
def calculate_maintainability_index(analysis_results: Dict[str, Any]) -> float:
    """Calculate maintainability index"""
    # Simplified maintainability calculation
    halstead = analysis_results.get('halstead_metrics', {})
    volume = halstead.get('volume', 1)
    complexity = sum(ctx.get('complexity_score', 0) for ctx in analysis_results.get('function_contexts', {}).values())
    
    # Simplified formula
    maintainability = max(0, 171 - 5.2 * (volume ** 0.23) - 0.23 * complexity - 16.2 * (volume ** 0.5))
    return min(100, maintainability)
def calculate_technical_debt_ratio(analysis_results: Dict[str, Any]) -> float:
    """Calculate technical debt ratio"""
    total_issues = sum(len(issues) for issues in analysis_results.get('issues_by_severity', {}).values())
    total_functions = len(analysis_results.get('function_contexts', {}))
    
    if total_functions == 0:
        return 0.0
    
    return (total_issues / total_functions) * 100
# Additional helper functions for dead code analysis
def prioritize_dead_code_removal(dead_code_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Prioritize dead code items for removal"""
    items = dead_code_analysis.get('dead_code_items', [])
    
    # Sort by blast radius (smaller blast radius = higher priority for removal)
    return sorted(items, key=lambda x: len(x.get('blast_radius', [])))
def analyze_dead_code_dependencies(dead_code_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze dependencies of dead code"""
    return {
        "safe_to_remove": [],
        "requires_careful_review": [],
        "dependency_chains": []
    }
def generate_cleanup_recommendations(dead_code_analysis: Dict[str, Any]) -> List[str]:
    """Generate recommendations for cleaning up dead code"""
    return [
        "Remove unused functions with no dependencies first",
        "Review functions with external dependencies before removal",
        "Update documentation after removing dead code",
        "Run comprehensive tests after cleanup"
    ]
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)