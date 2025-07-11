"""
Additional metrics and analysis functions for comprehensive codebase analysis.
"""

import math
from typing import Dict, List, Any
from graph_sitter.core.codebase import Codebase
from .models import FunctionMetrics, ClassMetrics, FileMetrics
from .analysis import calculate_cyclomatic_complexity, calculate_doi, calculate_cbo, calculate_lcom


def calculate_comprehensive_metrics(codebase: Codebase) -> Dict[str, Any]:
    """Calculate comprehensive metrics for functions, classes, and files."""
    function_metrics = []
    class_metrics = []
    file_metrics = []
    
    try:
        for file in codebase.files:
            # File metrics
            file_loc = len(file.source.splitlines()) if hasattr(file, 'source') else 0
            file_metrics.append(FileMetrics(
                file_path=file.filepath,
                lines_of_code=file_loc,
                functions_count=len(file.functions),
                classes_count=len(file.classes),
                imports_count=len(file.imports),
                complexity_score=sum(calculate_cyclomatic_complexity(f) for f in file.functions) / max(len(file.functions), 1),
                maintainability_index=calculate_maintainability_index(file),
                importance_score=calculate_file_importance(file, codebase)
            ))
            
            # Function metrics
            for func in file.functions:
                try:
                    complexity = calculate_cyclomatic_complexity(func)
                    halstead_metrics = calculate_halstead_metrics(func)
                    
                    function_metrics.append(FunctionMetrics(
                        function_name=func.name,
                        file_path=file.filepath,
                        line_number=getattr(func, 'line_number', None),
                        cyclomatic_complexity=complexity,
                        maintainability_index=calculate_function_maintainability(func),
                        lines_of_code=len(func.source.splitlines()) if hasattr(func, 'source') else 0,
                        halstead_volume=halstead_metrics.get('volume', 0.0),
                        halstead_difficulty=halstead_metrics.get('difficulty', 0.0),
                        halstead_effort=halstead_metrics.get('effort', 0.0),
                        parameters_count=len(func.parameters),
                        return_statements_count=len(func.return_statements),
                        importance_score=calculate_function_importance(func, codebase),
                        usage_frequency=len(func.call_sites)
                    ))
                except Exception as e:
                    print(f"Error calculating metrics for function {func.name}: {e}")
                    continue
            
            # Class metrics
            for cls in file.classes:
                try:
                    class_metrics.append(ClassMetrics(
                        class_name=cls.name,
                        file_path=file.filepath,
                        line_number=getattr(cls, 'line_number', None),
                        methods_count=len(cls.methods),
                        attributes_count=len(cls.attributes),
                        depth_of_inheritance=calculate_doi(cls),
                        coupling_between_objects=calculate_cbo(cls),
                        lack_of_cohesion=calculate_lcom(cls),
                        lines_of_code=len(cls.source.splitlines()) if hasattr(cls, 'source') else 0,
                        importance_score=calculate_class_importance(cls, codebase)
                    ))
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


def get_most_important_functions(codebase: Codebase, limit: int = 20) -> List[Dict[str, Any]]:
    """Get the most important functions based on various metrics."""
    important_functions = []
    
    try:
        for file in codebase.files:
            for func in file.functions:
                try:
                    # Calculate importance score based on multiple factors
                    complexity = calculate_cyclomatic_complexity(func)
                    usage_frequency = len(func.call_sites)
                    parameters_count = len(func.parameters)
                    loc = len(func.source.splitlines()) if hasattr(func, 'source') else 0
                    
                    # Weighted importance score
                    importance_score = (
                        complexity * 0.3 +
                        usage_frequency * 0.4 +
                        parameters_count * 0.1 +
                        (loc / 10) * 0.2
                    )
                    
                    important_functions.append({
                        'function_name': func.name,
                        'file_path': file.filepath,
                        'importance_score': importance_score,
                        'complexity': complexity,
                        'usage_frequency': usage_frequency,
                        'parameters_count': parameters_count,
                        'lines_of_code': loc,
                        'line_number': getattr(func, 'line_number', None)
                    })
                    
                except Exception as e:
                    print(f"Error analyzing function {func.name}: {e}")
                    continue
    
    except Exception as e:
        print(f"Error in get_most_important_functions: {e}")
    
    # Sort by importance score and return top functions
    important_functions.sort(key=lambda x: x['importance_score'], reverse=True)
    return important_functions[:limit]


def calculate_halstead_metrics(func) -> Dict[str, float]:
    """Calculate Halstead complexity metrics for a function."""
    try:
        if not hasattr(func, 'source') or not func.source:
            return {'volume': 0.0, 'difficulty': 0.0, 'effort': 0.0}
        
        # Simple approximation of Halstead metrics
        # In a real implementation, you'd parse operators and operands
        source_lines = func.source.splitlines()
        operators = 0
        operands = 0
        
        for line in source_lines:
            # Count basic operators
            operators += line.count('+') + line.count('-') + line.count('*') + line.count('/')
            operators += line.count('=') + line.count('==') + line.count('!=')
            operators += line.count('<') + line.count('>') + line.count('<=') + line.count('>=')
            
            # Count operands (rough approximation)
            words = line.split()
            operands += len([w for w in words if w.isidentifier()])
        
        # Halstead metrics calculations
        vocabulary = operators + operands
        length = operators + operands
        volume = length * math.log2(vocabulary) if vocabulary > 0 else 0
        difficulty = (operators / 2) * (operands / max(operands, 1)) if operands > 0 else 0
        effort = difficulty * volume
        
        return {
            'volume': volume,
            'difficulty': difficulty,
            'effort': effort
        }
        
    except Exception as e:
        print(f"Error calculating Halstead metrics: {e}")
        return {'volume': 0.0, 'difficulty': 0.0, 'effort': 0.0}


def calculate_function_maintainability(func) -> float:
    """Calculate maintainability index for a function."""
    try:
        complexity = calculate_cyclomatic_complexity(func)
        loc = len(func.source.splitlines()) if hasattr(func, 'source') else 1
        halstead_volume = calculate_halstead_metrics(func)['volume']
        
        # Maintainability Index formula (simplified)
        mi = max(0, (171 - 5.2 * math.log(halstead_volume) - 0.23 * complexity - 16.2 * math.log(loc)) * 100 / 171)
        return mi
        
    except Exception as e:
        print(f"Error calculating maintainability: {e}")
        return 0.0


def calculate_maintainability_index(file) -> float:
    """Calculate maintainability index for a file."""
    try:
        if not file.functions:
            return 100.0  # Empty file is perfectly maintainable
        
        # Average maintainability of all functions in the file
        total_mi = sum(calculate_function_maintainability(func) for func in file.functions)
        return total_mi / len(file.functions)
        
    except Exception as e:
        print(f"Error calculating file maintainability: {e}")
        return 0.0


def calculate_file_importance(file, codebase) -> float:
    """Calculate importance score for a file."""
    try:
        # Factors: number of functions, classes, imports, dependencies
        functions_count = len(file.functions)
        classes_count = len(file.classes)
        imports_count = len(file.imports)
        
        # Count how many other files depend on this file
        dependents_count = 0
        for other_file in codebase.files:
            if other_file != file:
                for imp in other_file.imports:
                    if file.name in str(imp):
                        dependents_count += 1
        
        # Weighted importance score
        importance = (
            functions_count * 2 +
            classes_count * 3 +
            imports_count * 1 +
            dependents_count * 4
        )
        
        return min(100.0, importance)  # Cap at 100
        
    except Exception as e:
        print(f"Error calculating file importance: {e}")
        return 0.0


def calculate_function_importance(func, codebase) -> float:
    """Calculate importance score for a function."""
    try:
        complexity = calculate_cyclomatic_complexity(func)
        usage_frequency = len(func.call_sites)
        parameters_count = len(func.parameters)
        
        # Weighted importance score
        importance = (
            complexity * 0.3 +
            usage_frequency * 0.5 +
            parameters_count * 0.2
        )
        
        return min(100.0, importance)  # Cap at 100
        
    except Exception as e:
        print(f"Error calculating function importance: {e}")
        return 0.0


def calculate_class_importance(cls, codebase) -> float:
    """Calculate importance score for a class."""
    try:
        methods_count = len(cls.methods)
        attributes_count = len(cls.attributes)
        inheritance_depth = calculate_doi(cls)
        
        # Count usage of this class
        usage_count = 0
        for file in codebase.files:
            for func in file.functions:
                if hasattr(func, 'source') and cls.name in func.source:
                    usage_count += 1
        
        # Weighted importance score
        importance = (
            methods_count * 2 +
            attributes_count * 1 +
            inheritance_depth * 1.5 +
            usage_count * 3
        )
        
        return min(100.0, importance)  # Cap at 100
        
    except Exception as e:
        print(f"Error calculating class importance: {e}")
        return 0.0
