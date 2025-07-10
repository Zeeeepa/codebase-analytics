"""
Helper functions for comprehensive codebase analysis.
"""

from typing import Dict, Any, List


def has_error_handling(function) -> bool:
    """Check if function has error handling (try-catch blocks)."""
    if not hasattr(function, 'code_block') or not function.code_block:
        return False
    
    source = getattr(function.code_block, 'source', '')
    return 'try:' in source or 'except' in source or 'finally:' in source


def has_potential_null_references(function) -> bool:
    """Check for potential null references in function."""
    if not hasattr(function, 'code_block') or not function.code_block:
        return False
    
    source = getattr(function.code_block, 'source', '')
    return has_potential_null_references_in_source(source)


def has_potential_null_references_in_source(source: str) -> bool:
    """Check for potential null references in source code."""
    # Look for attribute access without null checks
    patterns = ['.', '->']
    for pattern in patterns:
        if pattern in source and 'if' not in source and 'None' not in source:
            return True
    return False


def has_unhandled_critical_operations(function) -> bool:
    """Check for unhandled critical operations."""
    if not hasattr(function, 'code_block') or not function.code_block:
        return False
    
    source = getattr(function.code_block, 'source', '')
    critical_operations = ['open(', 'requests.', 'json.loads', 'int(', 'float(', 'dict[', 'list[']
    
    for op in critical_operations:
        if op in source and not is_wrapped_in_try_catch(source, op):
            return True
    return False


def is_wrapped_in_try_catch(source: str, operation: str) -> bool:
    """Check if operation is wrapped in try-catch."""
    lines = source.split('\n')
    for i, line in enumerate(lines):
        if operation in line:
            # Look backwards for try statement
            for j in range(i, -1, -1):
                if 'try:' in lines[j]:
                    return True
                if lines[j].strip() and not lines[j].startswith(' ') and not lines[j].startswith('\t'):
                    break
    return False


def is_special_function(function) -> bool:
    """Check if function is special (main, __init__, etc.)."""
    special_names = ['main', '__init__', '__main__', '__enter__', '__exit__', '__call__']
    return any(name in function.name for name in special_names)


def analyze_control_flow(function) -> Dict[str, Any]:
    """Analyze control flow complexity and patterns."""
    control_flow = {
        "if_statements": 0,
        "loops": 0,
        "try_catch_blocks": 0,
        "nested_depth": 0,
        "patterns": []
    }
    
    if not hasattr(function, 'code_block') or not function.code_block:
        return control_flow
    
    source = getattr(function.code_block, 'source', '')
    lines = source.split('\n')
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('if ') or stripped.startswith('elif '):
            control_flow["if_statements"] += 1
            control_flow["patterns"].append("conditional")
        elif stripped.startswith('for ') or stripped.startswith('while '):
            control_flow["loops"] += 1
            control_flow["patterns"].append("loop")
        elif stripped.startswith('try:'):
            control_flow["try_catch_blocks"] += 1
            control_flow["patterns"].append("error_handling")
    
    return control_flow


def analyze_error_patterns(function) -> Dict[str, Any]:
    """Analyze error handling patterns in function."""
    error_patterns = {
        "has_try_catch": False,
        "has_specific_exceptions": False,
        "has_generic_except": False,
        "has_finally": False,
        "error_handling_score": 0
    }
    
    if not hasattr(function, 'code_block') or not function.code_block:
        return error_patterns
    
    source = getattr(function.code_block, 'source', '')
    
    if 'try:' in source:
        error_patterns["has_try_catch"] = True
        error_patterns["error_handling_score"] += 2
    
    if 'except ' in source and 'Exception' not in source:
        error_patterns["has_specific_exceptions"] = True
        error_patterns["error_handling_score"] += 2
    
    if 'except:' in source or 'except Exception' in source:
        error_patterns["has_generic_except"] = True
        error_patterns["error_handling_score"] -= 1
    
    if 'finally:' in source:
        error_patterns["has_finally"] = True
        error_patterns["error_handling_score"] += 1
    
    return error_patterns


def analyze_performance_indicators(function) -> Dict[str, Any]:
    """Analyze performance indicators in function."""
    performance = {
        "nested_loops": 0,
        "database_calls": 0,
        "file_operations": 0,
        "network_calls": 0,
        "potential_bottlenecks": []
    }
    
    if not hasattr(function, 'code_block') or not function.code_block:
        return performance
    
    source = getattr(function.code_block, 'source', '')
    
    # Count nested loops (simplified)
    loop_depth = 0
    max_depth = 0
    for line in source.split('\n'):
        stripped = line.strip()
        if stripped.startswith('for ') or stripped.startswith('while '):
            loop_depth += 1
            max_depth = max(max_depth, loop_depth)
        elif not stripped or not line.startswith(' '):
            loop_depth = 0
    
    performance["nested_loops"] = max_depth
    if max_depth > 2:
        performance["potential_bottlenecks"].append("deep_nested_loops")
    
    # Check for database operations
    db_patterns = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', '.execute(', '.query(']
    for pattern in db_patterns:
        if pattern in source:
            performance["database_calls"] += source.count(pattern)
    
    # Check for file operations
    file_patterns = ['open(', 'read(', 'write(', 'close()']
    for pattern in file_patterns:
        if pattern in source:
            performance["file_operations"] += source.count(pattern)
    
    # Check for network calls
    network_patterns = ['requests.', 'urllib.', 'http.', 'socket.']
    for pattern in network_patterns:
        if pattern in source:
            performance["network_calls"] += source.count(pattern)
    
    if performance["database_calls"] > 5:
        performance["potential_bottlenecks"].append("many_database_calls")
    
    if performance["network_calls"] > 3:
        performance["potential_bottlenecks"].append("many_network_calls")
    
    return performance


def hop_through_imports(dep):
    """Hop through imports to find the root symbol source."""
    # This is a simplified implementation
    # In a real implementation, you would follow the import chain
    return dep

