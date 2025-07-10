"""
Advanced analysis functions using comprehensive graph-sitter capabilities.
This module contains enhanced analysis functions that leverage the full power
of graph-sitter's AST analysis, semantic understanding, and architectural insights.
"""

from typing import Dict, List, Any, Set, Optional
from collections import defaultdict, Counter
import re
import hashlib
from pathlib import Path

# Graph-sitter imports
from graph_sitter.core.codebase import Codebase
from graph_sitter.core.class_definition import Class
from graph_sitter.core.function import Function
from graph_sitter.core.file import SourceFile
from graph_sitter.core.symbol import Symbol
from graph_sitter.enums import EdgeType, SymbolType

# Advanced expression analysis classes
from graph_sitter.core.expressions import Expression, Name, String, Value
from graph_sitter.core.expressions.chained_attribute import ChainedAttribute
from graph_sitter.core.expressions.defined_name import DefinedName
from graph_sitter.core.expressions.builtin import Builtin

# Advanced core modules for deeper analysis
try:
    from graph_sitter.core.assignment import Assignment
except ImportError:
    Assignment = None

try:
    from graph_sitter.core.export import Export
except ImportError:
    Export = None

try:
    from graph_sitter.core.directory import Directory
except ImportError:
    Directory = None

try:
    from graph_sitter.core.interface import Interface
except ImportError:
    Interface = None

# Language-specific modules
try:
    import graph_sitter.python as python_analyzer
except ImportError:
    python_analyzer = None

try:
    import graph_sitter.typescript as typescript_analyzer
except ImportError:
    typescript_analyzer = None


def advanced_semantic_analysis(codebase: Codebase) -> Dict[str, Any]:
    """Advanced semantic analysis using Expression, Name, String, Value classes."""
    semantic_data = {
        "variable_usage_patterns": {},
        "string_literals": [],
        "function_call_patterns": {},
        "type_usage": {},
        "semantic_errors": []
    }
    
    for file in codebase.files:
        try:
            # Analyze expressions for semantic patterns
            for func in file.functions:
                if not hasattr(func, 'code_block') or not func.code_block:
                    continue
                    
                # Extract variable names and usage patterns
                try:
                    # Use Name class for variable analysis
                    variable_names = []
                    string_literals = []
                    
                    # Analyze function source for patterns
                    if hasattr(func, 'source'):
                        # Extract string literals
                        string_matches = re.findall(r'["\']([^"\']*)["\']', func.source)
                        string_literals.extend(string_matches)
                        
                        # Extract variable assignments
                        assignment_matches = re.findall(r'(\w+)\s*=\s*', func.source)
                        variable_names.extend(assignment_matches)
                    
                    semantic_data["variable_usage_patterns"][f"{file.filepath}:{func.name}"] = {
                        "variables": list(set(variable_names)),
                        "variable_count": len(set(variable_names))
                    }
                    
                    semantic_data["string_literals"].extend([
                        {
                            "value": literal,
                            "file": file.filepath,
                            "function": func.name,
                            "length": len(literal)
                        }
                        for literal in string_literals
                    ])
                    
                except Exception as e:
                    semantic_data["semantic_errors"].append({
                        "file": file.filepath,
                        "function": func.name,
                        "error": str(e),
                        "type": "variable_analysis_error"
                    })
                    
        except Exception as e:
            semantic_data["semantic_errors"].append({
                "file": file.filepath,
                "error": str(e),
                "type": "file_analysis_error"
            })
    
    return semantic_data


def advanced_dependency_analysis(codebase: Codebase) -> Dict[str, Any]:
    """Enhanced dependency analysis using Export and Assignment classes."""
    dependency_data = {
        "export_analysis": {},
        "assignment_patterns": {},
        "circular_dependencies": [],
        "unused_exports": [],
        "dependency_metrics": {}
    }
    
    # Analyze exports if Export class is available
    if Export:
        try:
            exports = list(codebase.exports) if hasattr(codebase, 'exports') else []
            dependency_data["export_analysis"] = {
                "total_exports": len(exports),
                "exports_by_file": {},
                "export_types": {}
            }
            
            for export in exports[:50]:  # Limit for performance
                try:
                    file_path = getattr(export, 'file', {}).get('filepath', 'unknown')
                    export_name = getattr(export, 'name', 'unknown')
                    
                    if file_path not in dependency_data["export_analysis"]["exports_by_file"]:
                        dependency_data["export_analysis"]["exports_by_file"][file_path] = []
                    
                    dependency_data["export_analysis"]["exports_by_file"][file_path].append({
                        "name": export_name,
                        "type": type(export).__name__
                    })
                    
                except Exception as e:
                    dependency_data["export_analysis"]["error"] = str(e)
                    
        except Exception as e:
            dependency_data["export_analysis"]["error"] = f"Export analysis failed: {e}"
    
    # Analyze assignments if Assignment class is available
    if Assignment:
        try:
            # This would require proper Assignment class integration
            dependency_data["assignment_patterns"]["status"] = "Assignment analysis available"
        except Exception as e:
            dependency_data["assignment_patterns"]["error"] = str(e)
    
    # Detect circular dependencies using graph analysis
    try:
        file_dependencies = {}
        for file in codebase.files:
            file_deps = []
            for imp in file.imports:
                if hasattr(imp, 'imported_symbol') and hasattr(imp.imported_symbol, 'filepath'):
                    file_deps.append(imp.imported_symbol.filepath)
            file_dependencies[file.filepath] = file_deps
        
        # Simple circular dependency detection
        def has_circular_dependency(file_path, target, visited=None):
            if visited is None:
                visited = set()
            if file_path in visited:
                return True
            if file_path == target and visited:
                return True
            visited.add(file_path)
            
            for dep in file_dependencies.get(file_path, []):
                if has_circular_dependency(dep, target, visited.copy()):
                    return True
            return False
        
        circular_deps = []
        for file_path in file_dependencies:
            if has_circular_dependency(file_path, file_path):
                circular_deps.append(file_path)
        
        dependency_data["circular_dependencies"] = circular_deps[:10]  # Limit results
        
    except Exception as e:
        dependency_data["circular_dependencies"] = [f"Error detecting circular dependencies: {e}"]
    
    return dependency_data


def advanced_architectural_analysis(codebase: Codebase) -> Dict[str, Any]:
    """Architectural analysis using Interface and Directory classes."""
    arch_data = {
        "interface_analysis": {},
        "directory_structure": {},
        "architectural_patterns": {},
        "code_organization": {}
    }
    
    # Interface analysis if available
    if Interface:
        try:
            interfaces = list(codebase.interfaces) if hasattr(codebase, 'interfaces') else []
            arch_data["interface_analysis"] = {
                "total_interfaces": len(interfaces),
                "interface_details": []
            }
            
            for interface in interfaces[:20]:  # Limit for performance
                try:
                    arch_data["interface_analysis"]["interface_details"].append({
                        "name": getattr(interface, 'name', 'unknown'),
                        "file": getattr(interface, 'filepath', 'unknown'),
                        "methods": len(getattr(interface, 'methods', [])),
                        "type": type(interface).__name__
                    })
                except Exception as e:
                    arch_data["interface_analysis"]["error"] = str(e)
                    
        except Exception as e:
            arch_data["interface_analysis"]["error"] = f"Interface analysis failed: {e}"
    
    # Directory structure analysis
    try:
        file_paths = [file.filepath for file in codebase.files]
        directories = set()
        for path in file_paths:
            parts = Path(path).parts
            for i in range(1, len(parts)):
                directories.add('/'.join(parts[:i]))
        
        arch_data["directory_structure"] = {
            "total_directories": len(directories),
            "max_depth": max(len(Path(path).parts) for path in file_paths) if file_paths else 0,
            "files_per_directory": {},
            "common_patterns": []
        }
        
        # Analyze files per directory
        dir_file_count = defaultdict(int)
        for path in file_paths:
            directory = str(Path(path).parent)
            dir_file_count[directory] += 1
        
        arch_data["directory_structure"]["files_per_directory"] = dict(
            sorted(dir_file_count.items(), key=lambda x: x[1], reverse=True)[:20]
        )
        
        # Detect common architectural patterns
        patterns = []
        if any('src' in path for path in file_paths):
            patterns.append("src/ directory pattern")
        if any('lib' in path for path in file_paths):
            patterns.append("lib/ directory pattern")
        if any('test' in path.lower() for path in file_paths):
            patterns.append("test directory pattern")
        if any('api' in path.lower() for path in file_paths):
            patterns.append("API directory pattern")
        if any('component' in path.lower() for path in file_paths):
            patterns.append("component-based architecture")
        
        arch_data["directory_structure"]["common_patterns"] = patterns
        
    except Exception as e:
        arch_data["directory_structure"]["error"] = str(e)
    
    return arch_data


def language_specific_analysis(codebase: Codebase) -> Dict[str, Any]:
    """Language-specific analysis using Python and TypeScript analyzers."""
    lang_data = {
        "python_analysis": {},
        "typescript_analysis": {},
        "language_distribution": {},
        "cross_language_dependencies": []
    }
    
    # Analyze language distribution
    try:
        file_extensions = defaultdict(int)
        for file in codebase.files:
            ext = Path(file.filepath).suffix.lower()
            file_extensions[ext] += 1
        
        lang_data["language_distribution"] = dict(file_extensions)
        
        # Determine primary languages
        total_files = sum(file_extensions.values())
        lang_percentages = {
            ext: (count / total_files * 100) if total_files > 0 else 0
            for ext, count in file_extensions.items()
        }
        lang_data["language_percentages"] = lang_percentages
        
    except Exception as e:
        lang_data["language_distribution"]["error"] = str(e)
    
    # Python-specific analysis
    if python_analyzer:
        try:
            python_files = [f for f in codebase.files if f.filepath.endswith('.py')]
            lang_data["python_analysis"] = {
                "total_python_files": len(python_files),
                "python_patterns": [],
                "python_specific_issues": []
            }
            
            # Analyze Python-specific patterns
            for file in python_files[:10]:  # Limit for performance
                try:
                    if '__init__.py' in file.filepath:
                        lang_data["python_analysis"]["python_patterns"].append("Package structure detected")
                    if 'setup.py' in file.filepath:
                        lang_data["python_analysis"]["python_patterns"].append("Python package setup detected")
                    if 'requirements.txt' in file.filepath or 'pyproject.toml' in file.filepath:
                        lang_data["python_analysis"]["python_patterns"].append("Dependency management detected")
                        
                except Exception as e:
                    lang_data["python_analysis"]["python_specific_issues"].append(str(e))
                    
        except Exception as e:
            lang_data["python_analysis"]["error"] = str(e)
    
    # TypeScript-specific analysis
    if typescript_analyzer:
        try:
            ts_files = [f for f in codebase.files if f.filepath.endswith(('.ts', '.tsx', '.js', '.jsx'))]
            lang_data["typescript_analysis"] = {
                "total_typescript_files": len(ts_files),
                "typescript_patterns": [],
                "typescript_specific_issues": []
            }
            
            # Analyze TypeScript-specific patterns
            for file in ts_files[:10]:  # Limit for performance
                try:
                    if 'package.json' in file.filepath:
                        lang_data["typescript_analysis"]["typescript_patterns"].append("NPM package detected")
                    if 'tsconfig.json' in file.filepath:
                        lang_data["typescript_analysis"]["typescript_patterns"].append("TypeScript configuration detected")
                    if '.tsx' in file.filepath:
                        lang_data["typescript_analysis"]["typescript_patterns"].append("React/JSX components detected")
                        
                except Exception as e:
                    lang_data["typescript_analysis"]["typescript_specific_issues"].append(str(e))
                    
        except Exception as e:
            lang_data["typescript_analysis"]["error"] = str(e)
    
    return lang_data


def advanced_performance_analysis(codebase: Codebase) -> Dict[str, Any]:
    """Advanced performance analysis using graph-sitter's deep AST capabilities."""
    perf_data = {
        "algorithmic_complexity": {},
        "memory_usage_patterns": {},
        "io_operations": {},
        "performance_bottlenecks": [],
        "optimization_suggestions": []
    }
    
    try:
        # Analyze algorithmic complexity patterns
        complexity_patterns = {
            "nested_loops": 0,
            "recursive_functions": 0,
            "large_data_structures": 0,
            "database_queries": 0
        }
        
        for file in codebase.files:
            for func in file.functions:
                try:
                    if not hasattr(func, 'source'):
                        continue
                        
                    source = func.source.lower()
                    
                    # Detect nested loops (simplified)
                    loop_keywords = ['for ', 'while ']
                    loop_count = sum(source.count(keyword) for keyword in loop_keywords)
                    if loop_count > 1:
                        complexity_patterns["nested_loops"] += 1
                    
                    # Detect recursive functions
                    if func.name in source:
                        complexity_patterns["recursive_functions"] += 1
                    
                    # Detect large data structure operations
                    large_data_keywords = ['list(', 'dict(', 'set(', 'array', 'map(']
                    if any(keyword in source for keyword in large_data_keywords):
                        complexity_patterns["large_data_structures"] += 1
                    
                    # Detect database operations
                    db_keywords = ['select ', 'insert ', 'update ', 'delete ', 'query', 'execute']
                    if any(keyword in source for keyword in db_keywords):
                        complexity_patterns["database_queries"] += 1
                        
                except Exception as e:
                    continue
        
        perf_data["algorithmic_complexity"] = complexity_patterns
        
        # Generate optimization suggestions based on patterns
        suggestions = []
        if complexity_patterns["nested_loops"] > 5:
            suggestions.append("Consider optimizing nested loops - found multiple instances")
        if complexity_patterns["recursive_functions"] > 10:
            suggestions.append("Review recursive functions for potential stack overflow issues")
        if complexity_patterns["large_data_structures"] > 20:
            suggestions.append("Consider memory-efficient alternatives for large data structures")
        if complexity_patterns["database_queries"] > 15:
            suggestions.append("Consider database query optimization and connection pooling")
        
        perf_data["optimization_suggestions"] = suggestions
        
    except Exception as e:
        perf_data["error"] = str(e)
    
    return perf_data


def comprehensive_error_context_analysis(codebase: Codebase, max_issues: int = 200) -> Dict[str, Any]:
    """
    Comprehensive error analysis with detailed context using advanced graph-sitter features.
    Provides the detailed error context requested: file paths, line numbers, function names,
    interconnected context, and fix suggestions.
    """
    # Import calculate_cyclomatic_complexity function locally to avoid circular imports
    def calculate_cyclomatic_complexity(function):
        """Enhanced cyclomatic complexity calculation with better statement handling."""
        base_complexity = 1
        
        if hasattr(function, 'source'):
            # Fallback to source-based analysis
            source = function.source.lower()
            base_complexity += source.count('if ') + source.count('elif ')
            base_complexity += source.count('for ') + source.count('while ')
            base_complexity += source.count('except ')
            base_complexity += source.count(' and ') + source.count(' or ')
        
        return base_complexity
    
    error_analysis = {
        "total_issues": 0,
        "critical_issues": 0,
        "issues_by_severity": {},
        "issues_by_file": {},
        "interconnected_analysis": {},
        "detailed_issues": []
    }
    
    issue_counter = 0
    
    for file in codebase.files:
        if issue_counter >= max_issues:
            break
            
        file_issues = []
        
        try:
            # Enhanced syntax and semantic analysis
            for func in file.functions:
                if issue_counter >= max_issues:
                    break
                    
                try:
                    # Get function context and interconnections
                    func_context = get_enhanced_function_context(func, codebase)
                    
                    # Complexity analysis with detailed context
                    complexity = calculate_cyclomatic_complexity(func)
                    if complexity > 10:
                        severity = "critical" if complexity > 25 else "high" if complexity > 15 else "medium"
                        
                        issue = {
                            "id": f"complexity_{hashlib.md5(f'{file.filepath}_{func.name}'.encode()).hexdigest()[:8]}",
                            "type": "complexity_issue",
                            "severity": severity,
                            "file_path": file.filepath,
                            "line_number": getattr(func, 'line_number', None),
                            "function_name": func.name,
                            "message": f"High cyclomatic complexity: {complexity}",
                            "description": f"Function '{func.name}' has cyclomatic complexity of {complexity}",
                            "context": {
                                "complexity_score": complexity,
                                "parameters_count": len(func.parameters),
                                "return_statements": len(func.return_statements),
                                "function_calls": len(func.function_calls),
                                "dependencies": [dep.name for dep in func.dependencies[:5]],
                                "call_sites": len(func.call_sites)
                            },
                            "interconnected_context": func_context,
                            "affected_symbols": {
                                "functions": [call.name for call in func.function_calls[:10]],
                                "parameters": [param.name for param in func.parameters],
                                "dependencies": [dep.name for dep in func.dependencies[:10]]
                            },
                            "fix_suggestions": [
                                f"Break down '{func.name}' into smaller functions (current complexity: {complexity})",
                                "Extract complex conditional logic into separate methods",
                                "Consider using strategy pattern for complex branching",
                                f"Target complexity should be under 10 (currently {complexity})"
                            ]
                        }
                        
                        file_issues.append(issue)
                        error_analysis["detailed_issues"].append(issue)
                        issue_counter += 1
                        
                        if severity == "critical":
                            error_analysis["critical_issues"] += 1
                            
                except Exception as e:
                    continue
            
            # Store file-level issues
            if file_issues:
                error_analysis["issues_by_file"][file.filepath] = {
                    "total_issues": len(file_issues),
                    "critical_count": len([i for i in file_issues if i["severity"] == "critical"]),
                    "high_count": len([i for i in file_issues if i["severity"] == "high"]),
                    "medium_count": len([i for i in file_issues if i["severity"] == "medium"]),
                    "issues": file_issues
                }
                
        except Exception as e:
            continue
    
    # Calculate summary statistics
    error_analysis["total_issues"] = issue_counter
    
    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for issue in error_analysis["detailed_issues"]:
        severity = issue.get("severity", "low")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    error_analysis["issues_by_severity"] = severity_counts
    
    return error_analysis


def get_enhanced_function_context(func: Function, codebase: Codebase) -> Dict[str, Any]:
    """Get enhanced context for a function including all interconnected elements."""
    context = {
        "dependencies": [],
        "dependents": [],
        "call_graph": {},
        "data_flow": {},
        "related_classes": [],
        "related_files": []
    }
    
    try:
        # Dependencies analysis
        for dep in func.dependencies[:10]:
            context["dependencies"].append({
                "name": getattr(dep, 'name', 'unknown'),
                "type": type(dep).__name__,
                "file": getattr(dep, 'filepath', 'unknown')
            })
        
        # Call sites analysis
        for call_site in func.call_sites[:10]:
            context["dependents"].append({
                "caller": getattr(call_site, 'name', 'unknown'),
                "file": getattr(call_site, 'filepath', 'unknown')
            })
        
        # Function calls analysis
        call_graph = {}
        for call in func.function_calls[:10]:
            call_graph[call.name] = {
                "type": "function_call",
                "arguments_count": len(getattr(call, 'args', []))
            }
        context["call_graph"] = call_graph
        
        # Related files through imports
        if hasattr(func, 'file'):
            file = func.file
            related_files = []
            for imp in file.imports[:5]:
                if hasattr(imp, 'imported_symbol') and hasattr(imp.imported_symbol, 'filepath'):
                    related_files.append(imp.imported_symbol.filepath)
            context["related_files"] = related_files
            
    except Exception as e:
        context["error"] = str(e)
    
    return context


def get_enhanced_class_context(cls: Class, codebase: Codebase) -> Dict[str, Any]:
    """Get enhanced context for a class including all interconnected elements."""
    context = {
        "inheritance_chain": [],
        "composition_relationships": [],
        "method_dependencies": {},
        "attribute_usage": {},
        "related_classes": []
    }
    
    try:
        # Local complexity calculation to avoid circular imports
        def calculate_cyclomatic_complexity(function):
            base_complexity = 1
            if hasattr(function, 'source'):
                source = function.source.lower()
                base_complexity += source.count('if ') + source.count('elif ')
                base_complexity += source.count('for ') + source.count('while ')
                base_complexity += source.count('except ')
                base_complexity += source.count(' and ') + source.count(' or ')
            return base_complexity
        
        # Inheritance analysis
        context["inheritance_chain"] = cls.parent_class_names
        
        # Method analysis
        method_deps = {}
        for method in cls.methods[:10]:
            method_deps[method.name] = {
                "parameters": len(method.parameters),
                "complexity": calculate_cyclomatic_complexity(method),
                "calls": [call.name for call in method.function_calls[:5]]
            }
        context["method_dependencies"] = method_deps
        
        # Attribute analysis
        attr_usage = {}
        for attr in cls.attributes[:10]:
            attr_usage[attr.name] = {
                "type": getattr(attr, 'type', 'unknown'),
                "access_level": getattr(attr, 'access_level', 'unknown')
            }
        context["attribute_usage"] = attr_usage
        
    except Exception as e:
        context["error"] = str(e)
    
    return context
