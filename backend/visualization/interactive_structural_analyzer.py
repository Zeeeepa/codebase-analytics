#!/usr/bin/env python3
"""
Interactive Structural Analyzer

This module provides functionality for analyzing the structure of a codebase
and generating interactive visualizations of the results.
"""

import logging
import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from enum import Enum

from codegen.sdk.core.codebase import Codebase
from codegen.sdk.core.file import SourceFile
from codegen.sdk.core.function import Function
from codegen.sdk.core.class_definition import Class
from codegen.sdk.core.symbol import Symbol
from codegen.sdk.enums import EdgeType, SymbolType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class StructuralAnalysisMode(str, Enum):
    """Modes for structural analysis."""
    
    OVERVIEW = "overview"
    DETAILED = "detailed"
    ERRORS_ONLY = "errors_only"
    DEPENDENCIES = "dependencies"
    COMPLEXITY = "complexity"


def analyze_repository_structure(
    repo_path: str,
    codebase: Codebase,
    mode: StructuralAnalysisMode = StructuralAnalysisMode.OVERVIEW,
) -> Dict[str, Any]:
    """
    Analyze the structure of a repository.
    
    Args:
        repo_path: Path to the repository
        codebase: Codebase object
        mode: Analysis mode
        
    Returns:
        Dictionary containing the analysis results
    """
    logger.info(f"Analyzing repository structure in {mode} mode")
    
    # Basic repository info
    repo_info = {
        "path": repo_path,
        "total_files": len(codebase.files),
        "total_functions": sum(len(file.functions) for file in codebase.files),
        "total_classes": sum(len(file.classes) for file in codebase.files),
        "total_lines": sum(len(file.source.split("\n")) if hasattr(file, "source") else 0 for file in codebase.files),
        "languages": get_repository_languages(codebase),
    }
    
    # Analyze imports and dependencies
    dependency_graph = analyze_dependencies(codebase)
    
    # Analyze code structure
    structure_tree = build_structure_tree(codebase)
    
    # Analyze errors and issues
    errors = analyze_errors(codebase)
    repo_info["total_errors"] = len(errors)
    
    # Analyze complexity
    complexity_metrics = analyze_complexity(codebase)
    
    # Build the result based on the mode
    result = {
        "repository_info": repo_info,
        "errors": errors,
    }
    
    if mode in [StructuralAnalysisMode.OVERVIEW, StructuralAnalysisMode.DETAILED, StructuralAnalysisMode.DEPENDENCIES]:
        result["dependency_graph"] = dependency_graph
    
    if mode in [StructuralAnalysisMode.OVERVIEW, StructuralAnalysisMode.DETAILED]:
        result["structure_tree"] = structure_tree
    
    if mode in [StructuralAnalysisMode.DETAILED, StructuralAnalysisMode.COMPLEXITY]:
        result["complexity_metrics"] = complexity_metrics
    
    return result


def get_repository_languages(codebase: Codebase) -> Dict[str, int]:
    """
    Get the languages used in the repository.
    
    Args:
        codebase: Codebase object
        
    Returns:
        Dictionary mapping language to number of files
    """
    languages = {}
    
    for file in codebase.files:
        ext = Path(file.filepath).suffix.lower()
        
        # Map extensions to languages
        language = None
        if ext in [".py"]:
            language = "Python"
        elif ext in [".js"]:
            language = "JavaScript"
        elif ext in [".ts"]:
            language = "TypeScript"
        elif ext in [".tsx", ".jsx"]:
            language = "React"
        elif ext in [".html"]:
            language = "HTML"
        elif ext in [".css", ".scss", ".sass"]:
            language = "CSS"
        elif ext in [".json"]:
            language = "JSON"
        elif ext in [".md"]:
            language = "Markdown"
        elif ext in [".java"]:
            language = "Java"
        elif ext in [".c", ".cpp", ".h", ".hpp"]:
            language = "C/C++"
        elif ext in [".go"]:
            language = "Go"
        elif ext in [".rb"]:
            language = "Ruby"
        elif ext in [".php"]:
            language = "PHP"
        elif ext in [".rs"]:
            language = "Rust"
        elif ext in [".swift"]:
            language = "Swift"
        elif ext in [".kt", ".kts"]:
            language = "Kotlin"
        else:
            language = "Other"
        
        languages[language] = languages.get(language, 0) + 1
    
    return languages


def analyze_dependencies(codebase: Codebase) -> Dict[str, Any]:
    """
    Analyze dependencies between files in the codebase.
    
    Args:
        codebase: Codebase object
        
    Returns:
        Dictionary containing dependency information
    """
    nodes = []
    edges = []
    
    # Create nodes for each file
    for file in codebase.files:
        nodes.append({
            "id": file.filepath,
            "label": Path(file.filepath).name,
            "type": "file",
            "size": len(file.source.split("\n")) if hasattr(file, "source") else 0,
            "metadata": {
                "file": file.filepath,
                "functions": len(file.functions),
                "classes": len(file.classes),
            }
        })
    
    # Create edges for imports
    for file in codebase.files:
        for imp in getattr(file, "imports", []):
            if hasattr(imp, "resolved_path") and imp.resolved_path:
                edges.append({
                    "source": file.filepath,
                    "target": imp.resolved_path,
                    "type": "import",
                })
    
    # Detect circular dependencies
    circular_deps = detect_circular_dependencies(nodes, edges)
    
    # Add circular dependency edges
    for cycle in circular_deps:
        for i in range(len(cycle)):
            source = cycle[i]
            target = cycle[(i + 1) % len(cycle)]
            
            edges.append({
                "source": source,
                "target": target,
                "type": "circular",
                "color": "#ef4444",  # Red
                "weight": 2.0,
                "metadata": {
                    "cycle": cycle,
                }
            })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "circular_dependencies": circular_deps,
        "metadata": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "total_circular_dependencies": len(circular_deps),
        }
    }


def detect_circular_dependencies(nodes, edges) -> List[List[str]]:
    """
    Detect circular dependencies in the dependency graph.
    
    Args:
        nodes: List of nodes
        edges: List of edges
        
    Returns:
        List of cycles, where each cycle is a list of node IDs
    """
    # Build adjacency list
    graph = {}
    for node in nodes:
        graph[node["id"]] = []
    
    for edge in edges:
        if edge["source"] in graph and edge["target"] in graph:
            graph[edge["source"]].append(edge["target"])
    
    # Find cycles using DFS
    def find_cycles(node, visited, path):
        if node in path:
            # Found a cycle
            cycle_start = path.index(node)
            return [path[cycle_start:]]
        
        if node in visited:
            return []
        
        visited.add(node)
        path.append(node)
        
        cycles = []
        for neighbor in graph.get(node, []):
            cycles.extend(find_cycles(neighbor, visited.copy(), path.copy()))
        
        return cycles
    
    all_cycles = []
    for node in graph:
        cycles = find_cycles(node, set(), [])
        for cycle in cycles:
            if cycle not in all_cycles:
                all_cycles.append(cycle)
    
    return all_cycles


def build_structure_tree(codebase: Codebase) -> Dict[str, Any]:
    """
    Build a tree representation of the codebase structure.
    
    Args:
        codebase: Codebase object
        
    Returns:
        Dictionary containing the structure tree
    """
    root = {
        "id": "root",
        "label": "Codebase",
        "type": "root",
        "children": []
    }
    
    # Group files by directory
    directories = {}
    
    for file in codebase.files:
        path_parts = Path(file.filepath).parts
        
        # Build directory structure
        current_path = ""
        current_node = root
        
        for i, part in enumerate(path_parts[:-1]):  # Exclude the filename
            current_path = os.path.join(current_path, part) if current_path else part
            
            if current_path not in directories:
                # Create new directory node
                dir_node = {
                    "id": current_path,
                    "label": part,
                    "type": "directory",
                    "children": []
                }
                
                # Add to parent
                current_node["children"].append(dir_node)
                
                # Update directories dict
                directories[current_path] = dir_node
            
            # Move to next level
            current_node = directories[current_path]
        
        # Add file node
        file_node = {
            "id": file.filepath,
            "label": path_parts[-1],
            "type": "file",
            "metadata": {
                "file": file.filepath,
                "functions": len(file.functions),
                "classes": len(file.classes),
            },
            "children": []
        }
        
        # Add classes and functions
        for cls in file.classes:
            class_node = {
                "id": f"{file.filepath}:{cls.name}",
                "label": cls.name,
                "type": "class",
                "metadata": {
                    "file": file.filepath,
                    "methods": len(cls.methods) if hasattr(cls, "methods") else 0,
                },
                "children": []
            }
            
            # Add methods
            if hasattr(cls, "methods"):
                for method in cls.methods:
                    method_node = {
                        "id": f"{file.filepath}:{cls.name}.{method.name}",
                        "label": method.name,
                        "type": "method",
                        "metadata": {
                            "file": file.filepath,
                            "class": cls.name,
                        }
                    }
                    class_node["children"].append(method_node)
            
            file_node["children"].append(class_node)
        
        # Add standalone functions
        for func in file.functions:
            # Skip methods (already added above)
            if hasattr(func, "parent") and func.parent:
                continue
            
            func_node = {
                "id": f"{file.filepath}:{func.name}",
                "label": func.name,
                "type": "function",
                "metadata": {
                    "file": file.filepath,
                }
            }
            file_node["children"].append(func_node)
        
        # Add file to parent directory
        current_node["children"].append(file_node)
    
    return root


def analyze_errors(codebase: Codebase) -> List[Dict[str, Any]]:
    """
    Analyze errors in the codebase.
    
    Args:
        codebase: Codebase object
        
    Returns:
        List of errors
    """
    errors = []
    
    # Check for import errors
    for file in codebase.files:
        for imp in getattr(file, "imports", []):
            if not hasattr(imp, "resolved_path") or not imp.resolved_path:
                errors.append({
                    "type": "import_error",
                    "severity": "error",
                    "message": f"Unresolved import: {imp.name}",
                    "location": {
                        "file": file.filepath,
                        "line": getattr(imp, "line", None),
                    }
                })
    
    # Check for undefined symbols
    for file in codebase.files:
        for func in file.functions:
            if hasattr(func, "calls"):
                for call in func.calls:
                    if not call.resolved:
                        errors.append({
                            "type": "undefined_symbol",
                            "severity": "warning",
                            "message": f"Undefined function call: {call.name}",
                            "location": {
                                "file": file.filepath,
                                "line": getattr(call, "line", None),
                            }
                        })
    
    # Check for circular dependencies
    dependency_graph = analyze_dependencies(codebase)
    for cycle in dependency_graph["circular_dependencies"]:
        cycle_str = " -> ".join(cycle)
        for file in cycle:
            errors.append({
                "type": "circular_dependency",
                "severity": "warning",
                "message": f"Circular dependency: {cycle_str}",
                "location": {
                    "file": file,
                }
            })
    
    return errors


def analyze_complexity(codebase: Codebase) -> Dict[str, Any]:
    """
    Analyze code complexity.
    
    Args:
        codebase: Codebase object
        
    Returns:
        Dictionary containing complexity metrics
    """
    file_complexity = {}
    function_complexity = []
    
    for file in codebase.files:
        file_metrics = {
            "file": file.filepath,
            "lines": len(file.source.split("\n")) if hasattr(file, "source") else 0,
            "functions": len(file.functions),
            "classes": len(file.classes),
            "complexity": 0,
        }
        
        # Calculate complexity for each function
        for func in file.functions:
            func_complexity = calculate_function_complexity(func)
            file_metrics["complexity"] += func_complexity
            
            function_complexity.append({
                "file": file.filepath,
                "function": func.name,
                "complexity": func_complexity,
                "lines": len(func.source.split("\n")) if hasattr(func, "source") else 0,
            })
        
        file_complexity[file.filepath] = file_metrics
    
    return {
        "file_complexity": file_complexity,
        "function_complexity": function_complexity,
        "summary": {
            "total_files": len(file_complexity),
            "total_functions": len(function_complexity),
            "avg_file_complexity": sum(f["complexity"] for f in file_complexity.values()) / len(file_complexity) if file_complexity else 0,
            "avg_function_complexity": sum(f["complexity"] for f in function_complexity) / len(function_complexity) if function_complexity else 0,
        }
    }


def calculate_function_complexity(func: Function) -> int:
    """
    Calculate cyclomatic complexity for a function.
    
    Args:
        func: Function object
        
    Returns:
        Complexity score
    """
    # This is a simplified calculation
    complexity = 1  # Base complexity
    
    # Count control flow statements
    if hasattr(func, "source"):
        source = func.source
        
        # Count if statements
        complexity += source.count("if ")
        
        # Count else if statements
        complexity += source.count("elif ")
        
        # Count for loops
        complexity += source.count("for ")
        
        # Count while loops
        complexity += source.count("while ")
        
        # Count and/or operators (each adds a path)
        complexity += source.count(" and ")
        complexity += source.count(" or ")
        
        # Count exception handlers
        complexity += source.count("except ")
    
    return complexity
