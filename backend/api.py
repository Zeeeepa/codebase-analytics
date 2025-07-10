from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List, Tuple, Any, Optional, Set
from codegen import Codebase
from codegen.configs.models.codebase import CodebaseConfig
from codegen.sdk.core.statements.for_loop_statement import ForLoopStatement
from codegen.sdk.core.statements.if_block_statement import IfBlockStatement
from codegen.sdk.core.statements.try_catch_statement import TryCatchStatement
from codegen.sdk.core.statements.while_statement import WhileStatement
from codegen.sdk.core.expressions.binary_expression import BinaryExpression
from codegen.sdk.core.expressions.unary_expression import UnaryExpression
from codegen.sdk.core.expressions.comparison_expression import ComparisonExpression
from collections import Counter, defaultdict
import math
import re
from .analysis_helpers import (
    has_error_handling, has_potential_null_references, has_unhandled_critical_operations,
    is_special_function, analyze_control_flow, analyze_error_patterns, 
    analyze_performance_indicators, has_potential_null_references_in_source
)
import requests
from datetime import datetime, timedelta
import subprocess
import os
import tempfile
from fastapi.middleware.cors import CORSMiddleware
import modal
import logging
import traceback

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "codegen", "fastapi", "uvicorn", "gitpython", "requests", "pydantic", "datetime"
    )
)

app = modal.App(name="analytics-app", image=image)

fastapi_app = FastAPI()

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ComprehensiveAnalysis:
    """Comprehensive codebase analysis using all graph-sitter capabilities."""
    
    def __init__(self, codebase: Codebase):
        self.codebase = codebase
        self.issues = []
        self.function_contexts = {}
        self.entry_points = []
        self.dead_code_items = []
        
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive analysis and return detailed results."""
        try:
            # Core analysis
            self._analyze_dead_code()
            self._analyze_entry_points()
            self._analyze_function_contexts()
            self._detect_issues()
            
            return {
                "repository_structure": self._get_repository_structure(),
                "summary": self._generate_summary(),
                "most_important_functions": self._find_most_important_functions(),
                "dead_code_analysis": self._analyze_dead_code(),
                "entry_points": self.entry_points,
                "issues_by_severity": self._group_issues_by_severity(),
                "function_contexts": self.function_contexts,
                "call_graph_metrics": self._analyze_call_graph(),
                "dependency_metrics": self._analyze_dependencies(),
                "import_analysis": self._analyze_imports(),
                "class_hierarchy": self._analyze_class_hierarchy(),
                "code_quality_metrics": self._analyze_code_quality(),
                # NEW: Enhanced analysis features
                "most_important_files": self.identify_critical_files(),
                "comprehensive_error_analysis": self.comprehensive_error_analysis()
            }
        except Exception as e:
            logging.error(f"Analysis failed: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _analyze_dead_code(self) -> Dict[str, Any]:
        """Analyze dead code in the codebase."""
        dead_functions = []
        dead_classes = []
        dead_imports = []
        
        # Find unused functions
        for func in self.codebase.functions:
            if not hasattr(func, 'usages') or len(func.usages) == 0:
                dead_functions.append({
                    'name': func.name,
                    'file': getattr(func, 'file', {}).get('filepath', 'Unknown'),
                    'type': 'function'
                })
                self.dead_code_items.append(func.name)
        
        # Find unused classes
        for cls in self.codebase.classes:
            if not hasattr(cls, 'usages') or len(cls.usages) == 0:
                dead_classes.append({
                    'name': cls.name,
                    'file': getattr(cls, 'file', {}).get('filepath', 'Unknown'),
                    'type': 'class'
                })
                self.dead_code_items.append(cls.name)
        
        # Find unused imports
        for file in self.codebase.files:
            for imp in getattr(file, 'imports', []):
                if not hasattr(imp, 'usages') or len(imp.usages) == 0:
                    dead_imports.append({
                        'name': getattr(imp, 'name', 'Unknown'),
                        'file': getattr(file, 'filepath', 'Unknown'),
                        'type': 'import'
                    })
        
        return {
            'dead_functions': dead_functions,
            'dead_classes': dead_classes,
            'dead_imports': dead_imports,
            'total_dead_items': len(dead_functions) + len(dead_classes) + len(dead_imports)
        }
    
    def _analyze_entry_points(self):
        """Identify entry points in the codebase."""
        entry_points = []
        
        # Common entry point patterns
        entry_patterns = ['main', '__main__', 'app', 'run', 'start', 'init']
        
        for func in self.codebase.functions:
            # Check if function name matches entry patterns
            if any(pattern in func.name.lower() for pattern in entry_patterns):
                entry_points.append({
                    'name': func.name,
                    'file': getattr(func, 'file', {}).get('filepath', 'Unknown'),
                    'type': 'function',
                    'reason': 'name_pattern'
                })
            
            # Check if function has no callers (potential entry point)
            if hasattr(func, 'call_sites') and len(func.call_sites) == 0:
                entry_points.append({
                    'name': func.name,
                    'file': getattr(func, 'file', {}).get('filepath', 'Unknown'),
                    'type': 'function',
                    'reason': 'no_callers'
                })
        
        self.entry_points = entry_points
    
    def _analyze_function_contexts(self):
        """Analyze detailed context for each function."""
        for func in self.codebase.functions:
            try:
                # Get function dependencies
                dependencies = []
                if hasattr(func, 'dependencies'):
                    dependencies = [getattr(dep, 'name', str(dep)) for dep in func.dependencies]
                
                # Get function calls made by this function
                function_calls = []
                if hasattr(func, 'function_calls'):
                    function_calls = [getattr(call, 'name', str(call)) for call in func.function_calls]
                
                # Get call sites (where this function is called)
                called_by = []
                if hasattr(func, 'call_sites'):
                    called_by = [getattr(site, 'name', str(site)) for site in func.call_sites]
                
                # Get parameters
                parameters = []
                if hasattr(func, 'parameters'):
                    parameters = [getattr(param, 'name', str(param)) for param in func.parameters]
                
                # Check if it's an entry point
                is_entry_point = func.name in [ep['name'] for ep in self.entry_points]
                
                # Check if it's dead code
                is_dead_code = func.name in self.dead_code_items
                
                # Build call chain
                max_call_chain = self._get_max_call_chain(func)
                
                # Detect issues for this function
                function_issues = self._detect_function_issues(func)
                
                self.function_contexts[func.name] = {
                    'filepath': getattr(func, 'file', {}).get('filepath', 'Unknown'),
                    'parameters': parameters,
                    'dependencies': dependencies,
                    'function_calls': function_calls,
                    'called_by': called_by,
                    'is_entry_point': is_entry_point,
                    'is_dead_code': is_dead_code,
                    'max_call_chain': max_call_chain,
                    'issues': function_issues,
                    'is_async': getattr(func, 'is_async', False),
                    'is_generator': getattr(func, 'is_generator', False),
                    'decorators': [getattr(d, 'name', str(d)) for d in getattr(func, 'decorators', [])],
                    'return_type': getattr(func, 'return_type', None)
                }
            except Exception as e:
                logging.warning(f"Failed to analyze function {func.name}: {str(e)}")
    
    def _get_max_call_chain(self, func, visited=None, depth=0) -> List[str]:
        """Get the maximum call chain for a function."""
        if visited is None:
            visited = set()
        
        if func.name in visited or depth > 10:  # Prevent infinite recursion
            return [func.name]
        
        visited.add(func.name)
        max_chain = [func.name]
        
        if hasattr(func, 'function_calls'):
            for call in func.function_calls:
                try:
                    called_func = getattr(call, 'function_definition', None)
                    if called_func:
                        sub_chain = self._get_max_call_chain(called_func, visited.copy(), depth + 1)
                        if len(sub_chain) > len(max_chain) - 1:
                            max_chain = [func.name] + sub_chain
                except:
                    continue
        
        return max_chain
    
    def _detect_issues(self):
        """Detect various code quality issues."""
        self.issues = []
        
        # Analyze functions for issues
        for func in self.codebase.functions:
            self.issues.extend(self._detect_function_issues(func))
        
        # Analyze classes for issues
        for cls in self.codebase.classes:
            self.issues.extend(self._detect_class_issues(cls))
        
        # Analyze imports for issues
        for file in self.codebase.files:
            self.issues.extend(self._detect_import_issues(file))
    
    def _detect_function_issues(self, func) -> List[Dict[str, Any]]:
        """Detect issues specific to a function."""
        issues = []
        
        try:
            # Check for unused parameters
            if hasattr(func, 'parameters'):
                for param in func.parameters:
                    if hasattr(param, 'usages') and len(param.usages) == 0:
                        issues.append({
                            'type': 'unused_parameter',
                            'severity': 'minor',
                            'message': f"Parameter '{param.name}' is unused",
                            'function': func.name,
                            'file': getattr(func, 'file', {}).get('filepath', 'Unknown')
                        })
            
            # Check for missing return type
            if not hasattr(func, 'return_type') or func.return_type is None:
                issues.append({
                    'type': 'missing_return_type',
                    'severity': 'minor',
                    'message': f"Function '{func.name}' missing return type annotation",
                    'function': func.name,
                    'file': getattr(func, 'file', {}).get('filepath', 'Unknown')
                })
            
            # Check for missing docstring
            if not hasattr(func, 'docstring') or not func.docstring:
                issues.append({
                    'type': 'missing_docstring',
                    'severity': 'minor',
                    'message': f"Function '{func.name}' missing docstring",
                    'function': func.name,
                    'file': getattr(func, 'file', {}).get('filepath', 'Unknown')
                })
            
            # Check for high complexity
            if hasattr(func, 'complexity') and func.complexity > 10:
                severity = 'critical' if func.complexity > 20 else 'major'
                issues.append({
                    'type': 'high_complexity',
                    'severity': severity,
                    'message': f"Function '{func.name}' has high complexity ({func.complexity})",
                    'function': func.name,
                    'file': getattr(func, 'file', {}).get('filepath', 'Unknown')
                })
            
            # Check for too many parameters
            if hasattr(func, 'parameters') and len(func.parameters) > 5:
                issues.append({
                    'type': 'too_many_parameters',
                    'severity': 'major',
                    'message': f"Function '{func.name}' has too many parameters ({len(func.parameters)})",
                    'function': func.name,
                    'file': getattr(func, 'file', {}).get('filepath', 'Unknown')
                })
            
            # Check for recursive functions without base case detection
            if hasattr(func, 'function_calls'):
                for call in func.function_calls:
                    if getattr(call, 'name', '') == func.name:
                        issues.append({
                            'type': 'potential_infinite_recursion',
                            'severity': 'critical',
                            'message': f"Function '{func.name}' may have infinite recursion",
                            'function': func.name,
                            'file': getattr(func, 'file', {}).get('filepath', 'Unknown')
                        })
                        break
            
        except Exception as e:
            logging.warning(f"Error detecting issues for function {func.name}: {str(e)}")
        
        return issues
    
    def _detect_class_issues(self, cls) -> List[Dict[str, Any]]:
        """Detect issues specific to a class."""
        issues = []
        
        try:
            # Check for classes with too many methods
            if hasattr(cls, 'methods') and len(cls.methods) > 20:
                issues.append({
                    'type': 'too_many_methods',
                    'severity': 'major',
                    'message': f"Class '{cls.name}' has too many methods ({len(cls.methods)})",
                    'class': cls.name,
                    'file': getattr(cls, 'file', {}).get('filepath', 'Unknown')
                })
            
            # Check for deep inheritance
            if hasattr(cls, 'superclasses') and len(cls.superclasses) > 5:
                issues.append({
                    'type': 'deep_inheritance',
                    'severity': 'major',
                    'message': f"Class '{cls.name}' has deep inheritance chain ({len(cls.superclasses)})",
                    'class': cls.name,
                    'file': getattr(cls, 'file', {}).get('filepath', 'Unknown')
                })
            
            # Check for missing docstring
            if not hasattr(cls, 'docstring') or not cls.docstring:
                issues.append({
                    'type': 'missing_docstring',
                    'severity': 'minor',
                    'message': f"Class '{cls.name}' missing docstring",
                    'class': cls.name,
                    'file': getattr(cls, 'file', {}).get('filepath', 'Unknown')
                })
                
        except Exception as e:
            logging.warning(f"Error detecting issues for class {cls.name}: {str(e)}")
        
        return issues
    
    def _detect_import_issues(self, file) -> List[Dict[str, Any]]:
        """Detect import-related issues."""
        issues = []
        
        try:
            # Check for unused imports
            for imp in getattr(file, 'imports', []):
                if hasattr(imp, 'usages') and len(imp.usages) == 0:
                    issues.append({
                        'type': 'unused_import',
                        'severity': 'minor',
                        'message': f"Import '{getattr(imp, 'name', 'Unknown')}' is unused",
                        'file': getattr(file, 'filepath', 'Unknown')
                    })
            
            # Check for circular imports (simplified detection)
            import_files = []
            for imp in getattr(file, 'imports', []):
                if hasattr(imp, 'resolved_symbol') and hasattr(imp.resolved_symbol, 'file'):
                    import_files.append(imp.resolved_symbol.file)
            
            # Check if any imported file imports back to this file
            for imp_file in import_files:
                for reverse_imp in getattr(imp_file, 'imports', []):
                    if (hasattr(reverse_imp, 'resolved_symbol') and 
                        hasattr(reverse_imp.resolved_symbol, 'file') and
                        reverse_imp.resolved_symbol.file == file):
                        issues.append({
                            'type': 'circular_import',
                            'severity': 'critical',
                            'message': f"Circular import detected between {getattr(file, 'filepath', 'Unknown')} and {getattr(imp_file, 'filepath', 'Unknown')}",
                            'file': getattr(file, 'filepath', 'Unknown')
                        })
                        
        except Exception as e:
            logging.warning(f"Error detecting import issues for file {getattr(file, 'filepath', 'Unknown')}: {str(e)}")
        
        return issues
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive codebase summary."""
        return {
            'total_files': len(list(self.codebase.files)),
            'total_functions': len(list(self.codebase.functions)),
            'total_classes': len(list(self.codebase.classes)),
            'total_imports': len(list(self.codebase.imports)),
            'total_symbols': len(list(self.codebase.symbols)),
            'total_issues': len(self.issues),
            'critical_issues': len([i for i in self.issues if i['severity'] == 'critical']),
            'major_issues': len([i for i in self.issues if i['severity'] == 'major']),
            'minor_issues': len([i for i in self.issues if i['severity'] == 'minor']),
            'dead_code_items': len(self.dead_code_items),
            'entry_points': len(self.entry_points),
            'external_modules': len(list(self.codebase.external_modules)) if hasattr(self.codebase, 'external_modules') else 0
        }
    
    def _find_most_important_functions(self) -> Dict[str, Any]:
        """Find the most important functions in the codebase."""
        most_calls = {'name': 'N/A', 'call_count': 0, 'calls': []}
        most_called = {'name': 'N/A', 'usage_count': 0}
        deepest_inheritance = {'name': 'N/A', 'chain_depth': 0}
        
        # Find function that makes the most calls
        for func in self.codebase.functions:
            if hasattr(func, 'function_calls'):
                call_count = len(func.function_calls)
                if call_count > most_calls['call_count']:
                    most_calls = {
                        'name': func.name,
                        'call_count': call_count,
                        'calls': [getattr(call, 'name', str(call)) for call in func.function_calls[:10]]
                    }
        
        # Find most called function
        for func in self.codebase.functions:
            if hasattr(func, 'usages'):
                usage_count = len(func.usages)
                if usage_count > most_called['usage_count']:
                    most_called = {
                        'name': func.name,
                        'usage_count': usage_count
                    }
        
        # Find class with deepest inheritance
        for cls in self.codebase.classes:
            if hasattr(cls, 'superclasses'):
                chain_depth = len(cls.superclasses)
                if chain_depth > deepest_inheritance['chain_depth']:
                    deepest_inheritance = {
                        'name': cls.name,
                        'chain_depth': chain_depth
                    }
        
        return {
            'most_calls': most_calls,
            'most_called': most_called,
            'deepest_inheritance': deepest_inheritance
        }
    
    def _group_issues_by_severity(self) -> Dict[str, List[Dict[str, Any]]]:
        """Group issues by severity level."""
        grouped = {'critical': [], 'major': [], 'minor': []}
        
        for issue in self.issues:
            severity = issue.get('severity', 'minor')
            if severity in grouped:
                grouped[severity].append(issue)
        
        return grouped
    
    def _analyze_call_graph(self) -> Dict[str, Any]:
        """Analyze call graph metrics."""
        total_calls = 0
        recursive_functions = []
        max_call_depth = 0
        
        for func in self.codebase.functions:
            if hasattr(func, 'function_calls'):
                total_calls += len(func.function_calls)
                
                # Check for recursion
                for call in func.function_calls:
                    if getattr(call, 'name', '') == func.name:
                        recursive_functions.append(func.name)
                        break
                
                # Calculate call depth
                call_chain = self._get_max_call_chain(func)
                if len(call_chain) > max_call_depth:
                    max_call_depth = len(call_chain)
        
        return {
            'total_function_calls': total_calls,
            'recursive_functions': recursive_functions,
            'max_call_depth': max_call_depth,
            'average_calls_per_function': total_calls / len(list(self.codebase.functions)) if len(list(self.codebase.functions)) > 0 else 0
        }
    
    def _analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze dependency metrics."""
        total_dependencies = 0
        circular_dependencies = []
        
        for symbol in self.codebase.symbols:
            if hasattr(symbol, 'dependencies'):
                total_dependencies += len(symbol.dependencies)
        
        # Simplified circular dependency detection
        for issue in self.issues:
            if issue['type'] == 'circular_import':
                circular_dependencies.append(issue['message'])
        
        return {
            'total_dependencies': total_dependencies,
            'circular_dependencies': circular_dependencies,
            'average_dependencies_per_symbol': total_dependencies / len(list(self.codebase.symbols)) if len(list(self.codebase.symbols)) > 0 else 0
        }
    
    def _analyze_imports(self) -> Dict[str, Any]:
        """Analyze import patterns and metrics."""
        total_imports = 0
        external_imports = 0
        unused_imports = 0
        
        for file in self.codebase.files:
            file_imports = getattr(file, 'imports', [])
            total_imports += len(file_imports)
            
            for imp in file_imports:
                # Check if external
                if hasattr(imp, 'is_external') and imp.is_external:
                    external_imports += 1
                
                # Check if unused
                if hasattr(imp, 'usages') and len(imp.usages) == 0:
                    unused_imports += 1
        
        return {
            'total_imports': total_imports,
            'external_imports': external_imports,
            'unused_imports': unused_imports,
            'import_efficiency': (total_imports - unused_imports) / total_imports if total_imports > 0 else 1.0
        }
    
    def _analyze_class_hierarchy(self) -> Dict[str, Any]:
        """Analyze class inheritance patterns."""
        total_classes = len(list(self.codebase.classes))
        classes_with_inheritance = 0
        max_inheritance_depth = 0
        abstract_classes = 0
        
        for cls in self.codebase.classes:
            if hasattr(cls, 'superclasses') and len(cls.superclasses) > 0:
                classes_with_inheritance += 1
                if len(cls.superclasses) > max_inheritance_depth:
                    max_inheritance_depth = len(cls.superclasses)
            
            if hasattr(cls, 'is_abstract') and cls.is_abstract:
                abstract_classes += 1
        
        return {
            'total_classes': total_classes,
            'classes_with_inheritance': classes_with_inheritance,
            'max_inheritance_depth': max_inheritance_depth,
            'abstract_classes': abstract_classes,
            'inheritance_ratio': classes_with_inheritance / total_classes if total_classes > 0 else 0
        }
    
    def _analyze_code_quality(self) -> Dict[str, Any]:
        """Analyze overall code quality metrics."""
        functions_with_docstrings = 0
        functions_with_type_hints = 0
        total_functions = len(list(self.codebase.functions))
        
        for func in self.codebase.functions:
            if hasattr(func, 'docstring') and func.docstring:
                functions_with_docstrings += 1
            
            if hasattr(func, 'return_type') and func.return_type:
                functions_with_type_hints += 1
        
        return {
            'documentation_coverage': functions_with_docstrings / total_functions if total_functions > 0 else 0,
            'type_hint_coverage': functions_with_type_hints / total_functions if total_functions > 0 else 0,
            'code_quality_score': self._calculate_quality_score()
        }
    
    def _calculate_quality_score(self) -> float:
        """Calculate overall code quality score (0-100)."""
        total_functions = len(list(self.codebase.functions))
        if total_functions == 0:
            return 100.0
        
        # Deduct points for issues
        score = 100.0
        critical_issues = len([i for i in self.issues if i['severity'] == 'critical'])
        major_issues = len([i for i in self.issues if i['severity'] == 'major'])
        minor_issues = len([i for i in self.issues if i['severity'] == 'minor'])
        
        score -= critical_issues * 10  # 10 points per critical issue
        score -= major_issues * 5      # 5 points per major issue
        score -= minor_issues * 1      # 1 point per minor issue
        
        # Deduct for dead code
        score -= len(self.dead_code_items) * 2
        
        return max(0.0, score)
    
    def _get_repository_structure(self) -> Dict[str, Any]:
        """Get repository structure information."""
        file_types = defaultdict(int)
        total_size = 0
        
        for file in self.codebase.files:
            filepath = getattr(file, 'filepath', '')
            if filepath:
                ext = filepath.split('.')[-1] if '.' in filepath else 'no_extension'
                file_types[ext] += 1
                
                # Estimate size based on source length
                source = getattr(file, 'source', '')
                total_size += len(source)
        
        return {
            'file_types': dict(file_types),
            'estimated_total_size': total_size,
            'average_file_size': total_size / len(list(self.codebase.files)) if len(list(self.codebase.files)) > 0 else 0
        }
    
    def identify_critical_files(self) -> Dict[str, Any]:
        """Identify the most critical files in the codebase."""
        critical_analysis = {
            "entry_points": [],
            "most_called_functions": [],
            "highest_dependency_files": [],
            "architectural_hubs": [],
            "error_prone_files": []
        }
        
        # Find entry points (main files, __init__.py, etc.)
        entry_patterns = ['main.py', '__main__.py', 'app.py', 'server.py', 'index.ts', 'index.js', '__init__.py']
        for file in self.codebase.files:
            filepath = getattr(file, 'filepath', '')
            if any(pattern in filepath.lower() for pattern in entry_patterns):
                critical_analysis["entry_points"].append({
                    "file": filepath,
                    "functions": len(getattr(file, 'functions', [])),
                    "classes": len(getattr(file, 'classes', [])),
                    "imports": len(getattr(file, 'imports', [])),
                    "importance_score": self._calculate_file_importance(file)
                })
        
        # Find most called functions across codebase
        function_call_counts = {}
        for function in self.codebase.functions:
            call_count = len(getattr(function, 'call_sites', []))
            if call_count > 0:
                function_call_counts[function] = call_count
        
        # Sort by call count and get top 20
        most_called = sorted(function_call_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        for func, count in most_called:
            critical_analysis["most_called_functions"].append({
                "function": func.name,
                "file": getattr(func, 'file', {}).get('filepath', 'Unknown'),
                "call_count": count,
                "dependencies": len(getattr(func, 'dependencies', [])),
                "error_indicators": self._analyze_function_errors(func)
            })
        
        # Find files with highest dependency counts
        file_dependency_counts = {}
        for file in self.codebase.files:
            symbols = getattr(file, 'symbols', [])
            total_deps = sum(len(getattr(symbol, 'dependencies', [])) for symbol in symbols)
            total_usages = sum(len(getattr(symbol, 'usages', [])) for symbol in symbols)
            file_dependency_counts[file] = total_deps + total_usages
        
        highest_deps = sorted(file_dependency_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        for file, dep_count in highest_deps:
            critical_analysis["highest_dependency_files"].append({
                "file": getattr(file, 'filepath', 'Unknown'),
                "dependency_score": dep_count,
                "symbols": len(getattr(file, 'symbols', [])),
                "potential_issues": self._analyze_file_issues(file)
            })
        
        return critical_analysis
    
    def _calculate_file_importance(self, file) -> int:
        """Calculate comprehensive importance score for a file."""
        importance = 0
        
        # Add importance from functions
        functions = getattr(file, 'functions', [])
        for func in functions:
            importance += len(getattr(func, 'call_sites', [])) * 2  # Called functions are important
            importance += len(getattr(func, 'dependencies', []))  # Dependencies add importance
        
        # Add importance from classes
        classes = getattr(file, 'classes', [])
        for cls in classes:
            importance += len(getattr(cls, 'methods', [])) * 2  # Methods add importance
            importance += len(getattr(cls, 'subclasses', [])) * 3  # Inheritance adds importance
        
        # Add import importance
        importance += len(getattr(file, 'imports', []))
        
        return importance
    
    def _analyze_function_errors(self, func) -> List[str]:
        """Analyze potential errors in a function."""
        errors = []
        
        # Check for missing error handling in async functions
        if getattr(func, 'is_async', False):
            if not has_error_handling(func):
                errors.append("missing_async_error_handling")
        
        # Check for potential null references
        if has_potential_null_references(func):
            errors.append("potential_null_references")
        
        # Check for unhandled critical operations
        if has_unhandled_critical_operations(func):
            errors.append("unhandled_critical_operations")
        
        return errors
    
    def _analyze_file_issues(self, file) -> List[str]:
        """Analyze potential issues in a file."""
        issues = []
        
        # Check for too many imports
        imports = getattr(file, 'imports', [])
        if len(imports) > 20:
            issues.append("too_many_imports")
        
        # Check for large files
        source = getattr(file, 'source', '')
        if len(source.split('\n')) > 500:
            issues.append("large_file")
        
        # Check for missing documentation
        functions = getattr(file, 'functions', [])
        undocumented_functions = [f for f in functions if not getattr(f, 'docstring', None)]
        if len(undocumented_functions) > len(functions) * 0.5:
            issues.append("poor_documentation")
        
        return issues
    
    def comprehensive_error_analysis(self) -> Dict[str, Any]:
        """Maximum depth error and issue analysis."""
        error_analysis = {
            "syntax_issues": [],
            "type_errors": [],
            "runtime_risks": [],
            "missing_parameters": [],
            "undefined_references": [],
            "circular_dependencies": [],
            "dead_code": [],
            "security_risks": [],
            "performance_issues": [],
            "maintainability_issues": []
        }
        
        # 1. Type-related errors
        for function in self.codebase.functions:
            # Missing type annotations
            if not getattr(function, 'return_type', None):
                error_analysis["type_errors"].append({
                    "type": "missing_return_type",
                    "function": function.name,
                    "file": getattr(function, 'file', {}).get('filepath', 'Unknown'),
                    "severity": "medium",
                    "context": self.get_function_context(function)
                })
            
            # Missing parameter types
            for param in getattr(function, 'parameters', []):
                if not getattr(param, 'type', None):
                    error_analysis["missing_parameters"].append({
                        "type": "missing_parameter_type",
                        "function": function.name,
                        "parameter": getattr(param, 'name', 'unknown'),
                        "file": getattr(function, 'file', {}).get('filepath', 'Unknown'),
                        "severity": "medium",
                        "context": self.get_function_context(function)
                    })
        
        # 2. Runtime risk analysis
        for function in self.codebase.functions:
            runtime_risks = self._analyze_runtime_risks(function)
            if runtime_risks:
                error_analysis["runtime_risks"].extend(runtime_risks)
        
        # 3. Circular dependency detection
        circular_deps = self._detect_circular_dependencies()
        error_analysis["circular_dependencies"].extend(circular_deps)
        
        # 4. Dead code detection
        for function in self.codebase.functions:
            usages = getattr(function, 'usages', [])
            call_sites = getattr(function, 'call_sites', [])
            if not usages and not call_sites:
                if not is_special_function(function):
                    error_analysis["dead_code"].append({
                        "type": "unused_function",
                        "function": function.name,
                        "file": getattr(function, 'file', {}).get('filepath', 'Unknown'),
                        "severity": "low",
                        "context": self.get_function_context(function)
                    })
        
        # 5. Security risk analysis
        security_risks = self._analyze_security_risks()
        error_analysis["security_risks"].extend(security_risks)
        
        return error_analysis
    
    def _analyze_runtime_risks(self, function) -> List[Dict[str, Any]]:
        """Analyze potential runtime errors in a function."""
        risks = []
        
        # Check for unhandled exceptions in async functions
        if getattr(function, 'is_async', False):
            if not has_error_handling(function):
                risks.append({
                    "type": "unhandled_async_exception",
                    "function": function.name,
                    "file": getattr(function, 'file', {}).get('filepath', 'Unknown'),
                    "severity": "high",
                    "description": "Async function without error handling",
                    "context": self.get_function_context(function)
                })
        
        # Check for potential null pointer exceptions
        if hasattr(function, 'code_block') and function.code_block:
            source = getattr(function.code_block, 'source', '')
            if has_potential_null_references_in_source(source):
                risks.append({
                    "type": "potential_null_reference",
                    "function": function.name,
                    "file": getattr(function, 'file', {}).get('filepath', 'Unknown'),
                    "severity": "medium",
                    "description": "Potential null reference in function",
                    "context": self.get_function_context(function)
                })
        
        # Check for missing error handling in critical operations
        if has_unhandled_critical_operations(function):
            risks.append({
                "type": "unhandled_critical_operation",
                "function": function.name,
                "file": getattr(function, 'file', {}).get('filepath', 'Unknown'),
                "severity": "high",
                "description": "Critical operation without error handling",
                "context": self.get_function_context(function)
            })
        
        return risks
    
    def _detect_circular_dependencies(self) -> List[Dict[str, Any]]:
        """Detect circular dependencies in imports."""
        circular_deps = []
        file_imports = {}
        
        # Build import graph
        for file in self.codebase.files:
            filepath = getattr(file, 'filepath', '')
            imports = getattr(file, 'imports', [])
            file_imports[filepath] = []
            
            for imp in imports:
                if hasattr(imp, 'resolved_symbol') and hasattr(imp.resolved_symbol, 'file'):
                    imported_file = getattr(imp.resolved_symbol.file, 'filepath', '')
                    if imported_file:
                        file_imports[filepath].append(imported_file)
        
        # Simple cycle detection
        for file, imports in file_imports.items():
            for imported_file in imports:
                if imported_file in file_imports:
                    if file in file_imports[imported_file]:
                        circular_deps.append({
                            "type": "circular_import",
                            "files": [file, imported_file],
                            "severity": "high",
                            "description": f"Circular import between {file} and {imported_file}"
                        })
        
        return circular_deps
    
    def _analyze_security_risks(self) -> List[Dict[str, Any]]:
        """Analyze potential security vulnerabilities."""
        security_risks = []
        
        dangerous_patterns = [
            ('eval(', 'code_injection'),
            ('exec(', 'code_injection'),
            ('os.system(', 'command_injection'),
            ('subprocess.call(', 'command_injection'),
            ('pickle.loads(', 'deserialization'),
            ('yaml.load(', 'unsafe_deserialization'),
            ('input(', 'user_input_risk'),
            ('raw_input(', 'user_input_risk')
        ]
        
        for function in self.codebase.functions:
            if hasattr(function, 'code_block') and function.code_block:
                source = getattr(function.code_block, 'source', '')
                for pattern, risk_type in dangerous_patterns:
                    if pattern in source:
                        security_risks.append({
                            "type": risk_type,
                            "function": function.name,
                            "file": getattr(function, 'file', {}).get('filepath', 'Unknown'),
                            "pattern": pattern,
                            "severity": "critical",
                            "description": f"Potentially dangerous pattern: {pattern}",
                            "context": self.get_function_context(function)
                        })
        
        return security_risks
    
    def get_function_context(self, function) -> Dict[str, Any]:
        """Get comprehensive context for a function including all relationships."""
        context = {
            "implementation": {
                "source": getattr(function, 'source', ''),
                "filepath": getattr(function, 'file', {}).get('filepath', 'Unknown'),
                "line_count": len(getattr(function, 'source', '').split('\n')),
            },
            "dependencies": [],
            "usages": [],
            "call_graph": {},
            "control_flow": {},
            "error_patterns": {},
            "performance_indicators": {}
        }
        
        # Add dependencies with full context
        for dep in getattr(function, 'dependencies', []):
            dep_info = {
                "name": getattr(dep, 'name', str(dep)),
                "type": type(dep).__name__
            }
            if hasattr(dep, 'source'):
                dep_info["source"] = dep.source
            if hasattr(dep, 'file') and hasattr(dep.file, 'filepath'):
                dep_info["filepath"] = dep.file.filepath
            context["dependencies"].append(dep_info)
        
        # Add usage information
        for usage in getattr(function, 'usages', []):
            usage_info = {
                "type": type(usage).__name__
            }
            if hasattr(usage, 'usage_symbol'):
                if hasattr(usage.usage_symbol, 'source'):
                    usage_info["source"] = usage.usage_symbol.source
                if hasattr(usage.usage_symbol, 'file') and hasattr(usage.usage_symbol.file, 'filepath'):
                    usage_info["filepath"] = usage.usage_symbol.file.filepath
            context["usages"].append(usage_info)
        
        # Analyze control flow
        context["control_flow"] = analyze_control_flow(function)
        
        # Analyze error patterns
        context["error_patterns"] = analyze_error_patterns(function)
        
        # Performance indicators
        context["performance_indicators"] = analyze_performance_indicators(function)
        
        return context


def get_monthly_commits(repo_path: str) -> Dict[str, int]:
    """
    Get the number of commits per month for the last 12 months.

    Args:
        repo_path: Path to the git repository

    Returns:
        Dictionary with month-year as key and number of commits as value
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    date_format = "%Y-%m-%d"
    since_date = start_date.strftime(date_format)
    until_date = end_date.strftime(date_format)
    repo_path = "https://github.com/" + repo_path

    try:
        original_dir = os.getcwd()

        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(["git", "clone", repo_path, temp_dir], check=True)
            os.chdir(temp_dir)

            cmd = [
                "git",
                "log",
                f"--since={since_date}",
                f"--until={until_date}",
                "--format=%aI",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            commit_dates = result.stdout.strip().split("\n")

            monthly_counts = {}
            current_date = start_date
            while current_date <= end_date:
                month_key = current_date.strftime("%Y-%m")
                monthly_counts[month_key] = 0
                current_date = (
                    current_date.replace(day=1) + timedelta(days=32)
                ).replace(day=1)

            for date_str in commit_dates:
                if date_str:  # Skip empty lines
                    commit_date = datetime.fromisoformat(date_str.strip())
                    month_key = commit_date.strftime("%Y-%m")
                    if month_key in monthly_counts:
                        monthly_counts[month_key] += 1

            os.chdir(original_dir)
            return dict(sorted(monthly_counts.items()))

    except subprocess.CalledProcessError as e:
        print(f"Error executing git command: {e}")
        return {}
    except Exception as e:
        print(f"Error processing git commits: {e}")
        return {}
    finally:
        try:
            os.chdir(original_dir)
        except:
            pass


def calculate_cyclomatic_complexity(function):
    def analyze_statement(statement):
        complexity = 0

        if isinstance(statement, IfBlockStatement):
            complexity += 1
            if hasattr(statement, "elif_statements"):
                complexity += len(statement.elif_statements)

        elif isinstance(statement, (ForLoopStatement, WhileStatement)):
            complexity += 1

        elif isinstance(statement, TryCatchStatement):
            complexity += len(getattr(statement, "except_blocks", []))

        if hasattr(statement, "condition") and isinstance(statement.condition, str):
            complexity += statement.condition.count(
                " and "
            ) + statement.condition.count(" or ")

        if hasattr(statement, "nested_code_blocks"):
            for block in statement.nested_code_blocks:
                complexity += analyze_block(block)

        return complexity

    def analyze_block(block):
        if not block or not hasattr(block, "statements"):
            return 0
        return sum(analyze_statement(stmt) for stmt in block.statements)

    return (
        1 + analyze_block(function.code_block) if hasattr(function, "code_block") else 1
    )


def cc_rank(complexity):
    if complexity < 0:
        raise ValueError("Complexity must be a non-negative value")

    ranks = [
        (1, 5, "A"),
        (6, 10, "B"),
        (11, 20, "C"),
        (21, 30, "D"),
        (31, 40, "E"),
        (41, float("inf"), "F"),
    ]
    for low, high, rank in ranks:
        if low <= complexity <= high:
            return rank
    return "F"


def calculate_doi(cls):
    """Calculate the depth of inheritance for a given class."""
    return len(cls.superclasses)


def get_operators_and_operands(function):
    operators = []
    operands = []

    for statement in function.code_block.statements:
        for call in statement.function_calls:
            operators.append(call.name)
            for arg in call.args:
                operands.append(arg.source)

        if hasattr(statement, "expressions"):
            for expr in statement.expressions:
                if isinstance(expr, BinaryExpression):
                    operators.extend([op.source for op in expr.operators])
                    operands.extend([elem.source for elem in expr.elements])
                elif isinstance(expr, UnaryExpression):
                    operators.append(expr.ts_node.type)
                    operands.append(expr.argument.source)
                elif isinstance(expr, ComparisonExpression):
                    operators.extend([op.source for op in expr.operators])
                    operands.extend([elem.source for elem in expr.elements])

        if hasattr(statement, "expression"):
            expr = statement.expression
            if isinstance(expr, BinaryExpression):
                operators.extend([op.source for op in expr.operators])
                operands.extend([elem.source for elem in expr.elements])
            elif isinstance(expr, UnaryExpression):
                operators.append(expr.ts_node.type)
                operands.append(expr.argument.source)
            elif isinstance(expr, ComparisonExpression):
                operators.extend([op.source for op in expr.operators])
                operands.extend([elem.source for elem in expr.elements])

    return operators, operands


def calculate_halstead_volume(operators, operands):
    n1 = len(set(operators))
    n2 = len(set(operands))

    N1 = len(operators)
    N2 = len(operands)

    N = N1 + N2
    n = n1 + n2

    if n > 0:
        volume = N * math.log2(n)
        return volume, N1, N2, n1, n2
    return 0, N1, N2, n1, n2


def count_lines(source: str):
    """Count different types of lines in source code."""
    if not source.strip():
        return 0, 0, 0, 0

    lines = [line.strip() for line in source.splitlines()]
    loc = len(lines)
    sloc = len([line for line in lines if line])

    in_multiline = False
    comments = 0
    code_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]
        code_part = line
        if not in_multiline and "#" in line:
            comment_start = line.find("#")
            if not re.search(r'["\'].*#.*["\']', line[:comment_start]):
                code_part = line[:comment_start].strip()
                if line[comment_start:].strip():
                    comments += 1

        if ('"""' in line or "'''" in line) and not (
            line.count('"""') % 2 == 0 or line.count("'''") % 2 == 0
        ):
            if in_multiline:
                in_multiline = False
                comments += 1
            else:
                in_multiline = True
                comments += 1
                if line.strip().startswith('"""') or line.strip().startswith("'''"):
                    code_part = ""
        elif in_multiline:
            comments += 1
            code_part = ""
        elif line.strip().startswith("#"):
            comments += 1
            code_part = ""

        if code_part.strip():
            code_lines.append(code_part)

        i += 1

    lloc = 0
    continued_line = False
    for line in code_lines:
        if continued_line:
            if not any(line.rstrip().endswith(c) for c in ("\\", ",", "{", "[", "(")):
                continued_line = False
            continue

        lloc += len([stmt for stmt in line.split(";") if stmt.strip()])

        if any(line.rstrip().endswith(c) for c in ("\\", ",", "{", "[", "(")):
            continued_line = True

    return loc, lloc, sloc, comments


def calculate_maintainability_index(
    halstead_volume: float, cyclomatic_complexity: float, loc: int
) -> int:
    """Calculate the normalized maintainability index for a given function."""
    if loc <= 0:
        return 100

    try:
        raw_mi = (
            171
            - 5.2 * math.log(max(1, halstead_volume))
            - 0.23 * cyclomatic_complexity
            - 16.2 * math.log(max(1, loc))
        )
        normalized_mi = max(0, min(100, raw_mi * 100 / 171))
        return int(normalized_mi)
    except (ValueError, TypeError):
        return 0


def get_maintainability_rank(mi_score: float) -> str:
    """Convert maintainability index score to a letter grade."""
    if mi_score >= 85:
        return "A"
    elif mi_score >= 65:
        return "B"
    elif mi_score >= 45:
        return "C"
    elif mi_score >= 25:
        return "D"
    else:
        return "F"


def get_github_repo_description(repo_url):
    api_url = f"https://api.github.com/repos/{repo_url}"

    response = requests.get(api_url)

    if response.status_code == 200:
        repo_data = response.json()
        return repo_data.get("description", "No description available")
    else:
        return ""


class RepoRequest(BaseModel):
    repo_url: str


@fastapi_app.post("/analyze_repo")
async def analyze_repo(request: RepoRequest) -> Dict[str, Any]:
    """Analyze a repository and return comprehensive metrics."""
    repo_url = request.repo_url
    
    try:
        # Initialize codebase with advanced configuration
        config = CodebaseConfig(
            method_usages=True,          # Enable method usage resolution
            generics=True,               # Enable generic type resolution
            sync_enabled=True,           # Enable graph sync during commits
            full_range_index=True,       # Full range-to-node mapping
            py_resolve_syspath=True,     # Resolve sys.path imports
            exp_lazy_graph=False,        # Lazy graph construction
        )
        
        codebase = Codebase.from_repo(repo_url, config=config)
        
        # Perform comprehensive analysis
        analyzer = ComprehensiveAnalysis(codebase)
        analysis_results = analyzer.analyze()
        
        # Get traditional metrics for backward compatibility
        monthly_commits = get_monthly_commits(repo_url)
        desc = get_github_repo_description(repo_url)
        
        # Calculate traditional line metrics
        total_loc = total_lloc = total_sloc = total_comments = 0
        total_complexity = 0
        total_volume = 0
        total_mi = 0
        total_doi = 0
        
        for file in codebase.files:
            try:
                source = getattr(file, 'source', '')
                if source:
                    loc, lloc, sloc, comments = count_lines(source)
                    total_loc += loc
                    total_lloc += lloc
                    total_sloc += sloc
                    total_comments += comments
            except Exception as e:
                logging.warning(f"Error processing file {getattr(file, 'filepath', 'Unknown')}: {str(e)}")
        
        # Calculate complexity metrics for functions
        num_callables = 0
        for func in codebase.functions:
            try:
                if hasattr(func, "code_block") and func.code_block:
                    complexity = calculate_cyclomatic_complexity(func)
                    operators, operands = get_operators_and_operands(func)
                    volume, _, _, _, _ = calculate_halstead_volume(operators, operands)
                    loc = len(getattr(func.code_block, 'source', '').splitlines())
                    mi_score = calculate_maintainability_index(volume, complexity, loc)

                    total_complexity += complexity
                    total_volume += volume
                    total_mi += mi_score
                    num_callables += 1
            except Exception as e:
                logging.warning(f"Error processing function {func.name}: {str(e)}")
        
        # Calculate class metrics
        for cls in codebase.classes:
            try:
                doi = calculate_doi(cls)
                total_doi += doi
            except Exception as e:
                logging.warning(f"Error processing class {cls.name}: {str(e)}")
        
        # Combine comprehensive analysis with traditional metrics
        results = {
            "repo_url": repo_url,
            "description": desc,
            "monthly_commits": monthly_commits,
            
            # Traditional metrics for backward compatibility
            "line_metrics": {
                "total": {
                    "loc": total_loc,
                    "lloc": total_lloc,
                    "sloc": total_sloc,
                    "comments": total_comments,
                    "comment_density": (total_comments / total_loc * 100) if total_loc > 0 else 0,
                },
            },
            "cyclomatic_complexity": {
                "average": total_complexity / num_callables if num_callables > 0 else 0,
            },
            "depth_of_inheritance": {
                "average": total_doi / len(list(codebase.classes)) if len(list(codebase.classes)) > 0 else 0,
            },
            "halstead_metrics": {
                "total_volume": int(total_volume),
                "average_volume": int(total_volume / num_callables) if num_callables > 0 else 0,
            },
            "maintainability_index": {
                "average": int(total_mi / num_callables) if num_callables > 0 else 0,
            },
            "num_files": len(list(codebase.files)),
            "num_functions": len(list(codebase.functions)),
            "num_classes": len(list(codebase.classes)),
            
            # NEW: Comprehensive analysis results
            **analysis_results
        }
        
        return results
        
    except Exception as e:
        logging.error(f"Repository analysis failed: {str(e)}")
        return {
            "error": f"Repository analysis failed: {str(e)}",
            "repo_url": repo_url,
            "traceback": traceback.format_exc()
        }


@app.function(image=image)
@modal.asgi_app()
def fastapi_modal_app():
    return fastapi_app


if __name__ == "__main__":
    app.deploy("analytics-app")
