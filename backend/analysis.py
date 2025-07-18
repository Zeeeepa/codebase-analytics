"""
Codebase Analytics - Core Analysis Engine
Comprehensive analysis engine for code quality, issue detection, and metrics calculation
"""

import re
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import json
from collections import defaultdict
import math

import graph_sitter
from graph_sitter.codebase.codebase_analysis import get_codebase_summary, get_file_summary, get_class_summary, get_function_summary, get_symbol_summary
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

from models import (
    CodeIssue, IssueType, IssueSeverity, FunctionContext,
    AnalysisResults, AutomatedResolution, AnalysisConfig
)



# ============================================================================
# ADVANCED ANALYSIS COMPONENTS
# ============================================================================

class GraphSitterAnalyzer:
    """Enhanced analyzer using graph-sitter's relationship tracking"""
    
    def __init__(self, codebase=None):
        self.codebase = codebase
        self.analysis_results = {}
        
    def analyze_repository(self, repo_path_or_url: str) -> Dict[str, Any]:
        """Perform comprehensive analysis of a repository"""
        print(f"🔍 Starting analysis of: {repo_path_or_url}")
        
        try:
            # Initialize codebase
            if GRAPH_SITTER_AVAILABLE:
                if repo_path_or_url.startswith('http'):
                    # Remote repository - extract repo name from URL
                    repo_name = repo_path_or_url.split('/')[-2] + '/' + repo_path_or_url.split('/')[-1]
                    if repo_name.endswith('.git'):
                        repo_name = repo_name[:-4]
                    self.codebase = Codebase.from_repo(repo_name)
                else:
                    # Local repository
                    self.codebase = Codebase(repo_path_or_url)
            else:
                raise ImportError("Graph-sitter not available")
            
            print(f"✅ Codebase loaded: {len(self.codebase.files)} files")
            
            # Perform comprehensive analysis
            results = {
                "repository_facts": self._analyze_repository_facts(),
                "most_important_files": self._find_most_important_files(),
                "entry_points": self._detect_entry_points(),
                "repository_structure": self._build_tree_structure(),
                "actual_errors": self._detect_runtime_errors(),
                "error_summary": self._generate_error_summary(),
                "analysis_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "analyzer_version": "2.0.0",
                    "graph_sitter_enabled": GRAPH_SITTER_AVAILABLE
                }
            }
            
            self.analysis_results = results
            return results
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            return {
                "error": str(e),
                "analysis_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "analyzer_version": "2.0.0",
                    "graph_sitter_enabled": GRAPH_SITTER_AVAILABLE
                }
            }
    
    def _analyze_repository_facts(self) -> Dict[str, Any]:
        """Analyze basic repository facts"""
        if not self.codebase:
            return {}
        
        # Get all files
        all_files = list(self.codebase.files)
        
        # Categorize files
        code_files = [f for f in all_files if self._is_code_file(f)]
        doc_files = [f for f in all_files if self._is_doc_file(f)]
        config_files = [f for f in all_files if self._is_config_file(f)]
        
        # Language analysis
        languages = defaultdict(int)
        for file in code_files:
            lang = self._detect_language(file)
            languages[lang] += 1
        
        # Function and class counts
        total_functions = 0
        total_classes = 0
        
        for file in code_files:
            try:
                if hasattr(file, 'functions'):
                    total_functions += len(file.functions)
                if hasattr(file, 'classes'):
                    total_classes += len(file.classes)
            except:
                continue
        
        return {
            "total_files": len(all_files),
            "code_files": len(code_files),
            "documentation_files": len(doc_files),
            "config_files": len(config_files),
            "total_functions": total_functions,
            "total_classes": total_classes,
            "languages": dict(languages),
            "entry_points_detected": 0  # Will be updated later
        }
    
    def _find_most_important_files(self) -> List[Dict[str, Any]]:
        """Find most important files using graph-sitter relationship data"""
        if not self.codebase:
            return []
        
        file_importance = []
        
        for file in self.codebase.files:
            if not self._is_code_file(file):
                continue
                
            try:
                # Calculate importance metrics using graph-sitter data
                total_usages = 0
                total_dependencies = 0
                entry_points_count = 0
                function_count = 0
                
                if hasattr(file, 'functions'):
                    for func in file.functions:
                        function_count += 1
                        
                        # Count usages (how often this function is called)
                        if hasattr(func, 'usages'):
                            total_usages += len(func.usages)
                        
                        # Count dependencies (what this function depends on)
                        if hasattr(func, 'dependencies'):
                            total_dependencies += len(func.dependencies)
                        
                        # Check if it's an entry point
                        if self._is_entry_point_function(func):
                            entry_points_count += 1
                
                # Calculate importance score
                importance_score = (
                    total_usages * 3 +           # How much it's used
                    total_dependencies * 1 +     # How complex it is
                    entry_points_count * 20 +    # Entry points bonus
                    function_count * 2           # Function density
                )
                
                file_importance.append({
                    "rank": 0,  # Will be set after sorting
                    "filepath": file.filepath,
                    "importance_score": importance_score,
                    "usage_count": total_usages,
                    "dependency_count": total_dependencies,
                    "function_count": function_count,
                    "entry_points": entry_points_count,
                    "is_entry_file": entry_points_count > 0,
                    "language": self._detect_language(file)
                })
                
            except Exception as e:
                print(f"Warning: Could not analyze file {file.filepath}: {e}")
                continue
        
        # Sort by importance and add ranks
        file_importance.sort(key=lambda x: x["importance_score"], reverse=True)
        for i, file_info in enumerate(file_importance[:8]):
            file_info["rank"] = i + 1
        
        return file_importance[:8]
    
    def _detect_entry_points(self) -> List[Dict[str, Any]]:
        """Detect entry points using graph-sitter's call graph analysis"""
        if not self.codebase:
            return []
        
        entry_points = []
        
        for file in self.codebase.files:
            if not self._is_code_file(file):
                continue
                
            try:
                if hasattr(file, 'functions'):
                    for func in file.functions:
                        if self._is_entry_point_function(func):
                            # Calculate metrics using graph-sitter data
                            usage_count = len(func.usages) if hasattr(func, 'usages') else 0
                            calls_count = len(func.function_calls) if hasattr(func, 'function_calls') else 0
                            
                            # Calculate importance score
                            importance_score = self._calculate_function_importance(func)
                            
                            entry_points.append({
                                "function_name": func.name,
                                "filepath": file.filepath,
                                "line_number": getattr(func, 'start_point', [0])[0],
                                "importance_score": importance_score,
                                "usage_count": usage_count,
                                "calls_count": calls_count,
                                "entry_type": self._determine_entry_type(func),
                                "call_chain_depth": self._calculate_call_chain_depth(func)
                            })
            except Exception as e:
                print(f"Warning: Could not analyze functions in {file.filepath}: {e}")
                continue
        
        # Sort by importance
        entry_points.sort(key=lambda x: x["importance_score"], reverse=True)
        return entry_points
    
    def _build_tree_structure(self) -> Dict[str, Any]:
        """Build repository tree structure with relationship data"""
        if not self.codebase:
            return {}
        
        tree_structure = {}
        
        for file in self.codebase.files:
            try:
                path_parts = file.filepath.split('/')
                current = tree_structure
                
                # Build directory structure
                for part in path_parts[:-1]:
                    if part not in current:
                        current[part] = {
                            "type": "directory",
                            "children": {},
                            "actual_errors": 0,
                            "total_functions": 0,
                            "total_usages": 0
                        }
                    current = current[part]["children"]
                
                # Add file with analysis
                filename = path_parts[-1]
                file_analysis = self._analyze_file_relationships(file)
                current[filename] = file_analysis
                
            except Exception as e:
                print(f"Warning: Could not process file {file.filepath}: {e}")
                continue
        
        return tree_structure
    
    def _analyze_file_relationships(self, file) -> Dict[str, Any]:
        """Analyze file using graph-sitter's relationship tracking"""
        file_data = {
            "type": "file",
            "filepath": file.filepath,
            "language": self._detect_language(file),
            "file_type": "code" if self._is_code_file(file) else "other",
            "functions": [],
            "classes": [],
            "imports": [],
            "actual_errors": 0,
            "error_types": [],
            "relationship_metrics": {
                "total_usages": 0,
                "total_dependencies": 0,
                "external_dependencies": 0,
                "internal_dependencies": 0
            }
        }
        
        if not self._is_code_file(file):
            return file_data
        
        try:
            # Analyze functions
            if hasattr(file, 'functions'):
                for func in file.functions:
                    func_data = self._analyze_function_relationships(func)
                    file_data["functions"].append(func_data)
                    file_data["relationship_metrics"]["total_usages"] += func_data.get("usage_count", 0)
                    file_data["relationship_metrics"]["total_dependencies"] += func_data.get("dependency_count", 0)
            
            # Analyze classes
            if hasattr(file, 'classes'):
                for cls in file.classes:
                    class_data = {
                        "name": cls.name,
                        "methods": [method.name for method in cls.methods] if hasattr(cls, 'methods') else [],
                        "attributes": [attr.name for attr in cls.attributes] if hasattr(cls, 'attributes') else [],
                        "usage_count": len(cls.usages) if hasattr(cls, 'usages') else 0,
                        "is_important": len(cls.usages) > 5 if hasattr(cls, 'usages') else False
                    }
                    file_data["classes"].append(class_data)
            
            # Analyze imports
            if hasattr(file, 'imports'):
                for imp in file.imports:
                    import_data = {
                        "module": getattr(imp, 'module', str(imp)),
                        "is_external": self._is_external_import(imp)
                    }
                    file_data["imports"].append(import_data)
                    
                    if import_data["is_external"]:
                        file_data["relationship_metrics"]["external_dependencies"] += 1
                    else:
                        file_data["relationship_metrics"]["internal_dependencies"] += 1
        
        except Exception as e:
            print(f"Warning: Could not analyze relationships in {file.filepath}: {e}")
        
        return file_data
    
    def _analyze_function_relationships(self, func) -> Dict[str, Any]:
        """Analyze function using graph-sitter's relationship data"""
        try:
            usage_count = len(func.usages) if hasattr(func, 'usages') else 0
            dependency_count = len(func.dependencies) if hasattr(func, 'dependencies') else 0
            calls_count = len(func.function_calls) if hasattr(func, 'function_calls') else 0
            
            # Detect actual errors in this function
            errors = self._detect_function_errors(func)
            
            return {
                "name": func.name,
                "line_start": getattr(func, 'start_point', [0])[0],
                "line_end": getattr(func, 'end_point', [0])[0],
                "is_entry_point": self._is_entry_point_function(func),
                "usage_count": usage_count,
                "dependency_count": dependency_count,
                "calls_count": calls_count,
                "actual_errors": len(errors),
                "errors": errors,
                "parameters": [{"name": p.name, "type": getattr(p, 'type', None)} 
                              for p in func.parameters] if hasattr(func, 'parameters') else [],
                "return_type": getattr(func, 'return_type', None)
            }
        except Exception as e:
            print(f"Warning: Could not analyze function {func.name}: {e}")
            return {
                "name": getattr(func, 'name', 'unknown'),
                "line_start": 0,
                "line_end": 0,
                "is_entry_point": False,
                "usage_count": 0,
                "dependency_count": 0,
                "calls_count": 0,
                "actual_errors": 0,
                "errors": [],
                "parameters": [],
                "return_type": None
            }
    
    def _detect_runtime_errors(self) -> List[Dict[str, Any]]:
        """Detect actual runtime errors using graph-sitter analysis"""
        if not self.codebase:
            return []
        
        errors = []
        error_id = 1
        
        for file in self.codebase.files:
            if not self._is_code_file(file):
                continue
                
            try:
                if hasattr(file, 'functions'):
                    for func in file.functions:
                        func_errors = self._detect_function_errors(func)
                        for error in func_errors:
                            error["id"] = error_id
                            error["filepath"] = file.filepath
                            error["function_name"] = func.name
                            errors.append(error)
                            error_id += 1
            except Exception as e:
                print(f"Warning: Could not detect errors in {file.filepath}: {e}")
                continue
        
        return errors
    
    def _detect_function_errors(self, func) -> List[Dict[str, Any]]:
        """Detect actual runtime errors in a function"""
        errors = []
        
        try:
            source = getattr(func, 'source', '')
            if not source:
                return errors
            
            # Null reference detection
            if '.get(' in source and 'if' not in source:
                errors.append({
                    "error_type": "null_reference",
                    "line_number": self._find_line_with_pattern(source, '.get('),
                    "description": "Potential null reference: .get() without null check",
                    "code_context": self._get_code_context(source, '.get('),
                    "auto_fix_available": True,
                    "fix_suggestion": "Add null check before using .get() result",
                    "runtime_impact": "AttributeError when object is None"
                })
            
            # Undefined function calls (using graph-sitter's resolution)
            if hasattr(func, 'function_calls'):
                for call in func.function_calls:
                    if not hasattr(call, 'function_definition') or not call.function_definition:
                        errors.append({
                            "error_type": "undefined_function",
                            "line_number": getattr(call, 'start_point', [0])[0],
                            "description": f"Undefined function call: {call.name}",
                            "code_context": f"Function '{call.name}' is called but not defined",
                            "auto_fix_available": False,
                            "fix_suggestion": f"Define function '{call.name}' or check import",
                            "runtime_impact": "NameError at runtime"
                        })
            
            # Division by zero detection
            if '/' in source and 'if' not in source:
                # Simple heuristic for potential division by zero
                if re.search(r'/\s*\w+(?!\w)', source):
                    errors.append({
                        "error_type": "division_by_zero",
                        "line_number": self._find_line_with_pattern(source, '/'),
                        "description": "Potential division by zero",
                        "code_context": self._get_code_context(source, '/'),
                        "auto_fix_available": False,
                        "fix_suggestion": "Add zero check before division",
                        "runtime_impact": "ZeroDivisionError at runtime"
                    })
            
            # Type mismatch detection (basic)
            if 'str(' in source and 'int(' in source:
                errors.append({
                    "error_type": "type_mismatch",
                    "line_number": self._find_line_with_pattern(source, 'str('),
                    "description": "Potential type mismatch: mixing str() and int() conversions",
                    "code_context": self._get_code_context(source, 'str('),
                    "auto_fix_available": True,
                    "fix_suggestion": "Ensure consistent type handling",
                    "runtime_impact": "TypeError when types don't match"
                })
            
        except Exception as e:
            print(f"Warning: Could not detect errors in function {func.name}: {e}")
        
        return errors
    
    def _generate_error_summary(self) -> Dict[str, Any]:
        """Generate summary of detected errors"""
        if "actual_errors" not in self.analysis_results:
            return {}
        
        errors = self.analysis_results.get("actual_errors", [])
        
        # Count by type
        by_type = defaultdict(int)
        auto_fixable = 0
        
        for error in errors:
            by_type[error["error_type"]] += 1
            if error.get("auto_fix_available", False):
                auto_fixable += 1
        
        return {
            "total_actual_errors": len(errors),
            "auto_fixable": auto_fixable,
            "manual_review_required": len(errors) - auto_fixable,
            "by_type": dict(by_type),
            "by_severity": {
                "will_crash_runtime": len([e for e in errors if e["error_type"] in 
                                         ["null_reference", "undefined_function", "division_by_zero"]]),
                "potential_runtime_issues": len([e for e in errors if e["error_type"] in 
                                               ["type_mismatch", "resource_leak"]])
            }
        }
    
    # Helper methods
    def _is_code_file(self, file) -> bool:
        """Check if file is a code file"""
        code_extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h']
        return any(file.filepath.endswith(ext) for ext in code_extensions)
    
    def _is_doc_file(self, file) -> bool:
        """Check if file is a documentation file"""
        doc_extensions = ['.md', '.rst', '.txt', '.doc', '.docx']
        return any(file.filepath.endswith(ext) for ext in doc_extensions)
    
    def _is_config_file(self, file) -> bool:
        """Check if file is a configuration file"""
        config_extensions = ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.env']
        config_names = ['requirements.txt', 'package.json', 'Dockerfile', 'Makefile']
        return (any(file.filepath.endswith(ext) for ext in config_extensions) or
                any(name in file.filepath for name in config_names))
    
    def _detect_language(self, file) -> str:
        """Detect programming language of file"""
        if file.filepath.endswith('.py'):
            return 'python'
        elif file.filepath.endswith(('.js', '.jsx')):
            return 'javascript'
        elif file.filepath.endswith(('.ts', '.tsx')):
            return 'typescript'
        elif file.filepath.endswith('.java'):
            return 'java'
        elif file.filepath.endswith(('.cpp', '.cc', '.cxx')):
            return 'cpp'
        elif file.filepath.endswith('.c'):
            return 'c'
        else:
            return 'unknown'
    
    def _is_entry_point_function(self, func) -> bool:
        """Check if function is an entry point"""
        entry_patterns = ['main', '__main__', 'app', 'run', 'start', 'cli', 'init']
        name = func.name.lower()
        
        # Pattern matching
        if any(pattern in name for pattern in entry_patterns):
            return True
        
        # HTTP endpoint detection
        if hasattr(func, 'is_method') and func.is_method:
            http_methods = ['get', 'post', 'put', 'delete', 'patch', 'head', 'options']
            if name in http_methods:
                return True
        
        # High usage indicates importance
        if hasattr(func, 'usages') and len(func.usages) > 10:
            return True
        
        return False
    
    def _calculate_function_importance(self, func) -> int:
        """Calculate function importance score"""
        score = 0
        
        # Usage frequency
        if hasattr(func, 'usages'):
            score += len(func.usages) * 5
        
        # Function calls (complexity)
        if hasattr(func, 'function_calls'):
            score += len(func.function_calls) * 2
        
        # Dependencies
        if hasattr(func, 'dependencies'):
            score += len(func.dependencies) * 1
        
        # Entry point bonus
        if self._is_entry_point_function(func):
            score += 50
        
        return min(score, 100)
    
    def _determine_entry_type(self, func) -> str:
        """Determine the type of entry point"""
        name = func.name.lower()
        
        if 'main' in name:
            return 'application_startup'
        elif name in ['app', 'create_app']:
            return 'application_factory'
        elif 'cli' in name:
            return 'command_line_interface'
        elif name in ['get', 'post', 'put', 'delete', 'patch']:
            return 'api_endpoint'
        elif 'run' in name or 'start' in name:
            return 'service_runner'
        else:
            return 'other'
    
    def _calculate_call_chain_depth(self, func) -> int:
        """Calculate maximum call chain depth"""
        try:
            if not hasattr(func, 'function_calls'):
                return 1
            
            # Simple depth calculation (avoid infinite recursion)
            max_depth = 1
            visited = set()
            
            def calculate_depth(f, depth=1):
                if depth > 10 or f.name in visited:  # Prevent infinite recursion
                    return depth
                
                visited.add(f.name)
                current_max = depth
                
                if hasattr(f, 'function_calls'):
                    for call in f.function_calls:
                        if hasattr(call, 'function_definition') and call.function_definition:
                            call_depth = calculate_depth(call.function_definition, depth + 1)
                            current_max = max(current_max, call_depth)
                
                return current_max
            
            return calculate_depth(func)
        except:
            return 1
    
    def _is_external_import(self, imp) -> bool:
        """Check if import is external (not from current codebase)"""
        try:
            module_name = getattr(imp, 'module', str(imp))
            # Simple heuristic: if it doesn't start with '.', it's likely external
            return not module_name.startswith('.')
        except:
            return True
    
    def _find_line_with_pattern(self, source: str, pattern: str) -> int:
        """Find line number containing pattern"""
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if pattern in line:
                return i + 1
        return 1
    
    def _get_code_context(self, source: str, pattern: str) -> str:
        """Get code context around pattern"""
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if pattern in line:
                # Return 3 lines of context
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                context_lines = []
                for j in range(start, end):
                    prefix = "► " if j == i else "  "
                    context_lines.append(f"Line {j+1}: {prefix}{lines[j]}")
                return "\n".join(context_lines)
        return f"Pattern '{pattern}' found in source"
        
class ImportResolver:
    """Advanced import resolution and automated import fixing"""
    
    def __init__(self):
        self.import_map = {}
        self.symbol_map = {}
        self.unused_imports = []
        self.missing_imports = []
    
    def analyze_imports(self, codebase: Codebase):
        """Analyze imports across the codebase"""
        self._build_import_map(codebase)
        self._build_symbol_map(codebase)
        self._detect_unused_imports(codebase)
        self._detect_missing_imports(codebase)
    
    def _build_import_map(self, codebase: Codebase):
        """Build map of available imports"""
        for source_file in codebase.files:
            if hasattr(source_file, 'imports'):
                for imp in source_file.imports:
                    if hasattr(imp, 'module_name'):
                        self.import_map[imp.module_name] = source_file.file_path
    
    def _build_symbol_map(self, codebase: Codebase):
        """Build map of available symbols"""
        for source_file in codebase.files:
            for symbol in source_file.symbols:
                if isinstance(symbol, Function):
                    self.symbol_map[symbol.name] = source_file.file_path
                elif isinstance(symbol, Class):
                    self.symbol_map[symbol.name] = source_file.file_path
    
    def _detect_unused_imports(self, codebase: Codebase):
        """Detect unused imports"""
        for source_file in codebase.files:
            if hasattr(source_file, 'imports') and source_file.source:
                for imp in source_file.imports:
                    if hasattr(imp, 'module_name'):
                        if imp.module_name not in source_file.source:
                            self.unused_imports.append({
                                'name': imp.module_name,
                                'file': source_file.file_path,
                                'line': getattr(imp, 'line_number', 1)
                            })
    
    def _detect_missing_imports(self, codebase: Codebase):
        """Detect missing imports"""
        for source_file in codebase.files:
            if source_file.source:
                for symbol, filepath in self.symbol_map.items():
                    if symbol in source_file.source and filepath != source_file.file_path:
                        if not self._is_imported(source_file, symbol):
                            self.missing_imports.append({
                                'symbol': symbol,
                                'file': source_file.file_path,
                                'source_file': filepath
                            })
    
    def _is_imported(self, source_file: SourceFile, symbol: str) -> bool:
        """Check if symbol is already imported"""
        if hasattr(source_file, 'imports'):
            for imp in source_file.imports:
                if hasattr(imp, 'module_name') and symbol in str(imp):
                    return True
        return False
    
    def generate_import_fixes(self) -> List[AutomatedResolution]:
        """Generate automated import fixes"""
        fixes = []
        
        # Remove unused imports
        for unused in self.unused_imports:
            fixes.append(AutomatedResolution(
                resolution_type="remove_unused_import",
                description=f"Remove unused import: {unused['name']}",
                original_code=f"import {unused['name']}",
                fixed_code="",
                confidence=0.95,
                file_path=unused['file'],
                line_number=unused['line'],
                is_safe=True,
                requires_validation=False
            ))
        
        # Add missing imports
        for missing in self.missing_imports:
            import_path = missing['source_file'].replace('/', '.').replace('.py', '')
            fixes.append(AutomatedResolution(
                resolution_type="add_missing_import",
                description=f"Add missing import: {missing['symbol']}",
                original_code="",
                fixed_code=f"from {import_path} import {missing['symbol']}",
                confidence=0.90,
                file_path=missing['file'],
                line_number=1,
                is_safe=True,
                requires_validation=False
            ))
        
        return fixes


class DeadCodeAnalyzer:
    """Advanced dead code detection with blast radius analysis"""
    
    def __init__(self):
        self.dead_functions = []
        self.dead_classes = []
        self.dead_variables = []
        self.blast_radius = {}
    
    def analyze_dead_code(self, codebase: Codebase, call_graph: Dict[str, List[str]]) -> Dict[str, Any]:
        """Perform comprehensive dead code analysis"""
        
        # Find unused functions
        self._find_dead_functions(codebase, call_graph)
        
        # Find unused classes
        self._find_dead_classes(codebase)
        
        # Calculate blast radius for removals
        self._calculate_blast_radius(codebase, call_graph)
        
        return {
            "dead_functions": self.dead_functions,
            "dead_classes": self.dead_classes,
            "dead_variables": self.dead_variables,
            "blast_radius": self.blast_radius,
            "safe_removals": len([f for f in self.dead_functions if self.blast_radius.get(f, {}).get('safe', False)]),
            "total_dead_code_lines": sum(self.blast_radius.get(f, {}).get('lines', 0) for f in self.dead_functions)
        }
    
    def _find_dead_functions(self, codebase: Codebase, call_graph: Dict[str, List[str]]):
        """Find functions that are never called"""
        all_functions = set()
        called_functions = set()
        
        # Collect all functions and called functions
        for source_file in codebase.files:
            for symbol in source_file.symbols:
                if isinstance(symbol, Function):
                    func_name = f"{source_file.file_path}::{symbol.name}"
                    all_functions.add(func_name)
                    
                    # Add called functions
                    if func_name in call_graph:
                        called_functions.update(call_graph[func_name])
        
        # Find entry points (functions that should not be considered dead)
        entry_points = set()
        for func_name in all_functions:
            simple_name = func_name.split("::")[-1].lower()
            if any(pattern in simple_name for pattern in ['main', 'init', 'setup', 'run', 'start', 'app', 'server']):
                entry_points.add(func_name)
        
        # Dead functions are those not called and not entry points
        self.dead_functions = list(all_functions - called_functions - entry_points)
    
    def _find_dead_classes(self, codebase: Codebase):
        """Find classes that are never instantiated"""
        all_classes = set()
        used_classes = set()
        
        for source_file in codebase.files:
            for symbol in source_file.symbols:
                if isinstance(symbol, Class):
                    class_name = f"{source_file.file_path}::{symbol.name}"
                    all_classes.add(class_name)
                    
                    # Simple heuristic: if class name appears in source, it's used
                    if source_file.source and symbol.name in source_file.source:
                        used_classes.add(class_name)
        
        self.dead_classes = list(all_classes - used_classes)
    
    def _calculate_blast_radius(self, codebase: Codebase, call_graph: Dict[str, List[str]]):
        """Calculate the impact of removing dead code"""
        for dead_func in self.dead_functions:
            # Find the function in codebase
            file_path, func_name = dead_func.split("::", 1)
            
            for source_file in codebase.files:
                if source_file.file_path == file_path:
                    for symbol in source_file.symbols:
                        if isinstance(symbol, Function) and symbol.name == func_name:
                            lines = symbol.line_end - symbol.line_start if hasattr(symbol, 'line_end') else 10
                            
                            self.blast_radius[dead_func] = {
                                'lines': lines,
                                'safe': lines < 50,  # Functions under 50 lines are safer to remove
                                'dependencies': call_graph.get(dead_func, []),
                                'dependents': []  # Functions that call this one
                            }
                            break


class RepositoryStructureAnalyzer:
    """Advanced repository structure analysis"""
    
    def __init__(self):
        self.structure = {}
        self.hotspots = []
        self.complexity_map = {}
    
    def analyze_structure(self, codebase: Codebase, issues: List[CodeIssue]) -> Dict[str, Any]:
        """Analyze repository structure with issue mapping"""
        
        # Build directory structure
        self._build_directory_structure(codebase)
        
        # Map issues to files
        self._map_issues_to_structure(issues)
        
        # Calculate complexity hotspots
        self._calculate_complexity_hotspots(codebase)
        
        return {
            "directory_structure": self.structure,
            "issue_hotspots": self.hotspots,
            "complexity_map": self.complexity_map,
            "total_directories": len(set(tuple(f.file_path.split("/")[:-1]) for f in codebase.files)),
            "files_by_extension": self._get_files_by_extension(codebase)
        }
    
    def _build_directory_structure(self, codebase: Codebase):
        """Build hierarchical directory structure"""
        for source_file in codebase.files:
            path_parts = source_file.file_path.split('/')
            current = self.structure
            
            for part in path_parts[:-1]:  # Directories
                if part not in current:
                    current[part] = {"type": "directory", "children": {}, "files": []}
                current = current[part]["children"]
            
            # Add file
            filename = path_parts[-1]
            current[filename] = {
                "type": "file",
                "path": source_file.file_path,
                "language": getattr(source_file, 'language', 'unknown'),
                "size": len(source_file.source.split('\n')) if source_file.source else 0,
                "issues": []
            }
    
    def _map_issues_to_structure(self, issues: List[CodeIssue]):
        """Map issues to files in structure"""
        issue_counts = {}
        
        for issue in issues:
            filepath = issue.filepath
            issue_counts[filepath] = issue_counts.get(filepath, 0) + 1
        
        # Sort by issue count to find hotspots
        self.hotspots = [
            {"filepath": filepath, "issue_count": count}
            for filepath, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        ]
    
    def _calculate_complexity_hotspots(self, codebase: Codebase):
        """Calculate complexity hotspots"""
        for source_file in codebase.files:
            complexity_score = 0
            
            # Simple complexity calculation
            if source_file.source:
                lines = len(source_file.source.split('\n'))
                functions = len([s for s in source_file.symbols if isinstance(s, Function)])
                classes = len([s for s in source_file.symbols if isinstance(s, Class)])
                
                complexity_score = lines * 0.1 + functions * 2 + classes * 3
            
            self.complexity_map[source_file.file_path] = complexity_score
    
    def _get_files_by_extension(self, codebase: Codebase) -> Dict[str, int]:
        """Get file count by extension"""
        extensions = {}
        for source_file in codebase.files:
            ext = source_file.file_path.split('.')[-1] if '.' in source_file.file_path else 'no_extension'
            extensions[ext] = extensions.get(ext, 0) + 1
        return extensions


# ============================================================================
# CORE ANALYSIS ENGINE
# ============================================================================

class AdvancedIssueDetector:
    """Advanced issue detection with automated resolution capabilities"""
    
    def __init__(self, codebase: Codebase):
        self.codebase = codebase
        self.issues: List[CodeIssue] = []
        self.automated_resolutions: List[AutomatedResolution] = []
        self.import_resolver = ImportResolver()
        
    def detect_all_issues(self) -> List[CodeIssue]:
        """Detect all types of issues with automated resolutions"""
        print("🔍 Starting comprehensive issue detection...")
        
        # Implementation errors
        self._detect_null_references()
        self._detect_type_mismatches()
        self._detect_undefined_variables()
        self._detect_missing_returns()
        self._detect_unreachable_code()
        
        # Function issues
        self._detect_function_issues()
        self._detect_parameter_issues()
        
        # Exception handling
        self._detect_exception_handling_issues()
        self._detect_resource_leaks()
        
        # Code quality
        self._detect_code_quality_issues()
        self._detect_magic_numbers()
        
        # Formatting & style
        self._detect_style_issues()
        self._detect_import_issues()
        
        # Runtime risks
        self._detect_runtime_risks()
        
        # Dead code
        self._detect_dead_code()
        
        print(f"✅ Detected {len(self.issues)} issues with {len(self.automated_resolutions)} automated resolutions")
        return self.issues
    
    def _detect_null_references(self):
        """Detect potential null reference issues"""
        for source_file in self.codebase.files:
            if source_file.source:
                lines = source_file.source.split('\n')
                for i, line in enumerate(lines):
                    # Check for .get() without null checks
                    if '.get(' in line and 'if' not in line and 'or' not in line:
                        issue = CodeIssue(
                            issue_type=IssueType.NULL_REFERENCE,
                            severity=IssueSeverity.MAJOR,
                            message="Potential null reference: .get() without null check",
                            filepath=source_file.file_path,
                            line_number=i + 1,
                            column_number=line.find('.get('),
                            context={"line": line.strip()},
                            suggested_fix="Add null check or provide default value"
                        )
                        
                        # Automated resolution
                        if '.get(' in line:
                            fixed_line = self._fix_null_reference(line)
                            issue.automated_resolution = AutomatedResolution(
                                resolution_type="null_check_addition",
                                description="Add null check with default value",
                                original_code=line.strip(),
                                fixed_code=fixed_line,
                                confidence=0.85,
                                file_path=source_file.file_path,
                                line_number=i + 1
                            )
                        
                        self.issues.append(issue)
    
    def _fix_null_reference(self, line: str) -> str:
        """Automatically fix null reference issues"""
        # Simple pattern: obj.get('key') -> obj.get('key', None)
        pattern = r'(\w+)\.get\([\'"]([^\'"]+)[\'"]\)'
        match = re.search(pattern, line)
        if match:
            obj, key = match.groups()
            return line.replace(f"{obj}.get('{key}')", f"{obj}.get('{key}', None)")
        return line
    
    def _detect_function_issues(self):
        """Detect function-related issues"""
        for source_file in self.codebase.files:
            for symbol in source_file.symbols:
                if isinstance(symbol, Function):
                    # Long function detection
                    line_count = symbol.line_end - symbol.line_start
                    if line_count > 50:
                        issue = CodeIssue(
                            issue_type=IssueType.LONG_FUNCTION,
                            severity=IssueSeverity.MAJOR,
                            message=f"Function '{symbol.name}' is too long ({line_count} lines)",
                            filepath=source_file.file_path,
                            line_number=symbol.line_start,
                            column_number=0,
                            function_name=symbol.name,
                            context={"line_count": line_count},
                            suggested_fix="Break down into smaller functions"
                        )
                        self.issues.append(issue)
                    
                    # Missing documentation
                    if not symbol.docstring:
                        issue = CodeIssue(
                            issue_type=IssueType.MISSING_DOCUMENTATION,
                            severity=IssueSeverity.MINOR,
                            message=f"Function '{symbol.name}' lacks documentation",
                            filepath=source_file.file_path,
                            line_number=symbol.line_start,
                            column_number=0,
                            function_name=symbol.name,
                            suggested_fix="Add docstring explaining function purpose"
                        )
                        
                        # Automated resolution - add basic docstring
                        docstring = self._generate_docstring(symbol)
                        issue.automated_resolution = AutomatedResolution(
                            resolution_type="add_docstring",
                            description=f"Add docstring to function '{symbol.name}'",
                            original_code="",
                            fixed_code=docstring,
                            confidence=0.80,
                            file_path=source_file.file_path,
                            line_number=symbol.line_start + 1
                        )
                        
                        self.issues.append(issue)
    
    def _generate_docstring(self, function: Function) -> str:
        """Generate basic docstring for a function"""
        params = ""
        if hasattr(function, 'parameters') and function.parameters:
            param_list = [f"        {param.name}: Description of {param.name}" 
                         for param in function.parameters if hasattr(param, 'name')]
            if param_list:
                params = f"\n    Args:\n" + "\n".join(param_list)
        
        return f'    """{function.name} function.\n    \n    Brief description of what this function does.{params}\n    \n    Returns:\n        Description of return value\n    """'
    
    def _detect_magic_numbers(self):
        """Detect magic numbers in code"""
        for source_file in self.codebase.files:
            if source_file.source:
                lines = source_file.source.split('\n')
                for i, line in enumerate(lines):
                    # Find numeric literals (excluding 0, 1, -1)
                    numbers = re.findall(r'\b(?<![.\w])\d{2,}\b(?![.\w])', line)
                    for number in numbers:
                        if int(number) not in [0, 1, -1, 100]:  # Common acceptable numbers
                            issue = CodeIssue(
                                issue_type=IssueType.MAGIC_NUMBER,
                                severity=IssueSeverity.MINOR,
                                message=f"Magic number detected: {number}",
                                filepath=source_file.file_path,
                                line_number=i + 1,
                                column_number=line.find(number),
                                context={"number": number, "line": line.strip()},
                                suggested_fix=f"Replace {number} with named constant"
                            )
                            self.issues.append(issue)
    
    def _detect_runtime_risks(self):
        """Detect potential runtime risks"""
        for source_file in self.codebase.files:
            if source_file.source:
                lines = source_file.source.split('\n')
                for i, line in enumerate(lines):
                    # Division by zero risk
                    if '/' in line and 'if' not in line:
                        # Simple heuristic for potential division by zero
                        if re.search(r'/\s*\w+(?!\w)', line):
                            issue = CodeIssue(
                                issue_type=IssueType.DIVISION_BY_ZERO,
                                severity=IssueSeverity.MAJOR,
                                message="Potential division by zero",
                                filepath=source_file.file_path,
                                line_number=i + 1,
                                column_number=line.find('/'),
                                context={"line": line.strip()},
                                suggested_fix="Add zero check before division"
                            )
                            self.issues.append(issue)
    
    def _detect_dead_code(self):
        """Detect dead code with automated removal suggestions"""
        # Find unused functions
        for source_file in self.codebase.files:
            for symbol in source_file.symbols:
                if isinstance(symbol, Function):
                    if hasattr(symbol, 'usages') and len(symbol.usages) == 0:
                        # Check if it's not an entry point
                        if not self._is_entry_point(symbol):
                            issue = CodeIssue(
                                issue_type=IssueType.DEAD_FUNCTION,
                                severity=IssueSeverity.MINOR,
                                message=f"Unused function: {symbol.name}",
                                filepath=source_file.file_path,
                                line_number=symbol.line_start,
                                column_number=0,
                                function_name=symbol.name,
                                context={"reason": "No usages found"},
                                suggested_fix="Remove unused function or verify if function is needed"
                            )
                            
                            # Automated resolution - mark for removal
                            issue.automated_resolution = AutomatedResolution(
                                resolution_type="remove_dead_function",
                                description=f"Remove unused function '{symbol.name}'",
                                original_code="",  # Will be filled with function source
                                fixed_code="",  # Empty means remove
                                confidence=0.75,
                                file_path=source_file.file_path,
                                line_number=symbol.line_start
                            )
                            
                            self.issues.append(issue)
    
    def _is_entry_point(self, function: Function) -> bool:
        """Check if function is an entry point"""
        entry_patterns = ['main', '__main__', 'run', 'start', 'execute', 'init', 'setup']
        return any(pattern in function.name.lower() for pattern in entry_patterns)
    
    def _detect_type_mismatches(self):
        """Detect type mismatch issues"""
        for source_file in self.codebase.files:
            if source_file.source:
                lines = source_file.source.split('\n')
                for i, line in enumerate(lines):
                    # Check for common type mismatches
                    if 'str(' in line and 'int(' in line:
                        issue = CodeIssue(
                            issue_type=IssueType.TYPE_MISMATCH,
                            severity=IssueSeverity.MAJOR,
                            message="Potential type mismatch: mixing str() and int() conversions",
                            filepath=source_file.file_path,
                            line_number=i + 1,
                            column_number=0,
                            context={"line": line.strip()},
                            suggested_fix="Ensure consistent type handling"
                        )
                        self.issues.append(issue)
    
    def _detect_undefined_variables(self):
        """Detect undefined variable usage"""
        for source_file in self.codebase.files:
            if source_file.source:
                lines = source_file.source.split("\n")
                for i, line in enumerate(lines):
                    # Simple check for common undefined variable patterns
                    if "NameError" in line or "undefined" in line.lower():
                        issue = CodeIssue(
                            issue_type=IssueType.UNDEFINED_VARIABLE,
                            severity=IssueSeverity.CRITICAL,
                            message="Potential undefined variable usage",
                            filepath=source_file.file_path,
                            line_number=i + 1,
                            column_number=0,
                            context={"line": line.strip()},
                            suggested_fix="Define variable before use or check spelling"
                        )
                        self.issues.append(issue)
        pass
    
    def _detect_missing_returns(self):
        """Detect functions missing return statements"""
        for source_file in self.codebase.files:
            for symbol in source_file.symbols:
                if isinstance(symbol, Function):
                    if hasattr(symbol, "source") and symbol.source:
                        # Check if function has return statements
                        if "return" not in symbol.source and "def " in symbol.source:
                            # Skip if function is a generator or has yield
                            if "yield" not in symbol.source:
                                issue = CodeIssue(
                                    issue_type=IssueType.MISSING_RETURN,
                                    severity=IssueSeverity.MINOR,
                                    message=f"Function {symbol.name} may be missing return statement",
                                    filepath=source_file.file_path,
                                    line_number=symbol.line_start,
                                    column_number=0,
                                    function_name=symbol.name,
                                    suggested_fix="Add explicit return statement"
                                )
                                self.issues.append(issue)
        pass
    
    def _detect_unreachable_code(self):
        """Detect unreachable code"""
        for source_file in self.codebase.files:
            if source_file.source:
                lines = source_file.source.split("\n")
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    # Check for code after return statements
                    if "return" in stripped and i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line and not next_line.startswith("#") and not next_line.startswith("}"):
                            # Check if it's not another function or class definition
                            if not any(keyword in next_line for keyword in ["def ", "class ", "if ", "else:", "elif ", "except:", "finally:"]):
                                issue = CodeIssue(
                                    issue_type=IssueType.UNREACHABLE_CODE,
                                    severity=IssueSeverity.MAJOR,
                                    message="Unreachable code after return statement",
                                    filepath=source_file.file_path,
                                    line_number=i + 2,
                                    column_number=0,
                                    context={"line": next_line},
                                    suggested_fix="Remove unreachable code or restructure logic"
                                )
                                self.issues.append(issue)
    
    def _detect_parameter_issues(self):
        """Detect parameter-related issues"""
        for source_file in self.codebase.files:
            for symbol in source_file.symbols:
                if isinstance(symbol, Function):
                    # Check for too many parameters
                    if hasattr(symbol, "parameters") and symbol.parameters:
                        param_count = len(symbol.parameters)
                        if param_count > 7:  # Generally considered too many
                            issue = CodeIssue(
                                issue_type=IssueType.LONG_PARAMETER_LIST,
                                severity=IssueSeverity.MAJOR,
                                message=f"Function '{symbol.name}' has too many parameters ({param_count})",
                                filepath=source_file.file_path,
                                line_number=symbol.line_start,
                                column_number=0,
                                function_name=symbol.name,
                                context={"parameter_count": param_count},
                                suggested_fix="Consider using a parameter object or breaking down the function"
                            )
                            self.issues.append(issue)
    
    def _detect_exception_handling_issues(self):
        """Detect exception handling issues"""
        for source_file in self.codebase.files:
            if source_file.source:
                lines = source_file.source.split("\n")
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    # Check for bare except clauses
                    if stripped == "except:" or stripped.startswith("except:"):
                        issue = CodeIssue(
                            issue_type=IssueType.BARE_EXCEPT,
                            severity=IssueSeverity.MAJOR,
                            message="Bare except clause catches all exceptions",
                            filepath=source_file.file_path,
                            line_number=i + 1,
                            column_number=0,
                            context={"line": stripped},
                            suggested_fix="Specify exception types or use 'except Exception:'"
                        )
                        self.issues.append(issue)
    
    def _detect_resource_leaks(self):
        """Detect resource leak issues"""
        for source_file in self.codebase.files:
            if source_file.source:
                lines = source_file.source.split("\n")
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    # Check for file operations without context managers
                    if "open(" in stripped and "with" not in stripped:
                        issue = CodeIssue(
                            issue_type=IssueType.RESOURCE_LEAK,
                            severity=IssueSeverity.MAJOR,
                            message="File opened without context manager (potential resource leak)",
                            filepath=source_file.file_path,
                            line_number=i + 1,
                            column_number=line.find("open("),
                            context={"line": stripped},
                            suggested_fix="Use 'with open(...) as f:' to ensure proper file closure"
                        )
                        self.issues.append(issue)
    
    def _detect_code_quality_issues(self):
        """Detect code quality issues"""
        for source_file in self.codebase.files:
            if source_file.source:
                lines = source_file.source.split("\n")
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    # Check for TODO/FIXME comments
                    if any(keyword in stripped.upper() for keyword in ["TODO", "FIXME", "HACK", "XXX"]):
                        issue = CodeIssue(
                            issue_type=IssueType.TODO_COMMENT,
                            severity=IssueSeverity.INFO,
                            message="TODO/FIXME comment found",
                            filepath=source_file.file_path,
                            line_number=i + 1,
                            column_number=0,
                            context={"line": stripped},
                            suggested_fix="Address the TODO/FIXME or create a proper issue"
                        )
                        self.issues.append(issue)
    
    def _detect_style_issues(self):
        """Detect style issues"""
        for source_file in self.codebase.files:
            if source_file.source:
                lines = source_file.source.split("\n")
                for i, line in enumerate(lines):
                    # Check for trailing whitespace
                    if line.endswith(" ") or line.endswith("\t"):
                        issue = CodeIssue(
                            issue_type=IssueType.TRAILING_WHITESPACE,
                            severity=IssueSeverity.INFO,
                            message="Trailing whitespace found",
                            filepath=source_file.file_path,
                            line_number=i + 1,
                            column_number=len(line.rstrip()),
                            context={"line": repr(line)},
                            suggested_fix="Remove trailing whitespace"
                        )
                        self.issues.append(issue)
    
    def _detect_import_issues(self):
        """Detect import-related issues"""
        self.import_resolver.analyze_imports(self.codebase)
        
        # Add unused import issues
        for unused in self.import_resolver.unused_imports:
            issue = CodeIssue(
                issue_type=IssueType.DEAD_IMPORT,
                severity=IssueSeverity.MINOR,
                message=f"Unused import: {unused['name']}",
                filepath=unused['file'],
                line_number=unused['line'],
                column_number=0,
                suggested_fix="Remove unused import"
            )
            
            # Automated resolution
            issue.automated_resolution = AutomatedResolution(
                resolution_type="remove_unused_import",
                description=f"Remove unused import '{unused['name']}'",
                original_code=f"import {unused['name']}",
                fixed_code="",
                confidence=0.90,
                file_path=unused['file'],
                line_number=unused['line']
            )
            
            self.issues.append(issue)


class CodebaseAnalyzer:
    """Main analysis engine for comprehensive codebase analysis"""
    
    def __init__(self):
        self.config = AnalysisConfig()
        self.issues: List[CodeIssue] = []
        self.function_contexts: Dict[str, FunctionContext] = {}
        self.call_graph: Dict[str, List[str]] = {}
        self.reverse_call_graph: Dict[str, List[str]] = {}
        self.import_resolver = ImportResolver()
        self.dead_code_analyzer = DeadCodeAnalyzer()
        self.repository_analyzer = RepositoryStructureAnalyzer()
        self.codebase_path = None  # Store for Graph-sitter integration
    
    def analyze(self, codebase_path: str, config: AnalysisConfig = None) -> AnalysisResults:
        """Main analyze method that takes a codebase path and returns analysis results"""
        if config:
            self.config = config
        
        self.codebase_path = codebase_path
        
        # Try to create Graph-sitter codebase
        gs_codebase = self._get_graph_sitter_codebase(codebase_path)
        
        if gs_codebase:
            # Use Graph-sitter based analysis
            return self.analyze_codebase(gs_codebase)
        else:
            # Fallback to basic analysis without Graph-sitter
            print("Falling back to basic analysis without Graph-sitter")
            # Create a minimal results object
            return AnalysisResults(
                issues=[],
                total_files=0,
                total_functions=0,
                total_classes=0,
                total_lines_of_code=0,
                most_important_functions=[],
                entry_points=[],
                complexity_metrics={},
                automated_resolutions=[]
            )

    def _get_graph_sitter_codebase(self, codebase_path: str):
        """Get Graph-sitter codebase instance with error handling"""
        try:
            # Check if graph_sitter module is available
            import graph_sitter
            from graph_sitter.core.codebase import Codebase
            return Codebase(codebase_path)
        except ImportError:
            print("Warning: Graph-sitter not available, using fallback analysis")
            return None
        except Exception as e:
            print(f"Warning: Could not initialize Graph-sitter codebase: {e}")
            return None
        self.repository_analyzer = RepositoryStructureAnalyzer()
        self.blast_radius_cache = {}
        self.advanced_issue_detector = None
        
    def analyze_codebase(self, codebase: Codebase) -> AnalysisResults:
        """Perform comprehensive codebase analysis"""
        
        # Initialize analysis results
        results = AnalysisResults(
            total_files=len(codebase.files),
            total_functions=0,
            total_classes=0,
            total_lines_of_code=0
        )
        
        # Phase 1: Basic metrics collection
        self._collect_basic_metrics(codebase, results)
        
        # Phase 2: Function analysis and context building
        self._analyze_functions(codebase, results)
        
        # Phase 3: Advanced issue detection with automated resolutions
        self.advanced_issue_detector = AdvancedIssueDetector(codebase)
        advanced_issues = self.advanced_issue_detector.detect_all_issues()
        
        # Phase 4: Traditional issue detection (for compatibility)
        self._detect_issues(codebase, results)
        
        # Merge advanced issues with traditional issues
        results.issues.extend(advanced_issues)
        
        # Phase 5: Call graph analysis
        self._analyze_call_graph(codebase, results)
        
        # Phase 6: Quality metrics calculation
        self._calculate_quality_metrics(codebase, results)
        
        # Phase 7: Health assessment
        self._assess_health(results)
        
        # Phase 8: Advanced import analysis
        self._analyze_imports(codebase, results)
        
        # Phase 9: Dead code analysis with blast radius
        self._analyze_dead_code_advanced(codebase, results)
        
        # Phase 10: Repository structure analysis
        self._analyze_repository_structure(codebase, results)
        
        # Phase 11: Generate comprehensive automated resolutions
        self._generate_automated_resolutions_advanced(results)
        
        # Phase 12: Collect automated resolutions from advanced detector
        if self.advanced_issue_detector:
            results.automated_resolutions.extend(self.advanced_issue_detector.automated_resolutions)
        
        return results
    
    def _collect_basic_metrics(self, codebase: Codebase, results: AnalysisResults):
        """Collect basic codebase metrics"""
        total_lines = 0
        total_functions = 0
        total_classes = 0
        
        for source_file in codebase.files:
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
        
        for source_file in codebase.files:
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
        
        # Enhanced analysis: coupling, cohesion, and importance
        self._calculate_coupling_cohesion_metrics()
        self._detect_function_importance(results)
        self._build_call_chains()
        
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
        
        for source_file in codebase.files:
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
    
    def _analyze_imports(self, codebase: Codebase, results: AnalysisResults):
        """Perform advanced import analysis"""
        print("🔍 Analyzing imports and dependencies...")
        
        self.import_resolver.analyze_imports(codebase)
        
        # Add import-related issues
        for unused in self.import_resolver.unused_imports:
            self._add_issue(
                IssueType.DEAD_IMPORT,
                IssueSeverity.MINOR,
                f"Unused import: {unused['name']}",
                unused['file'],
                unused['line'],
                0
            )
        
        for missing in self.import_resolver.missing_imports:
            self._add_issue(
                IssueType.UNDEFINED_VARIABLE,
                IssueSeverity.MAJOR,
                f"Missing import for symbol: {missing['symbol']}",
                missing['file'],
                1,
                0
            )
    
    def _analyze_dead_code_advanced(self, codebase: Codebase, results: AnalysisResults):
        """Perform advanced dead code analysis with blast radius"""
        print("💀 Analyzing dead code with blast radius...")
        
        dead_code_analysis = self.dead_code_analyzer.analyze_dead_code(codebase, self.call_graph)
        
        # Update results with dead code analysis
        results.dead_functions = dead_code_analysis["dead_functions"]
        
        # Add dead code issues
        for dead_func in dead_code_analysis["dead_functions"]:
            file_path, func_name = dead_func.split("::", 1)
            blast_info = dead_code_analysis["blast_radius"].get(dead_func, {})
            
            self._add_issue(
                IssueType.DEAD_FUNCTION,
                IssueSeverity.MINOR if blast_info.get('safe', False) else IssueSeverity.INFO,
                f"Dead function: {func_name} ({blast_info.get('lines', 0)} lines)",
                file_path,
                0,
                0,
                function_name=func_name
            )
        
        # Store blast radius information
        self.blast_radius_cache = dead_code_analysis["blast_radius"]
    
    def _analyze_repository_structure(self, codebase: Codebase, results: AnalysisResults):
        """Perform repository structure analysis"""
        print("🌳 Analyzing repository structure...")
        
        structure_analysis = self.repository_analyzer.analyze_structure(codebase, self.issues)
        
        # Store structure information for API response
        results.repository_structure = structure_analysis
    
    def _generate_automated_resolutions_advanced(self, results: AnalysisResults):
        """Generate comprehensive automated resolutions"""
        print("🤖 Generating automated resolutions...")
        
        resolutions = []
        
        # Import-based resolutions
        import_fixes = self.import_resolver.generate_import_fixes()
        resolutions.extend(import_fixes)
        
        # Issue-based resolutions
        for issue in self.issues:
            resolution = self._create_automated_resolution_advanced(issue)
            if resolution:
                resolutions.append(resolution)
                issue.automated_resolution = resolution
        
        # Dead code removal resolutions
        for dead_func in self.dead_code_analyzer.dead_functions:
            blast_info = self.blast_radius_cache.get(dead_func, {})
            if blast_info.get('safe', False):
                file_path, func_name = dead_func.split("::", 1)
                resolutions.append(AutomatedResolution(
                    resolution_type="remove_dead_function",
                    description=f"Safely remove unused function: {func_name}",
                    original_code="",  # Would need actual function source
                    fixed_code="",
                    confidence=0.85,
                    file_path=file_path,
                    line_number=0,
                    is_safe=True,
                    requires_validation=True
                ))
        
        results.automated_resolutions = resolutions
    
    def _create_automated_resolution_advanced(self, issue: CodeIssue) -> Optional[AutomatedResolution]:
        """Create advanced automated resolution for a specific issue"""
        
        if issue.issue_type == IssueType.MAGIC_NUMBER:
            return AutomatedResolution(
                resolution_type="extract_constant",
                description="Extract magic number to a named constant",
                original_code="",
                fixed_code="",
                confidence=0.8,
                file_path=issue.filepath,
                line_number=issue.line_number,
                is_safe=True,
                requires_validation=False
            )
        
        elif issue.issue_type == IssueType.MISSING_DOCUMENTATION:
            return AutomatedResolution(
                resolution_type="add_docstring",
                description=f"Add docstring to function '{issue.function_name}'",
                original_code="",
                fixed_code=self._generate_docstring_template(issue.function_name or "function"),
                confidence=0.9,
                file_path=issue.filepath,
                line_number=issue.line_number,
                is_safe=True,
                requires_validation=False
            )
        
        elif issue.issue_type == IssueType.LINE_LENGTH_VIOLATION:
            return AutomatedResolution(
                resolution_type="break_long_line",
                description="Break long line into multiple lines",
                original_code="",
                fixed_code="",
                confidence=0.7,
                file_path=issue.filepath,
                line_number=issue.line_number,
                is_safe=True,
                requires_validation=True
            )
        
        elif issue.issue_type == IssueType.INEFFICIENT_PATTERN:
            return AutomatedResolution(
                resolution_type="refactor_complex_function",
                description=f"Refactor complex function '{issue.function_name}' to reduce complexity",
                original_code="",
                fixed_code="",
                confidence=0.6,
                file_path=issue.filepath,
                line_number=issue.line_number,
                is_safe=False,
                requires_validation=True
            )
        
        return None
    
    def _generate_docstring_template(self, function_name: str) -> str:
        """Generate a basic docstring template"""
        return f'    """{function_name} function.\n    \n    Brief description of what this function does.\n    \n    Returns:\n        Description of return value\n    """'
    
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
    
    total_files = len(codebase.files)
    total_functions = sum(
        len([s for s in sf.symbols if isinstance(s, Function)])
        for sf in codebase.files
    )
    total_classes = sum(
        len([s for s in sf.symbols if isinstance(s, Class)])
        for sf in codebase.files
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


# ============================================================================
# ENHANCED ANALYSIS METHODS (Consolidated from enhanced_analysis.py)
# ============================================================================

def _calculate_coupling_cohesion_metrics(self):
    """Calculate coupling and cohesion metrics for functions"""
    
    for func_name, context in self.function_contexts.items():
        # Coupling score: based on number of external dependencies
        external_calls = len([call for call in context.function_calls 
                            if call not in self.function_contexts])
        internal_calls = len([call for call in context.function_calls 
                            if call in self.function_contexts])
        
        total_calls = external_calls + internal_calls
        context.coupling_score = external_calls / max(total_calls, 1)
        
        # Cohesion score: simplified metric based on function focus
        lines_of_code = context.line_end - context.line_start
        complexity = context.complexity_metrics.get('cyclomatic_complexity', 1)
        
        # Simple heuristic: cohesion inversely related to complexity per line
        context.cohesion_score = max(0, 1 - (complexity / max(lines_of_code, 1)))


def _detect_function_importance(self, results):
    """Detect and rank function importance"""
    
    importance_scores = {}
    
    for func_name, context in self.function_contexts.items():
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
            "filepath": self.function_contexts[func_name].filepath,
            "fan_in": self.function_contexts[func_name].fan_in,
            "fan_out": self.function_contexts[func_name].fan_out,
            "is_entry_point": self.function_contexts[func_name].is_entry_point,
            "coupling_score": self.function_contexts[func_name].coupling_score,
            "cohesion_score": self.function_contexts[func_name].cohesion_score
        }
        for func_name, score in sorted_functions[:20]  # Top 20
    ]
    
    # Detect entry points
    results.entry_points = [
        func_name for func_name, context in self.function_contexts.items()
        if context.is_entry_point
    ]


def _build_call_chains(self):
    """Build call chains for all functions"""
    
    for func_name, context in self.function_contexts.items():
        context.max_call_chain = self._build_call_chain(func_name, set())
        context.call_depth = len(context.max_call_chain) - 1


def _build_call_chain(self, function_name, visited):
    """Build the maximum call chain from a function"""
    if function_name in visited or function_name not in self.function_contexts:
        return [function_name]
    
    visited.add(function_name)
    context = self.function_contexts[function_name]
    
    max_chain = [function_name]
    for called_func in context.function_calls:
        if called_func not in visited:
            chain = self._build_call_chain(called_func, visited.copy())
            if len(chain) > len(max_chain) - 1:
                max_chain = [function_name] + chain
    
    return max_chain


def generate_codebase_summary(codebase: Codebase) -> str:
    """Generate a brief summary of the codebase"""
    
    file_count = len(codebase.files)
    
    # Count functions and classes
    function_count = 0
    class_count = 0
    
    for source_file in codebase.files:
        for symbol in source_file.symbols:
            if hasattr(symbol, '__class__'):
                if 'Function' in str(symbol.__class__):
                    function_count += 1
                elif 'Class' in str(symbol.__class__):
                    class_count += 1
    
    # Detect primary languages
    languages = set()
    for source_file in codebase.files:
        if hasattr(source_file, 'language') and source_file.language:
            languages.add(source_file.language)
    
    language_str = ", ".join(sorted(languages)) if languages else "Multiple languages"
    
    return f"Codebase with {file_count} files, {function_count} functions, {class_count} classes. Primary languages: {language_str}."


# Bind methods to CodebaseAnalyzer class
CodebaseAnalyzer._calculate_coupling_cohesion_metrics = _calculate_coupling_cohesion_metrics
CodebaseAnalyzer._detect_function_importance = _detect_function_importance
CodebaseAnalyzer._build_call_chains = _build_call_chains
CodebaseAnalyzer._build_call_chain = _build_call_chain
