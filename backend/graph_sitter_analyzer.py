"""
Enhanced Graph-sitter Analysis Engine
Leverages graph-sitter's pre-computed relationships for comprehensive codebase analysis
"""

import re
import json
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict
from datetime import datetime

try:
    from graph_sitter.codebase.codebase_analysis import Codebase
    GRAPH_SITTER_AVAILABLE = True
except ImportError:
    GRAPH_SITTER_AVAILABLE = False
    print("Warning: graph-sitter not available, using fallback analysis")

from .models import CodeIssue, IssueType, IssueSeverity, AutomatedResolution


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
                    # Local repository - use from_files for local analysis
                    import os
                    from pathlib import Path
                    
                    # Get all code files in the directory, grouped by language
                    code_files_by_lang = {
                        'typescript': [],
                        'python': [],
                        'javascript': []
                    }
                    
                    for root, dirs, files in os.walk(repo_path_or_url):
                        # Skip hidden directories and common build/cache directories
                        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'build', 'dist']]
                        
                        for file in files:
                            file_path = os.path.join(root, file)
                            if file.endswith(('.ts', '.tsx')):
                                code_files_by_lang['typescript'].append(file_path)
                            elif file.endswith('.py'):
                                code_files_by_lang['python'].append(file_path)
                            elif file.endswith(('.js', '.jsx')):
                                code_files_by_lang['javascript'].append(file_path)
                    
                    # Use the language with the most files
                    primary_lang = max(code_files_by_lang.keys(), key=lambda k: len(code_files_by_lang[k]))
                    primary_files = code_files_by_lang[primary_lang]
                    
                    if not primary_files:
                        raise ValueError("No supported code files found")
                    
                    # Convert file paths to file content dictionary
                    files_dict = {}
                    for file_path in primary_files:
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                # Use relative path as key
                                rel_path = os.path.relpath(file_path, repo_path_or_url)
                                files_dict[rel_path] = f.read()
                        except Exception as e:
                            print(f"Warning: Could not read {file_path}: {e}")
                            continue
                    
                    print(f"📝 Using {primary_lang} files: {len(files_dict)} files")
                    self.codebase = Codebase.from_files(files_dict)
            else:
                raise ImportError("Graph-sitter not available")
            
            print(f"✅ Codebase loaded: {len(self.codebase.files)} files")
            
            # Perform comprehensive analysis
            repository_facts = self._analyze_repository_facts()
            entry_points = self._detect_entry_points()
            actual_errors = self._detect_runtime_errors()
            
            # Update entry points count in repository facts
            repository_facts["entry_points_detected"] = len(entry_points)
            
            results = {
                "repository_facts": repository_facts,
                "most_important_files": self._find_most_important_files(),
                "entry_points": entry_points,
                "repository_structure": self._build_tree_structure(),
                "actual_errors": actual_errors,
                "error_summary": self._generate_error_summary_from_errors(actual_errors),
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
        
        # Get all files from the actual filesystem (not just graph-sitter files)
        import os
        all_files_on_disk = []
        code_files_on_disk = []
        doc_files_on_disk = []
        config_files_on_disk = []
        
        for root, dirs, files in os.walk('.'):
            # Skip hidden directories and common build/cache directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'build', 'dist']]
            
            for file in files:
                file_path = os.path.join(root, file)
                all_files_on_disk.append(file_path)
                
                if self._is_code_file_by_extension(file):
                    code_files_on_disk.append(file_path)
                elif self._is_doc_file_by_extension(file):
                    doc_files_on_disk.append(file_path)
                elif self._is_config_file_by_extension(file):
                    config_files_on_disk.append(file_path)
        
        # Language analysis from all code files
        languages = defaultdict(int)
        for file_path in code_files_on_disk:
            lang = self._detect_language_by_extension(file_path)
            languages[lang] += 1
        
        # Function and class counts using graph-sitter's direct access
        total_functions = len(self.codebase.functions) if hasattr(self.codebase, 'functions') else 0
        total_classes = len(self.codebase.classes) if hasattr(self.codebase, 'classes') else 0
        
        return {
            "total_files": len(all_files_on_disk),
            "code_files": len(code_files_on_disk),
            "documentation_files": len(doc_files_on_disk),
            "config_files": len(config_files_on_disk),
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
        
        # Use codebase.functions directly for better access
        if hasattr(self.codebase, 'functions'):
            for func in self.codebase.functions:
                try:
                    if self._is_entry_point_function(func):
                        # Calculate metrics using graph-sitter data
                        usage_count = len(func.usages) if hasattr(func, 'usages') else 0
                        calls_count = len(func.function_calls) if hasattr(func, 'function_calls') else 0
                        
                        # Calculate importance score
                        importance_score = self._calculate_function_importance(func)
                        
                        entry_points.append({
                            "function_name": func.name,
                            "filepath": func.filepath,
                            "line_number": getattr(func, 'start_point', [0])[0],
                            "importance_score": importance_score,
                            "usage_count": usage_count,
                            "calls_count": calls_count,
                            "entry_type": self._determine_entry_type(func),
                            "call_chain_depth": self._calculate_call_chain_depth(func)
                        })
                except Exception as e:
                    print(f"Warning: Could not analyze function {getattr(func, 'name', 'unknown')}: {e}")
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
        return self._generate_error_summary_from_errors(errors)
    
    def _generate_error_summary_from_errors(self, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary from a list of errors"""
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
    
    def _is_code_file_by_extension(self, filename: str) -> bool:
        """Check if file is a code file by extension"""
        code_extensions = ['.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.cpp', '.c', '.h', '.hpp', '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt']
        return any(filename.endswith(ext) for ext in code_extensions)
    
    def _is_doc_file_by_extension(self, filename: str) -> bool:
        """Check if file is a documentation file by extension"""
        doc_extensions = ['.md', '.txt', '.rst', '.doc', '.docx', '.pdf']
        return any(filename.endswith(ext) for ext in doc_extensions)
    
    def _is_config_file_by_extension(self, filename: str) -> bool:
        """Check if file is a config file by extension"""
        config_extensions = ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.xml']
        config_names = ['package.json', 'tsconfig.json', 'webpack.config.js', 'babel.config.js', '.eslintrc', '.gitignore', 'Dockerfile', 'docker-compose.yml']
        return any(filename.endswith(ext) for ext in config_extensions) or any(filename.endswith(name) for name in config_names)
    
    def _detect_language_by_extension(self, file_path: str) -> str:
        """Detect programming language by file extension"""
        if file_path.endswith(('.ts', '.tsx')):
            return 'typescript'
        elif file_path.endswith('.py'):
            return 'python'
        elif file_path.endswith(('.js', '.jsx')):
            return 'javascript'
        elif file_path.endswith('.java'):
            return 'java'
        elif file_path.endswith(('.cpp', '.c', '.h', '.hpp')):
            return 'c++'
        elif file_path.endswith('.cs'):
            return 'c#'
        elif file_path.endswith('.php'):
            return 'php'
        elif file_path.endswith('.rb'):
            return 'ruby'
        elif file_path.endswith('.go'):
            return 'go'
        elif file_path.endswith('.rs'):
            return 'rust'
        elif file_path.endswith('.swift'):
            return 'swift'
        elif file_path.endswith('.kt'):
            return 'kotlin'
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
