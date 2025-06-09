#!/usr/bin/env python3
"""
Graph-Sitter Integration Module for Advanced Code Analysis
Provides tree-sitter based parsing and analysis capabilities.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import re

try:
    import tree_sitter
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logging.warning("Tree-sitter not available. Falling back to regex-based analysis.")

logger = logging.getLogger(__name__)

class GraphSitterAnalyzer:
    """Advanced code analysis using tree-sitter when available"""
    
    def __init__(self):
        self.parsers = {}
        self.languages = {}
        self.supported_languages = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'tsx',
            '.jsx': 'jsx',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust'
        }
        
        if TREE_SITTER_AVAILABLE:
            self._initialize_parsers()
    
    def _initialize_parsers(self):
        """Initialize tree-sitter parsers for supported languages"""
        try:
            # This would normally load compiled language libraries
            # For now, we'll use fallback regex-based analysis
            logger.info("Tree-sitter parsers initialized (fallback mode)")
        except Exception as e:
            logger.error(f"Failed to initialize tree-sitter parsers: {e}")
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single file using tree-sitter or fallback methods"""
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return {}
        
        ext = file_path.suffix.lower()
        language = self.supported_languages.get(ext)
        
        if not language:
            return self._basic_analysis(content, file_path)
        
        if TREE_SITTER_AVAILABLE and language in self.parsers:
            return self._tree_sitter_analysis(content, file_path, language)
        else:
            return self._regex_based_analysis(content, file_path, language)
    
    def _tree_sitter_analysis(self, content: str, file_path: Path, language: str) -> Dict[str, Any]:
        """Perform tree-sitter based analysis (placeholder for actual implementation)"""
        # This would use actual tree-sitter parsing
        # For now, fall back to regex-based analysis
        return self._regex_based_analysis(content, file_path, language)
    
    def _regex_based_analysis(self, content: str, file_path: Path, language: str) -> Dict[str, Any]:
        """Perform regex-based analysis as fallback"""
        
        analysis = {
            'file_path': str(file_path),
            'language': language,
            'lines_of_code': len(content.split('\n')),
            'functions': [],
            'classes': [],
            'imports': [],
            'exports': [],
            'variables': [],
            'complexity_score': 0,
            'issues': []
        }
        
        if language == 'python':
            analysis.update(self._analyze_python(content))
        elif language in ['javascript', 'typescript', 'jsx', 'tsx']:
            analysis.update(self._analyze_javascript_typescript(content))
        elif language == 'java':
            analysis.update(self._analyze_java(content))
        elif language in ['cpp', 'c']:
            analysis.update(self._analyze_c_cpp(content))
        elif language == 'go':
            analysis.update(self._analyze_go(content))
        elif language == 'rust':
            analysis.update(self._analyze_rust(content))
        
        return analysis
    
    def _analyze_python(self, content: str) -> Dict[str, Any]:
        """Analyze Python code"""
        
        functions = []
        classes = []
        imports = []
        
        # Extract functions
        func_pattern = r'^\s*def\s+(\w+)\s*\([^)]*\):'
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            functions.append({
                'name': match.group(1),
                'line': content[:match.start()].count('\n') + 1,
                'type': 'function'
            })
        
        # Extract classes
        class_pattern = r'^\s*class\s+(\w+)(?:\([^)]*\))?:'
        for match in re.finditer(class_pattern, content, re.MULTILINE):
            classes.append({
                'name': match.group(1),
                'line': content[:match.start()].count('\n') + 1,
                'type': 'class'
            })
        
        # Extract imports
        import_patterns = [
            r'^\s*import\s+([^\s#]+)',
            r'^\s*from\s+([^\s#]+)\s+import'
        ]
        
        for pattern in import_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                imports.append({
                    'module': match.group(1),
                    'line': content[:match.start()].count('\n') + 1,
                    'type': 'import'
                })
        
        # Calculate complexity (simplified)
        complexity_keywords = ['if', 'elif', 'for', 'while', 'try', 'except', 'with']
        complexity_score = sum(len(re.findall(rf'\\b{keyword}\\b', content)) for keyword in complexity_keywords)
        
        return {
            'functions': functions,
            'classes': classes,
            'imports': imports,
            'complexity_score': complexity_score
        }
    
    def _analyze_javascript_typescript(self, content: str) -> Dict[str, Any]:
        """Analyze JavaScript/TypeScript code"""
        
        functions = []
        classes = []
        imports = []
        exports = []
        
        # Extract functions
        func_patterns = [
            r'function\s+(\w+)\s*\(',
            r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>',
            r'let\s+(\w+)\s*=\s*\([^)]*\)\s*=>',
            r'var\s+(\w+)\s*=\s*\([^)]*\)\s*=>',
            r'(\w+)\s*:\s*\([^)]*\)\s*=>'
        ]
        
        for pattern in func_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                functions.append({
                    'name': match.group(1),
                    'line': content[:match.start()].count('\n') + 1,
                    'type': 'function'
                })
        
        # Extract classes
        class_pattern = r'class\s+(\w+)(?:\s+extends\s+\w+)?'
        for match in re.finditer(class_pattern, content, re.MULTILINE):
            classes.append({
                'name': match.group(1),
                'line': content[:match.start()].count('\n') + 1,
                'type': 'class'
            })
        
        # Extract imports
        import_patterns = [
            r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'import\s+[\'"]([^\'"]+)[\'"]',
            r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)'
        ]
        
        for pattern in import_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                imports.append({
                    'module': match.group(1),
                    'line': content[:match.start()].count('\n') + 1,
                    'type': 'import'
                })
        
        # Extract exports
        export_patterns = [
            r'export\s+(?:default\s+)?(?:function\s+)?(\w+)',
            r'export\s*\{\s*([^}]+)\s*\}',
            r'module\.exports\s*=\s*(\w+)'
        ]
        
        for pattern in export_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                exports.append({
                    'name': match.group(1),
                    'line': content[:match.start()].count('\n') + 1,
                    'type': 'export'
                })
        
        # Calculate complexity
        complexity_keywords = ['if', 'else', 'for', 'while', 'switch', 'case', 'try', 'catch']
        complexity_score = sum(len(re.findall(rf'\\b{keyword}\\b', content)) for keyword in complexity_keywords)
        
        return {
            'functions': functions,
            'classes': classes,
            'imports': imports,
            'exports': exports,
            'complexity_score': complexity_score
        }
    
    def _analyze_java(self, content: str) -> Dict[str, Any]:
        """Analyze Java code"""
        
        functions = []
        classes = []
        imports = []
        
        # Extract methods
        method_pattern = r'(?:public|private|protected)?\s*(?:static)?\s*(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*\{'
        for match in re.finditer(method_pattern, content, re.MULTILINE):
            functions.append({
                'name': match.group(1),
                'line': content[:match.start()].count('\n') + 1,
                'type': 'method'
            })
        
        # Extract classes
        class_pattern = r'(?:public\s+)?class\s+(\w+)'
        for match in re.finditer(class_pattern, content, re.MULTILINE):
            classes.append({
                'name': match.group(1),
                'line': content[:match.start()].count('\n') + 1,
                'type': 'class'
            })
        
        # Extract imports
        import_pattern = r'import\s+([^;]+);'
        for match in re.finditer(import_pattern, content, re.MULTILINE):
            imports.append({
                'module': match.group(1),
                'line': content[:match.start()].count('\n') + 1,
                'type': 'import'
            })
        
        complexity_keywords = ['if', 'else', 'for', 'while', 'switch', 'case', 'try', 'catch']
        complexity_score = sum(len(re.findall(rf'\\b{keyword}\\b', content)) for keyword in complexity_keywords)
        
        return {
            'functions': functions,
            'classes': classes,
            'imports': imports,
            'complexity_score': complexity_score
        }
    
    def _analyze_c_cpp(self, content: str) -> Dict[str, Any]:
        """Analyze C/C++ code"""
        
        functions = []
        classes = []
        imports = []
        
        # Extract functions
        func_pattern = r'(?:^\s*(?:static\s+)?(?:inline\s+)?(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*\{)'
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            functions.append({
                'name': match.group(1),
                'line': content[:match.start()].count('\n') + 1,
                'type': 'function'
            })
        
        # Extract classes (C++)
        class_pattern = r'class\s+(\w+)'
        for match in re.finditer(class_pattern, content, re.MULTILINE):
            classes.append({
                'name': match.group(1),
                'line': content[:match.start()].count('\n') + 1,
                'type': 'class'
            })
        
        # Extract includes
        include_pattern = r'#include\s*[<"]([^>"]+)[>"]'
        for match in re.finditer(include_pattern, content, re.MULTILINE):
            imports.append({
                'module': match.group(1),
                'line': content[:match.start()].count('\n') + 1,
                'type': 'include'
            })
        
        complexity_keywords = ['if', 'else', 'for', 'while', 'switch', 'case']
        complexity_score = sum(len(re.findall(rf'\\b{keyword}\\b', content)) for keyword in complexity_keywords)
        
        return {
            'functions': functions,
            'classes': classes,
            'imports': imports,
            'complexity_score': complexity_score
        }
    
    def _analyze_go(self, content: str) -> Dict[str, Any]:
        """Analyze Go code"""
        
        functions = []
        imports = []
        
        # Extract functions
        func_pattern = r'func\s+(?:\([^)]*\)\s+)?(\w+)\s*\('
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            functions.append({
                'name': match.group(1),
                'line': content[:match.start()].count('\n') + 1,
                'type': 'function'
            })
        
        # Extract imports
        import_pattern = r'import\s+(?:\(\s*([^)]+)\s*\)|"([^"]+)")'
        for match in re.finditer(import_pattern, content, re.MULTILINE):
            module = match.group(1) or match.group(2)
            imports.append({
                'module': module,
                'line': content[:match.start()].count('\n') + 1,
                'type': 'import'
            })
        
        complexity_keywords = ['if', 'else', 'for', 'switch', 'case', 'select']
        complexity_score = sum(len(re.findall(rf'\\b{keyword}\\b', content)) for keyword in complexity_keywords)
        
        return {
            'functions': functions,
            'classes': [],  # Go doesn't have classes
            'imports': imports,
            'complexity_score': complexity_score
        }
    
    def _analyze_rust(self, content: str) -> Dict[str, Any]:
        """Analyze Rust code"""
        
        functions = []
        imports = []
        
        # Extract functions
        func_pattern = r'fn\s+(\w+)\s*\('
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            functions.append({
                'name': match.group(1),
                'line': content[:match.start()].count('\n') + 1,
                'type': 'function'
            })
        
        # Extract use statements
        use_pattern = r'use\s+([^;]+);'
        for match in re.finditer(use_pattern, content, re.MULTILINE):
            imports.append({
                'module': match.group(1),
                'line': content[:match.start()].count('\n') + 1,
                'type': 'use'
            })
        
        complexity_keywords = ['if', 'else', 'for', 'while', 'loop', 'match']
        complexity_score = sum(len(re.findall(rf'\\b{keyword}\\b', content)) for keyword in complexity_keywords)
        
        return {
            'functions': functions,
            'classes': [],  # Rust uses structs/traits instead
            'imports': imports,
            'complexity_score': complexity_score
        }
    
    def _basic_analysis(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Basic analysis for unsupported file types"""
        
        return {
            'file_path': str(file_path),
            'language': 'unknown',
            'lines_of_code': len(content.split('\n')),
            'functions': [],
            'classes': [],
            'imports': [],
            'exports': [],
            'variables': [],
            'complexity_score': 0,
            'issues': []
        }
    
    def get_inheritance_hierarchy(self, analyses: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Build inheritance hierarchy from analysis results"""
        
        hierarchy = {}
        
        for analysis in analyses:
            for cls in analysis.get('classes', []):
                class_name = cls['name']
                # This is a simplified version - real implementation would parse inheritance
                hierarchy[class_name] = []
        
        return hierarchy
    
    def get_function_call_graph(self, analyses: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Build function call graph from analysis results"""
        
        call_graph = {}
        
        for analysis in analyses:
            for func in analysis.get('functions', []):
                func_name = func['name']
                # This is a simplified version - real implementation would parse function calls
                call_graph[func_name] = []
        
        return call_graph

