#!/usr/bin/env python3
"""
Enhanced Interactive UI for Codebase Analytics

This module creates the interactive repository tree structure with:
- Clickable folder/file navigation
- Issue count tracking and display
- Symbol-level exploration
- Context statistical information panels
- Real-time issue severity classification
"""

import json
import os
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pathlib import Path

from codegen.sdk.core.codebase import Codebase
from codegen.sdk.core.class_definition import Class
from codegen.sdk.core.file import SourceFile
from codegen.sdk.core.function import Function
from codegen.sdk.core.symbol import Symbol

from analysis import (
    detect_comprehensive_issues, 
    calculate_cyclomatic_complexity,
    get_function_context,
    analyze_dependencies,
    get_advanced_codebase_statistics
)

@dataclass
class IssueCount:
    """Issue count tracking for UI display."""
    critical: int = 0
    major: int = 0
    minor: int = 0
    info: int = 0
    
    @property
    def total(self) -> int:
        return self.critical + self.major + self.minor + self.info
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "critical": self.critical,
            "major": self.major,
            "minor": self.minor,
            "info": self.info,
            "total": self.total
        }

@dataclass
class SymbolNode:
    """Represents a symbol (function/class) in the interactive tree."""
    name: str
    type: str  # "function", "class", "variable"
    line_number: int
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    complexity: int = 0
    call_count: int = 0
    callers: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    issues: IssueCount = field(default_factory=IssueCount)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "line_number": self.line_number,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "complexity": self.complexity,
            "call_count": self.call_count,
            "callers": self.callers,
            "dependencies": self.dependencies,
            "issues": self.issues.to_dict(),
            "context": self.context
        }

@dataclass
class FileNode:
    """Represents a file in the interactive tree."""
    name: str
    path: str
    type: str = "file"
    size: int = 0
    lines_of_code: int = 0
    symbols: List[SymbolNode] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    issues: IssueCount = field(default_factory=IssueCount)
    statistics: Dict[str, Any] = field(default_factory=dict)
    children: List['FileNode'] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "type": self.type,
            "size": self.size,
            "lines_of_code": self.lines_of_code,
            "symbols": [symbol.to_dict() for symbol in self.symbols],
            "imports": self.imports,
            "exports": self.exports,
            "issues": self.issues.to_dict(),
            "statistics": self.statistics,
            "children": [child.to_dict() for child in self.children]
        }

class EnhancedInteractiveTreeBuilder:
    """Builds the enhanced interactive repository tree structure."""
    
    def __init__(self, codebase: Codebase):
        self.codebase = codebase
        self.issue_map = {}
        self.symbol_context_map = {}
        self.dependency_graph = {}
        
    def build_interactive_tree(self) -> Dict[str, Any]:
        """Build the complete interactive tree structure."""
        # First, analyze the codebase for issues and context
        self._analyze_codebase_context()
        
        # Build the tree structure
        root_node = self._build_directory_structure()
        
        # Calculate summary statistics
        summary = self._calculate_summary_statistics(root_node)
        
        return {
            "repository": {
                "name": getattr(self.codebase, 'name', 'Unknown'),
                "type": "repository",
                "tree": root_node.to_dict(),
                "summary": summary,
                "ui_config": self._get_ui_config()
            }
        }
    
    def _analyze_codebase_context(self):
        """Analyze the codebase to gather context information."""
        try:
            # Detect comprehensive issues
            issues_result = detect_comprehensive_issues(self.codebase)
            self._process_issues(issues_result)
            
            # Analyze dependencies
            self.dependency_graph = analyze_dependencies(self.codebase)
            
            # Get symbol contexts
            for func in self.codebase.functions:
                try:
                    context = get_function_context(func)
                    self.symbol_context_map[func.name] = context
                except Exception as e:
                    print(f"Warning: Could not get context for function {func.name}: {e}")
                    
        except Exception as e:
            print(f"Warning: Could not analyze codebase context: {e}")
    
    def _process_issues(self, issues_result: Dict[str, Any]):
        """Process detected issues and map them to files/symbols."""
        detailed_issues = issues_result.get('detailed_issues', [])
        
        for issue in detailed_issues:
            file_path = issue.get('file_path', '')
            symbol_name = issue.get('symbol', '')
            severity = issue.get('severity', 'info').lower()
            
            # Map issues to files
            if file_path not in self.issue_map:
                self.issue_map[file_path] = {'file': IssueCount(), 'symbols': {}}
            
            # Increment file-level issue count
            if severity == 'critical':
                self.issue_map[file_path]['file'].critical += 1
            elif severity == 'major' or severity == 'high':
                self.issue_map[file_path]['file'].major += 1
            elif severity == 'minor' or severity == 'medium':
                self.issue_map[file_path]['file'].minor += 1
            else:
                self.issue_map[file_path]['file'].info += 1
            
            # Map issues to symbols
            if symbol_name:
                if symbol_name not in self.issue_map[file_path]['symbols']:
                    self.issue_map[file_path]['symbols'][symbol_name] = IssueCount()
                
                symbol_issues = self.issue_map[file_path]['symbols'][symbol_name]
                if severity == 'critical':
                    symbol_issues.critical += 1
                elif severity == 'major' or severity == 'high':
                    symbol_issues.major += 1
                elif severity == 'minor' or severity == 'medium':
                    symbol_issues.minor += 1
                else:
                    symbol_issues.info += 1
    
    def _build_directory_structure(self) -> FileNode:
        """Build the directory structure with files and symbols."""
        # Create root node
        root = FileNode(
            name=getattr(self.codebase, 'name', 'Repository'),
            path="/",
            type="directory"
        )
        
        # Group files by directory
        directory_map = {}
        
        for file in self.codebase.files:
            file_path = getattr(file, 'file_path', getattr(file, 'path', ''))
            if not file_path:
                continue
                
            # Create file node
            file_node = self._create_file_node(file)
            
            # Get directory path
            dir_path = str(Path(file_path).parent)
            if dir_path == '.':
                dir_path = '/'
            
            # Create directory structure
            if dir_path not in directory_map:
                directory_map[dir_path] = FileNode(
                    name=Path(dir_path).name or 'root',
                    path=dir_path,
                    type="directory"
                )
            
            directory_map[dir_path].children.append(file_node)
        
        # Build nested directory structure
        root.children = list(directory_map.values())
        
        # Calculate directory-level issue counts
        self._calculate_directory_issues(root)
        
        return root
    
    def _create_file_node(self, file) -> FileNode:
        """Create a file node with symbols and statistics."""
        file_path = getattr(file, 'file_path', getattr(file, 'path', ''))
        file_name = Path(file_path).name
        
        # Get file content and statistics
        content = getattr(file, 'content', '')
        lines_of_code = len(content.split('\n')) if content else 0
        file_size = len(content) if content else 0
        
        # Create file node
        file_node = FileNode(
            name=file_name,
            path=file_path,
            type="file",
            size=file_size,
            lines_of_code=lines_of_code
        )
        
        # Add file-level issues
        if file_path in self.issue_map:
            file_node.issues = self.issue_map[file_path]['file']
        
        # Add symbols (functions and classes)
        try:
            # Add functions
            for func in getattr(file, 'functions', []):
                symbol_node = self._create_symbol_node(func, 'function')
                file_node.symbols.append(symbol_node)
            
            # Add classes
            for cls in getattr(file, 'classes', []):
                symbol_node = self._create_symbol_node(cls, 'class')
                file_node.symbols.append(symbol_node)
                
        except Exception as e:
            print(f"Warning: Could not process symbols for file {file_path}: {e}")
        
        # Calculate file statistics
        file_node.statistics = self._calculate_file_statistics(file_node)
        
        return file_node
    
    def _create_symbol_node(self, symbol, symbol_type: str) -> SymbolNode:
        """Create a symbol node with detailed context information."""
        symbol_name = getattr(symbol, 'name', 'Unknown')
        
        # Create symbol node
        symbol_node = SymbolNode(
            name=symbol_name,
            type=symbol_type,
            line_number=getattr(symbol, 'line_number', 0)
        )
        
        # Add function-specific information
        if symbol_type == 'function':
            try:
                # Get parameters
                params = getattr(symbol, 'parameters', [])
                symbol_node.parameters = [str(p) for p in params]
                
                # Get return type
                symbol_node.return_type = getattr(symbol, 'return_type', None)
                
                # Calculate complexity
                try:
                    symbol_node.complexity = calculate_cyclomatic_complexity(symbol)
                except Exception:
                    symbol_node.complexity = 1
                
                # Get call information
                call_sites = getattr(symbol, 'call_sites', [])
                symbol_node.call_count = len(call_sites)
                symbol_node.callers = [str(c) for c in call_sites[:10]]  # Limit to 10
                
                # Get dependencies
                dependencies = getattr(symbol, 'dependencies', [])
                symbol_node.dependencies = [str(d) for d in dependencies[:10]]  # Limit to 10
                
            except Exception as e:
                print(f"Warning: Could not process function details for {symbol_name}: {e}")
        
        # Add class-specific information
        elif symbol_type == 'class':
            try:
                # Get methods
                methods = getattr(symbol, 'methods', [])
                symbol_node.context['methods'] = [m.name for m in methods[:10]]
                
                # Get inheritance
                bases = getattr(symbol, 'bases', [])
                symbol_node.context['inheritance'] = [str(b) for b in bases]
                
            except Exception as e:
                print(f"Warning: Could not process class details for {symbol_name}: {e}")
        
        # Add symbol-level issues
        file_path = getattr(symbol, 'file_path', '')
        if file_path in self.issue_map and symbol_name in self.issue_map[file_path]['symbols']:
            symbol_node.issues = self.issue_map[file_path]['symbols'][symbol_name]
        
        # Add context information
        if symbol_name in self.symbol_context_map:
            symbol_node.context.update(self.symbol_context_map[symbol_name])
        
        return symbol_node
    
    def _calculate_directory_issues(self, directory: FileNode):
        """Calculate issue counts for directories by aggregating child issues."""
        if directory.type != "directory":
            return
        
        total_issues = IssueCount()
        
        for child in directory.children:
            if child.type == "directory":
                self._calculate_directory_issues(child)
                child_issues = child.issues
            else:
                child_issues = child.issues
            
            total_issues.critical += child_issues.critical
            total_issues.major += child_issues.major
            total_issues.minor += child_issues.minor
            total_issues.info += child_issues.info
        
        directory.issues = total_issues
    
    def _calculate_file_statistics(self, file_node: FileNode) -> Dict[str, Any]:
        """Calculate detailed statistics for a file."""
        stats = {
            "functions_count": len([s for s in file_node.symbols if s.type == 'function']),
            "classes_count": len([s for s in file_node.symbols if s.type == 'class']),
            "average_complexity": 0,
            "max_complexity": 0,
            "total_parameters": 0,
            "maintainability_score": 0
        }
        
        # Calculate complexity statistics
        function_complexities = [s.complexity for s in file_node.symbols if s.type == 'function']
        if function_complexities:
            stats["average_complexity"] = sum(function_complexities) / len(function_complexities)
            stats["max_complexity"] = max(function_complexities)
        
        # Calculate parameter statistics
        stats["total_parameters"] = sum(len(s.parameters) for s in file_node.symbols if s.type == 'function')
        
        # Calculate maintainability score (simplified)
        if file_node.lines_of_code > 0:
            issue_density = file_node.issues.total / file_node.lines_of_code
            complexity_factor = stats["average_complexity"] / 10.0 if stats["average_complexity"] > 0 else 0
            stats["maintainability_score"] = max(0, 100 - (issue_density * 50) - (complexity_factor * 20))
        else:
            stats["maintainability_score"] = 100
        
        return stats
    
    def _calculate_summary_statistics(self, root_node: FileNode) -> Dict[str, Any]:
        """Calculate summary statistics for the entire repository."""
        def count_nodes(node: FileNode, counts: Dict[str, int]):
            if node.type == "file":
                counts["files"] += 1
                counts["lines_of_code"] += node.lines_of_code
                counts["functions"] += len([s for s in node.symbols if s.type == 'function'])
                counts["classes"] += len([s for s in node.symbols if s.type == 'class'])
            elif node.type == "directory":
                counts["directories"] += 1
            
            for child in node.children:
                count_nodes(child, counts)
        
        counts = {
            "files": 0,
            "directories": 0,
            "lines_of_code": 0,
            "functions": 0,
            "classes": 0
        }
        
        count_nodes(root_node, counts)
        
        return {
            "total_files": counts["files"],
            "total_directories": counts["directories"],
            "total_lines_of_code": counts["lines_of_code"],
            "total_functions": counts["functions"],
            "total_classes": counts["classes"],
            "total_issues": root_node.issues.total,
            "issues_by_severity": root_node.issues.to_dict()
        }
    
    def _get_ui_config(self) -> Dict[str, Any]:
        """Get UI configuration for the interactive tree."""
        return {
            "theme": "dark",
            "show_issue_details": True,
            "show_symbol_context": True,
            "expandable_tree": True,
            "issue_severity_colors": {
                "critical": "#ff4444",
                "major": "#ff8800",
                "minor": "#ffaa00",
                "info": "#4488ff"
            },
            "file_type_icons": {
                ".py": "🐍",
                ".js": "📜",
                ".ts": "📘",
                ".tsx": "⚛️",
                ".jsx": "⚛️",
                ".html": "🌐",
                ".css": "🎨",
                ".json": "📋",
                ".md": "📝",
                ".yml": "⚙️",
                ".yaml": "⚙️"
            },
            "symbol_type_icons": {
                "function": "🔧",
                "class": "🏗️",
                "variable": "📦",
                "method": "⚡"
            }
        }

def build_enhanced_interactive_structure(codebase: Codebase) -> Dict[str, Any]:
    """
    Build the enhanced interactive repository structure.
    
    This is the main entry point for creating the interactive tree
    with all the features requested: clickable navigation, issue tracking,
    symbol exploration, and context information.
    """
    builder = EnhancedInteractiveTreeBuilder(codebase)
    return builder.build_interactive_tree()

