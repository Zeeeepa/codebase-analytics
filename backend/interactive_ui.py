#!/usr/bin/env python3
"""
Interactive UI Module for Codebase Analytics

This module provides the interactive repository tree structure and statistical
information display as specified in the requirements.
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

class IssueSeverity(Enum):
    """Issue severity levels with emoji representations."""
    CRITICAL = ("⚠️", "Critical")
    MAJOR = ("👉", "Major") 
    MINOR = ("🔍", "Minor")
    INFO = ("ℹ️", "Info")

@dataclass
class IssueCount:
    """Issue count by severity."""
    critical: int = 0
    major: int = 0
    minor: int = 0
    info: int = 0
    
    @property
    def total(self) -> int:
        return self.critical + self.major + self.minor + self.info
    
    def to_display_string(self) -> str:
        """Convert to display string with emojis."""
        parts = []
        if self.critical > 0:
            parts.append(f"⚠️ Critical: {self.critical}")
        if self.major > 0:
            parts.append(f"👉 Major: {self.major}")
        if self.minor > 0:
            parts.append(f"🔍 Minor: {self.minor}")
        if self.info > 0:
            parts.append(f"ℹ️ Info: {self.info}")
        return f"[{'] ['.join(parts)}]" if parts else ""

@dataclass
class FileIssue:
    """Individual file issue details."""
    file_path: str
    function_name: Optional[str]
    line_number: int
    severity: IssueSeverity
    category: str
    description: str
    suggestion: Optional[str] = None

@dataclass
class SymbolInfo:
    """Symbol information for context display."""
    name: str
    symbol_type: str  # function, class, variable
    parameters: List[str] = None
    return_type: Optional[str] = None
    line_range: tuple = None
    complexity: Optional[float] = None
    usage_count: int = 0
    dependencies: List[str] = None
    issues: List[FileIssue] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = []
        if self.dependencies is None:
            self.dependencies = []
        if self.issues is None:
            self.issues = []

@dataclass
class FileNode:
    """File node in the repository tree."""
    name: str
    path: str
    is_directory: bool
    size: int = 0
    lines_of_code: int = 0
    issue_count: IssueCount = None
    symbols: List[SymbolInfo] = None
    children: List['FileNode'] = None
    
    def __post_init__(self):
        if self.issue_count is None:
            self.issue_count = IssueCount()
        if self.symbols is None:
            self.symbols = []
        if self.children is None:
            self.children = []
    
    @property
    def display_icon(self) -> str:
        """Get display icon for the node."""
        return "📁" if self.is_directory else "📄"
    
    def get_display_name(self) -> str:
        """Get formatted display name with issue counts."""
        base_name = f"{self.display_icon} {self.name}"
        if self.issue_count.total > 0:
            issue_display = self.issue_count.to_display_string()
            return f"{base_name} {issue_display}"
        return base_name
    
    def add_child(self, child: 'FileNode'):
        """Add a child node."""
        self.children.append(child)
        # Propagate issue counts up the tree
        self.issue_count.critical += child.issue_count.critical
        self.issue_count.major += child.issue_count.major
        self.issue_count.minor += child.issue_count.minor
        self.issue_count.info += child.issue_count.info

class InteractiveRepositoryTree:
    """Interactive repository tree structure generator."""
    
    def __init__(self, codebase):
        self.codebase = codebase
        self.root_node = None
        self.file_issues = {}  # file_path -> List[FileIssue]
        self.symbol_context = {}  # symbol_id -> SymbolInfo
    
    def build_tree_structure(self) -> FileNode:
        """Build the complete interactive repository tree."""
        # Create root node
        repo_name = getattr(self.codebase, 'name', 'repository')
        self.root_node = FileNode(
            name=repo_name,
            path="",
            is_directory=True
        )
        
        # Process all files in the codebase
        for file in self.codebase.files:
            self._process_file(file)
        
        # Build directory structure
        self._build_directory_structure()
        
        return self.root_node
    
    def _process_file(self, file):
        """Process a single file and extract symbols and issues."""
        file_path = file.filepath
        
        # Create file node
        file_node = FileNode(
            name=Path(file_path).name,
            path=file_path,
            is_directory=False,
            size=getattr(file, 'size', 0),
            lines_of_code=getattr(file, 'lines_of_code', 0)
        )
        
        # Extract symbols (functions, classes, variables)
        symbols = []
        
        # Process functions
        for func in getattr(file, 'functions', []):
            symbol = self._create_function_symbol(func, file_path)
            symbols.append(symbol)
            self.symbol_context[f"{file_path}:{func.name}"] = symbol
        
        # Process classes
        for cls in getattr(file, 'classes', []):
            symbol = self._create_class_symbol(cls, file_path)
            symbols.append(symbol)
            self.symbol_context[f"{file_path}:{cls.name}"] = symbol
            
            # Process class methods
            for method in getattr(cls, 'methods', []):
                method_symbol = self._create_function_symbol(method, file_path, cls.name)
                symbols.append(method_symbol)
                self.symbol_context[f"{file_path}:{cls.name}.{method.name}"] = method_symbol
        
        file_node.symbols = symbols
        
        # Detect and categorize issues for this file
        file_issues = self._detect_file_issues(file, file_path)
        self.file_issues[file_path] = file_issues
        
        # Update issue counts
        for issue in file_issues:
            if issue.severity == IssueSeverity.CRITICAL:
                file_node.issue_count.critical += 1
            elif issue.severity == IssueSeverity.MAJOR:
                file_node.issue_count.major += 1
            elif issue.severity == IssueSeverity.MINOR:
                file_node.issue_count.minor += 1
            else:
                file_node.issue_count.info += 1
        
        return file_node
    
    def _create_function_symbol(self, func, file_path: str, class_name: str = None) -> SymbolInfo:
        """Create symbol info for a function."""
        full_name = f"{class_name}.{func.name}" if class_name else func.name
        
        # Extract parameters
        parameters = []
        if hasattr(func, 'parameters'):
            parameters = [param.name for param in func.parameters]
        
        # Calculate complexity
        complexity = self._calculate_function_complexity(func)
        
        # Get usage count
        usage_count = len(getattr(func, 'usages', []))
        
        # Get dependencies
        dependencies = []
        if hasattr(func, 'function_calls'):
            dependencies = [call.name for call in func.function_calls]
        
        # Detect function-specific issues
        issues = self._detect_symbol_issues(func, file_path, 'function')
        
        return SymbolInfo(
            name=full_name,
            symbol_type='function',
            parameters=parameters,
            return_type=getattr(func, 'return_type', None),
            line_range=(getattr(func, 'start_line', 0), getattr(func, 'end_line', 0)),
            complexity=complexity,
            usage_count=usage_count,
            dependencies=dependencies,
            issues=issues
        )
    
    def _create_class_symbol(self, cls, file_path: str) -> SymbolInfo:
        """Create symbol info for a class."""
        # Get inheritance information
        dependencies = []
        if hasattr(cls, 'base_classes'):
            dependencies = [base.name for base in cls.base_classes]
        
        # Get method count
        method_count = len(getattr(cls, 'methods', []))
        
        # Detect class-specific issues
        issues = self._detect_symbol_issues(cls, file_path, 'class')
        
        return SymbolInfo(
            name=cls.name,
            symbol_type='class',
            parameters=[f"methods: {method_count}"],
            line_range=(getattr(cls, 'start_line', 0), getattr(cls, 'end_line', 0)),
            usage_count=len(getattr(cls, 'usages', [])),
            dependencies=dependencies,
            issues=issues
        )
    
    def _calculate_function_complexity(self, func) -> float:
        """Calculate cyclomatic complexity for a function."""
        # Simplified complexity calculation
        complexity = 1  # Base complexity
        
        if hasattr(func, 'source') and func.source:
            source = func.source
            # Count decision points
            complexity += source.count('if ')
            complexity += source.count('elif ')
            complexity += source.count('for ')
            complexity += source.count('while ')
            complexity += source.count('except ')
            complexity += source.count('and ')
            complexity += source.count('or ')
        
        return float(complexity)
    
    def _detect_file_issues(self, file, file_path: str) -> List[FileIssue]:
        """Detect issues in a file."""
        issues = []
        
        # Example issue detection patterns
        if hasattr(file, 'source'):
            source = file.source
            lines = source.split('\n')
            
            for i, line in enumerate(lines, 1):
                # Detect potential issues
                if 'TODO' in line or 'FIXME' in line:
                    issues.append(FileIssue(
                        file_path=file_path,
                        function_name=None,
                        line_number=i,
                        severity=IssueSeverity.MINOR,
                        category="Documentation",
                        description="TODO/FIXME comment found",
                        suggestion="Complete the TODO item or remove the comment"
                    ))
                
                if 'print(' in line and not line.strip().startswith('#'):
                    issues.append(FileIssue(
                        file_path=file_path,
                        function_name=None,
                        line_number=i,
                        severity=IssueSeverity.MINOR,
                        category="Code Quality",
                        description="Print statement found (potential debug code)",
                        suggestion="Use logging instead of print statements"
                    ))
                
                if 'except:' in line:
                    issues.append(FileIssue(
                        file_path=file_path,
                        function_name=None,
                        line_number=i,
                        severity=IssueSeverity.MAJOR,
                        category="Error Handling",
                        description="Bare except clause",
                        suggestion="Specify exception types to catch"
                    ))
        
        return issues
    
    def _detect_symbol_issues(self, symbol, file_path: str, symbol_type: str) -> List[FileIssue]:
        """Detect issues specific to a symbol."""
        issues = []
        
        if symbol_type == 'function':
            # Check for unused parameters
            if hasattr(symbol, 'parameters') and hasattr(symbol, 'source'):
                for param in symbol.parameters:
                    if param.name not in symbol.source:
                        issues.append(FileIssue(
                            file_path=file_path,
                            function_name=symbol.name,
                            line_number=getattr(symbol, 'start_line', 0),
                            severity=IssueSeverity.MINOR,
                            category="Code Quality",
                            description=f"Unused parameter '{param.name}'",
                            suggestion=f"Remove unused parameter or use it in the function"
                        ))
            
            # Check for high complexity
            complexity = self._calculate_function_complexity(symbol)
            if complexity > 10:
                issues.append(FileIssue(
                    file_path=file_path,
                    function_name=symbol.name,
                    line_number=getattr(symbol, 'start_line', 0),
                    severity=IssueSeverity.MAJOR,
                    category="Complexity",
                    description=f"High cyclomatic complexity ({complexity})",
                    suggestion="Consider breaking down the function into smaller functions"
                ))
        
        return issues
    
    def _build_directory_structure(self):
        """Build the hierarchical directory structure."""
        # This would organize files into their directory structure
        # For now, we'll create a flat structure
        pass
    
    def get_symbol_context(self, symbol_id: str) -> Optional[SymbolInfo]:
        """Get detailed context for a symbol."""
        return self.symbol_context.get(symbol_id)
    
    def get_file_issues(self, file_path: str) -> List[FileIssue]:
        """Get all issues for a specific file."""
        return self.file_issues.get(file_path, [])
    
    def generate_tree_html(self) -> str:
        """Generate HTML representation of the interactive tree."""
        if not self.root_node:
            return "<div>No repository data available</div>"
        
        html = """
        <div class="repository-tree">
            <style>
                .repository-tree {
                    font-family: 'Courier New', monospace;
                    background: #1e1e1e;
                    color: #d4d4d4;
                    padding: 20px;
                    border-radius: 8px;
                }
                .tree-node {
                    margin: 2px 0;
                    cursor: pointer;
                    padding: 2px 4px;
                    border-radius: 3px;
                }
                .tree-node:hover {
                    background: #2d2d30;
                }
                .tree-node.directory {
                    font-weight: bold;
                }
                .tree-node.file {
                    color: #9cdcfe;
                }
                .issue-badge {
                    font-size: 0.8em;
                    margin-left: 8px;
                }
                .issue-critical { color: #f14c4c; }
                .issue-major { color: #ff8c00; }
                .issue-minor { color: #ffcc02; }
                .issue-info { color: #0078d4; }
                .tree-indent { margin-left: 20px; }
                .symbol-list {
                    margin-left: 40px;
                    font-size: 0.9em;
                    color: #a277ff;
                }
                .symbol-item {
                    margin: 1px 0;
                    cursor: pointer;
                }
                .symbol-item:hover {
                    background: #2d2d30;
                    border-radius: 3px;
                }
            </style>
        """
        
        html += self._generate_node_html(self.root_node, 0)
        html += "</div>"
        
        return html
    
    def _generate_node_html(self, node: FileNode, depth: int) -> str:
        """Generate HTML for a single node and its children."""
        indent = "  " * depth
        node_class = "directory" if node.is_directory else "file"
        
        html = f'{indent}<div class="tree-node {node_class}" onclick="toggleNode(this)">\n'
        html += f'{indent}  {node.get_display_name()}\n'
        
        # Add symbols for files
        if not node.is_directory and node.symbols:
            html += f'{indent}  <div class="symbol-list">\n'
            for symbol in node.symbols:
                symbol_display = f"🔧 {symbol.name}"
                if symbol.symbol_type == 'function':
                    params = ", ".join(symbol.parameters) if symbol.parameters else ""
                    symbol_display = f"⚡ {symbol.name}({params})"
                elif symbol.symbol_type == 'class':
                    symbol_display = f"🏗️ {symbol.name}"
                
                # Add complexity info
                if symbol.complexity and symbol.complexity > 5:
                    symbol_display += f" [complexity: {symbol.complexity:.1f}]"
                
                # Add usage info
                if symbol.usage_count > 0:
                    symbol_display += f" [used: {symbol.usage_count}x]"
                
                html += f'{indent}    <div class="symbol-item" onclick="showSymbolContext(\'{node.path}:{symbol.name}\')">\n'
                html += f'{indent}      {symbol_display}\n'
                html += f'{indent}    </div>\n'
            html += f'{indent}  </div>\n'
        
        # Add children for directories
        if node.is_directory and node.children:
            html += f'{indent}  <div class="tree-indent">\n'
            for child in node.children:
                html += self._generate_node_html(child, depth + 1)
            html += f'{indent}  </div>\n'
        
        html += f'{indent}</div>\n'
        return html
    
    def generate_statistics_panel(self) -> Dict[str, Any]:
        """Generate statistical information panel."""
        if not self.root_node:
            return {}
        
        stats = {
            "repository_overview": {
                "total_files": self._count_files(self.root_node),
                "total_directories": self._count_directories(self.root_node),
                "total_lines_of_code": self._sum_lines_of_code(self.root_node),
                "total_size_bytes": self._sum_file_sizes(self.root_node)
            },
            "issue_summary": {
                "total_issues": self.root_node.issue_count.total,
                "critical_issues": self.root_node.issue_count.critical,
                "major_issues": self.root_node.issue_count.major,
                "minor_issues": self.root_node.issue_count.minor,
                "info_issues": self.root_node.issue_count.info
            },
            "symbol_summary": {
                "total_symbols": len(self.symbol_context),
                "functions": len([s for s in self.symbol_context.values() if s.symbol_type == 'function']),
                "classes": len([s for s in self.symbol_context.values() if s.symbol_type == 'class']),
                "average_complexity": self._calculate_average_complexity()
            },
            "top_issues": self._get_top_issues(),
            "most_complex_functions": self._get_most_complex_functions(),
            "most_used_symbols": self._get_most_used_symbols()
        }
        
        return stats
    
    def _count_files(self, node: FileNode) -> int:
        """Count total files in the tree."""
        count = 0 if node.is_directory else 1
        for child in node.children:
            count += self._count_files(child)
        return count
    
    def _count_directories(self, node: FileNode) -> int:
        """Count total directories in the tree."""
        count = 1 if node.is_directory else 0
        for child in node.children:
            count += self._count_directories(child)
        return count
    
    def _sum_lines_of_code(self, node: FileNode) -> int:
        """Sum total lines of code."""
        total = node.lines_of_code
        for child in node.children:
            total += self._sum_lines_of_code(child)
        return total
    
    def _sum_file_sizes(self, node: FileNode) -> int:
        """Sum total file sizes."""
        total = node.size
        for child in node.children:
            total += self._sum_file_sizes(child)
        return total
    
    def _calculate_average_complexity(self) -> float:
        """Calculate average function complexity."""
        complexities = [s.complexity for s in self.symbol_context.values() 
                       if s.symbol_type == 'function' and s.complexity]
        return sum(complexities) / len(complexities) if complexities else 0.0
    
    def _get_top_issues(self) -> List[Dict[str, Any]]:
        """Get top issues by severity."""
        all_issues = []
        for file_path, issues in self.file_issues.items():
            all_issues.extend(issues)
        
        # Sort by severity (Critical first)
        severity_order = {IssueSeverity.CRITICAL: 0, IssueSeverity.MAJOR: 1, 
                         IssueSeverity.MINOR: 2, IssueSeverity.INFO: 3}
        all_issues.sort(key=lambda x: severity_order[x.severity])
        
        return [asdict(issue) for issue in all_issues[:10]]  # Top 10 issues
    
    def _get_most_complex_functions(self) -> List[Dict[str, Any]]:
        """Get most complex functions."""
        functions = [s for s in self.symbol_context.values() 
                    if s.symbol_type == 'function' and s.complexity]
        functions.sort(key=lambda x: x.complexity, reverse=True)
        
        return [{"name": f.name, "complexity": f.complexity, "line_range": f.line_range} 
                for f in functions[:10]]
    
    def _get_most_used_symbols(self) -> List[Dict[str, Any]]:
        """Get most used symbols."""
        symbols = list(self.symbol_context.values())
        symbols.sort(key=lambda x: x.usage_count, reverse=True)
        
        return [{"name": s.name, "type": s.symbol_type, "usage_count": s.usage_count} 
                for s in symbols[:10] if s.usage_count > 0]

def build_interactive_repository_structure(codebase) -> Dict[str, Any]:
    """
    Build interactive repository structure with issue tracking.
    
    This is the main function that creates the complete interactive UI data
    as specified in the requirements.
    """
    tree_builder = InteractiveRepositoryTree(codebase)
    root_node = tree_builder.build_tree_structure()
    
    return {
        "repository_tree": {
            "html": tree_builder.generate_tree_html(),
            "data": asdict(root_node)
        },
        "statistics": tree_builder.generate_statistics_panel(),
        "symbol_context": {k: asdict(v) for k, v in tree_builder.symbol_context.items()},
        "file_issues": {k: [asdict(issue) for issue in v] for k, v in tree_builder.file_issues.items()}
    }

