"""
Repository Structure Builder
Interactive repository structure with issue indicators
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class DirectoryNode:
    """Represents a directory in the repository structure"""
    name: str
    path: str
    children: List['DirectoryNode'] = field(default_factory=list)
    files: List[Dict[str, Any]] = field(default_factory=list)
    issue_counts: Dict[str, int] = field(default_factory=dict)
    total_issues: int = 0
    health_score: float = 100.0
    lines_of_code: int = 0
    functions_count: int = 0
    classes_count: int = 0


class RepositoryStructureBuilder:
    """Builder for interactive repository structure"""
    
    def __init__(self, codebase):
        self.codebase = codebase
        
    def build_structure(self, issues: List[Any]) -> Dict[str, Any]:
        """Build interactive repository structure with issue indicators"""
        print("🏗️ Building repository structure...")
        
        # Group issues by file
        issues_by_file = defaultdict(list)
        for issue in issues:
            issues_by_file[issue.filepath].append(issue)
        
        # Build directory tree
        root = DirectoryNode(name="repository", path="")
        
        # Process each file
        for file in self.codebase.files:
            file_issues = issues_by_file[file.filepath]
            file_info = self._analyze_file(file, file_issues)
            self._add_file_to_tree(root, file.filepath, file_info)
        
        # Calculate directory metrics recursively
        self._calculate_directory_metrics(root)
        
        # Convert to serializable format
        structure_dict = self._serialize_directory_tree(root)
        
        print("✅ Repository structure built")
        return structure_dict
    
    def _analyze_file(self, file, issues: List[Any]) -> Dict[str, Any]:
        """Analyze a single file"""
        # Count lines of code
        loc = len(file.source.split('\n')) if hasattr(file, 'source') else 0
        
        # Count functions and classes
        functions_count = len([f for f in self.codebase.functions if f.filepath == file.filepath])
        classes_count = len([c for c in self.codebase.classes if c.filepath == file.filepath])
        
        # Group issues by severity
        issue_counts = defaultdict(int)
        for issue in issues:
            issue_counts[issue.severity.value] += 1
        
        # Calculate health score
        health_score = 100.0
        severity_penalties = {'critical': 15, 'major': 8, 'minor': 3, 'info': 1}
        for severity, count in issue_counts.items():
            health_score -= severity_penalties.get(severity, 1) * count
        health_score = max(0, health_score)
        
        return {
            'filepath': file.filepath,
            'filename': file.filepath.split('/')[-1],
            'lines_of_code': loc,
            'functions_count': functions_count,
            'classes_count': classes_count,
            'issues': [self._serialize_issue(issue) for issue in issues],
            'issue_counts': dict(issue_counts),
            'total_issues': len(issues),
            'health_score': health_score,
            'file_type': self._get_file_type(file.filepath)
        }
    
    def _add_file_to_tree(self, root: DirectoryNode, filepath: str, file_info: Dict[str, Any]):
        """Add a file to the directory tree"""
        path_parts = filepath.split('/')
        current_node = root
        
        # Navigate/create directory structure
        for part in path_parts[:-1]:
            # Find or create child directory
            child_node = None
            for child in current_node.children:
                if child.name == part:
                    child_node = child
                    break
            
            if child_node is None:
                child_path = f"{current_node.path}/{part}" if current_node.path else part
                child_node = DirectoryNode(name=part, path=child_path)
                current_node.children.append(child_node)
            
            current_node = child_node
        
        # Add file to the final directory
        current_node.files.append(file_info)
    
    def _calculate_directory_metrics(self, node: DirectoryNode):
        """Calculate metrics for directories recursively"""
        # Initialize counters
        node.issue_counts = defaultdict(int)
        node.total_issues = 0
        node.lines_of_code = 0
        node.functions_count = 0
        node.classes_count = 0
        
        # Aggregate from files
        for file_info in node.files:
            node.lines_of_code += file_info['lines_of_code']
            node.functions_count += file_info['functions_count']
            node.classes_count += file_info['classes_count']
            node.total_issues += file_info['total_issues']
            
            for severity, count in file_info['issue_counts'].items():
                node.issue_counts[severity] += count
        
        # Aggregate from subdirectories
        for child in node.children:
            self._calculate_directory_metrics(child)
            
            node.lines_of_code += child.lines_of_code
            node.functions_count += child.functions_count
            node.classes_count += child.classes_count
            node.total_issues += child.total_issues
            
            for severity, count in child.issue_counts.items():
                node.issue_counts[severity] += count
        
        # Calculate health score
        node.health_score = self._calculate_directory_health_score(node)
    
    def _calculate_directory_health_score(self, node: DirectoryNode) -> float:
        """Calculate health score for a directory"""
        if node.total_issues == 0:
            return 100.0
        
        health_score = 100.0
        severity_penalties = {'critical': 15, 'major': 8, 'minor': 3, 'info': 1}
        
        for severity, count in node.issue_counts.items():
            health_score -= severity_penalties.get(severity, 1) * count
        
        # Normalize by directory size
        if node.lines_of_code > 0:
            health_score = health_score * (1000 / max(node.lines_of_code, 1000))
        
        return max(0, min(100, health_score))
    
    def _serialize_directory_tree(self, node: DirectoryNode) -> Dict[str, Any]:
        """Convert directory tree to serializable format"""
        return {
            'name': node.name,
            'path': node.path,
            'type': 'directory',
            'children': [self._serialize_directory_tree(child) for child in node.children],
            'files': node.files,
            'metrics': {
                'total_issues': node.total_issues,
                'issue_counts': dict(node.issue_counts),
                'health_score': round(node.health_score, 1),
                'lines_of_code': node.lines_of_code,
                'functions_count': node.functions_count,
                'classes_count': node.classes_count
            },
            'indicators': self._get_directory_indicators(node)
        }
    
    def _get_directory_indicators(self, node: DirectoryNode) -> List[str]:
        """Get visual indicators for directory"""
        indicators = []
        
        if node.issue_counts.get('critical', 0) > 0:
            indicators.append('⚠️ Critical Issues')
        elif node.issue_counts.get('major', 0) > 0:
            indicators.append('👉 Major Issues')
        elif node.issue_counts.get('minor', 0) > 0:
            indicators.append('🔍 Minor Issues')
        
        if node.health_score < 50:
            indicators.append('🏥 Poor Health')
        elif node.health_score < 70:
            indicators.append('⚡ Needs Attention')
        
        if node.lines_of_code > 5000:
            indicators.append('📏 Large Directory')
        
        return indicators
    
    def _get_file_type(self, filepath: str) -> str:
        """Get file type from filepath"""
        if filepath.endswith('.py'):
            return 'python'
        elif filepath.endswith(('.ts', '.tsx')):
            return 'typescript'
        elif filepath.endswith(('.js', '.jsx')):
            return 'javascript'
        elif filepath.endswith('.md'):
            return 'markdown'
        elif filepath.endswith(('.yml', '.yaml')):
            return 'yaml'
        elif filepath.endswith('.json'):
            return 'json'
        else:
            return 'other'
    
    def _serialize_issue(self, issue) -> Dict[str, Any]:
        """Serialize issue for JSON output"""
        return {
            'type': issue.issue_type.value,
            'severity': issue.severity.value,
            'message': issue.message,
            'line_number': issue.line_number,
            'column_number': issue.column_number,
            'function_name': issue.function_name,
            'class_name': issue.class_name
        }

