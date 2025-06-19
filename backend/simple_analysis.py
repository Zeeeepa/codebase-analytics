#!/usr/bin/env python3
"""
Simplified Analysis Module without Codegen SDK dependencies.
"""

import os
import ast
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict
import re

def analyze_codebase(repo_path: str) -> Dict[str, Any]:
    """
    Analyze a codebase and return comprehensive metrics.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        Dictionary containing analysis results
    """
    repo_path = Path(repo_path)
    
    # Initialize results
    results = {
        "repo_path": str(repo_path),
        "total_files": 0,
        "total_lines": 0,
        "total_code_lines": 0,
        "total_comment_lines": 0,
        "total_blank_lines": 0,
        "languages": {},
        "file_sizes": {},
        "complexity_metrics": {},
        "files": []
    }
    
    # File extensions to analyze
    extensions = {
        '.py': 'Python',
        '.js': 'JavaScript',
        '.ts': 'TypeScript',
        '.jsx': 'JSX',
        '.tsx': 'TSX',
        '.java': 'Java',
        '.c': 'C',
        '.cpp': 'C++',
        '.h': 'C Header',
        '.hpp': 'C++ Header',
        '.go': 'Go',
        '.rs': 'Rust',
        '.php': 'PHP',
        '.rb': 'Ruby',
        '.swift': 'Swift',
        '.kt': 'Kotlin',
        '.scala': 'Scala',
        '.cs': 'C#',
        '.vb': 'VB.NET',
        '.html': 'HTML',
        '.css': 'CSS',
        '.scss': 'SCSS',
        '.sass': 'SASS',
        '.less': 'LESS',
        '.xml': 'XML',
        '.json': 'JSON',
        '.yaml': 'YAML',
        '.yml': 'YAML',
        '.md': 'Markdown',
        '.txt': 'Text',
        '.sh': 'Shell',
        '.bash': 'Bash',
        '.zsh': 'Zsh',
        '.fish': 'Fish',
        '.ps1': 'PowerShell',
        '.bat': 'Batch',
        '.cmd': 'Command',
        '.dockerfile': 'Dockerfile',
        '.sql': 'SQL',
        '.r': 'R',
        '.R': 'R',
        '.m': 'MATLAB',
        '.pl': 'Perl',
        '.lua': 'Lua',
        '.vim': 'Vim',
        '.el': 'Emacs Lisp',
        '.clj': 'Clojure',
        '.hs': 'Haskell',
        '.ml': 'OCaml',
        '.fs': 'F#',
        '.ex': 'Elixir',
        '.exs': 'Elixir',
        '.erl': 'Erlang',
        '.dart': 'Dart',
        '.jl': 'Julia',
        '.nim': 'Nim',
        '.cr': 'Crystal',
        '.zig': 'Zig'
    }
    
    # Walk through all files
    for file_path in repo_path.rglob('*'):
        if file_path.is_file():
            # Skip hidden files and directories
            if any(part.startswith('.') for part in file_path.parts):
                continue
                
            # Skip common non-source directories
            skip_dirs = {'node_modules', '__pycache__', '.git', 'venv', 'env', 'build', 'dist', 'target'}
            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue
            
            file_ext = file_path.suffix.lower()
            if file_ext in extensions or file_path.name.lower() in ['dockerfile', 'makefile', 'rakefile']:
                results["total_files"] += 1
                
                # Determine language
                if file_path.name.lower() in ['dockerfile', 'makefile', 'rakefile']:
                    language = file_path.name.capitalize()
                else:
                    language = extensions.get(file_ext, 'Unknown')
                
                # Analyze file
                file_analysis = analyze_file(file_path, language)
                results["files"].append(file_analysis)
                
                # Update totals
                results["total_lines"] += file_analysis["lines"]
                results["total_code_lines"] += file_analysis["code_lines"]
                results["total_comment_lines"] += file_analysis["comment_lines"]
                results["total_blank_lines"] += file_analysis["blank_lines"]
                
                # Update language stats
                if language not in results["languages"]:
                    results["languages"][language] = {
                        "files": 0,
                        "lines": 0,
                        "code_lines": 0,
                        "comment_lines": 0,
                        "blank_lines": 0
                    }
                
                results["languages"][language]["files"] += 1
                results["languages"][language]["lines"] += file_analysis["lines"]
                results["languages"][language]["code_lines"] += file_analysis["code_lines"]
                results["languages"][language]["comment_lines"] += file_analysis["comment_lines"]
                results["languages"][language]["blank_lines"] += file_analysis["blank_lines"]
                
                # File size analysis
                file_size = file_path.stat().st_size
                size_category = categorize_file_size(file_size)
                if size_category not in results["file_sizes"]:
                    results["file_sizes"][size_category] = 0
                results["file_sizes"][size_category] += 1
    
    # Calculate complexity metrics
    results["complexity_metrics"] = calculate_complexity_metrics(results)
    
    return results

def analyze_file(file_path: Path, language: str) -> Dict[str, Any]:
    """
    Analyze a single file.
    
    Args:
        file_path: Path to the file
        language: Programming language
        
    Returns:
        Dictionary containing file analysis
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        return {
            "path": str(file_path),
            "language": language,
            "lines": 0,
            "code_lines": 0,
            "comment_lines": 0,
            "blank_lines": 0,
            "size_bytes": 0,
            "error": str(e)
        }
    
    lines = content.split('\n')
    total_lines = len(lines)
    code_lines = 0
    comment_lines = 0
    blank_lines = 0
    
    # Language-specific comment patterns
    comment_patterns = {
        'Python': [r'^\s*#', r'^\s*"""', r'^\s*\'\'\''],
        'JavaScript': [r'^\s*//', r'^\s*/\*', r'^\s*\*'],
        'TypeScript': [r'^\s*//', r'^\s*/\*', r'^\s*\*'],
        'Java': [r'^\s*//', r'^\s*/\*', r'^\s*\*'],
        'C': [r'^\s*//', r'^\s*/\*', r'^\s*\*'],
        'C++': [r'^\s*//', r'^\s*/\*', r'^\s*\*'],
        'Go': [r'^\s*//', r'^\s*/\*', r'^\s*\*'],
        'Rust': [r'^\s*//', r'^\s*/\*', r'^\s*\*'],
        'Shell': [r'^\s*#'],
        'Bash': [r'^\s*#'],
        'HTML': [r'^\s*<!--'],
        'CSS': [r'^\s*/\*', r'^\s*\*'],
        'SQL': [r'^\s*--', r'^\s*/\*', r'^\s*\*']
    }
    
    patterns = comment_patterns.get(language, [r'^\s*#', r'^\s*//', r'^\s*/\*', r'^\s*\*'])
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank_lines += 1
        elif any(re.match(pattern, line) for pattern in patterns):
            comment_lines += 1
        else:
            code_lines += 1
    
    # Additional analysis for Python files
    functions = []
    classes = []
    if language == 'Python' and file_path.suffix == '.py':
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        "name": node.name,
                        "line": node.lineno,
                        "args": len(node.args.args),
                        "complexity": calculate_cyclomatic_complexity(node)
                    })
                elif isinstance(node, ast.ClassDef):
                    classes.append({
                        "name": node.name,
                        "line": node.lineno,
                        "methods": len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                    })
        except:
            pass  # Skip AST analysis if parsing fails
    
    return {
        "path": str(file_path.relative_to(file_path.parents[len(file_path.parents)-1])),
        "language": language,
        "lines": total_lines,
        "code_lines": code_lines,
        "comment_lines": comment_lines,
        "blank_lines": blank_lines,
        "size_bytes": file_path.stat().st_size,
        "functions": functions,
        "classes": classes
    }

def calculate_cyclomatic_complexity(node: ast.AST) -> int:
    """
    Calculate cyclomatic complexity for an AST node.
    
    Args:
        node: AST node
        
    Returns:
        Cyclomatic complexity
    """
    complexity = 1  # Base complexity
    
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
            complexity += 1
        elif isinstance(child, ast.ExceptHandler):
            complexity += 1
        elif isinstance(child, (ast.And, ast.Or)):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
    
    return complexity

def categorize_file_size(size_bytes: int) -> str:
    """
    Categorize file size.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Size category
    """
    if size_bytes < 1024:  # < 1KB
        return "tiny"
    elif size_bytes < 10240:  # < 10KB
        return "small"
    elif size_bytes < 102400:  # < 100KB
        return "medium"
    elif size_bytes < 1048576:  # < 1MB
        return "large"
    else:
        return "huge"

def calculate_complexity_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate overall complexity metrics.
    
    Args:
        results: Analysis results
        
    Returns:
        Complexity metrics
    """
    total_files = results["total_files"]
    total_lines = results["total_lines"]
    
    if total_files == 0:
        return {}
    
    # Calculate averages
    avg_lines_per_file = total_lines / total_files
    avg_code_lines_per_file = results["total_code_lines"] / total_files
    
    # Calculate ratios
    comment_ratio = results["total_comment_lines"] / total_lines if total_lines > 0 else 0
    blank_ratio = results["total_blank_lines"] / total_lines if total_lines > 0 else 0
    code_ratio = results["total_code_lines"] / total_lines if total_lines > 0 else 0
    
    # Language diversity
    language_count = len(results["languages"])
    
    # File size distribution
    size_distribution = results["file_sizes"]
    
    return {
        "avg_lines_per_file": round(avg_lines_per_file, 2),
        "avg_code_lines_per_file": round(avg_code_lines_per_file, 2),
        "comment_ratio": round(comment_ratio, 3),
        "blank_ratio": round(blank_ratio, 3),
        "code_ratio": round(code_ratio, 3),
        "language_count": language_count,
        "size_distribution": size_distribution
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
    else:
        repo_path = "."
    
    results = analyze_codebase(repo_path)
    print(json.dumps(results, indent=2, default=str))
