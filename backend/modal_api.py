#!/usr/bin/env python3
"""
Enhanced Modal-based Codebase Analytics API
Serverless deployment with Modal for scalable repository analysis
"""

import os
import sys
import json
import tempfile
import shutil
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from collections import defaultdict
import re
import asyncio

import modal
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, validator

# Modal image configuration with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        "fastapi==0.115.6",
        "uvicorn[standard]==0.32.1", 
        "pydantic==2.10.4",
        "httpx==0.28.1",
        "python-multipart==0.0.17",
        "gitpython==3.1.44",
        "tree-sitter==0.23.2",
        "structlog==25.4.0",
        "python-dotenv==1.0.1",
        "aiofiles==24.1.0"
    )
)

# Create Modal app
app = modal.App("codebase-analytics", image=image)

# Pydantic models (same as before)
class RepositoryRequest(BaseModel):
    repo_url: HttpUrl
    
    @validator('repo_url')
    def validate_github_url(cls, v):
        url_str = str(v)
        if not ('github.com' in url_str or 'gitlab.com' in url_str):
            raise ValueError('Only GitHub and GitLab repositories are supported')
        return v

class BasicMetrics(BaseModel):
    files: int = 0
    functions: int = 0
    classes: int = 0
    modules: int = 0

class LineMetrics(BaseModel):
    loc: int = 0
    lloc: int = 0
    sloc: int = 0
    comments: int = 0
    comment_density: float = 0.0

class ComplexityMetrics(BaseModel):
    cyclomatic_complexity: Dict[str, float] = {"average": 0.0}
    maintainability_index: Dict[str, float] = {"average": 0.0}
    halstead_metrics: Dict[str, Union[int, float]] = {"total_volume": 0, "average_volume": 0}

class IssueItem(BaseModel):
    file_path: str
    line_number: int
    severity: str
    issue_type: str
    description: str
    suggestion: Optional[str] = None

class RepositoryNode(BaseModel):
    name: str
    path: str
    type: str
    issue_count: int = 0
    critical_issues: int = 0
    functional_issues: int = 0
    minor_issues: int = 0
    children: Optional[List['RepositoryNode']] = None
    issues: Optional[List[IssueItem]] = None

class IssuesSummary(BaseModel):
    total: int = 0
    critical: int = 0
    functional: int = 0
    minor: int = 0

class RepositoryAnalysis(BaseModel):
    repo_url: str
    description: str = "Repository analysis"
    basic_metrics: BasicMetrics
    line_metrics: Dict[str, LineMetrics]
    complexity_metrics: ComplexityMetrics
    repository_structure: RepositoryNode
    issues_summary: IssuesSummary
    detailed_issues: List[IssueItem]
    monthly_commits: Dict[str, int] = {}

# Update forward references
RepositoryNode.model_rebuild()

# FastAPI app for Modal
fastapi_app = FastAPI(
    title="Enhanced Codebase Analytics API (Modal)",
    description="Serverless repository analysis with Modal",
    version="2.0.0"
)

# CORS middleware
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # More permissive for serverless
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

class CodeAnalyzer:
    """Enhanced code analyzer for Modal deployment"""
    
    def __init__(self):
        self.supported_extensions = {
            '.py': 'python',
            '.js': 'javascript', 
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php'
        }
        
        # Issue detection patterns
        self.issue_patterns = {
            'critical': [
                (r'def\s+(\w*commiter\w*)', 'Misspelled function name - should be "committer"'),
                (r'@staticmethod.*@property.*def\s+is_class_method', 'Incorrect implementation checking @staticmethod instead of @classmethod'),
                (r'assert\s+isinstance\(.*\)', 'Uses assert for runtime type checking'),
                (r'\.items\(\)\s*(?!.*isinstance)', 'Potential null reference - no type checking before calling .items()'),
                (r'return\s+"[^"]*\$\{[^}]*\}[^"]*"', 'Template literal syntax in regular string'),
            ],
            'functional': [
                (r'#\s*TODO[:\s]', 'Contains TODOs indicating incomplete implementation'),
                (r'@cached_property.*@reader\(cache=True\)', 'Potentially inefficient use of cached_property and reader decorator'),
                (r'def\s+\w+\([^)]*\).*:\s*pass', 'Empty function implementation'),
                (r'except\s*:\s*pass', 'Bare except clause'),
                (r'import_module.*#.*No validation', 'Missing validation for import_module'),
            ],
            'minor': [
                (r'def\s+\w+\([^)]*(\w+)[^)]*\).*?(?!.*\1)', 'Unused parameter'),
                (r'Args:\s*(\w+)\s*#.*typo', 'Typo in docstring'),
                (r'(\w+)\s*=\s*None\s*\n\s*if\s+\1\s*:=', 'Redundant variable initialization'),
                (r'#.*ISSUE:', 'Code marked with issue comment'),
                (r'\.removeprefix\([^)]+\).*(?!.*removeprefix)', 'Redundant code path'),
            ]
        }

    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single file for metrics and issues"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Basic metrics
            loc = len(lines)
            lloc = len([line for line in lines if line.strip()])
            comments = len([line for line in lines if line.strip().startswith('#') or line.strip().startswith('//')])
            
            # Function and class counting
            functions = len(re.findall(r'def\s+\w+\s*\(', content))
            classes = len(re.findall(r'class\s+\w+\s*[\(:]', content))
            
            # Issue detection
            issues = self.detect_issues(content, str(file_path))
            
            return {
                'loc': loc,
                'lloc': lloc,
                'comments': comments,
                'functions': functions,
                'classes': classes,
                'issues': issues,
                'complexity': self.calculate_complexity(content)
            }
            
        except Exception as e:
            print(f"Error analyzing file {file_path}: {e}")
            return {
                'loc': 0, 'lloc': 0, 'comments': 0, 
                'functions': 0, 'classes': 0, 'issues': [], 'complexity': 1.0
            }

    def detect_issues(self, content: str, file_path: str) -> List[IssueItem]:
        """Detect code issues using pattern matching"""
        issues = []
        lines = content.split('\n')
        
        for severity, patterns in self.issue_patterns.items():
            for pattern, description in patterns:
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append(IssueItem(
                            file_path=file_path,
                            line_number=line_num,
                            severity=severity,
                            issue_type=pattern.split('\\')[0][:20] + "...",
                            description=description,
                            suggestion=self.get_suggestion(severity, pattern)
                        ))
        
        return issues

    def get_suggestion(self, severity: str, pattern: str) -> str:
        """Get improvement suggestions for detected issues"""
        suggestions = {
            'critical': "This is a critical issue that may cause runtime errors. Please fix immediately.",
            'functional': "This affects functionality. Consider refactoring or completing the implementation.",
            'minor': "This is a minor issue that affects code quality. Consider cleaning up when convenient."
        }
        return suggestions.get(severity, "Consider reviewing this code section.")

    def calculate_complexity(self, content: str) -> float:
        """Calculate cyclomatic complexity"""
        complexity_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally', 'with']
        complexity = 1
        
        for keyword in complexity_keywords:
            complexity += len(re.findall(rf'\b{keyword}\b', content))
        
        return min(complexity, 50)

    def build_repository_tree(self, repo_path: Path, file_analyses: Dict[str, Dict]) -> RepositoryNode:
        """Build repository structure tree with issue counts"""
        
        def create_node(path: Path, relative_path: str = "") -> RepositoryNode:
            name = path.name if path.name else "repo"
            node_path = relative_path
            
            if path.is_file():
                analysis = file_analyses.get(str(path), {})
                issues = analysis.get('issues', [])
                
                return RepositoryNode(
                    name=name,
                    path=node_path,
                    type="file",
                    issue_count=len(issues),
                    critical_issues=len([i for i in issues if i.severity == 'critical']),
                    functional_issues=len([i for i in issues if i.severity == 'functional']),
                    minor_issues=len([i for i in issues if i.severity == 'minor']),
                    issues=issues
                )
            else:
                children = []
                total_issues = 0
                total_critical = 0
                total_functional = 0
                total_minor = 0
                
                try:
                    for child in sorted(path.iterdir()):
                        if child.name.startswith('.') and child.name not in ['.github', '.vscode']:
                            continue
                        
                        child_relative = f"{relative_path}/{child.name}" if relative_path else child.name
                        child_node = create_node(child, child_relative)
                        children.append(child_node)
                        
                        total_issues += child_node.issue_count
                        total_critical += child_node.critical_issues
                        total_functional += child_node.functional_issues
                        total_minor += child_node.minor_issues
                        
                except PermissionError:
                    pass
                
                return RepositoryNode(
                    name=name,
                    path=node_path,
                    type="directory",
                    issue_count=total_issues,
                    critical_issues=total_critical,
                    functional_issues=total_functional,
                    minor_issues=total_minor,
                    children=children
                )
        
        return create_node(repo_path)

@app.function(
    image=image,
    timeout=600,  # 10 minute timeout for large repos
    memory=2048,  # 2GB memory
    cpu=2.0       # 2 CPU cores
)
def clone_repository(repo_url: str) -> str:
    """Clone repository to temporary directory in Modal"""
    temp_dir = Path("/tmp") / f"repo_{datetime.now().timestamp()}"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Extract repo name from URL
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        clone_path = temp_dir / repo_name
        
        # Clone repository
        result = subprocess.run(
            ['git', 'clone', '--depth', '1', repo_url, str(clone_path)],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            raise Exception(f"Failed to clone repository: {result.stderr}")
        
        return str(clone_path)
        
    except subprocess.TimeoutExpired:
        raise Exception("Repository cloning timed out")
    except Exception as e:
        raise Exception(f"Error cloning repository: {str(e)}")

@app.function(
    image=image,
    timeout=600,
    memory=2048,
    cpu=2.0
)
def analyze_repository_modal(repo_url: str) -> Dict[str, Any]:
    """Main analysis function for Modal"""
    try:
        print(f"Starting analysis of repository: {repo_url}")
        
        # Clone repository
        repo_path = Path(clone_repository.remote(repo_url))
        
        # Initialize analyzer
        analyzer = CodeAnalyzer()
        
        # Analyze all files
        file_analyses = {}
        total_metrics = {
            'files': 0, 'functions': 0, 'classes': 0, 'modules': 0,
            'loc': 0, 'lloc': 0, 'comments': 0, 'complexity_sum': 0
        }
        all_issues = []
        
        for file_path in repo_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in analyzer.supported_extensions:
                analysis = analyzer.analyze_file(file_path)
                file_analyses[str(file_path)] = analysis
                
                # Aggregate metrics
                total_metrics['files'] += 1
                total_metrics['functions'] += analysis['functions']
                total_metrics['classes'] += analysis['classes']
                total_metrics['loc'] += analysis['loc']
                total_metrics['lloc'] += analysis['lloc']
                total_metrics['comments'] += analysis['comments']
                total_metrics['complexity_sum'] += analysis['complexity']
                
                all_issues.extend(analysis['issues'])
        
        # Count modules
        modules = len([
            d for d in repo_path.rglob('*') 
            if d.is_dir() and (
                (d / '__init__.py').exists() or 
                (d / 'package.json').exists()
            )
        ])
        total_metrics['modules'] = modules
        
        # Calculate averages
        avg_complexity = (
            total_metrics['complexity_sum'] / total_metrics['files'] 
            if total_metrics['files'] > 0 else 0
        )
        comment_density = (
            total_metrics['comments'] / total_metrics['loc'] 
            if total_metrics['loc'] > 0 else 0
        )
        
        # Build repository structure
        repo_structure = analyzer.build_repository_tree(repo_path, file_analyses)
        
        # Get git statistics
        monthly_commits = get_git_commit_stats(repo_path)
        
        # Summarize issues
        issues_summary = {
            'total': len(all_issues),
            'critical': len([i for i in all_issues if i.severity == 'critical']),
            'functional': len([i for i in all_issues if i.severity == 'functional']),
            'minor': len([i for i in all_issues if i.severity == 'minor'])
        }
        
        # Create response
        analysis = {
            'repo_url': repo_url,
            'description': "Repository analysis",
            'basic_metrics': {
                'files': total_metrics['files'],
                'functions': total_metrics['functions'],
                'classes': total_metrics['classes'],
                'modules': total_metrics['modules']
            },
            'line_metrics': {
                "total": {
                    'loc': total_metrics['loc'],
                    'lloc': total_metrics['lloc'],
                    'sloc': total_metrics['lloc'],
                    'comments': total_metrics['comments'],
                    'comment_density': comment_density
                }
            },
            'complexity_metrics': {
                'cyclomatic_complexity': {"average": round(avg_complexity, 1)},
                'maintainability_index': {"average": max(0, min(100, 100 - avg_complexity * 2))},
                'halstead_metrics': {
                    "total_volume": total_metrics['functions'] * 100,
                    "average_volume": 100
                }
            },
            'repository_structure': repo_structure.dict(),
            'issues_summary': issues_summary,
            'detailed_issues': [issue.dict() for issue in all_issues[:100]],
            'monthly_commits': monthly_commits
        }
        
        print(f"Analysis completed successfully for {repo_url}")
        return analysis
        
    except Exception as e:
        print(f"Error analyzing repository {repo_url}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def get_git_commit_stats(repo_path: Path) -> Dict[str, int]:
    """Get git commit statistics"""
    try:
        result = subprocess.run(
            ['git', 'log', '--since="12 months ago"', '--pretty=format:%ci'],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return {}
        
        monthly_commits = defaultdict(int)
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    date = datetime.strptime(line[:7], '%Y-%m')
                    month_key = date.strftime('%Y-%m')
                    monthly_commits[month_key] += 1
                except ValueError:
                    continue
        
        return dict(monthly_commits)
        
    except Exception as e:
        print(f"Error getting git stats: {e}")
        return {}

# FastAPI routes
@fastapi_app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "deployment": "modal",
        "timestamp": datetime.now().isoformat()
    }

@fastapi_app.post("/analyze_repo")
async def analyze_repository_endpoint(request: RepositoryRequest):
    """Analyze a repository using Modal serverless functions"""
    repo_url = str(request.repo_url)
    
    try:
        # Call the Modal function
        result = analyze_repository_modal.remote(repo_url)
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@fastapi_app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Enhanced Codebase Analytics API (Modal)",
        "version": "2.0.0",
        "deployment": "modal",
        "docs": "/docs",
        "health": "/health"
    }

# Mount the FastAPI app to Modal
@app.function(image=image)
@modal.asgi_app()
def fastapi_app_modal():
    return fastapi_app

if __name__ == "__main__":
  uvicorn.run(fastapi_app, host="0.0.0.0", port=9997)
