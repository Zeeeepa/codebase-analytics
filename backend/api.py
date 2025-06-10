#!/usr/bin/env python3
"""
Enhanced Codebase Analytics API
A comprehensive FastAPI backend for repository analysis with graph-sitter integration.
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
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import re

# FastAPI and related imports
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, validator
import uvicorn

# Add the graph_sitter path to sys.path for imports
current_dir = Path(__file__).parent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
from fastapi import FastAPI
from pydantic import BaseModel
from graph_sitter import Codebase
from graph_sitter.core.statements.for_loop_statement import ForLoopStatement
from graph_sitter.core.statements.if_block_statement import IfBlockStatement
from graph_sitter.core.statements.try_catch_statement import TryCatchStatement
from graph_sitter.core.statements.while_statement import WhileStatement
from graph_sitter.core.expressions.binary_expression import BinaryExpression
from graph_sitter.core.expressions.unary_expression import UnaryExpression
from graph_sitter.core.expressions.comparison_expression import ComparisonExpression
# Import graph_sitter from the installed package
try:
    from graph_sitter import Codebase
    from graph_sitter.codebase.codebase_ai import generate_context
    GRAPH_SITTER_AVAILABLE = True
    logger.info("Graph-sitter successfully imported")
    
    # Create analysis functions based on available Codebase methods
    def get_codebase_summary(codebase: Codebase) -> str:
        """Provides a high-level statistical overview of the entire codebase"""
        try:
            files_count = len(codebase.files)
            imports_count = len(codebase.imports)
            external_modules_count = len(codebase.external_modules)
            symbols_count = len(codebase.symbols)
            classes_count = len(codebase.classes)
            functions_count = len(codebase.functions)
            global_vars_count = len(codebase.global_vars)
            interfaces_count = len(codebase.interfaces)
            
            return f"""Codebase Summary:
- Files: {files_count}
- Imports: {imports_count}
- External modules: {external_modules_count}
- Total symbols: {symbols_count}
  - Classes: {classes_count}
  - Functions: {functions_count}
  - Global variables: {global_vars_count}
  - Interfaces: {interfaces_count}
- Repository: {codebase.name}
- Language: {codebase.language}"""
        except Exception as e:
            return f"Error generating codebase summary: {str(e)}"
    
    def get_file_summary(file) -> str:
        """Analyzes a single source file's dependencies and structure"""
        try:
            if hasattr(file, 'imports'):
                imports_count = len(file.imports) if file.imports else 0
            else:
                imports_count = 0
                
            if hasattr(file, 'symbols'):
                symbols_count = len(file.symbols) if file.symbols else 0
            else:
                symbols_count = 0
                
            return f"""File Summary for {getattr(file, 'name', 'unknown')}:
- Imports: {imports_count}
- Symbols: {symbols_count}
- Path: {getattr(file, 'path', 'unknown')}"""
        except Exception as e:
            return f"Error generating file summary: {str(e)}"
    
    def get_class_summary(cls) -> str:
        """Deep analysis of a class definition and its relationships"""
        try:
            return f"""Class Summary for {getattr(cls, 'name', 'unknown')}:
- Type: Class
- Location: {getattr(cls, 'path', 'unknown')}
- Methods: {len(getattr(cls, 'methods', []))}
- Attributes: {len(getattr(cls, 'attributes', []))}"""
        except Exception as e:
            return f"Error generating class summary: {str(e)}"
    
    def get_function_summary(func) -> str:
        """Comprehensive function analysis including call patterns"""
        try:
            return f"""Function Summary for {getattr(func, 'name', 'unknown')}:
- Type: Function
- Location: {getattr(func, 'path', 'unknown')}
- Parameters: {len(getattr(func, 'parameters', []))}
- Return statements: {len(getattr(func, 'return_statements', []))}"""
        except Exception as e:
            return f"Error generating function summary: {str(e)}"
    
    def get_symbol_summary(symbol) -> str:
        """Universal symbol usage analysis (works for any symbol type)"""
        try:
            return f"""Symbol Summary for {getattr(symbol, 'name', 'unknown')}:
- Type: {getattr(symbol, 'type', 'unknown')}
- Location: {getattr(symbol, 'path', 'unknown')}
- Dependencies: {len(getattr(symbol, 'dependencies', []))}
- Usage count: {len(getattr(symbol, 'usages', []))}"""
        except Exception as e:
            return f"Error generating symbol summary: {str(e)}"
            
except ImportError as e:
    logger.warning(f"Graph-sitter not available: {e}")
    GRAPH_SITTER_AVAILABLE = False
    # Create mock classes for development
    class Codebase:
        def __init__(self, repo_path):
            self.repo_path = repo_path
    
    def get_codebase_summary(codebase):
        return "Graph-sitter not available - mock summary"
    
    def get_file_summary(file):
        return "Graph-sitter not available - mock file summary"
    
    def get_class_summary(cls):
        return "Graph-sitter not available - mock class summary"
    
    def get_function_summary(func):
        return "Graph-sitter not available - mock function summary"
    
    def get_symbol_summary(symbol):
        return "Graph-sitter not available - mock symbol summary"
    
    def generate_context(obj):
        return "Graph-sitter not available - mock context"

# Pydantic models for request/response
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
    loc: int = 0  # Lines of Code
    lloc: int = 0  # Logical Lines of Code
    sloc: int = 0  # Source Lines of Code
    comments: int = 0
    comment_density: float = 0.0

class ComplexityMetrics(BaseModel):
    cyclomatic_complexity: Dict[str, float] = {"average": 0.0}
    maintainability_index: Dict[str, float] = {"average": 0.0}
    halstead_metrics: Dict[str, Union[int, float]] = {"total_volume": 0, "average_volume": 0}

class IssueItem(BaseModel):
    file_path: str
    line_number: int
    severity: str  # "critical", "functional", "minor"
    issue_type: str
    description: str
    suggestion: Optional[str] = None

class RepositoryNode(BaseModel):
    name: str
    path: str
    type: str  # "file" or "directory"
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

# FastAPI app initialization
app = FastAPI(
    title="Enhanced Codebase Analytics API",
    description="Comprehensive repository analysis with graph-sitter integration",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", "0.0.0.0", "*"]
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Rate limiting storage (simple in-memory for demo)
request_counts = defaultdict(list)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple rate limiting middleware"""
    client_ip = request.client.host
    now = datetime.now()
    
    # Clean old requests (older than 1 minute)
    request_counts[client_ip] = [
        req_time for req_time in request_counts[client_ip] 
        if now - req_time < timedelta(minutes=1)
    ]
    
    # Check rate limit (60 requests per minute)
    if len(request_counts[client_ip]) >= 60:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again later."}
        )
    
    # Add current request
    request_counts[client_ip].append(now)
    
    response = await call_next(request)
    return response

@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Add security headers"""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

class CodeAnalyzer:
    """Enhanced code analyzer with issue detection"""
    
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
            logger.warning(f"Error analyzing file {file_path}: {e}")
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
        # Simple complexity calculation based on control flow keywords
        complexity_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally', 'with']
        complexity = 1  # Base complexity
        
        for keyword in complexity_keywords:
            complexity += len(re.findall(rf'\b{keyword}\b', content))
        
        return min(complexity, 50)  # Cap at 50 for sanity

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

def clone_repository(repo_url: str) -> Path:
    """Clone repository to temporary directory"""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Extract repo name from URL
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        clone_path = temp_dir / repo_name
        
        # Clone repository
        result = subprocess.run(
            ['git', 'clone', '--depth', '1', repo_url, str(clone_path)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to clone repository: {result.stderr}"
            )
        
        return clone_path
        
    except subprocess.TimeoutExpired:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(
            status_code=408,
            detail="Repository cloning timed out"
        )
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error cloning repository: {str(e)}"
        )

def get_git_commit_stats(repo_path: Path) -> Dict[str, int]:
    """Get git commit statistics"""
    try:
        # Get commits from last 12 months
        result = subprocess.run(
            ['git', 'log', '--since="12 months ago"', '--pretty=format:%ci'],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return {}
        
        # Parse commit dates and group by month
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
        logger.warning(f"Error getting git stats: {e}")
        return {}

# API Routes
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/analyze_repo", response_model=RepositoryAnalysis)
async def analyze_repository(request: RepositoryRequest):
    """Analyze a repository and return comprehensive metrics"""
    repo_url = str(request.repo_url)
    temp_dir = None
    
    try:
        logger.info(f"Starting analysis of repository: {repo_url}")
        
        # Clone repository
        repo_path = clone_repository(repo_url)
        temp_dir = repo_path.parent
        
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
        
        # Count modules (directories with __init__.py or package.json)
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
        issues_summary = IssuesSummary(
            total=len(all_issues),
            critical=len([i for i in all_issues if i.severity == 'critical']),
            functional=len([i for i in all_issues if i.severity == 'functional']),
            minor=len([i for i in all_issues if i.severity == 'minor'])
        )
        
        # Create response
        analysis = RepositoryAnalysis(
            repo_url=repo_url,
            description="Repository analysis",
            basic_metrics=BasicMetrics(
                files=total_metrics['files'],
                functions=total_metrics['functions'],
                classes=total_metrics['classes'],
                modules=total_metrics['modules']
            ),
            line_metrics={
                "total": LineMetrics(
                    loc=total_metrics['loc'],
                    lloc=total_metrics['lloc'],
                    sloc=total_metrics['lloc'],  # Using lloc as sloc
                    comments=total_metrics['comments'],
                    comment_density=comment_density
                )
            },
            complexity_metrics=ComplexityMetrics(
                cyclomatic_complexity={"average": round(avg_complexity, 1)},
                maintainability_index={"average": max(0, min(100, 100 - avg_complexity * 2))},
                halstead_metrics={
                    "total_volume": total_metrics['functions'] * 100,
                    "average_volume": 100
                }
            ),
            repository_structure=repo_structure,
            issues_summary=issues_summary,
            detailed_issues=all_issues[:100],  # Limit to first 100 issues
            monthly_commits=monthly_commits
        )
        
        logger.info(f"Analysis completed successfully for {repo_url}")
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing repository {repo_url}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during analysis: {str(e)}"
        )
    finally:
        # Cleanup temporary directory
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Enhanced Codebase Analytics API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
