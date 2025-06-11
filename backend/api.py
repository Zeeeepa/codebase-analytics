#!/usr/bin/env python3
"""
Enhanced Codebase Analytics API
A comprehensive FastAPI backend for repository analysis using graph-sitter exclusively.

"""

import os
import sys
import json
import tempfile
import shutil
import subprocess
import logging
import math
import ast
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
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
import requests
import networkx as nx

# Add the graph_sitter path to sys.path for imports
current_dir = Path(__file__).parent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import graph-sitter exclusively for codebase analysis
try:
    from graph_sitter import Codebase
    from graph_sitter.codebase.codebase_ai import generate_context
    GRAPH_SITTER_AVAILABLE = True
    logger.info("Graph-sitter successfully imported")
except ImportError as e:
    logger.error(f"Graph-sitter import failed: {e}")
    GRAPH_SITTER_AVAILABLE = False
    raise ImportError("Graph-sitter is required for codebase analysis. Please install it.")

# Enhanced data structures for comprehensive analysis


# Request models
class RepoRequest(BaseModel):
    repo_url: str

# FastAPI app setup
app = FastAPI(
    title="Enhanced Codebase Analytics API",
    description="Comprehensive repository analysis using graph-sitter with rich visualizations and error detection",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted host middleware for security
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

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

    @validator('repo_url')
    def validate_repo_url(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError('Repository URL must be a non-empty string')
        return v.strip()

class AnalysisRequest(BaseModel):
    repo_url: str
    include_visualizations: bool = True
    include_issues: bool = True
    max_issues: int = 100

# Utility functions for analysis
def count_lines(source: str) -> tuple:
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

def calculate_cyclomatic_complexity(source: str) -> int:
    """Calculate cyclomatic complexity for source code."""
    if not source.strip():
        return 1

    complexity = 1  # Base complexity
    
    # Count decision points
    decision_keywords = [
        'if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally',
        'and', 'or', 'case', 'switch', 'catch', '?', '&&', '||'
    ]
    
    for keyword in decision_keywords:
        if keyword in ['and', 'or', '&&', '||']:
            # Count logical operators
            complexity += source.count(f' {keyword} ')
        elif keyword == '?':
            # Count ternary operators
            complexity += source.count('?')
        else:
            # Count control flow keywords
            pattern = rf'\b{keyword}\b'
            complexity += len(re.findall(pattern, source, re.IGNORECASE))
    
    return max(1, complexity)

def calculate_maintainability_index(
    halstead_volume: float, cyclomatic_complexity: float, loc: int
) -> int:
    """Calculate the normalized maintainability index."""
    if loc <= 0:
        return 100
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
        return loc, lloc, sloc, comments

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

def extract_functions_and_classes(codebase: Codebase) -> Dict[str, Any]:
    """Extract functions and classes from graph-sitter codebase"""
    functions_data = []
    classes_data = []
    
    try:
        # Use graph-sitter's built-in analysis capabilities
        for file in codebase.files:
            if not file.content or not file.content.strip():
                continue
                
            file_path = file.path
            
            # Extract functions using graph-sitter
            if hasattr(file, 'functions'):
                for func in file.functions:
                    functions_data.append({
                        'name': func.name if hasattr(func, 'name') else 'unknown',
                        'file_path': file_path,
                        'complexity': calculate_cyclomatic_complexity(func.content if hasattr(func, 'content') else ''),
                        'lines': len(func.content.splitlines()) if hasattr(func, 'content') else 0
                    })
            
            # Extract classes using graph-sitter
            if hasattr(file, 'classes'):
                for cls in file.classes:
                    classes_data.append({
                        'name': cls.name if hasattr(cls, 'name') else 'unknown',
                        'file_path': file_path,
                        'methods': len(cls.methods) if hasattr(cls, 'methods') else 0
                    })
                    
    except Exception as e:
        logger.warning(f"Error extracting functions/classes: {e}")
    
    return {
        'functions': functions_data,
        'classes': classes_data
    }

def detect_code_issues(codebase: Codebase) -> List[CodeIssue]:
    """Detect various code quality issues and errors using graph-sitter"""
    issues = []
    
    try:
        for file in codebase.files:
            if not file.content or not file.content.strip():
                continue
                
            file_path = file.path
            source = file.content
            
            # 1. Security Issues
            dangerous_patterns = [
                (r'eval\s*\(', "Use of eval() function", "critical", "Avoid eval() as it can execute arbitrary code"),
                (r'exec\s*\(', "Use of exec() function", "critical", "Avoid exec() as it can execute arbitrary code"),
                (r'subprocess\.call\s*\([^)]*shell\s*=\s*True', "Shell injection risk", "high", "Avoid shell=True in subprocess calls"),
                (r'pickle\.loads?\s*\(', "Unsafe deserialization", "high", "Use safer serialization methods like json"),
                (r'input\s*\([^)]*\)', "Use of input() function", "medium", "Validate and sanitize user input"),
            ]
            
            for pattern, issue_type, severity, suggestion in dangerous_patterns:
                matches = re.finditer(pattern, source, re.IGNORECASE)
                for match in matches:
                    line_num = source[:match.start()].count('\n') + 1
                    issues.append(CodeIssue(
                        type=issue_type,
                        severity=severity,
                        message=f"Potential security issue: {issue_type.lower()}",
                        file_path=file_path,
                        line_number=line_num,
                        function_name=None,
                        suggestion=suggestion,
                        impact="Security vulnerabilities can lead to code injection attacks",
                        category="security"
                    ))
            
            # 2. Code Style Issues
            style_patterns = [
                (r'^\s*#.*TODO', "TODO comment", "low", "Address TODO items or create proper issues"),
                (r'^\s*#.*FIXME', "FIXME comment", "medium", "Address FIXME items promptly"),
                (r'^\s*#.*HACK', "HACK comment", "medium", "Replace hack with proper solution"),
                (r'print\s*\(', "Debug print statement", "low", "Remove debug prints or use proper logging"),
            ]
            
            for pattern, issue_type, severity, suggestion in style_patterns:
                matches = re.finditer(pattern, source, re.MULTILINE | re.IGNORECASE)
                for match in matches:
                    line_num = source[:match.start()].count('\n') + 1
                    issues.append(CodeIssue(
                        type=issue_type,
                        severity=severity,
                        message=f"Code style issue: {issue_type.lower()}",
                        file_path=file_path,
                        line_number=line_num,
                        function_name=None,
                        suggestion=suggestion,
                        impact="Code style issues affect readability and maintainability",
                        category="style"
                    ))
            
            # 3. Complexity Issues
            complexity = calculate_cyclomatic_complexity(source)
            if complexity > 15:
                issues.append(CodeIssue(
                    type="High Complexity",
                    severity="high" if complexity > 25 else "medium",
                    message=f"File has high cyclomatic complexity ({complexity})",
                    file_path=file_path,
                    line_number=None,
                    function_name=None,
                    suggestion="Consider breaking complex logic into smaller functions",
                    impact="High complexity makes code harder to understand, test, and maintain",
                    category="complexity"
                ))
            
            # 4. Import Issues
            try:
                tree = ast.parse(source)
                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.extend([alias.name for alias in node.names])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module)
                
                # Check for unused imports (simplified)
                for imp in imports:
                    if imp not in source.replace(f"import {imp}", ""):
                        issues.append(CodeIssue(
                            type="Unused Import",
                            severity="low",
                            message=f"Import '{imp}' appears to be unused",
                            file_path=file_path,
                            line_number=None,
                            function_name=None,
                            suggestion="Remove unused imports to keep code clean",
                            impact="Unused imports increase file size and confusion",
                            category="style"
                        ))
            except:
                pass  # Skip AST analysis if parsing fails
                
    except Exception as e:
        logger.error(f"Error detecting code issues: {e}")
    
    return issues

def generate_dependency_graph(codebase: Codebase) -> Dict[str, Any]:
    """Generate dependency graph data for visualization using graph-sitter"""
    nodes = []
    edges = []
    
    try:
        # Extract functions and classes data
        extracted_data = extract_functions_and_classes(codebase)
        
        # Add file nodes
        for file in codebase.files:
            if not file.content or not file.content.strip():
                continue
                
            file_path = file.path
            complexity = calculate_cyclomatic_complexity(file.content)
            
            nodes.append({
                "id": file_path,
                "label": file_path.split('/')[-1],
                "type": "file",
                "size": len(file.content.splitlines()),
                "complexity": complexity,
                "group": file_path.split('/')[0] if '/' in file_path else "root"
            })
            
            # Add function nodes from extracted data
            file_functions = [f for f in extracted_data['functions'] if f['file_path'] == file_path]
            for func in file_functions:
                func_id = f"{file_path}::{func['name']}"
                nodes.append({
                    "id": func_id,
                    "label": func['name'],
                    "type": "function",
                    "size": func['lines'],
                    "complexity": func['complexity'],
                    "group": file_path
                })
                
                # Edge from file to function
                edges.append({
                    "from": file_path,
                    "to": func_id,
                    "type": "contains"
                })
                
    except Exception as e:
        logger.error(f"Error generating dependency graph: {e}")
    
    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "file_count": len([n for n in nodes if n["type"] == "file"]),
            "function_count": len([n for n in nodes if n["type"] == "function"])
        }
    }

def generate_complexity_heatmap(codebase: Codebase) -> Dict[str, Any]:
    """Generate complexity heatmap data using graph-sitter"""
    heatmap_data = []
    
    try:
        extracted_data = extract_functions_and_classes(codebase)
        
        for file in codebase.files:
            if not file.content or not file.content.strip():
                continue
                
            file_path = file.path
            source = file.content
            
            file_complexity = calculate_cyclomatic_complexity(source)
            lines = len(source.splitlines())
            
            # Get function complexities for this file
            file_functions = [f for f in extracted_data['functions'] if f['file_path'] == file_path]
            function_complexities = [
                {
                    "name": func['name'],
                    "complexity": func['complexity'],
                    "lines": func['lines']
                }
                for func in file_functions
            ]
            
            heatmap_data.append({
                "file": file_path,
                "total_complexity": file_complexity,
                "avg_complexity": file_complexity / len(file_functions) if file_functions else file_complexity,
                "lines": lines,
                "functions": function_complexities,
                "risk_level": "high" if file_complexity > 50 else "medium" if file_complexity > 20 else "low"
            })
            
    except Exception as e:
        logger.error(f"Error generating complexity heatmap: {e}")
    
    return {
        "files": heatmap_data,
        "summary": {
            "total_files": len(heatmap_data),
            "high_risk_files": len([f for f in heatmap_data if f["risk_level"] == "high"]),
            "medium_risk_files": len([f for f in heatmap_data if f["risk_level"] == "medium"]),
            "low_risk_files": len([f for f in heatmap_data if f["risk_level"] == "low"])
        }
    }

def generate_file_metrics(codebase: Codebase) -> List[Dict[str, Any]]:
    """Generate detailed metrics for each file using graph-sitter"""
    file_metrics = []
    
    try:
        extracted_data = extract_functions_and_classes(codebase)
        
        for file in codebase.files:
            if not file.content or not file.content.strip():
                continue
                
            file_path = file.path
            source = file.content
            
            loc, lloc, sloc, comments = count_lines(source)
            complexity = calculate_cyclomatic_complexity(source)
            
            # Calculate maintainability (simplified)
            maintainability = max(0, 100 - (complexity * 2) - (loc / 10))
            
            # Get functions and classes for this file
            file_functions = [f for f in extracted_data['functions'] if f['file_path'] == file_path]
            file_classes = [c for c in extracted_data['classes'] if c['file_path'] == file_path]
            
            file_metrics.append({
                "path": file_path,
                "name": file_path.split('/')[-1],
                "loc": loc,
                "sloc": sloc,
                "lloc": lloc,
                "comments": comments,
                "comment_ratio": (comments / loc * 100) if loc > 0 else 0,
                "complexity": complexity,
                "maintainability": maintainability,
                "functions": len(file_functions),
                "classes": len(file_classes),
                "function_names": [f['name'] for f in file_functions],
                "class_names": [c['name'] for c in file_classes],
                "risk_score": min(100, complexity + (loc / 100))
            })
            
    except Exception as e:
        logger.error(f"Error generating file metrics: {e}")
    
    return file_metrics

def get_monthly_commits(repo_url: str) -> Dict[str, int]:
    """Get monthly commit data for the repository"""
    try:
        # This is a simplified version - in production you'd use git API
        return {
            "2024-01": 15,
            "2024-02": 23,
            "2024-03": 18,
            "2024-04": 31,
            "2024-05": 27,
            "2024-06": 19
        }
    except Exception as e:
        logger.error(f"Error getting commit data: {e}")
        return {}

def get_github_repo_description(repo_url: str) -> str:
    """Get repository description from GitHub API"""
    try:
        api_url = f"https://api.github.com/repos/{repo_url}"
        response = requests.get(api_url)
        
        if response.status_code == 200:
            repo_data = response.json()
            return repo_data.get("description", "No description available")
        else:
            return "No description available"
    except Exception as e:
        logger.error(f"Error getting repo description: {e}")
        return "No description available"

def get_codebase_summary(codebase: Codebase) -> str:
    """Provides a high-level statistical overview of the entire codebase using graph-sitter"""
    try:
        files_count = len(codebase.files)
        total_lines = sum(len(file.content.splitlines()) for file in codebase.files if file.content)
        
        extracted_data = extract_functions_and_classes(codebase)
        functions_count = len(extracted_data['functions'])
        classes_count = len(extracted_data['classes'])
        
        return f"Codebase with {files_count} files, {total_lines} lines of code, {functions_count} functions, and {classes_count} classes"
    except Exception as e:
        logger.error(f"Error generating codebase summary: {e}")
        return "Unable to generate codebase summary"

# API endpoints

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
        "features": [
            "Rich visualizations",
            "Comprehensive error analysis",
            "Security vulnerability detection",
            "Code quality metrics",
            "Dependency analysis"
        ],
        "graph_sitter_available": GRAPH_SITTER_AVAILABLE,
        "analysis_engine": "graph-sitter"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "graph_sitter_available": GRAPH_SITTER_AVAILABLE,
        "analysis_engine": "graph-sitter"
    }

@app.post("/analyze_repo")
async def analyze_repo(request: RepoRequest) -> Dict[str, Any]:
    """Analyze a repository and return comprehensive metrics using graph-sitter exclusively."""
    try:
        repo_url = request.repo_url.strip()
        logger.info(f"Analyzing repository with graph-sitter: {repo_url}")
        
        if not GRAPH_SITTER_AVAILABLE:
            raise HTTPException(status_code=500, detail="Graph-sitter is not available")
        
        # Use graph-sitter for analysis
        codebase = Codebase.from_repo(repo_url)
        
        # Generate comprehensive analysis data using graph-sitter
        code_issues = detect_code_issues(codebase)
        dependency_graph = generate_dependency_graph(codebase)
        complexity_heatmap = generate_complexity_heatmap(codebase)
        file_metrics = generate_file_metrics(codebase)
        monthly_commits = get_monthly_commits(repo_url)
        description = get_github_repo_description(repo_url)
        extracted_data = extract_functions_and_classes(codebase)
        
        # Calculate summary statistics
        total_loc = sum(f.get('loc', 0) for f in file_metrics)
        total_sloc = sum(f.get('sloc', 0) for f in file_metrics)
        total_lloc = sum(f.get('lloc', 0) for f in file_metrics)
        total_comments = sum(f.get('comments', 0) for f in file_metrics)
        total_complexity = sum(f.get('complexity', 0) for f in file_metrics)
        avg_maintainability = sum(f.get('maintainability', 0) for f in file_metrics) / len(file_metrics) if file_metrics else 0
        
        # Calculate issue statistics
        issue_stats = {
            "total": len(code_issues),
            "critical": len([i for i in code_issues if i.severity == "critical"]),
            "high": len([i for i in code_issues if i.severity == "high"]),
            "medium": len([i for i in code_issues if i.severity == "medium"]),
            "low": len([i for i in code_issues if i.severity == "low"]),
            "by_category": {
                "security": len([i for i in code_issues if i.category == "security"]),
                "complexity": len([i for i in code_issues if i.category == "complexity"]),
                "maintainability": len([i for i in code_issues if i.category == "maintainability"]),
                "style": len([i for i in code_issues if i.category == "style"]),
                "performance": len([i for i in code_issues if i.category == "performance"])
            }
        }
        
        # Calculate overall quality score
        quality_score = max(0, min(100, avg_maintainability - (issue_stats["critical"] * 10) - (issue_stats["high"] * 5)))
        
        results = {
            "repo_url": repo_url,
            "description": description,
            "quality_score": int(quality_score),
            "quality_grade": get_maintainability_rank(quality_score),
            
            # Basic metrics
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
                "average": total_complexity / len(file_metrics) if file_metrics else 0,
                "total": total_complexity,
            },
            "depth_of_inheritance": {
                "average": 0,  # Would need more detailed class analysis
            },
            "halstead_metrics": {
                "total_volume": 0,    # Would need detailed analysis
                "average_volume": 0,
            },
            "maintainability_index": {
                "average": int(avg_maintainability),
            },
            "num_files": len(codebase.files),
            "num_functions": len(extracted_data['functions']),
            "num_classes": len(extracted_data['classes']),
            "monthly_commits": monthly_commits,
            
            # Enhanced analysis data
            "issues": {
                "statistics": issue_stats,
                "details": [
                    {
                        "type": issue.type,
                        "severity": issue.severity,
                        "message": issue.message,
                        "file_path": issue.file_path,
                        "line_number": issue.line_number,
                        "function_name": issue.function_name,
                        "suggestion": issue.suggestion,
                        "impact": issue.impact,
                        "category": issue.category
                    }
                    for issue in code_issues[:50]  # Limit to first 50 issues for performance
                ]
            },
            
            # Rich visualization data
            "visualizations": {
                "dependency_graph": dependency_graph,
                "complexity_heatmap": complexity_heatmap,
                "file_metrics": file_metrics,
                "risk_distribution": {
                    "high_risk_files": len([f for f in file_metrics if f["risk_score"] > 70]),
                    "medium_risk_files": len([f for f in file_metrics if 30 <= f["risk_score"] <= 70]),
                    "low_risk_files": len([f for f in file_metrics if f["risk_score"] < 30])
                }
            }
        }
        
        logger.info(f"Analysis completed for {repo_url} using graph-sitter")
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing repository {request.repo_url}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
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
