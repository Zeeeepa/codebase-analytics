#!/usr/bin/env python3
"""
Consolidated Codebase Analytics API with Graph-Sitter Integration
A comprehensive FastAPI backend for repository analysis using graph-sitter.
"""

import os
import sys
import json
import tempfile
import shutil
import subprocess
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from collections import defaultdict
import re
from urllib.parse import urlparse

# FastAPI and related imports
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, validator
import uvicorn

# Import our graph-sitter analyzer
from graph_sitter_analyzer import GraphSitterAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class RepositoryRequest(BaseModel):
    repo_url: HttpUrl
    analysis_type: Optional[str] = "full"  # "full", "basic", "graph_sitter"
    
    @validator('repo_url')
    def validate_github_url(cls, v):
        url_str = str(v)
        parsed = urlparse(url_str)
        if not (parsed.netloc in ['github.com', 'gitlab.com'] or 
                'github.com' in parsed.netloc or 'gitlab.com' in parsed.netloc):
            raise ValueError('Only GitHub and GitLab repositories are supported')
        return v

class BasicMetrics(BaseModel):
    files: int = 0
    functions: int = 0
    classes: int = 0
    modules: int = 0
    interfaces: int = 0
    enums: int = 0

class LineMetrics(BaseModel):
    loc: int = 0  # Lines of Code
    lloc: int = 0  # Logical Lines of Code
    sloc: int = 0  # Source Lines of Code
    comments: int = 0
    comment_density: float = 0.0

class ComplexityMetrics(BaseModel):
    cyclomatic_complexity: Dict[str, float] = {"average": 0.0, "max": 0.0, "min": 0.0}
    maintainability_index: Dict[str, float] = {"average": 0.0}
    halstead_metrics: Dict[str, Union[int, float]] = {"total_volume": 0, "average_volume": 0}

class IssueItem(BaseModel):
    file_path: str
    line_number: int
    severity: str  # "critical", "major", "minor"
    issue_type: str
    description: str
    suggestion: Optional[str] = None
    context: Optional[str] = None

class RepositoryNode(BaseModel):
    name: str
    path: str
    type: str  # "file" or "directory"
    issue_count: int = 0
    critical_issues: int = 0
    major_issues: int = 0
    minor_issues: int = 0
    children: Optional[List['RepositoryNode']] = None
    issues: Optional[List[IssueItem]] = None

class IssuesSummary(BaseModel):
    total: int = 0
    critical: int = 0
    major: int = 0
    minor: int = 0

class GraphSitterAnalysis(BaseModel):
    """Graph-sitter specific analysis results"""
    syntax_tree_depth: int = 0
    node_count: int = 0
    symbol_table: Dict[str, List[str]] = {}
    inheritance_hierarchy: Dict[str, List[str]] = {}
    function_calls: Dict[str, List[str]] = {}
    imports_exports: Dict[str, List[str]] = {}

class RepositoryAnalysis(BaseModel):
    repo_url: str
    description: str = "Repository analysis"
    basic_metrics: BasicMetrics
    line_metrics: Dict[str, LineMetrics]
    complexity_metrics: Dict[str, ComplexityMetrics]
    issues_summary: IssuesSummary
    repository_tree: RepositoryNode
    issues: List[IssueItem]
    graph_sitter_analysis: Optional[GraphSitterAnalysis] = None
    analysis_timestamp: str
    git_stats: Dict[str, Any] = {}

# Initialize FastAPI app
app = FastAPI(
    title="Consolidated Codebase Analytics API",
    description="Advanced repository analysis with graph-sitter integration",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConsolidatedCodeAnalyzer:
    """Unified code analyzer with graph-sitter integration"""
    
    def __init__(self):
        self.graph_sitter_analyzer = GraphSitterAnalyzer()
        self.supported_extensions = {
            '.py': 'python',
            '.js': 'javascript', 
            '.ts': 'typescript',
            '.tsx': 'tsx',
            '.jsx': 'jsx',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby'
        }
        
        # Issue detection patterns
        self.issue_patterns = {
            'critical': [
                (r'assert\s+isinstance\(.*\)', 'Uses assert for runtime type checking'),
                (r'except\s*:', 'Bare except clause'),
                (r'eval\s*\(', 'Uses eval() - security risk'),
                (r'exec\s*\(', 'Uses exec() - security risk'),
                (r'__import__\s*\(', 'Dynamic import - potential security risk'),
                (r'subprocess\.call\([^)]*shell=True', 'Shell injection vulnerability'),
                (r'os\.system\s*\(', 'Command injection vulnerability'),
                (r'pickle\.loads?\s*\(', 'Unsafe deserialization'),
                (r'yaml\.load\s*\([^,)]*\)', 'Unsafe YAML loading'),
            ],
            'major': [
                (r'#\s*TODO[:\s]', 'Contains TODOs indicating incomplete implementation'),
                (r'#\s*FIXME[:\s]', 'Contains FIXME comments'),
                (r'#\s*HACK[:\s]', 'Contains HACK comments'),
                (r'def\s+\w+\([^)]*\):\s*\n(\s*#.*\n)*\s*pass\s*$', 'Empty function implementation'),
                (r'class\s+\w+[^:]*:\s*\n(\s*#.*\n)*\s*pass\s*$', 'Empty class implementation'),
                (r'if\s+__name__\s*==\s*["\']__main__["\']:', 'Missing main guard'),
                (r'print\s*\(', 'Debug print statement'),
                (r'console\.log\s*\(', 'Debug console.log statement'),
                (r'debugger;?', 'Debugger statement left in code'),
            ],
            'minor': [
                (r'^\s*$\n^\s*$\n^\s*$', 'Multiple consecutive empty lines'),
                (r'\s+$', 'Trailing whitespace'),
                (r'\t', 'Tab character (consider using spaces)'),
                (r'var\s+', 'Using var instead of let/const'),
                (r'==\s*null', 'Loose equality with null'),
                (r'!=\s*null', 'Loose inequality with null'),
                (r'==\s*undefined', 'Loose equality with undefined'),
                (r'!=\s*undefined', 'Loose inequality with undefined'),
            ]
        }

    async def analyze_repository(self, repo_path: Path, analysis_type: str = "full") -> RepositoryAnalysis:
        """Main analysis function that coordinates all analysis types"""
        logger.info(f"Starting {analysis_type} analysis of repository: {repo_path}")
        
        # Basic file discovery
        source_files = self._discover_source_files(repo_path)
        
        # Initialize metrics
        basic_metrics = BasicMetrics()
        line_metrics = {}
        complexity_metrics = {}
        all_issues = []
        
        # Analyze each file
        file_analyses = {}
        for file_path in source_files:
            try:
                analysis = await self._analyze_file(file_path)
                file_analyses[str(file_path.relative_to(repo_path))] = analysis
                
                # Aggregate metrics
                basic_metrics.files += 1
                basic_metrics.functions += analysis.get('functions', 0)
                basic_metrics.classes += analysis.get('classes', 0)
                basic_metrics.modules += 1 if analysis.get('is_module', False) else 0
                
                # Collect issues
                all_issues.extend(analysis.get('issues', []))
                
            except Exception as e:
                logger.error(f"Error analyzing file {file_path}: {str(e)}")
                continue
        
        # Build repository tree
        repo_tree = self._build_repository_tree(repo_path, file_analyses)
        
        # Calculate issues summary
        issues_summary = self._calculate_issues_summary(all_issues)
        
        # Graph-sitter analysis (if requested)
        graph_sitter_analysis = None
        if analysis_type in ["full", "graph_sitter"]:
            graph_sitter_analysis = await self._perform_graph_sitter_analysis(repo_path, source_files)
        
        # Git statistics
        git_stats = await self._get_git_stats(repo_path)
        
        return RepositoryAnalysis(
            repo_url=str(repo_path),
            description=f"Analysis of {repo_path.name}",
            basic_metrics=basic_metrics,
            line_metrics=line_metrics,
            complexity_metrics=complexity_metrics,
            issues_summary=issues_summary,
            repository_tree=repo_tree,
            issues=all_issues,
            graph_sitter_analysis=graph_sitter_analysis,
            analysis_timestamp=datetime.now().isoformat(),
            git_stats=git_stats
        )

    def _discover_source_files(self, repo_path: Path) -> List[Path]:
        """Discover all source files in the repository"""
        source_files = []
        
        for ext in self.supported_extensions.keys():
            pattern = f"**/*{ext}"
            files = list(repo_path.glob(pattern))
            source_files.extend(files)
        
        # Filter out common ignore patterns
        ignore_patterns = [
            'node_modules', '.git', '__pycache__', '.pytest_cache',
            'venv', 'env', '.env', 'dist', 'build', '.next',
            'coverage', '.coverage', 'htmlcov'
        ]
        
        filtered_files = []
        for file_path in source_files:
            if not any(ignore in str(file_path) for ignore in ignore_patterns):
                filtered_files.append(file_path)
        
        return filtered_files

    async def _analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single file using graph-sitter integration"""
        try:
            # Use graph-sitter analyzer for detailed analysis
            gs_analysis = self.graph_sitter_analyzer.analyze_file(file_path)
            
            # Fallback to basic analysis if graph-sitter fails
            if not gs_analysis:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                gs_analysis = {
                    'path': str(file_path),
                    'size': len(content),
                    'lines': len(content.split('\n')),
                    'functions': [],
                    'classes': [],
                    'is_module': False,
                    'issues': []
                }
            
            # Convert graph-sitter results to our format
            analysis = {
                'path': gs_analysis.get('file_path', str(file_path)),
                'size': gs_analysis.get('lines_of_code', 0) * 50,  # Estimate
                'lines': gs_analysis.get('lines_of_code', 0),
                'functions': len(gs_analysis.get('functions', [])),
                'classes': len(gs_analysis.get('classes', [])),
                'is_module': len(gs_analysis.get('imports', [])) > 0,
                'issues': [],
                'complexity_score': gs_analysis.get('complexity_score', 0),
                'graph_sitter_data': gs_analysis
            }
            
            # Add traditional issue detection
            if 'content' not in gs_analysis:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    analysis['issues'] = self._detect_issues(content, str(file_path))
                except Exception:
                    pass
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {str(e)}")
            return {}

    def _detect_issues(self, content: str, file_path: str) -> List[IssueItem]:
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
                            issue_type=pattern.split('\\\\')[0][:20] + "...",
                            description=description,
                            suggestion=self._get_suggestion(severity, pattern),
                            context=line.strip()
                        ))
        
        return issues

    def _get_suggestion(self, severity: str, pattern: str) -> str:
        """Get improvement suggestions for detected issues"""
        suggestions = {
            'critical': "This is a critical issue that may cause runtime errors or security vulnerabilities. Please fix immediately.",
            'major': "This affects functionality or code quality. Consider refactoring or completing the implementation.",
            'minor': "This is a minor issue that affects code style or maintainability. Consider cleaning up when convenient."
        }
        return suggestions.get(severity, "Consider reviewing this code section.")

    def _build_repository_tree(self, repo_path: Path, file_analyses: Dict[str, Dict]) -> RepositoryNode:
        """Build hierarchical repository tree with issue counts"""
        
        def create_node(path: Path, relative_path: str = "") -> RepositoryNode:
            node = RepositoryNode(
                name=path.name,
                path=relative_path or path.name,
                type="directory" if path.is_dir() else "file"
            )
            
            if path.is_file():
                # File node - get analysis data
                rel_path = str(path.relative_to(repo_path))
                if rel_path in file_analyses:
                    analysis = file_analyses[rel_path]
                    issues = analysis.get('issues', [])
                    node.issues = issues
                    node.issue_count = len(issues)
                    node.critical_issues = len([i for i in issues if i.severity == 'critical'])
                    node.major_issues = len([i for i in issues if i.severity == 'major'])
                    node.minor_issues = len([i for i in issues if i.severity == 'minor'])
            else:
                # Directory node - aggregate children
                children = []
                for child_path in sorted(path.iterdir()):
                    if child_path.name.startswith('.'):
                        continue
                    child_relative = f"{relative_path}/{child_path.name}" if relative_path else child_path.name
                    child_node = create_node(child_path, child_relative)
                    children.append(child_node)
                    
                    # Aggregate issue counts
                    node.issue_count += child_node.issue_count
                    node.critical_issues += child_node.critical_issues
                    node.major_issues += child_node.major_issues
                    node.minor_issues += child_node.minor_issues
                
                node.children = children if children else None
            
            return node
        
        return create_node(repo_path)

    def _calculate_issues_summary(self, issues: List[IssueItem]) -> IssuesSummary:
        """Calculate summary statistics for issues"""
        summary = IssuesSummary(total=len(issues))
        
        for issue in issues:
            if issue.severity == 'critical':
                summary.critical += 1
            elif issue.severity == 'major':
                summary.major += 1
            elif issue.severity == 'minor':
                summary.minor += 1
        
        return summary

    async def _perform_graph_sitter_analysis(self, repo_path: Path, source_files: List[Path]) -> GraphSitterAnalysis:
        """Perform graph-sitter based analysis using the integrated analyzer"""
        
        analysis = GraphSitterAnalysis()
        all_analyses = []
        
        # Analyze each file with graph-sitter
        for file_path in source_files:
            try:
                gs_analysis = self.graph_sitter_analyzer.analyze_file(file_path)
                if gs_analysis:
                    all_analyses.append(gs_analysis)
            except Exception as e:
                logger.error(f"Error in graph-sitter analysis for {file_path}: {str(e)}")
                continue
        
        # Aggregate results
        analysis.node_count = sum(len(a.get('functions', [])) + len(a.get('classes', [])) for a in all_analyses)
        analysis.syntax_tree_depth = max((a.get('complexity_score', 0) for a in all_analyses), default=0)
        
        # Build symbol table
        for gs_analysis in all_analyses:
            file_path = gs_analysis.get('file_path', '')
            if file_path:
                rel_path = str(Path(file_path).relative_to(repo_path)) if repo_path in Path(file_path).parents else file_path
                symbols = []
                
                # Add functions
                for func in gs_analysis.get('functions', []):
                    symbols.append(f"function:{func.get('name', 'unknown')}")
                
                # Add classes
                for cls in gs_analysis.get('classes', []):
                    symbols.append(f"class:{cls.get('name', 'unknown')}")
                
                if symbols:
                    analysis.symbol_table[rel_path] = symbols
        
        # Build inheritance hierarchy (simplified)
        analysis.inheritance_hierarchy = self.graph_sitter_analyzer.get_inheritance_hierarchy(all_analyses)
        
        # Build function call graph (simplified)
        analysis.function_calls = self.graph_sitter_analyzer.get_function_call_graph(all_analyses)
        
        # Build imports/exports mapping
        for gs_analysis in all_analyses:
            file_path = gs_analysis.get('file_path', '')
            if file_path:
                rel_path = str(Path(file_path).relative_to(repo_path)) if repo_path in Path(file_path).parents else file_path
                imports = [imp.get('module', '') for imp in gs_analysis.get('imports', [])]
                exports = [exp.get('name', '') for exp in gs_analysis.get('exports', [])]
                
                if imports or exports:
                    analysis.imports_exports[rel_path] = imports + exports
        
        return analysis

    async def _get_git_stats(self, repo_path: Path) -> Dict[str, Any]:
        """Get git repository statistics"""
        stats = {}
        
        try:
            # Get commit count
            result = await asyncio.create_subprocess_exec(
                'git', 'rev-list', '--count', 'HEAD',
                cwd=repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                stats['total_commits'] = int(stdout.decode().strip())
            
            # Get contributors count
            result = await asyncio.create_subprocess_exec(
                'git', 'shortlog', '-sn', '--all',
                cwd=repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                contributors = stdout.decode().strip().split('\n')
                stats['contributors'] = len([c for c in contributors if c.strip()])
            
        except Exception as e:
            logger.error(f"Error getting git stats: {str(e)}")
            stats['error'] = str(e)
        
        return stats

async def clone_repository(repo_url: str) -> Path:
    """Clone repository to temporary directory with proper error handling"""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Extract repo name from URL
        parsed_url = urlparse(str(repo_url))
        repo_name = parsed_url.path.split('/')[-1].replace('.git', '')
        clone_path = temp_dir / repo_name
        
        # Clone repository with timeout
        process = await asyncio.create_subprocess_exec(
            'git', 'clone', '--depth', '1', str(repo_url), str(clone_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
        except asyncio.TimeoutError:
            process.kill()
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=408, detail="Repository cloning timed out")
        
        if process.returncode != 0:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(
                status_code=400,
                detail=f"Failed to clone repository: {stderr.decode()}"
            )
        
        return clone_path
        
    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error cloning repository: {str(e)}"
        )

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

@app.post("/analyze_repo", response_model=RepositoryAnalysis)
async def analyze_repository_endpoint(request: RepositoryRequest):
    """
    Main endpoint for repository analysis with graph-sitter integration
    """
    logger.info(f"Analyzing repository: {request.repo_url}")
    
    repo_path = None
    try:
        # Clone repository
        repo_path = await clone_repository(str(request.repo_url))
        
        # Initialize analyzer
        analyzer = ConsolidatedCodeAnalyzer()
        
        # Perform analysis
        analysis = await analyzer.analyze_repository(
            repo_path, 
            request.analysis_type or "full"
        )
        
        # Update repo_url in response
        analysis.repo_url = str(request.repo_url)
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )
    finally:
        # Cleanup
        if repo_path and repo_path.exists():
            shutil.rmtree(repo_path.parent, ignore_errors=True)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Consolidated Codebase Analytics API with Graph-Sitter Integration",
        "version": "2.0.0",
        "endpoints": {
            "analyze": "/analyze_repo",
            "health": "/health",
            "docs": "/docs"
        },
        "features": [
            "Repository cloning and analysis",
            "Graph-sitter integration",
            "Issue detection and classification",
            "Metrics calculation",
            "Git statistics",
            "Hierarchical repository tree"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
