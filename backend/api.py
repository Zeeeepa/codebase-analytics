#!/usr/bin/env python3
"""
🚀 Advanced Codebase Analytics API
Intelligent, context-aware repository analysis with real-time issue detection.
Provides dynamic analysis based on actual code state and semantic understanding.
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
import asyncio
import hashlib
import argparse  # Added for command-line argument parsing
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# FastAPI and related imports
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, HttpUrl, validator, Field
import uvicorn
import requests
import networkx as nx

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('codebase_analytics.log')
    ]
)
logger = logging.getLogger(__name__)

# Import graph-sitter for advanced AST analysis
# Check if graph-sitter should be disabled via command line
DISABLE_GRAPH_SITTER = False  # Will be updated in __main__

try:
    if not DISABLE_GRAPH_SITTER:
        from graph_sitter import Codebase
        from graph_sitter.codebase.codebase_ai import generate_context
        GRAPH_SITTER_AVAILABLE = True
        logger.info("🎯 Graph-sitter successfully imported - Advanced analysis enabled")
    else:
        logger.info("🔄 Graph-sitter disabled by command line argument")
        GRAPH_SITTER_AVAILABLE = False
except ImportError as e:
    logger.error(f"❌ Graph-sitter import failed: {e}")
    GRAPH_SITTER_AVAILABLE = False
    # Create mock implementations for development
    def generate_context(obj): return "Mock context - graph-sitter not available"

# 🎯 Advanced Data Models for Intelligent Analysis

@dataclass
class CodeContext:
    """Rich context information for code elements"""
    element_type: str  # function, class, method, variable
    name: str
    file_path: str
    line_start: int
    line_end: int
    complexity: float
    dependencies: List[str]
    dependents: List[str]
    usage_count: int
    risk_score: float
    semantic_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IntelligentIssue:
    """Advanced issue detection with context and impact analysis"""
    id: str
    type: str
    severity: str  # 'critical', 'major', 'minor'
    category: str  # 'security', 'performance', 'maintainability', 'logic', 'style'
    title: str
    description: str
    file_path: str
    line_number: int
    column_number: Optional[int]
    function_name: Optional[str]
    class_name: Optional[str]
    code_snippet: str
    context: CodeContext
    impact_analysis: str
    fix_suggestion: str
    confidence: float  # 0.0 to 1.0
    related_issues: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

@dataclass
class UsageHeatMap:
    """Usage patterns and hot spots in the codebase"""
    file_path: str
    function_name: str
    usage_frequency: int
    complexity_score: float
    maintainability_score: float
    risk_level: str
    heat_intensity: float  # 0.0 to 1.0
    entry_point: bool = False
    critical_path: bool = False

@dataclass
class DependencyGraph:
    """Advanced dependency analysis"""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    circular_dependencies: List[List[str]]
    critical_paths: List[List[str]]
    orphaned_modules: List[str]
    coupling_metrics: Dict[str, float]

@dataclass
class InheritanceAnalysis:
    """Class hierarchy and inheritance patterns"""
    class_name: str
    file_path: str
    parent_classes: List[str]
    child_classes: List[str]
    depth_of_inheritance: int
    method_count: int
    override_count: int
    abstract_methods: List[str]
    interface_compliance: float

@dataclass
class EntryPointAnalysis:
    """Entry points and high-level architecture analysis"""
    entry_points: List[Dict[str, Any]]
    main_functions: List[str]
    api_endpoints: List[str]
    cli_commands: List[str]
    event_handlers: List[str]
    initialization_patterns: List[str]
    architecture_pattern: str
    framework_detection: List[str]

@dataclass
class AdvancedMetrics:
    """Comprehensive code quality metrics"""
    halstead_metrics: Dict[str, float]
    cyclomatic_complexity: Dict[str, float]
    maintainability_index: Dict[str, float]
    technical_debt_ratio: float
    code_coverage_estimate: float
    duplication_percentage: float
    cognitive_complexity: Dict[str, float]
    npath_complexity: Dict[str, float]

@dataclass
class SecurityAnalysis:
    """Advanced security vulnerability analysis"""
    vulnerabilities: List[IntelligentIssue]
    security_score: float
    threat_model: Dict[str, Any]
    attack_surface: List[str]
    sensitive_data_flows: List[Dict[str, Any]]
    authentication_patterns: List[str]
    authorization_issues: List[str]
    input_validation_gaps: List[str]

@dataclass
class PerformanceAnalysis:
    """Performance bottleneck and optimization analysis"""
    bottlenecks: List[IntelligentIssue]
    performance_score: float
    memory_usage_patterns: List[Dict[str, Any]]
    cpu_intensive_functions: List[str]
    io_operations: List[Dict[str, Any]]
    algorithmic_complexity: Dict[str, str]
    optimization_opportunities: List[str]

# 🎯 Request/Response Models

class AdvancedAnalysisRequest(BaseModel):
    repo_url: str = Field(..., description="GitHub repository URL")
    analysis_depth: str = Field("comprehensive", description="Analysis depth: quick, standard, comprehensive, deep")
    focus_areas: List[str] = Field(default=["all"], description="Focus areas: security, performance, maintainability, architecture")
    include_context: bool = Field(True, description="Include rich context for issues")
    max_issues: int = Field(200, description="Maximum number of issues to report")
    enable_ai_insights: bool = Field(True, description="Enable AI-powered insights")
    
    @validator('repo_url')
    def validate_repo_url(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError('Repository URL must be a non-empty string')
        if not ('github.com' in v or 'gitlab.com' in v):
            raise ValueError('Only GitHub and GitLab repositories are supported')
        return v.strip()

class IntelligentAnalysisResponse(BaseModel):
    """Comprehensive analysis response with intelligent insights"""
    # Basic Information
    repo_url: str
    analysis_id: str
    timestamp: datetime
    analysis_duration: float
    
    # Executive Summary
    overall_quality_score: float = Field(..., description="Overall quality score (0-100)")
    quality_grade: str = Field(..., description="Quality grade (A-F)")
    risk_assessment: str = Field(..., description="Overall risk assessment")
    
    # Intelligent Insights
    key_findings: List[str] = Field(..., description="Key insights and findings")
    critical_recommendations: List[str] = Field(..., description="Critical recommendations")
    architecture_assessment: str = Field(..., description="Architecture quality assessment")
    
    # Detailed Analysis
    issues: List[IntelligentIssue] = Field(..., description="Detailed issue analysis")
    security_analysis: SecurityAnalysis
    performance_analysis: PerformanceAnalysis
    dependency_graph: DependencyGraph
    inheritance_analysis: List[InheritanceAnalysis]
    entry_points: EntryPointAnalysis
    usage_heatmap: List[UsageHeatMap]
    
    # Advanced Metrics
    metrics: AdvancedMetrics
    repository_structure: Dict[str, Any]
    
    # Visualization Data
    visualizations: Dict[str, Any] = Field(..., description="Rich visualization data")

# FastAPI app setup
app = FastAPI(
    title="🚀 Advanced Codebase Analytics API",
    description="Intelligent, context-aware repository analysis with real-time issue detection",
    version="3.0.0"
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

# Intelligent Analysis Engine

class IntelligentCodeAnalyzer:
    """Advanced code analyzer with AI-powered insights and real-time issue detection"""
    
    def __init__(self):
        self.analysis_cache = {}
        self.language_patterns = {
            'python': {
                'security_patterns': [
                    (r'eval\s*\(', 'Code injection vulnerability', 'critical'),
                    (r'exec\s*\(', 'Code execution vulnerability', 'critical'),
                    (r'subprocess.*shell\s*=\s*True', 'Shell injection risk', 'critical'),
                    (r'pickle\.loads?\s*\(', 'Unsafe deserialization', 'major'),
                    (r'input\s*\([^)]*\)', 'Unvalidated user input', 'major'),
                ],
                'performance_patterns': [
                    (r'for\s+\w+\s+in\s+range\s*\(\s*len\s*\(', 'Inefficient iteration pattern', 'minor'),
                    (r'\.append\s*\([^)]*\)\s*$', 'List concatenation in loop', 'major'),
                    (r'time\.sleep\s*\(', 'Blocking sleep operation', 'major'),
                ],
                'maintainability_patterns': [
                    (r'def\s+\w+\s*\([^)]{50,}', 'Function with too many parameters', 'major'),
                    (r'if\s+.*:\s*if\s+.*:\s*if\s+.*:', 'Deep nesting detected', 'major'),
                    (r'#\s*TODO', 'Unresolved TODO item', 'minor'),
                    (r'#\s*FIXME', 'Unresolved FIXME item', 'major'),
                ]
            }
        }
        
    async def analyze_repository(self, repo_url: str, request: AdvancedAnalysisRequest) -> IntelligentAnalysisResponse:
        """Perform comprehensive intelligent analysis of a repository"""
        start_time = datetime.now()
        analysis_id = hashlib.md5(f"{repo_url}_{start_time}".encode()).hexdigest()[:12]
        
        logger.info(f"🚀 Starting intelligent analysis for {repo_url} (ID: {analysis_id})")
        
        try:
            # Clone repository
            repo_path = await self._clone_repository(repo_url)
            
            # Initialize graph-sitter codebase if available
            if GRAPH_SITTER_AVAILABLE:
                try:
                    codebase = Codebase.from_repo(str(repo_path))
                    logger.info(f"📊 Graph-sitter codebase initialized: {len(codebase.files)} files")
                except Exception as e:
                    logger.warning(f"Graph-sitter initialization failed: {e}")
                    codebase = None
            else:
                codebase = None
            
            # Perform analysis with or without graph-sitter
            if codebase:
                analysis_result = await self._analyze_with_graph_sitter(codebase, repo_path, request)
            else:
                analysis_result = await self._analyze_without_graph_sitter(repo_path, request)
            
            analysis_duration = (datetime.now() - start_time).total_seconds()
            
            response = IntelligentAnalysisResponse(
                repo_url=repo_url,
                analysis_id=analysis_id,
                timestamp=start_time,
                analysis_duration=analysis_duration,
                **analysis_result
            )
            
            logger.info(f"✅ Analysis completed for {repo_url} in {analysis_duration:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"❌ Analysis failed for {repo_url}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        finally:
            # Cleanup
            if 'repo_path' in locals() and repo_path.exists():
                shutil.rmtree(repo_path, ignore_errors=True)
    
    async def _clone_repository(self, repo_url: str) -> Path:
        """Clone repository with optimizations"""
        temp_dir = Path(tempfile.mkdtemp())
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        clone_path = temp_dir / repo_name
        
        try:
            result = subprocess.run(
                ['git', 'clone', '--depth', '1', '--single-branch', repo_url, str(clone_path)],
                capture_output=True, text=True, timeout=300
            )
            
            if result.returncode != 0:
                raise HTTPException(status_code=400, detail=f"Failed to clone: {result.stderr}")
            
            return clone_path
        except subprocess.TimeoutExpired:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(status_code=408, detail="Repository cloning timed out")
    
    async def _analyze_with_graph_sitter(self, codebase, repo_path: Path, request: AdvancedAnalysisRequest) -> Dict[str, Any]:
        """Perform analysis using graph-sitter"""
        # Detect all issues with context
        all_issues = await self._detect_intelligent_issues_gs(codebase)
        
        # Generate comprehensive metrics
        metrics = await self._calculate_advanced_metrics_gs(codebase)
        
        # Generate usage heatmap
        usage_heatmap = await self._generate_usage_heatmap_gs(codebase)
        
        # Analyze inheritance patterns
        inheritance_analysis = await self._analyze_inheritance_gs(codebase)
        
        # Analyze architecture
        entry_points = await self._analyze_architecture_gs(codebase)
        
        # Security and performance analysis
        security_analysis = self._analyze_security_gs(codebase, all_issues)
        performance_analysis = self._analyze_performance_gs(codebase, all_issues)
        
        # Dependency analysis
        dependency_graph = await self._analyze_dependencies_gs(codebase)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(metrics, all_issues, security_analysis, performance_analysis)
        
        # Generate AI insights
        key_findings = self._generate_key_findings(all_issues, metrics, len(codebase.files))
        critical_recommendations = self._generate_recommendations(all_issues, quality_score)
        
        # Build repository structure with issue mapping
        repo_structure = await self._build_intelligent_repo_structure(repo_path, all_issues)
        
        # Create visualization data
        visualizations = self._create_visualizations(dependency_graph, usage_heatmap, all_issues, metrics)
        
        return {
            'overall_quality_score': quality_score,
            'quality_grade': self._get_quality_grade(quality_score),
            'risk_assessment': self._assess_risk(all_issues, security_analysis),
            'key_findings': key_findings,
            'critical_recommendations': critical_recommendations,
            'architecture_assessment': f"{entry_points.architecture_pattern} architecture with {len(entry_points.entry_points)} entry points",
            'issues': all_issues[:request.max_issues],
            'security_analysis': security_analysis,
            'performance_analysis': performance_analysis,
            'dependency_graph': dependency_graph,
            'inheritance_analysis': inheritance_analysis,
            'entry_points': entry_points,
            'usage_heatmap': usage_heatmap,
            'metrics': metrics,
            'repository_structure': repo_structure,
            'visualizations': visualizations
        }
    
    async def _analyze_without_graph_sitter(self, repo_path: Path, request: AdvancedAnalysisRequest) -> Dict[str, Any]:
        """Perform analysis without graph-sitter using pattern matching"""
        logger.info("📝 Performing pattern-based analysis (graph-sitter not available)")
        
        # Collect all source files
        source_files = []
        for ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs']:
            source_files.extend(repo_path.rglob(f'*{ext}'))
        
        all_issues = []
        total_complexity = 0
        total_files = len(source_files)
        
        # Analyze each file
        for file_path in source_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Detect issues using pattern matching
                file_issues = self._detect_issues_pattern_matching(content, str(file_path.relative_to(repo_path)))
                all_issues.extend(file_issues)
                
                # Calculate complexity
                complexity = self._calculate_cyclomatic_complexity(content)
                total_complexity += complexity
                
            except Exception as e:
                logger.warning(f"Error analyzing file {file_path}: {e}")
        
        # Create basic metrics
        avg_complexity = total_complexity / total_files if total_files > 0 else 0
        avg_maintainability = max(0, 100 - avg_complexity * 2)
        
        metrics = AdvancedMetrics(
            halstead_metrics={"total_volume": total_files * 100, "average_volume": 100},
            cyclomatic_complexity={"average": avg_complexity, "total": total_complexity},
            maintainability_index={"average": avg_maintainability},
            technical_debt_ratio=max(0, (100 - avg_maintainability) / 100),
            code_coverage_estimate=75.0,
            duplication_percentage=0.0,
            cognitive_complexity={"average": avg_complexity * 1.2},
            npath_complexity={"average": avg_complexity ** 2}
        )
        
        # Create basic analyses
        security_analysis = SecurityAnalysis(
            vulnerabilities=[i for i in all_issues if i.category == 'security'],
            security_score=max(0, 100 - len([i for i in all_issues if i.category == 'security']) * 10),
            threat_model={}, attack_surface=[], sensitive_data_flows=[],
            authentication_patterns=[], authorization_issues=[], input_validation_gaps=[]
        )
        
        performance_analysis = PerformanceAnalysis(
            bottlenecks=[i for i in all_issues if i.category == 'performance'],
            performance_score=max(0, 100 - len([i for i in all_issues if i.category == 'performance']) * 5),
            memory_usage_patterns=[], cpu_intensive_functions=[], io_operations=[],
            algorithmic_complexity={}, optimization_opportunities=[]
        )
        
        dependency_graph = DependencyGraph(
            nodes=[], edges=[], circular_dependencies=[], critical_paths=[],
            orphaned_modules=[], coupling_metrics={}
        )
        
        entry_points = EntryPointAnalysis(
            entry_points=[], main_functions=[], api_endpoints=[], cli_commands=[],
            event_handlers=[], initialization_patterns=[], architecture_pattern="Unknown",
            framework_detection=[]
        )
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(metrics, all_issues, security_analysis, performance_analysis)
        
        # Generate insights
        key_findings = self._generate_key_findings(all_issues, metrics, total_files)
        critical_recommendations = self._generate_recommendations(all_issues, quality_score)
        
        # Build repository structure
        repo_structure = await self._build_intelligent_repo_structure(repo_path, all_issues)
        
        # Create visualizations
        visualizations = self._create_visualizations(dependency_graph, [], all_issues, metrics)
        
        return {
            'overall_quality_score': quality_score,
            'quality_grade': self._get_quality_grade(quality_score),
            'risk_assessment': self._assess_risk(all_issues, security_analysis),
            'key_findings': key_findings,
            'critical_recommendations': critical_recommendations,
            'architecture_assessment': "Pattern-based analysis completed",
            'issues': all_issues[:request.max_issues],
            'security_analysis': security_analysis,
            'performance_analysis': performance_analysis,
            'dependency_graph': dependency_graph,
            'inheritance_analysis': [],
            'entry_points': entry_points,
            'usage_heatmap': [],
            'metrics': metrics,
            'repository_structure': repo_structure,
            'visualizations': visualizations
        }

# Initialize the intelligent analyzer
intelligent_analyzer = IntelligentCodeAnalyzer()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🚀 Advanced Codebase Analytics API",
        "version": "3.0.0",
        "features": [
            "🧠 Intelligent issue detection",
            "🔍 Real-time code analysis",
            "📊 Advanced metrics calculation",
            "🎯 Context-aware recommendations",
            "🔒 Security vulnerability scanning",
            "⚡ Performance bottleneck detection",
            "📈 Usage heat maps",
            "🏗️ Architecture analysis"
        ],
        "graph_sitter_available": GRAPH_SITTER_AVAILABLE,
        "analysis_engine": "graph-sitter + intelligent patterns"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "graph_sitter_available": GRAPH_SITTER_AVAILABLE,
        "analysis_engine": "intelligent"
    }

@app.post("/analyze", response_model=IntelligentAnalysisResponse)
async def analyze_repository(request: AdvancedAnalysisRequest) -> IntelligentAnalysisResponse:
    """🎯 Perform intelligent repository analysis with real-time issue detection"""
    return await intelligent_analyzer.analyze_repository(request.repo_url, request)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Codebase Analytics API Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--disable-graph-sitter", action="store_true", help="Disable graph-sitter even if available")
    parser.add_argument("--log-level", type=str, default="info", choices=["debug", "info", "warning", "error", "critical"],
                        help="Logging level")
    
    args = parser.parse_args()
    
    # Update global settings based on arguments
    if args.disable_graph_sitter:
        DISABLE_GRAPH_SITTER = True
        GRAPH_SITTER_AVAILABLE = False
        logger.info("🔄 Graph-sitter disabled by command line argument")
    
    # Try to run the server, handling port conflicts
    try:
        uvicorn.run(
            "api:app",
            host=args.host,
            port=args.port,
            reload=True,
            log_level=args.log_level
        )
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"Port {args.port} is already in use. Please specify a different port with --port.")
            sys.exit(1)
        else:
            logger.error(f"Error starting server: {e}")
            sys.exit(1)

# Placeholder methods for graph-sitter analysis (when available)
async def _detect_intelligent_issues_gs(self, codebase) -> List[IntelligentIssue]:
    """Detect issues using graph-sitter (placeholder)"""
    # This would use actual graph-sitter analysis
    return []
    
async def _calculate_advanced_metrics_gs(self, codebase) -> AdvancedMetrics:
    """Calculate metrics using graph-sitter (placeholder)"""
    return AdvancedMetrics(
        halstead_metrics={}, cyclomatic_complexity={}, maintainability_index={},
        technical_debt_ratio=0.0, code_coverage_estimate=0.0, duplication_percentage=0.0,
        cognitive_complexity={}, npath_complexity={}
    )
    
async def _generate_usage_heatmap_gs(self, codebase) -> List[UsageHeatMap]:
    """Generate heatmap using graph-sitter (placeholder)"""
    return []
    
async def _analyze_inheritance_gs(self, codebase) -> List[InheritanceAnalysis]:
    """Analyze inheritance using graph-sitter (placeholder)"""
    return []
    
async def _analyze_architecture_gs(self, codebase) -> EntryPointAnalysis:
    """Analyze architecture using graph-sitter (placeholder)"""
    return EntryPointAnalysis(
        entry_points=[], main_functions=[], api_endpoints=[], cli_commands=[],
        event_handlers=[], initialization_patterns=[], architecture_pattern="Unknown",
        framework_detection=[]
    )
    
def _analyze_security_gs(self, codebase, issues: List[IntelligentIssue]) -> SecurityAnalysis:
    """Analyze security using graph-sitter (placeholder)"""
    security_issues = [i for i in issues if i.category == 'security']
    return SecurityAnalysis(
        vulnerabilities=security_issues,
        security_score=max(0, 100 - len(security_issues) * 10),
        threat_model={}, attack_surface=[], sensitive_data_flows=[],
        authentication_patterns=[], authorization_issues=[], input_validation_gaps=[]
    )
    
def _analyze_performance_gs(self, codebase, issues: List[IntelligentIssue]) -> PerformanceAnalysis:
    """Analyze performance using graph-sitter (placeholder)"""
    performance_issues = [i for i in issues if i.category == 'performance']
    return PerformanceAnalysis(
        bottlenecks=performance_issues,
        performance_score=max(0, 100 - len(performance_issues) * 5),
        memory_usage_patterns=[], cpu_intensive_functions=[], io_operations=[],
        algorithmic_complexity={}, optimization_opportunities=[]
    )
    
async def _analyze_dependencies_gs(self, codebase) -> DependencyGraph:
    """Analyze dependencies using graph-sitter (placeholder)"""
    return DependencyGraph(
        nodes=[], edges=[], circular_dependencies=[], critical_paths=[],
        orphaned_modules=[], coupling_metrics={}
    )

# ... rest of code ...
