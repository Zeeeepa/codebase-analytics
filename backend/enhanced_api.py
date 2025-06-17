#!/usr/bin/env python3
"""
Enhanced API Module

This module extends the existing API with comprehensive analysis features
inspired by tree-sitter and graph-based code analysis capabilities.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Tuple, Any, Optional
import requests
from datetime import datetime, timedelta
import subprocess
import os
import tempfile
from fastapi.middleware.cors import CORSMiddleware
import modal
from pathlib import Path

# Import from Codegen SDK
from codegen.sdk.core.codebase import Codebase

# Import existing analysis modules
from analysis import (
    InheritanceAnalysis,
    RecursionAnalysis,
    calculate_cyclomatic_complexity,
    calculate_doi,
    get_operators_and_operands,
    calculate_halstead_volume,
    count_lines,
    calculate_maintainability_index,
    analyze_inheritance_patterns,
    analyze_recursive_functions,
    analyze_file_issues,
    build_repo_structure,
)

# Import new advanced analysis
from advanced_analysis import (
    DependencyAnalysis,
    CallGraphAnalysis,
    CodeQualityMetrics,
    ArchitecturalInsights,
    SecurityAnalysis,
    PerformanceAnalysis,
    AnalysisType,
    perform_comprehensive_analysis
)

from visualize import (
    VisualizationType,
    VisualizationConfig,
    create_call_graph,
    create_dependency_graph,
    create_class_hierarchy,
    create_complexity_heatmap,
    create_blast_radius,
)

# Enhanced Data Models

class EnhancedAnalysisResponse(BaseModel):
    """Enhanced analysis response with comprehensive metrics."""
    # Basic stats (existing)
    repo_url: str
    description: str
    num_files: int
    num_functions: int
    num_classes: int
    
    # Line metrics (existing)
    line_metrics: Dict[str, Dict[str, float]]
    
    # Complexity metrics (existing)
    cyclomatic_complexity: Dict[str, float]
    depth_of_inheritance: Dict[str, float]
    halstead_metrics: Dict[str, int]
    maintainability_index: Dict[str, int]
    
    # Git metrics (existing)
    monthly_commits: Dict[str, int]
    
    # Existing analysis features
    inheritance_analysis: InheritanceAnalysis
    recursion_analysis: RecursionAnalysis
    repo_structure: Dict[str, Any]
    
    # New comprehensive analysis features
    dependency_analysis: Optional[DependencyAnalysis] = None
    call_graph_analysis: Optional[CallGraphAnalysis] = None
    code_quality_metrics: Optional[CodeQualityMetrics] = None
    architectural_insights: Optional[ArchitecturalInsights] = None
    security_analysis: Optional[SecurityAnalysis] = None
    performance_analysis: Optional[PerformanceAnalysis] = None
    
    # Analysis metadata
    analysis_timestamp: str
    analysis_duration_seconds: float
    analysis_types_performed: List[str]

class AnalysisRequest(BaseModel):
    """Request model for enhanced analysis."""
    repo_url: str
    analysis_types: Optional[List[str]] = None  # If None, perform all analyses
    include_visualizations: bool = False
    max_analysis_time: int = 300  # Maximum analysis time in seconds

class ComprehensiveInsights(BaseModel):
    """Comprehensive insights derived from all analyses."""
    overall_quality_score: float
    technical_debt_level: str  # "Low", "Medium", "High", "Critical"
    maintainability_rating: str  # "Excellent", "Good", "Fair", "Poor"
    architectural_health: str
    security_risk_level: str
    performance_concerns: List[str]
    top_recommendations: List[str]
    complexity_distribution: Dict[str, int]
    dependency_health: str

# Enhanced API Functions

def get_github_repo_description(repo_url: str) -> str:
    """Get repository description from GitHub API."""
    try:
        api_url = f"https://api.github.com/repos/{repo_url}"
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('description', 'No description available')
        return 'Description not available'
    except Exception:
        return 'Description not available'

def get_monthly_commits(repo_url: str) -> Dict[str, int]:
    """Get monthly commit statistics from GitHub API."""
    try:
        api_url = f"https://api.github.com/repos/{repo_url}/stats/commit_activity"
        response = requests.get(api_url, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                monthly_commits = {}
                for week_data in data[-12:]:  # Last 12 weeks
                    week_timestamp = week_data.get('week', 0)
                    commits = week_data.get('total', 0)
                    
                    date = datetime.fromtimestamp(week_timestamp)
                    month_key = date.strftime('%Y-%m')
                    
                    if month_key not in monthly_commits:
                        monthly_commits[month_key] = 0
                    monthly_commits[month_key] += commits
                
                return monthly_commits
        
        # Fallback data
        return {
            datetime.now().strftime('%Y-%m'): 10,
            (datetime.now() - timedelta(days=30)).strftime('%Y-%m'): 15,
            (datetime.now() - timedelta(days=60)).strftime('%Y-%m'): 8
        }
    except Exception:
        return {'2024-01': 5, '2024-02': 12, '2024-03': 8}

def calculate_comprehensive_insights(analysis_results: Dict[str, Any]) -> ComprehensiveInsights:
    """Calculate comprehensive insights from all analysis results."""
    insights = ComprehensiveInsights(
        overall_quality_score=0.0,
        technical_debt_level="Unknown",
        maintainability_rating="Unknown",
        architectural_health="Unknown",
        security_risk_level="Unknown",
        performance_concerns=[],
        top_recommendations=[],
        complexity_distribution={},
        dependency_health="Unknown"
    )
    
    scores = []
    recommendations = []
    
    # Analyze code quality metrics
    if 'code_quality_metrics' in analysis_results:
        quality = analysis_results['code_quality_metrics']
        
        # Technical debt assessment
        debt_ratio = quality.technical_debt_ratio
        if debt_ratio < 5:
            insights.technical_debt_level = "Low"
            scores.append(90)
        elif debt_ratio < 15:
            insights.technical_debt_level = "Medium"
            scores.append(70)
        elif debt_ratio < 30:
            insights.technical_debt_level = "High"
            scores.append(40)
        else:
            insights.technical_debt_level = "Critical"
            scores.append(20)
            recommendations.append("Address technical debt by resolving TODO/FIXME items")
        
        # Documentation coverage
        if quality.documentation_coverage < 50:
            recommendations.append("Improve code documentation coverage")
        
        # Code duplication
        if quality.code_duplication_percentage > 10:
            recommendations.append("Reduce code duplication through refactoring")
    
    # Analyze dependency health
    if 'dependency_analysis' in analysis_results:
        deps = analysis_results['dependency_analysis']
        
        if deps.circular_dependencies:
            insights.dependency_health = "Poor"
            recommendations.append("Resolve circular dependencies")
            scores.append(30)
        elif deps.dependency_depth > 10:
            insights.dependency_health = "Fair"
            scores.append(60)
        else:
            insights.dependency_health = "Good"
            scores.append(80)
    
    # Analyze security
    if 'security_analysis' in analysis_results:
        security = analysis_results['security_analysis']
        
        vuln_count = len(security.potential_vulnerabilities)
        if vuln_count == 0:
            insights.security_risk_level = "Low"
            scores.append(90)
        elif vuln_count < 3:
            insights.security_risk_level = "Medium"
            scores.append(60)
        else:
            insights.security_risk_level = "High"
            scores.append(30)
            recommendations.append("Address security vulnerabilities")
    
    # Analyze performance
    if 'performance_analysis' in analysis_results:
        perf = analysis_results['performance_analysis']
        
        insights.performance_concerns = [
            hotspot['description'] for hotspot in perf.performance_hotspots
        ]
        
        if len(perf.performance_hotspots) > 5:
            recommendations.append("Optimize performance hotspots")
    
    # Analyze architecture
    if 'architectural_insights' in analysis_results:
        arch = analysis_results['architectural_insights']
        
        if arch.modularity_score > 70:
            insights.architectural_health = "Good"
            scores.append(80)
        elif arch.modularity_score > 40:
            insights.architectural_health = "Fair"
            scores.append(60)
        else:
            insights.architectural_health = "Poor"
            scores.append(40)
            recommendations.append("Improve code modularity and organization")
    
    # Calculate overall quality score
    if scores:
        insights.overall_quality_score = sum(scores) / len(scores)
    
    # Determine maintainability rating
    if insights.overall_quality_score >= 80:
        insights.maintainability_rating = "Excellent"
    elif insights.overall_quality_score >= 65:
        insights.maintainability_rating = "Good"
    elif insights.overall_quality_score >= 45:
        insights.maintainability_rating = "Fair"
    else:
        insights.maintainability_rating = "Poor"
    
    insights.top_recommendations = recommendations[:5]  # Top 5 recommendations
    
    return insights

# Enhanced FastAPI Application

app = FastAPI(
    title="Enhanced Codebase Analytics API",
    description="Comprehensive codebase analysis with advanced metrics and insights",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze_comprehensive")
async def analyze_comprehensive(request: AnalysisRequest) -> EnhancedAnalysisResponse:
    """Comprehensive repository analysis with advanced features."""
    start_time = datetime.now()
    
    repo_url = request.repo_url
    
    # Validate repo URL format
    if not repo_url or '/' not in repo_url:
        raise HTTPException(status_code=400, detail="Repository URL must be in format 'owner/repo'")
    
    # Clean repo URL
    if repo_url.startswith('https://github.com/'):
        repo_url = repo_url.replace('https://github.com/', '')
    if repo_url.endswith('.git'):
        repo_url = repo_url[:-4]
    
    parts = repo_url.split('/')
    if len(parts) < 2:
        raise HTTPException(status_code=400, detail="Repository URL must be in format 'owner/repo'")
    
    repo_url = f"{parts[0]}/{parts[1]}"
    
    try:
        codebase = Codebase.from_repo(repo_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to analyze repository: {str(e)}")

    # Basic analysis (existing functionality)
    num_files = len(codebase.files(extensions="*"))
    num_functions = len(codebase.functions)
    num_classes = len(codebase.classes)

    total_loc = total_lloc = total_sloc = total_comments = 0
    total_complexity = 0
    total_volume = 0
    total_mi = 0
    total_doi = 0

    monthly_commits = get_monthly_commits(repo_url)

    # Analyze files and collect symbols
    file_issues = {}
    file_symbols = {}
    
    for file in codebase.files:
        # Line metrics
        loc, lloc, sloc, comments = count_lines(file.source)
        total_loc += loc
        total_lloc += lloc
        total_sloc += sloc
        total_comments += comments

        # Analyze issues
        issues = analyze_file_issues(file)
        if any(len(v) > 0 for v in issues.values()):
            file_issues[file.filepath] = issues

        # Collect symbols (simplified for this example)
        symbols = []
        for func in file.functions:
            symbols.append({
                'id': str(hash(func.name + file.filepath)),
                'name': func.name,
                'type': 'function',
                'filepath': file.filepath,
                'start_line': func.start_point[0] if hasattr(func, 'start_point') else 0,
                'end_line': func.end_point[0] if hasattr(func, 'end_point') else 0
            })
        
        if symbols:
            file_symbols[file.filepath] = symbols

    # Build repository structure
    repo_structure = build_repo_structure(codebase.files, file_issues, file_symbols)

    # Calculate metrics
    callables = codebase.functions + [m for c in codebase.classes for m in c.methods]
    num_callables = 0
    
    for func in callables:
        if not hasattr(func, "code_block"):
            continue

        complexity = calculate_cyclomatic_complexity(func)
        operators, operands = get_operators_and_operands(func)
        volume, N1, N2, n1, n2 = calculate_halstead_volume(operators, operands)
        loc = len(func.code_block.source.splitlines())
        mi_score = calculate_maintainability_index(volume, complexity, loc)

        total_complexity += complexity
        total_volume += volume
        total_mi += mi_score
        num_callables += 1

    for cls in codebase.classes:
        doi = calculate_doi(cls)
        total_doi += doi

    desc = get_github_repo_description(repo_url)
    
    # Perform existing analysis features
    inheritance_analysis = analyze_inheritance_patterns(codebase)
    recursion_analysis = analyze_recursive_functions(codebase)

    # Determine which analysis types to perform
    analysis_types_to_perform = []
    if request.analysis_types:
        for analysis_type in request.analysis_types:
            try:
                analysis_types_to_perform.append(AnalysisType(analysis_type))
            except ValueError:
                pass  # Skip invalid analysis types
    else:
        analysis_types_to_perform = list(AnalysisType)  # Perform all analyses

    # Perform comprehensive analysis
    comprehensive_results = perform_comprehensive_analysis(codebase, analysis_types_to_perform)
    
    # Calculate analysis duration
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Build response
    response = EnhancedAnalysisResponse(
        repo_url=repo_url,
        description=desc,
        num_files=num_files,
        num_functions=num_functions,
        num_classes=num_classes,
        line_metrics={
            "total": {
                "loc": total_loc,
                "lloc": total_lloc,
                "sloc": total_sloc,
                "comments": total_comments,
                "comment_density": (total_comments / total_loc * 100) if total_loc > 0 else 0,
            },
        },
        cyclomatic_complexity={
            "average": total_complexity / num_callables if num_callables > 0 else 0,
        },
        depth_of_inheritance={
            "average": total_doi / len(codebase.classes) if codebase.classes else 0,
        },
        halstead_metrics={
            "total_volume": int(total_volume),
            "average_volume": int(total_volume / num_callables) if num_callables > 0 else 0,
        },
        maintainability_index={
            "average": int(total_mi / num_callables) if num_callables > 0 else 0,
        },
        monthly_commits=monthly_commits,
        inheritance_analysis=inheritance_analysis,
        recursion_analysis=recursion_analysis,
        repo_structure=repo_structure,
        analysis_timestamp=start_time.isoformat(),
        analysis_duration_seconds=duration,
        analysis_types_performed=[at.value for at in analysis_types_to_perform]
    )
    
    # Add comprehensive analysis results
    if 'dependency_analysis' in comprehensive_results:
        response.dependency_analysis = comprehensive_results['dependency_analysis']
    
    if 'call_graph_analysis' in comprehensive_results:
        response.call_graph_analysis = comprehensive_results['call_graph_analysis']
    
    if 'code_quality_metrics' in comprehensive_results:
        response.code_quality_metrics = comprehensive_results['code_quality_metrics']
    
    if 'architectural_insights' in comprehensive_results:
        response.architectural_insights = comprehensive_results['architectural_insights']
    
    if 'security_analysis' in comprehensive_results:
        response.security_analysis = comprehensive_results['security_analysis']
    
    if 'performance_analysis' in comprehensive_results:
        response.performance_analysis = comprehensive_results['performance_analysis']

    return response

@app.post("/insights")
async def get_comprehensive_insights(request: AnalysisRequest) -> ComprehensiveInsights:
    """Get high-level insights and recommendations."""
    # First perform comprehensive analysis
    analysis_response = await analyze_comprehensive(request)
    
    # Extract analysis results
    analysis_results = {}
    
    if analysis_response.dependency_analysis:
        analysis_results['dependency_analysis'] = analysis_response.dependency_analysis
    
    if analysis_response.call_graph_analysis:
        analysis_results['call_graph_analysis'] = analysis_response.call_graph_analysis
    
    if analysis_response.code_quality_metrics:
        analysis_results['code_quality_metrics'] = analysis_response.code_quality_metrics
    
    if analysis_response.architectural_insights:
        analysis_results['architectural_insights'] = analysis_response.architectural_insights
    
    if analysis_response.security_analysis:
        analysis_results['security_analysis'] = analysis_response.security_analysis
    
    if analysis_response.performance_analysis:
        analysis_results['performance_analysis'] = analysis_response.performance_analysis
    
    # Calculate comprehensive insights
    return calculate_comprehensive_insights(analysis_results)

@app.get("/analysis_types")
async def get_available_analysis_types() -> List[Dict[str, str]]:
    """Get available analysis types."""
    return [
        {
            "type": analysis_type.value,
            "description": get_analysis_type_description(analysis_type)
        }
        for analysis_type in AnalysisType
    ]

def get_analysis_type_description(analysis_type: AnalysisType) -> str:
    """Get description for analysis type."""
    descriptions = {
        AnalysisType.DEPENDENCY: "Analyze dependencies, circular dependencies, and dependency depth",
        AnalysisType.CALL_GRAPH: "Analyze function call relationships and call chains",
        AnalysisType.CODE_QUALITY: "Analyze code quality metrics, duplication, and technical debt",
        AnalysisType.ARCHITECTURAL: "Analyze architectural patterns, coupling, and modularity",
        AnalysisType.SECURITY: "Analyze potential security vulnerabilities and risks",
        AnalysisType.PERFORMANCE: "Analyze performance hotspots and optimization opportunities"
    }
    return descriptions.get(analysis_type, "Unknown analysis type")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "2.0.0"}

# Modal deployment
image = modal.Image.debian_slim().pip_install([
    "fastapi",
    "uvicorn",
    "requests",
    "pydantic",
    "networkx",
    "codegen"
])

@app.function(image=image)
@modal.asgi_app()
def modal_app():
    return app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Different port to avoid conflicts

