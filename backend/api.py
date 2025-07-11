"""
Comprehensive Codebase Analytics API - Single Endpoint Architecture
This module provides a unified codebase analysis endpoint that consolidates
all analysis capabilities into one powerful interface.
"""

import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import modal
from graph_sitter.core.codebase import Codebase

# Import analysis functions and models
from .analysis import (
    detect_entrypoints,
    analyze_code_issues,
    identify_critical_files,
    build_dependency_graph
)
from .metrics import (
    calculate_comprehensive_metrics,
    get_most_important_functions
)
from .comprehensive_analyzer import (
    ComprehensiveCodebaseAnalyzer,
    analyze_codebase
)
from .models import (
    CodebaseAnalysisRequest,
    CodebaseAnalysisResponse,
    CodebaseAnalysis,
    AnalysisSummary,
    IssueSeverity,
    IssueType
)

# Modal setup for deployment
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "codegen", "fastapi", "uvicorn", "gitpython", "requests", "pydantic", "datetime", "networkx"
    )
)

app = modal.App(name="analytics-app", image=image)
fastapi_app = FastAPI(
    title="Comprehensive Codebase Analytics API",
    description="Advanced codebase analysis with comprehensive issue detection and insights",
    version="2.0.0"
)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def create_analysis_summary(
    issues: List[Any],
    entry_points: List[Any],
    critical_files: List[Any],
    function_metrics: List[Any],
    class_metrics: List[Any],
    file_metrics: List[Any]
) -> AnalysisSummary:
    """Create a comprehensive analysis summary."""
    
    # Count issues by severity and type
    issues_by_severity = {}
    issues_by_type = {}
    
    for issue in issues:
        severity = issue.severity
        issue_type = issue.type
        
        issues_by_severity[severity] = issues_by_severity.get(severity, 0) + 1
        issues_by_type[issue_type] = issues_by_type.get(issue_type, 0) + 1
    
    critical_issues_count = issues_by_severity.get(IssueSeverity.CRITICAL, 0)
    high_priority_issues_count = (
        issues_by_severity.get(IssueSeverity.CRITICAL, 0) + 
        issues_by_severity.get(IssueSeverity.HIGH, 0)
    )
    
    # Calculate averages
    total_complexity = sum(getattr(f, 'cyclomatic_complexity', 0) for f in function_metrics)
    avg_complexity = total_complexity / len(function_metrics) if function_metrics else 0
    
    total_maintainability = sum(getattr(f, 'maintainability_index', 0) for f in function_metrics)
    avg_maintainability = total_maintainability / len(function_metrics) if function_metrics else 0
    
    total_loc = sum(getattr(f, 'lines_of_code', 0) for f in file_metrics)
    
    # Count files with issues
    files_with_issues = len(set(issue.file_path for issue in issues))
    
    return AnalysisSummary(
        total_issues=len(issues),
        issues_by_severity=issues_by_severity,
        issues_by_type=issues_by_type,
        files_with_issues=files_with_issues,
        critical_issues_count=critical_issues_count,
        high_priority_issues_count=high_priority_issues_count,
        total_files=len(file_metrics),
        total_functions=len(function_metrics),
        total_classes=len(class_metrics),
        total_lines_of_code=total_loc,
        average_complexity=avg_complexity,
        average_maintainability=avg_maintainability,
        entry_points_count=len(entry_points),
        critical_files_count=len(critical_files)
    )


@fastapi_app.post("/codebase_analysis", response_model=CodebaseAnalysisResponse)
async def comprehensive_codebase_analysis(request: CodebaseAnalysisRequest) -> CodebaseAnalysisResponse:
    """
    🎯 **SINGLE COMPREHENSIVE CODEBASE ANALYSIS ENDPOINT**
    
    This unified endpoint provides complete codebase analysis including:
    
    ✅ **Comprehensive Issue Detection** - All 169+ issues with full context
    ✅ **Critical File Identification** - Most important files with importance scoring  
    ✅ **Entry Point Detection** - All entry points with confidence scoring
    ✅ **Function Analysis** - Most important functions using Halstead metrics
    ✅ **Dependency Graph Analysis** - Complete architectural insights
    ✅ **Structured Storage Format** - Ready for CI/CD integration
    
    **Example Response Format:**
    ```json
    {
        "success": true,
        "analysis": {
            "summary": {
                "total_issues": 182,
                "critical_issues_count": 11,
                "files_with_issues": 23
            },
            "issues": [
                {
                    "id": "complexity_a1b2c3d4",
                    "type": "complexity_issue", 
                    "severity": "critical",
                    "file_path": "/path/to/file.py",
                    "line_number": 45,
                    "function_name": "complex_function",
                    "message": "High cyclomatic complexity: 28",
                    "interconnected_context": {
                        "dependencies": [...],
                        "dependents": [...],
                        "call_graph": {...}
                    },
                    "fix_suggestions": [...]
                }
            ],
            "critical_files": [...],
            "entry_points": [...],
            "function_metrics": [...]
        }
    }
    ```
    """
    
    start_time = time.time()
    
    try:
        # Load codebase using graph-sitter
        codebase = Codebase.from_repo(request.repo_url)
        
        # Initialize results containers
        issues = []
        entry_points = []
        critical_files = []
        function_metrics = []
        class_metrics = []
        file_metrics = []
        dependency_graph = None
        errors = []
        
        # 1. 🔍 COMPREHENSIVE ISSUE DETECTION
        if request.include_issues:
            try:
                issues = analyze_code_issues(codebase, max_issues=request.max_issues)
                
                # Filter by severity if specified
                if request.severity_filter:
                    issues = [issue for issue in issues if issue.severity in request.severity_filter]
                    
            except Exception as e:
                errors.append(f"Issue analysis failed: {str(e)}")
        
        # 2. 🎯 ENTRY POINT DETECTION  
        if request.include_entry_points:
            try:
                entry_points = detect_entrypoints(codebase)
            except Exception as e:
                errors.append(f"Entry point detection failed: {str(e)}")
        
        # 3. 📊 CRITICAL FILE IDENTIFICATION
        if request.include_critical_files:
            try:
                critical_files = identify_critical_files(codebase)
            except Exception as e:
                errors.append(f"Critical file analysis failed: {str(e)}")
        
        # 4. 📈 COMPREHENSIVE METRICS CALCULATION
        if request.include_metrics:
            try:
                metrics_result = calculate_comprehensive_metrics(codebase)
                function_metrics = metrics_result.get('function_metrics', [])
                class_metrics = metrics_result.get('class_metrics', [])
                file_metrics = metrics_result.get('file_metrics', [])
            except Exception as e:
                errors.append(f"Metrics calculation failed: {str(e)}")
        
        # 5. 🕸️ DEPENDENCY GRAPH ANALYSIS
        if request.include_dependency_graph:
            try:
                dependency_graph = build_dependency_graph(codebase)
            except Exception as e:
                errors.append(f"Dependency graph analysis failed: {str(e)}")
        
        # 6. 📋 CREATE COMPREHENSIVE ANALYSIS SUMMARY
        summary = create_analysis_summary(
            issues, entry_points, critical_files, 
            function_metrics, class_metrics, file_metrics
        )
        
        # 7. 🏗️ BUILD STRUCTURED ANALYSIS RESULT
        analysis_duration = time.time() - start_time
        
        # Group issues by file for better organization
        issues_by_file = {}
        for issue in issues:
            file_path = issue.file_path
            if file_path not in issues_by_file:
                issues_by_file[file_path] = []
            issues_by_file[file_path].append(issue)
        
        # Create the comprehensive analysis object
        analysis = CodebaseAnalysis(
            repo_url=request.repo_url,
            analysis_timestamp=datetime.now(),
            analysis_duration=analysis_duration,
            summary=summary,
            issues=issues,
            issues_by_file=issues_by_file,
            function_metrics=function_metrics,
            class_metrics=class_metrics,
            file_metrics=file_metrics,
            entry_points=entry_points,
            critical_files=critical_files,
            dependency_graph=dependency_graph,
            repository_info={
                "total_files": len(list(codebase.files)),
                "total_functions": len(list(codebase.functions)),
                "total_classes": len(list(codebase.classes)),
                "analysis_timestamp": datetime.now().isoformat()
            },
            analysis_config={
                "include_issues": request.include_issues,
                "include_entry_points": request.include_entry_points,
                "include_critical_files": request.include_critical_files,
                "include_metrics": request.include_metrics,
                "include_dependency_graph": request.include_dependency_graph,
                "max_issues": request.max_issues,
                "severity_filter": request.severity_filter,
                "file_extensions": request.file_extensions
            },
            errors=errors
        )
        
        return CodebaseAnalysisResponse(
            success=True,
            analysis=analysis,
            processing_time=analysis_duration
        )
        
    except Exception as e:
        return CodebaseAnalysisResponse(
            success=False,
            error_message=f"Codebase analysis failed: {str(e)}",
            processing_time=time.time() - start_time
        )


@fastapi_app.post("/comprehensive_analysis")
async def comprehensive_codebase_analysis(request: CodebaseAnalysisRequest) -> Dict[str, Any]:
    """
    🎯 **COMPREHENSIVE CODEBASE ANALYSIS**
    
    Perform complete codebase analysis with structured data output:
    
    ✅ **Advanced Issue Detection** - 30+ issue types with automated resolutions
    ✅ **Function Context Analysis** - Dependencies, call chains, importance scoring
    ✅ **Halstead Complexity Metrics** - Quantitative complexity measurements
    ✅ **Graph Analysis** - Call graphs and dependency analysis
    ✅ **Dead Code Detection** - With blast radius calculation
    ✅ **Health Assessment** - Overall health scoring and risk assessment
    ✅ **Repository Structure** - Interactive tree with issue indicators
    ✅ **Automated Resolutions** - Import fixes, dead code removal, refactoring suggestions
    
    Perfect for:
    - CI/CD integration
    - Health dashboards
    - Technical debt assessment
    - Automated code quality monitoring
    """
    
    start_time = time.time()
    
    try:
        # Load codebase using graph-sitter
        codebase = Codebase.from_repo(request.repo_url)
        
        # Perform comprehensive analysis
        analyzer = ComprehensiveCodebaseAnalyzer(codebase)
        results = analyzer.analyze()
        
        # Get structured data for API consumption
        structured_data = analyzer.get_structured_data()
        
        # Get health dashboard data
        health_dashboard = analyzer.get_health_dashboard_data()
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "analysis_results": structured_data,
            "health_dashboard": health_dashboard,
            "processing_time": processing_time,
            "repo_url": request.repo_url,
            "analysis_timestamp": datetime.now().isoformat(),
            "features_analyzed": [
                "Advanced issue detection (30+ types)",
                "Function context analysis",
                "Halstead complexity metrics",
                "Call graph analysis",
                "Dependency graph analysis", 
                "Dead code detection with blast radius",
                "Repository structure visualization",
                "Health assessment and risk analysis",
                "Automated resolution suggestions"
            ]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error_message": f"Comprehensive analysis failed: {str(e)}",
            "processing_time": time.time() - start_time
        }


@fastapi_app.get("/health")
async def health_check():
    """Health check endpoint to verify API status."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "features": {
            "comprehensive_issue_detection": True,
            "critical_file_identification": True,
            "entry_point_detection": True,
            "dependency_graph_analysis": True,
            "structured_analysis_storage": True,
            "halstead_metrics": True,
            "interconnected_context": True
        }
    }


@fastapi_app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Comprehensive Codebase Analytics API",
        "version": "2.0.0",
        "description": "Advanced codebase analysis with comprehensive issue detection",
        "endpoints": {
            "main": "/codebase_analysis",
            "comprehensive": "/comprehensive_analysis",
            "health": "/health",
            "docs": "/docs"
        },
        "features": [
            "🔍 Comprehensive Issue Detection - All issues with full context",
            "📊 Critical File Identification - Importance scoring and metrics", 
            "🎯 Entry Point Detection - All entry points with confidence scoring",
            "📈 Advanced Function Analysis - Halstead metrics and complexity",
            "🕸️ Dependency Graph Analysis - Complete architectural insights",
            "🏗️ Structured Storage Format - Ready for CI/CD integration"
        ]
    }


# Modal App Deployment
@app.function(image=image)
@modal.asgi_app()
def fastapi_modal_app():
    return fastapi_app


if __name__ == "__main__":
    # This allows running the app locally for development without Modal
    # Example: uvicorn api:fastapi_app --reload
    pass
