"""
Codebase Analytics - FastAPI Interface Layer
Clean API interface for comprehensive codebase analysis
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Any
import requests
import subprocess
import os
import tempfile
from datetime import datetime
import modal
import graph_sitter
from graph_sitter.core.codebase import Codebase

from .models import (
    RepoRequest, CodebaseAnalysisRequest, CodebaseAnalysisResponse,
    HealthCheckResponse, RootResponse
)
from .analysis import CodebaseAnalyzer, get_codebase_summary, create_health_dashboard


# ============================================================================
# MODAL SETUP AND CONFIGURATION
# ============================================================================

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "codegen", "fastapi", "uvicorn", "gitpython", "requests", "pydantic", "datetime"
    )
)

app = modal.App(name="analytics-app", image=image)

fastapi_app = FastAPI(
    title="Codebase Analytics API",
    description="Comprehensive codebase analysis with issue detection and automated resolutions",
    version="2.0.0"
)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clone_repo(repo_url: str) -> str:
    """Clone repository to temporary directory"""
    temp_dir = tempfile.mkdtemp()
    try:
        subprocess.run(
            ["git", "clone", repo_url, temp_dir],
            check=True,
            capture_output=True,
            text=True
        )
        return temp_dir
    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to clone repository: {e.stderr}"
        )


def get_repo_description(repo_url: str) -> str:
    """Get repository description from GitHub API"""
    try:
        if "github.com" in repo_url:
            # Extract owner/repo from URL
            parts = repo_url.rstrip('/').split('/')
            if len(parts) >= 2:
                owner, repo = parts[-2], parts[-1]
                if repo.endswith('.git'):
                    repo = repo[:-4]
                
                api_url = f"https://api.github.com/repos/{owner}/{repo}"
                response = requests.get(api_url, timeout=10)
                
                if response.status_code == 200:
                    repo_data = response.json()
                    return repo_data.get("description", "No description available")
        
        return "Repository description not available"
    except Exception:
        return "Repository description not available"


# ============================================================================
# API ENDPOINTS
# ============================================================================

@fastapi_app.get("/", response_model=RootResponse)
async def root():
    """Root endpoint with API information"""
    return RootResponse(
        message="🔍 Codebase Analytics API",
        description="Comprehensive codebase analysis with issue detection and automated resolutions",
        endpoints={
            "/": "API information",
            "/health": "Health check",
            "/analyze": "Single comprehensive analysis endpoint"
        },
        features=[
            "🔍 Advanced Issue Detection (30+ types with automated resolutions)",
            "🤖 Intelligent Import Resolution & Code Fixes",
            "📊 Comprehensive Quality Metrics & Health Scoring",
            "🕸️ Advanced Call Graph & Dependency Analysis",
            "📈 Technical Debt Calculation & Risk Assessment",
            "🎯 Entry Point Detection & Function Importance Scoring",
            "💀 Dead Code Detection with Blast Radius Analysis",
            "📋 Interactive Health Dashboard with Recommendations",
            "🌳 Repository Structure Analysis with Issue Mapping",
            "📈 Halstead Complexity Metrics & Maintainability Index",
            "🔄 Function Context Analysis with Call Chain Mapping"
        ]
    )


@fastapi_app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now().isoformat()
    )


@fastapi_app.post("/analyze", response_model=CodebaseAnalysisResponse)
async def analyze_repository(request: CodebaseAnalysisRequest):
    """
    🎯 **SINGLE COMPREHENSIVE CODEBASE ANALYSIS ENDPOINT**
    
    Performs complete codebase analysis with all advanced features:
    
    ✅ **Advanced Issue Detection** - 30+ issue types with automated resolutions
    ✅ **Intelligent Import Resolution** - Automatic import fixes and optimizations  
    ✅ **Function Context Analysis** - Dependencies, call chains, importance scoring
    ✅ **Halstead Complexity Metrics** - Quantitative complexity measurements
    ✅ **Advanced Graph Analysis** - Call graphs and dependency analysis
    ✅ **Dead Code Detection** - With blast radius calculation and safe removal
    ✅ **Health Assessment** - Overall health scoring and risk assessment
    ✅ **Repository Structure** - Interactive tree with issue indicators
    ✅ **Technical Debt Analysis** - Quantified debt with resolution estimates
    ✅ **Automated Resolutions** - High-confidence fixes for detected issues
    
    Perfect for:
    - CI/CD integration and quality gates
    - Health dashboards and monitoring
    - Technical debt assessment and planning
    - Automated code quality enforcement
    """
    start_time = datetime.now()
    
    try:
        # Clone repository
        repo_path = clone_repo(request.repo_url)
        
        # Initialize graph-sitter codebase
        codebase = Codebase.from_directory(repo_path)
        
        # Initialize comprehensive analyzer
        analyzer = CodebaseAnalyzer()
        
        # Perform comprehensive analysis with all features
        results = analyzer.analyze_codebase(codebase)
        
        # Get repository description
        repo_description = get_repo_description(request.repo_url)
        
        # Build comprehensive analysis results
        analysis_results = {
            "repository_overview": {
                "description": repo_description,
                "summary": get_codebase_summary(codebase),
                "total_files": results.total_files,
                "total_functions": results.total_functions,
                "total_classes": results.total_classes,
                "total_lines_of_code": results.total_lines_of_code,
                "languages": list(set(f.language for f in codebase.source_files if f.language))
            },
            "issues_analysis": {
                "total_issues": len(results.issues),
                "issues_by_severity": results.issues_by_severity,
                "issues_by_type": results.issues_by_type,
                "critical_issues": [
                    {
                        "type": issue.issue_type.value,
                        "message": issue.message,
                        "filepath": issue.filepath,
                        "line_number": issue.line_number,
                        "function_name": issue.function_name,
                        "has_automated_fix": issue.automated_resolution is not None
                    }
                    for issue in results.issues 
                    if issue.severity.value == "critical"
                ][:10],  # Top 10 critical issues
                "automated_resolutions": {
                    "total_available": len(results.automated_resolutions),
                    "high_confidence": len([r for r in results.automated_resolutions if r.confidence > 0.8]),
                    "safe_to_apply": len([r for r in results.automated_resolutions if r.is_safe]),
                    "resolutions": [
                        {
                            "type": res.resolution_type,
                            "description": res.description,
                            "confidence": res.confidence,
                            "file_path": res.file_path,
                            "line_number": res.line_number,
                            "is_safe": res.is_safe
                        }
                        for res in results.automated_resolutions[:20]  # Top 20 resolutions
                    ]
                }
            },
            "function_analysis": {
                "total_functions": len(results.function_contexts),
                "entry_points": results.entry_points,
                "dead_functions": results.dead_functions,
                "most_important_functions": results.most_important_functions[:10],
                "function_contexts": {
                    name: {
                        "filepath": context.filepath,
                        "line_start": context.line_start,
                        "line_end": context.line_end,
                        "complexity_metrics": context.complexity_metrics,
                        "fan_in": context.fan_in,
                        "fan_out": context.fan_out,
                        "is_entry_point": context.is_entry_point,
                        "function_calls": context.function_calls[:10]  # Limit for response size
                    }
                    for name, context in list(results.function_contexts.items())[:50]  # Top 50 functions
                }
            },
            "quality_metrics": {
                "halstead_metrics": results.halstead_metrics,
                "complexity_metrics": results.complexity_metrics,
                "maintainability_metrics": results.maintainability_metrics,
                "call_graph_metrics": results.call_graph_metrics
            },
            "health_assessment": {
                "health_score": results.health_score,
                "health_grade": results.health_grade,
                "risk_level": results.risk_level,
                "technical_debt_hours": results.technical_debt_hours,
                "maintainability_index": getattr(results, 'maintainability_index', 0)
            },
            "repository_structure": {
                "files_by_type": {},  # Will be populated by analyzer
                "directory_structure": [f.file_path for f in codebase.source_files[:100]],  # First 100 files
                "issue_hotspots": []  # Files with most issues
            }
        }
        
        # Add issue hotspots
        file_issue_counts = {}
        for issue in results.issues:
            file_issue_counts[issue.filepath] = file_issue_counts.get(issue.filepath, 0) + 1
        
        analysis_results["repository_structure"]["issue_hotspots"] = [
            {"filepath": filepath, "issue_count": count}
            for filepath, count in sorted(file_issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        # Create comprehensive health dashboard
        health_dashboard = create_health_dashboard(results)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # All features are always analyzed in this comprehensive endpoint
        features_analyzed = [
            "advanced_issue_detection",
            "automated_resolution_generation", 
            "import_resolution_analysis",
            "function_context_analysis",
            "halstead_complexity_metrics",
            "call_graph_analysis",
            "dependency_analysis",
            "dead_code_detection",
            "health_assessment",
            "technical_debt_calculation",
            "repository_structure_analysis",
            "maintainability_scoring",
            "entry_point_detection",
            "blast_radius_analysis",
            "health_dashboard_generation"
        ]
        
        return CodebaseAnalysisResponse(
            success=True,
            analysis_results=analysis_results,
            health_dashboard=health_dashboard,
            processing_time=processing_time,
            repo_url=request.repo_url,
            analysis_timestamp=datetime.now().isoformat(),
            features_analyzed=features_analyzed
        )
        
    except Exception as e:
        return CodebaseAnalysisResponse(
            success=False,
            analysis_results={},
            processing_time=(datetime.now() - start_time).total_seconds(),
            repo_url=request.repo_url,
            analysis_timestamp=datetime.now().isoformat(),
            features_analyzed=[],
            error_message=str(e)
        )
    
    finally:
        # Cleanup temporary directory
        if 'repo_path' in locals():
            subprocess.run(["rm", "-rf", repo_path], check=False)


# ============================================================================
# MODAL DEPLOYMENT
# ============================================================================

@app.function(image=image)
@modal.asgi_app()
def fastapi_app_modal():
    """Modal deployment function"""
    return fastapi_app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)
