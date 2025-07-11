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
            "/analyze": "Analyze repository",
            "/analyze-comprehensive": "Comprehensive analysis with health metrics"
        },
        features=[
            "🔍 Comprehensive Issue Detection (30+ types)",
            "🤖 Automated Resolution Generation",
            "📊 Quality Metrics & Health Scoring",
            "🕸️ Call Graph Analysis",
            "📈 Technical Debt Calculation",
            "🎯 Entry Point Detection",
            "💀 Dead Code Identification",
            "📋 Health Dashboard"
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
async def analyze_repository(request: RepoRequest):
    """Basic repository analysis"""
    start_time = datetime.now()
    
    try:
        # Clone repository
        repo_path = clone_repo(request.repo_url)
        
        # Initialize graph-sitter codebase
        codebase = Codebase.from_directory(repo_path)
        
        # Get basic summary
        summary = get_codebase_summary(codebase)
        description = get_repo_description(request.repo_url)
        
        # Basic analysis results
        analysis_results = {
            "summary": summary,
            "description": description,
            "total_files": len(codebase.source_files),
            "total_functions": len(list(codebase.functions)),
            "total_classes": len(list(codebase.classes)),
            "repository_structure": {
                "files": [f.file_path for f in codebase.source_files[:10]],  # First 10 files
                "languages": list(set(f.language for f in codebase.source_files if f.language))
            }
        }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return CodebaseAnalysisResponse(
            success=True,
            analysis_results=analysis_results,
            processing_time=processing_time,
            repo_url=request.repo_url,
            analysis_timestamp=datetime.now().isoformat(),
            features_analyzed=["basic_metrics", "repository_structure"]
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


@fastapi_app.post("/analyze-comprehensive", response_model=CodebaseAnalysisResponse)
async def analyze_comprehensive(request: CodebaseAnalysisRequest):
    """Comprehensive codebase analysis with all features"""
    start_time = datetime.now()
    
    try:
        # Clone repository
        repo_path = clone_repo(request.repo_url)
        
        # Initialize graph-sitter codebase
        codebase = Codebase.from_directory(repo_path)
        
        # Initialize analyzer
        analyzer = CodebaseAnalyzer()
        
        # Perform comprehensive analysis
        results = analyzer.analyze_codebase(codebase)
        
        # Convert results to dictionary for JSON serialization
        analysis_results = {
            "basic_metrics": {
                "total_files": results.total_files,
                "total_functions": results.total_functions,
                "total_classes": results.total_classes,
                "total_lines_of_code": results.total_lines_of_code
            },
            "issues_analysis": {
                "total_issues": len(results.issues),
                "issues_by_severity": results.issues_by_severity,
                "issues_by_type": results.issues_by_type,
                "automated_resolutions_count": len(results.automated_resolutions)
            },
            "function_analysis": {
                "total_functions": len(results.function_contexts),
                "entry_points": results.entry_points,
                "dead_functions": results.dead_functions,
                "most_important_functions": results.most_important_functions[:5]  # Top 5
            },
            "quality_metrics": {
                "halstead_metrics": results.halstead_metrics,
                "complexity_metrics": results.complexity_metrics,
                "maintainability_metrics": results.maintainability_metrics
            },
            "call_graph_metrics": results.call_graph_metrics,
            "health_assessment": {
                "health_score": results.health_score,
                "health_grade": results.health_grade,
                "risk_level": results.risk_level,
                "technical_debt_hours": results.technical_debt_hours
            },
            "repository_description": get_repo_description(request.repo_url)
        }
        
        # Create health dashboard if requested
        health_dashboard = None
        if request.include_health_metrics:
            health_dashboard = create_health_dashboard(results)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        features_analyzed = [
            "comprehensive_analysis",
            "issue_detection",
            "automated_resolutions",
            "quality_metrics",
            "call_graph_analysis",
            "health_assessment"
        ]
        
        if request.include_health_metrics:
            features_analyzed.append("health_dashboard")
        
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
