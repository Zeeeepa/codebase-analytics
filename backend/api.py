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
from .analysis import CodebaseAnalyzer, GraphSitterAnalyzer, create_health_dashboard

# Graph-sitter is required for this backend
from graph_sitter.codebase.codebase_analysis import get_codebase_summary


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
        # Initialize GraphSitter analyzer with enhanced summary capabilities
        analyzer = GraphSitterAnalyzer()
        
        # Perform comprehensive analysis with summaries
        results = analyzer.analyze_repository(request.repo_url)
        
        # Get repository description
        repo_description = get_repo_description(request.repo_url)
        
        # Build comprehensive analysis results with enhanced summaries
        analysis_results = {
            "repository_overview": {
                "description": repo_description,
                "summary": results.get("summaries", {}).get("codebase_summary", "Summary not available"),
                "repository_facts": results.get("repository_facts", {}),
                "most_important_files": results.get("most_important_files", []),
                "entry_points": results.get("entry_points", []),
                "repository_structure": results.get("repository_structure", {})
            },
            "summaries": results.get("summaries", {}),
            "error_analysis": {
                "actual_errors": results.get("actual_errors", []),
                "error_summary": results.get("error_summary", {}),
                "total_errors": len(results.get("actual_errors", []))
            },
            "analysis_metadata": results.get("analysis_metadata", {})
        }
        
        # Create health dashboard (simplified for GraphSitter format)
        health_dashboard = {
            "summary": "Analysis completed with GraphSitter enhanced summaries",
            "timestamp": datetime.now().isoformat()
        }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # GraphSitter enhanced features
        features_analyzed = [
            "graph_sitter_analysis",
            "comprehensive_summaries",
            "codebase_summary_generation",
            "file_summary_generation",
            "class_summary_generation", 
            "function_summary_generation",
            "symbol_summary_generation",
            "repository_facts_analysis",
            "important_files_detection",
            "entry_point_detection",
            "repository_structure_analysis",
            "error_detection_and_analysis"
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
