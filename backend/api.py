"""
Codebase Analytics - FastAPI Interface Layer
Clean API interface for comprehensive codebase analysis
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Any
import requests
from datetime import datetime
import modal

from .models import (
    RepoRequest, CodebaseAnalysisRequest, CodebaseAnalysisResponse,
    HealthCheckResponse, RootResponse
)
from .graph_sitter_analyzer import GraphSitterAnalyzer


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

# Repository cloning is now handled by graph-sitter analyzer


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
    🎯 **COMPREHENSIVE CODEBASE ANALYSIS ENDPOINT**
    
    Performs complete codebase analysis using graph-sitter's advanced capabilities:
    
    ✅ **Actual Runtime Error Detection** - Null references, undefined variables, type mismatches
    ✅ **Entry Point Detection** - Critical functions and application entry points
    ✅ **Most Important Files** - Usage-based importance scoring with relationship analysis
    ✅ **Repository Structure** - Complete tree with embedded error contexts
    ✅ **Function Analysis** - Dependencies, usages, call chains using graph-sitter
    ✅ **Auto-fix Detection** - Identifies which errors can be automatically resolved
    
    Perfect for:
    - Understanding new codebases quickly
    - Identifying critical runtime issues
    - Finding entry points and important functions
    - Error resolution and code quality improvement
    """
    start_time = datetime.now()
    
    try:
        # Initialize enhanced analyzer
        analyzer = GraphSitterAnalyzer()
        
        # Perform comprehensive analysis
        analysis_results = analyzer.analyze_repository(request.repo_url)
        
        # Check if analysis was successful
        if "error" in analysis_results:
            return CodebaseAnalysisResponse(
                success=False,
                analysis_results={},
                processing_time=(datetime.now() - start_time).total_seconds(),
                repo_url=request.repo_url,
                analysis_timestamp=datetime.now().isoformat(),
                features_analyzed=[],
                error_message=analysis_results["error"]
            )
        
        # Get repository description
        repo_description = get_repo_description(request.repo_url)
        analysis_results["repository_facts"]["description"] = repo_description
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Features analyzed by the enhanced system
        features_analyzed = [
            "graph_sitter_relationship_analysis",
            "actual_runtime_error_detection",
            "entry_point_detection",
            "importance_scoring",
            "repository_structure_analysis",
            "function_dependency_analysis",
            "usage_pattern_analysis",
            "auto_fix_detection",
            "call_chain_analysis",
            "import_resolution_analysis"
        ]
        
        return CodebaseAnalysisResponse(
            success=True,
            analysis_results=analysis_results,
            health_dashboard=None,  # Will be added later if needed
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
