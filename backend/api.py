"""
Comprehensive FastAPI Backend for Codebase Analysis
Consolidated from all analysis systems with single comprehensive endpoint
Includes Modal deployment support and all functionality from backend branches
"""

import os
import time
import asyncio
import modal
import requests
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Graph-sitter imports
from graph_sitter.core.codebase import Codebase

# Import consolidated models and analysis
from .models import (
    ComprehensiveAnalysisRequest, 
    ComprehensiveAnalysisResponse,
    AnalysisResults
)
<<<<<<< HEAD
from .analysis import ComprehensiveCodebaseAnalyzer
=======
from .analysis import (
    analyze_codebase_comprehensive, 
    get_codebase_summary, 
    create_health_dashboard,
    generate_repository_analysis_report,
    calculate_comprehensive_metrics
)

# Modal configuration for deployment
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "codegen", "fastapi", "uvicorn", "gitpython", "requests", 
        "pydantic", "datetime", "networkx"
    )
)

modal_app = modal.App(name="comprehensive-analytics-app", image=image)
>>>>>>> e76cfb7 (Consolidate comprehensive functionality from all backend branches)

# Initialize FastAPI app
app = FastAPI(
    title="Comprehensive Codebase Analytics API",
    description="Advanced codebase analysis with issue detection, health metrics, and automated resolutions",
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

# Global cache for analysis results
analysis_cache: Dict[str, Dict[str, Any]] = {}
CACHE_EXPIRY_HOURS = 24


def get_cache_key(repo_url: str, config: Dict[str, Any]) -> str:
    """Generate cache key for analysis results"""
    import hashlib
    config_str = str(sorted(config.items()))
    return hashlib.md5(f"{repo_url}:{config_str}".encode()).hexdigest()


def is_cache_valid(cache_entry: Dict[str, Any]) -> bool:
    """Check if cache entry is still valid"""
    if not cache_entry:
        return False
    
    cache_time = cache_entry.get('timestamp', 0)
    current_time = time.time()
    return (current_time - cache_time) < (CACHE_EXPIRY_HOURS * 3600)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Comprehensive Codebase Analytics API",
        "version": "2.0.0",
        "features": [
            "Comprehensive issue detection",
            "Entry point identification", 
            "Critical file analysis",
            "Function context analysis",
            "Halstead complexity metrics",
            "Graph analysis (NetworkX)",
            "Dead code detection",
            "Health assessment",
            "Automated resolution suggestions",
            "Repository structure analysis"
        ],
        "endpoints": {
            "/analyze": "POST - Comprehensive codebase analysis",
            "/health": "GET - API health check",
            "/cache/clear": "POST - Clear analysis cache"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache_entries": len(analysis_cache),
        "uptime": "running"
    }


@app.post("/analyze", response_model=ComprehensiveAnalysisResponse)
async def analyze_codebase(
    request: ComprehensiveAnalysisRequest,
    background_tasks: BackgroundTasks
) -> ComprehensiveAnalysisResponse:
    """
    Comprehensive codebase analysis endpoint
    Consolidates all analysis features into a single powerful endpoint
    """
    start_time = time.time()
    
    try:
        print(f"🚀 Starting analysis for: {request.repo_url}")
        
        # Generate cache key
        config = request.dict()
        cache_key = get_cache_key(request.repo_url, config)
        
        # Check cache if enabled
        if request.enable_caching and cache_key in analysis_cache:
            cache_entry = analysis_cache[cache_key]
            if is_cache_valid(cache_entry):
                print("📦 Returning cached results")
                return ComprehensiveAnalysisResponse(
                    success=True,
                    results=cache_entry['results'],
                    processing_time=time.time() - start_time,
                    cache_hit=True,
                    analysis_id=cache_key
                )
        
        # Load codebase
        print("📂 Loading codebase...")
        try:
            codebase = Codebase.from_repo_url(request.repo_url)
        except Exception as e:
            print(f"❌ Failed to load codebase: {str(e)}")
            return ComprehensiveAnalysisResponse(
                success=False,
                error_message=f"Failed to load codebase: {str(e)}",
                processing_time=time.time() - start_time
            )
        
        # Prepare analysis configuration
        analysis_config = {
            'include_basic_stats': True,
            'include_issues': request.include_issues,
            'include_entry_points': request.include_entry_points,
            'include_function_context': True,
            'include_halstead_metrics': request.include_halstead_metrics,
            'include_graph_analysis': request.include_graph_analysis,
            'include_dead_code_analysis': request.include_dead_code_analysis,
            'include_health_metrics': request.include_health_metrics,
            'include_repository_structure': True,
            'include_automated_resolutions': request.include_automated_resolutions,
            'max_issues': request.max_issues,
            'severity_filter': request.severity_filter,
            'file_extensions': request.file_extensions
        }
        
        # Perform comprehensive analysis
        print("🔬 Performing comprehensive analysis...")
        analyzer = ComprehensiveCodebaseAnalyzer(codebase)
        results = analyzer.analyze_comprehensive(analysis_config)
        
        # Cache results if enabled
        if request.enable_caching:
            analysis_cache[cache_key] = {
                'results': results,
                'timestamp': time.time()
            }
            print(f"💾 Results cached with key: {cache_key}")
        
        processing_time = time.time() - start_time
        print(f"✅ Analysis completed in {processing_time:.2f}s")
        
        return ComprehensiveAnalysisResponse(
            success=True,
            results=results,
            processing_time=processing_time,
            cache_hit=False,
            analysis_id=cache_key
        )
        
    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        print(f"❌ {error_msg}")
        
        return ComprehensiveAnalysisResponse(
            success=False,
            error_message=error_msg,
            processing_time=time.time() - start_time
        )


@app.post("/cache/clear")
async def clear_cache():
    """Clear the analysis cache"""
    global analysis_cache
    cache_count = len(analysis_cache)
    analysis_cache.clear()
    
    return {
        "message": f"Cache cleared successfully",
        "entries_removed": cache_count,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    valid_entries = 0
    expired_entries = 0
    
    for cache_entry in analysis_cache.values():
        if is_cache_valid(cache_entry):
            valid_entries += 1
        else:
            expired_entries += 1
    
    return {
        "total_entries": len(analysis_cache),
        "valid_entries": valid_entries,
        "expired_entries": expired_entries,
        "cache_expiry_hours": CACHE_EXPIRY_HOURS,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# ADDITIONAL ANALYSIS ENDPOINTS (from backend branches)
# ============================================================================

@app.post("/analyze/summary")
async def get_repository_summary(request: ComprehensiveAnalysisRequest):
    """Get a quick summary of the repository"""
    try:
        print(f"📊 Getting summary for: {request.repo_url}")
        
        # Load codebase
        codebase = Codebase.from_repo_url(request.repo_url)
        
        # Generate summary
        summary = get_codebase_summary(codebase)
        
        return {
            "success": True,
            "summary": summary,
            "repo_url": request.repo_url,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to generate summary: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }


@app.post("/analyze/report")
async def generate_analysis_report(request: ComprehensiveAnalysisRequest):
    """Generate a comprehensive analysis report"""
    try:
        print(f"📋 Generating report for: {request.repo_url}")
        
        # Load codebase
        codebase = Codebase.from_repo_url(request.repo_url)
        
        # Generate comprehensive report
        report = generate_repository_analysis_report(codebase, request.repo_url)
        
        return {
            "success": True,
            "report": report,
            "repo_url": request.repo_url,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to generate report: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }


@app.post("/analyze/metrics")
async def get_comprehensive_metrics(request: ComprehensiveAnalysisRequest):
    """Get comprehensive metrics for functions, classes, and files"""
    try:
        print(f"📈 Calculating metrics for: {request.repo_url}")
        
        # Load codebase
        codebase = Codebase.from_repo_url(request.repo_url)
        
        # Calculate comprehensive metrics
        metrics = calculate_comprehensive_metrics(codebase)
        
        return {
            "success": True,
            "metrics": metrics,
            "repo_url": request.repo_url,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to calculate metrics: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }


@app.post("/analyze/health")
async def get_health_dashboard(request: ComprehensiveAnalysisRequest):
    """Get health dashboard data"""
    try:
        print(f"🏥 Generating health dashboard for: {request.repo_url}")
        
        # Load codebase and perform analysis
        codebase = Codebase.from_repo_url(request.repo_url)
        results = analyze_codebase_comprehensive(codebase, {})
        
        # Create health dashboard
        dashboard = create_health_dashboard(results)
        
        return {
            "success": True,
            "dashboard": dashboard,
            "repo_url": request.repo_url,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to generate health dashboard: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# MODAL DEPLOYMENT CONFIGURATION
# ============================================================================

@modal_app.function(image=image)
@modal.web_endpoint(method="POST")
def modal_analyze_endpoint(item: Dict[str, Any]):
    """Modal deployment endpoint for comprehensive analysis"""
    try:
        # Extract request data
        repo_url = item.get("repo_url")
        if not repo_url:
            return {"success": False, "error": "repo_url is required"}
        
        # Load codebase
        codebase = Codebase.from_repo_url(repo_url)
        
        # Perform analysis
        config = item.get("config", {})
        results = analyze_codebase_comprehensive(codebase, config)
        
        return {
            "success": True,
            "results": results.dict() if hasattr(results, 'dict') else str(results),
            "repo_url": repo_url
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Modal analysis failed: {str(e)}"
        }


@modal_app.function(image=image)
@modal.web_endpoint(method="GET")
def modal_health_check():
    """Modal health check endpoint"""
    return {
        "status": "healthy",
        "service": "Comprehensive Codebase Analytics",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_repo_url(repo_url: str) -> bool:
    """Validate if the repository URL is accessible"""
    try:
        # Simple validation - check if URL is reachable
        response = requests.head(repo_url, timeout=10)
        return response.status_code == 200
    except Exception:
        return False


def cleanup_expired_cache():
    """Remove expired entries from cache"""
    global analysis_cache
    expired_keys = []
    
    for key, cache_entry in analysis_cache.items():
        if not is_cache_valid(cache_entry):
            expired_keys.append(key)
    
    for key in expired_keys:
        del analysis_cache[key]
    
    return len(expired_keys)


# Background task to clean up cache periodically
@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    print("🚀 Comprehensive Codebase Analytics API starting up...")
    print("✅ All analysis modules loaded successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    print("🛑 Comprehensive Codebase Analytics API shutting down...")
    cleanup_expired_cache()
    print("✅ Cleanup completed")


# Export the FastAPI app for Modal and other deployments
fastapi_app = app

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_message": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_message": f"Internal server error: {str(exc)}",
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    # Run the API server
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Starting Comprehensive Codebase Analytics API on {host}:{port}")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
