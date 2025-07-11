"""
Comprehensive FastAPI Backend for Codebase Analysis
Consolidated from all analysis systems with single comprehensive endpoint
"""

import os
import time
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

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
from .analysis import analyze_codebase_comprehensive

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
        results = analyze_codebase_comprehensive(codebase, analysis_config)
        
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


# Background task to clean expired cache entries
async def cleanup_expired_cache():
    """Remove expired cache entries"""
    global analysis_cache
    
    expired_keys = []
    for key, cache_entry in analysis_cache.items():
        if not is_cache_valid(cache_entry):
            expired_keys.append(key)
    
    for key in expired_keys:
        del analysis_cache[key]
    
    if expired_keys:
        print(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")


@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    print("🚀 Comprehensive Codebase Analytics API starting up...")
    print("📊 Features enabled:")
    print("   • Comprehensive issue detection")
    print("   • Entry point identification")
    print("   • Critical file analysis")
    print("   • Function context analysis")
    print("   • Halstead complexity metrics")
    print("   • Graph analysis (NetworkX)")
    print("   • Dead code detection")
    print("   • Health assessment")
    print("   • Automated resolution suggestions")
    print("   • Repository structure analysis")
    print("✅ API ready to serve requests!")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    print("🛑 Comprehensive Codebase Analytics API shutting down...")
    print(f"📊 Final stats: {len(analysis_cache)} cache entries")


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
