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
from .analysis import ComprehensiveCodebaseAnalyzer

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


def validate_repo_url(repo_url: str) -> bool:
    """Validate GitHub repository URL format"""
    import re
    github_pattern = r'^https://github\.com/[\w\-\.]+/[\w\-\.]+/?$'
    return bool(re.match(github_pattern, repo_url))


def preprocess_analysis_config(request: ComprehensiveAnalysisRequest) -> Dict[str, Any]:
    """Preprocess and validate analysis configuration"""
    config = {
        'include_basic_stats': True,
        'include_issues': request.include_issues,
        'include_entry_points': request.include_entry_points,
        'include_critical_files': request.include_critical_files,
        'include_metrics': request.include_metrics,
        'include_dependency_graph': request.include_dependency_graph,
        'include_function_context': True,
        'include_halstead_metrics': request.include_halstead_metrics,
        'include_graph_analysis': request.include_graph_analysis,
        'include_dead_code_analysis': request.include_dead_code_analysis,
        'include_health_metrics': request.include_health_metrics,
        'include_repository_structure': True,
        'include_automated_resolutions': request.include_automated_resolutions,
        'max_issues': min(request.max_issues, 1000),  # Cap at 1000 for performance
        'severity_filter': request.severity_filter or ['critical', 'high', 'medium', 'low'],
        'file_extensions': request.file_extensions or ['.py', '.js', '.ts', '.jsx', '.tsx']
    }
    
    return config


def format_analysis_results(results: AnalysisResults) -> Dict[str, Any]:
    """Format analysis results for JSON serialization"""
    try:
        formatted_results = {
            # Basic statistics
            'repository_stats': {
                'total_files': results.total_files,
                'total_functions': results.total_functions,
                'total_classes': results.total_classes,
                'total_lines_of_code': results.total_lines_of_code,
                'effective_lines_of_code': results.effective_lines_of_code
            },
            
            # Issue analysis
            'issues': {
                'total_issues': results.total_issues,
                'by_severity': results.issues_by_severity,
                'by_type': results.issues_by_type,
                'critical_issues': [
                    {
                        'type': issue.issue_type.value if hasattr(issue.issue_type, 'value') else str(issue.issue_type),
                        'severity': issue.severity.value if hasattr(issue.severity, 'value') else str(issue.severity),
                        'message': issue.message,
                        'file_path': issue.file_path,
                        'line_number': issue.line_number,
                        'confidence': getattr(issue, 'confidence', 0.8)
                    } for issue in results.critical_issues
                ]
            },
            
            # Automated resolutions
            'automated_resolutions': [
                {
                    'issue_type': res.issue_type.value if hasattr(res.issue_type, 'value') else str(res.issue_type),
                    'resolution_type': res.resolution_type,
                    'description': res.description,
                    'confidence': res.confidence,
                    'file_path': res.file_path,
                    'line_number': res.line_number,
                    'suggested_fix': res.suggested_fix,
                    'blast_radius': getattr(res, 'blast_radius', 'low')
                } for res in results.automated_resolutions
            ],
            
            # Function analysis
            'functions': {
                'entry_points': results.entry_points,
                'most_important_functions': results.most_important_functions,
                'function_contexts': {
                    name: {
                        'calls_made': ctx.calls_made,
                        'called_by': ctx.called_by,
                        'complexity': ctx.complexity,
                        'lines_of_code': ctx.lines_of_code,
                        'parameters': ctx.parameters,
                        'return_type': ctx.return_type,
                        'docstring_present': ctx.docstring_present
                    } for name, ctx in results.function_contexts.items()
                }
            },
            
            # Metrics
            'metrics': {
                'halstead_metrics': results.halstead_metrics,
                'complexity_metrics': results.complexity_metrics,
                'maintainability_score': results.maintainability_score,
                'technical_debt_score': results.technical_debt_score
            },
            
            # Graph analysis
            'graph_analysis': {
                'call_graph': results.call_graph,
                'dependency_graph': results.dependency_graph,
                'graph_metrics': {
                    k: v.__dict__ if hasattr(v, '__dict__') else v 
                    for k, v in results.graph_metrics.items()
                } if results.graph_metrics else {}
            },
            
            # Dead code analysis
            'dead_code_analysis': {
                'unused_functions': results.dead_code_analysis.unused_functions if results.dead_code_analysis else [],
                'unused_variables': results.dead_code_analysis.unused_variables if results.dead_code_analysis else [],
                'unused_imports': results.dead_code_analysis.unused_imports if results.dead_code_analysis else [],
                'blast_radius': results.dead_code_analysis.blast_radius if results.dead_code_analysis else {}
            },
            
            # Health metrics
            'health_metrics': {
                'overall_score': results.health_metrics.overall_score if results.health_metrics else 0,
                'maintainability_score': results.health_metrics.maintainability_score if results.health_metrics else 0,
                'technical_debt_score': results.health_metrics.technical_debt_score if results.health_metrics else 0,
                'complexity_score': results.health_metrics.complexity_score if results.health_metrics else 0,
                'documentation_coverage': results.health_metrics.documentation_coverage if results.health_metrics else 0,
                'risk_level': results.health_metrics.risk_level if results.health_metrics else 'unknown'
            },
            
            # Repository structure
            'repository_structure': {
                'directory_structure': results.repository_structure.directory_structure if results.repository_structure else {},
                'file_types': results.repository_structure.file_types if results.repository_structure else {},
                'largest_files': results.repository_structure.largest_files if results.repository_structure else []
            },
            
            # Analysis metadata
            'analysis_metadata': {
                'timestamp': results.analysis_timestamp.isoformat() if results.analysis_timestamp else datetime.now().isoformat(),
                'duration': results.analysis_duration,
                'errors': results.errors or []
            }
        }
        
        return formatted_results
        
    except Exception as e:
        print(f"⚠️ Error formatting results: {str(e)}")
        # Return basic structure if formatting fails
        return {
            'repository_stats': {
                'total_files': getattr(results, 'total_files', 0),
                'total_functions': getattr(results, 'total_functions', 0),
                'total_classes': getattr(results, 'total_classes', 0),
                'total_lines_of_code': getattr(results, 'total_lines_of_code', 0),
                'effective_lines_of_code': getattr(results, 'effective_lines_of_code', 0)
            },
            'issues': {
                'total_issues': getattr(results, 'total_issues', 0),
                'by_severity': getattr(results, 'issues_by_severity', {}),
                'by_type': getattr(results, 'issues_by_type', {}),
                'critical_issues': []
            },
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'duration': getattr(results, 'analysis_duration', 0),
                'errors': [f"Result formatting error: {str(e)}"]
            }
        }


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
        
        # Validate repository URL
        if not validate_repo_url(request.repo_url):
            return ComprehensiveAnalysisResponse(
                success=False,
                error_message="Invalid GitHub repository URL format",
                processing_time=time.time() - start_time
            )
        
        # Preprocess analysis configuration
        analysis_config = preprocess_analysis_config(request)
        cache_key = get_cache_key(request.repo_url, analysis_config)
        
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
            codebase = Codebase.from_repo(request.repo_url)
            print(f"✅ Successfully loaded codebase with {len(codebase.files)} files")
        except Exception as e:
            print(f"❌ Failed to load codebase: {str(e)}")
            return ComprehensiveAnalysisResponse(
                success=False,
                error_message=f"Failed to load codebase: {str(e)}",
                processing_time=time.time() - start_time
            )
        
        # Analysis configuration already preprocessed above
        
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
        
        # Format results for response
        formatted_results = format_analysis_results(results)
        
        return ComprehensiveAnalysisResponse(
            success=True,
            results=formatted_results,
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


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Comprehensive Codebase Analytics API",
        "version": "2.0.0",
        "features": [
            "comprehensive_issue_detection",
            "automated_resolutions", 
            "halstead_metrics",
            "graph_analysis",
            "health_assessment",
            "dead_code_detection",
            "repository_structure_analysis"
        ],
        "timestamp": datetime.now().isoformat()
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Comprehensive Codebase Analytics API",
        "version": "2.0.0",
        "description": "Advanced codebase analysis with issue detection, health metrics, and automated resolutions",
        "endpoints": {
            "analyze": "POST /analyze - Comprehensive codebase analysis",
            "health": "GET /health - Health check",
            "cache_stats": "GET /cache/stats - Cache statistics",
            "cache_clear": "POST /cache/clear - Clear analysis cache",
            "docs": "GET /docs - API documentation"
        },
        "features": {
            "issue_detection": "60+ comprehensive issue types",
            "automated_resolutions": "Automated fix suggestions with confidence scoring",
            "graph_analysis": "NetworkX-powered call and dependency graphs",
            "health_metrics": "Comprehensive health assessment with risk levels",
            "halstead_metrics": "Detailed complexity analysis",
            "dead_code_detection": "Unused code identification with blast radius",
            "caching": "Performance optimization with configurable caching"
        },
        "consolidation": {
            "pr_97": "Advanced issue detection and automated resolutions",
            "pr_99": "Single endpoint architecture and caching",
            "backend_branch": "Enhanced metrics and repository analysis"
        },
        "timestamp": datetime.now().isoformat()
    }


@app.post("/analyze/issues-only")
async def analyze_issues_only(request: ComprehensiveAnalysisRequest) -> ComprehensiveAnalysisResponse:
    """
    Lightweight endpoint for issue detection only
    Optimized for quick issue scanning without full analysis
    """
    start_time = time.time()
    
    try:
        print(f"🔍 Starting issues-only analysis for: {request.repo_url}")
        
        # Load codebase
        codebase = Codebase.from_repo(request.repo_url)
        
        # Configure for issues-only analysis
        analysis_config = {
            'include_basic_stats': True,
            'include_issues': True,
            'include_entry_points': False,
            'include_function_context': False,
            'include_halstead_metrics': False,
            'include_graph_analysis': False,
            'include_dead_code_analysis': False,
            'include_health_metrics': False,
            'include_repository_structure': False,
            'include_automated_resolutions': request.include_automated_resolutions,
            'max_issues': request.max_issues,
            'severity_filter': request.severity_filter,
            'file_extensions': request.file_extensions
        }
        
        # Perform lightweight analysis
        analyzer = ComprehensiveCodebaseAnalyzer(codebase)
        results = analyzer.analyze_comprehensive(analysis_config)
        
        processing_time = time.time() - start_time
        print(f"✅ Issues-only analysis completed in {processing_time:.2f}s")
        
        return ComprehensiveAnalysisResponse(
            success=True,
            results=results,
            processing_time=processing_time,
            cache_hit=False,
            analysis_id=f"issues_only_{int(time.time())}"
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        return ComprehensiveAnalysisResponse(
            success=False,
            error_message=f"Issues analysis failed: {str(e)}",
            processing_time=processing_time
        )


@app.post("/analyze/health-only")
async def analyze_health_only(request: ComprehensiveAnalysisRequest) -> ComprehensiveAnalysisResponse:
    """
    Lightweight endpoint for health assessment only
    Quick health check without detailed analysis
    """
    start_time = time.time()
    
    try:
        print(f"🏥 Starting health-only analysis for: {request.repo_url}")
        
        # Load codebase
        codebase = Codebase.from_repo(request.repo_url)
        
        # Configure for health-only analysis
        analysis_config = {
            'include_basic_stats': True,
            'include_issues': True,  # Need issues for health calculation
            'include_entry_points': False,
            'include_function_context': True,  # Need for health metrics
            'include_halstead_metrics': False,
            'include_graph_analysis': False,
            'include_dead_code_analysis': True,  # Need for health metrics
            'include_health_metrics': True,
            'include_repository_structure': False,
            'include_automated_resolutions': False,
            'max_issues': 100,  # Limit for performance
            'severity_filter': request.severity_filter,
            'file_extensions': request.file_extensions
        }
        
        # Perform health-focused analysis
        analyzer = ComprehensiveCodebaseAnalyzer(codebase)
        results = analyzer.analyze_comprehensive(analysis_config)
        
        processing_time = time.time() - start_time
        print(f"✅ Health-only analysis completed in {processing_time:.2f}s")
        
        return ComprehensiveAnalysisResponse(
            success=True,
            results=results,
            processing_time=processing_time,
            cache_hit=False,
            analysis_id=f"health_only_{int(time.time())}"
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        return ComprehensiveAnalysisResponse(
            success=False,
            error_message=f"Health analysis failed: {str(e)}",
            processing_time=processing_time
        )


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
