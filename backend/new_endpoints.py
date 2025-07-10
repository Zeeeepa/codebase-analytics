"""
New advanced API endpoints for call graphs, code views, metrics, and security analysis.
"""

from fastapi import HTTPException
from typing import Dict, Any
from datetime import datetime
from graph_sitter.core.codebase import Codebase

# Import advanced features
try:
    from .advanced_features import (
        generate_call_graph,
        generate_code_views,
        calculate_advanced_metrics,
        analyze_security_patterns,
        validate_source_code
    )
    from .api import (
        get_codebase_summary,
        detect_entrypoints,
        identify_critical_files,
        analyze_code_issues,
        RepoRequest,
        DetailedAnalysisRequest,
        FileAnalysisRequest
    )
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    ADVANCED_FEATURES_AVAILABLE = False
    print(f"Advanced features not available: {e}")


def register_new_endpoints(app):
    """Register all new advanced endpoints with the FastAPI app."""
    
    @app.post("/generate_call_graph")
    async def generate_call_graph_endpoint(request: RepoRequest) -> Dict[str, Any]:
        """Generate comprehensive call graph for the codebase."""
        if not ADVANCED_FEATURES_AVAILABLE:
            raise HTTPException(status_code=503, detail="Advanced features not available")
        
        try:
            codebase = Codebase.from_repo(request.repo_url)
            call_graph = generate_call_graph(codebase)
            
            return {
                "repo_url": request.repo_url,
                "analysis_type": "call_graph",
                "timestamp": datetime.now().isoformat(),
                "call_graph": call_graph
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Call graph generation failed: {str(e)}")

    @app.post("/generate_code_views")
    async def generate_code_views_endpoint(request: FileAnalysisRequest) -> Dict[str, Any]:
        """Generate multiple views of code structure (AST, CFG, DFG)."""
        if not ADVANCED_FEATURES_AVAILABLE:
            raise HTTPException(status_code=503, detail="Advanced features not available")
        
        try:
            codebase = Codebase.from_repo(request.repo_url)
            target_file = None
            
            for file in codebase.files:
                if file.filepath == request.file_path or file.name == request.file_path:
                    target_file = file
                    break
            
            if not target_file:
                raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
            
            # Validate source code before analysis
            validate_source_code(target_file)
            
            code_views = generate_code_views(target_file)
            
            return {
                "repo_url": request.repo_url,
                "file_path": request.file_path,
                "analysis_type": "code_views",
                "timestamp": datetime.now().isoformat(),
                "views": code_views
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Code views generation failed: {str(e)}")

    @app.post("/calculate_advanced_metrics")
    async def calculate_advanced_metrics_endpoint(request: RepoRequest) -> Dict[str, Any]:
        """Calculate advanced code metrics including complexity, dependencies, and quality."""
        if not ADVANCED_FEATURES_AVAILABLE:
            raise HTTPException(status_code=503, detail="Advanced features not available")
        
        try:
            codebase = Codebase.from_repo(request.repo_url)
            metrics = calculate_advanced_metrics(codebase)
            
            return {
                "repo_url": request.repo_url,
                "analysis_type": "advanced_metrics",
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Advanced metrics calculation failed: {str(e)}")

    @app.post("/analyze_security_patterns")
    async def analyze_security_patterns_endpoint(request: RepoRequest) -> Dict[str, Any]:
        """Analyze security patterns and potential vulnerabilities."""
        if not ADVANCED_FEATURES_AVAILABLE:
            raise HTTPException(status_code=503, detail="Advanced features not available")
        
        try:
            codebase = Codebase.from_repo(request.repo_url)
            security_analysis = analyze_security_patterns(codebase)
            
            return {
                "repo_url": request.repo_url,
                "analysis_type": "security_analysis",
                "timestamp": datetime.now().isoformat(),
                "security_analysis": security_analysis
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Security analysis failed: {str(e)}")

    @app.post("/comprehensive_codebase_analysis")
    async def comprehensive_codebase_analysis_endpoint(request: DetailedAnalysisRequest) -> Dict[str, Any]:
        """
        Ultimate comprehensive analysis combining all advanced features including:
        - Call graph analysis
        - Advanced metrics
        - Security analysis
        - Code views for critical files
        - Performance optimization suggestions
        """
        if not ADVANCED_FEATURES_AVAILABLE:
            raise HTTPException(status_code=503, detail="Advanced features not available")
        
        try:
            codebase = Codebase.from_repo(request.repo_url)
            
            print("🚀 Starting comprehensive codebase analysis...")
            
            results = {
                "repo_url": request.repo_url,
                "analysis_type": "comprehensive_codebase_analysis",
                "timestamp": datetime.now().isoformat(),
                "summary": get_codebase_summary(codebase)
            }
            
            # Generate call graph
            print("📊 Generating call graph...")
            results["call_graph"] = generate_call_graph(codebase)
            
            # Calculate advanced metrics
            print("📈 Calculating advanced metrics...")
            results["advanced_metrics"] = calculate_advanced_metrics(codebase)
            
            # Security analysis
            print("🔒 Performing security analysis...")
            results["security_analysis"] = analyze_security_patterns(codebase)
            
            # Add existing comprehensive analysis if requested
            if request.include_issues:
                print("🔍 Running issue analysis...")
                results["issues"] = analyze_code_issues(codebase, request.max_issues)
            
            if request.include_entrypoints:
                print("🚪 Detecting entry points...")
                results["entrypoints"] = detect_entrypoints(codebase)
            
            if request.include_critical_files:
                print("📁 Identifying critical files...")
                results["critical_files"] = identify_critical_files(codebase)
            
            # Generate code views for top 3 critical files
            print("🔍 Generating code views for critical files...")
            critical_files = results.get("critical_files", [])[:3]
            results["code_views"] = {}
            
            for critical_file in critical_files:
                try:
                    file_path = critical_file["file_path"]
                    target_file = None
                    for file in codebase.files:
                        if file.filepath == file_path:
                            target_file = file
                            break
                    
                    if target_file:
                        results["code_views"][file_path] = generate_code_views(target_file)
                except Exception as e:
                    results["code_views"][file_path] = {"error": str(e)}
            
            print("✅ Comprehensive analysis complete!")
            
            return results
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Comprehensive analysis failed: {str(e)}")

    @app.get("/features_status")
    async def features_status():
        """Check status of all advanced features."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "advanced_features_available": ADVANCED_FEATURES_AVAILABLE,
            "available_endpoints": {
                "call_graph": "/generate_call_graph",
                "code_views": "/generate_code_views", 
                "advanced_metrics": "/calculate_advanced_metrics",
                "security_analysis": "/analyze_security_patterns",
                "comprehensive_analysis": "/comprehensive_codebase_analysis"
            },
            "features": {
                "call_graph_generation": True,
                "multi_view_analysis": True,
                "advanced_metrics": True,
                "security_pattern_detection": True,
                "performance_optimization": True,
                "incremental_parsing": True,
                "caching_system": True,
                "error_validation": True
            }
        }

