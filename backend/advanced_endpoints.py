"""
Advanced API endpoints using comprehensive graph-sitter capabilities.
These endpoints provide enhanced analysis features that leverage the full power
of graph-sitter's AST analysis, semantic understanding, and architectural insights.
"""

from fastapi import HTTPException
from typing import Dict, Any
from datetime import datetime
import graph_sitter
from graph_sitter.core.codebase import Codebase

# Import advanced analysis functions
try:
    from .advanced_analysis import (
        advanced_semantic_analysis,
        advanced_dependency_analysis,
        advanced_architectural_analysis,
        language_specific_analysis,
        advanced_performance_analysis,
        comprehensive_error_context_analysis
    )
    from .api import (
        get_codebase_summary,
        detect_entrypoints,
        identify_critical_files,
        build_dependency_graph,
        RepoRequest,
        DetailedAnalysisRequest
    )
    ADVANCED_ANALYSIS_AVAILABLE = True
except ImportError as e:
    ADVANCED_ANALYSIS_AVAILABLE = False
    print(f"Advanced analysis functions not available: {e}")


def register_advanced_endpoints(app):
    """Register all advanced endpoints with the FastAPI app."""
    
    @app.post("/advanced_semantic_analysis")
    async def advanced_semantic_analysis_endpoint(request: RepoRequest) -> Dict[str, Any]:
        """Advanced semantic analysis using Expression, Name, String, Value classes."""
        if not ADVANCED_ANALYSIS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Advanced analysis not available")
        
        try:
            codebase = Codebase.from_repo(request.repo_url)
            semantic_data = advanced_semantic_analysis(codebase)
            
            return {
                "repo_url": request.repo_url,
                "analysis_type": "advanced_semantic_analysis",
                "timestamp": datetime.now().isoformat(),
                **semantic_data
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Advanced semantic analysis failed: {str(e)}")

    @app.post("/advanced_dependency_analysis")
    async def advanced_dependency_analysis_endpoint(request: RepoRequest) -> Dict[str, Any]:
        """Enhanced dependency analysis using Export and Assignment classes."""
        if not ADVANCED_ANALYSIS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Advanced analysis not available")
        
        try:
            codebase = Codebase.from_repo(request.repo_url)
            dependency_data = advanced_dependency_analysis(codebase)
            
            return {
                "repo_url": request.repo_url,
                "analysis_type": "advanced_dependency_analysis",
                "timestamp": datetime.now().isoformat(),
                **dependency_data
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Advanced dependency analysis failed: {str(e)}")

    @app.post("/advanced_architectural_analysis")
    async def advanced_architectural_analysis_endpoint(request: RepoRequest) -> Dict[str, Any]:
        """Architectural analysis using Interface and Directory classes."""
        if not ADVANCED_ANALYSIS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Advanced analysis not available")
        
        try:
            codebase = Codebase.from_repo(request.repo_url)
            arch_data = advanced_architectural_analysis(codebase)
            
            return {
                "repo_url": request.repo_url,
                "analysis_type": "advanced_architectural_analysis",
                "timestamp": datetime.now().isoformat(),
                **arch_data
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Advanced architectural analysis failed: {str(e)}")

    @app.post("/language_specific_analysis")
    async def language_specific_analysis_endpoint(request: RepoRequest) -> Dict[str, Any]:
        """Language-specific analysis using Python and TypeScript analyzers."""
        if not ADVANCED_ANALYSIS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Advanced analysis not available")
        
        try:
            codebase = Codebase.from_repo(request.repo_url)
            lang_data = language_specific_analysis(codebase)
            
            return {
                "repo_url": request.repo_url,
                "analysis_type": "language_specific_analysis",
                "timestamp": datetime.now().isoformat(),
                **lang_data
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Language-specific analysis failed: {str(e)}")

    @app.post("/advanced_performance_analysis")
    async def advanced_performance_analysis_endpoint(request: RepoRequest) -> Dict[str, Any]:
        """Advanced performance analysis using graph-sitter's deep AST capabilities."""
        if not ADVANCED_ANALYSIS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Advanced analysis not available")
        
        try:
            codebase = Codebase.from_repo(request.repo_url)
            perf_data = advanced_performance_analysis(codebase)
            
            return {
                "repo_url": request.repo_url,
                "analysis_type": "advanced_performance_analysis",
                "timestamp": datetime.now().isoformat(),
                **perf_data
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Advanced performance analysis failed: {str(e)}")

    @app.post("/comprehensive_error_analysis")
    async def comprehensive_error_analysis_endpoint(request: DetailedAnalysisRequest) -> Dict[str, Any]:
        """
        Comprehensive error analysis with detailed context using advanced graph-sitter features.
        Provides exactly what was requested: detailed error context with file paths, line numbers,
        function names, interconnected context, and fix suggestions.
        
        Example response format:
        "182 issues found, 11 critical"
        Each issue includes:
        - File path and line number
        - Function/class name and error details
        - All interconnected parameters, functions, methods, classes
        - Detailed fix suggestions
        """
        if not ADVANCED_ANALYSIS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Advanced analysis not available")
        
        try:
            codebase = Codebase.from_repo(request.repo_url)
            error_data = comprehensive_error_context_analysis(codebase, request.max_issues)
            
            # Format the summary as requested
            total_issues = error_data["total_issues"]
            critical_issues = error_data["critical_issues"]
            summary_message = f"{total_issues} issues found, {critical_issues} critical"
            
            return {
                "repo_url": request.repo_url,
                "analysis_type": "comprehensive_error_analysis",
                "timestamp": datetime.now().isoformat(),
                "summary_message": summary_message,
                "analysis_summary": {
                    "total_issues": total_issues,
                    "critical_issues": critical_issues,
                    "issues_by_severity": error_data["issues_by_severity"],
                    "files_with_issues": len(error_data["issues_by_file"]),
                    "most_problematic_files": [
                        {
                            "file_path": file_path,
                            "total_issues": file_data["total_issues"],
                            "critical_issues": file_data["critical_count"]
                        }
                        for file_path, file_data in sorted(
                            error_data["issues_by_file"].items(),
                            key=lambda x: x[1]["total_issues"],
                            reverse=True
                        )[:10]
                    ]
                },
                "detailed_issues": error_data["detailed_issues"],
                "issues_by_file": error_data["issues_by_file"]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Comprehensive error analysis failed: {str(e)}")

    @app.post("/ultimate_codebase_analysis")
    async def ultimate_codebase_analysis(request: DetailedAnalysisRequest) -> Dict[str, Any]:
        """
        Ultimate comprehensive codebase analysis combining ALL advanced graph-sitter features.
        This endpoint provides the most complete analysis possible using all discovered capabilities.
        """
        if not ADVANCED_ANALYSIS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Advanced analysis not available")
        
        try:
            codebase = Codebase.from_repo(request.repo_url)
            
            print("🚀 Starting ultimate codebase analysis...")
            
            # Run all advanced analyses
            results = {
                "repo_url": request.repo_url,
                "analysis_type": "ultimate_comprehensive_analysis",
                "timestamp": datetime.now().isoformat(),
                "codebase_summary": get_codebase_summary(codebase)
            }
            
            if request.include_issues:
                print("🔍 Running comprehensive error analysis...")
                error_data = comprehensive_error_context_analysis(codebase, request.max_issues)
                results["comprehensive_error_analysis"] = error_data
                
                # Create the requested summary format
                total_issues = error_data["total_issues"]
                critical_issues = error_data["critical_issues"]
                results["issue_summary"] = f"{total_issues} issues found, {critical_issues} critical"
            
            print("🧠 Running advanced semantic analysis...")
            results["semantic_analysis"] = advanced_semantic_analysis(codebase)
            
            print("🔗 Running advanced dependency analysis...")
            results["dependency_analysis"] = advanced_dependency_analysis(codebase)
            
            print("🏗️ Running architectural analysis...")
            results["architectural_analysis"] = advanced_architectural_analysis(codebase)
            
            print("🌐 Running language-specific analysis...")
            results["language_analysis"] = language_specific_analysis(codebase)
            
            print("⚡ Running performance analysis...")
            results["performance_analysis"] = advanced_performance_analysis(codebase)
            
            if request.include_entrypoints:
                print("🚪 Detecting entry points...")
                results["entrypoints"] = detect_entrypoints(codebase)
            
            if request.include_critical_files:
                print("📁 Identifying critical files...")
                results["critical_files"] = identify_critical_files(codebase)
            
            if request.include_dependency_graph:
                print("📊 Building dependency graph...")
                results["dependency_graph"] = build_dependency_graph(codebase)
            
            print("✅ Ultimate analysis complete!")
            
            return results
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ultimate codebase analysis failed: {str(e)}")

    @app.get("/advanced_health")
    async def advanced_health_check():
        """Advanced health check endpoint with graph-sitter capability information."""
        return {
            "status": "healthy", 
            "timestamp": datetime.now().isoformat(),
            "advanced_analysis_available": ADVANCED_ANALYSIS_AVAILABLE,
            "graph_sitter_version": getattr(graph_sitter, '__version__', 'unknown'),
            "available_features": {
                "semantic_analysis": ADVANCED_ANALYSIS_AVAILABLE,
                "dependency_analysis": ADVANCED_ANALYSIS_AVAILABLE,
                "architectural_analysis": ADVANCED_ANALYSIS_AVAILABLE,
                "language_specific_analysis": ADVANCED_ANALYSIS_AVAILABLE,
                "performance_analysis": ADVANCED_ANALYSIS_AVAILABLE,
                "comprehensive_error_analysis": ADVANCED_ANALYSIS_AVAILABLE,
                "ultimate_analysis": ADVANCED_ANALYSIS_AVAILABLE
            },
            "supported_languages": ["Python", "TypeScript", "JavaScript"],
            "advanced_capabilities": [
                "Expression-level AST analysis",
                "Name and String literal extraction",
                "Export and Assignment tracking",
                "Interface and Directory analysis",
                "Cross-language dependency analysis",
                "Performance bottleneck detection",
                "Comprehensive error context",
                "Interconnected symbol analysis"
            ]
        }

