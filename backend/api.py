"""
FastAPI application for the codebase analysis service.
This module defines all API endpoints and handles web requests,
utilizing the core analysis functions from analysis.py and data models
from models.py. It also includes Modal setup for deployment.
"""

import subprocess
import tempfile
from datetime import datetime, timedelta
import requests
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
import modal
import graph_sitter
from graph_sitter.core.codebase import Codebase

# Import analysis functions and models from local modules
from . import analysis
from .models import RepoRequest, DetailedAnalysisRequest


fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helper functions for API responses
def get_codebase_summary(codebase: Codebase) -> str:
    files = list(codebase.files)
    functions = list(codebase.functions)
    classes = list(codebase.classes)
    return (
        f"Contains {len(codebase.ctx.get_nodes())} nodes: "
        f"{len(files)} files, {len(functions)} functions, {len(classes)} classes."
    )


def register_advanced_endpoints(app):
    """Register all advanced endpoints with the FastAPI app."""

    @app.post("/advanced_semantic_analysis")
    async def advanced_semantic_analysis_endpoint(
        request: RepoRequest,
    ) -> Dict[str, Any]:
        """Advanced semantic analysis using Expression, Name, String, Value classes."""
        if not ADVANCED_ANALYSIS_AVAILABLE:
            raise HTTPException(
                status_code=503, detail="Advanced analysis not available"
            )

        try:
            codebase = Codebase.from_repo(request.repo_url)
            semantic_data = advanced_semantic_analysis(codebase)

            return {
                "repo_url": request.repo_url,
                "analysis_type": "advanced_semantic_analysis",
                "timestamp": datetime.now().isoformat(),
                **semantic_data,
            }
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Advanced semantic analysis failed: {str(e)}"
            )

    @app.post("/advanced_dependency_analysis")
    async def advanced_dependency_analysis_endpoint(
        request: RepoRequest,
    ) -> Dict[str, Any]:
        """Enhanced dependency analysis using Export and Assignment classes."""
        if not ADVANCED_ANALYSIS_AVAILABLE:
            raise HTTPException(
                status_code=503, detail="Advanced analysis not available"
            )

        try:
            codebase = Codebase.from_repo(request.repo_url)
            dependency_data = advanced_dependency_analysis(codebase)

            return {
                "repo_url": request.repo_url,
                "analysis_type": "advanced_dependency_analysis",
                "timestamp": datetime.now().isoformat(),
                **dependency_data,
            }
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Advanced dependency analysis failed: {str(e)}"
            )

    @app.post("/advanced_architectural_analysis")
    async def advanced_architectural_analysis_endpoint(
        request: RepoRequest,
    ) -> Dict[str, Any]:
        """Architectural analysis using Interface and Directory classes."""
        if not ADVANCED_ANALYSIS_AVAILABLE:
            raise HTTPException(
                status_code=503, detail="Advanced analysis not available"
            )

        try:
            codebase = Codebase.from_repo(request.repo_url)
            arch_data = advanced_architectural_analysis(codebase)

            return {
                "repo_url": request.repo_url,
                "analysis_type": "advanced_architectural_analysis",
                "timestamp": datetime.now().isoformat(),
                **arch_data,
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Advanced architectural analysis failed: {str(e)}",
            )

    @app.post("/language_specific_analysis")
    async def language_specific_analysis_endpoint(
        request: RepoRequest,
    ) -> Dict[str, Any]:
        """Language-specific analysis using Python and TypeScript analyzers."""
        if not ADVANCED_ANALYSIS_AVAILABLE:
            raise HTTPException(
                status_code=503, detail="Advanced analysis not available"
            )

        try:
            codebase = Codebase.from_repo(request.repo_url)
            lang_data = language_specific_analysis(codebase)

            return {
                "repo_url": request.repo_url,
                "analysis_type": "language_specific_analysis",
                "timestamp": datetime.now().isoformat(),
                **lang_data,
            }
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Language-specific analysis failed: {str(e)}"
            )

    @app.post("/advanced_performance_analysis")
    async def advanced_performance_analysis_endpoint(
        request: RepoRequest,
    ) -> Dict[str, Any]:
        """Advanced performance analysis using graph-sitter's deep AST capabilities."""
        if not ADVANCED_ANALYSIS_AVAILABLE:
            raise HTTPException(
                status_code=503, detail="Advanced analysis not available"
            )

        try:
            codebase = Codebase.from_repo(request.repo_url)
            perf_data = advanced_performance_analysis(codebase)

            return {
                "repo_url": request.repo_url,
                "analysis_type": "advanced_performance_analysis",
                "timestamp": datetime.now().isoformat(),
                **perf_data,
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Advanced performance analysis failed: {str(e)}",
            )

    @app.post("/comprehensive_error_analysis")
    async def comprehensive_error_analysis_endpoint(
        request: DetailedAnalysisRequest,
    ) -> Dict[str, Any]:
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
            raise HTTPException(
                status_code=503, detail="Advanced analysis not available"
            )

        try:
            codebase = Codebase.from_repo(request.repo_url)
            error_data = comprehensive_error_context_analysis(
                codebase, request.max_issues
            )

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
                            "critical_issues": file_data["critical_count"],
                        }
                        for file_path, file_data in sorted(
                            error_data["issues_by_file"].items(),
                            key=lambda x: x[1]["total_issues"],
                            reverse=True,
                        )[:10]
                    ],
                },
                "detailed_issues": error_data["detailed_issues"],
                "issues_by_file": error_data["issues_by_file"],
            }
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Comprehensive error analysis failed: {str(e)}"
            )

    @app.post("/ultimate_codebase_analysis")
    async def ultimate_codebase_analysis(
        request: DetailedAnalysisRequest,
    ) -> Dict[str, Any]:
        """
        Ultimate comprehensive codebase analysis combining ALL advanced graph-sitter features.
        This endpoint provides the most complete analysis possible using all discovered capabilities.
        """
        if not ADVANCED_ANALYSIS_AVAILABLE:
            raise HTTPException(
                status_code=503, detail="Advanced analysis not available"
            )

        try:
            codebase = Codebase.from_repo(request.repo_url)

            print("🚀 Starting ultimate codebase analysis...")

            # Run all advanced analyses
            results = {
                "repo_url": request.repo_url,
                "analysis_type": "ultimate_comprehensive_analysis",
                "timestamp": datetime.now().isoformat(),
                "codebase_summary": get_codebase_summary(codebase),
            }

            if request.include_issues:
                print("🔍 Running comprehensive error analysis...")
                error_data = comprehensive_error_context_analysis(
                    codebase, request.max_issues
                )
                results["comprehensive_error_analysis"] = error_data

                # Create the requested summary format
                total_issues = error_data["total_issues"]
                critical_issues = error_data["critical_issues"]
                results["issue_summary"] = (
                    f"{total_issues} issues found, {critical_issues} critical"
                )

            print("🧠 Running advanced semantic analysis...")
            results["semantic_analysis"] = advanced_semantic_analysis(codebase)

            print("🔗 Running advanced dependency analysis...")
            results["dependency_analysis"] = advanced_dependency_analysis(codebase)

            print("🏗️ Running architectural analysis...")
            results["architectural_analysis"] = advanced_architectural_analysis(
                codebase
            )

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
            raise HTTPException(
                status_code=500, detail=f"Ultimate codebase analysis failed: {str(e)}"
            )

    @app.get("/advanced_health")
    async def advanced_health_check():
        """Advanced health check endpoint with graph-sitter capability information."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "advanced_analysis_available": ADVANCED_ANALYSIS_AVAILABLE,
            "graph_sitter_version": getattr(graph_sitter, "__version__", "unknown"),
            "available_features": {
                "semantic_analysis": ADVANCED_ANALYSIS_AVAILABLE,
                "dependency_analysis": ADVANCED_ANALYSIS_AVAILABLE,
                "architectural_analysis": ADVANCED_ANALYSIS_AVAILABLE,
                "language_specific_analysis": ADVANCED_ANALYSIS_AVAILABLE,
                "performance_analysis": ADVANCED_ANALYSIS_AVAILABLE,
                "comprehensive_error_analysis": ADVANCED_ANALYSIS_AVAILABLE,
                "ultimate_analysis": ADVANCED_ANALYSIS_AVAILABLE,
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
                "Interconnected symbol analysis",
            ],
        }


def get_monthly_commits(repo_path: str) -> dict:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    date_format = "%Y-%m-%d"
    repo_url = f"https://github.com/{repo_path}"
    monthly_counts = {}
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                ["git", "clone", repo_url, temp_dir], check=True, capture_output=True
            )
            cmd = [
                "git",
                "log",
                f"--since={start_date.strftime(date_format)}",
                f"--until={end_date.strftime(date_format)}",
                "--format=%aI",
            ]
            result = subprocess.run(
                cmd, cwd=temp_dir, capture_output=True, text=True, check=True
            )
            commit_dates = result.stdout.strip().split("\n")
            for date_str in filter(None, commit_dates):
                month_key = datetime.fromisoformat(date_str.strip()).strftime("%Y-%m")
                monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error getting commits for {repo_path}: {e}")
        return {}
    return dict(sorted(monthly_counts.items()))


def get_github_repo_description(repo_url):
    api_url = f"https://api.github.com/repos/{repo_url}"
    try:
        response = requests.get(api_url, timeout=5)
        if response.status_code == 200:
            return response.json().get("description", "No description available")
    except requests.RequestException:
        pass
    return ""


def register_new_endpoints(app):
    """Register all new advanced endpoints with the FastAPI app."""

    @app.post("/generate_call_graph")
    async def generate_call_graph_endpoint(request: RepoRequest) -> Dict[str, Any]:
        """Generate comprehensive call graph for the codebase."""
        if not ADVANCED_FEATURES_AVAILABLE:
            raise HTTPException(
                status_code=503, detail="Advanced features not available"
            )

        try:
            codebase = Codebase.from_repo(request.repo_url)
            call_graph = generate_call_graph(codebase)

            return {
                "repo_url": request.repo_url,
                "analysis_type": "call_graph",
                "timestamp": datetime.now().isoformat(),
                "call_graph": call_graph,
            }
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Call graph generation failed: {str(e)}"
            )

    @app.post("/generate_code_views")
    async def generate_code_views_endpoint(
        request: FileAnalysisRequest,
    ) -> Dict[str, Any]:
        """Generate multiple views of code structure (AST, CFG, DFG)."""
        if not ADVANCED_FEATURES_AVAILABLE:
            raise HTTPException(
                status_code=503, detail="Advanced features not available"
            )

        try:
            codebase = Codebase.from_repo(request.repo_url)
            target_file = None

            for file in codebase.files:
                if file.filepath == request.file_path or file.name == request.file_path:
                    target_file = file
                    break

            if not target_file:
                raise HTTPException(
                    status_code=404, detail=f"File not found: {request.file_path}"
                )

            # Validate source code before analysis
            validate_source_code(target_file)

            code_views = generate_code_views(target_file)

            return {
                "repo_url": request.repo_url,
                "file_path": request.file_path,
                "analysis_type": "code_views",
                "timestamp": datetime.now().isoformat(),
                "views": code_views,
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Code views generation failed: {str(e)}"
            )

    @app.post("/calculate_advanced_metrics")
    async def calculate_advanced_metrics_endpoint(
        request: RepoRequest,
    ) -> Dict[str, Any]:
        """Calculate advanced code metrics including complexity, dependencies, and quality."""
        if not ADVANCED_FEATURES_AVAILABLE:
            raise HTTPException(
                status_code=503, detail="Advanced features not available"
            )

        try:
            codebase = Codebase.from_repo(request.repo_url)
            metrics = calculate_advanced_metrics(codebase)

            return {
                "repo_url": request.repo_url,
                "analysis_type": "advanced_metrics",
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
            }
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Advanced metrics calculation failed: {str(e)}"
            )

    @app.post("/analyze_security_patterns")
    async def analyze_security_patterns_endpoint(
        request: RepoRequest,
    ) -> Dict[str, Any]:
        """Analyze security patterns and potential vulnerabilities."""
        if not ADVANCED_FEATURES_AVAILABLE:
            raise HTTPException(
                status_code=503, detail="Advanced features not available"
            )

        try:
            codebase = Codebase.from_repo(request.repo_url)
            security_analysis = analyze_security_patterns(codebase)

            return {
                "repo_url": request.repo_url,
                "analysis_type": "security_analysis",
                "timestamp": datetime.now().isoformat(),
                "security_analysis": security_analysis,
            }
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Security analysis failed: {str(e)}"
            )

    @app.post("/comprehensive_codebase_analysis")
    async def comprehensive_codebase_analysis_endpoint(
        request: DetailedAnalysisRequest,
    ) -> Dict[str, Any]:
        """
        Ultimate comprehensive analysis combining all advanced features including:
        - Call graph analysis
        - Advanced metrics
        - Security analysis
        - Code views for critical files
        - Performance optimization suggestions
        """
        if not ADVANCED_FEATURES_AVAILABLE:
            raise HTTPException(
                status_code=503, detail="Advanced features not available"
            )

        try:
            codebase = Codebase.from_repo(request.repo_url)

            print("🚀 Starting comprehensive codebase analysis...")

            results = {
                "repo_url": request.repo_url,
                "analysis_type": "comprehensive_codebase_analysis",
                "timestamp": datetime.now().isoformat(),
                "summary": get_codebase_summary(codebase),
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
                file_path = critical_file["file_path"]
                target_file = None
                for file in codebase.files:
                    if file.filepath == file_path:
                        target_file = file
                        break

                if target_file:
                    results["code_views"][file_path] = generate_code_views(target_file)
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Critical file {file_path} not found in codebase",
                    )

            print("✅ Comprehensive analysis complete!")

            return results

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Comprehensive analysis failed: {str(e)}"
            )

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
                "comprehensive_analysis": "/comprehensive_codebase_analysis",
            },
            "features": {
                "call_graph_generation": ADVANCED_FEATURES_AVAILABLE,
                "multi_view_analysis": ADVANCED_FEATURES_AVAILABLE,
                "advanced_metrics": ADVANCED_FEATURES_AVAILABLE,
                "security_pattern_detection": ADVANCED_FEATURES_AVAILABLE,
                "performance_optimization": ADVANCED_FEATURES_AVAILABLE,
                "incremental_parsing": ADVANCED_FEATURES_AVAILABLE,
                "caching_system": ADVANCED_FEATURES_AVAILABLE,
                "error_validation": ADVANCED_FEATURES_AVAILABLE,
            },
        }


@fastapi_app.post("/analyze_repo")
async def analyze_repo(request: RepoRequest) -> Dict[str, Any]:
    """Analyze a repository and return comprehensive metrics."""
    repo_url = request.repo_url
    codebase = Codebase.from_repo(repo_url)

    num_files = len(codebase.files(extensions="*"))
    num_functions = len(codebase.functions)
    num_classes = len(codebase.classes)

    total_loc = total_lloc = total_sloc = total_comments = 0
    total_complexity = 0
    total_volume = 0
    total_mi = 0
    total_doi = 0

    monthly_commits = get_monthly_commits(repo_url)
    print(monthly_commits)

    for file in codebase.files:
        loc, lloc, sloc, comments = count_lines(file.source)
        total_loc += loc
        total_lloc += lloc
        total_sloc += sloc
        total_comments += comments

    callables = codebase.functions + [m for c in codebase.classes for m in c.methods]

    num_callables = 0
    for func in callables:
        if not hasattr(func, "code_block"):
            continue

        complexity = calculate_cyclomatic_complexity(func)
        operators, operands = get_operators_and_operands(func)
        volume, _, _, _, _ = calculate_halstead_volume(operators, operands)
        loc = len(func.code_block.source.splitlines())
        mi_score = calculate_maintainability_index(volume, complexity, loc)

        total_complexity += complexity
        total_volume += volume
        total_mi += mi_score
        num_callables += 1

    for cls in codebase.classes:
        doi = calculate_doi(cls)
        total_doi += doi

    desc = get_github_repo_description(repo_url)

    results = {
        "repo_url": repo_url,
        "line_metrics": {
            "total": {
                "loc": total_loc,
                "lloc": total_lloc,
                "sloc": total_sloc,
                "comments": total_comments,
                "comment_density": (total_comments / total_loc * 100)
                if total_loc > 0
                else 0,
            },
        },
        "cyclomatic_complexity": {
            "average": total_complexity if num_callables > 0 else 0,
        },
        "depth_of_inheritance": {
            "average": total_doi / len(codebase.classes) if codebase.classes else 0,
        },
        "halstead_metrics": {
            "total_volume": int(total_volume),
            "average_volume": int(total_volume / num_callables)
            if num_callables > 0
            else 0,
        },
        "maintainability_index": {
            "average": int(total_mi / num_callables) if num_callables > 0 else 0,
        },
        "description": desc,
        "num_files": num_files,
        "num_functions": num_functions,
        "num_classes": num_classes,
        "monthly_commits": monthly_commits,
    }

    return results


@fastapi_app.post("/comprehensive_analysis")
async def comprehensive_analysis(request: DetailedAnalysisRequest) -> Dict[str, Any]:
    """
    Perform comprehensive codebase analysis including:
    - Entry point detection
    - Critical file identification
    - Code issue analysis
    - Dependency graph analysis
    """
    try:
        codebase = Codebase.from_repo(request.repo_url)

        result = {
            "repo_url": request.repo_url,
            "analysis_timestamp": datetime.now().isoformat(),
            "summary": get_codebase_summary(codebase),
        }

        if request.include_entrypoints:
            print("Detecting entry points...")
            entrypoints = detect_entrypoints(codebase)
            result["entrypoints"] = {
                "total_count": len(entrypoints),
                "by_type": {
                    entry_type: len([ep for ep in entrypoints if ep.type == entry_type])
                    for entry_type in ["main", "cli", "web_endpoint", "test", "script"]
                },
                "details": [ep.dict() for ep in entrypoints],
            }

        if request.include_critical_files:
            print("Identifying critical files...")
            critical_files = identify_critical_files(codebase)
            result["critical_files"] = {
                "total_count": len(critical_files),
                "top_10": [cf.dict() for cf in critical_files[:10]],
                "summary": {
                    "avg_importance_score": sum(
                        cf.importance_score for cf in critical_files
                    )
                    / len(critical_files)
                    if critical_files
                    else 0,
                    "high_importance_count": len(
                        [cf for cf in critical_files if cf.importance_score > 70]
                    ),
                    "medium_importance_count": len(
                        [cf for cf in critical_files if 40 <= cf.importance_score <= 70]
                    ),
                    "low_importance_count": len(
                        [cf for cf in critical_files if cf.importance_score < 40]
                    ),
                },
            }

        if request.include_issues:
            print("Analyzing code issues...")
            issues = analyze_code_issues(codebase, request.max_issues)
            result["issues"] = {
                "total_count": len(issues),
                "by_severity": {
                    severity.value: len(
                        [issue for issue in issues if issue.severity == severity]
                    )
                    for severity in IssueSeverity
                },
                "by_type": {
                    issue_type.value: len(
                        [issue for issue in issues if issue.type == issue_type]
                    )
                    for issue_type in IssueType
                },
                "critical_issues": [
                    issue.dict()
                    for issue in issues
                    if issue.severity == IssueSeverity.CRITICAL
                ],
                "high_priority_issues": [
                    issue.dict()
                    for issue in issues
                    if issue.severity == IssueSeverity.HIGH
                ][:20],
                "all_issues": [issue.dict() for issue in issues]
                if len(issues) <= 50
                else [issue.dict() for issue in issues[:50]],
            }

        if request.include_dependency_graph:
            print("Building dependency graph...")
            dependency_graph = build_dependency_graph(codebase)

            # Get top nodes by centrality
            top_central_nodes = sorted(
                dependency_graph.values(),
                key=lambda x: x.centrality_score,
                reverse=True,
            )[:20]

            result["dependency_graph"] = {
                "total_nodes": len(dependency_graph),
                "node_types": {
                    "files": len(
                        [n for n in dependency_graph.values() if n.type == "file"]
                    ),
                    "functions": len(
                        [n for n in dependency_graph.values() if n.type == "function"]
                    ),
                    "classes": len(
                        [n for n in dependency_graph.values() if n.type == "class"]
                    ),
                },
                "most_central_nodes": [node.dict() for node in top_central_nodes],
                "graph_metrics": {
                    "avg_centrality": sum(
                        n.centrality_score for n in dependency_graph.values()
                    )
                    / len(dependency_graph),
                    "max_centrality": max(
                        n.centrality_score for n in dependency_graph.values()
                    )
                    if dependency_graph
                    else 0,
                    "highly_connected_nodes": len(
                        [
                            n
                            for n in dependency_graph.values()
                            if n.centrality_score > 0.1
                        ]
                    ),
                },
            }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@fastapi_app.post("/analyze_file")
async def analyze_file(request: FileAnalysisRequest) -> Dict[str, Any]:
    """Get detailed analysis for a specific file in the repository."""
    try:
        codebase = Codebase.from_repo(request.repo_url)
        return get_file_detailed_analysis(codebase, request.file_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File analysis failed: {str(e)}")


@fastapi_app.post("/detect_entrypoints")
async def detect_entrypoints_endpoint(request: RepoRequest) -> Dict[str, Any]:
    """Detect all entry points in the codebase."""
    try:
        codebase = Codebase.from_repo(request.repo_url)
        entrypoints = detect_entrypoints(codebase)

        return {
            "repo_url": request.repo_url,
            "total_entrypoints": len(entrypoints),
            "entrypoints_by_type": {
                entry_type: [ep.dict() for ep in entrypoints if ep.type == entry_type]
                for entry_type in ["main", "cli", "web_endpoint", "test", "script"]
            },
            "all_entrypoints": [ep.dict() for ep in entrypoints],
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Entry point detection failed: {str(e)}"
        )


@fastapi_app.post("/identify_critical_files")
async def identify_critical_files_endpoint(request: RepoRequest) -> Dict[str, Any]:
    """Identify the most critical files in the codebase."""
    try:
        codebase = Codebase.from_repo(request.repo_url)
        critical_files = identify_critical_files(codebase)

        return {
            "repo_url": request.repo_url,
            "total_files_analyzed": len(list(codebase.files)),
            "critical_files_count": len(critical_files),
            "critical_files": [cf.dict() for cf in critical_files],
            "summary": {
                "avg_importance_score": sum(
                    cf.importance_score for cf in critical_files
                )
                / len(critical_files)
                if critical_files
                else 0,
                "highest_score": max(cf.importance_score for cf in critical_files)
                if critical_files
                else 0,
                "files_above_80": len(
                    [cf for cf in critical_files if cf.importance_score > 80]
                ),
                "files_above_60": len(
                    [cf for cf in critical_files if cf.importance_score > 60]
                ),
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Critical file identification failed: {str(e)}"
        )


@fastapi_app.post("/analyze_issues")
async def analyze_issues_endpoint(request: DetailedAnalysisRequest) -> Dict[str, Any]:
    """Comprehensive code issue analysis."""
    try:
        codebase = Codebase.from_repo(request.repo_url)
        issues = analyze_code_issues(codebase, request.max_issues)

        # Group issues by file for better organization
        issues_by_file = defaultdict(list)
        for issue in issues:
            issues_by_file[issue.file_path].append(issue)

        # Get severity statistics
        severity_stats = {severity.value: 0 for severity in IssueSeverity}
        type_stats = {issue_type.value: 0 for issue_type in IssueType}

        for issue in issues:
            severity_stats[issue.severity.value] += 1
            type_stats[issue.type.value] += 1

        return {
            "repo_url": request.repo_url,
            "analysis_summary": {
                "total_issues": len(issues),
                "files_with_issues": len(issues_by_file),
                "critical_issues": severity_stats[IssueSeverity.CRITICAL.value],
                "high_priority_issues": severity_stats[IssueSeverity.HIGH.value],
                "medium_priority_issues": severity_stats[IssueSeverity.MEDIUM.value],
                "low_priority_issues": severity_stats[IssueSeverity.LOW.value],
            },
            "issues_by_severity": severity_stats,
            "issues_by_type": type_stats,
            "issues_by_file": {
                file_path: {
                    "issue_count": len(file_issues),
                    "critical_count": len(
                        [i for i in file_issues if i.severity == IssueSeverity.CRITICAL]
                    ),
                    "high_count": len(
                        [i for i in file_issues if i.severity == IssueSeverity.HIGH]
                    ),
                    "issues": [issue.dict() for issue in file_issues],
                }
                for file_path, file_issues in issues_by_file.items()
            },
            "most_problematic_files": [
                {
                    "file_path": file_path,
                    "total_issues": len(file_issues),
                    "critical_issues": len(
                        [i for i in file_issues if i.severity == IssueSeverity.CRITICAL]
                    ),
                    "high_issues": len(
                        [i for i in file_issues if i.severity == IssueSeverity.HIGH]
                    ),
                }
                for file_path, file_issues in sorted(
                    issues_by_file.items(), key=lambda x: len(x[1]), reverse=True
                )[:10]
            ],
            "all_issues": [issue.dict() for issue in issues],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Issue analysis failed: {str(e)}")


@fastapi_app.post("/dependency_graph")
async def dependency_graph_endpoint(request: RepoRequest) -> Dict[str, Any]:
    """Build and analyze the dependency graph of the codebase."""
    try:
        codebase = Codebase.from_repo(request.repo_url)
        dependency_graph = build_dependency_graph(codebase)

        # Calculate graph statistics
        total_edges = sum(len(node.dependencies) for node in dependency_graph.values())

        # Find most connected nodes
        most_connected = sorted(
            dependency_graph.values(),
            key=lambda x: len(x.dependencies) + len(x.dependents),
            reverse=True,
        )[:20]

        # Find nodes with highest centrality
        most_central = sorted(
            dependency_graph.values(), key=lambda x: x.centrality_score, reverse=True
        )[:20]

        return {
            "repo_url": request.repo_url,
            "graph_statistics": {
                "total_nodes": len(dependency_graph),
                "total_edges": total_edges,
                "average_connections": total_edges / len(dependency_graph)
                if dependency_graph
                else 0,
                "node_types": {
                    "files": len(
                        [n for n in dependency_graph.values() if n.type == "file"]
                    ),
                    "functions": len(
                        [n for n in dependency_graph.values() if n.type == "function"]
                    ),
                    "classes": len(
                        [n for n in dependency_graph.values() if n.type == "class"]
                    ),
                },
            },
            "most_connected_nodes": [
                {
                    "name": node.name,
                    "type": node.type,
                    "file_path": node.file_path,
                    "total_connections": len(node.dependencies) + len(node.dependents),
                    "dependencies_count": len(node.dependencies),
                    "dependents_count": len(node.dependents),
                }
                for node in most_connected
            ],
            "most_central_nodes": [node.dict() for node in most_central],
            "dependency_graph": {
                node_id: node.dict() for node_id, node in dependency_graph.items()
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Dependency graph analysis failed: {str(e)}"
        )


@fastapi_app.get("/health")
async def health_check():
    """Health check endpoint with enhanced graph-sitter capabilities."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "advanced_analysis_available": ADVANCED_ANALYSIS_AVAILABLE,
        "graph_sitter_version": getattr(graph_sitter, "__version__", "unknown"),
        "enhanced_features": {
            "working_imports": [
                "WhileStatement",
                "BinaryExpression",
                "UnaryExpression",
                "ComparisonExpression",
                "Expression",
                "Name",
                "String",
                "Value",
            ],
            "advanced_modules": [
                "ChainedAttribute",
                "DefinedName",
                "Builtin",
                "Assignment (conditional)",
                "Export (conditional)",
                "Directory (conditional)",
                "Interface (conditional)",
            ],
            "language_support": [
                "Python analyzer (conditional)",
                "TypeScript analyzer (conditional)",
            ],
        },
        "api_endpoints": {
            "basic": [
                "/analyze_repo",
                "/comprehensive_analysis",
                "/analyze_file",
                "/detect_entrypoints",
                "/identify_critical_files",
                "/analyze_issues",
                "/dependency_graph",
            ],
            "advanced": [
                "/advanced_semantic_analysis",
                "/advanced_dependency_analysis",
                "/advanced_architectural_analysis",
                "/language_specific_analysis",
                "/advanced_performance_analysis",
                "/comprehensive_error_analysis",
                "/ultimate_codebase_analysis",
            ],
        },
    }


# API Endpoints
@fastapi_app.post("/analyze_repo")
async def analyze_repo_endpoint(request: RepoRequest):
    codebase = Codebase.from_repo(request.repo_url)
    num_files = len(list(codebase.files))
    num_functions = len(list(codebase.functions))
    num_classes = len(list(codebase.classes))
    total_loc, total_sloc, total_comments = 0, 0, 0
    for file in codebase.files:
        loc, sloc, _, comments = analysis.count_lines(file.source)
        total_loc += loc
        total_sloc += sloc
        total_comments += comments

    return {
        "repo_url": request.repo_url,
        "description": get_github_repo_description(request.repo_url),
        "num_files": num_files,
        "num_functions": num_functions,
        "num_classes": num_classes,
        "line_metrics": {
            "loc": total_loc,
            "sloc": total_sloc,
            "comments": total_comments,
        },
        "monthly_commits": get_monthly_commits(request.repo_url),
    }


@fastapi_app.post("/comprehensive_analysis")
async def comprehensive_analysis_endpoint(request: DetailedAnalysisRequest):
    try:
        codebase = Codebase.from_repo(request.repo_url)
        result = {
            "repo_url": request.repo_url,
            "summary": get_codebase_summary(codebase),
        }
        if request.include_entrypoints:
            result["entrypoints"] = analysis.detect_entrypoints(codebase)
        if request.include_critical_files:
            result["critical_files"] = analysis.identify_critical_files(codebase)
        if request.include_issues:
            result["issues"] = analysis.analyze_code_issues(
                codebase, request.max_issues
            )
        if request.include_dependency_graph:
            result["dependency_graph"] = analysis.build_dependency_graph(codebase)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@fastapi_app.post("/generate_call_graph")
async def generate_call_graph_endpoint(request: RepoRequest):
    try:
        codebase = Codebase.from_repo(request.repo_url)
        return {"call_graph": analysis.generate_call_graph(codebase)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@fastapi_app.post("/calculate_advanced_metrics")
async def calculate_advanced_metrics_endpoint(request: RepoRequest):
    try:
        codebase = Codebase.from_repo(request.repo_url)
        return {"metrics": analysis.calculate_advanced_metrics(codebase)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@fastapi_app.post("/analyze_security_patterns")
async def analyze_security_patterns_endpoint(request: RepoRequest):
    try:
        codebase = Codebase.from_repo(request.repo_url)
        return {"security_analysis": analysis.analyze_security_patterns(codebase)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@fastapi_app.post("/comprehensive_error_analysis")
async def comprehensive_error_analysis_endpoint(request: DetailedAnalysisRequest):
    try:
        codebase = Codebase.from_repo(request.repo_url)
        return analysis.comprehensive_error_context_analysis(
            codebase, request.max_issues
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@fastapi_app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "graph_sitter_version": getattr(graph_sitter, "__version__", "unknown"),
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
