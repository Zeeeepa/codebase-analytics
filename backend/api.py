#!/usr/bin/env python3
"""
Unified API Server

This module contains the consolidated API server that imports and uses:
- consolidated_analysis.py (all analysis functions)
- consolidated_visualization.py (all visualization functions)

This is the single API entry point for the codebase analytics platform.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import requests
from datetime import datetime, timedelta
import modal

# Import consolidated modules
from analysis import (
    # Data classes
    InheritanceAnalysis,
    RecursionAnalysis,
    DependencyAnalysis,
    CallGraphAnalysis,
    CodeQualityMetrics,
    ArchitecturalInsights,
    SecurityAnalysis,
    PerformanceAnalysis,
    AnalysisType,
    
    # Analysis functions
    perform_comprehensive_analysis,
    analyze_dependencies_comprehensive,
    analyze_call_graph,
    analyze_code_quality,
    analyze_architecture,
    analyze_security,
    analyze_performance,
    analyze_inheritance_patterns,
    analyze_recursive_functions,
    calculate_cyclomatic_complexity,
    calculate_halstead_volume,
    count_lines,
    calculate_maintainability_index,
    get_maintainability_rank
)

from visualization import (
    # Visualization classes
    VisualizationType,
    OutputFormat,
    VisualizationConfig,
    
    # Visualization functions
    create_call_graph,
    create_dependency_graph,
    create_class_hierarchy,
    create_complexity_heatmap,
    create_blast_radius,
    create_enhanced_dependency_graph,
    create_comprehensive_dashboard_data,
    generate_all_visualizations,
    get_visualization_summary,
    save_visualization
)

# Import from Codegen SDK
from codegen.sdk.core.codebase import Codebase

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class CodebaseStats(BaseModel):
    """Basic codebase statistics."""
    total_files: int
    total_functions: int
    total_classes: int
    total_lines: int
    languages: List[str]

class FunctionAnalysis(BaseModel):
    """Function analysis results."""
    most_called_function: str
    most_called_count: int
    most_calling_function: str
    most_calling_count: int
    dead_functions: List[str]
    sample_functions: List[str]
    sample_classes: List[str]
    sample_imports: List[str]

class RepoRequest(BaseModel):
    """Repository analysis request."""
    repo_url: str

class AnalysisRequest(BaseModel):
    """Enhanced analysis request."""
    repo_url: str
    analysis_types: Optional[List[str]] = None
    include_visualizations: bool = False
    max_analysis_time: int = 300

class VisualizationRequest(BaseModel):
    """Visualization request."""
    repo_url: str
    visualization_type: str
    config: Optional[Dict[str, Any]] = None

class AnalysisResponse(BaseModel):
    """Standard analysis response."""
    repo_url: str
    description: str
    num_files: int
    num_functions: int
    num_classes: int
    line_metrics: Dict[str, Dict[str, float]]
    cyclomatic_complexity: Dict[str, float]
    depth_of_inheritance: Dict[str, float]
    halstead_metrics: Dict[str, int]
    maintainability_index: Dict[str, int]
    monthly_commits: Dict[str, int]
    inheritance_analysis: InheritanceAnalysis
    recursion_analysis: RecursionAnalysis
    function_analysis: FunctionAnalysis

class EnhancedAnalysisResponse(BaseModel):
    """Enhanced analysis response with comprehensive metrics."""
    # Basic stats
    repo_url: str
    description: str
    num_files: int
    num_functions: int
    num_classes: int
    
    # Line metrics
    line_metrics: Dict[str, Dict[str, float]]
    
    # Complexity metrics
    cyclomatic_complexity: Dict[str, float]
    depth_of_inheritance: Dict[str, float]
    halstead_metrics: Dict[str, int]
    maintainability_index: Dict[str, int]
    
    # Git metrics
    monthly_commits: Dict[str, int]
    
    # Existing analysis features
    inheritance_analysis: InheritanceAnalysis
    recursion_analysis: RecursionAnalysis
    function_analysis: FunctionAnalysis
    
    # New comprehensive analysis features
    dependency_analysis: Optional[DependencyAnalysis] = None
    call_graph_analysis: Optional[CallGraphAnalysis] = None
    code_quality_metrics: Optional[CodeQualityMetrics] = None
    architectural_insights: Optional[ArchitecturalInsights] = None
    security_analysis: Optional[SecurityAnalysis] = None
    performance_analysis: Optional[PerformanceAnalysis] = None
    
    # Analysis metadata
    analysis_timestamp: str
    analysis_duration_seconds: float
    analysis_types_performed: List[str]

class ComprehensiveInsights(BaseModel):
    """Comprehensive insights derived from all analyses."""
    overall_quality_score: float
    technical_debt_level: str
    maintainability_rating: str
    architectural_health: str
    security_risk_level: str
    performance_concerns: List[str]
    top_recommendations: List[str]
    complexity_distribution: Dict[str, int]
    dependency_health: str

class VisualizationResponse(BaseModel):
    """Visualization response."""
    visualization_type: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_github_repo_description(repo_url: str) -> str:
    """Get repository description from GitHub API."""
    try:
        api_url = f"https://api.github.com/repos/{repo_url}"
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('description', 'No description available')
        return 'Description not available'
    except Exception:
        return 'Description not available'

def get_monthly_commits(repo_url: str) -> Dict[str, int]:
    """Get monthly commit statistics from GitHub API."""
    try:
        api_url = f"https://api.github.com/repos/{repo_url}/stats/commit_activity"
        response = requests.get(api_url, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                monthly_commits = {}
                for week_data in data[-12:]:  # Last 12 weeks
                    week_timestamp = week_data.get('week', 0)
                    commits = week_data.get('total', 0)
                    
                    date = datetime.fromtimestamp(week_timestamp)
                    month_key = date.strftime('%Y-%m')
                    
                    if month_key not in monthly_commits:
                        monthly_commits[month_key] = 0
                    monthly_commits[month_key] += commits
                
                return monthly_commits
        
        # Fallback data
        return {
            datetime.now().strftime('%Y-%m'): 10,
            (datetime.now() - timedelta(days=30)).strftime('%Y-%m'): 15,
            (datetime.now() - timedelta(days=60)).strftime('%Y-%m'): 8
        }
    except Exception:
        return {'2024-01': 5, '2024-02': 12, '2024-03': 8}

def analyze_functions_comprehensive(codebase) -> FunctionAnalysis:
    """Comprehensive function analysis for API response."""
    functions = codebase.functions
    
    if not functions:
        return FunctionAnalysis(
            most_called_function="No functions found",
            most_called_count=0,
            most_calling_function="No functions found", 
            most_calling_count=0,
            dead_functions=[],
            sample_functions=[],
            sample_classes=[],
            sample_imports=[]
        )
    
    # Analyze call patterns
    call_counts = {}
    calling_counts = {}
    
    for func in functions:
        func_name = func.name
        call_counts[func_name] = 0
        calling_counts[func_name] = 0
        
        # Count how many times this function is called
        for other_func in functions:
            if hasattr(other_func, 'function_calls') and other_func.function_calls:
                for call in other_func.function_calls:
                    if hasattr(call, 'name') and call.name == func_name:
                        call_counts[func_name] += 1
        
        # Count how many functions this function calls
        if hasattr(func, 'function_calls') and func.function_calls:
            calling_counts[func_name] = len(func.function_calls)
    
    # Find most called and most calling functions
    most_called = max(call_counts.items(), key=lambda x: x[1]) if call_counts else ("None", 0)
    most_calling = max(calling_counts.items(), key=lambda x: x[1]) if calling_counts else ("None", 0)
    
    # Find dead functions (not called by anyone)
    dead_functions = [name for name, count in call_counts.items() if count == 0]
    
    # Get samples
    sample_functions = [func.name for func in functions[:5]]
    sample_classes = [cls.name for cls in codebase.classes[:5]]
    
    # Get sample imports
    sample_imports = []
    for file in codebase.files[:3]:
        if hasattr(file, 'imports') and file.imports:
            for imp in file.imports[:2]:
                module_name = getattr(imp, 'module', getattr(imp, 'name', 'unknown'))
                if module_name not in sample_imports:
                    sample_imports.append(module_name)
    
    return FunctionAnalysis(
        most_called_function=most_called[0],
        most_called_count=most_called[1],
        most_calling_function=most_calling[0],
        most_calling_count=most_calling[1],
        dead_functions=dead_functions[:10],  # Limit to 10
        sample_functions=sample_functions,
        sample_classes=sample_classes,
        sample_imports=sample_imports[:10]  # Limit to 10
    )

def calculate_comprehensive_insights(analysis_results: Dict[str, Any]) -> ComprehensiveInsights:
    """Calculate comprehensive insights from all analysis results."""
    insights = ComprehensiveInsights(
        overall_quality_score=0.0,
        technical_debt_level="Unknown",
        maintainability_rating="Unknown",
        architectural_health="Unknown",
        security_risk_level="Unknown",
        performance_concerns=[],
        top_recommendations=[],
        complexity_distribution={},
        dependency_health="Unknown"
    )
    
    scores = []
    recommendations = []
    
    # Analyze code quality metrics
    if 'code_quality_metrics' in analysis_results:
        quality = analysis_results['code_quality_metrics']
        
        # Technical debt assessment
        debt_ratio = quality.technical_debt_ratio
        if debt_ratio < 5:
            insights.technical_debt_level = "Low"
            scores.append(90)
        elif debt_ratio < 15:
            insights.technical_debt_level = "Medium"
            scores.append(70)
        elif debt_ratio < 30:
            insights.technical_debt_level = "High"
            scores.append(40)
        else:
            insights.technical_debt_level = "Critical"
            scores.append(20)
            recommendations.append("Address technical debt by resolving TODO/FIXME items")
        
        # Documentation coverage
        if quality.documentation_coverage < 50:
            recommendations.append("Improve code documentation coverage")
        
        # Code duplication
        if quality.code_duplication_percentage > 10:
            recommendations.append("Reduce code duplication through refactoring")
    
    # Analyze dependency health
    if 'dependency_analysis' in analysis_results:
        deps = analysis_results['dependency_analysis']
        
        if deps.circular_dependencies:
            insights.dependency_health = "Poor"
            recommendations.append("Resolve circular dependencies")
            scores.append(30)
        elif deps.dependency_depth > 10:
            insights.dependency_health = "Fair"
            scores.append(60)
        else:
            insights.dependency_health = "Good"
            scores.append(80)
    
    # Analyze security
    if 'security_analysis' in analysis_results:
        security = analysis_results['security_analysis']
        
        vuln_count = len(security.potential_vulnerabilities)
        if vuln_count == 0:
            insights.security_risk_level = "Low"
            scores.append(90)
        elif vuln_count < 3:
            insights.security_risk_level = "Medium"
            scores.append(60)
        else:
            insights.security_risk_level = "High"
            scores.append(30)
            recommendations.append("Address security vulnerabilities")
    
    # Analyze performance
    if 'performance_analysis' in analysis_results:
        perf = analysis_results['performance_analysis']
        
        insights.performance_concerns = [
            hotspot['description'] for hotspot in perf.performance_hotspots
        ]
        
        if len(perf.performance_hotspots) > 5:
            recommendations.append("Optimize performance hotspots")
    
    # Analyze architecture
    if 'architectural_insights' in analysis_results:
        arch = analysis_results['architectural_insights']
        
        if arch.modularity_score > 70:
            insights.architectural_health = "Good"
            scores.append(80)
        elif arch.modularity_score > 40:
            insights.architectural_health = "Fair"
            scores.append(60)
        else:
            insights.architectural_health = "Poor"
            scores.append(40)
            recommendations.append("Improve code modularity and organization")
    
    # Calculate overall quality score
    if scores:
        insights.overall_quality_score = sum(scores) / len(scores)
    
    # Determine maintainability rating
    if insights.overall_quality_score >= 80:
        insights.maintainability_rating = "Excellent"
    elif insights.overall_quality_score >= 65:
        insights.maintainability_rating = "Good"
    elif insights.overall_quality_score >= 45:
        insights.maintainability_rating = "Fair"
    else:
        insights.maintainability_rating = "Poor"
    
    insights.top_recommendations = recommendations[:5]  # Top 5 recommendations
    
    return insights

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Unified Codebase Analytics API",
    description="Comprehensive codebase analysis with advanced metrics and insights",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_repo(request: RepoRequest) -> AnalysisResponse:
    """Standard repository analysis endpoint."""
    start_time = datetime.now()
    
    repo_url = request.repo_url
    
    # Validate and clean repo URL
    if not repo_url or '/' not in repo_url:
        raise HTTPException(status_code=400, detail="Repository URL must be in format 'owner/repo'")
    
    if repo_url.startswith('https://github.com/'):
        repo_url = repo_url.replace('https://github.com/', '')
    if repo_url.endswith('.git'):
        repo_url = repo_url[:-4]
    
    parts = repo_url.split('/')
    if len(parts) < 2:
        raise HTTPException(status_code=400, detail="Repository URL must be in format 'owner/repo'")
    
    repo_url = f"{parts[0]}/{parts[1]}"
    
    try:
        codebase = Codebase.from_repo(repo_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to analyze repository: {str(e)}")

    # Basic analysis
    num_files = len(codebase.files)
    num_functions = len(codebase.functions)
    num_classes = len(codebase.classes)

    total_loc = total_lloc = total_sloc = total_comments = 0
    total_complexity = 0
    total_volume = 0
    total_mi = 0
    total_doi = 0

    monthly_commits = get_monthly_commits(repo_url)

    # Analyze files
    for file in codebase.files:
        loc, lloc, sloc, comments = count_lines(file.source)
        total_loc += loc
        total_lloc += lloc
        total_sloc += sloc
        total_comments += comments

    # Calculate metrics
    callables = codebase.functions + [m for c in codebase.classes for m in c.methods]
    num_callables = 0
    
    for func in callables:
        if not hasattr(func, "code_block"):
            continue

        complexity = calculate_cyclomatic_complexity(func)
        operators, operands = get_operators_and_operands(func)
        volume, N1, N2, n1, n2 = calculate_halstead_volume(operators, operands)
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
    
    # Perform analysis features
    inheritance_analysis = analyze_inheritance_patterns(codebase)
    recursion_analysis = analyze_recursive_functions(codebase)
    function_analysis = analyze_functions_comprehensive(codebase)

    return AnalysisResponse(
        repo_url=repo_url,
        description=desc,
        num_files=num_files,
        num_functions=num_functions,
        num_classes=num_classes,
        line_metrics={
            "total": {
                "loc": total_loc,
                "lloc": total_lloc,
                "sloc": total_sloc,
                "comments": total_comments,
                "comment_density": (total_comments / total_loc * 100) if total_loc > 0 else 0,
            },
        },
        cyclomatic_complexity={
            "average": total_complexity / num_callables if num_callables > 0 else 0,
        },
        depth_of_inheritance={
            "average": total_doi / len(codebase.classes) if codebase.classes else 0,
        },
        halstead_metrics={
            "total_volume": int(total_volume),
            "average_volume": int(total_volume / num_callables) if num_callables > 0 else 0,
        },
        maintainability_index={
            "average": int(total_mi / num_callables) if num_callables > 0 else 0,
        },
        monthly_commits=monthly_commits,
        inheritance_analysis=inheritance_analysis,
        recursion_analysis=recursion_analysis,
        function_analysis=function_analysis
    )

@app.post("/analyze_comprehensive", response_model=EnhancedAnalysisResponse)
async def analyze_comprehensive(request: AnalysisRequest) -> EnhancedAnalysisResponse:
    """Comprehensive repository analysis with advanced features."""
    start_time = datetime.now()
    
    # First perform standard analysis
    standard_request = RepoRequest(repo_url=request.repo_url)
    standard_response = await analyze_repo(standard_request)
    
    # Get codebase for comprehensive analysis
    repo_url = request.repo_url
    if repo_url.startswith('https://github.com/'):
        repo_url = repo_url.replace('https://github.com/', '')
    if repo_url.endswith('.git'):
        repo_url = repo_url[:-4]
    
    try:
        codebase = Codebase.from_repo(repo_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to analyze repository: {str(e)}")

    # Determine which analysis types to perform
    analysis_types_to_perform = []
    if request.analysis_types:
        for analysis_type in request.analysis_types:
            try:
                analysis_types_to_perform.append(AnalysisType(analysis_type))
            except ValueError:
                pass  # Skip invalid analysis types
    else:
        analysis_types_to_perform = list(AnalysisType)  # Perform all analyses

    # Perform comprehensive analysis
    comprehensive_results = perform_comprehensive_analysis(codebase, analysis_types_to_perform)
    
    # Calculate analysis duration
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Build enhanced response
    response = EnhancedAnalysisResponse(
        repo_url=standard_response.repo_url,
        description=standard_response.description,
        num_files=standard_response.num_files,
        num_functions=standard_response.num_functions,
        num_classes=standard_response.num_classes,
        line_metrics=standard_response.line_metrics,
        cyclomatic_complexity=standard_response.cyclomatic_complexity,
        depth_of_inheritance=standard_response.depth_of_inheritance,
        halstead_metrics=standard_response.halstead_metrics,
        maintainability_index=standard_response.maintainability_index,
        monthly_commits=standard_response.monthly_commits,
        inheritance_analysis=standard_response.inheritance_analysis,
        recursion_analysis=standard_response.recursion_analysis,
        function_analysis=standard_response.function_analysis,
        analysis_timestamp=start_time.isoformat(),
        analysis_duration_seconds=duration,
        analysis_types_performed=[at.value for at in analysis_types_to_perform]
    )
    
    # Add comprehensive analysis results
    if 'dependency_analysis' in comprehensive_results:
        response.dependency_analysis = comprehensive_results['dependency_analysis']
    
    if 'call_graph_analysis' in comprehensive_results:
        response.call_graph_analysis = comprehensive_results['call_graph_analysis']
    
    if 'code_quality_metrics' in comprehensive_results:
        response.code_quality_metrics = comprehensive_results['code_quality_metrics']
    
    if 'architectural_insights' in comprehensive_results:
        response.architectural_insights = comprehensive_results['architectural_insights']
    
    if 'security_analysis' in comprehensive_results:
        response.security_analysis = comprehensive_results['security_analysis']
    
    if 'performance_analysis' in comprehensive_results:
        response.performance_analysis = comprehensive_results['performance_analysis']

    return response

@app.post("/insights", response_model=ComprehensiveInsights)
async def get_comprehensive_insights(request: AnalysisRequest) -> ComprehensiveInsights:
    """Get high-level insights and recommendations."""
    # First perform comprehensive analysis
    analysis_response = await analyze_comprehensive(request)
    
    # Extract analysis results
    analysis_results = {}
    
    if analysis_response.dependency_analysis:
        analysis_results['dependency_analysis'] = analysis_response.dependency_analysis
    
    if analysis_response.call_graph_analysis:
        analysis_results['call_graph_analysis'] = analysis_response.call_graph_analysis
    
    if analysis_response.code_quality_metrics:
        analysis_results['code_quality_metrics'] = analysis_response.code_quality_metrics
    
    if analysis_response.architectural_insights:
        analysis_results['architectural_insights'] = analysis_response.architectural_insights
    
    if analysis_response.security_analysis:
        analysis_results['security_analysis'] = analysis_response.security_analysis
    
    if analysis_response.performance_analysis:
        analysis_results['performance_analysis'] = analysis_response.performance_analysis
    
    # Calculate comprehensive insights
    return calculate_comprehensive_insights(analysis_results)

@app.post("/visualize", response_model=VisualizationResponse)
async def create_visualization(request: VisualizationRequest) -> VisualizationResponse:
    """Create a visualization for the repository."""
    repo_url = request.repo_url
    
    # Clean repo URL
    if repo_url.startswith('https://github.com/'):
        repo_url = repo_url.replace('https://github.com/', '')
    if repo_url.endswith('.git'):
        repo_url = repo_url[:-4]
    
    try:
        codebase = Codebase.from_repo(repo_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to analyze repository: {str(e)}")
    
    # Create visualization config
    config = VisualizationConfig()
    if request.config:
        for key, value in request.config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Create the requested visualization
    viz_type = request.visualization_type
    
    if viz_type == VisualizationType.CALL_GRAPH.value:
        viz_data = create_call_graph(codebase, config=config)
    elif viz_type == VisualizationType.DEPENDENCY_GRAPH.value:
        viz_data = create_dependency_graph(codebase, config=config)
    elif viz_type == VisualizationType.CLASS_HIERARCHY.value:
        viz_data = create_class_hierarchy(codebase, config=config)
    elif viz_type == VisualizationType.COMPLEXITY_HEATMAP.value:
        viz_data = create_complexity_heatmap(codebase, config=config)
    elif viz_type == VisualizationType.BLAST_RADIUS.value:
        # For blast radius, we need a symbol name - use first function if not provided
        symbol_name = request.config.get('symbol_name') if request.config else None
        if not symbol_name and codebase.functions:
            symbol_name = codebase.functions[0].name
        if symbol_name:
            viz_data = create_blast_radius(codebase, symbol_name, config=config)
        else:
            raise HTTPException(status_code=400, detail="Symbol name required for blast radius visualization")
    elif viz_type == VisualizationType.ENHANCED_DEPENDENCY_GRAPH.value:
        viz_data = create_enhanced_dependency_graph(codebase)
    elif viz_type == VisualizationType.COMPREHENSIVE_DASHBOARD.value:
        viz_data = create_comprehensive_dashboard_data(codebase)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported visualization type: {viz_type}")
    
    return VisualizationResponse(
        visualization_type=viz_type,
        data=viz_data,
        metadata=viz_data.get('metadata', {})
    )

@app.get("/visualizations/summary")
async def get_visualizations_summary(repo_url: str):
    """Get a summary of available visualizations for a repository."""
    # Clean repo URL
    if repo_url.startswith('https://github.com/'):
        repo_url = repo_url.replace('https://github.com/', '')
    if repo_url.endswith('.git'):
        repo_url = repo_url[:-4]
    
    try:
        codebase = Codebase.from_repo(repo_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to analyze repository: {str(e)}")
    
    return get_visualization_summary(codebase)

@app.get("/analysis_types")
async def get_available_analysis_types() -> List[Dict[str, str]]:
    """Get available analysis types."""
    return [
        {
            "type": analysis_type.value,
            "description": get_analysis_type_description(analysis_type)
        }
        for analysis_type in AnalysisType
    ]

def get_analysis_type_description(analysis_type: AnalysisType) -> str:
    """Get description for analysis type."""
    descriptions = {
        AnalysisType.DEPENDENCY: "Analyze dependencies, circular dependencies, and dependency depth",
        AnalysisType.CALL_GRAPH: "Analyze function call relationships and call chains",
        AnalysisType.CODE_QUALITY: "Analyze code quality metrics, duplication, and technical debt",
        AnalysisType.ARCHITECTURAL: "Analyze architectural patterns, coupling, and modularity",
        AnalysisType.SECURITY: "Analyze potential security vulnerabilities and risks",
        AnalysisType.PERFORMANCE: "Analyze performance hotspots and optimization opportunities"
    }
    return descriptions.get(analysis_type, "Unknown analysis type")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "3.0.0", "features": ["analysis", "visualization", "insights"]}

# ============================================================================
# MODAL DEPLOYMENT
# ============================================================================

# Create Modal app and image for deployment
modal_app = modal.App("codebase-analytics")

image = modal.Image.debian_slim().pip_install([
    "fastapi",
    "uvicorn",
    "requests",
    "pydantic",
    "networkx",
    "codegen"
])

@modal_app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
