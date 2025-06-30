#!/usr/bin/env python3
"""
Clean API Module

This is the main API module that imports from consolidated analysis.py and visualize.py
modules to provide a clean, non-redundant interface for codebase analytics.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Tuple, Any, Optional
import requests
from datetime import datetime, timedelta
import subprocess
import os
import tempfile
from fastapi.middleware.cors import CORSMiddleware
import modal
from pathlib import Path

# Import from Codegen SDK
from codegen.sdk.core.codebase import Codebase
from codegen.sdk.core.class_definition import Class
from codegen.sdk.core.file import SourceFile
from codegen.sdk.core.function import Function
from codegen.sdk.core.symbol import Symbol

# Import from our consolidated modules
from analysis import (
    InheritanceAnalysis,
    RecursionAnalysis,
    SymbolInfo,
    calculate_cyclomatic_complexity,
    calculate_doi,
    get_operators_and_operands,
    calculate_halstead_volume,
    count_lines,
    calculate_maintainability_index,
    get_maintainability_rank,
    analyze_inheritance_patterns,
    analyze_recursive_functions,
    analyze_file_issues,
    build_repo_structure,
    get_file_type,
    get_detailed_symbol_context,
    get_max_call_chain
)

from visualize import (
    VisualizationType,
    OutputFormat,
    VisualizationConfig,
    create_call_graph,
    create_dependency_graph,
    create_class_hierarchy,
    create_complexity_heatmap,
    create_issues_heatmap,
    create_blast_radius,
    generate_all_visualizations,
    get_visualization_summary
)

# Modal setup
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "codegen", "fastapi", "uvicorn", "gitpython", "requests", "pydantic", "datetime",
        "networkx", "matplotlib"
    )
)

app = modal.App(name="analytics-app", image=image)
fastapi_app = FastAPI()

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models

class CodebaseStats(BaseModel):
    test_functions_count: int
    test_classes_count: int
    tests_per_file: float
    total_classes: int
    total_functions: int
    total_imports: int
    deepest_inheritance_class: Optional[Dict]
    recursive_functions: List[str]
    most_called_function: Dict
    function_with_most_calls: Dict
    unused_functions: List[Dict]
    dead_code: List[Dict]

class FileTestStats(BaseModel):
    filepath: str
    test_class_count: int
    file_length: int
    function_count: int

class FunctionContext(BaseModel):
    implementation: Dict
    dependencies: List[Dict]
    usages: List[Dict]

class TestAnalysis(BaseModel):
    total_test_functions: int
    total_test_classes: int
    tests_per_file: float
    top_test_files: List[Dict[str, Any]]

class FunctionAnalysis(BaseModel):
    total_functions: int
    most_called_function: Dict[str, Any]
    function_with_most_calls: Dict[str, Any]
    recursive_functions: List[str]
    unused_functions: List[Dict[str, str]]
    dead_code: List[Dict[str, str]]

class ClassAnalysis(BaseModel):
    total_classes: int
    deepest_inheritance: Optional[Dict[str, Any]]
    total_imports: int

class FileIssue(BaseModel):
    critical: List[Dict[str, str]]
    major: List[Dict[str, str]]
    minor: List[Dict[str, str]]

class ExtendedAnalysis(BaseModel):
    test_analysis: TestAnalysis
    function_analysis: FunctionAnalysis
    class_analysis: ClassAnalysis
    file_issues: Dict[str, FileIssue]
    repo_structure: Dict[str, Any]

class RepoRequest(BaseModel):
    repo_url: str

class Symbol(BaseModel):
    id: str
    name: str
    type: str  # 'function', 'class', or 'variable'
    filepath: str
    start_line: int
    end_line: int
    issues: Optional[List[Dict[str, str]]] = None

class FileNode(BaseModel):
    name: str
    type: str  # 'file' or 'directory'
    path: str
    issues: Optional[Dict[str, int]] = None
    symbols: Optional[List[Symbol]] = None
    children: Optional[Dict[str, 'FileNode']] = None

class AnalysisResponse(BaseModel):
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
    
    # New analysis features
    inheritance_analysis: InheritanceAnalysis
    recursion_analysis: RecursionAnalysis
    
    # Repository structure with symbols
    repo_structure: FileNode

class VisualizationRequest(BaseModel):
    repo_url: str
    visualization_type: VisualizationType
    config: Optional[Dict[str, Any]] = None

class VisualizationResponse(BaseModel):
    visualization_type: str
    data: Dict[str, Any]
    config: Dict[str, Any]

# Utility Functions

def get_monthly_commits(repo_path: str) -> Dict[str, int]:
    """Get the number of commits per month for the last 12 months."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    date_format = "%Y-%m-%d"
    since_date = start_date.strftime(date_format)
    until_date = end_date.strftime(date_format)
    repo_path = "https://github.com/" + repo_path

    try:
        original_dir = os.getcwd()

        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(["git", "clone", repo_path, temp_dir], check=True)
            os.chdir(temp_dir)

            cmd = [
                "git",
                "log",
                f"--since={since_date}",
                f"--until={until_date}",
                "--format=%aI",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            commit_dates = result.stdout.strip().split("\n")

            monthly_counts = {}
            current_date = start_date
            while current_date <= end_date:
                month_key = current_date.strftime("%Y-%m")
                monthly_counts[month_key] = 0
                current_date = (
                    current_date.replace(day=1) + timedelta(days=32)
                ).replace(day=1)

            for date_str in commit_dates:
                if date_str:  # Skip empty lines
                    commit_date = datetime.fromisoformat(date_str.strip())
                    month_key = commit_date.strftime("%Y-%m")
                    if month_key in monthly_counts:
                        monthly_counts[month_key] += 1

            os.chdir(original_dir)
            return dict(sorted(monthly_counts.items()))

    except subprocess.CalledProcessError as e:
        print(f"Error executing git command: {e}")
        return {}
    except Exception as e:
        print(f"Error processing git commits: {e}")
        return {}
    finally:
        try:
            os.chdir(original_dir)
        except:
            pass

def get_github_repo_description(repo_url: str) -> str:
    """Get repository description from GitHub API."""
    try:
        api_url = f"https://api.github.com/repos/{repo_url}"
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            return data.get("description", "No description available")
        else:
            return "Description not available"
    except Exception as e:
        print(f"Error fetching repo description: {e}")
        return "Description not available"

# Helper functions for symbol lookup (simplified implementations)
def get_function_by_id(function_id: str) -> Function:
    """Get function by ID - simplified implementation."""
    # This would need to be implemented with proper codebase context
    # For now, return a mock function
    raise HTTPException(status_code=501, detail="Function lookup not implemented")

def get_symbol_by_id(symbol_id: str) -> Symbol:
    """Get symbol by ID - simplified implementation."""
    # This would need to be implemented with proper codebase context
    # For now, return a mock symbol
    raise HTTPException(status_code=501, detail="Symbol lookup not implemented")

# API Endpoints

@fastapi_app.post("/analyze_repo")
async def analyze_repo(request: RepoRequest) -> AnalysisResponse:
    """Single entry point for repository analysis."""
    repo_url = request.repo_url
    
    # Validate repo URL format
    if not repo_url or '/' not in repo_url:
        raise HTTPException(status_code=400, detail="Repository URL must be in format 'owner/repo'")
    
    # Remove any GitHub URL prefix if present
    if repo_url.startswith('https://github.com/'):
        repo_url = repo_url.replace('https://github.com/', '')
    if repo_url.endswith('.git'):
        repo_url = repo_url[:-4]
    
    # Ensure it's in owner/repo format
    parts = repo_url.split('/')
    if len(parts) < 2:
        raise HTTPException(status_code=400, detail="Repository URL must be in format 'owner/repo'")
    
    repo_url = f"{parts[0]}/{parts[1]}"  # Take only owner/repo part
    
    try:
        codebase = Codebase.from_repo(repo_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to analyze repository: {str(e)}")

    # Basic analysis
    num_files = len(codebase.files(extensions="*"))
    num_functions = len(codebase.functions)
    num_classes = len(codebase.classes)

    total_loc = total_lloc = total_sloc = total_comments = 0
    total_complexity = 0
    total_volume = 0
    total_mi = 0
    total_doi = 0

    monthly_commits = get_monthly_commits(repo_url)

    # Analyze files and collect symbols
    file_issues = {}
    file_symbols = {}
    
    for file in codebase.files:
        # Line metrics
        loc, lloc, sloc, comments = count_lines(file.source)
        total_loc += loc
        total_lloc += lloc
        total_sloc += sloc
        total_comments += comments

        # Analyze issues
        issues = analyze_file_issues(file)
        if any(len(v) > 0 for v in issues.values()):
            file_issues[file.filepath] = issues

        # Collect symbols
        symbols = []
        
        # Add functions as symbols
        for func in file.functions:
            issues = []
            
            # Check for issues
            if not any(func.name in str(usage) for usage in func.usages):
                issues.append({
                    'type': 'minor',
                    'message': f'Unused function'
                })
            
            if hasattr(func, 'code_block'):
                code = func.code_block.source
                if 'None' in code and not any(s in code for s in ['is None', '== None', '!= None']):
                    issues.append({
                        'type': 'critical',
                        'message': f'Potential unsafe null reference'
                    })
                
                if 'TODO' in code or 'FIXME' in code:
                    issues.append({
                        'type': 'major',
                        'message': f'Incomplete implementation'
                    })

            symbols.append(Symbol(
                id=str(hash(func.name + file.filepath)),
                name=func.name,
                type='function',
                filepath=file.filepath,
                start_line=func.start_point[0] if hasattr(func, 'start_point') else 0,
                end_line=func.end_point[0] if hasattr(func, 'end_point') else 0,
                issues=issues if issues else None
            ))
        
        # Add classes as symbols
        for cls in file.classes:
            symbols.append(Symbol(
                id=str(hash(cls.name + file.filepath)),
                name=cls.name,
                type='class',
                filepath=file.filepath,
                start_line=cls.start_point[0] if hasattr(cls, 'start_point') else 0,
                end_line=cls.end_point[0] if hasattr(cls, 'end_point') else 0
            ))
        
        if symbols:
            file_symbols[file.filepath] = symbols

    # Build repository structure with symbols
    repo_structure = build_repo_structure(codebase.files, file_issues, file_symbols)

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
    
    # Perform new analysis features
    inheritance_analysis = analyze_inheritance_patterns(codebase)
    recursion_analysis = analyze_recursive_functions(codebase)

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
                "comment_density": (total_comments / total_loc * 100)
                if total_loc > 0
                else 0,
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
            "average_volume": int(total_volume / num_callables)
            if num_callables > 0
            else 0,
        },
        maintainability_index={
            "average": int(total_mi / num_callables) if num_callables > 0 else 0,
        },
        monthly_commits=monthly_commits,
        inheritance_analysis=inheritance_analysis,
        recursion_analysis=recursion_analysis,
        repo_structure=repo_structure
    )

@fastapi_app.post("/visualize")
async def create_visualization(request: VisualizationRequest) -> VisualizationResponse:
    """Create a visualization for a repository."""
    repo_url = request.repo_url
    viz_type = request.visualization_type
    config_dict = request.config or {}
    
    # Validate repo URL format
    if not repo_url or '/' not in repo_url:
        raise HTTPException(status_code=400, detail="Repository URL must be in format 'owner/repo'")
    
    # Clean repo URL
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

    # Create visualization config
    config = VisualizationConfig(**config_dict)
    
    # Generate visualization based on type
    try:
        if viz_type == VisualizationType.CALL_GRAPH:
            viz_data = create_call_graph(codebase, config=config)
        elif viz_type == VisualizationType.DEPENDENCY_GRAPH:
            viz_data = create_dependency_graph(codebase, config=config)
        elif viz_type == VisualizationType.CLASS_HIERARCHY:
            viz_data = create_class_hierarchy(codebase, config=config)
        elif viz_type == VisualizationType.COMPLEXITY_HEATMAP:
            viz_data = create_complexity_heatmap(codebase, config=config)
        elif viz_type == VisualizationType.BLAST_RADIUS:
            # For blast radius, we need a symbol name
            symbol_name = config_dict.get('symbol_name')
            if not symbol_name:
                raise HTTPException(status_code=400, detail="symbol_name required for blast radius visualization")
            viz_data = create_blast_radius(codebase, symbol_name, config=config)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported visualization type: {viz_type}")
        
        return VisualizationResponse(
            visualization_type=viz_type.value,
            data=viz_data,
            config=config.__dict__
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create visualization: {str(e)}")

@fastapi_app.get("/visualizations/summary/{repo_url:path}")
async def get_visualizations_summary(repo_url: str):
    """Get a summary of available visualizations for a repository."""
    # Clean repo URL
    if repo_url.startswith('https://github.com/'):
        repo_url = repo_url.replace('https://github.com/', '')
    if repo_url.endswith('.git'):
        repo_url = repo_url[:-4]
    
    try:
        codebase = Codebase.from_repo(repo_url)
        summary = get_visualization_summary(codebase)
        return summary
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to analyze repository: {str(e)}")

@fastapi_app.get("/function/{function_id}/call-chain")
async def get_function_call_chain(function_id: str) -> List[str]:
    """Get the maximum call chain for a function."""
    try:
        function = get_function_by_id(function_id)
        chain = get_max_call_chain(function)
        return [f.name for f in chain]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@fastapi_app.get("/function/{function_id}/context")
async def get_function_context(function_id: str) -> FunctionContext:
    """Get detailed context for a function."""
    try:
        function = get_function_by_id(function_id)
        context = get_detailed_symbol_context(function)
        return FunctionContext(
            implementation=context,
            dependencies=context.get('dependencies', []),
            usages=context.get('usages', [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@fastapi_app.get("/symbol/{symbol_id}/context")
async def get_symbol_context(symbol_id: str) -> Dict[str, Any]:
    """Get detailed context for any symbol."""
    try:
        symbol = get_symbol_by_id(symbol_id)
        return get_detailed_symbol_context(symbol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Legacy endpoints for backward compatibility
@fastapi_app.get("/api/codebase/stats")
async def get_codebase_stats(codebase_id: str) -> CodebaseStats:
    """Legacy endpoint - not implemented."""
    raise HTTPException(status_code=501, detail="Legacy endpoint not implemented")

@fastapi_app.get("/api/codebase/test-files")
async def get_test_file_stats(codebase_id: str) -> List[FileTestStats]:
    """Legacy endpoint - not implemented."""
    raise HTTPException(status_code=501, detail="Legacy endpoint not implemented")

@fastapi_app.get("/api/function/{function_id}/context")
async def get_function_context_legacy(function_id: str) -> FunctionContext:
    """Legacy endpoint - redirect to new endpoint."""
    return await get_function_context(function_id)

@fastapi_app.get("/api/function/{function_id}/call-chain")
async def get_function_call_chain_legacy(function_id: str) -> List[str]:
    """Legacy endpoint - redirect to new endpoint."""
    return await get_function_call_chain(function_id)

# Modal deployment
@app.function(image=image)
@modal.asgi_app()
def modal_app():
    return fastapi_app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)
