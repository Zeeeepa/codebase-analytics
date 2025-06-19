#!/usr/bin/env python3
"""
FastAPI server for the Codebase Analytics tool.
"""

import os
import tempfile
import subprocess
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from typing import Optional, Dict, Any, List

# Import consolidated analysis and visualization modules
from analysis import (
    analyze_codebase, analyze_code_quality, analyze_dependencies, analyze_call_graph,
    detect_issues, get_codebase_summary, get_dependency_graph, get_symbol_references,
    generate_dependency_graph_visualization, generate_call_graph_visualization,
    generate_issue_visualization, generate_code_quality_visualization,
    generate_repository_structure_visualization
)
from visualization import visualize_codebase, generate_html_report, run_visualization_analysis
from codegen.sdk.core.codebase import Codebase

# Create FastAPI app
app = FastAPI(
    title="Codebase Analytics API",
    description="API for analyzing codebases",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output directory
os.makedirs("output", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# Define request models
class AnalysisRequest(BaseModel):
    repo_url: str
    branch: Optional[str] = None
    output_dir: Optional[str] = "output"

class AnalysisResponse(BaseModel):
    status: str
    message: str
    analysis_id: Optional[str] = None

# Store analysis results
analysis_results = {}

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to the Codebase Analytics API"}

@app.get("/ui", response_class=HTMLResponse)
async def interactive_ui():
    """Serve the interactive analysis UI."""
    try:
        with open("../frontend/interactive-analysis.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>UI not found</h1>", status_code=404)

@app.get("/analyze/{repo_owner}/{repo_name}")
async def analyze_repository(repo_owner: str, repo_name: str):
    """
    Comprehensive analysis of a GitHub repository with all context.
    
    Args:
        repo_owner: GitHub repository owner
        repo_name: GitHub repository name
        
    Returns:
        Complete analysis results with interactive repository structure
    """
    try:
        # Construct repository URL
        repo_url = f"{repo_owner}/{repo_name}"
        
        # Load codebase using Codegen SDK
        codebase = Codebase.from_repo(repo_url)
        
        # Run comprehensive analysis
        print(f"🔍 Analyzing repository: {repo_url}")
        
        # 1. Detect issues
        issues = detect_issues(codebase)
        print(f"📊 Issues detected: {len(issues.issues)}")
        
        # 2. Analyze code quality
        code_quality = analyze_code_quality(codebase)
        print(f"📈 Code quality analyzed")
        
        # 3. Analyze dependencies
        dependencies = analyze_dependencies(codebase)
        print(f"🔗 Dependencies analyzed")
        
        # 4. Analyze call graph
        call_graph = analyze_call_graph(codebase)
        print(f"📞 Call graph analyzed")
        
        # 5. Generate visualizations
        dependency_viz = generate_dependency_graph_visualization(dependencies)
        call_graph_viz = generate_call_graph_visualization(call_graph)
        issue_viz = generate_issue_visualization(issues)
        quality_viz = generate_code_quality_visualization(code_quality)
        repo_structure_viz = generate_repository_structure_visualization(codebase)
        
        # 6. Build interactive repository structure with issue counts
        interactive_structure = build_interactive_repository_structure(codebase, issues)
        
        # 7. Get symbol details for interactive features
        symbol_details = build_symbol_details_map(codebase, issues, call_graph, dependencies)
        
        # Compile comprehensive response
        response = {
            "repository": {
                "owner": repo_owner,
                "name": repo_name,
                "url": f"https://github.com/{repo_url}",
                "total_files": len(codebase.files),
                "total_functions": len(codebase.functions),
                "total_classes": len(codebase.classes),
                "total_symbols": len(codebase.symbols)
            },
            "analysis": {
                "issues": {
                    "total": len(issues.issues),
                    "by_severity": issues.count_by_severity(),
                    "by_category": issues.count_by_category(),
                    "details": [
                        {
                            "id": issue.id,
                            "category": issue.category.value,
                            "severity": issue.severity.value,
                            "message": issue.message,
                            "file_path": str(issue.location.file_path),
                            "line_start": issue.location.line_start,
                            "line_end": issue.location.line_end
                        }
                        for issue in issues.issues
                    ]
                },
                "code_quality": {
                    "maintainability_index": code_quality.maintainability_index,
                    "cyclomatic_complexity": code_quality.cyclomatic_complexity,
                    "comment_density": code_quality.comment_density,
                    "source_lines_of_code": code_quality.source_lines_of_code,
                    "duplication_percentage": code_quality.duplication_percentage,
                    "technical_debt_ratio": code_quality.technical_debt_ratio
                },
                "dependencies": {
                    "total": dependencies.total_dependencies,
                    "circular": len(dependencies.circular_dependencies),
                    "external": len(dependencies.external_dependencies),
                    "internal": len(dependencies.internal_dependencies),
                    "depth": dependencies.dependency_depth,
                    "critical": len(dependencies.critical_dependencies)
                },
                "call_graph": {
                    "total_functions": call_graph.total_functions,
                    "entry_points": len(call_graph.entry_points),
                    "leaf_functions": len(call_graph.leaf_functions),
                    "max_call_depth": call_graph.max_call_depth,
                    "call_chains": len(call_graph.call_chains)
                }
            },
            "interactive_structure": interactive_structure,
            "symbol_details": symbol_details,
            "visualizations": {
                "dependency_graph": dependency_viz,
                "call_graph": call_graph_viz,
                "issues": issue_viz,
                "code_quality": quality_viz,
                "repository_structure": repo_structure_viz
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"❌ Error analyzing repository: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def build_interactive_repository_structure(codebase: Codebase, issues) -> Dict[str, Any]:
    """Build interactive repository structure with issue counts."""
    from pathlib import Path
    from collections import defaultdict
    
    # Group issues by file path
    issues_by_file = defaultdict(list)
    for issue in issues.issues:
        file_path = str(issue.location.file_path)
        issues_by_file[file_path].append(issue)
    
    # Build directory structure
    structure = {}
    
    for file in codebase.files:
        file_path = str(file.path)
        path_parts = Path(file_path).parts
        
        current = structure
        for part in path_parts[:-1]:  # All parts except filename
            if part not in current:
                current[part] = {"type": "directory", "children": {}, "issues": {"critical": 0, "major": 0, "minor": 0, "info": 0}}
            current = current[part]["children"]
        
        # Add file
        filename = path_parts[-1]
        file_issues = issues_by_file.get(file_path, [])
        issue_counts = {"critical": 0, "major": 0, "minor": 0, "info": 0}
        
        for issue in file_issues:
            severity = issue.severity.value.lower()
            if severity in issue_counts:
                issue_counts[severity] += 1
        
        current[filename] = {
            "type": "file",
            "path": file_path,
            "issues": issue_counts,
            "functions": [f.name for f in codebase.functions if f.filepath == file_path],
            "classes": [c.name for c in codebase.classes if c.filepath == file_path],
            "symbols": [s.name for s in codebase.symbols if s.filepath == file_path]
        }
        
        # Propagate issue counts up the directory tree
        propagate_issue_counts(structure, path_parts[:-1], issue_counts)
    
    return structure

def propagate_issue_counts(structure: Dict, path_parts: tuple, issue_counts: Dict):
    """Propagate issue counts up the directory tree."""
    current = structure
    for part in path_parts:
        if part in current and "issues" in current[part]:
            for severity, count in issue_counts.items():
                current[part]["issues"][severity] += count
        current = current[part]["children"]

def build_symbol_details_map(codebase: Codebase, issues, call_graph, dependencies) -> Dict[str, Any]:
    """Build detailed symbol information for interactive features."""
    symbol_details = {}
    
    # Process functions
    for func in codebase.functions:
        func_issues = [issue for issue in issues.issues if str(issue.location.file_path) == func.filepath and issue.location.line_start >= func.start_line and issue.location.line_start <= func.end_line]
        
        symbol_details[f"function:{func.name}"] = {
            "type": "function",
            "name": func.name,
            "filepath": func.filepath,
            "start_line": func.start_line,
            "end_line": func.end_line,
            "parameters": [param.name for param in func.parameters] if hasattr(func, 'parameters') else [],
            "issues": [
                {
                    "category": issue.category.value,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "line": issue.location.line_start
                }
                for issue in func_issues
            ],
            "calls": [],  # Will be populated from call graph
            "called_by": [],  # Will be populated from call graph
            "dependencies": []  # Will be populated from dependencies
        }
    
    # Process classes
    for cls in codebase.classes:
        cls_issues = [issue for issue in issues.issues if str(issue.location.file_path) == cls.filepath and issue.location.line_start >= cls.start_line and issue.location.line_start <= cls.end_line]
        
        symbol_details[f"class:{cls.name}"] = {
            "type": "class",
            "name": cls.name,
            "filepath": cls.filepath,
            "start_line": cls.start_line,
            "end_line": cls.end_line,
            "methods": [method.name for method in cls.methods] if hasattr(cls, 'methods') else [],
            "issues": [
                {
                    "category": issue.category.value,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "line": issue.location.line_start
                }
                for issue in cls_issues
            ],
            "dependencies": []
        }
    
    return symbol_details

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Analyze a codebase.
    
    Args:
        request: AnalysisRequest object
        background_tasks: BackgroundTasks object
        
    Returns:
        AnalysisResponse object
    """
    # Generate analysis ID
    import uuid
    analysis_id = str(uuid.uuid4())
    
    # Create output directory
    output_dir = os.path.join(request.output_dir, analysis_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Store analysis request
    analysis_results[analysis_id] = {
        "status": "pending",
        "message": "Analysis started",
        "repo_url": request.repo_url,
        "branch": request.branch,
        "output_dir": output_dir,
    }
    
    # Run analysis in background
    background_tasks.add_task(
        run_analysis,
        analysis_id,
        request.repo_url,
        request.branch,
        output_dir,
    )
    
    return {
        "status": "pending",
        "message": "Analysis started",
        "analysis_id": analysis_id,
    }

@app.get("/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """
    Get analysis results.
    
    Args:
        analysis_id: Analysis ID
        
    Returns:
        Analysis results
    """
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis_results[analysis_id]

@app.get("/analysis/{analysis_id}/report")
async def get_report(analysis_id: str):
    """
    Get analysis report.
    
    Args:
        analysis_id: Analysis ID
        
    Returns:
        HTML report
    """
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    if analysis_results[analysis_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed")
    
    report_path = os.path.join(analysis_results[analysis_id]["output_dir"], "report.html")
    
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(report_path)

@app.get("/analysis/{analysis_id}/visualization/{visualization_name}")
async def get_visualization(analysis_id: str, visualization_name: str):
    """
    Get visualization.
    
    Args:
        analysis_id: Analysis ID
        visualization_name: Visualization name
        
    Returns:
        Visualization image
    """
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    if analysis_results[analysis_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed")
    
    visualization_path = os.path.join(
        analysis_results[analysis_id]["output_dir"],
        "visualizations",
        f"{visualization_name}.png",
    )
    
    if not os.path.exists(visualization_path):
        raise HTTPException(status_code=404, detail="Visualization not found")
    
    return FileResponse(visualization_path)

@app.get("/cli/{repo_owner}/{repo_name}")
async def cli_analysis(repo_owner: str, repo_name: str):
    """
    CLI-friendly analysis endpoint that returns simplified text output.
    
    Args:
        repo_owner: GitHub repository owner
        repo_name: GitHub repository name
        
    Returns:
        Text-based analysis summary
    """
    try:
        # Get comprehensive analysis
        result = await analyze_repository(repo_owner, repo_name)
        
        # Extract data from JSONResponse
        import json
        if hasattr(result, 'body'):
            analysis_data = json.loads(result.body.decode())
        else:
            analysis_data = result
        
        repo = analysis_data["repository"]
        analysis = analysis_data["analysis"]
        
        cli_output = f"""
🔍 CODEBASE ANALYSIS REPORT
{'=' * 50}
📊 Repository: {repo["owner"]}/{repo["name"]}
🌐 URL: {repo["url"]}

📈 OVERVIEW:
- Files: {repo["total_files"]}
- Functions: {repo["total_functions"]}
- Classes: {repo["total_classes"]}
- Symbols: {repo["total_symbols"]}

🚨 ISSUES SUMMARY:
- Total Issues: {analysis["issues"]["total"]}
- Critical: {analysis["issues"]["by_severity"].get("critical", 0)}
- Major: {analysis["issues"]["by_severity"].get("major", 0)}
- Minor: {analysis["issues"]["by_severity"].get("minor", 0)}

📊 CODE QUALITY:
- Maintainability Index: {analysis["code_quality"]["maintainability_index"]:.1f}
- Cyclomatic Complexity: {analysis["code_quality"]["cyclomatic_complexity"]:.1f}
- Comment Density: {analysis["code_quality"]["comment_density"]:.2f}
- Source Lines of Code: {analysis["code_quality"]["source_lines_of_code"]}
- Technical Debt Ratio: {analysis["code_quality"]["technical_debt_ratio"]:.2f}

🔗 DEPENDENCIES:
- Total Dependencies: {analysis["dependencies"]["total"]}
- Circular Dependencies: {analysis["dependencies"]["circular"]}
- External Dependencies: {analysis["dependencies"]["external"]}
- Dependency Depth: {analysis["dependencies"]["depth"]}

📞 CALL GRAPH:
- Total Functions: {analysis["call_graph"]["total_functions"]}
- Entry Points: {analysis["call_graph"]["entry_points"]}
- Leaf Functions: {analysis["call_graph"]["leaf_functions"]}
- Max Call Depth: {analysis["call_graph"]["max_call_depth"]}

🎯 ANALYSIS COMPLETE!
"""
        
        return {"cli_output": cli_output}
        
    except Exception as e:
        return {"error": f"CLI analysis failed: {str(e)}"}

@app.get("/endpoint/{repo_name}/")
async def cli_analyze_endpoint(repo_name: str, background_tasks: BackgroundTasks, branch: Optional[str] = None):
    """
    CLI endpoint for codebase analysis.
    
    Args:
        repo_name: Repository name (format: owner/repo or just repo for GitHub)
        branch: Optional branch name (defaults to main/master)
        background_tasks: BackgroundTasks object
        
    Returns:
        Analysis results or analysis ID for background processing
    """
    # Generate analysis ID
    import uuid
    analysis_id = str(uuid.uuid4())
    
    # Construct GitHub URL if not a full URL
    if not repo_name.startswith(('http://', 'https://')):
        if '/' not in repo_name:
            # Assume it's a repo under current user/org context
            repo_url = f"https://github.com/Zeeeepa/{repo_name}"
        else:
            # Assume it's owner/repo format
            repo_url = f"https://github.com/{repo_name}"
    else:
        repo_url = repo_name
    
    # Create output directory
    output_dir = os.path.join("output", analysis_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Store analysis request
    analysis_results[analysis_id] = {
        "status": "pending",
        "message": "CLI analysis started",
        "repo_url": repo_url,
        "repo_name": repo_name,
        "branch": branch,
        "output_dir": output_dir,
    }
    
    # Run analysis in background
    background_tasks.add_task(
        run_analysis,
        analysis_id,
        repo_url,
        branch,
        output_dir,
    )
    
    return {
        "status": "pending",
        "message": f"CLI analysis started for {repo_name}",
        "analysis_id": analysis_id,
        "repo_url": repo_url,
        "check_status_url": f"/analysis/{analysis_id}",
        "report_url": f"/analysis/{analysis_id}/report"
    }

def run_analysis(analysis_id: str, repo_url: str, branch: Optional[str], output_dir: str):
    """
    Run analysis in background.
    
    Args:
        analysis_id: Analysis ID
        repo_url: Repository URL
        branch: Branch name
        output_dir: Output directory
    """
    try:
        # Update status
        analysis_results[analysis_id]["status"] = "running"
        analysis_results[analysis_id]["message"] = "Cloning repository"
        
        # Clone repository
        with tempfile.TemporaryDirectory() as temp_dir:
            # Clone repository
            clone_cmd = ["git", "clone", repo_url, temp_dir]
            if branch:
                clone_cmd.extend(["--branch", branch])
            
            subprocess.run(clone_cmd, check=True)
            
            # Update status
            analysis_results[analysis_id]["status"] = "running"
            analysis_results[analysis_id]["message"] = "Analyzing codebase"
            
            # Run analysis using the analysis module
            try:
                # Use the analyze_codebase function from analysis.py
                results = analyze_codebase(temp_dir)
                
                # Save analysis results
                with open(os.path.join(output_dir, "analysis.json"), "w") as f:
                    json.dump(results, f, indent=2, default=str)
                
                # Generate visualizations
                visualizations = visualize_codebase(temp_dir, output_dir)
                
                # Generate HTML report
                report_path = generate_html_report(results, visualizations, output_dir)
                
                # Update status
                analysis_results[analysis_id]["status"] = "completed"
                analysis_results[analysis_id]["message"] = "Analysis completed"
                analysis_results[analysis_id]["results"] = results
                analysis_results[analysis_id]["visualizations"] = visualizations
                analysis_results[analysis_id]["report_path"] = report_path
                
            except Exception as analysis_error:
                # If analysis fails, create a basic report
                error_results = {
                    "error": str(analysis_error),
                    "status": "analysis_failed",
                    "repo_url": repo_url,
                    "branch": branch,
                    "timestamp": str(datetime.now())
                }
                
                with open(os.path.join(output_dir, "analysis.json"), "w") as f:
                    json.dump(error_results, f, indent=2)
                
                analysis_results[analysis_id]["status"] = "completed_with_errors"
                analysis_results[analysis_id]["message"] = f"Analysis completed with errors: {str(analysis_error)}"
                analysis_results[analysis_id]["results"] = error_results
    
    except Exception as e:
        # Update status
        analysis_results[analysis_id]["status"] = "failed"
        analysis_results[analysis_id]["message"] = f"Analysis failed: {str(e)}"

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
