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
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Optional, Dict, Any, List

# Import consolidated analysis and visualization modules
from analysis import analyze_codebase, get_codebase_summary, get_dependency_graph, get_symbol_references
from visualization import visualize_codebase, generate_html_report, run_visualization_analysis, CodebaseVisualizer

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
