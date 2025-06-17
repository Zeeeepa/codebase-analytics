#!/usr/bin/env python3
"""
FastAPI server for the Codebase Analytics tool.
"""

import os
import tempfile
import subprocess
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Optional, Dict, Any, List

# Import analysis modules
from analysis import analyze_codebase
from visualize import visualize_codebase, generate_html_report

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
            
            # Run analysis
            # Note: This is a simplified version that doesn't use the Codegen SDK
            # In a real implementation, you would use the Codegen SDK to analyze the codebase
            
            # Get all files in the repository
            all_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith(('.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp')):
                        file_path = os.path.join(root, file)
                        all_files.append(file_path)
            
            # Create visualizations directory
            vis_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            # Run full analysis script
            subprocess.run(
                [
                    "python",
                    "run_full_analysis.py",
                    repo_url,
                    "--output-dir",
                    output_dir,
                ],
                check=True,
            )
            
            # Update status
            analysis_results[analysis_id]["status"] = "completed"
            analysis_results[analysis_id]["message"] = "Analysis completed"
            
            # Load analysis results
            with open(os.path.join(output_dir, "analysis.json"), "r") as f:
                analysis_results[analysis_id]["results"] = json.load(f)
    
    except Exception as e:
        # Update status
        analysis_results[analysis_id]["status"] = "failed"
        analysis_results[analysis_id]["message"] = f"Analysis failed: {str(e)}"

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

