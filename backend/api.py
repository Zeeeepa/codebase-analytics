#!/usr/bin/env python3
"""
API Module for Codebase Analysis

This module provides a FastAPI server for codebase analysis.
It includes endpoints for analyzing codebases, retrieving analysis results,
and generating visualizations.
"""

import os
import json
import tempfile
import shutil
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form, Query
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pathlib import Path

# Import analysis and visualization modules
from analysis import analyze_codebase
from visualize import visualize_codebase, generate_html_report

# Import from Codegen SDK
from codegen.sdk.core.codebase import Codebase
from codegen.sdk.core.file import SourceFile

# Create FastAPI app
app = FastAPI(
    title="Codebase Analysis API",
    description="API for analyzing codebases and generating visualizations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create storage directory
STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

# Create results directory
RESULTS_DIR = os.path.join(STORAGE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Create uploads directory
UPLOADS_DIR = os.path.join(STORAGE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Create visualizations directory
VISUALIZATIONS_DIR = os.path.join(STORAGE_DIR, "visualizations")
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Define models
class AnalysisRequest(BaseModel):
    """Request model for codebase analysis."""
    repo_url: str
    branch: Optional[str] = None
    include_visualizations: bool = True

class AnalysisResponse(BaseModel):
    """Response model for codebase analysis."""
    analysis_id: str
    status: str
    message: str

class AnalysisResult(BaseModel):
    """Model for analysis result."""
    analysis_id: str
    summary: Dict[str, Any]
    visualizations: Optional[Dict[str, Any]] = None

# Store analysis tasks
analysis_tasks = {}

# Background task for analyzing a codebase
def analyze_codebase_task(analysis_id: str, repo_url: str, branch: Optional[str], include_visualizations: bool):
    """
    Background task for analyzing a codebase.
    
    Args:
        analysis_id: ID of the analysis task
        repo_url: URL of the repository to analyze
        branch: Optional branch to analyze
        include_visualizations: Whether to include visualizations
    """
    try:
        # Update task status
        analysis_tasks[analysis_id] = {
            "status": "running",
            "message": "Cloning repository..."
        }
        
        # Create temporary directory for the repository
        with tempfile.TemporaryDirectory() as temp_dir:
            # Clone the repository
            import subprocess
            clone_cmd = ["git", "clone", repo_url, temp_dir]
            if branch:
                clone_cmd.extend(["--branch", branch])
            
            subprocess.run(clone_cmd, check=True)
            
            # Update task status
            analysis_tasks[analysis_id] = {
                "status": "running",
                "message": "Analyzing codebase..."
            }
            
            # Create codebase object
            codebase = Codebase.from_directory(temp_dir)
            
            # Analyze codebase
            analysis_result = analyze_codebase(codebase)
            
            # Create result directory
            result_dir = os.path.join(RESULTS_DIR, analysis_id)
            os.makedirs(result_dir, exist_ok=True)
            
            # Save analysis result
            with open(os.path.join(result_dir, "analysis.json"), "w") as f:
                # Convert to serializable format
                serializable_result = {
                    "summary": {
                        "total_files": analysis_result["summary"].total_files,
                        "total_lines": analysis_result["summary"].total_lines,
                        "total_functions": analysis_result["summary"].total_functions,
                        "total_classes": analysis_result["summary"].total_classes,
                        "total_issues": analysis_result["summary"].total_issues,
                        "issue_counts": analysis_result["summary"].issue_counts,
                        "metrics": analysis_result["summary"].metrics,
                        "recommendations": analysis_result["summary"].recommendations
                    },
                    "dependency_analysis": {
                        "total_dependencies": analysis_result["dependency_analysis"].total_dependencies,
                        "circular_dependencies": analysis_result["dependency_analysis"].circular_dependencies,
                        "dependency_depth": analysis_result["dependency_analysis"].dependency_depth,
                        "external_dependencies": analysis_result["dependency_analysis"].external_dependencies,
                        "internal_dependencies": analysis_result["dependency_analysis"].internal_dependencies,
                        "critical_dependencies": analysis_result["dependency_analysis"].critical_dependencies,
                        "unused_dependencies": analysis_result["dependency_analysis"].unused_dependencies
                    },
                    "call_graph_analysis": {
                        "total_functions": analysis_result["call_graph_analysis"].total_functions,
                        "entry_points": analysis_result["call_graph_analysis"].entry_points,
                        "leaf_functions": analysis_result["call_graph_analysis"].leaf_functions,
                        "max_call_depth": analysis_result["call_graph_analysis"].max_call_depth
                    },
                    "code_quality_result": {
                        "maintainability_index": analysis_result["code_quality_result"].maintainability_index,
                        "cyclomatic_complexity": analysis_result["code_quality_result"].cyclomatic_complexity,
                        "halstead_volume": analysis_result["code_quality_result"].halstead_volume,
                        "source_lines_of_code": analysis_result["code_quality_result"].source_lines_of_code,
                        "comment_density": analysis_result["code_quality_result"].comment_density,
                        "duplication_percentage": analysis_result["code_quality_result"].duplication_percentage,
                        "technical_debt_ratio": analysis_result["code_quality_result"].technical_debt_ratio
                    },
                    "issue_collection": {
                        "total_issues": len(analysis_result["issue_collection"].issues),
                        "by_severity": analysis_result["issue_collection"].count_by_severity(),
                        "by_category": analysis_result["issue_collection"].count_by_category(),
                        "by_status": analysis_result["issue_collection"].count_by_status()
                    },
                    "recommendations": analysis_result["recommendations"]
                }
                
                json.dump(serializable_result, f, indent=2)
            
            # Generate visualizations if requested
            if include_visualizations:
                # Update task status
                analysis_tasks[analysis_id] = {
                    "status": "running",
                    "message": "Generating visualizations..."
                }
                
                # Create visualizations directory
                vis_dir = os.path.join(VISUALIZATIONS_DIR, analysis_id)
                os.makedirs(vis_dir, exist_ok=True)
                
                # Generate visualizations
                visualization = visualize_codebase(codebase, vis_dir)
                
                # Generate HTML report
                html_report = generate_html_report(analysis_result, os.path.join(vis_dir, "report.html"))
            
            # Update task status
            analysis_tasks[analysis_id] = {
                "status": "completed",
                "message": "Analysis completed successfully",
                "result": {
                    "analysis_id": analysis_id,
                    "summary": serializable_result["summary"],
                    "visualizations": {
                        "html_report": f"/api/visualizations/{analysis_id}/report.html",
                        "dependency_graph": f"/api/visualizations/{analysis_id}/dependency_graph.png",
                        "call_graph": f"/api/visualizations/{analysis_id}/call_graph.png",
                        "issues": f"/api/visualizations/{analysis_id}/issues.png",
                        "issues_by_file": f"/api/visualizations/{analysis_id}/issues_by_file.png",
                        "code_quality": f"/api/visualizations/{analysis_id}/code_quality.png"
                    } if include_visualizations else None
                }
            }
    except Exception as e:
        # Update task status with error
        analysis_tasks[analysis_id] = {
            "status": "failed",
            "message": f"Analysis failed: {str(e)}"
        }
        
        # Log the error
        import traceback
        print(f"Error analyzing codebase: {str(e)}")
        print(traceback.format_exc())

# API endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Codebase Analysis API"}

@app.post("/api/analyze", response_model=AnalysisResponse)
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
    
    # Create analysis task
    analysis_tasks[analysis_id] = {
        "status": "pending",
        "message": "Analysis task created"
    }
    
    # Start analysis task in the background
    background_tasks.add_task(
        analyze_codebase_task,
        analysis_id,
        request.repo_url,
        request.branch,
        request.include_visualizations
    )
    
    # Return response
    return AnalysisResponse(
        analysis_id=analysis_id,
        status="pending",
        message="Analysis task created"
    )

@app.get("/api/analysis/{analysis_id}/status")
async def get_analysis_status(analysis_id: str):
    """
    Get the status of an analysis task.
    
    Args:
        analysis_id: ID of the analysis task
        
    Returns:
        Status of the analysis task
    """
    # Check if analysis task exists
    if analysis_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail="Analysis task not found")
    
    # Return status
    return {
        "analysis_id": analysis_id,
        "status": analysis_tasks[analysis_id]["status"],
        "message": analysis_tasks[analysis_id]["message"]
    }

@app.get("/api/analysis/{analysis_id}/result", response_model=AnalysisResult)
async def get_analysis_result(analysis_id: str):
    """
    Get the result of an analysis task.
    
    Args:
        analysis_id: ID of the analysis task
        
    Returns:
        Result of the analysis task
    """
    # Check if analysis task exists
    if analysis_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail="Analysis task not found")
    
    # Check if analysis task is completed
    if analysis_tasks[analysis_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis task not completed")
    
    # Return result
    return analysis_tasks[analysis_id]["result"]

@app.get("/api/analysis/{analysis_id}/report")
async def get_analysis_report(analysis_id: str):
    """
    Get the HTML report for an analysis task.
    
    Args:
        analysis_id: ID of the analysis task
        
    Returns:
        HTML report for the analysis task
    """
    # Check if analysis task exists
    if analysis_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail="Analysis task not found")
    
    # Check if analysis task is completed
    if analysis_tasks[analysis_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis task not completed")
    
    # Check if visualizations were generated
    if "visualizations" not in analysis_tasks[analysis_id]["result"] or analysis_tasks[analysis_id]["result"]["visualizations"] is None:
        raise HTTPException(status_code=400, detail="Visualizations not generated for this analysis")
    
    # Get HTML report path
    html_report_path = os.path.join(VISUALIZATIONS_DIR, analysis_id, "report.html")
    
    # Check if HTML report exists
    if not os.path.exists(html_report_path):
        raise HTTPException(status_code=404, detail="HTML report not found")
    
    # Return HTML report
    return FileResponse(html_report_path, media_type="text/html")

@app.get("/api/visualizations/{analysis_id}/{visualization_file}")
async def get_visualization(analysis_id: str, visualization_file: str):
    """
    Get a visualization file for an analysis task.
    
    Args:
        analysis_id: ID of the analysis task
        visualization_file: Name of the visualization file
        
    Returns:
        Visualization file
    """
    # Check if analysis task exists
    if analysis_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail="Analysis task not found")
    
    # Check if analysis task is completed
    if analysis_tasks[analysis_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis task not completed")
    
    # Check if visualizations were generated
    if "visualizations" not in analysis_tasks[analysis_id]["result"] or analysis_tasks[analysis_id]["result"]["visualizations"] is None:
        raise HTTPException(status_code=400, detail="Visualizations not generated for this analysis")
    
    # Get visualization file path
    visualization_path = os.path.join(VISUALIZATIONS_DIR, analysis_id, visualization_file)
    
    # Check if visualization file exists
    if not os.path.exists(visualization_path):
        raise HTTPException(status_code=404, detail="Visualization file not found")
    
    # Determine media type
    media_type = "image/png"
    if visualization_file.endswith(".html"):
        media_type = "text/html"
    elif visualization_file.endswith(".json"):
        media_type = "application/json"
    
    # Return visualization file
    return FileResponse(visualization_path, media_type=media_type)

@app.post("/api/upload")
async def upload_codebase(background_tasks: BackgroundTasks, file: UploadFile = File(...), include_visualizations: bool = Form(True)):
    """
    Upload and analyze a codebase.
    
    Args:
        background_tasks: BackgroundTasks object
        file: Uploaded file (zip or tar.gz)
        include_visualizations: Whether to include visualizations
        
    Returns:
        AnalysisResponse object
    """
    # Generate analysis ID
    import uuid
    analysis_id = str(uuid.uuid4())
    
    # Create analysis task
    analysis_tasks[analysis_id] = {
        "status": "pending",
        "message": "Analysis task created"
    }
    
    # Create upload directory
    upload_dir = os.path.join(UPLOADS_DIR, analysis_id)
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save uploaded file
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Extract uploaded file
    extract_dir = os.path.join(upload_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    
    # Update task status
    analysis_tasks[analysis_id] = {
        "status": "running",
        "message": "Extracting uploaded file..."
    }
    
    # Extract based on file type
    if file.filename.endswith(".zip"):
        import zipfile
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
    elif file.filename.endswith(".tar.gz") or file.filename.endswith(".tgz"):
        import tarfile
        with tarfile.open(file_path, "r:gz") as tar_ref:
            tar_ref.extractall(extract_dir)
    else:
        # Update task status with error
        analysis_tasks[analysis_id] = {
            "status": "failed",
            "message": "Unsupported file format. Please upload a zip or tar.gz file."
        }
        return AnalysisResponse(
            analysis_id=analysis_id,
            status="failed",
            message="Unsupported file format. Please upload a zip or tar.gz file."
        )
    
    # Start analysis task in the background
    background_tasks.add_task(
        analyze_uploaded_codebase_task,
        analysis_id,
        extract_dir,
        include_visualizations
    )
    
    # Return response
    return AnalysisResponse(
        analysis_id=analysis_id,
        status="pending",
        message="Analysis task created"
    )

# Background task for analyzing an uploaded codebase
def analyze_uploaded_codebase_task(analysis_id: str, extract_dir: str, include_visualizations: bool):
    """
    Background task for analyzing an uploaded codebase.
    
    Args:
        analysis_id: ID of the analysis task
        extract_dir: Directory containing the extracted codebase
        include_visualizations: Whether to include visualizations
    """
    try:
        # Update task status
        analysis_tasks[analysis_id] = {
            "status": "running",
            "message": "Analyzing codebase..."
        }
        
        # Create codebase object
        codebase = Codebase.from_directory(extract_dir)
        
        # Analyze codebase
        analysis_result = analyze_codebase(codebase)
        
        # Create result directory
        result_dir = os.path.join(RESULTS_DIR, analysis_id)
        os.makedirs(result_dir, exist_ok=True)
        
        # Save analysis result
        with open(os.path.join(result_dir, "analysis.json"), "w") as f:
            # Convert to serializable format
            serializable_result = {
                "summary": {
                    "total_files": analysis_result["summary"].total_files,
                    "total_lines": analysis_result["summary"].total_lines,
                    "total_functions": analysis_result["summary"].total_functions,
                    "total_classes": analysis_result["summary"].total_classes,
                    "total_issues": analysis_result["summary"].total_issues,
                    "issue_counts": analysis_result["summary"].issue_counts,
                    "metrics": analysis_result["summary"].metrics,
                    "recommendations": analysis_result["summary"].recommendations
                },
                "dependency_analysis": {
                    "total_dependencies": analysis_result["dependency_analysis"].total_dependencies,
                    "circular_dependencies": analysis_result["dependency_analysis"].circular_dependencies,
                    "dependency_depth": analysis_result["dependency_analysis"].dependency_depth,
                    "external_dependencies": analysis_result["dependency_analysis"].external_dependencies,
                    "internal_dependencies": analysis_result["dependency_analysis"].internal_dependencies,
                    "critical_dependencies": analysis_result["dependency_analysis"].critical_dependencies,
                    "unused_dependencies": analysis_result["dependency_analysis"].unused_dependencies
                },
                "call_graph_analysis": {
                    "total_functions": analysis_result["call_graph_analysis"].total_functions,
                    "entry_points": analysis_result["call_graph_analysis"].entry_points,
                    "leaf_functions": analysis_result["call_graph_analysis"].leaf_functions,
                    "max_call_depth": analysis_result["call_graph_analysis"].max_call_depth
                },
                "code_quality_result": {
                    "maintainability_index": analysis_result["code_quality_result"].maintainability_index,
                    "cyclomatic_complexity": analysis_result["code_quality_result"].cyclomatic_complexity,
                    "halstead_volume": analysis_result["code_quality_result"].halstead_volume,
                    "source_lines_of_code": analysis_result["code_quality_result"].source_lines_of_code,
                    "comment_density": analysis_result["code_quality_result"].comment_density,
                    "duplication_percentage": analysis_result["code_quality_result"].duplication_percentage,
                    "technical_debt_ratio": analysis_result["code_quality_result"].technical_debt_ratio
                },
                "issue_collection": {
                    "total_issues": len(analysis_result["issue_collection"].issues),
                    "by_severity": analysis_result["issue_collection"].count_by_severity(),
                    "by_category": analysis_result["issue_collection"].count_by_category(),
                    "by_status": analysis_result["issue_collection"].count_by_status()
                },
                "recommendations": analysis_result["recommendations"]
            }
            
            json.dump(serializable_result, f, indent=2)
        
        # Generate visualizations if requested
        if include_visualizations:
            # Update task status
            analysis_tasks[analysis_id] = {
                "status": "running",
                "message": "Generating visualizations..."
            }
            
            # Create visualizations directory
            vis_dir = os.path.join(VISUALIZATIONS_DIR, analysis_id)
            os.makedirs(vis_dir, exist_ok=True)
            
            # Generate visualizations
            visualization = visualize_codebase(codebase, vis_dir)
            
            # Generate HTML report
            html_report = generate_html_report(analysis_result, os.path.join(vis_dir, "report.html"))
        
        # Update task status
        analysis_tasks[analysis_id] = {
            "status": "completed",
            "message": "Analysis completed successfully",
            "result": {
                "analysis_id": analysis_id,
                "summary": serializable_result["summary"],
                "visualizations": {
                    "html_report": f"/api/visualizations/{analysis_id}/report.html",
                    "dependency_graph": f"/api/visualizations/{analysis_id}/dependency_graph.png",
                    "call_graph": f"/api/visualizations/{analysis_id}/call_graph.png",
                    "issues": f"/api/visualizations/{analysis_id}/issues.png",
                    "issues_by_file": f"/api/visualizations/{analysis_id}/issues_by_file.png",
                    "code_quality": f"/api/visualizations/{analysis_id}/code_quality.png"
                } if include_visualizations else None
            }
        }
    except Exception as e:
        # Update task status with error
        analysis_tasks[analysis_id] = {
            "status": "failed",
            "message": f"Analysis failed: {str(e)}"
        }
        
        # Log the error
        import traceback
        print(f"Error analyzing codebase: {str(e)}")
        print(traceback.format_exc())

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

