#!/usr/bin/env python3
"""
Codebase Analytics API

This module provides a FastAPI-based API for codebase analysis and visualization.
It imports functionality from the analysis and visualize modules, providing a
clean separation of concerns while maintaining a unified API for clients.
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Union
import json
import logging
import os
import tempfile
import subprocess
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
import modal
from pathlib import Path

# Import from analysis and visualize modules
from analysis import (
    analyze_codebase,
    AnalysisType,
    AnalysisResult,
    Issue,
    IssueCollection,
    IssueSeverity,
    IssueCategory,
    IssueStatus,
    CodeLocation,
)
from visualize import (
    visualize_analysis,
    VisualizationType,
    OutputFormat,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = modal.App(name="analytics-app")
fastapi_app = FastAPI(
    title="Codebase Analytics API",
    description="Visual codebase exploration and analysis API",
    version="1.0.0"
)

# Add CORS middleware
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@fastapi_app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Codebase Analytics API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

#######################################################
# API Models
#######################################################

class RepoRequest(BaseModel):
    """Request model for repository analysis."""
    repo_url: str
    analysis_types: Optional[List[str]] = None
    output_format: Optional[str] = "json"

class VisualizationRequest(BaseModel):
    """Request model for visualization."""
    repo_url: str
    visualization_type: str = "comprehensive"
    output_format: str = "interactive"

class StructuralAnalysisRequest(BaseModel):
    """Request model for structural analysis."""
    repo_url: str
    mode: str = "structural_overview"

class BlastRadiusRequest(BaseModel):
    """Request model for blast radius analysis."""
    repo_url: str
    symbol_name: str

class AnalysisResponse(BaseModel):
    """Response model for analysis results."""
    repo_url: str
    summary: Dict[str, Any]
    code_quality: Optional[Dict[str, Any]] = None
    dependencies: Optional[Dict[str, Any]] = None
    security: Optional[Dict[str, Any]] = None
    performance: Optional[Dict[str, Any]] = None
    type_analysis: Optional[Dict[str, Any]] = None
    issues: Optional[Dict[str, Any]] = None

class VisualizationResponse(BaseModel):
    """Response model for visualization results."""
    repo_url: str
    visualization_type: str
    output_format: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]

#######################################################
# API Endpoints
#######################################################

@fastapi_app.post("/analyze", response_model=AnalysisResponse)
async def analyze_repository(request: RepoRequest):
    """
    Analyze a repository and return the results.
    
    This endpoint performs a comprehensive analysis of the specified repository,
    including code quality, dependencies, security, and performance.
    """
    try:
        logger.info(f"Analyzing repository: {request.repo_url}")
        
        # Perform analysis
        result = analyze_codebase(
            repo_url=request.repo_url,
            analysis_types=request.analysis_types,
            output_format=request.output_format,
        )
        
        # Convert to response model
        response = {
            "repo_url": request.repo_url,
            "summary": asdict(result.summary) if hasattr(result, "summary") else {},
            "code_quality": asdict(result.code_quality) if hasattr(result, "code_quality") and result.code_quality else None,
            "dependencies": asdict(result.dependencies) if hasattr(result, "dependencies") and result.dependencies else None,
            "security": asdict(result.security) if hasattr(result, "security") and result.security else None,
            "performance": asdict(result.performance) if hasattr(result, "performance") and result.performance else None,
            "type_analysis": asdict(result.type_analysis) if hasattr(result, "type_analysis") and result.type_analysis else None,
            "issues": result.issues.to_dict() if hasattr(result, "issues") and result.issues else None,
        }
        
        return response
    except Exception as e:
        logger.error(f"Error analyzing repository: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@fastapi_app.post("/visualize", response_model=VisualizationResponse)
async def visualize_repository(request: VisualizationRequest):
    """
    Visualize a repository analysis and return the results.
    
    This endpoint generates visualizations of the analysis results for the
    specified repository, including issue treemaps, dependency graphs, and more.
    """
    try:
        logger.info(f"Visualizing repository: {request.repo_url}")
        
        # First analyze the repository
        analysis_result = analyze_codebase(
            repo_url=request.repo_url,
            analysis_types=["comprehensive"],
            output_format="json",
        )
        
        # Then visualize the analysis
        visualization_result = visualize_analysis(
            analysis_result=analysis_result,
            visualization_type=request.visualization_type,
            output_format=request.output_format,
        )
        
        # Convert to response model
        response = {
            "repo_url": request.repo_url,
            "visualization_type": request.visualization_type,
            "output_format": request.output_format,
            "data": visualization_result,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "analysis_time": analysis_result.summary.analysis_time,
            },
        }
        
        return response
    except Exception as e:
        logger.error(f"Error visualizing repository: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")

@fastapi_app.post("/analyze_structural")
async def analyze_structural(request: StructuralAnalysisRequest):
    """
    Perform comprehensive structural analysis with interactive visualization.
    
    This endpoint provides detailed error detection, structural tree generation,
    and contextual issue reporting similar to advanced code analysis tools.
    """
    try:
        from visualization.interactive_structural_analyzer import analyze_repository_structure
        from visualization.visual_codebase_explorer import create_visual_exploration, analyze_error_blast_radius, ExplorationMode
        
        repo_url = request.repo_url
        logger.info(f"Starting structural analysis for: {repo_url}")
        
        # Clone repository to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = clone_repo(repo_url, temp_dir)
            
            # Load codebase with Codegen SDK
            codebase = Codebase.from_path(repo_path)
            
            # Perform comprehensive structural analysis
            analysis_result = analyze_repository_structure(repo_path, codebase)
            
            logger.info(f"Structural analysis completed. Found {analysis_result['repository_info']['total_errors']} issues.")
            
            return analysis_result
            
    except Exception as e:
        logger.error(f"Error in structural analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Structural analysis failed: {str(e)}")

@fastapi_app.post("/explore_visual")
async def explore_visual_codebase(request: StructuralAnalysisRequest):
    """
    Perform visual exploration of codebase structure, errors, and relationships.
    
    This endpoint provides comprehensive visual analysis focused on immediate insights
    rather than trends, including error detection, blast radius analysis, and 
    interactive structural navigation.
    """
    try:
        from visualization.visual_codebase_explorer import create_visual_exploration, ExplorationMode
        
        repo_url = request.repo_url
        mode = request.mode
        logger.info(f"Starting visual exploration for: {repo_url} in mode: {mode}")
        
        # Validate exploration mode
        try:
            exploration_mode = ExplorationMode(mode)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid exploration mode: {mode}")
        
        # Handle local path or clone repository
        if repo_url == "." or repo_url.startswith("/") or repo_url.startswith("./"):
            # Local path
            repo_path = repo_url
            logger.info(f"Loading local codebase from: {repo_path}")
            codebase = Codebase(repo_path)
            logger.info(f"Loaded codebase with {len(codebase.files)} files")
            
            # Perform visual exploration
            logger.info(f"Performing visual exploration in {mode} mode...")
            exploration_data = create_visual_exploration(codebase, exploration_mode)
        else:
            # Clone repository to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                repo_path = clone_repo(repo_url, temp_dir)
                
                # Load codebase with Codegen SDK
                logger.info(f"Loading codebase from: {repo_path}")
                codebase = Codebase(repo_path)
                logger.info(f"Loaded codebase with {len(codebase.files)} files")
                
                # Perform visual exploration
                logger.info(f"Performing visual exploration in {mode} mode...")
                exploration_data = create_visual_exploration(codebase, exploration_mode)
        
        logger.info(f"Visual exploration completed successfully")
        logger.info(f"Found {exploration_data['summary']['total_nodes']} nodes and {exploration_data['summary']['total_issues']} issues")
        
        return exploration_data
            
    except Exception as e:
        logger.error(f"Error in visual exploration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Visual exploration failed: {str(e)}")

@fastapi_app.post("/analyze_blast_radius")
async def analyze_symbol_blast_radius(request: BlastRadiusRequest):
    """
    Analyze the blast radius of a specific symbol to understand impact of changes.
    
    This endpoint shows how changes to a specific function, class, or symbol would
    affect other parts of the codebase, providing visual impact analysis.
    """
    try:
        from visualization.visual_codebase_explorer import analyze_error_blast_radius
        
        repo_url = request.repo_url
        symbol_name = request.symbol_name
        logger.info(f"Analyzing blast radius for symbol '{symbol_name}' in: {repo_url}")
        
        # Handle local path or clone repository
        if repo_url == "." or repo_url.startswith("/") or repo_url.startswith("./"):
            # Local path
            repo_path = repo_url
            logger.info(f"Loading local codebase from: {repo_path}")
            codebase = Codebase(repo_path)
            logger.info(f"Loaded codebase with {len(codebase.files)} files")
            
            # Analyze blast radius
            logger.info(f"Analyzing blast radius for symbol: {symbol_name}")
            blast_radius_data = analyze_error_blast_radius(codebase, symbol_name)
        else:
            # Clone repository to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                repo_path = clone_repo(repo_url, temp_dir)
                
                # Load codebase with Codegen SDK
                logger.info(f"Loading codebase from: {repo_path}")
                codebase = Codebase(repo_path)
                logger.info(f"Loaded codebase with {len(codebase.files)} files")
                
                # Analyze blast radius
                logger.info(f"Analyzing blast radius for symbol: {symbol_name}")
                blast_radius_data = analyze_error_blast_radius(codebase, symbol_name)
        
        if "error" in blast_radius_data:
            raise HTTPException(status_code=404, detail=blast_radius_data["error"])
        
        logger.info(f"Blast radius analysis completed successfully")
        logger.info(f"Symbol affects {blast_radius_data['blast_radius']['affected_nodes']} nodes")
        
        return blast_radius_data
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in blast radius analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Blast radius analysis failed: {str(e)}")

#######################################################
# Helper Functions
#######################################################

def clone_repo(repo_url: str, target_dir: str) -> str:
    """
    Clone a repository to a target directory.
    
    Args:
        repo_url: URL of the repository to clone
        target_dir: Directory to clone the repository to
        
    Returns:
        Path to the cloned repository
    """
    logger.info(f"Cloning repository: {repo_url} to {target_dir}")
    
    try:
        # Extract repo name from URL
        repo_name = repo_url.split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
        
        # Clone the repository
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, target_dir],
            check=True,
            capture_output=True,
        )
        
        logger.info(f"Repository cloned successfully to {target_dir}")
        return target_dir
    except subprocess.CalledProcessError as e:
        logger.error(f"Error cloning repository: {e.stderr.decode()}")
        raise ValueError(f"Failed to clone repository: {e.stderr.decode()}")
    except Exception as e:
        logger.error(f"Error cloning repository: {str(e)}")
        raise ValueError(f"Failed to clone repository: {str(e)}")

def asdict(obj):
    """
    Convert an object to a dictionary.
    
    Args:
        obj: Object to convert
        
    Returns:
        Dictionary representation of the object
    """
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    elif hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    else:
        return obj

@app.function()
@modal.asgi_app()
def fastapi_modal_app():
    """Create a Modal ASGI app for the FastAPI app."""
    return fastapi_app

if __name__ == "__main__":
    import uvicorn
    import socket
    
    def find_available_port(start_port=8000, max_port=8100):
        """Find an available port starting from start_port"""
        for port in range(start_port, max_port):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('0.0.0.0', port))
                    return port
            except OSError:
                continue
        raise RuntimeError(f"No available ports found between {start_port} and {max_port}")
    
    # Find an available port
    port = find_available_port()
    print(f"🚀 Starting FastAPI server on http://localhost:{port}")
    print(f"📚 API documentation available at http://localhost:{port}/docs")
    
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)
