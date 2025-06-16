from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn

# Mock classes for demonstration
class MockCodebase:
    def __init__(self, path):
        self.path = path
        self.files = [f"file_{i}.py" for i in range(10)]
        self.functions = [f"function_{i}" for i in range(50)]
        self.classes = [f"Class_{i}" for i in range(20)]

# Pydantic models
class AnalysisRequest(BaseModel):
    repo_url: str
    mode: str = "comprehensive"

class BlastRadiusRequest(BaseModel):
    repo_url: str
    symbol_name: str

class Issue(BaseModel):
    type: str
    severity: str
    message: str
    suggestion: str

class VisualNode(BaseModel):
    id: str
    name: str
    type: str
    path: str
    issues: List[Issue]
    blast_radius: int
    metadata: Dict[str, Any]

class ExplorationData(BaseModel):
    summary: Dict[str, int]
    nodes: List[VisualNode]
    error_hotspots: List[Dict[str, Any]]
    critical_paths: List[Dict[str, Any]]

class BlastRadiusData(BaseModel):
    target_symbol: Dict[str, str]
    blast_radius: Dict[str, Any]
    affected_nodes: List[VisualNode]

# Create FastAPI app
app = FastAPI(title="Interactive Codebase Analytics API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Interactive Codebase Analytics API"}

@app.post("/explore_visual")
async def explore_visual(request: AnalysisRequest) -> ExplorationData:
    """Visual exploration of codebase with error detection."""
    
    # Mock data for demonstration
    mock_issues = [
        Issue(
            type="parameter_mismatch",
            severity="high",
            message="Function expects 3 parameters but receives 2",
            suggestion="Add missing parameter or update function signature"
        ),
        Issue(
            type="unused_variable",
            severity="medium", 
            message="Variable 'temp_data' is declared but never used",
            suggestion="Remove unused variable or implement its usage"
        ),
        Issue(
            type="missing_return",
            severity="high",
            message="Function should return a value but has no return statement",
            suggestion="Add return statement or change function to void"
        )
    ]
    
    mock_nodes = [
        VisualNode(
            id="node_1",
            name="user_authentication.py",
            type="file",
            path="/src/auth/user_authentication.py",
            issues=mock_issues[:2],
            blast_radius=15,
            metadata={"lines": 245, "complexity": 8}
        ),
        VisualNode(
            id="node_2", 
            name="validate_user",
            type="function",
            path="/src/auth/user_authentication.py:45",
            issues=[mock_issues[2]],
            blast_radius=8,
            metadata={"parameters": 2, "calls": 12}
        ),
        VisualNode(
            id="node_3",
            name="UserManager",
            type="class",
            path="/src/models/user.py:12",
            issues=[],
            blast_radius=25,
            metadata={"methods": 8, "inheritance_depth": 2}
        )
    ]
    
    return ExplorationData(
        summary={
            "total_nodes": 45,
            "total_issues": 23,
            "error_hotspots_count": 8,
            "critical_paths_count": 3
        },
        nodes=mock_nodes,
        error_hotspots=[
            {"file": "user_authentication.py", "issues": 5, "severity": "high"},
            {"file": "data_processor.py", "issues": 3, "severity": "medium"},
            {"file": "api_handler.py", "issues": 4, "severity": "high"}
        ],
        critical_paths=[
            {"path": "login -> validate -> authenticate", "risk_score": 9.2},
            {"path": "data_input -> process -> output", "risk_score": 8.7},
            {"path": "api_call -> validate -> response", "risk_score": 8.1}
        ]
    )

@app.post("/analyze_blast_radius")
async def analyze_blast_radius(request: BlastRadiusRequest) -> BlastRadiusData:
    """Analyze the blast radius of a specific symbol."""
    
    affected_nodes = [
        VisualNode(
            id="affected_1",
            name="login_handler.py",
            type="file", 
            path="/src/handlers/login_handler.py",
            issues=[],
            blast_radius=5,
            metadata={"depends_on_target": True}
        ),
        VisualNode(
            id="affected_2",
            name="user_service.py", 
            type="file",
            path="/src/services/user_service.py",
            issues=[],
            blast_radius=3,
            metadata={"depends_on_target": True}
        )
    ]
    
    return BlastRadiusData(
        target_symbol={
            "name": request.symbol_name,
            "type": "function",
            "path": f"/src/auth/{request.symbol_name}.py"
        },
        blast_radius={
            "affected_nodes": 12,
            "risk_level": "medium",
            "change_impact": 7.5
        },
        affected_nodes=affected_nodes
    )

@app.post("/analyze_structural")
async def analyze_structural(request: AnalysisRequest):
    """Analyze structural issues in the codebase."""
    return {
        "circular_dependencies": [
            {"modules": ["auth.py", "user.py", "session.py"], "severity": "high"},
            {"modules": ["data.py", "processor.py"], "severity": "medium"}
        ],
        "coupling_issues": [
            {"class": "UserManager", "coupling_score": 8.5, "suggestion": "Extract interfaces"},
            {"class": "DataProcessor", "coupling_score": 7.2, "suggestion": "Use dependency injection"}
        ],
        "complexity_hotspots": [
            {"function": "process_user_data", "complexity": 15, "threshold": 10},
            {"function": "validate_input", "complexity": 12, "threshold": 10}
        ]
    }

@app.post("/analyze_repo")
async def analyze_repo(request: AnalysisRequest):
    """Comprehensive repository analysis."""
    return {
        "repository": request.repo_url,
        "analysis_mode": request.mode,
        "statistics": {
            "total_files": 156,
            "total_functions": 423,
            "total_classes": 89,
            "total_lines": 15420
        },
        "quality_metrics": {
            "test_coverage": 78.5,
            "code_duplication": 12.3,
            "maintainability_index": 82.1
        },
        "issues_summary": {
            "critical": 5,
            "high": 12,
            "medium": 28,
            "low": 45
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

