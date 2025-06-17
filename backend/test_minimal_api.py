#!/usr/bin/env python3
"""
Minimal API test to isolate the issue.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI()

class RepoRequest(BaseModel):
    repo_url: str

class FunctionDetail(BaseModel):
    name: str
    parameters: List[str] = []
    return_type: Optional[str] = None
    call_count: int = 0
    calls_made: int = 0

class ImportDetail(BaseModel):
    module: str
    imported_symbols: List[str] = []

class FunctionAnalysis(BaseModel):
    total_functions: int = 0
    most_called_function: Optional[FunctionDetail] = None
    most_calling_function: Optional[FunctionDetail] = None
    dead_functions_count: int = 0
    sample_functions: List[FunctionDetail] = []
    sample_classes: List[dict] = []
    sample_imports: List[ImportDetail] = []

class AnalysisResponse(BaseModel):
    function_analysis: FunctionAnalysis

@app.post("/analyze_repo")
async def analyze_repo(request: RepoRequest) -> AnalysisResponse:
    """Test endpoint that returns mock data."""
    
    print(f"🧪 Testing with repo: {request.repo_url}")
    
    # Create mock function analysis
    mock_function_analysis = FunctionAnalysis(
        total_functions=42,
        most_called_function=FunctionDetail(
            name="test_function",
            parameters=["param1", "param2"],
            return_type="str",
            call_count=15,
            calls_made=3
        ),
        most_calling_function=FunctionDetail(
            name="main_function",
            parameters=["args"],
            return_type="None",
            call_count=5,
            calls_made=20
        ),
        dead_functions_count=7,
        sample_functions=[
            FunctionDetail(name="func1", call_count=5),
            FunctionDetail(name="func2", call_count=3),
        ],
        sample_imports=[
            ImportDetail(module="os", imported_symbols=["path"]),
            ImportDetail(module="sys", imported_symbols=["argv"]),
        ]
    )
    
    return AnalysisResponse(function_analysis=mock_function_analysis)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Minimal API is working"}

if __name__ == "__main__":
    print("🚀 Starting minimal test API server...")
    uvicorn.run(app, host="0.0.0.0", port=8002)
