from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Tuple, Any, Optional
from codegen import Codebase
from codegen.sdk.core.statements.for_loop_statement import ForLoopStatement
from codegen.sdk.core.statements.if_block_statement import IfBlockStatement
from codegen.sdk.core.statements.try_catch_statement import TryCatchStatement
from codegen.sdk.core.statements.while_statement import WhileStatement
from codegen.sdk.core.expressions.binary_expression import BinaryExpression
from codegen.sdk.core.expressions.unary_expression import UnaryExpression
from codegen.sdk.core.expressions.comparison_expression import ComparisonExpression
import math
import re
import requests
import json
from datetime import datetime, timedelta
import subprocess
import os
import tempfile
from fastapi.middleware.cors import CORSMiddleware
import modal
from collections import Counter
import networkx as nx
from pathlib import Path
from codegen.sdk.core.class_definition import Class
from codegen.sdk.core.codebase import Codebase
from codegen.sdk.core.external_module import ExternalModule
from codegen.sdk.core.file import SourceFile
from codegen.sdk.core.function import Function
from codegen.sdk.core.import_resolution import Import
from codegen.sdk.core.symbol import Symbol
from codegen.sdk.enums import EdgeType, SymbolType

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "codegen", "fastapi", "uvicorn", "gitpython", "requests", "pydantic", "datetime",
        "networkx"  # Added for call chain analysis
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

# Mock data for development and testing
MOCK_DATA = {
    "name": "codebase-analytics",
    "description": "Analytics for codebase maintainability and complexity",
    "stars": 128,
    "forks": 42,
    "issues": 15,
    "language": "TypeScript",
    "contributors": 8,
    "commits": 342,
    "codeQuality": {
        "complexity": 72,
        "maintainability": 85,
        "testCoverage": 68,
        "technicalDebt": 45
    },
    "languageBreakdown": [
        {"name": "TypeScript", "value": 0.65, "color": "#3178c6"},
        {"name": "JavaScript", "value": 0.15, "color": "#f7df1e"},
        {"name": "Python", "value": 0.12, "color": "#3572A5"},
        {"name": "CSS", "value": 0.08, "color": "#563d7c"}
    ],
    "commitActivity": [
        {"month": "Jan", "commits": 42},
        {"month": "Feb", "commits": 38},
        {"month": "Mar", "commits": 56},
        {"month": "Apr", "commits": 32},
        {"month": "May", "commits": 45},
        {"month": "Jun", "commits": 62},
        {"month": "Jul", "commits": 58},
        {"month": "Aug", "commits": 40},
        {"month": "Sep", "commits": 35},
        {"month": "Oct", "commits": 48},
        {"month": "Nov", "commits": 52},
        {"month": "Dec", "commits": 30}
    ]
}

class AnalysisRequest(BaseModel):
    owner: str
    repo: str

@app.function()
@fastapi_app.get("/api/analyze")
async def analyze_repo(owner: str, repo: str):
    """
    Analyze a GitHub repository and return metrics
    """
    try:
        # For development/demo purposes, return mock data
        # In production, this would call the actual analysis functions
        return MOCK_DATA
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing repository: {str(e)}")

@app.function()
@fastapi_app.get("/api/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# Additional endpoints would be added here for specific analysis features

# Mount the FastAPI app
@app.route("/", method=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
def fastapi_handler(raw_request: modal.Request):
    """
    Handle all FastAPI requests
    """
    scope = {
        "type": "http",
        "path": raw_request.path,
        "raw_path": raw_request.path.encode(),
        "root_path": "",
        "scheme": "https",
        "query_string": raw_request.query_string.encode(),
        "headers": [[k.lower().encode(), v.encode()] for k, v in raw_request.headers.items()],
        "method": raw_request.method,
        "client": None,
        "server": None,
    }
    
    async def receive():
        return {"type": "http.request", "body": raw_request.body}
    
    response_body = []
    response_status = None
    response_headers = []
    
    async def send(message):
        nonlocal response_body, response_status, response_headers
        if message["type"] == "http.response.start":
            response_status = message["status"]
            response_headers = message.get("headers", [])
        elif message["type"] == "http.response.body":
            response_body.append(message.get("body", b""))
    
    from fastapi.applications import ASGIApp
    asgi_app: ASGIApp = fastapi_app
    
    import asyncio
    asyncio.run(asgi_app(scope, receive, send))
    
    headers = {k.decode(): v.decode() for k, v in response_headers}
    
    return modal.Response(
        body=b"".join(response_body),
        status_code=response_status,
        headers=headers,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

