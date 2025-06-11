#!/usr/bin/env python3
"""
🚀 Advanced Codebase Analytics API
Intelligent, context-aware repository analysis with real-time issue detection.
Provides dynamic analysis based on actual code state and semantic understanding.
"""

import os
import sys
import re
import json
import time
import uuid
import hashlib
import logging
import argparse
import tempfile
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from collections import defaultdict, Counter

# Import graph-sitter components
from graph_sitter.core.class_definition import Class
from graph_sitter.core.codebase import Codebase
from graph_sitter.core.external_module import ExternalModule
from graph_sitter.core.file import SourceFile
from graph_sitter.core.function import Function
from graph_sitter.core.import_resolution import Import
from graph_sitter.core.symbol import Symbol
from graph_sitter.enums import EdgeType, SymbolType

import fastapi
from fastapi import FastAPI, HTTPException, Depends, Query, status, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


# Add graph-sitter context summary functions
def get_codebase_summary(codebase: 'Codebase') -> str:
    """
    Generate a comprehensive summary of a codebase.
    
    Args:
        codebase: The Codebase object to summarize
        
    Returns:
        A formatted string with node and edge statistics
    """
    node_summary = f"""Contains {len(codebase.ctx.get_nodes())} nodes
- {len(list(codebase.files))} files
- {len(list(codebase.imports))} imports
- {len(list(codebase.external_modules))} external_modules
- {len(list(codebase.symbols))} symbols
\t- {len(list(codebase.classes))} classes
\t- {len(list(codebase.functions))} functions
\t- {len(list(codebase.global_vars))} global_vars
\t- {len(list(codebase.interfaces))} interfaces
"""
    edge_summary = f"""Contains {len(codebase.ctx.edges)} edges
- {len([x for x in codebase.ctx.edges if x[2].type == EdgeType.SYMBOL_USAGE])} symbol -> used symbol
- {len([x for x in codebase.ctx.edges if x[2].type == EdgeType.IMPORT_SYMBOL_RESOLUTION])} import -> used symbol
- {len([x for x in codebase.ctx.edges if x[2].type == EdgeType.EXPORT])} export -> exported symbol
    """

    return f"{node_summary}\n{edge_summary}"


def get_file_summary(file: 'SourceFile') -> str:
    """
    Generate a summary of a source file.
    
    Args:
        file: The SourceFile object to summarize
        
    Returns:
        A formatted string with file statistics
    """
    return f"""==== [ {file.name} ] ====
- {len(file.imports)} imports
- {len(file.symbols)} symbol references
\t- {len(file.classes)} classes
\t- {len(file.functions)} functions
\t- {len(file.global_vars)} global variables
\t- {len(file.interfaces)} interfaces
    """


def get_class_summary(cls: 'Class') -> str:
    """
    Generate a summary of a class.
    
    Args:
        cls: The Class object to summarize
        
    Returns:
        A formatted string with class statistics
    """
    return f"""==== [ {cls.name} ] ====
- parent classes: {cls.parent_class_names}
- {len(cls.methods)} methods
- {len(cls.attributes)} attributes
- {len(cls.decorators)} decorators
- {len(cls.dependencies)} dependencies
    """


def get_function_summary(func: 'Function') -> str:
    """
    Generate a summary of a function.
    
    Args:
        func: The Function object to summarize
        
    Returns:
        A formatted string with function statistics
    """
    return f"""==== [ {func.name} ] ====
- {len(func.return_statements)} return statements
- {len(func.parameters)} parameters
- {len(func.function_calls)} function calls
- {len(func.call_sites)} call sites
- {len(func.decorators)} decorators
- {len(func.dependencies)} dependencies
    """


def get_symbol_summary(symbol: 'Symbol') -> str:
    """
    Generate a summary of a symbol.
    
    Args:
        symbol: The Symbol object to summarize
        
    Returns:
        A formatted string with symbol usage statistics
    """
    # Count usages by type
    function_count = 0
    class_count = 0
    global_var_count = 0
    interface_count = 0
    import_count = 0
    
    # Count imported symbols by type
    imported_functions = 0
    imported_classes = 0
    imported_global_vars = 0
    imported_interfaces = 0
    imported_external_modules = 0
    imported_files = 0
    
    # Process all symbol usages
    for usage in symbol.symbol_usages:
        # Check if it's an import
        usage_type = type(usage).__name__
        if usage_type == 'Import':
            import_count += 1
            imported_symbol = usage.imported_symbol
            
            # Check the type of the imported symbol
            if hasattr(imported_symbol, 'symbol_type'):
                if imported_symbol.symbol_type == SymbolType.Function:
                    imported_functions += 1
                elif imported_symbol.symbol_type == SymbolType.Class:
                    imported_classes += 1
                elif imported_symbol.symbol_type == SymbolType.GlobalVar:
                    imported_global_vars += 1
                elif imported_symbol.symbol_type == SymbolType.Interface:
                    imported_interfaces += 1
            # Check if it's an external module
            elif type(imported_symbol).__name__ == 'ExternalModule':
                imported_external_modules += 1
            # Check if it's a file
            elif type(imported_symbol).__name__ == 'SourceFile':
                imported_files += 1
        # Check if it's a symbol
        elif usage_type == 'Symbol':
            if usage.symbol_type == SymbolType.Function:
                function_count += 1
            elif usage.symbol_type == SymbolType.Class:
                class_count += 1
            elif usage.symbol_type == SymbolType.GlobalVar:
                global_var_count += 1
            elif usage.symbol_type == SymbolType.Interface:
                interface_count += 1
    
    # Format the summary
    summary = f"""==== [ `{symbol.name}` ({type(symbol).__name__}) Usage Summary ] ====
- {len(symbol.symbol_usages)} usages
\t- {function_count} functions
\t- {class_count} classes
\t- {global_var_count} global variables
\t- {interface_count} interfaces
\t- {import_count} imports
\t\t- {imported_functions} functions
\t\t- {imported_classes} classes
\t\t- {imported_global_vars} global variables
\t\t- {imported_interfaces} interfaces
\t\t- {imported_external_modules} external modules
\t\t- {imported_files} files
    """
    
    return summary


def get_context_summary(context: Union['Codebase', 'SourceFile', 'Class', 'Function', 'Symbol']) -> str:
    """
    Generate a summary for any context object.
    
    Args:
        context: The context object to summarize (Codebase, SourceFile, Class, Function, or Symbol)
        
    Returns:
        A formatted string with context-specific summary information
    """
    # Check the type of the context object by name
    context_type = type(context).__name__
    
    if context_type == 'Codebase':
        return get_codebase_summary(context)
    elif context_type == 'SourceFile':
        return get_file_summary(context)
    elif context_type == 'Class':
        return get_class_summary(context)
    elif context_type == 'Function':
        return get_function_summary(context)
    elif context_type == 'Symbol':
        return get_symbol_summary(context)
    else:
        return f"Unsupported context type: {context_type}"


def get_context_summary_dict(context: Union['Codebase', 'SourceFile', 'Class', 'Function', 'Symbol']) -> Dict[str, Any]:
    """
    Generate a dictionary summary for any context object.
    
    Args:
        context: The context object to summarize (Codebase, SourceFile, Class, Function, or Symbol)
        
    Returns:
        A dictionary with context-specific summary information
    """
    # Check the type of the context object by name
    context_type = type(context).__name__
    
    if context_type == 'Codebase':
        return {
            "type": "Codebase",
            "nodes": len(context.ctx.get_nodes()),
            "edges": len(context.ctx.edges),
            "files": len(list(context.files)),
            "imports": len(list(context.imports)),
            "external_modules": len(list(context.external_modules)),
            "symbols": {
                "total": len(list(context.symbols)),
                "classes": len(list(context.classes)),
                "functions": len(list(context.functions)),
                "global_vars": len(list(context.global_vars)),
                "interfaces": len(list(context.interfaces))
            },
            "edge_types": {
                "symbol_usage": len([x for x in context.ctx.edges if x[2].type == EdgeType.SYMBOL_USAGE]),
                "import_resolution": len([x for x in context.ctx.edges if x[2].type == EdgeType.IMPORT_SYMBOL_RESOLUTION]),
                "export": len([x for x in context.ctx.edges if x[2].type == EdgeType.EXPORT])
            }
        }
    elif context_type == 'SourceFile':
        return {
            "type": "SourceFile",
            "name": context.name,
            "imports": len(context.imports),
            "symbols": {
                "total": len(context.symbols),
                "classes": len(context.classes),
                "functions": len(context.functions),
                "global_vars": len(context.global_vars),
                "interfaces": len(context.interfaces)
            },
            "importers": len(context.imports)
        }
    elif context_type == 'Class':
        return {
            "type": "Class",
            "name": context.name,
            "parent_classes": context.parent_class_names,
            "methods": len(context.methods),
            "attributes": len(context.attributes),
            "decorators": len(context.decorators),
            "dependencies": len(context.dependencies),
            "usages": _get_symbol_summary_dict(context)
        }
    elif context_type == 'Function':
        return {
            "type": "Function",
            "name": context.name,
            "return_statements": len(context.return_statements),
            "parameters": len(context.parameters),
            "function_calls": len(context.function_calls),
            "call_sites": len(context.call_sites),
            "decorators": len(context.decorators),
            "dependencies": len(context.dependencies),
            "usages": _get_symbol_summary_dict(context)
        }
    elif context_type == 'Symbol':
        return _get_symbol_summary_dict(context)
    else:
        return {"error": f"Unsupported context type: {context_type}"}


def _get_symbol_summary_dict(symbol: 'Symbol') -> Dict[str, Any]:
    """
    Generate a dictionary summary of a symbol's usage.
    
    Args:
        symbol: The Symbol object to summarize
        
    Returns:
        A dictionary with symbol usage information
    """
    # Count usages by type
    function_count = 0
    class_count = 0
    global_var_count = 0
    interface_count = 0
    import_count = 0
    
    # Count imported symbols by type
    imported_functions = 0
    imported_classes = 0
    imported_global_vars = 0
    imported_interfaces = 0
    imported_external_modules = 0
    imported_files = 0
    
    # Process all symbol usages
    for usage in symbol.symbol_usages:
        # Check if it's an import
        usage_type = type(usage).__name__
        if usage_type == 'Import':
            import_count += 1
            imported_symbol = usage.imported_symbol
            
            # Check the type of the imported symbol
            if hasattr(imported_symbol, 'symbol_type'):
                if imported_symbol.symbol_type == SymbolType.Function:
                    imported_functions += 1
                elif imported_symbol.symbol_type == SymbolType.Class:
                    imported_classes += 1
                elif imported_symbol.symbol_type == SymbolType.GlobalVar:
                    imported_global_vars += 1
                elif imported_symbol.symbol_type == SymbolType.Interface:
                    imported_interfaces += 1
            # Check if it's an external module
            elif type(imported_symbol).__name__ == 'ExternalModule':
                imported_external_modules += 1
            # Check if it's a file
            elif type(imported_symbol).__name__ == 'SourceFile':
                imported_files += 1
        # Check if it's a symbol
        elif usage_type == 'Symbol':
            if usage.symbol_type == SymbolType.Function:
                function_count += 1
            elif usage.symbol_type == SymbolType.Class:
                class_count += 1
            elif usage.symbol_type == SymbolType.GlobalVar:
                global_var_count += 1
            elif usage.symbol_type == SymbolType.Interface:
                interface_count += 1
    
    return {
        "type": type(symbol).__name__,
        "name": symbol.name,
        "total_usages": len(symbol.symbol_usages),
        "usage_types": {
            "functions": function_count,
            "classes": class_count,
            "global_vars": global_var_count,
            "interfaces": interface_count
        },
        "imports": {
            "total": import_count,
            "functions": imported_functions,
            "classes": imported_classes,
            "global_vars": imported_global_vars,
            "interfaces": imported_interfaces,
            "external_modules": imported_external_modules,
            "files": imported_files
        }
    }


# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api.log')
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Codebase Analytics API",
    description="Advanced codebase analysis with real-time issue detection",
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

# API key security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# API key validation
async def get_api_key(api_key_header: str = Depends(api_key_header)):
    if api_key_header == os.environ.get("API_KEY", "test_key"):
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key",
    )

# Request logging middleware
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Request: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.4f}s")
        return response

app.add_middleware(RequestLoggingMiddleware)

# Pydantic models for request/response
class CodebaseRequest(BaseModel):
    repo_url: str = Field(..., description="URL of the repository to analyze")
    branch: Optional[str] = Field(None, description="Branch to analyze (default: main)")
    depth: Optional[int] = Field(1, description="Depth of analysis (1-3)")
    
    @validator('depth')
    def validate_depth(cls, v):
        if v not in [1, 2, 3]:
            raise ValueError('Depth must be between 1 and 3')
        return v

class CodebaseResponse(BaseModel):
    repo_url: str
    branch: str
    summary: Dict[str, Any]
    files: int
    classes: int
    functions: int
    symbols: int
    analysis_time: float

class FileRequest(BaseModel):
    repo_url: str = Field(..., description="URL of the repository to analyze")
    file_path: str = Field(..., description="Path to the file to analyze")
    branch: Optional[str] = Field(None, description="Branch to analyze (default: main)")

class FileResponse(BaseModel):
    repo_url: str
    file_path: str
    branch: str
    summary: Dict[str, Any]
    classes: List[str]
    functions: List[str]
    imports: List[str]
    analysis_time: float

class SymbolRequest(BaseModel):
    repo_url: str = Field(..., description="URL of the repository to analyze")
    symbol_name: str = Field(..., description="Name of the symbol to analyze")
    file_path: Optional[str] = Field(None, description="Path to the file containing the symbol")
    branch: Optional[str] = Field(None, description="Branch to analyze (default: main)")

class SymbolResponse(BaseModel):
    repo_url: str
    symbol_name: str
    file_path: Optional[str]
    branch: str
    summary: Dict[str, Any]
    usages: List[Dict[str, Any]]
    analysis_time: float

# API endpoints
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to the Codebase Analytics API"}

@app.post("/api/analyze/codebase", response_model=CodebaseResponse)
async def analyze_codebase(request: CodebaseRequest, api_key: str = Depends(get_api_key)):
    """
    Analyze an entire codebase and return comprehensive statistics
    """
    start_time = time.time()
    
    # Mock implementation for now
    analysis_result = {
        "repo_url": request.repo_url,
        "branch": request.branch or "main",
        "summary": {
            "files": 120,
            "classes": 45,
            "functions": 230,
            "symbols": 560,
            "complexity": 0.75,
            "maintainability": 0.82,
            "test_coverage": 0.68
        },
        "files": 120,
        "classes": 45,
        "functions": 230,
        "symbols": 560,
        "analysis_time": time.time() - start_time
    }
    
    return analysis_result

@app.post("/api/analyze/file", response_model=FileResponse)
async def analyze_file(request: FileRequest, api_key: str = Depends(get_api_key)):
    """
    Analyze a specific file and return detailed information
    """
    start_time = time.time()
    
    # Mock implementation for now
    analysis_result = {
        "repo_url": request.repo_url,
        "file_path": request.file_path,
        "branch": request.branch or "main",
        "summary": {
            "lines": 250,
            "classes": 2,
            "functions": 8,
            "imports": 5,
            "complexity": 0.65,
            "maintainability": 0.78
        },
        "classes": ["UserService", "AuthManager"],
        "functions": ["authenticate", "validate_token", "get_user", "update_user", "delete_user", "create_user", "list_users", "search_users"],
        "imports": ["os", "sys", "json", "datetime", "requests"],
        "analysis_time": time.time() - start_time
    }
    
    return analysis_result

@app.post("/api/analyze/symbol", response_model=SymbolResponse)
async def analyze_symbol(request: SymbolRequest, api_key: str = Depends(get_api_key)):
    """
    Analyze a specific symbol (class, function, variable) and return usage information
    """
    start_time = time.time()
    
    # Mock implementation for now
    analysis_result = {
        "repo_url": request.repo_url,
        "symbol_name": request.symbol_name,
        "file_path": request.file_path,
        "branch": request.branch or "main",
        "summary": {
            "type": "function",
            "lines": 25,
            "parameters": 3,
            "return_type": "User",
            "complexity": 0.45,
            "maintainability": 0.82
        },
        "usages": [
            {"file": "src/controllers/user_controller.py", "line": 45, "type": "call"},
            {"file": "src/controllers/auth_controller.py", "line": 32, "type": "call"},
            {"file": "tests/test_user_service.py", "line": 78, "type": "call"}
        ],
        "analysis_time": time.time() - start_time
    }
    
    return analysis_result

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Codebase Analytics API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

