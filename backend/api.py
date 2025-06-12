# Add BaseModelWithConfig at the top
from pydantic import BaseModel, ConfigDict
from fastapi import FastAPI, HTTPException
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
from datetime import datetime, timedelta
import subprocess
import os
import tempfile
from fastapi.middleware.cors import CORSMiddleware
import modal
from collections import Counter
import networkx as nx
from pathlib import Path

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

# Base models for codebase analysis
class BaseModelWithConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

class CodebaseStats(BaseModelWithConfig):
    test_functions_count: int
    test_classes_count: int
    tests_per_file: float
    total_classes: int
    total_functions: int
    total_imports: int
    deepest_inheritance_class: Optional[Dict[str, Any]]
    recursive_functions: List[str]
    most_called_function: Dict[str, Any]
    function_with_most_calls: Dict[str, Any]
    unused_functions: List[Dict[str, str]]
    dead_code: List[Dict[str, str]]

class FileTestStats(BaseModelWithConfig):
    filepath: str
    test_class_count: int
    file_length: int
    function_count: int

class FunctionContext(BaseModelWithConfig):
    implementation: Dict[str, Any]
    dependencies: List[Dict[str, Any]]
    usages: List[Dict[str, Any]]

# Models for extended analysis
class TestAnalysis(BaseModelWithConfig):
    total_test_functions: int
    total_test_classes: int
    tests_per_file: float
    top_test_files: List[Dict[str, Any]]

class FunctionAnalysis(BaseModelWithConfig):
    total_functions: int
    most_called_function: Dict[str, Any]
    function_with_most_calls: Dict[str, Any]
    recursive_functions: List[str]
    unused_functions: List[Dict[str, str]]
    dead_code: List[Dict[str, str]]

class ClassAnalysis(BaseModelWithConfig):
    total_classes: int
    deepest_inheritance: Optional[Dict[str, Any]]
    total_imports: int

class FileIssue(BaseModelWithConfig):
    critical: List[Dict[str, str]]
    major: List[Dict[str, str]]
    minor: List[Dict[str, str]]

class ExtendedAnalysis(BaseModelWithConfig):
    test_analysis: TestAnalysis
    function_analysis: FunctionAnalysis
    class_analysis: ClassAnalysis
    file_issues: Dict[str, FileIssue]
    repo_structure: Dict[str, Any]

class RepoRequest(BaseModelWithConfig):
    repo_url: str

class Symbol(BaseModelWithConfig):
    id: str
    name: str
    type: str  # function, class, or variable
    filepath: str
    start_line: int
    end_line: int
    issues: Optional[List[Dict[str, str]]] = None

class FileNode(BaseModelWithConfig):
    name: str
    type: str  # file or directory
    path: str
    issues: Optional[Dict[str, int]] = None
    symbols: Optional[List[Symbol]] = None
    children: Optional[Dict[str, "FileNode"]] = None

class AnalysisResponse(BaseModelWithConfig):
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
    
    # Repository structure with symbols
    repo_structure: FileNode
