from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List, Tuple, Any, Optional
import graph_sitter
from graph_sitter.core.class_definition import Class
from graph_sitter.core.codebase import Codebase
from graph_sitter.core.external_module import ExternalModule
from graph_sitter.core.file import SourceFile
from graph_sitter.core.function import Function
from graph_sitter.core.import_resolution import Import
from graph_sitter.core.symbol import Symbol
from graph_sitter.enums import EdgeType, SymbolType
from graph_sitter.core.statements.for_loop_statement import ForLoopStatement
from graph_sitter.core.statements.if_block_statement import IfBlockStatement
from graph_sitter.core.statements.try_catch_statement import TryCatchStatement
from graph_sitter.core.statements.while_statement import WhileStatement
# Skip expressions for now due to circular import issues
import math
import re
import requests
from datetime import datetime, timedelta
import subprocess
import os
import tempfile
from fastapi.middleware.cors import CORSMiddleware
import modal
import networkx as nx
import json
import time
import ast
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, Counter

# Modal image configuration
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "graph_sitter", "fastapi", "uvicorn", "gitpython", "requests", "pydantic", "networkx"
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

# Pydantic models
class RepoRequest(BaseModel):
    repo_url: str

class AnalysisResult(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: float
    timestamp: str

# Helper functions
def get_codebase_summary(codebase: Codebase) -> str:
    try:
        nodes = codebase.ctx.get_nodes()
        files = list(codebase.files)
        imports = list(codebase.imports)
        external_modules = list(codebase.external_modules)
        symbols = list(codebase.symbols)
        classes = list(codebase.classes)
        functions = list(codebase.functions)
        global_vars = list(codebase.global_vars)
        interfaces = list(codebase.interfaces)
        
        node_summary = f"""Contains {len(nodes)} nodes
- {len(files)} files
- {len(imports)} imports
- {len(external_modules)} external_modules
- {len(symbols)} symbols
\t- {len(classes)} classes
\t- {len(functions)} functions
\t- {len(global_vars)} global_vars
\t- {len(interfaces)} interfaces
"""
        
        edges = codebase.ctx.edges
        symbol_usage_edges = [x for x in edges if x[2].type == EdgeType.SYMBOL_USAGE]
        import_symbol_edges = [x for x in edges if x[2].type == EdgeType.IMPORT_SYMBOL_RESOLUTION]
        export_edges = [x for x in edges if x[2].type == EdgeType.EXPORT]
        
        edge_summary = f"""Contains {len(edges)} edges
- {len(symbol_usage_edges)} symbol -> used symbol
- {len(import_symbol_edges)} import -> used symbol
- {len(export_edges)} export -> exported symbol
    """
        return f"{node_summary}\n{edge_summary}"
    except Exception as e:
        return f"Error generating codebase summary: {str(e)}"

def get_file_summary(file: SourceFile) -> str:
    try:
        return f"""==== [ `{file.name}` (SourceFile) Dependency Summary ] ====
- {len(file.imports)} imports
- {len(file.symbols)} symbol references
\t- {len(file.classes)} classes
\t- {len(file.functions)} functions
\t- {len(file.global_vars)} global variables
\t- {len(file.interfaces)} interfaces

==== [ `{file.name}` Usage Summary ] ====
- {len(file.imports)} importers
"""
    except Exception as e:
        return f"Error generating file summary for {file.name}: {str(e)}"

def get_class_summary(cls: Class) -> str:
    try:
        return f"""==== [ `{cls.name}` (Class) Dependency Summary ] ====
- parent classes: {getattr(cls, 'parent_class_names', [])}
- {len(cls.methods)} methods
- {len(cls.attributes)} attributes
- {len(cls.decorators)} decorators
- {len(cls.dependencies)} dependencies

{get_symbol_summary(cls)}
    """
    except Exception as e:
        return f"Error generating class summary for {cls.name}: {str(e)}"

def get_function_summary(func: Function) -> str:
    try:
        return f"""==== [ `{func.name}` (Function) Dependency Summary ] ====
- {len(func.return_statements)} return statements
- {len(func.parameters)} parameters
- {len(func.function_calls)} function calls
- {len(func.call_sites)} call sites
- {len(func.decorators)} decorators
- {len(func.dependencies)} dependencies

{get_symbol_summary(func)}
        """
    except Exception as e:
        return f"Error generating function summary for {func.name}: {str(e)}"

def get_symbol_summary(symbol: Symbol) -> str:
    try:
        usages = symbol.symbol_usages
        imported_symbols = [x.imported_symbol for x in usages if isinstance(x, Import)]

        return f"""==== [ `{symbol.name}` ({type(symbol).__name__}) Usage Summary ] ====
- {len(usages)} usages
\t- {len([x for x in usages if isinstance(x, Symbol) and x.symbol_type == SymbolType.Function])} functions
\t- {len([x for x in usages if isinstance(x, Symbol) and x.symbol_type == SymbolType.Class])} classes
\t- {len([x for x in usages if isinstance(x, Symbol) and x.symbol_type == SymbolType.GlobalVar])} global variables
\t- {len([x for x in usages if isinstance(x, Symbol) and x.symbol_type == SymbolType.Interface])} interfaces
\t- {len(imported_symbols)} imports
\t\t- {len([x for x in imported_symbols if isinstance(x, Symbol) and x.symbol_type == SymbolType.Function])} functions
\t\t- {len([x for x in imported_symbols if isinstance(x, Symbol) and x.symbol_type == SymbolType.Class])} classes
\t\t- {len([x for x in imported_symbols if isinstance(x, Symbol) and x.symbol_type == SymbolType.GlobalVar])} global variables
\t\t- {len([x for x in imported_symbols if isinstance(x, Symbol) and x.symbol_type == SymbolType.Interface])} interfaces
\t\t- {len([x for x in imported_symbols if isinstance(x, ExternalModule)])} external modules
\t\t- {len([x for x in imported_symbols if isinstance(x, SourceFile)])} files
    """
    except Exception as e:
        return f"Error generating symbol summary for {symbol.name}: {str(e)}"

def get_function_context(function: Function) -> dict:
    """Get the implementation, dependencies, and usages of a function."""
    try:
        context = {
            "name": function.name,
            "implementation": {"source": getattr(function, 'source', ''), "filepath": getattr(function, 'filepath', '')},
            "dependencies": [],
            "usages": [],
            "parameters": len(function.parameters),
            "return_statements": len(function.return_statements),
            "function_calls": len(function.function_calls),
            "call_sites": len(function.call_sites)
        }
        
        # Add dependencies
        for dep in function.dependencies:
            if isinstance(dep, Import):
                dep = hop_through_imports(dep)
            context["dependencies"].append({
                "name": getattr(dep, 'name', 'unknown'),
                "source": getattr(dep, 'source', ''),
                "filepath": getattr(dep, 'filepath', '')
            })
        
        # Add usages
        for usage in function.usages:
            context["usages"].append({
                "source": getattr(usage.usage_symbol, 'source', ''),
                "filepath": getattr(usage.usage_symbol, 'filepath', ''),
            })
        
        return context
    except Exception as e:
        return {
            "name": function.name,
            "error": str(e),
            "implementation": {"source": "", "filepath": ""},
            "dependencies": [],
            "usages": [],
            "parameters": 0,
            "return_statements": 0,
            "function_calls": 0,
            "call_sites": 0
        }

def hop_through_imports(imp: Import) -> Symbol | ExternalModule:
    """Finds the root symbol for an import."""
    try:
        if isinstance(imp.imported_symbol, Import):
            return hop_through_imports(imp.imported_symbol)
        return imp.imported_symbol
    except Exception as e:
        return imp

def calculate_doi(cls: Class) -> int:
    """Calculate the depth of inheritance for a given class."""
    try:
        return len(getattr(cls, 'superclasses', [])) if hasattr(cls, 'superclasses') else 0
    except Exception:
        return 0

def calculate_halstead_metrics(function: Function) -> Dict[str, float]:
    """Calculate Halstead complexity metrics for a function."""
    try:
        operators, operands = get_operators_and_operands(function)
        
        n1 = len(set(operators))  # Number of distinct operators
        n2 = len(set(operands))   # Number of distinct operands
        N1 = len(operators)       # Total number of operators
        N2 = len(operands)        # Total number of operands
        
        if n1 == 0 or n2 == 0:
            return {
                "vocabulary": 0,
                "length": 0,
                "volume": 0,
                "difficulty": 0,
                "effort": 0
            }
        
        vocabulary = n1 + n2
        length = N1 + N2
        volume = length * math.log2(vocabulary) if vocabulary > 0 else 0
        difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
        effort = difficulty * volume
        
        return {
            "vocabulary": vocabulary,
            "length": length,
            "volume": volume,
            "difficulty": difficulty,
            "effort": effort
        }
    except Exception as e:
        return {
            "vocabulary": 0,
            "length": 0,
            "volume": 0,
            "difficulty": 0,
            "effort": 0,
            "error": str(e)
        }

def get_operators_and_operands(function: Function) -> Tuple[List[str], List[str]]:
    """Extract operators and operands from a function."""
    operators = []
    operands = []
    
    try:
        # Add function name as operand
        operands.append(function.name)
        
        # Add parameters as operands
        for param in function.parameters:
            operands.append(getattr(param, 'name', str(param)))
        
        # Add function calls as operators
        for call in function.function_calls:
            operators.append(getattr(call, 'name', str(call)))
        
        # Basic operators (simplified approach)
        operators.extend(['=', '+', '-', '*', '/', '==', '!=', '<', '>', '<=', '>='])
        
    except Exception as e:
        # Fallback to basic analysis
        operators = ['function_def']
        operands = [function.name]
    
    return operators, operands

def find_dead_code(codebase: Codebase) -> List[Dict[str, Any]]:
    """Find potentially dead/unused functions."""
    dead_functions = []
    
    try:
        for function in codebase.functions:
            # Check if function has no call sites (not called by other functions)
            if len(function.call_sites) == 0:
                # Skip main functions and special methods
                if function.name not in ['main', '__main__', '__init__', '__str__', '__repr__']:
                    dead_functions.append({
                        "name": function.name,
                        "filepath": getattr(function, 'filepath', ''),
                        "line_number": getattr(function, 'start_point', [0])[0] if hasattr(function, 'start_point') else 0,
                        "reason": "No call sites found"
                    })
    except Exception as e:
        print(f"Error in dead code detection: {e}")
    
    return dead_functions

def analyze_repository_structure(codebase: Codebase) -> Dict[str, Any]:
    """Analyze repository structure and organization."""
    structure = {
        "total_files": 0,
        "file_types": {},
        "directories": set(),
        "largest_files": [],
        "file_details": []
    }
    
    try:
        files = list(codebase.files)
        structure["total_files"] = len(files)
        
        for file in files:
            # File extension analysis
            ext = os.path.splitext(file.name)[1] if '.' in file.name else 'no_extension'
            structure["file_types"][ext] = structure["file_types"].get(ext, 0) + 1
            
            # Directory analysis
            dir_path = os.path.dirname(getattr(file, 'filepath', ''))
            if dir_path:
                structure["directories"].add(dir_path)
            
            # File details
            file_info = {
                "name": file.name,
                "path": getattr(file, 'filepath', ''),
                "classes": len(file.classes),
                "functions": len(file.functions),
                "imports": len(file.imports),
                "lines_of_code": len(getattr(file, 'source', '').split('\n'))
            }
            structure["file_details"].append(file_info)
        
        # Convert set to list for JSON serialization
        structure["directories"] = list(structure["directories"])
        
        # Sort files by size and get largest
        structure["file_details"].sort(key=lambda x: x["lines_of_code"], reverse=True)
        structure["largest_files"] = structure["file_details"][:10]
        
    except Exception as e:
        print(f"Error in repository structure analysis: {e}")
    
    return structure

def detect_code_issues(codebase: Codebase) -> Dict[str, Any]:
    """Detect various code quality issues."""
    issues = {
        "total_issues": 0,
        "by_severity": {"critical": 0, "high": 0, "medium": 0, "low": 0},
        "by_type": {},
        "details": []
    }
    
    try:
        # Check for functions without documentation
        undocumented_functions = 0
        for function in codebase.functions:
            if not hasattr(function, 'docstring') or not getattr(function, 'docstring', None):
                undocumented_functions += 1
                issues["details"].append({
                    "type": "missing_documentation",
                    "severity": "medium",
                    "message": f"Function '{function.name}' lacks documentation",
                    "file": getattr(function, 'filepath', ''),
                    "line": getattr(function, 'start_point', [0])[0] if hasattr(function, 'start_point') else 0
                })
        
        issues["by_type"]["missing_documentation"] = undocumented_functions
        issues["by_severity"]["medium"] += undocumented_functions
        
        # Check for long functions (>50 lines)
        long_functions = 0
        for function in codebase.functions:
            source = getattr(function, 'source', '')
            if source:
                lines = len(source.split('\n'))
                if lines > 50:
                    long_functions += 1
                    issues["details"].append({
                        "type": "long_function",
                        "severity": "high",
                        "message": f"Function '{function.name}' is too long ({lines} lines)",
                        "file": getattr(function, 'filepath', ''),
                        "line": getattr(function, 'start_point', [0])[0] if hasattr(function, 'start_point') else 0
                    })
        
        issues["by_type"]["long_function"] = long_functions
        issues["by_severity"]["high"] += long_functions
        
        # Check for complex functions (many parameters)
        complex_functions = 0
        for function in codebase.functions:
            if len(function.parameters) > 5:
                complex_functions += 1
                issues["details"].append({
                    "type": "too_many_parameters",
                    "severity": "medium",
                    "message": f"Function '{function.name}' has too many parameters ({len(function.parameters)})",
                    "file": getattr(function, 'filepath', ''),
                    "line": getattr(function, 'start_point', [0])[0] if hasattr(function, 'start_point') else 0
                })
        
        issues["by_type"]["too_many_parameters"] = complex_functions
        issues["by_severity"]["medium"] += complex_functions
        
        issues["total_issues"] = sum(issues["by_severity"].values())
        
    except Exception as e:
        print(f"Error in issue detection: {e}")
    
    return issues

def calculate_health_score(codebase: Codebase, issues: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate overall codebase health score."""
    try:
        total_functions = len(list(codebase.functions))
        total_classes = len(list(codebase.classes))
        total_files = len(list(codebase.files))
        total_issues = issues["total_issues"]
        
        # Base score
        health_score = 100
        
        # Deduct points for issues
        health_score -= min(issues["by_severity"]["critical"] * 10, 30)
        health_score -= min(issues["by_severity"]["high"] * 5, 25)
        health_score -= min(issues["by_severity"]["medium"] * 2, 20)
        health_score -= min(issues["by_severity"]["low"] * 1, 15)
        
        # Ensure score doesn't go below 0
        health_score = max(health_score, 0)
        
        # Determine grade
        if health_score >= 90:
            grade = "A"
            risk_level = "Low"
        elif health_score >= 80:
            grade = "B"
            risk_level = "Low"
        elif health_score >= 70:
            grade = "C"
            risk_level = "Medium"
        elif health_score >= 60:
            grade = "D"
            risk_level = "High"
        else:
            grade = "F"
            risk_level = "Critical"
        
        # Estimate technical debt (hours to fix issues)
        tech_debt_hours = (
            issues["by_severity"]["critical"] * 4 +
            issues["by_severity"]["high"] * 2 +
            issues["by_severity"]["medium"] * 1 +
            issues["by_severity"]["low"] * 0.5
        )
        
        return {
            "health_score": health_score,
            "health_grade": grade,
            "risk_level": risk_level,
            "technical_debt_hours": tech_debt_hours,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "total_files": total_files,
            "total_issues": total_issues
        }
    except Exception as e:
        return {
            "health_score": 0,
            "health_grade": "F",
            "risk_level": "Critical",
            "technical_debt_hours": 0,
            "error": str(e)
        }

def analyze_codebase_comprehensive(codebase: Codebase) -> Dict[str, Any]:
    """Perform comprehensive codebase analysis."""
    start_time = time.time()
    
    try:
        # Basic metrics
        total_functions = len(list(codebase.functions))
        total_classes = len(list(codebase.classes))
        total_files = len(list(codebase.files))
        
        print(f"Analyzing codebase with {total_functions} functions, {total_classes} classes, {total_files} files")
        
        # Function analysis
        function_contexts = []
        halstead_metrics = []
        
        functions_list = list(codebase.functions)[:20]  # Limit to first 20 for performance
        for i, function in enumerate(functions_list):
            try:
                print(f"Analyzing function {i+1}/{len(functions_list)}: {function.name}")
                context = get_function_context(function)
                function_contexts.append(context)
                
                halstead = calculate_halstead_metrics(function)
                halstead["function_name"] = function.name
                halstead_metrics.append(halstead)
            except Exception as e:
                print(f"Error analyzing function {function.name}: {e}")
        
        # Dead code detection
        print("Detecting dead code...")
        dead_functions = find_dead_code(codebase)
        
        # Repository structure
        print("Analyzing repository structure...")
        repo_structure = analyze_repository_structure(codebase)
        
        # Issue detection
        print("Detecting code issues...")
        issues = detect_code_issues(codebase)
        
        # Health assessment
        print("Calculating health metrics...")
        health_metrics = calculate_health_score(codebase, issues)
        
        # Most important functions (by call sites)
        important_functions = []
        try:
            functions_by_calls = sorted(
                list(codebase.functions), 
                key=lambda f: len(f.call_sites), 
                reverse=True
            )[:10]
            
            for func in functions_by_calls:
                important_functions.append({
                    "name": func.name,
                    "call_sites": len(func.call_sites),
                    "filepath": getattr(func, 'filepath', ''),
                    "importance_score": len(func.call_sites) + len(func.function_calls)
                })
        except Exception as e:
            print(f"Error finding important functions: {e}")
        
        # Entry points detection
        entry_points = []
        try:
            entry_patterns = ['main', '__main__', 'run', 'start', 'execute', 'init', 'setup', 'app', 'server', 'cli']
            for function in codebase.functions:
                if any(pattern in function.name.lower() for pattern in entry_patterns):
                    entry_points.append({
                        "name": function.name,
                        "filepath": getattr(function, 'filepath', ''),
                        "type": "potential_entry_point"
                    })
        except Exception as e:
            print(f"Error finding entry points: {e}")
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "summary": {
                "total_functions": total_functions,
                "total_classes": total_classes,
                "total_files": total_files,
                "processing_time": processing_time
            },
            "function_analysis": {
                "contexts": function_contexts,
                "most_important": important_functions,
                "entry_points": entry_points,
                "dead_functions": dead_functions
            },
            "quality_metrics": {
                "halstead_metrics": halstead_metrics,
                "issues": issues,
                "health_assessment": health_metrics
            },
            "repository_structure": repo_structure,
            "codebase_summary": get_codebase_summary(codebase),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "processing_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat()
        }

def save_analysis_to_json(analysis_result: Dict[str, Any], filename: str = None) -> str:
    """Save analysis results to JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"codebase_analysis_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(analysis_result, f, indent=2, default=str)
        return filename
    except Exception as e:
        print(f"Error saving to JSON: {e}")
        return ""

# API Endpoints
@fastapi_app.post("/analyze_repo")
async def analyze_repository(request: RepoRequest) -> AnalysisResult:
    """
    Analyze a repository using graph_sitter and return comprehensive results.
    """
    start_time = time.time()
    
    try:
        # Load codebase using graph_sitter
        print(f"Loading codebase from: {request.repo_url}")
        codebase = Codebase.from_repo(request.repo_url)
        print("Codebase loaded successfully!")
        
        # Perform comprehensive analysis
        analysis_result = analyze_codebase_comprehensive(codebase)
        
        # Save results to JSON
        json_filename = save_analysis_to_json(analysis_result)
        analysis_result["json_file"] = json_filename
        
        processing_time = time.time() - start_time
        
        return AnalysisResult(
            success=True,
            data=analysis_result,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Analysis failed: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        
        return AnalysisResult(
            success=False,
            error=error_msg,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )

@fastapi_app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "codebase-analytics-api",
        "graph_sitter_available": True
    }

@fastapi_app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Codebase Analytics API",
        "version": "3.0.0",
        "description": "Advanced codebase analysis using graph_sitter",
        "endpoints": {
            "analyze_repo": "/analyze_repo",
            "health": "/health",
            "docs": "/docs"
        },
        "features": [
            "Comprehensive function analysis",
            "Dead code detection",
            "Halstead complexity metrics",
            "Code quality issue detection",
            "Repository structure analysis",
            "Health assessment and scoring",
            "JSON result export"
        ],
        "powered_by": "graph_sitter"
    }

# Modal deployment
@app.function(image=image)
@modal.asgi_app()
def fastapi_modal_app():
    return fastapi_app

if __name__ == "__main__":
    # For local testing
    import uvicorn
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

