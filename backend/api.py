from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
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
from graph_sitter.core.class_definition import Class
from graph_sitter.core.codebase import Codebase
from graph_sitter.core.external_module import ExternalModule
from graph_sitter.core.file import SourceFile
from graph_sitter.core.function import Function
from graph_sitter.core.import_resolution import Import
from graph_sitter.core.symbol import Symbol
from graph_sitter.enums import EdgeType, SymbolType

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

class FileTestStats(BaseModelWithConfig):
    filepath: str
    test_class_count: int
    file_length: int
    function_count: int

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
    type: str  # 'function', 'class', or 'variable'
    filepath: str
    start_line: int
    end_line: int
    issues: Optional[List[Dict[str, str]]] = None

class FileNode(BaseModelWithConfig):
    name: str
    type: str  # 'file' or 'directory'
    path: str
    issues: Optional[Dict[str, int]] = None
    symbols: Optional[List[Symbol]] = None
    children: Optional[Dict[str, 'FileNode']] = None

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

# ... [Keep all existing helper functions] ...

def get_file_type(filename: str) -> str:
    """Get the type of file based on its extension."""
    ext = Path(filename).suffix.lower()
    if ext in ['.py', '.pyi', '.pyx']:
        return 'python'
    elif ext in ['.js', '.jsx', '.ts', '.tsx']:
        return 'javascript'
    elif ext in ['.java']:
        return 'java'
    elif ext in ['.c', '.cpp', '.h', '.hpp']:
        return 'cpp'
    elif ext in ['.go']:
        return 'go'
    elif ext in ['.rs']:
        return 'rust'
    elif ext in ['.rb']:
        return 'ruby'
    elif ext in ['.php']:
        return 'php'
    elif ext in ['.cs']:
        return 'csharp'
    elif ext in ['.swift']:
        return 'swift'
    elif ext in ['.kt']:
        return 'kotlin'
    elif ext in ['.scala']:
        return 'scala'
    elif ext in ['.html', '.htm']:
        return 'html'
    elif ext in ['.css', '.scss', '.sass', '.less']:
        return 'css'
    elif ext in ['.json']:
        return 'json'
    elif ext in ['.xml']:
        return 'xml'
    elif ext in ['.md', '.markdown']:
        return 'markdown'
    elif ext in ['.yml', '.yaml']:
        return 'yaml'
    elif ext in ['.sh', '.bash']:
        return 'shell'
    elif ext in ['.sql']:
        return 'sql'
    elif ext in ['.dockerfile', '.containerfile']:
        return 'docker'
    elif ext in ['.gitignore', '.dockerignore']:
        return 'config'
    elif ext in ['.txt']:
        return 'text'
    else:
        return 'unknown'

def get_detailed_symbol_context(symbol: Symbol) -> Dict[str, Any]:
    """Get detailed context for any symbol type."""
    base_info = {
        'id': str(hash(symbol.name + symbol.filepath)),
        'name': symbol.name,
        'type': symbol.__class__.__name__,
        'filepath': symbol.filepath,
        'start_line': symbol.start_point[0] if hasattr(symbol, 'start_point') else 0,
        'end_line': symbol.end_point[0] if hasattr(symbol, 'end_point') else 0,
        'source': symbol.source if hasattr(symbol, 'source') else None,
    }

    # Get usage statistics
    usages = symbol.symbol_usages
    imported_symbols = [x.imported_symbol for x in usages if isinstance(x, Import)]
    
    usage_stats = {
        'total_usages': len(usages),
        'usage_breakdown': {
            'functions': len([x for x in usages if isinstance(x, Symbol) and x.symbol_type == SymbolType.Function]),
            'classes': len([x for x in usages if isinstance(x, Symbol) and x.symbol_type == SymbolType.Class]),
            'global_vars': len([x for x in usages if isinstance(x, Symbol) and x.symbol_type == SymbolType.GlobalVar]),
            'interfaces': len([x for x in usages if isinstance(x, Symbol) and x.symbol_type == SymbolType.Interface])
        },
        'imports': {
            'total': len(imported_symbols),
            'breakdown': {
                'functions': len([x for x in imported_symbols if isinstance(x, Symbol) and x.symbol_type == SymbolType.Function]),
                'classes': len([x for x in imported_symbols if isinstance(x, Symbol) and x.symbol_type == SymbolType.Class]),
                'global_vars': len([x for x in imported_symbols if isinstance(x, Symbol) and x.symbol_type == SymbolType.GlobalVar]),
                'interfaces': len([x for x in imported_symbols if isinstance(x, Symbol) and x.symbol_type == SymbolType.Interface]),
                'external_modules': len([x for x in imported_symbols if isinstance(x, ExternalModule)]),
                'files': len([x for x in imported_symbols if isinstance(x, SourceFile)])
            }
        }
    }

    # Add type-specific information
    if isinstance(symbol, Function):
        base_info.update({
            'function_info': {
                'return_statements': len(symbol.return_statements),
                'parameters': [
                    {
                        'name': p.name,
                        'type': p.type if hasattr(p, 'type') else None,
                        'default_value': p.default_value if hasattr(p, 'default_value') else None
                    }
                    for p in symbol.parameters
                ],
                'function_calls': [
                    {
                        'name': call.name,
                        'args': [arg.source for arg in call.args] if hasattr(call, 'args') else [],
                        'line': call.start_point[0] if hasattr(call, 'start_point') else 0
                    }
                    for call in symbol.function_calls
                ],
                'call_sites': [
                    {
                        'caller': site.parent_function.name if hasattr(site, 'parent_function') else None,
                        'line': site.start_point[0] if hasattr(site, 'start_point') else 0,
                        'file': site.filepath if hasattr(site, 'filepath') else None
                    }
                    for site in symbol.call_sites
                ],
                'decorators': [d.source for d in symbol.decorators] if hasattr(symbol, 'decorators') else [],
                'dependencies': [
                    {
                        'name': dep.name,
                        'type': dep.__class__.__name__,
                        'filepath': dep.filepath if hasattr(dep, 'filepath') else None
                    }
                    for dep in symbol.dependencies
                ] if hasattr(symbol, 'dependencies') else []
            }
        })

        # Add complexity metrics
        if hasattr(symbol, 'code_block'):
            complexity = calculate_cyclomatic_complexity(symbol)
            operators, operands = get_operators_and_operands(symbol)
            volume, N1, N2, n1, n2 = calculate_halstead_volume(operators, operands)
            loc = len(symbol.code_block.source.splitlines())
            mi_score = calculate_maintainability_index(volume, complexity, loc)

            base_info['metrics'] = {
                'cyclomatic_complexity': {
                    'value': complexity,
                    'rank': cc_rank(complexity)
                },
                'halstead_metrics': {
                    'volume': volume,
                    'unique_operators': n1,
                    'unique_operands': n2,
                    'total_operators': N1,
                    'total_operands': N2
                },
                'maintainability_index': {
                    'value': mi_score,
                    'rank': get_maintainability_rank(mi_score)
                },
                'lines_of_code': {
                    'total': loc,
                    'code': len([l for l in symbol.code_block.source.splitlines() if l.strip()]),
                    'comments': len([l for l in symbol.code_block.source.splitlines() if l.strip().startswith('#')])
                }
            }

    elif isinstance(symbol, Class):
        base_info.update({
            'class_info': {
                'parent_classes': symbol.parent_class_names,
                'methods': [
                    {
                        'name': m.name,
                        'parameters': len(m.parameters) if hasattr(m, 'parameters') else 0,
                        'line': m.start_point[0] if hasattr(m, 'start_point') else 0
                    }
                    for m in symbol.methods
                ],
                'attributes': [
                    {
                        'name': a.name,
                        'type': a.type if hasattr(a, 'type') else None,
                        'line': a.start_point[0] if hasattr(a, 'start_point') else 0
                    }
                    for a in symbol.attributes
                ],
                'decorators': [d.source for d in symbol.decorators] if hasattr(symbol, 'decorators') else [],
                'dependencies': [
                    {
                        'name': dep.name,
                        'type': dep.__class__.__name__,
                        'filepath': dep.filepath if hasattr(dep, 'filepath') else None
                    }
                    for dep in symbol.dependencies
                ] if hasattr(symbol, 'dependencies') else [],
                'inheritance_depth': len(symbol.superclasses) if hasattr(symbol, 'superclasses') else 0,
                'inheritance_chain': [
                    {
                        'name': s.name,
                        'filepath': s.filepath if hasattr(s, 'filepath') else None
                    }
                    for s in symbol.superclasses
                ] if hasattr(symbol, 'superclasses') else []
            }
        })

    base_info['usage_stats'] = usage_stats
    return base_info

@fastapi_app.get("/symbol/{symbol_id}/context")
async def get_symbol_context(symbol_id: str) -> Dict[str, Any]:
    """Get detailed context for any symbol."""
    try:
        symbol = get_symbol_by_id(symbol_id)  # You'll need to implement this
        return get_detailed_symbol_context(symbol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ... [Keep all existing endpoints and helper functions] ...

@app.function(image=image)
@modal.asgi_app()
def fastapi_modal_app():
    return fastapi_app

if __name__ == "__main__":
    app.deploy("analytics-app")

