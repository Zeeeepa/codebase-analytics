from collections import Counter
from typing import Dict, List, Optional
import networkx as nx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class CodebaseStats(BaseModel):
    test_functions_count: int
    test_classes_count: int
    tests_per_file: float
    total_classes: int
    total_functions: int
    total_imports: int
    deepest_inheritance_class: Optional[Dict]
    recursive_functions: List[str]
    most_called_function: Dict
    function_with_most_calls: Dict
    unused_functions: List[Dict]
    dead_code: List[Dict]

class FileTestStats(BaseModel):
    filepath: str
    test_class_count: int
    file_length: int
    function_count: int

class FunctionContext(BaseModel):
    implementation: Dict
    dependencies: List[Dict]
    usages: List[Dict]

@app.get("/api/codebase/stats")
async def get_codebase_stats(codebase_id: str) -> CodebaseStats:
    """Get comprehensive statistics about the codebase."""
    try:
        # Filter test functions and classes
        test_functions = [x for x in codebase.functions if x.name.startswith('test_')]
        test_classes = [x for x in codebase.classes if x.name.startswith('Test')]
        
        # Calculate tests per file
        tests_per_file = len(test_functions) / len(codebase.files) if codebase.files else 0
        
        # Find class with deepest inheritance
        deepest_class = None
        if codebase.classes:
            deepest = max(codebase.classes, key=lambda x: len(x.superclasses))
            deepest_class = {
                'name': deepest.name,
                'depth': len(deepest.superclasses),
                'chain': [s.name for s in deepest.superclasses]
            }
        
        # Find recursive functions
        recursive = [f.name for f in codebase.functions 
                    if any(call.name == f.name for call in f.function_calls)][:5]
        
        # Find most called function
        most_called = max(codebase.functions, key=lambda f: len(f.call_sites))
        most_called_info = {
            'name': most_called.name,
            'call_count': len(most_called.call_sites),
            'callers': [{'function': call.parent_function.name, 
                        'line': call.start_point[0]} 
                       for call in most_called.call_sites]
        }
        
        # Find function with most calls
        most_calls = max(codebase.functions, key=lambda f: len(f.function_calls))
        most_calls_info = {
            'name': most_calls.name,
            'calls_count': len(most_calls.function_calls),
            'called_functions': [call.name for call in most_calls.function_calls]
        }
        
        # Find unused functions
        unused = [{'name': f.name, 'filepath': f.filepath} 
                 for f in codebase.functions if len(f.call_sites) == 0]
        
        # Find dead code
        dead_code = find_dead_code(codebase)
        dead_code_info = [{'name': f.name, 'filepath': f.filepath} for f in dead_code]
        
        return CodebaseStats(
            test_functions_count=len(test_functions),
            test_classes_count=len(test_classes),
            tests_per_file=tests_per_file,
            total_classes=len(codebase.classes),
            total_functions=len(codebase.functions),
            total_imports=len(codebase.imports),
            deepest_inheritance_class=deepest_class,
            recursive_functions=recursive,
            most_called_function=most_called_info,
            function_with_most_calls=most_calls_info,
            unused_functions=unused,
            dead_code=dead_code_info
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/codebase/test-files")
async def get_test_file_stats(codebase_id: str) -> List[FileTestStats]:
    """Get statistics about test files in the codebase."""
    try:
        test_classes = [x for x in codebase.classes if x.name.startswith('Test')]
        file_test_counts = Counter([x.file for x in test_classes])
        
        stats = []
        for file, num_tests in file_test_counts.most_common()[:5]:
            stats.append(FileTestStats(
                filepath=file.filepath,
                test_class_count=num_tests,
                file_length=len(file.source),
                function_count=len(file.functions)
            ))
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/function/{function_id}/context")
async def get_function_context(function_id: str) -> FunctionContext:
    """Get detailed context for a specific function."""
    try:
        function = get_function_by_id(function_id)  # You'll need to implement this
        
        context = {
            "implementation": {
                "source": function.source,
                "filepath": function.filepath
            },
            "dependencies": [],
            "usages": []
        }
        
        # Add dependencies
        for dep in function.dependencies:
            if isinstance(dep, Import):
                dep = hop_through_imports(dep)  # You'll need to implement this
            context["dependencies"].append({
                "source": dep.source,
                "filepath": dep.filepath
            })
        
        # Add usages
        for usage in function.usages:
            context["usages"].append({
                "source": usage.usage_symbol.source,
                "filepath": usage.usage_symbol.filepath
            })
        
        return FunctionContext(**context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/function/{function_id}/call-chain")
async def get_function_call_chain(function_id: str) -> List[str]:
    """Get the maximum call chain for a function."""
    try:
        function = get_function_by_id(function_id)
        chain = get_max_call_chain(function)
        return [f.name for f in chain]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def find_dead_code(codebase) -> List:
    """Find functions that are never called."""
    dead_functions = []
    for function in codebase.functions:
        if not any(function.function_calls):
            dead_functions.append(function)
    return dead_functions

def get_max_call_chain(function) -> List:
    """Get the longest call chain starting from a function."""
    G = nx.DiGraph()
    
    def build_graph(func, depth=0):
        if depth > 10:  # Prevent infinite recursion
            return
        for call in func.function_calls:
            called_func = call.function_definition
            G.add_edge(func, called_func)
            build_graph(called_func, depth + 1)
    
    build_graph(function)
    return nx.dag_longest_path(G)

# Helper function to get a function by ID (you'll need to implement this)
def get_function_by_id(function_id: str):
    # Implementation depends on how you store/retrieve functions
    pass

# Helper function to resolve imports (you'll need to implement this)
def hop_through_imports(import_symbol):
    # Implementation depends on how you handle imports
    pass

