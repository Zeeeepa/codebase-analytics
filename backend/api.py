from collections import Counter
from typing import Dict, List, Optional
import networkx as nx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class TestAnalysis(BaseModel):
    total_test_functions: int
    total_test_classes: int
    tests_per_file: float
    top_test_files: List[Dict[str, any]]

class FunctionAnalysis(BaseModel):
    total_functions: int
    most_called_function: Dict[str, any]
    function_with_most_calls: Dict[str, any]
    recursive_functions: List[str]
    unused_functions: List[Dict[str, str]]
    dead_code: List[Dict[str, str]]

class ClassAnalysis(BaseModel):
    total_classes: int
    deepest_inheritance: Optional[Dict[str, any]]
    total_imports: int

class FileIssue(BaseModel):
    critical: List[Dict[str, str]]
    major: List[Dict[str, str]]
    minor: List[Dict[str, str]]

class CodebaseAnalysis(BaseModel):
    test_analysis: TestAnalysis
    function_analysis: FunctionAnalysis
    class_analysis: ClassAnalysis
    file_issues: Dict[str, FileIssue]
    repo_structure: Dict[str, any]

@app.get("/api/analyze")
async def analyze_codebase(codebase_id: str) -> CodebaseAnalysis:
    """Comprehensive codebase analysis endpoint that returns all metrics and analysis results."""
    try:
        # Test Analysis
        test_functions = [x for x in codebase.functions if x.name.startswith('test_')]
        test_classes = [x for x in codebase.classes if x.name.startswith('Test')]
        tests_per_file = len(test_functions) / len(codebase.files) if codebase.files else 0
        
        # Get top test files
        file_test_counts = Counter([x.file for x in test_classes])
        top_test_files = [
            {
                'filepath': file.filepath,
                'test_count': num_tests,
                'file_length': len(file.source),
                'function_count': len(file.functions)
            }
            for file, num_tests in file_test_counts.most_common(5)
        ]

        test_analysis = TestAnalysis(
            total_test_functions=len(test_functions),
            total_test_classes=len(test_classes),
            tests_per_file=tests_per_file,
            top_test_files=top_test_files
        )

        # Function Analysis
        recursive = [
            f.name for f in codebase.functions 
            if any(call.name == f.name for call in f.function_calls)
        ][:5]

        most_called = max(codebase.functions, key=lambda f: len(f.call_sites))
        most_called_info = {
            'name': most_called.name,
            'call_count': len(most_called.call_sites),
            'callers': [
                {
                    'function': call.parent_function.name,
                    'line': call.start_point[0]
                }
                for call in most_called.call_sites
            ]
        }

        most_calls = max(codebase.functions, key=lambda f: len(f.function_calls))
        most_calls_info = {
            'name': most_calls.name,
            'calls_count': len(most_calls.function_calls),
            'called_functions': [call.name for call in most_calls.function_calls]
        }

        unused = [
            {'name': f.name, 'filepath': f.filepath}
            for f in codebase.functions if len(f.call_sites) == 0
        ]

        dead_code = [
            {'name': f.name, 'filepath': f.filepath}
            for f in find_dead_code(codebase)
        ]

        function_analysis = FunctionAnalysis(
            total_functions=len(codebase.functions),
            most_called_function=most_called_info,
            function_with_most_calls=most_calls_info,
            recursive_functions=recursive,
            unused_functions=unused,
            dead_code=dead_code
        )

        # Class Analysis
        deepest_class = None
        if codebase.classes:
            deepest = max(codebase.classes, key=lambda x: len(x.superclasses))
            deepest_class = {
                'name': deepest.name,
                'depth': len(deepest.superclasses),
                'chain': [s.name for s in deepest.superclasses]
            }

        class_analysis = ClassAnalysis(
            total_classes=len(codebase.classes),
            deepest_inheritance=deepest_class,
            total_imports=len(codebase.imports)
        )

        # File Issues Analysis
        file_issues = {}
        for file in codebase.files:
            issues = analyze_file_issues(file)
            if any(len(v) > 0 for v in issues.values()):
                file_issues[file.filepath] = FileIssue(**issues)

        # Repository Structure
        repo_structure = build_repo_structure(codebase.files, file_issues)

        return CodebaseAnalysis(
            test_analysis=test_analysis,
            function_analysis=function_analysis,
            class_analysis=class_analysis,
            file_issues=file_issues,
            repo_structure=repo_structure
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def find_dead_code(codebase) -> List:
    """Find functions that are never called."""
    dead_functions = []
    for function in codebase.functions:
        if not any(function.function_calls):
            dead_functions.append(function)
    return dead_functions

def analyze_file_issues(file) -> Dict[str, List[Dict[str, str]]]:
    """Analyze a file for various types of issues."""
    issues = {
        'critical': [],
        'major': [],
        'minor': []
    }
    
    # Check for implementation errors
    for function in file.functions:
        # Check for unused parameters
        for param in function.parameters:
            if not any(param.name in str(usage) for usage in function.usages):
                issues['minor'].append({
                    'type': 'unused_parameter',
                    'message': f'Unused parameter "{param.name}" in function "{function.name}"'
                })

        # Check for null references
        if hasattr(function, 'code_block'):
            code = function.code_block.source
            if 'None' in code and not any(s in code for s in ['is None', '== None', '!= None']):
                issues['critical'].append({
                    'type': 'unsafe_null_check',
                    'message': f'Potential unsafe null reference in function "{function.name}"'
                })

        # Check for incomplete implementations
        if 'TODO' in function.source or 'FIXME' in function.source:
            issues['major'].append({
                'type': 'incomplete_implementation',
                'message': f'Incomplete implementation in function "{function.name}"'
            })

    # Check for code duplication
    seen_blocks = {}
    for function in file.functions:
        if hasattr(function, 'code_block'):
            code = function.code_block.source.strip()
            if len(code) > 50:  # Only check substantial blocks
                if code in seen_blocks:
                    issues['major'].append({
                        'type': 'code_duplication',
                        'message': f'Code duplication between functions "{function.name}" and "{seen_blocks[code]}"'
                    })
                else:
                    seen_blocks[code] = function.name

    return issues

def build_repo_structure(files, file_issues) -> Dict:
    """Build a hierarchical repository structure with issue counts."""
    root = {'name': 'root', 'children': {}}
    
    for file in files:
        path_parts = file.filepath.split('/')
        current = root
        
        # Build the tree structure
        for i, part in enumerate(path_parts[:-1]):
            if part not in current['children']:
                current['children'][part] = {
                    'name': part,
                    'type': 'directory',
                    'children': {},
                    'issues': {'critical': 0, 'major': 0, 'minor': 0}
                }
            current = current['children'][part]
        
        # Add the file
        filename = path_parts[-1]
        file_node = {
            'name': filename,
            'type': 'file',
            'issues': {'critical': 0, 'major': 0, 'minor': 0}
        }
        
        # Add issue counts if present
        if file.filepath in file_issues:
            issues = file_issues[file.filepath]
            file_node['issues'] = {
                'critical': len(issues.critical),
                'major': len(issues.major),
                'minor': len(issues.minor)
            }
            
            # Propagate counts up the tree
            temp = root
            for part in path_parts[:-1]:
                temp = temp['children'][part]
                for severity in ['critical', 'major', 'minor']:
                    temp['issues'][severity] += file_node['issues'][severity]
        
        current['children'][filename] = file_node
    
    return root

