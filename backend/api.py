from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Tuple, Any, Optional
import os
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import modal

# Import our comprehensive analysis engine
from analysis import (
    ComprehensiveCodebaseAnalyzer,
    AnalysisType,
    AnalysisResult,
    analyze_repository,
    analyze_repository_to_text
)

# Import Codegen SDK
from codegen.sdk.core.codebase import Codebase

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

# API Models for the new comprehensive analysis engine
class RepoRequest(BaseModel):
    repo_url: str
    analysis_types: Optional[List[str]] = None

class AnalysisRequest(BaseModel):
    repo_path: str
    analysis_types: Optional[List[str]] = None
    output_format: str = "json"  # "json" or "text"

class IssueResponse(BaseModel):
    id: str
    severity: str
    category: str
    message: str
    file_path: str
    line_number: Optional[int] = None
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class FileAnalysisResponse(BaseModel):
    path: str
    issues: List[IssueResponse]
    symbols: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    dependencies: List[str]

class RepositoryStructureResponse(BaseModel):
    name: str
    type: str
    path: str
    children: List['RepositoryStructureResponse'] = []
    issue_counts: Dict[str, int] = {}
    symbols: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {}

class ComprehensiveAnalysisResponse(BaseModel):
    repository_structure: RepositoryStructureResponse
    total_issues: Dict[str, int]
    file_analyses: List[FileAnalysisResponse]
    call_graph: Optional[Dict[str, Any]] = None
    dependency_graph: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any] = {}
    timestamp: str

class InteractiveTreeNode(BaseModel):
    """Interactive tree node for the UI."""
    name: str
    type: str  # 'file' or 'directory'
    path: str
    icon: str  # emoji icon
    issue_counts: Dict[str, int] = {}
    children: List['InteractiveTreeNode'] = []
    symbols: List[Dict[str, Any]] = []
    is_expanded: bool = False

class SymbolDetail(BaseModel):
    """Detailed symbol information for UI."""
    name: str
    type: str
    file_path: str
    line_number: Optional[int] = None
    parameters: List[str] = []
    return_type: Optional[str] = None
    call_chain: List[str] = []
    dependencies: List[str] = []
    issues: List[IssueResponse] = []
    context: Dict[str, Any] = {}

# Fix forward references
RepositoryStructureResponse.model_rebuild()
InteractiveTreeNode.model_rebuild()

#######################################################
# Helper Functions
#######################################################

def clone_repository(repo_url: str) -> str:
    """Clone a repository and return the local path."""
    temp_dir = tempfile.mkdtemp()
    try:
        subprocess.run(["git", "clone", repo_url, temp_dir], check=True, capture_output=True)
        return temp_dir
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, f"Failed to clone repository: {e}")

def convert_analysis_result_to_response(result: AnalysisResult) -> ComprehensiveAnalysisResponse:
    """Convert internal AnalysisResult to API response format."""
    
    def convert_repository_structure(structure) -> RepositoryStructureResponse:
        return RepositoryStructureResponse(
            name=structure.name,
            type=structure.type,
            path=structure.path,
            children=[convert_repository_structure(child) for child in structure.children],
            issue_counts={severity.value: count for severity, count in structure.issue_counts.items()},
            symbols=structure.symbols,
            metrics=structure.metrics
        )
    
    def convert_issue(issue) -> IssueResponse:
        return IssueResponse(
            id=issue.id,
            severity=issue.severity.value,
            category=issue.category.value,
            message=issue.message,
            file_path=issue.file_path,
            line_number=issue.line_number,
            function_name=issue.function_name,
            class_name=issue.class_name,
            context=issue.context
        )
    
    def convert_file_analysis(file_analysis) -> FileAnalysisResponse:
        return FileAnalysisResponse(
            path=file_analysis.path,
            issues=[convert_issue(issue) for issue in file_analysis.issues],
            symbols=file_analysis.symbols,
            metrics=file_analysis.metrics,
            dependencies=file_analysis.dependencies
        )
    
    return ComprehensiveAnalysisResponse(
        repository_structure=convert_repository_structure(result.repository_structure),
        total_issues={severity.value: count for severity, count in result.total_issues.items()},
        file_analyses=[convert_file_analysis(fa) for fa in result.file_analyses],
        call_graph=result.call_graph,
        dependency_graph=result.dependency_graph,
        metrics=result.metrics,
        timestamp=result.timestamp
    )

def build_interactive_tree(structure: RepositoryStructureResponse) -> InteractiveTreeNode:
    """Convert repository structure to interactive tree format."""
    
    # Determine icon based on type and issues
    if structure.type == "directory":
        icon = "📁"
    else:
        # File icons based on extension
        if structure.path.endswith('.py'):
            icon = "🐍"
        elif structure.path.endswith(('.js', '.ts', '.jsx', '.tsx')):
            icon = "📜"
        elif structure.path.endswith(('.md', '.txt')):
            icon = "📝"
        elif structure.path.endswith(('.json', '.yaml', '.yml')):
            icon = "⚙️"
        else:
            icon = "📄"
    
    return InteractiveTreeNode(
        name=structure.name,
        type=structure.type,
        path=structure.path,
        icon=icon,
        issue_counts=structure.issue_counts,
        children=[build_interactive_tree(child) for child in structure.children],
        symbols=structure.symbols,
        is_expanded=False
    )


#######################################################
# API Endpoints
#######################################################

def get_monthly_commits(repo_path: str) -> Dict[str, int]:
    """
    Get the number of commits per month for the last 12 months.

    Args:
        repo_path: Path to the git repository

    Returns:
        Dictionary with month-year as key and number of commits as value
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    date_format = "%Y-%m-%d"
    since_date = start_date.strftime(date_format)
    until_date = end_date.strftime(date_format)
    repo_path = "https://github.com/" + repo_path

    try:
        original_dir = os.getcwd()

        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(["git", "clone", repo_path, temp_dir], check=True)
            os.chdir(temp_dir)

            cmd = [
                "git",
                "log",
                f"--since={since_date}",
                f"--until={until_date}",
                "--format=%aI",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            commit_dates = result.stdout.strip().split("\n")

            monthly_counts = {}
            current_date = start_date
            while current_date <= end_date:
                month_key = current_date.strftime("%Y-%m")
                monthly_counts[month_key] = 0
                current_date = (
                    current_date.replace(day=1) + timedelta(days=32)
                ).replace(day=1)

            for date_str in commit_dates:
                if date_str:  # Skip empty lines
                    commit_date = datetime.fromisoformat(date_str.strip())
                    month_key = commit_date.strftime("%Y-%m")
                    if month_key in monthly_counts:
                        monthly_counts[month_key] += 1

            os.chdir(original_dir)
            return dict(sorted(monthly_counts.items()))

    except subprocess.CalledProcessError as e:
        print(f"Error executing git command: {e}")
        return {}
    except Exception as e:
        print(f"Error processing git commits: {e}")
        return {}
    finally:
        try:
            os.chdir(original_dir)
        except:
            pass

def calculate_cyclomatic_complexity(function):
    def analyze_statement(statement):
        complexity = 0

        if isinstance(statement, IfBlockStatement):
            complexity += 1
            if hasattr(statement, "elif_statements"):
                complexity += len(statement.elif_statements)

        elif isinstance(statement, (ForLoopStatement, WhileStatement)):
            complexity += 1

        elif isinstance(statement, TryCatchStatement):
            complexity += len(getattr(statement, "except_blocks", []))

        if hasattr(statement, "condition") and isinstance(statement.condition, str):
            complexity += statement.condition.count(
                " and "
            ) + statement.condition.count(" or ")

        if hasattr(statement, "nested_code_blocks"):
            for block in statement.nested_code_blocks:
                complexity += analyze_block(block)

        return complexity

    def analyze_block(block):
        if not block or not hasattr(block, "statements"):
            return 0
        return sum(analyze_statement(stmt) for stmt in block.statements)

    return (
        1 + analyze_block(function.code_block) if hasattr(function, "code_block") else 1
    )

def cc_rank(complexity):
    if complexity < 0:
        raise ValueError("Complexity must be a non-negative value")

    ranks = [
        (1, 5, "A"),
        (6, 10, "B"),
        (11, 20, "C"),
        (21, 30, "D"),
        (31, 40, "E"),
        (41, float("inf"), "F"),
    ]
    for low, high, rank in ranks:
        if low <= complexity <= high:
            return rank
    return "F"

def calculate_doi(cls):
    """Calculate the depth of inheritance for a given class."""
    return len(cls.superclasses)

def get_operators_and_operands(function):
    operators = []
    operands = []

    for statement in function.code_block.statements:
        for call in statement.function_calls:
            operators.append(call.name)
            for arg in call.args:
                operands.append(arg.source)

        if hasattr(statement, "expressions"):
            for expr in statement.expressions:
                if isinstance(expr, BinaryExpression):
                    operators.extend([op.source for op in expr.operators])
                    operands.extend([elem.source for elem in expr.elements])
                elif isinstance(expr, UnaryExpression):
                    operators.append(expr.ts_node.type)
                    operands.append(expr.argument.source)
                elif isinstance(expr, ComparisonExpression):
                    operators.extend([op.source for op in expr.operators])
                    operands.extend([elem.source for elem in expr.elements])

        if hasattr(statement, "expression"):
            expr = statement.expression
            if isinstance(expr, BinaryExpression):
                operators.extend([op.source for op in expr.operators])
                operands.extend([elem.source for elem in expr.elements])
            elif isinstance(expr, UnaryExpression):
                operators.append(expr.ts_node.type)
                operands.append(expr.argument.source)
            elif isinstance(expr, ComparisonExpression):
                operators.extend([op.source for op in expr.operators])
                operands.extend([elem.source for elem in expr.elements])

    return operators, operands

def calculate_halstead_volume(operators, operands):
    n1 = len(set(operators))
    n2 = len(set(operands))

    N1 = len(operators)
    N2 = len(operands)

    N = N1 + N2
    n = n1 + n2

    if n > 0:
        volume = N * math.log2(n)
        return volume, N1, N2, n1, n2
    return 0, N1, N2, n1, n2

def count_lines(source: str):
    """Count different types of lines in source code."""
    if not source.strip():
        return 0, 0, 0, 0

    lines = [line.strip() for line in source.splitlines()]
    loc = len(lines)
    sloc = len([line for line in lines if line])

    in_multiline = False
    comments = 0
    code_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]
        code_part = line
        if not in_multiline and "#" in line:
            comment_start = line.find("#")
            if not re.search(r'["\'].*#.*["\']', line[:comment_start]):
                code_part = line[:comment_start].strip()
                if line[comment_start:].strip():
                    comments += 1

        if ('"""' in line or "'''" in line) and not (
            line.count('"""') % 2 == 0 or line.count("'''") % 2 == 0
        ):
            if in_multiline:
                in_multiline = False
                comments += 1
            else:
                in_multiline = True
                comments += 1
                if line.strip().startswith('"""') or line.strip().startswith("'''"):
                    code_part = ""
        elif in_multiline:
            comments += 1
            code_part = ""
        elif line.strip().startswith("#"):
            comments += 1
            code_part = ""

        if code_part.strip():
            code_lines.append(code_part)

        i += 1

    lloc = 0
    continued_line = False
    for line in code_lines:
        if continued_line:
            if not any(line.rstrip().endswith(c) for c in ("\\", ",", "{", "[", "(")):
                continued_line = False
            continue

        lloc += len([stmt for stmt in line.split(";") if stmt.strip()])

        if any(line.rstrip().endswith(c) for c in ("\\", ",", "{", "[", "(")):
            continued_line = True

    return loc, lloc, sloc, comments

def calculate_maintainability_index(
    halstead_volume: float, cyclomatic_complexity: float, loc: int
) -> int:
    """Calculate the normalized maintainability index for a given function."""
    if loc <= 0:
        return 100

    try:
        raw_mi = (
            171
            - 5.2 * math.log(max(1, halstead_volume))
            - 0.23 * cyclomatic_complexity
            - 16.2 * math.log(max(1, loc))
        )
        normalized_mi = max(0, min(100, raw_mi * 100 / 171))
        return int(normalized_mi)
    except (ValueError, TypeError):
        return 0

def get_maintainability_rank(mi_score: float) -> str:
    """Convert maintainability index score to a letter grade."""
    if mi_score >= 85:
        return "A"
    elif mi_score >= 65:
        return "B"
    elif mi_score >= 45:
        return "C"
    elif mi_score >= 25:
        return "D"
    else:
        return "F"

def get_github_repo_description(repo_url):
    api_url = f"https://api.github.com/repos/{repo_url}"

    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            repo_data = response.json()
            description = repo_data.get("description")
            return description if description else "No description available"
        else:
            return "No description available"
    except Exception as e:
        print(f"Error fetching repo description: {e}")
        return "No description available"

def analyze_inheritance_patterns(codebase) -> InheritanceAnalysis:
    """Analyze class inheritance patterns to find the deepest inheritance chain."""
    if not codebase.classes:
        return InheritanceAnalysis()
    
    try:
        # Find class with most inheritance
        deepest_class = max(codebase.classes, key=lambda x: len(x.superclasses))
        
        return InheritanceAnalysis(
            deepest_class_name=deepest_class.name,
            deepest_class_depth=len(deepest_class.superclasses),
            inheritance_chain=[s.name for s in deepest_class.superclasses]
        )
    except Exception as e:
        print(f"Error analyzing inheritance patterns: {e}")
        return InheritanceAnalysis()

def analyze_recursive_functions(codebase) -> RecursionAnalysis:
    """Analyze functions to identify recursive patterns."""
    if not codebase.functions:
        return RecursionAnalysis()
    
    try:
        # Find recursive functions (functions that call themselves)
        recursive_functions = []
        for func in codebase.functions:
            # Check if function calls itself directly
            if any(call.name == func.name for call in func.function_calls):
                recursive_functions.append(func.name)
                if len(recursive_functions) >= 5:  # Limit to first 5
                    break
        
        return RecursionAnalysis(
            recursive_functions=recursive_functions,
            total_recursive_count=len(recursive_functions)
        )
    except Exception as e:
        print(f"Error analyzing recursive functions: {e}")
        return RecursionAnalysis()

def analyze_functions_comprehensive(codebase) -> FunctionAnalysis:
    """Perform comprehensive function analysis."""
    if not codebase.functions:
        return FunctionAnalysis()
    
    try:
        # Find most called function
        most_called = None
        max_call_count = 0
        for func in codebase.functions:
            call_count = len(func.call_sites) if hasattr(func, 'call_sites') else 0
            if call_count > max_call_count:
                max_call_count = call_count
                most_called = func
        
        most_called_detail = None
        if most_called:
            most_called_detail = FunctionDetail(
                name=most_called.name,
                parameters=[p.name for p in most_called.parameters] if hasattr(most_called, 'parameters') else [],
                return_type=getattr(most_called, 'return_type', None),
                call_count=max_call_count,
                calls_made=len(most_called.function_calls)
            )
        
        # Find function that makes the most calls
        most_calling = None
        max_calls_made = 0
        for func in codebase.functions:
            calls_made = len(func.function_calls)
            if calls_made > max_calls_made:
                max_calls_made = calls_made
                most_calling = func
        
        most_calling_detail = None
        if most_calling:
            most_calling_detail = FunctionDetail(
                name=most_calling.name,
                parameters=[p.name for p in most_calling.parameters] if hasattr(most_calling, 'parameters') else [],
                return_type=getattr(most_calling, 'return_type', None),
                call_count=len(most_calling.call_sites) if hasattr(most_calling, 'call_sites') else 0,
                calls_made=max_calls_made
            )
        
        # Find dead functions (functions with no callers)
        dead_functions = []
        for func in codebase.functions:
            call_count = len(func.call_sites) if hasattr(func, 'call_sites') else 0
            if call_count == 0:
                dead_functions.append(func.name)
                if len(dead_functions) >= 10:  # Limit to first 10
                    break
        
        # Sample functions (first 5)
        sample_functions = []
        for func in codebase.functions[:5]:
            sample_functions.append(FunctionDetail(
                name=func.name,
                parameters=[p.name for p in func.parameters] if hasattr(func, 'parameters') else [],
                return_type=getattr(func, 'return_type', None),
                call_count=len(func.call_sites) if hasattr(func, 'call_sites') else 0,
                calls_made=len(func.function_calls)
            ))
        
        # Sample classes (first 5)
        sample_classes = []
        if hasattr(codebase, 'classes') and codebase.classes:
            for cls in codebase.classes[:5]:
                sample_classes.append(ClassDetail(
                    name=cls.name,
                    methods=[m.name for m in cls.methods] if hasattr(cls, 'methods') else [],
                    attributes=[a.name for a in cls.attributes] if hasattr(cls, 'attributes') else []
                ))
        
        # Sample imports (first 5)
        sample_imports = []
        for file in codebase.files[:5]:  # Check first 5 files for imports
            if hasattr(file, 'imports') and file.imports:
                for imp in file.imports[:2]:  # Max 2 imports per file
                    sample_imports.append(ImportDetail(
                        module=getattr(imp, 'module', 'unknown'),
                        imported_symbols=[s.name for s in imp.imported_symbol] if hasattr(imp, 'imported_symbol') else []
                    ))
                    if len(sample_imports) >= 5:
                        break
            if len(sample_imports) >= 5:
                break
        
        return FunctionAnalysis(
            total_functions=len(codebase.functions),
            most_called_function=most_called_detail,
            most_calling_function=most_calling_detail,
            dead_functions=dead_functions,
            dead_functions_count=len(dead_functions),
            sample_functions=sample_functions,
            sample_classes=sample_classes,
            sample_imports=sample_imports
        )
    
    except Exception as e:
        print(f"Error analyzing functions comprehensively: {e}")
        return FunctionAnalysis(total_functions=len(codebase.functions) if codebase.functions else 0)

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

def build_repo_structure(files, file_issues, file_symbols) -> Dict:
    """Build a hierarchical repository structure with issue counts and symbols."""
    root = {
        'name': 'root',
        'type': 'directory',
        'path': '',
        'children': {},
        'issues': {'critical': 0, 'major': 0, 'minor': 0},
        'stats': {
            'files': 0,
            'directories': 0,
            'symbols': 0,
            'issues': 0
        }
    }
    
    # First pass: Create all directories
    all_dirs = set()
    for file in files:
        dir_path = os.path.dirname(file.filepath)
        if dir_path:
            parts = dir_path.split('/')
            current_path = ''
            for part in parts:
                current_path = os.path.join(current_path, part) if current_path else part
                all_dirs.add(current_path)
    
    # Create directory nodes
    for dir_path in sorted(all_dirs):
        parts = dir_path.split('/')
        current = root
        current_path = ''
        
        for part in parts:
            current_path = os.path.join(current_path, part) if current_path else part
            if part not in current['children']:
                current['children'][part] = {
                    'name': part,
                    'type': 'directory',
                    'path': current_path,
                    'children': {},
                    'issues': {'critical': 0, 'major': 0, 'minor': 0},
                    'stats': {
                        'files': 0,
                        'directories': 0,
                        'symbols': 0,
                        'issues': 0
                    }
                }
                current['stats']['directories'] += 1
            current = current['children'][part]
    
    # Add files
    for file in sorted(files, key=lambda f: f.filepath):
        dir_path = os.path.dirname(file.filepath)
        filename = os.path.basename(file.filepath)
        
        # Navigate to the correct directory
        current = root
        if dir_path:
            for part in dir_path.split('/'):
                current = current['children'][part]
        
        # Create file node
        file_node = {
            'name': filename,
            'type': 'file',
            'file_type': get_file_type(filename),
            'path': file.filepath,
            'issues': {'critical': 0, 'major': 0, 'minor': 0},
            'stats': {
                'symbols': 0,
                'issues': 0
            }
        }
        
        # Add issue counts
        if file.filepath in file_issues:
            issues = file_issues[file.filepath]
            file_node['issues'] = {
                'critical': len(issues['critical']),
                'major': len(issues['major']),
                'minor': len(issues['minor'])
            }
            file_node['stats']['issues'] = sum(file_node['issues'].values())
            
            # Propagate issue counts up the tree
            temp_path = dir_path
            temp = current
            while temp is not None:
                for severity in ['critical', 'major', 'minor']:
                    temp['issues'][severity] += file_node['issues'][severity]
                temp['stats']['issues'] += file_node['stats']['issues']
                if temp_path:
                    parent_path = os.path.dirname(temp_path)
                    temp = root
                    if parent_path:
                        for part in parent_path.split('/'):
                            temp = temp['children'][part]
                    temp_path = parent_path
                else:
                    temp = None
        
        # Add symbols
        if file.filepath in file_symbols:
            file_node['symbols'] = file_symbols[file.filepath]
            file_node['stats']['symbols'] = len(file_symbols[file.filepath])
            
            # Propagate symbol counts up the tree
            temp_path = dir_path
            temp = current
            while temp is not None:
                temp['stats']['symbols'] += file_node['stats']['symbols']
                if temp_path:
                    parent_path = os.path.dirname(temp_path)
                    temp = root
                    if parent_path:
                        for part in parent_path.split('/'):
                            temp = temp['children'][part]
                    temp_path = parent_path
                else:
                    temp = None
        
        current['children'][filename] = file_node
        current['stats']['files'] += 1
        root['stats']['files'] += 1
    
    return root

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

@fastapi_app.get("/api/codebase/stats")
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

@fastapi_app.get("/api/codebase/test-files")
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

@fastapi_app.get("/api/function/{function_id}/context")
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

@fastapi_app.get("/api/function/{function_id}/call-chain")
async def get_function_call_chain(function_id: str) -> List[str]:
    """Get the maximum call chain for a function."""
    try:
        function = get_function_by_id(function_id)
        chain = get_max_call_chain(function)
        return [f.name for f in chain]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@fastapi_app.post("/analyze_repo")
async def analyze_repo(request: RepoRequest) -> ComprehensiveAnalysisResponse:
    """Comprehensive repository analysis using the new analysis engine."""
    repo_url = request.repo_url
    
    # Validate repo URL format
    if not repo_url or '/' not in repo_url:
        raise HTTPException(status_code=400, detail="Repository URL must be in format 'owner/repo'")
    
    # Remove any GitHub URL prefix if present
    if repo_url.startswith('https://github.com/'):
        repo_url = repo_url.replace('https://github.com/', '')
    if repo_url.endswith('.git'):
        repo_url = repo_url[:-4]
    
    # Ensure it's in owner/repo format
    parts = repo_url.split('/')
    if len(parts) < 2:
        raise HTTPException(status_code=400, detail="Repository URL must be in format 'owner/repo'")
    
    repo_url = f"{parts[0]}/{parts[1]}"  # Take only owner/repo part
    
    try:
        # Clone repository to temporary directory
        repo_path = clone_repository(f"https://github.com/{repo_url}.git")
        
        # Convert analysis types from request
        analysis_types = None
        if request.analysis_types:
            analysis_types = [AnalysisType(at) for at in request.analysis_types]
        
        # Perform comprehensive analysis
        result = analyze_repository(repo_path, analysis_types)
        
        # Convert to API response format
        response = convert_analysis_result_to_response(result)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to analyze repository: {str(e)}")
    finally:
        # Clean up temporary directory
        if 'repo_path' in locals():
            import shutil
            try:
                shutil.rmtree(repo_path)
            except:
                pass

# New API endpoints for interactive features

@fastapi_app.get("/api/repository/{repo_owner}/{repo_name}/tree")
async def get_interactive_tree(repo_owner: str, repo_name: str) -> InteractiveTreeNode:
    """Get interactive repository tree structure with issue counts."""
    repo_url = f"{repo_owner}/{repo_name}"
    
    try:
        # Clone and analyze repository
        repo_path = clone_repository(f"https://github.com/{repo_url}.git")
        result = analyze_repository(repo_path, [AnalysisType.INTERACTIVE_TREE, AnalysisType.ISSUE_DETECTION])
        
        # Convert to API response and build interactive tree
        response = convert_analysis_result_to_response(result)
        tree = build_interactive_tree(response.repository_structure)
        
        return tree
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to get repository tree: {str(e)}")
    finally:
        if 'repo_path' in locals():
            import shutil
            try:
                shutil.rmtree(repo_path)
            except:
                pass

@fastapi_app.get("/api/repository/{repo_owner}/{repo_name}/file/{file_path:path}")
async def get_file_analysis(repo_owner: str, repo_name: str, file_path: str) -> FileAnalysisResponse:
    """Get detailed analysis for a specific file."""
    repo_url = f"{repo_owner}/{repo_name}"
    
    try:
        # Clone and analyze repository
        repo_path = clone_repository(f"https://github.com/{repo_url}.git")
        result = analyze_repository(repo_path)
        
        # Find the specific file analysis
        for file_analysis in result.file_analyses:
            if file_analysis.path == file_path:
                response = convert_analysis_result_to_response(result)
                # Find corresponding file analysis in response
                for fa in response.file_analyses:
                    if fa.path == file_path:
                        return fa
        
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to analyze file: {str(e)}")
    finally:
        if 'repo_path' in locals():
            import shutil
            try:
                shutil.rmtree(repo_path)
            except:
                pass

@fastapi_app.get("/api/repository/{repo_owner}/{repo_name}/symbol/{symbol_name}")
async def get_symbol_detail(repo_owner: str, repo_name: str, symbol_name: str) -> SymbolDetail:
    """Get detailed information about a specific symbol (function/class)."""
    repo_url = f"{repo_owner}/{repo_name}"
    
    try:
        # Clone and analyze repository
        repo_path = clone_repository(f"https://github.com/{repo_url}.git")
        result = analyze_repository(repo_path, [AnalysisType.CALL_GRAPH, AnalysisType.DEPENDENCY_ANALYSIS])
        
        # Search for the symbol across all files
        for file_analysis in result.file_analyses:
            for symbol in file_analysis.symbols:
                if symbol.get('name') == symbol_name:
                    # Build symbol detail
                    symbol_issues = [
                        IssueResponse(
                            id=issue.id,
                            severity=issue.severity.value,
                            category=issue.category.value,
                            message=issue.message,
                            file_path=issue.file_path,
                            line_number=issue.line_number,
                            function_name=issue.function_name,
                            class_name=issue.class_name,
                            context=issue.context
                        )
                        for issue in file_analysis.issues 
                        if issue.function_name == symbol_name or issue.class_name == symbol_name
                    ]
                    
                    return SymbolDetail(
                        name=symbol_name,
                        type=symbol.get('type', 'unknown'),
                        file_path=file_analysis.path,
                        line_number=symbol.get('line_number'),
                        parameters=symbol.get('parameters', []),
                        return_type=symbol.get('return_type'),
                        call_chain=[],  # Would need call graph analysis
                        dependencies=[],  # Would need dependency analysis
                        issues=symbol_issues,
                        context=symbol
                    )
        
        raise HTTPException(status_code=404, detail=f"Symbol not found: {symbol_name}")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to analyze symbol: {str(e)}")
    finally:
        if 'repo_path' in locals():
            import shutil
            try:
                shutil.rmtree(repo_path)
            except:
                pass

@fastapi_app.post("/api/analyze/text")
async def analyze_repository_text(request: AnalysisRequest) -> str:
    """Analyze repository and return results as formatted text."""
    try:
        # Clone repository if URL provided, otherwise use local path
        if request.repo_path.startswith('http'):
            repo_path = clone_repository(request.repo_path)
        else:
            repo_path = request.repo_path
        
        # Convert analysis types
        analysis_types = None
        if request.analysis_types:
            analysis_types = [AnalysisType(at) for at in request.analysis_types]
        
        # Get text output
        if request.output_format == "text":
            result_text = analyze_repository_to_text(repo_path)
            return result_text
        else:
            result = analyze_repository(repo_path, analysis_types)
            import json
            from dataclasses import asdict
            return json.dumps(asdict(result), indent=2, default=str)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to analyze repository: {str(e)}")
    finally:
        if 'repo_path' in locals() and request.repo_path.startswith('http'):
            import shutil
            try:
                shutil.rmtree(repo_path)
            except:
                pass


# Modal app configuration for deployment
@app.function(image=image)
@modal.web_endpoint(method="POST")
def analyze_repo_modal(request: RepoRequest):
    """Modal endpoint for repository analysis."""
    return analyze_repo(request)

@app.function(image=image)
@modal.web_endpoint(method="GET")
def get_tree_modal(repo_owner: str, repo_name: str):
    """Modal endpoint for getting repository tree."""
    return get_interactive_tree(repo_owner, repo_name)

# Local development server
if __name__ == "__main__":
    import uvicorn
    import socket
    
    def find_available_port(start_port=8000, max_port=8100):
        """Find an available port starting from start_port."""
        for port in range(start_port, max_port + 1):
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
    print(f"🔍 New comprehensive analysis endpoints:")
    print(f"  - POST /analyze_repo - Comprehensive repository analysis")
    print(f"  - GET /api/repository/{{owner}}/{{repo}}/tree - Interactive tree structure")
    print(f"  - GET /api/repository/{{owner}}/{{repo}}/file/{{path}} - File analysis")
    print(f"  - GET /api/repository/{{owner}}/{{repo}}/symbol/{{name}} - Symbol details")
    print(f"  - POST /api/analyze/text - Text format analysis")
    
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port)
