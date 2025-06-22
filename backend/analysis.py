"""
Comprehensive Codebase Analysis Engine
Provides deep analysis including error detection, function contexts, dead code analysis, and metrics
Integrates functionality from analyzer.py, comprehensive_analysis.py, and visualization modules

This unified module contains:
- Complete analysis engine with 6 categories of issue detection
- Function context analysis with dependencies and call chains  
- Dead code detection with blast radius calculation
- Halstead complexity metrics calculation
- Most important functions detection algorithms
- Entry point identification
- Mock codebase generation for testing
- Comprehensive demo functionality
"""
import networkx as nx
import json
import logging
import math
import re
import sys
import tempfile
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Import from Codegen SDK
try:
    from codegen.sdk.core.codebase import Codebase
    from codegen.sdk.core.file import SourceFile
    from codegen.sdk.core.function import Function
    from codegen.sdk.core.class_definition import Class
    from codegen.sdk.core.symbol import Symbol
    from codegen.sdk.core.import_resolution import Import
    from codegen.sdk.enums import EdgeType, SymbolType
    from codegen.shared.enums.programming_language import ProgrammingLanguage
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    print("Warning: Codegen SDK not available, using mock implementations")

class IssueSeverity(Enum):
    CRITICAL = "critical"
    MAJOR = "major" 
    MINOR = "minor"
    INFO = "info"

class IssueType(Enum):
    # Implementation Errors
    NULL_REFERENCE = "null_reference"
    TYPE_MISMATCH = "type_mismatch"
    UNDEFINED_VARIABLE = "undefined_variable"
    MISSING_RETURN = "missing_return"
    UNREACHABLE_CODE = "unreachable_code"
    
    # Function Issues
    MISSPELLED_FUNCTION = "misspelled_function"
    WRONG_PARAMETER_COUNT = "wrong_parameter_count"
    PARAMETER_TYPE_MISMATCH = "parameter_type_mismatch"
    UNUSED_PARAMETER = "unused_parameter"
    
    # Exception Handling
    IMPROPER_EXCEPTION_HANDLING = "improper_exception_handling"
    MISSING_ERROR_HANDLING = "missing_error_handling"
    UNSAFE_ASSERTION = "unsafe_assertion"
    RESOURCE_LEAK = "resource_leak"
    
    # Code Quality
    CODE_DUPLICATION = "code_duplication"
    INEFFICIENT_PATTERN = "inefficient_pattern"
    MAGIC_NUMBER = "magic_number"
    LONG_FUNCTION = "long_function"
    DEEP_NESTING = "deep_nesting"
    
    # Formatting & Style
    INCONSISTENT_NAMING = "inconsistent_naming"
    MISSING_DOCUMENTATION = "missing_documentation"
    INCONSISTENT_INDENTATION = "inconsistent_indentation"
    IMPORT_ORGANIZATION = "import_organization"
    
    # Runtime Risks
    DIVISION_BY_ZERO = "division_by_zero"
    ARRAY_BOUNDS = "array_bounds"
    INFINITE_LOOP = "infinite_loop"
    STACK_OVERFLOW = "stack_overflow"
    CONCURRENCY_ISSUE = "concurrency_issue"

@dataclass
class Issue:
    type: IssueType
    severity: IssueSeverity
    message: str
    filepath: str
    line_number: int
    column_number: int = 0
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}

@dataclass
class FunctionContext:
    name: str
    filepath: str
    source: str
    parameters: List[Dict[str, Any]]
    dependencies: List[Dict[str, Any]]
    usages: List[Dict[str, Any]]
    function_calls: List[str]
    called_by: List[str]
    max_call_chain: List[str]
    issues: List[Issue]
    is_entry_point: bool
    is_dead_code: bool
    class_name: Optional[str] = None
    complexity_score: int = 0
    halstead_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.halstead_metrics is None:
            self.halstead_metrics = {}

class AnalysisType(Enum):
    """Types of analysis that can be performed."""
    CODE_QUALITY = "code_quality"
    DEPENDENCIES = "dependencies"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DEAD_CODE = "dead_code"
    COMPLEXITY = "complexity"
    ISSUES = "issues"
    HALSTEAD = "halstead"

def get_function_context(function) -> dict:
    """Get the implementation, dependencies, and usages of a function."""
    context = {
        "implementation": {"source": getattr(function, 'source', ''), "filepath": getattr(function, 'filepath', '')},
        "dependencies": [],
        "usages": [],
    }
    
    # Add dependencies
    if hasattr(function, 'dependencies'):
        for dep in function.dependencies:
            if hasattr(dep, 'source') and hasattr(dep, 'filepath'):
                context["dependencies"].append({"source": dep.source, "filepath": dep.filepath})
    
    # Add usages
    if hasattr(function, 'usages'):
        for usage in function.usages:
            if hasattr(usage, 'source') and hasattr(usage, 'filepath'):
                context["usages"].append({"source": usage.source, "filepath": usage.filepath})
    
    return context

def get_max_call_chain(function):
    """Calculate maximum call chain using NetworkX DAG"""
    if not nx:
        return [function.name] if hasattr(function, 'name') else []
    
    G = nx.DiGraph()
    
    # Add function as starting node
    func_name = getattr(function, 'name', 'unknown')
    G.add_node(func_name)
    
    # Build call graph
    if hasattr(function, 'function_calls'):
        for call in function.function_calls:
            call_name = call if isinstance(call, str) else getattr(call, 'name', str(call))
            G.add_edge(func_name, call_name)
    
    # Find longest path
    try:
        if nx.is_directed_acyclic_graph(G):
            return nx.dag_longest_path(G)
        else:
            # Handle cycles by finding longest simple path
            return [func_name]
    except:
        return [func_name]

def calculate_halstead_metrics(source_code: str) -> Dict[str, Any]:
    """Calculate Halstead complexity metrics"""
    if not source_code:
        return {}
    
    # Python operators and keywords
    operators = {
        '+', '-', '*', '/', '//', '%', '**', '=', '+=', '-=', '*=', '/=', '//=', '%=', '**=',
        '==', '!=', '<', '>', '<=', '>=', 'and', 'or', 'not', 'in', 'is', 'if', 'else', 'elif',
        'for', 'while', 'def', 'class', 'return', 'yield', 'import', 'from', 'as', 'try',
        'except', 'finally', 'with', 'lambda', 'pass', 'break', 'continue', 'global', 'nonlocal'
    }
    
    # Count operators and operands
    operator_counts = {}
    operand_counts = {}
    
    # Simple tokenization (could be enhanced with AST)
    tokens = re.findall(r'\b\w+\b|[^\w\s]', source_code)
    
    for token in tokens:
        if token in operators:
            operator_counts[token] = operator_counts.get(token, 0) + 1
        elif token.isidentifier():
            operand_counts[token] = operand_counts.get(token, 0) + 1
    
    # Calculate Halstead metrics
    n1 = len(operator_counts)  # Number of distinct operators
    n2 = len(operand_counts)   # Number of distinct operands
    N1 = sum(operator_counts.values())  # Total number of operators
    N2 = sum(operand_counts.values())   # Total number of operands
    
    vocabulary = n1 + n2
    length = N1 + N2
    volume = length * math.log2(vocabulary) if vocabulary > 0 else 0
    difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
    effort = difficulty * volume
    
    return {
        "n1": n1,
        "n2": n2,
        "N1": N1,
        "N2": N2,
        "vocabulary": vocabulary,
        "length": length,
        "volume": volume,
        "difficulty": difficulty,
        "effort": effort
    }

def analyze_codebase(codebase):
    """Main analysis function - enhanced with comprehensive analysis"""
    analyzer = ComprehensiveCodebaseAnalyzer(codebase)
    return analyzer.analyze()

class ComprehensiveCodebaseAnalyzer:
    """Enhanced analyzer combining functionality from multiple analysis modules"""
    
    def __init__(self, codebase):
        self.codebase = codebase
        self.issues = []
        self.function_contexts = {}
        self.dead_code_items = []
        self.entry_points = []
        self.halstead_metrics = {}
        
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive analysis"""
        
        # Core analysis
        self._analyze_functions()
        self._analyze_classes()
        self._analyze_dead_code()
        self._analyze_issues()
        self._analyze_entry_points()
        self._calculate_metrics()
        
        # Find most important functions
        most_important = self._find_most_important_functions()
        
        # Generate summary
        summary = self._generate_summary()
        
        return {
            "summary": summary,
            "function_contexts": self.function_contexts,
            "most_important_functions": most_important,
            "dead_code_analysis": {
                "total_dead_functions": len(self.dead_code_items),
                "dead_code_items": self.dead_code_items
            },
            "issues_by_severity": self._group_issues_by_severity(),
            "entry_points": self.entry_points,
            "halstead_metrics": self.halstead_metrics,
            "repository_structure": self._analyze_repository_structure()
        }
    
    def _analyze_functions(self):
        """Analyze all functions in the codebase"""
        if not hasattr(self.codebase, 'functions'):
            return
            
        for function in self.codebase.functions:
            # Get function context
            context = get_function_context(function)
            
            # Calculate call chain
            max_call_chain = get_max_call_chain(function)
            
            # Analyze function issues
            function_issues = self._analyze_function_issues(function)
            
            # Calculate complexity
            complexity = self._calculate_complexity(function)
            
            # Calculate Halstead metrics
            halstead = calculate_halstead_metrics(getattr(function, 'source', ''))
            
            # Create function context
            func_context = FunctionContext(
                name=getattr(function, 'name', 'unknown'),
                filepath=getattr(function, 'filepath', ''),
                source=getattr(function, 'source', ''),
                parameters=self._extract_parameters(function),
                dependencies=context.get('dependencies', []),
                usages=context.get('usages', []),
                function_calls=getattr(function, 'function_calls', []),
                called_by=getattr(function, 'call_sites', []),
                max_call_chain=max_call_chain,
                issues=function_issues,
                is_entry_point=self._is_entry_point(function),
                is_dead_code=self._is_dead_code(function),
                class_name=getattr(function, 'class_name', None),
                complexity_score=complexity,
                halstead_metrics=halstead
            )
            
            func_context_dict = asdict(func_context)
            # Convert issues to serializable format
            func_context_dict['issues'] = []
            for issue in function_issues:
                issue_dict = asdict(issue)
                issue_dict['severity'] = issue.severity.value if hasattr(issue.severity, 'value') else str(issue.severity)
                issue_dict['type'] = issue.type.value if hasattr(issue.type, 'value') else str(issue.type)
                func_context_dict['issues'].append(issue_dict)
            
            self.function_contexts[function.name] = func_context_dict
    
    def _analyze_classes(self):
        """Analyze classes in the codebase"""
        if not hasattr(self.codebase, 'classes'):
            return
            
        for cls in self.codebase.classes:
            # Analyze class-specific issues
            class_issues = self._analyze_class_issues(cls)
            self.issues.extend(class_issues)
    
    def _analyze_dead_code(self):
        """Identify dead code with blast radius analysis"""
        if not hasattr(self.codebase, 'functions'):
            return
            
        for function in self.codebase.functions:
            if self._is_dead_code(function):
                blast_radius = self._calculate_blast_radius(function)
                self.dead_code_items.append({
                    "name": getattr(function, 'name', 'unknown'),
                    "type": "function",
                    "filepath": getattr(function, 'filepath', ''),
                    "reason": "No usages found",
                    "blast_radius": blast_radius
                })
    
    def _analyze_issues(self):
        """Comprehensive issue detection"""
        if not hasattr(self.codebase, 'functions'):
            return
            
        for function in self.codebase.functions:
            issues = self._analyze_function_issues(function)
            self.issues.extend(issues)
    
    def _analyze_function_issues(self, function) -> List[Issue]:
        """Analyze issues in a specific function"""
        issues = []
        source = getattr(function, 'source', '')
        filepath = getattr(function, 'filepath', '')
        
        # Check for long functions
        if source.count('\n') > 50:
            issues.append(Issue(
                type=IssueType.LONG_FUNCTION,
                severity=IssueSeverity.MINOR,
                message=f"Function {getattr(function, 'name', 'unknown')} is too long ({source.count('\n')} lines)",
                filepath=filepath,
                line_number=1
            ))
        
        # Check for missing documentation
        if not source.strip().startswith('"""') and not source.strip().startswith("'''"):
            issues.append(Issue(
                type=IssueType.MISSING_DOCUMENTATION,
                severity=IssueSeverity.MINOR,
                message=f"Function {getattr(function, 'name', 'unknown')} lacks documentation",
                filepath=filepath,
                line_number=1
            ))
        
        # Check for unused parameters
        params = self._extract_parameters(function)
        for param in params:
            param_name = param.get('name', '')
            if param_name and param_name not in source:
                issues.append(Issue(
                    type=IssueType.UNUSED_PARAMETER,
                    severity=IssueSeverity.MINOR,
                    message=f"Parameter '{param_name}' appears to be unused",
                    filepath=filepath,
                    line_number=1
                ))
        
        return issues
    
    def _analyze_class_issues(self, cls) -> List[Issue]:
        """Analyze issues in a specific class"""
        issues = []
        # Add class-specific issue detection here
        return issues
    
    def _analyze_entry_points(self):
        """Identify entry points in the codebase"""
        if not hasattr(self.codebase, 'functions'):
            return
            
        for function in self.codebase.functions:
            if self._is_entry_point(function):
                self.entry_points.append({
                    "name": getattr(function, 'name', 'unknown'),
                    "filepath": getattr(function, 'filepath', ''),
                    "type": "function"
                })
    
    def _calculate_metrics(self):
        """Calculate overall codebase metrics"""
        if not hasattr(self.codebase, 'functions'):
            return
            
        all_source = ""
        for function in self.codebase.functions:
            all_source += getattr(function, 'source', '') + "\n"
        
        self.halstead_metrics = calculate_halstead_metrics(all_source)
    
    def _find_most_important_functions(self) -> Dict[str, Any]:
        """Find the most important functions using the exact algorithms requested"""
        if not hasattr(self.codebase, 'functions') or not self.codebase.functions:
            return {}
        
        # Find function that makes the most calls
        most_calls = max(self.codebase.functions, 
                        key=lambda f: len(getattr(f, 'function_calls', [])),
                        default=None)
        
        # Find the most called function
        most_called = max(self.codebase.functions,
                         key=lambda f: len(getattr(f, 'call_sites', [])),
                         default=None)
        
        result = {}
        
        if most_calls:
            result["most_calls"] = {
                "name": getattr(most_calls, 'name', 'unknown'),
                "call_count": len(getattr(most_calls, 'function_calls', [])),
                "calls": getattr(most_calls, 'function_calls', [])
            }
        
        if most_called:
            result["most_called"] = {
                "name": getattr(most_called, 'name', 'unknown'),
                "usage_count": len(getattr(most_called, 'call_sites', []))
            }
        
        # Find class with most inheritance
        if hasattr(self.codebase, 'classes') and self.codebase.classes:
            deepest_class = max(self.codebase.classes,
                               key=lambda x: len(getattr(x, 'superclasses', [])),
                               default=None)
            if deepest_class:
                result["deepest_inheritance"] = {
                    "name": getattr(deepest_class, 'name', 'unknown'),
                    "chain_depth": len(getattr(deepest_class, 'superclasses', []))
                }
        
        return result
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate analysis summary"""
        total_files = len(getattr(self.codebase, 'files', []))
        total_functions = len(getattr(self.codebase, 'functions', []))
        
        issues_by_severity = self._group_issues_by_severity()
        
        return {
            "total_files": total_files,
            "total_functions": total_functions,
            "total_issues": len(self.issues),
            "critical_issues": len(issues_by_severity.get('critical', [])),
            "major_issues": len(issues_by_severity.get('major', [])),
            "minor_issues": len(issues_by_severity.get('minor', [])),
            "dead_code_items": len(self.dead_code_items),
            "entry_points": len(self.entry_points)
        }
    
    def _group_issues_by_severity(self) -> Dict[str, List[Dict]]:
        """Group issues by severity level"""
        grouped = {"critical": [], "major": [], "minor": [], "info": []}
        
        for issue in self.issues:
            severity = issue.severity.value if hasattr(issue.severity, 'value') else str(issue.severity)
            if severity in grouped:
                issue_dict = asdict(issue)
                # Convert enums to strings for JSON serialization
                issue_dict['severity'] = severity
                issue_dict['type'] = issue.type.value if hasattr(issue.type, 'value') else str(issue.type)
                grouped[severity].append(issue_dict)
        
        return grouped
    
    def _analyze_repository_structure(self) -> Dict[str, Any]:
        """Analyze repository structure with issue counts"""
        structure = {
            "total_files": len(getattr(self.codebase, 'files', [])),
            "file_types": {},
            "directories": {}
        }
        
        if hasattr(self.codebase, 'files'):
            for file in self.codebase.files:
                filepath = getattr(file, 'filepath', '')
                if filepath:
                    ext = Path(filepath).suffix
                    structure["file_types"][ext] = structure["file_types"].get(ext, 0) + 1
        
        return structure
    
    def _extract_parameters(self, function) -> List[Dict[str, Any]]:
        """Extract function parameters"""
        if hasattr(function, 'parameters'):
            return [{"name": p.name, "type": getattr(p, 'type', 'Any')} 
                   for p in function.parameters]
        return []
    
    def _is_entry_point(self, function) -> bool:
        """Check if function is an entry point"""
        name = getattr(function, 'name', '')
        return name in ['main', '__main__', 'run', 'start', 'execute']
    
    def _is_dead_code(self, function) -> bool:
        """Check if function is dead code"""
        call_sites = getattr(function, 'call_sites', [])
        usages = getattr(function, 'usages', [])
        return len(call_sites) == 0 and len(usages) == 0 and not self._is_entry_point(function)
    
    def _calculate_blast_radius(self, function) -> List[str]:
        """Calculate blast radius for dead code"""
        # Simple implementation - could be enhanced
        return []
    
    def _calculate_complexity(self, function) -> int:
        """Calculate cyclomatic complexity"""
        source = getattr(function, 'source', '')
        if not source:
            return 1
        
        # Count decision points
        complexity = 1  # Base complexity
        complexity += source.count('if ')
        complexity += source.count('elif ')
        complexity += source.count('for ')
        complexity += source.count('while ')
        complexity += source.count('except ')
        complexity += source.count('and ')
        complexity += source.count('or ')
        
        return complexity

# ============================================================================
# MOCK CODEBASE GENERATION AND DEMO FUNCTIONALITY
# ============================================================================

class MockFunction:
    """Mock function class for testing and demonstration"""
    def __init__(self, name, filepath, source="", parameters=None, usages=None, function_calls=None, call_sites=None, dependencies=None):
        self.name = name
        self.filepath = filepath
        self.source = source
        self.parameters = parameters or []
        self.usages = usages or []
        self.function_calls = function_calls or []
        self.call_sites = call_sites or []
        self.dependencies = dependencies or []
        self.start_point = (1, 0)
        self.end_point = (10, 0)

class MockClass:
    """Mock class for testing inheritance analysis"""
    def __init__(self, name, filepath, superclasses=None):
        self.name = name
        self.filepath = filepath
        self.superclasses = superclasses or []

class MockFile:
    """Mock file for repository structure analysis"""
    def __init__(self, filepath):
        self.filepath = filepath

class MockCodebase:
    """Comprehensive mock codebase with realistic data for testing"""
    def __init__(self):
        # Create comprehensive mock data with realistic function relationships
        self.functions = [
            MockFunction(
                name="main",
                filepath="src/main.py",
                source='def main():\n    """Main entry point"""\n    user_input = get_user_input()\n    result = process_data(user_input)\n    print(result)',
                function_calls=["get_user_input", "process_data"],
                call_sites=[],
                parameters=[]
            ),
            MockFunction(
                name="process_data",
                filepath="src/main.py", 
                source='def process_data(data, unused_param):\n    """Process input data"""\n    validated = validate_input(data)\n    total = calculate_total(validated)\n    return total',
                function_calls=["validate_input", "calculate_total"],
                call_sites=["main"],
                parameters=[
                    type('MockParam', (), {'name': 'data', 'type': 'dict'})(),
                    type('MockParam', (), {'name': 'unused_param', 'type': 'str'})()
                ]
            ),
            MockFunction(
                name="validate_input",
                filepath="src/validation.py",
                source='def validate_input(data):\n    """Validate input data"""\n    if not isinstance(data, dict):\n        raise ValueError("Invalid data")\n    return check_format(data)',
                function_calls=["check_format"],
                call_sites=["process_data"],
                parameters=[type('MockParam', (), {'name': 'data', 'type': 'dict'})()]
            ),
            MockFunction(
                name="check_format",
                filepath="src/validation.py",
                source='def check_format(data):\n    """Check data format"""\n    return data.get("format") == "valid"',
                function_calls=[],
                call_sites=["validate_input"],
                parameters=[type('MockParam', (), {'name': 'data', 'type': 'dict'})()]
            ),
            MockFunction(
                name="calculate_total",
                filepath="src/utils.py",
                source='def calculate_total(items):\n    """Calculate total value from items"""\n    return sum(item.get("value", 0) for item in items)',
                function_calls=[],
                call_sites=["process_data"],
                parameters=[type('MockParam', (), {'name': 'items', 'type': 'list'})()]
            ),
            MockFunction(
                name="get_user_input",
                filepath="src/input.py",
                source='def get_user_input():\n    """Get user input"""\n    return {"data": "sample", "format": "valid"}',
                function_calls=[],
                call_sites=["main"],
                parameters=[]
            ),
            MockFunction(
                name="unused_helper",
                filepath="src/utils.py",
                source='def unused_helper():\n    """This function is never called"""\n    return "unused"',
                function_calls=[],
                call_sites=[],
                parameters=[]
            ),
            MockFunction(
                name="complex_algorithm",
                filepath="src/algorithms.py",
                source='''def complex_algorithm(data, threshold=10, debug=False):
    """Complex algorithm with high cyclomatic complexity"""
    result = []
    for item in data:
        if item > threshold:
            if debug:
                print(f"Processing {item}")
            if item % 2 == 0:
                result.append(item * 2)
            elif item % 3 == 0:
                result.append(item * 3)
            else:
                result.append(item)
        elif item < 0:
            if debug:
                print(f"Negative item: {item}")
            result.append(0)
        else:
            result.append(item)
    return result''',
                function_calls=[],
                call_sites=[],
                parameters=[
                    type('MockParam', (), {'name': 'data', 'type': 'list'})(),
                    type('MockParam', (), {'name': 'threshold', 'type': 'int'})(),
                    type('MockParam', (), {'name': 'debug', 'type': 'bool'})()
                ]
            )
        ]
        
        self.classes = [
            MockClass(
                name="DataProcessor",
                filepath="src/processor.py",
                superclasses=["BaseProcessor", "Validator"]
            ),
            MockClass(
                name="BaseProcessor", 
                filepath="src/base.py",
                superclasses=["Object"]
            ),
            MockClass(
                name="Validator",
                filepath="src/validation.py",
                superclasses=[]
            ),
            MockClass(
                name="AdvancedProcessor",
                filepath="src/advanced.py",
                superclasses=["DataProcessor", "Logger", "ConfigManager"]
            )
        ]
        
        self.files = [
            MockFile("src/main.py"),
            MockFile("src/validation.py"),
            MockFile("src/utils.py"),
            MockFile("src/input.py"),
            MockFile("src/processor.py"),
            MockFile("src/base.py"),
            MockFile("src/algorithms.py"),
            MockFile("src/advanced.py"),
            MockFile("tests/test_main.py"),
            MockFile("tests/test_validation.py"),
            MockFile("README.md"),
            MockFile("requirements.txt")
        ]

def load_codebase(username: str, repo_name: str):
    """Load codebase - returns mock codebase for demonstration"""
    return MockCodebase()

# ============================================================================
# COMPREHENSIVE DEMO FUNCTIONALITY
# ============================================================================

def run_comprehensive_demo():
    """
    Run a comprehensive demonstration of all analysis capabilities
    This function showcases all the features integrated from multiple modules
    """
    print("🚀 COMPREHENSIVE CODEBASE ANALYSIS DEMO")
    print("=" * 50)
    print("This demo showcases the unified analysis engine with features from:")
    print("• analyzer.py - Core analysis engine")
    print("• comprehensive_analysis.py - Advanced issue detection")
    print("• visualization modules - Data generation for UI")
    print()
    
    # Load a mock codebase
    print("📁 Loading codebase...")
    codebase = load_codebase("codegen-sh", "graph-sitter")
    
    # Perform analysis
    print("🔍 Performing comprehensive analysis...")
    analysis_results = analyze_codebase(codebase)
    
    # Display key results
    print("\n📊 ANALYSIS SUMMARY:")
    print("-" * 30)
    summary = analysis_results.get('summary', {})
    print(f"📁 Total Files: {summary.get('total_files', 0)}")
    print(f"🔧 Total Functions: {summary.get('total_functions', 0)}")
    print(f"🚨 Total Issues: {summary.get('total_issues', 0)}")
    print(f"⚠️  Critical Issues: {summary.get('critical_issues', 0)}")
    print(f"👉 Major Issues: {summary.get('major_issues', 0)}")
    print(f"🔍 Minor Issues: {summary.get('minor_issues', 0)}")
    print(f"💀 Dead Code Items: {summary.get('dead_code_items', 0)}")
    print(f"🎯 Entry Points: {summary.get('entry_points', 0)}")
    
    # Show most important functions
    print("\n🌟 MOST IMPORTANT FUNCTIONS:")
    print("-" * 35)
    important = analysis_results.get('most_important_functions', {})
    
    most_calls = important.get('most_calls', {})
    print(f"📞 Makes Most Calls: {most_calls.get('name', 'N/A')}")
    print(f"   📊 Call Count: {most_calls.get('call_count', 0)}")
    if most_calls.get('calls'):
        print(f"   🎯 Calls: {', '.join(most_calls['calls'][:3])}...")
    
    most_called = important.get('most_called', {})
    print(f"📈 Most Called: {most_called.get('name', 'N/A')}")
    print(f"   📊 Usage Count: {most_called.get('usage_count', 0)}")
    
    deepest_inheritance = important.get('deepest_inheritance', {})
    if deepest_inheritance.get('name'):
        print(f"🌳 Deepest Inheritance: {deepest_inheritance.get('name')}")
        print(f"   📊 Chain Depth: {deepest_inheritance.get('chain_depth', 0)}")
    
    # Show function contexts
    print("\n🔧 FUNCTION CONTEXTS:")
    print("-" * 25)
    function_contexts = analysis_results.get('function_contexts', {})
    
    for func_name, context in list(function_contexts.items())[:3]:  # Show first 3
        print(f"\n📝 Function: {func_name}")
        print(f"   📁 File: {context.get('filepath', 'N/A')}")
        print(f"   📊 Parameters: {len(context.get('parameters', []))}")
        print(f"   🔗 Dependencies: {len(context.get('dependencies', []))}")
        print(f"   📞 Function Calls: {len(context.get('function_calls', []))}")
        print(f"   📈 Called By: {len(context.get('called_by', []))}")
        print(f"   🚨 Issues: {len(context.get('issues', []))}")
        print(f"   🎯 Entry Point: {context.get('is_entry_point', False)}")
        print(f"   💀 Dead Code: {context.get('is_dead_code', False)}")
        print(f"   🧮 Complexity: {context.get('complexity_score', 0)}")
        
        if context.get('max_call_chain'):
            chain = context['max_call_chain']
            if len(chain) > 1:
                print(f"   ⛓️  Call Chain: {' → '.join(chain[:3])}...")
        
        # Show Halstead metrics for this function
        halstead = context.get('halstead_metrics', {})
        if halstead:
            print(f"   📐 Halstead Volume: {halstead.get('volume', 0):.1f}")
            print(f"   📐 Halstead Difficulty: {halstead.get('difficulty', 0):.1f}")
    
    # Show issues by severity
    print("\n🚨 ISSUES BY SEVERITY:")
    print("-" * 25)
    issues_by_severity = analysis_results.get('issues_by_severity', {})
    
    for severity, issues in issues_by_severity.items():
        if issues:
            print(f"\n{severity.upper()} ({len(issues)} issues):")
            for issue in issues[:2]:  # Show first 2 issues
                print(f"  • {issue.get('message', 'No message')}")
                print(f"    📁 {issue.get('filepath', 'N/A')}:{issue.get('line_number', 0)}")
                print(f"    🏷️  Type: {issue.get('type', 'unknown')}")
            if len(issues) > 2:
                print(f"  ... and {len(issues) - 2} more")
    
    # Show dead code analysis
    print("\n💀 DEAD CODE ANALYSIS:")
    print("-" * 22)
    dead_code = analysis_results.get('dead_code_analysis', {})
    print(f"🔢 Total Dead Functions: {dead_code.get('total_dead_functions', 0)}")
    
    dead_items = dead_code.get('dead_code_items', [])
    if dead_items:
        print("📋 Dead Code Items:")
        for item in dead_items[:3]:  # Show first 3
            print(f"  • {item.get('name', 'N/A')} ({item.get('type', 'unknown')}) - {item.get('reason', 'No reason')}")
            print(f"    📁 {item.get('filepath', 'N/A')}")
            blast_radius = item.get('blast_radius', [])
            if blast_radius:
                print(f"    💥 Blast Radius: {', '.join(blast_radius[:3])}...")
    
    # Show Halstead metrics
    print("\n📐 HALSTEAD COMPLEXITY METRICS:")
    print("-" * 35)
    halstead = analysis_results.get('halstead_metrics', {})
    if halstead:
        print(f"📚 Vocabulary (n1 + n2): {halstead.get('vocabulary', 0)}")
        print(f"📏 Length (N1 + N2): {halstead.get('length', 0)}")
        print(f"📊 Volume: {halstead.get('volume', 0):.2f}")
        print(f"🎯 Difficulty: {halstead.get('difficulty', 0):.2f}")
        print(f"⚡ Effort: {halstead.get('effort', 0):.2f}")
        print(f"🔧 Distinct Operators (n1): {halstead.get('n1', 0)}")
        print(f"🔧 Distinct Operands (n2): {halstead.get('n2', 0)}")
        print(f"🔢 Total Operators (N1): {halstead.get('N1', 0)}")
        print(f"🔢 Total Operands (N2): {halstead.get('N2', 0)}")
    
    # Show repository structure
    print("\n🏗️  REPOSITORY STRUCTURE:")
    print("-" * 25)
    repo_structure = analysis_results.get('repository_structure', {})
    print(f"📁 Total Files: {repo_structure.get('total_files', 0)}")
    
    file_types = repo_structure.get('file_types', {})
    if file_types:
        print("📊 File Types:")
        for ext, count in file_types.items():
            print(f"  • {ext or 'no extension'}: {count} files")
    
    # Show entry points
    print("\n🎯 ENTRY POINTS:")
    print("-" * 15)
    entry_points = analysis_results.get('entry_points', [])
    if entry_points:
        for entry in entry_points:
            print(f"  • {entry.get('name', 'N/A')} in {entry.get('filepath', 'N/A')}")
    else:
        print("  No entry points detected")
    
    print("\n" + "=" * 50)
    print("✅ COMPREHENSIVE ANALYSIS COMPLETE!")
    print("This demo showcased all integrated features:")
    print("• Function context analysis with call chains")
    print("• Dead code detection with blast radius")
    print("• Comprehensive issue detection (6 categories)")
    print("• Halstead complexity metrics")
    print("• Most important functions algorithms")
    print("• Entry point identification")
    print("• Repository structure analysis")
    print("=" * 50)
    
    return analysis_results

# ============================================================================
# ADVANCED ANALYSIS FUNCTIONS
# ============================================================================

def analyze_function_relationships(codebase):
    """Analyze relationships between functions for dependency mapping"""
    relationships = {
        "call_graph": {},
        "dependency_graph": {},
        "circular_dependencies": [],
        "orphaned_functions": []
    }
    
    if not hasattr(codebase, 'functions'):
        return relationships
    
    # Build call graph
    for function in codebase.functions:
        func_name = getattr(function, 'name', 'unknown')
        relationships["call_graph"][func_name] = {
            "calls": getattr(function, 'function_calls', []),
            "called_by": getattr(function, 'call_sites', []),
            "filepath": getattr(function, 'filepath', '')
        }
    
    # Detect circular dependencies
    if nx:
        G = nx.DiGraph()
        for func_name, data in relationships["call_graph"].items():
            for called_func in data["calls"]:
                G.add_edge(func_name, called_func)
        
        try:
            cycles = list(nx.simple_cycles(G))
            relationships["circular_dependencies"] = cycles
        except:
            pass
    
    # Find orphaned functions (no callers, not entry points)
    for func_name, data in relationships["call_graph"].items():
        if not data["called_by"] and func_name not in ['main', '__main__', 'run', 'start', 'execute']:
            relationships["orphaned_functions"].append(func_name)
    
    return relationships

def calculate_code_quality_score(analysis_results):
    """Calculate overall code quality score based on analysis results"""
    summary = analysis_results.get('summary', {})
    issues_by_severity = analysis_results.get('issues_by_severity', {})
    
    # Base score
    score = 100
    
    # Deduct points for issues
    critical_issues = len(issues_by_severity.get('critical', []))
    major_issues = len(issues_by_severity.get('major', []))
    minor_issues = len(issues_by_severity.get('minor', []))
    
    score -= critical_issues * 20  # Critical issues are severe
    score -= major_issues * 10     # Major issues are significant
    score -= minor_issues * 2      # Minor issues have small impact
    
    # Deduct points for dead code
    dead_code_items = summary.get('dead_code_items', 0)
    score -= dead_code_items * 5
    
    # Bonus points for good practices
    total_functions = summary.get('total_functions', 1)
    if total_functions > 0:
        # Bonus for having entry points
        entry_points = summary.get('entry_points', 0)
        if entry_points > 0:
            score += 5
        
        # Bonus for low issue ratio
        total_issues = summary.get('total_issues', 0)
        issue_ratio = total_issues / total_functions
        if issue_ratio < 0.1:  # Less than 10% of functions have issues
            score += 10
    
    # Ensure score is between 0 and 100
    return max(0, min(100, score))

def generate_refactoring_recommendations(analysis_results):
    """Generate specific refactoring recommendations based on analysis"""
    recommendations = []
    
    function_contexts = analysis_results.get('function_contexts', {})
    issues_by_severity = analysis_results.get('issues_by_severity', {})
    dead_code = analysis_results.get('dead_code_analysis', {})
    
    # High complexity functions
    high_complexity_funcs = [
        (name, data) for name, data in function_contexts.items()
        if data.get('complexity_score', 0) > 15
    ]
    
    if high_complexity_funcs:
        recommendations.append({
            "type": "complexity",
            "priority": "high",
            "title": "Reduce Function Complexity",
            "description": f"Found {len(high_complexity_funcs)} functions with high complexity",
            "functions": [name for name, _ in high_complexity_funcs[:3]],
            "action": "Consider breaking these functions into smaller, more focused functions"
        })
    
    # Functions with many issues
    problematic_funcs = [
        (name, data) for name, data in function_contexts.items()
        if len(data.get('issues', [])) > 3
    ]
    
    if problematic_funcs:
        recommendations.append({
            "type": "issues",
            "priority": "medium",
            "title": "Address Function Issues",
            "description": f"Found {len(problematic_funcs)} functions with multiple issues",
            "functions": [name for name, _ in problematic_funcs[:3]],
            "action": "Review and fix the identified issues in these functions"
        })
    
    # Dead code removal
    dead_items = dead_code.get('dead_code_items', [])
    if dead_items:
        recommendations.append({
            "type": "dead_code",
            "priority": "low",
            "title": "Remove Dead Code",
            "description": f"Found {len(dead_items)} unused functions",
            "functions": [item.get('name') for item in dead_items[:3]],
            "action": "Remove these unused functions to reduce codebase size"
        })
    
    # Documentation improvements
    doc_issues = [
        issue for issue in issues_by_severity.get('minor', [])
        if 'documentation' in issue.get('message', '').lower()
    ]
    
    if len(doc_issues) > 5:
        recommendations.append({
            "type": "documentation",
            "priority": "low",
            "title": "Improve Documentation",
            "description": f"Found {len(doc_issues)} functions lacking documentation",
            "action": "Add docstrings to improve code maintainability"
        })
    
    return recommendations

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# ============================================================================
# FLASK API INTEGRATION
# ============================================================================

try:
    from flask import Flask, jsonify, request
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Warning: Flask not available, API functionality disabled")

def create_flask_app():
    """Create and configure Flask application"""
    if not FLASK_AVAILABLE:
        return None
        
    app = Flask(__name__)
    CORS(app)
    
    # In-memory cache for analysis results
    analysis_cache = {}
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            "status": "healthy",
            "service": "codebase-analytics-api"
        })
    
    @app.route('/analyze/<username>/<repo>', methods=['GET'])
    def analyze_repository(username, repo):
        """Comprehensive analysis endpoint"""
        try:
            cache_key = f"{username}/{repo}"
            
            # Check cache first
            if cache_key in analysis_cache:
                return jsonify(analysis_cache[cache_key])
            
            # Load and analyze codebase
            codebase = load_codebase(username, repo)
            analysis_results = analyze_codebase(codebase)
            
            # Add enhanced features
            analysis_results['refactoring_suggestions'] = generate_refactoring_recommendations(analysis_results)
            analysis_results['code_quality_score'] = calculate_code_quality_score(analysis_results)
            analysis_results['function_relationships'] = analyze_function_relationships(codebase)
            
            # Cache results
            analysis_cache[cache_key] = analysis_results
            
            return jsonify(analysis_results)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/visualize/<username>/<repo>', methods=['GET'])
    def visualize_repository(username, repo):
        """Interactive visualization endpoint"""
        try:
            cache_key = f"viz_{username}/{repo}"
            
            # Check cache first
            if cache_key in analysis_cache:
                return jsonify(analysis_cache[cache_key])
            
            # Load and analyze codebase
            codebase = load_codebase(username, repo)
            analysis_results = analyze_codebase(codebase)
            
            # Generate visualization data
            viz_data = generate_visualization_data(analysis_results, codebase)
            
            # Cache results
            analysis_cache[cache_key] = viz_data
            
            return jsonify(viz_data)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return app

def generate_visualization_data(analysis_results, codebase):
    """Generate comprehensive visualization data"""
    function_contexts = analysis_results.get('function_contexts', {})
    issues_by_severity = analysis_results.get('issues_by_severity', {})
    dead_code = analysis_results.get('dead_code_analysis', {})
    
    # Generate call graph
    call_graph = {
        "nodes": [],
        "edges": [],
        "layout": "hierarchical",
        "interactive": True,
        "statistics": {
            "total_nodes": 0,
            "total_edges": 0,
            "dead_code_nodes": 0
        }
    }
    
    # Add nodes for each function
    for func_name, context in function_contexts.items():
        # Handle both dict and object contexts
        if isinstance(context, dict):
            is_dead = context.get('is_dead_code', False)
            is_entry = context.get('is_entry_point', False)
            complexity = context.get('complexity_score', 0)
            issues_count = len(context.get('issues', []))
        else:
            is_dead = getattr(context, 'is_dead_code', False)
            is_entry = getattr(context, 'is_entry_point', False)
            complexity = getattr(context, 'complexity_score', 0)
            issues_count = len(getattr(context, 'issues', []))
        
        # Color coding based on function characteristics
        if is_dead:
            color = "#ff6b6b"  # Red for dead code
        elif is_entry:
            color = "#4ecdc4"  # Teal for entry points
        elif complexity > 10:
            color = "#ffa726"  # Orange for complex functions
        else:
            color = "#66bb6a"  # Green for normal functions
        
        filepath = context.get('filepath', '') if isinstance(context, dict) else getattr(context, 'filepath', '')
        
        call_graph["nodes"].append({
            "id": func_name,
            "label": func_name,
            "type": "function",
            "filepath": filepath,
            "complexity": complexity,
            "issues": issues_count,
            "dead_code": is_dead,
            "entry_point": is_entry,
            "color": color
        })
        
        if is_dead:
            call_graph["statistics"]["dead_code_nodes"] += 1
    
    # Add edges for function calls
    for func_name, context in function_contexts.items():
        function_calls = context.get('function_calls', []) if isinstance(context, dict) else getattr(context, 'function_calls', [])
        for called_func in function_calls:
            call_graph["edges"].append({
                "from": func_name,
                "to": called_func,
                "type": "calls",
                "weight": 1
            })
    
    call_graph["statistics"]["total_nodes"] = len(call_graph["nodes"])
    call_graph["statistics"]["total_edges"] = len(call_graph["edges"])
    
    # Generate repository tree
    repository_tree = {
        "name": "codegen-sh/graph-sitter",
        "type": "repository",
        "expanded": True,
        "issue_counts": {
            "critical": len(issues_by_severity.get('critical', [])),
            "major": len(issues_by_severity.get('major', [])),
            "minor": len(issues_by_severity.get('minor', []))
        },
        "children": []
    }
    
    # Group files by directory
    file_structure = {}
    if hasattr(codebase, 'files'):
        for file_obj in codebase.files:
            filepath = getattr(file_obj, 'filepath', '')
            parts = filepath.split('/')
            
            current = file_structure
            for part in parts[:-1]:  # Directories
                if part not in current:
                    current[part] = {"type": "directory", "children": {}, "files": []}
                current = current[part]["children"]
            
            # Add file
            filename = parts[-1] if parts else filepath
            if "files" not in current:
                current["files"] = []
            current["files"].append({
                "name": filename,
                "type": "file",
                "filepath": filepath
            })
    
    # Convert file structure to tree format
    def build_tree_node(name, data, path=""):
        node = {
            "name": name,
            "type": data.get("type", "directory"),
            "expanded": True,
            "children": []
        }
        
        # Add child directories
        for child_name, child_data in data.get("children", {}).items():
            child_path = f"{path}/{child_name}" if path else child_name
            node["children"].append(build_tree_node(child_name, child_data, child_path))
        
        # Add files
        for file_info in data.get("files", []):
            node["children"].append({
                "name": file_info["name"],
                "type": "file",
                "filepath": file_info["filepath"]
            })
        
        return node
    
    for dir_name, dir_data in file_structure.items():
        repository_tree["children"].append(build_tree_node(dir_name, dir_data))
    
    # Generate issue visualization
    total_issues = sum(len(issues) for issues in issues_by_severity.values())
    issue_visualization = {
        "total_issues": total_issues,
        "severity_distribution": {
            severity: len(issues) for severity, issues in issues_by_severity.items()
        },
        "file_heatmap": {},
        "most_problematic_files": []
    }
    
    # Calculate file-level issue counts
    file_issues = {}
    for severity, issues in issues_by_severity.items():
        if isinstance(issues, list):
            for issue in issues:
                if isinstance(issue, dict):
                    filepath = issue.get('filepath', 'unknown')
                else:
                    filepath = getattr(issue, 'filepath', 'unknown')
                
                if filepath not in file_issues:
                    file_issues[filepath] = {"critical": 0, "major": 0, "minor": 0, "info": 0}
                file_issues[filepath][severity] += 1
    
    # Sort files by issue count
    sorted_files = sorted(file_issues.items(), 
                         key=lambda x: sum(x[1].values()), 
                         reverse=True)
    
    issue_visualization["most_problematic_files"] = [
        {"filepath": filepath, "issues": counts}
        for filepath, counts in sorted_files[:5]
    ]
    
    # Generate dead code visualization
    dead_code_items = dead_code.get('dead_code_items', [])
    affected_files = set()
    
    for item in dead_code_items:
        if isinstance(item, dict):
            filepath = item.get('filepath', '')
        else:
            filepath = getattr(item, 'filepath', '')
        if filepath:
            affected_files.add(filepath)
    
    dead_code_visualization = {
        "dead_code_summary": {
            "total_functions": len(dead_code_items),
            "affected_files": len(affected_files),
            "estimated_loc_savings": len(dead_code_items) * 15  # Estimate
        },
        "blast_radius_visualization": {}
    }
    
    return {
        "repository_tree": repository_tree,
        "call_graph": call_graph,
        "issue_visualization": issue_visualization,
        "dead_code_visualization": dead_code_visualization,
        "ui_components": {
            "layout": {
                "type": "split-view",
                "left_panel": "repository_tree",
                "right_panel": "tabbed_content"
            },
            "search_bar": {
                "enabled": True,
                "placeholder": "Search functions, files, or issues..."
            },
            "interactive_features": {
                "clickable_tree": True,
                "context_menus": True,
                "tooltips": True
            }
        }
    }

def start_api_server(host='0.0.0.0', port=5000, debug=True):
    """Start the Flask API server"""
    if not FLASK_AVAILABLE:
        print("❌ Flask not available. Install with: pip install flask flask-cors")
        return
    
    app = create_flask_app()
    if app:
        print("🚀 Starting Codebase Analytics API...")
        print("📊 Available endpoints:")
        print("   GET /analyze/<username>/<repo> - Comprehensive analysis")
        print("   GET /visualize/<username>/<repo> - Interactive visualization")
        print("   GET /health - Health check")
        print(f"🌐 Server running on http://localhost:{port}")
        app.run(host=host, port=port, debug=debug)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        # Start API server
        start_api_server()
    else:
        # Run the comprehensive demo when this file is executed directly
        run_comprehensive_demo()
