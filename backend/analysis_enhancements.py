"""
Codebase Analytics - Enhanced Analysis Functions
Additional functionality to complete the analysis engine
"""

import re
import ast
import os
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, Counter
import networkx as nx

from .models import (
    CodeIssue, IssueType, IssueSeverity, FunctionContext, 
    AutomatedResolution, AnalysisConfig
)

# Graph-sitter integration with graceful fallback
try:
    import graph_sitter
    from graph_sitter.core.codebase import Codebase
    from graph_sitter.core.function import Function
    from graph_sitter.core.file import SourceFile
    GRAPH_SITTER_AVAILABLE = True
except ImportError:
    GRAPH_SITTER_AVAILABLE = False


class EnhancedAnalysisEngine:
    """Enhanced analysis capabilities to complement the main analyzer"""
    
    def __init__(self, codebase):
        self.codebase = codebase
        self.cache = {}
        
    def enhanced_undefined_variable_detection(self, source_file) -> List[CodeIssue]:
        """Enhanced undefined variable detection with scope tracking"""
        issues = []
        if not source_file.source:
            return issues
            
        lines = source_file.source.split("\n")
        defined_vars = set()
        scopes = [set()]  # Stack of scopes
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Track scope changes
            indent_level = len(line) - len(line.lstrip())
            
            # Track variable definitions
            assignment_patterns = [
                r'(\w+)\s*=',  # variable = value
                r'for\s+(\w+)\s+in',  # for var in
                r'def\s+\w+\([^)]*(\w+)',  # function parameters
                r'import\s+(\w+)',  # imports
                r'from\s+\w+\s+import\s+(\w+)',  # from imports
                r'with\s+.*\s+as\s+(\w+)',  # with statements
                r'except\s+.*\s+as\s+(\w+)',  # exception handling
            ]
            
            for pattern in assignment_patterns:
                matches = re.findall(pattern, stripped)
                defined_vars.update(matches)
                scopes[-1].update(matches)
            
            # Check for potential undefined variable usage
            if "NameError" in line or "undefined" in line.lower():
                issues.append(CodeIssue(
                    issue_type=IssueType.UNDEFINED_VARIABLE,
                    severity=IssueSeverity.CRITICAL,
                    message="Potential undefined variable usage",
                    filepath=source_file.file_path,
                    line_number=i + 1,
                    column_number=0,
                    context={"line": line.strip()},
                    suggested_fix="Define variable before use or check spelling"
                ))
            
            # Advanced undefined variable detection
            var_usage = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', stripped)
            builtin_names = {
                'if', 'else', 'elif', 'for', 'while', 'def', 'class', 'import', 'from', 
                'return', 'yield', 'break', 'continue', 'pass', 'try', 'except', 'finally',
                'print', 'len', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple',
                'range', 'enumerate', 'zip', 'map', 'filter', 'sum', 'max', 'min',
                'True', 'False', 'None', 'and', 'or', 'not', 'in', 'is'
            }
            
            for var in var_usage:
                if (var not in defined_vars and 
                    var not in builtin_names and
                    not var.isupper() and  # Skip constants
                    '=' not in stripped and  # Skip assignment lines
                    not stripped.startswith('#') and  # Skip comments
                    not stripped.startswith('def ') and  # Skip function definitions
                    not stripped.startswith('class ')):  # Skip class definitions
                    
                    # Only flag if variable appears to be used, not defined
                    if re.search(rf'\b{var}\b(?!\s*[=\(])', stripped):
                        issues.append(CodeIssue(
                            issue_type=IssueType.UNDEFINED_VARIABLE,
                            severity=IssueSeverity.MAJOR,
                            message=f"Potentially undefined variable: {var}",
                            filepath=source_file.file_path,
                            line_number=i + 1,
                            column_number=stripped.find(var),
                            context={"variable": var, "line": line.strip()},
                            suggested_fix=f"Define variable '{var}' before use or check if it's imported"
                        ))
        
        return issues
    
    def enhanced_missing_returns_detection(self, source_file) -> List[CodeIssue]:
        """Enhanced missing return statement detection"""
        issues = []
        
        for symbol in source_file.symbols:
            if hasattr(symbol, '__class__') and 'Function' in str(symbol.__class__):
                if hasattr(symbol, "source") and symbol.source:
                    # More sophisticated return detection
                    source_lines = symbol.source.split('\n')
                    has_return = False
                    has_yield = False
                    is_generator = False
                    
                    for line in source_lines:
                        stripped = line.strip()
                        if stripped.startswith('return'):
                            has_return = True
                        elif stripped.startswith('yield'):
                            has_yield = True
                            is_generator = True
                        elif 'yield' in stripped:
                            is_generator = True
                    
                    # Check if function should have return statement
                    function_name = getattr(symbol, 'name', 'unknown')
                    
                    # Skip certain function types
                    skip_patterns = ['__init__', '__del__', 'setUp', 'tearDown', 'test_']
                    should_skip = any(pattern in function_name for pattern in skip_patterns)
                    
                    if (not has_return and not is_generator and not should_skip and 
                        len(source_lines) > 3):  # Only check non-trivial functions
                        
                        # Check if function has meaningful logic that should return something
                        has_logic = any(
                            any(keyword in line for keyword in ['if', 'for', 'while', 'try', 'with'])
                            for line in source_lines
                        )
                        
                        if has_logic:
                            issues.append(CodeIssue(
                                issue_type=IssueType.MISSING_RETURN,
                                severity=IssueSeverity.MINOR,
                                message=f"Function {function_name} may be missing return statement",
                                filepath=source_file.file_path,
                                line_number=getattr(symbol, 'line_start', 0),
                                column_number=0,
                                function_name=function_name,
                                suggested_fix="Add explicit return statement or return None",
                                context={"function_length": len(source_lines), "has_logic": has_logic}
                            ))
        
        return issues
    
    def enhanced_nesting_depth_calculation(self, source: str) -> int:
        """Enhanced nesting depth calculation with proper indentation tracking"""
        lines = source.split('\n')
        max_depth = 0
        current_depth = 0
        indent_stack = [0]  # Stack to track indentation levels
        
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
                
            # Calculate current indentation
            current_indent = len(line) - len(line.lstrip())
            
            # Handle indentation changes
            if current_indent > indent_stack[-1]:
                # Increased indentation - new nesting level
                indent_stack.append(current_indent)
                current_depth = len(indent_stack) - 1
            elif current_indent < indent_stack[-1]:
                # Decreased indentation - pop levels
                while len(indent_stack) > 1 and current_indent < indent_stack[-1]:
                    indent_stack.pop()
                current_depth = len(indent_stack) - 1
            
            # Check for nesting keywords
            nesting_keywords = ['if', 'elif', 'else:', 'for', 'while', 'try:', 'except', 'finally:', 'with', 'def', 'class']
            if any(stripped.startswith(keyword) for keyword in nesting_keywords):
                max_depth = max(max_depth, current_depth + 1)
        
        return max_depth
    
    def extract_function_calls(self, function) -> List[str]:
        """Extract function calls from a function using AST and regex analysis"""
        calls = []
        
        if not hasattr(function, 'source') or not function.source:
            return calls
        
        try:
            # Try AST parsing first
            tree = ast.parse(function.source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        calls.append(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        calls.append(node.func.attr)
        except SyntaxError:
            # Fallback to regex-based extraction
            call_patterns = [
                r'(\w+)\s*\(',  # function_name(
                r'\.(\w+)\s*\(',  # object.method(
                r'(\w+)\s*=\s*(\w+)\s*\(',  # var = function(
            ]
            
            for pattern in call_patterns:
                matches = re.findall(pattern, function.source)
                if isinstance(matches[0], tuple) if matches else False:
                    calls.extend([match[0] for match in matches if match[0]])
                else:
                    calls.extend(matches)
        
        # Filter out common keywords and built-ins
        filtered_calls = []
        builtin_functions = {
            'print', 'len', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple',
            'range', 'enumerate', 'zip', 'map', 'filter', 'sum', 'max', 'min',
            'open', 'input', 'type', 'isinstance', 'hasattr', 'getattr', 'setattr'
        }
        
        for call in calls:
            if call not in builtin_functions and len(call) > 1:
                filtered_calls.append(call)
        
        return list(set(filtered_calls))  # Remove duplicates
    
    def detect_security_vulnerabilities(self, source_file) -> List[CodeIssue]:
        """Detect basic security vulnerability patterns"""
        issues = []
        if not source_file.source:
            return issues
        
        lines = source_file.source.split('\n')
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # SQL Injection risks
            if re.search(r'execute\s*\(\s*["\'].*%.*["\']', stripped, re.IGNORECASE):
                issues.append(CodeIssue(
                    issue_type=IssueType.SQL_INJECTION_RISK,
                    severity=IssueSeverity.CRITICAL,
                    message="Potential SQL injection vulnerability",
                    filepath=source_file.file_path,
                    line_number=i + 1,
                    column_number=0,
                    context={"line": stripped},
                    suggested_fix="Use parameterized queries instead of string formatting"
                ))
            
            # Hardcoded secrets
            secret_patterns = [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']',
            ]
            
            for pattern in secret_patterns:
                if re.search(pattern, stripped, re.IGNORECASE):
                    issues.append(CodeIssue(
                        issue_type=IssueType.HARDCODED_CONFIG,
                        severity=IssueSeverity.MAJOR,
                        message="Hardcoded secret detected",
                        filepath=source_file.file_path,
                        line_number=i + 1,
                        column_number=0,
                        context={"line": stripped},
                        suggested_fix="Use environment variables or secure configuration"
                    ))
            
            # Unsafe eval/exec usage
            if re.search(r'\b(eval|exec)\s*\(', stripped):
                issues.append(CodeIssue(
                    issue_type=IssueType.UNSAFE_ASSERTION,
                    severity=IssueSeverity.CRITICAL,
                    message="Unsafe eval/exec usage detected",
                    filepath=source_file.file_path,
                    line_number=i + 1,
                    column_number=0,
                    context={"line": stripped},
                    suggested_fix="Avoid eval/exec or validate input thoroughly"
                ))
        
        return issues
    
    def analyze_test_coverage(self, codebase) -> Dict[str, Any]:
        """Analyze test coverage and patterns"""
        test_files = []
        test_functions = []
        tested_functions = set()
        
        for source_file in codebase.files:
            file_path = source_file.file_path.lower()
            
            # Identify test files
            if ('test' in file_path or 
                file_path.endswith('_test.py') or 
                file_path.startswith('test_') or
                '/tests/' in file_path):
                test_files.append(source_file.file_path)
                
                # Extract test functions
                for symbol in source_file.symbols:
                    if hasattr(symbol, 'name') and symbol.name.startswith('test_'):
                        test_functions.append(symbol.name)
                        
                        # Try to identify what function is being tested
                        tested_func = symbol.name.replace('test_', '')
                        tested_functions.add(tested_func)
        
        # Calculate coverage metrics
        total_functions = len([f for file in codebase.files for f in file.symbols 
                             if hasattr(f, '__class__') and 'Function' in str(f.__class__)])
        
        coverage_ratio = len(tested_functions) / max(total_functions, 1)
        
        return {
            "test_files": test_files,
            "test_functions": test_functions,
            "tested_functions": list(tested_functions),
            "total_test_files": len(test_files),
            "total_test_functions": len(test_functions),
            "estimated_coverage": coverage_ratio,
            "untested_functions": total_functions - len(tested_functions)
        }
    
    def enhanced_repository_structure_analysis(self, codebase, issues) -> Dict[str, Any]:
        """Enhanced repository structure analysis"""
        structure = {
            "directories": defaultdict(list),
            "file_types": defaultdict(int),
            "architecture_patterns": [],
            "hotspots": [],
            "organization_score": 0.0
        }
        
        # Analyze directory structure
        for source_file in codebase.files:
            path_parts = source_file.file_path.split('/')
            directory = '/'.join(path_parts[:-1]) if len(path_parts) > 1 else 'root'
            filename = path_parts[-1]
            
            structure["directories"][directory].append(filename)
            
            # File type analysis
            extension = filename.split('.')[-1] if '.' in filename else 'no_extension'
            structure["file_types"][extension] += 1
        
        # Detect architecture patterns
        directories = list(structure["directories"].keys())
        
        # MVC pattern detection
        mvc_indicators = ['models', 'views', 'controllers', 'templates']
        if any(indicator in ' '.join(directories) for indicator in mvc_indicators):
            structure["architecture_patterns"].append("MVC")
        
        # Microservices pattern detection
        service_indicators = ['services', 'api', 'handlers', 'endpoints']
        if any(indicator in ' '.join(directories) for indicator in service_indicators):
            structure["architecture_patterns"].append("Service-Oriented")
        
        # Test organization
        test_indicators = ['tests', 'test', 'spec']
        if any(indicator in ' '.join(directories) for indicator in test_indicators):
            structure["architecture_patterns"].append("Test-Organized")
        
        # Calculate issue hotspots
        file_issue_counts = defaultdict(int)
        for issue in issues:
            file_issue_counts[issue.filepath] += 1
        
        structure["hotspots"] = [
            {"filepath": filepath, "issue_count": count}
            for filepath, count in sorted(file_issue_counts.items(), 
                                        key=lambda x: x[1], reverse=True)[:10]
        ]
        
        # Organization score (0-100)
        score = 50  # Base score
        
        # Bonus for good organization
        if len(structure["architecture_patterns"]) > 0:
            score += 20
        if "Test-Organized" in structure["architecture_patterns"]:
            score += 15
        if len(structure["directories"]) > 1:  # Not everything in root
            score += 10
        
        # Penalty for poor organization
        if len(structure["directories"]["root"]) > 10:  # Too many files in root
            score -= 15
        if structure["file_types"].get("no_extension", 0) > 5:  # Too many files without extensions
            score -= 10
        
        structure["organization_score"] = max(0, min(100, score))
        
        return structure
    
    def generate_enhanced_automated_resolutions(self, issues) -> List[AutomatedResolution]:
        """Generate enhanced automated resolutions"""
        resolutions = []
        
        for issue in issues:
            resolution = None
            
            if issue.issue_type == IssueType.HARDCODED_CONFIG:
                resolution = AutomatedResolution(
                    resolution_type="extract_to_config",
                    description="Extract hardcoded value to configuration",
                    original_code=issue.context.get("line", ""),
                    fixed_code="# TODO: Move to environment variable or config file",
                    confidence=0.7,
                    file_path=issue.filepath,
                    line_number=issue.line_number,
                    is_safe=False,
                    requires_validation=True
                )
            
            elif issue.issue_type == IssueType.MISSING_RETURN:
                resolution = AutomatedResolution(
                    resolution_type="add_return_none",
                    description="Add explicit return None statement",
                    original_code="",
                    fixed_code="    return None",
                    confidence=0.8,
                    file_path=issue.filepath,
                    line_number=issue.line_number,
                    is_safe=True,
                    requires_validation=False
                )
            
            elif issue.issue_type == IssueType.UNDEFINED_VARIABLE:
                var_name = issue.context.get("variable", "unknown")
                resolution = AutomatedResolution(
                    resolution_type="add_variable_definition",
                    description=f"Add definition for variable '{var_name}'",
                    original_code="",
                    fixed_code=f"    {var_name} = None  # TODO: Define appropriate value",
                    confidence=0.6,
                    file_path=issue.filepath,
                    line_number=max(1, issue.line_number - 1),
                    is_safe=False,
                    requires_validation=True
                )
            
            if resolution:
                resolutions.append(resolution)
        
        return resolutions


def enhance_codebase_analyzer(analyzer_class):
    """Enhance the CodebaseAnalyzer class with additional methods"""
    
    def enhanced_detect_undefined_variables(self):
        """Enhanced undefined variable detection"""
        enhancer = EnhancedAnalysisEngine(self.codebase)
        for source_file in self.codebase.files:
            new_issues = enhancer.enhanced_undefined_variable_detection(source_file)
            self.issues.extend(new_issues)
    
    def enhanced_detect_missing_returns(self):
        """Enhanced missing return detection"""
        enhancer = EnhancedAnalysisEngine(self.codebase)
        for source_file in self.codebase.files:
            new_issues = enhancer.enhanced_missing_returns_detection(source_file)
            self.issues.extend(new_issues)
    
    def enhanced_calculate_nesting_depth(self, source: str) -> int:
        """Enhanced nesting depth calculation"""
        enhancer = EnhancedAnalysisEngine(self.codebase)
        return enhancer.enhanced_nesting_depth_calculation(source)
    
    def enhanced_extract_function_calls(self, function) -> List[str]:
        """Enhanced function call extraction"""
        enhancer = EnhancedAnalysisEngine(self.codebase)
        return enhancer.extract_function_calls(function)
    
    def detect_security_vulnerabilities(self):
        """Detect security vulnerabilities"""
        enhancer = EnhancedAnalysisEngine(self.codebase)
        for source_file in self.codebase.files:
            new_issues = enhancer.detect_security_vulnerabilities(source_file)
            self.issues.extend(new_issues)
    
    def analyze_test_coverage(self):
        """Analyze test coverage"""
        enhancer = EnhancedAnalysisEngine(self.codebase)
        return enhancer.analyze_test_coverage(self.codebase)
    
    def enhanced_repository_structure_analysis(self, issues):
        """Enhanced repository structure analysis"""
        enhancer = EnhancedAnalysisEngine(self.codebase)
        return enhancer.enhanced_repository_structure_analysis(self.codebase, issues)
    
    def generate_enhanced_automated_resolutions(self, issues):
        """Generate enhanced automated resolutions"""
        enhancer = EnhancedAnalysisEngine(self.codebase)
        return enhancer.generate_enhanced_automated_resolutions(issues)
    
    # Bind methods to the analyzer class
    analyzer_class._detect_undefined_variables = enhanced_detect_undefined_variables
    analyzer_class._detect_missing_returns = enhanced_detect_missing_returns
    analyzer_class._calculate_nesting_depth = enhanced_calculate_nesting_depth
    analyzer_class._extract_function_calls = enhanced_extract_function_calls
    analyzer_class.detect_security_vulnerabilities = detect_security_vulnerabilities
    analyzer_class.analyze_test_coverage = analyze_test_coverage
    analyzer_class.enhanced_repository_structure_analysis = enhanced_repository_structure_analysis
    analyzer_class.generate_enhanced_automated_resolutions = generate_enhanced_automated_resolutions
    
    return analyzer_class
