"""
Advanced Issue Detection Framework
Comprehensive error detection, context analysis, and automated resolution
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import re
import ast
from collections import defaultdict, Counter


class IssueSeverity(Enum):
    CRITICAL = "critical"  # ⚠️
    MAJOR = "major"       # 👉
    MINOR = "minor"       # 🔍
    INFO = "info"         # ℹ️


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
    MISSING_REQUIRED_PARAMETER = "missing_required_parameter"
    UNUSED_PARAMETER = "unused_parameter"
    
    # Exception Handling
    IMPROPER_EXCEPTION_HANDLING = "improper_exception_handling"
    MISSING_ERROR_HANDLING = "missing_error_handling"
    UNSAFE_ASSERTION = "unsafe_assertion"
    RESOURCE_LEAK = "resource_leak"
    MEMORY_MANAGEMENT = "memory_management"
    
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
    LINE_LENGTH_VIOLATION = "line_length_violation"
    IMPORT_ORGANIZATION = "import_organization"
    
    # Runtime Risks
    DIVISION_BY_ZERO = "division_by_zero"
    ARRAY_INDEX_OUT_OF_BOUNDS = "array_index_out_of_bounds"
    INFINITE_LOOP = "infinite_loop"
    STACK_OVERFLOW = "stack_overflow"
    CONCURRENCY_ISSUE = "concurrency_issue"
    
    # Dead Code
    DEAD_FUNCTION = "dead_function"
    DEAD_VARIABLE = "dead_variable"
    DEAD_CLASS = "dead_class"
    DEAD_IMPORT = "dead_import"


@dataclass
class AutomatedResolution:
    """Represents an automated fix that can be applied"""
    resolution_type: str
    description: str
    original_code: str
    fixed_code: str
    confidence: float
    file_path: str
    line_number: int
    is_safe: bool = True
    requires_validation: bool = False


@dataclass
class CodeIssue:
    """Represents a code issue with full context and automated resolution"""
    issue_type: IssueType
    severity: IssueSeverity
    message: str
    filepath: str
    line_number: int
    column_number: int
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    suggested_fix: Optional[str] = None
    blast_radius: List[str] = field(default_factory=list)
    automated_resolution: Optional[AutomatedResolution] = None
    related_issues: List[str] = field(default_factory=list)
    impact_score: float = 0.0
    fix_effort: str = "low"  # low, medium, high


class AdvancedIssueDetector:
    """Advanced issue detection with automated resolution capabilities"""
    
    def __init__(self, codebase):
        self.codebase = codebase
        self.issues = []
        self.automated_resolutions = []
        self.import_resolver = ImportResolver(codebase)
        
    def detect_all_issues(self) -> List[CodeIssue]:
        """Detect all types of issues with automated resolutions"""
        print("🔍 Starting comprehensive issue detection...")
        
        # Implementation errors
        self._detect_null_references()
        self._detect_type_mismatches()
        self._detect_undefined_variables()
        self._detect_missing_returns()
        self._detect_unreachable_code()
        
        # Function issues
        self._detect_function_issues()
        self._detect_parameter_issues()
        
        # Exception handling
        self._detect_exception_handling_issues()
        self._detect_resource_leaks()
        
        # Code quality
        self._detect_code_quality_issues()
        self._detect_magic_numbers()
        
        # Formatting & style
        self._detect_style_issues()
        self._detect_import_issues()
        
        # Runtime risks
        self._detect_runtime_risks()
        
        # Dead code
        self._detect_dead_code()
        
        # Apply automated resolutions
        self._apply_automated_resolutions()
        
        print(f"✅ Detected {len(self.issues)} issues with {len(self.automated_resolutions)} automated resolutions")
        return self.issues
    
    def _detect_null_references(self):
        """Detect potential null reference issues"""
        for file in self.codebase.files:
            if hasattr(file, 'source'):
                lines = file.source.split('\n')
                for i, line in enumerate(lines):
                    # Check for .get() without null checks
                    if '.get(' in line and 'if' not in line and 'or' not in line:
                        issue = CodeIssue(
                            issue_type=IssueType.NULL_REFERENCE,
                            severity=IssueSeverity.MAJOR,
                            message="Potential null reference: .get() without null check",
                            filepath=file.filepath,
                            line_number=i + 1,
                            column_number=line.find('.get('),
                            context={"line": line.strip()},
                            suggested_fix="Add null check or provide default value",
                            impact_score=7.5
                        )
                        
                        # Automated resolution
                        if '.get(' in line:
                            fixed_line = self._fix_null_reference(line)
                            issue.automated_resolution = AutomatedResolution(
                                resolution_type="null_check_addition",
                                description="Add null check with default value",
                                original_code=line.strip(),
                                fixed_code=fixed_line,
                                confidence=0.85,
                                file_path=file.filepath,
                                line_number=i + 1
                            )
                        
                        self.issues.append(issue)
    
    def _fix_null_reference(self, line: str) -> str:
        """Automatically fix null reference issues"""
        # Simple pattern: obj.get('key') -> obj.get('key', default_value)
        pattern = r'(\w+)\.get\([\'"]([^\'"]+)[\'"]\)'
        match = re.search(pattern, line)
        if match:
            obj, key = match.groups()
            return line.replace(f"{obj}.get('{key}')", f"{obj}.get('{key}', None)")
        return line
    
    def _detect_import_issues(self):
        """Detect and automatically resolve import issues"""
        for file in self.codebase.files:
            if hasattr(file, 'imports'):
                # Detect unused imports
                unused_imports = self.import_resolver.find_unused_imports(file)
                for unused_import in unused_imports:
                    issue = CodeIssue(
                        issue_type=IssueType.DEAD_IMPORT,
                        severity=IssueSeverity.MINOR,
                        message=f"Unused import: {unused_import['name']}",
                        filepath=file.filepath,
                        line_number=unused_import.get('line', 1),
                        column_number=0,
                        context={"import_name": unused_import['name']},
                        impact_score=2.0
                    )
                    
                    # Automated resolution - remove unused import
                    issue.automated_resolution = AutomatedResolution(
                        resolution_type="remove_unused_import",
                        description=f"Remove unused import: {unused_import['name']}",
                        original_code=unused_import.get('source', ''),
                        fixed_code="",  # Remove the line
                        confidence=0.95,
                        file_path=file.filepath,
                        line_number=unused_import.get('line', 1),
                        is_safe=True
                    )
                    
                    self.issues.append(issue)
                
                # Detect missing imports
                missing_imports = self.import_resolver.find_missing_imports(file)
                for missing_import in missing_imports:
                    issue = CodeIssue(
                        issue_type=IssueType.UNDEFINED_VARIABLE,
                        severity=IssueSeverity.CRITICAL,
                        message=f"Missing import for: {missing_import['symbol']}",
                        filepath=file.filepath,
                        line_number=missing_import.get('line', 1),
                        column_number=0,
                        context={"symbol": missing_import['symbol']},
                        impact_score=9.0
                    )
                    
                    # Automated resolution - add missing import
                    resolved_import = self.import_resolver.resolve_import(missing_import['symbol'])
                    if resolved_import:
                        issue.automated_resolution = AutomatedResolution(
                            resolution_type="add_missing_import",
                            description=f"Add import: {resolved_import}",
                            original_code="",
                            fixed_code=resolved_import,
                            confidence=0.90,
                            file_path=file.filepath,
                            line_number=1,  # Add at top of file
                            is_safe=True
                        )
                    
                    self.issues.append(issue)
    
    def _detect_function_issues(self):
        """Detect function-related issues"""
        for function in self.codebase.functions:
            # Long function detection
            if hasattr(function, 'start_point') and hasattr(function, 'end_point'):
                line_count = function.end_point[0] - function.start_point[0]
                if line_count > 50:
                    issue = CodeIssue(
                        issue_type=IssueType.LONG_FUNCTION,
                        severity=IssueSeverity.MAJOR,
                        message=f"Function '{function.name}' is too long ({line_count} lines)",
                        filepath=function.filepath,
                        line_number=function.start_point[0],
                        column_number=0,
                        function_name=function.name,
                        context={"line_count": line_count},
                        suggested_fix="Break down into smaller functions",
                        impact_score=6.0,
                        fix_effort="high"
                    )
                    self.issues.append(issue)
            
            # Missing documentation
            if hasattr(function, 'source'):
                if not ('"""' in function.source or "'''" in function.source):
                    issue = CodeIssue(
                        issue_type=IssueType.MISSING_DOCUMENTATION,
                        severity=IssueSeverity.MINOR,
                        message=f"Function '{function.name}' lacks documentation",
                        filepath=function.filepath,
                        line_number=function.start_point[0] if hasattr(function, 'start_point') else 0,
                        column_number=0,
                        function_name=function.name,
                        impact_score=3.0
                    )
                    
                    # Automated resolution - add basic docstring
                    docstring = self._generate_docstring(function)
                    issue.automated_resolution = AutomatedResolution(
                        resolution_type="add_docstring",
                        description=f"Add docstring to function '{function.name}'",
                        original_code="",
                        fixed_code=docstring,
                        confidence=0.80,
                        file_path=function.filepath,
                        line_number=function.start_point[0] + 1 if hasattr(function, 'start_point') else 1
                    )
                    
                    self.issues.append(issue)
    
    def _generate_docstring(self, function) -> str:
        """Generate basic docstring for a function"""
        params = ""
        if hasattr(function, 'parameters') and function.parameters:
            param_list = [f"        {param.name}: Description of {param.name}" 
                         for param in function.parameters if hasattr(param, 'name')]
            if param_list:
                params = f"\n    Args:\n" + "\n".join(param_list)
        
        return f'    """{function.name} function.\n    \n    Brief description of what this function does.{params}\n    \n    Returns:\n        Description of return value\n    """'
    
    def _detect_magic_numbers(self):
        """Detect magic numbers in code"""
        for file in self.codebase.files:
            if hasattr(file, 'source'):
                lines = file.source.split('\n')
                for i, line in enumerate(lines):
                    # Find numeric literals (excluding 0, 1, -1)
                    numbers = re.findall(r'\b(?<![\w.])\d{2,}\b(?![\w.])', line)
                    for number in numbers:
                        if int(number) not in [0, 1, -1, 100]:  # Common acceptable numbers
                            issue = CodeIssue(
                                issue_type=IssueType.MAGIC_NUMBER,
                                severity=IssueSeverity.MINOR,
                                message=f"Magic number detected: {number}",
                                filepath=file.filepath,
                                line_number=i + 1,
                                column_number=line.find(number),
                                context={"number": number, "line": line.strip()},
                                suggested_fix=f"Replace {number} with named constant",
                                impact_score=2.5
                            )
                            self.issues.append(issue)
    
    def _detect_runtime_risks(self):
        """Detect potential runtime risks"""
        for file in self.codebase.files:
            if hasattr(file, 'source'):
                lines = file.source.split('\n')
                for i, line in enumerate(lines):
                    # Division by zero risk
                    if '/' in line and 'if' not in line:
                        # Simple heuristic for potential division by zero
                        if re.search(r'/\s*\w+(?!\w)', line):
                            issue = CodeIssue(
                                issue_type=IssueType.DIVISION_BY_ZERO,
                                severity=IssueSeverity.MAJOR,
                                message="Potential division by zero",
                                filepath=file.filepath,
                                line_number=i + 1,
                                column_number=line.find('/'),
                                context={"line": line.strip()},
                                suggested_fix="Add zero check before division",
                                impact_score=8.0
                            )
                            self.issues.append(issue)
    
    def _detect_dead_code(self):
        """Detect dead code with automated removal suggestions"""
        # Find unused functions
        for function in self.codebase.functions:
            if hasattr(function, 'usages') and len(function.usages) == 0:
                # Check if it's not an entry point
                if not self._is_entry_point(function):
                    issue = CodeIssue(
                        issue_type=IssueType.DEAD_FUNCTION,
                        severity=IssueSeverity.MINOR,
                        message=f"Unused function: {function.name}",
                        filepath=function.filepath,
                        line_number=function.start_point[0] if hasattr(function, 'start_point') else 0,
                        column_number=0,
                        function_name=function.name,
                        context={"reason": "No usages found"},
                        impact_score=1.0
                    )
                    
                    # Automated resolution - mark for removal
                    issue.automated_resolution = AutomatedResolution(
                        resolution_type="remove_dead_function",
                        description=f"Remove unused function '{function.name}'",
                        original_code=function.source if hasattr(function, 'source') else "",
                        fixed_code="",
                        confidence=0.75,
                        file_path=function.filepath,
                        line_number=function.start_point[0] if hasattr(function, 'start_point') else 0,
                        requires_validation=True
                    )
                    
                    self.issues.append(issue)
    
    def _is_entry_point(self, function) -> bool:
        """Check if function is an entry point"""
        entry_patterns = ['main', '__main__', 'run', 'start', 'execute', 'init', 'setup', 'app', 'server', 'cli']
        return any(pattern in function.name.lower() for pattern in entry_patterns)
    
    def _detect_code_quality_issues(self):
        """Detect general code quality issues"""
        pass  # Placeholder for additional quality checks
    
    def _detect_type_mismatches(self):
        """Detect type mismatch issues"""
        pass  # Placeholder for type analysis
    
    def _detect_undefined_variables(self):
        """Detect undefined variable usage"""
        pass  # Placeholder for variable analysis
    
    def _detect_missing_returns(self):
        """Detect missing return statements"""
        pass  # Placeholder for return analysis
    
    def _detect_unreachable_code(self):
        """Detect unreachable code"""
        pass  # Placeholder for reachability analysis
    
    def _detect_parameter_issues(self):
        """Detect parameter-related issues"""
        pass  # Placeholder for parameter analysis
    
    def _detect_exception_handling_issues(self):
        """Detect exception handling problems"""
        pass  # Placeholder for exception analysis
    
    def _detect_resource_leaks(self):
        """Detect resource leak issues"""
        pass  # Placeholder for resource analysis
    
    def _detect_style_issues(self):
        """Detect style and formatting issues"""
        pass  # Placeholder for style analysis
    
    def _apply_automated_resolutions(self):
        """Apply automated resolutions where safe"""
        for issue in self.issues:
            if issue.automated_resolution and issue.automated_resolution.is_safe:
                self.automated_resolutions.append(issue.automated_resolution)


class ImportResolver:
    """Automated import resolution system"""
    
    def __init__(self, codebase):
        self.codebase = codebase
        self.import_map = self._build_import_map()
        self.symbol_map = self._build_symbol_map()
    
    def _build_import_map(self) -> Dict[str, str]:
        """Build map of available imports"""
        import_map = {}
        for file in self.codebase.files:
            if hasattr(file, 'imports'):
                for imp in file.imports:
                    if hasattr(imp, 'module_name'):
                        import_map[imp.module_name] = file.filepath
        return import_map
    
    def _build_symbol_map(self) -> Dict[str, str]:
        """Build map of available symbols"""
        symbol_map = {}
        for file in self.codebase.files:
            # Map functions
            if hasattr(file, 'functions'):
                for func in file.functions:
                    symbol_map[func.name] = file.filepath
            
            # Map classes
            if hasattr(file, 'classes'):
                for cls in file.classes:
                    symbol_map[cls.name] = file.filepath
        
        return symbol_map
    
    def find_unused_imports(self, file) -> List[Dict[str, Any]]:
        """Find unused imports in a file"""
        unused = []
        if hasattr(file, 'imports') and hasattr(file, 'source'):
            for imp in file.imports:
                if hasattr(imp, 'module_name'):
                    # Simple check if import is used in source
                    if imp.module_name not in file.source:
                        unused.append({
                            'name': imp.module_name,
                            'source': str(imp),
                            'line': getattr(imp, 'line_number', 1)
                        })
        return unused
    
    def find_missing_imports(self, file) -> List[Dict[str, Any]]:
        """Find missing imports in a file"""
        missing = []
        if hasattr(file, 'source'):
            # Find undefined symbols that could be imports
            for symbol, filepath in self.symbol_map.items():
                if symbol in file.source and filepath != file.filepath:
                    # Check if already imported
                    if not self._is_imported(file, symbol):
                        missing.append({
                            'symbol': symbol,
                            'source_file': filepath,
                            'line': 1  # Will be added at top
                        })
        return missing
    
    def _is_imported(self, file, symbol: str) -> bool:
        """Check if symbol is already imported"""
        if hasattr(file, 'imports'):
            for imp in file.imports:
                if hasattr(imp, 'module_name') and symbol in str(imp):
                    return True
        return False
    
    def resolve_import(self, symbol: str) -> Optional[str]:
        """Resolve the correct import statement for a symbol"""
        if symbol in self.symbol_map:
            source_file = self.symbol_map[symbol]
            # Convert file path to import statement
            import_path = source_file.replace('/', '.').replace('.py', '')
            return f"from {import_path} import {symbol}"
        return None

