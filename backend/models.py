"""
Codebase Analytics - Data Models and Structures
Comprehensive data models for codebase analysis, issue detection, and metrics
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel


# ============================================================================
# ISSUE DETECTION ENUMS AND TYPES
# ============================================================================

class IssueSeverity(Enum):
    """Severity levels for detected issues"""
    CRITICAL = "critical"  # ⚠️
    MAJOR = "major"       # 👉
    MINOR = "minor"       # 🔍
    INFO = "info"         # ℹ️


class IssueType(Enum):
    """Comprehensive issue type classification"""
    
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
    
    # Additional Issue Types
    LONG_PARAMETER_LIST = "long_parameter_list"
    BARE_EXCEPT = "bare_except"
    EMPTY_EXCEPT = "empty_except"
    TODO_COMMENT = "todo_comment"
    PRINT_STATEMENT = "print_statement"
    HARDCODED_CONFIG = "hardcoded_config"
    TRAILING_WHITESPACE = "trailing_whitespace"
    MIXED_INDENTATION = "mixed_indentation"
    EXCESSIVE_BLANK_LINES = "excessive_blank_lines"


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

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


@dataclass
class FunctionContext:
    """Complete context for a function with all relationships"""
    name: str
    filepath: str
    line_start: int
    line_end: int
    source: str
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: Optional[str] = None
    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    usages: List[Dict[str, Any]] = field(default_factory=list)
    function_calls: List[str] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)
    class_name: Optional[str] = None
    max_call_chain: List[str] = field(default_factory=list)
    issues: List[Any] = field(default_factory=list)
    complexity_metrics: Dict[str, Any] = field(default_factory=dict)
    is_entry_point: bool = False
    is_dead_code: bool = False
    call_depth: int = 0
    fan_in: int = 0  # Number of functions calling this one
    fan_out: int = 0  # Number of functions this one calls
    coupling_score: float = 0.0
    cohesion_score: float = 0.0


@dataclass
class AnalysisResults:
    """Structured analysis results for API consumption"""
    
    # Basic Statistics
    total_files: int
    total_functions: int
    total_classes: int
    total_lines_of_code: int
    
    # Issues Analysis
    issues: List[CodeIssue] = field(default_factory=list)
    issues_by_severity: Dict[str, int] = field(default_factory=dict)
    issues_by_type: Dict[str, int] = field(default_factory=dict)
    automated_resolutions: List[AutomatedResolution] = field(default_factory=list)
    
    # Function Analysis
    function_contexts: Dict[str, FunctionContext] = field(default_factory=dict)
    most_important_functions: List[Dict[str, Any]] = field(default_factory=list)
    entry_points: List[str] = field(default_factory=list)
    dead_functions: List[str] = field(default_factory=list)
    
    # Quality Metrics
    halstead_metrics: Dict[str, Any] = field(default_factory=dict)
    complexity_metrics: Dict[str, Any] = field(default_factory=dict)
    maintainability_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Graph Analysis
    call_graph_metrics: Dict[str, Any] = field(default_factory=dict)
    dependency_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Health Assessment
    health_score: float = 0.0
    health_grade: str = "F"
    risk_level: str = "high"
    technical_debt_hours: float = 0.0
    
    # Repository Structure
    repository_structure: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# API REQUEST/RESPONSE MODELS
# ============================================================================

class RepoRequest(BaseModel):
    """Request model for repository analysis"""
    repo_url: str


class CodebaseAnalysisRequest(BaseModel):
    """Request model for comprehensive codebase analysis"""
    repo_url: str
    analysis_depth: str = "comprehensive"
    include_automated_fixes: bool = True
    include_health_metrics: bool = True


class CodebaseAnalysisResponse(BaseModel):
    """Response model for codebase analysis"""
    success: bool
    analysis_results: Dict[str, Any]
    health_dashboard: Optional[Dict[str, Any]] = None
    processing_time: float
    repo_url: str
    analysis_timestamp: str
    features_analyzed: List[str]
    error_message: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str
    timestamp: str
    version: str = "1.0.0"


class RootResponse(BaseModel):
    """Response model for root endpoint"""
    message: str
    description: str
    endpoints: Dict[str, str]
    features: List[str]
    version: str = "1.0.0"


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

class AnalysisConfig:
    """Configuration constants for analysis"""
    
    # Issue Detection Thresholds
    LONG_FUNCTION_THRESHOLD = 50
    MAGIC_NUMBER_EXCLUSIONS = {0, 1, -1, 100}
    MAX_CALL_DEPTH = 10
    
    # Health Scoring Weights
    CRITICAL_ISSUE_WEIGHT = 10.0
    MAJOR_ISSUE_WEIGHT = 5.0
    MINOR_ISSUE_WEIGHT = 1.0
    INFO_ISSUE_WEIGHT = 0.1
    
    # Technical Debt Estimation (hours)
    CRITICAL_ISSUE_HOURS = 4.0
    MAJOR_ISSUE_HOURS = 2.0
    MINOR_ISSUE_HOURS = 0.5
    INFO_ISSUE_HOURS = 0.1
    
    # Entry Point Patterns
    ENTRY_POINT_PATTERNS = [
        'main', '__main__', 'run', 'start', 'execute', 
        'init', 'setup', 'app', 'server', 'cli'
    ]
    
    # Health Grade Thresholds
    HEALTH_GRADES = {
        90: "A+",
        85: "A",
        80: "A-",
        75: "B+",
        70: "B",
        65: "B-",
        60: "C+",
        55: "C",
        50: "C-",
        45: "D+",
        40: "D",
        35: "D-",
        0: "F"
    }


# ============================================================================
# UTILITY TYPES
# ============================================================================

# Type aliases for better code readability
FilePathType = str
FunctionNameType = str
ClassNameType = str
MetricsDict = Dict[str, Any]
IssuesList = List[CodeIssue]
ResolutionsList = List[AutomatedResolution]

