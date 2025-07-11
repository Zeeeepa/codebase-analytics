"""
Comprehensive Data Models and Enumerations for Codebase Analysis API
Consolidated from all analysis systems with enhanced functionality
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from datetime import datetime
from dataclasses import dataclass


# Enums for issue severity and types
class IssueSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueType(str, Enum):
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
    
    # Legacy types for compatibility
    SYNTAX_ERROR = "syntax_error"
    TYPE_ERROR = "type_error"
    SECURITY_VULNERABILITY = "security_vulnerability"
    CODE_SMELL = "code_smell"
    PERFORMANCE_ISSUE = "performance_issue"
    MAINTAINABILITY_ISSUE = "maintainability_issue"
    COMPLEXITY_ISSUE = "complexity_issue"
    DEPENDENCY_ISSUE = "dependency_issue"


# Data models for API requests and responses
class CodeIssue(BaseModel):
    id: str = Field(..., description="Unique identifier for the issue")
    type: IssueType
    severity: IssueSeverity
    file_path: str
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    message: str
    description: str
    context: Dict[str, Any] = Field(default_factory=dict)
    related_symbols: List[str] = Field(default_factory=list)
    affected_functions: List[str] = Field(default_factory=list)
    affected_classes: List[str] = Field(default_factory=list)
    fix_suggestions: List[str] = Field(default_factory=list)


class EntryPoint(BaseModel):
    type: str  # "main", "cli", "web_endpoint", "test", "script"
    file_path: str
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    line_number: Optional[int] = None
    description: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    dependencies: List[str] = Field(default_factory=list)


class CriticalFile(BaseModel):
    file_path: str
    importance_score: float = Field(..., ge=0.0, le=100.0)
    reasons: List[str]
    metrics: Dict[str, Any] = Field(default_factory=dict)
    dependencies_count: int = 0
    dependents_count: int = 0
    complexity_score: float = 0.0
    lines_of_code: int = 0


class DependencyNode(BaseModel):
    name: str
    type: str  # "file", "function", "class", "module"
    file_path: str
    dependencies: List[str] = Field(default_factory=list)
    dependents: List[str] = Field(default_factory=list)
    centrality_score: float = 0.0


class RepoRequest(BaseModel):
    repo_url: str


class DetailedAnalysisRequest(BaseModel):
    repo_url: str
    include_issues: bool = Field(default=True, description="Include code issue analysis")
    include_entrypoints: bool = Field(default=True, description="Include entrypoint detection")
    include_critical_files: bool = Field(default=True, description="Include critical file analysis")
    include_dependency_graph: bool = Field(default=True, description="Include dependency graph analysis")
    max_issues: int = Field(default=100, description="Maximum number of issues to return")


class FileAnalysisRequest(BaseModel):
    repo_url: str
    file_path: str


# New comprehensive analysis models
class CodebaseAnalysisRequest(BaseModel):
    repo_url: str
    include_issues: bool = Field(default=True, description="Include comprehensive issue detection")
    include_entry_points: bool = Field(default=True, description="Include entry point detection")
    include_critical_files: bool = Field(default=True, description="Include critical file identification")
    include_metrics: bool = Field(default=True, description="Include function and class metrics")
    include_dependency_graph: bool = Field(default=True, description="Include dependency graph analysis")
    max_issues: int = Field(default=200, description="Maximum number of issues to return")
    severity_filter: Optional[List[IssueSeverity]] = Field(default=None, description="Filter issues by severity")
    file_extensions: Optional[List[str]] = Field(default=None, description="Filter files by extensions")


class AnalysisSummary(BaseModel):
    total_issues: int
    issues_by_severity: Dict[str, int]
    issues_by_type: Dict[str, int]
    files_with_issues: int
    critical_issues_count: int
    high_priority_issues_count: int
    total_files: int
    total_functions: int
    total_classes: int
    total_lines_of_code: int
    average_complexity: float
    average_maintainability: float
    entry_points_count: int
    critical_files_count: int


class FunctionMetrics(BaseModel):
    function_name: str
    file_path: str
    line_number: Optional[int] = None
    cyclomatic_complexity: float = 0.0
    maintainability_index: float = 0.0
    lines_of_code: int = 0
    halstead_volume: float = 0.0
    halstead_difficulty: float = 0.0
    halstead_effort: float = 0.0
    parameters_count: int = 0
    return_statements_count: int = 0
    importance_score: float = 0.0
    usage_frequency: int = 0


class ClassMetrics(BaseModel):
    class_name: str
    file_path: str
    line_number: Optional[int] = None
    methods_count: int = 0
    attributes_count: int = 0
    depth_of_inheritance: int = 0
    coupling_between_objects: int = 0
    lack_of_cohesion: float = 0.0
    lines_of_code: int = 0
    importance_score: float = 0.0


class FileMetrics(BaseModel):
    file_path: str
    lines_of_code: int = 0
    functions_count: int = 0
    classes_count: int = 0
    imports_count: int = 0
    complexity_score: float = 0.0
    maintainability_index: float = 0.0
    importance_score: float = 0.0


class CodebaseAnalysis(BaseModel):
    repo_url: str
    analysis_timestamp: datetime
    analysis_duration: float
    summary: AnalysisSummary
    issues: List[CodeIssue]
    issues_by_file: Dict[str, List[CodeIssue]]
    function_metrics: List[FunctionMetrics]
    class_metrics: List[ClassMetrics]
    file_metrics: List[FileMetrics]
    entry_points: List[EntryPoint]
    critical_files: List[CriticalFile]
    dependency_graph: Optional[Dict[str, Any]] = None
    repository_info: Dict[str, Any]
    analysis_config: Dict[str, Any]
    errors: List[str] = Field(default_factory=list)


class CodebaseAnalysisResponse(BaseModel):
    success: bool
    analysis: Optional[CodebaseAnalysis] = None
    error_message: Optional[str] = None
    processing_time: float


# Enhanced models from PR 96 & 97 - Automated Resolution System
class AutomatedResolution(BaseModel):
    issue_id: str
    resolution_type: str  # "fix", "refactor", "optimize", "remove"
    confidence: float = Field(..., ge=0.0, le=1.0)
    description: str
    fix_code: Optional[str] = None
    blast_radius: Dict[str, Any] = Field(default_factory=dict)
    validation_status: str = "pending"  # "pending", "validated", "failed"
    rollback_info: Optional[Dict[str, Any]] = None


class HealthMetrics(BaseModel):
    overall_score: float = Field(..., ge=0.0, le=100.0)
    maintainability_score: float = Field(..., ge=0.0, le=100.0)
    technical_debt_score: float = Field(..., ge=0.0, le=100.0)
    complexity_score: float = Field(..., ge=0.0, le=100.0)
    test_coverage_score: float = Field(default=0.0, ge=0.0, le=100.0)
    documentation_score: float = Field(default=0.0, ge=0.0, le=100.0)
    trend_direction: str = "neutral"  # "improving", "declining", "neutral"
    risk_level: str = "medium"  # "low", "medium", "high", "critical"


class FunctionContext(BaseModel):
    function_name: str
    file_path: str
    line_number: Optional[int] = None
    signature: str
    docstring: Optional[str] = None
    parameters: List[Dict[str, Any]] = Field(default_factory=list)
    return_type: Optional[str] = None
    calls_made: List[str] = Field(default_factory=list)
    called_by: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    complexity_metrics: Dict[str, float] = Field(default_factory=dict)
    usage_patterns: Dict[str, Any] = Field(default_factory=dict)
    blast_radius: Dict[str, Any] = Field(default_factory=dict)


class HalsteadMetrics(BaseModel):
    volume: float = 0.0
    difficulty: float = 0.0
    effort: float = 0.0
    time_to_program: float = 0.0
    bugs_delivered: float = 0.0
    operators: Dict[str, int] = Field(default_factory=dict)
    operands: Dict[str, int] = Field(default_factory=dict)
    unique_operators: int = 0
    unique_operands: int = 0
    total_operators: int = 0
    total_operands: int = 0


class GraphMetrics(BaseModel):
    betweenness_centrality: float = 0.0
    closeness_centrality: float = 0.0
    degree_centrality: float = 0.0
    eigenvector_centrality: float = 0.0
    clustering_coefficient: float = 0.0
    pagerank: float = 0.0


class DeadCodeAnalysis(BaseModel):
    unused_functions: List[str] = Field(default_factory=list)
    unused_classes: List[str] = Field(default_factory=list)
    unused_imports: List[str] = Field(default_factory=list)
    unused_variables: List[str] = Field(default_factory=list)
    dead_code_percentage: float = 0.0
    removal_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    blast_radius_analysis: Dict[str, Any] = Field(default_factory=dict)


class RepositoryStructure(BaseModel):
    total_files: int = 0
    files_by_type: Dict[str, int] = Field(default_factory=dict)
    directory_structure: Dict[str, Any] = Field(default_factory=dict)
    largest_files: List[Dict[str, Any]] = Field(default_factory=list)
    most_complex_files: List[Dict[str, Any]] = Field(default_factory=list)
    architectural_patterns: List[str] = Field(default_factory=list)


# Comprehensive Analysis Results (from PR 96)
@dataclass
class AnalysisResults:
    """Structured analysis results for API consumption"""
    
    # Basic Statistics
    total_files: int
    total_functions: int
    total_classes: int
    total_lines_of_code: int
    effective_lines_of_code: int
    
    # Issue Analysis
    total_issues: int
    issues_by_severity: Dict[str, int]
    issues_by_type: Dict[str, int]
    critical_issues: List[CodeIssue]
    automated_resolutions: List[AutomatedResolution]
    
    # Function Analysis
    most_important_functions: List[Dict[str, Any]]
    entry_points: List[Dict[str, Any]]
    function_contexts: Dict[str, FunctionContext]
    
    # Code Quality Metrics
    halstead_metrics: Dict[str, Any]
    complexity_metrics: Dict[str, Any]
    maintainability_score: float
    technical_debt_score: float
    
    # Graph Analysis
    call_graph: Dict[str, Any]
    dependency_graph: Dict[str, Any]
    graph_metrics: Dict[str, GraphMetrics]
    
    # Dead Code Analysis
    dead_code_analysis: DeadCodeAnalysis
    
    # Health Metrics
    health_metrics: HealthMetrics
    
    # Repository Structure
    repository_structure: RepositoryStructure
    
    # Analysis Metadata
    analysis_timestamp: datetime
    analysis_duration: float
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


# Enhanced Request Models
class ComprehensiveAnalysisRequest(BaseModel):
    repo_url: str
    include_issues: bool = Field(default=True, description="Include comprehensive issue detection")
    include_entry_points: bool = Field(default=True, description="Include entry point detection")
    include_critical_files: bool = Field(default=True, description="Include critical file identification")
    include_metrics: bool = Field(default=True, description="Include function and class metrics")
    include_dependency_graph: bool = Field(default=True, description="Include dependency graph analysis")
    include_dead_code_analysis: bool = Field(default=True, description="Include dead code detection")
    include_health_metrics: bool = Field(default=True, description="Include health assessment")
    include_automated_resolutions: bool = Field(default=True, description="Include automated fix suggestions")
    include_halstead_metrics: bool = Field(default=True, description="Include Halstead complexity metrics")
    include_graph_analysis: bool = Field(default=True, description="Include NetworkX graph analysis")
    max_issues: int = Field(default=200, description="Maximum number of issues to return")
    severity_filter: Optional[List[IssueSeverity]] = Field(default=None, description="Filter issues by severity")
    file_extensions: Optional[List[str]] = Field(default=None, description="Filter files by extensions")
    enable_caching: bool = Field(default=True, description="Enable analysis caching for performance")


class ComprehensiveAnalysisResponse(BaseModel):
    success: bool
    results: Optional[AnalysisResults] = None
    error_message: Optional[str] = None
    processing_time: float
    cache_hit: bool = False
    analysis_id: Optional[str] = None


# ============================================================================
# ADDITIONAL MODELS FROM BACKEND BRANCHES
# ============================================================================

class FunctionMetrics(BaseModel):
    """Comprehensive function metrics"""
    function_name: str
    file_path: str
    line_number: Optional[int] = None
    cyclomatic_complexity: int
    maintainability_index: float
    lines_of_code: int
    halstead_volume: float
    halstead_difficulty: float
    halstead_effort: float
    parameters_count: int
    return_statements_count: int = 0
    importance_score: float
    usage_frequency: int = 0


class ClassMetrics(BaseModel):
    """Comprehensive class metrics"""
    class_name: str
    file_path: str
    line_number: Optional[int] = None
    methods_count: int
    attributes_count: int
    depth_of_inheritance: int
    coupling_between_objects: int
    lack_of_cohesion: float
    lines_of_code: int
    importance_score: float


class FileMetrics(BaseModel):
    """Comprehensive file metrics"""
    file_path: str
    lines_of_code: int
    functions_count: int
    classes_count: int
    imports_count: int
    complexity_score: float
    maintainability_index: float
    importance_score: float


class AnalysisConfig(BaseModel):
    """Configuration for analysis parameters"""
    # Entry point patterns
    ENTRY_POINT_PATTERNS: List[str] = Field(default_factory=lambda: [
        'main', 'run', 'start', 'init', 'setup', 'launch', 'execute',
        'app', 'server', 'cli', 'command', 'handler', 'endpoint'
    ])
    
    # Issue detection thresholds
    LONG_FUNCTION_THRESHOLD: int = 50
    HIGH_COMPLEXITY_THRESHOLD: int = 10
    MAGIC_NUMBER_EXCLUSIONS: List[int] = Field(default_factory=lambda: [0, 1, -1, 2, 10, 100])
    
    # Health scoring weights
    CRITICAL_ISSUE_WEIGHT: int = 10
    MAJOR_ISSUE_WEIGHT: int = 5
    MINOR_ISSUE_WEIGHT: int = 2
    INFO_ISSUE_WEIGHT: int = 1
    
    # Technical debt estimation (hours)
    CRITICAL_ISSUE_HOURS: float = 8.0
    MAJOR_ISSUE_HOURS: float = 4.0
    MINOR_ISSUE_HOURS: float = 1.0
    INFO_ISSUE_HOURS: float = 0.25
    
    # Health grades
    HEALTH_GRADES: Dict[int, str] = Field(default_factory=lambda: {
        90: "A+",
        80: "A",
        70: "B",
        60: "C",
        50: "D",
        0: "F"
    })


class RepoRequest(BaseModel):
    """Simple repository request model"""
    repo_url: str
    branch: Optional[str] = "main"


class CodebaseAnalysisRequest(BaseModel):
    """Request model for codebase analysis"""
    repo_url: str
    analysis_type: str = "comprehensive"
    include_metrics: bool = True
    include_issues: bool = True
    include_health: bool = True
    max_issues: int = 100


class CodebaseAnalysisResponse(BaseModel):
    """Response model for codebase analysis"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: float
    timestamp: datetime


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: datetime
    version: str = "2.0.0"
    features: List[str] = Field(default_factory=lambda: [
        "comprehensive_analysis",
        "issue_detection", 
        "health_metrics",
        "automated_resolutions",
        "graph_analysis"
    ])


class RootResponse(BaseModel):
    """Root endpoint response model"""
    message: str
    version: str
    documentation: str = "/docs"
    health: str = "/health"
    endpoints: Dict[str, str]


# Enhanced issue types from backend analysis
class IssueTypeExtended(str, Enum):
    """Extended issue types from backend analysis"""
    SYNTAX_ERROR = "syntax_error"
    TYPE_ERROR = "type_error"
    SECURITY_VULNERABILITY = "security_vulnerability"
    CODE_SMELL = "code_smell"
    PERFORMANCE_ISSUE = "performance_issue"
    MAINTAINABILITY_ISSUE = "maintainability_issue"
    COMPLEXITY_ISSUE = "complexity_issue"
    DEPENDENCY_ISSUE = "dependency_issue"
    LONG_FUNCTION = "long_function"
    MISSING_DOCUMENTATION = "missing_documentation"
    INEFFICIENT_PATTERN = "inefficient_pattern"
    MAGIC_NUMBER = "magic_number"
    LINE_LENGTH_VIOLATION = "line_length_violation"
    DEAD_CODE = "dead_code"
    UNUSED_IMPORT = "unused_import"
    DUPLICATE_CODE = "duplicate_code"


class ResolutionType(str, Enum):
    """Types of automated resolutions"""
    EXTRACT_CONSTANT = "extract_constant"
    ADD_DOCSTRING = "add_docstring"
    BREAK_FUNCTION = "break_function"
    REMOVE_DEAD_CODE = "remove_dead_code"
    OPTIMIZE_IMPORT = "optimize_import"
    REFACTOR_COMPLEXITY = "refactor_complexity"
    FIX_FORMATTING = "fix_formatting"


class AnalysisMode(str, Enum):
    """Analysis modes"""
    QUICK = "quick"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    DEEP = "deep"


class RepositoryType(str, Enum):
    """Repository types"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    MIXED = "mixed"
    UNKNOWN = "unknown"


# Dataclass models for internal use
@dataclass
class AnalysisContext:
    """Context for analysis operations"""
    repo_url: str
    branch: str = "main"
    analysis_mode: AnalysisMode = AnalysisMode.STANDARD
    start_time: datetime = None
    cache_enabled: bool = True
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()


@dataclass
class ProcessingStats:
    """Statistics for processing operations"""
    files_processed: int = 0
    functions_analyzed: int = 0
    classes_analyzed: int = 0
    issues_detected: int = 0
    resolutions_generated: int = 0
    processing_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0


# Export all models for easy importing
__all__ = [
    # Enums
    "IssueSeverity", "IssueType", "IssueTypeExtended", "ResolutionType", 
    "AnalysisMode", "RepositoryType",
    
    # Core Models
    "CodeIssue", "EntryPoint", "CriticalFile", "DependencyNode", 
    "AutomatedResolution", "FunctionContext", "HalsteadMetrics",
    "GraphMetrics", "DeadCodeAnalysis", "HealthMetrics", 
    "RepositoryStructure", "AnalysisResults",
    
    # Metrics Models
    "FunctionMetrics", "ClassMetrics", "FileMetrics",
    
    # Request/Response Models
    "ComprehensiveAnalysisRequest", "ComprehensiveAnalysisResponse",
    "RepoRequest", "CodebaseAnalysisRequest", "CodebaseAnalysisResponse",
    "HealthCheckResponse", "RootResponse",
    
    # Configuration
    "AnalysisConfig",
    
    # Dataclasses
    "AnalysisContext", "ProcessingStats"
]
