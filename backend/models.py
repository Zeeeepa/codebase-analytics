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
