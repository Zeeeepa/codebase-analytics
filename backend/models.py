"""
Data models and enumerations for the codebase analysis API.
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from enum import Enum


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