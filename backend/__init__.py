"""
Codebase Analytics Backend Package
Comprehensive codebase analysis with issue detection and automated resolutions
"""

from .models import (
    CodeIssue, IssueType, IssueSeverity, FunctionContext, 
    AnalysisResults, AutomatedResolution, AnalysisConfig,
    RepoRequest, CodebaseAnalysisRequest, CodebaseAnalysisResponse,
    HealthCheckResponse, RootResponse
)

from .analysis import (
    CodebaseAnalyzer, get_codebase_summary, create_health_dashboard
)

# Import enhanced analysis capabilities
try:
    from .analysis_enhancements import EnhancedAnalysisEngine
    __all__.append("EnhancedAnalysisEngine")
except ImportError:
    pass

from .api import fastapi_app

__version__ = "2.0.0"
__author__ = "Codebase Analytics Team"
__description__ = "Comprehensive codebase analysis with issue detection and automated resolutions"

__all__ = [
    # Models
    "CodeIssue", "IssueType", "IssueSeverity", "FunctionContext",
    "AnalysisResults", "AutomatedResolution", "AnalysisConfig",
    "RepoRequest", "CodebaseAnalysisRequest", "CodebaseAnalysisResponse",
    "HealthCheckResponse", "RootResponse",
    
    # Analysis Engine
    "CodebaseAnalyzer", "get_codebase_summary", "create_health_dashboard",
    
    # API
    "fastapi_app"
]
