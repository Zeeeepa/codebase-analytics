"""
Comprehensive Data Models for Codebase Analytics
Enhanced with structured summary fields and graph-sitter integration
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Set
from datetime import datetime
from enum import Enum
# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================
class IssueSeverity(Enum):
    """Issue severity levels"""
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"
class IssueType(Enum):
    """Types of code issues"""
    # Critical Issues
    NULL_REFERENCE = "null_reference"
    DIVISION_BY_ZERO = "division_by_zero"
    UNDEFINED_VARIABLE = "undefined_variable"
    TYPE_MISMATCH = "type_mismatch"
    
    # Major Issues
    MISSING_RETURN = "missing_return"
    UNREACHABLE_CODE = "unreachable_code"
    LONG_FUNCTION = "long_function"
    LONG_PARAMETER_LIST = "long_parameter_list"
    BARE_EXCEPT = "bare_except"
    RESOURCE_LEAK = "resource_leak"
    
    # Minor Issues
    MISSING_DOCUMENTATION = "missing_documentation"
    MAGIC_NUMBER = "magic_number"
    DEAD_FUNCTION = "dead_function"
    UNUSED_IMPORT = "unused_import"
    UNUSED_VARIABLE = "unused_variable"
    STYLE_VIOLATION = "style_violation"
class SummaryType(Enum):
    """Types of summaries generated"""
    REPOSITORY_OVERVIEW = "repository_overview"
    FUNCTION_SUMMARY = "function_summary"
    CLASS_SUMMARY = "class_summary"
    FILE_SUMMARY = "file_summary"
    DEPENDENCY_SUMMARY = "dependency_summary"
    ISSUE_SUMMARY = "issue_summary"
    HEALTH_SUMMARY = "health_summary"
    COMPLEXITY_SUMMARY = "complexity_summary"
    ENTRY_POINTS_SUMMARY = "entry_points_summary"
    ARCHITECTURE_SUMMARY = "architecture_summary"
class AnalysisScope(Enum):
    """Scope of analysis"""
    FULL_REPOSITORY = "full_repository"
    DIRECTORY = "directory"
    FILE = "file"
    FUNCTION = "function"
    CLASS = "class"
# ============================================================================
# CORE DATA MODELS
# ============================================================================
@dataclass
class CodeIssue:
    """Represents a code issue with context and resolution"""
    id: str
    issue_type: IssueType
    severity: IssueSeverity
    message: str
    filepath: str
    line_number: int
    column_number: int = 0
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    suggested_fix: str = ""
    automated_resolution: Optional['AutomatedResolution'] = None
    confidence_score: float = 0.0
    blast_radius: Dict[str, Any] = field(default_factory=dict)
    related_issues: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
@dataclass
class AutomatedResolution:
    """Automated resolution for code issues"""
    resolution_type: str
    description: str
    original_code: str
    fixed_code: str
    confidence: float
    file_path: str
    line_number: int
    validation_status: str = "pending"
    estimated_time_saved: int = 0  # in minutes
    risk_level: str = "low"
    dependencies: List[str] = field(default_factory=list)
@dataclass
class HalsteadMetrics:
    """Halstead complexity metrics"""
    vocabulary: int = 0
    length: int = 0
    volume: float = 0.0
    difficulty: float = 0.0
    effort: float = 0.0
    time_seconds: float = 0.0
    estimated_bugs: float = 0.0
    operators: Dict[str, int] = field(default_factory=dict)
    operands: Dict[str, int] = field(default_factory=dict)
    unique_operators: int = 0
    unique_operands: int = 0
    total_operators: int = 0
    total_operands: int = 0
@dataclass
class FunctionContext:
    """Comprehensive function context and analysis"""
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
    issues: List[CodeIssue] = field(default_factory=list)
    complexity_metrics: HalsteadMetrics = field(default_factory=HalsteadMetrics)
    is_entry_point: bool = False
    is_dead_code: bool = False
    importance_score: int = 0
    usage_heat_score: float = 0.0
    docstring: str = ""
    decorators: List[str] = field(default_factory=list)
    async_function: bool = False
    generator_function: bool = False
@dataclass
class EntryPoint:
    """Entry point detection and analysis"""
    function_name: str
    filepath: str
    line_number: int
    entry_type: str  # main, api, cli, init, etc.
    importance_score: int
    usage_count: int
    call_depth: int
    dependencies: List[str] = field(default_factory=list)
    heat_map_data: Dict[str, float] = field(default_factory=dict)
    is_critical: bool = False
    framework_type: Optional[str] = None  # fastapi, flask, cli, etc.
    endpoint_pattern: Optional[str] = None
    http_methods: List[str] = field(default_factory=list)
@dataclass
class CriticalFile:
    """Critical file identification and analysis"""
    filepath: str
    importance_score: int
    file_type: str  # code, config, documentation
    language: str
    lines_of_code: int
    function_count: int
    class_count: int
    import_count: int
    export_count: int
    dependency_count: int
    usage_count: int
    issue_count: int
    critical_issues: int
    entry_points: List[str] = field(default_factory=list)
    key_functions: List[str] = field(default_factory=list)
    complexity_score: float = 0.0
    maintainability_index: float = 0.0
    test_coverage: float = 0.0
@dataclass
class DependencyNode:
    """Dependency graph node"""
    name: str
    type: str  # function, class, module, file
    filepath: str
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    depth: int = 0
    circular_dependencies: List[str] = field(default_factory=list)
    external_dependencies: List[str] = field(default_factory=list)
    weight: float = 0.0
    centrality_score: float = 0.0
# ============================================================================
# SUMMARY DATA MODELS
# ============================================================================
@dataclass
class RepositorySummary:
    """Comprehensive repository summary using graph-sitter functions"""
    # Basic Statistics
    total_files: int = 0
    code_files: int = 0
    documentation_files: int = 0
    config_files: int = 0
    test_files: int = 0
    
    # Code Metrics
    total_functions: int = 0
    total_classes: int = 0
    total_lines_of_code: int = 0
    total_imports: int = 0
    total_exports: int = 0
    
    # Language Breakdown
    languages: Dict[str, int] = field(default_factory=dict)
    primary_language: str = ""
    language_distribution: Dict[str, float] = field(default_factory=dict)
    
    # Architecture Overview
    entry_points_count: int = 0
    critical_files_count: int = 0
    dependency_depth: int = 0
    circular_dependencies: int = 0
    
    # Quality Metrics
    total_issues: int = 0
    critical_issues: int = 0
    major_issues: int = 0
    minor_issues: int = 0
    automated_fixes_available: int = 0
    
    # Complexity Analysis
    average_function_complexity: float = 0.0
    highest_complexity_function: str = ""
    complexity_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Health Indicators
    health_score: float = 0.0
    maintainability_index: float = 0.0
    technical_debt_ratio: float = 0.0
    test_coverage: float = 0.0
    
    # Generated Summaries
    executive_summary: str = ""
    architecture_overview: str = ""
    key_insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    analysis_duration: float = 0.0
    graph_sitter_version: str = ""
    analyzer_version: str = "2.0.0"
@dataclass
class FunctionSummary:
    """Function-level summary with graph-sitter integration"""
    name: str
    filepath: str
    line_count: int
    complexity_score: float
    importance_score: int
    usage_count: int
    issue_count: int
    is_entry_point: bool
    is_critical: bool
    halstead_metrics: HalsteadMetrics
    dependencies_count: int
    call_chain_length: int
    summary_text: str = ""
    key_responsibilities: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
@dataclass
class FileSummary:
    """File-level summary with comprehensive analysis"""
    filepath: str
    file_type: str
    language: str
    lines_of_code: int
    function_count: int
    class_count: int
    import_count: int
    issue_count: int
    complexity_score: float
    importance_score: int
    entry_points: List[str] = field(default_factory=list)
    key_functions: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    summary_text: str = ""
    purpose_description: str = ""
    architectural_role: str = ""
@dataclass
class IssueSummary:
    """Comprehensive issue analysis summary"""
    total_issues: int = 0
    by_severity: Dict[str, int] = field(default_factory=dict)
    by_type: Dict[str, int] = field(default_factory=dict)
    by_file: Dict[str, int] = field(default_factory=dict)
    by_function: Dict[str, int] = field(default_factory=dict)
    
    # Automated Resolution Analysis
    auto_fixable: int = 0
    manual_review_required: int = 0
    high_confidence_fixes: int = 0
    estimated_fix_time: int = 0  # in minutes
    
    # Trend Analysis
    hotspot_files: List[str] = field(default_factory=list)
    hotspot_functions: List[str] = field(default_factory=list)
    issue_patterns: List[str] = field(default_factory=list)
    
    # Priority Analysis
    critical_path_issues: List[str] = field(default_factory=list)
    blocking_issues: List[str] = field(default_factory=list)
    quick_wins: List[str] = field(default_factory=list)
    
    # Summary Text
    summary_text: str = ""
    priority_recommendations: List[str] = field(default_factory=list)
@dataclass
class HealthSummary:
    """Codebase health summary with actionable insights"""
    overall_score: float = 0.0
    grade: str = "F"  # A, B, C, D, F
    risk_level: str = "high"  # low, medium, high, critical
    
    # Component Scores
    code_quality_score: float = 0.0
    maintainability_score: float = 0.0
    reliability_score: float = 0.0
    security_score: float = 0.0
    performance_score: float = 0.0
    
    # Trend Indicators
    improving_areas: List[str] = field(default_factory=list)
    declining_areas: List[str] = field(default_factory=list)
    stable_areas: List[str] = field(default_factory=list)
    
    # Action Items
    immediate_actions: List[str] = field(default_factory=list)
    short_term_goals: List[str] = field(default_factory=list)
    long_term_objectives: List[str] = field(default_factory=list)
    
    # Summary
    health_summary: str = ""
    key_concerns: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
# ============================================================================
# TREE STRUCTURE MODELS
# ============================================================================
@dataclass
class TreeNode:
    """Tree structure node for visualization"""
    name: str
    type: str  # directory, file, function, class
    path: str
    children: List['TreeNode'] = field(default_factory=list)
    
    # Issue Information
    issues: Dict[str, int] = field(default_factory=dict)  # severity -> count
    total_issues: int = 0
    
    # Importance Markers
    is_entry_point: bool = False
    is_critical_file: bool = False
    importance_score: int = 0
    
    # Visualization Data
    color_code: str = ""  # yellow, orange, red, etc.
    icon: str = ""
    badges: List[str] = field(default_factory=list)
    
    # Interactive Data
    click_action: str = ""
    tooltip_data: Dict[str, Any] = field(default_factory=dict)
    
    # Summary Information
    summary: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
@dataclass
class RepositoryStructure:
    """Complete repository structure with analysis"""
    root: TreeNode
    total_nodes: int = 0
    max_depth: int = 0
    
    # File Type Distribution
    file_types: Dict[str, int] = field(default_factory=dict)
    directory_count: int = 0
    
    # Issue Distribution
    issue_hotspots: List[str] = field(default_factory=list)
    clean_areas: List[str] = field(default_factory=list)
    
    # Navigation Data
    entry_points_map: Dict[str, str] = field(default_factory=dict)
    critical_files_map: Dict[str, str] = field(default_factory=dict)
    
    # Visualization Settings
    color_scheme: Dict[str, str] = field(default_factory=dict)
    icon_mapping: Dict[str, str] = field(default_factory=dict)
    
    # Generated Structure
    tree_visualization: str = ""
    interactive_map: Dict[str, Any] = field(default_factory=dict)
# ============================================================================
# COMPREHENSIVE ANALYSIS RESULTS
# ============================================================================
@dataclass
class AnalysisResults:
    """Complete analysis results with all summaries"""
    # Core Analysis Data
    repository_facts: RepositorySummary
    function_contexts: Dict[str, FunctionContext] = field(default_factory=dict)
    entry_points: List[EntryPoint] = field(default_factory=list)
    critical_files: List[CriticalFile] = field(default_factory=list)
    issues: List[CodeIssue] = field(default_factory=list)
    
    # Structured Summaries
    summaries: Dict[SummaryType, Any] = field(default_factory=dict)
    
    # Repository Structure
    repository_structure: RepositoryStructure = field(default_factory=RepositoryStructure)
    
    # Dependency Analysis
    dependency_graph: Dict[str, DependencyNode] = field(default_factory=dict)
    
    # Health and Quality
    health_metrics: HealthSummary = field(default_factory=HealthSummary)
    issue_summary: IssueSummary = field(default_factory=IssueSummary)
    
    # Generated Reports
    executive_summary: str = ""
    technical_summary: str = ""
    architecture_summary: str = ""
    
    # Metadata
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_summary(self, summary_type: SummaryType, summary_data: Any):
        """Add a summary to the results"""
        self.summaries[summary_type] = summary_data
    
    def get_summary(self, summary_type: SummaryType) -> Optional[Any]:
        """Get a specific summary"""
        return self.summaries.get(summary_type)
    
    def get_all_summaries(self) -> Dict[SummaryType, Any]:
        """Get all summaries"""
        return self.summaries.copy()
# ============================================================================
# CONFIGURATION AND SETTINGS
# ============================================================================
@dataclass
class AnalysisConfig:
    """Configuration for analysis execution"""
    # Scope Settings
    scope: AnalysisScope = AnalysisScope.FULL_REPOSITORY
    target_path: str = ""
    
    # Analysis Options
    include_tests: bool = True
    include_documentation: bool = True
    include_config_files: bool = True
    
    # Issue Detection
    detect_critical_issues: bool = True
    detect_code_quality_issues: bool = True
    detect_style_issues: bool = False
    
    # Summary Generation
    generate_summaries: bool = True
    summary_types: List[SummaryType] = field(default_factory=lambda: list(SummaryType))
    
    # Performance Settings
    max_file_size: int = 1024 * 1024  # 1MB
    timeout_seconds: int = 300
    parallel_processing: bool = True
    
    # Output Settings
    include_source_code: bool = False
    include_raw_data: bool = False
    compress_output: bool = False
    
    # Graph-sitter Settings
    use_graph_sitter_summaries: bool = True
    fallback_to_local: bool = True
    cache_results: bool = True
# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def create_default_config() -> AnalysisConfig:
    """Create default analysis configuration"""
    return AnalysisConfig(
        scope=AnalysisScope.FULL_REPOSITORY,
        generate_summaries=True,
        use_graph_sitter_summaries=True,
        fallback_to_local=True
    )
def severity_to_score(severity: IssueSeverity) -> int:
    """Convert severity to numeric score"""
    scores = {
        IssueSeverity.CRITICAL: 10,
        IssueSeverity.MAJOR: 5,
        IssueSeverity.MINOR: 2,
        IssueSeverity.INFO: 1
    }
    return scores.get(severity, 0)
def calculate_health_grade(score: float) -> str:
    """Calculate health grade from score"""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"
def format_issue_badge(issues: Dict[str, int]) -> str:
    """Format issue counts as badges"""
    badges = []
    if issues.get('critical', 0) > 0:
        badges.append(f"[⚠️ Critical: {issues['critical']}]")
    if issues.get('major', 0) > 0:
        badges.append(f"[👉 Major: {issues['major']}]")
    if issues.get('minor', 0) > 0:
        badges.append(f"[🔍 Minor: {issues['minor']}]")
    return " ".join(badges)
def create_tree_node(name: str, node_type: str, path: str) -> TreeNode:
    """Create a tree node with default values"""
    return TreeNode(
        name=name,
        type=node_type,
        path=path,
        issues={},
        color_code="",
        icon="📁" if node_type == "directory" else "📄",
        badges=[],
        tooltip_data={}
    )
