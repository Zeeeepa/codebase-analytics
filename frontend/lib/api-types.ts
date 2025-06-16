// API Types for codebase analytics

export interface RepoRequest {
  repo_url: string;
}

export interface AnalysisResponse {
  repo_url: string;
  line_metrics: {
    total: {
      loc: number;
      lloc: number;
      sloc: number;
      comments: number;
      comment_density: number;
    }
  };
  cyclomatic_complexity: { average: number };
  depth_of_inheritance: { average: number };
  halstead_metrics: { 
    total_volume: number;
    average_volume: number;
  };
  maintainability_index: { average: number };
  description: string;
  num_files: number;
  num_functions: number;
  num_classes: number;
  monthly_commits: Record<string, number>;
}

export interface CodebaseStats {
  total_lines: number;
  code_lines: number;
  comment_lines: number;
  blank_lines: number;
  comment_ratio: number;
  file_count: number;
  function_count: number;
  class_count: number;
  complexity_average: number;
  maintainability_index: number;
}

export interface FunctionContext {
  id: string;
  name: string;
  file_path: string;
  start_line: number;
  end_line: number;
  parameters: string[];
  return_type: string;
  docstring: string;
  complexity: number;
  calls: string[];
  called_by: string[];
}

export interface ClassContext {
  id: string;
  name: string;
  file_path: string;
  start_line: number;
  end_line: number;
  methods: string[];
  attributes: string[];
  base_classes: string[];
  derived_classes: string[];
  complexity: number;
}

export interface VisualNode {
  id: string;
  name: string;
  type: string;
  path: string;
  issues: Issue[];
  blast_radius: number;
  metadata: any;
  location?: {
    start_line: number;
    end_line: number;
    start_col?: number;
    end_col?: number;
  };
}

export interface Issue {
  id: string;
  type: IssueType;
  severity: IssueSeverity;
  message: string;
  suggestion: string;
  location: {
    file_path: string;
    start_line: number;
    end_line: number;
    start_col?: number;
    end_col?: number;
  };
  code_snippet?: string;
  related_symbols?: string[];
  fix_examples?: string[];
  category: IssueCategory;
  impact_score: number; // 1-10 score indicating impact
}

export enum IssueType {
  UNUSED_IMPORT = "unused_import",
  UNUSED_VARIABLE = "unused_variable",
  UNUSED_FUNCTION = "unused_function",
  UNUSED_PARAMETER = "unused_parameter",
  UNDEFINED_VARIABLE = "undefined_variable",
  UNDEFINED_FUNCTION = "undefined_function",
  PARAMETER_MISMATCH = "parameter_mismatch",
  TYPE_ERROR = "type_error",
  CIRCULAR_DEPENDENCY = "circular_dependency",
  DEAD_CODE = "dead_code",
  COMPLEXITY_ISSUE = "complexity_issue",
  STYLE_ISSUE = "style_issue",
  SECURITY_ISSUE = "security_issue",
  PERFORMANCE_ISSUE = "performance_issue",
  OTHER = "other"
}

export enum IssueSeverity {
  CRITICAL = "critical",
  HIGH = "high",
  MEDIUM = "medium",
  LOW = "low",
  INFO = "info"
}

export enum IssueCategory {
  FUNCTIONAL = "functional", // Issues that affect functionality
  STRUCTURAL = "structural", // Issues related to code structure
  QUALITY = "quality",       // Issues related to code quality
  SECURITY = "security",     // Security vulnerabilities
  PERFORMANCE = "performance" // Performance issues
}

export interface ExplorationData {
  summary: {
    total_nodes: number;
    total_issues: number;
    error_hotspots_count: number;
    critical_paths_count: number;
    issues_by_severity: Record<IssueSeverity, number>;
    issues_by_category: Record<IssueCategory, number>;
  };
  error_hotspots: VisualNode[];
  exploration_insights: Insight[];
  critical_paths: any[];
  issues_by_file: Record<string, Issue[]>;
}

export interface Insight {
  id: string;
  type: string;
  priority: IssueSeverity;
  title: string;
  description: string;
  affected_nodes: string[];
  recommendation: string;
  impact_score: number; // 1-10 score indicating impact
}

export interface BlastRadiusData {
  target_symbol: VisualNode;
  blast_radius: {
    affected_nodes: number;
    affected_edges: number;
    impact_score: number; // 1-10 score indicating impact
    affected_files: number;
    affected_functions: number;
    affected_classes: number;
  };
  affected_nodes: VisualNode[];
  impact_graph: {
    nodes: any[];
    edges: any[];
  };
}

export interface IssueCount {
  critical: number;
  high: number;
  medium: number;
  low: number;
  info: number;
}

export interface Stats {
  lines: number;
  functions: number;
  classes: number;
  complexity: number;
}

export interface FileNode {
  name: string;
  type: 'file' | 'directory';
  file_type?: string;
  path: string;
  issues: IssueCount;
  stats: Stats;
  symbols?: Symbol[];
  children?: { [key: string]: FileNode };
}

export interface Symbol {
  id: string;
  name: string;
  type: 'function' | 'class' | 'variable' | 'module';
  location: {
    start_line: number;
    end_line: number;
  };
  issues?: Issue[];
}

export interface RepoData {
  name: string;
  description: string;
  linesOfCode: number;
  cyclomaticComplexity: number;
  depthOfInheritance: number;
  halsteadVolume: number;
  maintainabilityIndex: number;
  commentDensity: number;
  sloc: number;
  lloc: number;
  numberOfFiles: number;
  numberOfFunctions: number;
  numberOfClasses: number;
  issues: {
    total: number;
    by_severity: IssueCount;
    by_category: Record<IssueCategory, number>;
  };
}

