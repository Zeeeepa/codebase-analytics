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
  return_type: string | null;
  docstring: string | null;
  code: string;
  complexity: number;
  calls: string[];
  called_by: string[];
}

export interface VisualNode {
  id: string;
  name: string;
  type: string;
  path: string;
  issues: Issue[];
  blast_radius: number;
  metadata: any;
}

export interface Issue {
  type: string;
  severity: string;
  message: string;
  suggestion: string;
}

export interface ExplorationData {
  summary: {
    total_nodes: number;
    total_issues: number;
    error_hotspots_count: number;
    critical_paths_count: number;
  };
  error_hotspots: VisualNode[];
  exploration_insights: Insight[];
  critical_paths: any[];
}

export interface Insight {
  type: string;
  priority: string;
  title: string;
  description: string;
  affected_nodes: string[];
}

export interface BlastRadiusData {
  target_symbol: VisualNode;
  blast_radius: {
    affected_nodes: number;
    affected_edges: number;
  };
}

export interface Symbol {
  id: string;
  name: string;
  type: 'function' | 'class' | 'variable';
  filepath: string;
  start_line: number;
  end_line: number;
  issues?: {
    type: 'critical' | 'major' | 'minor';
    message: string;
  }[];
}

export interface Stats {
  files: number;
  directories: number;
  symbols: number;
  issues: number;
}

export interface IssueCount {
  critical: number;
  major: number;
  minor: number;
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
}

