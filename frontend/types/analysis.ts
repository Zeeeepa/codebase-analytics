// Analysis data types for the enhanced modal component

export interface RepositoryMetrics {
  files: number;
  functions: number;
  classes: number;
  modules: number;
}

export interface IssueLocation {
  file: string;
  line?: number;
  column?: number;
  function?: string;
  method?: string;
}

export interface CodeIssue {
  id: string;
  severity: 'critical' | 'functional' | 'minor';
  type: string;
  title: string;
  description: string;
  location: IssueLocation;
  code_snippet?: string;
  suggestion?: string;
  category: string;
}

export interface FileNode {
  name: string;
  path: string;
  type: 'file' | 'directory';
  issues: {
    critical: number;
    functional: number;
    minor: number;
  };
  children?: FileNode[];
  size?: number;
  language?: string;
}

export interface RepositoryAnalysis {
  repository: {
    name: string;
    description?: string;
    url?: string;
  };
  metrics: RepositoryMetrics;
  structure: FileNode;
  issues: CodeIssue[];
  summary: {
    total_issues: number;
    critical_issues: number;
    functional_issues: number;
    minor_issues: number;
  };
  analysis_timestamp: string;
}

export interface AnalysisModalProps {
  analysis: RepositoryAnalysis | null;
  isOpen: boolean;
  onClose: () => void;
  onRefresh?: () => void;
}

export interface ModalTab {
  id: string;
  label: string;
  icon: React.ComponentType<any>;
  content: React.ComponentType<{ analysis: RepositoryAnalysis }>;
}

export interface GraphSitterConfig {
  languages: string[];
  analysis_depth: 'shallow' | 'deep';
  include_patterns: string[];
  exclude_patterns: string[];
}

