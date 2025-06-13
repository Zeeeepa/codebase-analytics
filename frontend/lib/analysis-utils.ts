// Utility functions for analysis and data processing

import { 
  Issue, 
  IssueType, 
  IssueSeverity, 
  IssueCategory, 
  VisualNode, 
  ExplorationData,
  BlastRadiusData,
  Insight
} from './api-types';

// Filter issues based on severity, category, type, and search query
export function filterIssues(
  issues: Issue[],
  severityFilter: IssueSeverity | 'all' = 'all',
  categoryFilter: IssueCategory | 'all' = 'all',
  typeFilter: IssueType | 'all' = 'all',
  searchQuery: string = ''
): Issue[] {
  return issues.filter(issue => {
    const matchesSeverity = severityFilter === 'all' || issue.severity === severityFilter;
    const matchesCategory = categoryFilter === 'all' || issue.category === categoryFilter;
    const matchesType = typeFilter === 'all' || issue.type === typeFilter;
    
    const matchesSearch = !searchQuery || 
      issue.message.toLowerCase().includes(searchQuery.toLowerCase()) ||
      issue.type.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (issue.location?.file_path?.toLowerCase().includes(searchQuery.toLowerCase()) ?? false);
    
    return matchesSeverity && matchesCategory && matchesType && matchesSearch;
  });
}

// Filter nodes based on issues, type, and search query
export function filterNodes(
  nodes: VisualNode[],
  severityFilter: IssueSeverity | 'all' = 'all',
  categoryFilter: IssueCategory | 'all' = 'all',
  typeFilter: IssueType | 'all' = 'all',
  searchQuery: string = ''
): VisualNode[] {
  return nodes.filter(node => {
    // Filter out nodes with no issues
    if (!node.issues || node.issues.length === 0) return false;
    
    // Filter issues within the node
    const filteredIssues = filterIssues(
      node.issues,
      severityFilter,
      categoryFilter,
      typeFilter,
      searchQuery
    );
    
    // Check if node name matches search query
    const nodeNameMatches = !searchQuery || 
      node.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      node.path.toLowerCase().includes(searchQuery.toLowerCase());
    
    // Return true if node has matching issues or its name matches
    return filteredIssues.length > 0 || nodeNameMatches;
  });
}

// Sort issues by different criteria
export function sortIssues(
  issues: Issue[],
  sortBy: 'impact' | 'severity' | 'type' | 'name' = 'impact'
): Issue[] {
  return [...issues].sort((a, b) => {
    if (sortBy === 'impact') {
      return (b.impact_score || 0) - (a.impact_score || 0);
    } else if (sortBy === 'severity') {
      const severityOrder = {
        [IssueSeverity.CRITICAL]: 5,
        [IssueSeverity.HIGH]: 4,
        [IssueSeverity.MEDIUM]: 3,
        [IssueSeverity.LOW]: 2,
        [IssueSeverity.INFO]: 1
      };
      return (severityOrder[b.severity] || 0) - (severityOrder[a.severity] || 0);
    } else if (sortBy === 'type') {
      return a.type.localeCompare(b.type);
    } else {
      // Sort by message text
      return a.message.localeCompare(b.message);
    }
  });
}

// Sort nodes by different criteria
export function sortNodes(
  nodes: VisualNode[],
  sortBy: 'impact' | 'severity' | 'type' | 'name' = 'impact'
): VisualNode[] {
  return [...nodes].sort((a, b) => {
    if (sortBy === 'impact') {
      // Sort by number of issues as a proxy for impact
      return (b.issues?.length || 0) - (a.issues?.length || 0);
    } else if (sortBy === 'severity') {
      // Sort by highest severity issue in each node
      const getHighestSeverity = (node: VisualNode): number => {
        if (!node.issues || node.issues.length === 0) return 0;
        
        const severityOrder = {
          [IssueSeverity.CRITICAL]: 5,
          [IssueSeverity.HIGH]: 4,
          [IssueSeverity.MEDIUM]: 3,
          [IssueSeverity.LOW]: 2,
          [IssueSeverity.INFO]: 1
        };
        
        return Math.max(...node.issues.map(issue => severityOrder[issue.severity] || 0));
      };
      
      return getHighestSeverity(b) - getHighestSeverity(a);
    } else if (sortBy === 'type') {
      // Sort by node type
      return a.type.localeCompare(b.type);
    } else {
      // Sort by name
      return a.name.localeCompare(b.name);
    }
  });
}

// Calculate summary statistics from exploration data
export function calculateSummaryStats(data: ExplorationData) {
  if (!data) return null;
  
  const totalIssues = data.summary.total_issues || 0;
  const issuesBySeverity = data.summary.issues_by_severity || {};
  const issuesByCategory = data.summary.issues_by_category || {};
  
  return {
    totalIssues,
    issuesBySeverity,
    issuesByCategory
  };
}

// Calculate blast radius statistics
export function calculateBlastRadiusStats(data: BlastRadiusData) {
  if (!data) return null;
  
  const { affected_nodes, affected_edges, impact_score } = data.blast_radius;
  
  // Calculate percentage of codebase affected (assuming 100 nodes total as a placeholder)
  // In a real implementation, you would use the total number of nodes from the exploration data
  const percentageAffected = (affected_nodes / 100) * 100;
  
  return {
    affectedNodes: affected_nodes,
    affectedEdges: affected_edges,
    impactScore: impact_score,
    percentageAffected
  };
}

// Get color for severity level
export function getSeverityColor(severity: IssueSeverity | string): string {
  switch (severity) {
    case IssueSeverity.CRITICAL:
      return 'bg-red-600';
    case IssueSeverity.HIGH:
      return 'bg-orange-500';
    case IssueSeverity.MEDIUM:
      return 'bg-yellow-500';
    case IssueSeverity.LOW:
      return 'bg-blue-500';
    case IssueSeverity.INFO:
      return 'bg-gray-500';
    default:
      return 'bg-gray-500';
  }
}

// Get label for severity level
export function getSeverityLabel(severity: IssueSeverity | string): string {
  switch (severity) {
    case IssueSeverity.CRITICAL:
      return 'Critical';
    case IssueSeverity.HIGH:
      return 'High';
    case IssueSeverity.MEDIUM:
      return 'Medium';
    case IssueSeverity.LOW:
      return 'Low';
    case IssueSeverity.INFO:
      return 'Info';
    default:
      return severity.toString();
  }
}

// Get color for category
export function getCategoryColor(category: IssueCategory | string): string {
  switch (category) {
    case IssueCategory.FUNCTIONAL:
      return 'bg-purple-500';
    case IssueCategory.STRUCTURAL:
      return 'bg-indigo-500';
    case IssueCategory.QUALITY:
      return 'bg-teal-500';
    case IssueCategory.SECURITY:
      return 'bg-red-500';
    case IssueCategory.PERFORMANCE:
      return 'bg-amber-500';
    default:
      return 'bg-gray-500';
  }
}

// Get label for category
export function getCategoryLabel(category: IssueCategory | string): string {
  switch (category) {
    case IssueCategory.FUNCTIONAL:
      return 'Functional';
    case IssueCategory.STRUCTURAL:
      return 'Structural';
    case IssueCategory.QUALITY:
      return 'Quality';
    case IssueCategory.SECURITY:
      return 'Security';
    case IssueCategory.PERFORMANCE:
      return 'Performance';
    default:
      return category.toString();
  }
}

// Get icon component name for issue type
export function getIssueTypeIcon(type: IssueType | string): string {
  switch (type) {
    case IssueType.UNUSED_IMPORT:
      return 'Ban';
    case IssueType.UNUSED_VARIABLE:
    case IssueType.UNUSED_FUNCTION:
    case IssueType.UNUSED_PARAMETER:
      return 'Code2';
    case IssueType.UNDEFINED_VARIABLE:
    case IssueType.UNDEFINED_FUNCTION:
      return 'AlertCircle';
    case IssueType.PARAMETER_MISMATCH:
      return 'AlertTriangle';
    case IssueType.TYPE_ERROR:
      return 'FileWarning';
    case IssueType.CIRCULAR_DEPENDENCY:
      return 'Repeat';
    case IssueType.DEAD_CODE:
      return 'Unlink';
    case IssueType.COMPLEXITY_ISSUE:
      return 'FileCode';
    case IssueType.STYLE_ISSUE:
      return 'Sparkles';
    case IssueType.SECURITY_ISSUE:
      return 'Shield';
    case IssueType.PERFORMANCE_ISSUE:
      return 'Gauge';
    default:
      return 'Bug';
  }
}

// Get label for issue type
export function getIssueTypeLabel(type: IssueType | string): string {
  switch (type) {
    case IssueType.UNUSED_IMPORT:
      return 'Unused Import';
    case IssueType.UNUSED_VARIABLE:
      return 'Unused Variable';
    case IssueType.UNUSED_FUNCTION:
      return 'Unused Function';
    case IssueType.UNUSED_PARAMETER:
      return 'Unused Parameter';
    case IssueType.UNDEFINED_VARIABLE:
      return 'Undefined Variable';
    case IssueType.UNDEFINED_FUNCTION:
      return 'Undefined Function';
    case IssueType.PARAMETER_MISMATCH:
      return 'Parameter Mismatch';
    case IssueType.TYPE_ERROR:
      return 'Type Error';
    case IssueType.CIRCULAR_DEPENDENCY:
      return 'Circular Dependency';
    case IssueType.DEAD_CODE:
      return 'Dead Code';
    case IssueType.COMPLEXITY_ISSUE:
      return 'Complexity Issue';
    case IssueType.STYLE_ISSUE:
      return 'Style Issue';
    case IssueType.SECURITY_ISSUE:
      return 'Security Issue';
    case IssueType.PERFORMANCE_ISSUE:
      return 'Performance Issue';
    default:
      return type.toString().replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  }
}

// Get impact badge color and label based on score
export function getImpactBadgeInfo(score: number): { color: string; label: string } {
  if (score >= 9) {
    return { color: 'bg-red-600', label: 'Critical' };
  } else if (score >= 7) {
    return { color: 'bg-orange-500', label: 'High' };
  } else if (score >= 5) {
    return { color: 'bg-yellow-500', label: 'Medium' };
  } else if (score >= 3) {
    return { color: 'bg-blue-500', label: 'Low' };
  } else {
    return { color: 'bg-gray-500', label: 'Minimal' };
  }
}

// Group issues by file
export function groupIssuesByFile(issues: Issue[]): Record<string, Issue[]> {
  return issues.reduce((acc, issue) => {
    const filePath = issue.location?.file_path || 'unknown';
    if (!acc[filePath]) {
      acc[filePath] = [];
    }
    acc[filePath].push(issue);
    return acc;
  }, {} as Record<string, Issue[]>);
}

// Find related issues (issues in the same file or with similar types)
export function findRelatedIssues(issue: Issue, allIssues: Issue[]): Issue[] {
  if (!issue) return [];
  
  return allIssues.filter(otherIssue => {
    // Skip the current issue
    if (otherIssue.id === issue.id) return false;
    
    // Check if in the same file
    const sameFile = otherIssue.location?.file_path === issue.location?.file_path;
    
    // Check if same type
    const sameType = otherIssue.type === issue.type;
    
    // Check if related symbols overlap
    const hasOverlappingSymbols = 
      issue.related_symbols && 
      otherIssue.related_symbols && 
      issue.related_symbols.some(symbol => otherIssue.related_symbols?.includes(symbol));
    
    return sameFile || sameType || hasOverlappingSymbols;
  });
}

// Extract insights from exploration data
export function extractInsights(data: ExplorationData): Insight[] {
  if (!data || !data.exploration_insights) return [];
  
  return data.exploration_insights;
}

// Find hotspots (files with the most issues)
export function findHotspots(data: ExplorationData, limit: number = 10): VisualNode[] {
  if (!data || !data.error_hotspots) return [];
  
  return [...data.error_hotspots]
    .sort((a, b) => (b.issues?.length || 0) - (a.issues?.length || 0))
    .slice(0, limit);
}

// Calculate code quality score (0-100) based on issues
export function calculateCodeQualityScore(data: ExplorationData): number {
  if (!data) return 0;
  
  const { total_issues = 0 } = data.summary;
  const issuesBySeverity = data.summary.issues_by_severity || {};
  
  // Weight issues by severity
  const weightedIssues = 
    (issuesBySeverity[IssueSeverity.CRITICAL] || 0) * 10 +
    (issuesBySeverity[IssueSeverity.HIGH] || 0) * 5 +
    (issuesBySeverity[IssueSeverity.MEDIUM] || 0) * 2 +
    (issuesBySeverity[IssueSeverity.LOW] || 0) * 1 +
    (issuesBySeverity[IssueSeverity.INFO] || 0) * 0.1;
  
  // Calculate score (higher is better)
  // This is a simple formula that can be adjusted based on project size
  const baseScore = 100;
  const penalty = Math.min(weightedIssues, 100); // Cap penalty at 100
  
  return Math.max(0, baseScore - penalty);
}

