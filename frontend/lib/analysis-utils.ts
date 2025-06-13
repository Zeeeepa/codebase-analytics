import { 
  Issue, 
  IssueSeverity, 
  IssueCategory, 
  IssueType, 
  VisualNode, 
  ExplorationData, 
  BlastRadiusData,
  Insight
} from './api-types';

// Filtering functions
export function filterIssues(
  issues: Issue[],
  severityFilter: IssueSeverity | 'all' = 'all',
  categoryFilter: IssueCategory | 'all' = 'all',
  typeFilter: IssueType | 'all' = 'all',
  searchQuery: string = ''
): Issue[] {
  if (!issues || issues.length === 0) return [];
  
  return issues.filter(issue => {
    // Apply severity filter
    if (severityFilter !== 'all' && issue.severity !== severityFilter) {
      return false;
    }
    
    // Apply category filter
    if (categoryFilter !== 'all' && issue.category !== categoryFilter) {
      return false;
    }
    
    // Apply type filter
    if (typeFilter !== 'all' && issue.type !== typeFilter) {
      return false;
    }
    
    // Apply search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      const messageMatch = issue.message?.toLowerCase().includes(query);
      const suggestionMatch = issue.suggestion?.toLowerCase().includes(query);
      const locationMatch = issue.location?.file_path?.toLowerCase().includes(query);
      
      return messageMatch || suggestionMatch || locationMatch;
    }
    
    return true;
  });
}

export function filterNodes(
  nodes: VisualNode[],
  severityFilter: IssueSeverity | 'all' = 'all',
  categoryFilter: IssueCategory | 'all' = 'all',
  typeFilter: IssueType | 'all' = 'all',
  searchQuery: string = ''
): VisualNode[] {
  if (!nodes || nodes.length === 0) return [];
  
  return nodes.filter(node => {
    // If no filters are applied and no search query, return all nodes
    if (severityFilter === 'all' && categoryFilter === 'all' && typeFilter === 'all' && !searchQuery) {
      return true;
    }
    
    // Filter by issues if the node has issues
    if (node.issues && node.issues.length > 0) {
      // For severity, category, and type filters, check if any issue matches
      if (severityFilter !== 'all') {
        if (!node.issues.some(issue => issue.severity === severityFilter)) {
          return false;
        }
      }
      
      if (categoryFilter !== 'all') {
        if (!node.issues.some(issue => issue.category === categoryFilter)) {
          return false;
        }
      }
      
      if (typeFilter !== 'all') {
        if (!node.issues.some(issue => issue.type === typeFilter)) {
          return false;
        }
      }
    } else if (severityFilter !== 'all' || categoryFilter !== 'all' || typeFilter !== 'all') {
      // If the node has no issues and any filter is active, exclude it
      return false;
    }
    
    // Apply search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      const nameMatch = node.name?.toLowerCase().includes(query);
      const pathMatch = node.path?.toLowerCase().includes(query);
      const typeMatch = node.type?.toLowerCase().includes(query);
      
      // Also check if any issue message matches
      const issueMatch = node.issues?.some(issue => 
        issue.message?.toLowerCase().includes(query)
      );
      
      return nameMatch || pathMatch || typeMatch || issueMatch;
    }
    
    return true;
  });
}

// Sorting functions
export function sortIssues(
  issues: Issue[],
  sortBy: 'impact' | 'severity' | 'type' | 'name' = 'impact'
): Issue[] {
  if (!issues || issues.length === 0) return [];
  
  const sortedIssues = [...issues];
  
  switch (sortBy) {
    case 'impact':
      // Sort by impact score (high to low)
      return sortedIssues.sort((a, b) => {
        const scoreA = a.impact_score || 0;
        const scoreB = b.impact_score || 0;
        return scoreB - scoreA;
      });
      
    case 'severity':
      // Sort by severity (critical to info)
      const severityOrder = {
        [IssueSeverity.CRITICAL]: 5,
        [IssueSeverity.HIGH]: 4,
        [IssueSeverity.MEDIUM]: 3,
        [IssueSeverity.LOW]: 2,
        [IssueSeverity.INFO]: 1
      };
      
      return sortedIssues.sort((a, b) => {
        const orderA = severityOrder[a.severity] || 0;
        const orderB = severityOrder[b.severity] || 0;
        return orderB - orderA;
      });
      
    case 'type':
      // Sort alphabetically by type
      return sortedIssues.sort((a, b) => {
        return a.type.localeCompare(b.type);
      });
      
    case 'name':
      // Sort alphabetically by message
      return sortedIssues.sort((a, b) => {
        return a.message.localeCompare(b.message);
      });
      
    default:
      return sortedIssues;
  }
}

export function sortNodes(
  nodes: VisualNode[],
  sortBy: 'impact' | 'severity' | 'type' | 'name' = 'impact'
): VisualNode[] {
  if (!nodes || nodes.length === 0) return [];
  
  const sortedNodes = [...nodes];
  
  switch (sortBy) {
    case 'impact':
      // Sort by issue count (high to low)
      return sortedNodes.sort((a, b) => {
        const countA = a.issues?.length || 0;
        const countB = b.issues?.length || 0;
        
        // If issue counts are equal, sort by blast radius
        if (countA === countB) {
          const radiusA = a.blast_radius || 0;
          const radiusB = b.blast_radius || 0;
          return radiusB - radiusA;
        }
        
        return countB - countA;
      });
      
    case 'severity':
      // Sort by highest severity issue
      const severityOrder = {
        [IssueSeverity.CRITICAL]: 5,
        [IssueSeverity.HIGH]: 4,
        [IssueSeverity.MEDIUM]: 3,
        [IssueSeverity.LOW]: 2,
        [IssueSeverity.INFO]: 1,
        'none': 0
      };
      
      return sortedNodes.sort((a, b) => {
        // Find highest severity for each node
        const highestSeverityA = a.issues && a.issues.length > 0
          ? Math.max(...a.issues.map(issue => severityOrder[issue.severity] || 0))
          : 0;
          
        const highestSeverityB = b.issues && b.issues.length > 0
          ? Math.max(...b.issues.map(issue => severityOrder[issue.severity] || 0))
          : 0;
          
        return highestSeverityB - highestSeverityA;
      });
      
    case 'type':
      // Sort alphabetically by type
      return sortedNodes.sort((a, b) => {
        return a.type.localeCompare(b.type);
      });
      
    case 'name':
      // Sort alphabetically by name
      return sortedNodes.sort((a, b) => {
        return a.name.localeCompare(b.name);
      });
      
    default:
      return sortedNodes;
  }
}

// Calculation functions
export function calculateSummaryStats(data: ExplorationData | null) {
  if (!data) return null;
  
  return {
    totalIssues: data.summary.total_issues,
    issuesBySeverity: data.summary.issues_by_severity,
    issuesByCategory: data.summary.issues_by_category
  };
}

export function calculateBlastRadiusStats(data: BlastRadiusData | null) {
  if (!data) return null;
  
  // Assuming there's a total number of nodes in the codebase
  // This could be provided in the data or set to a default value
  const totalNodes = 100; // Default value, should be replaced with actual data
  
  return {
    affectedNodes: data.blast_radius.affected_nodes,
    affectedEdges: data.blast_radius.affected_edges,
    impactScore: data.blast_radius.impact_score,
    percentageAffected: (data.blast_radius.affected_nodes / totalNodes) * 100
  };
}

export function calculateCodeQualityScore(data: ExplorationData | null): number {
  if (!data) return 0;
  
  // Base score starts at 100
  let score = 100;
  
  // Deduct points based on issue severity
  const severityWeights = {
    [IssueSeverity.CRITICAL]: 10,
    [IssueSeverity.HIGH]: 5,
    [IssueSeverity.MEDIUM]: 2,
    [IssueSeverity.LOW]: 1,
    [IssueSeverity.INFO]: 0.1
  };
  
  // Calculate deductions
  let deduction = 0;
  
  for (const [severity, count] of Object.entries(data.summary.issues_by_severity)) {
    deduction += (count as number) * (severityWeights[severity as IssueSeverity] || 0);
  }
  
  // Ensure score doesn't go below 0
  score = Math.max(0, score - deduction);
  
  return score;
}

// Styling and labeling functions
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
      return severity;
  }
}

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
      return category;
  }
}

export function getIssueTypeIcon(type: IssueType | string): string {
  switch (type) {
    case IssueType.UNUSED_IMPORT:
      return 'Ban';
    case IssueType.UNUSED_VARIABLE:
      return 'Code2';
    case IssueType.UNDEFINED_VARIABLE:
      return 'AlertCircle';
    case IssueType.PARAMETER_MISMATCH:
      return 'AlertTriangle';
    case IssueType.CIRCULAR_DEPENDENCY:
      return 'Repeat';
    case IssueType.PERFORMANCE_ISSUE:
      return 'Gauge';
    case IssueType.SECURITY_ISSUE:
      return 'Shield';
    case IssueType.SYNTAX_ERROR:
      return 'FileWarning';
    case IssueType.STYLE_ISSUE:
      return 'Sparkles';
    default:
      return 'Bug';
  }
}

export function getIssueTypeLabel(type: IssueType | string): string {
  // Convert snake_case to Title Case
  return type
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ');
}

export function getImpactBadgeInfo(score: number): { color: string, label: string } {
  if (score >= 9) return { color: 'bg-red-600', label: 'Critical' };
  if (score >= 7) return { color: 'bg-orange-500', label: 'High' };
  if (score >= 5) return { color: 'bg-yellow-500', label: 'Medium' };
  if (score >= 3) return { color: 'bg-blue-500', label: 'Low' };
  return { color: 'bg-gray-500', label: 'Minimal' };
}

// Data processing functions
export function groupIssuesByFile(issues: Issue[]): Record<string, Issue[]> {
  if (!issues || issues.length === 0) return {};
  
  const groupedIssues: Record<string, Issue[]> = {};
  
  issues.forEach(issue => {
    const filePath = issue.location?.file_path || 'unknown';
    
    if (!groupedIssues[filePath]) {
      groupedIssues[filePath] = [];
    }
    
    groupedIssues[filePath].push(issue);
  });
  
  return groupedIssues;
}

export function findRelatedIssues(issue: Issue | null, allIssues: Issue[]): Issue[] {
  if (!issue || !allIssues || allIssues.length === 0) return [];
  
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

// Analysis functions
export function extractInsights(data: ExplorationData | null): Insight[] {
  if (!data || !data.exploration_insights) return [];
  
  return data.exploration_insights;
}

export function findHotspots(data: ExplorationData | null, limit: number = 5): VisualNode[] {
  if (!data || !data.error_hotspots) return [];
  
  // Sort hotspots by issue count (descending)
  const sortedHotspots = [...data.error_hotspots].sort((a, b) => {
    const countA = a.issues?.length || 0;
    const countB = b.issues?.length || 0;
    return countB - countA;
  });
  
  // Return the top N hotspots
  return sortedHotspots.slice(0, limit);
}
