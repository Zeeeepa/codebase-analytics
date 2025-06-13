// API service for codebase analytics
import { 
  RepoRequest, 
  AnalysisResponse, 
  CodebaseStats, 
  FunctionContext, 
  ExplorationData,
  BlastRadiusData,
  FileNode,
  Issue,
  IssueType,
  IssueSeverity,
  IssueCategory
} from './api-types';

// Get backend URL - uses environment variable or falls back to defaults
const getBackendUrl = () => {
  // Check for environment variable first
  if (typeof process !== 'undefined' && process.env.NEXT_PUBLIC_BACKEND_URL) {
    return process.env.NEXT_PUBLIC_BACKEND_URL;
  }
  
  // In development, try local backend first
  if (typeof window !== 'undefined' && window.location.hostname === 'localhost') {
    // Connect to our deployed backend on port 8000
    return 'http://localhost:8000';
  }
  
  // Fallback to Modal deployment for production
  return 'https://zeeeepa--analytics-app-fastapi-modal-app-dev.modal.run';
};

// Parse repo URL to get owner/repo format
export const parseRepoUrl = (input: string): string => {
  if (input.includes('github.com')) {
    try {
      const url = new URL(input);
      const pathParts = url.pathname.split('/').filter(Boolean);
      if (pathParts.length >= 2) {
        return `${pathParts[0]}/${pathParts[1]}`;
      }
    } catch (e) {
      // If URL parsing fails, just return the input
      console.error('Error parsing URL:', e);
    }
  }
  return input;
};

// API methods
export const apiService = {
  // Analyze repository
  async analyzeRepo(repoUrl: string): Promise<AnalysisResponse> {
    const parsedRepoUrl = parseRepoUrl(repoUrl);
    const response = await fetch(`${getBackendUrl()}/analyze_repo`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({ repo_url: parsedRepoUrl }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  },

  // Get codebase stats
  async getCodebaseStats(codebaseId: string): Promise<CodebaseStats> {
    const response = await fetch(`${getBackendUrl()}/get_codebase_stats/${codebaseId}`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  },

  // Get function context
  async getFunctionContext(functionId: string): Promise<FunctionContext> {
    const response = await fetch(`${getBackendUrl()}/get_function_context/${functionId}`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  },

  // Get function call chain
  async getFunctionCallChain(functionId: string): Promise<string[]> {
    const response = await fetch(`${getBackendUrl()}/get_function_call_chain/${functionId}`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  },

  // Visual exploration
  async exploreVisual(repoUrl: string, mode: string = 'error_focused'): Promise<ExplorationData> {
    const parsedRepoUrl = parseRepoUrl(repoUrl);
    const response = await fetch(`${getBackendUrl()}/explore_visual`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        repo_url: parsedRepoUrl,
        mode: mode
      })
    });

    if (!response.ok) {
      throw new Error(`Error exploring repository: ${response.statusText}`);
    }

    const data = await response.json();
    
    // Enhance the data with additional issue information
    if (data.error_hotspots) {
      data.error_hotspots = data.error_hotspots.map((hotspot: any) => {
        if (hotspot.issues) {
          hotspot.issues = hotspot.issues.map((issue: any) => {
            // Generate a unique ID if not provided
            if (!issue.id) {
              issue.id = `issue-${Math.random().toString(36).substr(2, 9)}`;
            }
            
            // Ensure issue has proper type and severity enums
            if (typeof issue.type === 'string' && !Object.values(IssueType).includes(issue.type as IssueType)) {
              issue.type = mapLegacyIssueType(issue.type);
            }
            
            if (typeof issue.severity === 'string' && !Object.values(IssueSeverity).includes(issue.severity as IssueSeverity)) {
              issue.severity = mapLegacySeverity(issue.severity);
            }
            
            // Add category if not present
            if (!issue.category) {
              issue.category = determineIssueCategory(issue.type);
            }
            
            // Add impact score if not present
            if (!issue.impact_score) {
              issue.impact_score = calculateImpactScore(issue.severity, issue.type);
            }
            
            // Add location if not present
            if (!issue.location) {
              issue.location = {
                file_path: hotspot.path,
                start_line: 0,
                end_line: 0
              };
            }
            
            return issue;
          });
        }
        return hotspot;
      });
    }
    
    // Add summary statistics if not present
    if (!data.summary.issues_by_severity) {
      data.summary.issues_by_severity = countIssuesBySeverity(data.error_hotspots);
    }
    
    if (!data.summary.issues_by_category) {
      data.summary.issues_by_category = countIssuesByCategory(data.error_hotspots);
    }
    
    // Add issues by file if not present
    if (!data.issues_by_file) {
      data.issues_by_file = groupIssuesByFile(data.error_hotspots);
    }
    
    return data;
  },

  // Blast radius analysis
  async analyzeBlastRadius(repoUrl: string, symbolName: string): Promise<BlastRadiusData> {
    const parsedRepoUrl = parseRepoUrl(repoUrl || '.');
    const response = await fetch(`${getBackendUrl()}/analyze_blast_radius`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        repo_url: parsedRepoUrl,
        symbol_name: symbolName
      })
    });

    if (!response.ok) {
      throw new Error(`Error analyzing blast radius: ${response.statusText}`);
    }

    const data = await response.json();
    
    // Enhance the blast radius data with impact scores
    if (!data.blast_radius.impact_score) {
      data.blast_radius.impact_score = calculateBlastRadiusImpact(
        data.blast_radius.affected_nodes,
        data.blast_radius.affected_edges
      );
    }
    
    // Add impact graph if not present
    if (!data.impact_graph) {
      data.impact_graph = {
        nodes: [],
        edges: []
      };
    }
    
    return data;
  },

  // Get repository structure
  async getRepoStructure(repoUrl: string): Promise<FileNode> {
    const parsedRepoUrl = parseRepoUrl(repoUrl);
    const response = await fetch(`${getBackendUrl()}/get_repo_structure`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        repo_url: parsedRepoUrl
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  }
};

// Helper functions for issue enhancement

function mapLegacyIssueType(type: string): IssueType {
  const typeMap: Record<string, IssueType> = {
    'unused_parameter': IssueType.UNUSED_PARAMETER,
    'unused_function': IssueType.UNUSED_FUNCTION,
    'unused_variable': IssueType.UNUSED_VARIABLE,
    'unused_import': IssueType.UNUSED_IMPORT,
    'undefined_variable': IssueType.UNDEFINED_VARIABLE,
    'undefined_function_call': IssueType.UNDEFINED_FUNCTION,
    'missing_required_arguments': IssueType.PARAMETER_MISMATCH,
    'mutable_default_parameter': IssueType.PARAMETER_MISMATCH,
    'circular_dependency': IssueType.CIRCULAR_DEPENDENCY,
    'dead_code': IssueType.DEAD_CODE,
    'complexity': IssueType.COMPLEXITY_ISSUE,
    'style': IssueType.STYLE_ISSUE,
    'security': IssueType.SECURITY_ISSUE,
    'performance': IssueType.PERFORMANCE_ISSUE
  };
  
  return typeMap[type] || IssueType.OTHER;
}

function mapLegacySeverity(severity: string): IssueSeverity {
  const severityMap: Record<string, IssueSeverity> = {
    'critical': IssueSeverity.CRITICAL,
    'high': IssueSeverity.HIGH,
    'major': IssueSeverity.HIGH,
    'medium': IssueSeverity.MEDIUM,
    'minor': IssueSeverity.LOW,
    'low': IssueSeverity.LOW,
    'info': IssueSeverity.INFO
  };
  
  return severityMap[severity] || IssueSeverity.MEDIUM;
}

function determineIssueCategory(type: IssueType | string): IssueCategory {
  const categoryMap: Record<string, IssueCategory> = {
    [IssueType.UNDEFINED_VARIABLE]: IssueCategory.FUNCTIONAL,
    [IssueType.UNDEFINED_FUNCTION]: IssueCategory.FUNCTIONAL,
    [IssueType.PARAMETER_MISMATCH]: IssueCategory.FUNCTIONAL,
    [IssueType.TYPE_ERROR]: IssueCategory.FUNCTIONAL,
    
    [IssueType.CIRCULAR_DEPENDENCY]: IssueCategory.STRUCTURAL,
    [IssueType.DEAD_CODE]: IssueCategory.STRUCTURAL,
    
    [IssueType.UNUSED_IMPORT]: IssueCategory.QUALITY,
    [IssueType.UNUSED_VARIABLE]: IssueCategory.QUALITY,
    [IssueType.UNUSED_FUNCTION]: IssueCategory.QUALITY,
    [IssueType.UNUSED_PARAMETER]: IssueCategory.QUALITY,
    [IssueType.COMPLEXITY_ISSUE]: IssueCategory.QUALITY,
    [IssueType.STYLE_ISSUE]: IssueCategory.QUALITY,
    
    [IssueType.SECURITY_ISSUE]: IssueCategory.SECURITY,
    
    [IssueType.PERFORMANCE_ISSUE]: IssueCategory.PERFORMANCE
  };
  
  return categoryMap[type] || IssueCategory.QUALITY;
}

function calculateImpactScore(severity: IssueSeverity | string, type: IssueType | string): number {
  // Base score by severity
  const severityScores: Record<string, number> = {
    [IssueSeverity.CRITICAL]: 10,
    [IssueSeverity.HIGH]: 8,
    [IssueSeverity.MEDIUM]: 5,
    [IssueSeverity.LOW]: 3,
    [IssueSeverity.INFO]: 1
  };
  
  // Adjustment by category
  const categoryAdjustments: Record<IssueCategory, number> = {
    [IssueCategory.FUNCTIONAL]: 2,
    [IssueCategory.SECURITY]: 2,
    [IssueCategory.PERFORMANCE]: 1,
    [IssueCategory.STRUCTURAL]: 0,
    [IssueCategory.QUALITY]: -1
  };
  
  const category = determineIssueCategory(type);
  const baseScore = severityScores[severity] || 5;
  const adjustment = categoryAdjustments[category] || 0;
  
  // Ensure score is between 1-10
  return Math.max(1, Math.min(10, baseScore + adjustment));
}

function calculateBlastRadiusImpact(affectedNodes: number, affectedEdges: number): number {
  // Simple formula: base score based on affected nodes, adjusted by edges
  const baseScore = Math.min(10, Math.ceil(affectedNodes / 5));
  const edgeAdjustment = Math.min(3, Math.ceil(affectedEdges / 10));
  
  return Math.max(1, Math.min(10, baseScore + edgeAdjustment));
}

function countIssuesBySeverity(errorHotspots: any[]): Record<IssueSeverity, number> {
  const counts: Record<IssueSeverity, number> = {
    [IssueSeverity.CRITICAL]: 0,
    [IssueSeverity.HIGH]: 0,
    [IssueSeverity.MEDIUM]: 0,
    [IssueSeverity.LOW]: 0,
    [IssueSeverity.INFO]: 0
  };
  
  errorHotspots.forEach(hotspot => {
    if (hotspot.issues) {
      hotspot.issues.forEach((issue: any) => {
        counts[issue.severity] = (counts[issue.severity] || 0) + 1;
      });
    }
  });
  
  return counts;
}

function countIssuesByCategory(errorHotspots: any[]): Record<IssueCategory, number> {
  const counts: Record<IssueCategory, number> = {
    [IssueCategory.FUNCTIONAL]: 0,
    [IssueCategory.STRUCTURAL]: 0,
    [IssueCategory.QUALITY]: 0,
    [IssueCategory.SECURITY]: 0,
    [IssueCategory.PERFORMANCE]: 0
  };
  
  errorHotspots.forEach(hotspot => {
    if (hotspot.issues) {
      hotspot.issues.forEach((issue: any) => {
        counts[issue.category] = (counts[issue.category] || 0) + 1;
      });
    }
  });
  
  return counts;
}

function groupIssuesByFile(errorHotspots: any[]): Record<string, Issue[]> {
  const issuesByFile: Record<string, Issue[]> = {};
  
  errorHotspots.forEach(hotspot => {
    if (hotspot.issues && hotspot.issues.length > 0) {
      const filePath = hotspot.path;
      if (!issuesByFile[filePath]) {
        issuesByFile[filePath] = [];
      }
      
      issuesByFile[filePath].push(...hotspot.issues);
    }
  });
  
  return issuesByFile;
}
