import { 
  filterIssues, 
  filterNodes, 
  sortIssues, 
  sortNodes, 
  calculateSummaryStats, 
  calculateBlastRadiusStats, 
  getSeverityColor, 
  getSeverityLabel, 
  getCategoryColor, 
  getCategoryLabel, 
  getIssueTypeIcon, 
  getIssueTypeLabel, 
  getImpactBadgeInfo, 
  groupIssuesByFile, 
  findRelatedIssues, 
  extractInsights, 
  findHotspots, 
  calculateCodeQualityScore 
} from '@/lib/analysis-utils';
import { 
  Issue, 
  IssueType, 
  IssueSeverity, 
  IssueCategory, 
  VisualNode, 
  ExplorationData, 
  BlastRadiusData 
} from '@/lib/api-types';

// Mock data for testing
const mockIssues: Issue[] = [
  {
    id: 'issue-1',
    type: IssueType.UNUSED_VARIABLE,
    severity: IssueSeverity.CRITICAL,
    category: IssueCategory.QUALITY,
    message: 'Unused variable x',
    suggestion: 'Remove the variable',
    location: {
      file_path: 'src/file1.ts',
      start_line: 10,
      end_line: 10
    },
    impact_score: 8
  },
  {
    id: 'issue-2',
    type: IssueType.SECURITY_ISSUE,
    severity: IssueSeverity.HIGH,
    category: IssueCategory.SECURITY,
    message: 'Potential XSS vulnerability',
    suggestion: 'Sanitize input',
    location: {
      file_path: 'src/file2.ts',
      start_line: 20,
      end_line: 25
    },
    impact_score: 9
  },
  {
    id: 'issue-3',
    type: IssueType.PERFORMANCE_ISSUE,
    severity: IssueSeverity.MEDIUM,
    category: IssueCategory.PERFORMANCE,
    message: 'Inefficient loop',
    suggestion: 'Use map instead',
    location: {
      file_path: 'src/file1.ts',
      start_line: 30,
      end_line: 35
    },
    impact_score: 5
  },
  {
    id: 'issue-4',
    type: IssueType.UNUSED_IMPORT,
    severity: IssueSeverity.LOW,
    category: IssueCategory.QUALITY,
    message: 'Unused import',
    suggestion: 'Remove the import',
    location: {
      file_path: 'src/file3.ts',
      start_line: 5,
      end_line: 5
    },
    impact_score: 2
  }
];

const mockNodes: VisualNode[] = [
  {
    id: 'node-1',
    name: 'function1',
    type: 'function',
    path: 'src/file1.ts',
    issues: [mockIssues[0], mockIssues[2]],
    blast_radius: 5,
    metadata: {}
  },
  {
    id: 'node-2',
    name: 'function2',
    type: 'function',
    path: 'src/file2.ts',
    issues: [mockIssues[1]],
    blast_radius: 10,
    metadata: {}
  },
  {
    id: 'node-3',
    name: 'class1',
    type: 'class',
    path: 'src/file3.ts',
    issues: [mockIssues[3]],
    blast_radius: 3,
    metadata: {}
  },
  {
    id: 'node-4',
    name: 'function3',
    type: 'function',
    path: 'src/file4.ts',
    issues: [],
    blast_radius: 1,
    metadata: {}
  }
];

const mockExplorationData: ExplorationData = {
  repo_url: 'test/repo',
  summary: {
    total_nodes: 10,
    total_issues: 4,
    error_hotspots_count: 2,
    critical_paths_count: 1,
    issues_by_severity: {
      [IssueSeverity.CRITICAL]: 1,
      [IssueSeverity.HIGH]: 1,
      [IssueSeverity.MEDIUM]: 1,
      [IssueSeverity.LOW]: 1,
      [IssueSeverity.INFO]: 0
    },
    issues_by_category: {
      [IssueCategory.QUALITY]: 2,
      [IssueCategory.SECURITY]: 1,
      [IssueCategory.PERFORMANCE]: 1,
      [IssueCategory.FUNCTIONAL]: 0,
      [IssueCategory.STRUCTURAL]: 0
    }
  },
  error_hotspots: mockNodes,
  exploration_insights: [
    {
      id: 'insight-1',
      type: 'quality',
      priority: IssueSeverity.HIGH,
      title: 'Code Quality Issues',
      description: 'Multiple code quality issues found',
      affected_nodes: ['node-1', 'node-3'],
      recommendation: 'Review code quality',
      impact_score: 7
    }
  ],
  critical_paths: [],
  issues_by_file: {
    'src/file1.ts': [mockIssues[0], mockIssues[2]],
    'src/file2.ts': [mockIssues[1]],
    'src/file3.ts': [mockIssues[3]]
  }
};

const mockBlastRadiusData: BlastRadiusData = {
  target_symbol: mockNodes[0],
  blast_radius: {
    affected_nodes: 5,
    affected_edges: 8,
    impact_score: 7,
    affected_files: 3,
    affected_functions: 4,
    affected_classes: 1
  },
  affected_nodes: [mockNodes[1], mockNodes[2]],
  impact_graph: {
    nodes: [],
    edges: []
  }
};

describe('filterIssues', () => {
  test('should filter issues by severity', () => {
    const result = filterIssues(mockIssues, IssueSeverity.CRITICAL);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe('issue-1');
  });

  test('should filter issues by category', () => {
    const result = filterIssues(mockIssues, 'all', IssueCategory.QUALITY);
    expect(result).toHaveLength(2);
    expect(result[0].id).toBe('issue-1');
    expect(result[1].id).toBe('issue-4');
  });

  test('should filter issues by type', () => {
    const result = filterIssues(mockIssues, 'all', 'all', IssueType.PERFORMANCE_ISSUE);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe('issue-3');
  });

  test('should filter issues by search query', () => {
    const result = filterIssues(mockIssues, 'all', 'all', 'all', 'xss');
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe('issue-2');
  });

  test('should return all issues when no filters are applied', () => {
    const result = filterIssues(mockIssues);
    expect(result).toHaveLength(4);
  });

  test('should handle empty issues array', () => {
    const result = filterIssues([]);
    expect(result).toHaveLength(0);
  });
});

describe('filterNodes', () => {
  test('should filter nodes by issue severity', () => {
    const result = filterNodes(mockNodes, IssueSeverity.CRITICAL);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe('node-1');
  });

  test('should filter nodes by issue category', () => {
    const result = filterNodes(mockNodes, 'all', IssueCategory.SECURITY);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe('node-2');
  });

  test('should filter out nodes with no issues', () => {
    const result = filterNodes(mockNodes);
    expect(result).toHaveLength(3);
    expect(result.find(node => node.id === 'node-4')).toBeUndefined();
  });

  test('should handle empty nodes array', () => {
    const result = filterNodes([]);
    expect(result).toHaveLength(0);
  });
});

describe('sortIssues', () => {
  test('should sort issues by impact', () => {
    const result = sortIssues(mockIssues, 'impact');
    expect(result[0].id).toBe('issue-2'); // impact_score: 9
    expect(result[1].id).toBe('issue-1'); // impact_score: 8
    expect(result[2].id).toBe('issue-3'); // impact_score: 5
    expect(result[3].id).toBe('issue-4'); // impact_score: 2
  });

  test('should sort issues by severity', () => {
    const result = sortIssues(mockIssues, 'severity');
    expect(result[0].id).toBe('issue-1'); // CRITICAL
    expect(result[1].id).toBe('issue-2'); // HIGH
    expect(result[2].id).toBe('issue-3'); // MEDIUM
    expect(result[3].id).toBe('issue-4'); // LOW
  });

  test('should sort issues by type', () => {
    const result = sortIssues(mockIssues, 'type');
    // Sort alphabetically by type
    expect(result.map(issue => issue.type)).toEqual([
      IssueType.PERFORMANCE_ISSUE,
      IssueType.SECURITY_ISSUE,
      IssueType.UNUSED_IMPORT,
      IssueType.UNUSED_VARIABLE
    ]);
  });

  test('should sort issues by name (message)', () => {
    const result = sortIssues(mockIssues, 'name');
    // Sort alphabetically by message
    expect(result.map(issue => issue.message)).toEqual([
      'Inefficient loop',
      'Potential XSS vulnerability',
      'Unused import',
      'Unused variable x'
    ]);
  });
});

describe('sortNodes', () => {
  test('should sort nodes by impact (issue count)', () => {
    const result = sortNodes(mockNodes, 'impact');
    expect(result[0].id).toBe('node-1'); // 2 issues
    expect(result[1].id).toBe('node-2'); // 1 issue
    expect(result[2].id).toBe('node-3'); // 1 issue
    expect(result[3].id).toBe('node-4'); // 0 issues
  });

  test('should sort nodes by severity', () => {
    const result = sortNodes(mockNodes, 'severity');
    expect(result[0].id).toBe('node-1'); // Has CRITICAL issue
    expect(result[1].id).toBe('node-2'); // Has HIGH issue
    expect(result[2].id).toBe('node-3'); // Has LOW issue
    expect(result[3].id).toBe('node-4'); // No issues
  });

  test('should sort nodes by type', () => {
    const result = sortNodes(mockNodes, 'type');
    // Sort alphabetically by type
    expect(result.map(node => node.type)).toEqual([
      'class',
      'function',
      'function',
      'function'
    ]);
  });

  test('should sort nodes by name', () => {
    const result = sortNodes(mockNodes, 'name');
    // Sort alphabetically by name
    expect(result.map(node => node.name)).toEqual([
      'class1',
      'function1',
      'function2',
      'function3'
    ]);
  });
});

describe('calculateSummaryStats', () => {
  test('should calculate summary statistics from exploration data', () => {
    const result = calculateSummaryStats(mockExplorationData);
    expect(result).toEqual({
      totalIssues: 4,
      issuesBySeverity: {
        [IssueSeverity.CRITICAL]: 1,
        [IssueSeverity.HIGH]: 1,
        [IssueSeverity.MEDIUM]: 1,
        [IssueSeverity.LOW]: 1,
        [IssueSeverity.INFO]: 0
      },
      issuesByCategory: {
        [IssueCategory.QUALITY]: 2,
        [IssueCategory.SECURITY]: 1,
        [IssueCategory.PERFORMANCE]: 1,
        [IssueCategory.FUNCTIONAL]: 0,
        [IssueCategory.STRUCTURAL]: 0
      }
    });
  });

  test('should handle null data', () => {
    const result = calculateSummaryStats(null);
    expect(result).toBeNull();
  });
});

describe('calculateBlastRadiusStats', () => {
  test('should calculate blast radius statistics', () => {
    const result = calculateBlastRadiusStats(mockBlastRadiusData);
    expect(result).toEqual({
      affectedNodes: 5,
      affectedEdges: 8,
      impactScore: 7,
      percentageAffected: 5 // (5/100) * 100
    });
  });

  test('should handle null data', () => {
    const result = calculateBlastRadiusStats(null);
    expect(result).toBeNull();
  });
});

describe('getSeverityColor', () => {
  test('should return correct color for each severity level', () => {
    expect(getSeverityColor(IssueSeverity.CRITICAL)).toBe('bg-red-600');
    expect(getSeverityColor(IssueSeverity.HIGH)).toBe('bg-orange-500');
    expect(getSeverityColor(IssueSeverity.MEDIUM)).toBe('bg-yellow-500');
    expect(getSeverityColor(IssueSeverity.LOW)).toBe('bg-blue-500');
    expect(getSeverityColor(IssueSeverity.INFO)).toBe('bg-gray-500');
    expect(getSeverityColor('unknown')).toBe('bg-gray-500');
  });
});

describe('getSeverityLabel', () => {
  test('should return correct label for each severity level', () => {
    expect(getSeverityLabel(IssueSeverity.CRITICAL)).toBe('Critical');
    expect(getSeverityLabel(IssueSeverity.HIGH)).toBe('High');
    expect(getSeverityLabel(IssueSeverity.MEDIUM)).toBe('Medium');
    expect(getSeverityLabel(IssueSeverity.LOW)).toBe('Low');
    expect(getSeverityLabel(IssueSeverity.INFO)).toBe('Info');
    expect(getSeverityLabel('unknown')).toBe('unknown');
  });
});

describe('getCategoryColor', () => {
  test('should return correct color for each category', () => {
    expect(getCategoryColor(IssueCategory.FUNCTIONAL)).toBe('bg-purple-500');
    expect(getCategoryColor(IssueCategory.STRUCTURAL)).toBe('bg-indigo-500');
    expect(getCategoryColor(IssueCategory.QUALITY)).toBe('bg-teal-500');
    expect(getCategoryColor(IssueCategory.SECURITY)).toBe('bg-red-500');
    expect(getCategoryColor(IssueCategory.PERFORMANCE)).toBe('bg-amber-500');
    expect(getCategoryColor('unknown')).toBe('bg-gray-500');
  });
});

describe('getCategoryLabel', () => {
  test('should return correct label for each category', () => {
    expect(getCategoryLabel(IssueCategory.FUNCTIONAL)).toBe('Functional');
    expect(getCategoryLabel(IssueCategory.STRUCTURAL)).toBe('Structural');
    expect(getCategoryLabel(IssueCategory.QUALITY)).toBe('Quality');
    expect(getCategoryLabel(IssueCategory.SECURITY)).toBe('Security');
    expect(getCategoryLabel(IssueCategory.PERFORMANCE)).toBe('Performance');
    expect(getCategoryLabel('unknown')).toBe('unknown');
  });
});

describe('getIssueTypeIcon', () => {
  test('should return correct icon name for each issue type', () => {
    expect(getIssueTypeIcon(IssueType.UNUSED_IMPORT)).toBe('Ban');
    expect(getIssueTypeIcon(IssueType.UNUSED_VARIABLE)).toBe('Code2');
    expect(getIssueTypeIcon(IssueType.UNDEFINED_VARIABLE)).toBe('AlertCircle');
    expect(getIssueTypeIcon(IssueType.PARAMETER_MISMATCH)).toBe('AlertTriangle');
    expect(getIssueTypeIcon(IssueType.CIRCULAR_DEPENDENCY)).toBe('Repeat');
    expect(getIssueTypeIcon('unknown')).toBe('Bug');
  });
});

describe('getIssueTypeLabel', () => {
  test('should return correct label for each issue type', () => {
    expect(getIssueTypeLabel(IssueType.UNUSED_IMPORT)).toBe('Unused Import');
    expect(getIssueTypeLabel(IssueType.UNUSED_VARIABLE)).toBe('Unused Variable');
    expect(getIssueTypeLabel(IssueType.PARAMETER_MISMATCH)).toBe('Parameter Mismatch');
    expect(getIssueTypeLabel(IssueType.SECURITY_ISSUE)).toBe('Security Issue');
    expect(getIssueTypeLabel('CUSTOM_TYPE')).toBe('Custom Type');
  });
});

describe('getImpactBadgeInfo', () => {
  test('should return correct badge info for different impact scores', () => {
    expect(getImpactBadgeInfo(10)).toEqual({ color: 'bg-red-600', label: 'Critical' });
    expect(getImpactBadgeInfo(8)).toEqual({ color: 'bg-orange-500', label: 'High' });
    expect(getImpactBadgeInfo(6)).toEqual({ color: 'bg-yellow-500', label: 'Medium' });
    expect(getImpactBadgeInfo(4)).toEqual({ color: 'bg-blue-500', label: 'Low' });
    expect(getImpactBadgeInfo(2)).toEqual({ color: 'bg-gray-500', label: 'Minimal' });
  });
});

describe('groupIssuesByFile', () => {
  test('should group issues by file path', () => {
    const result = groupIssuesByFile(mockIssues);
    expect(Object.keys(result)).toHaveLength(3);
    expect(result['src/file1.ts']).toHaveLength(2);
    expect(result['src/file2.ts']).toHaveLength(1);
    expect(result['src/file3.ts']).toHaveLength(1);
  });

  test('should handle issues with no location', () => {
    const issuesWithNoLocation = [
      { ...mockIssues[0], location: undefined },
      mockIssues[1]
    ];
    const result = groupIssuesByFile(issuesWithNoLocation);
    expect(Object.keys(result)).toHaveLength(2);
    expect(result['unknown']).toHaveLength(1);
    expect(result['src/file2.ts']).toHaveLength(1);
  });
});

describe('findRelatedIssues', () => {
  test('should find issues in the same file', () => {
    const result = findRelatedIssues(mockIssues[0], mockIssues);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe('issue-3'); // Same file: src/file1.ts
  });

  test('should find issues with the same type', () => {
    // Create a test issue with the same type but different file
    const testIssue = {
      ...mockIssues[0],
      id: 'test-issue',
      location: { file_path: 'src/different.ts', start_line: 1, end_line: 1 }
    };
    const result = findRelatedIssues(testIssue, [...mockIssues, testIssue]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe('issue-1'); // Same type: UNUSED_VARIABLE
  });

  test('should handle null issue', () => {
    const result = findRelatedIssues(null, mockIssues);
    expect(result).toHaveLength(0);
  });
});

describe('extractInsights', () => {
  test('should extract insights from exploration data', () => {
    const result = extractInsights(mockExplorationData);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe('insight-1');
  });

  test('should handle null data', () => {
    const result = extractInsights(null);
    expect(result).toHaveLength(0);
  });
});

describe('findHotspots', () => {
  test('should find hotspots from exploration data', () => {
    const result = findHotspots(mockExplorationData, 2);
    expect(result).toHaveLength(2);
    expect(result[0].id).toBe('node-1'); // 2 issues
    expect(result[1].id).toBe('node-2'); // 1 issue (tied with node-3, but comes first in the array)
  });

  test('should handle null data', () => {
    const result = findHotspots(null);
    expect(result).toHaveLength(0);
  });
});

describe('calculateCodeQualityScore', () => {
  test('should calculate code quality score', () => {
    const result = calculateCodeQualityScore(mockExplorationData);
    // Expected calculation:
    // Base score: 100
    // Weighted issues: 1*10 (CRITICAL) + 1*5 (HIGH) + 1*2 (MEDIUM) + 1*1 (LOW) + 0*0.1 (INFO) = 18
    // Final score: 100 - 18 = 82
    expect(result).toBe(82);
  });

  test('should handle null data', () => {
    const result = calculateCodeQualityScore(null);
    expect(result).toBe(0);
  });
});

