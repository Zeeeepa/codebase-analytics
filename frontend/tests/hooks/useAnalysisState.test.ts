import { renderHook, act } from '@testing-library/react-hooks';
import { 
  useAnalysisState, 
  useIssueSelection, 
  useNodeSelection, 
  useFilters, 
  useViewOptions, 
  useNavigation 
} from '@/hooks/useAnalysisState';
import { IssueSeverity, IssueCategory, IssueType } from '@/lib/api-types';

// Mock issue and node data
const mockIssue = {
  id: 'issue-1',
  type: IssueType.UNUSED_VARIABLE,
  severity: IssueSeverity.MEDIUM,
  category: IssueCategory.QUALITY,
  message: 'Unused variable',
  suggestion: 'Remove the variable',
  location: {
    file_path: 'src/file.ts',
    start_line: 10,
    end_line: 10
  }
};

const mockNode = {
  id: 'node-1',
  name: 'testFunction',
  type: 'function',
  path: 'src/file.ts',
  issues: [],
  blast_radius: 5,
  metadata: {}
};

describe('useAnalysisState', () => {
  beforeEach(() => {
    // Reset the store before each test
    act(() => {
      useAnalysisState.getState().resetAll();
    });
  });

  test('should initialize with default values', () => {
    const { result } = renderHook(() => useAnalysisState());
    
    expect(result.current.selectedIssue).toBeNull();
    expect(result.current.selectedNode).toBeNull();
    expect(result.current.selectedFilePath).toBeNull();
    expect(result.current.selectedSymbolName).toBeNull();
    expect(result.current.severityFilter).toBe('all');
    expect(result.current.categoryFilter).toBe('all');
    expect(result.current.typeFilter).toBe('all');
    expect(result.current.searchQuery).toBe('');
    expect(result.current.sortBy).toBe('impact');
    expect(result.current.viewMode).toBe('list');
    expect(result.current.navigationHistory).toEqual([]);
    expect(result.current.currentHistoryIndex).toBe(-1);
  });

  test('should update selected issue', () => {
    const { result } = renderHook(() => useAnalysisState());
    
    act(() => {
      result.current.setSelectedIssue(mockIssue);
    });
    
    expect(result.current.selectedIssue).toEqual(mockIssue);
  });

  test('should update selected node', () => {
    const { result } = renderHook(() => useAnalysisState());
    
    act(() => {
      result.current.setSelectedNode(mockNode);
    });
    
    expect(result.current.selectedNode).toEqual(mockNode);
  });

  test('should update filters', () => {
    const { result } = renderHook(() => useAnalysisState());
    
    act(() => {
      result.current.setSeverityFilter(IssueSeverity.CRITICAL);
      result.current.setCategoryFilter(IssueCategory.SECURITY);
      result.current.setTypeFilter(IssueType.SECURITY_ISSUE);
      result.current.setSearchQuery('test');
    });
    
    expect(result.current.severityFilter).toBe(IssueSeverity.CRITICAL);
    expect(result.current.categoryFilter).toBe(IssueCategory.SECURITY);
    expect(result.current.typeFilter).toBe(IssueType.SECURITY_ISSUE);
    expect(result.current.searchQuery).toBe('test');
  });

  test('should reset filters', () => {
    const { result } = renderHook(() => useAnalysisState());
    
    act(() => {
      result.current.setSeverityFilter(IssueSeverity.CRITICAL);
      result.current.setCategoryFilter(IssueCategory.SECURITY);
      result.current.setTypeFilter(IssueType.SECURITY_ISSUE);
      result.current.setSearchQuery('test');
      result.current.resetFilters();
    });
    
    expect(result.current.severityFilter).toBe('all');
    expect(result.current.categoryFilter).toBe('all');
    expect(result.current.typeFilter).toBe('all');
    expect(result.current.searchQuery).toBe('');
  });

  test('should update view options', () => {
    const { result } = renderHook(() => useAnalysisState());
    
    act(() => {
      result.current.setSortBy('severity');
      result.current.setViewMode('graph');
    });
    
    expect(result.current.sortBy).toBe('severity');
    expect(result.current.viewMode).toBe('graph');
  });

  test('should handle navigation', () => {
    const { result } = renderHook(() => useAnalysisState());
    
    act(() => {
      result.current.navigateTo('/page1');
      result.current.navigateTo('/page2');
      result.current.navigateTo('/page3');
    });
    
    expect(result.current.navigationHistory).toEqual(['/page1', '/page2', '/page3']);
    expect(result.current.currentHistoryIndex).toBe(2);
    
    act(() => {
      result.current.navigateBack();
    });
    
    expect(result.current.currentHistoryIndex).toBe(1);
    
    act(() => {
      result.current.navigateForward();
    });
    
    expect(result.current.currentHistoryIndex).toBe(2);
  });

  test('should reset all state', () => {
    const { result } = renderHook(() => useAnalysisState());
    
    act(() => {
      result.current.setSelectedIssue(mockIssue);
      result.current.setSelectedNode(mockNode);
      result.current.setSeverityFilter(IssueSeverity.CRITICAL);
      result.current.navigateTo('/page1');
      result.current.resetAll();
    });
    
    expect(result.current.selectedIssue).toBeNull();
    expect(result.current.selectedNode).toBeNull();
    expect(result.current.severityFilter).toBe('all');
    expect(result.current.navigationHistory).toEqual([]);
  });
});

describe('useIssueSelection', () => {
  test('should provide issue selection functionality', () => {
    const { result } = renderHook(() => useIssueSelection());
    
    act(() => {
      result.current.setSelectedIssue(mockIssue);
    });
    
    expect(result.current.selectedIssue).toEqual(mockIssue);
  });
});

describe('useNodeSelection', () => {
  test('should provide node selection functionality', () => {
    const { result } = renderHook(() => useNodeSelection());
    
    act(() => {
      result.current.setSelectedNode(mockNode);
    });
    
    expect(result.current.selectedNode).toEqual(mockNode);
  });
});

describe('useFilters', () => {
  test('should provide filtering functionality', () => {
    const { result } = renderHook(() => useFilters());
    
    act(() => {
      result.current.setSeverityFilter(IssueSeverity.HIGH);
      result.current.setCategoryFilter(IssueCategory.PERFORMANCE);
    });
    
    expect(result.current.severityFilter).toBe(IssueSeverity.HIGH);
    expect(result.current.categoryFilter).toBe(IssueCategory.PERFORMANCE);
    
    act(() => {
      result.current.resetFilters();
    });
    
    expect(result.current.severityFilter).toBe('all');
    expect(result.current.categoryFilter).toBe('all');
  });
});

describe('useViewOptions', () => {
  test('should provide view options functionality', () => {
    const { result } = renderHook(() => useViewOptions());
    
    act(() => {
      result.current.setSortBy('name');
      result.current.setViewMode('tree');
    });
    
    expect(result.current.sortBy).toBe('name');
    expect(result.current.viewMode).toBe('tree');
  });
});

describe('useNavigation', () => {
  test('should provide navigation functionality', () => {
    const { result } = renderHook(() => useNavigation());
    
    expect(result.current.canGoBack).toBe(false);
    expect(result.current.canGoForward).toBe(false);
    
    act(() => {
      result.current.navigateTo('/page1');
      result.current.navigateTo('/page2');
    });
    
    expect(result.current.currentPath).toBe('/page2');
    expect(result.current.canGoBack).toBe(true);
    expect(result.current.canGoForward).toBe(false);
    
    act(() => {
      result.current.navigateBack();
    });
    
    expect(result.current.currentPath).toBe('/page1');
    expect(result.current.canGoBack).toBe(false);
    expect(result.current.canGoForward).toBe(true);
  });
});

