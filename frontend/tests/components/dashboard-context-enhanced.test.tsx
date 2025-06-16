import React from 'react';
import { render, act } from '@testing-library/react';
import { DashboardProviderEnhanced, useDashboardEnhanced } from '@/components/dashboard-context-enhanced';
import { IssueType, IssueSeverity, IssueCategory } from '@/lib/api-types';

// Mock the useSharedAnalysisState hook
jest.mock('@/hooks/useSharedAnalysisState', () => ({
  useSharedAnalysisState: () => ({
    selectedIssue: null,
    selectedNode: null,
    selectedFilePath: null,
    selectedSymbolName: null,
    setSelectedIssue: jest.fn(),
    setSelectedNode: jest.fn(),
    setSelectedFilePath: jest.fn(),
    setSelectedSymbolName: jest.fn()
  })
}));

// Test component that uses the dashboard context
const TestComponent = () => {
  const {
    repoUrl,
    setRepoUrl,
    activeView,
    setActiveView,
    selectedSymbol,
    setSelectedSymbol,
    navigateToView,
    navigateBack,
    shareDataBetweenViews,
    getSharedData
  } = useDashboardEnhanced();

  return (
    <div>
      <div data-testid="repo-url">{repoUrl}</div>
      <div data-testid="active-view">{activeView}</div>
      <div data-testid="selected-symbol">{selectedSymbol}</div>
      <button
        data-testid="set-repo-url"
        onClick={() => setRepoUrl('test/repo')}
      >
        Set Repo URL
      </button>
      <button
        data-testid="set-active-view"
        onClick={() => setActiveView('explorer')}
      >
        Set Active View
      </button>
      <button
        data-testid="set-selected-symbol"
        onClick={() => setSelectedSymbol('testSymbol')}
      >
        Set Selected Symbol
      </button>
      <button
        data-testid="navigate-to-view"
        onClick={() => navigateToView('blast-radius', { symbol: 'testSymbol' })}
      >
        Navigate To View
      </button>
      <button
        data-testid="navigate-back"
        onClick={() => navigateBack()}
      >
        Navigate Back
      </button>
      <button
        data-testid="share-data"
        onClick={() => shareDataBetweenViews('testKey', 'testValue')}
      >
        Share Data
      </button>
      <div data-testid="shared-data">{getSharedData('testKey')}</div>
    </div>
  );
};

describe('DashboardProviderEnhanced', () => {
  test('provides initial state values', () => {
    const { getByTestId } = render(
      <DashboardProviderEnhanced>
        <TestComponent />
      </DashboardProviderEnhanced>
    );
    
    expect(getByTestId('repo-url').textContent).toBe('');
    expect(getByTestId('active-view').textContent).toBe('metrics');
    expect(getByTestId('selected-symbol').textContent).toBe('');
    expect(getByTestId('shared-data').textContent).toBe('');
  });

  test('updates repoUrl when setRepoUrl is called', () => {
    const { getByTestId } = render(
      <DashboardProviderEnhanced>
        <TestComponent />
      </DashboardProviderEnhanced>
    );
    
    act(() => {
      getByTestId('set-repo-url').click();
    });
    
    expect(getByTestId('repo-url').textContent).toBe('test/repo');
  });

  test('updates activeView when setActiveView is called', () => {
    const { getByTestId } = render(
      <DashboardProviderEnhanced>
        <TestComponent />
      </DashboardProviderEnhanced>
    );
    
    act(() => {
      getByTestId('set-active-view').click();
    });
    
    expect(getByTestId('active-view').textContent).toBe('explorer');
  });

  test('updates selectedSymbol when setSelectedSymbol is called', () => {
    const { getByTestId } = render(
      <DashboardProviderEnhanced>
        <TestComponent />
      </DashboardProviderEnhanced>
    );
    
    act(() => {
      getByTestId('set-selected-symbol').click();
    });
    
    expect(getByTestId('selected-symbol').textContent).toBe('testSymbol');
  });

  test('navigates to view and updates state when navigateToView is called', () => {
    const { getByTestId } = render(
      <DashboardProviderEnhanced>
        <TestComponent />
      </DashboardProviderEnhanced>
    );
    
    act(() => {
      getByTestId('navigate-to-view').click();
    });
    
    expect(getByTestId('active-view').textContent).toBe('blast-radius');
    expect(getByTestId('selected-symbol').textContent).toBe('testSymbol');
  });

  test('shares data between views when shareDataBetweenViews is called', () => {
    const { getByTestId } = render(
      <DashboardProviderEnhanced>
        <TestComponent />
      </DashboardProviderEnhanced>
    );
    
    act(() => {
      getByTestId('share-data').click();
    });
    
    expect(getByTestId('shared-data').textContent).toBe('testValue');
  });

  test('throws error when useDashboardEnhanced is used outside of DashboardProviderEnhanced', () => {
    // Suppress console.error for this test
    const originalConsoleError = console.error;
    console.error = jest.fn();
    
    expect(() => {
      render(<TestComponent />);
    }).toThrow('useDashboardEnhanced must be used within a DashboardProviderEnhanced');
    
    // Restore console.error
    console.error = originalConsoleError;
  });
});

describe('Enhanced functionality', () => {
  // Mock issues and nodes for testing
  const mockIssues = [
    {
      id: 'issue-1',
      type: IssueType.UNUSED_VARIABLE,
      severity: IssueSeverity.CRITICAL,
      category: IssueCategory.QUALITY,
      message: 'Unused variable x',
      location: { file_path: 'src/file1.ts', start_line: 10, end_line: 10 },
      related_symbols: ['symbol1', 'symbol2']
    },
    {
      id: 'issue-2',
      type: IssueType.SECURITY_ISSUE,
      severity: IssueSeverity.HIGH,
      category: IssueCategory.SECURITY,
      message: 'Security vulnerability',
      location: { file_path: 'src/file2.ts', start_line: 20, end_line: 25 },
      related_symbols: ['symbol3']
    }
  ];

  const mockNodes = [
    {
      id: 'node-1',
      name: 'function1',
      type: 'function',
      path: 'src/file1.ts',
      issues: [mockIssues[0]],
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
    }
  ];

  // Test component that uses the enhanced functionality
  const EnhancedFunctionalityTestComponent = () => {
    const {
      getAllIssues,
      getAllNodes,
      findNodeById,
      findIssueById,
      findRelatedIssues,
      findRelatedNodes,
      setExplorationData,
      setBlastRadiusData
    } = useDashboardEnhanced();

    // Set mock data
    React.useEffect(() => {
      setExplorationData({
        repo_url: 'test/repo',
        summary: {
          total_nodes: 2,
          total_issues: 2,
          error_hotspots_count: 2,
          critical_paths_count: 0,
          issues_by_severity: {},
          issues_by_category: {}
        },
        error_hotspots: mockNodes,
        exploration_insights: [],
        critical_paths: [],
        issues_by_file: {}
      });

      setBlastRadiusData({
        target_symbol: mockNodes[0],
        blast_radius: {
          affected_nodes: 1,
          affected_edges: 1,
          impact_score: 5,
          affected_files: 1,
          affected_functions: 1,
          affected_classes: 0
        },
        affected_nodes: [mockNodes[1]],
        impact_graph: { nodes: [], edges: [] }
      });
    }, [setExplorationData, setBlastRadiusData]);

    return (
      <div>
        <div data-testid="all-issues-count">{getAllIssues().length}</div>
        <div data-testid="all-nodes-count">{getAllNodes().length}</div>
        <div data-testid="find-node-by-id">
          {findNodeById('node-1')?.name || 'not found'}
        </div>
        <div data-testid="find-issue-by-id">
          {findIssueById('issue-1')?.message || 'not found'}
        </div>
        <div data-testid="find-related-issues-count">
          {findRelatedIssues('issue-1').length}
        </div>
        <div data-testid="find-related-nodes-count">
          {findRelatedNodes('node-1').length}
        </div>
      </div>
    );
  };

  test('getAllIssues returns all issues from exploration and blast radius data', () => {
    const { getByTestId } = render(
      <DashboardProviderEnhanced>
        <EnhancedFunctionalityTestComponent />
      </DashboardProviderEnhanced>
    );
    
    // Wait for useEffect to run
    act(() => {
      jest.runAllTimers();
    });
    
    expect(getByTestId('all-issues-count').textContent).toBe('2');
  });

  test('getAllNodes returns all nodes from exploration and blast radius data', () => {
    const { getByTestId } = render(
      <DashboardProviderEnhanced>
        <EnhancedFunctionalityTestComponent />
      </DashboardProviderEnhanced>
    );
    
    // Wait for useEffect to run
    act(() => {
      jest.runAllTimers();
    });
    
    expect(getByTestId('all-nodes-count').textContent).toBe('2');
  });

  test('findNodeById returns the correct node', () => {
    const { getByTestId } = render(
      <DashboardProviderEnhanced>
        <EnhancedFunctionalityTestComponent />
      </DashboardProviderEnhanced>
    );
    
    // Wait for useEffect to run
    act(() => {
      jest.runAllTimers();
    });
    
    expect(getByTestId('find-node-by-id').textContent).toBe('function1');
  });

  test('findIssueById returns the correct issue', () => {
    const { getByTestId } = render(
      <DashboardProviderEnhanced>
        <EnhancedFunctionalityTestComponent />
      </DashboardProviderEnhanced>
    );
    
    // Wait for useEffect to run
    act(() => {
      jest.runAllTimers();
    });
    
    expect(getByTestId('find-issue-by-id').textContent).toBe('Unused variable x');
  });

  // Additional tests for findRelatedIssues and findRelatedNodes would go here
  // but they require more complex setup due to their implementation details
});

