import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { NodeCard } from '@/components/ui/node-card';
import { IssueType, IssueSeverity, IssueCategory } from '@/lib/api-types';

// Mock the useNodeSelection hook
jest.mock('@/hooks/useAnalysisState', () => ({
  useNodeSelection: () => ({
    selectedNode: null,
    setSelectedNode: jest.fn()
  })
}));

describe('NodeCard', () => {
  const mockIssues = [
    {
      id: 'issue-1',
      type: IssueType.UNUSED_VARIABLE,
      severity: IssueSeverity.CRITICAL,
      category: IssueCategory.QUALITY,
      message: 'Unused variable x',
      suggestion: 'Remove the variable',
      location: {
        file_path: 'src/file.ts',
        start_line: 10,
        end_line: 10
      }
    },
    {
      id: 'issue-2',
      type: IssueType.SECURITY_ISSUE,
      severity: IssueSeverity.HIGH,
      category: IssueCategory.SECURITY,
      message: 'Security vulnerability',
      suggestion: 'Fix security issue',
      location: {
        file_path: 'src/file.ts',
        start_line: 20,
        end_line: 25
      }
    }
  ];

  const mockNode = {
    id: 'node-1',
    name: 'testFunction',
    type: 'function',
    path: 'src/file.ts',
    issues: mockIssues,
    blast_radius: 5,
    metadata: {
      complexity: 8,
      lines_of_code: 25,
      dependencies: 3
    }
  };

  const mockHandlers = {
    onViewIssues: jest.fn(),
    onViewBlastRadius: jest.fn(),
    onViewInExplorer: jest.fn()
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders node name and path', () => {
    render(<NodeCard node={mockNode} />);
    
    expect(screen.getByText('testFunction')).toBeInTheDocument();
    expect(screen.getByText('src/file.ts')).toBeInTheDocument();
  });

  test('renders node type icon', () => {
    render(<NodeCard node={mockNode} />);
    
    // The icon is rendered as an SVG, so we can't easily check its content
    // But we can check that there's an SVG element
    const svg = document.querySelector('svg');
    expect(svg).toBeInTheDocument();
  });

  test('renders issue count badge when showIssueCount is true', () => {
    render(<NodeCard node={mockNode} showIssueCount={true} />);
    
    expect(screen.getByText('2 issues')).toBeInTheDocument();
  });

  test('does not render issue count badge when showIssueCount is false', () => {
    render(<NodeCard node={mockNode} showIssueCount={false} />);
    
    expect(screen.queryByText('2 issues')).not.toBeInTheDocument();
  });

  test('renders issue distribution when showIssueCount is true', () => {
    render(<NodeCard node={mockNode} showIssueCount={true} />);
    
    expect(screen.getByText('Issue severity')).toBeInTheDocument();
    // The distribution is rendered as colored divs, which we can't easily check
    // But we can check that the container div exists
    expect(document.querySelector('.flex.h-2.overflow-hidden.rounded-full.bg-muted')).toBeInTheDocument();
  });

  test('renders blast radius when showBlastRadius is true', () => {
    render(<NodeCard node={mockNode} showBlastRadius={true} />);
    
    expect(screen.getByText('Blast Radius:')).toBeInTheDocument();
    expect(screen.getByText('5')).toBeInTheDocument();
    expect(screen.getByText('nodes')).toBeInTheDocument();
  });

  test('does not render blast radius when showBlastRadius is false', () => {
    render(<NodeCard node={mockNode} showBlastRadius={false} />);
    
    expect(screen.queryByText('Blast Radius:')).not.toBeInTheDocument();
  });

  test('renders metadata when showMetadata is true', () => {
    render(<NodeCard node={mockNode} showMetadata={true} />);
    
    expect(screen.getByText('complexity:')).toBeInTheDocument();
    expect(screen.getByText('8')).toBeInTheDocument();
    expect(screen.getByText('lines of code:')).toBeInTheDocument();
    expect(screen.getByText('25')).toBeInTheDocument();
    expect(screen.getByText('dependencies:')).toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument();
  });

  test('does not render metadata when showMetadata is false', () => {
    render(<NodeCard node={mockNode} showMetadata={false} />);
    
    expect(screen.queryByText('complexity:')).not.toBeInTheDocument();
  });

  test('calls onViewIssues when Issues button is clicked', () => {
    render(<NodeCard node={mockNode} onViewIssues={mockHandlers.onViewIssues} />);
    
    fireEvent.click(screen.getByText('Issues'));
    
    expect(mockHandlers.onViewIssues).toHaveBeenCalledWith(mockNode);
  });

  test('calls onViewBlastRadius when Blast Radius button is clicked', () => {
    render(<NodeCard node={mockNode} onViewBlastRadius={mockHandlers.onViewBlastRadius} />);
    
    fireEvent.click(screen.getByText('Blast Radius'));
    
    expect(mockHandlers.onViewBlastRadius).toHaveBeenCalledWith(mockNode);
  });

  test('calls onViewInExplorer when Explorer button is clicked', () => {
    render(<NodeCard node={mockNode} onViewInExplorer={mockHandlers.onViewInExplorer} />);
    
    fireEvent.click(screen.getByText('Explorer'));
    
    expect(mockHandlers.onViewInExplorer).toHaveBeenCalledWith(mockNode);
  });

  test('applies custom className when provided', () => {
    const { container } = render(<NodeCard node={mockNode} className="custom-class" />);
    
    expect(container.firstChild).toHaveClass('custom-class');
  });

  test('handles node with no issues', () => {
    const nodeWithNoIssues = {
      ...mockNode,
      issues: []
    };
    
    render(<NodeCard node={nodeWithNoIssues} showIssueCount={true} />);
    
    expect(screen.queryByText('issues')).not.toBeInTheDocument();
    expect(screen.queryByText('Issue severity')).not.toBeInTheDocument();
  });

  test('handles node with no metadata', () => {
    const nodeWithNoMetadata = {
      ...mockNode,
      metadata: {}
    };
    
    render(<NodeCard node={nodeWithNoMetadata} showMetadata={true} />);
    
    expect(screen.queryByText('complexity:')).not.toBeInTheDocument();
  });
});
