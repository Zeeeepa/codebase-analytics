import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { IssueCard } from '@/components/ui/issue-card';
import { IssueType, IssueSeverity, IssueCategory } from '@/lib/api-types';

// Mock the useIssueSelection hook
jest.mock('@/hooks/useSharedAnalysisState', () => ({
  useIssueSelection: () => ({
    selectedIssue: null,
    setSelectedIssue: jest.fn()
  })
}));

describe('IssueCard', () => {
  const mockIssue = {
    id: 'issue-1',
    type: IssueType.UNUSED_VARIABLE,
    severity: IssueSeverity.MEDIUM,
    category: IssueCategory.QUALITY,
    message: 'Unused variable x',
    suggestion: 'Remove the variable',
    location: {
      file_path: 'src/file.ts',
      start_line: 10,
      end_line: 10
    },
    impact_score: 6,
    code_snippet: 'const x = 5;',
    fix_examples: ['// Remove this line'],
    related_symbols: ['functionA', 'classB']
  };

  const mockHandlers = {
    onViewInFile: jest.fn(),
    onViewRelatedIssues: jest.fn()
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders issue message and location', () => {
    render(<IssueCard issue={mockIssue} />);
    
    expect(screen.getByText('Unused variable x')).toBeInTheDocument();
    expect(screen.getByText('src/file.ts:10')).toBeInTheDocument();
  });

  test('renders severity and category badges', () => {
    render(<IssueCard issue={mockIssue} />);
    
    expect(screen.getByText('Medium')).toBeInTheDocument();
    expect(screen.getByText('Quality')).toBeInTheDocument();
  });

  test('renders impact score badge when provided', () => {
    render(<IssueCard issue={mockIssue} />);
    
    expect(screen.getByText(/Impact: Medium/)).toBeInTheDocument();
    expect(screen.getByText(/\(6\/10\)/)).toBeInTheDocument();
  });

  test('renders suggestion when showSuggestion is true', () => {
    render(<IssueCard issue={mockIssue} showSuggestion={true} />);
    
    // Open the collapsible content
    fireEvent.click(screen.getByText('Unused variable x'));
    
    expect(screen.getByText('💡 Suggestion:')).toBeInTheDocument();
    expect(screen.getByText('Remove the variable')).toBeInTheDocument();
  });

  test('does not render suggestion when showSuggestion is false', () => {
    render(<IssueCard issue={mockIssue} showSuggestion={false} />);
    
    // Open the collapsible content
    fireEvent.click(screen.getByText('Unused variable x'));
    
    expect(screen.queryByText('💡 Suggestion:')).not.toBeInTheDocument();
  });

  test('renders code snippet when showCodeSnippet is true', () => {
    render(<IssueCard issue={mockIssue} showCodeSnippet={true} />);
    
    // Open the collapsible content
    fireEvent.click(screen.getByText('Unused variable x'));
    
    expect(screen.getByText('Code snippet:')).toBeInTheDocument();
    expect(screen.getByText('const x = 5;')).toBeInTheDocument();
  });

  test('does not render code snippet when showCodeSnippet is false', () => {
    render(<IssueCard issue={mockIssue} showCodeSnippet={false} />);
    
    // Open the collapsible content
    fireEvent.click(screen.getByText('Unused variable x'));
    
    expect(screen.queryByText('Code snippet:')).not.toBeInTheDocument();
  });

  test('renders fix examples when showFixExamples is true', () => {
    render(<IssueCard issue={mockIssue} showFixExamples={true} />);
    
    // Open the collapsible content
    fireEvent.click(screen.getByText('Unused variable x'));
    
    expect(screen.getByText('Fix example:')).toBeInTheDocument();
    expect(screen.getByText('// Remove this line')).toBeInTheDocument();
  });

  test('renders related symbols when showRelatedSymbols is true', () => {
    render(<IssueCard issue={mockIssue} showRelatedSymbols={true} />);
    
    // Open the collapsible content
    fireEvent.click(screen.getByText('Unused variable x'));
    
    expect(screen.getByText('Related symbols:')).toBeInTheDocument();
    expect(screen.getByText('functionA')).toBeInTheDocument();
    expect(screen.getByText('classB')).toBeInTheDocument();
  });

  test('calls onViewInFile when View in File button is clicked', () => {
    render(<IssueCard issue={mockIssue} onViewInFile={mockHandlers.onViewInFile} />);
    
    // Open the collapsible content
    fireEvent.click(screen.getByText('Unused variable x'));
    
    // Click the View in File button
    fireEvent.click(screen.getByText('View in File'));
    
    expect(mockHandlers.onViewInFile).toHaveBeenCalledWith(mockIssue);
  });

  test('calls onViewRelatedIssues when Find Related Issues button is clicked', () => {
    render(<IssueCard issue={mockIssue} onViewRelatedIssues={mockHandlers.onViewRelatedIssues} />);
    
    // Open the collapsible content
    fireEvent.click(screen.getByText('Unused variable x'));
    
    // Click the Find Related Issues button
    fireEvent.click(screen.getByText('Find Related Issues'));
    
    expect(mockHandlers.onViewRelatedIssues).toHaveBeenCalledWith(mockIssue);
  });

  test('applies custom className when provided', () => {
    const { container } = render(<IssueCard issue={mockIssue} className="custom-class" />);
    
    expect(container.firstChild).toHaveClass('custom-class');
  });
});

