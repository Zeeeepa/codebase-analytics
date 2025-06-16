"use client"

import { useState } from 'react';
import { Issue, IssueSeverity, IssueCategory, IssueType } from '@/lib/api-types';
import { useIssueSelection } from '@/hooks/useAnalysisState';
import { 
  AlertTriangle, 
  Code2, 
  Bug, 
  AlertCircle,
  Ban,
  Unlink,
  Repeat,
  Gauge,
  Shield,
  FileWarning,
  Sparkles,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Search,
  FileCode
} from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { cn } from '@/lib/utils';

interface IssueCardProps {
  issue: Issue;
  showSuggestion?: boolean;
  showCodeSnippet?: boolean;
  showFixExamples?: boolean;
  showRelatedSymbols?: boolean;
  onViewInFile?: (issue: Issue) => void;
  onViewRelatedIssues?: (issue: Issue) => void;
  className?: string;
}

export function IssueCard({
  issue,
  showSuggestion = true,
  showCodeSnippet = true,
  showFixExamples = true,
  showRelatedSymbols = true,
  onViewInFile,
  onViewRelatedIssues,
  className
}: IssueCardProps) {
  const [isOpen, setIsOpen] = useState(false);
  const { selectedIssue, setSelectedIssue } = useIssueSelection();
  
  const isSelected = selectedIssue?.id === issue.id;
  
  const handleSelect = () => {
    setSelectedIssue(isSelected ? null : issue);
  };
  
  const getIssueTypeIcon = (type: IssueType | string) => {
    switch (type) {
      case IssueType.UNUSED_IMPORT:
        return <Ban className="h-4 w-4" />;
      case IssueType.UNUSED_VARIABLE:
        return <Code2 className="h-4 w-4" />;
      case IssueType.UNDEFINED_VARIABLE:
        return <AlertCircle className="h-4 w-4" />;
      case IssueType.PARAMETER_MISMATCH:
        return <AlertTriangle className="h-4 w-4" />;
      case IssueType.CIRCULAR_DEPENDENCY:
        return <Repeat className="h-4 w-4" />;
      case IssueType.PERFORMANCE_ISSUE:
        return <Gauge className="h-4 w-4" />;
      case IssueType.SECURITY_ISSUE:
        return <Shield className="h-4 w-4" />;
      case IssueType.SYNTAX_ERROR:
        return <FileWarning className="h-4 w-4" />;
      case IssueType.STYLE_ISSUE:
        return <Sparkles className="h-4 w-4" />;
      default:
        return <Bug className="h-4 w-4" />;
    }
  };
  
  const getSeverityColor = (severity: IssueSeverity | string) => {
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
  };
  
  const getCategoryColor = (category: IssueCategory | string) => {
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
  };
  
  const getImpactBadgeInfo = (score: number) => {
    if (score >= 9) return { color: 'bg-red-600', label: 'Critical' };
    if (score >= 7) return { color: 'bg-orange-500', label: 'High' };
    if (score >= 5) return { color: 'bg-yellow-500', label: 'Medium' };
    if (score >= 3) return { color: 'bg-blue-500', label: 'Low' };
    return { color: 'bg-gray-500', label: 'Minimal' };
  };
  
  const impactInfo = issue.impact_score ? getImpactBadgeInfo(issue.impact_score) : null;
  
  return (
    <Card 
      className={cn(
        "transition-all duration-200",
        isSelected ? "border-primary shadow-md" : "",
        className
      )}
    >
      <Collapsible
        open={isOpen}
        onOpenChange={setIsOpen}
        className="w-full"
      >
        <div className="flex items-start p-4">
          <div className="mr-3 mt-0.5">
            {getIssueTypeIcon(issue.type)}
          </div>
          
          <div className="flex-1">
            <CollapsibleTrigger className="flex items-center justify-between w-full text-left">
              <div>
                <h3 className="font-medium">{issue.message}</h3>
                <p className="text-sm text-muted-foreground">
                  {issue.location?.file_path}:{issue.location?.start_line}
                </p>
              </div>
              {isOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </CollapsibleTrigger>
          </div>
        </div>
        
        <div className="px-4 pb-2 flex flex-wrap gap-2">
          <Badge variant="outline" className={`${getSeverityColor(issue.severity)} text-white`}>
            {issue.severity}
          </Badge>
          
          <Badge variant="outline" className={`${getCategoryColor(issue.category)} text-white`}>
            {issue.category}
          </Badge>
          
          {impactInfo && (
            <Badge variant="outline" className={`${impactInfo.color} text-white`}>
              Impact: {impactInfo.label} ({issue.impact_score}/10)
            </Badge>
          )}
        </div>
        
        <CollapsibleContent>
          <CardContent className="pt-0 pb-4">
            <div className="space-y-4">
              {showSuggestion && issue.suggestion && (
                <div>
                  <p className="text-sm font-medium mb-1">💡 Suggestion:</p>
                  <p className="text-sm text-muted-foreground">{issue.suggestion}</p>
                </div>
              )}
              
              {showCodeSnippet && issue.code_snippet && (
                <div>
                  <p className="text-sm font-medium mb-1">Code snippet:</p>
                  <pre className="text-xs bg-muted p-2 rounded-md overflow-x-auto">
                    {issue.code_snippet}
                  </pre>
                </div>
              )}
              
              {showFixExamples && issue.fix_examples && issue.fix_examples.length > 0 && (
                <div>
                  <p className="text-sm font-medium mb-1">Fix example:</p>
                  <pre className="text-xs bg-muted p-2 rounded-md overflow-x-auto">
                    {issue.fix_examples[0]}
                  </pre>
                </div>
              )}
              
              {showRelatedSymbols && issue.related_symbols && issue.related_symbols.length > 0 && (
                <div>
                  <p className="text-sm font-medium mb-1">Related symbols:</p>
                  <div className="flex flex-wrap gap-2">
                    {issue.related_symbols.map((symbol, index) => (
                      <Badge key={index} variant="secondary">
                        {symbol}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
              
              <div className="flex flex-wrap gap-2 mt-4">
                {onViewInFile && (
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={() => onViewInFile(issue)}
                    className="flex items-center gap-1"
                  >
                    <FileCode className="h-3 w-3" />
                    View in File
                  </Button>
                )}
                
                {onViewRelatedIssues && (
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={() => onViewRelatedIssues(issue)}
                    className="flex items-center gap-1"
                  >
                    <Search className="h-3 w-3" />
                    Find Related Issues
                  </Button>
                )}
                
                <Button 
                  variant={isSelected ? "default" : "outline"} 
                  size="sm" 
                  onClick={handleSelect}
                  className="flex items-center gap-1"
                >
                  {isSelected ? 'Deselect' : 'Select'}
                </Button>
              </div>
            </div>
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
}

