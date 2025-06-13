"use client"

import { Issue, IssueType, IssueSeverity, IssueCategory } from '@/lib/api-types'
import { 
  getSeverityColor, 
  getSeverityLabel, 
  getCategoryColor, 
  getCategoryLabel, 
  getIssueTypeIcon, 
  getIssueTypeLabel,
  getImpactBadgeInfo
} from '@/lib/analysis-utils'
import { useIssueSelection } from '@/hooks/useSharedAnalysisState'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { Button } from '@/components/ui/button'
import { Info, AlertTriangle, AlertCircle, Ban, Code2, FileWarning, Repeat, Unlink, FileCode, Sparkles, Shield, Gauge, Bug, ExternalLink } from 'lucide-react'
import { useState } from 'react'

interface IssueCardProps {
  issue: Issue
  showLocation?: boolean
  showCodeSnippet?: boolean
  showSuggestion?: boolean
  showRelatedSymbols?: boolean
  showFixExamples?: boolean
  onViewInFile?: (issue: Issue) => void
  onViewRelatedIssues?: (issue: Issue) => void
  className?: string
}

export function IssueCard({
  issue,
  showLocation = true,
  showCodeSnippet = true,
  showSuggestion = true,
  showRelatedSymbols = true,
  showFixExamples = true,
  onViewInFile,
  onViewRelatedIssues,
  className = ''
}: IssueCardProps) {
  const [isOpen, setIsOpen] = useState(false)
  const { selectedIssue, setSelectedIssue } = useIssueSelection()
  
  const isSelected = selectedIssue?.id === issue.id
  
  const handleSelect = () => {
    setSelectedIssue(isSelected ? null : issue)
  }
  
  const getIconComponent = (type: IssueType | string) => {
    switch (type) {
      case IssueType.UNUSED_IMPORT:
        return <Ban className="h-4 w-4" />;
      case IssueType.UNUSED_VARIABLE:
      case IssueType.UNUSED_FUNCTION:
      case IssueType.UNUSED_PARAMETER:
        return <Code2 className="h-4 w-4" />;
      case IssueType.UNDEFINED_VARIABLE:
      case IssueType.UNDEFINED_FUNCTION:
        return <AlertCircle className="h-4 w-4" />;
      case IssueType.PARAMETER_MISMATCH:
        return <AlertTriangle className="h-4 w-4" />;
      case IssueType.TYPE_ERROR:
        return <FileWarning className="h-4 w-4" />;
      case IssueType.CIRCULAR_DEPENDENCY:
        return <Repeat className="h-4 w-4" />;
      case IssueType.DEAD_CODE:
        return <Unlink className="h-4 w-4" />;
      case IssueType.COMPLEXITY_ISSUE:
        return <FileCode className="h-4 w-4" />;
      case IssueType.STYLE_ISSUE:
        return <Sparkles className="h-4 w-4" />;
      case IssueType.SECURITY_ISSUE:
        return <Shield className="h-4 w-4" />;
      case IssueType.PERFORMANCE_ISSUE:
        return <Gauge className="h-4 w-4" />;
      default:
        return <Bug className="h-4 w-4" />;
    }
  }
  
  const renderImpactBadge = (score: number) => {
    const { color, label } = getImpactBadgeInfo(score);
    
    return (
      <Badge variant="outline" className={`text-white ${color}`}>
        Impact: {label} ({score}/10)
      </Badge>
    );
  }
  
  return (
    <Card className={`overflow-hidden ${isSelected ? 'ring-2 ring-primary' : ''} ${className}`}>
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <CollapsibleTrigger className="w-full text-left">
          <CardHeader className="pb-2">
            <div className="flex items-start justify-between">
              <div className="flex items-start gap-3">
                <div className="mt-1">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        {getIconComponent(issue.type)}
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>{getIssueTypeLabel(issue.type)}</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <div>
                  <CardTitle className="text-base">{issue.message}</CardTitle>
                  {issue.location && (
                    <CardDescription className="text-xs mt-1">
                      {issue.location.file_path}:{issue.location.start_line}
                      {issue.location.end_line !== issue.location.start_line && 
                        `-${issue.location.end_line}`}
                    </CardDescription>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Info className="h-4 w-4 text-muted-foreground" />
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2 mt-2">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger>
                    <Badge 
                      variant="outline" 
                      className={`text-white ${getSeverityColor(issue.severity)}`}
                    >
                      {getSeverityLabel(issue.severity)}
                    </Badge>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Issue severity level</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger>
                    <Badge 
                      variant="outline" 
                      className={`text-white ${getCategoryColor(issue.category)}`}
                    >
                      {getCategoryLabel(issue.category)}
                    </Badge>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Issue category</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              
              {issue.impact_score && (
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger>
                      {renderImpactBadge(issue.impact_score)}
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Impact score indicates the potential effect of this issue</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              )}
            </div>
          </CardHeader>
        </CollapsibleTrigger>
        
        <CollapsibleContent>
          <CardContent className="pt-0 pb-4">
            <div className="space-y-4 text-sm">
              {showSuggestion && issue.suggestion && (
                <div className="bg-muted p-3 rounded-md">
                  <p className="font-medium text-blue-600">💡 Suggestion:</p>
                  <p>{issue.suggestion}</p>
                </div>
              )}
              
              {showCodeSnippet && issue.code_snippet && (
                <div className="space-y-1">
                  <p className="text-muted-foreground">Code snippet:</p>
                  <pre className="bg-muted p-3 rounded-md overflow-x-auto font-mono text-xs">
                    {issue.code_snippet}
                  </pre>
                </div>
              )}
              
              {showFixExamples && issue.fix_examples && issue.fix_examples.length > 0 && (
                <div className="space-y-1">
                  <p className="text-muted-foreground">Fix example:</p>
                  <pre className="bg-muted p-3 rounded-md overflow-x-auto font-mono text-xs">
                    {issue.fix_examples[0]}
                  </pre>
                </div>
              )}
              
              {showRelatedSymbols && issue.related_symbols && issue.related_symbols.length > 0 && (
                <div className="space-y-1">
                  <p className="text-muted-foreground">Related symbols:</p>
                  <div className="flex flex-wrap gap-2">
                    {issue.related_symbols.map((symbol, index) => (
                      <Badge key={index} variant="secondary">
                        {symbol}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
              
              <div className="flex items-center justify-between pt-2">
                <div className="flex items-center gap-2">
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={handleSelect}
                  >
                    {isSelected ? 'Deselect' : 'Select'}
                  </Button>
                  
                  {onViewInFile && (
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        onViewInFile(issue);
                      }}
                    >
                      <ExternalLink className="h-4 w-4 mr-1" />
                      View in File
                    </Button>
                  )}
                </div>
                
                {onViewRelatedIssues && (
                  <Button 
                    variant="ghost" 
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      onViewRelatedIssues(issue);
                    }}
                  >
                    Find Related Issues
                  </Button>
                )}
              </div>
            </div>
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  )
}

