"use client"

import { useState, useEffect, useMemo } from 'react';
import { useDashboard } from '@/components/dashboard-context';
import { apiService } from '@/lib/api-service';
import { 
  Search, 
  Code2, 
  AlertTriangle, 
  Target, 
  Zap, 
  FileCode, 
  Bug, 
  AlertCircle,
  Ban,
  Unlink,
  Repeat,
  Gauge,
  Shield,
  FileWarning,
  Sparkles,
  Filter,
  SortDesc,
  Info
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Separator } from '@/components/ui/separator';
import { 
  Issue, 
  IssueType, 
  IssueSeverity, 
  IssueCategory,
  VisualNode,
  ExplorationData
} from '@/lib/api-types';

export function ExplorerView() {
  const { 
    repoUrl, 
    explorationData, 
    setExplorationData, 
    isLoading, 
    setIsLoading, 
    error, 
    setError,
    analysisMode
  } = useDashboard();

  // Local state for filtering and sorting
  const [severityFilter, setSeverityFilter] = useState<IssueSeverity | 'all'>('all');
  const [categoryFilter, setCategoryFilter] = useState<IssueCategory | 'all'>('all');
  const [sortBy, setSortBy] = useState<'impact' | 'severity' | 'type'>('impact');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedIssue, setSelectedIssue] = useState<Issue | null>(null);
  const [activeTab, setActiveTab] = useState('issues');

  const handleVisualExploration = async () => {
    if (!repoUrl.trim()) {
      setError("Please enter a repository URL or use '.' for current directory");
      return;
    }

    setIsLoading(true);
    setError("");
    
    try {
      const data = await apiService.exploreVisual(repoUrl, analysisMode);
      setExplorationData(data);
    } catch (error) {
      console.error('Error during visual exploration:', error);
      setError('Failed to analyze repository. Please check the URL and try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Filter and sort issues
  const filteredHotspots = useMemo(() => {
    if (!explorationData?.error_hotspots) return [];
    
    return explorationData.error_hotspots.filter(hotspot => {
      // Filter out hotspots with no issues
      if (!hotspot.issues || hotspot.issues.length === 0) return false;
      
      // Filter by severity and category if needed
      if (severityFilter !== 'all' || categoryFilter !== 'all' || searchQuery) {
        hotspot.issues = hotspot.issues.filter(issue => {
          const matchesSeverity = severityFilter === 'all' || issue.severity === severityFilter;
          const matchesCategory = categoryFilter === 'all' || issue.category === categoryFilter;
          const matchesSearch = !searchQuery || 
            issue.message.toLowerCase().includes(searchQuery.toLowerCase()) ||
            issue.type.toLowerCase().includes(searchQuery.toLowerCase()) ||
            hotspot.name.toLowerCase().includes(searchQuery.toLowerCase());
          
          return matchesSeverity && matchesCategory && matchesSearch;
        });
        
        return hotspot.issues.length > 0;
      }
      
      return true;
    });
  }, [explorationData, severityFilter, categoryFilter, searchQuery]);

  // Sort issues within each hotspot
  const sortedHotspots = useMemo(() => {
    return filteredHotspots.map(hotspot => {
      const sortedIssues = [...hotspot.issues].sort((a, b) => {
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
        } else {
          return a.type.localeCompare(b.type);
        }
      });
      
      return {
        ...hotspot,
        issues: sortedIssues
      };
    });
  }, [filteredHotspots, sortBy]);

  // Calculate summary statistics
  const summaryStats = useMemo(() => {
    if (!explorationData) return null;
    
    const totalIssues = explorationData.summary.total_issues || 0;
    const issuesBySeverity = explorationData.summary.issues_by_severity || {};
    const issuesByCategory = explorationData.summary.issues_by_category || {};
    
    return {
      totalIssues,
      issuesBySeverity,
      issuesByCategory
    };
  }, [explorationData]);

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

  const getSeverityLabel = (severity: IssueSeverity | string) => {
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

  const getCategoryLabel = (category: IssueCategory | string) => {
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
  };

  const getIssueTypeIcon = (type: IssueType | string) => {
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
  };

  const getIssueTypeLabel = (type: IssueType | string) => {
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
  };

  const getImpactBadge = (score: number) => {
    let color = 'bg-gray-500';
    let label = 'Unknown';
    
    if (score >= 9) {
      color = 'bg-red-600';
      label = 'Critical';
    } else if (score >= 7) {
      color = 'bg-orange-500';
      label = 'High';
    } else if (score >= 5) {
      color = 'bg-yellow-500';
      label = 'Medium';
    } else if (score >= 3) {
      color = 'bg-blue-500';
      label = 'Low';
    } else {
      color = 'bg-gray-500';
      label = 'Minimal';
    }
    
    return (
      <Badge variant="outline" className={`text-white ${color}`}>
        Impact: {label} ({score}/10)
      </Badge>
    );
  };

  return (
    <div className="space-y-6">
      {/* Analysis Controls */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-xl flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-orange-500" />
            Code Issue Explorer
          </CardTitle>
          <CardDescription>
            Detect and visualize code issues, parameter mismatches, and other problems
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-4">
            <div className="flex gap-2">
              <Button 
                onClick={handleVisualExploration} 
                disabled={isLoading || !repoUrl}
                className="flex items-center gap-2"
              >
                <Zap className="h-4 w-4" />
                {isLoading ? 'Analyzing...' : 'Analyze Code Issues'}
              </Button>
              
              <Input
                placeholder="Search issues..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="flex-1"
                disabled={isLoading || !explorationData}
              />
            </div>
            
            {error && (
              <Alert variant="destructive">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
            
            {isLoading && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Analyzing codebase for issues...</span>
                  <span>Please wait</span>
                </div>
                <Progress value={undefined} className="w-full" />
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      {explorationData && !isLoading && (
        <div className="space-y-6">
          {/* Summary Statistics */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">Analysis Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card>
                  <CardContent className="pt-6">
                    <div className="text-2xl font-bold">{summaryStats?.totalIssues || 0}</div>
                    <p className="text-xs text-muted-foreground">Total Issues</p>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardContent className="pt-6">
                    <div className="text-2xl font-bold text-red-600">
                      {summaryStats?.issuesBySeverity?.[IssueSeverity.CRITICAL] || 0}
                    </div>
                    <p className="text-xs text-muted-foreground">Critical Issues</p>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardContent className="pt-6">
                    <div className="text-2xl font-bold text-purple-600">
                      {summaryStats?.issuesByCategory?.[IssueCategory.FUNCTIONAL] || 0}
                    </div>
                    <p className="text-xs text-muted-foreground">Functional Issues</p>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardContent className="pt-6">
                    <div className="text-2xl font-bold text-amber-600">
                      {summaryStats?.issuesByCategory?.[IssueCategory.PERFORMANCE] || 0}
                    </div>
                    <p className="text-xs text-muted-foreground">Performance Issues</p>
                  </CardContent>
                </Card>
              </div>
            </CardContent>
          </Card>

          {/* Tabs for different views */}
          <Tabs defaultValue="issues" value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="issues">Issues List</TabsTrigger>
              <TabsTrigger value="hotspots">Error Hotspots</TabsTrigger>
              <TabsTrigger value="insights">Insights</TabsTrigger>
            </TabsList>
            
            {/* Issues List Tab */}
            <TabsContent value="issues" className="space-y-4">
              <div className="flex flex-wrap gap-2 mb-4">
                <Select value={severityFilter} onValueChange={(value) => setSeverityFilter(value as any)}>
                  <SelectTrigger className="w-[180px]">
                    <SelectValue placeholder="Filter by Severity" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Severities</SelectItem>
                    <SelectItem value={IssueSeverity.CRITICAL}>Critical</SelectItem>
                    <SelectItem value={IssueSeverity.HIGH}>High</SelectItem>
                    <SelectItem value={IssueSeverity.MEDIUM}>Medium</SelectItem>
                    <SelectItem value={IssueSeverity.LOW}>Low</SelectItem>
                    <SelectItem value={IssueSeverity.INFO}>Info</SelectItem>
                  </SelectContent>
                </Select>
                
                <Select value={categoryFilter} onValueChange={(value) => setCategoryFilter(value as any)}>
                  <SelectTrigger className="w-[180px]">
                    <SelectValue placeholder="Filter by Category" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Categories</SelectItem>
                    <SelectItem value={IssueCategory.FUNCTIONAL}>Functional</SelectItem>
                    <SelectItem value={IssueCategory.STRUCTURAL}>Structural</SelectItem>
                    <SelectItem value={IssueCategory.QUALITY}>Quality</SelectItem>
                    <SelectItem value={IssueCategory.SECURITY}>Security</SelectItem>
                    <SelectItem value={IssueCategory.PERFORMANCE}>Performance</SelectItem>
                  </SelectContent>
                </Select>
                
                <Select value={sortBy} onValueChange={(value) => setSortBy(value as any)}>
                  <SelectTrigger className="w-[180px]">
                    <SelectValue placeholder="Sort by" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="impact">Sort by Impact</SelectItem>
                    <SelectItem value="severity">Sort by Severity</SelectItem>
                    <SelectItem value="type">Sort by Type</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <ScrollArea className="h-[600px]">
                <div className="space-y-4">
                  {sortedHotspots.length > 0 ? (
                    sortedHotspots.map((hotspot) => (
                      <Card key={hotspot.id} className="overflow-hidden">
                        <CardHeader className="pb-3">
                          <CardTitle className="text-lg flex items-center gap-2">
                            <FileCode className="h-5 w-5 text-blue-500" />
                            {hotspot.name}
                            <Badge variant="outline" className="ml-2">
                              {hotspot.issues.length} issues
                            </Badge>
                          </CardTitle>
                          <CardDescription>{hotspot.path}</CardDescription>
                        </CardHeader>
                        <CardContent className="p-0">
                          <div className="divide-y">
                            {hotspot.issues.map((issue) => (
                              <Collapsible key={issue.id} className="px-6 py-3 hover:bg-muted/50 transition-colors">
                                <CollapsibleTrigger className="flex items-start justify-between w-full text-left">
                                  <div className="flex items-start gap-3">
                                    <div className="mt-1">
                                      <TooltipProvider>
                                        <Tooltip>
                                          <TooltipTrigger>
                                            {getIssueTypeIcon(issue.type)}
                                          </TooltipTrigger>
                                          <TooltipContent>
                                            <p>{getIssueTypeLabel(issue.type)}</p>
                                          </TooltipContent>
                                        </Tooltip>
                                      </TooltipProvider>
                                    </div>
                                    <div className="flex-1">
                                      <div className="flex flex-wrap items-center gap-2 mb-1">
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
                                                {getImpactBadge(issue.impact_score)}
                                              </TooltipTrigger>
                                              <TooltipContent>
                                                <p>Impact score indicates the potential effect of this issue</p>
                                              </TooltipContent>
                                            </Tooltip>
                                          </TooltipProvider>
                                        )}
                                      </div>
                                      <p className="text-sm">{issue.message}</p>
                                    </div>
                                  </div>
                                  <Info className="h-4 w-4 text-muted-foreground shrink-0 ml-2" />
                                </CollapsibleTrigger>
                                <CollapsibleContent className="pt-3">
                                  <div className="space-y-3 text-sm">
                                    {issue.suggestion && (
                                      <div className="bg-muted p-3 rounded-md">
                                        <p className="font-medium text-blue-600">💡 Suggestion:</p>
                                        <p>{issue.suggestion}</p>
                                      </div>
                                    )}
                                    
                                    {issue.location && (
                                      <div className="flex items-center gap-2 text-muted-foreground">
                                        <span>Location:</span>
                                        <span className="font-mono">
                                          {issue.location.file_path}:{issue.location.start_line}
                                          {issue.location.end_line !== issue.location.start_line && 
                                            `-${issue.location.end_line}`}
                                        </span>
                                      </div>
                                    )}
                                    
                                    {issue.code_snippet && (
                                      <div className="space-y-1">
                                        <p className="text-muted-foreground">Code snippet:</p>
                                        <pre className="bg-muted p-3 rounded-md overflow-x-auto font-mono text-xs">
                                          {issue.code_snippet}
                                        </pre>
                                      </div>
                                    )}
                                    
                                    {issue.fix_examples && issue.fix_examples.length > 0 && (
                                      <div className="space-y-1">
                                        <p className="text-muted-foreground">Fix example:</p>
                                        <pre className="bg-muted p-3 rounded-md overflow-x-auto font-mono text-xs">
                                          {issue.fix_examples[0]}
                                        </pre>
                                      </div>
                                    )}
                                    
                                    {issue.related_symbols && issue.related_symbols.length > 0 && (
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
                                  </div>
                                </CollapsibleContent>
                              </Collapsible>
                            ))}
                          </div>
                        </CardContent>
                      </Card>
                    ))
                  ) : (
                    <div className="text-center py-8">
                      <p className="text-muted-foreground">
                        {explorationData ? 
                          'No issues found matching the current filters.' : 
                          'No issues data available. Run analysis first.'}
                      </p>
                    </div>
                  )}
                </div>
              </ScrollArea>
            </TabsContent>
            
            {/* Error Hotspots Tab */}
            <TabsContent value="hotspots">
              <Card>
                <CardHeader>
                  <CardTitle>Error Hotspots</CardTitle>
                  <CardDescription>Files and functions with the most issues</CardDescription>
                </CardHeader>
                <CardContent>
                  {explorationData.error_hotspots && explorationData.error_hotspots.length > 0 ? (
                    <div className="space-y-4">
                      {explorationData.error_hotspots
                        .sort((a, b) => (b.issues?.length || 0) - (a.issues?.length || 0))
                        .slice(0, 10)
                        .map((hotspot) => (
                          <div key={hotspot.id} className="flex items-center gap-4 p-4 border rounded-lg">
                            <div className="flex-1">
                              <div className="flex items-center gap-2">
                                <FileCode className="h-5 w-5 text-blue-500" />
                                <span className="font-medium">{hotspot.name}</span>
                                <Badge variant="destructive">{hotspot.issues?.length || 0} issues</Badge>
                              </div>
                              <p className="text-sm text-muted-foreground mt-1">{hotspot.path}</p>
                            </div>
                            <div className="w-32">
                              <div className="text-sm text-muted-foreground mb-1">Issue severity</div>
                              <div className="flex h-2 overflow-hidden rounded-full bg-muted">
                                {Object.values(IssueSeverity).map(severity => {
                                  const count = hotspot.issues?.filter(i => i.severity === severity).length || 0;
                                  const percentage = hotspot.issues?.length ? (count / hotspot.issues.length) * 100 : 0;
                                  return percentage > 0 ? (
                                    <div 
                                      key={severity}
                                      className={`${getSeverityColor(severity)}`}
                                      style={{ width: `${percentage}%` }}
                                    />
                                  ) : null;
                                })}
                              </div>
                            </div>
                          </div>
                        ))
                      }
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <p className="text-muted-foreground">No error hotspots data available.</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
            
            {/* Insights Tab */}
            <TabsContent value="insights">
              <Card>
                <CardHeader>
                  <CardTitle>Analysis Insights</CardTitle>
                  <CardDescription>Key findings and recommendations</CardDescription>
                </CardHeader>
                <CardContent>
                  {explorationData.exploration_insights && explorationData.exploration_insights.length > 0 ? (
                    <div className="space-y-4">
                      {explorationData.exploration_insights.map((insight, index) => (
                        <Card key={insight.id || index}>
                          <CardHeader className="pb-2">
                            <div className="flex items-center justify-between">
                              <CardTitle className="text-lg">{insight.title}</CardTitle>
                              <Badge 
                                variant="outline" 
                                className={`text-white ${getSeverityColor(insight.priority)}`}
                              >
                                {getSeverityLabel(insight.priority)}
                              </Badge>
                            </div>
                          </CardHeader>
                          <CardContent>
                            <p className="text-sm mb-3">{insight.description}</p>
                            {insight.recommendation && (
                              <div className="bg-muted p-3 rounded-md mb-3">
                                <p className="font-medium text-blue-600">💡 Recommendation:</p>
                                <p className="text-sm">{insight.recommendation}</p>
                              </div>
                            )}
                            {insight.affected_nodes && insight.affected_nodes.length > 0 && (
                              <div className="text-sm text-muted-foreground">
                                Affects {insight.affected_nodes.length} nodes
                              </div>
                            )}
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <p className="text-muted-foreground">No insights data available.</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      )}
    </div>
  );
}

import { useState } from 'react';

