"use client"

import { useState, useEffect, useMemo } from 'react';
import { useDashboard } from '@/components/dashboard-context';
import { apiService } from '@/lib/api-service';
import { 
  filterIssues, 
  filterNodes, 
  sortIssues, 
  sortNodes, 
  calculateSummaryStats, 
  findHotspots, 
  extractInsights 
} from '@/lib/analysis-utils';
import { 
  useFilters, 
  useViewOptions, 
  useIssueSelection, 
  useNodeSelection 
} from '@/hooks/useSharedAnalysisState';
import { 
  Search, 
  Zap, 
  AlertTriangle, 
  FileCode, 
  Info,
  BarChart3,
  Target
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { FilterBar } from '@/components/ui/filter-bar';
import { SortBar } from '@/components/ui/sort-bar';
import { IssueCard } from '@/components/ui/issue-card';
import { NodeCard } from '@/components/ui/node-card';
import { Issue, VisualNode } from '@/lib/api-types';

export function ExplorerViewEnhanced() {
  const { 
    repoUrl, 
    explorationData, 
    setExplorationData, 
    isLoading, 
    setIsLoading, 
    error, 
    setError,
    analysisMode,
    setActiveView,
    setSelectedSymbol,
    setBlastRadiusData
  } = useDashboard();

  // Get shared state from hooks
  const { 
    severityFilter, 
    categoryFilter, 
    typeFilter, 
    searchQuery 
  } = useFilters();
  
  const { sortBy, viewMode } = useViewOptions();
  const { selectedIssue } = useIssueSelection();
  const { selectedNode } = useNodeSelection();
  
  // Local state
  const [activeTab, setActiveTab] = useState('issues');

  // Handle visual exploration
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
  const filteredIssues = useMemo(() => {
    if (!explorationData) return [];
    
    // Collect all issues from all hotspots
    const allIssues: Issue[] = [];
    explorationData.error_hotspots.forEach(hotspot => {
      if (hotspot.issues && hotspot.issues.length > 0) {
        allIssues.push(...hotspot.issues);
      }
    });
    
    // Filter issues
    return filterIssues(allIssues, severityFilter, categoryFilter, typeFilter, searchQuery);
  }, [explorationData, severityFilter, categoryFilter, typeFilter, searchQuery]);

  // Sort issues
  const sortedIssues = useMemo(() => {
    return sortIssues(filteredIssues, sortBy);
  }, [filteredIssues, sortBy]);

  // Filter and sort hotspots
  const filteredHotspots = useMemo(() => {
    if (!explorationData?.error_hotspots) return [];
    
    return filterNodes(
      explorationData.error_hotspots,
      severityFilter,
      categoryFilter,
      typeFilter,
      searchQuery
    );
  }, [explorationData, severityFilter, categoryFilter, typeFilter, searchQuery]);

  // Sort hotspots
  const sortedHotspots = useMemo(() => {
    return sortNodes(filteredHotspots, sortBy);
  }, [filteredHotspots, sortBy]);

  // Calculate summary statistics
  const summaryStats = useMemo(() => {
    return calculateSummaryStats(explorationData);
  }, [explorationData]);

  // Get insights
  const insights = useMemo(() => {
    return extractInsights(explorationData);
  }, [explorationData]);

  // Get top hotspots
  const topHotspots = useMemo(() => {
    return findHotspots(explorationData, 10);
  }, [explorationData]);

  // Handle view in file
  const handleViewInFile = (issue: Issue) => {
    // Navigate to structure view and select the file
    if (issue.location?.file_path) {
      // Set the selected file in the dashboard context
      // This would typically be handled by the structure view
      // For now, we'll just log it
      console.log('View in file:', issue.location.file_path);
    }
  };

  // Handle view related issues
  const handleViewRelatedIssues = (issue: Issue) => {
    // Filter issues to show only related ones
    // This could be implemented by setting a special filter
    console.log('View related issues:', issue.id);
  };

  // Handle view blast radius
  const handleViewBlastRadius = (node: VisualNode) => {
    // Navigate to blast radius view and analyze the selected node
    setSelectedSymbol(node.name);
    setActiveView('blast-radius');
    
    // Optionally, trigger the blast radius analysis immediately
    // This would require modifying the blast-radius-view component
    // to accept a node parameter
  };

  // Handle view in explorer
  const handleViewInExplorer = (node: VisualNode) => {
    // This could open a modal or navigate to a detailed view
    console.log('View in explorer:', node.id);
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
                      {summaryStats?.issuesBySeverity?.CRITICAL || 0}
                    </div>
                    <p className="text-xs text-muted-foreground">Critical Issues</p>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardContent className="pt-6">
                    <div className="text-2xl font-bold text-purple-600">
                      {summaryStats?.issuesByCategory?.FUNCTIONAL || 0}
                    </div>
                    <p className="text-xs text-muted-foreground">Functional Issues</p>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardContent className="pt-6">
                    <div className="text-2xl font-bold text-amber-600">
                      {summaryStats?.issuesByCategory?.PERFORMANCE || 0}
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
              <TabsTrigger value="issues" className="flex items-center gap-1">
                <AlertTriangle className="h-4 w-4" />
                Issues List
              </TabsTrigger>
              <TabsTrigger value="hotspots" className="flex items-center gap-1">
                <FileCode className="h-4 w-4" />
                Error Hotspots
              </TabsTrigger>
              <TabsTrigger value="insights" className="flex items-center gap-1">
                <BarChart3 className="h-4 w-4" />
                Insights
              </TabsTrigger>
            </TabsList>
            
            {/* Issues List Tab */}
            <TabsContent value="issues" className="space-y-4">
              <div className="flex flex-wrap justify-between gap-2 mb-4">
                <FilterBar />
                <SortBar />
              </div>
              
              <ScrollArea className="h-[600px]">
                <div className="space-y-4">
                  {sortedIssues.length > 0 ? (
                    sortedIssues.map((issue) => (
                      <IssueCard
                        key={issue.id}
                        issue={issue}
                        onViewInFile={handleViewInFile}
                        onViewRelatedIssues={handleViewRelatedIssues}
                      />
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
            <TabsContent value="hotspots" className="space-y-4">
              <div className="flex flex-wrap justify-between gap-2 mb-4">
                <FilterBar />
                <SortBar />
              </div>
              
              <ScrollArea className="h-[600px]">
                <div className="space-y-4">
                  {sortedHotspots.length > 0 ? (
                    sortedHotspots.map((hotspot) => (
                      <NodeCard
                        key={hotspot.id}
                        node={hotspot}
                        onViewIssues={() => {
                          setActiveTab('issues');
                          // Ideally, we would filter issues to show only those from this hotspot
                        }}
                        onViewBlastRadius={handleViewBlastRadius}
                        onViewInExplorer={handleViewInExplorer}
                      />
                    ))
                  ) : (
                    <div className="text-center py-8">
                      <p className="text-muted-foreground">
                        {explorationData ? 
                          'No hotspots found matching the current filters.' : 
                          'No hotspots data available. Run analysis first.'}
                      </p>
                    </div>
                  )}
                </div>
              </ScrollArea>
            </TabsContent>
            
            {/* Insights Tab */}
            <TabsContent value="insights" className="space-y-4">
              <ScrollArea className="h-[600px]">
                <div className="space-y-4">
                  {insights && insights.length > 0 ? (
                    insights.map((insight, index) => (
                      <Card key={insight.id || index}>
                        <CardHeader className="pb-2">
                          <div className="flex items-center justify-between">
                            <CardTitle className="text-lg">{insight.title}</CardTitle>
                            <div className={`px-2 py-1 rounded-full text-xs text-white ${getSeverityColor(insight.priority)}`}>
                              {getSeverityLabel(insight.priority)}
                            </div>
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
                    ))
                  ) : (
                    <div className="text-center py-8">
                      <p className="text-muted-foreground">No insights data available.</p>
                    </div>
                  )}
                </div>
              </ScrollArea>
            </TabsContent>
          </Tabs>
        </div>
      )}
    </div>
  );
}

// Helper functions
function getSeverityColor(severity: string): string {
  switch (severity) {
    case 'CRITICAL':
      return 'bg-red-600';
    case 'HIGH':
      return 'bg-orange-500';
    case 'MEDIUM':
      return 'bg-yellow-500';
    case 'LOW':
      return 'bg-blue-500';
    case 'INFO':
      return 'bg-gray-500';
    default:
      return 'bg-gray-500';
  }
}

function getSeverityLabel(severity: string): string {
  switch (severity) {
    case 'CRITICAL':
      return 'Critical';
    case 'HIGH':
      return 'High';
    case 'MEDIUM':
      return 'Medium';
    case 'LOW':
      return 'Low';
    case 'INFO':
      return 'Info';
    default:
      return severity;
  }
}

